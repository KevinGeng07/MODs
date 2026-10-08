"""Tables, a deterministic batch sampler, and a Dataset that marks rows QUEUED
in modsdb as DataLoader workers load them from the user's Dataset.

The samples themselves never enter modsdb. Each run gets its own table
`run<N>` with one row per training sample (row i + 1 = sample i) holding only
that run's training state; it is dropped when the run ends. The history tables
hold only the current run (see clear_history)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset, get_worker_info

from .db import IDLE, QUEUED, DBError, put

# One row per training sample in a run's table `run<N>` (id = sample index + 1): its state in this run.
RUN_ROWS_SCHEMA = {"status": "int64", "epoch": "int64", "worker": "int64", "batch": "int64", "seen": "int64",
                   "loss": "float64", "pred": "int64", "step": "int64"}
META_SCHEMA = {"runs": "int64"}  # row 1: number of runs ever started
# One row per run (id = run): its settings and latest step.
RUNS_SCHEMA = {"batch_size": "int64", "workers": "int64", "step": "int64"}
# One row per consumed batch: id = step_id(run, step).
STEPS_SCHEMA = {"run": "int64", "step": "int64", "epoch": "int64", "batch": "int64", "worker": "int64",
                "n": "int64", "loss": "float64", "acc": "float64"}
# One row per sample of a consumed batch: id = step_id * 1000 + position in batch.
BATCH_SAMPLES_SCHEMA = {"run": "int64", "step": "int64", "sample": "int64", "label": "int64", "pred": "int64",
                        "loss": "float64", "correct": "int64"}
VALSET_SCHEMA = {"val_id": "int64"}  # id = run * 100 + k: the run's 10 shown validation samples
EVALS_SCHEMA = {"run": "int64", "step": "int64", "epoch": "int64", "acc": "float64", "n": "int64"}  # id = run * 100000 + n
MAX_BATCH = 999
MAX_STEPS = 10_000_000  # per run


def run_table(run: int) -> str:
    return f"run{run}"


def step_id(run: int, step: int) -> int:
    return run * MAX_STEPS + step


def sample_row_id(sid: int, k: int) -> int:
    return sid * 1000 + k


HISTORY_TABLES = (("runs", RUNS_SCHEMA), ("steps", STEPS_SCHEMA), ("batch_samples", BATCH_SAMPLES_SCHEMA),
                  ("valset", VALSET_SCHEMA), ("evals", EVALS_SCHEMA))


def setup_tables(client) -> None:
    """Create any missing shared tables (CreateTable is idempotent)."""
    for name, schema in (("meta", META_SCHEMA), *HISTORY_TABLES):
        client.create_table(name, schema)
    if not client.get_rows("meta", [1]):
        client.apply([put("meta", 1, {"runs": 0})])


def clear_history(client) -> None:
    """Forget every earlier run: drop their state tables and empty the history tables.
    The run counter in `meta` keeps counting, so an old tab can never reach a new run."""
    for r in client.scan("runs"):
        try:
            client.drop_table(run_table(r.id))
        except DBError:  # already dropped when that run ended
            pass
    for name, schema in HISTORY_TABLES:
        client.drop_table(name)
        client.create_table(name, schema)


def fresh_run_rows(table: str, ids) -> list:
    """Puts that give a run one clean training-state row per sample."""
    return [put(table, i, {"status": IDLE, "epoch": -1, "worker": -1, "batch": -1, "seen": 0, "loss": 0.0,
                           "pred": -1, "step": -1}) for i in ids]


def as_input(x) -> torch.Tensor:
    """A sample's input as float32. uint8 images (0-255) are scaled to [0, 1]."""
    x = torch.as_tensor(x)
    return x.float() / 255 if x.dtype == torch.uint8 else x.float()


class EpochSampler:
    """Yields batches as lists of (epoch, batch_idx, row_id). Epoch e always
    uses the same permutation, and iteration starts at `start`."""

    def __init__(self, ids: Sequence[int], batch_size: int, seed: int = 0):
        self.ids, self.batch_size, self.seed = sorted(ids), batch_size, seed
        self.epoch, self.start = 0, 0

    def __len__(self) -> int:
        return (len(self.ids) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        perm = torch.randperm(len(self.ids), generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        order = [self.ids[i] for i in perm]
        for b in range(self.start, len(self)):
            yield [(self.epoch, b, rid) for rid in order[b * self.batch_size:(b + 1) * self.batch_size]]


def collate(batch):
    return batch  # DBDataset.__getitems__ already returns a whole batch


class DBDataset(Dataset):
    """Per batch: one FetchAndMark on the run's table (marks the rows QUEUED
    with the fetching worker and batch index, atomically), then the samples
    are loaded from the user's Dataset."""

    def __init__(self, client, table: str, dataset):
        self.client, self.table, self.dataset = client, table, dataset

    def __getitems__(self, items):
        epoch, batch = items[0][0], items[0][1]
        info = get_worker_info()
        worker = info.id if info else -1
        marked = self.client.fetch_and_mark(self.table, [rid for _, _, rid in items],
                                            {"status": QUEUED, "epoch": epoch, "worker": worker, "batch": batch},
                                            columns=["status"], tag=f"fetch:{self.table}:w{worker}")
        ids = [r.id for r in marked]
        samples = [self.dataset[rid - 1] for rid in ids]
        return {"ids": ids, "x": torch.stack([as_input(x) for x, _ in samples]),
                "y": torch.tensor([int(y) for _, y in samples]), "epoch": epoch, "batch": batch, "worker": worker}
