"""Tables, a deterministic batch sampler, and a Dataset that marks rows QUEUED
in modsdb as DataLoader workers prefetch them.

Layout: `train` / `val` hold the dataset (x, y). Each run (one browser tab)
gets its own table `run<N>` with one row per training sample holding only that
run's training state, so concurrent runs never touch each other's rows. The
pixels are read from `train`; a run's table is dropped when the run ends."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset, get_worker_info

from .db import IDLE, QUEUED, put

DATA_SCHEMA = {"x": "tensor", "y": "int64"}
RUN_ROWS_SCHEMA = {"status": "int64", "epoch": "int64", "worker": "int64", "batch": "int64", "seen": "int64",
                   "loss": "float64", "pred": "int64", "step": "int64"}
META_SCHEMA = {"runs": "int64"}  # row 1: number of runs ever started
# One row per run (id = run): settings plus its live position.
RUNS_SCHEMA = {"batch_size": "int64", "workers": "int64", "prefetch": "int64", "started_ms": "int64",
               "epoch": "int64", "step": "int64", "batch": "int64"}
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


def setup_tables(client) -> None:
    """Create any missing shared tables (CreateTable is idempotent)."""
    for name, schema in (("train", DATA_SCHEMA), ("val", DATA_SCHEMA), ("meta", META_SCHEMA), ("runs", RUNS_SCHEMA),
                         ("steps", STEPS_SCHEMA), ("batch_samples", BATCH_SAMPLES_SCHEMA),
                         ("valset", VALSET_SCHEMA), ("evals", EVALS_SCHEMA)):
        client.create_table(name, schema)
    if not client.get_rows("meta", [1]):
        client.apply([put("meta", 1, {"runs": 0})])


def load_samples(client, table: str, xs: torch.Tensor, ys: Sequence[int]) -> None:
    rows = [put(table, i + 1, {"x": xs[i], "y": int(ys[i])}) for i in range(len(ys))]
    for i in range(0, len(rows), 500):
        client.apply(rows[i:i + 500], tag="load")


def fresh_run_rows(table: str, ids) -> list:
    """Puts that give a run one clean training-state row per sample."""
    return [put(table, i, {"status": IDLE, "epoch": -1, "worker": -1, "batch": -1, "seen": 0, "loss": 0.0,
                           "pred": -1, "step": -1}) for i in ids]


def as_input(x: torch.Tensor) -> torch.Tensor:
    """Stored images are uint8 (0-255); the model sees float32 in [0, 1]."""
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
    with the fetching worker and batch index, atomically), then one read of
    the pixels and labels from the shared `train` table."""

    def __init__(self, client, table: str, data_table: str = "train"):
        self.client, self.table, self.data_table = client, table, data_table

    def __getitems__(self, items):
        epoch, batch = items[0][0], items[0][1]
        info = get_worker_info()
        worker = info.id if info else -1
        marked = self.client.fetch_and_mark(self.table, [rid for _, _, rid in items],
                                            {"status": QUEUED, "epoch": epoch, "worker": worker, "batch": batch},
                                            columns=["status"], tag=f"fetch:{self.table}:w{worker}")
        rows = self.client.get_rows(self.data_table, [r.id for r in marked], ["x", "y"])
        return {"ids": [r.id for r in rows], "x": torch.stack([as_input(r["x"]) for r in rows]),
                "y": torch.tensor([r["y"] for r in rows]), "epoch": epoch, "batch": batch, "worker": worker}
