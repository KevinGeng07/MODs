"""A training session is one run: a fresh model, its own copy of the sample
table (`run<N>`), and a DataLoader feeding from it. Every public method takes
the session's lock, so HTTP handlers and the Run loop never race.

The module-level history functions read a run's logged batches."""

from __future__ import annotations

import random
import threading
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (MAX_BATCH, RUN_ROWS_SCHEMA, DBDataset, EpochSampler, as_input, collate, fresh_run_rows,
                   run_table, sample_row_id, step_id)
from .db import CONSUMED, QUEUED, incr, patch, put

VAL_SET_SIZE = 10


class Session:
    def __init__(self, client, model, optimizer, batch_size=16, num_workers=2, prefetch_factor=2, seed=0):
        """Start a new run. Callers must not create two Sessions concurrently
        (the run number is allocated from a shared counter)."""
        if not 1 <= batch_size <= MAX_BATCH:
            raise ValueError(f"batch_size must be 1..{MAX_BATCH}")
        self.client, self.model, self.optimizer = client, model, optimizer
        self.batch_size, self.num_workers, self.prefetch_factor = batch_size, num_workers, prefetch_factor
        self.lock = threading.RLock()
        self.running, self.delay_ms, self.last_step = False, 300, None
        self.closed = False
        self.epoch = self.step_no = self.batch = self.evals = 0

        # One atomic commit: allocate the run, register it, and give it a fresh
        # copy of the dataset with clean per-row state.
        self.run = client.get_rows("meta", [1])[0]["runs"] + 1
        self.table = run_table(self.run)
        client.create_table(self.table, RUN_ROWS_SCHEMA)
        train_ids = [r.id for r in client.scan("train", columns=["y"])]
        client.apply([patch("meta", 1, {"runs": self.run}),
                      put("runs", self.run, {"batch_size": batch_size, "workers": num_workers,
                                             "prefetch": prefetch_factor, "started_ms": int(time.time() * 1000),
                                             "epoch": 0, "step": 0, "batch": 0}),
                      *fresh_run_rows(self.table, train_ids)], tag=f"run:{self.run}")

        self.val_ids = [r.id for r in client.scan("val", columns=["y"])]
        self._shuffle_rng = random.Random(seed + self.run)
        self.shuffle()
        self.sampler = EpochSampler(train_ids, batch_size, seed + self.run)
        self.loader, self.it = None, None
        self._start_loader()
        threading.Thread(target=self._run_loop, daemon=True).start()

    # ---- loader
    def _start_loader(self):
        self.sampler.epoch, self.sampler.start = self.epoch, self.batch
        if self.loader is None:
            kw = dict(num_workers=self.num_workers, prefetch_factor=self.prefetch_factor, persistent_workers=True,
                      multiprocessing_context="spawn") if self.num_workers else {}
            self.loader = DataLoader(DBDataset(self.client, self.table), batch_sampler=self.sampler,
                                     collate_fn=collate, **kw)
        with torch.random.fork_rng(devices=[]):  # iter() draws from the global RNG
            self.it = iter(self.loader)  # workers start prefetching (and marking rows QUEUED) now

    def close(self):
        """End the run: stop the Run loop, shut down the DataLoader workers and
        drop the run's state table. Its training history is kept."""
        with self.lock:
            if self.closed:
                return
            self.closed, self.running = True, False
            if self.it is not None and hasattr(self.it, "_shutdown_workers"):
                self.it._shutdown_workers()
            self.loader = self.it = None
            try:
                self.client.drop_table(self.table)
            except Exception:  # noqa: BLE001 - the database may already be shutting down
                pass

    def _check_open(self):
        if self.closed:
            raise RuntimeError(f"run {self.run} has ended")

    # ---- commands
    def step(self) -> dict:
        with self.lock:
            self._check_open()
            try:
                b = next(self.it)
            except StopIteration:  # epoch done: next epoch, same workers
                self.epoch, self.batch = self.epoch + 1, 0
                self._start_loader()
                b = next(self.it)
            self.model.train()
            logits = self.model(b["x"])
            losses = F.cross_entropy(logits, b["y"], reduction="none")
            self.optimizer.zero_grad()
            losses.mean().backward()
            self.optimizer.step()
            self.step_no, self.batch = self.step_no + 1, b["batch"] + 1
            sid = step_id(self.run, self.step_no)
            preds, losses, labels = logits.argmax(1).tolist(), losses.detach().tolist(), b["y"].tolist()
            loss, acc = sum(losses) / len(losses), sum(p == y for p, y in zip(preds, labels)) / len(preds)

            # One atomic commit: the sample rows, the batch log row, its samples, and the run position.
            ops = []
            for k, (rid, y, p, l) in enumerate(zip(b["ids"], labels, preds, losses)):
                ops.append(patch(self.table, rid, {"status": CONSUMED, "loss": l, "pred": p, "step": self.step_no}))
                ops.append(incr(self.table, rid, {"seen": 1}))
                ops.append(put("batch_samples", sample_row_id(sid, k),
                               {"run": self.run, "step": self.step_no, "sample": rid, "label": y, "pred": p,
                                "loss": l, "correct": int(p == y)}))
            ops.append(put("steps", sid, {"run": self.run, "step": self.step_no, "epoch": self.epoch,
                                          "batch": b["batch"], "worker": b["worker"], "n": len(preds),
                                          "loss": loss, "acc": acc}))
            ops.append(patch("runs", self.run, {"epoch": self.epoch, "step": self.step_no, "batch": self.batch}))
            lsn = self.client.apply(ops, tag=f"consume:{self.table}")
            self.last_step = {"step": self.step_no, "seq": sid, "batch": b["batch"], "worker": b["worker"],
                              "ids": b["ids"], "loss": loss, "acc": acc, "lsn": lsn}
            return self.last_step

    def _run_loop(self):
        while not self.closed:
            if self.running:
                try:
                    self.step()
                except Exception as e:  # noqa: BLE001
                    if not self.closed:
                        print(f"run {self.run}: step failed:", e)
                    self.running = False
                time.sleep(self.delay_ms / 1000)
            else:
                time.sleep(0.05)

    # ---- validation
    def evaluate(self) -> dict:
        """Run the 10 shown validation samples through the current model. Only
        the accuracy is stored (one row in `evals`); per-sample class
        distributions are returned to the caller and not persisted."""
        with self.lock:
            self._check_open()
            rows = {r.id: r for r in self.client.get_rows("val", self.val_set, ["x", "y"])}
            ids = [i for i in self.val_set if i in rows]
            x = torch.stack([as_input(rows[i]["x"]) for i in ids])
            y = torch.tensor([rows[i]["y"] for i in ids])
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                probs = torch.softmax(self.model(x), 1)
            self.model.train(was_training)
            preds = probs.argmax(1)
            acc = int((preds == y).sum()) / len(ids)
            self.evals += 1
            self.client.apply([put("evals", self.run * 100_000 + self.evals,
                                   {"run": self.run, "step": self.step_no, "epoch": self.epoch, "acc": acc,
                                    "n": len(ids)})], tag="eval")
            return {"acc": acc, "step": self.step_no, "epoch": self.epoch,
                    "samples": [{"val_id": i, "target": int(y[k]), "pred": int(preds[k]), "probs": probs[k].tolist()}
                                for k, i in enumerate(ids)]}

    def shuffle(self) -> list[int]:
        """Pick a new random set of validation samples (one atomic write)."""
        with self.lock:
            self._check_open()
            self.val_set = self._shuffle_rng.sample(self.val_ids, min(VAL_SET_SIZE, len(self.val_ids)))
            self.client.apply([put("valset", self.run * 100 + k, {"val_id": v}) for k, v in enumerate(self.val_set)],
                              tag="shuffle")
            return self.val_set

    def val_samples(self) -> list[dict]:
        rows = {r.id: r for r in self.client.get_rows("val", self.val_set, ["x", "y"])}
        return [{"val_id": i, "x": rows[i]["x"].squeeze().tolist(), "y": rows[i]["y"]} for i in self.val_set if i in rows]

    # ---- status
    def queue(self) -> list[dict]:
        """Batches currently prefetched, per worker, read from the database."""
        rows = self.client.scan(self.table, columns=["worker", "batch"], status=QUEUED, epoch=self.epoch)
        workers = {w: {} for w in (range(self.num_workers) if self.num_workers else [-1])}
        for r in rows:
            workers.setdefault(r["worker"], {}).setdefault(r["batch"], []).append(r.id)
        return [{"worker": w, "batches": [{"batch": b, "ids": ids} for b, ids in sorted(bs.items())]}
                for w, bs in sorted(workers.items())]

    def status(self) -> dict:
        evals = self.client.get_rows("evals", [self.run * 100_000 + n for n in range(1, self.evals + 1)][-12:], ["acc"])
        return {"running": self.running, "delay_ms": self.delay_ms, "run": self.run, "epoch": self.epoch,
                "step": self.step_no, "batch": self.batch, "batches_per_epoch": len(self.sampler),
                "batch_size": self.batch_size, "num_workers": self.num_workers,
                "prefetch_factor": self.prefetch_factor, "val_set": self.val_set,
                "accuracy_history": [r["acc"] for r in evals], "last_step": self.last_step,
                "queue": self.queue(), "db": self.client.stats()}


# ---- history

def batch_log(client, run: int, limit: int = 50) -> list[dict]:
    """A run's most recent batches, newest first (looked up by id range, no scan)."""
    rows = client.get_rows("runs", [run], ["step"])
    if not rows:
        return []
    last = rows[0]["step"]
    ids = [step_id(run, s) for s in range(max(1, last - limit + 1), last + 1)]
    return [{"seq": r.id, "lsn": r.version, **r} for r in reversed(client.get_rows("steps", ids))]


def batch_samples(client, sid: int) -> dict:
    """The samples of one logged batch: index, label, prediction, loss."""
    step = client.get_rows("steps", [sid])
    if not step:
        raise KeyError(f"no batch {sid}")
    rows = client.get_rows("batch_samples", [sample_row_id(sid, k) for k in range(step[0]["n"])])
    return {"seq": sid, **step[0], "samples": [dict(r) for r in rows]}
