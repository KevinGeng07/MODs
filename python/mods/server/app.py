"""FastAPI app. Every page load starts a new run (POST /api/runs) with its own
fresh model; the tab then drives that run through /api/runs/{run}/...

Runs end when their tab closes (beacon), goes silent for IDLE_S seconds, or
when more than MAX_LIVE runs are live. Nothing is cached: every response is
Cache-Control: no-store."""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..data import MAX_BATCH, run_table
from ..db import DBError
from ..session import Session, batch_log, batch_samples, list_runs

INDEX = Path(__file__).parent / "static" / "index.html"
IDLE_S = 120  # background tabs may only poll once a minute
MAX_LIVE = 4


class NewRun(BaseModel):
    batch_size: int = Field(64, ge=1, le=MAX_BATCH)
    workers: int = Field(2, ge=0, le=8)
    prefetch: int = Field(2, ge=1, le=8)


class Runs:
    """Live sessions keyed by run number."""

    def __init__(self, client, make_model: Callable):
        self.client, self.make_model = client, make_model
        self.live: dict[int, Session] = {}
        self.seen: dict[int, float] = {}
        self.lock = threading.Lock()
        # Runs left over from a previous server process are over: free their state tables.
        for r in list_runs(client):
            try:
                client.drop_table(run_table(r["run"]))
            except DBError:
                pass

    def start(self, cfg: NewRun) -> Session:
        with self.lock:  # run numbers come from a shared counter: create one at a time
            while len(self.live) >= MAX_LIVE:
                self._end(min(self.seen, key=self.seen.get))
            model, optimizer = self.make_model()
            s = Session(self.client, model, optimizer, batch_size=cfg.batch_size, num_workers=cfg.workers,
                        prefetch_factor=cfg.prefetch)
            self.live[s.run], self.seen[s.run] = s, time.monotonic()
            return s

    def get(self, run: int) -> Session:
        with self.lock:
            s = self.live.get(run)
            if s is None:
                raise HTTPException(410, f"run {run} has ended; reload the page to start a new run")
            self.seen[run] = time.monotonic()
            return s

    def _end(self, run: int) -> None:
        s = self.live.pop(run, None)
        self.seen.pop(run, None)
        if s is not None:
            threading.Thread(target=s.close, daemon=True).start()  # worker shutdown takes a moment

    def end(self, run: int) -> None:
        with self.lock:
            self._end(run)

    def reap(self) -> None:
        while True:
            time.sleep(5)
            with self.lock:
                for run in [r for r, t in self.seen.items() if time.monotonic() - t > IDLE_S]:
                    self._end(run)

    def end_all(self) -> None:
        with self.lock:
            sessions = list(self.live.values())
            self.live.clear()
            self.seen.clear()
        for s in sessions:
            s.close()


def create_app(client, make_model: Callable, graph: dict) -> FastAPI:
    """make_model() must return a fresh (model, optimizer) pair each call."""
    runs = Runs(client, make_model)

    @asynccontextmanager
    async def lifespan(_app):
        threading.Thread(target=runs.reap, daemon=True).start()
        yield
        runs.end_all()

    app = FastAPI(title="MODs", lifespan=lifespan)

    @app.middleware("http")
    async def no_store(request, call_next):
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def guard(fn, *args):
        try:
            return fn(*args)
        except (DBError, ValueError) as e:
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            raise HTTPException(410, str(e))

    @app.get("/")
    def index():
        return FileResponse(INDEX)

    @app.get("/api/graph")
    def get_graph():
        return graph

    @app.get("/api/runs")
    def all_runs():
        return list_runs(client, set(runs.live))

    @app.post("/api/runs")
    def new_run(cfg: NewRun):
        s = guard(runs.start, cfg)
        return {"run": s.run}

    @app.post("/api/runs/{run}/close")
    def close_run(run: int):
        runs.end(run)
        return {"closed": run}

    @app.get("/api/runs/{run}/status")
    def status(run: int):
        return guard(runs.get(run).status)

    @app.get("/api/runs/{run}/val")
    def val_samples(run: int):
        return guard(runs.get(run).val_samples)

    @app.post("/api/runs/{run}/step")
    def step(run: int):
        s = runs.get(run)
        s.running = False
        return guard(s.step)

    @app.post("/api/runs/{run}/run")
    def run_training(run: int, delay_ms: int = 300):
        s = runs.get(run)
        s.delay_ms, s.running = max(0, delay_ms), True
        return {"running": True}

    @app.post("/api/runs/{run}/pause")
    def pause(run: int):
        runs.get(run).running = False
        return {"running": False}

    @app.post("/api/runs/{run}/evaluate")
    def evaluate(run: int):
        return guard(runs.get(run).evaluate)

    @app.post("/api/runs/{run}/shuffle")
    def shuffle(run: int):
        return {"val_set": guard(runs.get(run).shuffle)}

    # History works for any run, live or ended.
    @app.get("/api/runs/{run}/batches")
    def batches(run: int, limit: int = 50):
        return batch_log(client, run, min(limit, 1000))

    @app.get("/api/batches/{seq}/samples")
    def samples(seq: int):
        try:
            return batch_samples(client, seq)
        except KeyError as e:
            raise HTTPException(404, str(e))

    app.state.runs = runs
    return app
