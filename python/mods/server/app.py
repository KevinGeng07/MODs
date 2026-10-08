"""FastAPI app. There is only ever one run: every page load starts it
(POST /api/runs) with a fresh model, ending and forgetting the previous one.
The tab then drives it through /api/runs/{run}/...; an older tab gets 410.

The run ends when its tab closes (beacon) or goes silent for IDLE_S seconds.
Nothing is cached: every response is Cache-Control: no-store."""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..data import MAX_BATCH, clear_history
from ..db import DBError
from ..session import Session, batch_log, batch_samples

INDEX = Path(__file__).parent / "static" / "index.html"
IDLE_S = 120  # background tabs may only poll once a minute


class NewRun(BaseModel):
    batch_size: int = Field(64, ge=1, le=MAX_BATCH)
    workers: int = Field(2, ge=0, le=8)
    prefetch: int = Field(2, ge=1, le=8)


class Runs:
    """Holds the one live run, if any."""

    def __init__(self, client, setup):
        self.client, self.setup = client, setup
        self.live: Session | None = None
        self.seen = 0.0
        self.lock = threading.Lock()

    def start(self, cfg: NewRun) -> Session:
        with self.lock:
            self._end()
            clear_history(self.client)
            model, optimizer = self.setup.make_model()
            self.live = Session(self.client, model, optimizer, self.setup.train, self.setup.val,
                                batch_size=cfg.batch_size, num_workers=cfg.workers, prefetch_factor=cfg.prefetch)
            self.seen = time.monotonic()
            return self.live

    def get(self, run: int) -> Session:
        with self.lock:
            if self.live is None or self.live.run != run:
                raise HTTPException(410, "this run has ended; reload the page to start again")
            self.seen = time.monotonic()
            return self.live

    def _end(self) -> None:
        if self.live is not None:
            self.live.close()
            self.live = None

    def end(self, run: int | None = None) -> None:
        """End the live run (only if it is `run`, when given)."""
        with self.lock:
            if self.live is not None and run in (None, self.live.run):
                self._end()

    def reap(self) -> None:
        while True:
            time.sleep(5)
            with self.lock:
                if self.live is not None and time.monotonic() - self.seen > IDLE_S:
                    self._end()


def create_app(client, setup) -> FastAPI:
    """`setup` is a mods.watch.Setup: the model factory, datasets, classes and graph."""
    runs = Runs(client, setup)

    @asynccontextmanager
    async def lifespan(_app):
        threading.Thread(target=runs.reap, daemon=True).start()
        yield
        runs.end()

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
        return setup.graph

    @app.get("/api/info")
    def info():
        return {"name": setup.name, "classes": setup.classes}

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

    @app.get("/api/runs/{run}/batches")
    def batches(run: int, limit: int = 50):
        return batch_log(client, runs.get(run).run, min(limit, 1000))

    @app.get("/api/batches/{seq}/samples")
    def samples(seq: int):
        try:
            return batch_samples(client, seq)
        except KeyError as e:
            raise HTTPException(404, str(e))

    return app
