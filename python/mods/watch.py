"""The public API: decorate a PyTorch classifier with @watch, then serve() it to
start, pause and step its training from the dashboard at http://127.0.0.1:8000.

    @mods.watch(train=lambda: make_train_set(), optimizer=lambda m: SGD(m.parameters(), lr=0.02))
    class Net(nn.Module): ...

    if __name__ == "__main__":
        Net.serve()
"""

from __future__ import annotations

import operator
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import random_split

from .data import as_input, clear_history, setup_tables
from .graph import model_graph

DB_FORMAT = "6"  # bump when the table layout changes; older databases are replaced
VAL_FRACTION = 0.1


@dataclass
class Setup:
    """Everything the server needs about the user's model, checked and resolved."""
    name: str
    make_model: Callable  # () -> (model, optimizer), a fresh pair every run
    train: object
    val: object
    classes: list[str]
    graph: dict


def watch(train, val=None, optimizer=None, classes=None):
    """Class decorator. `train` / `val` are Datasets of (x, integer label), or
    zero-argument functions returning one (recommended: DataLoader workers
    re-run the script's top-level code, so a factory avoids building the
    data in every worker). Without `val`, 10% of `train` is held out.
    `optimizer(model)` returns the optimizer (default: Adam, lr 1e-3).
    `classes` names the labels (default "0", "1", ...).

    The class itself is unchanged; it gains `Net.serve(port=8000, data_dir=".mods")`."""
    def decorate(cls):
        cls.__mods__ = dict(train=train, val=val, optimizer=optimizer, classes=classes)
        cls.serve = classmethod(serve)
        return cls
    return decorate


def _resolve(ds):
    """A Dataset, or a zero-argument function returning one."""
    return ds() if callable(ds) and not hasattr(ds, "__getitem__") else ds


def _label(y) -> int:
    try:
        return operator.index(y)
    except TypeError:
        raise ValueError(f"labels must be integers, got {y!r}") from None


def prepare(model_cls, train, val=None, optimizer=None, classes=None) -> Setup:
    """Resolve the datasets and check that the model and data fit together."""
    train, val = _resolve(train), _resolve(val)
    if train is None or len(train) == 0:
        raise ValueError("the training dataset is empty")
    if val is None:
        n_val = min(len(train) - 1, max(10, int(len(train) * VAL_FRACTION)))
        if n_val < 1:
            raise ValueError("pass val=..., the training dataset is too small to split")
        train, val = random_split(train, [len(train) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    if len(val) == 0:
        raise ValueError("the validation dataset is empty")

    sample = train[0]
    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
        raise ValueError("the dataset must yield (input, label) pairs")
    try:
        x = as_input(sample[0])
    except (TypeError, ValueError, RuntimeError):
        raise ValueError(f"inputs must be tensors (or arrays), got {type(sample[0]).__name__}; "
                         "for images, use transform=ToTensor()") from None
    y = _label(sample[1])

    model = model_cls()
    model.eval()
    with torch.no_grad():
        out = model(x.unsqueeze(0))
    if out.ndim != 2 or out.shape[0] != 1:
        raise ValueError(f"{model_cls.__name__} must return [batch, classes] scores, got shape {list(out.shape)}")
    n = out.shape[1]
    if classes is not None and len(classes) != n:
        raise ValueError(f"{model_cls.__name__} outputs {n} scores but {len(classes)} classes were named")
    if not 0 <= y < n:
        raise ValueError(f"{model_cls.__name__} outputs {n} scores, so labels must be 0..{n - 1}; got {y}")

    make_optimizer = optimizer or (lambda m: torch.optim.Adam(m.parameters(), lr=1e-3))

    def make_model():
        m = model_cls()
        return m, make_optimizer(m)

    return Setup(name=model_cls.__name__, make_model=make_model, train=train, val=val,
                 classes=[str(c) for c in classes] if classes is not None else [str(i) for i in range(n)],
                 graph=model_graph(model_cls(), x.unsqueeze(0)))


def build_app(client, model_cls, **settings):
    """The FastAPI app for `model_cls`, on a running modsdb. Settings default to
    the ones given to @watch."""
    from .server.app import create_app
    setup_tables(client)
    clear_history(client)
    if not settings and not hasattr(model_cls, "__mods__"):
        raise ValueError(f"decorate {model_cls.__name__} with @mods.watch(...) or pass train=...")
    return create_app(client, prepare(model_cls, **(settings or model_cls.__mods__)))


def serve(model_cls, port: int = 8000, host: str = "127.0.0.1", data_dir: str = ".mods", **settings) -> None:
    """Start modsdb and the dashboard for `model_cls`; blocks until Ctrl+C."""
    import uvicorn

    from .db import Client
    from .launch import DBProcess

    data_dir = Path(data_dir)
    fmt = data_dir / "FORMAT"
    if data_dir.exists() and (not fmt.exists() or fmt.read_text().strip() != DB_FORMAT):
        shutil.rmtree(data_dir)  # an older table layout; it only ever held one run's history
    data_dir.mkdir(parents=True, exist_ok=True)
    db = DBProcess(data_dir)
    fmt.write_text(DB_FORMAT)
    try:
        app = build_app(Client(db.addr), model_cls, **settings)
        print(f"mods_ UI for {model_cls.__name__}: http://{host}:{port}/", flush=True)
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        db.stop()
