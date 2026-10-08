# mods_ as a package: `@mods.watch` — Design Spec

**Date:** 2026-10-07
**Status:** Approved in chat, implemented in the same session

## 1. Goal

Let anyone with a PyTorch classifier open the mods_ dashboard on their own model:

```python
import mods

@mods.watch(
    train=lambda: MNIST("data", train=True, download=True, transform=ToTensor()),
    val=lambda: MNIST("data", train=False, download=True, transform=ToTensor()),
    optimizer=lambda m: torch.optim.SGD(m.parameters(), lr=0.02),
)
class Net(nn.Module):
    ...

if __name__ == "__main__":
    Net.serve()              # http://127.0.0.1:8000: start / pause / step training from the page
```

The current MNIST dashboard becomes the demo: `python -m mods demo` (and `./mods start`) serves it,
built with the same decorator. There is no separate landing page.

### Scope decisions

- **Classification only.** A dataset yields `(x, label)` pairs with an integer label (or a 0-d
  tensor). The loss is always `cross_entropy`. Every dashboard panel keeps its meaning.
- **Installed from this repo** (`pip install -e .`). The modsdb binary is built from `db/` with Go on
  first use, as before. No prebuilt binaries, no PyPI release.
- **Data stays in the user's Dataset (option C).** modsdb no longer holds samples. It stores only
  what the dashboard shows: each sample's training state, the step history, evaluations and
  validation picks. No copy of the data, no load step.
- One run at a time; every page load starts a fresh run and clears the previous one (unchanged).

### Non-goals

Regression or custom loss functions, custom training steps, multiple models per server, a landing
page, PyPI packaging, persisting history across runs.

## 2. Public API (`mods/watch.py`)

`watch(train, val=None, optimizer=None, classes=None)` returns a class decorator.

| Argument | Meaning |
|---|---|
| `train` | A `Dataset` of `(x, label)`, or a zero-argument callable returning one. |
| `val` | Same, optional. Default: a fixed random 10% of `train` (at least 10 samples), removed from training. |
| `optimizer` | `callable(model) -> Optimizer`. Default: `Adam(model.parameters(), lr=1e-3)`. |
| `classes` | Optional list of label names. Default: `"0"`, `"1"`, ... |

- The decorator returns the same class, unchanged except for an added `serve` classmethod and a
  `__mods__` attribute holding the settings. `Net()` still builds an ordinary module.
- `Net.serve(port=8000, data_dir=".mods", host="127.0.0.1")` blocks until Ctrl+C.
- `mods.serve(model_cls, ..., port=..., data_dir=...)` is the function underneath (also used by
  tests via `build_app`).

**Factories are recommended.** On macOS, DataLoader workers start by re-running the script's
top-level code, so a dataset built at module level would be built again in every worker. A
factory is only called inside `serve()`.

## 3. Startup (`serve`)

1. Resolve the datasets (call factories). Split off `val` if not given.
2. Check, and stop with one clear `ValueError` before anything starts:
   - `train` is non-empty and `train[0]` is `(x, label)` with `x` convertible to a tensor and
     `label` an integer.
   - The model accepts `x.unsqueeze(0)` and returns `[1, C]` logits.
   - `C` equals `len(classes)` when `classes` is given; the first sample's label is in `0..C-1`.
3. Start modsdb in `data_dir` on a free port (`DBProcess`), create the tables, clear old history.
4. Build the static model graph from `x.unsqueeze(0)`.
5. Build the FastAPI app (`build_app`) and run uvicorn on `host:port`.
6. On exit: end the run (stops DataLoader workers), stop modsdb.

## 4. Data flow

- Sample `i` of the training Dataset is row `i + 1` of the run's state table `run<N>`. At run start
  each row gets `{status: IDLE, epoch, worker, batch, seen, loss, pred, step}` (unchanged schema).
- `DBDataset(client, table, dataset)`: for each batch, one `FetchAndMark` marks the rows QUEUED with
  worker and batch index (atomic, as before), then the samples are loaded from the user's Dataset
  and stacked. Labels come from the Dataset, not the database.
- Workers receive the Dataset by pickling (normal PyTorch behaviour with `num_workers > 0`).
- Validation: the run picks 10 indices of `val`. `evaluate` and the previews read `val[i]` directly.
  A preview image is sent only when the squeezed input is `H×W` or `3×H×W`; otherwise `x` is `null`
  and the page shows just the label.
- Tables: `meta`, `runs`, `steps`, `batch_samples`, `valset`, `evals`, and `run<N>`. The `train` and
  `val` tables are gone.

## 5. HTTP / UI changes

- New `GET /api/info` → `{"name": "<model class name>", "classes": [...]}`.
- Header shows the model name. "digit 6" becomes the class name (`"6"` by default). The confidence
  bars and the history detail use class names. "each digit 0–9" becomes "each class".
- Non-image validation samples render a text tile instead of a canvas.
- Layout and styling unchanged.

## 6. Demo

- `mods/demo.py`: the MNIST CNN from before, written as an ordinary `@mods.watch` user, with dataset
  factories (needs `torchvision`, the `examples` extra).
- `python -m mods demo [--port 8000] [--data-dir .mods]`.
- `./mods start|stop|...` runs `python -m mods demo`. The fixed modsdb port 7070 goes away.
- `examples/mnist_cnn.py` becomes a user-style test script: a different model (a small MLP) on
  MNIST, served with `@mods.watch`, to show the decorator on code that is not the demo.

## 7. Errors

- Startup problems (bad dataset, wrong output size, Go missing for the modsdb build) raise before
  the server starts.
- A failing training step returns HTTP 400 and its message shows next to the worker panel title.
- Ctrl+C shuts down the workers and modsdb cleanly.

## 8. Testing

- pytest, with a tiny in-memory `TensorDataset` (no data tables):
  - the decorator leaves the class usable and attaches `serve`;
  - startup checks reject a bad label, a wrong output size and a `classes` mismatch;
  - stepping marks rows QUEUED/CONSUMED in modsdb while samples load from the user's Dataset;
  - validation previews: image inputs get pixels, vector inputs get `null`;
  - the HTTP API end to end through `build_app` (start, step, evaluate, info, single-run rules).
- Manual: run the demo and the example in the browser.
