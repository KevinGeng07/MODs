# mods_: (m)odel (o)bservability for (d)ata

mods_ lets you watch your own PyTorch classifier train, and start, pause or step its training from a
dashboard on localhost. Decorate the model class with `@mods.watch`, call `.serve()`, and open
http://127.0.0.1:8000. As the DataLoader's workers load batches from your Dataset, they mark those
samples as queued in **modsdb**, a small database written in Go. Each training step then records
which samples it used, their predictions, the loss and the accuracy. The page shows all of it live,
read straight from the database.

![mods_ dashboard](assets/dashboard.png)

## Use it on your model

```python
import torch
from torch import nn
import mods

@mods.watch(
    train=lambda: make_train_dataset(),     # a Dataset of (input, integer label), or a function returning one
    val=lambda: make_val_dataset(),         # optional: defaults to 10% held out of train
    optimizer=lambda m: torch.optim.SGD(m.parameters(), lr=0.02),  # optional: defaults to Adam, lr 1e-3
    classes=["cat", "dog"],                 # optional label names
)
class Net(nn.Module):
    ...                                     # must return [batch, classes] scores

if __name__ == "__main__":                  # required: DataLoader workers re-run this file
    Net.serve()                             # http://127.0.0.1:8000; Ctrl+C to stop
```

- The class is unchanged: `Net()` still builds an ordinary module. Each page load builds a fresh
  `Net()` and optimizer, and trains it with cross-entropy loss.
- Pass datasets as functions (`lambda: ...`) so they are built once, not again in every worker.
- Your data never enters the database. It only stores each sample's state, the step history and
  evaluations. uint8 inputs (0–255) are scaled to [0, 1]; other inputs are used as given.
- Validation samples show as pictures when inputs are images (H×W, 1×H×W or 3×H×W).
- `Net.serve(port=8000, host="127.0.0.1", data_dir=".mods")`: `data_dir` is where modsdb keeps
  its log and snapshots.
- Setup problems (non-integer labels, a model output that doesn't match the labels) are reported
  before the server starts.

## Demo

The demo trains a tiny CNN on MNIST: one 3×3 convolution, ReLU, max pooling, then two linear
layers. Its code (`python/mods/demo.py`) uses `@mods.watch` like any other model.

```bash
./mods start    # sets everything up on first run, then serves the demo at http://127.0.0.1:8000
./mods stop     # also: status, restart, logs, open
# or, without the script:
.venv/bin/python -m mods demo --port 8000
```

You need Python 3.10+ with PyTorch, and Go 1.22+ (to build modsdb on first use). The first start
downloads MNIST (about 11 MB) to `data/mnist`. Everything stays on your machine.

## What you can do

- Pick a batch size and number of workers each time you open the page. Opening the page starts
  the one run with a fresh model; the previous run and its history are dropped.
- Step through training one batch at a time, or let it run.
- See which batches each worker has prepared and which one is next.
- Check the model on 10 random validation samples, and see how confident it is for each class.
- Browse the last 50 training steps and expand any step to see its samples.

## How modsdb works

modsdb keeps its tables in memory and makes every change durable before acknowledging it:

- **Write-ahead log:** every change is appended to the log first, and many concurrent writes share
  one disk sync (group commit).
- **Snapshots:** the full state is saved periodically, and older log segments are deleted.
- **Recovery:** after a crash, it loads the latest snapshot and replays the rest of the log.

Each training step is written as one atomic batch, so the history and the sample states never
disagree.

## Testing

**Automated:**

```bash
(cd db && go test ./...)    # log, snapshots, crash recovery, concurrent writers
.venv/bin/python -m pytest  # the decorator and its checks, training, worker queues, the web API
```

**By hand, on a model that isn't the demo:** `examples/mnist_mlp.py` is a small MLP on MNIST,
written the way you would use mods_ on your own model (with class names).

1. `./mods stop` if the demo is running (both use port 8000).
2. `.venv/bin/python examples/mnist_mlp.py` and open http://127.0.0.1:8000.
3. Start training with the defaults. The header should read `model: MnistMLP`.
4. Press **Step** a few times (or `S`): the step count, loss and Training History update, and the
   worker panel shows the queued batches.
5. Press **Run** (`R`), then **Pause**. Move **Step Interval** to change the speed.
6. Press **Evaluate** (`E`): the 10 validation digits show `label seven`, `guess seven ✓` and so
   on, and clicking one shows a confidence bar per class name.
7. Expand a row in Training History: each sample shows `true → predicted` with class names.
8. Reload the page: a fresh run starts and the old history is gone.
9. Ctrl+C in the terminal stops the server and the database.

To try your own model, copy `examples/mnist_mlp.py` and swap in your model and datasets.

## Layout

- `db/`: modsdb (Go). The API is in `db/proto/modsdb/v1/modsdb.proto`.
- `python/mods/`: the package. `watch.py` is the public API (`@watch`, `serve`); `session.py`
  trains one run; `server/` is the web app; `demo.py` is the MNIST demo.
- `examples/mnist_mlp.py`: a user-style example to test with.
- `mods`: start/stop script for the demo.
