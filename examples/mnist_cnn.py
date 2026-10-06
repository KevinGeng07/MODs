"""mods_ demo: a small CNN learning MNIST, every image served from modsdb.
Open http://127.0.0.1:8000: every page load asks for batch size and workers and
starts the one run with a fresh model; the previous run and its history are dropped.

    .venv/bin/python examples/mnist_cnn.py                      # start the server
    .venv/bin/python examples/mnist_cnn.py --wipe               # delete the database first
    .venv/bin/python examples/mnist_cnn.py --init weights.pt    # every run starts from these weights

The first start downloads MNIST (about 11 MB) into ./data/mnist and loads it into modsdb.
"""

import argparse
import shutil
from pathlib import Path

import torch
from torch import nn

DB_FORMAT = "4"  # bump when the table layout changes; older databases are replaced


class MnistCNN(nn.Module):
    """conv -> relu -> max pool -> flatten -> linear -> linear (output)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3)   # 1x28x28 -> 4x26x26
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)               # -> 4x13x13
        self.flatten = nn.Flatten()                  # -> 676
        self.fc = nn.Linear(4 * 13 * 13, 6)          # -> 6
        self.out = nn.Linear(6, 10)                  # -> 10 classes

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv(x)))
        return self.out(self.fc(self.flatten(x)))


def main():
    import uvicorn
    from torchvision.datasets import MNIST

    from mods.data import load_samples, setup_tables
    from mods.db import Client
    from mods.graph import model_graph
    from mods.launch import DBProcess
    from mods.server.app import create_app

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "modsdb-data"))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--init", help="state_dict .pt file with precomputed weights")
    p.add_argument("--wipe", action="store_true", help="delete the database first")
    a = p.parse_args()

    data_dir = Path(a.data_dir)
    fmt = data_dir / "FORMAT"
    if a.wipe or (data_dir.exists() and (not fmt.exists() or fmt.read_text().strip() != DB_FORMAT)):
        if not a.wipe:
            print(f"{data_dir} uses an older table layout; starting a new database")
        shutil.rmtree(data_dir, ignore_errors=True)
    db = DBProcess(data_dir, port=7070)
    fmt.write_text(DB_FORMAT)
    try:
        client = Client(db.addr)
        setup_tables(client)
        if not client.get_rows("train", [1]):
            root = Path(__file__).resolve().parents[1] / "data" / "mnist"
            train, test = MNIST(root, train=True, download=True), MNIST(root, train=False, download=True)
            print("loading MNIST into modsdb (one time)...", flush=True)
            # uint8 images, shape [1, 28, 28]: 784 bytes per sample in the database.
            load_samples(client, "train", train.data.unsqueeze(1), train.targets.tolist())
            load_samples(client, "val", test.data.unsqueeze(1), test.targets.tolist())
            print(f"loaded {len(train)} train / {len(test)} validation images into modsdb", flush=True)

        def make_model():
            """A fresh model every time the page starts the run."""
            model = MnistCNN()
            if a.init:
                model.load_state_dict(torch.load(a.init))
            return model, torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

        app = create_app(client, make_model, model_graph(MnistCNN(), torch.zeros(1, 1, 28, 28)))
        print(f"mods_ UI: http://127.0.0.1:{a.port}/", flush=True)
        uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
    finally:
        db.stop()


if __name__ == "__main__":  # required: DataLoader workers are spawned and re-import this file
    main()
