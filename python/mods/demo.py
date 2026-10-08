"""The mods_ demo: a small CNN learning MNIST, written as an ordinary @mods.watch user.
Run it with `python -m mods demo`. The first start downloads MNIST (about 11 MB) to ./data/mnist."""

import torch
from torch import nn
from torch.utils.data import TensorDataset

from .watch import watch

DATA_DIR = "data/mnist"


def mnist(train: bool) -> TensorDataset:
    """MNIST as uint8 [1, 28, 28] images (mods_ scales uint8 inputs to [0, 1])."""
    from torchvision.datasets import MNIST
    ds = MNIST(DATA_DIR, train=train, download=True)
    return TensorDataset(ds.data.unsqueeze(1), ds.targets)


@watch(
    train=lambda: mnist(train=True),
    val=lambda: mnist(train=False),
    optimizer=lambda m: torch.optim.SGD(m.parameters(), lr=0.02, momentum=0.9),
)
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
