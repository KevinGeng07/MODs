"""Test mods_ on your own model: a small MLP learning MNIST, served with @mods.watch.

    .venv/bin/python examples/mnist_mlp.py      # then open http://127.0.0.1:8000

The first run downloads MNIST (about 11 MB) to ./data/mnist.
"""

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import mods

CLASSES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


# Factories (lambdas) rather than datasets: DataLoader workers re-run this file's
# top-level code, and a factory is only called once, inside serve().
@mods.watch(
    train=lambda: MNIST("data/mnist", train=True, download=True, transform=ToTensor()),
    val=lambda: MNIST("data/mnist", train=False, download=True, transform=ToTensor()),
    optimizer=lambda m: torch.optim.Adam(m.parameters(), lr=1e-3),
    classes=CLASSES,
)
class MnistMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()          # 1x28x28 -> 784
        self.hidden = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        return self.out(self.relu(self.hidden(self.flatten(x))))


if __name__ == "__main__":  # required: DataLoader workers re-import this file
    MnistMLP.serve(port=8000)
