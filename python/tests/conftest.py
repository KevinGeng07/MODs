import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from mods.data import setup_tables
from mods.db import Client
from mods.launch import DBProcess


@pytest.fixture
def client(tmp_path):
    """A real modsdb with the history tables created."""
    with DBProcess(tmp_path / "data") as p:
        c = Client(p.addr)
        setup_tables(c)
        yield c


def datasets():
    """40 train / 20 val samples: 4x4 inputs, label = whether they sum above 0."""
    g = torch.Generator().manual_seed(0)
    xs = torch.randn(60, 4, 4, generator=g)
    ys = (xs.sum((1, 2)) > 0).long()
    return TensorDataset(xs[:40], ys[:40]), TensorDataset(xs[40:], ys[40:])


class Tiny(nn.Sequential):
    def __init__(self):
        torch.manual_seed(0)
        super().__init__(nn.Flatten(), nn.Linear(16, 12), nn.ReLU(), nn.Dropout(0.1), nn.Linear(12, 2))


def tiny_model():
    m = Tiny()
    return m, torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
