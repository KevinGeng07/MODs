import pytest
import torch
from torch import nn

from mods.data import load_samples, setup_tables
from mods.db import Client
from mods.launch import DBProcess


@pytest.fixture
def client(tmp_path):
    """A real modsdb seeded with 40 train / 20 val samples."""
    with DBProcess(tmp_path / "data") as p:
        c = Client(p.addr)
        setup_tables(c)
        g = torch.Generator().manual_seed(0)
        xs = torch.randn(60, 4, 4, generator=g)
        ys = (xs.sum((1, 2)) > 0).long().tolist()
        load_samples(c, "train", xs[:40], ys[:40])
        load_samples(c, "val", xs[40:], ys[40:])
        yield c


def tiny_model():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Flatten(), nn.Linear(16, 12), nn.ReLU(), nn.Dropout(0.1), nn.Linear(12, 2))
    return m, torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
