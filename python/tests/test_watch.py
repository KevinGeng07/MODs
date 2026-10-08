import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

import mods
from mods.session import preview
from mods.watch import prepare

from conftest import Tiny, datasets


def test_watch_leaves_the_class_usable():
    train, val = datasets()

    @mods.watch(train=train, val=val, classes=["low", "high"])
    class Net(Tiny):
        pass

    assert Net.__mods__["classes"] == ["low", "high"] and callable(Net.serve)
    assert Net()(torch.zeros(3, 4, 4)).shape == (3, 2)  # still an ordinary module


def test_prepare_resolves_factories_and_splits_val():
    train, _ = datasets()
    s = prepare(Tiny, train=lambda: train)
    assert (len(s.train), len(s.val)) == (30, 10)  # 10 held out of 40
    assert s.name == "Tiny" and s.classes == ["0", "1"] and s.graph["nodes"]
    model, opt = s.make_model()
    assert isinstance(model, Tiny) and isinstance(opt, torch.optim.Adam)
    assert s.make_model()[0] is not model  # a fresh model every run


@pytest.mark.parametrize("ds, classes, message", [
    (TensorDataset(torch.zeros(4, 4, 4), torch.tensor([0.5] * 4)), None, "labels must be integers"),
    (TensorDataset(torch.zeros(4, 4, 4), torch.tensor([7] * 4)), None, "labels must be 0..1"),
    (datasets()[0], ["a", "b", "c"], "outputs 2 scores but 3 classes"),
    (TensorDataset(torch.zeros(0, 4, 4), torch.zeros(0).long()), None, "empty"),
])
def test_prepare_rejects_bad_setups(ds, classes, message):
    with pytest.raises(ValueError, match=message):
        prepare(Tiny, train=ds, val=datasets()[1], classes=classes)


def test_prepare_rejects_wrong_output_shape():
    class Flat(nn.Module):
        def forward(self, x):
            return x.flatten()

    with pytest.raises(ValueError, match=r"must return \[batch, classes\]"):
        prepare(Flat, train=datasets()[0], val=datasets()[1])


def test_previews_only_for_images():
    assert len(preview(torch.rand(1, 8, 8))) == 8                    # grayscale, channel squeezed
    assert len(preview(torch.rand(3, 5, 5))) == 3                    # RGB
    assert preview(torch.rand(16)) is None                           # feature vector
    assert preview(torch.full((1, 2, 2), 255, dtype=torch.uint8))[0][0] == 1.0  # uint8 scaled
