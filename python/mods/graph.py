"""The model's layers in order, for the Model Layers panel: traced with torch.fx
(with output shapes), or the list of submodules when the model can't be traced."""

from __future__ import annotations

import torch
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp


def _label(gm: fx.GraphModule, n: fx.Node) -> str:
    if n.op == "placeholder":
        return f"input: {n.target}"
    if n.op == "output":
        return "output"
    if n.op == "call_module":
        return f"{n.target}: {type(gm.get_submodule(n.target)).__name__}"
    if n.op == "call_function":
        return getattr(n.target, "__name__", str(n.target))
    return f".{n.target}()"  # call_method


def model_graph(model: torch.nn.Module, example: torch.Tensor) -> dict:
    """{"nodes": [{"op", "label", "shape"}], "total_params"}, nodes in execution order."""
    total = sum(p.numel() for p in model.parameters())
    try:
        gm = fx.symbolic_trace(model)
        model.eval()
        with torch.no_grad():
            ShapeProp(gm).propagate(example)
        nodes = []
        for n in gm.graph.nodes:
            meta = n.meta.get("tensor_meta")
            nodes.append({"op": n.op, "label": _label(gm, n), "shape": list(meta.shape) if hasattr(meta, "shape") else None})
        return {"nodes": nodes, "total_params": total}
    except Exception:  # noqa: BLE001 - untraceable models (data-dependent control flow) fall back
        return {"nodes": [{"op": "module", "label": f"{name}: {type(m).__name__}", "shape": None}
                          for name, m in model.named_modules() if name and not list(m.children())],
                "total_params": total}
