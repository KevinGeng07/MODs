"""Static model graph: torch.fx dataflow with output shapes, falling back to
the module hierarchy when the model cannot be symbolically traced."""

from __future__ import annotations

import torch
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp


def _param_count(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters(recurse=False))


def _label(gm: fx.GraphModule, n: fx.Node) -> str:
    if n.op == "placeholder":
        return f"input: {n.target}"
    if n.op == "output":
        return "output"
    if n.op == "call_module":
        return f"{n.target}: {type(gm.get_submodule(n.target)).__name__}"
    if n.op == "call_function":
        return getattr(n.target, "__name__", str(n.target))
    if n.op == "call_method":
        return f".{n.target}()"
    return f"{n.op}: {n.target}"


def model_graph(model: torch.nn.Module, example: torch.Tensor) -> dict:
    """Return {kind, nodes: [{id, op, label, shape, params}], edges: [[src, dst]], error?}."""
    try:
        gm = fx.symbolic_trace(model)
        was = model.training
        model.eval()
        try:
            with torch.no_grad():
                ShapeProp(gm).propagate(example)
        finally:
            model.train(was)
        nodes, edges = [], []
        for n in gm.graph.nodes:
            meta = n.meta.get("tensor_meta")
            shape = list(meta.shape) if hasattr(meta, "shape") else None
            params = _param_count(gm.get_submodule(n.target)) if n.op == "call_module" else 0
            nodes.append({"id": n.name, "op": n.op, "label": _label(gm, n), "shape": shape, "params": params})
            edges.extend([src.name, n.name] for src in n.all_input_nodes)
        return {"kind": "fx", "nodes": nodes, "edges": edges,
                "total_params": sum(p.numel() for p in model.parameters())}
    except Exception as e:  # noqa: BLE001 - any tracing failure falls back
        nodes = [{"id": name or "model", "op": "module", "label": f"{name or 'model'}: {type(m).__name__}",
                  "shape": None, "params": _param_count(m)} for name, m in model.named_modules()]
        edges = [[name.rpartition(".")[0] or "model", name] for name, _ in model.named_modules() if name]
        return {"kind": "modules", "nodes": nodes, "edges": edges, "error": f"{type(e).__name__}: {e}",
                "total_params": sum(p.numel() for p in model.parameters())}
