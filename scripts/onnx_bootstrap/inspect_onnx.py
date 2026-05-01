#!/usr/bin/env python3
"""
Inspect a CeresNet ONNX file to infer the CeresTrain config that produced it.

Run from WSL (where the Python deps live):
  wsl.exe -d Ubuntu -- bash -lc "source ~/cerestrain-env/bin/activate && python3 /mnt/c/Dev/Chess/CeresTrain/inspect_onnx.py /mnt/c/Dev/Chess/Networks/CeresNet/C1-640-34-I8.onnx"
"""
import sys
from collections import Counter

import onnx
from onnx import numpy_helper


def infer_architecture(model):
    """Infer ModelDim, NumLayers, NumHeads, FFNMult, etc. from tensor shapes + names."""
    init_map = {t.name: t for t in model.graph.initializer}

    shapes = {name: list(t.dims) for name, t in init_map.items()}

    # 1. Count transformer layers: look for tensor names like "transformer_layer.N.*"
    import re
    layer_re = re.compile(r'transformer_layer\.(\d+)')
    layer_indices = set()
    for name in shapes:
        m = layer_re.search(name)
        if m:
            layer_indices.add(int(m.group(1)))
    num_layers = max(layer_indices) + 1 if layer_indices else None

    # 2. ModelDim: find attention.qkvLN.scale or similar per-layer norm whose size = d_model
    # CeresNet uses RMSNorm whose shape is [d_model]
    dmodel_candidates = []
    for name, dims in shapes.items():
        if 'qkvLN.scale' in name or 'ln2.scale' in name:
            if len(dims) == 1:
                dmodel_candidates.append(dims[0])
    from collections import Counter
    dmodel = Counter(dmodel_candidates).most_common(1)[0][0] if dmodel_candidates else None

    # 3. NumHeads: look at sm1 (smolgen) weight shape [num_heads, d_model]
    num_heads = None
    for name, dims in shapes.items():
        if 'attention.sm1.weight' in name and len(dims) == 2:
            num_heads = dims[0]
            break

    # 4. FFN multiplier: look at mlp.linear1.weight shape [ffn_dim, d_model]. FFN mult = ffn_dim / d_model.
    ffn_mult = None
    for name, dims in shapes.items():
        if 'mlp.linear1.weight' in name and len(dims) == 2 and dmodel:
            ffn_dim = dims[0]
            if ffn_dim % dmodel == 0:
                ffn_mult = ffn_dim // dmodel
                break

    # 5. Head output sizes: policy head output size (1858), value (3), action (1858*3=5574), etc.
    # Check output tensor shapes from model.graph.output
    output_info = {}
    for out in model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else '?' for d in out.type.tensor_type.shape.dim]
        output_info[out.name] = shape

    return {
        'num_layers': num_layers,
        'd_model': dmodel,
        'num_heads': num_heads,
        'ffn_mult': ffn_mult,
        'output_names': list(output_info.keys()),
        'output_shapes': output_info,
    }


def main(path):
    print(f"Loading {path}...")
    model = onnx.load(path)
    print(f"  nodes: {len(model.graph.node)}")
    print(f"  initializers (weight tensors): {len(model.graph.initializer)}")

    arch = infer_architecture(model)
    print("\n=== Inferred architecture ===")
    for k, v in arch.items():
        print(f"  {k}: {v}")

    # Dump all transformer_layer.0.* tensor names and shapes so we can cross-reference
    print("\n=== Layer 0 tensor inventory ===")
    from collections import OrderedDict
    layer0 = OrderedDict()
    for t in model.graph.initializer:
        if 'transformer_layer.0.' in t.name:
            layer0[t.name] = list(t.dims)
    for name, dims in layer0.items():
        print(f"  {name:<80s} {dims}")

    # Count input planes by looking at first input tensor / first matmul
    print("\n=== Graph inputs ===")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else '?' for d in inp.type.tensor_type.shape.dim]
        elem_t = inp.type.tensor_type.elem_type
        print(f"  {inp.name}: shape={shape}, elem_type={elem_t}")

    # List top-level params (not in transformer_layer.*)
    print("\n=== Top-level initializers (excl. transformer_layer) ===")
    top_level = [(t.name, list(t.dims)) for t in model.graph.initializer
                 if 'transformer_layer' not in t.name]
    top_level.sort(key=lambda x: x[0])
    for name, dims in top_level[:40]:
        print(f"  {name:<80s} {dims}")
    if len(top_level) > 40:
        print(f"  ... and {len(top_level) - 40} more")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/c/Dev/Chess/Networks/CeresNet/C1-640-34-I8.onnx'
    main(path)
