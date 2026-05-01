#!/usr/bin/env python3
"""v8 fold+export with UINT8 input AND decomposed Mish — fork of export_v8_uint8.py.

Mish(x) = x * tanh(softplus(x)). PyTorch exports `torch.nn.Mish` as a single
opset-18 `Mish` op. TensorRT may not have an optimized kernel for `Mish`
across all versions, falling back to a generic implementation. Orig's ONNX
(`C1-640-34-I8.onnx`) was exported pre-decomposed into 75 × (Softplus + Tanh + Mul)
triples — TRT fuses these into an efficient kernel.

This script monkey-patches `torch.nn.Mish` and `torch.nn.functional.mish`
to emit the decomposed form before model construction, so the exported ONNX
matches orig's activation pattern.

Reads LORA_BIN and OUT from env vars V8_LORA_BIN / V8_OUT.
"""
import os, sys, math, struct, torch, torch.nn as nn

# ===== Mish decomposition patch (must happen BEFORE model construction) =====
def _mish_decomposed(x):
    """Mish(x) = x * tanh(softplus(x))."""
    return x * torch.tanh(torch.nn.functional.softplus(x))

# Patch the functional API used in dot_product_attention.py:286
torch.nn.functional.mish = _mish_decomposed

# Patch the module form used by activation_functions.py:28 (returned by to_activation('Mish'))
class _MishDecomposed(torch.nn.Module):
    def forward(self, x):
        return _mish_decomposed(x)

# Replace torch.nn.Mish with our decomposed module so any net building code
# that calls torch.nn.Mish() gets the decomposed version.
torch.nn.Mish = _MishDecomposed

CERES_PY_DIR = '/mnt/c/Users/lepne/source/repos/CeresTrain/src/CeresTrainPy'
sys.path.insert(0, CERES_PY_DIR)
from config import Configuration
from ceres_net import CeresNet

BASE_CKPT = os.environ.get('V8_BASE_CKPT', '/mnt/c/Dev/Chess/CeresTrain/nets/ckpt_c1_640_34_from_onnx_0')
LORA_BIN  = os.environ.get('V8_LORA_BIN', '')   # optional — empty means no LoRA fold (e.g. GTAB-only run)
OUT       = os.environ['V8_OUT']
# Optional comma-separated list of name-prefixes to skip when folding the bin.
# Use case: when V8_BASE_CKPT already has body LoRA folded in (e.g. via
# extract_body_lora_from_ckpt.py), the bin's body modules would re-apply the
# delta. Set V8_LORA_SKIP_PREFIX=transformer_layer. to skip body fold.
SKIP_PREFIX = os.environ.get('V8_LORA_SKIP_PREFIX', '')
print(f'[export] BASE_CKPT={BASE_CKPT}')
print(f'[export] LORA_BIN={LORA_BIN if LORA_BIN else "(none — direct export of base ckpt)"}')
print(f'[export] SKIP_PREFIX={SKIP_PREFIX if SKIP_PREFIX else "(none)"}')


def deserialize_lora_bin(path):
    out = {}
    with open(path, 'rb') as f:
        n = struct.unpack('I', f.read(4))[0]
        for _ in range(n):
            nl = struct.unpack('I', f.read(4))[0]
            name = f.read(nl).decode('utf-8')
            alpha = struct.unpack('f', f.read(4))[0]
            ra, ca = struct.unpack('II', f.read(8))
            A = torch.tensor([struct.unpack(f'{ca}f', f.read(4*ca)) for _ in range(ra)], dtype=torch.float32)
            rb, cb = struct.unpack('II', f.read(8))
            B = torch.tensor([struct.unpack(f'{cb}f', f.read(4*cb)) for _ in range(rb)], dtype=torch.float32)
            out[name] = {'A': A, 'B': B, 'alpha': float(alpha)}
    return out


base = {k.replace('_forward_module._orig_mod.', ''): v.to(torch.float32)
        for k, v in torch.load(BASE_CKPT, map_location='cpu', weights_only=False)['model'].items()}

folded = dict(base)
if LORA_BIN:
    lora = deserialize_lora_bin(LORA_BIN)
    skip_prefixes = [p.strip() for p in SKIP_PREFIX.split(',') if p.strip()]
    skipped = 0
    applied = 0
    for name, d in lora.items():
        if any(name.startswith(p) for p in skip_prefixes):
            skipped += 1
            continue
        wk = f"{name}.weight"
        if wk not in folded: continue
        r = d['A'].shape[1]
        scaling = d['alpha'] / math.sqrt(r)
        update = scaling * (d['A'] @ d['B'])
        folded[wk] = folded[wk] + update
        applied += 1
    print(f'[export] folded {applied} bin entries, skipped {skipped} per SKIP_PREFIX')

config = Configuration('/mnt/c/Dev/Chess/CeresTrain/configs', 'c1_640_34')
config.Opt_LoRARankDivisor = 0
from lightning.fabric import Fabric
fabric = Fabric(accelerator='cpu', devices=1)
net = CeresNet(fabric, config,
    policy_loss_weight=1.5, value_loss_weight=1.0, moves_left_loss_weight=0,
    unc_loss_weight=0.01, value2_loss_weight=0.04, q_deviation_loss_weight=0.02,
    value_diff_loss_weight=0, value2_diff_loss_weight=0, action_loss_weight=0,
    uncertainty_policy_weight=0.01, action_uncertainty_loss_weight=0, q_ratio=0.0)
net = net.to(torch.float32).eval()
net.load_state_dict({k: v.to(torch.float32) for k,v in folded.items()}, strict=False)


class SingleInputNetUINT8(nn.Module):
    def __init__(self, inner, real_dim):
        super().__init__()
        self.inner = inner
        self.real_dim = real_dim
    def forward(self, squares_byte):
        squares = squares_byte.to(torch.float32) / 100.0
        ps = torch.zeros(squares.shape[0], 64, self.real_dim, dtype=squares.dtype, device=squares.device)
        return self.inner(squares, ps)


wrapper = SingleInputNetUINT8(net, config.NetDef_PriorStateDim).eval()

with torch.no_grad():
    sq_test = torch.randint(0, 256, (2, 64, 137), dtype=torch.uint8)
    out = wrapper(sq_test)
    print(f"wrapper test: {len(out)} outputs, policy shape {out[0].shape}, value shape {out[1].shape}")

sample_inputs = (torch.randint(0, 256, (256, 64, 137), dtype=torch.uint8),)
head_out = ['policy','value','mlh','unc','value2','q_deviation_lower',
            'q_deviation_upper','uncertainty_policy','action','prior_state','action_uncertainty']
axes = {n: {0:'batch_size'} for n in ['squares_byte']+head_out}

torch.onnx.export(wrapper, sample_inputs, OUT,
                  do_constant_folding=True,
                  export_params=True, opset_version=17,
                  input_names=['squares_byte'], output_names=head_out,
                  dynamic_axes=axes)
print("FP32 exported. Converting to FP16...")

from onnxconverter_common.float16 import convert_float_to_float16
import onnx as _onnx
m = _onnx.load(OUT)
m16 = convert_float_to_float16(m, keep_io_types=False, min_positive_val=1e-10, max_finite_val=1e4,
                               op_block_list=['Cast'])
_onnx.save(m16, OUT)
print(f"Saved: {OUT}")

m2 = _onnx.load(OUT)
print("Inputs:", [(i.name, _onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type)) for i in m2.graph.input])
print("Outputs:", [(o.name, _onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type)) for o in m2.graph.output])
op_counts = {}
for n in m2.graph.node:
    op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
print(f"Mish op count: {op_counts.get('Mish', 0)}")
print(f"Softplus op count: {op_counts.get('Softplus', 0)}")
print(f"Tanh op count: {op_counts.get('Tanh', 0)}")
print(f"Total nodes: {len(m2.graph.node)}")
