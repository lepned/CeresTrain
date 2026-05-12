#!/usr/bin/env python3
"""
Reconstruct a PyTorch CeresNet checkpoint from an ONNX file.

Strategy:
  1. Instantiate a CeresNet with the config that matches the ONNX's architecture
  2. Enumerate model.state_dict() keys and their expected shapes
  3. For each PyTorch key, find the corresponding ONNX tensor:
     - Exact name match: direct (biases, norm scales, some named linear weights)
     - Anonymous matmul weights: trace from the adjacent named bias via ONNX graph walk
  4. Convert FP16 → FP32 (no precision loss in that direction)
  5. Transpose anonymous matmul weights to PyTorch's [out_features, in_features] layout
  6. Load into the CeresNet, report any unmatched keys
  7. Save as a fabric-compatible checkpoint {'model': state_dict, 'optimizer': {...}, 'num_pos': '0'}

Run inside WSL:
  wsl.exe -d Ubuntu -- bash -lc "source ~/cerestrain-env/bin/activate && \
    python3 /mnt/c/Dev/Chess/CeresTrain/reconstruct_ckpt_from_onnx.py \
      /mnt/c/Dev/Chess/Networks/CeresNet/C1-640-34-I8.onnx c1_640_34 /mnt/c/Dev/Chess/CeresTrain"
"""
import sys
import os
import re
import traceback

import torch
import onnx
from onnx import numpy_helper

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
CERES_PY_DIR = os.environ.get('CERES_PY_DIR') or os.path.join(_REPO_ROOT, 'src', 'CeresTrainPy')
sys.path.insert(0, CERES_PY_DIR)

from config import Configuration
from ceres_net import CeresNet


def build_weight_bias_pairs(model):
    """Trace ONNX graph: for each Add that combines (MatMul_output, named_bias),
    record {bias_name: weight_initializer_name}. Also maps bare MatMul nodes (Linears
    with no bias, e.g. q2/k2/v2 secondary projections under NonLinearAttention) by
    deriving a PyTorch param name from the MatMul node name path."""
    produces = {o: n for n in model.graph.node for o in n.output}
    inits = {t.name for t in model.graph.initializer}

    pairs = {}  # key (PyTorch param name fragment e.g. '.qkv.bias' or '.q2.weight') → weight_init_name
    for node in model.graph.node:
        if node.op_type != 'Add':
            continue
        bias_name = None
        value_src = None
        for inp in node.input:
            if inp in inits and inp.endswith('.bias'):
                bias_name = inp
            else:
                value_src = inp
        if bias_name is None or value_src is None:
            continue
        producer = produces.get(value_src)
        if producer is None or producer.op_type not in ('MatMul', 'Gemm'):
            continue
        for inp in producer.input:
            if inp in inits and inp.startswith('onnx::MatMul'):
                pairs[bias_name] = inp
                break

    # Now also cover bare MatMul Linears (no bias) — the q2/k2/v2 pattern under
    # NonLinearAttention. Derive a synthetic PyTorch-like name from the MatMul
    # node's path and record a suggestion under a distinct 'bare' dict.
    bare = {}  # pytorch_name_fragment → onnx::MatMul_N
    for node in model.graph.node:
        if node.op_type != 'MatMul':
            continue
        # e.g. name="/transformer_layer.0/attention/q2/MatMul"
        n = node.name.strip('/').replace('/', '.')
        if not n.endswith('.MatMul'):
            continue
        synth = n[:-len('.MatMul')] + '.weight'
        if synth in inits:
            continue  # already has a named weight; not a bare case
        # The weight initializer is the input that starts with 'onnx::MatMul'
        weight_init = None
        for inp in node.input:
            if inp in inits and inp.startswith('onnx::MatMul'):
                weight_init = inp
                break
        if weight_init is not None:
            bare[synth] = weight_init
    return pairs, bare


def load_onnx_tensors(onnx_path):
    """Load all initializers from an ONNX file as a dict {name: torch.Tensor(FP32)}."""
    print(f"Loading ONNX initializers from {onnx_path}...")
    model = onnx.load(onnx_path)
    tensors = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        tensors[init.name] = torch.from_numpy(arr.copy()).to(torch.float32)
    print(f"  loaded {len(tensors)} tensors")
    return model, tensors


def find_onnx_source(pytorch_key, torch_shape, onnx_tensors, bias_weight_pairs,
                    shared_sources=None, bare_weight_pairs=None):
    """Find the ONNX tensor name that should feed into this PyTorch param.
    Returns (onnx_name, needs_transpose) or (None, None) on no-match."""

    # 1. Exact match
    if pytorch_key in onnx_tensors:
        onnx_shape = list(onnx_tensors[pytorch_key].shape)
        if onnx_shape == list(torch_shape):
            return pytorch_key, False
        # Some ONNX-named Linear weights are stored in PyTorch format already; shapes match → no transpose.
        # If shapes match after transpose (2D), try that.
        if len(onnx_shape) == 2 and len(torch_shape) == 2:
            if onnx_shape == [torch_shape[1], torch_shape[0]]:
                return pytorch_key, True

    # 2. For .weight params, look up via adjacent bias in the traced graph
    if pytorch_key.endswith('.weight'):
        bias_key = pytorch_key[:-7] + '.bias'
        onnx_weight = bias_weight_pairs.get(bias_key)
        if onnx_weight is not None:
            # onnx::MatMul initializers are stored transposed vs PyTorch Linear.weight
            return onnx_weight, True

    # 3. Shared smolgenPrepLayer per-transformer-layer case: the shared weight has name
    # "onnx::MatMul_6816" (for layer 0). Any transformer_layer.N.smolgenPrepLayer.weight
    # maps to the same shared weight. Check if PyTorch key contains 'smolgenPrepLayer'.
    if 'smolgenPrepLayer.weight' in pytorch_key and shared_sources:
        return shared_sources.get('smolgenPrepLayer'), True

    # 4. Bare MatMul weights (no bias): try matching by suffix path (q2/k2/v2 etc).
    # pytorch_key example: 'transformer_layer.0.attention.q2.weight'
    # bare pair keys look like: 'transformer_layer.0.attention.q2.weight' (synth name).
    if bare_weight_pairs and pytorch_key in bare_weight_pairs:
        return bare_weight_pairs[pytorch_key], True

    return None, None


def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/c/Dev/Chess/Networks/CeresNet/C1-640-34-I8.onnx'
    config_name = sys.argv[2] if len(sys.argv) > 2 else 'c1_640_34'
    outputs_dir = sys.argv[3] if len(sys.argv) > 3 else '/mnt/c/Dev/Chess/CeresTrain'

    # 1. Load ONNX
    model_onnx, onnx_tensors = load_onnx_tensors(onnx_path)
    bias_weight_pairs, bare_weight_pairs = build_weight_bias_pairs(model_onnx)
    print(f"Traced {len(bias_weight_pairs)} bias→weight pairs, {len(bare_weight_pairs)} bare-MatMul Linears.")

    # Special-case: smolgenPrepLayer is defined once on CeresNet but used N times per layer.
    # The shared weight init is the single 'onnx::MatMul' that feeds the first occurrence.
    shared = {'smolgenPrepLayer': bias_weight_pairs.get('smolgenPrepLayer.bias')}

    # 2. Build Configuration and CeresNet
    print(f"\nInstantiating CeresNet from config '{config_name}'...")
    config_dir = os.path.join(outputs_dir, 'configs')
    config = Configuration(config_dir, config_name)

    from lightning.fabric import Fabric
    fabric = Fabric(accelerator='cpu', devices=1)

    # Constructor signature (from train.py): CeresNet(fabric, config, policy_loss_weight=..., value_loss_weight=..., etc.)
    net = CeresNet(
        fabric, config,
        policy_loss_weight=config.Opt_LossPolicyMultiplier,
        value_loss_weight=config.Opt_LossValueMultiplier,
        moves_left_loss_weight=config.Opt_LossMLHMultiplier,
        unc_loss_weight=config.Opt_LossUNCMultiplier,
        value2_loss_weight=config.Opt_LossValue2Multiplier,
        q_deviation_loss_weight=config.Opt_LossQDeviationMultiplier,
        value_diff_loss_weight=config.Opt_LossValueDMultiplier,
        value2_diff_loss_weight=config.Opt_LossValue2DMultiplier,
        action_loss_weight=config.Opt_LossActionMultiplier,
        uncertainty_policy_weight=config.Opt_LossUncertaintyPolicyMultiplier,
        action_uncertainty_loss_weight=config.Opt_LossActionUncertaintyMultiplier,
        q_ratio=0.0,
    )
    net = net.to(torch.float32)
    print(f"  built CeresNet: {sum(p.numel() for p in net.parameters())/1e6:.1f} M params")

    # 3. Build PyTorch key → ONNX tensor mapping
    sd = net.state_dict()
    print(f"\nCeresNet state_dict has {len(sd)} entries.")

    new_sd = {}
    matched = 0
    mismatched_shape = []
    unmatched = []

    for key, tensor in sd.items():
        src_name, needs_transpose = find_onnx_source(key, tensor.shape, onnx_tensors,
                                                     bias_weight_pairs, shared, bare_weight_pairs)
        if src_name is None:
            unmatched.append(key)
            new_sd[key] = tensor  # keep random init
            continue
        src = onnx_tensors[src_name]
        if needs_transpose and src.ndim == 2:
            src = src.t().contiguous()
        if list(src.shape) != list(tensor.shape):
            mismatched_shape.append((key, list(tensor.shape), list(src.shape), src_name))
            new_sd[key] = tensor  # keep random init
            continue
        new_sd[key] = src.to(torch.float32)
        matched += 1

    print(f"\n=== Mapping summary ===")
    print(f"  matched:  {matched} / {len(sd)}")
    print(f"  shape mismatches: {len(mismatched_shape)}")
    print(f"  unmatched keys:   {len(unmatched)}")

    if mismatched_shape:
        print("\n=== First 20 shape mismatches ===")
        for key, torch_shape, src_shape, src_name in mismatched_shape[:20]:
            print(f"  {key:<70s} torch={torch_shape} onnx={src_shape} src={src_name}")

    if unmatched:
        print("\n=== First 30 unmatched PyTorch keys ===")
        for k in unmatched[:30]:
            print(f"  {k}  shape={list(sd[k].shape)}")

    # 4. Load into net (non-strict)
    net.load_state_dict(new_sd, strict=False)

    # 5. Steal optimizer dict structure from an existing puzzle-trained checkpoint
    # so param_groups structure matches exactly what train.py builds (decay/no_decay
    # split with the correct counts of param IDs). Replace 'state' with empty so
    # the optimizer starts fresh; only the structural metadata is reused.
    template_ckpt = '/mnt/c/Dev/Chess/CeresTrain/nets/ckpt_lepdev_puzzle_arch_G_50000896'
    if os.path.exists(template_ckpt):
        template = torch.load(template_ckpt, map_location='cpu', weights_only=False)
        opt_state = template['optimizer']
        opt_state['state'] = {}  # fresh optimizer state
        print(f"  using optimizer template from {template_ckpt} ({len(opt_state['param_groups'])} param groups)")
    else:
        # Fallback: minimal structure with no param groups (will fail on resume but
        # still allows the file to be loaded for inspection)
        opt_state = {'state': {}, 'param_groups': []}

    # 6. Save a fabric-format checkpoint
    out_path = os.path.join(outputs_dir, 'nets', 'ckpt_c1_640_34_from_onnx_0')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    state = {'model': new_sd, 'optimizer': opt_state, 'num_pos': '0'}
    torch.save(state, out_path)
    print(f"\nSaved reconstructed checkpoint to {out_path}")
    print(f"  matched={matched}, unmatched={len(unmatched)}, shape_mismatches={len(mismatched_shape)}")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
