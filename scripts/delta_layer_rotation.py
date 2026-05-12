#!/usr/bin/env python3
"""Per-layer rotation delta between orig and a LoRA fine-tune.

Pipeline:
  1. Build CeresNet at Opt_LoRARankDivisor=0 (raw nn.Linear modules).
  2. Load orig ckpt → snapshot W_orig per Linear.
  3. Forward-pass on TPG sample → alignment/eff_rank per layer (orig CSV).
  4. Read LoRA bin (format from lora.serialize_lora_to_binary) → merge
     ΔW = (alpha / sqrt(rank)) · A · B into the matching Linear.weight.
  5. Repeat the forward pass on the SAME positions → FT alignment/eff_rank.
  6. Write a delta CSV: alignment_orig, alignment_ft, dalignment,
     dw_frob, dw_rel, dw_eff_rank_95, lora_targeted.

Reads cleanly from .lora_*.bin files emitted by save_model.py post-training.
The LoRA-untargeted layers serve as a control: dw_frob ≈ 0, dalignment
should be ~0 modulo numerical noise from running the same forward twice.

Env vars:
  V8_BASE_CKPT       Orig pytorch ckpt.
  V8_LORA_BIN        .lora_*.bin file from training run.
  V8_TPG_DIR         TPG shards dir.
  V8_OUT_DELTA_CSV   Per-layer delta CSV.
  V8_OUT_FT_CSV      Per-layer raw FT CSV (optional, full rotation table).
  V8_NUM_POSITIONS   Forward-pass position budget (default 10240).
  V8_RANK            Truncation rank for alignment SVD (default 32).
  V8_BATCH_SIZE      (default 64).
  V8_DEVICE          (default cuda).

Usage example:
  V8_BASE_CKPT=/mnt/c/Dev/Chess/CeresTrain/nets/ckpt_c1_640_34_from_onnx_0  \\
  V8_LORA_BIN=/mnt/c/Dev/Chess/CeresTrain/nets/lepdev_c1_640_34_KL30_body32_PONLY_aug_2450up_1600K.lora_last.bin  \\
  V8_TPG_DIR=/mnt/d/c1_640_34_2600up_aug/tpg                                \\
  V8_OUT_DELTA_CSV=/mnt/c/Dev/Chess/CeresTrain/layer_rotation_delta_KL30_2450up_1600K.csv  \\
  python3 delta_layer_rotation.py
"""
import csv
import math
import os
import struct
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

# Decompose Mish so the inference graph matches export_v8_uint8_mish.py.
def _mish_decomposed(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))
torch.nn.functional.mish = _mish_decomposed
class _MishDecomposed(torch.nn.Module):
    def forward(self, x):
        return _mish_decomposed(x)
torch.nn.Mish = _MishDecomposed

CERES_PY_DIR = '/mnt/c/Users/lepne/source/repos/CeresTrain/src/CeresTrainPy'
sys.path.insert(0, CERES_PY_DIR)
from config import Configuration  # noqa: E402
from ceres_net import CeresNet    # noqa: E402

BASE_CKPT      = os.environ['V8_BASE_CKPT']
LORA_BIN       = os.environ['V8_LORA_BIN']
TPG_DIR        = os.environ.get('V8_TPG_DIR',  '/mnt/d/c1_640_34_2600up_aug/tpg')
OUT_DELTA_CSV  = os.environ['V8_OUT_DELTA_CSV']
OUT_FT_CSV     = os.environ.get('V8_OUT_FT_CSV')   # optional
NUM_POSITIONS  = int(os.environ.get('V8_NUM_POSITIONS', '10240'))
RANK           = int(os.environ.get('V8_RANK', '32'))
BATCH_SIZE     = int(os.environ.get('V8_BATCH_SIZE', '64'))
DEVICE         = os.environ.get('V8_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    print(msg, flush=True)

log(f'[delta] BASE_CKPT      = {BASE_CKPT}')
log(f'[delta] LORA_BIN       = {LORA_BIN}')
log(f'[delta] OUT_DELTA_CSV  = {OUT_DELTA_CSV}')
log(f'[delta] NUM_POSITIONS  = {NUM_POSITIONS}')
log(f'[delta] RANK           = {RANK}')
log(f'[delta] DEVICE         = {DEVICE}')

# ============================================================
# LoRA bin reader — inverse of serialize_lora_to_binary in lora.py.
# ============================================================
def read_lora_bin(path):
    """Return dict layer_name → {'A': tensor(out, r), 'B': tensor(r, in), 'alpha': float}."""
    out = {}
    with open(path, 'rb') as f:
        (n_layers,) = struct.unpack('I', f.read(4))
        for _ in range(n_layers):
            (name_len,) = struct.unpack('I', f.read(4))
            name = f.read(name_len).decode('utf-8')
            (alpha,) = struct.unpack('f', f.read(4))
            rows_a, cols_a = struct.unpack('II', f.read(8))
            A = np.frombuffer(f.read(rows_a * cols_a * 4), dtype=np.float32).reshape(rows_a, cols_a)
            rows_b, cols_b = struct.unpack('II', f.read(8))
            B = np.frombuffer(f.read(rows_b * cols_b * 4), dtype=np.float32).reshape(rows_b, cols_b)
            out[name] = {
                'A': torch.from_numpy(A.copy()),     # (out_features, rank)
                'B': torch.from_numpy(B.copy()),     # (rank,         in_features)
                'alpha': float(alpha),
            }
    return out

lora_dict = read_lora_bin(LORA_BIN)
log(f'[delta] LoRA bin: {len(lora_dict)} layers')

# ============================================================
# Build raw CeresNet, load orig
# ============================================================
config = Configuration('/mnt/c/Dev/Chess/CeresTrain/configs', 'c1_640_34')
config.Opt_LoRARankDivisor = 0

from lightning.fabric import Fabric  # noqa: E402
fabric = Fabric(accelerator='cpu', devices=1)
net = CeresNet(fabric, config,
    policy_loss_weight=1.5, value_loss_weight=1.0, moves_left_loss_weight=0,
    unc_loss_weight=0.01, value2_loss_weight=0.04, q_deviation_loss_weight=0.02,
    value_diff_loss_weight=0, value2_diff_loss_weight=0, action_loss_weight=0,
    uncertainty_policy_weight=0.01, action_uncertainty_loss_weight=0, q_ratio=0.0)

state = {k.replace('_forward_module._orig_mod.', ''): v.to(torch.float32)
         for k, v in torch.load(BASE_CKPT, map_location='cpu', weights_only=False)['model'].items()}
net.load_state_dict(state, strict=False)
net = net.to(DEVICE).to(torch.float32).eval()
log(f'[delta] orig net loaded — {sum(p.numel() for p in net.parameters()):,} params on {DEVICE}')

# ============================================================
# Snapshot orig weights per Linear (key = module name)
# ============================================================
linear_modules = OrderedDict()
W_orig_snapshot = {}
for name, module in net.named_modules():
    if isinstance(module, nn.Linear):
        linear_modules[name] = module
        W_orig_snapshot[name] = module.weight.detach().clone().to(torch.float64).cpu()
log(f'[delta] {len(linear_modules)} Linear modules')

# Sanity-check that LoRA names match Linear names
unmatched = [n for n in lora_dict if n not in linear_modules]
if unmatched:
    log(f'[delta] WARNING: {len(unmatched)} LoRA names not found in net: {unmatched[:5]}{"..." if len(unmatched)>5 else ""}')
covered = sum(1 for n in lora_dict if n in linear_modules)
log(f'[delta] {covered}/{len(lora_dict)} LoRA layers matched to Linear modules')

# ============================================================
# Activation-covariance accumulator (same as diagnose_layer_rotation.py)
# ============================================================
class CovAccumulator:
    def __init__(self, dim_in, dim_out, device):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.x_cov = torch.zeros(dim_in, dim_in, dtype=torch.float64, device=device)
        self.y_cov = torch.zeros(dim_out, dim_out, dtype=torch.float64, device=device)
        self.n_samples = 0
    def update(self, x, y):
        x_flat = x.reshape(-1, self.dim_in)
        y_flat = y.reshape(-1, self.dim_out)
        self.x_cov += (x_flat.T @ x_flat).to(torch.float64)
        self.y_cov += (y_flat.T @ y_flat).to(torch.float64)
        self.n_samples += x_flat.shape[0]
    def reset(self):
        self.x_cov.zero_()
        self.y_cov.zero_()
        self.n_samples = 0

accumulators = OrderedDict(
    (name, CovAccumulator(m.in_features, m.out_features, DEVICE))
    for name, m in linear_modules.items()
)
hook_handles = []
for name, module in linear_modules.items():
    acc = accumulators[name]
    def make_hook(_acc):
        def hook(_mod, _inp, _out):
            _acc.update(_inp[0].detach(), _out.detach())
        return hook
    hook_handles.append(module.register_forward_hook(make_hook(acc)))

# ============================================================
# TPG streaming (same protocol as diagnose_layer_rotation.py)
# ============================================================
import zstandard  # noqa: E402
BYTES_PER_POS = 9378
SIZE_SQUARE = 137

def iter_squares(tpg_dir, batch_size, target_total):
    files = sorted([f for f in os.listdir(tpg_dir) if f.endswith('.zst')])
    if not files:
        raise RuntimeError(f'no .zst files in {tpg_dir}')
    decompressor = zstandard.ZstdDecompressor()
    emitted = 0
    buf = bytearray()
    sq_offset = BYTES_PER_POS - 64 * SIZE_SQUARE
    for fname in files:
        with open(os.path.join(tpg_dir, fname), 'rb') as f:
            stream_reader = decompressor.stream_reader(f)
            while True:
                chunk = stream_reader.read(BYTES_PER_POS * batch_size * 2)
                if not chunk:
                    break
                buf.extend(chunk)
                while len(buf) >= BYTES_PER_POS * batch_size:
                    block = bytes(buf[:BYTES_PER_POS * batch_size])
                    del buf[:BYTES_PER_POS * batch_size]
                    arr = np.frombuffer(block, dtype=np.int8).reshape(batch_size, BYTES_PER_POS)
                    sq = arr[:, sq_offset:sq_offset + 64*SIZE_SQUARE].reshape(batch_size, 64, SIZE_SQUARE).astype(np.float32) / 100.0
                    yield torch.from_numpy(sq)
                    emitted += batch_size
                    if emitted >= target_total:
                        return

# ============================================================
# One full forward-pass sweep, returns dict[name] → metrics
# ============================================================
def run_pass_and_score(tag):
    for acc in accumulators.values():
        acc.reset()
    log(f'[delta] [{tag}] forward-passing {NUM_POSITIONS} positions in batches of {BATCH_SIZE}')
    t0 = time.time()
    n_done = 0
    prior_state_dim = config.NetDef_PriorStateDim
    with torch.no_grad():
        for sq_cpu in iter_squares(TPG_DIR, BATCH_SIZE, NUM_POSITIONS):
            sq = sq_cpu.to(DEVICE)
            ps = torch.zeros(sq.shape[0], 64, prior_state_dim, dtype=sq.dtype, device=sq.device)
            net(sq, ps)
            n_done += sq.shape[0]
    log(f'[delta] [{tag}] forward-pass done — {n_done} pos in {time.time()-t0:.1f}s')

    out = {}
    for name, acc in accumulators.items():
        if acc.n_samples == 0:
            continue
        x_cov = acc.x_cov / acc.n_samples
        y_cov = acc.y_cov / acc.n_samples
        try:
            evals_x, evecs_x = torch.linalg.eigh(x_cov)
            evals_y, evecs_y = torch.linalg.eigh(y_cov)
        except Exception:
            continue
        evals_x = evals_x.flip(0).clamp(min=0.0); evecs_x = evecs_x.flip(1)
        evals_y = evals_y.flip(0).clamp(min=0.0); evecs_y = evecs_y.flip(1)
        sv_y = evals_y.sqrt()
        r = min(RANK, evecs_x.shape[1], evecs_y.shape[1])
        A_in  = evecs_x[:, :r]
        A_out = evecs_y[:, :r]
        W = linear_modules[name].weight.detach().to(torch.float64).to(DEVICE)
        WA_in = W @ A_in
        M = A_out.T @ WA_in
        align = (M.norm() / (WA_in.norm() + 1e-12)).item()
        s_pos = sv_y[sv_y > 1e-12].to(torch.float64)
        if s_pos.numel() == 0:
            eff_rank_y = 0.0
        else:
            p = s_pos / s_pos.sum()
            eff_rank_y = float(np.exp(-(p * (p + 1e-20).log()).sum().item()))
        out[name] = {'alignment': align, 'eff_rank_y': eff_rank_y}
    return out

# ============================================================
# (1) Score orig
# ============================================================
orig_scores = run_pass_and_score('orig')

# ============================================================
# (2) Apply LoRA merges into each Linear.weight (in place)
# ΔW = (alpha / sqrt(rank)) · A · B    where A: (out, r), B: (r, in)
# ============================================================
n_merged = 0
total_dw_frob = 0.0
dw_norms = {}
dw_eff_rank95 = {}
for name, lora in lora_dict.items():
    if name not in linear_modules:
        continue
    A = lora['A']    # (out, r)
    B = lora['B']    # (r, in)
    rank = A.shape[1]
    scaling = lora['alpha'] / math.sqrt(rank)
    dW = (scaling * (A @ B)).to(torch.float64)   # (out, in)
    # numerical norm + effective rank of ΔW
    fro = dW.norm().item()
    dw_norms[name] = fro
    total_dw_frob += fro * fro
    try:
        sv_dW = torch.linalg.svdvals(dW.to(torch.float32))
        s = sv_dW.to(torch.float64)
        s2 = s * s
        cum = torch.cumsum(s2, dim=0) / s2.sum().clamp(min=1e-30)
        # smallest k such that cum[k] >= 0.95
        k95 = int((cum >= 0.95).nonzero()[0].item()) + 1 if (cum >= 0.95).any() else int(s.numel())
        dw_eff_rank95[name] = k95
    except Exception:
        dw_eff_rank95[name] = -1
    # apply
    with torch.no_grad():
        linear_modules[name].weight.data.add_(dW.to(linear_modules[name].weight.dtype).to(linear_modules[name].weight.device))
    n_merged += 1
log(f'[delta] merged {n_merged} LoRA deltas, ‖ΔW‖_F (sum of squares) = {math.sqrt(total_dw_frob):.4f}')

# ============================================================
# (3) Score FT
# ============================================================
ft_scores = run_pass_and_score('ft')

# ============================================================
# (4) Write delta CSV
# ============================================================
fields = [
    'name', 'lora_targeted',
    'alignment_orig', 'alignment_ft', 'dalignment',
    'eff_rank_y_orig', 'eff_rank_y_ft', 'd_eff_rank_y',
    'dw_frob', 'w_orig_frob', 'dw_rel', 'dw_eff_rank_95',
]
rows = []
for name in linear_modules:
    if name not in orig_scores or name not in ft_scores:
        continue
    o = orig_scores[name]; f = ft_scores[name]
    targeted = name in lora_dict
    w_orig_fro = float(W_orig_snapshot[name].norm().item())
    dw_fro = dw_norms.get(name, 0.0)
    rows.append({
        'name': name,
        'lora_targeted': int(targeted),
        'alignment_orig': round(o['alignment'], 6),
        'alignment_ft':   round(f['alignment'], 6),
        'dalignment':     round(f['alignment'] - o['alignment'], 6),
        'eff_rank_y_orig': round(o['eff_rank_y'], 3),
        'eff_rank_y_ft':   round(f['eff_rank_y'], 3),
        'd_eff_rank_y':    round(f['eff_rank_y'] - o['eff_rank_y'], 3),
        'dw_frob':       round(dw_fro, 6),
        'w_orig_frob':   round(w_orig_fro, 6),
        'dw_rel':        round(dw_fro / max(w_orig_fro, 1e-12), 6),
        'dw_eff_rank_95': dw_eff_rank95.get(name, 0),
    })

with open(OUT_DELTA_CSV, 'w', newline='') as fout:
    w = csv.DictWriter(fout, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
log(f'[delta] wrote {len(rows)} rows to {OUT_DELTA_CSV}')

# ============================================================
# Summary print
# ============================================================
targeted_rows = [r for r in rows if r['lora_targeted']]
if targeted_rows:
    by_dw = sorted(targeted_rows, key=lambda r: -r['dw_rel'])
    print()
    log('[delta] === TOP 12 LoRA-targeted layers by ‖ΔW‖_F / ‖W_orig‖_F (largest relative weight movement) ===')
    for r in by_dw[:12]:
        print(f"       dw_rel={r['dw_rel']:.4f}  Δalign={r['dalignment']:+.4f}  k95={r['dw_eff_rank_95']:3d}  {r['name']}")

    by_da = sorted(targeted_rows, key=lambda r: r['dalignment'])
    print()
    log('[delta] === TOP 8 layers by Δalignment (puzzle FT pulled rotation FURTHER off-axis) ===')
    for r in by_da[:8]:
        print(f"       Δalign={r['dalignment']:+.4f}  align: {r['alignment_orig']:.3f}→{r['alignment_ft']:.3f}  dw_rel={r['dw_rel']:.4f}  {r['name']}")

# Optional FT raw CSV (mirrors diagnose_layer_rotation.csv columns subset)
if OUT_FT_CSV:
    with open(OUT_FT_CSV, 'w', newline='') as fout:
        w = csv.DictWriter(fout, fieldnames=['name', 'alignment_in_subspace', 'eff_rank_y'])
        w.writeheader()
        for name in linear_modules:
            if name in ft_scores:
                w.writerow({
                    'name': name,
                    'alignment_in_subspace': round(ft_scores[name]['alignment'], 6),
                    'eff_rank_y':            round(ft_scores[name]['eff_rank_y'], 3),
                })
    log(f'[delta] wrote FT CSV to {OUT_FT_CSV}')
