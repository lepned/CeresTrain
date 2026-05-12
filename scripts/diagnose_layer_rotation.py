#!/usr/bin/env python3
"""Diagnose per-layer input/output subspace rotation in a Ceres net.

Implements Kovax's "active concentrator" measurement: for each Linear layer
in the network, capture input X and output Y activations on a sample of
training positions, then compute:

  A_in  = top-r right singular vectors of E[xxᵀ]   (input principal subspace)
  A_out = top-r left  singular vectors of E[yyᵀ]   (output principal subspace)

The mediated alignment S = ||A_outᵀ W A_in||_F / r tells us how much of
the layer's top-r input subspace is mapped by W into the top-r output
subspace. Interpretation:

  S close to 1: W maps top input directions to top output directions
                → "passive" — the layer mostly preserves the dominant
                  input subspace through to output.
  S close to 0: W rotates input variance into directions OUTSIDE the
                top output subspace → unusual (would mean output's top
                directions come from non-top input directions).
  S in middle:  partial alignment — W actively concentrates / reweights.

We also report effective rank of Y (entropy-based) and Y singular-value
decay slope, both proxies for how concentrated / low-dimensional the
layer's output distribution is. A low effective rank means LoRA capacity
beyond that rank is wasted on this layer.

Usage:
  V8_BASE_CKPT=/mnt/c/Dev/Chess/CeresTrain/nets/ckpt_c1_640_34_from_onnx_0  \\
  V8_TPG_DIR=/mnt/d/c1_640_34_2600up_aug/tpg                                \\
  V8_OUT_CSV=/mnt/c/Dev/Chess/CeresTrain/layer_rotation_c1_640_34.csv       \\
  V8_NUM_POSITIONS=10240                                                     \\
  V8_RANK=32                                                                 \\
  python3 diagnose_layer_rotation.py

Env vars:
  V8_BASE_CKPT      Path to base ckpt (orig).
  V8_TPG_DIR        Directory of .zst TPG shards (e.g. corpus dir).
  V8_OUT_CSV        Output CSV path.
  V8_NUM_POSITIONS  How many positions to forward-pass for the diagnostic
                    (default 10240 = 16 batches of 640). 10K is enough for
                    stable top-32 subspace estimation on a 640-dim residual
                    stream when each position contributes 64 token rows.
  V8_RANK           Truncation rank for A_in / A_out (default 32 = matches
                    body LoRA r-div=32 in our flagship recipe).
  V8_BATCH_SIZE     Batch size for forward passes (default 64).
  V8_DEVICE         cuda or cpu (default cuda if available).
"""

import os
import sys
import csv
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# Decompose Mish exactly as export_v8_uint8_mish.py does, so the model we
# build matches the inference-time graph (avoids any drift from a single
# Mish op vs Softplus+Tanh+Mul triples).
# ============================================================
def _mish_decomposed(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))

torch.nn.functional.mish = _mish_decomposed

class _MishDecomposed(torch.nn.Module):
    def forward(self, x):
        return _mish_decomposed(x)

torch.nn.Mish = _MishDecomposed

# ============================================================
# Bring in CeresTrainPy
# ============================================================
CERES_PY_DIR = '/mnt/c/Users/lepne/source/repos/CeresTrain/src/CeresTrainPy'
sys.path.insert(0, CERES_PY_DIR)
from config import Configuration  # noqa: E402
from ceres_net import CeresNet    # noqa: E402

# ============================================================
# Env-var configuration
# ============================================================
BASE_CKPT       = os.environ.get('V8_BASE_CKPT', '/mnt/c/Dev/Chess/CeresTrain/nets/ckpt_c1_640_34_from_onnx_0')
TPG_DIR         = os.environ.get('V8_TPG_DIR',  '/mnt/d/c1_640_34_2600up_aug/tpg')
OUT_CSV         = os.environ.get('V8_OUT_CSV',  '/mnt/c/Dev/Chess/CeresTrain/layer_rotation_c1_640_34.csv')
NUM_POSITIONS   = int(os.environ.get('V8_NUM_POSITIONS', '10240'))
RANK            = int(os.environ.get('V8_RANK', '32'))
BATCH_SIZE      = int(os.environ.get('V8_BATCH_SIZE', '64'))
DEVICE          = os.environ.get('V8_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
PROGRESS_LOG    = os.environ.get('V8_PROGRESS_LOG', '/mnt/c/Dev/Chess/CeresTrain/diag_progress.log')

# Write progress to a dedicated file as well as stdout, so visibility doesn't
# depend on tee/pipe buffering across the WSL→Windows boundary.
_progress_fh = open(PROGRESS_LOG, 'w', buffering=1)  # line-buffered

def log(msg):
    print(msg, flush=True)
    _progress_fh.write(msg + '\n')
    _progress_fh.flush()
    try:
        os.fsync(_progress_fh.fileno())
    except OSError:
        pass

log(f'[diag] BASE_CKPT      = {BASE_CKPT}')
log(f'[diag] TPG_DIR        = {TPG_DIR}')
log(f'[diag] OUT_CSV        = {OUT_CSV}')
log(f'[diag] PROGRESS_LOG   = {PROGRESS_LOG}')
log(f'[diag] NUM_POSITIONS  = {NUM_POSITIONS}')
log(f'[diag] RANK           = {RANK}')
log(f'[diag] BATCH_SIZE     = {BATCH_SIZE}')
log(f'[diag] DEVICE         = {DEVICE}')

# ============================================================
# Build CeresNet, load orig weights, eval mode
# ============================================================
config = Configuration('/mnt/c/Dev/Chess/CeresTrain/configs', 'c1_640_34')
config.Opt_LoRARankDivisor = 0  # raw net, no LoRA wrappers, so we inspect the orig Linear layers directly

from lightning.fabric import Fabric  # noqa: E402
fabric = Fabric(accelerator='cpu', devices=1)
net = CeresNet(fabric, config,
    policy_loss_weight=1.5, value_loss_weight=1.0, moves_left_loss_weight=0,
    unc_loss_weight=0.01, value2_loss_weight=0.04, q_deviation_loss_weight=0.02,
    value_diff_loss_weight=0, value2_diff_loss_weight=0, action_loss_weight=0,
    uncertainty_policy_weight=0.01, action_uncertainty_loss_weight=0, q_ratio=0.0)

log(f'[diag] loading ckpt from {BASE_CKPT}')
state = {k.replace('_forward_module._orig_mod.', ''): v.to(torch.float32)
         for k, v in torch.load(BASE_CKPT, map_location='cpu', weights_only=False)['model'].items()}
log(f'[diag] ckpt loaded — {len(state)} tensors, applying to net...')
net.load_state_dict(state, strict=False)
log(f'[diag] state_dict applied, moving to {DEVICE}...')
net = net.to(DEVICE).to(torch.float32).eval()
log(f'[diag] net loaded — {sum(p.numel() for p in net.parameters()):,} params on {DEVICE}')

# ============================================================
# Hook every nn.Linear: accumulate sum(X^T X), sum(Y^T Y), and sample count.
# Memory-efficient: we never store raw activations, only the running covariance
# matrices. For dim 640 each cov is 1.6 MB float32; total across ~200 layers
# stays well under 1 GB.
# ============================================================
class CovAccumulator:
    """Accumulate XᵀX, YᵀY (sums, not means — we'll normalize at the end)."""
    def __init__(self, name, dim_in, dim_out, device):
        self.name = name
        self.dim_in = dim_in
        self.dim_out = dim_out
        # Use the device the activations live on; converted to fp64 for stability.
        self.x_cov = torch.zeros(dim_in, dim_in, dtype=torch.float64, device=device)
        self.y_cov = torch.zeros(dim_out, dim_out, dtype=torch.float64, device=device)
        self.n_samples = 0

    def update(self, x, y):
        # x: (..., dim_in), y: (..., dim_out)
        # Do the inner GEMM in fp32 (fast on consumer GPU; fp64 is ~1/64 throughput).
        # Then upcast the small dim×dim outer-product result to fp64 once and accumulate.
        # Per-batch fp32 inner-product error for ~4K rows: ~6e-6 relative — well below
        # what matters for top-r eigenvector estimation on a 640×640 cov.
        x_flat = x.reshape(-1, self.dim_in)
        y_flat = y.reshape(-1, self.dim_out)
        self.x_cov += (x_flat.T @ x_flat).to(torch.float64)
        self.y_cov += (y_flat.T @ y_flat).to(torch.float64)
        self.n_samples += x_flat.shape[0]


accumulators = OrderedDict()
linear_modules = []  # (name, module) pairs we'll iterate after capture

for name, module in net.named_modules():
    if not isinstance(module, nn.Linear):
        continue
    # Skip the embedding-input linear and head outputs at the very end? No —
    # we want to inspect ALL of them so the user can see where active vs
    # passive layers live. The CSV will let them filter post-hoc.
    accumulators[name] = CovAccumulator(name, module.in_features, module.out_features, DEVICE)
    linear_modules.append((name, module))

log(f'[diag] hooked {len(accumulators)} Linear modules')

# Register forward hooks. We capture both input and output. Hook signature:
# (module, inputs, output) — inputs is a tuple, output is the result tensor.
hook_handles = []
for name, module in linear_modules:
    acc = accumulators[name]
    def make_hook(_acc):
        def hook(_mod, _inp, _out):
            x = _inp[0]
            _acc.update(x.detach(), _out.detach())
        return hook
    hook_handles.append(module.register_forward_hook(make_hook(acc)))

# ============================================================
# Stream batches of squares from TPG files
# ============================================================
import zstandard  # noqa: E402

BYTES_PER_POS = 9378
SIZE_SQUARE = 137

def iter_squares(tpg_dir, batch_size, target_total):
    """Yield (B, 64, 137) float32 squares tensors until target_total positions
    have been emitted. Reads .zst files in directory order; each record is
    BYTES_PER_POS bytes; squares are the last 64*137=8768 bytes of each record."""
    files = sorted([f for f in os.listdir(tpg_dir) if f.endswith('.zst')])
    if not files:
        raise RuntimeError(f'no .zst files in {tpg_dir}')
    log(f'[diag] streaming from {len(files)} TPG file(s)')
    decompressor = zstandard.ZstdDecompressor()
    emitted = 0
    buf = bytearray()
    sq_offset_in_record = BYTES_PER_POS - 64 * SIZE_SQUARE  # squares are at the tail of each record
    for fname in files:
        path = os.path.join(tpg_dir, fname)
        with open(path, 'rb') as f:
            stream_reader = decompressor.stream_reader(f)
            while True:
                chunk = stream_reader.read(BYTES_PER_POS * batch_size * 2)
                if not chunk:
                    break
                buf.extend(chunk)
                # Yield as many full batches as we have
                while len(buf) >= BYTES_PER_POS * batch_size:
                    block = bytes(buf[:BYTES_PER_POS * batch_size])
                    del buf[:BYTES_PER_POS * batch_size]
                    arr = np.frombuffer(block, dtype=np.int8).reshape(batch_size, BYTES_PER_POS)
                    sq_bytes = arr[:, sq_offset_in_record : sq_offset_in_record + 64 * SIZE_SQUARE]
                    sq = sq_bytes.reshape(batch_size, 64, SIZE_SQUARE).astype(np.float32) / 100.0
                    yield torch.from_numpy(sq)
                    emitted += batch_size
                    if emitted >= target_total:
                        return
    # Fell off end of files — that's fine; just emitted < target.

# ============================================================
# Forward-pass loop
# ============================================================
log(f'[diag] forward-passing up to {NUM_POSITIONS} positions in batches of {BATCH_SIZE}')
t0 = time.time()
n_done = 0
prior_state_dim = config.NetDef_PriorStateDim
with torch.no_grad():
    for sq_cpu in iter_squares(TPG_DIR, BATCH_SIZE, NUM_POSITIONS):
        sq = sq_cpu.to(DEVICE)
        ps = torch.zeros(sq.shape[0], 64, prior_state_dim, dtype=sq.dtype, device=sq.device)
        net(sq, ps)
        n_done += sq.shape[0]
        if n_done % (BATCH_SIZE * 8) == 0:
            log(f'[diag]   {n_done}/{NUM_POSITIONS} positions ({(time.time()-t0):.1f}s)')

elapsed = time.time() - t0
log(f'[diag] forward-pass done — {n_done} positions in {elapsed:.1f}s')

# Remove hooks now that we have all the data
for h in hook_handles:
    h.remove()

# ============================================================
# Per-layer SVD + rotation analysis
# ============================================================
def effective_rank(singular_values, eps=1e-12):
    """Roy & Vetterli effective rank: exp(H(p)) where p = s_i / sum(s)."""
    s = singular_values.to(torch.float64)
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    H = -(p * (p + eps).log()).sum().item()
    return float(np.exp(H))

def topk_decay_ratio(singular_values, k):
    """Fraction of total singular-value mass captured by top-k."""
    s = singular_values.to(torch.float64)
    if s.numel() == 0:
        return 0.0
    return (s[:k].sum() / s.sum()).item() if s.sum() > 0 else 0.0

log(f'[diag] computing SVD + rotation metrics for {len(accumulators)} layers')

rows = []
for name, acc in accumulators.items():
    if acc.n_samples == 0:
        # Layer wasn't reached on the forward pass (e.g., disabled head).
        continue

    n = acc.n_samples
    # Normalize to covariance estimates (mean of outer products).
    # Using sum is fine for SVD-direction extraction; we just want eigenvectors.
    # But we use mean for singular-value MAGNITUDE comparisons.
    x_cov = acc.x_cov / n
    y_cov = acc.y_cov / n

    # SVD of symmetric PSD matrices: eigendecomposition is more stable.
    # eigh returns eigenvalues ascending — flip to descending.
    try:
        evals_x, evecs_x = torch.linalg.eigh(x_cov)
        evals_y, evecs_y = torch.linalg.eigh(y_cov)
    except Exception as e:
        log(f'[diag]   {name}: eigh failed ({e}), skipping')
        continue
    # Descending order
    evals_x = evals_x.flip(0).clamp(min=0.0)
    evecs_x = evecs_x.flip(1)  # columns are eigenvectors
    evals_y = evals_y.flip(0).clamp(min=0.0)
    evecs_y = evecs_y.flip(1)

    # Singular values of activation distribution = sqrt of eigenvalues of cov.
    sv_x = evals_x.sqrt()
    sv_y = evals_y.sqrt()

    # Truncation
    r = min(RANK, evecs_x.shape[1], evecs_y.shape[1])
    A_in  = evecs_x[:, :r]   # (dim_in,  r)  — top-r right singular vectors of X
    A_out = evecs_y[:, :r]   # (dim_out, r)  — top-r left  singular vectors of Y

    # Get the layer's weight (and bias for completeness).
    module = dict(net.named_modules())[name]
    W = module.weight.detach().to(torch.float64).to(DEVICE)  # (dim_out, dim_in)

    # Mediated alignment: how much of W's action on top-r input subspace
    # lands inside the top-r output subspace.
    # M = A_outᵀ W A_in   (r × r). If perfect alignment, M is unitary up to scale.
    M = A_out.T @ W @ A_in  # (r, r)

    # Frobenius-norm-based alignment score in [0, 1]:
    #   numerator   = ||A_outᵀ W A_in||_F   (component of W·A_in inside span(A_out))
    #   denominator = ||W A_in||_F          (total action of W on top input subspace)
    WA_in = W @ A_in            # (dim_out, r)
    num = M.norm().item()
    den = WA_in.norm().item() + 1e-12
    alignment_in_subspace = num / den   # 1.0 = perfect alignment, 0 = orthogonal

    # Principal angles: SVD of M gives cosines of principal angles between
    # span(A_out) and span(W A_in / ||...||). Each singular value ∈ [0, 1].
    try:
        # Normalize M's columns by per-column norm of WA_in to get a direction-
        # only comparison.
        col_norm = WA_in.norm(dim=0) + 1e-12
        M_norm = M / col_norm  # (r, r), columns now lie in unit balls
        sv_M = torch.linalg.svdvals(M_norm)
        principal_cos_mean = sv_M.mean().item()
        principal_cos_min  = sv_M.min().item()
    except Exception:
        principal_cos_mean = float('nan')
        principal_cos_min  = float('nan')

    # Effective rank of Y (how concentrated is the output distribution)
    eff_rank_y = effective_rank(sv_y)
    eff_rank_x = effective_rank(sv_x)

    # Top-r mass: fraction of Y singular-value mass in top-r
    topr_mass_y = topk_decay_ratio(sv_y, r)

    rows.append({
        'name': name,
        'dim_in': acc.dim_in,
        'dim_out': acc.dim_out,
        'n_samples': n,
        'rank_r': r,
        'alignment_in_subspace': round(alignment_in_subspace, 6),
        'principal_cos_mean': round(principal_cos_mean, 6),
        'principal_cos_min':  round(principal_cos_min,  6),
        'eff_rank_x': round(eff_rank_x, 3),
        'eff_rank_y': round(eff_rank_y, 3),
        'topr_mass_y_frac': round(topr_mass_y, 6),
        'sv_y_top1': round(sv_y[0].item(), 6),
        'sv_y_topr': round(sv_y[r-1].item(), 6) if r > 0 else 0.0,
        'sv_y_decay_topr_to_top1': round((sv_y[r-1] / (sv_y[0] + 1e-12)).item(), 6) if r > 0 else 0.0,
        'w_norm': round(W.norm().item(), 4),
    })

# Write CSV
with open(OUT_CSV, 'w', newline='') as f:
    if not rows:
        log('[diag] WARNING: no rows to write — did the forward pass reach any Linear layers?')
    else:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

log(f'[diag] wrote {len(rows)} rows to {OUT_CSV}')

# Quick textual summary
if rows:
    by_align = sorted(rows, key=lambda r: r['alignment_in_subspace'])
    print()
    log('[diag] === TOP 8 most ROTATED layers (lowest alignment_in_subspace) ===')
    print('       (W maps top input subspace OUT of top output subspace — active concentrators)')
    for r in by_align[:8]:
        print(f"       {r['alignment_in_subspace']:.3f}  eff_rank_y={r['eff_rank_y']:6.1f}  {r['name']}")

    print()
    log('[diag] === TOP 8 most ALIGNED layers (highest alignment_in_subspace) ===')
    print('       (W maps top input subspace INTO top output subspace — passive)')
    for r in by_align[-8:][::-1]:
        print(f"       {r['alignment_in_subspace']:.3f}  eff_rank_y={r['eff_rank_y']:6.1f}  {r['name']}")
