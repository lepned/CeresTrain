"""Unit tests for survival-loss modes in losses.py (exact-ply CE and bucket CE paths).

losses.py reads its CERES_SURVIVAL_* env vars at import time, so each mode runs in a
subprocess with its own environment. Run the driver with no args:

    python3 test_survival_loss.py

NOTE: two further modes (ordinal all-threshold BCE; piece-value square weighting) were
implemented, A/B-tested (t91s1k4i20M, 2026-07-07), found Pareto-negative, and removed
along with their tests. See SURVIVAL_TARGET_SPEC.md §6.1.
"""
import os
import subprocess
import sys

K = 8
C = K + 2  # classes: 0 empty, 1..K captured-at-ply, K+1 survives


def build_case():
  """Small deterministic batch: 2 positions x 64 squares.
  Returns (target [B,64], output [B,64,C])."""
  import torch
  torch.manual_seed(7)
  B = 2
  target = torch.zeros(B, 64, dtype=torch.uint8)
  # (square, fate): a handful of occupied squares; the rest stay empty (masked).
  layout = [(0, K + 1), (7, K + 1), (10, 2), (11, 2), (20, K + 1), (30, 5), (40, K + 1), (50, 1)]
  for sq, fate in layout:
    for b in range(B):
      target[b, sq] = fate
  output = torch.randn(B, 64, C, requires_grad=True)
  return target, output


def make_calc():
  import torch.nn as nn
  from losses import LossCalculator
  return LossCalculator(nn.Linear(4, 4))


def mode_exact():
  """No envs: loss must equal plain masked CE over all C classes."""
  import torch
  import torch.nn.functional as F
  target, output = build_case()
  lc = make_calc()
  loss = lc.survival_loss(target, output, False, 0.3)
  mask = target > 0
  expected = F.cross_entropy(output[mask].float(), target[mask].long())
  assert torch.allclose(loss, expected, atol=1e-6), f'{loss} vs {expected}'
  loss.backward()
  assert output.grad is not None and torch.isfinite(output.grad).all()
  print('exact OK')


def mode_bucket():
  """Buckets+capture-weight: must equal an independently computed bucket CE."""
  import torch
  import torch.nn.functional as F
  target, output = build_case()
  lc = make_calc()
  loss = lc.survival_loss(target, output, False, 0.3)
  # Independent reimplementation.
  bounds = [2, 4, 8]
  c2b = torch.zeros(C, dtype=torch.long)
  for c in range(1, K + 1):
    c2b[c] = next(i for i, bnd in enumerate(bounds) if c <= bnd)
  c2b[K + 1] = len(bounds)
  mask = target > 0
  om = output[mask].float()
  tm = c2b[target[mask].long()]
  bl = torch.stack([torch.logsumexp(om[:, (c2b == b).nonzero(as_tuple=True)[0]], dim=1)
                    for b in range(len(bounds) + 1)], dim=1)
  w = torch.ones(len(bounds) + 1)
  w[:len(bounds)] = 4.0
  expected = F.cross_entropy(bl, tm, weight=w)
  assert torch.allclose(loss, expected, atol=1e-6), f'{loss} vs {expected}'
  print('bucket OK')


MODES = {
  'exact': ({}, mode_exact),
  'bucket': ({'CERES_SURVIVAL_LOSS_BUCKETS': '2,4,8', 'CERES_SURVIVAL_CAPTURE_WEIGHT': '4'}, mode_bucket),
}


if __name__ == '__main__':
  if len(sys.argv) > 1:
    MODES[sys.argv[1]][1]()
  else:
    for name, (env, _) in MODES.items():
      full_env = {k: v for k, v in os.environ.items() if not k.startswith('CERES_SURVIVAL')}
      full_env.update(env)
      r = subprocess.run([sys.executable, os.path.abspath(__file__), name], env=full_env)
      if r.returncode != 0:
        print(f'FAIL: {name}')
        sys.exit(1)
    print('ALL SURVIVAL LOSS TESTS PASS')
