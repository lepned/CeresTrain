# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

import torch
from torch import Tensor

class RMSNorm(torch.nn.Module):
  def __init__(self, d_model : int, eps : float =1e-6):
    super().__init__()

    self.d_model = d_model
    self.eps = eps
    self.scale = torch.nn.Parameter(torch.ones(d_model))

  def forward(self, x : Tensor) -> Tensor:
    # Explicit decomposed RMSNorm (Pow→Mean→Add→rsqrt→Mul→Mul). Mathematically
    # identical to F.rms_norm (same scale, same eps).
    #
    # We deliberately do NOT use torch.nn.functional.rms_norm here: that lowers
    # to aten._fused_rms_norm, which ONNX export emits as a single fused opset-23
    # RMSNormalization op. TensorRT runs that fused op in PURE FP16 — it does NOT
    # auto-promote the mean(x^2) reduction to FP32 — so the unnormalized residual
    # stream (|x| can exceed 256) overflows FP16 inside the norm, producing
    # NaN/Inf that poisons the value head (garbage WDL) while the argmax policy
    # still looks plausible. The decomposed chain below is recognised and forced
    # FP32 by TensorRTWrapper's structural scale-constant marker, matching how
    # every working production net (opset≤18, decomposed norms) behaves.
    var = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(var + self.eps)
    if getattr(self, '_scale_folded', False):
      # export_folds.py folded `scale` into the consumer Linear's input columns.
      return x
    return x * self.scale


def make_norm(norm_type: str, d_model: int, eps: float = 1e-6) -> torch.nn.Module:
  """Factory for the configured normalization layer.

  Replaces the ad-hoc `LayerNorm if X else RMSNorm` ternary that appeared in
  ~11 sites across ceres_net, encoder_layer, dot_product_attention, and
  mlp2_layer. Adding a new norm type is now a single-file change here.

  Supported norm_type values:
    'LayerNorm' — torch.nn.LayerNorm (per-channel affine, mean+var stats)
    'RMSNorm'   — RMSNorm (per-channel scale, RMS stat only)
    'Derf'      — DerfNorm (DyT-style: gamma*erf(alpha*x)+beta, no stats)
    'DyT'       — DyTNorm (DyT-style: gamma*tanh(alpha*x)+beta, no stats)
  """
  if norm_type == 'LayerNorm':
    return torch.nn.LayerNorm(d_model, eps=eps)
  if norm_type == 'RMSNorm':
    return RMSNorm(d_model, eps=eps)
  if norm_type == 'Derf':
    from derf_norm import DerfNorm
    return DerfNorm(d_model, eps=eps)
  if norm_type == 'DyT':
    from dyt_norm import DyTNorm
    return DyTNorm(d_model, eps=eps)
  raise ValueError(f"Unknown norm_type: {norm_type!r} (expected one of "
                   "'LayerNorm', 'RMSNorm', 'Derf', 'DyT')")


# Every module class whose parameters are NORM GAINS (scale/gamma/alpha/beta).
# wd_partition.py routes params owned by these to no_decay by ownership, so a
# norm type added to make_norm above MUST be added here too (review 2026-09-11:
# the list used to live only in wd_partition.py and a new type would have been
# swept into `decay` by the trunk catch-all with no assert firing).
# L2NormScaled = SoftMoE normPhi (dormant: SMOE_USE_NORMALIZATION is hard-off),
# included so that path is right the day it is armed.
from derf_norm import DerfNorm
from dyt_norm import DyTNorm
from l2norm_scaled import L2NormScaled
NORM_MODULE_TYPES = (torch.nn.LayerNorm, RMSNorm, DerfNorm, DyTNorm, L2NormScaled)
