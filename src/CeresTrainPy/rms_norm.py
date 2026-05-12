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
      rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
      return x / rms * self.scale


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
