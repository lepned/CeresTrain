# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

import os
import torch
from torch import Tensor

class DerfNorm(torch.nn.Module):
  """DyT-style normalization replacement using erf as the squash function.

  Drop-in for LayerNorm/RMSNorm: same (d_model, eps) signature.

  Per Zhu et al. "Transformers without Normalization" (2024), normalization
  layers can be replaced with `gamma * tanh(alpha * x) + beta`, eliminating
  the variance-reduction op entirely. This variant uses erf instead of tanh
  (matched derivative at origin via tanh(2/sqrt(pi) * x) ≈ erf(x), but erf
  saturates marginally faster in the tail — sharper outlier clipping).

  Parameters:
    alpha:  learnable squash-strength. Init value read from env var
            DERF_ALPHA_INIT (default 1.5). Shape is scalar by default;
            DERF_ALPHA_PER_CHANNEL=1 makes it per-channel vector
            (matches RMSNorm's per-channel adaptive capacity — addresses
            the diagnosed deficit where scalar alpha couldn't track
            per-channel variance differentiation as training progressed).
    gamma:  per-channel learnable scale (init=1, like LayerNorm/RMSNorm affine).
    beta:   per-channel learnable shift (init=0).

  No reductions, no statistics — TRT/cudnn fuses to a single elementwise op
  chain on FP16, no synchronization overhead. eps argument is accepted for
  API compatibility with LayerNorm/RMSNorm but unused (no division).
  """
  def __init__(self, d_model: int, eps: float = 1e-6):
    super().__init__()
    self.d_model = d_model
    alpha_init = float(os.environ.get('DERF_ALPHA_INIT', '1.5'))
    per_channel = os.environ.get('DERF_ALPHA_PER_CHANNEL', '0') == '1'
    if per_channel:
      self.alpha = torch.nn.Parameter(torch.full((d_model,), alpha_init))
    else:
      self.alpha = torch.nn.Parameter(torch.tensor(alpha_init))
    self.gamma = torch.nn.Parameter(torch.ones(d_model))
    self.beta  = torch.nn.Parameter(torch.zeros(d_model))

  def forward(self, x: Tensor) -> Tensor:
    return self.gamma * torch.erf(self.alpha * x) + self.beta
