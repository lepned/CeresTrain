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

class DyTNorm(torch.nn.Module):
  """Dynamic Tanh normalization (DyT). Same family as DerfNorm but uses
  tanh as the squash function instead of erf.

  Per Zhu et al. "Transformers without Normalization" (2024), this is the
  paper's exact recipe: gamma * tanh(alpha * x) + beta. Tanh saturates
  more gently than erf (gentler outlier clipping). DyT is the reference
  no-reduction normalization method in the literature.

  Same hardware story as DerfNorm: pure pointwise, no reductions, fuses
  with surrounding elementwise ops in TRT FP16/FP8.

  Parameters:
    alpha:  learnable squash-strength. Init from env var DYT_ALPHA_INIT
            (default 0.5 — paper's pre-norm recommendation). Shape is
            scalar by default; DYT_ALPHA_PER_CHANNEL=1 makes it
            per-channel (matches RMSNorm's per-channel adaptive capacity).
    gamma:  per-channel learnable scale (init=1).
    beta:   per-channel learnable shift (init=0).
  """
  def __init__(self, d_model: int, eps: float = 1e-6):
    super().__init__()
    self.d_model = d_model
    alpha_init = float(os.environ.get('DYT_ALPHA_INIT', '0.5'))
    per_channel = os.environ.get('DYT_ALPHA_PER_CHANNEL', '0') == '1'
    if per_channel:
      self.alpha = torch.nn.Parameter(torch.full((d_model,), alpha_init))
    else:
      self.alpha = torch.nn.Parameter(torch.tensor(alpha_init))
    self.gamma = torch.nn.Parameter(torch.ones(d_model))
    self.beta  = torch.nn.Parameter(torch.zeros(d_model))

  def forward(self, x: Tensor) -> Tensor:
    return self.gamma * torch.tanh(self.alpha * x) + self.beta
