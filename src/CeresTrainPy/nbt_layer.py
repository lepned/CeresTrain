# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

"""Nested bottleneck trunk block (KataGo `bottlenest3transformerropesg`, v1.17 main run).

    x (d) ─────────────────────────────────────────────┐
      └ norm → Linear d→d_mid                          │
          ├ EncoderLayer at d_mid (attention + FFN)    │  x NBTInnerLayers,
          ├ ...                                        │  each with its own residual
      └ norm → Linear d_mid→d  (ZERO-init)             │
    x + out ◄──────────────────────────────────────────┘

KataGo: "the nested-bottleneck design is substantially stronger than a plain stack at
equal cost, for both convnets and transformers". Inner layers run at half width (a
quarter of the matmul FLOPs each). With only 64 tokens the per-layer overhead (smolgen,
attention scores, norms) does not shrink with the width, so the cost-neutral point is
lower than the FLOP count says: measured on an untrained 1024x10 (TRT, M=80, 2026-09-28)
10 blocks x 2 inner and 7 blocks x 3 inner serve at 1.00x the plain net, 10 x 3 at 0.72x.

The zero-init up projection makes every block an exact identity at step 0; train.py keeps
it out of Muon (same class as the other zero-init couplings). The outer residual is
un-normalized (pre-norm style, as in KataGo), so ceres_net adds the trunk-end norm
whenever NBT is on. Same forward signature as EncoderLayer, so the trunk loop (LoopCount,
DenseFormer, depth-state collection, move-token trunk mix) is unchanged: all of those see
only the full-width block outputs. Inner layers are numbered as the real stack
(block * NBTInnerLayers + j), so per-layer settings are per layer, not per block.
"""

from typing import Callable, Tuple

import torch

from rms_norm import make_norm


class NestedBottleneckLayer(torch.nn.Module):
  def __init__(self, model_dim: int, mid_dim: int, make_inner_layer: Callable[[int], torch.nn.Module],
               num_inner_layers: int, norm_type: str, layernorm_eps: float):
    super().__init__()
    assert num_inner_layers >= 1, num_inner_layers
    self.model_dim = model_dim
    self.mid_dim = mid_dim
    self.norm_in = make_norm(norm_type, model_dim, eps=layernorm_eps)
    self.down = torch.nn.Linear(model_dim, mid_dim, bias=False)
    self.inner = torch.nn.ModuleList([make_inner_layer(j) for j in range(num_inner_layers)])
    self.norm_out = make_norm(norm_type, mid_dim, eps=layernorm_eps)
    self.up = torch.nn.Linear(mid_dim, model_dim, bias=False)
    torch.nn.init.zeros_(self.up.weight)


  def forward(self, x: torch.Tensor, piece_relation_bias: torch.Tensor = None,
              film: Tuple[torch.Tensor, torch.Tensor] = None,
              rpe_src: torch.Tensor = None,
              rpe_precomputed: bool = False,
              vis_edge: torch.Tensor = None) -> torch.Tensor:
    # rpe_src is the full-width post-embedding state; the inner attentions project it with
    # their own d_mid-wide qkv, so it cannot be passed through (ceres_net rejects the
    # combination). piece_relation_bias is per head and width-independent: every inner
    # layer gets it, as every plain layer would.
    assert rpe_src is None, 'NBT: rpe_src (RPE from embedding) is not supported inside a nested bottleneck block'
    h = self.down(self.norm_in(x))
    for layer in self.inner:
      h = layer(h, piece_relation_bias=piece_relation_bias, rpe_precomputed=rpe_precomputed, vis_edge=vis_edge)
    out = self.up(self.norm_out(h))
    # Phase-FiLM is [B, 1, model_dim]: applied to the block's full-width branch output
    # before the residual add -- as a pre-norm plain layer applies it to its FFN output
    # before `out1 + mlp`. Like there, a learned beta lands on the un-normalized stream
    # (it is zero-init, so the block stays an identity at step 0).
    if film is not None:
      out = out * (1.0 + film[0]) + film[1]
    return x + out
