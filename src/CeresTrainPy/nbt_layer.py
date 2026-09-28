# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

"""Nested bottleneck trunk block (KataGo NestedBottleneckTransformerBlock, model_pytorch.py;
their trained transformer nets tf2-b10c384 / tf3-b10c512 / tf3-b11c768 are nbt / nbt3).

    x (d) ─────────────────────────────────────────────────┐
      └ norm → [activation] → Linear d→d_mid                │
          ├ EncoderLayer at d_mid (attention + FFN)        │  x NBTInnerLayers,
          ├ ...                                            │  each with its own residual
      └ norm → [activation] → Linear d_mid→d  (ZERO-init)   │
    x + out ◄──────────────────────────────────────────────┘

[activation] = NBTProjActivation (None by default; KataGo uses SiLU = 'Swish'). KataGo's
NormMask carries a per-channel beta in front of the activation; RMSNorm has none and none is
added (judged not worth a knob, 09-28; the Affine norm has its own).

KataGo's evidence (docs/KataGoMethods.md, "Nested Bottleneck Residual Nets"): for their
convnets, at constant total compute, one inner conv per pair of 1x1 projections does not
pay back the projections, two do, three or four are slightly better. Inner layers here run
at half width (a quarter of the attention/FFN matmul FLOPs each), so K=2 is about 0.64x and
K=3 about 0.92x the trunk FLOPs of the plain stack at 256x10 (torch flop_counter, 09-28); a
K=2 arm is NOT a cost-matched comparison. With only 64 tokens the per-layer overhead
(smolgen, attention scores, norms) does not shrink with the width either: measured on an
untrained 1024x10 (TRT, M=80, 2026-09-28) 10 blocks x 2 inner and 7 blocks x 3 inner serve at
1.00x the plain net, 10 x 3 at 0.72x.

The zero-init up projection makes every block an exact identity at step 0 (KataGo's legacy
fixup mode; their fixscaleonenorm mode inits it normally). It is trained with Muon like the
rest of the trunk (see train.py: in AdamW the blocks stayed near identity for millions of
positions and the smoke learned clearly slower). The outer residual is un-normalized
(pre-norm style, as in KataGo), so ceres_net adds the trunk-end norm whenever NBT is on.
Same forward signature as EncoderLayer, so the trunk loop (LoopCount, DenseFormer,
depth-state collection, move-token trunk mix) is unchanged: all of those see only the
full-width block outputs. Inner layers are numbered as the real stack
(block * NBTInnerLayers + j), so per-layer settings are per layer, not per block.

KataGo-style options (compared against a local KataGo clone, model_pytorch.py
NestedBottleneckTransformerBlock): NBTInnerHeads (KataGo keeps head dim 32 at the inner width,
e.g. b4c256h4nbttflrs = mid 128 / 4 heads), NBTInnerPreNorm (their inner stream is pure
pre-norm: x + attn(norm(x)), x + ffn(norm(x))) and NBTProjActivation (their projections are
norm -> activation -> 1x1). The first port (8 heads at 128 = dim 16, halved again by Diff++
half-dim, post-norm inner layers, no projection activation) trailed the plain 256x10 smoke --
at 0.71x its trunk FLOPs. Remaining deliberate deviations: smolgen in every inner layer
instead of 2D RoPE, SoftCap/QK-clip on the inner attentions, a per-token RMSNorm (no
activation) at the trunk end, biases on W_h and the FFN linears, no beta before the
projection activations.
"""

from typing import Callable, Tuple

import torch

from activation_functions import to_activation
from rms_norm import make_norm, ChannelAffine


def _make_proj_norm(proj_norm: str, norm_type: str, d: int, eps: float, fixed_scale: float) -> torch.nn.Module:
  # 'Affine' = KataGo's fixscale NormMask: fixed_scale * (1 + gamma) * x + beta, no statistics;
  # the only true norms in a KataGo nbt transformer sit inside the attention/FFN sublayers.
  if proj_norm == 'Affine':
    return ChannelAffine(d, fixed_scale=fixed_scale)
  assert proj_norm == 'Norm', proj_norm
  return make_norm(norm_type, d, eps=eps)


class NestedBottleneckLayer(torch.nn.Module):
  def __init__(self, model_dim: int, mid_dim: int, make_inner_layer: Callable[[int], torch.nn.Module],
               num_inner_layers: int, norm_type: str, layernorm_eps: float, proj_activation: str = 'None',
               proj_norm: str = 'Norm', down_init_scale: float = 1.0, block_index: int = 0):
    super().__init__()
    assert num_inner_layers >= 1, num_inner_layers
    self.model_dim = model_dim
    self.mid_dim = mid_dim
    # NBTProjNorm: 'Norm' = the trunk's NormType (RMSNorm normalizes what KataGo's fixed
    # fixscale factor only rescales); 'Affine' = KataGo's fixscale constants, 1/sqrt(block+1)
    # in front of the down projection and 1/sqrt(K+1) in front of the up projection
    # (model_pytorch.py: normactconvp scale, NestedBottleneckTransformerBlock normactconvq scale).
    self.norm_in = _make_proj_norm(proj_norm, norm_type, model_dim, layernorm_eps, 1.0 / (block_index + 1) ** 0.5)
    self.down = torch.nn.Linear(model_dim, mid_dim, bias=False)
    # KataGo legacy fixup: the down projection is initialized at fixup_scale^(1/(1+K)) with
    # fixup_scale = 1/sqrt(num_blocks) (NestedBottleneckTransformerBlock.initialize).
    if down_init_scale != 1.0:
      with torch.no_grad():
        self.down.weight.mul_(down_init_scale)
    self.inner = torch.nn.ModuleList([make_inner_layer(j) for j in range(num_inner_layers)])
    self.norm_out = _make_proj_norm(proj_norm, norm_type, mid_dim, layernorm_eps, 1.0 / (num_inner_layers + 1) ** 0.5)
    self.up = torch.nn.Linear(mid_dim, model_dim, bias=False)
    torch.nn.init.zeros_(self.up.weight)
    # KataGo's projections are NormActConv: norm -> activation -> 1x1 (NBTProjActivation, same
    # vocabulary as FFNActivationType; 'None' gives an identity).
    self.act_in = to_activation(proj_activation)
    self.act_out = to_activation(proj_activation)
    self.inner_heads = self.inner[0].num_attention_heads


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
    # A per-head bias is built with the trunk head count; it only fits inner layers that use it.
    assert piece_relation_bias is None or piece_relation_bias.shape[1] == self.inner_heads, \
      f'NBT: per-head attention bias has {piece_relation_bias.shape[1]} heads, inner layers use {self.inner_heads} (NBTInnerHeads)'
    h = self.down(self.act_in(self.norm_in(x)))
    for layer in self.inner:
      h = layer(h, piece_relation_bias=piece_relation_bias, rpe_precomputed=rpe_precomputed, vis_edge=vis_edge)
    out = self.up(self.act_out(self.norm_out(h)))
    # Phase-FiLM is [B, 1, model_dim]: applied to the block's full-width branch output
    # before the residual add -- as a pre-norm plain layer applies it to its FFN output
    # before `out1 + mlp`. Like there, a learned beta lands on the un-normalized stream
    # (it is zero-init, so the block stays an identity at step 0).
    if film is not None:
      out = out * (1.0 + film[0]) + film[1]
    return x + out
