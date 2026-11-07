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
at 0.71x its trunk FLOPs. Remaining deliberate deviations: smolgen (per inner layer, or one
per block with NBTSharedSmolgen) instead of 2D RoPE (learnable 2D RoPE exists: RoPELearnable),
SoftCap/QK-clip on the inner attentions, a per-token RMSNorm (no activation) at the trunk end, biases on W_h and the FFN linears, no beta before the
projection activations.
"""

from typing import Callable, Tuple

import math
import torch

from activation_functions import to_activation
from rms_norm import make_norm, ChannelAffine
from dot_product_attention import LinearWrapper
from egt_edge import edge_gate_bias


def gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
  """x [B, 64, C], idx [B, K] square indices -> [B, K, C]."""
  return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def scatter_tokens(base: torch.Tensor, idx: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
  """Out-of-place: base [B, 64, C] with rows idx [B, K] replaced by src [B, K, C] (indices are distinct)."""
  return torch.scatter(base, 1, idx.unsqueeze(-1).expand(-1, -1, base.shape[-1]), src.to(base.dtype))


def gather_pair_bias(bias: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
  """Per-head square-pair bias [B, H, 64, 64] -> [B, H, K, K] at (idx_i, idx_j): rows, then columns."""
  B, H, T, _ = bias.shape
  K = idx.shape[1]
  rows = torch.gather(bias, 2, idx.reshape(B, 1, K, 1).expand(-1, H, -1, T))
  return torch.gather(rows, 3, idx.reshape(B, 1, 1, K).expand(-1, H, K, -1))


class SharedSmolgen(torch.nn.Module):
  """One smolgen generator per NBT block (config NBTSharedSmolgen), shared by its K inner layers.

  Same three stages as the per-layer generator in dot_product_attention.smolgen (sm1 per square,
  sm2 over the flattened board, sm3 to a per-head vector, activation + norm after sm2/sm3, then
  the GLOBAL smolgenPrepLayer to a 64x64 logit bias per head), but it reads the BLOCK INPUT --
  the full-width residual stream after the block's input norm (256/512/1536 wide) instead of
  the K x narrower inner states -- and its bias is added to every inner attention's logits
  (via the piece_relation_bias path: the same MatMul -> Add -> softmax graph as smolgen itself).
  Motivation (09-30/10-01): the per-layer generators do not shrink with the inner width (sm2 is
  2048 -> SmolgenDim whatever the width): 30 of them are 40 % of the params on the 512x10x3 net
  and 60 % on the 256x10x3 smoke net, and ~2/3 of the smolgen kernels at serving. The block
  input is the richer signal (3x wider than the inner state on the 1536/3 net). Whether
  per-layer adaptivity matters is what the paired smoke measures (NB the shared arm is also
  the smaller net: 20M vs 33M params on 256x10x3).
  `hidden_dim` = the sm2 width (NBTSharedSmolgenDim, default SmolgenDim; may be wider than the
  per-layer one since there are K x fewer generators); sm3 must still emit
  heads x (SmolgenDim // SmolgenToHeadDivisor) because the prep layer is shared with that width."""
  def __init__(self, model_dim: int, num_heads: int, per_square_dim: int, hidden_dim: int, prep_in_dim: int,
               prep_layer: torch.nn.Linear, activation: str, norm_type: str, layernorm_eps: float, num_tokens: int = 64):
    super().__init__()
    assert isinstance(prep_layer, torch.nn.Linear), f'SharedSmolgen: prep layer must be a plain nn.Linear (LoRA-wrapped smolgen is not supported), was {type(prep_layer).__name__}'
    assert prep_layer.in_features == prep_in_dim and prep_layer.out_features == num_tokens * num_tokens, (
      f'shared smolgenPrepLayer must be Linear({prep_in_dim} -> {num_tokens * num_tokens}), was {prep_layer}')
    self.num_heads = num_heads
    self.num_tokens = num_tokens
    self.per_square_dim = per_square_dim
    self.prep_in_dim = prep_in_dim
    self.sm1 = torch.nn.Linear(model_dim, per_square_dim)
    self.sm2 = torch.nn.Linear(num_tokens * per_square_dim, hidden_dim)
    self.ln1 = make_norm(norm_type, hidden_dim, eps=layernorm_eps)
    self.sm3 = torch.nn.Linear(hidden_dim, num_heads * prep_in_dim)
    self.ln2 = make_norm(norm_type, num_heads * prep_in_dim, eps=layernorm_eps)
    assert activation in ('None', 'ReLU', 'ReLUSquared', 'Swish', 'SwiGLU'), f'SharedSmolgen: activation {activation!r} not in the per-layer smolgen set'
    self.act = to_activation(activation)
    self._prep = LinearWrapper(prep_layer)     # not a submodule: the prep layer is registered once, at the net root

  @property
  def prep_layer(self):
    return self._prep.linear

  def forward(self, x: torch.Tensor, token_idx: torch.Tensor = None, fill: torch.Tensor = None,
              num_registers: int = 0) -> torch.Tensor:
    """x: [B, 64, model_dim] (block input, normalized) -> [B, heads, 64, 64] logit bias.
    Piece tokens: x is [B, K, model_dim] at squares token_idx [B, K]; sm1 is scattered back to its squares (zeros
    elsewhere = 'no token') so sm2 sees the fixed square layout, and the bias is returned at the K tokens [B, heads, K, K].
    fill [B, 64, model_dim] (PieceTokenSmolgenFill 'stale'): the non-token squares take sm1 of this (normalized) 64-square
    state instead of zeros -- what a converted 64-square net's smolgen was trained on.
    num_registers (RegisterTokens): the last num_registers tokens of x are registers; the generator reads only the
    squares and the bias is zero-padded on the register rows/columns ([B, heads, T+R, T+R])."""
    if num_registers:
      x = x[:, :x.shape[1] - num_registers]
    s = self.sm1(x)
    if token_idx is not None:
      # one-hot matmul, not ScatterElements into zeros: the zeros' ConstantOfShape stays fp32 through the fp16
      # export conversion and TRT then rejects the mixed-type scatter (11-07); [B,64,K] @ [B,K,S] is as cheap.
      onehot = (token_idx.unsqueeze(1) == torch.arange(self.num_tokens, device=s.device).reshape(1, -1, 1)).to(s.dtype)
      s = torch.matmul(onehot, s)
      if fill is not None:
        s = s + self.sm1(fill) * (1.0 - onehot.sum(dim=-1, keepdim=True))
    s = s.reshape(-1, self.num_tokens * self.per_square_dim)
    s = self.ln1(self.act(self.sm2(s)))
    s = self.ln2(self.act(self.sm3(s)))
    s = s.reshape(-1, self.num_heads, self.prep_in_dim)
    bias = self.prep_layer(s).reshape(-1, self.num_heads, self.num_tokens, self.num_tokens)
    if token_idx is not None:
      bias = gather_pair_bias(bias, token_idx)
    if num_registers:
      bias = torch.nn.functional.pad(bias, (0, num_registers, 0, num_registers))
    return bias


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
               proj_norm: str = 'Norm', down_init_scale: float = 1.0, block_index: int = 0,
               shared_smolgen: torch.nn.Module = None, proj_activation_shift: bool = False, mod_capacity: int = 0):
    super().__init__()
    assert num_inner_layers >= 1, num_inner_layers
    # NBTSharedSmolgen: one generator per block on the (normalized) block input; the inner
    # layers are then built WITHOUT their own smolgen and receive this bias instead.
    self.shared_smolgen = shared_smolgen
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
    # NBTProjActivationShift: per-token mean over channels subtracted after the activation.
    self.proj_activation_shift = proj_activation_shift
    self.inner_heads = self.inner[0].num_attention_heads
    # NBTMoDCapacity (Mixture-of-Depths): a router on the normalized block input picks mod_capacity squares; only
    # they run the block, scaled by 2*sigmoid(score) (1 at a zero score; the gradient path that trains the router).
    # Init std 0.02: scores ~N(0, ~0.9) on the normalized input, i.e. random routing and gates ~0.5-1.5 at step 0
    # (harmless: the block's up projection is zero-init).
    self.mod_capacity = mod_capacity
    if mod_capacity > 0:
      self.mod_router = torch.nn.Linear(model_dim, 1)
      torch.nn.init.normal_(self.mod_router.weight, std=0.02)
      torch.nn.init.zeros_(self.mod_router.bias)
    if self.shared_smolgen is not None:
      assert self.shared_smolgen.num_heads == self.inner_heads, 'shared smolgen head count must equal the inner head count'
      assert not any(getattr(l.attention, 'use_smolgen', False) for l in self.inner), (
        'NBTSharedSmolgen: inner layers must be built without their own smolgen')


  def _shift(self, a: torch.Tensor) -> torch.Tensor:
    return a - a.mean(dim=-1, keepdim=True) if self.proj_activation_shift else a


  def fold_activation_shift_(self) -> bool:
    """Export-time fold of the shift into the consuming Linears (in place, for an export copy):
    W (a - mean_c(a)) = (W - W.mean(dim=1)) a, so the graph needs no ReduceMean/Sub."""
    if not self.proj_activation_shift:
      return False
    with torch.no_grad():
      for lin in (self.down, self.up):
        w = lin.weight.float()
        lin.weight.copy_((w - w.mean(dim=1, keepdim=True)).to(lin.weight.dtype))
    self.proj_activation_shift = False
    return True


  def forward(self, x: torch.Tensor, piece_relation_bias: torch.Tensor = None,
              film: Tuple[torch.Tensor, torch.Tensor] = None,
              rpe_src: torch.Tensor = None,
              rpe_precomputed: bool = False,
              vis_edge: torch.Tensor = None,
              edge_scales = None, want_logits: bool = False, edge_gates = None, token_idx: torch.Tensor = None,
              token_fill: torch.Tensor = None, num_registers: int = 0):
    # rpe_src is the full-width post-embedding state; the inner attentions project it with
    # their own d_mid-wide qkv, so it cannot be passed through (ceres_net rejects the
    # combination). piece_relation_bias is per head and width-independent: every inner
    # layer gets it, as every plain layer would.
    assert rpe_src is None, 'NBT: rpe_src (RPE from embedding) is not supported inside a nested bottleneck block'
    # A per-head bias is built with the trunk head count; it only fits inner layers that use it.
    assert piece_relation_bias is None or piece_relation_bias.shape[1] == self.inner_heads, \
      f'NBT: per-head attention bias has {piece_relation_bias.shape[1]} heads, inner layers use {self.inner_heads} (NBTInnerHeads)'
    # token_idx: piece tokens (ceres_net) -- x holds the K selected squares; only the smolgen needs their squares.
    assert token_idx is None or self.mod_capacity == 0, 'NBT: piece tokens and MoD routing do not combine'
    assert not ((token_idx is not None or self.mod_capacity > 0) and (want_logits or edge_scales is not None or edge_gates is not None
                or piece_relation_bias is not None or vis_edge is not None)),       'NBT piece tokens / MoD: no EGT, edge or incoming per-head square-pair bias (they are 64x64)'
    # num_registers: the last num_registers tokens of x are register tokens (ceres_net RegisterTokens)
    assert not (num_registers and (self.mod_capacity > 0 or want_logits or edge_scales is not None or edge_gates is not None
                or piece_relation_bias is not None or vis_edge is not None)),       'NBT register tokens: no MoD, EGT, edge or incoming per-head square-pair bias (they are 64x64)'
    h0 = self.norm_in(x)
    if self.shared_smolgen is not None:
      # Block-level content bias, summed with any incoming per-head bias (piece-relation etc.).
      sb = self.shared_smolgen(h0, token_idx, None if token_fill is None else self.norm_in(token_fill), num_registers)
      piece_relation_bias = sb if piece_relation_bias is None else piece_relation_bias + sb.to(piece_relation_bias.dtype)
    mod_idx = None
    if self.mod_capacity > 0:
      score = self.mod_router(h0).squeeze(-1)                                     # [B, 64]
      mod_idx = torch.topk(score.float(), k=self.mod_capacity, dim=1).indices     # [B, C]
      mod_gate = 2.0 * torch.sigmoid(torch.gather(score, 1, mod_idx)).unsqueeze(-1)   # [B, C, 1]
      h0 = gather_tokens(h0, mod_idx)
      if piece_relation_bias is not None:
        piece_relation_bias = gather_pair_bias(piece_relation_bias, mod_idx)
    h = self.down(self._shift(self.act_in(h0)))
    logits = None
    for j, layer in enumerate(self.inner):
      _eg = None if edge_gates is None else edge_gates[j]
      if want_logits and j == len(self.inner) - 1:
        logits = self._raw_logits(layer, h, piece_relation_bias, edge_scales, _eg)
      h = layer(h, piece_relation_bias=piece_relation_bias, rpe_precomputed=rpe_precomputed, vis_edge=vis_edge,
                edge_scales=edge_scales, edge_gate=_eg)
    out = self.up(self._shift(self.act_out(self.norm_out(h))))
    # Phase-FiLM is [B, 1, model_dim]: applied to the block's full-width branch output
    # before the residual add -- as a pre-norm plain layer applies it to its FFN output
    # before `out1 + mlp`. Like there, a learned beta lands on the un-normalized stream
    # (it is zero-init, so the block stays an identity at step 0).
    if film is not None:
      out = out * (1.0 + film[0]) + film[1]
    if mod_idx is not None:
      # unrouted squares: identity; routed: x + gate * block output
      return x + scatter_tokens(torch.zeros_like(x), mod_idx, out * mod_gate.to(out.dtype))
    if want_logits:
      return x + out, logits
    return x + out

  @staticmethod
  def _raw_logits(layer, h, piece_relation_bias, edge_scales, edge_gate=None):
    """Pre-softmax scores [B,H,64,64] of one inner layer, for the EGT edge-stream readback (egt_edge.py). Recomputes
    Q/K with the layer's own projection (one extra qkv GEMM per site) instead of returning them from the attention,
    which keeps the attention's fused score path and its signature unchanged. Standard QKV attention only."""
    att = layer.attention
    assert (att.use_qkv and not att.use_nonlinear_attention and not att.use_diff_attention and not att.use_rope
            and not att.use_rpe and not att.use_smolgen), 'EGT readback: plain QKV inner attention only'
    inp = layer.ln1(h) if layer.pre_norm else h
    B = inp.shape[0]
    dk = att.d_k * att.attention_multiplier
    qkv = att.qkv(inp).reshape(B, -1, att.num_heads, 3 * dk).permute(0, 2, 1, 3)
    Q, K, _ = qkv.chunk(3, dim=-1)
    if att.use_qk_norm:
      Q, K = att.qLN(Q), att.kLN(K)
    if edge_scales is not None:
      Q = Q * edge_scales[0].unsqueeze(-1).to(Q.dtype)
      K = K * edge_scales[1].unsqueeze(-1).to(K.dtype)
    s = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(dk)
    if piece_relation_bias is not None:
      s = s + piece_relation_bias.to(s.dtype)
    if edge_gate is not None:
      _gb = edge_gate_bias(Q, K, edge_gate)
      if _gb is not None:
        s = s + _gb.to(s.dtype)
    # same post-bias steps as sdp_and_smol_or_rpe (review 2026-10-07): the readback must see exactly the scores the
    # softmax saw, so the per-head logit temperature and the softcap are applied here too
    if att.use_head_logit_temp:
      s = s * torch.exp(att.head_logit_temp.clamp(-2.0, 2.0)).reshape(1, att.num_heads, 1, 1).to(s.dtype)
    if att.softcap_cutoff > 0:
      s = att.soft_cap(s, att.softcap_cutoff)
    return s
