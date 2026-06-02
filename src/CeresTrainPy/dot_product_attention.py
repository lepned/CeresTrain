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
import math
import numpy as np

import torch

from einops import einsum, rearrange, repeat
from rms_norm import RMSNorm, make_norm

from activation_functions import Swish, ReLUSquared
from lora import LoRALinear

# Attention-LoRA gate. Reads CERES_LORA_ATTN_RANK_DIV first (specific knob),
# falls back to CERES_LORA_TRANSFORMER_RANK_DIV (legacy unified knob) for
# backward compatibility. Allows attention-only transformer-LoRA experiments
# (combine with CERES_LORA_FFN_RANK_DIV=0 in mlp2_layer.py).
# Optional layer-range gating via CERES_LORA_LAYER_MIN / CERES_LORA_LAYER_MAX
# (inclusive, 0-indexed). Layers outside the range receive no LoRA.
def _maybe_wrap_lora(layer, layer_num=None):
    n_attn  = os.environ.get("CERES_LORA_ATTN_RANK_DIV")
    n_legacy = os.environ.get("CERES_LORA_TRANSFORMER_RANK_DIV", "0")
    n = int((n_attn if n_attn is not None else n_legacy) or "0")
    if n <= 0:
      return layer
    if layer_num is not None:
      lo = os.environ.get("CERES_LORA_LAYER_MIN")
      hi = os.environ.get("CERES_LORA_LAYER_MAX")
      if lo is not None and layer_num < int(lo):
        return layer
      if hi is not None and layer_num > int(hi):
        return layer
    return LoRALinear(layer, n, True)


# Smolgen-LoRA gate. Wraps per-attention sm1/sm2/sm3 (and the shared
# smolgenPrepLayer in ceres_net.py) when CERES_LORA_SMOLGEN_RANK_DIV>0.
# Not subject to LAYER_MIN/MAX gating — smolgen is a network-wide attention
# bias mechanism, restricting it per-layer doesn't have a clean meaning.
def _maybe_wrap_smolgen_lora(layer):
    n = int(os.environ.get("CERES_LORA_SMOLGEN_RANK_DIV", "0") or "0")
    return LoRALinear(layer, n, True) if n > 0 else layer

class SmolgenPerLayerDelta(torch.nn.Module):
  """Per-layer low-rank zero-init delta added on top of the shared
  smolgenPrepLayer output.

  Variant A of the per-layer mini-smolgen experiment: the shared
  smolgenPrepLayer (the one architectural "concentration" point in the
  otherwise per-layer smolgen pipeline) is kept as-is; each layer additionally
  produces a small low-rank per-head correction `A_l @ B_l^T` from its own
  pre-prep per-head state, which is added to the prep-layer output before
  reshape.

  Zero-init: both W_A and W_B are zero-initialized so the delta is exactly 0
  at step 0 → model is bit-identical to baseline smolgen at init. Deltas grow
  from training signal if and only if per-layer output adaptation is useful;
  otherwise they remain near zero and the model degenerates to baseline.

  Parameters:
    sm_per_head_dim: per-head intermediate dim feeding smolgenPrepLayer
                     (smolgen_intermediate_dim // smolgen_head_divisor).
    num_heads: attention heads (delta is produced per head).
    num_tokens: square count (64 for chess).
    rank: low-rank factorization rank of the [64, 64] per-head delta.
    bottleneck: per-head compression dim before A/B projection.
  """
  def __init__(self, sm_per_head_dim, num_heads, num_tokens=64, rank=4, bottleneck=32):
    super().__init__()
    self.num_heads = num_heads
    self.num_tokens = num_tokens
    self.rank = rank
    # Per-head compression (weights shared across heads, applied to each
    # head's intermediate state independently).
    self.W_compress = torch.nn.Linear(sm_per_head_dim, bottleneck, bias=False)
    self.W_A = torch.nn.Linear(bottleneck, num_tokens * rank, bias=False)
    self.W_B = torch.nn.Linear(bottleneck, num_tokens * rank, bias=False)
    # LoRA-style init: zero-init only W_B; leave W_A and W_compress at
    # standard init. delta = (W_A·h) @ (W_B·h)^T = ... @ 0 = 0 at step 0
    # (bit-identical to baseline), but gradient flows: dL/dW_B is non-zero
    # because W_A is non-zero. Once W_B starts to move, W_A also receives
    # non-zero gradient. Double zero-init creates a dead-unit (neither
    # receives gradient) — that bug was caught by the post-train diagnostic
    # showing exact zeros on both matrices across all layers.
    torch.nn.init.xavier_uniform_(self.W_compress.weight)
    torch.nn.init.xavier_uniform_(self.W_A.weight)
    torch.nn.init.zeros_(self.W_B.weight)

  def forward(self, smolgen_per_head_state):
    # smolgen_per_head_state: [B, num_heads, sm_per_head_dim]
    h = self.W_compress(smolgen_per_head_state)                              # [B, H, bottleneck]
    A = self.W_A(h).reshape(-1, self.num_heads, self.num_tokens, self.rank)  # [B, H, T, r]
    B = self.W_B(h).reshape(-1, self.num_heads, self.num_tokens, self.rank)  # [B, H, T, r]
    delta = torch.matmul(A, B.transpose(-1, -2))                             # [B, H, T, T]
    return delta


class LinearWrapper:
  def __init__(self, linear_layer):
    self._layer = linear_layer

  @property
  def linear(self):
    return self._layer


class ParameterWrapper:
  def __init__(self, parameter):
    self._parameter = parameter

  @property
  def parameter(self):
    return self._parameter


class DotProductAttention(torch.nn.Module):
  """
  Implements (scaled) Dot Product Attention.

  Parameters:
      num_attention_heads (int): Number of attention heads in the module.
      kv_channels (int): Number of channels (dimensions) in each key and value vector.
      norm_type (str): Type of normalization to apply within the attention mechanism.
      layernorm_eps (float): Epsilon value for layer normalization to prevent division by zero.
      attention_multiplier (int, optional): Scaling factor for attention scores. Defaults to 1.
      smolgen_per_square_dim (int, optional): Dimensionality for Smolgen per-square processing. Defaults to 0.
      smolgen_intermediate_dim (int, optional): Intermediate dimensionality for Smolgen processing. Defaults to 0.
      smolgenPrepLayer: Optional layer for preprocessing in the Smolgen context.
  """
  def __init__(self, num_tokens_q : int, num_tokens_kv : int,
               num_attention_heads: int, kv_channels: int, norm_type : str, 
               layernorm_eps : float, 
               use_qkv : bool = True,
               softcap_cutoff : float = 0, 
               use_qk_norm : bool = False,
               attention_multiplier : int = 1,
               smolgen_per_square_dim : int = 0, smolgen_intermediate_dim : int = 0,
               smolgen_head_divisor : int = 1, smolgenPrepLayer = None,
               smolgen_activation_type : str = 'None',
               smolgen_delta_rank : int = 0,
               use_rpe : bool = False,
               use_rpe_v : bool = True,
               rpe_factor_shared  = None,
               use_rel_bias: bool = False,
               use_nonlinear_attention: bool = False,
               use_rope : bool = False,
               test : bool = False,
               layer_num : int = None,
               use_diff_attention : bool = False) -> None:
    super().__init__()

    self.num_tokens_q = num_tokens_q
    self.num_tokens_kv = num_tokens_kv
    self.num_heads = num_attention_heads
    self.attention_multiplier = attention_multiplier
    self.d_model = num_attention_heads * kv_channels
    self.d_output = num_attention_heads * kv_channels
    self.d_k = kv_channels
    self.softmax = torch.nn.Softmax(-1)
    self.smolgen_head_divisor = smolgen_head_divisor
    self.test = test    
    self.use_qkv = use_qkv
    self.use_smolgen = smolgenPrepLayer is not None    
    self.use_rpe = use_rpe
    self.use_rpe_v = use_rpe_v
    self.use_rel_bias = use_rel_bias
    self.use_rope = use_rope
    self.use_nonlinear_attention = use_nonlinear_attention
    self.use_qk_norm = use_qk_norm
    self.softcap_cutoff = softcap_cutoff
    self.layer_num = layer_num

    # smolgen + RoPE coexistence allowed: RoPE rotates Q/K before scores;
    # smolgen adds learned bias to scores after. Compose cleanly (verified 2026-05-22).

    if self.use_rope:
      from rope import precompute_rope_freqs
      d_per_head = kv_channels * attention_multiplier
      cos_table, sin_table = precompute_rope_freqs(d_per_head)
      # buffers, not parameters: move with module to GPU but no gradients
      self.register_buffer('rope_cos', cos_table, persistent=False)
      self.register_buffer('rope_sin', sin_table, persistent=False)
    
    if self.use_smolgen:
      if (smolgen_activation_type == 'None'):
        self.smolgen_activation_fn = torch.nn.Identity()
      elif (smolgen_activation_type == 'ReLU'):
        self.smolgen_activation_fn = torch.nn.ReLU()
      elif (smolgen_activation_type == 'ReLUSquared'):
        self.smolgen_activation_fn = ReLUSquared()
      elif (smolgen_activation_type == 'Swish'):
        self.smolgen_activation_fn = Swish()
      elif (smolgen_activation_type == 'SwiGLU'):
        self.smolgen_activation_fn = torch.nn.SiLU() # First of SwiGLU here
      else:
        raise Exception('Unknown activation type', smolgen_activation_type)


    # Implementations often but not always use no bias
    USE_BIAS = False

    if not self.use_qkv:
      assert self.use_smolgen, "smolgen must be used when not use_qkv"
      assert not self.use_nonlinear_attention, "nonlinear_attention not allowed when not use_qkv"

    # Fused Q, K, and V linear projection for improved efficiency.
    # DiffAttention V2 (Microsoft Apr 2026): doubles Q (Q1, Q2 split) while KV
    # stays single — produces two attention maps, subtracts with per-token
    # sigmoid(lambda) gate to cancel attention noise.
    self.use_diff_attention = use_diff_attention
    if self.use_diff_attention:
      assert self.use_qkv, "DiffAttention requires use_qkv"
      self.qkv_multiplier = 4  # Q1, Q2, K, V (works in both linear and nonlinear QKV paths)
    else:
      self.qkv_multiplier = 3 if self.use_qkv else 1 # only contains V if not using QKV
    self.qkv = _maybe_wrap_lora(torch.nn.Linear(self.d_model, self.qkv_multiplier * self.d_model * self.attention_multiplier, bias = True if self.use_nonlinear_attention else USE_BIAS), self.layer_num)
    if self.use_diff_attention:
      # Per-token per-head lambda gate. Sigmoid output gives lambda in [0,1].
      # Bias init -2.2 → sigmoid ≈ 0.1 at start → small differential subtraction
      # initially (~pure attn1), training can grow lambda as the noise-cancellation
      # signal becomes useful. Zero-init weight keeps lambda input-independent at
      # init, layer-dependent via the bias only.
      self.lambda_proj = torch.nn.Linear(self.d_model, num_attention_heads, bias=True)
      torch.nn.init.zeros_(self.lambda_proj.weight)
      torch.nn.init.constant_(self.lambda_proj.bias, -2.2)
    self.W_h = _maybe_wrap_lora(torch.nn.Linear(self.d_model * self.attention_multiplier, self.d_output), self.layer_num)

    if self.use_nonlinear_attention:
      self.qkvLN = make_norm(norm_type, self.d_model * self.attention_multiplier)
      self.q2 = _maybe_wrap_lora(torch.nn.Linear(self.d_model * self.attention_multiplier, self.d_model * self.attention_multiplier, bias=USE_BIAS), self.layer_num)
      self.k2 = _maybe_wrap_lora(torch.nn.Linear(self.d_model * self.attention_multiplier, self.d_model * self.attention_multiplier, bias=USE_BIAS), self.layer_num)
      self.v2 = _maybe_wrap_lora(torch.nn.Linear(self.d_model * self.attention_multiplier, self.d_model * self.attention_multiplier, bias=USE_BIAS), self.layer_num)
      if self.use_diff_attention:
        # Second Q projection for the differential head, mirrors q2.
        self.q2b = _maybe_wrap_lora(torch.nn.Linear(self.d_model * self.attention_multiplier, self.d_model * self.attention_multiplier, bias=USE_BIAS), self.layer_num)

    if self.use_qk_norm:
      # extra layernorm for enahnced training stability
      self.qLN = make_norm(norm_type, self.d_k * self.attention_multiplier)
      self.kLN = make_norm(norm_type, self.d_k * self.attention_multiplier)

    RPE_INNER_DIM = 16 # rounded up to power of 2 (there are only 15 possible values of a -  b where a and b are 0...7)

    if self.use_rpe:
      assert self.use_qkv, "rpe requires use_qkv"
      self.wrapped_rpe_factor_shared = ParameterWrapper(rpe_factor_shared) # wrap so shared layer is not re-registered
      self.rpe_q = torch.nn.Parameter(torch.zeros(self.d_k * self.attention_multiplier * self.num_heads, RPE_INNER_DIM * RPE_INNER_DIM))
      self.rpe_k = torch.nn.Parameter(torch.zeros(self.d_k * self.attention_multiplier * self.num_heads, RPE_INNER_DIM * RPE_INNER_DIM))
      self.rpe_v = torch.nn.Parameter(torch.zeros(self.d_k * self.attention_multiplier * self.num_heads, RPE_INNER_DIM * RPE_INNER_DIM)) if self.use_rpe_v else None

      torch.nn.init.kaiming_uniform_(self.rpe_q, a=0.1)
      torch.nn.init.kaiming_uniform_(self.rpe_k, a=0.1)
      if self.use_rpe_v:
        torch.nn.init.kaiming_uniform_(self.rpe_v, a=0.1)

    if self.use_rel_bias:
      self.rel_bias = torch.nn.Parameter(torch.zeros(self.num_heads, RPE_INNER_DIM * RPE_INNER_DIM))

    self.smolgen_per_square_dim = smolgen_per_square_dim
    self.smolgen_intermediate_dim = smolgen_intermediate_dim


    if self.use_smolgen:
      self.wrapped_smolgen_prep_layer = LinearWrapper(smolgenPrepLayer) # wrap so shared layer is not re-registered
      self.sm1 = _maybe_wrap_smolgen_lora(torch.nn.Linear(self.d_model, smolgen_per_square_dim))
      self.sm2 = _maybe_wrap_smolgen_lora(torch.nn.Linear(num_tokens_q * smolgen_per_square_dim, smolgen_intermediate_dim))
      self.ln1 = make_norm(norm_type, smolgen_intermediate_dim, eps=layernorm_eps)
      self.sm3 = _maybe_wrap_smolgen_lora(torch.nn.Linear(smolgen_intermediate_dim, num_attention_heads * smolgen_intermediate_dim // smolgen_head_divisor))
      self.ln2 = make_norm(norm_type, num_attention_heads * smolgen_intermediate_dim // smolgen_head_divisor, eps=layernorm_eps)

    # Variant A: per-layer low-rank zero-init delta added to the shared
    # smolgenPrepLayer output. Only active when smolgen is on AND rank > 0.
    self.use_smolgen_delta = self.use_smolgen and smolgen_delta_rank > 0
    if self.use_smolgen_delta:
      self.smolgen_delta = SmolgenPerLayerDelta(
        sm_per_head_dim=smolgen_intermediate_dim // smolgen_head_divisor,
        num_heads=num_attention_heads,
        num_tokens=num_tokens_q,
        rank=smolgen_delta_rank,
        bottleneck=32,
      )



  @property
  def smolgenPrepLayer(self):
    return self.wrapped_smolgen_prep_layer.linear

  @property
  def rpeFactorShared(self):
    return self.wrapped_rpe_factor_shared.parameter.data

  # Function to cap logit scores (as used in the grok and gemma models).
  def soft_cap(self, score, softcap):
    score = score / softcap
    score = torch.tanh(score)
    score = score * softcap
    return score

 
  def sdp_diff(self, Q1:torch.Tensor, Q2:torch.Tensor, K:torch.Tensor, V:torch.Tensor,
               smolgen:torch.Tensor, x:torch.Tensor,
               piece_relation_bias:torch.Tensor = None):
    """DiffAttention V2 (Microsoft Apr 2026): two attention maps from Q1 / Q2,
    differential subtraction with per-token sigmoid(lambda) gate cancels
    attention noise. Smolgen bias added to BOTH attention maps (Option A) —
    both branches inherit the same per-position smolgen prior; the differential
    cancels Q1-vs-Q2 noise on top of it. Softcap unsupported in this path
    (assert in __init__ if needed)."""
    # Two attention score matrices using the same K
    scores1 = torch.matmul(Q1, K.transpose(2, 3)) / math.sqrt(self.d_k)
    scores2 = torch.matmul(Q2, K.transpose(2, 3)) / math.sqrt(self.d_k)

    # Smolgen bias added to BOTH branches (Option A)
    if smolgen is not None:
      assert self.num_tokens_q == self.num_tokens_kv, "use_smolgen requires equal number of tokens for Q and K"
      scores1 = scores1 + smolgen
      scores2 = scores2 + smolgen

    # Piece-relation bias also added to both (same rationale)
    if piece_relation_bias is not None:
      prb = piece_relation_bias.to(scores1.dtype)
      scores1 = scores1 + prb
      scores2 = scores2 + prb

    attn1 = self.softmax(scores1)
    attn2 = self.softmax(scores2)

    # Per-token per-head lambda gate (sigmoid). x: [B, T, d_model].
    # lambda_proj outputs [B, T, H] → reshape to [B, H, T, 1] to broadcast over K-dim.
    lambda_t = torch.sigmoid(self.lambda_proj(x))           # [B, T, H]
    lambda_t = lambda_t.permute(0, 2, 1).unsqueeze(-1)      # [B, H, T, 1]

    attn = attn1 - lambda_t * attn2                          # [B, H, T_q, T_k]
    H = torch.matmul(attn, V)
    return H, attn

  def sdp_and_smol_or_rpe(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, smolgen:torch.Tensor, piece_relation_bias:torch.Tensor = None): # -> torch.Tensor, torch.Tensor:
    # Note that scaling could be done separately on Q and K to possibly improve stability. See:
    #   https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118
    #scaleDivisor = 1 # math.pow(self.d_k, 0.25) # apply sqrt twice since we are dividing twice
    #Q = Q / scaleDivisor
    #K = K / scaleDivisor

    if self.use_qkv:
      scores = torch.matmul(Q, K.transpose(2, 3))

    if self.use_rpe:
      rpe_q = self.rpe_q @ self.rpeFactorShared
      rpe_q = rpe_q.reshape(self.d_k * self.attention_multiplier, self.num_heads, 64, 64)

      rpe_k = self.rpe_k @ self.rpeFactorShared
      rpe_k = rpe_k.reshape(self.d_k * self.attention_multiplier, self.num_heads, 64, 64)
      
      scores = scores + einsum(Q, rpe_q, "b h q d, d h q k->b h q k")
      scores = scores + einsum(K, rpe_k, "b h k d, d h q k->b h q k")
      # consider using scaling below as (3 * self.d_k) due to extra terms
       
    if self.use_qkv:
      scores = scores / math.sqrt(self.d_k)

    if self.use_rel_bias:
      scores = scores + torch.reshape(self.rel_bias @ self.rpe_factor, [-1, 64, 64])

    if not self.use_qkv:
      scores = smolgen / math.sqrt(self.d_k)
    elif self.use_smolgen:
      assert self.num_tokens_q == self.num_tokens_kv, "use_smolgen requires equal number of tokens for Q and K"
      smolgen_logits_repeated = smolgen
      scores = scores + smolgen_logits_repeated

    # Plan 3: chess-specific piece-relation bias. Computed once per forward by
    # the parent network from the squares input and passed unchanged to every
    # encoder layer's attention. Shape [B, num_heads, 64, 64], same as scores.
    if piece_relation_bias is not None:
      scores = scores + piece_relation_bias.to(scores.dtype)

    if self.softcap_cutoff > 0:
      #softcap logits for enhanced training stability
      scores = self.soft_cap(scores, self.softcap_cutoff)

    A = self.softmax(scores)

    # Get the weighted average of the values
    H = torch.matmul(A, V)

    if self.use_rpe and self.use_rpe_v:
      rpe_v = self.rpe_v @ self.rpeFactorShared
      rpe_v = rpe_v.reshape(self.d_k * self.attention_multiplier, self.num_heads, 64, 64)
      
      H = H + einsum(A, rpe_v, "b h q k, d h q k->b h q d")

    return H, A
  

  def calc_smolgen(self, x:torch.Tensor) -> torch.Tensor:
    smolgen = self.sm1(x)
    smolgen = smolgen.reshape(-1, self.num_tokens_q * self.smolgen_per_square_dim)

    smolgen = self.sm2(smolgen)
    smolgen = self.smolgen_activation_fn(smolgen)
    smolgen = self.ln1(smolgen)

    smolgen = self.sm3(smolgen)
    smolgen = self.smolgen_activation_fn(smolgen)
    smolgen = self.ln2(smolgen)

    smolgen = smolgen.reshape(-1, self.num_heads, self.smolgen_intermediate_dim // self.smolgen_head_divisor)

    # Variant A: capture the per-head pre-prep state to feed the delta.
    # Compute delta BEFORE smolgenPrepLayer so both branches use the same
    # intermediate state. Branch is a no-op when delta is disabled.
    if self.use_smolgen_delta:
      delta = self.smolgen_delta(smolgen)  # [B, H, T, T]

    smolgen = self.smolgenPrepLayer(smolgen)
    smolgen = smolgen.reshape(-1, self.num_heads, self.num_tokens_q, self.num_tokens_q)

    if self.use_smolgen_delta:
      smolgen = smolgen + delta

    return smolgen


  def forward(self, x:torch.Tensor, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              piece_relation_bias: torch.Tensor = None) -> torch.Tensor:
    batch_size = query.size(0)

    qkv_x = query    

    # Linear projections (Q, K, V jointly).
    qkv = self.qkv(qkv_x)

    if not self.use_qkv:
      Q = None
      K = None
      V = qkv.reshape(batch_size, -1, self.num_heads, self.d_k * self.attention_multiplier)
      V = V.permute(0, 2, 1, 3)
    elif not self.use_nonlinear_attention:
      if self.use_diff_attention:
        # DiffAttention V2: 4-way split (Q1, Q2, K, V); Q is doubled.
        qkv = qkv.reshape(batch_size, -1, self.num_heads, 4*self.d_k * self.attention_multiplier)
        qkv = qkv.permute(0, 2, 1, 3)
        Q1, Q2, K, V = qkv.chunk(4, dim=-1)
        Q = (Q1, Q2)  # pass as tuple; sdp_diff will unpack
      else:
        # Split apart Q, K, V (with heads on the left)
        qkv = qkv.reshape(batch_size, -1, self.num_heads, 3*self.d_k * self.attention_multiplier)
        qkv = qkv.permute(0, 2, 1, 3)
        Q, K, V = qkv.chunk(3, dim=-1)
    else:
      # Idea of introducing nonlinearity in the QKV was proposed in:
      #   "Neural Attention : Enhancing QKV Calculation in Self-Attention Mechanism with Neural Networks"
      #   Muhan Zhang, 2023
      if self.use_diff_attention:
        # 4-way split: q1, q2, k, v through shared LN+Mish, then per-stream Linear projections.
        qkv = qkv.reshape(batch_size, -1, 4, self.d_model * self.attention_multiplier)
        qkv = self.qkvLN(qkv)
        qkv = torch.nn.functional.mish(qkv)
        q1, q2, k, v = torch.unbind(qkv, dim=-2)
        Q1 = self.q2 (q1).reshape(batch_size, -1, self.num_heads, self.d_k * self.attention_multiplier).permute(0, 2, 1, 3)
        Q2 = self.q2b(q2).reshape(batch_size, -1, self.num_heads, self.d_k * self.attention_multiplier).permute(0, 2, 1, 3)
        K  = self.k2 (k ).reshape(batch_size, -1, self.num_heads, self.d_k * self.attention_multiplier).permute(0, 2, 1, 3)
        V  = self.v2 (v ).reshape(batch_size, -1, self.num_heads, self.d_k * self.attention_multiplier).permute(0, 2, 1, 3)
        Q = (Q1, Q2)
      else:
        qkv = qkv.reshape(batch_size, -1, 3, self.d_model * self.attention_multiplier)
        qkv = self.qkvLN(qkv)
        qkv = torch.nn.functional.mish(qkv)
        q, k, v = torch.unbind(qkv, dim=-2)

        Q = self.q2(q).reshape(batch_size, -1, self.num_heads, self.d_k * self.attention_multiplier).permute(0, 2, 1, 3)
        K = self.k2(k).reshape(batch_size, -1, self.num_heads, self.d_k * self.attention_multiplier).permute(0, 2, 1, 3)
        V = self.v2(v).reshape(batch_size, -1, self.num_heads, self.d_k * self.attention_multiplier).permute(0, 2, 1, 3)

    if self.use_qk_norm:
      Q = self.qLN(Q)
      K = self.kLN(K)

    if self.use_rope:
      # Apply rotation to Q and K (not V). Position info is intrinsic to
      # rotated Q/K — no bias addition needed. Stays on the fast SDPA path.
      from rope import apply_rope
      Q = apply_rope(Q, self.rope_cos, self.rope_sin)
      K = apply_rope(K, self.rope_cos, self.rope_sin)

    if self.use_smolgen:
      smolgen = self.calc_smolgen(x)
      if self.use_diff_attention:
        Q1, Q2 = Q  # unpack tuple
        H_cat, A = self.sdp_diff(Q1, Q2, K, V, smolgen, qkv_x, piece_relation_bias=piece_relation_bias)
      else:
        H_cat, A = self.sdp_and_smol_or_rpe(Q, K, V, smolgen, piece_relation_bias=piece_relation_bias)
    else:
      # Always route through the explicit Q·Kᵀ → softmax → ·V form. The previous
      # branch called torch.nn.functional.scaled_dot_product_attention, which
      # PyTorch ≥ 2.10's dynamo ONNX exporter auto-fuses into the opset-23
      # `Attention` op — and TRT 10.15's Attention plugin requires the network
      # to be built in strongly-typed mode, which the C++ wrapper does not use,
      # so engine build aborts with API Usage Error 3.
      # The explicit form is mathematically equivalent (no mask, no dropout),
      # exports cleanly to opset 23 as MatMul→Softmax→MatMul, and also gains
      # softcap support that the F.sdpa path was lacking.
      if self.use_diff_attention:
        Q1, Q2 = Q  # unpack tuple
        H_cat, A = self.sdp_diff(Q1, Q2, K, V, None, qkv_x, piece_relation_bias=piece_relation_bias)
      else:
        H_cat, A = self.sdp_and_smol_or_rpe(Q, K, V, None, piece_relation_bias=piece_relation_bias)

    # Put all the heads back together by concat (with heads moved back to the right)
    H_cat =  H_cat.transpose(1, 2).contiguous().view(batch_size, -1, self.d_output * self.attention_multiplier)
      
    # Final linear layer  
    H = self.W_h(H_cat)

    return H



