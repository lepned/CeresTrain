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

# Vis edge-bias B/C gate fusion crossover (see forward): fusing both gate terms
# into one E contraction was measured −11% TRT engine time at C=12 per-layer
# but +6% at C=4 (cat/split overhead exceeds the saved E-read); measured on
# RTX 5090 / TRT 10.16, B=512. Only C=12 and C=4 were measured — C=8 takes the
# fused path by extrapolation. Re-measure if TRT/GPU generation changes.
VIS_GATE_FUSE_MIN_CHANNELS = 8

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
               use_diff_attention : bool = False,
               vis_gate_channels : int = 0,
               vis_gate_mode : str = 'qk',
               graph_route_channels : int = 0,
               softmin_heads : int = 0,
               softmax_agg_heads : int = 0,
               use_head_logit_temp : bool = False) -> None:
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
      assert not use_rpe,           ('DiffAttention + RPE unsupported: sdp_diff never receives Q_rpe/K_rpe, so RPE '
           'would be SILENTLY dropped while its parameters still exist (unused). Set '
           'UseRPE false for diff arms, or extend sdp_diff first. (guard 2026-08-26)')
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
      # DIFF++ (2026-08-26, UseDiffAttention >= 2): de to komponentene fra
      # originalpapiret (2410.05258) som V2 mangler — (a) lambda-init-PLAN per
      # lag (0.2 -> 0.76 med dybden; papirets 0.8-0.6*exp(-0.3 l)) i stedet for
      # flat 0.1, (b) per-hode RMSNorm paa den kombinerte differansen skalert
      # med (1-lambda_init) (papirets GroupNorm — rapportert viktig for
      # stabilitet og gevinst). Delt K beholdes fra V2.
      self.diff_pp = int(use_diff_attention) >= 2
      if self.diff_pp:
        _l = self.layer_num if self.layer_num is not None else 0
        _p = 0.8 - 0.6 * math.exp(-0.3 * _l)
        torch.nn.init.constant_(self.lambda_proj.bias, math.log(_p / (1.0 - _p)))
        self.diff_norm = RMSNorm(self.d_k * self.attention_multiplier)  # HUS-norm, IKKE nn.RMSNorm: fused aten-op mangler ONNX-oversettelse og TRT kjoerer den i ren FP16 (se rms_norm.py)
        self.diff_norm_scale = 1.0 - _p
        if not self.layer_num:
          print(f'[dot_product_attention] DIFF++ enabled: lambda-init-plan per lag (l0 p=0.2 -> l9 p~0.76) '
                f'+ per-hode RMSNorm * (1-lambda_init) paa differansen')
    self.W_h = _maybe_wrap_lora(torch.nn.Linear(self.d_model * self.attention_multiplier, self.d_output), self.layer_num)

    # Gated attention output (Kimi K3 "Gated MLA" / Qwen gated-attention family).
    # Env-gated: CERES_GATED_ATTENTION_OUTPUT=1 (default 0 = exactly current behavior).
    # Elementwise sigmoid gate computed from the attention INPUT, applied to the
    # concatenated head outputs before W_h. Softmax forces every head to put its
    # mass somewhere on every square; ungated heads therefore inject noise into the
    # residual stream on positions where they have nothing to say. The gate gives
    # each channel an explicit learned "stay silent here" option.
    # Init: zero weight + bias 4.0 -> gate = sigmoid(4) ~ 0.982 everywhere at step 0,
    # i.e. a near-identity, input-independent start (same pattern as lambda_proj);
    # channels close only where gradients ask for it.
    self.use_gated_attn_out = int(os.environ.get('CERES_GATED_ATTENTION_OUTPUT', '0') or 0) > 0
    if self.use_gated_attn_out:
      self.attn_out_gate = torch.nn.Linear(self.d_model, self.d_model * self.attention_multiplier, bias=True)
      torch.nn.init.zeros_(self.attn_out_gate.weight)
      torch.nn.init.constant_(self.attn_out_gate.bias, 4.0)
      if not self.layer_num:  # print once (layer 0 or None), not per layer
        print(f'[dot_product_attention] GATED ATTENTION OUTPUT enabled: elementwise sigmoid '
              f'gate [{self.d_model} -> {self.d_model * self.attention_multiplier}] per layer, '
              f'zero-init weight / bias 4.0 (gate~0.982 at step 0)')

    # Visibility edge-bias B/C content gates (Kovax visibility program,
    # VISIBILITY_PROGRAM.md sec. 4/16): per-layer, per-head LINEAR readout of
    # this layer's own Q (form B, query-gated) and/or K (form C, key-gated)
    # per-head content, contracted against the shared pairwise edge channels
    # E[b, q, k, c] and ADDED to the attention logits alongside the
    # content-free per-layer projection (form A) that lives in ceres_net:
    #   B: logits[h,q,k] += sum_c (Q[h,q,:] . gate_q[h,c,:]) * E[q,k,c]
    #   C: logits[h,q,k] += sum_c (K[h,k,:] . gate_k[h,c,:]) * E[q,k,c]
    # Zero-init => exact step-0 no-op; the terms are LINEAR in the gate params
    # so gradient flows from step 1 (no zero-times-zero trap — that only
    # afflicts the refuted multiplicative B*C form, which is deliberately not
    # implemented; B+C reached the same ceiling at half the complexity).
    # Gates read Q/K after qk_norm but BEFORE RoPE rotation (content, not
    # position, is what the gate is meant to condition on).
    self.vis_gate_channels = vis_gate_channels
    self.attack_gate_q = None
    self.attack_gate_k = None
    if vis_gate_channels > 0:
      assert self.use_qkv, 'vis edge gates require use_qkv'
      assert not use_diff_attention, 'vis edge gates unsupported with DiffAttention (Q is a tuple)'
      assert vis_gate_mode in ('q', 'k', 'qk'), f'bad vis_gate_mode: {vis_gate_mode}'
      # The raw-E gate formulations in forward contract with matmul batch dims
      # (B, Nq) resp. (B, Nk) against the SAME square E — self-attention only.
      assert num_tokens_q == num_tokens_kv, \
          'vis edge gates require num_tokens_q == num_tokens_kv (square E)'
      _dkm = self.d_k * self.attention_multiplier
      if 'q' in vis_gate_mode:
        self.attack_gate_q = torch.nn.Parameter(torch.zeros(self.num_heads, vis_gate_channels, _dkm))
      if 'k' in vis_gate_mode:
        self.attack_gate_k = torch.nn.Parameter(torch.zeros(self.num_heads, vis_gate_channels, _dkm))
        # out<->in channel swap: lets the C-term contract gate_k against the
        # raw shared E (see forward). The permutation and the emission-order
        # invariant it depends on are owned and construction-verified by
        # VisibilityChannels.
        from chess_geometry import VisibilityChannels
        self.register_buffer('vis_gate_swap_idx',
                             VisibilityChannels.out_in_swap_index(vis_gate_channels),
                             persistent=False)

    # Graph-route heads (2026-08 tactical program): per-head gated blend of the
    # softmax attention map with an EXACT row-stochastic routing matrix built
    # from the shared visibility edge channels E:
    #   A_hard[h] = rownorm( sum_c relu(w[h,c]) * E[:, :, c]  +  eps * I )
    #   A[h]      = A_soft[h] + tanh(g[h]) * (A_hard[h] - A_soft[h])
    # This upgrades the edge channels from a logit BIAS (which must out-shout
    # content scores inside softmax) to GUARANTEED routing: with the gate
    # open, information provably flows along attack/vis edges each layer, and
    # stacking layers gives exact multi-hop propagation over the attack graph
    # (exchange chains, coverage nets). Gate is tanh of a ZERO-INIT raw scalar
    # per head: exact step-0 no-op (the TSB/GTAB/vis convention) with full
    # gradient at init; the blend coefficient is sign-free during training
    # (negative = route AWAY from edges, the DiffAttention precedent for
    # signed attention). For gate in [0, 1] the blend stays row-stochastic.
    # relu on the mixture keeps A_hard nonnegative; the eps self-loop keeps
    # rows of empty squares well-defined. Post-softmax => no interaction with
    # QK-clip (which monitors pre-softmax logits) and composes with smolgen/
    # RPE/vis biases unchanged. Export: matmul/relu/tanh/div only.
    # NB LoRA-style two-phase start: dL/dw is ~0 while the gate is ~0, so the
    # gate moves first and w follows — the SmolgenPerLayerDelta pattern, not
    # a dead unit (the gate's own gradient is nonzero from step 1).
    self.graph_route_channels = graph_route_channels
    if graph_route_channels > 0:
      assert not use_diff_attention, 'graph-route heads unsupported with DiffAttention'
      self.graph_route_w = torch.nn.Parameter(
          torch.full((self.num_heads, graph_route_channels), 1.0 / graph_route_channels))
      self.graph_route_gate = torch.nn.Parameter(torch.zeros(self.num_heads))
      self.register_buffer('graph_route_eye', torch.eye(num_tokens_q), persistent=False)
      if not self.layer_num:
        print(f'[dot_product_attention] GRAPH-ROUTE HEADS enabled: {self.num_heads} gated '
              f'heads/layer over {graph_route_channels} edge channels, tanh gate zero-init (exact step-0 no-op)')

    # Soft-min ("AND-logic") value aggregation (2026-08 tactical program).
    # Coverage/safety facts are universally quantified — "no defender reaches
    # h7", "every flight square is covered" — and the softmax aggregation
    # H = A @ V can only express weighted ORs (means) over the attended set.
    # The measured check/flight value gain was exactly a HAND-CODED such AND
    # (flight = coverage closure of the enemy king's neighborhood); soft-min
    # heads make the AND learnable and motif-general instead. The FIRST
    # `softmin_heads` heads aggregate V with an attention-weighted soft
    # minimum in place of the weighted mean:
    #   H[i,c] = -(1/tau) * log( sum_j A[i,j] * exp(-tau * V[j,c]) )
    # Same A @ V' matmul as standard aggregation with exp before / log after,
    # so serving cost is ~0 (unlike the vis qk gates' E contraction). tau is
    # per-head learnable (log-parameterized, init tau=1): tau->0 recovers the
    # weighted mean exactly, tau->inf approaches the hard min over the
    # attention support, so each head interpolates mean<->min as training
    # asks. LSE max-shift for stability; computed in fp32 (tiny: k x 64 x d_k).
    # NOT a zero-init no-op: aggregation differs from step 0, so this is an
    # ARCH key (like NormType) — no resume fresh-init path, and strict load
    # correctly refuses warm starts across a SoftMinHeads config change.
    # Signed-tau generalization (T1.1): soft-min detects universally
    # quantified failures; the dual soft-MAX detects existential threats
    # ("SOME piece attacks h7" — one attacker suffices, the mean dilutes it).
    # Identical formula with tau < 0: heads [softmin_heads,
    # softmin_heads+softmax_agg_heads) use tau = -exp(log_tau), init -1.
    # Same ARCH-key semantics as SoftMinHeads (not a zero-init no-op).
    self.softmin_heads = softmin_heads
    self.softmax_agg_heads = softmax_agg_heads
    if softmin_heads > 0 or softmax_agg_heads > 0:
      assert not use_diff_attention, \
          'soft-agg heads unsupported with DiffAttention (differential A can be negative — log undefined)'
      assert graph_route_channels == 0, \
          ('soft-agg heads unsupported with graph-route heads: the route blend '
           'A + tanh(g)*(A_hard - A) can produce NEGATIVE attention entries for '
           'tanh(g) < 0, and the soft-agg log of an A-weighted sum silently NaNs on them')
      assert softmin_heads >= 0 and softmax_agg_heads >= 0 and \
          0 < softmin_heads + softmax_agg_heads <= self.num_heads, \
          (f'SoftMinHeads+SoftMaxAggHeads must total 1..{self.num_heads}, '
           f'got {softmin_heads}+{softmax_agg_heads}')
      if softmin_heads > 0:
        self.softmin_log_tau = torch.nn.Parameter(torch.zeros(softmin_heads))
      if softmax_agg_heads > 0:
        self.softmax_log_tau = torch.nn.Parameter(torch.zeros(softmax_agg_heads))
      if not self.layer_num:
        print(f'[dot_product_attention] SOFT-AGG HEADS enabled: {softmin_heads} soft-min '
              f'+ {softmax_agg_heads} soft-max of {self.num_heads} heads '
              f'(learnable per-head tau, init +1.0 / -1.0)')

    # Per-head logit temperature (2026-08 tactics toolbox T4.1): learnable
    # multiplicative sharpness on the fully-assembled pre-softmax logits.
    # Tactical attention must COMMIT (near-argmax routing); general attention
    # must blend — one global 1/sqrt(d_k) cannot serve both. temp =
    # exp(log_temp), init log 0 => temp 1 => EXACT step-0 bit-identity with
    # the baseline (unlike the soft-agg heads this IS a zero-effect init).
    # Applied BEFORE the QK-clip monitor stash, so QKClipTau sees the true
    # effective logits. ⚠ CAVEAT (review 2026-08-20 #8): the clip answers by
    # shrinking W_q/W_k only, while temp also scales the ADDITIVE bias terms
    # (smolgen/RPE/vis) the clip cannot touch — a persistently hot temp can
    # squeeze content attention toward the bias-only solution while the clip
    # counter looks healthy. Mechanism is parked (pht verdict); if revived,
    # consider clipping on the pre-temp content term instead.
    self.use_head_logit_temp = use_head_logit_temp
    if use_head_logit_temp:
      self.head_logit_temp = torch.nn.Parameter(torch.zeros(self.num_heads))
      if not self.layer_num:
        print(f'[dot_product_attention] PER-HEAD LOGIT TEMP enabled: {self.num_heads} heads/layer, '
              f'temp=exp(log_temp) init 1.0 (exact step-0 no-op), clamp exp(±2)')

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
    # NB: return the parameter itself, NOT .data. The factor is a fixed constant
    # (requires_grad=False) so this is training-neutral, but .data detaches it into
    # a constant that torch.export lifts as a FakeTensor → ONNX save fails
    # ("Cannot take content out from the FakeTensor ... lifted_tensor_0").
    return self.wrapped_rpe_factor_shared.parameter

  # Per-head QK-clip (Kimi K2 "MuonClip", retained in the K3 recipe): if a head's
  # observed max attention logit exceeds tau, rescale that head's Q and K
  # projection rows by sqrt(tau/max) so the QK^T logits shrink by tau/max.
  # Weight-level: zero inference-graph footprint, no train/export divergence,
  # inert once training is stable (gamma clamps to 1). Called from train.py
  # after optimizer.step(); consumes the _last_max_logit stash.
  # Layout notes (mirrors the per-head Muon spec builder):
  #   nonlinear path: q2/k2 rows are head-major blocks of d_k*mult
  #   linear path:    fused qkv rows are per-head [Q|K|V] (or [Q1|Q2|K|V]) chunks
  #                   of d_k*mult each — scale all chunks except the final V chunk
  #   V-only path (no QK) and LoRA-wrapped layers: skipped
  # Known partial coverage: RPE/smolgen/rel-bias additive logit terms are not
  # rescaled (Q/K scaling shrinks the RPE cross terms by sqrt(gamma) only);
  # the monitored max is the FULL pre-softcap logit, so clipping is conservative.
  # The same applies to piece_relation_bias and everything folded into it (PRB,
  # ray bias, vis edge bias + B/C gates). ⚠ With QK-clip ARMED this is a real
  # feedback risk, not just waste: once a head's bias-driven max exceeds tau,
  # gamma < 1 every step and the clip multiplicatively shrinks that head's Q/K
  # rows while the additive bias does not shrink at all — progressive decay of
  # content attention toward the bias-only solution. If QKClipTau is combined
  # with large learnable logit biases, the monitor should stash the max BEFORE
  # the bias add (or those runs should not arm QK-clip).
  @torch.no_grad()
  def apply_qk_clip(self, tau):
    """Returns number of heads clipped this call (0 if none / not applicable)."""
    m = getattr(self, '_last_max_logit', None)
    if m is None or not self.use_qkv:
      return 0
    if self.use_qk_norm:
      # qLN/kLN renormalize Q/K after projection, so weight scaling cannot
      # change the logits — QK-clip is structurally ineffective here.
      return 0
    self._last_max_logit = None
    gamma = (tau / m.clamp(min=1e-6)).clamp(max=1.0)
    clipped = int((gamma < 0.9999).sum().item())
    if clipped == 0:
      return 0
    s = gamma.sqrt()
    dkm = self.d_k * self.attention_multiplier
    if self.use_nonlinear_attention:
      layers = [self.q2, self.k2] + ([self.q2b] if self.use_diff_attention else [])
      if not all(isinstance(l, torch.nn.Linear) for l in layers):
        return 0  # LoRA-wrapped: skip rather than corrupt adapter/base split
      for lin in layers:
        for h in range(self.num_heads):
          lin.weight[h * dkm:(h + 1) * dkm].mul_(s[h])
    else:
      if not isinstance(self.qkv, torch.nn.Linear):
        return 0
      w = self.qkv.weight
      per_head = self.qkv_multiplier * dkm
      for h in range(self.num_heads):
        base = h * per_head
        for j in range(self.qkv_multiplier - 1):  # every Q and K chunk, never V
          w[base + j * dkm: base + (j + 1) * dkm].mul_(s[h])
    return clipped

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
    if getattr(self, 'diff_pp', False):
      H = self.diff_norm(H) * self.diff_norm_scale
    return H, attn

  def sdp_and_smol_or_rpe(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, smolgen:torch.Tensor, piece_relation_bias:torch.Tensor = None,
                          Q_rpe:torch.Tensor = None, K_rpe:torch.Tensor = None,
                          rpe_precomputed:bool = False,
                          graph_route = None): # -> torch.Tensor, torch.Tensor:
    # Note that scaling could be done separately on Q and K to possibly improve stability. See:
    #   https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118
    #scaleDivisor = 1 # math.pow(self.d_k, 0.25) # apply sqrt twice since we are dividing twice
    #Q = Q / scaleDivisor
    #K = K / scaleDivisor

    if self.use_qkv:
      scores = torch.matmul(Q, K.transpose(2, 3))

    # rpe_precomputed (RPE-fromEmb ARCHITECTED serving graph): the QK rpe terms
    # were computed in a generator phase by the parent (from static embedding
    # content) and arrive via piece_relation_bias — skip the in-attention einsums
    # entirely so the score path stays a plain fusable QK^T (+bias). rpe_v is
    # intentionally skipped too (measured dead weight, pvsmoke8).
    if self.use_rpe and not rpe_precomputed:
      rpe_q = self.rpe_q @ self.rpeFactorShared
      rpe_q = rpe_q.reshape(self.d_k * self.attention_multiplier, self.num_heads, 64, 64)

      rpe_k = self.rpe_k @ self.rpeFactorShared
      rpe_k = rpe_k.reshape(self.d_k * self.attention_multiplier, self.num_heads, 64, 64)
      
      # RPE-from-embedding experiment (2026-08): when Q_rpe/K_rpe are supplied
      # (projections of the LAYER-0 embedding through this layer's own qkv
      # weights), the RPE content-coupling reads STATIC input content instead of
      # layer-computed content. Measures how much of RPE's win needs per-layer
      # content — the discriminator between cheap once-materialized geometry
      # gating and expensive per-layer schemes.
      _Qr = Q if Q_rpe is None else Q_rpe
      _Kr = K if K_rpe is None else K_rpe
      scores = scores + einsum(_Qr, rpe_q, "b h q d, d h q k->b h q k")
      scores = scores + einsum(_Kr, rpe_k, "b h k d, d h q k->b h q k")
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

    # Per-head logit temperature (see __init__): scale the assembled logits
    # before the clip monitor and softcap, so both see effective magnitudes.
    if self.use_head_logit_temp:
      _temp = torch.exp(self.head_logit_temp.clamp(-2.0, 2.0)).reshape(1, self.num_heads, 1, 1)
      scores = scores * _temp.to(scores.dtype)

    # QK-clip monitor (K2 MuonClip / K3 recipe): stash the per-head max PRE-softcap
    # logit for the weight-level clip applied after the optimizer step (train.py).
    # Training-only attribute stash (placement-head pattern) — contributes NOTHING
    # to the eval/export graph, which is the whole point of weight-level clipping.
    if self.training and getattr(self, 'qk_clip_monitor', False):
      self._last_max_logit = scores.detach().amax(dim=(0, 2, 3))  # [num_heads]

    if self.softcap_cutoff > 0:
      #softcap logits for enhanced training stability
      scores = self.soft_cap(scores, self.softcap_cutoff)

    A = self.softmax(scores)

    # Graph-route heads (see __init__): row-stochastic blend toward the exact
    # attack-graph routing matrix. Applied POST-softmax; the downstream V
    # aggregation and the rpe_v term read the blended A (intended — routed
    # heads route everything they carry).
    if graph_route is not None:
      _A_hard, _lam = graph_route          # [B, H, T, T], [1, H, 1, 1]
      A = A + _lam.to(A.dtype) * (_A_hard.to(A.dtype) - A)

    # Get the weighted average of the values. Soft-min heads (see __init__)
    # aggregate with an attention-weighted soft minimum instead: the same
    # matmul against exp-transformed V, log after, fp32 for the
    # transcendentals (k heads x 64 x d_k — negligible).
    if self.softmin_heads > 0 or self.softmax_agg_heads > 0:
      # Signed tau per soft-agg head: +exp for soft-min heads, -exp for
      # soft-max heads. The LSE identity below is sign-agnostic (max-shift
      # keeps exp in range either way): tau<0 turns the soft-min into its
      # dual soft-max exactly.
      k_agg = self.softmin_heads + self.softmax_agg_heads
      taus = []
      if self.softmin_heads > 0:
        taus.append(torch.exp(self.softmin_log_tau.clamp(-4.0, 4.0)))
      if self.softmax_agg_heads > 0:
        taus.append(-torch.exp(self.softmax_log_tau.clamp(-4.0, 4.0)))
      tau = torch.cat(taus).reshape(1, k_agg, 1, 1)
      A_agg = A[:, :k_agg].float()
      V_agg = V[:, :k_agg].float()
      negtv = -tau * V_agg
      m = negtv.amax(dim=2, keepdim=True)              # max over source tokens j
      s = torch.matmul(A_agg, torch.exp(negtv - m))    # rows of A sum to 1 -> s in (0, 1]
      H_agg = -(torch.log(s.clamp_min(1e-20)) + m) / tau
      H = torch.cat([H_agg.to(V.dtype), torch.matmul(A[:, k_agg:], V[:, k_agg:])], dim=1)
    else:
      H = torch.matmul(A, V)

    if self.use_rpe and self.use_rpe_v and not rpe_precomputed:
      rpe_v = self.rpe_v @ self.rpeFactorShared
      rpe_v = rpe_v.reshape(self.d_k * self.attention_multiplier, self.num_heads, 64, 64)

      _k_agg0 = self.softmin_heads + self.softmax_agg_heads
      if _k_agg0 > 0:
        # Soft-agg heads (review finding #7): the rpe_v term is a WEIGHTED-MEAN
        # positional add — splicing it onto an LSE min/max aggregate corrupts
        # the quantifier semantics. Apply it to the plain-mean heads only.
        _rpe_add = einsum(A[:, _k_agg0:], rpe_v[:, _k_agg0:], "b h q k, d h q k->b h q d")
        H = torch.cat([H[:, :_k_agg0], H[:, _k_agg0:] + _rpe_add], dim=1)
      else:
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
              piece_relation_bias: torch.Tensor = None, rpe_src: torch.Tensor = None,
              rpe_precomputed: bool = False, vis_edge: torch.Tensor = None) -> torch.Tensor:
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

    # RPE-from-embedding experiment: project the layer-0 embedding through THIS
    # layer's own qkv weights and route the results into the RPE einsums only
    # (zero new params; standard-path only). See sdp_and_smol_or_rpe.
    Q_rpe = None; K_rpe = None
    if self.use_rpe and rpe_src is not None:
      assert self.use_qkv and not self.use_diff_attention, \
        'rpe_src experiment: standard or nonlinear QKV paths only'
      if self.use_nonlinear_attention:
        qkv_e = self.qkv(rpe_src).reshape(batch_size, -1, 3, self.d_model * self.attention_multiplier)
        qkv_e = torch.nn.functional.mish(self.qkvLN(qkv_e))
        _qe, _ke, _ = torch.unbind(qkv_e, dim=-2)
        Q_rpe = self.q2(_qe).reshape(batch_size, -1, self.num_heads, self.d_k * self.attention_multiplier).permute(0, 2, 1, 3)
        K_rpe = self.k2(_ke).reshape(batch_size, -1, self.num_heads, self.d_k * self.attention_multiplier).permute(0, 2, 1, 3)
      else:
        qkv_e = self.qkv(rpe_src)
        qkv_e = qkv_e.reshape(batch_size, -1, self.num_heads, 3*self.d_k * self.attention_multiplier).permute(0, 2, 1, 3)
        Q_rpe, K_rpe, _ = qkv_e.chunk(3, dim=-1)
      if self.use_qk_norm:
        Q_rpe = self.qLN(Q_rpe)
        K_rpe = self.kLN(K_rpe)

    # Visibility edge-bias B/C content gates (see __init__): contract this
    # layer's per-head Q/K content against the shared edge channels and fold
    # the result into piece_relation_bias (added to scores post-1/sqrt(d_k),
    # the same injection point as the source program). Matmul-only
    # formulations (no einsum) for export friendliness:
    #   gq[b,h,q,c] = Q[b,h,q,:] @ gate_q[h,c,:]^T
    #   B-term[b,h,q,k] = sum_c gq[b,h,q,c] * E[b,q,k,c]   (batch dims b,q)
    #   C-term[b,h,q,k] = sum_c gk[b,h,k,c] * E[b,q,k,c]   (batch dims b,k)
    # Both terms contract against the RAW shared E — no per-layer E permutes.
    # B-term: batch dim q is already E's leading square axis. C-term: the
    # transposed operand E.permute(0,2,1,3) equals E with out<->in channels
    # swapped (VisibilityChannels emits *_in = *_out^T), so the swap is folded
    # into gate_k via vis_gate_swap_idx (a constant gather on a parameter,
    # constant-folded at export) instead of materializing [B,64,64,C] per layer.
    if self.vis_gate_channels > 0 and vis_edge is not None:
      E = vis_edge.to(Q.dtype)                                    # [B, 64q, 64k, C]
      vis_gate_bias = None
      gqP = gkP = None
      if self.attack_gate_q is not None:
        gq = torch.matmul(Q, self.attack_gate_q.transpose(-1, -2).unsqueeze(0).to(Q.dtype))  # [B,H,64,C]
        gqP = gq.permute(0, 2, 3, 1)                              # [B, 64q, C, H]
      if self.attack_gate_k is not None:
        gk_w = self.attack_gate_k.index_select(1, self.vis_gate_swap_idx)  # [H, C, d], channels swapped
        gk = torch.matmul(K, gk_w.transpose(-1, -2).unsqueeze(0).to(K.dtype))  # [B,H,64,C]
        gkP = gk.permute(0, 2, 3, 1)                              # [B, 64k, C, H]
      if gqP is not None and gkP is not None and self.vis_gate_channels >= VIS_GATE_FUSE_MIN_CHANNELS:
        # qk mode, wide E: ONE E contraction for both terms — the [B,64,64,C]
        # re-read per matmul dominates the gate's serving cost when C is large,
        # so halving the number of E-reading matmuls halves it. First H outputs
        # read with batch dim = q (B-term), last H with batch dim = k (C-term).
        term = torch.matmul(E, torch.cat([gqP, gkP], dim=-1))     # [B, s1, s2, 2H]
        vis_gate_bias = (term[..., :self.num_heads].permute(0, 3, 1, 2)
                         + term[..., self.num_heads:].permute(0, 3, 2, 1))
      else:
        # B-term: batch dim = q. C-term: batch dim = k (channel-swapped gate).
        terms = []
        if gqP is not None:
          terms.append(torch.matmul(E, gqP).permute(0, 3, 1, 2))  # [B, H, 64q, 64k]
        if gkP is not None:
          terms.append(torch.matmul(E, gkP).permute(0, 3, 2, 1))  # [B, H, 64q, 64k]
        vis_gate_bias = terms[0] if len(terms) == 1 else terms[0] + terms[1]
      piece_relation_bias = vis_gate_bias if piece_relation_bias is None \
          else piece_relation_bias + vis_gate_bias

    if self.use_rope:
      # Apply rotation to Q and K (not V). Position info is intrinsic to
      # rotated Q/K — no bias addition needed. Stays on the fast SDPA path.
      from rope import apply_rope
      Q = apply_rope(Q, self.rope_cos, self.rope_sin)
      K = apply_rope(K, self.rope_cos, self.rope_sin)

    # Graph-route heads (see __init__): build the row-stochastic routing matrix
    # from the raw shared E once per layer. relu(w) mixture keeps entries
    # nonnegative; eps self-loop guarantees a nonzero row for squares with no
    # edges (empty board regions) so the normalization is well-defined.
    _graph_route = None
    if self.graph_route_channels > 0 and vis_edge is not None:
      _E_gr = vis_edge.to(self.graph_route_w.dtype)                         # [B, T, T, C]
      _M = torch.matmul(_E_gr, torch.relu(self.graph_route_w).transpose(0, 1))  # [B, T, T, H]
      _M = _M.permute(0, 3, 1, 2) + 1e-3 * self.graph_route_eye.to(_M.dtype)   # [B, H, T, T]
      _A_hard = _M / _M.sum(dim=-1, keepdim=True).clamp(min=1e-6)
      _lam = torch.tanh(self.graph_route_gate).reshape(1, self.num_heads, 1, 1)
      _graph_route = (_A_hard, _lam)

    if self.use_smolgen:
      smolgen = self.calc_smolgen(x)
      if self.use_diff_attention:
        Q1, Q2 = Q  # unpack tuple
        H_cat, A = self.sdp_diff(Q1, Q2, K, V, smolgen, qkv_x, piece_relation_bias=piece_relation_bias)
      else:
        H_cat, A = self.sdp_and_smol_or_rpe(Q, K, V, smolgen, piece_relation_bias=piece_relation_bias, Q_rpe=Q_rpe, K_rpe=K_rpe, rpe_precomputed=rpe_precomputed, graph_route=_graph_route)
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
        H_cat, A = self.sdp_and_smol_or_rpe(Q, K, V, None, piece_relation_bias=piece_relation_bias, Q_rpe=Q_rpe, K_rpe=K_rpe, rpe_precomputed=rpe_precomputed, graph_route=_graph_route)

    # Put all the heads back together by concat (with heads moved back to the right)
    H_cat =  H_cat.transpose(1, 2).contiguous().view(batch_size, -1, self.d_output * self.attention_multiplier)

    # Gated attention output (see __init__): per-channel sigmoid gate from the
    # attention input scales head outputs before the final projection.
    if self.use_gated_attn_out:
      H_cat = H_cat * torch.sigmoid(self.attn_out_gate(qkv_x))

    # Final linear layer
    H = self.W_h(H_cat)

    return H



