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
from activation_functions import to_activation
from rms_norm import RMSNorm, make_norm
from lora import LoRALinear

# FFN-LoRA gate. Reads CERES_LORA_FFN_RANK_DIV first (specific knob), falls
# back to CERES_LORA_TRANSFORMER_RANK_DIV (legacy unified knob) for backward
# compatibility. Set CERES_LORA_FFN_RANK_DIV=0 to disable FFN LoRA while
# keeping attention LoRA on (per AGZO-style attribution: FFN1 cross-layer
# coherent perturbation is the value-damaging component).
# Optional layer-range gating via CERES_LORA_LAYER_MIN / CERES_LORA_LAYER_MAX
# (inclusive, 0-indexed). Layers outside the range receive no LoRA.
def _maybe_wrap_lora(layer, layer_num=None):
    n_ffn  = os.environ.get("CERES_LORA_FFN_RANK_DIV")
    n_legacy = os.environ.get("CERES_LORA_TRANSFORMER_RANK_DIV", "0")
    n = int((n_ffn if n_ffn is not None else n_legacy) or "0")
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

# An intuitive explanation of why biases are important can be found in 
# the YouTube video "How might LLMs store facts" by 3Blue1Brown (at about 9:00).
USE_BIAS = True # Daniel Moore reported biases useful in FFN

MLP_GLOBAL_PER_SQUARE_DIVISOR = 8; # reduces DIM ==> DIM / MLP_GLOBAL_PER_SQUARE_DIVISOR before flatten
MLP_GLOBAL_DIVISOR = 1; # divisor used to determine size of model dimension versus concatenated global dimension
MLP_GLOBAL_LN_EPS = 1e-6


class MLP2Layer(torch.nn.Module):
  def __init__(self, model_dim: int, ffn_inner_dim: int, out_dim : int, activation_type : str, norm_type : str, use_global : bool, use_te : bool = False, layer_num : int = None) -> None:
    super().__init__()

    self.activation_type = activation_type
    self.use_te = use_te
    self.use_global = use_global
    self.layer_num = layer_num

    if self.use_te:
      import transformer_engine.pytorch as te
      from transformer_engine.common.recipe import Format, DelayedScaling

      # TODO: Lift restriction that activation function must be 'gelu' for TE
      self.te_mlp_ln = te.LayerNormMLP(model_dim, ffn_inner_dim, bias=USE_BIAS, 
                                       return_layernorm_output = True, activation='gelu') 
      fp8_format = Format.HYBRID  # E4M3 during forward pass, E5M2 during backward pass
      self.fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    
    else:
      if self.use_global:
        mlpGlobalPerSquare = model_dim // MLP_GLOBAL_PER_SQUARE_DIVISOR
        mlpGlobalDim = 64 * mlpGlobalPerSquare
        self.mlpGlobalSquareReduce = torch.nn.Linear(model_dim, mlpGlobalPerSquare, bias=USE_BIAS)
        self.mlpGlobalReduce = torch.nn.Linear(mlpGlobalDim, model_dim // MLP_GLOBAL_DIVISOR, bias=USE_BIAS)
        self.mlpGlobalLN = make_norm(norm_type, model_dim // MLP_GLOBAL_DIVISOR, eps=MLP_GLOBAL_LN_EPS)

      self.linear1 = _maybe_wrap_lora(torch.nn.Linear(model_dim + (model_dim // MLP_GLOBAL_DIVISOR if self.use_global else 0), ffn_inner_dim, bias=USE_BIAS), self.layer_num)
      self.linear2 = _maybe_wrap_lora(torch.nn.Linear(ffn_inner_dim, out_dim, bias=USE_BIAS), self.layer_num)
      if activation_type == 'SwiGLU':
        # SwiGLU multiplicative gate: y = linear2(act(linear1(x)) * linear3(x))
        # Wrap with CERES_LORA_FFN_RANK_DIV (same gate as linear1/linear2) so
        # FFN-LoRA experiments adapt the gate too — without this, prior FFN
        # ablations only adapted one of the two FFN paths.
        # Runde-3-fiks 2026-08-29: linear3 leser samme post-global-concat-input
        # som linear1 (jf. forward) — in-dim maa matche, ellers krasjer
        # FFNUseGlobalEveryNLayers + SwiGLU paa foerste forward.
        self.linear3 = _maybe_wrap_lora(torch.nn.Linear(model_dim + (model_dim // MLP_GLOBAL_DIVISOR if self.use_global else 0), ffn_inner_dim, bias=False), self.layer_num)

    self.activation_fn = to_activation(activation_type)

    # SiTU-GLU (Kimi K3 report §2.3.2, arXiv 2607.24653): both multiplicative
    # branches of SwiGLU are unbounded, so coincident large coordinates create
    # activation outliers / low-precision overflow. SiTU bounds each branch with
    # the smooth cap softcap(z, b) = b*tanh(z/b):
    #   Swish(z) = z*sigmoid(z)   ->   softcap(z, b1)*sigmoid(z)
    #   up branch u              ->   softcap(u, b2)
    # Near the origin softcap(z,b) ~ z, so normal-range behavior matches SwiGLU;
    # only outliers are compressed. Env CERES_FFN_SOFTCAP: unset/'0' = off
    # (exact legacy SwiGLU), '1' = K3 defaults b1=4 (gate), b2=25 (up), or
    # explicit 'b1,b2'. SwiGLU-only; other activations ignore the knob.
    _sc = os.environ.get('CERES_FFN_SOFTCAP', '0') or '0'
    self.ffn_softcap = None
    if activation_type == 'SwiGLU' and _sc != '0':
      self.ffn_softcap = (4.0, 25.0) if _sc == '1' else tuple(float(v) for v in _sc.split(','))
      assert len(self.ffn_softcap) == 2 and all(b > 0 for b in self.ffn_softcap)
      if not self.layer_num:  # print once (layer 0 or None)
        print(f'[mlp2_layer] SiTU-GLU FFN softcap enabled: b1={self.ffn_softcap[0]} (gate branch), '
              f'b2={self.ffn_softcap[1]} (up branch)')


  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.use_te:
      with te.fp8_autocast(self.training, fp8_recipe=self.fp8_recipe):
        return self.te_mlp_ln(x) # TODO: figure out why returning a singleton here is required vs. tuple below
    else:
      if self.use_global:
        mlpGlobal = self.mlpGlobalSquareReduce(x);
        mlpGlobal = torch.flatten(mlpGlobal, 1);
        mlpGlobal = self.mlpGlobalReduce(mlpGlobal);
        mlpGlobal = self.activation_fn(mlpGlobal)
        mlpGlobal = self.mlpGlobalLN(mlpGlobal);
        x = torch.concat((x, mlpGlobal.unsqueeze(1).expand(-1, 64, -1)), dim=-1)

      # SwiGLU: y = linear2(SiLU(linear1(x)) * linear3(x)) — both linears
      # operate on the SAME original input x (post-global-concat). Save the
      # input before linear1 overwrites it.
      x_in = x
      _l13 = getattr(self, 'linear13', None)
      if _l13 is not None:
        # export_folds.py: one (in -> 2*inner) GEMM for gate|up, then SiLU(g) * u.
        g, u = _l13(x_in).chunk(2, dim=-1)
        before_linear2 = self.activation_fn(g) * u
        return before_linear2, self.linear2(before_linear2)
      x = self.linear1(x_in)
      if self.ffn_softcap is not None:
        # SiTU-GLU (see __init__): capped Swish gate x capped up branch.
        b1, b2 = self.ffn_softcap
        up = self.linear3(x_in)
        before_linear2 = (b1 * torch.tanh(x / b1)) * torch.sigmoid(x) * (b2 * torch.tanh(up / b2))
      else:
        before_linear2 = self.activation_fn(x)
        if (self.activation_type == 'SwiGLU'):
            before_linear2 = before_linear2 * self.linear3(x_in)

      x_out = self.linear2(before_linear2)

    return before_linear2, x_out


class TSBSwiGLU(torch.nn.Module):
  """Tactical SwiGLU Bypass: per-block parallel SwiGLU FFN + per-block scalar gate.

  Sits beside the original frozen SwiGLU FFN (SP_FFN). The encoder layer combines
  the two outputs via additive residual:

      output = sp_ffn_out + g * tactical_ffn_out

  where g = sigmoid(MLP(meanpool(x))). This additive form (rather than the convex
  blend (1-g)*A + g*B) is bit-identical to SP-only behavior at init regardless of
  the gate value — because tactical_ffn_out is exactly zero at init (zero-initted
  linear3.weight makes the SwiGLU multiplicative gate produce zero).

  Convention: parameter names are prefixed `tactical_ffn_*` and `tactical_gate_*`,
  used by train.py freeze logic and resume strict=False filtering.
  """

  def __init__(self, model_dim: int, ffn_inner_dim: int,
               activation_type: str = 'SwiGLU',
               gate_hidden_divisor: int = 8,
               gate_bias_init: float = -4.0):
    super().__init__()
    assert activation_type == 'SwiGLU', "TSBSwiGLU requires SwiGLU activation"

    # Parallel SwiGLU FFN (same structure as MLP2Layer's SwiGLU branch but no LoRA wrap).
    self.tactical_ffn_linear1 = torch.nn.Linear(model_dim, ffn_inner_dim, bias=USE_BIAS)
    self.tactical_ffn_linear2 = torch.nn.Linear(ffn_inner_dim, model_dim, bias=USE_BIAS)
    self.tactical_ffn_linear3 = torch.nn.Linear(model_dim, ffn_inner_dim, bias=False)
    self.tactical_activation = to_activation(activation_type)
    # CRITICAL: zero-init multiplicative gate => tactical FFN output is exactly 0
    # at step 0, regardless of linear1/linear2 init. Combined with the additive
    # residual in the encoder, this guarantees bit-identical forward at init.
    torch.nn.init.zeros_(self.tactical_ffn_linear3.weight)
    # ... and linear2's BIAS must also be zero: with h == 0 the output is
    # linear2(0) = bias, so a default-init bias silently injects a constant
    # into every block at step 0, violating the invariant above (caught by
    # tools/tsb_smoke.py once its config drift was repaired, 2026-08-17).
    if self.tactical_ffn_linear2.bias is not None:
      torch.nn.init.zeros_(self.tactical_ffn_linear2.bias)

    # Per-block scalar gate: model_dim -> hidden -> 1, sigmoid.
    gate_hidden = max(1, model_dim // gate_hidden_divisor)
    self.tactical_gate_fc1 = torch.nn.Linear(model_dim, gate_hidden, bias=True)
    self.tactical_gate_fc2 = torch.nn.Linear(gate_hidden, 1, bias=True)
    # Closed-init: zero fc2 weight + bias = -4 makes g = sigmoid(-4) ~= 0.018,
    # combined with zero tactical output makes (g * 0) == 0 anyway.
    torch.nn.init.zeros_(self.tactical_gate_fc2.weight)
    torch.nn.init.constant_(self.tactical_gate_fc2.bias, float(gate_bias_init))

  def forward(self, x: torch.Tensor):
    """Returns (tactical_out, gate_value) tensors.

    tactical_out: [B, T, model_dim] — same shape as SP_FFN output.
    gate_value:   [B, 1, 1] — per-batch-element scalar, broadcast over T,model_dim.
    """
    # Tactical SwiGLU path: y = linear2(activation(linear1(x)) * linear3(x))
    h = self.tactical_activation(self.tactical_ffn_linear1(x)) * self.tactical_ffn_linear3(x)
    tactical_out = self.tactical_ffn_linear2(h)

    # Per-block scalar gate from mean-pooled hidden state.
    pooled = x.mean(dim=1)                                         # [B, model_dim]
    gate_hidden = torch.nn.functional.gelu(self.tactical_gate_fc1(pooled))
    gate_value = torch.sigmoid(self.tactical_gate_fc2(gate_hidden))  # [B, 1]
    gate_value = gate_value.unsqueeze(-1)                            # [B, 1, 1]

    return tactical_out, gate_value
