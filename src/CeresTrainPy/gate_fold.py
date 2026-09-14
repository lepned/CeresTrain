# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""Warm-start fold for the gated attention output (2026-09-14).

The gate `H_cat * sigmoid(attn_out_gate(x))` starts with ZERO weight and a constant
bias b, i.e. it multiplies every attention output channel by g0 = sigmoid(b) (0.982
for the default b = 4) — a near-identity, not an identity. When the gate is switched
on at a RESUME of a trained checkpoint that lacked it, that 2 % scaling of every
attention branch would perturb the net. Folding 1/g0 into each layer's loaded output
projection W_h makes the switched net function-identical to the checkpoint:

    W_h (g0 * H_cat) + b_h  ==  (W_h / g0) (g0 * H_cat) + b_h  ==  W_h H_cat + b_h

(W_h's bias is untouched: the gate acts before W_h.) Exact up to float rounding
(relative ~1e-7 in fp32), and the gate keeps its normal gradient because b is
unchanged. Only valid while the gate weight is still all-zero (fresh init).
"""
import torch


def fold_gate_on_warm_start(model) -> tuple:
  """Scale W_h by 1/sigmoid(gate bias) in every attention layer whose gate is fresh
  (zero weight). Returns (num_layers_folded, g0). Raises when W_h is not a plain
  Linear (LoRA-wrapped) or a gate weight is already non-zero (trained: folding would
  double-count)."""
  n = 0
  g0 = None
  for lyr in model.transformer_layer:
    att = getattr(lyr, 'attention', None)
    if att is None or not getattr(att, 'use_gated_attn_out', False):
      continue
    gate = att.attn_out_gate
    if float(gate.weight.detach().abs().max()) != 0.0:
      raise RuntimeError('gate fold: attn_out_gate.weight is non-zero (already trained) — the fold is only exact for a fresh gate')
    if not isinstance(att.W_h, torch.nn.Linear):
      raise RuntimeError(f'gate fold: W_h is {type(att.W_h).__name__}, not a plain Linear — fold unsupported')
    b = gate.bias.detach().float()
    if float(b.max() - b.min()) != 0.0:
      raise RuntimeError('gate fold: attn_out_gate.bias is not constant across channels — the fold assumes the fresh constant init')
    g = float(torch.sigmoid(b[0]))
    if g0 is None:
      g0 = g
    with torch.no_grad():
      att.W_h.weight.mul_(1.0 / g)
    n += 1
  return n, g0
