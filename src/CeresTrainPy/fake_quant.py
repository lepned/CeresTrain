# License Notice
#
# This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
# Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
#
# Ceres is free software distributed under the terms of the GNU General Public License v3.0.
# You should have received a copy of the GNU General Public License along with CeresTrain.
# If not, see <http://www.gnu.org/licenses/>.
#
# End of License Notice

"""INT8 Quantization-Aware Training (QAT) via in-place fake-quant on nn.Linear.

WHY
  Post-training quantization (PTQ) of the C1-640-34 flagship hits a ceiling
  (~-0.6pp policy / -0.8pp value vs FP16) because policy and value share the
  trunk but prefer opposite activation-clipping percentiles — a single static
  scale can't serve both. QAT dissolves this by adapting the *weights* (under
  the straight-through estimator) so both heads tolerate the deployed INT8
  scale, instead of us hand-tuning the scale at calibration time.

WHAT THIS MATCHES (must mirror scripts/qdq_export.py exactly, no train/deploy skew)
  - weights      : per-OUTPUT-channel, SYMMETRIC int8 (qmin/qmax -127..127),
                   scale = max|W_c| / 127, recomputed every forward from the
                   live FP weights (standard QAT).
  - activations  : per-TENSOR, SYMMETRIC int8, range from a percentile observer
                   (default 99.999, the deployed PTQ sweet spot), then frozen.
  - scope        : every nn.Linear GEMM (qkv, W_h, FFN linear1/2/3, smolgen
                   sm1/2/3 + prep, global reduces, heads). The attention inner
                   matmuls (Q·Kᵀ, A·V) are activation×activation and are
                   quantized by the *export* (op_types_to_quantize=['MatMul'])
                   but NOT here — PTQ already quantizes them cleanly (value
                   survives at 90.85%), so they are not the gap and adding them
                   would mean editing every attention branch. Documented skew,
                   empirically negligible.

DESIGN
  In-place __class__ swap (nn.Linear -> FakeQuantLinear) rather than module
  replacement. This preserves Parameter identity (the optimizer keeps working
  unchanged) AND keeps shared/aliased modules correct: the smolgenPrepLayer is
  referenced through a LinearWrapper in every attention layer, so REPLACING the
  registered module would leave those wrappers pointing at the old un-quantized
  object. Mutating the object in place updates every reference at once.

  Fake-quant is gated on `self.training`, so save_model()'s eval()+no_grad
  export emits a CLEAN FP16 graph (no Q/DQ ops) — the QAT'd weights are then
  quantized for real by scripts/qdq_export.py -> strongly-typed TRT. Zero
  changes needed at any of the export call sites.

USAGE (env-gated, matching the CERES_LORA_* / CERES_GTAB idiom; no config-schema
or C# change):
  CERES_QAT_INT8=1                 enable QAT
  CERES_QAT_PERCENTILE=99.999      activation calibration percentile (deploy match)
  CERES_QAT_CALIB_POS=200000       observe activations for this many positions, then freeze
  CERES_QAT_EXCLUDE=substr,substr  skip nn.Linear whose qualified name contains any substring
  CERES_QAT_WEIGHTS_ONLY=0         1 = quantize weights only (skip activation fake-quant)

  In train.py:  convert_to_fake_quant(model_nocompile, ...) after resume; then
  once per step  freeze_if_ready(model_nocompile, num_pos, calib_pos).
"""

import os
import torch
import torch.nn.functional as F


_QMIN = -127
_QMAX = 127


def _fake_quant_ste(x, scale, qmin=_QMIN, qmax=_QMAX):
  """Symmetric fake-quant with a straight-through-estimator gradient.

  Forward: dequant(quant(x)). Backward: identity (grad flows through unchanged).
  Computed IN x's dtype (no fp32 upcast) — critical for memory: upcasting every
  activation to fp32 and retaining it for backward across hundreds of layers
  blows out GPU memory (a 250M net OOM'd at 40 GiB). bf16/fp16 both have >=8
  mantissa bits, so round(x/scale) is exact over the int8 range [-127, 127]
  anyway — the upcast bought no accuracy. `scale` broadcasts against x
  (per-tensor scalar or per-out-channel column); cast to x.dtype to avoid
  silent fp32 promotion from the fp32 scale buffer.
  """
  s = scale.to(x.dtype).clamp_min(1e-12)
  q = torch.clamp(torch.round(x / s), qmin, qmax)
  dq = q * s
  # STE: value is the dequantized result, gradient is d(dq)/dx == 1.
  return x + (dq - x).detach()


class FakeQuantLinear(torch.nn.Linear):
  """nn.Linear with INT8 fake-quant on weights (per-out-channel) and inputs
  (per-tensor), active only in training mode. Instances are produced by
  convert_to_fake_quant() via in-place __class__ assignment — never constructed
  directly — so __init__ is intentionally not overridden."""

  def forward(self, x):
    # Clean path for eval/export and when QAT is inactive: bit-identical nn.Linear.
    if not (self.training and getattr(self, '_fq_active', False)):
      return F.linear(x, self.weight, self.bias)

    # --- weight fake-quant: per-output-channel symmetric int8 ---
    if self._fq_quant_weights:
      w = self.weight
      if getattr(self, '_fq_w_scale', None) is not None:
        # Frozen scales (computed once at conversion). Live recompute creates a
        # feedback loop — weight updates move the row max, which shifts the whole
        # row's quantization grid discontinuously — that diverged the weights-only
        # run at ~2M positions (loss 0.28 -> 2.4 in ~25K pos). v1 was accidentally
        # damped by its activation clipping.
        w_scale = self._fq_w_scale
      else:
        w_absmax = w.detach().abs().amax(dim=1, keepdim=True)        # [out, 1]
        w_scale = w_absmax / _QMAX
      w = _fake_quant_ste(w, w_scale)
    else:
      w = self.weight

    # --- activation fake-quant: per-tensor symmetric int8 ---
    if self._fq_quant_acts:
      if not self._fq_frozen:
        self._observe(x)
      a_scale = (self._fq_act_range / _QMAX)
      x = _fake_quant_ste(x, a_scale)

    return F.linear(x, w, self.bias)

  @torch.no_grad()
  def _observe(self, x):
    """Update the EMA percentile estimate of |activation| for this tensor.

    Approximates ORT's global-histogram percentile with a per-batch
    torch.quantile (subsampled to cap cost), EMA-smoothed across calib steps.
    """
    a = x.detach().abs().reshape(-1).float()
    n = a.numel()
    if n > 1_000_000:
      a = a[:: (n // 1_000_000) + 1]
    q = torch.quantile(a, self._fq_percentile)
    if self._fq_obs_count == 0:
      self._fq_act_range.copy_(q)
    else:
      self._fq_act_range.mul_(0.9).add_(0.1 * q)
    self._fq_obs_count += 1


def _qualified_linear_names(model):
  return {id(m): name for name, m in model.named_modules()
          if isinstance(m, torch.nn.Linear)}


def convert_to_fake_quant(model, percentile=99.999, exclude=None,
                          quant_weights=True, quant_acts=True,
                          freeze_weight_scales=False):
  """In-place: swap every plain nn.Linear in `model` to FakeQuantLinear.

  Idempotent (already-converted modules are skipped). `exclude` is a list of
  substrings; any nn.Linear whose qualified module name contains one is left as
  plain FP. Returns (n_converted, n_excluded).
  """
  exclude = exclude or []
  names = {}
  for name, m in model.named_modules():
    names[id(m)] = name

  n_conv = n_excl = 0
  seen = set()
  for m in model.modules():
    if type(m) is not torch.nn.Linear:
      continue  # FakeQuantLinear (already converted) and subclasses are skipped
    if id(m) in seen:
      continue
    seen.add(id(m))
    qname = names.get(id(m), '?')
    if any(sub in qname for sub in exclude):
      n_excl += 1
      continue
    m.__class__ = FakeQuantLinear
    # state lives on the instance; buffer moves with .to(device)/.cuda()
    m.register_buffer('_fq_act_range', torch.ones(1, device=m.weight.device,
                                                  dtype=torch.float32))
    if freeze_weight_scales and quant_weights:
      with torch.no_grad():
        m.register_buffer('_fq_w_scale',
                          (m.weight.detach().abs().amax(dim=1, keepdim=True)
                           / _QMAX).clamp_min(1e-12))
    else:
      m._fq_w_scale = None
    m._fq_percentile = float(percentile) / 100.0
    m._fq_obs_count = 0
    m._fq_frozen = False
    m._fq_active = True
    m._fq_quant_weights = bool(quant_weights)
    m._fq_quant_acts = bool(quant_acts)
    m._fq_name = qname
    n_conv += 1
  return n_conv, n_excl


def freeze(model, verbose=True):
  """Freeze activation ranges (stop observing) on all FakeQuantLinear modules."""
  ranges = []
  for m in model.modules():
    if isinstance(m, FakeQuantLinear):
      m._fq_frozen = True
      if m._fq_quant_acts:
        ranges.append((m._fq_name, float(m._fq_act_range.item()), m._fq_obs_count))
  if verbose and ranges:
    vals = sorted(r[1] for r in ranges)
    lo, med, hi = vals[0], vals[len(vals) // 2], vals[-1]
    print(f"INFO: QAT_FREEZE froze {len(ranges)} activation ranges "
          f"(min {lo:.3f}  median {med:.3f}  max {hi:.3f}); "
          f"obs_steps={ranges[0][2]}", flush=True)
  return len(ranges)


def freeze_if_ready(model, num_pos, calib_pos, _state={'frozen': False}):
  """Call once per training step. Freezes activation ranges after `calib_pos`
  positions of observation. No-op after the first freeze. Returns True on the
  step that performs the freeze."""
  if _state['frozen'] or num_pos < calib_pos:
    return False
  freeze(model, verbose=True)
  _state['frozen'] = True
  return True


def set_active(model, active):
  """Globally toggle fake-quant (e.g. force-clean for an explicit FP export)."""
  for m in model.modules():
    if isinstance(m, FakeQuantLinear):
      m._fq_active = bool(active)


def summary(model):
  n = sum(1 for m in model.modules() if isinstance(m, FakeQuantLinear))
  return f"{n} FakeQuantLinear modules"
