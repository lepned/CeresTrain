# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""Learning-rate schedule as a pure function of position (factored out of
train.py's lr_lambda 2026-09-11 so it can be unit-tested and printed at boot).

Shape (factor of LearningRateBase):
  warmup      : inverse-sqrt ramp over WARMUP_POS positions
  hold        : 1.0 until the first knot (or the decay start when no knots)
  knots       : piecewise-LINEAR between (fraction_complete, factor) knots
  final decay : from the last knot's factor (1.0 without knots) down to
                min_lr over [decay_start, 1.0], shape 'linear' or 'cosine'

`LRKnots` (config, optional) = [[frac, factor], ...]; the last knot's frac must
equal LRBeginDecayAtFractionComplete (one source of truth for where the final
decay begins), the first knot's factor must be 1.0 (no step off the hold), fracs
strictly increasing and after warmup, factors in (0, 1] and non-increasing with
the last one at or above LRMinFactor (the LR never rises). Two-slope example
(8B run, 09-11):
  LRKnots [[0.25, 1.0], [0.5, 0.75]] + LRBeginDecayAtFractionComplete 0.5 + cosine
  = hold to 2B, linear 8e-4 -> 6e-4 over 2B-4B, cosine 6e-4 -> floor over 4B-8B.
"""

import math


def warmup_positions(max_pos):
  """Warmup length: 5 % of the run, capped at 100M (train.py's num_warmup_positions)."""
  return int(min(100_000_000, 0.05 * max_pos))


def validate_knots(knots, frac_start_decay, min_lr=0.0, max_pos=None):
  """Returns the knots as a list of (frac, factor) float tuples, or None when
  absent/empty. Raises AssertionError with a specific message otherwise.
  `min_lr` / `max_pos` (when given) add the floor and warmup checks."""
  if not knots:
    return None
  ks = []
  for k in knots:
    assert isinstance(k, (list, tuple)) and len(k) == 2, f"LRKnots entries must be [fraction, factor] pairs, got {k!r}"
    ks.append((float(k[0]), float(k[1])))
  for i, (f, v) in enumerate(ks):
    assert 0.0 < f < 1.0, f"LRKnots[{i}] fraction {f} must be in (0, 1)"
    assert 0.0 < v <= 1.0, f"LRKnots[{i}] factor {v} must be in (0, 1]"
    if i > 0:
      assert f > ks[i - 1][0], f"LRKnots fractions must be strictly increasing ({ks[i-1][0]} -> {f})"
      assert v <= ks[i - 1][1], f"LRKnots factors must be non-increasing ({ks[i-1][1]} -> {v} at fraction {f}): the LR never rises"
  assert ks[0][1] == 1.0, (f"LRKnots[0] factor must be 1.0 (the hold ends there; a lower value would be a step), got {ks[0][1]}")
  assert ks[-1][1] >= float(min_lr), (f"the last LRKnots factor ({ks[-1][1]}) is below LRMinFactor ({min_lr}): the final decay would rise")
  if max_pos:
    assert ks[0][0] * max_pos >= warmup_positions(max_pos), (
        f"LRKnots[0] ({ks[0][0]} = {ks[0][0]*max_pos/1e6:.0f}M) lies inside the warmup ({warmup_positions(max_pos)/1e6:.0f}M)")
  assert abs(ks[-1][0] - float(frac_start_decay)) < 1e-9, (
      f"the last LRKnots fraction ({ks[-1][0]}) must equal LRBeginDecayAtFractionComplete ({frac_start_decay}): "
      f"the final decay starts at the last knot")
  return ks


def lr_factor(num_pos, max_pos, warmup_pos, frac_start_decay, min_lr, shape='linear', knots=None):
  """LR multiplier at `num_pos` positions into a run of `max_pos`."""
  if num_pos < warmup_pos:
    return (float(num_pos) / float(warmup_pos)) ** 0.5   # inverse square root warmup
  frac = num_pos / float(max_pos)
  if frac > 1.0:
    return min_lr   # shouldn't happen
  top = 1.0   # factor at which the final decay starts
  if knots:
    if frac < knots[0][0]:
      return 1.0
    for (f0, v0), (f1, v1) in zip(knots[:-1], knots[1:]):
      if frac < f1:
        return v0 + (v1 - v0) * (frac - f0) / (f1 - f0)
    top = knots[-1][1]
  elif frac < frac_start_decay:
    return 1.0
  prog = (frac - frac_start_decay) / (1.0 - frac_start_decay) if frac_start_decay < 1.0 else 1.0
  prog = min(max(prog, 0.0), 1.0)
  if shape == 'cosine':
    return min_lr + (top - min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))
  return top + (min_lr - top) * prog   # linear


def describe(lr_base, max_pos, warmup_pos, frac_start_decay, min_lr, shape, knots):
  """One-line-per-landmark description for the boot log."""
  marks = [("warmup end", warmup_pos)]
  if knots:
    marks += [(f"knot {i}", int(round(f * max_pos))) for i, (f, _) in enumerate(knots)]
  else:
    marks.append(("decay start", int(round(frac_start_decay * max_pos))))
  marks.append(("mid decay", int(round((frac_start_decay + (1.0 - frac_start_decay) / 2) * max_pos))))
  marks.append(("end", max_pos))
  rows = []
  for name, pos in marks:
    fac = lr_factor(pos, max_pos, warmup_pos, frac_start_decay, min_lr, shape, knots)
    rows.append(f"{name} @{pos/1e6:.0f}M: {lr_base*fac:.3g} (x{fac:.3f})")
  kind = f"{len(knots)} knots, then {shape}" if knots else f"hold, then {shape}"
  return f"[lr-schedule] {kind} to floor x{min_lr}: " + " | ".join(rows)
