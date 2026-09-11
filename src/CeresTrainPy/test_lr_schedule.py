# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""Unit test for lr_schedule.py (pure, no torch). Run: python test_lr_schedule.py

1. Without knots the factor equals the legacy train.py lr_lambda formula
   (linear and cosine) at every position — the refactor is behaviour-preserving.
2. Knots: hold to the first knot, linear between knots, final decay from the
   last knot's factor; continuous at every knot.
3. Validation rejects steps, unsorted knots, and a last knot that disagrees with
   LRBeginDecayAtFractionComplete.
4. The 8B two-slope example lands on the intended landmarks.
"""

import math
from lr_schedule import lr_factor, validate_knots, describe


def legacy(num_pos, max_pos, warmup, frac_start, min_lr, shape):
  fraction_complete = num_pos / max_pos
  if num_pos < warmup:
    return (float(num_pos) / float(warmup)) ** 0.5
  elif fraction_complete < frac_start:
    return 1.0
  elif fraction_complete > 1:
    return min_lr
  elif shape == 'cosine':
    prog = (fraction_complete - frac_start) / (1.0 - frac_start)
    return min_lr + (1.0 - min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))
  else:
    slope = (min_lr - 1.0) / (1.0 - frac_start)
    return 1.0 + slope * (fraction_complete - frac_start)


def test_legacy_identical():
  max_pos, warmup = 8_000_000_000, 100_000_000
  for shape in ('linear', 'cosine'):
    for frac_start in (0.0, 0.25, 0.5, 0.6):
      for min_lr in (0.0, 0.05, 0.1):
        for pos in range(0, max_pos + 1, 25_000_000):
          a = lr_factor(pos, max_pos, warmup, frac_start, min_lr, shape, None)
          b = legacy(pos, max_pos, warmup, frac_start, min_lr, shape)
          assert abs(a - b) < 1e-12, (shape, frac_start, min_lr, pos, a, b)
  print('  legacy shapes identical (linear/cosine x 4 starts x 3 floors, 321 positions each)')


def test_knots_shape():
  max_pos, warmup, min_lr = 8_000_000_000, 100_000_000, 0.05
  knots = validate_knots([[0.25, 1.0], [0.5, 0.75]], 0.5)
  f = lambda pos, shape='cosine': lr_factor(pos, max_pos, warmup, 0.5, min_lr, shape, knots)
  assert f(1_000_000_000) == 1.0 and f(1_999_999_999) == 1.0           # hold to the first knot
  assert abs(f(2_000_000_000) - 1.0) < 1e-12                            # continuous at knot 0
  assert abs(f(3_000_000_000) - 0.875) < 1e-12                          # midway on the ramp
  assert abs(f(4_000_000_000) - 0.75) < 1e-12                           # knot 1 = decay start
  assert abs(f(4_000_000_000 - 1) - 0.75) < 1e-9                        # continuous at knot 1
  assert abs(f(6_000_000_000) - (min_lr + (0.75 - min_lr) * 0.5)) < 1e-12   # cosine midpoint from 0.75
  assert abs(f(8_000_000_000) - min_lr) < 1e-12                         # floor at the end
  assert abs(f(6_000_000_000, 'linear') - (0.75 + (min_lr - 0.75) * 0.5)) < 1e-12
  # monotone non-increasing after warmup
  prev = 2.0
  for pos in range(warmup, max_pos + 1, 10_000_000):
    cur = f(pos); assert cur <= prev + 1e-12, pos; prev = cur
  # three knots: linear between each pair
  k3 = validate_knots([[0.2, 1.0], [0.4, 0.9], [0.6, 0.5]], 0.6)
  g = lambda pos: lr_factor(pos, max_pos, warmup, 0.6, min_lr, 'linear', k3)
  assert abs(g(2_400_000_000) - 0.95) < 1e-12 and abs(g(4_000_000_000) - 0.7) < 1e-12
  print('  knot interpolation, continuity, monotonicity OK')


def test_validation():
  def rejects(knots, start, needle):
    try:
      validate_knots(knots, start)
    except AssertionError as e:
      assert needle in str(e), (needle, str(e)); return
    raise AssertionError(f'accepted {knots!r}')
  rejects([[0.25, 0.9], [0.5, 0.75]], 0.5, 'factor must be 1.0')
  rejects([[0.5, 1.0], [0.25, 0.75]], 0.25, 'strictly increasing')
  rejects([[0.25, 1.0], [0.5, 0.75]], 0.6, 'must equal LRBeginDecayAtFractionComplete')
  rejects([[0.25, 1.0], [0.5, 0.0]], 0.5, 'factor 0.0 must be in (0, 1]')
  rejects([[0.25, 1.0], [1.0, 0.5]], 1.0, 'fraction 1.0 must be in (0, 1)')
  rejects([[0.25]], 0.25, 'pairs')
  rejects([[0.25, 1.0], [0.4, 0.5], [0.5, 0.9]], 0.5, 'non-increasing')          # LR would rise mid-run
  # floor and warmup checks (only when min_lr / max_pos are passed, as config.py does)
  def rejects_full(knots, start, min_lr, max_pos, needle):
    try:
      validate_knots(knots, start, min_lr, max_pos)
    except AssertionError as e:
      assert needle in str(e), (needle, str(e)); return
    raise AssertionError(f'accepted {knots!r}')
  rejects_full([[0.25, 1.0], [0.5, 0.03]], 0.5, 0.05, 8e9, 'below LRMinFactor')     # final decay would rise
  rejects_full([[0.01, 1.0], [0.5, 0.75]], 0.5, 0.05, 8e9, 'inside the warmup')     # 80M < 100M warmup
  assert validate_knots([[0.25, 1.0], [0.5, 0.75]], 0.5, 0.05, 8e9) == [(0.25, 1.0), (0.5, 0.75)]
  assert validate_knots([[0.0125, 1.0], [0.5, 0.75]], 0.5, 0.05, 8e9)[0] == (0.0125, 1.0)  # exactly at warmup end (100M) is fine
  assert validate_knots(None, 0.5) is None and validate_knots([], 0.5) is None
  print('  validation OK (9 rejections incl. rising factors / floor / warmup, absent/empty -> None)')


def test_8b_example():
  lr, max_pos, warmup = 8e-4, 8_000_000_000, 100_000_000
  knots = validate_knots([[0.25, 1.0], [0.5, 0.75]], 0.5)
  at = lambda pos: lr * lr_factor(pos, max_pos, warmup, 0.5, 0.05, 'cosine', knots)
  assert abs(at(2_000_000_000) - 8e-4) < 1e-12
  assert abs(at(4_000_000_000) - 6e-4) < 1e-12
  assert abs(at(8_000_000_000) - 4e-5) < 1e-12
  line = describe(lr, max_pos, warmup, 0.5, 0.05, 'cosine', knots)
  assert '2 knots, then cosine' in line and '@4000M: 0.0006' in line, line
  print('  8B two-slope example: 8e-4 @2B, 6e-4 @4B, 4e-5 @8B; describe():', line)


if __name__ == '__main__':
  test_legacy_identical(); test_knots_shape(); test_validation(); test_8b_example()
  print('ALL OK')
