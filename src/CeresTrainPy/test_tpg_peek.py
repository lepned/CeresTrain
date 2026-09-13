# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""Unit test for scripts/tpg_peek.py (pure numpy, no torch). Run: python test_tpg_peek.py

tpg_peek.py reads raw TPG shards with its own record offsets, which is exactly how
scripts/tpg_eval.py became untrustworthy (hand-derived constants, never cross-checked,
silently misaligned -> reversed checkpoint ranking on 2026-08-17). These tests are the
cross-check that file never had:

1. The layout table accounts for every byte before the squares, at both widths.
2. Its constants still agree with tpg_dataset.py (the validated reader) -- read out
   of that file's source, so a change there fails here instead of drifting silently.
3. A synthetic shard with known field values decodes back to those values.
4. Decoding at the WRONG square width is REJECTED rather than reported -- the
   specific failure mode that poisoned tpg_eval.py.
"""

import os
import re
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

import tpg_peek  # noqa: E402


def _dataset_source():
  with open(os.path.join(THIS_DIR, 'tpg_dataset.py'), encoding='utf-8') as fh:
    return fh.read()


def test_layout_accounts_for_every_byte():
  """The fields must exactly fill the space before the squares, at 137 and 141."""
  for sq in tpg_peek.SQUARE_BYTES_CHOICES:
    expected = tpg_peek.bytes_per_pos(sq) - 64 * sq
    assert tpg_peek.HEADER_AND_POLICY_BYTES == expected, (sq, expected)
  # Header alone (everything but the two policy blocks) is 242 bytes.
  assert tpg_peek.HEADER_AND_POLICY_BYTES - 4 * tpg_peek.MAX_MOVES == 242
  print('  layout: 242 B header + 4*MAX_MOVES policy + 64*sq == BYTES_PER_POS at both widths')


def test_constants_match_tpg_dataset():
  """MAX_MOVES and the record-size formula must still match the validated reader."""
  src = _dataset_source()

  m = re.search(r'^MAX_MOVES\s*=\s*(\d+)', src, re.M)
  assert m, 'MAX_MOVES not found in tpg_dataset.py'
  assert int(m.group(1)) == tpg_peek.MAX_MOVES, (m.group(1), tpg_peek.MAX_MOVES)

  m = re.search(r'_BYTES_PER_POS_ONDISK\s*=\s*(\d+)\s*if\s*_USE_V3_TPG\s*else\s*(\d+)', src)
  assert m, '_BYTES_PER_POS_ONDISK not found in tpg_dataset.py'
  v3, v2 = int(m.group(1)), int(m.group(2))
  assert tpg_peek.bytes_per_pos(137) == v2, (tpg_peek.bytes_per_pos(137), v2)
  assert tpg_peek.bytes_per_pos(141) == v3, (tpg_peek.bytes_per_pos(141), v3)

  # The in-loop formula tpg_dataset.py actually decodes with.
  assert re.search(r'BYTES_PER_POS\s*=\s*9378\s*\+\s*\(self\.square_bytes\s*-\s*137\)\s*\*\s*64', src), \
      'tpg_dataset.py record-size formula changed -- update tpg_peek.bytes_per_pos'
  print(f'  constants: MAX_MOVES={tpg_peek.MAX_MOVES}, V2={v2}, V3={v3} agree with tpg_dataset.py')


def _synth_shard(square_bytes, n=64, seed=0):
  """Build n records with known values; returns (raw_bytes, expected dict)."""
  rng = np.random.default_rng(seed)
  bpp = tpg_peek.bytes_per_pos(square_bytes)
  rec = np.zeros((n, bpp), dtype=np.uint8)

  wdl = rng.random((n, 3)).astype(np.float32)
  wdl /= wdl.sum(axis=1, keepdims=True)
  subopt = rng.random(n).astype(np.float32)
  kld = rng.random(n).astype(np.float32)

  # Two moves per position carrying 0.75 / 0.25, then the C# padding convention:
  # unused slots REPLICATE the last (index, value) pair.
  idx = np.zeros((n, tpg_peek.MAX_MOVES), dtype=np.int16)
  val = np.zeros((n, tpg_peek.MAX_MOVES), dtype=np.float16)
  idx[:, 0] = 11
  idx[:, 1:] = 22
  val[:, 0] = np.float16(0.75)
  val[:, 1:] = np.float16(0.25)

  def put(offset, arr):
    b = np.ascontiguousarray(arr).view(np.uint8).reshape(n, -1)
    rec[:, offset : offset + b.shape[1]] = b

  off = 0
  for name, size in tpg_peek.LAYOUT:
    if name == 'wdl_nondeblundered':
      put(off, wdl)
    elif name == 'wdl_deblundered':
      put(off, wdl)          # identical -> frac_deblundered must be 0
    elif name == 'wdl_q':
      put(off, wdl)
    elif name == 'played_q_suboptimality':
      put(off, subopt)
    elif name == 'kld_policy':
      put(off, kld)
    elif name == 'policies_indices':
      put(off, idx)
    elif name == 'policies_values':
      put(off, val)
    off += size

  return rec.tobytes(), {'wdl': wdl, 'subopt': subopt, 'kld': kld}


def test_synthetic_roundtrip():
  """Known values must survive the decode, with the padding mask applied."""
  for sq in tpg_peek.SQUARE_BYTES_CHOICES:
    raw, exp = _synth_shard(sq)
    n, f = tpg_peek._decode(raw, sq)
    assert n == 64, n
    assert np.allclose(f['wdl_q'], exp['wdl'], atol=1e-6)
    assert np.allclose(f['played_q_suboptimality'], exp['subopt'], atol=1e-6)
    assert np.allclose(f['kld_policy'], exp['kld'], atol=1e-6)
    assert tpg_peek._wdl_validity(f) == 1.0

    # Padding is a replicated suffix: 2 distinct moves, renormalised to 0.75/0.25.
    idx = f['policies_indices']
    valid = np.ones_like(idx, dtype=bool)
    valid[:, 1:] = idx[:, 1:] != idx[:, :-1]
    assert valid.sum(axis=1).max() == 2 and valid.sum(axis=1).min() == 2
    probs = np.where(valid, f['policies_values'], 0.0)
    probs /= probs.sum(axis=1, keepdims=True)
    assert np.allclose(probs.max(axis=1), 0.75, atol=1e-3)
  print('  synthetic round-trip: wdl / subopt / KLD / padded policy decode exactly at both widths')


def test_wrong_width_is_rejected():
  """The tpg_eval.py failure mode: a misaligned stride must be loud, not silent."""
  for sq in tpg_peek.SQUARE_BYTES_CHOICES:
    other = 141 if sq == 137 else 137
    raw, _ = _synth_shard(sq, n=256)

    # The wrong stride must not pass the probability-vector check...
    _, bad = tpg_peek._decode(raw, other)
    assert tpg_peek._wdl_validity(bad) < 0.98, (sq, other, tpg_peek._wdl_validity(bad))

    # ...detection must pick the true width...
    detected, scores = tpg_peek.detect_square_bytes(raw)
    assert detected == sq, (detected, sq, scores)

    # ...and an explicit wrong override must raise instead of returning numbers.
    raised = False
    try:
      tpg_peek.decode_checked(raw, other, label=f'synthetic-{sq}')
    except ValueError:
      raised = True
    assert raised, f'wrong width {other} on a {sq} shard was not rejected'

    # The right width still goes through.
    n, _, validity = tpg_peek.decode_checked(raw, sq, label=f'synthetic-{sq}')
    assert n == 256 and validity == 1.0, (n, validity)
  print('  wrong width: validity collapses, detection recovers the truth, override raises')


if __name__ == '__main__':
  test_layout_accounts_for_every_byte()
  test_constants_match_tpg_dataset()
  test_synthetic_roundtrip()
  test_wrong_width_is_rejected()
  print('ALL OK')
