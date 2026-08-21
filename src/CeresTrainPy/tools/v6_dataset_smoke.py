# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""Smoke tests for v6_dataset.py (DirectFromV6), incl. the C#-parity fixes
from V6_DATASET_REVIEW.md. Run from CeresTrainPy with
CERES_AUX_FEATURES_PER_SQUARE=0:

    CERES_AUX_FEATURES_PER_SQUARE=0 python3 tools/v6_dataset_smoke.py \
        [--v6-dir DIR] [--v7-dir DIR]

Synthetic-record tests always run; real-data tests run when the dirs are
provided (default local paths used when present).
"""

import argparse
import glob
import gzip as std_gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from v6_dataset import (V6ChunkDataset, V6_DTYPE, V7_DTYPE, POLICY_FLOOR, MAX_MOVES)


def _bare(skip=1, zmax=0.0):
  ds = V6ChunkDataset.__new__(V6ChunkDataset)
  ds.skip_count = skip
  ds.max_resultq_delta = zmax
  ds._skipped_formats = 0
  ds._skipped_other_version = 0
  ds._read_errors = 0
  ds._zfiltered = 0
  ds._version = None
  ds._tar_handles = {}
  return ds


def _startpos_planes():
  """LC0 bit planes for the initial position (our perspective), one history pos."""
  def bb(*squares):
    v = np.uint64(0)
    for sq in squares:
      r, f = sq // 8, sq % 8
      v |= np.uint64(1) << np.uint64(r * 8 + (7 - f))   # byte=rank, MSB=file a
    return v
  ours = [bb(8, 9, 10, 11, 12, 13, 14, 15), bb(1, 6), bb(2, 5), bb(0, 7), bb(3), bb(4)]
  theirs = [bb(48, 49, 50, 51, 52, 53, 54, 55), bb(57, 62), bb(58, 61), bb(56, 63), bb(59), bb(60)]
  return ours + theirs + [np.uint64(0)]     # + repetition plane


def _mk_record(planes_hist, rule50=0, best_q=0.0, best_d=0.3, played_q=0.0,
               result_q=0.0, result_d=1.0, orig_q=0.0, plies_left=50.0,
               nlegal=20):
  r = np.zeros(1, dtype=V6_DTYPE)[0]
  r['version'] = 6
  r['input_format'] = 1
  probs = np.full(1858, -1.0, dtype=np.float32)
  probs[:nlegal] = 1.0 / nlegal
  r['probs'] = probs
  pl = np.zeros(104, dtype=np.uint64)
  for h, hist in enumerate(planes_hist[:8]):
    pl[h * 13:(h + 1) * 13] = hist
  r['planes'] = pl
  r['us_oo'] = r['us_ooo'] = r['them_oo'] = r['them_ooo'] = 1
  r['rule50'] = rule50
  r['best_q'], r['best_d'] = best_q, best_d
  r['played_q'] = played_q
  r['result_q'], r['result_d'] = result_q, result_d
  r['orig_q'] = orig_q
  r['plies_left'] = plies_left
  return r


def synthetic_tests():
  ds = _bare()
  sp = _startpos_planes()

  # --- history FILL_IN: only h0 present -> h1..h7 copied from h0, not "empty board"
  rec = _mk_record([sp])
  out = ds._records_to_arrays(np.array([rec], dtype=V6_DTYPE))
  sq = out[9][0]
  assert np.array_equal(sq[:, 0:13], sq[:, 13:26]), 'history FILL_IN: h1 != h0'
  assert np.array_equal(sq[:, 0:13], sq[:, 91:104]), 'history FILL_IN: h7 != h0'
  assert list(np.where(sq[:, 6] > 0)[0]) == [4] and list(np.where(sq[:, 12] > 0)[0]) == [60]
  print('PASS history FILL_IN cascade + startpos anchor')

  # --- Move50 scale: min(count,100)/50
  for r50, want in ((30, 0.6), (100, 2.0), (120, 2.0)):
    rec = _mk_record([sp], rule50=r50)
    sq = ds._records_to_arrays(np.array([rec], dtype=V6_DTYPE))[9][0]
    assert abs(float(sq[0, 116]) - want) < 1e-6, f'Move50({r50}) = {sq[0,116]}, want {want}'
  print('PASS Move50 = min(count,100)/50')

  # --- EP: their pawn just double-pushed on file 2: h1 has pawn 48+2, h0 has 32+2.
  def with_their_pawn_move(f):
    theirs_now = list(_startpos_planes())
    # move their pawn from 48+f to 32+f in h0; h1 = startpos
    def bbmod(v, clear_sq, set_sq):
      def bit(sq):
        r_, f_ = sq // 8, sq % 8
        return np.uint64(1) << np.uint64(r_ * 8 + (7 - f_))
      return (v & ~bit(clear_sq)) | bit(set_sq)
    theirs_now[6] = bbmod(theirs_now[6], 48 + f, 32 + f)
    return [theirs_now, _startpos_planes()]
  rec = _mk_record(with_their_pawn_move(2))
  sq = ds._records_to_arrays(np.array([rec], dtype=V6_DTYPE))[9][0]
  ep_squares = list(np.where(sq[:, 118] > 0)[0])
  assert ep_squares == [32 + 2], f'EP flag on {ep_squares}, want [34] (opponent pawn square)'
  print('PASS EP flag on opponent pawn square (32+f), single file')

  # --- EP negative: doubled their-pawns on the file (pawn still on 48+f in h0)
  hist = with_their_pawn_move(3)
  def bit(sq):
    r_, f_ = sq // 8, sq % 8
    return np.uint64(1) << np.uint64(r_ * 8 + (7 - f_))
  hist[0][6] = hist[0][6] | bit(48 + 3)          # extra pawn still on 48+f now
  rec = _mk_record(hist)
  sq = ds._records_to_arrays(np.array([rec], dtype=V6_DTYPE))[9][0]
  assert not (sq[:, 118] > 0).any(), 'EP 4th condition (!h0[48+f]) violated'
  print('PASS EP 4-condition suppression (doubled pawn case)')

  # --- policy floor: every legal move >= floor pre-normalization; no kept-mass renorm
  rec = _mk_record([sp], nlegal=30)
  probs = np.full(1858, -1.0, dtype=np.float32)
  probs[:30] = 0.0                                # legal but zero-visit moves
  probs[0] = 1.0
  rec['probs'] = probs
  pi, pv = (ds._records_to_arrays(np.array([rec], dtype=V6_DTYPE))[i] for i in (0, 1))
  vals = pv[0].astype(np.float32)
  nz = vals[vals > 0]
  assert len(nz) == 30, f'{len(nz)} nonzero policy slots, want 30 (floored legals)'
  expected_floor = POLICY_FLOOR / (1.0 + 29 * POLICY_FLOOR)
  assert abs(float(nz.min()) - expected_floor) < 1e-6, 'floor value wrong'
  assert abs(float(vals.sum()) - 1.0) < 1e-3, 'full mass should be kept (<=92 legals)'
  print('PASS policy floor 0.0005 per legal, no kept-mass renorm')

  # --- MLH clamp at 255
  rec = _mk_record([sp], plies_left=400.0)
  mlh = ds._records_to_arrays(np.array([rec], dtype=V6_DTYPE))[4]
  assert abs(float(mlh[0, 0]) - 2.55) < 1e-6, f'MLH clamp: {mlh[0,0]}'
  print('PASS MLH clamp at 255 plies')

  # --- unc NaN fallback 0.15
  rec = _mk_record([sp], orig_q=np.nan, best_q=0.7)
  unc = ds._records_to_arrays(np.array([rec], dtype=V6_DTYPE))[5]
  assert abs(float(unc[0, 0]) - 0.15) < 1e-6, f'unc NaN fallback: {unc[0,0]}'
  print('PASS uncertainty NaN-orig_q fallback = 0.15')

  # --- played_q_suboptimality from PREVIOUS record (game-order, pre-sampling)
  r0 = _mk_record([sp], best_q=0.5, played_q=0.1)
  r1 = _mk_record([sp], best_q=0.0, played_q=0.0)
  chunk = np.array([r0, r1], dtype=V6_DTYPE).tobytes()
  recs = ds._decode_chunk(chunk)
  pqs = ds._records_to_arrays(recs)[11]
  assert abs(float(pqs[0, 0])) < 1e-6 and abs(float(pqs[1, 0]) - 0.4) < 1e-6, \
      f'pqs shift wrong: {pqs.ravel()}'
  print('PASS played_q_suboptimality uses previous record')

  # --- z-integrity filter
  dsz = _bare(zmax=1.2)
  r_bad = _mk_record([sp], best_q=0.9, result_q=-1.0, result_d=0.0)   # delta 1.9
  r_ok = _mk_record([sp], best_q=0.2, result_q=1.0, result_d=0.0)     # delta 0.8
  recs = dsz._decode_chunk(np.array([r_bad, r_ok], dtype=V6_DTYPE).tobytes())
  assert len(recs) == 1 and dsz._zfiltered == 1, 'z-filter miscounted'
  print('PASS z-integrity filter')

  # --- version pinning: v6 chunk then v7 chunk
  dsv = _bare()
  assert dsv._decode_chunk(np.array([_mk_record([sp])], dtype=V6_DTYPE).tobytes()) is not None
  v7 = np.zeros(1, dtype=V7_DTYPE)
  v7['version'] = 7
  v7['input_format'] = 1
  assert dsv._decode_chunk(v7.tobytes()) is None and dsv._skipped_other_version == 1
  print('PASS version pinning (mixed v6/v7 rejected)')

  # --- unreadable data counted, not raised
  dse = _bare()
  assert dse._read_entry(('fs', '/nonexistent/x.gz')) is None and dse._read_errors == 1
  print('PASS read-error counting')


def real_data_tests(v6_dir, v7_dir):
  if v6_dir and os.path.isdir(v6_dir):
    ds = _bare(skip=10)
    files = sorted(glob.glob(os.path.join(v6_dir, '**', '*.gz'), recursive=True))[:50] \
        or [e for e in V6ChunkDataset._index_tar(sorted(
            glob.glob(os.path.join(v6_dir, '**', '*.tar'), recursive=True))[0])[:50]]
    seen = 0
    for f in files:
      data = ds._read_entry(f if isinstance(f, tuple) else ('fs', f))
      recs = ds._decode_chunk(data)
      if recs is None:
        continue
      out = ds._records_to_arrays(recs)
      assert np.allclose(out[9][:, :, 0:13].sum(axis=2), 1.0)
      assert np.isfinite(out[3]).all() and np.isfinite(out[4]).all()
      pv = out[1].astype(np.float32)
      assert float(pv.sum(axis=1).max()) <= 1.0 + 1e-3
      seen += len(recs)
    print(f'PASS real v6 data ({seen} records, finite targets, valid simplex)')
  if v7_dir and os.path.isdir(v7_dir):
    tars = sorted(glob.glob(os.path.join(v7_dir, '**', '*.tar'), recursive=True))
    if tars:
      ds = _bare(skip=10)
      entries = V6ChunkDataset._index_tar(tars[0])[:50]
      got_v7x = False
      for e in entries:
        recs = ds._decode_chunk(ds._read_entry(e))
        if recs is None:
          continue
        v7x = ds._records_to_arrays(recs)[13]
        if v7x is not None:
          got_v7x = True
          assert np.isfinite(v7x[0]).all() and np.isfinite(v7x[1]).all()
      assert got_v7x, 'no v7x extracted from v7 tar'
      print('PASS real v7 tar (v7x extracted, finite)')


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('--v6-dir', default='/mnt/f/cout/v6_chunks')
  ap.add_argument('--v7-dir', default='/mnt/e/_t91_v7_op1_sample')
  args = ap.parse_args()
  synthetic_tests()
  real_data_tests(args.v6_dir, args.v7_dir)
  print('V6_DATASET SMOKE: ALL PASS')
