"""q-deviation targets computed by the V6 loader (2026-10-05).

    python test_v6_qdev.py /path/to/v8_corpus [/path/to/tpg_dir]      (CPU)

1. Vectorized per-game targets == a literal port of the C# TPG generator loop
   (TrainingPositionGeneratorGameRescorer.CalcForwardBlunders) on real games.
2. Ply perspective: best_q alternates side to move (corr(best_q[i], best_q[i+1]) strongly negative).
3. Distribution next to the TPG corpus' stored targets (optional second argument).
"""
import os, sys
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
import numpy as np
from v6_dataset import V6ChunkDataset


def ref_qdev(bq):
  # literal C# loop, NaN included (a NaN comparison is false -> that ply is skipped)
  n = len(bq); lo = np.zeros(n); hi = np.zeros(n)
  for i in range(n):
    mn = mx = 0.0
    for j in range(i, n):
      q = bq[j] if (j - i) % 2 == 0 else -bq[j]
      d = q - bq[i]
      if d < 0 and abs(d) > mn: mn = abs(d)
      elif d > 0 and abs(d) > mx: mx = abs(d)
    lo[i], hi[i] = mn, mx
  return lo, hi


# 0. NaN handling on a synthetic game (review 2026-10-05): matches the C# loop
import v6_dataset as v6
syn = np.zeros(4, dtype=v6.V6_DTYPE); syn['version'] = 6; syn['input_format'] = 1
syn['best_q'] = [0.3, -0.2, np.nan, 0.9]
_ds0 = V6ChunkDataset.__new__(V6ChunkDataset)
for k, v in dict(_version=None, _skipped_other_version=0, _ragged_chunks=0, max_resultq_delta=0.0,
                 _zfiltered=0, skip_count=1, _skipped_formats=0).items():
  setattr(_ds0, k, v)
r = _ds0._decode_chunk(syn.tobytes())
lo0, hi0 = ref_qdev(np.array([0.3, -0.2, np.nan, 0.9]))
assert np.allclose(r['best_m'], lo0) and np.allclose(r['orig_m'], hi0), (r['best_m'], lo0, r['orig_m'], hi0)
print('OK NaN: a NaN ply is skipped and a NaN baseline gives 0, as in C#', list(np.round(r['best_m'], 3)))


ds = V6ChunkDataset(sys.argv[1], 256, 0.0, 0, 1, 0, 1, 0, False, max_resultq_delta=0.0)
ds.skip_count = 1                     # whole games: the reference needs every ply
games, maxerr, corr_pairs = 0, 0.0, []
all_lo, all_hi = [], []
for e in ds.files[:400]:
  recs = ds._decode_chunk(ds._read_entry(e))
  if recs is None or len(recs) < 4:
    continue
  bq = recs['best_q'].astype(np.float64)
  lo, hi = ref_qdev(bq)
  maxerr = max(maxerr, float(np.abs(lo - recs['best_m']).max()), float(np.abs(hi - recs['orig_m']).max()))
  corr_pairs.append(np.corrcoef(bq[:-1], bq[1:])[0, 1])
  arr = ds._records_to_arrays(recs)
  assert arr[7].dtype == np.float16 and arr[7].shape == (len(recs), 1) and arr[8].shape == (len(recs), 1)
  assert np.allclose(arr[7].ravel(), lo, atol=2e-3) and np.allclose(arr[8].ravel(), hi, atol=2e-3), 'tuple slots 7/8'
  all_lo.append(arr[7].astype(np.float32).ravel()); all_hi.append(arr[8].astype(np.float32).ravel())
  games += 1
assert games > 100, games
assert maxerr < 1e-5, maxerr
print(f'OK reference: {games} games, max |vectorized - C# port| = {maxerr:.2e}; tuple slots 7/8 = lower/upper, float16 [n,1]')
c = float(np.nanmedian(corr_pairs))
assert c < -0.5, c
print(f'OK perspective: median corr(best_q[i], best_q[i+1]) = {c:.3f} (side to move alternates)')
lo, hi = np.concatenate(all_lo), np.concatenate(all_hi)
q = lambda x: ' '.join(f'{v:.3f}' for v in np.quantile(x, [0.1, 0.5, 0.9, 0.99]))
print(f'cv4  lower mean {lo.mean():.3f} q10/50/90/99 {q(lo)} | upper mean {hi.mean():.3f} q {q(hi)}  (n {len(lo)})')
if len(sys.argv) > 2:
  from tpg_dataset import TPGDataset
  t = TPGDataset(sys.argv[2], 1024, 0.0, 0, 1, 0, 1, 0, False)
  tl, th = [], []
  for k in range(20):
    b = t[k]; b = b[0] if isinstance(b, tuple) else b
    tl.append(b['q_deviation_lower'].numpy().ravel()); th.append(b['q_deviation_upper'].numpy().ravel())
  tl, th = np.concatenate(tl), np.concatenate(th)
  print(f'TPG  lower mean {tl.mean():.3f} q10/50/90/99 {q(tl)} | upper mean {th.mean():.3f} q {q(th)}  (n {len(tl)})')
