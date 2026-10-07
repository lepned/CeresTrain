"""Contract tests for the rotating disjoint slots of the DirectFromV6 loader (V6SampleSlots, 2026-10-07).

    python test_v6_sample_slots.py            (CPU, synthetic v8 games, no corpus needed)

1. slot_of_records: balanced (each slot n/S +-1), deterministic for the same chunk key, different across keys.
2. Over S consecutive epochs the kept records of a game are DISJOINT and cover the WHOLE game exactly once; epoch e and
   e + S keep the same records (the rotation repeats; resume re-derives it from the epoch alone).
3. Game-order targets (q-deviation lower/upper, played_q suboptimality) of a kept record equal those of the full-game
   decode, i.e. they are still computed on the whole game BEFORE the slot filter.
4. A record's slot does not depend on the z-integrity filter (filter-on kept set == filter-off kept set minus the dropped).
5. Config guards: S = 1 and S with V6SkipCount > 1 are refused.
"""
import os, sys
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v6_dataset import V6ChunkDataset, V8_DTYPE


def game_bytes(n, seed):
  rng = np.random.default_rng(seed)
  recs = np.zeros(n, dtype=V8_DTYPE)
  recs['version'] = 8
  recs['input_format'] = 1
  recs['played_idx'] = np.arange(n)
  recs['best_q'] = rng.uniform(-0.9, 0.9, n).astype(np.float32)
  recs['played_q'] = recs['best_q'] - rng.uniform(0, 0.2, n).astype(np.float32)
  recs['result_q'] = np.where(rng.random(n) < 0.3, 1.0, recs['best_q']).astype(np.float32)
  return recs.tobytes()


def bare(slots, zfilter=0.0):
  d = object.__new__(V6ChunkDataset)
  d._version = None
  d._skipped_other_version = d._skipped_formats = d._ragged_chunks = d._zfiltered = 0
  d.max_resultq_delta = zfilter
  d.skip_count = 1
  d.sample_slots = slots
  return d


def ids(recs):
  return set(int(i) for i in recs['played_idx']) if recs is not None else set()


def main():
  # 1
  for n, S in ((101, 4), (64, 8), (7, 8)):
    s = V6ChunkDataset.slot_of_records(n, 'k', S)
    counts = np.bincount(s, minlength=S)
    assert counts.max() - counts.min() <= 1, counts
    assert np.array_equal(s, V6ChunkDataset.slot_of_records(n, 'k', S))
  assert not np.array_equal(V6ChunkDataset.slot_of_records(101, 'a', 4), V6ChunkDataset.slot_of_records(101, 'b', 4))
  print('OK slots: balanced, deterministic per chunk key, different across keys')

  # 2 + 3
  n, S = 111, 4
  data = game_bytes(n, 1)
  entry = ('fs', '/corpus/cell/game_000123.gz')
  full = bare(0)._decode_chunk(data)
  by_id = {int(r['played_idx']): r for r in full}
  seen = []
  for e in range(S):
    recs = bare(S)._decode_chunk(data, entry=entry, epoch=e)
    seen.append(ids(recs))
    for r in recs:
      f = by_id[int(r['played_idx'])]
      for fld in ('best_m', 'orig_m', 'played_m'):
        assert r[fld] == f[fld], (fld, int(r['played_idx']))
  for a in range(S):
    for b in range(a + 1, S):
      assert not (seen[a] & seen[b]), (a, b)
  assert set().union(*seen) == ids(full) and len(ids(full)) == n   # decode marks the last ply's played_idx 0xFFFF
  assert ids(bare(S)._decode_chunk(data, entry=entry, epoch=S + 2)) == seen[2], 'rotation must repeat with period S'
  # no entry/epoch (startup diagnosis path) -> no slot filtering
  assert len(bare(S)._decode_chunk(data)) == n
  print(f'OK rotation: {S} epochs disjoint, cover all {n} records once, period S; game-order targets computed on the full game')

  # 4 z-filter independence
  for e in range(S):
    off = ids(bare(S)._decode_chunk(data, entry=entry, epoch=e))
    d = bare(S, zfilter=0.5)
    on = ids(d._decode_chunk(data, entry=entry, epoch=e))
    zdrop = {i for i, r in by_id.items() if abs(float(r['best_q']) - float(r['result_q'])) > 0.5}
    assert on == off - zdrop, e
  print('OK z-filter: slots assigned before filtering (kept set = unfiltered kept set minus filtered records)')

  # 5 guards (raised before any corpus access)
  for kw in ({'sample_slots': 1, 'skip_count': 1}, {'sample_slots': 4, 'skip_count': 4}):
    try:
      V6ChunkDataset('/nonexistent', 16, 0.0, 0, 1, 1, 1, **kw)
      raise SystemExit(f'FAIL: {kw} accepted')
    except ValueError:
      pass
  print('OK guards: S=1 and S with V6SkipCount>1 refused')
  print('ALL OK')


if __name__ == '__main__':
  main()
