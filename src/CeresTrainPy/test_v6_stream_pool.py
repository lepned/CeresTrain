"""Contract test for the streaming shuffle pool of the DirectFromV6 loader (V6StreamPool, 2026-10-07).

    python test_v6_stream_pool.py            (CPU, synthetic chunks; no corpus)

Runs the REAL item_generator with the chunk reader / decoder / converter replaced by stubs that carry a unique record id.
1. No record is emitted twice within an epoch; records are conserved (emitted + still pooled == read).
2. Steady output: after the pool is full, the number of chunks read between two emitted blocks is small
   (~EMIT / records-per-chunk), versus ~pool_size / records-per-chunk for the original fill/flush mode.
3. Mixing: emitted records come from across the pool (mean age ~ pool size), not FIFO.
4. Original mode (stream off) still emits every record exactly once per flush cycle.
"""
import os, sys
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v6_dataset
from v6_dataset import V6ChunkDataset

DT = np.dtype([('id', '<i8')])
RPC = 13           # records per chunk (~ a game at slots 8)
NCHUNK = 20000
POOL = 50000       # production pool size


def make(stream, pool=POOL, B=64):
  d = object.__new__(V6ChunkDataset)
  d.batch_size, d.rank, d.world_size, d.num_workers, d.worker_id = B, 0, 1, 1, 0
  d.files = [('fs', f'/fake/{i}.gz') for i in range(NCHUNK)]
  d.root_dir = '/fake'; d._v6_key_prefix = '__v6__:test:'; d._v6_resume = {}
  d.pool_size = pool; d.stream_pool = stream
  d._read_errors = d._skipped_other_version = d._skipped_formats = d._ragged_chunks = d._zfiltered = 0
  d.reads = 0
  d.arrived = 0                               # arrival order of records (the pool's notion of age)
  d.order = {}

  def read_entry(entry):
    d.reads += 1
    return entry

  def decode(entry, entry2=None, epoch=0):
    i = int(entry[1].split('/')[-1].split('.')[0])
    r = np.zeros(RPC, dtype=DT)
    r['id'] = (epoch * NCHUNK + i) * RPC + np.arange(RPC)
    for k, rid in enumerate(r['id'].tolist()):
      d.order[rid] = d.arrived + k
    d.arrived += RPC
    return r

  def to_arrays(recs):
    ids = recs['id']
    return tuple([ids] + [ids] * 12 + [None, None])

  d._read_entry = read_entry
  d._decode_chunk = lambda data, entry=None, epoch=None: decode(data, epoch=epoch)
  d._records_to_arrays = to_arrays
  return d


def drive(d, n_batches):
  np.random.seed(0)
  it = d.item_generator()
  out, reads_at = [], []
  for _ in range(n_batches):
    b = next(it)
    out.append(b[0].copy())
    reads_at.append(d.reads)
  return out, reads_at


def main():
  # 1-3 streaming
  d = make(True)
  out, reads_at = drive(d, 2500)
  ids = np.concatenate(out)
  assert len(ids) == len(set(ids.tolist())), 'duplicate emission'
  assert all(len(b) == 64 for b in out)
  read_records = d.reads * RPC
  # conservation: everything read is either emitted or still in the pool (pool between pool_size and pool+EMIT+chunk)
  pooled = read_records - len(ids)
  assert POOL <= pooled <= POOL + 2048 + RPC, pooled
  gaps = np.diff(reads_at)
  steady = gaps[len(gaps) // 2:]
  max_gap = int(steady.max())
  assert max_gap <= 2048 // RPC + 2, f'streaming gap {max_gap} chunks'
  # age = records that arrived after it, at the moment of emission (true arrival order; review 10-07: the file-index
  # id was uncorrelated with read order because item_generator shuffles the files)
  ages = []
  for b, ra in zip(out[len(out) // 2:], reads_at[len(out) // 2:]):
    ages += [ra * RPC - 1 - d.order[int(x)] for x in b]
  ages = np.array(ages)
  assert 0.7 * POOL < ages.mean() < 1.5 * (POOL + 2048), f'mean age {ages.mean():.0f} vs pool {POOL}'
  assert (ages < 2048).mean() < 0.15 and ages.max() > 2 * POOL, 'age distribution should be reservoir-like (FIFO would be ~constant)'
  print(f'OK streaming: no duplicates, conserved (pooled {pooled}), max chunks between emits {max_gap}, '
        f'mean record age {ages.mean():.0f} (pool {POOL})')

  # 4 original mode, for contrast + regression
  d0 = make(False)
  out0, reads0 = drive(d0, 2500)
  ids0 = np.concatenate(out0)
  assert len(ids0) == len(set(ids0.tolist()))
  gap0 = int(np.diff(reads0).max())
  assert gap0 >= POOL // RPC - 2, gap0
  print(f'OK original mode: no duplicates; max chunks between emits {gap0} (whole pool refill) vs {max_gap} streaming')
  # epoch wrap + production batch 384 (EMIT 1920): a small corpus read for several epochs, ids unique per epoch
  global NCHUNK
  NCHUNK = 600
  d2 = make(True, pool=3000, B=384)
  out2, _ = drive(d2, 120)
  ids2 = np.concatenate(out2)
  assert all(len(b) == 384 for b in out2)
  assert len(ids2) == len(set(ids2.tolist())), 'duplicate across epoch wrap'
  epochs = set((ids2 // RPC // NCHUNK).tolist())
  assert len(epochs) >= 3, epochs
  print(f'OK epoch wrap: {len(ids2)} records over epochs {sorted(epochs)}, B=384, no duplicates')
  # dtype change mid-stream raises instead of silently dropping the pool
  d3 = make(True, pool=200)
  it = d3.item_generator(); next(it)
  d3._decode_chunk = lambda data, entry=None, epoch=None: np.zeros(RPC, dtype=np.dtype([('id', '<i4')]))
  try:
    for _ in range(50):
      next(it)
    raise SystemExit('FAIL: dtype change accepted')
  except RuntimeError as e:
    assert 'dtype changed' in str(e)
  print('OK dtype change mid-stream raises')
  print('ALL OK')


if __name__ == '__main__':
  main()
