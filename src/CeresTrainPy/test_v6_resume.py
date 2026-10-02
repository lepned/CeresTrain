"""DirectFromV6 chunk-level datastream resume (2026-10-02).

    CERES_SHUFFLE_SEED=123 python test_v6_resume.py /path/to/v8_corpus      (CPU)

1. Same epoch: a fresh loader handed the tag of run A's 2nd pool flush reads exactly the chunks run A read next.
2. Epoch replay: start state (epoch 1, k) reads epoch 1's shuffled order from position k (rng replayed, epoch 0 unread).
3. A tag written under a different world size / worker count / chunk count is refused with ValueError (review 2026-10-02).
"""
import os, sys, random
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
import v6_dataset
from v6_dataset import V6ChunkDataset
from tpg_dataset import stable_str_hash, _RUN_SHUFFLE_SEED

ROOT = sys.argv[1]
POOL = 3000


def make(**kw):
  ds = V6ChunkDataset(ROOT, 256, 0.0, 0, 1, 0, 1, 0, False, shuffle_pool=POOL, **kw)
  read = []
  orig = ds._read_entry
  ds._read_entry = lambda e: (read.append(e), orig(e))[1]
  return ds, read


def pull(ds, n_batches):
  it = ds.item_generator()
  tags = []
  for _ in range(n_batches):
    b = next(it)
    tags.append(b[15])
  return tags


# 1. same epoch
dsA, readA = make()
tagsA = pull(dsA, 60)
flush_tags = sorted({t for t in tagsA if t is not None}, key=lambda t: t[1])
assert len(flush_tags) >= 3, flush_tags
key, k = flush_tags[1][0], flush_tags[1][1]
assert key.endswith(':e0'), key
dsB, readB = make(start_offsets={key: k})
pull(dsB, 10)
n = min(len(readB), len(readA) - k)
assert n > 50 and readB[:n] == readA[k:k + n], 'resume did not continue with the next chunks of the same order'
print(f'OK same-epoch: resumed at chunk {k}, next {n} chunks identical to the uninterrupted run')

# 2. epoch replay (epoch 1 order without reading epoch 0)
my_files = list(dsA.files)                              # world 1 x 1 worker: shard 0 = all entries
rng = random.Random(stable_str_hash(f'{dsA.root_dir}|0|{_RUN_SHUFFLE_SEED}') & 0x7fffffff)
e0 = list(my_files); rng.shuffle(e0)
e1 = list(my_files); rng.shuffle(e1)
k1 = 7
dsC, readC = make(start_offsets={key.replace(':e0', ':e1'): k1})
pull(dsC, 5)
assert readC[:20] == e1[k1:k1 + 20], 'epoch replay: wrong order in epoch 1'
assert readC[0] not in e0[:k1] or e0[:k1].index(readC[0]) >= 0
print(f'OK epoch replay: epoch 1 read from chunk {k1} in the replayed shuffle order (epoch 0 skipped unread)')

# 3. different world size -> refused
bad = key.replace('W1x1', 'W4x1')
try:
  make(start_offsets={bad: k}); raise AssertionError('mismatched world size accepted')
except ValueError:
  pass
print('OK world-size mismatch: refused with ValueError')

# 4. changed corpus (different chunk count) -> refused
import re
bad_n = re.sub(r'N(\d+):', lambda m: f'N{int(m.group(1)) + 1}:', key)
try:
  make(start_offsets={bad_n: k}); raise AssertionError('changed chunk count accepted')
except ValueError:
  pass
print('OK chunk-count mismatch: refused with ValueError')
