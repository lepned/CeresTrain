# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""Direct-from-LC0-chunk training dataset (2026-08-21).

Reads loose .gz game chunks (LC0 v6 training records, one game per file — the
format the lczero-training pipeline consumes and the format LC0 training data
is distributed in) and yields batches in EXACTLY the same tuple layout as
tpg_dataset.TPGDataset.item_generator, so train.py can consume either source
interchangeably. Selected via Data config `"SourceType": "DirectFromV6"` with
`TrainingFilesDirectory` pointing at a directory tree containing .gz chunks.

WHY: storage. The TPG path materializes a second full-size corpus on disk;
this path trains straight from the chunk files already on disk (e.g. a nearly
full server holding pre-rescored LC0 data). Throughput measured 2026-08-21:
gen-tpg full-analysis conversion runs ~16M pos/h — below training consumption
— so real-time C#-side streaming was rejected in favor of this direct reader.

WHAT IT DOES NOT DO (by design, documented trade-offs vs the TPG path):
  - No tablebase rescoring / deblunder: train on PRE-RESCORED chunks (the
    LC0 rescorer pipeline), like lczero-training does. Fields that TPG's
    deblunder pass would rewrite (wdl_deblundered) fall back to the chunk's
    (possibly rescored) result.
  - No aux feature channels: 137-byte square layout only (requires
    CERES_AUX_FEATURES_PER_SQUARE=0). Rationale: the qz no-aux ablation
    showed aux ≈ no-aux.
  - Blunder-counter squares bytes (QPositiveBlunders/QNegativeBlunders),
    PlySinceLastMove and q_deviation targets are ZERO (not derivable from v6;
    our own TPG gen writes zeros for ply-since already; set
    LossQDeviationMultiplier 0 in configs used with this source).
  - Survival sidecars / v7x: not available (None).

Square encoding matches TPGSquareRecord (V2, 137 B) with the trainer-side
divisor already applied (floats, one-hot = 1.0):
    [0..103]   8 history positions x 13 piece one-hot (empty, our PNBRQK, their pnbrqk)
    [104..111] per-history repetition flags
    [112..115] CanOO, CanOOO, OpponentCanOO, OpponentCanOOO
    [116]      Move50Count / 100
    [117]      PlySinceLastMove (zeros)
    [118]      IsEnPassant (derived from history-plane diff: their double pawn push)
    [119,120]  QPositiveBlunders, QNegativeBlunders (zeros)
    [121..128] rank one-hot
    [129..136] file one-hot

Input format support: v6 records with input_format 1 (classic castling
booleans + stm byte). Other formats are counted and skipped loudly.
"""

import os
import glob
import random
import zlib
import numpy as np

# Profiled 2026-08-21: gzip decompression is ~93% of decode cost. python-isal
# (libdeflate bindings) decompresses 2-3x faster and is a drop-in replacement;
# use it opportunistically. Scaling beyond that = dataloader workers (each
# worker decompresses its own disjoint file shard).
try:
  from isal import igzip as gzip
  _GZIP_IMPL = 'isal'
except ImportError:
  import gzip
  _GZIP_IMPL = 'gzip'

from config import NUM_AUX_FEATURES_PER_SQUARE
from tpg_dataset import stable_str_hash, TPGDataset

MAX_MOVES = 92          # TPGRecord.MAX_MOVES — top-K policy slots
V6_RECORD_BYTES = 8356

# LC0 v6 record as a numpy structured dtype (little-endian throughout).
V6_DTYPE = np.dtype([
    ('version', '<u4'), ('input_format', '<u4'),
    ('probs', '<f4', (1858,)),
    ('planes', '<u8', (104,)),
    ('us_ooo', 'u1'), ('us_oo', 'u1'), ('them_ooo', 'u1'), ('them_oo', 'u1'),
    ('stm_or_ep', 'u1'), ('rule50', 'u1'), ('invariance', 'u1'), ('dep_result', 'i1'),
    ('root_q', '<f4'), ('best_q', '<f4'), ('root_d', '<f4'), ('best_d', '<f4'),
    ('root_m', '<f4'), ('best_m', '<f4'), ('plies_left', '<f4'),
    ('result_q', '<f4'), ('result_d', '<f4'),
    ('played_q', '<f4'), ('played_d', '<f4'), ('played_m', '<f4'),
    ('orig_q', '<f4'), ('orig_d', '<f4'), ('orig_m', '<f4'),
    ('visits', '<u4'), ('played_idx', '<u2'), ('best_idx', '<u2'),
    ('reserved', '<u2', (4,)),
])
assert V6_DTYPE.itemsize == V6_RECORD_BYTES, V6_DTYPE.itemsize

# LC0 v7 record = the v6 record + a 40-byte tail (EncodedTrainingPositionExtraV7):
# short-term D, opp/next played move indices, z-provenance (0=orig result,
# 1=syzygy, 2=deblunder-noise, 3=deblunder-unintended, 4=op1-8man; stored as
# float), censored short-term q/d, q after played move, 4 reserved floats.
# The censored/provenance fields are EXACTLY what the TPG pipeline ships as
# .v7x sidecars, so v7 chunks feed the trainer's v7x consumers natively.
V7_RECORD_BYTES = 8396
V7_DTYPE = np.dtype(V6_DTYPE.descr + [
    ('d_short_term', '<f4'),
    ('opp_played_idx', '<u2'), ('next_played_idx', '<u2'),
    ('z_provenance', '<f4'),
    ('cens_q_st', '<f4'), ('cens_d_st', '<f4'),
    ('q_after_played', '<f4'),
    ('reserved47', '<f4', (4,)),
])
assert V7_DTYPE.itemsize == V7_RECORD_BYTES, V7_DTYPE.itemsize

# Config keys V6SkipCount / V6ShufflePool bridge to these envs via
# config_bootstrap (config-first convention; env override still works).
_SKIP_COUNT = int(os.environ.get('CERES_V6_SKIP_COUNT', '30') or 30)
# Record-level shuffle pool (positions pooled + shuffled before batches are
# emitted), ON TOP of per-worker shuffled file order. Pools hold RAW records
# (8.4 KB each): RAM = pool * workers * 8.4 KB — 200k * 8 workers = 13 GB
# OOM-killed a worker on 2026-08-21; 50k * 8 = 3.3 GB is the safe default.
_POOL_SIZE = int(os.environ.get('CERES_V6_SHUFFLE_POOL', '50000') or 50000)


def _wdl_from_qd(q, d):
  """(q, d) -> [w, d, l] rows, clipped to valid simplex."""
  w = np.clip((1.0 + q - d) * 0.5, 0.0, 1.0)
  l = np.clip((1.0 - q - d) * 0.5, 0.0, 1.0)
  dd = np.clip(1.0 - w - l, 0.0, 1.0)
  return np.stack([w, dd, l], axis=1).astype(np.float32)


class V6ChunkDataset(TPGDataset):
  """LC0 v6 .gz chunk directories as a TPGDataset.

  Subclasses TPGDataset to inherit __getitem__ (the batch -> torch-dict
  conversion incl. policy scatter and boards_per_batch splitting), __len__
  and set_worker_id; overrides __init__ (no super().__init__ — different
  discovery) and item_generator (v6 decode instead of shard reads). The
  generator attribute is created here in the parent process but its body
  first executes inside the DataLoader worker (after fork + set_worker_id),
  same lazy-sharding pattern as TPGDataset itself."""

  def __init__(self, root_dir,
               batch_size: int,
               wdl_smoothing: float,
               rank: int,
               world_size: int,
               num_workers: int,
               boards_per_batch: int,
               num_files_to_skip: int = 0,
               test: bool = False,
               file_mirror_prob: float = None,
               **_ignored):
    assert NUM_AUX_FEATURES_PER_SQUARE == 0, \
        'DirectFromV6 requires CERES_AUX_FEATURES_PER_SQUARE=0 (137-channel model)'
    if wdl_smoothing:
      raise NotImplementedError('WDLLabelSmoothing not supported by DirectFromV6 (set 0)')
    if file_mirror_prob:
      raise NotImplementedError('FileMirrorAug not supported by DirectFromV6 yet (set 0)')
    self.root_dir = root_dir
    self.batch_size = batch_size
    self.rank = rank
    self.world_size = world_size
    self.num_workers = max(1, num_workers)
    self.worker_id = 0
    self.skip_count = _SKIP_COUNT
    # Two source kinds, freely mixed under root_dir:
    #   ('fs',  path)                       — a loose .gz chunk file
    #   ('tar', tar_path, offset, size)     — a .gz member INSIDE an (uncompressed)
    #                                         LC0 tar, read via seek+read with no
    #                                         extraction (zero extra storage).
    # Tar member tables are indexed once at startup (headers only) and cached
    # as <tar>.chunkindex.npz next to the tar when the directory is writable.
    entries = [('fs', p) for p in
               sorted(glob.glob(os.path.join(root_dir, '**', '*.gz'), recursive=True))]
    for tp in sorted(glob.glob(os.path.join(root_dir, '**', '*.tar'), recursive=True)):
      entries += self._index_tar(tp)
    if num_files_to_skip:
      entries = entries[num_files_to_skip:]
    if not entries:
      raise FileNotFoundError(f'DirectFromV6: no .gz chunks or .tar archives under {root_dir}')
    # Run-level entry shuffle (same CERES_SHUFFLE_SEED convention as
    # tpg_dataset.try_shuffle, which itself can't be used here — it groups
    # by '.tpg' filename prefixes and our entries are tuples).
    _seed = os.environ.get('CERES_SHUFFLE_SEED')
    random.Random(int(_seed) if _seed else None).shuffle(entries)
    self.files = entries
    self._tar_handles = {}
    self._skipped_formats = 0
    self.boards_per_batch = boards_per_batch
    _n_tar = sum(1 for e in entries if e[0] == 'tar')
    print(f'[v6_dataset] {root_dir}: {len(entries):,} chunks ({_n_tar:,} in-tar), '
          f'skip_count={self.skip_count}, pool={_POOL_SIZE:,}, '
          f'rank {rank}/{world_size}, gzip={_GZIP_IMPL}')
    self.generator = self.item_generator()

  # ---- tar support ------------------------------------------------------

  @staticmethod
  def _index_tar(tar_path):
    """Member table for an uncompressed LC0 tar: [('tar', path, offset, size), ...]."""
    import tarfile
    cache = tar_path + '.chunkindex.npz'
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(tar_path):
      z = np.load(cache)
      return [('tar', tar_path, int(o), int(s)) for o, s in zip(z['offsets'], z['sizes'])]
    offsets, sizes = [], []
    with tarfile.open(tar_path, 'r:') as tf:      # 'r:' = uncompressed only (no surprise gunzip of the container)
      for m in tf:
        if m.isfile() and m.name.endswith('.gz'):
          offsets.append(m.offset_data)
          sizes.append(m.size)
    try:
      np.savez_compressed(cache, offsets=np.asarray(offsets, dtype=np.int64),
                          sizes=np.asarray(sizes, dtype=np.int64))
    except OSError:
      pass                                        # read-only dir (nearly-full server): index stays in-memory
    print(f'[v6_dataset] indexed {os.path.basename(tar_path)}: {len(offsets):,} chunk members')
    return [('tar', tar_path, o, s) for o, s in zip(offsets, sizes)]

  def _read_entry(self, entry):
    """Entry -> decompressed chunk bytes (None on read error)."""
    try:
      if entry[0] == 'fs':
        with gzip.open(entry[1], 'rb') as f:
          return f.read()
      _, tar_path, offset, size = entry
      fh = self._tar_handles.get(tar_path)
      if fh is None:
        fh = open(tar_path, 'rb', buffering=0)
        self._tar_handles[tar_path] = fh
      fh.seek(offset)
      return gzip.decompress(fh.read(size))
    except (OSError, EOFError, zlib.error):
      return None

  # ---- decoding ---------------------------------------------------------

  def _decode_chunk(self, data: bytes):
    """Bytes of one .gz-decompressed chunk -> structured record array (downsampled)."""
    if len(data) < 8:
      return None
    ver = int(np.frombuffer(data[:4], dtype='<u4')[0])
    if ver == 6:
      rec_bytes, dtype = V6_RECORD_BYTES, V6_DTYPE
    elif ver == 7:
      rec_bytes, dtype = V7_RECORD_BYTES, V7_DTYPE
    else:
      return None
    n = len(data) // rec_bytes
    if n == 0:
      return None
    recs = np.frombuffer(data, dtype=dtype, count=n)
    # Downsample 1/skip_count to decorrelate sequential game positions
    # (same role as TPG PositionSkipCount / lc0 SKIP).
    if self.skip_count > 1:
      keep = np.random.random(n) < (1.0 / self.skip_count)
      if not keep.any():
        return None
      recs = recs[keep]
    # Only classic input format 1 supported (see module header).
    fmt_ok = recs['input_format'] == 1
    if not fmt_ok.all():
      self._skipped_formats += int((~fmt_ok).sum())
      recs = recs[fmt_ok]
      if len(recs) == 0:
        return None
    return recs

  def _records_to_arrays(self, recs):
    """Structured records -> the per-position target/input arrays (numpy)."""
    n = len(recs)

    # ---- planes -> squares[137] ----
    # LC0 u64 planes, our perspective: byte b of the (little-endian) u64 is
    # rank b, and within each byte the MSB is file a (verified empirically
    # via the startpos anchor: LSB-first decoding file-mirrors the board —
    # kings land on d-file).
    plane_bytes = recs['planes'].view(np.uint8).reshape(n, 104, 8)
    bits = np.unpackbits(plane_bytes, axis=2, bitorder='big')           # [n, 104, 64]
    planes = bits.astype(np.float32).transpose(0, 2, 1)                 # [n, 64, 104]

    squares = np.zeros((n, 64, 137), dtype=np.float32)
    # 8 history positions: v6 plane order per position = our PNBRQK, their pnbrqk, repetition
    for h in range(8):
      p0 = h * 13
      pc = planes[:, :, p0:p0 + 12]                                     # 12 piece planes
      occ = pc.sum(axis=2)
      squares[:, :, h * 13 + 0] = 1.0 - np.minimum(occ, 1.0)            # empty flag
      squares[:, :, h * 13 + 1: h * 13 + 13] = pc                       # our 1-6, their 7-12
      squares[:, :, 104 + h] = planes[:, :, p0 + 12]                    # repetition flag
    # castling (broadcast to all squares): CanOO, CanOOO, OppCanOO, OppCanOOO
    squares[:, :, 112] = (recs['us_oo'] > 0)[:, None]
    squares[:, :, 113] = (recs['us_ooo'] > 0)[:, None]
    squares[:, :, 114] = (recs['them_oo'] > 0)[:, None]
    squares[:, :, 115] = (recs['them_ooo'] > 0)[:, None]
    squares[:, :, 116] = (recs['rule50'].astype(np.float32) / 100.0)[:, None]
    # [117] PlySinceLastMove: zeros (matches our own TPG gen).
    # [118] IsEnPassant: their pawn double push detected from history diff —
    # their pawn present at 32+f now (h0), present at 48+f before (h1), and
    # absent at 32+f before. EP square = 40+f.
    their_p_h0 = planes[:, :, 6]                                        # their pawns, current
    their_p_h1 = planes[:, :, 13 + 6]                                   # their pawns, 1 ply ago
    for f in range(8):
      dbl = (their_p_h0[:, 32 + f] > 0) & (their_p_h1[:, 48 + f] > 0) & (their_p_h1[:, 32 + f] == 0)
      squares[dbl, 40 + f, 118] = 1.0
    # [119,120] blunder counters: zeros.
    ranks = np.arange(64) // 8
    files_ = np.arange(64) % 8
    squares[:, np.arange(64), 121 + ranks] = 1.0
    squares[:, np.arange(64), 129 + files_] = 1.0

    # ---- policy: dense 1858 (visits dist, -1 = illegal) -> top-92 ----
    probs = np.nan_to_num(recs['probs'], nan=0.0)                       # NaN rows exist in raw data
    probs[probs < 0] = 0.0                                              # illegal -> 0
    k = MAX_MOVES
    part = np.argpartition(-probs, k - 1, axis=1)[:, :k]                # top-k indices (unordered)
    vals = np.take_along_axis(probs, part, axis=1)
    order = np.argsort(-vals, axis=1)                                   # sort slots by prob desc
    policies_indices = np.take_along_axis(part, order, axis=1).astype(np.int16)
    policies_values = np.take_along_axis(vals, order, axis=1)
    s = policies_values.sum(axis=1, keepdims=True)
    s = np.where(s > 0, s, 1.0)
    policies_values = (policies_values / s).astype(np.float16)          # renormalize over kept mass

    # ---- value / aux targets ----
    wdl_q = _wdl_from_qd(recs['best_q'], recs['best_d'])
    wdl_result = _wdl_from_qd(recs['result_q'], recs['result_d'])       # (rescored) game outcome
    wdl_deblundered = wdl_result                                        # no deblunder pass (see header)
    wdl_nondeblundered = wdl_result
    played_q_subopt = np.maximum(recs['best_q'] - recs['played_q'], 0.0).astype(np.float32).reshape(-1, 1)
    unc_policy = np.nan_to_num(np.frombuffer(recs['reserved'].tobytes(), dtype='<f4')
                               .reshape(-1, 2)[:, 0]).astype(np.float32).reshape(-1, 1)
    # ^ v6 stores policy_kld in the first 4 reserved bytes (chunkparser: pol_kld).
    unc_policy = np.abs(unc_policy)
    orig_q = np.nan_to_num(recs['orig_q'], nan=0.0)
    uncertainty = np.abs(recs['best_q'] - orig_q).astype(np.float32).reshape(-1, 1)
    mlh = (recs['plies_left'].astype(np.float32) / 100.0).reshape(-1, 1)
    zeros16 = np.zeros((n, 1), dtype=np.float16)
    pip = np.full((n, 1), -1, dtype=np.int16)                           # policy_index_in_parent: unused

    # v7 tail -> v7x extras (same fields the TPG pipeline ships as .v7x
    # sidecars: censored short-term q/d + z-provenance code).
    v7x = None
    if 'cens_q_st' in recs.dtype.names:
      v7x = (np.nan_to_num(recs['cens_q_st']).astype(np.float32).reshape(-1, 1),
             np.nan_to_num(recs['cens_d_st']).astype(np.float32).reshape(-1, 1),
             np.nan_to_num(recs['z_provenance']).astype(np.uint8).reshape(-1, 1))

    return (policies_indices, policies_values, wdl_deblundered, wdl_q, mlh,
            uncertainty, wdl_nondeblundered, zeros16, zeros16.copy(), squares,
            pip, played_q_subopt, unc_policy, v7x)

  # ---- generation -------------------------------------------------------

  def item_generator(self):
    B = self.batch_size
    # Worker/rank file sharding: same convention as TPGDataset (disjoint slices).
    shard = self.rank * self.num_workers + self.worker_id
    num_shards = self.world_size * self.num_workers
    my_files = self.files[shard::num_shards]
    if not my_files:
      my_files = self.files
    rng = random.Random(stable_str_hash(f'{self.root_dir}|{shard}') & 0x7fffffff)

    # Pool RAW sampled records (structured arrays) and convert to target
    # arrays in one vectorized pass per pool flush — per-chunk conversion
    # calls were pure numpy-call overhead (few sampled records per game).
    rec_pool = []
    pool_count = 0
    carry = None       # leftover converted arrays from previous flush

    def _flush():
      nonlocal rec_pool, pool_count, carry
      recs = np.concatenate(rec_pool)
      rec_pool = []
      pool_count = 0
      out = self._records_to_arrays(recs)
      arrays = list(out[:13])
      v7x = out[13]
      # Flatten v7x components into the array list so shuffle/batch/carry
      # treat them uniformly; reassemble per batch below.
      has_v7x = v7x is not None
      if has_v7x:
        arrays += list(v7x)
      if carry is not None:
        assert len(carry) == len(arrays), 'v6/v7 chunks mixed in one dataset'
        arrays = [np.concatenate([c, a], axis=0) for c, a in zip(carry, arrays)]
      perm = np.random.permutation(arrays[0].shape[0])
      arrays = [a[perm] for a in arrays]
      nfull = arrays[0].shape[0] // B
      for bi in range(nfull):
        sl = slice(bi * B, (bi + 1) * B)
        rows = [a[sl] for a in arrays]
        v7x_b = tuple(rows[13:16]) if has_v7x else None
        yield tuple(rows[:13]) + (None, v7x_b)              # survival=None, v7x
      rem = arrays[0].shape[0] - nfull * B
      carry = [a[nfull * B:] for a in arrays] if rem else None

    epoch = 0
    while True:
      files = list(my_files)
      rng.shuffle(files)
      for fp in files:
        data = self._read_entry(fp)
        if data is None:
          continue
        recs = self._decode_chunk(data)
        if recs is None or len(recs) == 0:
          continue
        if rec_pool and rec_pool[0].dtype != recs.dtype:
          self._skipped_formats += len(recs)   # v6/v7 mixed in one dir: keep pool's version
          continue
        rec_pool.append(np.asarray(recs))
        pool_count += len(recs)
        if pool_count >= _POOL_SIZE:
          yield from _flush()
      epoch += 1
      if self._skipped_formats:
        print(f'[v6_dataset] epoch {epoch}: skipped {self._skipped_formats} non-format-1 records so far')
