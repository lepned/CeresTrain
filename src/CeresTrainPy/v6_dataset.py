# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""Direct-from-LC0-chunk training dataset (2026-08-21; review pass same day —
see V6_DATASET_REVIEW.md, all 15 findings addressed here).

Reads LC0 v6/v7 training records — loose .gz game chunks AND members inside
uncompressed .tar archives (in-place via seek+read, zero extra storage) — and
yields batches in exactly the same tuple layout as TPGDataset.item_generator.
Selected via Data config "SourceType": "DirectFromV6".

WHY: storage. The TPG path materializes a second full-size corpus; this path
trains straight from the (pre-rescored) LC0 data already on disk. gen-tpg
conversion measured ~16M pos/h — below training consumption — so C#-side
real-time streaming was rejected in favor of this reader.

INPUT PARITY with the C# TPG writer (review findings 1-5): EP flag on the
opponent pawn's square (32+f) with the full 4-condition detection; Move50 =
min(count,100)/50; missing history FILL_IN by repeating the nearest real
board (TPGRecordConverter cascade); per-legal-move policy floor 0.0005
without kept-mass renormalization; MLH clamped at 255 plies; NaN orig_q ->
0.15 uncertainty fallback; played_q_suboptimality from the PREVIOUS record.

NOT provided by this path (by design): TB rescoring/deblunder (train on
pre-rescored chunks; see the z-integrity filter below for non-deblundered
sets), aux feature channels (requires CERES_AUX_FEATURES_PER_SQUARE=0; the
qz ablation showed aux-neutrality), survival sidecars, PlySinceLastMove and
blunder-counter square bytes (zeros), q-deviation targets (zeros — the
trainer asserts LossQDeviationMultiplier is 0 for this source).

Square encoding (post-divisor floats, one-hot = 1.0), matches TPGSquareRecord:
    [0..103]   8 history positions x 13 piece one-hot (empty, our PNBRQK, their pnbrqk)
    [104..111] per-history repetition flags
    [112..115] CanOO, CanOOO, OpponentCanOO, OpponentCanOOO
    [116]      min(Move50Count,100)/50
    [117]      PlySinceLastMove (zeros)
    [118]      IsEnPassant (opponent double-pushed pawn's square)
    [119,120]  QPositiveBlunders, QNegativeBlunders (zeros)
    [121..128] rank one-hot
    [129..136] file one-hot
"""

import os
import glob
import random
import zlib
import numpy as np

# Profiled: gzip decompression is ~93% of decode cost. python-isal
# (libdeflate) is 2-3x faster; use opportunistically. NOTE (review finding
# 8): isal raises its own error classes (not zlib.error), so the catchable
# error tuple is built alongside the import.
try:
  from isal import igzip as gzip
  from isal import isal_zlib
  _DECOMP_ERRORS = (OSError, EOFError, zlib.error, isal_zlib.error)
  _GZIP_IMPL = 'isal'
except ImportError:
  import gzip
  _DECOMP_ERRORS = (OSError, EOFError, zlib.error)
  _GZIP_IMPL = 'gzip'

from config import NUM_AUX_FEATURES_PER_SQUARE
from tpg_dataset import stable_str_hash, TPGDataset, _RUN_SHUFFLE_SEED

MAX_MOVES = 92               # TPGRecord.MAX_MOVES — top-K policy slots
POLICY_FLOOR = 0.0005        # CompressedPolicyVector.DEFAULT_MIN_PROBABILITY_LEGAL_MOVE
V6_RECORD_BYTES = 8356
V7_RECORD_BYTES = 8396
_MAX_OPEN_TARS = 64          # LRU cap on per-worker tar handles (finding 7)
_FAILFAST_CHUNKS = 200       # all of the first N chunks unusable -> raise (finding 13)

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

# LC0 v7 = v6 + 40-byte tail (EncodedTrainingPositionExtraV7). The censored/
# provenance fields are exactly what the TPG pipeline ships as .v7x sidecars.
V7_DTYPE = np.dtype(V6_DTYPE.descr + [
    ('d_short_term', '<f4'),
    ('opp_played_idx', '<u2'), ('next_played_idx', '<u2'),
    ('z_provenance', '<f4'),
    ('cens_q_st', '<f4'), ('cens_d_st', '<f4'),
    ('q_after_played', '<f4'),
    ('reserved47', '<f4', (4,)),
])
assert V7_DTYPE.itemsize == V7_RECORD_BYTES, V7_DTYPE.itemsize


def _wdl_from_qd(q, d):
  """(q, d) -> [w, d, l] rows, clipped to valid simplex."""
  w = np.clip((1.0 + q - d) * 0.5, 0.0, 1.0)
  l = np.clip((1.0 - q - d) * 0.5, 0.0, 1.0)
  dd = np.clip(1.0 - w - l, 0.0, 1.0)
  return np.stack([w, dd, l], axis=1).astype(np.float32)


class V6ChunkDataset(TPGDataset):
  """LC0 v6/v7 .gz chunks and tar archives as a TPGDataset.

  Subclasses TPGDataset to inherit __getitem__ (batch -> torch-dict incl.
  policy scatter and boards_per_batch splitting), __len__ and set_worker_id;
  overrides __init__ and item_generator. The generator attribute is created
  in the parent process but its body first executes inside the DataLoader
  worker (after fork + set_worker_id) — the same lazy-sharding pattern as
  TPGDataset. NOTE (review, sub-threshold): boards_per_batch > 1 relies on
  record adjacency that the pool shuffle breaks — assert against it."""

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
               skip_count: int = None,
               shuffle_pool: int = None,
               max_resultq_delta: float = None,
               **_ignored):
    assert NUM_AUX_FEATURES_PER_SQUARE == 0, \
        'DirectFromV6 requires CERES_AUX_FEATURES_PER_SQUARE=0 (137-channel model)'
    if wdl_smoothing:
      raise NotImplementedError('WDLLabelSmoothing not supported by DirectFromV6 (set 0)')
    if file_mirror_prob:
      raise NotImplementedError('FileMirrorAug not supported by DirectFromV6 yet (set 0)')
    if boards_per_batch != 1:
      raise NotImplementedError('DirectFromV6 requires BoardsPerBatch=1 (pool shuffle '
                                'breaks the record adjacency multi-board batches assume)')
    # Loudly reject knobs the TPG path honors but this path ignores (finding 15).
    if float(os.environ.get('CERES_POLICY_TARGET_ALPHA', '0') or 0) > 0:
      raise NotImplementedError('CERES_POLICY_TARGET_ALPHA is not implemented in DirectFromV6')
    if float(os.environ.get('CERES_KEEP_DRAW_PROB', '1') or 1) < 1.0:
      raise NotImplementedError('CERES_KEEP_DRAW_PROB (decisive oversampling) is not '
                                'implemented in DirectFromV6')
    self.root_dir = root_dir
    self.batch_size = batch_size
    self.rank = rank
    self.world_size = world_size
    self.num_workers = max(1, num_workers)
    self.worker_id = 0
    self.boards_per_batch = boards_per_batch
    # Knobs: constructor (from Data config via train.py) beats env, env beats default.
    def _knob(explicit, env, default, cast):
      if explicit is not None:
        return cast(explicit)
      v = os.environ.get(env)
      return cast(v) if v not in (None, '') else default
    self.skip_count = _knob(skip_count, 'CERES_V6_SKIP_COUNT', 30, int)
    self.pool_size = _knob(shuffle_pool, 'CERES_V6_SHUFFLE_POOL', 50000, int)
    self.max_resultq_delta = _knob(max_resultq_delta, 'CERES_V6_MAX_RESULTQ_DELTA', 0.0, float)
    # NOTE: num_files_to_skip counts CHUNKS (games, ~100 pos) here, not TPG
    # shards (millions of pos) — semantically different from the TPG path.
    entries = [('fs', p) for p in
               sorted(glob.glob(os.path.join(root_dir, '**', '*.gz'), recursive=True))]
    for tp in sorted(glob.glob(os.path.join(root_dir, '**', '*.tar'), recursive=True)):
      entries += self._index_tar(tp)
    if num_files_to_skip:
      entries = entries[num_files_to_skip:]
    if not entries:
      raise FileNotFoundError(f'DirectFromV6: no .gz chunks or .tar archives under {root_dir}')
    # Rank-consistent shuffle (finding 6): all ranks must produce the SAME
    # order for the strided rank/worker partition to be a partition. Reuse
    # the run-level seed the TPG path uses. Its default is time-based and
    # only single-process-safe, so DDP without an explicit seed is refused.
    if world_size > 1 and not os.environ.get('CERES_SHUFFLE_SEED'):
      raise RuntimeError('DirectFromV6 under DDP requires CERES_SHUFFLE_SEED '
                         '(rank-divergent shuffles silently skip/duplicate corpus slices)')
    random.Random(_RUN_SHUFFLE_SEED).shuffle(entries)
    self.files = entries
    self._tar_handles = {}           # tar_path -> fh, LRU-bounded (finding 7)
    self._version = None             # pinned record version (finding 9)
    self._skipped_other_version = 0
    self._skipped_formats = 0
    self._read_errors = 0
    self._zfiltered = 0
    if self.max_resultq_delta > 0:
      print(f'[v6_dataset] z-integrity filter ON: drop |best_q - result_q| > {self.max_resultq_delta}')
    self._startup_diagnosis(entries)
    _n_tar = sum(1 for e in entries if e[0] == 'tar')
    print(f'[v6_dataset] {root_dir}: {len(entries):,} chunks ({_n_tar:,} in-tar), '
          f'skip_count={self.skip_count}, pool={self.pool_size:,}, '
          f'rank {rank}/{world_size}, gzip={_GZIP_IMPL}')
    self.generator = self.item_generator()

  # ---- startup diagnosis ------------------------------------------------

  def _startup_diagnosis(self, entries, sample_n=100):
    """Sample chunks and report what this corpus IS: record version(s) and
    rescore/deblunder status. Nothing in v6 marks a set as (not-)deblundered,
    but deblundering writes CONTINUOUS z values — any non-integer result_q
    proves it; v7 provenance codes 2/3 prove it directly (1/4 = rescored).
    Every run self-reports its data provenance instead of relying on
    folklore about a directory."""
    step = max(1, len(entries) // sample_n)
    versions, provs, rq_nonint, rq_n = set(), set(), 0, 0
    for e in entries[::step][:sample_n]:
      data = self._read_entry(e)
      if data is None or len(data) < 8:
        continue
      ver = int(np.frombuffer(data[:4], dtype='<u4')[0])
      if ver == 6:
        recs = np.frombuffer(data, dtype=V6_DTYPE, count=len(data) // V6_RECORD_BYTES)
      elif ver == 7:
        recs = np.frombuffer(data, dtype=V7_DTYPE, count=len(data) // V7_RECORD_BYTES)
      else:
        versions.add(ver)
        continue
      versions.add(ver)
      rq = np.nan_to_num(recs['result_q'])
      rq_nonint += int((np.abs(rq - np.round(rq)) > 1e-6).sum())
      rq_n += len(rq)
      if ver == 7:
        provs.update(np.unique(np.nan_to_num(recs['z_provenance']).astype(np.uint8)).tolist())
    # Close parent-process tar handles so forked workers do not inherit
    # shared file offsets (each worker reopens lazily).
    for fh in self._tar_handles.values():
      fh.close()
    self._tar_handles = {}
    self._diag_versions = versions            # consumed by train.py sidecar preflights
    deblundered = (rq_nonint > 0) or bool(provs & {2, 3})
    rescored = bool(provs & {1, 4})
    parts = [f'versions={sorted(versions)}']
    parts.append('deblundered=YES' if deblundered else
                 'deblundered=NO (z-integrity filter recommended: V6MaxResultQDelta)')
    parts.append(('rescored=YES' if rescored else 'rescored=unknown/no') if 7 in versions
                 else 'rescored=undetectable (v6)')
    if provs:
      parts.append(f'provenance={sorted(provs)}')
    print(f'[v6_dataset] corpus diagnosis ({rq_n:,} pos sampled): ' + ', '.join(parts))

  # ---- tar support ------------------------------------------------------

  @staticmethod
  def _index_tar(tar_path):
    """Member table for an uncompressed LC0 tar: [('tar', path, offset, size), ...].
    Cached as <tar>.chunkindex.npz; written atomically (tmp + os.replace,
    finding 11) so concurrent ranks can never observe a truncated cache, and
    a corrupt cache is rebuilt instead of crashing."""
    import tarfile
    cache = tar_path + '.chunkindex.npz'
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(tar_path):
      try:
        z = np.load(cache)
        return [('tar', tar_path, int(o), int(s)) for o, s in zip(z['offsets'], z['sizes'])]
      except Exception:
        pass                                      # corrupt/truncated cache: rebuild below
    offsets, sizes = [], []
    with tarfile.open(tar_path, 'r:') as tf:      # 'r:' = uncompressed container only
      for m in tf:
        if m.isfile() and m.name.endswith('.gz'):
          offsets.append(m.offset_data)
          sizes.append(m.size)
    try:
      tmp = cache + f'.tmp.{os.getpid()}'
      np.savez_compressed(tmp, offsets=np.asarray(offsets, dtype=np.int64),
                          sizes=np.asarray(sizes, dtype=np.int64))
      os.replace(tmp + '.npz' if os.path.exists(tmp + '.npz') else tmp, cache)
    except OSError:
      pass                                        # read-only dir: index stays in-memory
    print(f'[v6_dataset] indexed {os.path.basename(tar_path)}: {len(offsets):,} chunk members')
    return [('tar', tar_path, o, s) for o, s in zip(offsets, sizes)]

  def _read_entry(self, entry):
    """Entry -> decompressed chunk bytes (None on read error, counted)."""
    try:
      if entry[0] == 'fs':
        with gzip.open(entry[1], 'rb') as f:
          return f.read()
      _, tar_path, offset, size = entry
      fh = self._tar_handles.get(tar_path)
      if fh is None:
        if len(self._tar_handles) >= _MAX_OPEN_TARS:      # LRU cap (finding 7)
          old_path, old_fh = next(iter(self._tar_handles.items()))
          old_fh.close()
          del self._tar_handles[old_path]
        fh = open(tar_path, 'rb', buffering=0)
      else:
        del self._tar_handles[tar_path]                    # re-insert = mark recently used
      self._tar_handles[tar_path] = fh
      fh.seek(offset)
      return gzip.decompress(fh.read(size))
    except _DECOMP_ERRORS:
      self._read_errors += 1
      if self._read_errors in (1, 10, 100) or self._read_errors % 10000 == 0:
        print(f'[v6_dataset] WARNING: {self._read_errors} unreadable chunks so far '
              f'(latest: {entry[1] if entry[0] == "fs" else entry[1]})')
      return None

  # ---- decoding ---------------------------------------------------------

  def _decode_chunk(self, data: bytes):
    """Bytes of one chunk -> structured record array (game order preserved;
    downsampled; played_q_suboptimality precomputed game-wise into the
    otherwise-unused 'played_m' field — finding 5: C# takes it from the
    PREVIOUS record, so it must be computed before sampling)."""
    if data is None or len(data) < 8:
      return None
    ver = int(np.frombuffer(data[:4], dtype='<u4')[0])
    if ver not in (6, 7):
      return None
    if self._version is None:
      self._version = ver                          # pin corpus version (finding 9)
    elif ver != self._version:
      self._skipped_other_version += 1
      return None
    rec_bytes, dtype = ((V6_RECORD_BYTES, V6_DTYPE) if ver == 6
                        else (V7_RECORD_BYTES, V7_DTYPE))
    n = len(data) // rec_bytes
    if n == 0:
      return None
    recs = np.frombuffer(data, dtype=dtype, count=n)
    # played_q_suboptimality of the PREVIOUS record (game order), before any
    # filtering/sampling. NaN-guarded; first record of the game gets 0.
    bq = np.nan_to_num(recs['best_q'])
    pq = np.nan_to_num(recs['played_q'])
    pqs = np.zeros(n, dtype=np.float32)
    if n > 1:
      pqs[1:] = np.maximum(bq[:-1] - pq[:-1], 0.0)
    recs = recs.copy()                             # frombuffer view is read-only
    recs['played_m'] = pqs                         # repurposed (see docstring)
    # z-integrity filter (deblunder-lite; ~neutral on deblundered data).
    if self.max_resultq_delta > 0:
      zok = np.abs(bq - np.nan_to_num(recs['result_q'])) <= self.max_resultq_delta
      self._zfiltered += int((~zok).sum())
      recs = recs[zok]
      if len(recs) == 0:
        return None
    # Downsample 1/skip_count (decorrelation, random per pass).
    if self.skip_count > 1:
      keep = np.random.random(len(recs)) < (1.0 / self.skip_count)
      if not keep.any():
        return None
      recs = recs[keep]
    # Only classic input format 1 supported.
    fmt_ok = recs['input_format'] == 1
    if not fmt_ok.all():
      self._skipped_formats += int((~fmt_ok).sum())
      recs = recs[fmt_ok]
      if len(recs) == 0:
        return None
    return recs

  def _records_to_arrays(self, recs):
    """Structured records -> per-position target/input arrays (numpy)."""
    n = len(recs)

    # ---- planes -> squares[137] ----
    # LC0 u64 planes, our perspective: byte b of the little-endian u64 is
    # rank b; within each byte the MSB is file a (startpos-anchor verified).
    plane_bytes = recs['planes'].view(np.uint8).reshape(n, 104, 8)
    bits = np.unpackbits(plane_bytes, axis=2, bitorder='big')           # [n, 104, 64]
    planes = bits.astype(np.float32).transpose(0, 2, 1)                 # [n, 64, 104]

    squares = np.zeros((n, 64, 137), dtype=np.float32)
    # History FILL_IN (finding 3): C# repeats the nearest real board for
    # missing history. LC0 zero-pads instead, so an all-zero piece block
    # means "missing" — cascade-copy from the previous (newer) position,
    # exactly TPGRecordConverter's h_k = fill(h_{k-1}) chain. Repetition
    # flags of filled positions stay 0 (a copied board is not a repetition).
    hist = planes.reshape(n, 64, 8, 13)                                 # [n, 64, hist, 12+rep]
    piece_blocks = [hist[:, :, h, :12] for h in range(8)]
    for h in range(1, 8):
      missing = piece_blocks[h].sum(axis=(1, 2)) == 0                   # [n]
      if missing.any():
        piece_blocks[h] = np.where(missing[:, None, None],
                                   piece_blocks[h - 1], piece_blocks[h])
    for h in range(8):
      pc = piece_blocks[h]
      occ = pc.sum(axis=2)
      squares[:, :, h * 13 + 0] = 1.0 - np.minimum(occ, 1.0)            # empty flag
      squares[:, :, h * 13 + 1: h * 13 + 13] = pc                       # our 1-6, their 7-12
      squares[:, :, 104 + h] = hist[:, :, h, 12]                        # repetition flag
    # castling: CanOO, CanOOO, OpponentCanOO, OpponentCanOOO
    squares[:, :, 112] = (recs['us_oo'] > 0)[:, None]
    squares[:, :, 113] = (recs['us_ooo'] > 0)[:, None]
    squares[:, :, 114] = (recs['them_oo'] > 0)[:, None]
    squares[:, :, 115] = (recs['them_ooo'] > 0)[:, None]
    # Move50 (finding 2): C# = min(count,100)/50, range [0, 2].
    squares[:, :, 116] = (np.minimum(recs['rule50'].astype(np.float32), 100.0) / 50.0)[:, None]
    # [117] PlySinceLastMove: zeros (matches our own TPG gen).
    # [118] IsEnPassant (finding 1): flag on the OPPONENT PAWN'S square
    # (32+f), full 4-condition detection between history boards h0/h1
    # (EncodedPositionBoards.EnPassantOpportunityBetweenBoards): pawn was on
    # 48+f, not on 32+f; now gone from 48+f, present on 32+f.
    their_p_h0 = planes[:, :, 6]                                        # their pawns, current
    their_p_h1 = planes[:, :, 13 + 6]                                   # their pawns, 1 ply ago
    for f in range(8):
      dbl = ((their_p_h0[:, 32 + f] > 0) & (their_p_h1[:, 48 + f] > 0)
             & (their_p_h1[:, 32 + f] == 0) & (their_p_h0[:, 48 + f] == 0))
      squares[dbl, 32 + f, 118] = 1.0
    # [119,120] blunder counters: zeros.
    ranks = np.arange(64) // 8
    files_ = np.arange(64) % 8
    squares[:, np.arange(64), 121 + ranks] = 1.0
    squares[:, np.arange(64), 129 + files_] = 1.0

    # ---- policy (finding 4): per-legal-move floor 0.0005 (soft legality),
    # full-vector renormalize, then top-92 WITHOUT kept-mass renormalization
    # (matches CompressedPolicyVector; the tail mass stays missing exactly
    # as in TPG shards).
    raw = np.nan_to_num(recs['probs'], nan=-1.0)
    legal = raw >= 0
    probs = np.where(legal, np.maximum(raw, POLICY_FLOOR), 0.0).astype(np.float32)
    tot = probs.sum(axis=1, keepdims=True)
    tot = np.where(tot > 0, tot, 1.0)
    probs = probs / tot
    k = MAX_MOVES
    part = np.argpartition(-probs, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(probs, part, axis=1)
    order = np.argsort(-vals, axis=1)
    policies_indices = np.take_along_axis(part, order, axis=1).astype(np.int16)
    policies_values = np.take_along_axis(vals, order, axis=1).astype(np.float16)

    # ---- value / aux targets (finding 5/10: NaN-guard every field used) ----
    bq = np.nan_to_num(recs['best_q'])
    bd = np.nan_to_num(recs['best_d'])
    rq = np.nan_to_num(recs['result_q'])
    rd = np.nan_to_num(recs['result_d'])
    wdl_q = _wdl_from_qd(bq, bd)
    wdl_result = _wdl_from_qd(rq, rd)
    wdl_deblundered = wdl_result
    wdl_nondeblundered = wdl_result
    played_q_subopt = np.nan_to_num(recs['played_m']).astype(np.float32).reshape(-1, 1)
    unc_policy = np.abs(np.nan_to_num(np.frombuffer(recs['reserved'].tobytes(), dtype='<f4')
                                      .reshape(-1, 2)[:, 0])).astype(np.float32).reshape(-1, 1)
    # DeltaQVersusV: |best_q - orig_q|; C# fills 0.15 when orig_q is NaN
    # (cache miss) rather than treating the eval as perfect.
    orig_q = recs['orig_q']
    unc = np.abs(bq - np.nan_to_num(orig_q))
    unc = np.where(np.isnan(orig_q), 0.15, unc).astype(np.float32).reshape(-1, 1)
    # MLH: clamp at 255 plies (TPG encoding range [0, 2.55]).
    mlh = (np.minimum(np.nan_to_num(recs['plies_left']), 255.0)
           .astype(np.float32) / 100.0).reshape(-1, 1)
    zeros16 = np.zeros((n, 1), dtype=np.float16)
    pip = np.full((n, 1), -1, dtype=np.int16)

    v7x = None
    if 'cens_q_st' in recs.dtype.names:
      v7x = (np.nan_to_num(recs['cens_q_st']).astype(np.float32).reshape(-1, 1),
             np.nan_to_num(recs['cens_d_st']).astype(np.float32).reshape(-1, 1),
             np.nan_to_num(recs['z_provenance']).astype(np.uint8).reshape(-1, 1))

    return (policies_indices, policies_values, wdl_deblundered, wdl_q, mlh,
            unc, wdl_nondeblundered, zeros16, zeros16.copy(), squares,
            pip, played_q_subopt, unc_policy, v7x)

  # ---- generation -------------------------------------------------------

  def item_generator(self):
    B = self.batch_size
    shard = self.rank * self.num_workers + self.worker_id
    num_shards = self.world_size * self.num_workers
    my_files = self.files[shard::num_shards]
    if not my_files:
      # Refuse silent whole-corpus duplication (review, sub-threshold): the
      # TPG path raises here too.
      raise RuntimeError(f'DirectFromV6: {len(self.files)} chunks < {num_shards} '
                         f'rank*worker shards — reduce workers or add data')
    # Per-worker rng: run seed + shard so resumes reshuffle differently per
    # run but identically across ranks' partitions.
    rng = random.Random(stable_str_hash(f'{self.root_dir}|{shard}|{_RUN_SHUFFLE_SEED}') & 0x7fffffff)

    # Pool RAW records; shuffle structured ROWS; convert in blocks (finding
    # 14: converting the whole pool at once spikes ~70 KB/record transient).
    # Carry is RECORDS, not arrays, so version handling stays trivial.
    CONVERT_BLOCK = 8192
    rec_pool = []
    pool_count = 0
    processed = 0

    def _flush(final=False):
      nonlocal rec_pool, pool_count
      recs = np.concatenate(rec_pool)
      perm = np.random.permutation(len(recs))
      recs = recs[perm]
      nbatch = (len(recs) // B) * B
      leftover = recs[nbatch:]
      recs = recs[:nbatch]
      for i0 in range(0, len(recs), CONVERT_BLOCK):
        block = recs[i0:i0 + CONVERT_BLOCK]
        out = self._records_to_arrays(block)
        v7x = out[13]
        for b0 in range(0, len(block), B):
          sl = slice(b0, b0 + B)
          v7x_b = tuple(a[sl] for a in v7x) if v7x is not None else None
          yield tuple(a[sl] for a in out[:13]) + (None, v7x_b)
      rec_pool = [leftover] if len(leftover) and not final else []
      pool_count = len(leftover) if not final else 0

    epoch = 0
    while True:
      files = list(my_files)
      rng.shuffle(files)
      chunks_seen = 0
      for fp in files:
        recs = self._decode_chunk(self._read_entry(fp))
        chunks_seen += 1
        # Fail fast on a wrong-corpus mistake (finding 13) instead of idling
        # the GPUs for a full silent pass.
        if processed == 0 and chunks_seen == _FAILFAST_CHUNKS and not rec_pool:
          raise RuntimeError(
              f'DirectFromV6: first {_FAILFAST_CHUNKS} chunks yielded no usable records '
              f'(read errors {self._read_errors}, other-version {self._skipped_other_version}, '
              f'non-format-1 {self._skipped_formats}) — wrong corpus?')
        if recs is None or len(recs) == 0:
          continue
        rec_pool.append(recs)
        pool_count += len(recs)
        if pool_count >= self.pool_size:
          processed += pool_count
          yield from _flush()
      epoch += 1
      diags = []
      if self._skipped_other_version:
        diags.append(f'other-version chunks {self._skipped_other_version}')
      if self._skipped_formats:
        diags.append(f'non-format-1 records {self._skipped_formats}')
      if self._read_errors:
        diags.append(f'read errors {self._read_errors}')
      if self._zfiltered:
        diags.append(f'z-filtered {self._zfiltered}')
      if diags:
        print(f'[v6_dataset] epoch {epoch} (worker {self.worker_id}): ' + ', '.join(diags))

  def __del__(self):
    for fh in getattr(self, '_tar_handles', {}).values():
      try:
        fh.close()
      except Exception:
        pass
