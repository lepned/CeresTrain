# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

# NOTE: this code derived from: https://github.com/Rocketknight1/minimal_lczero.

import os, fnmatch
import numpy as np
import zstandard

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

# stable hash function for strings so all worker processes use same function
def stable_str_hash(s: str) -> int:
    hash_value = 0
    for char in s:
        hash_value = (hash_value * 31 + ord(char)) % (59275)
    return hash_value


# Method to enhance shuffling of TPG files.
# Try to avoid having files from same TPG set appearing more than once within blocks of 8.
# Make repeated passes with random perturbations to achieve this to a large or complete degree.
#
# Seed selection: per-restart randomness eliminates the bias where a kill-mid-file
# always re-reads the same beginning-of-file records on every resume. Each launch
# of train.py picks a fresh seed (from time_ns), so the file-iteration order shifts
# between runs. Set CERES_SHUFFLE_SEED to override (e.g. for reproducibility tests).
import time as _time
_RUN_SHUFFLE_SEED = int(os.environ.get('CERES_SHUFFLE_SEED', str(_time.time_ns() & 0xFFFFFFFF)))
print(f'[try_shuffle] run-level shuffle seed = {_RUN_SHUFFLE_SEED} (override with CERES_SHUFFLE_SEED)')

def try_shuffle(file_list):
    import random
    BLOCK_SIZE = 8  # Assume max 8 GPUs (TPG reading workers) per node
    SEED = _RUN_SHUFFLE_SEED  # randomized per training launch (was fixed 42)
    NUM_PASSES = 50
    rand_gen = random.Random(SEED)

    for _ in range(NUM_PASSES):
        index = 0
        while index < len(file_list):
            block = file_list[index:index + BLOCK_SIZE]
            prefixes = {}
            duplicates = []
            for i, filename in enumerate(block):
                prefix = filename.split('.tpg')[0]
                if prefix in prefixes:
                    duplicates.append(index + i)
                else:
                    prefixes[prefix] = True
            # Move one of the duplicates to a deterministic "random" position
            for dup_index in sorted(duplicates, reverse=True):
                new_pos = rand_gen.randint(0, len(file_list) - 1)
                file_list.insert(new_pos, file_list.pop(dup_index))
            index += BLOCK_SIZE
    return file_list


MAX_MOVES = 92 # Maximum number of policy moves in a position that can be stored (TPGRecord.MAX_MOVES)

# SINGLE SOURCE OF TRUTH: import the aux-feature count from config rather than
# re-reading the env here. This guarantees the data width (how many aux channels
# we keep per square) always matches the model width (config.py's embedding sizing).
# Defaults to 4 (full V3); override with CERES_AUX_FEATURES_PER_SQUARE. Resolved at
# module import so spawned worker processes inherit the same value.
from config import NUM_AUX_FEATURES_PER_SQUARE as _NUM_AUX_FEATURES_PER_SQUARE

# Optional: pick specific aux-channel INDICES (1-based-into-aux-slice) instead of
# the default "first N channels". Comma-separated; count must match
# CERES_AUX_FEATURES_PER_SQUARE. Example: CERES_AUX_CHANNEL_INDICES=7 with
# CERES_AUX_FEATURES_PER_SQUARE=1 → use only aux[7] (SEE-only ablation).
_AUX_INDICES_ENV = os.environ.get('CERES_AUX_CHANNEL_INDICES', '').strip()
_AUX_CHANNEL_INDICES = None
if _AUX_INDICES_ENV:
  _AUX_CHANNEL_INDICES = [int(x.strip()) for x in _AUX_INDICES_ENV.split(',') if x.strip()]
  if len(_AUX_CHANNEL_INDICES) != _NUM_AUX_FEATURES_PER_SQUARE:
    raise ValueError(f'CERES_AUX_CHANNEL_INDICES count ({len(_AUX_CHANNEL_INDICES)}) must equal '
                     f'CERES_AUX_FEATURES_PER_SQUARE ({_NUM_AUX_FEATURES_PER_SQUARE})')

if _NUM_AUX_FEATURES_PER_SQUARE > 0:
  if _AUX_CHANNEL_INDICES is not None:
    print(f'[tpg_dataset] AUX_FEATURES enabled: {_NUM_AUX_FEATURES_PER_SQUARE} channels '
          f'at INDICES {_AUX_CHANNEL_INDICES} (non-default selection)')
  else:
    print(f'[tpg_dataset] AUX_FEATURES enabled: +{_NUM_AUX_FEATURES_PER_SQUARE} baked V3 aux channels per square (read directly from the TPG record, no recompute)')

# TPG FILE format: bytes per square IN THE SHARD FILES (not the model width).
#   141 (default) = V3 shards (137 base + 4 baked aux bytes per square, 9634 B/pos)
#   137           = upstream V2 shards (no aux bytes, 9378 B/pos)
# For 137, CERES_AUX_FEATURES_PER_SQUARE must be 0 (the file carries no aux to serve).
# Default honors CERES_TPG_V3=0 (whole-corpus V2 toggle from upstream) when the
# per-dataset env is not set explicitly.
_TPG_SQUARE_BYTES = int(os.environ.get('CERES_TPG_SQUARE_BYTES',
                                       '137' if os.environ.get('CERES_TPG_V3', '1') == '0' else '141'))
if _TPG_SQUARE_BYTES not in (137, 141):
  raise ValueError(f'CERES_TPG_SQUARE_BYTES must be 137 (V2 shards) or 141 (V3 shards), got {_TPG_SQUARE_BYTES}')
if _TPG_SQUARE_BYTES == 137:
  if _NUM_AUX_FEATURES_PER_SQUARE != 0:
    raise ValueError('CERES_TPG_SQUARE_BYTES=137 (V2 shards, no aux bytes) requires CERES_AUX_FEATURES_PER_SQUARE=0, '
                     f'got {_NUM_AUX_FEATURES_PER_SQUARE}')
  print('[tpg_dataset] V2 TPG shard format: 137 bytes/square (9378 bytes/pos), no aux channels')

# K-ply survival target sidecars (SURVIVAL_TARGET_SPEC.md): companion '<shard minus .zst>.tgt.zst'
# files with a 16-byte header (TPGT|ver|channels|K|reserved) followed by [numRecords, 64] uint8
# fate labels in the SAME record order. Exposed as batch['survival'] for the survival aux head.
# Modes (CERES_TPG_TARGET_SIDECAR):
#   0 / unset : off — sidecars ignored entirely (legacy)
#   1         : required — EVERY shard must have a sidecar (hard error otherwise)
#   auto      : per-shard — shards with sidecars supply survival targets, shards without
#               simply yield batches with no 'survival' key (the survival loss skips them).
#               Enables mixing a huge sidecar-less primary with survival-labeled secondaries.
_TPG_TARGET_SIDECAR_ENV = (os.environ.get('CERES_TPG_TARGET_SIDECAR', '0') or '0').strip().lower()
if _TPG_TARGET_SIDECAR_ENV in ('0', ''):
  _TPG_TARGET_SIDECAR_MODE = 'off'
elif _TPG_TARGET_SIDECAR_ENV == '1':
  _TPG_TARGET_SIDECAR_MODE = 'required'
elif _TPG_TARGET_SIDECAR_ENV == 'auto':
  _TPG_TARGET_SIDECAR_MODE = 'auto'
else:
  raise ValueError(f"CERES_TPG_TARGET_SIDECAR must be 0, 1 or 'auto', got {_TPG_TARGET_SIDECAR_ENV!r}")
_SURVIVAL_HORIZON = int(os.environ.get('CERES_SURVIVAL_HORIZON', '8') or 8)
if _TPG_TARGET_SIDECAR_MODE != 'off':
  print(f'[tpg_dataset] survival target sidecars ENABLED, mode={_TPG_TARGET_SIDECAR_MODE} (expect K={_SURVIVAL_HORIZON} in headers)')
# Public alias (callers must not import the underscore name): the process-wide default
# mode; per-dataset override via TPGDataset(sidecar_mode=...).
TPG_TARGET_SIDECAR_MODE = _TPG_TARGET_SIDECAR_MODE

# Optional policy-target sharpening: target = alpha * one_hot(solver) + (1-alpha) * teacher.
# Set CERES_POLICY_TARGET_ALPHA > 0 (e.g. 0.5) to enable. Default 0 = no sharpening.
_POLICY_TARGET_ALPHA = float(os.environ.get('CERES_POLICY_TARGET_ALPHA', '0.0'))

# On-disk TPG record format: V3 (141 B/square, 9634 B/pos, 4 aux baked in) vs upstream
# V2 (137 B/square, 9378 B/pos, no aux). Mirrors Ceres.Chess TPGRecord.USE_V3_TPG_RECORD.
# Default '1' (V3) preserves prior behavior; set CERES_TPG_V3=0 to read a V2 corpus.
# NB: a V2 corpus has no aux on disk, so it requires CERES_AUX_FEATURES_PER_SQUARE=0.
_USE_V3_TPG = os.environ.get('CERES_TPG_V3', '1') != '0'
_SIZE_SQUARE_ONDISK = 141 if _USE_V3_TPG else 137
_BYTES_PER_POS_ONDISK = 9634 if _USE_V3_TPG else 9378
if not _USE_V3_TPG and _NUM_AUX_FEATURES_PER_SQUARE != 0:
  raise ValueError('CERES_TPG_V3=0 (V2 corpus) requires CERES_AUX_FEATURES_PER_SQUARE=0 '
                   f'(got {_NUM_AUX_FEATURES_PER_SQUARE}) — V2 has no aux bytes on disk.')
if _POLICY_TARGET_ALPHA > 0:
  print(f"[tpg_dataset] policy-target sharpening: alpha = {_POLICY_TARGET_ALPHA}")

# Decisive-position oversampling: keep all win/loss records but keep DRAW records only with
# probability CERES_KEEP_DRAW_PROB (default 1.0 = off). Counters draw-saturation of strong
# self-play data (e.g. 0.40 lifts decisive share ~28% -> ~50%). Draw class = argmax(wdl_q)==1.
_KEEP_DRAW_PROB = float(os.environ.get('CERES_KEEP_DRAW_PROB', '1.0'))
if _KEEP_DRAW_PROB < 1.0:
  print(f"[tpg_dataset] decisive oversampling: keep_draw_prob = {_KEEP_DRAW_PROB}")


class TPGDataset(Dataset):
  """
  TPGDataset is a subclass of the PyTorch Dataset for efficiently reading and parsing raw binary
  TPGRecords files (compressed in ZST format) containing training data for chess positions.
  It's optimized for high throughput (typically about 50,000 positions per second per worker) 
  and takes care to partition data without overlap across the possibly multiple workers in a distributed setup. 

  The class uses numpy for initial parsing of values (using ascontiguousarray, view, reshape) 
  and converts them into PyTorch tensors on the CPU. It's designed to work with a specific 
  binary format of chess training data (TPG).

  Parameters:
      root_dir (str): Directory containing TPGRecords files with ZST extension.
      batch_size (int): Size of the batch to be read and processed.
      wdl_smoothing (float): Smoothing parameter for win/draw/loss data.
      rank (int): The rank of the process in distributed training.
      world_size (int): Total number of processes in distributed training.
      num_workers (int): Number of worker processes for data loading.
      boards_per_batch (int): Number of boards per batch.
      num_files_to_skip: Optional number of TPG files to be skipped by this worker (to avoids reprocessing files already processed)
      test (bool): If the Exec test flag is enabled.
  """
  def __init__(self, root_dir,
               batch_size: int,
               wdl_smoothing : bool,
               rank : int,
               world_size : int,
               num_workers : int,
               boards_per_batch : int,
               num_files_to_skip : int = 0,
               test : bool = False,
               square_bytes : int = None,
               sidecar_mode : str = None):

    self.root_dir = root_dir
    self.batch_size = batch_size
    self.wdl_smoothing = wdl_smoothing
    self.num_workers = num_workers
    # Per-dataset shard record format (137 = upstream V2, 141 = V3 with baked aux).
    # Default = the process-wide CERES_TPG_SQUARE_BYTES; override per dataset to mix
    # e.g. a V3 primary with a V2 secondary. A 137-byte dataset carries no aux bytes,
    # so any aux-channel model input requires all datasets to be 141.
    self.square_bytes = int(square_bytes) if square_bytes is not None else _TPG_SQUARE_BYTES
    if self.square_bytes not in (137, 141):
      raise ValueError(f'square_bytes must be 137 (V2) or 141 (V3), got {self.square_bytes}')
    # Per-dataset survival-sidecar mode (default = process-wide CERES_TPG_TARGET_SIDECAR).
    # Lets a combined recipe run the survival-labeled primary in 'required' (fail-loud on
    # any missing sidecar) while a sidecar-less secondary (e.g. puzzle TPG) runs 'auto' —
    # process-wide 'required' used to FileNotFoundError on the secondary's first batch.
    self.sidecar_mode = sidecar_mode if sidecar_mode is not None else _TPG_TARGET_SIDECAR_MODE
    if self.sidecar_mode not in ('off', 'required', 'auto'):
      raise ValueError(f"sidecar_mode must be 'off', 'required' or 'auto', got {self.sidecar_mode!r}")
    if self.sidecar_mode != _TPG_TARGET_SIDECAR_MODE:
      print(f'[tpg_dataset] {root_dir}: survival sidecar mode override: '
            f'{_TPG_TARGET_SIDECAR_MODE} -> {self.sidecar_mode}', flush=True)
    if self.square_bytes == 137 and _NUM_AUX_FEATURES_PER_SQUARE != 0:
      raise ValueError(f'dataset {root_dir}: 137-byte (V2) shards carry no aux bytes; '
                       f'CERES_AUX_FEATURES_PER_SQUARE must be 0 (got {_NUM_AUX_FEATURES_PER_SQUARE})')
    self.generator = self.item_generator()
    self.boards_per_batch = boards_per_batch
    self.test = test

    # State retained for re-enumeration (so files added to root_dir during a
    # long training run get picked up automatically — see item_generator).
    self.rank = rank
    self.world_size = world_size
    self.num_files_to_skip = num_files_to_skip

    # Get initial list of files and select the rank-subset for this worker.
    self.files = self._discover_files(initial=True)

    self.worker_id = None
    self._last_seen_count = len(self.files)

    print('Creating TPGDataset at', root_dir, ' found', len(self.files), 'files matching this worker', rank, 'of', world_size, 'to be split among', num_workers, 'workers.')


  def _discover_files(self, initial=False):
    """Enumerate the directory, sort/shuffle/skip per config, then return the
    rank-partitioned subset for this rank. (Worker-id filtering happens
    separately in item_generator.) Called both from __init__ and at the start
    of each pass through the data so that new .zst files dropped into root_dir
    during a long run get included automatically."""
    all_files = fnmatch.filter(os.listdir(self.root_dir), '*.zst')
    # Survival target sidecars (<shard>.tgt.zst) are companion label files, never shards.
    all_files = [f for f in all_files if not f.endswith('.tgt.zst')]
    all_files.sort(key=lambda f: stable_str_hash(f))  # deterministic shuffle
    if initial:
      assert len(all_files) >= self.num_files_to_skip + self.num_workers, f"Trying to skip more files than available: {len(all_files)} available, {self.num_files_to_skip} to skip, {self.num_workers} workers"
    all_files = all_files[self.num_files_to_skip:]
    all_files = try_shuffle(all_files)
    files_per_worker = len(all_files) // self.world_size
    start_index = self.rank * files_per_worker
    end_index = start_index + files_per_worker
    return all_files[start_index:end_index]


  def set_worker_id(self, worker_id):
    self.worker_id = worker_id


  def __len__(self):
        # There is no actual limit (we repeat if necessary), so we just return a large number.
        # N.B. under some circumstances PyTorch will construct a data structure of this length, 
        #      so we return a number large enough to be more than any reasonable training session, but not excessive.
        return 10_000_000 # probably large enough (e.g. 10 million batches of size 1024 ==> 20 billion positions)


  def item_generator(self):
    DTYPE = np.float32
    BATCH_SIZE = self.batch_size
    # Fixed size of TPGRecord (V3 format with USE_V2 + USE_V3 both true, post-2026-06-01 cleanup):
    # 9250 (original) + 2*64 (V2 PlyBin arrays) + 4*64 (V3 aux feature bytes) = 9634
    # The 4 aux channels per square are:
    #   [0] mobility            — pseudo-legal move count of piece on square
    #   [1] defender_count      — same-color attackers of piece on square
    #   [2] is_pinned           — boolean (0/100), pinned to own king by opp slider
    #   [3] is_threatened       — boolean (0/100), attacked by strictly-lower-value opp piece
    # Must match Ceres TPGRecord.TOTAL_BYTES (9634 for V3, 9378 for V2).
    # Per-dataset (self.square_bytes) so a V3 primary can mix with V2 secondaries in one run.
    BYTES_PER_POS = 9378 + (self.square_bytes - 137) * 64
    POS_PER_BLOCK = 24576//2 # read this many positions per loop iteration (somewhat arbitrary, each block about 115MB)
    BYTES_PER_BLOCK = POS_PER_BLOCK * BYTES_PER_POS

    wdl_smoothing_transform = np.array([
        [1-self.wdl_smoothing, self.wdl_smoothing*0.75, self.wdl_smoothing*0.25],
        [self.wdl_smoothing*0.5, 1-self.wdl_smoothing, self.wdl_smoothing*0.5],
        [self.wdl_smoothing*0.25, self.wdl_smoothing*0.75, 1-self.wdl_smoothing]])

    while True:
      # Re-enumerate the directory at the start of each pass so new .zst files
      # dropped into root_dir during long training runs get picked up
      # automatically (no restart needed). The original implementation cached
      # the file list at __init__ time only.
      rank_files = self._discover_files()
      my_files = [file for index, file in enumerate(rank_files)
                  if self.num_workers == 0 or (index % self.num_workers == self.worker_id)]
      if len(my_files) != self._last_seen_count:
        print(f'DATASET WORKER {self.worker_id} re-enumerated {self.root_dir}: '
              f'now has {len(my_files)} files (was {self._last_seen_count})')
        self._last_seen_count = len(my_files)

      # Sidecar coverage visibility: in 'auto' a missing/misnamed sidecar dir trains
      # silently with a starved survival head, so report the ratio once per pass.
      if self.sidecar_mode != 'off' and my_files:
        _n_sidecars = sum(1 for f in my_files
                          if os.path.exists(os.path.join(self.root_dir, f[:-4] + '.tgt.zst')))
        print(f'[tpg_dataset] worker {self.worker_id} {self.root_dir}: '
              f'{_n_sidecars}/{len(my_files)} shards carry survival sidecars '
              f'(mode={self.sidecar_mode})', flush=True)

      def _read_exact(reader, n):
        """Read exactly n bytes from a zstd stream_reader (short reads happen at
        frame boundaries and are NOT EOF). Raises on premature end of stream."""
        parts = []
        remaining = n
        while remaining > 0:
          piece = reader.read(remaining)
          if not piece:
            raise RuntimeError(f'survival sidecar ended prematurely ({remaining} of {n} bytes missing)')
          parts.append(piece)
          remaining -= len(piece)
        return b''.join(parts) if len(parts) != 1 else parts[0]

      for file_name in my_files:
        print()
        print('DATASET WORKER', self.worker_id, 'PROCESSING TPG FILE', file_name)

        surv_file = None
        surv_reader = None
        if self.sidecar_mode != 'off':
          surv_path = os.path.join(self.root_dir, file_name[:-4] + '.tgt.zst')
          if not os.path.exists(surv_path):
            if self.sidecar_mode == 'required':
              raise FileNotFoundError(f'survival sidecar mode=required but sidecar missing: {surv_path}')
            # mode=auto: shard has no sidecar -> its batches carry no survival targets.
          else:
            surv_file = open(surv_path, 'rb')
            surv_reader = zstandard.ZstdDecompressor().stream_reader(surv_file)
            hdr = _read_exact(surv_reader, 16)
            if hdr[:4] != b'TPGT' or hdr[4] != 1 or hdr[5] != 1:
              raise ValueError(f'bad survival sidecar header in {surv_path}: {hdr[:8].hex()}')
            surv_sidecar_K = hdr[6]
            if surv_sidecar_K < _SURVIVAL_HORIZON:
              # Upward is impossible without regen: "survives beyond sidecar-K" cannot be
              # split into captured-later vs survives-longer.
              raise ValueError(f'survival sidecar K={surv_sidecar_K} < CERES_SURVIVAL_HORIZON={_SURVIVAL_HORIZON} '
                               f'in {surv_path}: shrinking K is lossless, growing K requires gen-tpg regen')
            # surv_sidecar_K > _SURVIVAL_HORIZON is allowed: labels are remapped losslessly
            # below (captured at ply > K == survived the K-ply horizon). Enables K sweeps
            # on one K=8 corpus without regeneration.

        with open(os.path.join(self.root_dir, file_name),'rb') as file:
          dctx = zstandard.ZstdDecompressor()
          stream_reader = dctx.stream_reader(file)

          leftover = b''
          while True:
            chunk = stream_reader.read(BYTES_PER_BLOCK)
            if not chunk:
              break  # true end of stream; any leftover is < one batch and is dropped (as before)
            # NOTE: a short read does NOT mean end of stream — zstandard's stream_reader
            # stops at zstd FRAME boundaries (read_across_frames defaults to False), so a
            # multi-frame shard returns short chunks mid-stream. Carry the sub-batch
            # remainder across reads so record alignment is preserved and no data is
            # silently dropped mid-stream. This also makes shards smaller than one block
            # (small corpora) and BATCH_SIZE values that do not divide POS_PER_BLOCK work.
            decompressed_data = leftover + chunk if leftover else chunk
            usable_bytes = (len(decompressed_data) // (BATCH_SIZE * BYTES_PER_POS)) * (BATCH_SIZE * BYTES_PER_POS)
            leftover = decompressed_data[usable_bytes:]
            if usable_bytes == 0:
              continue

            dd = np.frombuffer(decompressed_data, dtype=np.uint8, count=usable_bytes)  # zero-copy prefix view
            batches = dd.reshape(-1, BATCH_SIZE, BYTES_PER_POS)

            # Survival sidecar rows for exactly the records consumed this iteration,
            # read in lockstep so record order stays aligned with the main stream.
            surv_batches = None
            if surv_reader is not None:
              num_records_this_block = usable_bytes // BYTES_PER_POS
              surv_bytes = _read_exact(surv_reader, num_records_this_block * 64)
              surv_batches = np.frombuffer(surv_bytes, dtype=np.uint8).reshape(-1, BATCH_SIZE, 64)
              if surv_sidecar_K > _SURVIVAL_HORIZON:
                # Lossless downward remap to the configured horizon K': captured at ply
                # d <= K' keeps its label; captured later or survives-beyond-sidecar-K
                # both mean "survived the K'-ply horizon" = class K'+1. Empty (0) unchanged.
                surv_batches = surv_batches.copy()
                surv_batches[surv_batches > _SURVIVAL_HORIZON] = _SURVIVAL_HORIZON + 1

            for batch_num in range(batches.shape[0]):
              this_batch = batches[batch_num,:,:]
              survival = surv_batches[batch_num] if surv_batches is not None else None
              
              offset = 0 # running offset of where we are within the record

              if (self.wdl_smoothing == 0.5):
                assert 1==2, "wdl_smoothing == 0.5 not supported"

              # Layout matches the V2 TPGRecord struct in TPGRecord.cs
              # (LayoutKind.Sequential, Pack=1, USE_V2_TPG_RECORD=true).
              # Total = 9378 bytes per record.

              wdl_nondeblundered = np.ascontiguousarray(this_batch[:, offset : offset + 3*4]).view(dtype=np.float32).reshape(-1, 3)
              offset+= 3 * 4
              if (self.wdl_smoothing > 0):
                wdl_nondeblundered = np.matmul(wdl_nondeblundered, wdl_smoothing_transform)

              wdl_deblundered = np.ascontiguousarray(this_batch[:, offset : offset + 3*4]).view(dtype=np.float32).reshape(-1, 3)
              offset+= 3 * 4
              if (self.wdl_smoothing > 0):
                wdl_deblundered = np.matmul(wdl_deblundered, wdl_smoothing_transform)

              wdl_q = np.ascontiguousarray(this_batch[:, offset : offset + 3*4]).view(dtype=np.float32).reshape(-1, 3)
              offset+= 3 * 4
              if (self.wdl_smoothing > 0):
                wdl_q = np.matmul(wdl_q, wdl_smoothing_transform)

              played_q_suboptimality = np.ascontiguousarray(this_batch[:, offset : offset + 1*4]).view(dtype=np.float32).reshape(-1, 1)
              offset+= 1 * 4

              # IsWhiteToMove (1 byte) + Unused1 (1 byte) + PUNIMSelf (1 byte) + PUNIMOpponent (1 byte)
              # + UnusedArray[42] (42 bytes). Skipped — not currently consumed by trainer.
              offset+= 4 + 42

              # NumSearchNodes (int32), RefModel1NumNodes/Value (2x float16), RefModel1BestMove (ushort).
              # Skipped — reference-model fields not consumed by trainer.
              offset+= 4 + 2 + 2 + 2

              # KLDPolicy (float32): KL divergence between policy head and search visits.
              # Reused as the "uncertainty_policy" value for back-compat with trainer consumers.
              uncertainty_policy = np.ascontiguousarray(this_batch[:, offset : offset + 1*4]).view(dtype=np.float32).reshape(-1, 1)
              uncertainty_policy = np.abs(uncertainty_policy)
              offset+= 1 * 4

              mlh = np.ascontiguousarray(this_batch[:, offset : offset + 1*4]).view(dtype=np.float32).reshape(-1, 1)
              mlh = np.square(mlh / 0.1) # undo preprocessing
              mlh = mlh / 100.
              offset+= 1 * 4

              # DeltaQVersusV (float32): serves as the "uncertainty" value for back-compat.
              uncertainty = np.ascontiguousarray(this_batch[:, offset : offset + 1*4]).view(dtype=np.float32).reshape(-1, 1)
              uncertainty = np.abs(uncertainty)
              offset+= 1 * 4

              q_deviation_lower = np.ascontiguousarray(this_batch[:, offset : offset + 1*2]).view(dtype=np.float16).reshape(-1, 1)
              offset+= 1 * 2
              q_deviation_upper = np.ascontiguousarray(this_batch[:, offset : offset + 1*2]).view(dtype=np.float16).reshape(-1, 1)
              offset+= 1 * 2

              policy_index_in_parent = np.ascontiguousarray(this_batch[:, offset : offset + 1*2]).view(dtype=np.int16).reshape(-1, 1)
              offset+= 1 * 2

              # Two PlyBinPerSquare64 arrays (each 64 bytes): PlyUntilSquareChangePiece + PlyUntilSquarePieceCapture.
              # Skipped — currently unused by trainer.
              offset+= 64 + 64

              policies_indices = np.ascontiguousarray(this_batch[:, offset : offset + MAX_MOVES*2]).view(dtype=np.int16).reshape(-1, MAX_MOVES)
              # much faster, but tries to reinitialize CUDA and fails:
              #   policies = torch.from_numpy(np.ascontiguousarray(this_batch[:, offset : offset + 1858*2]).view(dtype=np.float16)).cuda().reshape(-1,1858)
              offset+= MAX_MOVES * 2

              policies_values = np.ascontiguousarray(this_batch[:, offset : offset + MAX_MOVES*2]).view(dtype=np.float16).reshape(-1, MAX_MOVES)
              offset+= MAX_MOVES * 2

              # Optional policy-target sharpening (loss-alignment fix). When
              # CERES_POLICY_TARGET_ALPHA > 0, mix in a one-hot on the solver
              # move (the slot with the highest probability — guaranteed to be
              # the Lichess-prescribed solver move per the rank-1 nudge applied
              # at TPG generation):
              #     target = alpha * one_hot(argmax) + (1 - alpha) * teacher
              # alpha=0.5 puts half the policy mass directly on solver while
              # keeping teacher's nuance among alternatives. Skipped for rows
              # whose teacher distribution is all zero (OppDef value-only records).
              if _POLICY_TARGET_ALPHA > 0.0:
                row_max = policies_values.max(axis=1)        # [B]
                active = row_max > 0                          # mask out value-only rows
                if active.any():
                  pv32 = policies_values.astype(np.float32)
                  solver_slot = pv32.argmax(axis=1)           # [B]
                  pv32[active] = pv32[active] * (1.0 - _POLICY_TARGET_ALPHA)
                  rows_active = np.where(active)[0]
                  pv32[rows_active, solver_slot[active]] += _POLICY_TARGET_ALPHA
                  # Belt-and-suspenders normalization: scrub any FP16 drift in
                  # the input distribution that survives the sharpen step.
                  row_sum = pv32[active].sum(axis=1, keepdims=True)
                  row_sum = np.where(row_sum > 0, row_sum, 1.0)
                  pv32[active] = pv32[active] / row_sum
                  policies_values = pv32.astype(np.float16)

              # Square records: 141 bytes/sq for V3 shards (137 base + 4 baked aux),
              # 137 for upstream V2 shards (per-dataset self.square_bytes). For
              # 137-channel models reading V3 shards, the aux tail is sliced below.
              SIZE_SQUARE = self.square_bytes
              squares = np.ascontiguousarray(this_batch[:, offset : offset + 64 * SIZE_SQUARE * 1]).view(dtype=np.byte).reshape(-1, 64, SIZE_SQUARE).astype(DTYPE)
              DIVISOR = 100
              squares = np.divide(squares, DIVISOR).astype(DTYPE)
              offset+= 64 * SIZE_SQUARE

              assert offset == BYTES_PER_POS, f"Layout mismatch: offset={offset} expected={BYTES_PER_POS}"

              # Drop unused trailing aux channels (V3 shards only; V2 shards have none).
              # V3 carries 4 aux bytes; trainers using CERES_AUX_FEATURES_PER_SQUARE < 4
              # slice the tail.
              #   0 = legacy 137-channel model (no aux)
              #   4 = full V3 (mobility / defender / is_pinned / is_threatened)
              # CERES_AUX_CHANNEL_INDICES can override the default first-N selection
              # (cherry-pick specific channels for ablation; indices are into the 4-channel
              # aux slice, mapping to absolute positions 137..140).
              if SIZE_SQUARE > 137:
                if _AUX_CHANNEL_INDICES is not None:
                  idx_abs = [137 + i for i in _AUX_CHANNEL_INDICES]
                  squares = np.concatenate([squares[:, :, :137], squares[:, :, idx_abs]], axis=2)
                else:
                  keep_channels = 137 + _NUM_AUX_FEATURES_PER_SQUARE
                  if keep_channels < SIZE_SQUARE:
                    squares = squares[:, :, :keep_channels]

              # Decisive oversampling: drop a fraction of DRAW records (argmax(wdl_q)==1),
              # keep all win/loss. Variable-size batch downstream is fine (pos counted as kept).
              if _KEEP_DRAW_PROB < 1.0:
                _is_draw = (wdl_q.argmax(axis=1) == 1)
                _keep = (~_is_draw) | (np.random.random(_is_draw.shape[0]) < _KEEP_DRAW_PROB)
                if not _keep.all():
                  policies_indices = policies_indices[_keep]; policies_values = policies_values[_keep]
                  wdl_deblundered = wdl_deblundered[_keep]; wdl_q = wdl_q[_keep]; mlh = mlh[_keep]
                  uncertainty = uncertainty[_keep]; wdl_nondeblundered = wdl_nondeblundered[_keep]
                  q_deviation_lower = q_deviation_lower[_keep]; q_deviation_upper = q_deviation_upper[_keep]
                  squares = squares[_keep]; policy_index_in_parent = policy_index_in_parent[_keep]
                  played_q_suboptimality = played_q_suboptimality[_keep]; uncertainty_policy = uncertainty_policy[_keep]
                  if survival is not None:
                    survival = survival[_keep]

              yield  ((policies_indices, policies_values, wdl_deblundered, wdl_q, mlh, uncertainty,
                       wdl_nondeblundered, q_deviation_lower, q_deviation_upper, squares,policy_index_in_parent, played_q_suboptimality,
                       uncertainty_policy, survival))

        if surv_file is not None:
          surv_reader.close()
          surv_file.close()


  def __getitem__(self, idx):
    batch = next(self.generator)
    policies_indices = batch[0]
    policies_values = batch[1]
    wdl_deblundered = batch[2]
    wdl_q = batch[3]
    mlh = batch[4]
    uncertainty = batch[5]
    wdl_nondeblundered = batch[6]
    q_deviation_lower = batch[7]
    q_deviation_upper = batch[8]
    squares = batch[9]
    policy_index_in_parent = batch[10]
    played_q_suboptimality = batch[11]
    uncertainty_policy = batch[12]
    survival = batch[13] if len(batch) > 13 else None
    
    _nb = policies_indices.shape[0]   # actual row count (may be < batch_size after decisive draw-filtering)
    policies_indices = torch.tensor(policies_indices, dtype=torch.int64).reshape(_nb, MAX_MOVES)
    policies_values  = torch.tensor(policies_values, dtype=torch.float16).reshape(_nb, MAX_MOVES)

    # TO DO: do this on GPU?
    policies = torch.zeros(_nb, 1858, dtype=torch.float16)
    policies.scatter_(1, policies_indices, policies_values)

   
    def create_filtered_dict(mod_value):
      # Function to filter tensor elements with indices modulo boards_per_batch equal to mod_value
      def filter_tensor(tensor, mod_value):
          indices = torch.arange(len(tensor))
          filtered_indices = indices[indices % self.boards_per_batch == mod_value]
          return tensor[filtered_indices]

      # Creating the new dictionary with filtered tensors
      filtered_dict = {
          'policies': filter_tensor(policies, mod_value),
          'wdl_deblundered': filter_tensor(torch.tensor(wdl_deblundered), mod_value),
          'wdl_q': filter_tensor(torch.tensor(wdl_q), mod_value),
          'mlh': filter_tensor(torch.tensor(mlh), mod_value),
          'unc': filter_tensor(torch.tensor(uncertainty), mod_value),
          'wdl_nondeblundered': filter_tensor(torch.tensor(wdl_nondeblundered), mod_value),
          'q_deviation_lower': filter_tensor(torch.tensor(q_deviation_lower), mod_value).to(torch.float32),
          'q_deviation_upper': filter_tensor(torch.tensor(q_deviation_upper), mod_value).to(torch.float32),
          'squares': filter_tensor(torch.tensor(squares), mod_value),
          'policy_index_in_parent': filter_tensor(torch.tensor(policy_index_in_parent), mod_value),
          'played_q_suboptimality': filter_tensor(torch.tensor(played_q_suboptimality), mod_value),
          'uncertainty_policy': filter_tensor(torch.tensor(uncertainty_policy), mod_value)
      }
      if survival is not None:
        filtered_dict['survival'] = filter_tensor(torch.tensor(np.ascontiguousarray(survival), dtype=torch.uint8), mod_value)
      return filtered_dict
    
    return [create_filtered_dict(i) for i in range(self.boards_per_batch)]


class TPGMixedDataset(Dataset):
  """Mixes batches from two TPG datasets at a configurable ratio.

  Yields `ratio_1_to_2` batches from the primary dataset, then 1 batch from the
  secondary, repeating. If `secondary` is None or `ratio_1_to_2 <= 0`, yields only
  from `primary` (matches single-source legacy behaviour exactly).

  Use case: train on T80 self-play (primary, ~95% of batches) plus puzzle data
  (secondary, ~5%) at ratio 19:1 — the trainer sees one puzzle batch per 20 steps.

  Note: each DataLoader worker gets its own counter (worker process state isn't
  shared). At the dataset level the ratio is honoured per-worker; aggregate ratio
  across all workers is the same.
  """
  def __init__(self, primary, secondary=None, ratio_1_to_2: int = 0):
    self.primary = primary
    if secondary is not None and ratio_1_to_2 > 0:
      self.secondary = secondary
      self.ratio = int(ratio_1_to_2)
    else:
      self.secondary = None
      self.ratio = 0
    self.counter = 0

  def __len__(self):
    return self.primary.__len__()

  def set_worker_id(self, worker_id):
    self.primary.set_worker_id(worker_id)
    if self.secondary is not None:
      self.secondary.set_worker_id(worker_id)

  def __getitem__(self, idx):
    if self.secondary is None:
      return self.primary[idx]
    # Cycle of length (ratio+1): positions 0..ratio-1 are primary; position ratio is secondary.
    cycle_pos = self.counter % (self.ratio + 1)
    self.counter += 1
    if cycle_pos == self.ratio:
      item = self.secondary[idx]
      # Tag so the trainer can apply per-stream loss multipliers (see train.py
      # CERES_SECONDARY_LOSS_*_MULT). Plain bool survives DataLoader passthrough
      # (batch_size=None) and _move_batch_to_device (non-tensor leaves returned as-is).
      for board_dict in item:
        board_dict['is_secondary'] = True
      return item
    return self.primary[idx]


def worker_init_fn(worker_id):
  """
    Initialize a worker function for a data loader.

    This function sets a global variable `WORKER_ID` to the ID of the worker.
    This method will be called in a multi-process data loading scenarios,
    allowing us to record the identifier if this worker for later coordination use.

    Args:
        worker_id (int): An integer identifier for the worker process.
    """
  global WORKER_ID
  WORKER_ID = worker_id




if __name__ == "__main__": 
  import time
  import sys

  print('Beginning performance test of tpg_dataset.py.')
  TPG_TRAIN_DIR = "/mnt/i/tpg_16man" #"./test_data"
  devices = [0]
  BATCH_SIZE = 1024 * 4
  

  def worker_init_fn(worker_id):
    dataset.set_worker_id(worker_id)

  # Use two concurrent dataset workers (if more than one training data file is available)
  count_zst_files = len(fnmatch.filter(os.listdir(TPG_TRAIN_DIR), '*.zst'))
  NUM_DATASET_WORKERS = 8 if not sys.platform.startswith("win") else 0 # Not available on Windows. 1 meansone parallel worker always processing in advance (change with caution).
  PREFETCH_FACTOR = None if NUM_DATASET_WORKERS == 0 else 2 # to keep GPU busy
 
  world_size = len(devices)
  rank = 0 if world_size == 1 else dist.get_rank()
  dataset = TPGDataset(TPG_TRAIN_DIR, BATCH_SIZE // world_size, False, 
                       rank, world_size, NUM_DATASET_WORKERS, 
                       1, 0, False)
  dataloader = DataLoader(dataset, batch_size=None, pin_memory=False, num_workers=NUM_DATASET_WORKERS, worker_init_fn=worker_init_fn, prefetch_factor=PREFETCH_FACTOR)


  BATCH_COUNT_PER_INTERVAL = 100
  start = time.time_ns()
  i = 0
  for batch_idx, (batch) in enumerate(dataloader):
    if i % BATCH_COUNT_PER_INTERVAL == BATCH_COUNT_PER_INTERVAL - 1:
      end = time.time_ns()
      time_sec = (end-start)*0.001*0.001*0.001
      print (i, ' ', (BATCH_SIZE * BATCH_COUNT_PER_INTERVAL) / time_sec, '/sec')
      start = time.time_ns()
    i+=1

