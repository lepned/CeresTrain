# Code review: c37804e DirectFromV6 (v6_dataset.py) — 2026-08-21

Full adversarial review (max effort): 10 finder angles, C# ground truth from the
Ceres fork the build actually references, empirical decoding of real T80 tars
AND a production TPG shard (`D:\t91_skip1_v2_surv`), crash reproductions that
were actually run, and a 3-batch end-to-end smoke on real data.

**VERDICT: not ready for the 4xA100 run yet.** The core is byte-exact; 15
findings must be addressed, several of which are triggered by EXACTLY the
intended production setup (t91_v7_op1, 1896 tars, 4 ranks, isal installed, no
explicit seed).

> **Status 2026-08-22: all 15 findings are fixed** (commit `0ae5ed0`, with a
> follow-up review closed in `1e77d15`). This document is retained as the
> record of what was found and why. See `ACTION_HEAD_REVIEW.md` for the
> follow-up round.

## Verified CLEAN

- V6/V7 record dtypes byte-exact against `EncodedPositionEvalMiscInfoV6` /
  `ExtraV7` (struct-cited, itemsize-asserted)
- Plane bit order (`unpackbits` big-endian, MSB = the a-file) empirically
  correct; square layout (8×13 blocks, repetition at 104-111, castling order,
  rank/file one-hots at 121-136), newest-first history, STM orientation with
  rank flip — all match both the C# and a real shard
- `wdl_q` / `wdl_nondeblundered` algebra, the KLDPolicy source, the MLH source
  and its `0.1*sqrt` decoding, the v7x tail
- 15-slot yield parity with `TPGDataset`; the TPG path is byte-identically
  untouched
- Truncation and gzip-CRC handling complete FOR the stdlib gzip fallback

## FINDINGS (ranked)

### Silent input-parity divergences (the worst class)

1. **EP flag on the WRONG SQUARE**: Python uses 40+f (the capture square), C#
   uses 32+f (the opponent pawn's square, `TPGSquareRecord.cs:351-361`;
   empirically confirmed in a production shard). Python also lacks C#'s 4th
   condition (`!TheirPawns(48+f)`, `EncodedPositionBoards.cs:137-154`) → 6.1 %
   false flags measured on real T80 data. (`v6_dataset.py:302`)
2. **Move50 at half scale and unclamped**: Python uses `rule50/100`; C# uses
   `min(count,100)/50` (`TPGRecordEncoding.cs:42-47`). Empirically: the TPG
   bytes in channel 116 are all even. (`:294`)
3. **History fill diverges**: LC0 zero-pads missing history; Python encodes an
   "empty board"; C# with `FILL_IN=true` repeats the nearest real board
   (`TPGRecordConverter.cs:536`; 0 of 50 production records have an all-zero
   h7). Affects every position under 8 plies. (`:286`)
4. **The policy floor is missing**: C# gives EVERY legal move a 0.0005 floor
   (soft legality, `CompressedPolicyVector.cs:68`) and renormalises
   conditionally; Python takes the top 92 with exact zeros and renormalises
   unconditionally → a different MCTS prior from every TPG-trained net.
   (`:319`)
5. **Aux targets**: MLH not clamped at 255 plies (5.9 % of real data exceeds
   it → targets up to 4.5 against TPG's [0, 2.55]); `unc` uses `abs(best_q)`
   when `orig_q` is NaN where C# fills 0.15; `played_q_suboptimality` uses
   record *i* where C# uses *i−1*; qdev zeros with no guard against
   `LossQDeviationMultiplier > 0`. (`:335`)

### Operational killers for the production setup

6. **Rank-divergent shuffle**: `random.Random(None)` per rank plus strided
   sharding → without `CERES_SHUFFLE_SEED`, ~32 % of the corpus is NEVER read
   and ~26 % is read twice on 4 ranks (simulated), silently. The
   `_default_shuffle_seed` + all_reduce check from `tpg_dataset` was dropped —
   reuse `_RUN_SHUFFLE_SEED`. A fixed seed alone revives resume bias (the
   per-worker RNG lacks a run component). (`:183`/`:358`)
7. **fd exhaustion**: `_tar_handles` opens 1 fd per tar and never closes them;
   each worker's stride spans ~all 1896 tars → past `ulimit 1024` early in
   epoch 1; after that the `open()` OSError is swallowed into `return None`
   WITHOUT logging → ~46 % of the corpus silently gone for the whole run.
   Needs an LRU cap + an error counter (and possibly `RLIMIT_NOFILE`). (`:227`)
8. **isal errors slip through**: the net catches `(OSError, EOFError,
   zlib.error)`, but python-isal (the recommended fast path!) raises
   `IsalError` (only an `Exception` subclass) on corrupt deflate — probed live.
   One corrupt member kills the entire DDP run. Catch `isal_zlib.error` when
   the import succeeded. (`:231`)
9. **Mixed v6+v7 crashes** (REPRODUCED: `assert len(carry)==len(arrays)`, 13 vs
   16, right after a flush); in the lottery case the yield tuples flip silently
   between v7x and non-v7x. (`:384`/`:408`)
10. **NaN propagation**: `best_q/d`, `result_q/d`, `played_q`, `plies_left` are
    used raw (only `probs`/`orig_q` are `nan_to_num`'d); one NaN record → NaN
    loss → all weights NaN. C# has explicit NaN fallbacks for these fields.
    (`:324`)
11. **Index-cache race**: all ranks write the same `.chunkindex.npz` without
    tmp + `os.replace`; a truncated cache → an unrecognised
    `zipfile.BadZipFile` → a crash on every later start until someone deletes
    the file. Plus 4× redundant cold-start work. (`:211`/`:202`)
12. **Sidecar preflight conflict**: the survival/stvalue/prov preflights list
    `.v7x.zst` files that a tar corpus never has → honest configs DIE at
    startup; the `'=1'` workaround makes the survival head train with NO
    supervision at all, silently. (`train.py:1293-1322`)
13. **Version skip without a counter**: v3/v4/v5 records are dropped with no
    diagnosis; the final report only comes after a COMPLETE pass (never at
    1.45B) → the wrong corpus means GPUs idle for hours without a line of
    output. Fail fast after N fully-skipped chunks. (`:246`/`:261`)
14. **RAM budget ~8-10× too low**: the comment's "50k × 8 workers = 3.3 GB"
    counts only raw records; `_flush` materialises ~70 KB/record transiently
    (~4 GB/worker spike, synchronised across workers at the first flush).
    (`:116`)
15. **Silently ignored knobs**: `KeepDrawProb` / `CERES_KEEP_DRAW_PROB`,
    `POLICY_TARGET_ALPHA` and `FILE_MIRROR_AUG` are ignored by the v6 path
    WHILE the `tpg_dataset` banners still claim they are on; the recipe
    documents `V6SkipCount` / `V6ShufflePool` in the DATA config but the
    bootstrap only bridges the OPT config (the example values equalling the
    defaults masks the no-op). (`:153`)

### Below the top-15 cut (verified, worth mentioning)

- The C# `DataSourceType` enum lacks `DirectFromV6` → all C# tooling throws
  `JsonException` on the config
- `boards_per_batch > 1` / action configs: the pool permutation destroys record
  adjacency; `policy_index_in_parent = -1` silently indexes slot 1857
- `NumTPGFilesToSkip` now counts GAMES (~5 positions) rather than shards
  (~millions)
- The starvation fallback lets starved workers read the WHOLE corpus
  (duplication) where `TPGDataset` raises
- `recover_export`'s `CERES_HOST_PREFIX` default of `lepdev` versus `train.py`'s
  `gethostname()` → two naming schemes, and a `FileNotFoundError` on the server
- Per-pass re-enumeration of directories was dropped (tars added mid-run are
  ignored)

## Recommended fix order

1. Parity: EP square + condition, Move50 scale, history FILL_IN, policy floor
   (findings 1-4)
2. DDP seed: reuse `_RUN_SHUFFLE_SEED` + the all_reduce check (finding 6)
3. Robustness: isal catch, fd LRU, NaN guard, atomic cache write
   (findings 8, 7, 10, 11)
4. The rest, plus validation: a parity A/B (the same positions through both the
   TPG path and the v6 path, comparing tensors bit for bit) is the GOLD
   STANDARD before a server run.
