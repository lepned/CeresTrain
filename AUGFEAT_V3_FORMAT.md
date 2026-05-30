# V3 TPG Format — Augmented Input Features

**Status**: Implemented + validated end-to-end (Phase 4 EngineBattle eval: OOD +26 Pol Perf at −5% NPS on 384×12 quietmix-10M).
**Repos**:
- Ceres: `augfeat-mvp` branch (5 commits ahead of int8-impl/main)
- CeresTrain: `main` branch (3 augfeat commits)

---

## What V3 adds

3 augmented input feature bytes per square, baked into the TPG file format. These are chess-engine-computed semantic features that the network previously had to derive from raw piece placement:

| Channel | Encoding | Range | Meaning |
|---|---|---|---|
| `our_attackers` | `count * 100 / 8` | byte [0, 100] → float [0, 1] | # of our pieces attacking this square (incl. defenders) |
| `opp_attackers` | `count * 100 / 8` | byte [0, 100] → float [0, 1] | # of opp pieces attacking this square |
| `net_attackers` | `(our - opp + 8) * 100 / 16` | byte [0, 100] → float [0, 1] | shifted-positive net (our-opp), centered at 0.5 |

All three quantized via integer divide so Python (training) and C# (inference) match bit-for-bit through the `byte / 100` pipeline.

---

## Format diff: V2 (137 bytes/sq) → V3 (140 bytes/sq)

```
Per-square TPGSquareRecord layout:
  bytes [0..137)   = unchanged from V2 (piece history, castling, rank/file encoding, etc.)
  bytes [137..140) = NEW V3 augmented feature bytes (our/opp/net attackers)

Per-record TPGRecord total:
  V2: 9378 bytes  (9250 base + 2*64 V2 PlyBin arrays)
  V3: 9570 bytes  (9250 base + 2*64 V2 PlyBin + 3*64 V3 aug bytes)
```

Constants in `Ceres.Chess.NNEvaluators.Ceres.TPG.TPGRecord`:
- `USE_V3_TPG_RECORD = true` (compile-time const)
- `NUM_AUG_FEATURE_BYTES_PER_SQUARE = 3`
- `BYTES_PER_SQUARE_RECORD = 140`
- `TOTAL_BYTES = 9570`

---

## Code that touches V3

### Ceres (chess-engine + inference)
| File | Role |
|---|---|
| `Ceres.Chess/NNEvaluators/Ceres/TPG/TPGRecord.cs` | V3 const flags, byte sizing |
| `Ceres.Chess/NNEvaluators/Ceres/TPG/TPGSquareRecord.cs` | Added `augFeatureBytes[3]` fixed buffer + `AugFeatureBytesSetter`/`ReadOnly` accessors. `WritePosPieces` now bakes aug bytes at record-write time. |
| `Ceres.Chess/Position/PerSquareAttacks.cs` | The bitboard math. Three entry points: `Compute(in MGPosition)` (live position), `ComputeFromBitboards(...)` (already-extracted bitboards), `ComputeFromTpgSquareBytes(...)` (used by V2→V3 upgrade) |
| `Ceres.Chess/NNEvaluators/Ceres/TPG/TPGConvertersToFlat.cs` | Live-inference path; auto-detects V3 140-byte vs V2 137-byte caller buffer + slices when feeding a legacy 137-channel model |
| `Ceres.Chess/NNEvaluators/Base/TensorRT/NNEvaluatorTensorRT.cs` | Buffer-size now derived from ONNX `inputElementsPerPosition` instead of hardcoded 137 |
| `Ceres.Chess/NNBackends/ONNXRuntime/ONNXNetExecutor.cs` | `TPG_BYTES_PER_SQUARE_RECORD` references `TPGRecord.BYTES_PER_SQUARE_RECORD` (auto-tracks V3) |
| `tests/AugFeatSanity/` | Test project: 4-phase validation (layout sanity, starting-pos truth, Python↔C# byte equality on 25 FENs, WritePosPieces orientation correctness) |

### CeresTrain (data pipeline + training)
| File | Role |
|---|---|
| `src/Tasks/TPGConvertV2ToV3.cs` | V2→V3 upgrade tool. `UpgradeFile()` / `UpgradeDirectory()`. Multi-threaded across positions, configurable across files. |
| `src/CeresTrainCommands/CeresTrainCommandLauncher.cs` | CLI command `upgrade-tpg-v2-v3` |
| `src/CeresTrainPy/aug_features.py` | Python reference + vectorized impl. Used as oracle for cross-language validation. Dead-code at training time when V3 corpus is used (features in TPG bytes). |
| `src/CeresTrainPy/tpg_dataset.py` | Loader reads `BYTES_PER_POS=9570`. When `CERES_AUG_FEATURES_PER_SQUARE=0`, slices to 137 (legacy net compat). |
| `src/CeresTrainPy/config.py` | New env-driven constants `NUM_AUG_FEATURES_PER_SQUARE` / `TOTAL_INPUT_FEATURES_PER_SQUARE` |
| `src/CeresTrainPy/ceres_net.py` | Embedding layer width = `TOTAL_INPUT_FEATURES_PER_SQUARE` (137 or 140) |
| `src/CeresTrainPy/save_model.py` | ONNX export dummy input shape uses `TOTAL_INPUT_FEATURES_PER_SQUARE` |
| `scripts/aug_features_dump_for_fen.py` | Python oracle for cross-language equality test |

---

## V2 → V3 conversion: usage

The CLI command reads V2 .zst shards, computes aug bytes from the piece data **already in the records** (no re-labeling, no MCTS search), writes V3 .zst.

```bash
CeresTrain.exe upgrade-tpg-v2-v3 \
    --tpg-dir-in  <V2 source directory> \
    --tpg-dir-out <V3 destination directory> \
    [--zstd-level 5|11|...]  \
    [--max-files-parallel N]
```

### Defaults
- `--zstd-level 5` — fast, +69% size vs V2. Right for one-off testing.
- `--max-files-parallel 4` — balanced for multi-core + SSD.

### For billion-scale upgrades (the other machine):

```bash
CeresTrain.exe upgrade-tpg-v2-v3 \
    --tpg-dir-in  /path/to/V2_corpus \
    --tpg-dir-out /path/to/V3_corpus \
    --zstd-level 11 \
    --max-files-parallel 8
```

- `--zstd-level 11` matches gen-tpg's "Optimal". Storage delta vs V2: **+35%** (vs +69% at lvl 5).
- `--max-files-parallel 8` saturates more cores. Raise to 16+ on big-iron.

### Throughput (measured on this machine, 5090, fast SSD)

| Config | Throughput | Per-billion ETA |
|---|---|---|
| `--zstd-level 5  --max-files-parallel 4` | 199K positions/sec | ~84 min |
| `--zstd-level 11 --max-files-parallel 1` | 25K positions/sec | ~11 hours |
| `--zstd-level 11 --max-files-parallel 8` (extrapolated) | ~150K positions/sec | ~110 min |

### Storage cost for V3 (rough)

| Compression | Cost vs V2 | Notes |
|---|---|---|
| zstd 5 | +69% | Fast; ok for dev |
| zstd 11 | +35% | gen-tpg parity; recommended for production storage |
| zstd 19+ | Marginal further reduction | 2-3× slower; usually not worth it |

The intrinsic **floor** of V3-over-V2 is ~30-35% — the new 3 bytes/sq carry real information (varies per-square per-position), so they can't compress as efficiently as V2's sparse marker bytes. **Even max compression can't go below +30%.**

---

## What's needed on the other machine

Pull both repos with the augfeat commits + rebuild:

```bash
# Ceres (need branch with augfeat-mvp commits, or merge them into main)
cd Ceres
git checkout augfeat-mvp     # or main once merged
dotnet build src/Ceres.Chess/Ceres.Chess.csproj -c Release
dotnet build src/Ceres/Ceres.csproj -c Release

# CeresTrain (main branch)
cd CeresTrain
git pull
dotnet build src/CeresTrain.csproj -c Release
```

### Sanity-check it built correctly

Run the test project (validates TPGRecord size = 140, aug encoding matches python-chess):

```bash
cd Ceres
dotnet run --project tests/AugFeatSanity/AugFeatSanity.csproj -c Release
```

Expected output (last line):
```
ALL TESTS PASSED (Phase 1 + Phase 2 + Phase 3 v3-bake-in)
```

### Run the upgrade

```bash
artifacts/release/net10.0/CeresTrain.exe upgrade-tpg-v2-v3 \
    --tpg-dir-in  /path/to/big_V2_corpus \
    --tpg-dir-out /path/to/V3_corpus \
    --zstd-level 11 \
    --max-files-parallel 8
```

Watch throughput in the per-shard log lines:
```
file.zst: 1,000,000 positions in 5.0s (200,000/sec)
```

### What gets preserved vs computed

| | Preserved from V2 | Computed fresh |
|---|---|---|
| Policy targets (MCTS visits) | ✓ | |
| Value targets (WDL) | ✓ | |
| Move50/EP/castling state | ✓ | |
| Piece placement (history planes) | ✓ | |
| Q-blunder annotations | ✓ | |
| `augFeatureBytes[3]` per square | | derived from piece placement |

**No labels are lost. No MCTS search is rerun.** Pure byte-level transformation.

---

## Currently implemented aug features (V3 ships with these 3)

The 3 ships-with-V3 features are intentionally the minimum-viable set: chosen for unambiguous semantics + cheap compute + high signal in the MVP test (Phase 4: +26 OOD Pol Perf, no regressions on any band, +268 Val Perf on mate puzzles).

| # | Name | Bytes | Semantic |
|---|---|---|---|
| 0 | `our_attackers` | 1 | Count of our pieces with attack-mask covering this square (incl. defenders) |
| 1 | `opp_attackers` | 1 | Count of opp pieces with attack-mask covering this square |
| 2 | `net_attackers` | 1 | Shifted-positive (our - opp + 8) / 16 |

All three derived from chess.Board attackers_mask (Python) / PerSquareAttacks (C#) — bit-identical, validated by `tests/AugFeatSanity`.

---

## Future features (V4 format ideas, not yet implemented)

If/when we want to expand the feature set, the additions go in a hypothetical V4 layout (new bytes appended after the V3 aug bytes — same compile-time-flag pattern). Each upgrade tool gets a similar V3→V4 path.

Ranked by ease + expected signal-to-noise:

| Feature | Bytes/sq | Computed via | Notes |
|---|---|---|---|
| **mobility** | 1 | popcount(piece-on-sq attack mask) | "How many squares can the piece here move to?" Trivial to compute once we have per-piece attack masks. |
| **is-pinned** (1 bit) | 1 | SEE-style pin detection | "Is the piece on this square pinned to its own king?" Unambiguous for absolute pins; tricker for relative pins. |
| **is-hanging** (1 bit) | 1 | `our_attackers < opp_attackers` (already derivable from V3) | Could be V3-derivable; encoding it explicitly might still help learning. |
| **defender-count** | 1 | Same as our_attackers but only counts pieces defending same-color piece on the square | Decomposes V3 channel 0 into "defenders" vs "attacks on empty/opp squares" |
| **x-ray-attackers** | 1 each color | Attack mask through one blocker | Useful for pin/skewer recognition |
| **king-distance** | 1 each color | Chebyshev distance to enemy king | Useful for king safety + endgame king-activity |
| **passed-pawn flag** | 1 (per square) | Pawn structure check | Heavy lift, encodes high-level pawn-structure signal |
| **piece-square value (PSQT)** | 1 | Classical positional eval | Brings in handcrafted-evaluation prior |

### Global features (broadcast to all 64 squares, or 1 extra "global" token)

| Feature | Bytes | Notes |
|---|---|---|
| **material imbalance** | 5 (W-B for P/N/B/R/Q) | Material delta per piece type |
| **phase indicator** | 1 | Opening/middlegame/endgame interpolated |
| **open/semi-open file mask** | 1 per file | Where pawn structure breaks down |
| **king-safety** | 2 (one per king) | Standard pawn-shield + open-files-near-king score |

### Implementation pattern for V4

When adding new aug features:
1. Bump `USE_V3_TPG_RECORD → USE_V4_TPG_RECORD = true`, add `NUM_AUG_FEATURE_BYTES_PER_SQUARE_V4 = N_NEW`
2. Extend `TPGSquareRecord.augFeatureBytes[N]` to the new size (compile-time constant)
3. Add computation in `PerSquareAttacks.Compute*` (or a new helper class for non-attack features)
4. `WritePosPieces` bakes them at record-write time
5. Python training: `aug_features.py` adds the new channels (or skip — they're already in the bytes)
6. Embedding layer width = `137 + 3 + N_NEW`
7. CLI: `upgrade-tpg-v3-v4` command for in-place upgrade of V3 corpora

The V2→V3 pattern is the template — the new converter is ~80 LoC of bitboard math.

---

## Performance characteristics (Phase 4 result on Phase 1-3 implementation)

| Metric | Value |
|---|---|
| OOD avg Pol Perf delta | **+26.3** (acceptance bar +20) ✓ |
| In-dist avg Pol Perf delta | +29.7 |
| Mate-band Pol Perf delta | +71 Pol / +268 Val |
| Universal across bands | ✓ no regressions |
| KLD universally down | ✓ network more calibrated |
| NPS startpos | −5% (cap −10%) ✓ |
| Param delta | +1,152 (of 43M) |
| Training overhead | **0 ms / batch** (V3 baked) |
| Inference overhead | ~64 popcount + ray ops per position |

---

## Open questions / known limitations

1. **`ApplyPlySinceLastMoveTransformationToTPGBuffer`** casts buffer as `Span<TPGSquareRecord>` (140-byte stride with V3). Works fine for V3 + PlySinceLastMoveMode != Zero (most common). The earlier worry about misalignment was V2-only.

2. **Backward compat**: existing 137-channel ONNX models can still be served on V3-formatted TPG data — `TPGConvertersToFlat` auto-slices the trailing 3 bytes per square when caller's output buffer is sized for 137. So you don't have to retrain everything to use V3 inference.

3. **No automatic V3-detection in Python loader yet**. `tpg_dataset.py` hardcodes `BYTES_PER_POS=9570` (V3 only). Reading old V2 files needs the upgrader first. (Adding per-file auto-detect is a 30-min change if needed.)

4. **The Phase 4 result was achieved with a now-fixed orientation bug** (Phase 1C used XOR-56 rank-only flip; should have been 63-i 180-rotation, per `TPGSquareRecord.WritePosPieces` line 363). V3 bake-in does the right thing. Future Phase 4 reruns on V3 should show **stronger** signal.

---

## Memory + git references

- Memory note: `project_augmented_input_features_mvp.md` (Phase 4 result)
- Memory note: `project_aug_features_vectorize_plan.md` (numpy vectorization plan)
- Ceres branch: `augfeat-mvp`, latest: `a9b63e63` (V2→V3 upgrade helper API)
- CeresTrain branch: `main`, latest: `84dfb7c` (CLI zstd-level + max-files-parallel options)
