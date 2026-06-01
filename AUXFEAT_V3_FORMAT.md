# V3 TPG Format — Auxiliary Input Features

**Status**: Finalized 2026-06-01 after ablation-driven cleanup. The format previously included 7-8 aux channels (V3-MVP attackers, V3-extended tactical features, SEE) and was reduced to **the 4 channels that demonstrably earned their place via tournament + puzzle ablation**.

---

## What V3 adds

4 auxiliary input feature bytes per square, baked into the TPG file format. These are tactical-motif features that the network can't compute cheaply in 1-2 attention layers:

| Channel | Encoding | Range | Meaning |
|---|---|---|---|
| `mobility`        | `raw * 100 / 27` (capped at 100) | byte [0, 100] → float [0, 1] | Pseudo-legal move count of piece on this square (captures-only for pawns) |
| `defender_count`  | `count * 100 / 8`                | byte [0, 100] → float [0, 1] | Same-color attackers of this square (defending the piece on it) |
| `is_pinned`       | 0 or 100                         | byte {0, 100} → float {0, 1} | Boolean: piece pinned to own king by an opp slider |
| `is_threatened`   | 0 or 100                         | byte {0, 100} → float {0, 1} | Boolean: piece attacked by opp piece of *strictly lower* value (NNUE-spirit). Pawn never threatened; King threatened if attacked at all. |

All four are quantized via integer divide so Python (training oracle) and C# (TPG generation + inference) match bit-for-bit through the `byte / 100` pipeline.

**Why these four:** ablation tested 0/3/7/8-aux feature sets at 256×10 + smolgen + SwiGLU + post-norm + Muon-LR8e4 + decay@0.7 + 10M positions. Findings:
- 3 attacker counts (V3-MVP): tournament-tied vs 0-aux (dead weight — model derives internally)
- 4 tactical features (mobility/defender/pinned/threatened): **+39 Elo tournament @ n=100 (CFS 99.6%)**, **+14 OOD puzzle Pol Perf** vs 0-aux
- SEE (8th channel): bit-exact validated but tournament-negative under AdamW; puzzle-tied under Muon. Redundant with the 4 above.

---

## Format diff: V2 (137 bytes/sq) → V3 (141 bytes/sq)

```
Per-square TPGSquareRecord layout:
  bytes [0..137)   = unchanged from V2 (piece history, castling, rank/file encoding, etc.)
  bytes [137..141) = V3 aux feature bytes:
                       [137] mobility
                       [138] defender_count
                       [139] is_pinned
                       [140] is_threatened

Per-record TPGRecord total:
  V2: 9378 bytes  (9250 base + 2*64 V2 PlyBin arrays)
  V3: 9634 bytes  (9250 base + 2*64 V2 PlyBin + 4*64 V3 aux bytes)
```

Constants in `Ceres.Chess.NNEvaluators.Ceres.TPG.TPGRecord`:
- `USE_V3_TPG_RECORD = true` (compile-time const)
- `NUM_AUX_FEATURE_BYTES_PER_SQUARE = 4`
- `BYTES_PER_SQUARE_RECORD = 141`
- `TOTAL_BYTES = 9634`

---

## Code that touches V3

### Ceres (chess-engine + inference)
| File | Role |
|---|---|
| `Ceres.Chess/NNEvaluators/Ceres/TPG/TPGRecord.cs` | V3 const flags, byte sizing |
| `Ceres.Chess/NNEvaluators/Ceres/TPG/TPGSquareRecord.cs` | `AuxFeatureBytesSetter` writes 4 aux bytes per square. `WritePosPieces` bakes them at record-write time via `PerSquareAttacks.ComputeExtendedFeatures`. |
| `Ceres.Chess/Position/PerSquareAttacks.cs` | The bitboard math. Three entry points: `Compute(in MGPosition)` (attackers only, live use), `ComputeExtendedFeatures(in MGPosition, ...)` (the 4 V3 aux, used by WritePosPieces), `ComputeExtendedFromTpgSquareBytes(...)` (used by V2→V3 upgrade) |
| `Ceres.Chess/NNEvaluators/Ceres/TPG/TPGConvertersToFlat.cs` | Live-inference path. Auto-detects two model widths: 137 (legacy aux-blind) or 141 (V3 with all 4 aux). Slices off aux tail when feeding a legacy 137-channel model. |
| `tests/AugFeatSanity/` | Test project: Phase 0 (layout sanity, 141 bytes/sq, 9634 bytes/record), Phase 1 (starting-pos attacker truth), Phase 2 (Python↔C# attacker-byte equality on 25 FENs). |

### CeresTrain (data pipeline + training)
| File | Role |
|---|---|
| `src/Tasks/TPGConvertV2ToV3.cs` | V2→V3 upgrade tool. Constants: `V3_BYTES_PER_POS=9634`, `SQ_BYTES_V3=141`, `NUM_AUX_BYTES=4`. `UpgradeOnePosition` writes the 4 aux bytes via `ComputeExtendedFromTpgSquareBytes`. |
| `src/CeresTrainPy/tpg_dataset.py` | Loader reads `BYTES_PER_POS=9634`, `SIZE_SQUARE=141`. When `CERES_AUX_FEATURES_PER_SQUARE=0`, slices to 137 (legacy net compat). `CERES_AUX_CHANNEL_INDICES` env var supports cherry-picking specific aux channels for ablation. |
| `src/CeresTrainPy/config.py` | `NUM_AUX_FEATURES_PER_SQUARE` env-var-driven; widens model input embed by N channels. |
| `scripts/validate_v3ext_aux_bytes.py` | Python oracle that bit-exactly verifies the 4 aux bytes written by C# `WritePosPieces` against an independent python-chess implementation. Tested on 5K+ positions, zero mismatches. |

---

## CLI commands

### Upgrade an existing V2 corpus to V3 (preserves labels, no MCTS rerun needed)

```bash
./CeresTrain.exe upgrade-tpg-v2-v3 \
  --tpg-dir-in  <V2-corpus-path> \
  --tpg-dir-out <V3-output-path> \
  --zstd-level 5 \
  --max-files-parallel 4
```

Throughput: ~100K positions/sec on a modern multi-core machine. 5M positions → ~50 sec.

### Generate V3 corpus from scratch (TAR → TPG)

The `gen-tpg` command routes through `WritePosPieces` which auto-writes all 4 aux channels. Use the standard tool the same way as before; the new format is automatic when `USE_V3_TPG_RECORD=true` (current default).

### Training

Toggle V3 aux consumption via env var:
```bash
CERES_AUX_FEATURES_PER_SQUARE=4 python3 train.py <config-id> <ceres-train-root>
```
Setting `=0` reproduces a legacy 137-channel model on the same V3 corpus (auto-slices the aux tail).

---

## Validation

End-to-end correctness verified by `scripts/validate_v3ext_aux_bytes.py`:

```bash
python3 scripts/validate_v3ext_aux_bytes.py /path/to/V3-corpus.zst 5000
```

Decompresses the corpus, decodes positions to a chess.Board, computes the 4 aux channels independently via python-chess, compares byte-by-byte to the C#-baked values. On 5500 positions × 64 squares × 4 channels = 1,408,000 byte comparisons across self-play + puzzle shards: **0 mismatches**.

---

## Historical (dropped) features

The following features were tested and **removed** from V3 after ablation. Kept here for posterity:

- **`our_attackers`, `opp_attackers`, `net_attackers`** (V3-MVP, May 2026): the original 3-channel V3 design. Showed +73.9 OOD Pol Perf in early under-trained ablations (LR=2e-4), but at the proper prod LR (Muon-LR8e-4) gave **tournament-tied vs 0-aux**. Model derives attacker counts cheaply in 1 attention layer; baking them as inputs is dead weight at this scale.

- **`SEE`** (Static Exchange Evaluation, May-June 2026): the 8th channel. Bit-exact validated against a Python SEE oracle. Tournament-negative under AdamW; puzzle-tied with no-SEE under Muon. Captures ~half the signal of the 4 tactical features as a standalone, but adds nothing marginal when combined with them (model can integrate `is_threatened` + position structure into SEE-like evaluation internally).

Implementations of both are preserved in git history (commit prior to the 2026-06-01 cleanup) if revisiting at billion-position scale ever changes the picture.

---

## Notes

1. **Backward-compat with 137-channel models is preserved.** TPGConvertersToFlat auto-detects the caller's buffer width (137 or 141) and slices the aux tail when serving a legacy model. Existing production nets (like `lepned_320_15_rope_v3_*`) continue to serve correctly without rebuild.

2. **NPS impact: negligible.** Inference NPS test on 137-ch prod net: aux-baking overhead is within 1% noise. The CPU work in `WritePosPieces` is well-amortized vs GPU forward pass.

3. **Format is stable.** This is the final V3 design after extensive ablation. Further channel additions would constitute V4.
