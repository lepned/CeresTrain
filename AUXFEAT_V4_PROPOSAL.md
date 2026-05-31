# V4 Aux-Feature Proposal — Survey + Recommended Roadmap

**Status**: Research draft (2026-05-31, autonomous). Not yet implemented. Goal is to identify which next features to add after the V3 baseline.
**Context**: V3 ships with 3 per-square bytes (our/opp/net attackers). Phase 4: OOD +26 Pol Perf, NPS −5%. The format is extensible: V4 = more bytes per square, same bake-in pattern.

---

## 1. Survey of feature engineering across engines

### 1.1 NNUE-family (Stockfish, Berserk, Obsidian, Carp)

**Idea**: every piece's contribution depends heavily on KING POSITIONS. Encode each piece-square as a feature indexed by both king squares.

**HalfKAv2_hm** (Stockfish 17): for each side, ~45K sparse binary features = (king_sq × piece_type × piece_sq). Horizontal-mirror symmetry halves the feature space.

**Architecture fit**: NNUE feature transformer is a sparse linear layer (~50M params). Our transformer takes DENSE per-square tokens — so we can't import NNUE features directly. But we CAN distill the *spirit* (king-relative encoding) into per-square dense features:

| NNUE concept | Transformer-friendly per-square encoding |
|---|---|
| HalfKA(king_sq, piece_sq) | KingDistanceFromUs[s], KingDistanceFromOpp[s] |
| KingZone (queen + king radius around our king) | IsInKingZoneUs[s], IsInKingZoneOpp[s] (boolean) |
| KingAttackers count | NumOurAttackersOnOppKingZone (global, broadcast) |
| Material imbalance vector | Global material features (broadcast to all squares) |

### 1.2 Classical HCE (Komodo, classical Stockfish before NNUE)

Per-square / per-piece features computed at eval time:
- **Mobility per piece** (count of legal moves)
- **Pinned/skewered detection**
- **Hanging-piece detection** (attackers > defenders + piece value)
- **Outpost squares** (squares defended by pawn that opponent can't attack with pawn)
- **Bishop pair bonus**
- **Doubled/isolated/passed pawns**
- **Open/semi-open files for rooks**
- **King safety** (pawn shield count, attacker pressure on king zone)
- **PSQT** (piece-square tables for positional eval)

### 1.3 AlphaZero / Lc0 (no engineered features)

Raw board planes only. The network learns its own representations. **No special features added.**

This is the philosophy our project departed from with V3. The MVP win (OOD +26) validates "explicit features help at our data scale" (10M positions).

### 1.4 Modern research approaches

- **DeepMind Searchless Chess (2024)**: distilled from Stockfish action-values. No engineered features.
- **Anthropic chess GPT (2024)**: tokenize moves directly. Skip board features entirely.
- **Chessformer (2023)**: smolgen (already in our arch) + standard transformer. No engineered features.
- **BT4 (Lc0 dev)**: history planes only, smolgen for attention bias.

The unique contribution of V3 is **bringing back HCE-style features in a way that composes with transformer attention**. No published work does exactly this AFAIK.

---

## 2. Compatibility filter: what fits V3+ format

Constraint: features must be encodable per-square in 1 byte (0-255). For booleans, 0 or 100 (to match the `/100 → float` pipeline).

| Feature class | Per-square fit | Global fit |
|---|---|---|
| Attack counts | ✓ (V3 ships this) | (already there via aggregation) |
| Mobility per piece | ✓ (depends on piece on square) | — |
| Defender count | ✓ | — |
| Pinned/hanging flags | ✓ (boolean) | — |
| King distances | ✓ (Chebyshev × scaling) | — |
| Material imbalance | ✗ (not per-square) | ✓ (broadcast, or 65th token) |
| Phase indicator | ✗ | ✓ (broadcast scalar) |
| Pawn-structure flags | ✓ (per pawn square) | — |
| PSQT values | ✓ (1 byte per square) | — |

**Per-square ones are easier** — drop right into the V4 layout. **Global ones need either broadcast (wasteful) or an architectural change** (add a 65th token).

For first V4 iteration, stick to per-square only. Global features can be V5.

---

## 3. Recommended V4 feature set (ranked by signal-per-byte)

Each row: byte cost / implementation difficulty / expected signal.

| Rank | Feature | Bytes | Compute | Expected signal | Why |
|---|---|---|---|---|---|
| **1** | **Mobility per piece** | 1 | `popcount(attacks_from(piece, sq, occupancy))` | High | Mobility is the strongest classical positional signal. Free to compute (we already compute attacks). Sliding pieces benefit most. |
| **2** | **Defender count** | 1 | `our_attackers[sq]` restricted to "this is our own piece on sq" | High for tactics | Decomposes V3's `our_attackers` into pure defenders (when own piece is here) vs attacks (when empty/opp piece). Network currently has to learn this decomposition. |
| **3** | **Is-pinned** | 1 (bool) | Standard pin detection (line from king through piece to opp slider, no blockers in between) | Medium-high for tactics | Pinned pieces have constrained mobility — explicit encoding helps tactical eval. |
| **4** | **King distance — us** | 1 | Chebyshev distance to our king × 14 (range 0-98 → byte 0-98) | Medium for endgame | Endgame king activity. NNUE-spirit feature. |
| **5** | **King distance — opp** | 1 | Same for opp king | Medium for endgame | Together with #4 enables king-pair-relative reasoning. |
| **6** | **Is-hanging** | 1 (bool) | `opp_attackers > our_attackers` AND there's our piece here | Low-medium (derivable from V3) | Network could derive this from V3 attacker counts, but explicit flag might accelerate learning. Test if it helps. |
| **7** | **PSQT value** | 1 | Lookup table per (piece_type, sq) — classical | Medium | Cheap classical positional eval. Has been shown to bootstrap learning faster. |
| **8** | **X-ray attackers — opp** | 1 | Sliders attacking through 1 blocker | Low-medium tactical | Detects pin/skewer threats one move ahead. Real but smaller signal than direct attacks. |

### Recommended V4 layout (cumulative bytes)

- **V4-mini**: features 1-3 (mobility, defender, is-pinned) = 3 bytes/sq → 6 bytes/sq aug total → 9762 bytes/record (+2% over V3)
- **V4**: features 1-5 (add king distances) = 5 bytes/sq → 8 bytes/sq aug total → 9890 bytes/record (+3.3% over V3)
- **V4-full**: features 1-7 (add is-hanging, PSQT) = 7 bytes/sq → 10 bytes/sq aug total → 10018 bytes/record (+4.7% over V3)

### Bonus features for hypothetical V5

If V4 wins big, V5 candidates:
- Global material imbalance vector (would need 65th token or broadcast)
- Phase indicator (global)
- King-pressure (attack count of opp on our king zone)
- Pawn-structure flags (passed/isolated/doubled — only meaningful for pawn squares)

---

## 4. Implementation pattern (per feature)

Each new V4 feature follows the V3 V2→V3 template. Steps for each:

1. **Compute logic** in `Ceres.Chess.PositionDataInfo.PerSquareAttacks` (or a new helper class for non-attack features):
   ```csharp
   public static void ComputeMobilityPerSquare(in MGPosition pos, Span<byte> mobility)
   {
     // For each piece, count its attack-mask bits = mobility
     // Empty squares get 0
   }
   ```
   ~40 LoC per feature.

2. **Bake into V4 record** in `TPGSquareRecord.WritePosPieces`:
   ```csharp
   if (TPGRecord.USE_V4_TPG_RECORD) {
     // ... existing V3 aug bytes ...
     // ... new V4 bytes ...
     pieceRecord.AuxFeatureBytesSetter[3] = (byte)mobility[squareNum];
     pieceRecord.AuxFeatureBytesSetter[4] = (byte)defenderCount[squareNum];
     // ...
   }
   ```

3. **Bump format constant** in `TPGRecord.cs`:
   ```csharp
   public const bool USE_V4_TPG_RECORD = true;
   internal const int NUM_AUX_FEATURE_BYTES_PER_SQUARE = USE_V4_TPG_RECORD ? 8 : (USE_V3_TPG_RECORD ? 3 : 0);
   ```

4. **V3→V4 upgrade tool** in `CeresTrain/Tasks/TPGConvertV3ToV4.cs` (~mirror of V2→V3 converter):
   - Reads V3 .zst
   - Re-computes features from existing piece data
   - Writes V4 .zst
   - CLI command `upgrade-tpg-v3-v4`

5. **Python loader** updates:
   - `tpg_dataset.py`: BYTES_PER_POS = 9890 (or 10018)
   - SIZE_SQUARE = 145 (or 147)
   - `config.py`: `NUM_AUX_FEATURES_PER_SQUARE = 8` (or 10)

6. **Sanity test** in `tests/AugFeatSanity`: extend with V4 layout check + new feature-byte-correctness validation against python-chess oracle.

7. **MVP train + eval** following the same 10M smoke → 30M scale pattern as V3.

**Time estimate per feature**: 4-6 hours from spec to validated.

---

## 5. Critical question — is this worth doing?

The V3 result was clean (+26 OOD). But each new feature has diminishing returns:
- V3 (3 channels) → +26 OOD
- V4 (5 channels) → expected +30-40 OOD?
- V5 (8 channels) → expected +35-50 OOD?

After some point, additional features don't help (the network can learn the patterns from existing inputs at sufficient data scale). The "Bitter Lesson" warns against this.

**My recommendation**: test V4-mini (3 more bytes: mobility + defender + is-pinned) first as a 10M smoke. If it adds +20 OOD on top of V3, ship it. If not, the diminishing-returns curve says stop adding features.

**Skip if**: Phase 5 30M result shows V3 alone is already production-grade (>+40 OOD at scale). At that point invest in other levers (arch changes, training methodology) instead.

**Don't skip if**: Phase 5 30M shows V3 win shrunk vs Phase 4's +26 (data scaling diminished aug benefit), suggesting we need MORE features to keep the signal. Adding V4 mobility/defender/pinned is the natural extension.

---

## 6. What I'd implement first if I were continuing autonomously

Order of operations (each is an independent commit):

1. **Add `Mobility` field to PerSquareAttacks** (~40 LoC). Validate against python-chess. Pure addition — no V4 yet.
2. **Add `DefenderCount` field** (~30 LoC). Same.
3. **Add `IsPinned` flag** (~80 LoC — pin detection is harder). Validate via known pinned-piece test positions.
4. **Define V4 layout** (TPGRecord const, struct size). 4-phase test for V4.
5. **Bake V4 in WritePosPieces** — append the 3 new bytes after V3's 3.
6. **Write `TPGConvertV3ToV4`** mirroring V2→V3 (mostly copy-paste).
7. **Add CLI command** `upgrade-tpg-v3-v4`.
8. **Update Python loader** for V4 byte count.
9. **Upgrade quietmix V3 → V4** (already small, fast).
10. **10M MVP train** with V4 + eval.

Per-step cost: ~30 min to 2 hours. Full V4-mini: ~1 day.

---

## 7. NNUE-specific deeper dive (king-relative ideas)

NNUE's success comes from the **king-square indexed feature transformer**. The first layer learns piece-square embeddings RELATIVE TO each king position. For 6-piece-type × 64-square × 64-king-square = 24K features per side, the first layer has ~24M params learning these embeddings.

Adapting to transformer per-square encoding:
- **KingZoneFlagUs[s]**: 1 byte boolean — is square s in the 3×3 zone around our king?
- **KingZoneFlagOpp[s]**: same for opp king
- **KingDistanceUs[s]**: Chebyshev distance × 14 (max 98)
- **KingDistanceOpp[s]**: same for opp

These 4 features (4 bytes/sq) bring NNUE-spirit king-relativity to our transformer. Cheap to compute, mechanical. Could be V4 alternative or V5.

Why this MIGHT not work as well as in NNUE: our transformer already has SMOLGEN, which is a content-aware attention bias. Smolgen probably learns king-relative patterns implicitly (it has access to all 64 squares' embeddings simultaneously). Explicit king-distance might be redundant.

To test: implement king-distance and king-zone features. Run 10M ablation. If they add >+10 OOD on top of V3, smolgen wasn't capturing them. If they add <+5, smolgen already had it.

---

## 8. Long-tail features (probably skip)

Things that would technically work but are unlikely to move the needle:

- **Bishop pair flag** (global boolean): too coarse, smolgen learns it
- **Castled-side flag**: in TPG metadata already
- **Phase indicator** (broadcast global): material counts already implicit in piece bitboards
- **Opponent threat squares** (squares opp can attack next move): would need 1-ply lookahead per generation. Expensive. Could be V6 if everything else stalls.
- **Captureable values**: position-dependent; better learned by the net than handcrafted

---

## 9. Storage budget perspective

If we ever add ALL the features I've listed (mobility, defender, is-pinned, king-dist×2, king-zone×2, is-hanging, PSQT, x-ray×2 = 11 extra bytes/sq), we'd be at:
- 14 aug bytes/sq × 64 = 896 aug bytes/record
- Plus 9250 base + 128 V2 = 9378
- Total: 10,274 bytes/record = +9.5% over V3 (and +9.6% over V2)

At billion scale: ~10 GB extra over V3 (~45 GB extra over V2). Still very manageable.

**The byte budget is generous.** The constraint is signal-to-noise, not storage.

---

## 10. Decision tree for next session

```
After Phase 5 30M completes, look at the OOD Pol Perf delta vs Phase 4's +26:

  ≥ +50 OOD: V3 alone is enough. Don't add more features.
              Move to next architectural lever (arch search, training methodology).

  +25 to +50 OOD: V3 holds at scale. Adding V4 might compound — try V4-mini.

  +10 to +25 OOD: V3 partially scales. V4 likely needed.
                  Implement V4-mini (mobility + defender + pinned).

  < +10 OOD: V3 didn't scale well. Either change recipe or invest in
              architectural levers (smolgen tuning, depth, etc.).
              Adding features alone won't fix this — fundamental issue elsewhere.
```

---

## Appendix: example king-distance compute (POC)

```csharp
public static class KingDistanceFeatures
{
  // Precomputed Chebyshev distance lookup: 64*64 = 4096 bytes
  private static readonly byte[,] DISTANCE_TABLE = InitDistanceTable();

  private static byte[,] InitDistanceTable()
  {
    var t = new byte[64, 64];
    for (int a = 0; a < 64; a++)
      for (int b = 0; b < 64; b++)
      {
        int dr = Math.Abs((a >> 3) - (b >> 3));
        int df = Math.Abs((a & 7) - (b & 7));
        t[a, b] = (byte)Math.Max(dr, df);
      }
    return t;
  }

  public static void Compute(in MGPosition pos,
                              Span<byte> distFromOurKing,    // 64 bytes
                              Span<byte> distFromOppKing)    // 64 bytes
  {
    // Find king squares (using existing piece bitboards)
    int ourKingSq = ExtractKingSquare(pos, isOurs: true);
    int oppKingSq = ExtractKingSquare(pos, isOurs: false);
    for (int s = 0; s < 64; s++)
    {
      distFromOurKing[s] = (byte)(DISTANCE_TABLE[s, ourKingSq] * 14);  // scale 0..7 → 0..98
      distFromOppKing[s] = (byte)(DISTANCE_TABLE[s, oppKingSq] * 14);
    }
  }
}
```

40 LoC. Mechanical. Validated by symmetry tests (in starting pos, dist[a8] from our K should = dist[a1] from opp K under us=WHITE convention).

---

## Notes for tomorrow

- Phase 5 30M training result is the gate. Read decision tree (section 9) first.
- If proceeding with V4: implement features 1-3 (mobility, defender, is-pinned) as the V4-mini smoke.
- The V2→V3 converter pattern works as a direct template for V3→V4.
- This doc + V3 format doc (`AUXFEAT_V3_FORMAT.md`) together are the full feature-engineering handoff.
