# K-Ply Survival Targets — Research Spec (v0.1, 2026-07-06)

Dense per-square **auxiliary prediction targets** for value-head research: for every
training position, predict the short-horizon *fate* of each piece (and each king's
safety). Labels are derived for free from the game continuation already present in
the training data at TPG-generation time. The head(s) are training-only auxiliaries
in the placement-value-head mold (env-gated, stash-pattern, never exported).

**SCOPE (user decision 2026-07-06): pure tactics research.** The goal of this phase
is a tactics-aware architecture/training pipeline, judged ONLY on tactical
performance (puzzle bands). Positional aspects and general-play cost (tournament
Elo, MCTS-fit calibration) are explicitly out of scope until a tactical
breakthrough exists; then the general-play cost gets its own evaluation phase.

Motivation: the value head must stop reciting corpus priors and become a function of
piece placement (see `project_kovax_corpus_skip_ablation` / placement-head work).
Survival-to-K-plies is an almost purely *tactical* target — hanging pieces, capture
sequences, pins, overloads, mating nets — the dynamic version of the V3 aux INPUT
features (`is_threatened`, `defender_count`) that were worth +39 Elo, but demanded
as an *output*, so the trunk must compute the tactics itself. Prior art: KataGo's
ownership/score auxiliary targets (large documented value-learning efficiency gains);
this is the tactical-horizon chess translation.

## 1. Label definition

### Channel S — piece fate (per square, uint8)

For the piece standing on square `s` in position `P_i` of a game, look at the actual
continuation `P_{i+1} … P_{i+K}` (K plies) of the SAME game:

| value | meaning |
|---|---|
| 0 | square empty in `P_i` (masked out of the loss) |
| d = 1..K | piece is **captured** exactly `d` plies later |
| K+1 | piece **survives** the horizon (or survives to game end if the game ends sooner) |

Rules (piece identity tracked by replaying the game's actual moves):
- A piece that **moves** stays itself (follow it to its new square).
- **Promotion**: the pawn survives as the promoted piece.
- **Castling**: king and rook both simply move (KTR encoding in FRC cells — replay
  must use the engine's Chess960-aware make-move, NOT naive from-to).
- **En passant**: the captured pawn's square is the EP-captured square, not the target square.
- Game ends before `i+K` (mate/draw/adjudication): pieces alive at the final
  position are labeled K+1 (survives). No extrapolation.

Labels are **two-sided** (both colors on all 64 squares) and physical (no
side-to-move flip; the input squares are already stm-oriented in TPG — the sidecar
follows the SAME square ordering as the record's 64 square slots, so square index
alignment is automatic).

### Channel A — king attack (per square, uint8; v1.1, same replay pass)

0 everywhere except the two king squares of `P_i`:
`min(number of checks this king receives within K plies, 3) + 4 * (this side is MATED within K plies)`.
Captures the attacking dimension explicitly (see §2). Ships immediately after
Channel S (identical plumbing; labels computed in the same replay).

Default **K = 8** (research knob `--survival-horizon`; sweep 4/8/16 later).

## 2. Sacrifice semantics — why sacs are NOT mislabeled (design position)

Objection: tactics often *sacrifice* material — survival labeling marks the
sacrificed rook "captured in 2", indistinguishable from a blunder.

Resolution: the label is **descriptive, not evaluative**. It answers "what happens
to this piece", never "was that good". Three consequences:

1. A successful sacrifice is *richly* labeled, not mislabeled: my rook dies at ply 2
   (S=2 on its square), AND the defender's knight dies at ply 3, pawn shield dies at
   ply 4-5, king gets checked 3 times and mated (Channel A) — the enemy casualties
   and king events ARE the attack's payoff, visible because labeling is two-sided.
   A head that predicts this whole pattern has understood the combination.
2. "Good vs bad" remains the job of the value target (`wdl_q`), trained jointly as
   ever. The pairing is exactly the compensation lesson: trunk sees (my rook dies,
   their king falls, wdl_q says winning) → material deficit + these attack patterns
   = won. Survival supplies the *mechanism*; wdl_q supplies the *verdict*.
3. What the horizon genuinely misses: slow attacking build-ups where nothing is
   captured within K plies, and long-delayed positional-sac payoffs. Channel A
   (checks/mate-within-K) covers part of the first; the rest is out of scope for
   this target and stays with wdl_q. This is a tactical-horizon target by design.

Known label-noise sources (accepted for v1): the label reflects the single actual
game continuation (including exploration blunders — noise 0.1/0.12, ptemp 1.45 in
the kovax data); voluntary equal trades are labeled the same as material losses
(descriptive semantics make this acceptable — trades are real capture dynamics).
Mitigation available later: skip labels on positions the deblunderer flags
(NoiseBlund), or teacher-verify a subsample.

## 3. Storage — sidecar stream (NOT a record-format change)

**`<shard>.tpg_setN.tgt.zst`** alongside each `<shard>.tpg_setN.zst`:

```
header (16 bytes): magic "TPGT" | uint8 version=1 | uint8 numChannels C | uint8 K | 9 reserved
body: [num_records, 64, C] uint8, zstd-compressed, record order IDENTICAL to the main shard
```

Rationale: keeps the main shards **pure upstream V2** (user requirement; shareable
with kovax/dje), targets are an optional overlay (loader trains without them if
absent), no `TPGRecord` struct change / no `USE_*` define, trivially strippable, and
new channels only bump C. Rejected alternative: appending target bytes per square in
the record (V3-style) — conflates inputs with targets, breaks V2 purity, resurrects
the define-mismatch trap.

**Order guarantee**: the sidecar row MUST be appended at the exact point the main
record is appended, by the same writer (`TrainingPositionWriter`), per concurrent
set — never in a parallel pass (the generator's threading/deblunder/skip logic makes
any out-of-band ordering assumption wrong).

## 4. Generation (C#) — implementation map

1. **Compute** per game, once: in `TrainingPositionGenerator` where the per-game
   loop runs with full game context (`TrainingPositionGenerator.cs:453`,
   `gameAnalyzer.PositionRef(i)` for every ply). Replay the game's moves
   (`EncodedTrainingPositionGame` → MGMove path, Chess960-aware) building
   `fate[ply][64]` + `kingEvents[ply][64]` arrays in one O(plies) pass, then slice
   per emitted position (fate at ply i = f(captures at plies i+1..i+K)).
2. **Carry**: add `byte[] SurvivalTargets` (64*C) to `TPGTrainingTargetNonPolicyInfo`
   (already threaded per-position into the writer: `TrainingPositionWriter.Write(record, targetInfo, ...)`
   at `TrainingPositionWriter.cs:216`).
3. **Write**: in the writer's per-set append path (same place
   `TPGRecordConverter.ConvertToTPGRecord` output is buffered,
   `TrainingPositionWriter.cs:437`), append the 64*C bytes to a parallel per-set
   zstd stream; open/close alongside the main stream.
4. **CLI**: `gen-tpg --survival-horizon K` (0 = off, default 0 → bit-identical
   current behavior and no sidecar files).
5. **Validation tool** (python, one-shot): for a sample of records, decode the
   shard's square bytes to a board, replay via python-chess from the raw kovax tar
   by position match, recompute fate independently, compare bit-for-bit (the same
   oracle-validation pattern used for the V3 aux bytes, `validate_v3ext_aux_bytes.py`).

## 5. Loader (`tpg_dataset.py`)

- Env `CERES_TPG_TARGET_SIDECAR=1`: for each shard, require `<shard>.tgt.zst`,
  stream it in lockstep (same remainder-carry pattern as the main stream; validate
  header + equal record counts; hard error on mismatch — silent misalignment is the
  known corruption class here).
- Yield an extra `survival_targets` [B, 64, C] uint8 tensor in the batch dict.
- The `CERES_KEEP_DRAW_PROB` filter and any future row filter MUST index it with the
  same `_keep` mask (add to the filter list — this list is a known desync hazard).

## 6. Model + loss (`ceres_net.py`, placement-head patterns throughout)

- `survival_head = nn.Linear(EMBEDDING_DIM, K+2)` applied per square on trunk flow
  → [B, 64, K+2] logits; CE against Channel S with **empty squares masked**
  (class 0 excluded from loss; mean over piece squares). ~2.6K params at K=8.
- Channel A head analogous (8-class), king squares only, v1.1.
- Env `CERES_SURVIVAL_TARGET_WEIGHT` (default 0 = off; start 0.3). Reuse verbatim:
  `self.training`-gated stash (export safety), `LossCalculator.survival_loss` with
  PENDING/LAST accumulator (+ per-class accuracy for monitoring), `SURV:` log line,
  resume key handling (generalize the placement `placement_value_` prefix strip/init
  to an aux-prefix list: `('placement_value_', 'survival_')`), DDP loud-error guard,
  4-board loud-error guard, per-stream routing multiplier `CERES_SECONDARY_LOSS_SURVIVAL_MULT`.

### 6.1 Loss-shaping variants (all pure loss-side; head shape and checkpoints unchanged)

- `CERES_SURVIVAL_LOSS_BUCKETS="2,4,8"` — ordinal-bucket CE: exact-ply logits pooled by
  logsumexp into capture-distance buckets [1-2],[3-4],[5-8],[survives]; CE at bucket
  granularity. Adopted default (exact timing is move-order noise; measured 13% exact-ply).
- `CERES_SURVIVAL_CAPTURE_WEIGHT=4` — CE class weight on capture classes/buckets
  (survives stays 1); counters the ~10:1 survive:capture imbalance. Adopted default.
- ❌ REMOVED after falsification (2026-07-07, t91s1k4i20M 20M A/B vs t91s1k420M): ordinal
  all-threshold loss + piece-value square weighting, tested TOGETHER = Pareto-negative
  (value −11..−52 all bands, policy −9..−14, KLD worse) → both deleted from losses.py.
  Post-mortem hypotheses: piece-value weighting inverts the signal economics (pawn/minor
  fates are the dense reliable signal; major fates rare+noisy), and the ordinal objective's
  smaller magnitude acted as an accidental aux-weight REDUCTION (train loss 0.153 vs 0.246)
  — the latter observation motivated the (successful) weight sweep. Do not re-add without
  a new mechanism argument.
- Unit tests: `src/CeresTrainPy/test_survival_loss.py` (subprocess-per-env-mode; pins the
  exact-ply and bucket CE paths against independent reimplementations).
- Survival weight: **0.6 adopted** (2026-07-07 sweep 0.3/0.6/1.0 at K=4/20M: mate value
  +50/+42 in two independent arms, quiet bands unmoved, policy free; saturates by 0.6).
- K sweeps WITHOUT regen: the loader remaps sidecar labels losslessly downward when
  `CERES_SURVIVAL_HORIZON` < sidecar K (captured at ply > K' == survived the K'-ply
  horizon -> class K'+1). One K=8 corpus serves any K' <= 8; K' > sidecar K errors
  loudly (regen required — "survives beyond 8" cannot be split). Bucket bounds must
  end at the CONFIGURED K' (e.g. K'=4 -> CERES_SURVIVAL_LOSS_BUCKETS="2,4").

### 6.2 Per-head gradient-norm measurement

`CERES_LOG_GRAD_NORMS_EVERY=N` — on every Nth stats interval, run a diagnostic pass that
backprops each head's loss separately and prints `GRADNORM: <head> , <raw> , <weighted>`
lines. Single-GPU + `PyTorchCompileMode` off only; each pass costs ~one backward per head.
Intended use: short measurement runs to put numbers on how much trunk-shaping each head
actually does (basis for weight choices instead of loss-magnitude guessing).

## 7. Experiment plan

1. Regenerate the kovax skip-1 corpus with `--survival-horizon 8` (~4 min + replay
   cost; sidecars ≈ 64B/pos raw, ~a few MB/shard compressed).
2. Oracle-validate 4K sampled labels (python-chess, incl. FRC cells) before any run.
3. Arms at 20M positions, 256×10 prodclone recipe (baselines already exist):
   a. baseline (done: `kvx20M_s1`) · b. +survival@0.3 · c. +survival+placement.
4. Read: puzzle bands (a *tactical* target should show there, unlike placement —
   mate + rg bands are the honest yardstick for this one); survival-head accuracy
   by class (does it actually learn captures-at-d?); value/policy non-regression.
5. Success gate (TACTICS-ONLY phase): value Perf +≥50 on ≥2 tactical bands over
   baseline (outside single-seed noise), or clear policy gains (threat-awareness
   helps policy too). Then: K sweep, weight sweep, Channel A, aggressive combos
   (survival aux + puzzle-policy stream via loss routing). Tournament/general-play
   evaluation is DEFERRED by design to a later phase (v66-v70 lesson still applies
   before any production/shipping claim — but not to this research loop).
6. Interpretability: per-square predicted-fate heatmaps from any checkpoint ("the
   net thinks this knight is lost in 3") — use real TPG records as input, never
   hand-encoded FENs (known standalone-input trap).

## 8. Open questions / v2 ideas (parked)

- Distance-bucketed classes {1-2, 3-4, 5-8, survives} vs exact-ply (v1 = exact; buckets if class imbalance hurts).
- Second horizon channel (game-end survival = positional variant) for later A/B with tournament gate.
- Blunder-masked labels (skip fate labels crossing a NoiseBlund ply).
- Counterfactual (search-verified) fates for a subsample via teacher MCGS — turns descriptive labels into "best-play" fates where budget allows.
