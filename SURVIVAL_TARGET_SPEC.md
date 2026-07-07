# K-Ply Survival Targets — Spec + As-Built Reference (v1.0, 2026-07-07)

> **STATUS: IMPLEMENTED, VALIDATED, TOURNAMENT-CONFIRMED.** Channel S (piece fate) is fully
> built and committed (gen 565ada1, puzzles da2ba11, training 3a7a479, scripts 0ff2ec5).
> Channel A (king attack) is DESIGN ONLY — not implemented; sidecars on disk have C=1.
> §9 is the production data-preparation quickstart — start there if you are the other machine.

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

### Channel A — king attack (per square, uint8) — ⚠️ DESIGN ONLY, NOT IMPLEMENTED

0 everywhere except the two king squares of `P_i`:
`min(number of checks this king receives within K plies, 3) + 4 * (this side is MATED within K plies)`.
Captures the attacking dimension explicitly (see §2). Same replay pass as Channel S;
the sidecar header's numChannels byte reserves room. Queued in the corpus-v2 regen
bundle; all existing sidecars are single-channel (C=1, Channel S only).

**K guidance (post-sweep, 2026-07-07): GENERATE at K=8, TRAIN at any K' ≤ 8.**
The loader remaps labels losslessly downward (see §6.1), so one K=8 corpus serves every
smaller horizon; K' > 8 would require regeneration. A 2/4/8 sweep at 20M was null (K
insensitive; even 2-ply induces the threat features) — training default is K=4 (easier
task, smaller head), but always emit sidecars at K=8 to keep the corpus future-proof.

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

## 4. Generation (C#) — AS BUILT (commit 565ada1)

1. **Compute**: `SurvivalLabeler.ComputeGameSurvival(in game, horizonPlies)`
   (`src/TPG/TPGGenerator/SurvivalLabeler.cs`) — one O(plies) pass per game using
   position-DIFF piece tracking (compares consecutive board states rather than
   replaying moves; FRC/EP/promotion-safe, asserts on incoherent transitions).
   Returns `byte[][] gameSurvival` = per-ply [64] fate labels in REAL-BOARD indexing.
   Called from the per-game loop in `TrainingPositionGenerator.Read` (gated on
   `Options.SurvivalTargetHorizon > 0`).
2. **Carry**: the per-square labels travel as an explicit `byte[] survivalBySquares`
   element of the writer's item tuple — NOT inside `TPGTrainingTargetNonPolicyInfo`
   (the spec's original plan; the tuple slot was cleaner). Real-board → record-slot
   remap happens in `PreparePosition` (black-to-move slot = realSquare ^ 56).
3. **Write**: `TrainingPositionWriter` opens a parallel per-set zstd stream
   `<base>_setN.tgt.zst`, writes the 16-byte header at open, buffers one 64-byte row
   per record in `bufferSurvivalBySquare`, and flushes rows at the exact same
   4096-record flush points as the main stream (order lockstep by construction).
   Guards: sidecars require zstd output; incompatible with evaluator/postprocessor
   record-omission modes (throws at construction).
4. **CLI**: `gen-tpg ... --survival-horizon K` (0 = off, default 0 → bit-identical
   legacy behavior, no sidecar files). See §9 for the full production command.
5. **Validation** (as built): `scripts/check_survival_sidecars.py` — structural gate
   (header, shard/sidecar row lockstep, empty-mask ≡ label-0 which catches any
   desync or slot-flip, kings-always-survive, class stats). The empty-mask
   equivalence is the load-bearing check: it fails on ANY record-order or
   orientation error. An orientation bug was in fact caught this way pre-v1
   (the s^56 slot mapping). Plus training-side sanity: fate accuracy must clear the
   trivial all-survive floor (~91% at K=8, ~96% at K=4 bucket-graded).

## 5. Loader (`tpg_dataset.py`) — AS BUILT (commit 3a7a479)

- Env `CERES_TPG_TARGET_SIDECAR`: `0`/unset = off (legacy) · `1` = required (every
  shard must have `<shard>.tgt.zst`, hard error otherwise) · `auto` = per-shard
  (sidecar-less shards yield batches with NO 'survival' key and the loss skips them
  — enables a huge sidecar-less primary mixed with survival-labeled secondaries).
- Sidecar streamed in lockstep with the shard via `read_exact` (zstd stream_reader
  stops at FRAME boundaries — a naive short-read check silently drops data; this bit
  us once). Header validated: magic/version/channels, and K per §6.1's remap rule.
- Batch key: **`batch['survival']`**, shape **[B, 64] uint8** (C=1 is implicit —
  a channels dimension appears only if/when Channel A ships).
- `.tgt.zst` files are EXCLUDED from shard discovery (a sidecar being picked up as a
  training shard was a real bug, now guarded).
- The draw filter and any row filter index survival with the same `_keep` mask
  (this filter list is a known desync hazard — extend it for any new filter).

## 6. Model + loss (`ceres_net.py`, placement-head patterns throughout)

- `survival_head = nn.Linear(EMBEDDING_DIM, K+2)` applied per square on trunk flow
  → [B, 64, K+2] logits; CE against Channel S with **empty squares masked**
  (class 0 excluded from loss; mean over piece squares). ~2.6K params at K=8.
- Channel A head analogous (8-class), king squares only, v1.1.
- Env `CERES_SURVIVAL_TARGET_WEIGHT` (default 0 = off; start 0.3). Reuse verbatim:
  `self.training`-gated stash (export safety), `LossCalculator.survival_loss` with
  PENDING/LAST accumulator (+ per-class accuracy for monitoring), `SURV:` log line,
  resume key handling (aux-prefix list, as built: `('placement_value_', 'survival_head.')`
  in train.py), DDP loud-error guard,
  4-board loud-error guard, per-stream routing multiplier `CERES_SECONDARY_LOSS_SURVIVAL_MULT`.

### 6.1 Loss-shaping variants (all pure loss-side; head shape and checkpoints unchanged)

- `CERES_SURVIVAL_LOSS_BUCKETS` — ordinal-bucket CE: exact-ply logits pooled by
  logsumexp into capture-distance buckets; CE at bucket granularity. Adopted default
  (exact timing is move-order noise; measured 13% exact-ply). Bounds must be ascending
  and END AT the configured horizon: `"2,4,8"` at K=8, `"2,4"` at K=4 (current default).
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

## 7. Results record (what has been established, 2026-07-06/07)

| experiment | result |
|---|---|
| survival@0.3 vs baseline, 20M (kovax) | value +17..+46 ALL 4 bands, policy free |
| + puzzle-surv secondary 32:1 value-masked ("combined") | policy +70..+194, value +40..+63; **tournament +38 Elo** over survival-only |
| epochs on 22.5M corpus (60M→200M pos) | policy keeps climbing; **value plateaus ~60M** |
| 100M on FRESH 191.5M corpus (14M net) | value plateau UNCHANGED → not data exhaustion |
| grad-norm measurement | value1 = 48% of trunk gradient, policy 33%, survival 1.5% → aux works by INFORMATION, not gradient share |
| K sweep 2/4/8, 20M | null (K insensitive) → train K=4, generate K=8 |
| ordinal + piece-value loss | Pareto-negative → removed (§6.1) |
| weight sweep 0.3/0.6/1.0 | mate value +50/+42, saturates at 0.6 → **0.6 adopted** |
| **43M SwiGLU/smolgen × 200M fresh (t91s1swg200M)** | **value plateau BROKEN +64..+86 all bands (capacity was binding); mate value beats prod-600mm; +68 Elo prelim (26 games, CFS 97%, TC 60+1) over prod despite 35% nps deficit** |

Interpretability: `scripts/survival_heatmap.py` renders per-square predicted-fate
grids from real TPG records (never hand-encoded FENs — standalone-input trap).

## 8a. Tablebase perfect-play survival — IMPLEMENTED + A/B-CONFIRMED (2026-07-07)

> **STATUS: BUILT, VALIDATED, RESULT POSITIVE.** `gen-endgame-tpg --survival-horizon K`
> emits `<file>.dat.tgt.zst` sidecars (same header/row format; loader-compatible by naming,
> zero loader changes). Implementation: `TablebaseSurvivalWalker` (DTZ-optimal K-ply walk)
> + `SurvivalLabeler.ComputeSurvivalForLine` reuse (identity slot map — TB positions are
> normalized white-to-move) + writer `batchPostprocessorWithTargetsDelegate` for streaming
> consumers + paired-queue transport. Also fixed: TB path resolved the tablebase dir via
> `SyzygyPath` only and threw on `DirTablebases`-style configs; now via `TablebaseDirectory`.
>
> **A/B (endgame-only, 256x10, 20M single-epoch, imbalanced 5/6-man mix, single variable):
> survival ON cuts held-out value errors by ~1/3** — holdout probe-value vs exact Syzygy WDL
> (2 x 16,384 unseen positions): acc 97.3/97.1 vs 95.8/95.5, CE -35%, corrEV 0.968 vs 0.949,
> ~10-sigma, train==holdout (no memorization). First perfect-label test of the survival
> mechanism: the effect is LARGER than on blunder-noisy game corpora.
>
> Ops notes: generate DECISIVE-friendly piece mixes (equal material is heavily drawn — an
> imbalanced mix like [KRPkr .25][KRPkrp .15][KPPkp .15][KQPkq .15][KBPkb .15][KNPkn .15]
> measured 62.4% supervised vs ~55% for KRPkrp-only); ~5-6K pos/s with walks on a 16-thread
> box; TB-arm training shows policy acc ~100% + slightly negative policy loss (near-one-hot
> DTZ policy targets; cosmetic).

### Original design rationale (kept for context)

Extend the endgame-TPG stream (`gen-endgame-tpg`, positions sampled from piece configs and
labeled by Syzygy) with survival sidecars. TB records have no game continuation — so we
SYNTHESIZE one: from each position, walk K plies of TB-OPTIMAL play (winning side picks
DTZ-minimizing moves, losing side DTZ-maximizing; deterministic tie-break), then run the
existing position-diff fate labeling over the walked line.

Why this is attractive — the labels are **perfect-play ground truth**:
- Zero label noise: game-derived fates include blunders (~6% of plies flagged in T91);
  TB-walked fates are exact. This is §8's "counterfactual best-play fates" idea, exact
  rather than teacher-approximated.
- DTZ-optimal play is naturally FORCING (DTZ counts down to the next capture/pawn event),
  so lines are rich in capture dynamics. A WDL-only walk would NOT work (winning side
  could shuffle; every label would read "survives").
- Verdict+mechanism pairing in the purest-tactics domain: the TB stream's historic role is
  endgame VALUE (verdict); this adds the perfect-play mechanism to the same positions.

Design decisions:
- v1 labels DECISIVE positions only; draws emit all-zero (unsupervised) rows — "any
  WDL-preserving move" makes drawn-position fates highly line-dependent.
- Optimal-move ties: deterministic tie-break for reproducibility; bucket loss absorbs
  residual exact-ply jitter (same move-order-noise argument as game corpora).
- Mixed semantics (game shards = realistic-play fates, TB shards = perfect-play fates,
  one head): assumed benign — piece count tells the net the regime — but A/B-gated like
  everything else (endgame stream ± survival, 20M).
- Cost: ~K x branching DTZ probes per position (memory-mapped, fast); TB streams are
  small (millions of positions) so throughput is a non-issue.
- Plumbing note: the TB generator (`TablebaseTPGBatchGenerator`) streams raw records to
  `.dat.zst` rather than going through `TrainingPositionWriter` — sidecar emission and
  loader consumption for this stream need their own wiring (see implementation).

## 8. Open questions / v2 ideas (parked)

- ~~Distance-bucketed classes~~ → ADOPTED as default (§6.1, buckets + capture weight).
- Channel A (king attack) — designed (§1), queued in the corpus-v2 regen bundle.
- Blunder-TRUNCATED labels (cut fate windows at NoiseBlund plies; truncate, don't
  discard — and only on noise_blund, since err_blund masking would bias against
  sharp positions; requires updating the validator's empty≡0 invariant).
- Second horizon channel (game-end survival = positional variant), tournament-gated.
- Counterfactual (search-verified) fates for a subsample via teacher MCGS.
- Attribution: Mish-twin of t91s1swg200M (isolate SwiGLU's share of the breakthrough).

## 9. PRODUCTION DATA-PREP QUICKSTART (for the other machine)

Prereqs: CeresTrain at commit `0ff2ec5` or later (survival series: 565ada1 → 0ff2ec5),
release build. The survival changes are gen-side C# + python; no Ceres-engine change.

**Generate** (V2 shards + K=8 sidecars; always K=8 — see §1 K guidance):

```
CeresTrain.exe gen-tpg --tar-dir <TAR_DIR> --tpg-dir <OUT_DIR> \
    --num-pos <N> --skip-count 1 --survival-horizon 8 --include-frc
```

- `--num-pos` MUST be a multiple of 4096 (writer hard-errors otherwise).
- CLI uses NAMED options only (positional args are rejected).
- `--include-frc` is the PRODUCTION DEFAULT (matches dje's recipe; the labeler is
  Chess960-safe). Omitting it DROPS all FRC games (legacy filter) — only do that
  deliberately, e.g. to reproduce the 2026-07 dev-box tactics corpora.
- Memory: the dedup dictionary is capped at 100M entries (~5GB; commit 6e221d4).
  Budget roughly 30-40GB RAM for a 200M-position skip-1 run; measured throughput
  ~110K pos/s on a 24-core box (~30 min per 200M).
- Output: `TPG_<id>.tpg_setN.zst` (pure V2, 9378 B/pos — shareable with any V2
  consumer, sidecars are an optional overlay) + `TPG_<id>.tpg_setN.tgt.zst`.

**Validate before training** (non-negotiable; catches desync/orientation classes):

```
python check_survival_sidecars.py <OUT_DIR>              # single process, or
python check_survival_sidecars.py <OUT_DIR> "setN."      # one process per set, parallel
```

PASS requires: 0 empty-square mismatches, 0 king violations. Expect capture-within-8
rates ~7.5% pawns / ~10.5% pieces, side-symmetric (game corpora; puzzle corpora are
legitimately asymmetric).

**Train** (adopted env block, 2026-07-07):

```
CERES_TPG_TARGET_SIDECAR=1        # or 'auto' for mixed sidecar-less primaries
CERES_SURVIVAL_HORIZON=4          # trains K'=4 from K=8 sidecars (lossless remap)
CERES_SURVIVAL_TARGET_WEIGHT=0.6
CERES_SURVIVAL_LOSS_BUCKETS=2,4   # bounds MUST end at the configured horizon
CERES_SURVIVAL_CAPTURE_WEIGHT=4
# combined recipe (puzzle secondary, value-masked):
CERES_SECONDARY_LOSS_VALUE_MULT=0
CERES_SECONDARY_LOSS_VALUE2_MULT=0
CERES_SECONDARY_LOSS_AUX_MULT=0
```

Aux heads are single-GPU only (loud error under DDP). Expect a `SURV:` log line per
stats interval; bucket-graded accuracy ~96% at K=4 is healthy (trivial floor ~91%).
