# Training Recommendations — Value/Policy Balance in Chess Transformer Training

*CeresTrain project notes, August 2026. Distilled from the s1–s6 run series (256×10 and
384×12 nets, 0.6–1B positions per run on T80/T91 lc0 data), a systematic value-head
investigation (2026-08-06/07), source review of lczero-training (`vdapda-heads` branch),
and Kovax's live training configs. Evidence level is marked per item:*
**[confirmed]** *= measured by us with paired gates,* **[practice]** *= established
lc0-community practice verified in live configs,* **[open]** *= promising but unproven.*

Our benchmark throughout: 2000-puzzle paired gates at fixed rating bands (±3 Elo
same-gate noise, ≥10 = real effect), value tested at Nodes=1 — which evaluates **all
child positions** of each puzzle position, making it a breadth/generalization test,
not a root-eval test.

---

## 1. The central finding: value vs policy is an allocation problem

- At **matched model size and matched training positions**, our value head equals the
  reference (dje/Kovax-lineage) nets exactly — in-distribution AND out-of-distribution.
  The famous "value 150 Elo above policy" profile of those nets is **policy restraint,
  not value magic**, at any mid-training snapshot. **[confirmed]**
- Late in training the trade-off becomes **zero-sum**: policy and value improvements
  anti-correlate step for step (observed in every run of ours *and* reported for the
  reference runs). Early training is positive-sum; the fight starts when the trunk
  saturates. Total P+V (puzzle Elo) of very different recipes lands within ~20 Elo —
  same Pareto frontier, different operating point. **[confirmed]**
- Consequently: **capacity does not fix value.** A 2.5× larger net moved the operating
  point toward policy, not value (s2 vs Official). Whatever capacity you add, the
  dominant gradient eats it. **[confirmed]**
- The trunk gradient at nominal loss weights 1:1 is ~75 % policy / ~25 % value
  (1858-class CE vs 3-class CE). The reference recipes counteract this with **loss
  mass**: Kovax's live objective carries ≈ 50 % value-family weight (three value heads
  + two value-error heads + two categorical heads + depth-supervised probes) vs our
  historical ~7 %. This is the single largest recipe difference we found. **[practice]**

**Recommendation:** treat the value/policy split as a *dial you must actively set*,
and set it with many value-family loss expressions (see §3) rather than by cranking
one value weight. Judge recipes at the END of the schedule — mid-run reads are
misleading (our runs showed reference-class V−P at 100–600M and lost it all in the
final anneal; a 200M compressed-schedule test arm reproduces this collapse in ~8h
and makes a cheap testbed). **[confirmed]**

## 2. Weight averaging (EMA/SWA): the cheapest real gain we measured

- Running EMA with the lc0 formula (`ema = ema·n/(n+1) + w/(n+1)`, n capped) at
  **period 100 optimizer steps, cap n = 10**, exporting the averaged weights alongside
  the raw ones: the averaged net beat the raw net on **every metric in every rating
  band** — policy +21/+35, top-3 +36/+63, value +31/+35 (mid-run, peak LR). Pure
  Pareto gain, zero training cost. **[confirmed]**
- These exact settings (`period_steps: 100, num_averages: 10, export_swa_model: true`)
  are what Kovax runs today, and lc0's published nets are the averaged weights (the
  `-swa-` files). **[practice]**
- Post-hoc checkpoint averaging (averaging the last few saved checkpoints of a
  finished run) follows a **phase rule**: it pays (+30 Elo value, policy unchanged)
  when the endpoint is *past* its value peak (late-collapse runs), and it *hurts*
  (−36) when the endpoint IS the peak (anneal-crest runs) — the average dilutes the
  crest with weaker earlier weights. Running EMA has no such failure mode. **[confirmed]**

**Recommendation:** always train with EMA on and export both; serve the EMA net.
If you only have saved checkpoints, average them only when the run ended past its
value peak.

## 3. Build the value side out of many loss expressions

Single-target value CE saturates: with soft search-Q targets the per-position KL floors
at the teacher-noise level (~0.06–0.10 nats for us) early, after which most value
gradient is noise-fitting while the policy CE still has real signal — this is *why*
policy wins the late-phase fight. **[confirmed]** The remedies that fit the evidence:

| Expression | What it adds | Status |
|---|---|---|
| Dual/triple value heads (game result z; search Q; short-term value) | independent targets, more value gradient | **[practice]** (all at weight 1.0 in live configs; we historically ran z at 0.04) |
| Value-error (uncertainty) heads, one per value head | forces error-awareness; enables optimistic policy | **[practice]** (weight 1.0–10) |
| Categorical/HL-Gauss value (bucketed q with Gaussian-smoothed targets) | fine-grained value *resolution* the 3-class CE never demands; CE conditioning | **[practice]** (0.1) / our implementation **[open]** |
| Deep supervision: linear WDL probes on every trunk depth | value gradient into every layer, not just the top | **[practice]** (0.1 per probe) |
| Short-term **blunder-censored** value target (backward EMA of root_q, θ = 5/6, carry reset at deblunder triggers) | much cleaner label than game outcome | **[practice]** (V7 rescorer field) |
| Mirror-consistency regularizer (KL between value of position and its a↔h mirror, castling-free positions only) | label-free variance reduction; our best OOD-value effect early-run (+43..+76 OOD at 100M) | **[confirmed early / open late]** |
| Hard-position replay (online buffer of highest-value-KL rows, reinjected with bounded reuse + continuous churn) | extra optimization passes exactly where value fails; adaptive to the live net | **[open]** (mechanism validated, Elo pending) |

Things that did **not** work for us — equally worth knowing:

- **Value-head-only fine-tuning of a finished net: zero effect.** The head is small;
  the ceiling is the trunk representation. **[confirmed]**
- **Cold head-LR under a sane trunk LR (ratio ~0.25 at 8e-4): no effect at all**
  (indistinguishable from baseline at every checkpoint). Note Kovax runs an
  internal-Adam ratio of **1/3 under a hot 3e-3 trunk** — the ratio may only matter
  at hot LR. **[confirmed / open]**
- **In-batch pairwise ranking loss on value: learns the task, transfers nothing.**
  Ranking random positions ≠ ranking the *children* of one position, which is what
  Nodes=1 value tests (and search) actually require. Sibling-level supervision needs
  sibling-level data (per-move Q, or parent/child pairs from game trajectories). **[confirmed]**
- **Hardness-weighted (focal) value CE at γ=1 for a full run: trended slightly
  behind baseline** on value absolutes mid-run; dropped. May still have a role late
  in the schedule only. **[confirmed mid-run / open]**
- Fat smolgen / general capacity as a value fix: see §1. **[confirmed]**

## 4. Learning-rate schedule

- V−P at matched checkpoints follows trunk LR monotonically in our runs
  (2e-4 → +105, 8e-4 → +40..+115 mid-run, 2.4–3e-3 → +3..+13): hotter trunks buy
  policy speed and starve the value profile. **[confirmed]**
- The reference schedule has **no peak plateau at all**: linear warmup (~4000 steps at
  batch 4096) then **cosine over the entire run** to a floor of peak/10. Long runs
  (≈4B positions) mean the decay is slow *in positions* — value, the slow-maturing
  head (its test requires every child position to be decently evaluated → it buys
  skill with volume), gets paid at every LR level. Our hold-then-drop compressed
  anneals produced a late policy sprint that converted the entire mid-run value
  surplus into policy. **[practice + confirmed-symptom; head-to-head open]**
- Muon specifics from live configs: ns_steps 5, momentum 0.95, **weight_decay 0.01**
  (from-scratch norm-growth insurance), `adaptive: false` (breaks Muon's LR
  calibration), internal-Adam beta2 0.98 / eps 1e-7, internal-Adam LR = trunk/3.
  **[practice]**

**Recommendation:** for value-critical production runs, prefer long cosine-to-floor
schedules over hold-then-drop; if you must compress, expect the policy sprint and
counter it (EMA export, and consider policy restraint late — soft-policy aux is the
anti-over-peaking tool: KataGo's auxiliary soft policy target, weight 8.0 at T=4 in
live lc0 configs).

## 5. Mechanisms with serving impact zero (train-only heads + export blends)

All auxiliary heads above can be training-only (stripped from the export graph):
deep-supervision probes, categorical value, soft policy, optimistic policy, replay,
mirror — **none cost a single serving FLOP**. Two exceptions worth having as
*optional* export features:

- **Optimistic policy** (per-sample policy CE weighted `sigmoid((z−2)·3)` with
  `z = (target_q − q_pred)/σ`, σ from the value-error head): lc0 *serves* this head.
  We recommend implementing it as a separate head plus an export-time **logit blend**
  `(1−λ)·vanilla + λ·optimistic` so λ can be swept from the same checkpoint at zero
  training cost. Same pattern works for the soft-policy head. **[practice / our blends open]**
- A hysteresis controller ("thermostat") makes expensive regularizers affordable:
  enforce every step only while the measured effect-size is above a threshold, drop
  to sparse probing when it is quiet, auto-reactivate on drift. Measured for the
  mirror term: symmetry drifts 20× within ~25k positions in a young net without
  enforcement, but a mature net stays quiet for long stretches — the controller's
  duty cycle *is* the measurement of how much maintenance the net needs. **[confirmed]**

## 6. Measurement hygiene (the part that saved us repeatedly)

- **Pair everything.** Same puzzle set, same harness binary, same rating band; then
  same-gate noise is ±3 Elo and single-digit effects are readable. Never compare
  Perf numbers across rating bands. **[confirmed]**
- **Compare value ABSOLUTES at matched positions, not V−P**: V−P moves when *either*
  head moves; most of our "value crises" were policy surpluses. **[confirmed]**
- **Puzzle Perf is not a tournament proxy** — validate with Nodes=1 head-to-head
  before shipping. (Three separate incidents where a big puzzle edge lost the match.)
  **[confirmed]**
- Keep an out-of-distribution rating band **below your training-puzzle floor** in
  every gate: it is where generalization effects (mirror, replay) show first and
  where in-distribution contamination cannot reach. **[confirmed]**
- When a mechanism changes the *composition* of the training batch (replay injection)
  or the *definition* of a logged loss (focal weighting, EMA-labeled exports), log
  the affected metrics separately (fresh vs replayed rows) or expect to misread your
  own TensorBoard. **[confirmed]**
- Late-phase gains on a 0.1–1 % *repeated* puzzle stream can be replay-overfit
  inflating puzzle-policy numbers (we measured a 2 %-stream net that gained +396
  puzzle Elo and lost the tournament). Keep such streams at ≤0.1 % and read their
  cells skeptically. **[confirmed]**

## 7. Data-side notes

- Deblunder parameters in live use: trigger threshold q≈0.10–0.20, width 0.05–0.06,
  unintended-blunder 0.30; short-term value EMA θ = 5/6; syzygy WDL+DTZ rescore.
  Provenance labels (original / tablebase / deblunder-noise / deblunder-unintended)
  enable per-sample value-loss weighting. **[practice]**
- Draw downsampling (keep-probability on draw games) sharpens the value signal per
  position; lc0's pipeline instead offers diff-focus weighting (upweight positions
  where search disagreed with the static eval — data-recorded hardness, the static
  cousin of our live-net replay buffer). **[open / practice]**
- If your loader supports mixed corpora with optional sidecar targets, make every
  hardness/replay mechanism key-set aware — injecting a row into a batch with
  different target columns silently mislabels the auxiliaries. **[confirmed the hard way]**

---

*Everything in our stack described here is config-driven (one JSON defines the run,
including an `Env{}` escape hatch for legacy env-gated switches) so that resumes and
re-exports reproduce the training environment exactly. We recommend that property to
anyone maintaining a long-lived training stack: our worst debugging days all started
with an environment variable that did not travel with the checkpoint.*
