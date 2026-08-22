# Server recipe: 512-16 + dual-plane deg4 (best configuration as of 2026-08-20)

Distillate of the dp2 campaign's best validated configuration, adapted to the
server's 512-16 net (4xA100, DDP). Source evidence: the dpdeg series + the
mirror matrix @5M (`F:\cout\puzzle_tacs_*_gate.log`) and EB cmp measurements
(`cmp_toolbox` / `cmp_*_20s.log`). Everything below was trained/measured
locally on 256-10 and 512-15; the code is DDP-validated.

## Why this recipe

| Finding | Evidence |
|---|---|
| RelDegrees = replicated policy lift | +52/+29 and +22/+16 in-dist (2 seeds), 0 % cost |
| P-plane depth = generalisation lever | OOD-pT3 −13 → +30 → +49 for 2→3→4 blocks (monotone) |
| 5th block gives in-dist only (memorisation) | +75 in-dist, OOD flat → 4 blocks is the right depth |
| The P-plane is ~free at 512 scale | EB cmp both orders: no measurable EPS cost (512-15) |
| deg3 survives anneal better (less OOD loss) | deg310 vs dp210 control @10M: +71/+59 in-dist, +19/+8 OOD |

## Net config (ceres_net.json) — COMPLETE chassis, not just the delta

IMPORTANT (learned 2026-08-20 evening): the first version of this file listed
only the dual-plane delta — the server then inherited its own defaults for
everything else (RPE off, among others). Below is the WHOLE chassis, adapted
to 512-16. Copy the block as it stands.

⚠ ONE DEVIATION FROM THE MEASUREMENT CHASSIS: `UseRPE` read `true` here until
08-22, because ALL local ablations were measured with RPE on. It is now
`false` — RPE is dropped for production (see the cost measurement below). So
this block describes the RUN chassis for the NEXT RUN, not the measurement
chassis; the difference is this one key, and it is a known confounder on the
deg4 deltas (see below).

```json
{
    "ModelDim": 512,
    "NumLayers": 16,
    "NumHeads": 8,
    "PreNorm": false,
    "NormType": "RMSNorm",
    "FFNMultiplier": 6,
    "FFNActivationType": "Mish",
    "HeadsActivationType": "Mish",
    "NonLinearAttention": true,
    "SoftCapCutoff": 100,
    "SmolgenDimPerSquare": 32,
    "SmolgenDim": 256,
    "SmolgenToHeadDivisor": 4,
    "SmolgenActivationType": "Swish",
    "UseRPE": false,
    "UseRPE_V": false,
    "UseRoPE": false,

    "UseVisEdgeBias": false,
    "VisEdgeFamilies": "vis,xray,pinray,check,flight",
    "VisEdgeGates": "",

    "UseDualPlane": true,
    "DualPlanePolicyDecode": true,
    "DualPlaneRelDegrees": true,
    "DualPlaneLayers": 4,
    "DualPlaneSoftMinHeads": 2
}
```

### Component importance (measured @5M, dp2 chassis, ablations dp2ns/dp2nr)

| Component | Cost of removal | Verdict |
|---|---|---|
| Smolgen | P −71, pT3 −59 in-dist; KLD 1.05→1.43 | **REQUIRED** |
| RPE | P −3 in-dist, but OOD −35/−36 | **DROPPED** (see below) |

THE RPE DECISION (final as of 2026-08-22). Three independent arguments point
the same way, and they replace the earlier "recommended, not critical / worth
trying on the server" wording:

1. **COST:** TRT serving cost measured at ~11 % EPS on 256-10 and ~10-16 % on
   512-15 — it does NOT shrink with width (details in the next-run section).
2. **GAIN uncertain:** the OOD cost of removal (−35/−36) was measured in the
   puzzle regime, where the proxy has been wrong four times. In-dist it is −3.
3. **REDUNDANCY** (Kovax's bench, 08-16, read 08-22): RPE and smolgen —
   opposite mechanisms — landed within 3 Elo of each other (+42..+51 policy /
   +52..+56 value over baseline). If you are already paying for smolgen, RPE
   buys little. We keep smolgen (REQUIRED above) and drop RPE.

Precedent: v4r (2.5B, smolgen + no positional encoding = policy parity). The
P-plane also carries its own geometry (file/rank embeddings + relation biases).

⚠ KNOWN CONFOUNDER: the deg4 deltas (+49 OOD etc.) were measured WITH RPE on.
The next run goes without. The deltas are therefore not guaranteed additive
with the chassis above — read them as directional, not as a forecast.

Notes:
- `UseVisEdgeBias: false` = bare chassis. `VisEdgeFamilies` must still be set:
  dual-plane builds its 20 relation channels privately from the family list
  (`dp_private_vis`); without it ceres_net asserts at startup.
- No cf (`VisEdgeGates: ""`): gates cost 26-39 % NPS. **Strengthened 2026-08-22
  by the 7-arm ladder** (4000 samples, rg2100 OOD): `cf3g` scores 1942 policy
  against `dpdeg4`'s 1925 — a +17 gap that sits INSIDE the 14-Elo seed noise
  floor measured in the same run. The free dual-plane ladder has caught up with
  the expensive gates chassis on OOD policy, so the decision no longer rests on
  cost alone.
  ⚠ CORRECTION: the earlier "puzzle specialist" justification was WRONG and is
  withdrawn. The per-theme table refutes it — cf3g's gains are concentrated in
  the LARGEST, broadest buckets (middlegame +15.5 pp at 10.0 sigma, crushing
  +12.3 at 8.0), not in check/flight-flavoured themes. It is a generally strong
  and generally expensive mechanism. Right call, wrong reason.
- The P-plane is a fixed dp=128 / 32 tokens regardless of trunk width, so its
  cost at 512-16 measures ~0 % (EB cmp, both orders, 20 s from startpos).
- Zero-init contract: every inject/decode coupling starts as an exact no-op —
  warm-starting from an existing 512-16 checkpoint is safe (the dual-plane
  family is in `_AUX_HEAD_PREFIXES` and is fresh-initialised when resuming
  from a non-dp checkpoint).

## Opt config — recommendations

```json
{
    "LearningRateBase": 0.0005
}
```

- **LR 5e-4 (REVISED 08-22 together with WSD), not 6e-4 and certainly not
  1.2e-3.** The basis for 6e-4 still holds: (1) the sqrt width rule from
  8e-4@256 gives ~6e-4@512; (2) the QK-clip analysis showed heavy clipping on
  the 512 net at 1.2e-3 — clipping is the symptom of an LR too hot for the
  width. The 200-500M value collapse on the server baseline is consistent with
  the same diagnosis.
- **Why lower to 5e-4 once WSD is on:** the schedule shape changes the
  TIME-INTEGRATED LR exposure, not just the peak. Mean multiplier over the run:
  cosine-from-step-0 with floor 0.1 gives 0.1 + 0.9·0.5 = **0.55**, whereas WSD
  0.8/0.2 gives 0.8·1.0 + 0.2·0.55 = **0.91**. The same peak would therefore
  deliver **1.65× higher average LR** than anything we have run. The net now
  sits 800M positions at FULL peak — a longer continuous hot exposure than any
  previous run. 5e-4 only partly compensates (5e-4·0.91 = 4.55e-4 against
  6e-4·0.55 = 3.3e-4, i.e. still **~1.4× hotter on average**) — which is the
  intent: the point of WSD is more time at high LR, just not as much as the
  naive substitution would give. The risk is asymmetric: too cold costs a
  little learning speed, too hot gives QK clipping and the instability we have
  already measured on 512.
- **NO FileMirrorAug** (user decision, correct): the benefit is
  anti-memorisation in limited-data regimes; the T80 corpus (4B pos) does not
  have the memorisation problem, the qualifying fraction is low on game data
  (castling rights), and mirroring on game data is untested at scale.
- `BatchSizeForwardPass` is GLOBAL and split across ranks under DDP — use the
  same value as the existing server configs, do not multiply it per GPU.

## Split-LR status (as of 2026-08-20)

Implemented and correctness-validated, NOT effect-validated:

- Code: `muon.py` (`lr_ratios` per param), `train.py` (FAMILY-LR block with a
  membership dump), config keys `Opt_LearningRateHeadsRatio` /
  `Opt_LearningRateCouplingsRatio`, `MuonAdamWScope: "ffn-only"` (the Kovax
  partition, AdamW for attention).
- Validation: `tools/splitlr_smoke.py` green; micro-run slr0 trained
  end-to-end including checkpoint resume, banner confirmed
  (`FAMILY-LR: heads ratio=0.333 (32 params), couplings ratio=2.0 (28 params)`).
- Effect: the H1-H4 arms (fixed-budget 150M, plan in
  `F:\cout\Findings\split_lr_plan_2026-08.md`) have NOT been run yet.

**Recommendation for the first server run: leave split-LR OFF** (omit the
keys). One variable at a time — first the dual-plane delta against the 512-16
baseline you already have, then optionally H1 (heads ratio 1/3) as the next arm
once that delta is known.

## DirectFromV6: train straight from LC0 tars/chunks — zero TPG storage (2026-08-21)

For servers without disk space for TPG shards, the data config

```json
{
    "SourceType": "DirectFromV6",
    "TrainingFilesDirectory": "/path/to/tars-or-chunks",
    "V6SkipCount": 30,
    "V6ShufflePool": 50000
}
```

reads LC0 v6 AND v7 records directly — both loose `.gz` chunks and UNPACKED
`.tar` archives (members are read in place via seek+read; the index is cached
as `<tar>.chunkindex.npz` when the directory is writable). The v7 tail feeds
the trainer's v7x consumers (censored q/d + z-provenance) with no sidecar files.

Requirements / choices:

- `AuxFeaturesPerSquare: 0` (137-channel model; the qz ablation showed aux
  neutrality)
- `LossQDeviationMultiplier: 0` (the field does not exist in v6)
- `pip install isal` (2-3× faster gunzip; used automatically if installed)
- `CERES_NUM_DATASET_WORKERS` by core count: ~2.2M pos/h/worker (loose files,
  native fs) and up to ~19M pos/h/worker DIRECTLY FROM TAR (one file handle, no
  per-file open cost — the fastest path, measured 2026-08-21)
- `V6ShufflePool`: RAM = pool × workers × 8.4 KB (200k × 8 = 13 GB OOM'd
  locally; 50k is a safe default)
- No rescore/deblunder in the path: use pre-rescored tars
- Skipping is RANDOM per pass (the whole corpus is reached across epochs,
  unlike gen-tpg's permanent selection)

Validated end-to-end locally 2026-08-21: startpos anchor + field sanity (v6 and
v7), 5M training from raw chunks on the deg4 chassis (v6smk: value 1972/1719 =
the deg4mx class at the same LR phase; policy limited by the one-day corpus as
expected), ORT load green, gate run.

## Value oscillation (observed 400M-1B on the first 512-16 deg4 run) — diagnosis and cure

**Symptom:** EB value alternates with a ~100M period from 400M (500M down, 600M
up, ...); TB shows value/value2/unc oscillating while policy and qk-clip are
healthy. The baseline without dual-plane: worse level but more stable.

**Diagnosis (two components that compose):**

1. THE SHARED-POOL MECHANISM: the P-plane (~500k params) is shared by the
   policy decode (strong, committing gradients) and the value injects (weak).
   Policy continuously reshapes the representation value reads → dp gives a
   better value LEVEL but higher variance than baseline (trunk-only value =
   sluggish but stable). That value2 oscillates in lockstep despite its 0.04
   weight proves this is representation-driven, not loss competition.
2. Era non-stationarity in the corpus (T91: draw rate / endgame density / eval
   sharpness drift with tar date; shard-granular reading) produces the ~100M
   periodicity. DirectFromV6 mixes at game level and removes this component in
   later runs.

**Cure (surgically validated locally, commit 18e5b84):**

```json
net-config:  "DualPlanePolicyGradScale": 0.25   ⚠ SUPERSEDED 08-22 -> 1.0
opt-config:  "LossValueMultiplier": 2.0         ⚠ see the 08-22 revision below
```

⚠ **THE GRAD-SCALE VALUE IS SUPERSEDED** by the 2026-08-22 revision further
down ("the loss-mass accounting" and the correction after it). In short:
value 2.0 delivered only 23 % of the objective because the opp-policy head ate
the step, and grad-scale 0.25 throttled the ONLY dual-plane mechanism that has
replicated (policy). The diagnosis below still stands; it is the PRESCRIPTION
that changed. Use the block in the next-run section.

- Grad-scale: forward IDENTITY (zero inference cost, bit-identical function);
  scales only the policy decode's gradients INTO the shared P-tokens (verified
  exactly: P-plane grad × α, decode weights × 1.0, value grad × 1.0). Damps
  policy's reshaping power over the pool; the 5M smoke showed a moderate early
  policy lag (−52 P) as expected when the fastest learner is damped — the
  claimed gain was stability at 400M+ scale.
- Value mass 2.0: Kovax adoption #1 (his objective is ~50 % value; ours was
  ~20-25 %) — addresses the trunk moving for policy at value's expense.
- Survival anchor: DROPPED on the server (decided 2026-08-21): the labels exist
  only as TPG sidecars, and the server's data path is chunks/tars (v6/v7) which
  do not carry them — the preflight refuses the combination by design. The
  value family's fallback is therefore the value-private P-block alone.

**Recommended design (decided 2026-08-21 evening): a FRESH 1B run, not an
extension arm.** Rationale: (1) a resume tail lowers LR at the same time as the
stabilisers are switched on — attribution drowns in the s3 law (value
consolidates at low LR regardless); (2) the value representation is FORMED in
the 100-400M window — the stabilisers must protect the formation phase, not
just repair the tail; (3) the loss-mass rebalance (value 23 % → parity, see the
08-22 revision) changes the objective itself — switching it on mid-run loses
attribution for the same reason as (1).

~~(3) RPE belongs from step 0 and cannot be warm-started in.~~ VOID as of
08-22: RPE is dropped, so the argument is empty. Note also that the aux heads
(oppp/action) CAN now be warm-started — `action_head.` entered
`_AUX_HEAD_PREFIXES` in `1e77d15` — so an extension arm is technically
possible if it is ever wanted. The fresh-run decision rests on (1)-(3) above,
not on a technical blocker.

## 🅰️🅱️ THE NEXT RUN IS AN A/B: 2 x 2.5B, `srv_512_16_t91ab_{ctrl,ffn}`

Committed configs, ready to launch. This supersedes the single-5B-run framing
below — the knob rationale in every other section still applies verbatim.

| | arm A `ctrl` | arm B `ffn` |
|---|---|---|
| `MuonAdamWScope` | `all-non-trunk` | **`ffn-only`** |
| Muon / AdamW params | 120 / 202 | **20 / 302** |
| GPUs | 0,1 | 2,3 (NVLink pairs) |
| positions | 2.5B | 2.5B |
| everything else | — | **byte-identical, same seeds** |

Verified programmatically: the opt configs differ in `MuonAdamWScope` alone,
and the net and data configs not at all.

⚠ **THIS IS NOT "SPLIT-LR".** `LearningRateBase` is 0.0005 in BOTH arms and
`LearningRateBaseHeads` / `LearningRateHeadsRatio` / `LearningRateCouplingsRatio`
are all unset. The variable is which OPTIMIZER each parameter gets, not what
learning rate it gets. The name has caused confusion twice; keep them apart.

**What `ffn-only` does:** Muon keeps only the FFN linears; attention qkv/proj
and smolgen move to the internal AdamW. That is the Kovax partition ("NAdam for
attention, Muon for FFN", our AdamW standing in for his NAdam), and it acts on
the QK-clip evidence — the load-bearing choice is taking attention matrices out
of Muon's orthogonalization. Note how drastic it is: Muon drops to 20 params.

⚠ Scope naming is inconsistent in the code: `all-non-trunk` and `final-only`
describe what **AdamW** gets, while `ffn-only` describes what **Muon** gets.
Read `train.py:616-648` before assuming.

**Why an A/B rather than one 5B run.** The dev box's finding #5 ruled out LR
LEVEL as the value-freeze lever (two arms 2x apart froze at the same onset; a
5x in-run cut did not help), and moved the weight to the partition. Spending
our largest compute on a run that tests nothing on the value axis would leave
us in the same place a week later. 2.5B is still 2.5x any previous production
run and 12x the ~200M floor for a readable value verdict.

⚠ `MuonAdamWScope` CANNOT be changed mid-run: the optimizer-state guard resets
momentum when the partition differs from the checkpoint. It must be set from
step 0 — another reason this is an A/B rather than a mid-run switch.

**Local de-risking done:** `ffn-only` smoked clean on T91 TPG at 1M (within
noise of control); a separate arm at Muon 5e-4 / AdamW 8e-4 confirmed the
partition tolerates 1.6x on attention with only sporadic QK clipping (1 head),
so 5e-4 is not handicapping arm B. Resume correctness verified end-to-end.

**Launch:** `scripts/server/launch_tactical_ab.sh <OUT> 2` (2 GPUs per arm,
separate ports). Edit the data path in BOTH data configs first, and check
shards >= 2 ranks x workers per arm before choosing worker count.

**If arm B wins**, the validated follow-on is raising the AdamW-branch LR
(`LearningRateBaseHeads`) — already shown safe, not folded in here because it
would have broken the single-key design.

## 🔄 DATA-PATH CHANGE 2026-08-22 EVENING: Hetzner **T91 TPG**, not v7 chunks

The next run will train on T91 **TPG shards** being downloaded to the server,
not on v7 chunks via DirectFromV6. TPG records do NOT carry the v7 tail, so two
features built this week have to be switched OFF. The block below supersedes the
DirectFromV6 package that follows it.

```json
data-config: "SourceType": "DirectFromPositionGenerator",   (the TPG path; this is the default)
             "TrainingFilesDirectory": "/path/to/t91_tpg",
             "FractionQ": 1, "WDLLabelSmoothing": 0,
             "NumTPGFilesToSkip": 0
             // V6SkipCount / V6ShufflePool / V6MaxResultQDelta are INERT here — delete them
net-config:  unchanged (chassis above, DualPlanePolicyGradScale 1.0)
opt-config:  "LearningRateBase": 0.0005,
             "LRBeginDecayAtFractionComplete": 0.8,
             "LRDecayShape": "cosine", "LRMinFactor": 0.1,
             "LossValueMultiplier": 2.0,
             "LossOppPolicyMultiplier": 0,        ← WAS 0.02, MUST be 0 on TPG
             "LossActionPlayedMultiplier": 0,     ← WAS 0.1,  MUST be 0 on TPG
             "AuxFeaturesPerSquare": 0,
             "LossQDeviationMultiplier": 0        (see note 4)
env:         CERES_SHUFFLE_SEED=<fixed>, CERES_NUM_DATASET_WORKERS per note 3
             CERES_DDP_STATIC_GRAPH is NO LONGER REQUIRED (note 2) but is harmless
```

**1. Why oppp and action must be 0.** Both targets live in the v7 record tail
(`OppPlayedIndex`, `QAfterPlayedMove`) which only DirectFromV6 reads in-band.
The TPG `.v7x` sidecar carries only the three-field triple
(`cens_q`, `cens_d`, `prov`) — see the `V7Extras` contract in `tpg_dataset.py`,
whose last four fields default to `None` on the sidecar path. There is no
sidecar format that supplies them. The preflight added in `1e77d15` raises at
startup ("requires SourceType DirectFromV6 with a v7 corpus") rather than
training a head with zero supervision, so a mistake here fails loudly — but set
them to 0 and the run just starts.

**2. `CERES_DDP_STATIC_GRAPH` requirement lapses with them.** The guard fires on
stash-only heads; with oppp off and no survival/stvalue/soft/HLG/vc/vda/refiner
enabled, none are active. Keeping the env var set costs nothing and guards
against re-enabling something later, so leaving it in the launch script is the
safer habit.

**3. Shard count is now a hard launch constraint.** The TPG preflight requires
**shards >= nproc x workers, PER corpus**. With 4 ranks and 4 workers that is 16
shards. Count the downloaded shards before choosing `CERES_NUM_DATASET_WORKERS`;
this has bitten us before (the 5-shard puzzle corpus).

**4. `LossQDeviationMultiplier` becomes available again.** It was 0 only because
the field does not exist in v6 records; TPG V2/V3 carries it. Leaving it at 0
for now — one variable at a time — but it is no longer a forced zero, so this is
a candidate for a later arm.

**5. Survival anchor: still OFF, and still for a data reason.** The extraction
DOES carry survival K=8 sidecars — but they exist **only on the local training
box**, never uploaded to the storage box (user, 2026-08-22). The server sees
the `.zst` data shards without their `.tgt.zst` twins, so
`CERES_TPG_TARGET_SIDECAR` must stay off and `survival_target_weight` 0.
Should survival ever become interesting it is a cheap fix: the sidecars are
~680 MB per shard, so ~9.5 GB for all 14 — trivial next to the 322 GiB of data
— but they exist only for the `TPG_88552` extraction, so a second extraction's
shards could not be covered without regenerating.

### ✅ THE ERA REGRESSION DOES NOT APPLY — checked, 2026-08-22

The value-oscillation diagnosis names TWO composing causes, and cause 2 is era
non-stationarity read at **shard granularity**. DirectFromV6's game-level mixing
was the stated mitigation, so dropping back to TPG looked like it would
re-introduce the problem. **It does not.** Verified against the actual
generation options of the local copy of the same extraction
(`E:\T91_survival_tpg\TPG_88552.tpg.options.txt`):

- **1265 source tars, fed in RANDOMISED order.** The date sequence in the
  options file jumps: 2026-03-31, 03-18, 05-01, 03-09, 03-12, 04-30, 03-20,
  03-11, 04-02, 03-12, 04-01, 03-25 — not chronological.
- gen-tpg additionally scatters consecutive positions across worker threads and
  output sets (the same property that makes TPG unusable for ply-order
  analysis).

Together those mean **every output shard already draws from the whole date
range**, so the era component is mixed away at GENERATION time rather than at
read time. Shard-granular reading is therefore harmless for this corpus.

⚠ This is a property of THIS extraction, not of TPG in general. A corpus built
from a chronologically-ordered tar list would have the problem. Check the
`.options.txt` before assuming it holds for a different shard set.

Extraction parameters for the record: `PositionSkipCount: 20`, ~1.0B positions,
all-variants (standard + FRC), survival K=8 sidecars present. Note the skip is
already baked into the shards — `NumTPGFilesToSkip` in the data config is a
different knob (it skips whole shards) and stays 0.

Cause 1 (shared-pool) is unaffected either way: grad-scale 1.0 stands on the
three independent arguments above, none of which involve the corpus.

---

The config package below is the SUPERSEDED DirectFromV6 variant, kept because
its rationale for every non-data knob (LR, WSD, grad-scale, value mass, RPE)
still applies verbatim.

The config package for the fresh run (every point backed by measured evidence).
DATA PATH: DirectFromV6 on the v7 corpora directly (NOT TPG) — the new v7
fields (OppPlayedIndex and others) exist only there, and game-level mixing
counteracts the era component of the oscillation. Multiple directories are
separated by ';' — mixing is volume-proportional (cv2 ~1/3 of T91 ⇒ ~25 % of
the stream):

```json
data-config: "SourceType": "DirectFromV6",
             "TrainingFilesDirectory": "/path/t91_v7_op1;/path/cv2",
             "V6SkipCount": 30, "V6ShufflePool": 50000,
             "V6MaxResultQDelta": 1.2
net-config:  "DualPlanePolicyGradScale": 1.0          (REVISED 08-22, see below)
opt-config:  "LearningRateBase": 0.0005,              (REVISED 08-22 with WSD)
             "LossValueMultiplier": 2.0,              (5.0 was an OVERSHOOT —
                                                       gradient probe, see below)
             "LossOppPolicyMultiplier": 0.02,         (REVISED 08-22)
             "LossActionPlayedMultiplier": 0.1,       (corpus-dependent, see below)
             "LRBeginDecayAtFractionComplete": 0.8,   (WSD — NEW 08-22)
             "LRDecayShape": "cosine", "LRMinFactor": 0.1,
             "AuxFeaturesPerSquare": 0, "LossQDeviationMultiplier": 0
env:         CERES_SHUFFLE_SEED=<fixed number>, CERES_NUM_DATASET_WORKERS=4 per rank,
             CERES_DDP_STATIC_GRAPH=1  (REQUIRED with LossOppPolicyMultiplier>0
             under torchrun: the oppp head is stash-only and invisible to DDP's
             default reducer; the guard in train.py fails loudly without it.
             Review 2026-08-21 findings 1+2 — participation terms are in, so
             mixed / all-−1 batches are safe.)
```

### 📉 WSD SCHEDULE (new 08-22): hold the peak, then decay

`LRBeginDecayAtFractionComplete: 0.8` — ONE number, zero code. The LR lambda
(`train.py:763-787`) IS already a WSD scheduler: warmup → `return 1.0` while
`fraction_complete < FRAC_START_DECAY` → half-cosine to MIN_LR. We have been
running 0.0, which the config comment itself describes as "cosine from step 0,
no hold". **0.8 (user's choice, 08-22)** holds to ~800M and then cosines down
to 0.1× peak over the last ~200M (~48,800 of ~244,000 steps).

Why:

1. Our own s3 law — "value only improves under falling LR". Cosine from step 0
   smears that signal thinly across the whole run; WSD concentrates it.
2. The dev box's finding #5 rules out the LR LEVEL as the freeze lever (two
   arms, 0.0012 and 0.0006, identical freeze onset) — the shape is the untested
   candidate.
3. Kovax runs hold-then-drop in production.

Note this is a DELIBERATE deviation from Kovax's 90/10: we give twice the decay
window. The reason is the s3 law again — value consolidates under falling LR,
and value is our weak point, so the window where that happens is worth more to
us than the last 100M at peak.

⚠ His peak of 1e-3 is a NAdam LR and ours is Muon — adopt the SHAPE, not the
number (and see the LR section above for why the peak drops to 5e-4).
⚠ Warmup is untouched (ours is 5 % = 50M at 1B; his ~1.6 %) — one variable at
a time.

Bonus: WSD makes later fixed-budget arms practical (hold one trunk, branch
decays off arbitrary points).

### ⚖️ REVISION 2026-08-22: the loss-mass accounting — both knobs pointed the wrong way

Decomposed the objective in the 1M smoke (actsmk3) against the multipliers. The
sum hit the logged `total_loss` of 1.9588 EXACTLY, so this is not approximate:

| term | contribution | share |
|---|---|---|
| policy | 0.964 | 49 % |
| **opp-policy** | **0.495** | **25 %** |
| value + value2 | 0.450 | 23 % |
| action | 0.036 | 2 % |

**The opp-policy head became the second largest term in the whole objective and
contributed more than the ENTIRE value family** — because its CE sits at ~5.0
while value loss sits at 0.21, so "weight 0.1" means something very different
from what it looks like. Net, the objective was ~75 % policy-flavoured against
23 % value, while Kovax runs ~50 % value (adoption #1). We thought
`LossValueMultiplier` 2.0 was a step toward that, but the aux head ate the
entire step.

At the same time grad-scale 0.25 throttled the policy decode — the ONLY
dual-plane mechanism that has replicated (A3: +110-127 OOD policy across
seeds/GPUs). The value synergy NEVER replicated (dp2r 14 < dp1f 20 = seed
variance). So we were sacrificing a proven policy gain for a value gain that
does not exist, while simultaneously underweighting value.

**Revised position (user-approved 08-22):** treat the P-plane as what the
evidence says it is — a POLICY mechanism (grad-scale 1.0) — and fix value with
MASS, not with damping.

### 🔴 CORRECTION THE SAME EVENING (dev box measurements, FINDINGS_2026-08-21_22.md)

The loss-mass accounting above is arithmetically right, but **loss share is the
wrong proxy for what shapes the trunk**. A direct gradient measurement on
deg4@100M (143 samples) is the authority:

| group | cos(g_policy, g_value) | \|g_v\|/\|g_p\| |
|---|---|---|
| trunk | **+0.0703** (~550 sigma POSITIVE) | **0.556** |

Two consequences, both against what I proposed:

1. **`LossValueMultiplier` 5.0 is an OVERSHOOT.** At mult 1.0 value already
   accounts for 36 % of trunk gradient magnitude; mult **2.0 gives roughly
   PARITY**. Mult 5.0 would give a ratio of 2.78, i.e. value at ~74 % of the
   trunk gradient — a far heavier tilt than the Kovax parity we were after.
   **Set 2.0, not 5.0.** (Loss CE and gradient magnitude are not coupled:
   3-class value CE and 1858-class policy CE scale differently.)
2. **The grad-scale premise is FALSIFIED.** The knob was built on the
   tug-of-war hypothesis. The cosine is +0.0703 at ~550 sigma — the objectives
   pull the trunk in COMPATIBLE directions, with 1 % negative samples. There is
   no tug-of-war. Grad-scale 1.0 is therefore doubly confirmed, and the dial
   should be RETIRED, not explored further at 0.5/0.75 — there is no theory
   left behind it.

**A third, independent argument arrived the same day (knockout on v7@800M, run
on the training box; full numbers in
`C:\Users\Navn\OneDrive\Chess\Kovax\SHARED_NOTES.md`):** zeroing the policy
decode leaves the net within 0.35 % of maximum entropy over legal moves, with
logit sd down to 8.1 % of control and cross-position logit correlation rising
from +0.129 to +0.591 — i.e. the base policy head has degenerated into a fixed
per-move prior. **Grad-scale 0.25 did not prevent that capture**; it only made
the plane learn policy 4× slower. Value, by contrast, is merely SHARED (55 % of
its q spread survives the value knockout). So the throttle handicapped the
pathway policy actually depends on, while the value pathway it was meant to
protect was never captured in the same sense.

The part of the loss-mass accounting that SURVIVES: opp-policy at 0.1 really
did take 25 % of the loss mass, which nobody intended. Its gradient share is
unmeasured, so **0.02 stands** as a cautious value until someone measures it.

**Current values for the NEXT RUN: grad-scale 1.0, LossValueMultiplier 2.0,
LossOppPolicyMultiplier 0.02, LearningRateBase 5e-4, WSD hold 0.8.**

- Opponent-policy aux (`LossOppPolicyMultiplier` **0.02**): the Monroe idea
  (+5 % at LC0), target = v7 OppPlayedIndex (99 % populated in both corpora),
  training-only head, −1 masked. Validated: unit smoke + 1M integrated training
  (opp_policy_loss 6.34 → 4.80; random baseline 7.5). The weight was REDUCED
  from 0.1 to 0.02 per the mass accounting above — 0.1 gave it 25 % of the
  objective.
- Grad-scale: surgically verified (P-plane grad × α, decode weights × 1.0,
  value grad × 1.0), zero inference cost. **Set to 1.0 in the next run** (i.e.
  the mechanism is inactive); the knob is kept for a later A/B once the value
  mass is settled.
- RPE: DROPPED after cost measurement (2026-08-21 evening): TRT serving cost
  measured at ~11 % EPS on 256-10 (dp2 vs dp2nr, both orders) and ~10-16 % on
  512-15 (rpe512 probe vs deg3-512) — the cost does NOT shrink with width.
  Against +15-36 OOD policy (Nodes=1, uncertain transfer to play) the net is
  marginal to negative at fixed time. Archived as measured-and-rejected for
  serving nets.
- Value mass 2.0: Kovax adoption #1.
- Survival anchor: NOT in this run (user decision) — remains a later candidate.
- Data path (revised 2026-08-21): DirectFromV6 on v7 DIRECTLY (no TPG) — user
  decision; the new v7 fields (OppPlayedIndex, QAfterPlayedMove) exist only
  there. Multi-root: cv2 + T91-v7 in one `TrainingFilesDirectory`
  (';'-separated, volume-proportional mixing). The whole package was
  1M-smoke-tested together (run2pkg + actsmk variants).

FREE DIAGNOSTIC BEFORE STARTING: run EB on the 900M and 1B checkpoints from the
old run — their own cosine is already down at ~1e-4 there. If value still
alternates, low LR alone is proven insufficient (the stabilisers are
necessary); if it settles on its own, the diagnosis is more LR-coupled than
shared-pool-coupled. One EB run, high information value.

Success criterion for the fresh run: the EB value curve 400M-1B flat or
monotone (no level alternation), TB value/value2/unc without a sawtooth.
Fallback if the damping is not enough: a value-private P-block (fork after the
last shared block, value gradients only).

### The action head — built and validated 2026-08-21

- ⚠ **CORPUS-DEPENDENT. V7 FORMAT DOES NOT GUARANTEE THE FIELD IS FILLED.**
  Measured 08-22: t91_v7_op1 and cv2 are 99 % populated, but **`/mnt/t82data`
  (server5, 605 GiB / ~27M games) reports `action(q_after) populated=0.0 %`** —
  v7-formatted, field never written. There `LossActionPlayedMultiplier` MUST be
  0 (`LossOppPolicyMultiplier` is fine at 99.1 %). The preflight catches this
  automatically (raises below 5 %), and the startup line
  `[v6_dataset] corpus diagnosis ... action(q_after) populated=X%` is the
  authority. READ THAT LINE before trusting the head.
- `LossActionPlayedMultiplier` (opt config, recommended 0.1): the action head
  ([B,1858,3] WDL per move) gets a masked soft-CE at the PLAYED move slot
  against the WDL built from v7 q/d-after-played (verified `q_after ==
  -next.best_q` to MAE 0.000). 1M smoke: `action_played_loss` 0.918 → 0.834,
  policy/value unaffected.
- ⚠ TEMPERED EXPECTATION (dev box finding #3, 08-22): `q_after_played(t) =
  −best_q(t+1)`, and adjacent plies correlate ~0.99 in magnitude once the
  perspective flip is corrected. So the action target is **~99 % redundant with
  the value target we already train on**. What is new is the ROUTING (per-move
  structure), not the information. The bet is that the routing forces a better
  representation — a weaker bet than new signal.
- The head is EXPORTED as the ONNX output `action` — Ceres TRT detects it by
  name and MCTS consumes it via `ActionWDLForMove`.
- TRT cost MEASURED (EB cmp actsmk vs run2pkg, both orders): ~9-12 % EPS. The
  cost is OPTIONAL at serving: train with the head (value signal into the
  trunk), strip it from the export for full speed — a per-net decision, not a
  per-run one. Strip with `CERES_EXPORT_STRIP_ACTION=1` in front of
  `recover_export` (or the training export) — verified end-to-end: a graph
  without the head (91.9 vs 97.9 MB) whose `action` output is the [B,1] alias,
  exactly as when the head is disabled.
- `action_played_loss` is logged as a true KL (target entropy subtracted, house
  convention). It reads ~0.49 lower than the raw CE did — that is the
  convention change, not a regression. Do not compare across that boundary.

## Gate rule (unchanged from the Stage C protocol)

Compare against the 512-16 baseline at the same position count; the value rule
of ≥ +30 @200M rg2700 stands. Read OOD (the rg2100/2300 class) for
generalisation verdicts and hits in the floor regime — and for policy read pT3
+ the KLD matrix, not just P.

⚠ **DO NOT CONFLATE TWO DIFFERENT RULES** (I mixed them up on 08-22, the user
corrected it):

- **Puzzle-only SMOKES (5-10M, puzzle as the TRAINING corpus):** value is NOT
  read. There the value labels are puzzle outcomes, and the user's rule is "if
  it lifts policy, we are happy". This applies to our ablations, not to
  production runs.
- **1B runs on a GAME corpus, evaluated ON the puzzle gate:** here the value
  number is the primary yardstick, and **value IS expected to move**. The
  labels are Q/z from real games; the puzzle gate is merely the measuring
  instrument. Our entire production history is read this way (s1 V2545 @818M,
  s3b V2660 @800M = all-time, s5 V2568 @600M). A 1B run should land in that
  class or better.

### 📏 NOISE FLOOR, measured 2026-08-22 (use this, not folklore)

The 7-arm ladder included `dp2` and `dp2r` — the SAME mechanism at two seeds —
so every delta between them is noise by construction. At 4000 samples, rg2100:

- **Headline policy: 14 Elo. pTop3: 12 Elo.** Seed-to-seed, across separate
  training runs. Our old "±3 on the same gate" figure was a REPEAT-measurement
  floor (same net twice); it does not apply when comparing two trained arms.
  Several historical +15..+20 "wins" were therefore barely above noise.
- **Per-theme, the floor scales with n:** ~1.5 pp for n≈2000 themes
  (crushing +9.7 vs +8.9, middlegame +9.2 vs +7.9, endgame +8.8 vs +7.3), but
  ~5-7 pp for n≈50 (knightEndgame +15.6 vs +22.2). **Do not believe a small
  theme cell below ~10 pp.**

Consequence for the gate rule above: the "≥ +30 value @200M" bar is only ~2x
the policy noise floor. Prefer reading a TRAJECTORY over >=3 checkpoints
(`EngineBattle puzzletrend`, min-steps 3) rather than a single-point delta, and
run 4000 samples rather than 2000 where theme-level attribution is wanted.

Caveats that do NOT override the second point: rg2700 value is the floor regime
(read hits alongside the rating), noise is ±3 on the same gate while ≥10 is
real, and the final PRODUCTION verdict requires a Nodes=1 head-to-head because
puzzle is not a tournament proxy (three runs have contradicted the puzzle
numbers). But that is a verdict about PLAYING STRENGTH — it is not an argument
for declining to read value on the gate.

## ⚠ EXPORT BUG FOUND AND FIXED (b2c6af6, 2026-08-20 evening) — read before EB-testing dp nets

**Symptom:** deg4@100M from the server run "played badly" in EB gameplay. The
cause was NOT the net — the training metrics were healthy — but a corrupted
ONNX export:

- `convert_float_to_float16` (save_model) rewrites tensor types in the graph
  but leaves existing Cast nodes' `to` attribute untouched. Dual-plane's
  internal `.float()` casts (dynamo → chained `_to_copy` nodes) therefore
  produced a self-contradictory graph.
- onnxruntime REFUSES to load the file ("Type Error: Type (tensor(float16)) of
  output arg (_to_copy_1) ... does not match expected type (tensor(float))").
- TensorRT parses it anyway and builds a SUBTLY WRONG engine → a net that
  "works" but plays weakly. That is the trap: EB/TRT give no error.

**Fix (in-tree):** a reconciliation pass in `save_model.py` after the fp16
conversion — all FLOAT↔FLOAT16 Cast attributes are set to match the converted
output type. Logs `INFO: ONNX_FP16_CAST_RECONCILED N` when it contributes.
Systemic: all future module code with internal `.float()`/`.half()` is covered.

**Validation (locally, against the server checkpoint deg4@100M):** before = ORT
load fails exactly as above; after = loads cleanly, 0 NaN, and measures as the
training metrics imply (top1 54.9 % / pTop3 82.3 % / valAcc 91.6 % on
T91-skip1 — parity with 256ctrl@100M). ⇒ the gameplay verdict on deg4@100M is
INVALID; retest with the fixed export.

**Practical:**

- Running training processes hold the old `save_model` in memory: their exports
  must be re-exported with `recover_export.py <ID> <outputs_dir> <numpos>`
  after pulling. The checkpoints are healthy; no retraining.
- The loadability test is a perfect CI gate:
  `onnxruntime.InferenceSession(path)` is green/red within a second.

**Side finding (applies without dual-plane too):** the 512 BASELINE's early
exports (100M/200M, LR 1.2e-3) are FP16-FRAGILE: strict fp16 (ORT) gives
garbage (top1 11 %!), while TRT worked because it selects fp32 where needed.
Mechanism: extreme weights/logits in the hot phase (the QK-clip regime)
saturate under fp16 conversion; from ~500M the files are healthy. The EB
numbers (TRT) stand, but do NOT use early 512@1.2e-3 ONNX files in ORT-based
measurements. deg4 (LR 0.6e-3) is healthy already at 100M — another argument
for a lower peak LR.
