# VISEDGE 200M pair — overnight run spec for the training computer

**Goal:** adjudicate the visibility edge-bias mechanism (commit `1aa4f12`) at the
horizon where value becomes readable. The 20M smoke ladder (2026-08-14, dev box)
showed structure fully consistent with the Kovax visibility program (all channel
families in active use, pinray > xray weight ranking, gates absorbing the
content-free bias) but value deltas inside single-run noise — exactly as the
source program's calibration predicts: its readable value results needed ~200M
samples. This pair is that test.

Two runs, sequential, single GPU:

| run | what | config delta vs the 20M smokes |
|---|---|---|
| `raybase200M` | control | `NumTrainingPositions` 200M |
| `visgates200M` | vis edge bias + B+C content gates (the source program's best value arm) | 200M + `UseVisEdgeBias`/`VisEdgeGates` in net json |

**Everything else identical between the two arms and to the 20M smokes**
(256x10 post-norm RMSNorm + RPE, no smolgen, Muon 2e-3 decay@0.6, batch
fwd 512 / bwd 4096, survival w=0.6 K=4, FractionQ=1.0, t91 skip-1 corpus +
puzzle-aug 32:1). Do not "improve" one arm without the other — the pair is the
experiment.

## Prerequisites

1. `git pull` to **at least `1aa4f12`** (visibility edge-bias v2). Older code
   lacks the `NetDef VisEdge*` config fields AND the Muon fix — the gates arm
   will either silently ignore the fields or crash Muon with `assert p.ndim == 2`.
2. Corpus `t91_skip1_v2_surv` (191.5M pos, V2 137 B/sq, survival K=8 sidecars)
   and the puzzle-aug secondary `c1_640_34_2350up_aug_v2_surv/tpg` present on
   local NVMe. Adjust `TrainingFilesDirectory`/`2` in the data configs to the
   local paths.
3. Single GPU (`DeviceIDs: [0]`). DDP with the vis modules is untested — do not
   torchrun this pair.
4. ~7 h per arm at ~8K pos/s (dev-box rate; scale to local NPS). Disk: each
   ckpt ~120 MB, 4 ckpts/arm + exports.

## Configs

Create the quartet `<ID>_ceres_{exec,net,opt,data}.json` in `configs/` for each
ID. Clone from the in-repo 20M smoke quartet if present (`raybase20M_*`), else
use these exact contents:

### `<ID>_ceres_exec.json` (both arms)
```json
{
  "ID": "visedge200M_pair", "DeviceType": "cuda", "DeviceIDs": [0],
  "DataType": "BFloat16", "UseFP8": false, "UseHistory": true,
  "RunInDocker": false, "DropoutRate": 0, "DropoutDuringInference": false,
  "EngineType": "CSharpViaTorchscript",
  "SaveNetwork1FileName": null, "SaveNetwork2FileName": null,
  "ActivationMonitorDumpSkipCount": 0, "SupplementaryStat": "None",
  "TrackFinalLayerIntrinsicDimensionality": false, "MonitorActivationStats": false,
  "ExportOnly": false, "TestFlag": false, "TestValue": 0
}
```

### `<ID>_ceres_net.json`
Both arms share this base:
```json
{
  "TrainOn4BoardSequences": false, "ModelDim": 256, "NumLayers": 10,
  "NumHeads": 8, "LoRARankDivisor": 0, "UseQKV": true,
  "DualAttentionMode": "None", "PreNorm": false, "NormType": "RMSNorm",
  "AttentionMultiplier": 1, "FFNMultiplier": 6, "FFNActivationType": "Mish",
  "FFNUseGlobalEveryNLayers": 0, "HeadsActivationType": "Mish",
  "PriorStateDim": 0, "NonLinearAttention": false, "SoftCapCutoff": 0,
  "UseQKNorm": false, "DeepNorm": false, "DenseFormer": false,
  "SmolgenDimPerSquare": 0, "SmolgenDim": 0, "SmolgenToHeadDivisor": 1,
  "SmolgenActivationType": "Swish",
  "UseRPE": true, "UseRPE_V": true, "UseRelBias": false, "UseRoPE": false,
  "HeadWidthMultiplier": 4,
  "SoftMoEConfig": { "MoEMode": "None", "OnlyForAlternatingLayers": false,
    "NumExperts": 0, "NumSlotsPerExpert": 0, "UseNormalization": false,
    "UseBias": false },
  "TestValue": 0, "LoopCount": 1, "UsePieceRelationBias": false
}
```
**`visgates200M` net json additionally sets** (these fields ARE the second arm —
config-only, the old `CERES_VIS_EDGE_*` env vars are retired and assert):
```json
  "UseVisEdgeBias": true,
  "VisEdgeFamilies": "vis,xray,pinray",
  "VisEdgeGates": "qk",
  "VisEdgeSharedProjection": false
```
`raybase200M` net json must NOT contain the VisEdge fields (or set
`UseVisEdgeBias: false`).

### `<ID>_ceres_opt.json` (both arms)
```json
{
  "LoRARankDivisor": 0, "LoRARestrictPolicyValueOnly": false,
  "LoRARestrictValueOnly": false,
  "NumTrainingPositions": 200000000,
  "BatchSizeForwardPass": 512, "BatchSizeBackwardPass": 4096,
  "Optimizer": "Muon",
  "CheckpointFrequencyNumPositions": 50000000,
  "CheckpointResumeFromFileName": null,
  "PyTorchCompileMode": "default", "WeightDecay": 0.005,
  "LearningRateBase": 0.002, "LRBeginDecayAtFractionComplete": 0.6,
  "LRWarmupPhaseMultiplier": 0.1,
  "Beta1": 0.95, "Beta2": 0.999, "Beta3": 0.9999, "Alpha": 5,
  "GradientClipLevel": 1,
  "LossValueMultiplier": 1.0, "LossValue2Multiplier": 0.04,
  "LossPolicyMultiplier": 1.0, "LossMLHMultiplier": 0,
  "LossUNCMultiplier": 0.01, "LossQDeviationMultiplier": 0.02,
  "LossUncertaintyPolicyMultiplier": 0.01,
  "LossValueDMultiplier": 0, "LossValue2DMultiplier": 0,
  "LossActionMultiplier": 0, "LossActionUncertaintyMultiplier": 0,
  "TestValue": 0
}
```
`CheckpointFrequencyNumPositions: 50000000` is a deliberate deviation from the
final-only default: ckpts at 50/100/150/200M give the value-vs-exposure curve
(the source program benches exactly such curve nets to separate "value gain" from
"value overfit-then-collapse"). Keep it.

### `<ID>_ceres_data.json` (both arms — adjust paths to local disk)
```json
{
  "SourceType": "DirectFromTPG",
  "TPGFixedSet1": null, "TPGFixedSet2": null,
  "NumTPGFixedSet1BatchesReturnedForSet2": 0,
  "TrainingFilesDirectory": "/mnt/d/t91_skip1_v2_surv",
  "TrainingFilesDirectory2": "/mnt/d/c1_640_34_2350up_aug_v2_surv/tpg",
  "RatioSet1ToSet2": 32,
  "NumTPGFilesToSkip": 0, "TARPositionSkipCount": 0,
  "FractionQ": 1.0, "WDLLabelSmoothing": 0
}
```

## Launch

Per-run script (survival/loader knobs are runtime env, still env-based;
only the VisEdge arch knobs moved to config):

```bash
#!/usr/bin/env bash
export CERES_AUX_FEATURES_PER_SQUARE=0     # V2 corpus, 137 B/sq
export CERES_TPG_SQUARE_BYTES=137
export CERES_TPG_TARGET_SIDECAR=1
export CERES_SURVIVAL_HORIZON=4
export CERES_SURVIVAL_TARGET_WEIGHT=0.6
export CERES_SURVIVAL_LOSS_BUCKETS=2,4
export CERES_SURVIVAL_CAPTURE_WEIGHT=4
export CERES_SECONDARY_LOSS_VALUE_MULT=0
export CERES_SECONDARY_LOSS_VALUE2_MULT=0
export CERES_SECONDARY_LOSS_AUX_MULT=0
cd <repo>/src/CeresTrainPy
exec <python> -u train.py <ID> <OUTPUTS_DIR> > <OUTPUTS_DIR>/<ID>.log 2>&1
```

Run `raybase200M` first, `visgates200M` after it exits (sequential wrapper or a
`while pgrep -f 'train.py raybase200M'; do sleep 60; done` queue script). Both
fresh — `CheckpointResumeFromFileName: null`; do NOT resume from the 20M smokes.

## Launch verification (check within the first minutes, in the log)

1. `visgates200M` only: `[ceres_net] VISIBILITY EDGE BIAS enabled: families=('vis', 'xray', 'pinray') (12 channels), per-layer zero-init projection (960 params), content gates: qk`
   — if this line is missing, the config fields didn't take (old code or wrong json): **abort**.
2. `visgates200M` only: `[train] Muon partition scope=all-non-trunk: 70 muon / 117 adamw params`
   (baseline: 70/97). If Muon crashes with `assert p.ndim == 2` the code is pre-`1aa4f12`: **pull and restart**.
3. Both: first TRAIN line magnitudes sane (value_loss ~0.6, policy ~1.6 at pos 512;
   field order: pos, total, value, policy, ... — value BEFORE policy).
4. Both: `... shards carry survival sidecars (mode=required)` for the main corpus.

## After both finish (`INFO: EXIT_STATUS SUCCESS`)

1. Confirm both exported `.onnx` per checkpoint (auto-export; ~88–89 MB each).
2. Report back / sync the nets; the dev box runs the EB 4-band suite
   (mate + rg2340/2500/2700, policy + value, served blend) on the final nets and
   the 50/100/150M curve ckpts of both arms.
3. Decision rule (pre-registered): visgates value bands +X over raybase across
   ≥3 of 4 bands with policy flat → mechanism confirmed at horizon → next step
   is the 384x12 smolgen+SwiGLU production recipe (pre-derisked: the source
   program's +188-value-at-500k held WITH RPE and smolgen present) + the
   serving-graph efficiency work (visgates currently −34% NPS on TRT;
   known fixes documented in the review findings). Value flat at 200M →
   mechanism is refuted for our stack at this scale; close the arc.

## Known non-issues

- visgates trains ~identically fast to baseline (the −34% is TRT serving only).
- `WARN`-free resume/re-export requires the VisEdge fields to stay in the
  net json — they are the architecture record; never strip them.
- Sign-flipping small value deltas on TB during the run are expected; judge on
  the EB bands at the end, not the curves.
