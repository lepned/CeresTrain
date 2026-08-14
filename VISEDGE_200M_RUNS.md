# VISEDGE 200M pair — run spec for the training computer (rev 2, 2026-08-14)

**Goal:** adjudicate the visibility edge-bias mechanism at the horizon where
value becomes readable (the source program's readable value results needed
~200M samples; the 20M ladder's value deltas sit inside single-run noise).

**Rev 2 changes the test arm.** The 5-arm 20M ladder (dev box, 2026-08-14) now
includes `visqk20M` = **vis-only channels + shared projection + qk content
gates** — the source program's exact champion configuration (its 2333 value
arm). On the ladder it is (a) the best OOD-value arm (rg2340/2500/2700 sum +53
over control, vs +20 for the full-12ch gates arm), (b) the ONLY arm value-led
at all four bands (the production-like operating point), and (c) nearly free to
serve: 1.28x engine time, ~−5% search NPS (vs 1.92x / −33% for full-12ch).
The full-12ch `visgates200M` arm from rev 1 is DEFERRED — it is an
xray/pinray-attribution experiment, not the value-per-NPS candidate.

Two runs, sequential, single GPU:

| run | what | config delta vs the 20M smokes |
|---|---|---|
| `raybase200M` | control | `NumTrainingPositions` 200M |
| `visqk200M` | vis-only + shared projection + qk gates | 200M + the four `VisEdge*` fields in net json |

**Everything else identical between the two arms and to the 20M smokes**
(256x10 post-norm RMSNorm + RPE, no smolgen, Muon 2e-3 decay@0.6, batch
fwd 512 / bwd 4096, survival w=0.6 K=4, FractionQ=1.0, t91 skip-1 corpus +
puzzle-aug 32:1). Do not "improve" one arm without the other — the pair is the
experiment.

## Prerequisites

1. `git pull` to **at least `45cb87e`** (vis edge-bias serving-graph
   optimizations). That commit also guarantees `1aa4f12` (the VisEdge config
   fields + the Muon ndim fix). Older code either lacks the fields (arm
   silently becomes the control) or exports the slow serving graph.
2. Corpus `t91_skip1_v2_surv` (191.5M pos, V2 137 B/sq, survival K=8 sidecars)
   and the puzzle-aug secondary `c1_640_34_2350up_aug_v2_surv/tpg` present on
   local NVMe. Adjust `TrainingFilesDirectory`/`2` in the data configs to the
   local paths.
3. Single GPU (`DeviceIDs: [0]`). DDP with the vis modules is untested — do not
   torchrun this pair.
4. ~7 h per arm at ~8K pos/s (dev-box rate; scale to local NPS). Disk: each
   ckpt ~120 MB, 4 ckpts/arm + exports.

## Configs — the exact 8 files

train.py identifies a run by the config **filename prefix** given on the
command line (`train.py visqk200M ...` reads `configs/visqk200M_ceres_*.json`);
the `"ID"` field inside exec json is a legacy label and not load-bearing.
Create these 8 files in `<OUTPUTS_DIR>/configs/` (contents complete — no
assembly needed). Only the two net jsons differ between the arms.

### `raybase200M_ceres_exec.json` AND `visqk200M_ceres_exec.json` (identical contents)
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

### `raybase200M_ceres_net.json` (control — note: NO VisEdge fields)
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

### `visqk200M_ceres_net.json` (test arm — identical plus the last four fields,
which ARE the experiment; config-only, the old `CERES_VIS_EDGE_*` env vars are
retired and assert)
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
  "TestValue": 0, "LoopCount": 1, "UsePieceRelationBias": false,
  "UseVisEdgeBias": true,
  "VisEdgeFamilies": "vis",
  "VisEdgeGates": "qk",
  "VisEdgeSharedProjection": true
}
```

⚠ Write the json files WITHOUT a UTF-8 BOM (PowerShell 5.1 `Out-File -Encoding
utf8` adds one and train.py's json load dies on it; use
`[System.IO.File]::WriteAllText(path, text, [System.Text.UTF8Encoding]::new($false))`
or write from bash).

### `raybase200M_ceres_opt.json` AND `visqk200M_ceres_opt.json` (identical contents)
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
(the source program benches exactly such curve nets to separate "value gain"
from "value overfit-then-collapse"). Keep it.

### `raybase200M_ceres_data.json` AND `visqk200M_ceres_data.json` (identical contents — adjust the two directory paths to the local corpus locations)
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
only the VisEdge arch knobs are config):

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

`<ID>` is literally `raybase200M` resp. `visqk200M` (must match the config
filename prefixes above); `<OUTPUTS_DIR>` is the directory CONTAINING
`configs/` — logs, ckpts (`nets/`) and TB logs land under it.

Run `raybase200M` first, `visqk200M` after it exits (sequential wrapper or a
`while pgrep -f 'train.py raybase200M'; do sleep 60; done` queue script). Both
fresh — `CheckpointResumeFromFileName: null`; do NOT resume from the 20M smokes.

## Launch verification (check within the first minutes, in the log)

1. `visqk200M` only — this exact line (verified on the dev 20M run):
   `[ceres_net] VISIBILITY EDGE BIAS enabled: families=('vis',) (4 channels), shared zero-init projection (32 params), content gates: qk`
   — if missing, the config fields didn't take (old code or wrong json): **abort**.
   If it says 12 channels or per-layer, the wrong net json was cloned: **abort**.
2. `visqk200M` only: `[train] Muon partition scope=all-non-trunk: 70 muon / 108 adamw params`
   (control: 70/97). A Muon crash `assert p.ndim == 2` means pre-`1aa4f12` code:
   **pull and restart**.
3. Both: first TRAIN line magnitudes sane (value_loss ~0.6, policy ~1.6 at pos
   512; field order: pos, total, value, policy, ... — value BEFORE policy).
4. Both: `... shards carry survival sidecars (mode=required)` for the main corpus.

## After both finish (`INFO: EXIT_STATUS SUCCESS`)

1. Confirm both exported `.onnx` per checkpoint (auto-export; ~88 MB each).
2. Report back / sync the nets; the dev box runs the EB 4-band suite
   (mate + rg2340/2500/2700, policy + value, served blend) on the final nets and
   the 50/100/150M curve ckpts of both arms.
3. Decision rule (pre-registered): visqk value bands up vs raybase across ≥3 of
   4 bands with policy flat → mechanism confirmed at horizon → next step is the
   384x12 smolgen+SwiGLU production recipe with the same four VisEdge fields
   (pre-derisked: the source program's +188-value-at-500k held WITH RPE and
   smolgen present), plus one replace-smolgen arm (source program measured its
   graph 14.9% FASTER than smolgen — the mechanism may be a net-free smolgen
   substitute). Value flat at 200M → mechanism refuted for our stack at this
   scale; close the arc.

## Known non-issues

- Serving cost of this arm is small and already measured on dev: 1.28x engine
  time vs control (trtexec B=512), ~−5% search NPS. The −34%/1.92x numbers
  belong to the deferred full-12ch arm only.
- Speed probes on throwaway short-trained nets UNDERSTATE search NPS (bad
  policy → worse tree/batch shapes): judge serving cost by trtexec engine
  ratio, or by search cmp on properly trained nets only.
- `WARN`-free resume/re-export requires the VisEdge fields to stay in the
  net json — they are the architecture record; never strip them.
- Sign-flipping small value deltas on TB during the run are expected; judge on
  the EB bands at the end, not the curves.
