# Getting started: puzzle fine-tuning with CeresTrain

End-to-end guide for fine-tuning a Ceres net on Lichess puzzles using the
CeresTrain pipeline. Covers: cloning, building, generating training data
(label → enrich → TPG), launching a training run (head-only and full-body
LoRA recipes), folding the LoRA bin into a deployable ONNX, and running the
puzzle test harnesses.

---

## 0. Prerequisites

- **Windows 10/11** with WSL2 (Ubuntu 22.04+ recommended) for the Python
  trainer, plus native Windows for the C# data-pipeline binary.
- **NVIDIA GPU** with TensorRT support (training and labeling were validated
  on RTX 5090; any 16+ GB GPU should work).
- **.NET 10 SDK** (for the CeresTrain C# binary).
- **Python 3.10+** in WSL with PyTorch 2.x + CUDA, lightning, onnx, etc.
- **Ceres** (the engine) — built and runnable. Used by puzzle test scripts.
- **Lichess puzzles CSV**: download from
  <https://database.lichess.org/#puzzles> and place at a known path.

---

## 1. Clone CeresTrain and Ceres

```bash
# CeresTrain (this repo)
git clone https://github.com/lepned/CeresTrain.git C:/Users/<you>/source/repos/CeresTrain

# Ceres (engine; needed by puzzle tests and at training time for data
# loaders that share types with CeresTrain).
git clone https://github.com/dje-dev/Ceres.git C:/Users/<you>/source/repos/Ceres
```

Build both:

```bash
cd C:/Users/<you>/source/repos/Ceres/src
dotnet build -c Release

cd C:/Users/<you>/source/repos/CeresTrain/src
dotnet build -c Release
```

Verify the CeresTrain CLI:

```bash
C:/Users/<you>/source/repos/CeresTrain/artifacts/release/net10.0/CeresTrain.exe --help
```

---

## 2. Set up the Python environment (WSL)

```bash
# In WSL
python3 -m venv ~/cerestrain-env
source ~/cerestrain-env/bin/activate
pip install --upgrade pip
pip install torch lightning onnx numpy zstandard python-chess
```

Note: the trainer expects `~/cerestrain-env/bin/activate` by default. If you
use a different path, edit it in the run scripts under
`C:/Dev/Chess/CeresTrain/run_v*.sh`.

---

## 3. Project layout (suggested)

The training-side scripts assume:

```
C:/Dev/Chess/Puzzles/
    lichess_db_puzzle_<date>.csv          # downloaded Lichess CSV

C:/Dev/Chess/CeresTrain/
    configs/
        c1_640_34_ceres_data.json         # which TPG dir to train from
        c1_640_34_ceres_net.json          # net architecture (SoftCapCutoff etc.)
        c1_640_34_ceres_opt.json          # LoRA + optimizer + LR
        c1_640_34_ceres_exec.json
        c1_640_34_ceres_monitoring.json
    nets/
        ckpt_c1_640_34_from_onnx_0        # base orig checkpoint (warm start)
        lepdev_c1_640_34_v<N>_folded_trt.onnx  # output nets (TRT-deployable)
    compare_value_eb_full_line.py         # value puzzle harness
    compare_policy_eb_full_line.py        # policy puzzle harness
    export_v8_uint8_mish.py               # LoRA → folded ONNX deployment
    run_v<N>.sh                           # per-experiment driver scripts

C:/Dev/Chess/Networks/CeresNet/
    C3-384-12-I8.onnx                     # smaller teacher (fast labeling)
    C3-768-30-pre3-I8.onnx                # larger teacher (sharper labels)

D:/Puzzles/                                # data lives on a fast SSD
    c3_<rating-band>/                     # one dir per labeling run
        labeled.jsonl                     # raw teacher labels
        labeled_clamped.jsonl             # post-clamp+rank-1 fix
        labeled_with_oppdef.jsonl         # + OppDef enrichment
        tpg/puzzles_*.tpg_set0.zst        # TPG training shards
```

Adjust paths to taste; `Network/CeresNet/*.onnx` and the orig checkpoint are
distributed separately (ask repo owner).

---

## 4. Generating TPG training data

The puzzle pipeline runs in three stages, each driven by a JSON config:

### 4a. Stage 1: label puzzles with a search-backed teacher

Create `c3_<band>.json` (rating range, output dir, teacher net, search nodes):

```json
{
  "LichessCsvPath": "C:/Dev/Chess/Puzzles/lichess_db_puzzle_<date>.csv",
  "MinRating": 2600,
  "MaxRating": 2700,
  "ThemeIncludeAny": null,
  "ThemeExcludeAny": null,
  "MaxPuzzlesToRead": 1000000000,
  "MaxRecordsToLabel": 0,
  "SkipMining": true,
  "ResumeFromCheckpoint": false,
  "OutDir": "D:/Puzzles/c3_2600_2700",
  "NetSpec": "C:/Dev/Chess/Networks/CeresNet/C3-384-12-I8.onnx",
  "Device": "GPU:0#TensorRTNative",
  "TeacherNodes": 200,
  "MineBatchSize": 512,
  "TeacherWorkerThreads": 1
}
```

Run:

```bash
CeresTrain.exe label-puzzles --puzzle-config=C:/Dev/Chess/CeresTrain/c3_2600_2700.json
```

Output: `D:/Puzzles/c3_2600_2700/labeled.jsonl`. ETA ≈ 1 minute per ~1,000
puzzle records on RTX 5090 at 200 nodes.

### 4b. Stage 2: post-process (clamp + rank-1 nudge)

```bash
python C:/Dev/Chess/CeresTrain/clamp_wdl.py \
    D:/Puzzles/c3_2600_2700/labeled.jsonl \
    D:/Puzzles/c3_2600_2700/labeled_clamped.jsonl
```

This clamps any negative WDL components from MCGS-aggregation rounding and
ensures the Lichess solution is the unique top of the policy distribution.

### 4c. Stage 3a (optional but recommended): OppDefence enrichment

Adds opp-to-move records with search-backed WDL targets so the value head
gets calibration coverage on the positions MCTS reaches at inference. Use
`labeled_clamped.jsonl` as input via `LabeledJsonlFileName`:

```json
{
  "LichessCsvPath": "C:/Dev/Chess/Puzzles/lichess_db_puzzle_<date>.csv",
  "MinRating": 2600,
  "MaxRating": 2700,
  "SkipMining": true,
  "OutDir": "D:/Puzzles/c3_2600_2700",
  "LabeledJsonlFileName": "labeled_clamped.jsonl",
  "NetSpec": "C:/Dev/Chess/Networks/CeresNet/C3-384-12-I8.onnx",
  "Device": "GPU:0#TensorRTNative",
  "TeacherNodes": 100,
  "TeacherWorkerThreads": 1
}
```

```bash
CeresTrain.exe enrich-opp-defence --puzzle-config=C:/Dev/Chess/CeresTrain/c3_2600_2700_oppdef.json
```

Output: `D:/Puzzles/c3_2600_2700/labeled_with_oppdef.jsonl`. ETA ≈ 70 min for
~200K records. OppDef records are emitted with all-zero policy targets
downstream, so the policy head trains only on solver-side puzzle moves while
the value head sees both.

### 4d. Stage 3b: convert to TPG

Same config as enrichment, just point `LabeledJsonlFileName` at the file you
want to train on (typically `labeled_with_oppdef.jsonl`):

```bash
CeresTrain.exe puzzles-to-tpg --puzzle-config=C:/Dev/Chess/CeresTrain/c3_2600_2700_tpg.json
```

Output: `D:/Puzzles/c3_2600_2700/tpg/puzzles_<timestamp>.tpg_set0.zst`. About
1-2 minutes for ~350K records.

### 4e. (Optional) merge multiple rating bands

To train on a broader rating range, label each band separately, then collect
the resulting `.zst` shards into one directory:

```bash
mkdir -p D:/Puzzles/c3_combined/tpg
cp D:/Puzzles/c3_2600_2700/tpg/*.zst D:/Puzzles/c3_combined/tpg/
cp D:/Puzzles/c3_above_2700/tpg/*.zst D:/Puzzles/c3_combined/tpg/
```

The trainer reads every `.zst` in the configured directory.

---

## 5. Setting up a training experiment

The trainer reads three JSON configs:

| File | Role |
|---|---|
| `c1_640_34_ceres_data.json` | `TrainingFilesDirectory` — points to your TPG dir |
| `c1_640_34_ceres_net.json` | architecture (must match the orig you fold onto) |
| `c1_640_34_ceres_opt.json` | LoRA rank, loss weights, LR, NumTrainingPositions, warm-start ckpt |

Always start from the **orig checkpoint** (never resume from a prior LoRA bin):

```json
"CheckpointResumeFromFileName": "/mnt/c/Dev/Chess/CeresTrain/nets/ckpt_c1_640_34_from_onnx_0"
```

### 5a. Recipe: head-only PV LoRA (safest, current best Pareto)

In `c1_640_34_ceres_opt.json`:

```json
{
  "LoRARankDivisor": 128,
  "LoRARestrictPolicyValueOnly": true,
  "LoRARestrictValueOnly": false,
  "LossValueMultiplier": 1.0,
  "LossPolicyMultiplier": 1.0,
  "LearningRateBase": 0.0001,
  "NumTrainingPositions": 1000000,
  "CheckpointFrequencyNumPositions": 1000000,
  "Beta2": 0.999
}
```

In `c1_640_34_ceres_net.json` ensure `"SoftCapCutoff": 0` to match orig.

Launch (WSL):

```bash
source ~/cerestrain-env/bin/activate
cd /mnt/c/Users/<you>/source/repos/CeresTrain/src/CeresTrainPy
python train.py c1_640_34 /mnt/c/Dev/Chess/CeresTrain
```

### 5b. Recipe: full-body LoRA + PV head LoRA

Same `opt.json` but typically use a tighter body rank, controlled via env
vars before launching `train.py`:

```bash
CERES_LORA_ATTN_RANK_DIV=64 \
CERES_LORA_FFN_RANK_DIV=64 \
python train.py c1_640_34 /mnt/c/Dev/Chess/CeresTrain
```

To restrict body LoRA to a layer range (e.g., skip the last 3 layers of a
34-layer net):

```bash
CERES_LORA_ATTN_RANK_DIV=64 \
CERES_LORA_FFN_RANK_DIV=64 \
CERES_LORA_LAYER_MIN=0 \
CERES_LORA_LAYER_MAX=30 \
python train.py c1_640_34 /mnt/c/Dev/Chess/CeresTrain
```

### 5c. Recipe: value-head-only diagnostic

For isolating "is body-LoRA-shift the value killer?" — train only the value
head with value-only loss:

```json
{
  "LoRARankDivisor": 128,
  "LoRARestrictValueOnly": true,
  "LoRARestrictPolicyValueOnly": false,
  "LossValueMultiplier": 1.0,
  "LossPolicyMultiplier": 0,
  "LearningRateBase": 0.0001
}
```

Body env vars unset (no body LoRA at all). Total trainable params: ~few-K.

### 5d. Driver script template

The `run_v<N>.sh` scripts under `C:/Dev/Chess/CeresTrain/` chain
config-mutation, training, fold, and puzzle tests in one shot. Copy the
nearest match (e.g., `run_v205.sh` for head-only PV) and edit:

- `VID` — net id used in output filenames
- `NUMPOS` — training positions
- `TPG_DIR_LINUX` — TPG dir to read
- `LR`, `HEAD_DIV`, optional `BODY_DIV` — recipe knobs

Each run produces:

- `nets/lepdev_c1_640_34.lora_v<N>.bin` — the LoRA delta bin
- `nets/lepdev_c1_640_34_v<N>_folded_trt.onnx` — bin folded onto orig,
  TRT-deployable
- `training_v<N>.log`, `fold_v<N>.log`, `value_v<N>.log`, `policy_v<N>.log`
- `v<N>_results.txt` — summary of value/policy puzzle metrics

---

## 6. Folding LoRA delta into a deployable ONNX

After training emits `lepdev_c1_640_34.lora_<N>.bin`, fold it onto orig with
`export_v8_uint8_mish.py` (UINT8 input + decomposed Mish to match orig's
graph structure exactly):

```bash
source ~/cerestrain-env/bin/activate
V8_BASE_CKPT=/mnt/c/Dev/Chess/CeresTrain/nets/ckpt_c1_640_34_from_onnx_0 \
V8_LORA_BIN=/mnt/c/Dev/Chess/CeresTrain/nets/lepdev_c1_640_34.lora_<N>.bin \
V8_OUT=/mnt/c/Dev/Chess/CeresTrain/nets/lepdev_c1_640_34_v<N>_folded_trt.onnx \
python3 /mnt/c/Dev/Chess/CeresTrain/export_v8_uint8_mish.py
```

The output ONNX is loadable by Ceres for both EngineBattle and the puzzle
test scripts.

---

## 7. Running puzzle tests

Two harnesses live in `C:/Dev/Chess/CeresTrain/`:

- **`compare_value_eb_full_line.py`** — value-head puzzle solving.
  Walks every solver-to-move position; counts a puzzle solved only if the
  net plays the correct UCI move at all of them.
- **`compare_policy_eb_full_line.py`** — policy-head equivalent.

### 7a. Configure

Edit the `CONFIGS` dict at the top of each script:

```python
CONFIGS = {
    "v205": _cfg("C:/Dev/Chess/CeresTrain/nets/lepdev_c1_640_34_v205_folded_trt.onnx"),
}
N_PUZZLES = 5000
START_RATING = 2710
```

The driver scripts auto-write CONFIGS via sed/regex; if running manually,
edit it by hand.

### 7b. Run

```bash
python compare_value_eb_full_line.py > value_v205.log 2>&1
python compare_policy_eb_full_line.py > policy_v205.log 2>&1
```

Each finishes in ~5-7 minutes for 5K puzzles. Final lines look like:

```
FINAL: 3740/5000 puzzles solved = 74.80%
       11563/13750 per-move correct = 84.09%
```

Standard reference: orig at 5K narrow ≥2710 = **value 74.44 / policy 56.26
per-puzzle, 84.11 per-move**. Always compare your fine-tune vs orig (not vs
prior fine-tunes).

### 7c. EB tournament (the real ship gate)

The puzzle metric is necessary but not sufficient. To validate a candidate
ships, run an EngineBattle tournament against orig — see the EngineDef JSONs
under `C:/Dev/Chess/Engines/EngineDefs/Ceres_lepdev_c1_640_34_v<N>.json`.

---

## 8. Reference values (orig baseline)

5K narrow ≥2710, EB-aligned full-line harness, TRT-Native:

| metric | orig (`Ceres_c1_640_34_orig_trt.json`) |
|---|---|
| value per-puzzle | 74.44% |
| policy per-puzzle | 56.26% |
| policy per-move | 84.11% |

---

## 9. Pitfalls and conventions

- **Always start from orig**, never from a prior `.lora_last.bin`. The
  `CheckpointResumeFromFileName` field of `opt.json` should always be the
  orig checkpoint path. Per-experiment LoRA bins are saved under their `vID`
  for record-keeping.
- **Fold every run** before puzzle-testing. Untouched `.lora_last.bin` is
  not deployable.
- **`SoftCapCutoff` must match orig** (=0 for the c1_640_34 family).
  Mismatched values silently change NPS and can confound tournament results.
- **Use `Device: GPU:0#TensorRTNative`** in test configs to match orig's
  inference path.
- **Never resume a labeling run mid-band** without `ResumeFromCheckpoint:
  true` — otherwise you'll concatenate duplicate records.
- **Each LoRA experiment should sweep one knob.** Mixing recipe and data
  changes makes attribution impossible (see project history for several
  arcs that had to be retracted because of this).

---

## 10. Reading the results

A clean Pareto-positive net beats orig on **both** value and policy puzzle
metrics. Negative deltas on either axis are a regression even if the other
improves — historically, "policy gain at value cost" candidates have
consistently lost in EB tournament. The puzzle metric is the cheap pre-filter;
tournament is the additional gate. Both are required to ship.
