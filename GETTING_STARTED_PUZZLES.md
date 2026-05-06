# Getting started: puzzle fine-tuning with CeresTrain

End-to-end guide for fine-tuning a Ceres net on Lichess puzzles using the
CeresTrain pipeline. Covers: cloning, building, generating training data
(label → enrich → TPG), launching a training run (head-only and full-body
LoRA recipes), folding the LoRA bin into a deployable ONNX, and running the
puzzle test harnesses.

---

## 0. Prerequisites

### 0a. System

- **Windows 10/11** with WSL2 (Ubuntu 22.04+ recommended) for the Python
  trainer, plus native Windows for the C# data-pipeline binary.
- **NVIDIA GPU** with TensorRT support. Training and labeling were validated
  on RTX 5090 (32 GB). Smaller cards work but reduce
  `BatchSizeBackwardPass` / `BatchSizeForwardPass` in
  `c1_640_34_ceres_opt.json` if you OOM.
- **NVIDIA driver + CUDA Toolkit + TensorRT** versions matching your
  PyTorch+ONNX install. The repo's TRT engine cache is bound to driver
  major version — first run on a new machine recompiles the cache (5-7 min
  per batch profile, ~40 min total — looks like a hang, isn't).

### 0b. Toolchain

- **.NET 10 SDK** (for the CeresTrain C# binary). `dotnet --version` must
  print `10.x`.
- **Python 3.10+** in WSL with: `torch` (2.x, CUDA-matched), `lightning`,
  `onnx`, `onnxruntime` (CPU is fine for export), `numpy`, `zstandard`,
  `python-chess` (for validation scripts only).

### 0c. External dependencies

- **Ceres** (the engine): clone <https://github.com/dje-dev/Ceres> and
  build alongside CeresTrain. Used at training time (shared types) and by
  the puzzle test scripts (UCI engine).
- **Lichess puzzles CSV**: download from
  <https://database.lichess.org/#puzzles>. The pipeline expects the
  uncompressed CSV.
- **Syzygy tablebases** (optional but referenced by all EngineDefs and
  puzzle test scripts under `SyzygyPath`). 3-4-5-piece is fine for puzzle
  testing. If you don't have them, set `SyzygyPath` to an empty string in
  the EngineDef + harness scripts.

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

Adjust paths to taste.

---

## 3.5. Stage 0 — get the orig artifacts (REQUIRED before training)

The pipeline depends on three artifacts that are NOT in this repo:

| Artifact | What it is | Used by |
|---|---|---|
| `C3-384-12-I8.onnx` | Smaller teacher net (12 layers) | Labeling Stage 1 (fast) |
| `C3-768-30-pre3-I8.onnx` | Larger teacher net (30 layers) | Optional sharper labeling |
| `ckpt_c1_640_34_from_onnx_0` | Lightning checkpoint of the orig net we fine-tune | Every training run (warm-start) and every fold (`V8_BASE_CKPT`) |

Get them from the repo owner (private distribution). They live under:

```
C:/Dev/Chess/Networks/CeresNet/C3-384-12-I8.onnx
C:/Dev/Chess/Networks/CeresNet/C3-768-30-pre3-I8.onnx
C:/Dev/Chess/Networks/CeresNet/Ceres_c1_640_34_orig_trt.onnx     # the orig itself, for puzzle-baseline tests
C:/Dev/Chess/CeresTrain/nets/ckpt_c1_640_34_from_onnx_0           # warm-start ckpt
```

### Bootstrap your own orig from a public ONNX

Don't have a checkpoint from the maintainer? You can build one from any
compatible Ceres ONNX (e.g. `C1-640-34-I8.onnx` available in the public
[Ceres-nets](https://github.com/dje-dev/Ceres-nets) repo). Two scripts under
`scripts/onnx_bootstrap/` handle this:

1. **`inspect_onnx.py`** — infer the architecture (ModelDim, NumLayers,
   NumHeads, FFNMultiplier, NonLinearAttention, etc.) by walking the ONNX
   graph. Use the output to populate `configs/<your-id>_ceres_net.json`.

   ```bash
   wsl.exe -- bash -lc "source ~/cerestrain-env/bin/activate && \
       python3 /mnt/c/Users/<you>/source/repos/CeresTrain/scripts/onnx_bootstrap/inspect_onnx.py \
         /mnt/c/Dev/Chess/Networks/CeresNet/C1-640-34-I8.onnx"
   ```

2. **`reconstruct_ckpt_from_onnx.py`** — instantiate a `CeresNet` matching
   the architecture, walk the ONNX graph to map each initializer to its
   PyTorch parameter (handling the bias-MatMul pair pattern AND bare
   MatMul nodes for q2/k2/v2 under NonLinearAttention), transpose
   anonymous matmul weights to PyTorch's `[out_features, in_features]`
   layout, and write a fabric-compatible
   `{'model': state_dict, 'optimizer': {...}, 'num_pos': '0'}` checkpoint.

   ```bash
   wsl.exe -- bash -lc "source ~/cerestrain-env/bin/activate && \
       python3 /mnt/c/Users/<you>/source/repos/CeresTrain/scripts/onnx_bootstrap/reconstruct_ckpt_from_onnx.py \
         /mnt/c/Dev/Chess/Networks/CeresNet/C1-640-34-I8.onnx \
         c1_640_34 \
         /mnt/c/Dev/Chess/CeresTrain"
   # Output: /mnt/c/Dev/Chess/CeresTrain/nets/ckpt_c1_640_34_from_onnx_0
   ```

   After running, verify forward-pass parity vs the source ONNX (the
   script reports max abs/rel diffs on policy/value heads — should be
   within FP16 noise). See `scripts/onnx_bootstrap/NOTES.md` for the
   debugging history of one bug that's been fixed (NonLinearAttention's
   bias-less q2/k2/v2 weights were initially missed).

The reverse path (ckpt→ONNX) lives in `src/CeresTrainPy/reconvert_onnx.py`
and is invoked at end-of-training and during fold; that's what
`export_v8_uint8_mish.py` uses.

### 3.5a. Auto-loaded settings files

Both binaries auto-load user-settings JSONs at first run:

- **CeresTrain**: `C:/Users/<you>/CeresTrain.json` — initialize on first
  run; fields cover scratch dirs and any host-specific overrides.
- **Ceres engine**: `C:/Users/<you>/source/repos/Ceres/artifacts/release/net10.0/Ceres.json`
  (and a copy may show up at `C:/Users/<you>/CeresTrain/Ceres.json`).
  Holds default tablebase paths and TRT cache locations.

If either binary errors with "user settings not found," run it once with
`--help`; it'll create a default file you can edit.

### 3.5b. TensorRT engine cache

First load of any ONNX through `Device: GPU:0#TensorRTNative` compiles
TRT engines for each batch profile (`[1, 8, 20, 42, 64, 88, 116, 240]` by
default), writing them to:

```
C:/Dev/Chess/Networks/CeresNet/trt_engines/<HOST>/
```

Expect 5-7 minutes per profile on first run. Subsequent loads are <5
seconds. If you upgrade your NVIDIA driver, delete this cache — engines
are driver-version-bound.

### 3.5c. Sanity-test the orig before training

Verify the orig net loads and serves UCI before launching anything:

```bash
C:/Users/<you>/source/repos/Ceres/artifacts/release/net10.0/Ceres.exe UCI
> setoption name Network value C:/Dev/Chess/Networks/CeresNet/Ceres_c1_640_34_orig_trt.onnx
> setoption name Device value GPU:0#TensorRTNative
> isready
> position startpos
> go nodes 100
```

You should get a `bestmove` reply within a few seconds (after the
one-time TRT compile). If TRT compile fails, double-check CUDA/TensorRT
versions match your PyTorch install.

### 3.5d. Run the puzzle harness against orig (baseline reference)

Before any fine-tune, capture the orig's puzzle metric on your machine:

```bash
# Edit C:/Users/<you>/source/repos/CeresTrain/scripts/compare_value_eb_full_line.py:
#   - CERES = path to your Ceres.exe
#   - CSV_PATH = path to your Lichess CSV
#   - SyzygyPath = path to your tablebases (or "")
#   - CONFIGS = { "orig": _cfg("...Ceres_c1_640_34_orig_trt.onnx") }
python C:/Users/<you>/source/repos/CeresTrain/scripts/compare_value_eb_full_line.py
```

Expected (5K narrow ≥2710, RTX-class GPU): **74.44%**. If your number is
significantly different, your TRT/driver stack is producing different
inference outputs — fix that before fine-tuning.

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

### 4b. Stage 2 (RECOMMENDED): augment 2× via vertical-flip + side-swap

The session-best recipe uses solver-only puzzle data with 2× augmentation. No
post-processing is needed: the labeler now writes `StartFen` + `PriorUciMoves`
natively (the old `patch_jsonl_history.py` step is no longer required, and
`clamp_wdl.py` was a workaround for an MCGS-aggregation rounding bug that's
since been fixed).

Use `scripts/augment_realhist_jsonl.py` (or a path-edited copy) to emit each
record twice — original and mirrored:

```bash
python C:/Users/<you>/source/repos/CeresTrain/scripts/augment_realhist_jsonl.py
```

Inputs and outputs are hardcoded paths near the top of the script — edit
`SRC` and `DST` to your dirs.

Output: `D:/Puzzles/c3_<band>_aug/labeled_aug.jsonl` (~2× the labeled count).

### 4c. Stage 3 (optional, legacy): OppDefence enrichment

The session demonstrated that **PONLY (`LossValueMultiplier=0`) + KL anchor**
on solver-only data outperforms OppDef-enriched corpora for the recipe that
ships. OppDef enrichment is left in for completeness but is no longer
recommended for the default puzzle FT pipeline.

If you do want OppDef enrichment (for an experimental run or on an older
recipe), it adds opp-to-move records with search-backed WDL targets:

```bash
CeresTrain.exe enrich-opp-defence --puzzle-config=C:/Dev/Chess/CeresTrain/c3_<band>_oppdef.json
```

The opp-side records get all-zero policy targets, so the policy head trains
only on solver moves while the value head sees both. With the modern recipe
`LossValueMultiplier=0` makes those records zero-loss no-ops anyway, hence
the simplification.

### 4d. Stage 4: convert to TPG

Point `LabeledJsonlFileName` at the file you want to train on (modern
recipe: `labeled.jsonl` for non-aug or `labeled_aug.jsonl` for aug):

```bash
CeresTrain.exe puzzles-to-tpg --puzzle-config=C:/Dev/Chess/CeresTrain/c3_<band>_tpg.json
```

Output: `D:/Puzzles/c3_<band>/tpg/puzzles_<timestamp>.tpg_set0.zst`. About
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

### 5a. ⭐ RECIPE: KL anchor + body+head LoRA + PONLY (CURRENT SHIP)

This is the session-best recipe — tournament-confirmed +18.4 ±9.6 Elo over
orig at 200n. Combines:

- Body LoRA r-div=32 + head LoRA r-div=32 (about 12.3M trainable params)
- KL-divergence anchor to orig (β=3.0 on policy AND value)
- `LossValueMultiplier=0` ("PONLY") to avoid gradient-conflict between
  puzzle policy and value losses
- Trained on the solver-only 2x-aug corpus

In `c1_640_34_ceres_opt.json`:

```json
{
  "LoRARankDivisor": 32,
  "LoRARestrictPolicyValueOnly": false,
  "LoRARestrictValueOnly": false,
  "NumTrainingPositions": 1000000,
  "CheckpointFrequencyNumPositions": 250000,
  "CheckpointResumeFromFileName": "/mnt/c/Dev/Chess/CeresTrain/nets/ckpt_c1_640_34_from_onnx_0",
  "WeightDecay": 0.005,
  "LearningRateBase": 0.0001,
  "LRBeginDecayAtFractionComplete": 0.5,
  "LRWarmupPhaseMultiplier": 0.1,
  "Beta1": 0.95,
  "Beta2": 0.999,
  "LossValueMultiplier": 0.0,
  "LossPolicyMultiplier": 1.0,
  "LossUNCMultiplier": 0.01,
  "LossQDeviationMultiplier": 0.02,
  "LossUncertaintyPolicyMultiplier": 0.01,
  "KLAnchorRefCheckpoint": "/mnt/c/Dev/Chess/CeresTrain/nets/ckpt_c1_640_34_from_onnx_0",
  "KLAnchorPolicyWeight": 3.0,
  "KLAnchorValueWeight": 3.0
}
```

In `c1_640_34_ceres_net.json` ensure `"SoftCapCutoff": 0` to match orig.

Launch (WSL) with body-LoRA env var set:

```bash
source ~/cerestrain-env/bin/activate
cd /mnt/c/Users/<you>/source/repos/CeresTrain/src/CeresTrainPy
CERES_LORA_TRANSFORMER_RANK_DIV=32 \
  python train.py c1_640_34 /mnt/c/Dev/Chess/CeresTrain
```

**β-sweep notes (1M positions, solver-only 2x-aug corpus):**

The tournament Elo curve at 200n is unimodal in β with a plateau in
[β=1.5, β=3.0] hitting roughly +16 to +18 Elo. β=1.0 collapses to ~+2 Elo
(extreme drift, search can't recover). β=4.0 saturates at smaller gains.
β=3.0 has the best raw policy quality at nodes=1 (smallest deficit vs orig)
and is the safer ship despite being statistically tied with β=1.5 at 200n.

### 5b. Legacy recipe: head-only PV LoRA (pre-KL-anchor era)

Older recipe, kept for completeness. Achieves Val +1.28 / Pol par at hard-tail
≥2710 in puzzle metrics. **No tournament-confirmed positive Elo** — use
recipe 5a instead.

```json
{
  "LoRARankDivisor": 128,
  "LoRARestrictPolicyValueOnly": true,
  "LoRARestrictValueOnly": false,
  "LossValueMultiplier": 1.0,
  "LossPolicyMultiplier": 1.0,
  "LearningRateBase": 0.0001,
  "NumTrainingPositions": 1000000,
  "Beta2": 0.999
}
```

Launch (WSL), no body-LoRA env vars:

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

### 5d. Full LoRA env-var reference

These are read by `dot_product_attention.py`, `mlp2_layer.py`,
`ceres_net.py`, and `save_model.py`. All default to `0` (= disabled) when
unset.

| Env var | Effect |
|---|---|
| `CERES_LORA_ATTN_RANK_DIV` | LoRA rank divisor for attention QKV projections in transformer body. r-div=64 → rank ≈ ModelDim/64. |
| `CERES_LORA_FFN_RANK_DIV` | LoRA rank divisor for MLP/FFN linear layers (linear1, linear2, plus SwiGLU gate linear3) in transformer body. |
| `CERES_LORA_TRANSFORMER_RANK_DIV` | Legacy unified knob that sets both attn and ffn at once if the specific knobs are unset. |
| `CERES_LORA_LAYER_MIN` | Inclusive minimum layer index that body LoRA wraps (default: 0). |
| `CERES_LORA_LAYER_MAX` | Inclusive maximum layer index that body LoRA wraps (default: NumLayers-1). |
| `CERES_LORA_HEADFRONT_RANK_DIV` | LoRA on the shared head-front projection that feeds every head (after the body, before the heads split). |
| `CERES_LORA_SMOLGEN_RANK_DIV` | LoRA on the smolgen prep + per-attention sm1/sm2/sm3 modules. |
| `V8_BASE_CKPT` | (export-time only) Path to the orig Lightning checkpoint to fold the LoRA delta onto. |
| `V8_LORA_BIN` | (export-time only) Path to the trained `lepdev_c1_640_34.lora_*.bin`. |
| `V8_OUT` | (export-time only) Output path for the folded TRT-deployable ONNX. |
| `V8_LORA_SKIP_PREFIX` | (export-time only) Optional prefix; LoRA modules whose names start with this prefix are skipped during fold. Used in two-stage flows where body is already folded into the ckpt and shouldn't be folded again from the bin. Typical: `transformer_layer.`. |

Head LoRA (the "PV head" — policy and value heads) is controlled by the
`opt.json` field `LoRARankDivisor` plus the boolean restrictors
`LoRARestrictPolicyValueOnly` / `LoRARestrictValueOnly`.

KL anchor (added in this session) is controlled by three opt.json fields:
`KLAnchorRefCheckpoint` (path to frozen reference, typically the same as the
warm-start ckpt), `KLAnchorPolicyWeight` (β_pol), `KLAnchorValueWeight`
(β_val). All default to 0 / null when unset (no anchor). See
`project_kl10_body32_ponly_breakthrough.md` in memory for full β-sweep
results.

### 5e. Path-edit checklist (before first run on a new machine)

Several paths are hardcoded across the repo. Search-and-replace these
before training/testing on a new layout:

| File | What to edit |
|---|---|
| `C:/Dev/Chess/CeresTrain/configs/c1_640_34_ceres_data.json` | `TrainingFilesDirectory` — your TPG dir (WSL `/mnt/...` form) |
| `C:/Dev/Chess/CeresTrain/configs/c1_640_34_ceres_opt.json` | `CheckpointResumeFromFileName` — full path to your orig ckpt (WSL form) |
| `C:/Dev/Chess/CeresTrain/run_v*.sh` | `LOG_DIR`, `NETS`, `ENGINE_DIR`, `ORIG_CKPT_LINUX`, `TPG_DIR_LINUX` |
| `scripts/compare_value_eb_full_line.py` | `CERES` (path to Ceres.exe), `CSV_PATH` (Lichess CSV), `SyzygyPath`, `CONFIGS` (the net under test) |
| `scripts/compare_policy_eb_full_line.py` | Same fields as the value harness |
| `C:/Dev/Chess/CeresTrain/c3_*.json` (puzzle configs) | `LichessCsvPath`, `OutDir`, `NetSpec` |
| `C:/Dev/Chess/Engines/EngineDefs/Ceres_*_*.json` | `Path` (Ceres.exe), `NetworkPath`, `Network`, `SyzygyPath` |

### 5f. Driver script template

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
python3 /mnt/c/Users/<you>/source/repos/CeresTrain/scripts/export_v8_uint8_mish.py
```

The output ONNX is loadable by Ceres for both EngineBattle and the puzzle
test scripts.

---

## 7. Running puzzle tests

Two equivalent paths exist:

### 7a (RECOMMENDED). EngineBattle Console `puzzlejson`

The cleanest cross-engine puzzle comparison runner. Set up a JSON config
(see `C:/Dev/Chess/EB/PuzzleLichess_CeresAlt.json` for a template) listing
the engines and test types, then:

```bash
EngineBattle.Console.exe puzzlejson C:/Dev/Chess/EB/PuzzleLichess_<MyConfig>.json
```

Supports `Type: "policy, policy3, value"` for top-1/top-3 policy + value-head
tests on the same set, multiple engines in one run via the `Engines[]` array,
optional `RatingGroups` for cohort-specific output rows. Add `--json
<output-path>` for stable structured JSON output (see PuzzleJsonSchema.md in
the EngineBattle repo).

For multi-net sweeps (one engine def, many networks), use `EngineWithNets`
with `ListOfNetsWithPaths`.

### 7b (LEGACY). Repo-local Python harnesses

Two scripts live in `C:/Dev/Chess/CeresTrain/`:

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

## 8. Reference values

### 8a. Orig baseline (legacy harness)

5K narrow ≥2710, EB-aligned full-line harness, TRT-Native:

| metric | orig (`Ceres_c1_640_34_orig_trt.json`) |
|---|---|
| value per-puzzle | 74.44% |
| policy per-puzzle | 56.26% |
| policy per-move | 84.11% |

### 8b. EB Console puzzlejson (modern harness)

2000 puzzles, AvgR ~2498, RatingGroups "2499", Type `value, policy, policy3`,
Nodes 1:

| metric | orig (`Ceres_C1-640-34-I8_orig_default.json`) |
|---|---|
| Policy Perf | 2649 |
| Policy accuracy | 70.5% |
| pTop3 Perf | 3077 |
| pTop3 accuracy | 96.6% |
| Value Perf | 2822 |
| Value accuracy | 86.6% |
| Pol KLD | 0.8874 |

### 8c. Current shipping fine-tune (KL30 PONLY 2500up)

Same harness as 8b:

| metric | KL30 2500up | Δ vs orig |
|---|---|---|
| Policy Perf | 2679 | **+30** |
| Policy accuracy | 74.0% | +3.5 pp |
| pTop3 Perf | 3124 | +47 |
| Value Perf | 2820 | par |

Tournament: **+18.4 ±9.6 Elo over orig at 200n** (1000 games, CFS 100%).

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
