# Getting Started with Training (CeresTrain, 2.5B-position recipe)

A cold-start guide for training a Ceres transformer net using the
`c1_256_12_v1_off_pt3` recipe. Targets a single 24 GB NVIDIA GPU (e.g.
RTX 4090) on Windows with WSL2.

This doc is intentionally end-to-end: prerequisites, data prep, configs,
launch, monitor, resume. If something fails, the recipe was validated
2026-05-12 by training the pt2 1.5B FINAL net to completion.

---

## 1. Prerequisites

### Hardware

- **1× NVIDIA GPU with ≥ 24 GB VRAM.** Validated on RTX 4090.
  At batch size 2048 in bf16, peak allocation is ~22 GB.
- **≥ 64 GB system RAM.** TPG dataloader workers fit in a few GB each;
  the host JSON-config / log paths are negligible.
- **SSD for TPG output and checkpoints.** ~50 GB for the trained net's
  checkpoints over the run; faster disks help during checkpoint saves.
  TPG input can live on slower drives (HDD ok for read-only).
- **Fast storage for TPG input** improves throughput. A 4090 saturates
  at ~9 K positions/sec wall-clock, which is roughly 84 MB/s of TPG
  read; even a SATA SSD covers it.

### Software

- **Windows 11** with **WSL2** (Ubuntu recommended).
- **CUDA 12.x** drivers on Windows host; WSL inherits the driver.
- **Python 3.10+** in WSL with **PyTorch 2.7 or newer** (CUDA 12.x build)
  and the listed extras (TensorBoard, NumPy, etc.). The current pinned
  requirements file targets PyTorch 2.11.x, which is what's actively
  validated; 2.7–2.10 also work with minor flexibility in the pin file.
- **.NET 10** on Windows (for the C# launcher and the Ceres binary, if
  you want the Spectre live status UI).

Install Python deps inside WSL:

```bash
cd /mnt/c/Users/<you>/source/repos/CeresTrain/src/CeresTrainPy
pip3 install --user -r requirements-2.11.txt
```

`requirements-2.11.txt` is the actively validated pin set. If you already
have a working PyTorch 2.7+ install you'd rather keep, you can install
only the non-torch extras by manually picking the lines you need from
that file.

### Repositories

```bash
git clone https://github.com/lepned/CeresTrain.git
git clone https://github.com/lepned/Ceres.git    # optional — needed for inference/EB
```

---

## 2. Training data (TPG files)

You need a TPG corpus. Two paths:

### A. Convert existing Lc0 T80 TARs

If you have downloaded LC0 training TARs (~1.5 GB each):

```powershell
$env:TPG_NUM_THREADS = "3"      # see "HDD note" below
& "C:/.../CeresTrain.exe" gen-tpg `
    --tar-dir D:/LZGames/T80 `
    --tpg-dir E:/T80_tpg `
    --num-tpg-sets 1
```

This produces 16 ZSTD-compressed `.zst` shards in the target dir, total
~65 GB, containing ~200 M training positions.

**HDD note.** The `TPG_NUM_THREADS` env var was added because the
hardcoded default (`16 + min(ProcessorCount, 50)`) causes head-thrashing
on spinning disks. On a single HDD source, **NumThreads=3 peaks
throughput**; 8+ threads collapses it by 3×. NVMe/SSD sources can use
the default. See `CHANGELOG.md` 2026-05-12 entry.

For Chess960-only extraction, add `--frc-only`.

### B. Use an existing TPG corpus

Just point the trainer at a directory containing `*.zst` TPG files. The
trainer auto-discovers files at the start of each pass, so you can add
new shards mid-run.

### Puzzle TPG corpus (optional secondary)

The training config supports mixing in a secondary TPG corpus (e.g.
labeled Lichess puzzles) at a fixed batch ratio. See
`GETTING_STARTED_PUZZLES.md` for how to generate puzzle TPGs.

---

## 3. Configuration files

CeresTrain reads **five JSON files per training run**, all named
`<config_id>_ceres_<part>.json`. For the pt3 recipe these live in
`F:/cout/configs/`:

```
c1_256_12_v1_off_pt3_ceres_exec.json         # device, dtype, run ID
c1_256_12_v1_off_pt3_ceres_net.json          # architecture
c1_256_12_v1_off_pt3_ceres_opt.json          # optimizer, schedule, losses
c1_256_12_v1_off_pt3_ceres_data.json         # TPG sources + ratios
c1_256_12_v1_off_pt3_ceres_monitoring.json   # eval / dump settings
```

### Key settings in `_opt.json`

| field | pt3 value | what it does |
|---|---:|---|
| `NumTrainingPositions` | `2500000000` | total positions to train on (2.5 B) |
| `BatchSizeForwardPass` | `2048` | batch size; bf16 fits at 24 GB VRAM |
| `BatchSizeBackwardPass` | `2048` | gradient accumulation equal to forward |
| `Optimizer` | `"AdamW"` | only Beta1/Beta2 are read; Beta3/Alpha are vestigial here |
| `LearningRateBase` | `0.0005` | peak LR after warmup |
| `LRWarmupPhaseMultiplier` | `0.1` | warmup = 10 % of run = 250 M positions |
| `LRBeginDecayAtFractionComplete` | `0.7` | linear decay starts at 70 % of run, runs to LR 0 at end |
| `WeightDecay` | `0.01` | AdamW decoupled WD |
| `Beta1`, `Beta2` | `0.9`, `0.95` | AdamW momentum terms |
| `GradientClipLevel` | `0` | no global grad clipping |
| `CheckpointFrequencyNumPositions` | `100000000` | save every 100 M positions (25 saves total) |
| `CheckpointResumeFromFileName` | `null` | fresh start; set to a path to resume |
| `PyTorchCompileMode` | `"default"` | `max-autotune` OOMs at bs=2048 on 4090 — don't use |
| `LossPolicyMultiplier` | `1.5` | policy CE loss weight |
| `LossValueMultiplier` | `0.5` | WDL CE loss weight |
| `LossValue2Multiplier` | `0.04` | secondary value loss |
| `LossValueDMultiplier` | `0.1` | deblundered value loss |
| `LossMLHMultiplier` | `0.05` | moves-left-head loss |
| `LossUNCMultiplier` | `0.005` | value-uncertainty loss |
| `LossUncertaintyPolicyMultiplier` | `0.02` | policy-uncertainty loss |
| `LossQDeviationMultiplier` | `0.01` | search-Q deviation loss |

### Key settings in `_net.json`

| field | pt3 value | what it does |
|---|---:|---|
| `ModelDim` | `256` | transformer hidden width |
| `NumLayers` | `12` | transformer depth |
| `NumHeads` | `8` | attention heads (head_dim = 32) |
| `FFNMultiplier` | `4` | FFN inner width = 4× ModelDim |
| `FFNActivationType` | `"SwiGLU"` | gated FFN (better than GELU/ReLU empirically) |
| `HeadsActivationType` | `"Mish"` | smooth activation in policy/value heads |
| `HeadWidthMultiplier` | `4` | heads are 4× the body width |
| `NormType` | `"RMSNorm"` | (vs LayerNorm) |
| `PreNorm` | `false` | post-norm (norm applied after residual add) |
| `NonLinearAttention` | `true` | extra non-linear path in attention |
| `SmolgenDim` | `0` | smolgen disabled (saves ~80% params for ~3% EPS, slightly worse acc) |
| `LoRARankDivisor` | `0` | LoRA disabled (fresh-start, not fine-tuning) |
| `UseQKV` | `true` | combined QKV projection |

Total parameters: **~16.66 M**.

### Key settings in `_data.json`

| field | example value | what it does |
|---|---|---|
| `SourceType` | `"DirectFromPositionGenerator"` | streaming TPG reader (not pre-buffered) |
| `TrainingFilesDirectory` | `"/mnt/e/T80_tpg"` | primary TPG dir (WSL path) |
| `TrainingFilesDirectory2` | `"/mnt/c/Dev/Chess/Puzzles/.../tpg"` | optional secondary corpus |
| `RatioSet1ToSet2` | `49` | 49 primary batches per 1 secondary batch |
| `TARPositionSkipCount` | `30` | sample 1 in 30 positions when traversing TPG |
| `FractionQ` | `0` | use deblundered WDL targets, not Q estimates |
| `NumTPGFilesToSkip` | `0` | skip first N files of deterministic sort (rarely useful) |

### Key settings in `_exec.json`

| field | pt3 value | what it does |
|---|---|---|
| `ID` | `"c1_256_12_v1_off_pt3"` | run identifier; checkpoint filenames include this |
| `DeviceType` | `"cuda"` | only `cuda` is real |
| `DeviceIDs` | `[0]` | GPU index visible to PyTorch |
| `DataType` | `"BFloat16"` | bf16 mixed precision via `torch.amp.autocast` |
| `UseHistory` | `true` | include 7 previous board positions in input |

To run on a different physical GPU, leave `DeviceIDs: [0]` and set the
WSL env `CUDA_VISIBLE_DEVICES=1` (or whichever) before launching. The
config's index 0 then maps to your chosen physical GPU.

---

## 4. Launch training

Two ways. Pick whichever works on your machine.

### A. Direct Python via WSL (no Spectre UI, easier to script)

```bash
wsl bash -c "cd /mnt/c/Users/<you>/source/repos/CeresTrain/src/CeresTrainPy && \
  nohup python3 -u train.py \
    /mnt/f/cout/configs/c1_256_12_v1_off_pt3 \
    /mnt/f/cout \
    > /mnt/f/cout/pt3_train.log 2>&1 &"
```

- First positional arg: config basename (no `_ceres_*.json` suffix).
- Second positional arg: output root. Checkpoints land in
  `<out>/nets/`, TensorBoard logs in `<out>/logs/`.
- `nohup ... &`: detaches; the run survives terminal exit.

### B. C# launcher (with Spectre live status UI)

```powershell
& "C:/.../CeresTrain.exe" train --config c1_256_12_v1_off_pt3 --host wsl
```

- Reads `f:/cout/CeresTrainHosts.json` to find the `wsl` host entry,
  then spawns the same Python command as method A inside WSL, wrapped
  in a Spectre live status table.
- Caveat: redirecting the launcher's stdout to a file breaks Spectre
  with "handle is invalid". Use method A if you need redirection.

### Override `NumTrainingPositions` on the command line

If you want to smoke-test the recipe without committing to the full
2.5 B, pass `--num-pos 100000000` to the C# launcher. It rewrites the
config file in place, then the Python reads the rewritten value.
(Direct Python invocation does **not** support this — the config is the
source of truth.)

---

## 5. Monitor progress

### Log

Each training step logs a line like:

```
TRAIN: <positions>, <total_loss>, <value_loss>, <policy_loss>, <policy_acc%>, <value_acc%>, ..., <lr>
```

Tail it:

```bash
tail -f /mnt/f/cout/pt3_train.log
```

### TensorBoard

```bash
tensorboard --logdir /mnt/f/cout/logs --port 6006
```

Open http://localhost:6006 in a browser. Loss/accuracy curves, gradient
norms, learning rate, etc. update live.

### Sanity checks at key milestones

| milestone | expected |
|---|---|
| First minute | `NUM_PARAMETERS 16664669` logged; `DATASET WORKER 0 PROCESSING TPG FILE` appears |
| First training step | total loss ~2.5–3.0, policy acc ~50–65 % |
| Warmup end (250 M / 10 %) | LR reaches base `5e-4`, total loss < 1.5 |
| Decay start (1.75 B / 70 %) | LR begins linear decay toward 0 |
| End (2.5 B / 100 %) | final checkpoint saved, ONNX exported |

### Throughput

Steady-state on a 4090 with `default` compile is **~8.9 K positions/sec**.
Multiply by `BatchSizeBackwardPass` (2048) to get steps/sec ≈ 4.3.

If you see < 5 K pos/sec sustained, suspect:
- TPG storage I/O (HDD, network share, etc.)
- Other process holding the GPU (Windows desktop with heavy GPU UI use,
  another training run, an inference engine, etc.)
- WSL memory pressure (close other heavy WSL processes)

---

## 6. Resume from a checkpoint

If training is interrupted (kernel update, power loss, kill):

1. Find the latest checkpoint:

   ```bash
   ls -t /mnt/f/cout/nets/ckpt_GodLikePC_c1_256_12_v1_off_pt3_*
   ```

2. Edit `c1_256_12_v1_off_pt3_ceres_opt.json`, set:

   ```json
   "CheckpointResumeFromFileName": "/mnt/f/cout/nets/ckpt_GodLikePC_c1_256_12_v1_off_pt3_<step>"
   ```

3. Relaunch with the same command as section 4. The trainer restores
   model weights, optimizer state, and step counter, then continues.

The per-launch shuffle seed re-randomizes file iteration order on each
restart, so resumes don't bias toward beginning-of-file records. To
make a resume fully deterministic (debugging only), set
`CERES_SHUFFLE_SEED=<int>` in the launch env.

---

## 7. Output artifacts

When training completes, expect:

```
/mnt/f/cout/nets/
  ckpt_GodLikePC_c1_256_12_v1_off_pt3_100000000        # every 100 M
  ckpt_GodLikePC_c1_256_12_v1_off_pt3_200000000
  ...
  ckpt_GodLikePC_c1_256_12_v1_off_pt3_2500000000       # final
  GodLikePC_c1_256_12_v1_off_pt3_2500000000.ts         # TorchScript trace
  GodLikePC_c1_256_12_v1_off_pt3_2500000000.onnx       # ONNX export
```

The `.onnx` is what Ceres / EB Console / Lc0 consume for inference.

---

## 8. Expected wall-clock

At 8.9 K positions/sec on a 4090:

| phase | positions | wall time |
|---|---:|---:|
| Warmup (10 %) | 0 → 250 M | ~7.8 h |
| Constant LR (60 %) | 250 M → 1.75 B | ~46.8 h |
| Linear decay (30 %) | 1.75 B → 2.5 B | ~23.4 h |
| **Total** | **2.5 B** | **~78 h ≈ 3.25 days** |

Plan for ~3.5 days if you want buffer for one or two interruption/resume
cycles. Checkpointing every 100 M means a worst-case loss of ~3 hours
on a hard crash.

---

## 9. Empirical reference (pt2 1.5B FINAL net)

The same recipe at 1.5 B positions produced lepned pt2 1.5B FINAL.
Compared to the older C1-256-10 baseline (15.45 M params, trained with a
prior recipe), pt2 wins on a Lichess-puzzle AvgR 2350 evaluation
(in-band relative to current puzzle corpus, Nodes=1, 2000 puzzles):

| metric | pt2 1.5 B FINAL | C1-256-10 | Δ |
|---|---:|---:|---:|
| Policy Perf | 2426 | 2368 | **+58 Elo** |
| Policy acc | 60.8 % | 52.6 % | +8.2 pp |
| pTop3 Perf | 2849 | 2769 | **+80 Elo** |
| pTop3 acc | 94.7 % | 91.8 % | +2.9 pp |
| Value Perf | 2426 | 2368 | **+58 Elo** |
| Value acc | 60.8 % | 52.7 % | +8.1 pp |

The expectation for pt3 (2.5 B, longer hold, longer decay) is a further
+5–15 Elo, with diminishing returns past 1.5 B.

---

## 10. Common failure modes

| symptom | likely cause | fix |
|---|---|---|
| `CUDA out of memory` mid-compile | `PyTorchCompileMode: "max-autotune"` with bs=2048 | use `"default"` |
| "Spectre handle is invalid" exception from launcher | redirected stdout breaks Spectre | use direct Python (method A) |
| `lr_scheduler.step()` before `optimizer.step()` warning | benign; pre-existing | ignore |
| Training freezes after N hours | rare WSL+CUDA hang | resume from latest checkpoint |
| EPS suddenly drops below 5 K/sec | another GPU process started | check `nvidia-smi`; pause other workloads |
| `KeyError: grad_2.0_norm_total` | pre-fix `_grad_norm` helper | fixed in current `train.py`; ensure you're on latest `main` |

---

## 11. References

- `CHANGELOG.md` — recent changes and their rationale.
- `GETTING_STARTED_PUZZLES.md` — building the secondary puzzle TPG corpus.
- `training_config.txt` (in `artifacts/release/net10.0/`) — single-page
  shareable summary of the recipe.
- The five `c1_256_12_v1_off_pt3_ceres_*.json` files in `F:/cout/configs/`
  are the source of truth for the current recipe.
