# Multi-GPU training (DistributedDataParallel)

`train.py` supports data-parallel training across multiple GPUs via PyTorch
DistributedDataParallel (DDP). It is **opt-in**: launched the normal way (a
plain `python train.py ...`) nothing changes — it runs single-process /
single-GPU exactly as before. DDP activates only when launched under `torchrun`
(which sets `WORLD_SIZE`/`RANK`/`LOCAL_RANK` in the environment).

The model is small (~52M params) and fits comfortably on one GPU, so this is
**pure throughput scaling** (data-parallel), not model sharding. Each rank holds
a full copy of the model on its own GPU, trains on a disjoint shard of the data,
and DDP all-reduces (averages) the gradients across ranks before each optimizer
step so the copies stay identical.

## Launch — real multi-GPU (e.g. 4×A100)

```bash
export CERES_AUX_FEATURES_PER_SQUARE=4          # required for V3-aux nets
torchrun --standalone --nproc_per_node=4 \
    train.py <CONFIG_ID> /path/to/OUTPUTS_DIR
```

- `--nproc_per_node=N` = one process per GPU. Use N = number of GPUs.
- Default collective backend is **NCCL** (the right choice for real GPUs).
- Each rank logs a `[ddp] rank=i/N ...` line at startup; only **rank 0** writes
  checkpoints, ONNX/TS exports, tensorboard, and the parsed `TRAIN:` lines.

## Launch — single-GPU simulation (validate before deploying)

You can exercise the entire DDP code path on a single-GPU box. NCCL refuses to
put two ranks on one GPU, so use the **gloo** (CPU) backend; the code maps every
rank onto `cuda:0` automatically.

```bash
export CERES_AUX_FEATURES_PER_SQUARE=4
export CERES_DDP_BACKEND=gloo
torchrun --standalone --nproc_per_node=2 \
    train.py <CONFIG_ID> /path/to/OUTPUTS_DIR
```

This validates: process-group init, gradient all-reduce, the no-sync gradient
accumulation path, rank-0-only checkpoint/export gating, dataset file-sharding,
and global position counting / LR-schedule timing. It does **not** measure real
NCCL performance or give any speedup (everything runs on one GPU), and it needs
N model+optimizer+activation copies to fit in VRAM — trivial at this model size;
just use a small `Opt_BatchSizeForwardPass`.

> ⚠ **The simulation is necessary but not sufficient.** With `gloo` on CPU there
> is no CUDA context, so it cannot see CUDA/NCCL-specific faults. A real 2-GPU
> NCCL run found a crash the gloo simulation had passed cleanly minutes earlier
> (a collective executing inside a forked DataLoader worker). **Always do one
> short real multi-GPU run before committing a production window to a new box.**

To confirm equivalence, run a short single-GPU baseline and a 2-rank gloo run
from the same config/seed and check the `TRAIN:` loss curve and the position
count advance at the same rate.

## Semantics (what stays identical to single-GPU)

- `Opt_BatchSizeForwardPass` and `Opt_BatchSizeBackwardPass` are **global**. Each
  rank processes `BatchSizeForwardPass // world_size` per micro-step, so the
  effective optimization batch — and therefore the LR schedule — is unchanged.
  **No LR retune is needed** vs. a single-GPU run of the same config.
- `num_pos` counts **global** positions (per-rank × world_size), so
  `NumTrainingPositions`, checkpoint cadence, and LR decay timing all match a
  single-GPU run rather than running `world_size`× too long.
- Requirements: `BatchSizeForwardPass` divisible by `nproc_per_node`, and at
  least `nproc_per_node` TPG files in the corpus (ideally ≥ nproc × dataset
  workers) so every rank gets a shard.

## Environment knobs

| Var | Default | Meaning |
|---|---|---|
| `CERES_DDP_BACKEND` | `nccl` | Collective backend. Use `gloo` for single-GPU simulation. |
| `CERES_DDP_STATIC_GRAPH` | `1` if 4-board else `0` | `static_graph=True` DDP. Auto-on for `TrainOn4BoardSequences` (the action head calls the model 3–4× before one backward, which the default reducer mishandles). |
| `CERES_DDP_FIND_UNUSED` | `1` | `find_unused_parameters`. Needed when a head/output never reaches the loss. Ignored when static_graph is on. Set `0` once a config is verified to use every parameter (slightly faster). |

## Reproducible A/B runs (seeds)

Two config keys make an ablation attributable. Both live in `_ceres_opt.json`
and are omitted by default (omitting them reproduces the historical behaviour
exactly).

| Key | Governs |
|---|---|
| `TorchSeed` | weight init, dropout masks, hard-replay / mirror row sampling, file-mirror coin flips. Applied identically on every rank. |
| `ShuffleSeed` | file ordering, hence **which shards each rank reads**. |

Set **both, to the same values, in both arms** of an A/B and the only difference
left is the mechanism under test. Until these existed nothing called
`torch.manual_seed` anywhere in `train.py`, which is where the ±2-3 Elo
"seed noise" between same-config arms came from.

Note this is reproducibility *between runs*, not bit-exact determinism — that
would additionally need `torch.use_deterministic_algorithms` and disabling cuDNN
autotuning, which costs throughput and is deliberately not done.

## Corpus requirements (read this before launching)

Files are **partitioned** across ranks, not shared, and the rank slice is split
again by DataLoader worker. The real precondition is:

```
shards_per_corpus  >=  nproc_per_node * CERES_NUM_DATASET_WORKERS
```

This applies to **every** corpus, including a small puzzle secondary — that one
is the usual offender (a 5-shard puzzle set is fine at 4 ranks × 1 worker and
starves at 4 × 2). Violations now fail loudly at startup; before, every rank got
an empty file list and the run simply hung.

Shards that do not divide evenly are **never read for the whole run** (the
ordering is fixed per run), and startup prints how many were dropped. Prefer a
shard count divisible by `nproc_per_node`.

## EMA under DDP

`EMAPeriodSteps` counts **optimizer steps**, and each step consumes
`world_size`× more positions than single-GPU. To reproduce a 1-GPU-validated
recipe, **divide it by the GPU count** (e.g. 100 → 25 on 4 GPUs). The shadow
lives on rank 0 only; every rank holds identical weights, so nothing is lost.

## Mixed V2/V3 corpora

Per-corpus shard format is config-driven, so it travels with the run instead of
living in the launching shell:

```jsonc
"TPGV3": 1,                 // primary = V3 (141 bytes/square)
"SquareBytes2": 137,        // secondary = V2
"AuxFeaturesPerSquare": 0   // REQUIRED as soon as any corpus is V2
```

A 137-byte corpus carries no aux bytes, so a single V2 dataset forces aux 0 for
the entire run. Mismatches fail at startup with the exact key to change.

## First experiment on a new multi-GPU box

`configs/srv_256_10_tactical_{ctrl,cf}_*` are a ready-made 2-arm ablation, one
arm per GPU pair. They differ in **exactly one key** (`VisEdgeFamilies`) — the
check/flight tactical edge families — with `TorchSeed`/`ShuffleSeed` pinned
identically so the gate delta is attributable to that key alone.

Background: on a 5M puzzle-only smoke, check/flight gave **+383 value Elo
in-dist at zero inference cost**, replicated across three runs. But that was
measured where value labels come from puzzle outcomes; on a game corpus the
labels are Q/z, so the effect is **unvalidated outside puzzle-only data**. This
A/B answers that in a few hours instead of risking a full production window.

Before launching, edit in **both** `_ceres_data.json` files:

```jsonc
"TrainingFilesDirectory":  "/EDIT/ME/PATH/TO/GAME_CORPUS_TPG",
"TrainingFilesDirectory2": "/EDIT/ME/PATH/TO/PUZZLE_CORPUS_TPG"
```

Then launch both arms with one command — it copies the repo configs into the
outputs dir (without overwriting), validates them, and starts each arm on its
own GPU pair with distinct rendezvous ports:

```bash
scripts/server/launch_tactical_ab.sh /path/to/OUTPUTS 2
```

Individual runs, and the validator on its own:

```bash
scripts/server/launch_ddp.sh <CONFIG_ID> /path/to/OUTPUTS 4          # all 4 GPUs
scripts/server/launch_ddp.sh <CONFIG_ID> /path/to/OUTPUTS 2 2,3 29501
scripts/server/preflight.sh  <CONFIG_ID> /path/to/OUTPUTS 4          # check only
```

`preflight.sh` refuses to launch on: unedited `/EDIT/ME` paths, a corpus with
fewer shards than `ranks × workers`, a V2 corpus with `AuxFeaturesPerSquare != 0`,
`BatchSizeForwardPass` not divisible by the rank count, and mirror-consistency
under DDP. It warns about never-read remainder shards and about `EMAPeriodSteps`
needing division by the rank count. Distinct `--master_port` per concurrent job
is mandatory — a shared port makes the second job join the first's process group.

### Corpus format: start on V3, add V2 later without changing the net

The A/B configs ship as `TPGV3: 1` (V3 primary) with `AuxFeaturesPerSquare: 0`.
That combination is deliberate: the V3 reader slices the aux tail, so the model
is **137 channels wide whether the shards are V2 or V3**. Adding a V2 corpus
later is then a one-line change with no architectural consequence and no loss of
comparability against the runs that came before:

```jsonc
"SquareBytes2": 137     // secondary corpus is V2; everything else unchanged
```

Had aux been left at 4, introducing any V2 data would have forced it to 0 and
silently changed the input width — i.e. a different network, not comparable with
earlier gates. (Project measurement is that aux buys nothing anyway.)

### Environment variables: none required

Everything the run needs now lives in its config and travels with it through
resume, ExportOnly and the checkpoint tools. `CERES_*` variables remain only as
fallbacks for configs that predate a key. Optional, if you want them:

| Var | When |
|---|---|
| `CERES_NUM_DATASET_WORKERS` | raise if the dataloader caps throughput (remember the shard requirement scales with it) |
| `CERES_DDP_BACKEND=gloo` | single-GPU simulation only |
| `CERES_DDP_STATIC_GRAPH=1` | required if you enable placement/survival/stvalue/depth-probe aux heads |

In particular `CERES_AUX_FEATURES_PER_SQUARE` no longer needs exporting — it
comes from `AuxFeaturesPerSquare` in the config, which is authoritative over the
environment.

**Pre-registered decision rule:** check/flight proceeds to production if the
value delta at the 200M gate (rg2700, raw weights) is **>= +30 Elo**. Below
that, treat the puzzle-only result as distribution-specific and park the
mechanism. Same-gate noise is ±3 Elo; ≥10 is real.

Two mechanisms are deliberately **off** in these configs: graph-route heads
(`UseGraphRouteHeads`, −8% serving throughput, and adjacency-flavoured
mechanisms have failed twice before) and the iterated refiner (`RefinerIters`,
−4%, only +20 policy at 5M). Enable them only as a follow-up, one at a time.

## Startup checklist

Each of these prints at startup — check them in the log before walking away:

```
[try_shuffle] run-level shuffle seed = ...   # SAME on every rank
[train] TORCH SEED set to ...                # if you pinned it
[ddp] rank=i/N local_rank=i gpu=cuda:i backend=nccl
[ddp] model wrapped in DistributedDataParallel (static_graph=..., find_unused_parameters=...)
```

If ranks print **different** shuffle seeds, stop — the rank slices are not a
partition and the run is silently training on duplicated data. (A startup
`all_reduce` check now fails loudly on this, but verify anyway.)

## Troubleshooting

| Symptom | Cause |
|---|---|
| `expect_autograd_hooks_ INTERNAL ASSERT FAILED` | `static_graph` combined with the no-sync accumulation path. Fixed — sync is left on under `static_graph` (mathematically identical, one all-reduce per micro-step instead of per optimizer step). If it reappears, something re-introduced `require_backward_grad_sync`. |
| `Cannot re-initialize CUDA in forked subprocess` | Something is running a CUDA op (often a collective) inside a **forked DataLoader worker**. Collectives belong in the main process only. |
| Run hangs at startup, no error | A corpus has fewer shards than `ranks × workers`. Now a hard failure; older checkouts hang. |
| `Expected to mark a variable ready only once` | A loss built from a **stashed** tensor (placement/survival/stvalue/depth-probe heads) invisible to the default reducer. Use `CERES_DDP_STATIC_GRAPH=1`. |
| Ranks disagree on shard ordering | `ShuffleSeed` differs across ranks, or the default seed derivation stopped being rank-consistent. |

## Optimizer notes

- **Muon** (production): `muon.py` updates each parameter from its local `.grad`;
  DDP averages grads across ranks *before* `optimizer.step()`, so Muon operates
  on the global-mean gradient. No double reduction — its `import torch.distributed`
  is unused. ✅ DDP-safe.
- **AdEMAMixShampoo**: calls `all_reduce` itself. **Review before using under
  DDP** — it may double-reduce or need a process-group guard. Not yet validated.

## Known caveats

- **Mirror-consistency (`MirrorConsistencyWeight > 0`) is single-GPU only** and
  refuses to start under DDP. Three separate blockers, all real: its second
  forward through the wrapped model needs `static_graph`; whether it runs at all
  is data-dependent per rank (so ranks can disagree and deadlock); and its
  probe/hysteresis mode varies the graph between iterations, which `static_graph`
  forbids. A port must clear all three, and loses the thermostat that makes the
  feature cheap.
- **AdEMAMixShampoo** calls `all_reduce` itself — still unvalidated under DDP.
  Muon (production) is DDP-safe.
- **Dataloader is the bottleneck on this corpus** (~59% GPU util single-GPU =
  zstd decompress + TPG parse). DDP gives each rank its own dataloader workers,
  but all ranks read from the same disk — disk read + decompress throughput may
  cap the speedup below `nproc`× . Provision fast NVMe and raise
  `CERES_NUM_DATASET_WORKERS`; measure actual scaling before assuming linear.
- `recover_export.py` / `ExportOnly` runs are single-process by design (DDP is
  disabled when `Exec_ExportOnly` is set) — run those without `torchrun`.
