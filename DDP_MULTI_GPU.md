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

## Optimizer notes

- **Muon** (production): `muon.py` updates each parameter from its local `.grad`;
  DDP averages grads across ranks *before* `optimizer.step()`, so Muon operates
  on the global-mean gradient. No double reduction — its `import torch.distributed`
  is unused. ✅ DDP-safe.
- **AdEMAMixShampoo**: calls `all_reduce` itself. **Review before using under
  DDP** — it may double-reduce or need a process-group guard. Not yet validated.

## Known caveats / to validate in simulation first

- **4-board sequence mode + gradient accumulation**: static_graph + the no-sync
  accumulation path is the least-exercised combination. If a 4-board config with
  `BatchSizeBackwardPass > BatchSizeForwardPass` errors at the first optimizer
  step, test with `CERES_DDP_STATIC_GRAPH=0` and/or a config without accumulation
  to isolate it.
- **Dataloader is the bottleneck on this corpus** (~59% GPU util single-GPU =
  zstd decompress + TPG parse). DDP gives each rank its own dataloader workers,
  but all ranks read from the same disk — disk read + decompress throughput may
  cap the speedup below `nproc`× . Provision fast NVMe and raise
  `CERES_NUM_DATASET_WORKERS`; measure actual scaling before assuming linear.
- `recover_export.py` / `ExportOnly` runs are single-process by design (DDP is
  disabled when `Exec_ExportOnly` is set) — run those without `torchrun`.
