# Changelog

Notable changes to CeresTrain. Latest commits at the top.

## 2026-05-12

### TPG generation: `--frc-only` flag + `TPG_NUM_THREADS` env override (`f513033`)
- New `--frc-only` flag on `gen-tpg` inverts the standard variant filter, producing a Chess960-only TPG corpus from existing TARs. Ceres now supports 960 castling, but the default TPG path still filters FRC out, so this gives a path to build a 960 training set.
- `NumThreads` can be overridden via `TPG_NUM_THREADS=N` env var. The hardcoded default (`16 + min(ProcessorCount, 50)`) catastrophically head-thrashes HDD source dirs — empirically, on a D: HDD, throughput peaks at **3 threads (~40 MB/s)** vs **8+ threads (~14 MB/s)** due to seek contention.

### Trainer: Lightning Fabric removed; plain PyTorch only (`f26b6f6`)
- `lightning.fabric.Fabric` and `pl.LightningModule` fully removed from `src/CeresTrainPy/`. Two of three observed training hangs traced to Fabric defaults:
  - `clip_gradients` with `error_if_nonfinite=True` (Fabric default).
  - The recursive `_apply_to_collection_slow` walk in `setup_dataloaders`' auto-move.
- Both classes of failure are now structurally impossible.
- `train.py`: replaced ~30 `fabric.*` sites with idiomatic torch (`autocast`, `SummaryWriter`, `loss.backward`, `clip_grad_norm_` without `error_if_nonfinite`). New `_grad_norm` and `_move_batch_to_device` helpers replace Lightning utilities.
- `ceres_net.py`: `CeresNet` base class `pl.LightningModule` → `nn.Module`; constructor takes `writer: SummaryWriter` instead of `fabric: Fabric`; ~25 `self.fabric.log()` sites collapsed to a single `_log` helper.
- `save_model.py`: `fabric.save` → `torch.save` of state_dicts (avoids pickling SummaryWriter's thread lock); `model.to_torchscript` → `torch.jit.trace`.
- Validated by a 1.5B-position production training run end-to-end on a 4090.

## 2026-05-08

### `save_model`: emit `.lora_*.bin` under env-var-driven LoRA modes (`203c64a`)
- Widens the `.bin` save gate to fire whenever any of `CERES_LORA_ATTN_RANK_DIV`, `CERES_LORA_FFN_RANK_DIV`, `CERES_LORA_TRANSFORMER_RANK_DIV`, `CERES_LORA_HEADFRONT_RANK_DIV`, or `CERES_LORA_SMOLGEN_RANK_DIV` is set, not only when `Opt_LoRARankDivisor > 0`. Closes a gap from the earlier env-var LoRA work.
- Adds `scripts/extract_lora_bin_from_ckpt.py`: post-hoc extractor that reads a checkpoint and writes the equivalent `.lora_<step>.bin`, for recovering artifacts from nets trained before this fix.

## 2026-05-06

### KL-anchor + body-LoRA env-var controls + TSB scaffolding + puzzle pipeline configs (`09d5115`)
- KL-anchor losses for puzzle fine-tuning (PONLY pipeline): policy and value KL terms with configurable reference checkpoint.
- LoRA controls split per-region via environment variables: `CERES_LORA_ATTN_RANK_DIV` (attention QKV+output), `CERES_LORA_FFN_RANK_DIV` (FFN matrices), `CERES_LORA_HEADFRONT_RANK_DIV` (head-front), `CERES_LORA_SMOLGEN_RANK_DIV` (smolgen); legacy `CERES_LORA_TRANSFORMER_RANK_DIV` retained.
- TSB (Token-State-Bottleneck) scaffolding for upcoming experiments.
- Puzzle pipeline configs added for replay + label-fast modes.

## 2026-05-03

### Puzzle TPG pipeline byte-perfect parity fixes (`0e8ba43`)
- Byte-117 plane restored (was being mis-written).
- History planes filled with real game history instead of placeholders.
- V-only WDL floor logic preserved through the puzzle path.
- SwiGLU activation fixes in the conversion code paths.
- All pre-2026-05-03 puzzle TPG shards must be regenerated to be byte-compatible with the trainer.

## 2026-05-01

### Tools and scripts
- `export_v8_uint8_mish.py`: V8 weight export with uint8 quantization + Mish activations (`13f085a`).
- `scripts/clamp_wdl.py`: clamp utility for WDL targets in labeled JSONL (`9ef0f44`).
- `scripts/onnx_bootstrap/`: ONNX → ckpt bootstrap pipeline for reconstructing training checkpoints from exported ONNX nets (`e0248e4`).
- `GETTING_STARTED_PUZZLES.md`: cold-start guide for the puzzle replay pipeline (`acf03a1`, updated `c20c507`).
- Opp-to-move calibration pipeline + LoRA control upgrades (`80693ca`).
- OppDef enricher: log search-cascade exceptions; recovery utilities for the state-accumulation bug at long runs (`01af885`).

## 2026-04-28

### Action-head value enrichment (`a820c63`)
- Theme-floor solver targets for the action head.
- LoRA splits for finer fine-tuning control.

## Empirical notes (not commit-tied)

### Validated production recipe (2026-05-11)
- Config family: `c1_256_12_v1_off_pt2`
- Arch: C1-256-12, SwiGLU FFN, smolgen OFF, RMSNorm, 16.66M params
- Optimizer: AdamW, LR 5e-4, WD 0.01, β(0.9, 0.95); 10% warmup, decay starts at 50% complete
- Batch: 2048, bf16
- Training: 1.5B positions on T80 self-play TPGs (49 parts) mixed with puzzle TPGs (1 part)
- Result: lepned pt2 1.5B FINAL net beats lepned C1-256-10 (15.45M params) by ~+58 Elo (Policy/Value) and +80 Elo (pTop3) at AvgR 2350 on Lichess puzzles, despite being slightly larger but trained with a more modern recipe.

### Next-run recipe (pt3, in progress)
- Same arch, schedule tweaks: LR decay starts at 0.7 (was 0.5), checkpoint frequency 100M (was 25M).
- `max-autotune` compile mode tried and rejected: OOMs at bs=2048 on a 4090 (24 GB VRAM) during cudagraph capture. Reverted to `default`.
- Target: 2.5B positions, projected ~3.25 days on a single 4090 at ~8.9K positions/sec steady-state.
