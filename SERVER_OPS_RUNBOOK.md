# Server runbook: AIME 4xA100 (shared with Kovax)

Operational reference: connection, status, launch/resume, restart.
Updated 2026-08-21.

## Connection

```
ssh -p 21031 admin@gate05.aime.info
```

- Key: `~/.ssh/id_ed25519` (same as the Hetzner box, no passphrase).
- From Windows: connect from **WSL** (git-bash has its own known_hosts and
  fails the first time; alternatively use
  `-o StrictHostKeyChecking=accept-new`).
- **Shared box.** Kovax runs his own jobs — check `nvidia-smi` before taking
  GPUs. Our tmux/byobu session is named `lepned`; his is `1`. TensorBoard
  ports: 6006-6008 are his, **6009 is ours**.

## Directories

| what | where |
|---|---|
| repo | `~/repos/CeresTrain` |
| venv | `~/cerestrain-env` (not activated — the launcher finds torchrun itself) |
| outputs | `~/ceres_out/{configs,nets,tblogs,logs}` |
| T91 (22 shards) | `/mnt/lepned/T91` |
| puzzles, 4-shard (USE THIS ONE) | `/mnt/lepned/puzzles_2600up_v3/tpg4` |
| puzzles, original 5-shard (untouched, for other users) | `/mnt/lepned/puzzles_2600up_v3/tpg` |

## Status checks

```
grep -a '^TRAIN:' ~/ceres_out/logs/<ID>_launch.log | tail -1        # position count
pgrep -af "train.py"                                                # processes
nvidia-smi                                                          # GPU
grep -a '^cpu MHz' /proc/cpuinfo | awk '$4<1500' | wc -l            # throttle check
```

- Throughput: measure over **>= 180 s** (60 s windows lie because of
  checkpoint pauses).
- Throttle history: the 399 MHz lock was BIOS-related, fixed 2026-08-18.
  Individual cores dipping transiently is normal; the fault signature was MANY
  cores, PERSISTENT, and the same cores each time.

## Launch

```
cd ~/repos/CeresTrain
CERES_NUM_DATASET_WORKERS=<W> nohup bash scripts/server/launch_ddp.sh \
  <ID> ~/ceres_out <NPROC> <GPU-list> <PORT> > /tmp/l_<ID>.out 2>&1 &
```

- Configs must live at
  `~/ceres_out/configs/<ID>_ceres_{net,opt,data,exec,monitoring}.json`
  (the launcher prefers these over the repo copies).
- **Preflight requirement: shards >= NPROC × workers PER corpus.**
  4 ranks → `CERES_NUM_DATASET_WORKERS=1` (the puzzle corpus has 4 shards).
  2 ranks → workers=2 is fine.
- Ports: 29500 (and 29600 for a second arm).
- **Shape rule:** one big run = 1 job × 4 GPUs (half the wall clock).
  An A/B pair = 2+2 on the NVLink pairs (GPU 0,1 / 2,3) — ~15 % better overall.
  A small net (256x10) on 4 GPUs = COLLAPSE (comm-bound); never use that shape.
- Batch: `BatchSizeForwardPass` is GLOBAL and split across ranks. For 512 nets:
  fwd 4096 on 4 ranks (1024/rank ≈ 23 GB/GPU = safe; 2048/rank = OOM on 40 GB).

## Resume / restarting a run

Set in `<ID>_ceres_opt.json`:

```
"CheckpointResumeFromFileName": "/home/admin/ceres_out/nets/ckpt_a4000-21bn11_<ID>_<numpos>"
```

and launch as usual. Restored: weights, optimizer state (Muon momentum),
`num_pos` (→ the LR schedule and checkpoint cadence continue), and the data
stream is fast-forwarded.

- **The EMA shadow is NOT persisted** — it re-warms over ~`EMAMaxN` periods
  (~20M positions). Do not compare EMA exports for the first ~20M after a
  resume.
- Aux-width guard: 137-channel nets require `CERES_AUX_FEATURES_PER_SQUARE=0`
  in the environment.
- The first export after a resume is weight-identical to the source
  checkpoint (a good sanity check).

## Restarting the SERVER (reboot)

After a reboot nothing of ours is persistent:

1. Check clocks: `grep -a '^cpu MHz' /proc/cpuinfo | sort -n | head`
   (all >= 1500).
2. Relaunch training with resume from the last checkpoint (see above).
3. TensorBoard if needed:
   `nohup tensorboard --logdir ~/ceres_out/tblogs --port 6009 --host 127.0.0.1 &`
   and tunnel from your machine:
   `ssh -p 21031 -L 6009:localhost:6009 admin@gate05.aime.info -N`
   → http://localhost:6009. Use `ServerAliveInterval=60` — tunnels otherwise
   die silently overnight.

## Export pitfalls (important)

- Validate ONNX exports with `onnxruntime.InferenceSession(path)` — one
  second, green or red. TRT will happily build from a DEFECTIVE graph with no
  error message (a subtly wrong engine → "plays strangely").
- The Cast/FP16 bug is fixed in `save_model.py` (b2c6af6) — but a RUNNING
  process started before a fix holds the old code in memory: its exports must
  be re-exported afterwards with
  `python recover_export.py <ID> ~/ceres_out <numpos>` (first copy the
  checkpoint to `ckpt_lepdev_<ID>_<numpos>` — `recover_export` expects the
  `lepdev` prefix, or set `CERES_HOST_PREFIX`).
- Early checkpoints from hot-LR runs (512 @ 1.2e-3, 100-200M) are FP16-fragile:
  fine in TRT, garbage in strict fp16. Do not use them in ORT-based
  measurements.

## Miscellaneous

- `EMAPeriodSteps` counts OPTIMIZER STEPS; 1 step = `BatchSizeBackwardPass`
  positions REGARDLESS of rank count. The preflight warning about dividing by
  nproc is WRONG — ignore it.
- The zstd CLI is not installed on the server; use the `zstandard` Python
  module from the venv.
- Disk: `/` (916 GB) = our outputs; `/mnt` (7.3 TB, ~99 % full) = corpora. Ask
  Kovax before putting anything on `/mnt`.
