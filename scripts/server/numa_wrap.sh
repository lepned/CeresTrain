#!/usr/bin/env bash
# Pin a torchrun rank to the NUMA node local to ITS GPU, then exec the trainer.
#
# Used as the torchrun entrypoint:
#   torchrun --nproc_per_node=4 scripts/server/numa_wrap.sh train.py <ID> <OUT>
#
# Why: on a dual-socket box the GPUs are split across sockets (e.g. 0,1 on
# socket 0 and 2,3 on socket 1). Without binding, the Linux scheduler is free to
# place a rank's process — and its dataloader workers, and NCCL's staging
# buffers — on the socket that does NOT own its GPU. Every batch then crosses
# the inter-socket link twice: once to parse into remote memory, once to reach
# the device. Two ranks on one socket never expose this; four ranks spanning
# both do, which is the classic "2 GPUs fine, 4 GPUs collapse" signature.
#
# NCCL tuning cannot fix it: the misplacement happens at allocation time, in
# both the dataloader and the collective's host buffers.
#
# Set CERES_NUMA_BIND=0 to disable (useful for A/B measuring the effect).
set -euo pipefail

if [ "${CERES_NUMA_BIND:-1}" = "0" ] || ! command -v numactl > /dev/null 2>&1; then
  [ "${CERES_NUMA_BIND:-1}" != "0" ] && echo "[numa] numactl not found — running unbound" >&2
  exec python3 "$@"
fi

RANK="${LOCAL_RANK:-0}"
# GPU index this rank will use. Honour CUDA_VISIBLE_DEVICES so a job restricted
# to a subset (e.g. "2,3") maps rank 0 -> physical GPU 2.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f$((RANK + 1)))
else
  GPU="$RANK"
fi

NODE=""
BUS=$(nvidia-smi --id="$GPU" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | tr 'A-Z' 'a-z' | sed 's/^0000//' || true)
if [ -n "$BUS" ]; then
  # nvidia-smi prints 00000000:41:00.0 -> sysfs wants 0000:41:00.0
  SYS="/sys/bus/pci/devices/0000:${BUS#*:}/numa_node"
  [ -r "$SYS" ] && NODE=$(cat "$SYS")
fi

if [ -z "$NODE" ] || [ "$NODE" = "-1" ]; then
  echo "[numa] rank $RANK gpu $GPU: no NUMA affinity reported — running unbound" >&2
  exec python3 "$@"
fi

echo "[numa] rank $RANK -> gpu $GPU -> NUMA node $NODE (cpu+mem bound)" >&2
exec numactl --cpunodebind="$NODE" --membind="$NODE" python3 "$@"
