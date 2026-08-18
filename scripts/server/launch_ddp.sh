#!/usr/bin/env bash
# Launch one training run under torchrun, after validating the config.
#
#   scripts/server/launch_ddp.sh <CONFIG_ID> <OUTPUTS_DIR> [NPROC] [GPUS] [PORT]
#
# Examples:
#   scripts/server/launch_ddp.sh srv_256_10_tactical_cf /data/cout 4
#   scripts/server/launch_ddp.sh srv_256_10_tactical_cf /data/cout 2 2,3 29501
#
# GPUS is a CUDA_VISIBLE_DEVICES list; omit to use all. PORT must be UNIQUE per
# concurrent job — two jobs on the same rendezvous port join one process group
# and hang or corrupt each other.
set -euo pipefail

ID="${1:?usage: launch_ddp.sh <CONFIG_ID> <OUTPUTS_DIR> [NPROC] [GPUS] [PORT]}"
OUT="${2:?missing OUTPUTS_DIR}"
NPROC="${3:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
GPUS="${4:-}"
PORT="${5:-29500}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYDIR="$REPO/src/CeresTrainPy"
LOGDIR="$OUT/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/${ID}_launch.log"

# Repo configs are the source of truth; copy any that are missing from the
# outputs dir (train.py reads <OUTPUTS_DIR>/configs/). Never overwrites: an
# edited config in the outputs dir wins, so a live run's record is preserved.
mkdir -p "$OUT/configs"
for f in "$REPO"/configs/${ID}_ceres_*.json; do
  [ -e "$f" ] || continue
  dest="$OUT/configs/$(basename "$f")"
  [ -e "$dest" ] || { cp "$f" "$dest"; echo "copied $(basename "$f") from repo"; }
done

bash "$REPO/scripts/server/preflight.sh" "$ID" "$OUT" "$NPROC"

# Resolve torchrun EXPLICITLY. A bare `torchrun` needs an activated venv, and a
# detached setsid/nohup launch does not inherit one — on the first real server
# deploy this failed with "nohup: failed to run command 'torchrun'". Prefer the
# venv (CERES_VENV, default ~/cerestrain-env), then PATH.
VENV="${CERES_VENV:-$HOME/cerestrain-env}"
if [ -x "$VENV/bin/torchrun" ]; then
  TORCHRUN="$VENV/bin/torchrun"
elif command -v torchrun > /dev/null 2>&1; then
  TORCHRUN="$(command -v torchrun)"
else
  echo "torchrun not found: activate the venv, or set CERES_VENV=/path/to/venv" >&2
  exit 1
fi

[ -n "$GPUS" ] && export CUDA_VISIBLE_DEVICES="$GPUS"
cd "$PYDIR"

echo "launching $ID: nproc=$NPROC gpus=${GPUS:-all} port=$PORT"
echo "  torchrun: $TORCHRUN"
echo "  log: $LOG"
# NUMA binding: on a dual-socket box the GPUs are split across sockets, and an
# unbound rank can land on the socket that does not own its GPU — dataloader
# parsing into remote memory and NCCL staging buffers on the wrong side. That is
# invisible with 2 ranks on one socket and severe with 4 spanning both. The
# wrapper pins each rank to its own GPU's node; CERES_NUMA_BIND=0 disables it
# for an A/B. Falls back to a plain launch if the wrapper is missing.
NUMA_WRAP="$REPO/scripts/server/numa_wrap.sh"
if [ -x "$NUMA_WRAP" ] && [ "${CERES_NUMA_BIND:-1}" != "0" ]; then
  ENTRY=("$NUMA_WRAP" train.py)
else
  ENTRY=(train.py)
fi

setsid nohup $TORCHRUN --standalone --nproc_per_node="$NPROC" --master_port="$PORT"     "${ENTRY[@]}" "$ID" "$OUT" > "$LOG" 2>&1 < /dev/null &

sleep 45
if pgrep -f "train.py $ID" > /dev/null; then
  echo "$ID RUNNING"
  echo "--- startup checklist (all ranks must agree on the seed) ---"
  grep -E '\[numa\]|\[try_shuffle\]|TORCH SEED|\[ddp\]|VISIBILITY EDGE|shards are NEVER' "$LOG" | head -16 || true
else
  echo "$ID FAILED TO START — log tail:"; tail -n 25 "$LOG"; exit 1
fi
