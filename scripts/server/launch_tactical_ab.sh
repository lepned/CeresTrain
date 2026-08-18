#!/usr/bin/env bash
# Launch the check/flight A/B: two arms, half the GPUs each, concurrently.
#
#   scripts/server/launch_tactical_ab.sh <OUTPUTS_DIR> [GPUS_PER_ARM]
#
# The arms differ in exactly one config key (VisEdgeFamilies) with
# TorchSeed/ShuffleSeed pinned identically, so the gate delta is attributable
# to the mechanism. Decision rule and background: DDP_MULTI_GPU.md.
set -euo pipefail

OUT="${1:?usage: launch_tactical_ab.sh <OUTPUTS_DIR> [GPUS_PER_ARM]}"
PER="${2:-2}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TOTAL=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$TOTAL" -ge $((PER * 2)) ] || { echo "need $((PER*2)) GPUs, found $TOTAL" >&2; exit 1; }

A_GPUS=$(seq -s, 0 $((PER - 1)))
B_GPUS=$(seq -s, "$PER" $((PER * 2 - 1)))

echo "=== arm A (ctrl, no check/flight) on GPUs $A_GPUS ==="
bash "$REPO/scripts/server/launch_ddp.sh" srv_256_10_tactical_ctrl "$OUT" "$PER" "$A_GPUS" 29500
echo
echo "=== arm B (cf, with check/flight) on GPUs $B_GPUS ==="
bash "$REPO/scripts/server/launch_ddp.sh" srv_256_10_tactical_cf   "$OUT" "$PER" "$B_GPUS" 29501
echo
echo "Both arms launched. Watch:  tail -f $OUT/logs/srv_256_10_tactical_{ctrl,cf}_launch.log"
echo "Gate at 200M, compare VALUE at rg2700 raw. Ship check/flight if delta >= +30 Elo."
