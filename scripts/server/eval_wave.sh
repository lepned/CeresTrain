#!/usr/bin/env bash
# Dev-box side of the overnight edge-aux campaign: fetch the final EMA nets of
# the given server arms and puzzle them (EB 4-band, served blend).
#
#   bash scripts/server/eval_wave.sh <POS> <ARM_ID> [<ARM_ID> ...]
#   e.g. bash scripts/server/eval_wave.sh 100003840 srv_320_ea_A srv_320_ea_B
#
# Run from the repo root in git-bash. scp goes through WSL (its known_hosts has
# the server); the path is passed INSIDE the wsl bash -c string (git-bash
# mangles bare /mnt/c arguments). Nets land in the usual Server/ folder.
set -uo pipefail
POS="${1:?usage: eval_wave.sh <POS> <ARM_ID>...}"; shift
DST_WIN="C:/Dev/Chess/Networks/CeresNet/lepned/Server"
DST_WSL="/mnt/c/Dev/Chess/Networks/CeresNet/lepned/Server"
PREFIX="a4000-21bn11"
SK=".claude/skills/eval-net/eval_net.py"
for ID in "$@"; do
  # EVAL_RAW=1 reads the RAW net (the EMA auto-window at a 50M ckpt cadence averages
  # across the whole decay window of a 100M run — raw is the honest read there).
  if [ "${EVAL_RAW:-0}" = "1" ]; then NET="${PREFIX}_${ID}_${POS}.onnx"; TAGSUF="_raw"; else NET="${PREFIX}_${ID}_${POS}ema.onnx"; TAGSUF=""; fi
  if [ ! -s "$DST_WIN/$NET" ]; then
    wsl bash -c "scp -q -P 21031 admin@gate05.aime.info:/mnt/lepned/ceres_out/nets/$NET '$DST_WSL/'" \
      || { echo "FETCH FAILED $NET"; continue; }
  fi
  echo "=== $ID ($NET)"
  PYTHONIOENCODING=utf-8 python "$SK" puzzle "$DST_WIN/$NET" "ea_${ID#srv_320_ea_}${TAGSUF}" 2>&1 | tail -12
done
