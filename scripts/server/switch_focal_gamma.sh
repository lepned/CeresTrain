#!/bin/bash
# Switch the live 8B run onto ValueFocalGamma = 0.25 at the next checkpoint.
#
# STRICT SINGLE KNOB: the five configs in the repo were verified byte-identical (modulo
# line endings) to the running ones before this change, and only ValueFocalGamma moved
# 0 -> 0.25. Anything else differing means someone edited a config in between -- the
# pre-flight diff below will refuse the switch in that case.
#
# Run it detached:   setsid nohup bash ~/scan/switch_focal_gamma.sh > /tmp/switch_fg.log 2>&1 &
#
# Why 0.25 and not 1.0: weight_i = (KL_i + 1e-3)**gamma with the loss normalised by
# sum(w), so gamma redistributes emphasis rather than changing magnitude. Measured on
# the 3.0B net over 24.7K rows, the value-KL distribution is extremely skewed (median
# 0.0075, 55% of rows below 0.01), so effective batch size (ESS) collapses with gamma:
#   0.25 -> 82%   0.50 -> 47%   0.75 -> 22%   1.00 -> 10%   2.00 -> 0.8%
# The value target is already ESS-limited (redundancy ~100x policy, driven by game
# count), so gamma 1 would shrink an already-binding constraint by 10x. 0.25 doubles
# top-decile emphasis (10% -> 20.5%) while keeping 82% of the batch effective.

set -u
OUT=/mnt/lepned/ceres_out
ID=prod_1024_10_f2_h16_t80t91_8B_hr07d05
REPO=~/repos/CeresTrain
LOG=$OUT/logs/${ID}_launch.log
SEG=seg8_focalgamma025
MINPOS=${MINPOS:-3100000000}      # switch at the first checkpoint at/after this

say() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---- 1. wait for a checkpoint >= MINPOS with sidecar + BOTH exports size-stable ----
say "waiting for checkpoint >= $MINPOS"
CK=""
for i in $(seq 1 360); do
  for c in $(ls -1 $OUT/nets/ckpt_a4000-21bn11_${ID}_* 2>/dev/null | grep -v datastream | sort); do
    pos=${c##*_}
    [ "$pos" -ge "$MINPOS" ] 2>/dev/null || continue
    onnx=$OUT/nets/a4000-21bn11_${ID}_${pos}.onnx
    ema=$OUT/nets/a4000-21bn11_${ID}_${pos}ema.onnx
    side=$c.datastream.json
    [ -s "$onnx" ] && [ -s "$ema" ] && [ -s "$side" ] || continue
    s1=$(stat -c %s "$c")$(stat -c %s "$onnx"); sleep 30
    s2=$(stat -c %s "$c")$(stat -c %s "$onnx")
    [ "$s1" = "$s2" ] && { CK=$c; break; }
  done
  [ -n "$CK" ] && break
  sleep 60
done
[ -z "$CK" ] && { say "TIMEOUT: no stable checkpoint >= $MINPOS"; exit 1; }
say "checkpoint ready: $CK"

# ---- 2. pull the repo and PROVE the only change is ValueFocalGamma ----
cd $REPO || exit 1
git fetch origin && git reset --hard origin/main || exit 1
say "repo at $(git rev-parse --short HEAD)"
CHANGED=0
for f in data exec monitoring net opt; do
  n=${ID}_ceres_$f.json
  if ! diff -q <(tr -d '\r' < $REPO/configs/$n) <(tr -d '\r' < $OUT/configs/$n) >/dev/null 2>&1; then
    say "DIFF in $f:"; diff <(tr -d '\r' < $REPO/configs/$n) <(tr -d '\r' < $OUT/configs/$n) | sed 's/^/    /'
    CHANGED=$((CHANGED+1))
  fi
done
# Exactly one file (opt) may differ, and only on the ValueFocalGamma line.
ONLY=$(diff <(tr -d '\r' < $REPO/configs/${ID}_ceres_opt.json) <(tr -d '\r' < $OUT/configs/${ID}_ceres_opt.json) \
       | grep -E '^[<>]' | grep -vc 'ValueFocalGamma')
if [ "$CHANGED" -ne 1 ] || [ "$ONLY" -ne 0 ]; then
  say "ABORT: expected exactly one config to differ on the ValueFocalGamma line only"
  say "       (files differing: $CHANGED, non-gamma diff lines: $ONLY)"
  exit 1
fi
say "pre-flight OK: single-knob change confirmed"

# ---- 3. install configs, point the resume at the real checkpoint ----
cp $REPO/configs/${ID}_ceres_*.json $OUT/configs/ || exit 1
# NOTE: CheckpointResumeFromFileName lives in the OPT config (config.py:152 reads it from
# config_opt), NOT exec. Sed-ing exec would match nothing and silently restart from 0.
sed -i "s#\"CheckpointResumeFromFileName\":.*#\"CheckpointResumeFromFileName\": \"$CK\",#" \
    $OUT/configs/${ID}_ceres_opt.json
if ! grep -q "\"CheckpointResumeFromFileName\": \"$CK\"," $OUT/configs/${ID}_ceres_opt.json; then
  say "ABORT: resume path was not rewritten to $CK -- refusing to relaunch from scratch"
  exit 1
fi
python3 -c "import json;d=json.load(open('$OUT/configs/${ID}_ceres_opt.json'));print('    resume =',d['CheckpointResumeFromFileName']);print('    gamma  =',d['ValueFocalGamma'])" || exit 1

# ---- 4. stop the old ranks, preserve the log ----
say "stopping ranks"
pkill -f "train.py $ID"
sleep 20
pkill -9 -f "train.py $ID" 2>/dev/null
sleep 5
say "procs left: $(pgrep -fc "train.py $ID")"
cp $LOG $OUT/logs/${ID}_launch_${SEG}_pre.log
mv $LOG $OUT/logs/${ID}_launch_${SEG}.log

# ---- 5. relaunch ----
say "relaunching"
bash ~/scan/launch_ea_wave.sh --nproc 4 --gpus 0,1,2,3 $ID
sleep 180

# ---- 6. verify the boot lines that matter ----
say "boot check:"
grep -E "LOAD_CHECKPOINT|FOCAL hardness|Muon honors|wd partition|lr-schedule|datastream. resume|collective timeout" \
     $LOG | tail -12 | sed 's/^/    /'
if grep -q "FOCAL hardness weighting enabled: gamma=0.25" $LOG; then
  say "OK: focal weighting ACTIVE at gamma 0.25"
else
  say "WARNING: focal weighting NOT confirmed in the log -- check before trusting the run"
fi
say "procs: $(pgrep -fc "train.py $ID")  tracebacks: $(grep -c 'Traceback\|Watchdog caught' $LOG)"
