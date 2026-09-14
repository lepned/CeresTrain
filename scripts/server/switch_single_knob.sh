#!/bin/bash
# Switch the live 8B run onto a ONE-FIELD config change at the next checkpoint.
#
# Generalised from switch_focal_gamma.sh so the same, verified machinery serves the whole
# sequence of single-knob experiments (HLGaussWeight now, ValueFocalGamma next).
#
#   KEY=HLGaussWeight VAL=0 MINPOS=3100000000 SEG=seg8_hlgauss_off \
#     setsid nohup bash ~/scan/switch_single_knob.sh > /tmp/switch.log 2>&1 &
#
# THE GUARD IS THE POINT: it aborts unless exactly one config file differs from the
# running one and the only differing lines mention KEY. Two knobs at once makes the
# readout uninterpretable whichever way the metric moves -- the failure that made the
# first hard-replay arm worthless.
#
# Resume note: CheckpointResumeFromFileName lives in the OPT config (config.py:152 reads
# it from config_opt), NOT exec. Sed-ing exec matches nothing and silently restarts from 0,
# so the rewrite is verified below and the script aborts if it did not take.

set -u
OUT=/mnt/lepned/ceres_out
ID=prod_1024_10_f2_h16_t80t91_8B_hr07d05
REPO=~/repos/CeresTrain
LOG=$OUT/logs/${ID}_launch.log
KEY=${KEY:?set KEY: one field name, or a regex alternation for a deliberate
             multi-field change, e.g. "HLGaussWeight|ValueFocalGamma"}
MINPOS=${MINPOS:-3100000000}
SEG=${SEG:-seg8_change}

say() { echo "[$(date '+%H:%M:%S')] $*"; }
say "target: fields matching /$KEY/ at first checkpoint >= $MINPOS  (seg $SEG)"

# ---- 1. wait for a checkpoint >= MINPOS with sidecar + BOTH exports size-stable ----
CK=""
for i in $(seq 1 360); do
  for c in $(ls -1 $OUT/nets/ckpt_a4000-21bn11_${ID}_* 2>/dev/null | grep -v datastream | sort); do
    pos=${c##*_}
    [ "$pos" -ge "$MINPOS" ] 2>/dev/null || continue
    onnx=$OUT/nets/a4000-21bn11_${ID}_${pos}.onnx
    ema=$OUT/nets/a4000-21bn11_${ID}_${pos}ema.onnx
    [ -s "$onnx" ] && [ -s "$ema" ] && [ -s "$c.datastream.json" ] || continue
    s1=$(stat -c %s "$c")$(stat -c %s "$onnx"); sleep 30
    s2=$(stat -c %s "$c")$(stat -c %s "$onnx")
    [ "$s1" = "$s2" ] && { CK=$c; break; }
  done
  [ -n "$CK" ] && break
  sleep 60
done
[ -z "$CK" ] && { say "TIMEOUT: no stable checkpoint >= $MINPOS"; exit 1; }
say "checkpoint ready: $CK"

# ---- 2. pull the repo and PROVE this is a single-knob change ----
cd $REPO || exit 1
git fetch origin && git reset --hard origin/main || exit 1
say "repo at $(git rev-parse --short HEAD)"
CHANGED=0
for f in data exec monitoring net opt; do
  n=${ID}_ceres_$f.json
  if ! diff -q <(tr -d '\r' < $REPO/configs/$n) <(tr -d '\r' < $OUT/configs/$n) >/dev/null 2>&1; then
    say "differs: $f"
    diff <(tr -d '\r' < $REPO/configs/$n) <(tr -d '\r' < $OUT/configs/$n) | sed 's/^/    /'
    CHANGED=$((CHANGED+1))
    OTHER=$(diff <(tr -d '\r' < $REPO/configs/$n) <(tr -d '\r' < $OUT/configs/$n) \
            | grep -E '^[<>]' | grep -vEc "$KEY")
  fi
done
if [ "$CHANGED" -ne 1 ] || [ "${OTHER:-1}" -ne 0 ]; then
  say "ABORT: expected exactly ONE file differing, on $KEY lines only"
  say "       (files differing: $CHANGED, non-$KEY diff lines: ${OTHER:-n/a})"
  exit 1
fi
say "pre-flight OK: single-knob change on $KEY confirmed"

# ---- 3. install configs, point the resume at the real checkpoint ----
cp $REPO/configs/${ID}_ceres_*.json $OUT/configs/ || exit 1
sed -i "s#\"CheckpointResumeFromFileName\":.*#\"CheckpointResumeFromFileName\": \"$CK\",#" \
    $OUT/configs/${ID}_ceres_opt.json
grep -q "\"CheckpointResumeFromFileName\": \"$CK\"," $OUT/configs/${ID}_ceres_opt.json || {
  say "ABORT: resume path not rewritten -- refusing to relaunch from scratch"; exit 1; }
python3 -c "
import json,re
d=json.load(open('$OUT/configs/${ID}_ceres_opt.json'))
print('    resume =', d['CheckpointResumeFromFileName'])
for k in d:
    if re.search(r'$KEY', k): print('   ', k, '=', d[k])" || exit 1

# ---- 4. stop the old ranks, preserve the log ----
say "stopping ranks"
pkill -f "train.py $ID"; sleep 20
pkill -9 -f "train.py $ID" 2>/dev/null; sleep 5
say "procs left: $(pgrep -fc "train.py $ID")"
mv $LOG $OUT/logs/${ID}_launch_${SEG}.log

# ---- 5. relaunch ----
say "relaunching"
bash ~/scan/launch_ea_wave.sh --nproc 4 --gpus 0,1,2,3 $ID
sleep 210

# ---- 6. verify the boot lines that matter ----
say "boot check:"
grep -E "LOAD_CHECKPOINT|HL-GAUSS|FOCAL hardness|dropped checkpoint tensors|Muon honors|wd partition|lr-schedule|datastream. resume|collective timeout|static_graph" \
     $LOG | tail -14 | sed 's/^/    /'
say "procs: $(pgrep -fc "train.py $ID")  tracebacks: $(grep -c 'Traceback\|Watchdog caught' $LOG)"
say "pos: $(grep '^TRAIN: ' $LOG | tail -1 | cut -d, -f1)"
