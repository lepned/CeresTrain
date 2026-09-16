#!/bin/bash
# Switch the live 8B run onto a ONE-FIELD config change at the next checkpoint.
#
# Generalised from switch_focal_gamma.sh so the same, verified machinery serves the whole
# sequence of single-knob experiments (HLGaussWeight now, ValueFocalGamma next).
#
#   KEY=HLGaussWeight VAL=0 MINPOS=3100000000 SEG=seg8_hlgauss_off \
#     setsid nohup bash ~/scan/switch_single_knob.sh > /tmp/switch.log 2>&1 &
#
#   ID=<id>_ng KEY=MoveTokenLayers MINPOS=3900000000 SEG=seg10_dec6 \
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
# ID is overridable: arm B (the live run since 3.3B) is the "_ng" training id.
ID=${ID:-prod_1024_10_f2_h16_t80t91_8B_hr07d05}
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
# Compare configs as JSON, KEY BY KEY -- not as text. A text diff reports formatting as
# a change (adding a key puts a comma on the previous line) and cannot exempt a field, so
# it flagged CheckpointResumeFromFileName, which this script rewrites itself a few lines
# below and which is therefore ALWAYS stale in the repo once a switch has run. Both bit a
# real launch on 2026-09-15. Key-wise comparison is exact: every key whose VALUE differs
# must name a KEY field, and at least one must.
python3 - "$REPO/configs" "$OUT/configs" "$ID" "$KEY" <<'PYGUARD'
import json, os, re, sys
repo, out, rid, keyre = sys.argv[1:5]
pat = re.compile(keyre)
IGNORE = {"CheckpointResumeFromFileName"}   # rewritten below, never a real knob
matched, other = [], []
for f in ("data", "exec", "monitoring", "net", "opt"):
    n = f"{rid}_ceres_{f}.json"
    a = json.load(open(os.path.join(repo, n), encoding="utf-8"))
    b = json.load(open(os.path.join(out, n), encoding="utf-8"))
    for k in sorted(set(a) | set(b)):
        if k in IGNORE:
            continue
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if va != vb:
            (matched if pat.search(k) else other).append(f"{f}.{k}: {vb!r} -> {va!r}")
for m in matched:
    print("    CHANGE  " + m)
for o in other:
    print("    UNEXPECTED  " + o)
if other or not matched:
    print(f"ABORT: {len(matched)} intended change(s), {len(other)} unexpected")
    sys.exit(1)
print(f"pre-flight OK: {len(matched)} change(s), all naming /{keyre}/")
PYGUARD
if [ $? -ne 0 ]; then
  say "ABORT: config pre-flight failed (see above)"
  exit 1
fi

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
grep -E "LOAD_CHECKPOINT|DECODER GROWN|post-move attn|MT_EXPORT_MAX|HL-GAUSS|FOCAL hardness|dropped checkpoint tensors|Muon honors|wd partition|lr-schedule|datastream. resume|collective timeout|static_graph" \
     $LOG | tail -16 | sed 's/^/    /'
say "procs: $(pgrep -fc "train.py $ID")  tracebacks: $(grep -c 'Traceback\|Watchdog caught' $LOG)"
say "pos: $(grep '^TRAIN: ' $LOG | tail -1 | cut -d, -f1)"
