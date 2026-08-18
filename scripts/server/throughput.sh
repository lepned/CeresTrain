#!/usr/bin/env bash
# Steady-state throughput of a RUNNING training job, measured by sampling the
# position counter twice.
#
#   scripts/server/throughput.sh <LOGFILE> [SECONDS]      (default 120)
#
# Why not wall-time / total positions: torch.compile spends 1-2 minutes building
# graphs at startup, so on a short smoke that startup IS most of the wall time
# and swamps whatever you were comparing. Sampling a running job measures the
# steady state directly and is immune to it.
set -euo pipefail
LOG="${1:?usage: throughput.sh <LOGFILE> [SECONDS]}"
WINDOW="${2:-120}"

pos_now() { grep '^TRAIN:' "$LOG" 2>/dev/null | tail -1 | sed 's/^TRAIN: *//' | cut -d',' -f1 | tr -d ' '; }

A=$(pos_now); [ -n "$A" ] || { echo "no TRAIN lines yet in $LOG (still compiling?)" >&2; exit 1; }
echo "sampling ${WINDOW}s from position $A ..."
sleep "$WINDOW"
B=$(pos_now)

python3 - "$A" "$B" "$WINDOW" <<'PY'
import sys
a, b, w = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
d = b - a
if d <= 0:
    print(f'no progress in {w:.0f}s (positions still {b:,}) — stalled, or the log only '
          f'writes a TRAIN line every few minutes; retry with a longer window')
    raise SystemExit(1)
print(f'  advanced   : {d:,} positions in {w:.0f}s')
print(f'  throughput : {d/w:,.0f} pos/s   ({d/w*3600/1e6:.2f} M pos/h)')
PY
