#!/usr/bin/env bash
# Pre-launch validation for a multi-GPU run. Checks the mistakes that are
# expensive on a rented box: unedited data paths, a corpus with too few shards
# for the rank/worker count (which used to hang rather than fail), and V2/V3
# format combinations that silently misparse records.
#
#   scripts/server/preflight.sh <CONFIG_ID> <OUTPUTS_DIR> <NPROC>
set -euo pipefail

ID="${1:?usage: preflight.sh <CONFIG_ID> <OUTPUTS_DIR> <NPROC>}"
OUT="${2:?missing OUTPUTS_DIR}"
NPROC="${3:-1}"
WORKERS="${CERES_NUM_DATASET_WORKERS:-1}"
CFG="$OUT/configs"
fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }

for part in net data opt exec monitoring; do
  [ -f "$CFG/${ID}_ceres_${part}.json" ] || fail "missing $CFG/${ID}_ceres_${part}.json"
done

python3 - "$CFG" "$ID" "$NPROC" "$WORKERS" <<'PY'
import json, os, sys, glob
cfg_dir, cid, nproc, workers = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
def load(p): return json.load(open(os.path.join(cfg_dir, f'{cid}_ceres_{p}.json')))
data, opt, net = load('data'), load('opt'), load('net')
errs, warns = [], []

# 1. Data paths actually point somewhere.
dirs = [('primary', data.get('TrainingFilesDirectory'))]
if data.get('TrainingFilesDirectory2') and int(data.get('RatioSet1ToSet2') or 0) > 0:
    dirs.append(('secondary', data['TrainingFilesDirectory2']))
is_v6 = str(data.get('SourceType', '')) == 'DirectFromV6'
for label, d in dirs:
    if not d:
        errs.append(f'{label} corpus not set'); continue
    if 'EDIT/ME' in d:
        errs.append(f'{label} corpus is still the placeholder: {d}'); continue
    # DirectFromV6 accepts ';'-separated multi-root lists (see config.split_roots);
    # validate each root's existence but skip the TPG shard-count math — v6 sources
    # shard at the chunk-entry level, not the file level, so nproc*workers does not
    # constrain them the same way.
    roots = [r.strip() for r in str(d).split(';') if r.strip()] if is_v6 else [d]
    if not roots:
        errs.append(f'{label} corpus is an empty/separator-only list: {d!r}'); continue
    if len(roots) > 1 and not is_v6:
        errs.append(f"{label}: ';'-separated lists are only supported for SourceType "
                    f'DirectFromV6 (got {data.get("SourceType")!r}): {d}'); continue
    missing = [r for r in roots if not os.path.isdir(r)]
    if missing:
        errs.append(f'{label} corpus director{"ies" if len(missing)>1 else "y"} do(es) not exist: '
                    + '; '.join(missing)); continue
    if is_v6:
        continue
    shards = [f for f in glob.glob(os.path.join(d, '*.zst'))
              if not f.endswith('.tgt.zst') and not f.endswith('.v7x.zst')]
    need = nproc * workers
    if len(shards) < need:
        errs.append(f'{label} corpus has {len(shards)} shard(s); needs >= nproc*workers = '
                    f'{nproc}*{workers} = {need} (files are partitioned, not shared)')
    elif len(shards) % nproc:
        warns.append(f'{label}: {len(shards) % nproc} of {len(shards)} shards will NEVER be '
                     f'read (not divisible by nproc={nproc})')

# 2. V2/V3 consistency. A 137-byte corpus carries no aux bytes.
aux = int(opt.get('AuxFeaturesPerSquare', 4) or 0)
v2_primary = int(opt.get('TPGV3', 1) or 0) == 0 or int(opt.get('SquareBytes', 141) or 141) == 137
v2_secondary = int(opt.get('SquareBytes2', 0) or 0) == 137
if (v2_primary or v2_secondary) and aux != 0:
    errs.append(f'a V2 (137-byte) corpus is configured but AuxFeaturesPerSquare={aux}; '
                f'V2 shards carry no aux bytes -> set it to 0 for the whole run')

# 3. Batch divisibility.
fwd = int(opt['BatchSizeForwardPass'])
if fwd % nproc:
    errs.append(f'BatchSizeForwardPass {fwd} not divisible by nproc {nproc}')

# 4. Things that refuse to run under DDP.
if nproc > 1:
    if float(opt.get('MirrorConsistencyWeight', 0) or 0) > 0:
        errs.append('MirrorConsistencyWeight > 0 is single-GPU only (see DDP_MULTI_GPU.md)')
    for k in ('SurvivalTargetWeight', 'PlacementValueWeight', 'StValueWeight'):
        if float(opt.get(k, 0) or 0) > 0:
            warns.append(f'{k} > 0 needs CERES_DDP_STATIC_GRAPH=1 under DDP')
    ema = int(opt.get('EMAPeriodSteps', 0) or 0)
    if ema:
        warns.append(f'EMAPeriodSteps={ema} counts OPTIMIZER steps; a 1-GPU recipe of {ema*nproc} '
                     f'corresponds to {ema} here (verify it was divided by nproc={nproc})')

print(f'  corpora     : {", ".join(l for l, _ in dirs)}')
print(f'  format      : TPGV3={opt.get("TPGV3")} SquareBytes2={opt.get("SquareBytes2")} aux={aux}')
print(f'  seeds       : TorchSeed={opt.get("TorchSeed")} ShuffleSeed={opt.get("ShuffleSeed")}')
print(f'  vis families: {net.get("VisEdgeFamilies")}  gates={net.get("VisEdgeGates")}')
print(f'  batch       : {fwd} global -> {fwd // nproc}/rank, backward {opt["BatchSizeBackwardPass"]}')
for w in warns: print(f'  WARN: {w}')
if errs:
    for e in errs: print(f'  ERROR: {e}')
    sys.exit(1)
PY
echo "PREFLIGHT OK: $ID on $NPROC rank(s)"
