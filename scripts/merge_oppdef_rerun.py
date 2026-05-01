#!/usr/bin/env python3
"""Merge OppDef records from a re-run into the original enriched output.

The original `labeled_with_oppdef.jsonl` has Standard records interleaved
with OppDef children, but some Standard records are MISSING their OppDef
children (the ones that failed search). This script:

  1. Reads the rerun output (which contains pairs Standard + OppDef for
     the previously-failed records).
  2. Builds a map (PuzzleId, Standard.FEN, Standard.SolutionUci) → OppDef record.
  3. Walks the original output. For each Standard record, if it doesn't already
     have an OppDef child immediately following AND we have one in the map,
     insert the OppDef from the map directly after.
  4. Writes the merged output to a new file.

Usage:
  python merge_oppdef_rerun.py <orig_enriched.jsonl> <rerun_enriched.jsonl> <out_merged.jsonl>
"""
import json
import sys

if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(1)

ORIG = sys.argv[1]
RERUN = sys.argv[2]
OUT = sys.argv[3]


# 1. Build (PuzzleId, FEN, SolutionUci) -> OppDef record map from re-run output.
print(f"Reading rerun: {RERUN}")
oppdef_map = {}
last_std_key = None
with open(RERUN) as f:
    for line in f:
        r = json.loads(line)
        if r.get('Kind') == 0:
            last_std_key = (r['PuzzleId'], r['FEN'], r['SolutionUci'])
        elif r.get('Kind') == 1 and last_std_key is not None:
            oppdef_map[last_std_key] = line  # store raw json line (preserves formatting)
            last_std_key = None
print(f"  collected {len(oppdef_map):,} OppDef records from rerun")


# 2. Walk original. For each Standard, if no OppDef partner already AND we
#    have one in our map, insert it.
print(f"Merging into: {ORIG}")
print(f"Writing to:   {OUT}")

n_input_records = 0
n_added_oppdef = 0

with open(ORIG) as fin, open(OUT, 'w') as fout:
    pending_std = None
    pending_std_key = None
    for line in fin:
        r = json.loads(line)
        n_input_records += 1
        if r.get('Kind') == 0:
            # Flush any prior pending Standard that didn't get its OppDef yet.
            if pending_std is not None:
                fout.write(pending_std)
                # Try to insert from map.
                if pending_std_key in oppdef_map:
                    fout.write(oppdef_map[pending_std_key])
                    n_added_oppdef += 1
            pending_std = line
            pending_std_key = (r['PuzzleId'], r['FEN'], r['SolutionUci'])
        elif r.get('Kind') == 1:
            # OppDef immediately following a Standard → already paired.
            if pending_std is not None:
                fout.write(pending_std)
                pending_std = None
                pending_std_key = None
            fout.write(line)
        else:
            # Other Kinds: pass through.
            if pending_std is not None:
                fout.write(pending_std)
                if pending_std_key in oppdef_map:
                    fout.write(oppdef_map[pending_std_key])
                    n_added_oppdef += 1
                pending_std = None
                pending_std_key = None
            fout.write(line)

    # Flush trailing.
    if pending_std is not None:
        fout.write(pending_std)
        if pending_std_key in oppdef_map:
            fout.write(oppdef_map[pending_std_key])
            n_added_oppdef += 1

print(f"\n=== Results ===")
print(f"  Input records read:   {n_input_records:,}")
print(f"  OppDef records added: {n_added_oppdef:,}")
print(f"  (expected: {len(oppdef_map):,})")
