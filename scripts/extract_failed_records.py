#!/usr/bin/env python3
"""Identify Standard records whose OppDef enrichment failed (search exception)
and emit them as a fresh labeled_clamped.jsonl for re-processing.

A Standard record is "failed" if:
  - It's followed in the output by another record (so we can compare its
    next neighbor), AND
  - That neighbor is NOT an OppDef record with the same PuzzleId, AND
  - The Standard is not the puzzle's final solver move (which would be
    an expected SkipNoNext, not a failure).

We compare against the puzzle's CSV moves list to determine "is this the
final solver move."

Usage:
  python extract_failed_records.py <input_clamped.jsonl> <enriched_output.jsonl> <csv> <out_failed.jsonl>
"""
import json
import sys
import csv as csvlib

if len(sys.argv) != 5:
    print(__doc__)
    sys.exit(1)

INPUT_CLAMPED = sys.argv[1]
ENRICHED_OUTPUT = sys.argv[2]
CSV_PATH = sys.argv[3]
OUT_FAILED = sys.argv[4]


def fenkey(f):
    return ' '.join(f.split(' ')[:4])


# 1. Load CSV puzzle move-list map (for identifying final-solver-move case).
print(f"Loading CSV: {CSV_PATH}")
csv_map = {}
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        cols = line.split(',')
        if len(cols) >= 3:
            csv_map[cols[0]] = (cols[1], cols[2].split())
print(f"  {len(csv_map):,} puzzles")


# 2. Walk the enriched output to determine which Standard records have an
#    OppDef partner (success) vs not.
print(f"Scanning enriched output: {ENRICHED_OUTPUT}")
records = []
with open(ENRICHED_OUTPUT) as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except:
            pass
print(f"  {len(records):,} total records")

# Build set of "successful" (PuzzleId, Standard.FEN, Standard.SolutionUci) keys.
# A Standard is successful if the next record is its OppDef child (Kind=1) with
# matching PuzzleId.
successful = set()
for i, r in enumerate(records):
    if r.get('Kind') != 0:
        continue
    if i + 1 < len(records):
        nxt = records[i + 1]
        if nxt.get('Kind') == 1 and nxt.get('PuzzleId') == r.get('PuzzleId'):
            successful.add((r['PuzzleId'], r['FEN'], r['SolutionUci']))


# 3. Walk the input clamped file. For each Standard record:
#    - If in successful set → skip (already enriched).
#    - Else, check if it's the puzzle's final solver move using CSV.
#      - If yes → skip (expected SkipNoNext).
#      - Else → emit to OUT_FAILED (real search failure).
print(f"Scanning input: {INPUT_CLAMPED}")
n_input = 0
n_already_succeeded = 0
n_expected_no_next = 0
n_failed = 0
n_no_csv = 0
n_no_match = 0

with open(INPUT_CLAMPED) as fin, open(OUT_FAILED, 'w') as fout:
    for line in fin:
        rec = json.loads(line)
        n_input += 1
        if rec.get('Kind') != 0:
            # Pass-through any non-Standard (shouldn't happen in a clamped file).
            fout.write(line)
            continue

        key = (rec['PuzzleId'], rec['FEN'], rec['SolutionUci'])
        if key in successful:
            n_already_succeeded += 1
            continue

        # Need CSV to know if this is the final solver move.
        pid = rec['PuzzleId']
        if pid not in csv_map:
            n_no_csv += 1
            continue
        csv_fen, csv_moves = csv_map[pid]

        # Find the move index by walking the line. We need StartFen + PriorUciMoves
        # from the record (populated by the labeler now) OR reconstruct.
        sf = rec.get('StartFen') or csv_fen
        pum = rec.get('PriorUciMoves') or ''
        prior = pum.split() if pum else []

        if rec.get('StartFen') and rec.get('PriorUciMoves') is not None:
            move_idx = len(prior)
        else:
            # Reconstruct by walking and matching FEN-key.
            try:
                import chess
                b = chess.Board(sf)
                tk = fenkey(rec['FEN'])
                move_idx = -1
                for i in range(len(csv_moves)):
                    if fenkey(b.fen(en_passant='fen')) == tk and csv_moves[i] == rec['SolutionUci']:
                        move_idx = i
                        break
                    try: b.push_uci(csv_moves[i])
                    except: break
                if move_idx < 0:
                    n_no_match += 1
                    continue
            except ImportError:
                print("python-chess required for legacy records without StartFen")
                sys.exit(1)

        # Final solver move = move_idx + 1 >= len(csv_moves) (no opp reply after).
        if move_idx + 1 >= len(csv_moves):
            n_expected_no_next += 1
            continue

        # Real failure → emit for re-run.
        fout.write(line)
        n_failed += 1

print(f"\n=== Results ===")
print(f"  Input Standard records:                {n_input:,}")
print(f"  Already enriched (paired OppDef):      {n_already_succeeded:,}")
print(f"  Expected SkipNoNext (final solver):    {n_expected_no_next:,}")
print(f"  No CSV / no match:                     {n_no_csv} / {n_no_match}")
print(f"  REAL FAILURES (to re-process):         {n_failed:,}")
print(f"\nWrote: {OUT_FAILED}")
