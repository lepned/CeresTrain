"""Post-process a labeled.jsonl file:
  1. Clamp TeacherW/D/L to >=0, renormalize sum to 1.0, recompute V = W - L.
  2. Ensure Lichess solution is unique top of TeacherPolicy by RANK_ONE_EPSILON
     margin. If solution is missing entirely (immediate-no-search mate cases
     where MCGS short-circuits without populating root visit counts for it),
     add it with dominating probability and renormalize.

Fixes two bugs in PuzzleTeacherLabeler observed on 2026-04-30 phase-1:
  - WDL: 52.7% records had L<0 (MCGS aggregation rounding).
  - Policy: 9/92360 records had solution missing from the distribution
    (immediate mate detection short-circuit).

Usage:
  python clamp_wdl.py <input.jsonl> <output.jsonl>
Or with default paths:
  python clamp_wdl.py
"""
import json, sys

DEFAULT_IN  = r"D:/Puzzles/c3_pilot_2600_2700/labeled.jsonl"
DEFAULT_OUT = r"D:/Puzzles/c3_pilot_2600_2700/labeled_clamped.jsonl"
RANK_ONE_EPSILON = 0.03

inp  = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IN
outp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

print(f"Reading: {inp}")
print(f"Writing: {outp}")

n_total = 0
n_clamped_w = 0
n_clamped_d = 0
n_clamped_l = 0
n_renormalized = 0
n_solution_missing = 0
n_solution_below_top = 0

with open(inp,  'r', encoding='utf-8') as f_in, \
     open(outp, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        line = line.rstrip('\n')
        if not line: continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip partial/bad lines
        n_total += 1

        # --- WDL clamp + renormalize ---
        w = float(r.get('TeacherW', 0.0))
        d = float(r.get('TeacherD', 0.0))
        l = float(r.get('TeacherL', 0.0))
        if w < 0: n_clamped_w += 1; w = 0.0
        if d < 0: n_clamped_d += 1; d = 0.0
        if l < 0: n_clamped_l += 1; l = 0.0
        s = w + d + l
        if s > 0:
            if abs(s - 1.0) > 1e-6: n_renormalized += 1
            w /= s; d /= s; l /= s
        r['TeacherW'] = w
        r['TeacherD'] = d
        r['TeacherL'] = l
        r['TeacherV'] = w - l

        # --- Policy rank-1 nudge: ensure Lichess solution is unique top ---
        sol_uci = r.get('SolutionUci', '')
        policy = r.get('TeacherPolicy', [])
        if sol_uci and policy:
            sol_idx = next((i for i, e in enumerate(policy) if e['Uci'] == sol_uci), -1)
            other_max_p = max((e['P'] for e in policy if e['Uci'] != sol_uci), default=0.0)
            if sol_idx < 0:
                # Solution missing from distribution — likely immediate-mate short-circuit.
                # Add with dominating probability.
                n_solution_missing += 1
                policy.append({'Uci': sol_uci, 'P': other_max_p + RANK_ONE_EPSILON})
                # renormalize
                tot = sum(e['P'] for e in policy)
                if tot > 0:
                    for e in policy:
                        e['P'] = e['P'] / tot
            elif policy[sol_idx]['P'] < other_max_p + RANK_ONE_EPSILON:
                # Solution present but not dominant by epsilon — boost it.
                n_solution_below_top += 1
                policy[sol_idx]['P'] = other_max_p + RANK_ONE_EPSILON
                tot = sum(e['P'] for e in policy)
                if tot > 0:
                    for e in policy:
                        e['P'] = e['P'] / tot
            r['TeacherPolicy'] = policy

        f_out.write(json.dumps(r) + '\n')

print(f"\nProcessed: {n_total} records")
print(f"  W<0 clamped: {n_clamped_w} ({n_clamped_w/n_total*100:.1f}%)")
print(f"  D<0 clamped: {n_clamped_d} ({n_clamped_d/n_total*100:.1f}%)")
print(f"  L<0 clamped: {n_clamped_l} ({n_clamped_l/n_total*100:.1f}%)")
print(f"  WDL renormalized (sum drift): {n_renormalized}")
print(f"  Solution missing from policy (added): {n_solution_missing}")
print(f"  Solution below top (boosted):         {n_solution_below_top}")
