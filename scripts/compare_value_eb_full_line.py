#!/usr/bin/env python3
"""EB-aligned value-head harness. Walks the full puzzle line: a puzzle counts as
'solved' only if the net plays the correct UCI move at EVERY solver-to-move position
in the line. This replaces compare_value_eb_exact.py (which only tested the first
solver move) and brings the in-tree harness into agreement with EB Console.

EB selection:
  - Sort puzzles by rating DESCENDING
  - Find first puzzle with rating == start_rating (e.g., 2710)
  - Take next N puzzles (which spans start_rating downward)

EB scoring:
  - Each puzzle line: opp setup move, then alternating solver/opp moves.
  - 'Solved' = engine's `go value` bestmove matches the puzzle's solver move at
    EVERY solver-to-move position. Any mismatch = puzzle wrong, break.
  - (TODO: add 'accept any mate at the final solver move' lenience to fully match
    EB. For now, strict match — slight conservative bias.)

Configure CONFIGS at top to test multiple nets vs orig in one pass.
"""
import subprocess, time, threading, queue, csv

CERES = r"C:/Dev/Chess/Ceres/artifacts/release/net10.0/Ceres.exe"   # match EB binary

_COMMON = {
    "SyzygyPath":       "D:/sygyzy",
    "VerboseMoveStats": "true",
    "LogLiveStats":     "true",
    "UCI_ShowWDL":      "true",
    "RamLimitMb":       "10096",
}
def _cfg(net, device="GPU:0#TensorRTNative"):
    return {"Network": net, "Device": device, **_COMMON}

CONFIGS = {
    "orig": _cfg("C:/Dev/Chess/Networks/CeresNet/C1-640-34-I8.onnx"),
    "v52":  _cfg("C:/Dev/Chess/CeresTrain/nets/lepdev_c1_640_34_v52_folded_trt.onnx"),
    "v58":  _cfg("C:/Dev/Chess/CeresTrain/nets/lepdev_c1_640_34_v58_folded_trt.onnx"),
    "v60":  _cfg("C:/Dev/Chess/CeresTrain/nets/lepdev_c1_640_34_v60_folded_trt.onnx"),
    "v65":  _cfg("C:/Dev/Chess/CeresTrain/nets/lepdev_c1_640_34_v65_folded_trt.onnx"),
}
CSV_PATH    = "C:/Dev/Chess/Puzzles/lichess_db_puzzle_july2025.csv"
N_PUZZLES   = 5000   # default; bump to 5K+ only when nets are within ~1 pp
START_RATING = 2710


def load_puzzles(n, start_rating):
    """EB-style: sort by rating desc, find first rating == start_rating, take n."""
    all_p = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try: rating = int(row.get("Rating", "0"))
            except: continue
            fen = row.get("FEN"); mv = row.get("Moves")
            if not fen or not mv: continue
            mvs = mv.split()
            # Need at least one solver move (so >= 2 total: opp setup + solver1)
            if len(mvs) < 2: continue
            all_p.append((row.get("PuzzleId",""), fen, mvs, rating))
    # Sort by rating descending; stable so within same rating, CSV order preserved
    all_p.sort(key=lambda p: -p[3])
    start = next((i for i, p in enumerate(all_p) if p[3] == start_rating), 0)
    selected = all_p[start : start + n]
    if selected:
        rs = [p[3] for p in selected]
        print(f"  Selected {len(selected)} puzzles starting at rating {start_rating}: "
              f"min={min(rs)}, max={max(rs)}, avg={sum(rs)/len(rs):.1f}")
    return selected


def run_net(label, opts, puzzles):
    print(f"\n=== [{label}] {len(puzzles)} puzzles (full-line, EB-aligned) ===")
    proc = subprocess.Popen([CERES, "UCI"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    q = queue.Queue()
    def pump():
        try:
            for line in proc.stdout: q.put(line.rstrip())
        finally: q.put(None)
    threading.Thread(target=pump, daemon=True).start()
    def send(c):
        try: proc.stdin.write(c+"\n"); proc.stdin.flush(); return True
        except (OSError, ValueError): return False
    def read_until(pred, timeout=60):
        lines=[]; t=time.time()+timeout
        while time.time()<t:
            try: line=q.get(timeout=max(0.05, t-time.time()))
            except queue.Empty: return lines, False
            if line is None: return lines, False
            lines.append(line)
            if pred(line): return lines, True
        return lines, False

    send("uci"); read_until(lambda l: l == "uciok", 30)
    for k, v in opts.items():
        send(f"setoption name {k} value {v}")
    send("isready")
    _, ok = read_until(lambda l: l == "readyok", 600)
    if not ok:
        print("  [engine did not readyok]")
        try: proc.kill()
        except: pass
        return 0, 0

    correct = 0
    total = 0
    moves_tested = 0
    moves_correct = 0
    for pi, (pid, start_fen, all_moves, rat) in enumerate(puzzles):
        # solver moves are at odd indices: 1, 3, 5, ... (move 0 is opp setup).
        solver_indices = list(range(1, len(all_moves), 2))
        if not solver_indices:
            continue
        total += 1

        puzzle_solved = True
        for si in solver_indices:
            # Build prefix: all moves before this solver move, applied to start_fen.
            prefix = " ".join(all_moves[:si])
            cmd_pos = f"position fen {start_fen} moves {prefix}"
            expected_move = all_moves[si]

            if not send("ucinewgame"): puzzle_solved = False; break
            if not send("isready"): puzzle_solved = False; break
            _, ok = read_until(lambda l: l == "readyok", 10)
            if not ok: puzzle_solved = False; break
            if not send(cmd_pos): puzzle_solved = False; break
            if not send("go value"): puzzle_solved = False; break
            lines, ok = read_until(
                lambda l: l.startswith("bestmove") or "Unhandled" in l or "Exception" in l, 30)
            if not ok: puzzle_solved = False; break
            if any(("Unhandled" in l or "NullReference" in l) for l in lines):
                puzzle_solved = False; break
            bm = next((l for l in lines if l.startswith("bestmove")), "")
            parts = bm.split()
            picked = parts[1] if len(parts) >= 2 else ""

            moves_tested += 1
            if picked == expected_move:
                moves_correct += 1
            else:
                puzzle_solved = False
                break  # any wrong solver move = puzzle wrong

        if puzzle_solved:
            correct += 1

        if (pi + 1) % 500 == 0:
            print(f"  [{pi+1}] solved: {correct}/{total} = {100*correct/max(1,total):.2f}%  "
                  f"per-move: {moves_correct}/{moves_tested} = {100*moves_correct/max(1,moves_tested):.2f}%")

    print(f"  FINAL: {correct}/{total} puzzles solved = {100*correct/max(1,total):.2f}%")
    print(f"         {moves_correct}/{moves_tested} per-move correct = {100*moves_correct/max(1,moves_tested):.2f}%")
    try: send("quit"); proc.wait(timeout=10)
    except Exception:
        try: proc.kill()
        except Exception: pass
    return correct, total


pz = load_puzzles(N_PUZZLES, START_RATING)
print(f"Loaded {len(pz)} puzzles (EB-style sort: rating desc starting at {START_RATING})")
results = {}
for label, opts in CONFIGS.items():
    results[label] = run_net(label, opts, pz)

print(f"\n=== Summary (EB-aligned full-line, start_rating={START_RATING}, n={N_PUZZLES}) ===")
for label, (c, t) in results.items():
    print(f"  {label:<12s}  {c}/{t} puzzles solved = {100*c/max(1,t):.2f}%")
