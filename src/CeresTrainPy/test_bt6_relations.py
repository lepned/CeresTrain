"""Tests for bt6_relations.py: the 34 BT6 relation channels against two independent slow references.

Run from src/CeresTrainPy (CPU):

    python test_bt6_relations.py [fen file] [positions]

References
1. `reference_definitions`: plain Python written straight from the channel DEFINITIONS (walk the rays square by
   square); shares no code and no formulation with the module.
2. `reference_kernel_port`: a literal transcription of the lc0ex kernel `_prologue_egt_rows_kernel`
   (lczero_triton/bt4/kernels/prologue_egt.py) to Python integers used as 64-bit boards: the same ray masks,
   lowest / highest set bit scans, pin rule and ring dilations. This is what the real BT6 program executes.

Cases: >= 2,000 positions of the FEN file with (a) the engine's FEN-only history (copies of the position, e.p. pawn
put back), (b) the previous position replaced by ANOTHER position of the file (exercises gained / lost), (c) no
history; plus hand-made edge cases with explicit expected cells (pins, x-rays through two blockers, adjacent
kings, pawns on the promotion rank, the start position).
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bt6_planes import encode_fens, expand_planes, read_fens  # noqa: E402
from bt6_relations import BASE_CHANNELS, CHANNEL_NAMES, OPPOSITE_DIRECTION, SLIDER_DIRECTIONS, BT6Relations  # noqa: E402

DEFAULT_FENS = "/mnt/c/lc0_bt6/check/selftest_7040.fen"
MASK64 = (1 << 64) - 1


# --------------------------------------------------------------------------- reference 1: from the definitions
def _walk(square, direction):
  """Squares from `square` along `direction`, nearest first."""
  row, column = divmod(square, 8)
  row_step, column_step = direction
  out = []
  while True:
    row, column = row + row_step, column + column_step
    if not (0 <= row < 8 and 0 <= column < 8):
      return out
    out.append(8 * row + column)


def _chebyshev(a, b):
  return max(abs(a // 8 - b // 8), abs(a % 8 - b % 8))


def _board(pieces):
  """bool [12, 64] -> list of (side, kind) or None per square; side 0 = stm, kind 0..5 = P N B R Q K."""
  board = [None] * 64
  for plane in range(12):
    for square in range(64):
      if pieces[plane][square]:
        board[square] = (plane // 6, plane % 6)
  return board


def _attack_sets(board):
  """attacks[side][i] = set of targets, straight from the piece movement rules."""
  attacks = [[set() for _ in range(64)] for _ in range(2)]
  for square, piece in enumerate(board):
    if piece is None:
      continue
    side, kind = piece
    row, column = divmod(square, 8)
    targets = attacks[side][square]
    if kind == 0:
      forward = 1 if side == 0 else -1
      for dc in (-1, 1):
        r, c = row + forward, column + dc
        if 0 <= r < 8 and 0 <= c < 8:
          targets.add(8 * r + c)
    elif kind == 1:
      for dr, dc in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
        r, c = row + dr, column + dc
        if 0 <= r < 8 and 0 <= c < 8:
          targets.add(8 * r + c)
    elif kind == 5:
      for other in range(64):
        if _chebyshev(square, other) == 1:
          targets.add(other)
    else:
      directions = {2: SLIDER_DIRECTIONS[:4], 3: SLIDER_DIRECTIONS[4:], 4: SLIDER_DIRECTIONS}[kind]
      for direction in directions:
        for target in _walk(square, direction):
          targets.add(target)
          if board[target] is not None:
            break
  return attacks


def reference_definitions(pieces, previous):
  """bool [12, 64] x2 -> bool [34, 64, 64]."""
  board = _board(pieces)
  out = np.zeros((34, 64, 64), dtype=bool)
  attacks = _attack_sets(board)
  for side in range(2):
    for i in range(64):
      for j in attacks[side][i]:
        out[side, i, j] = True
        if board[j] is not None and board[j][0] == side:
          out[2 + side, i, j] = True
  for square, piece in enumerate(board):
    if piece is None or piece[1] not in (2, 3, 4):
      continue
    side, kind = piece
    directions = {2: SLIDER_DIRECTIONS[:4], 3: SLIDER_DIRECTIONS[4:], 4: SLIDER_DIRECTIONS}[kind]
    for direction in directions:
      blockers = 0
      first = None
      for target in _walk(square, direction):
        if blockers == 1:
          out[4 + side, square, target] = True          # strictly beyond the first blocker ...
        if board[target] is not None:
          blockers += 1
          if blockers == 1:
            first = target
          elif blockers == 2:
            out[6 + side, first, target] = True          # pinray [first blocker, first occupied beyond it]
            break                                        # ... up to and including the second
  kings = [[s for s in range(64) if board[s] == (side, 5)] for side in range(2)]
  for q in range(64):
    for distance, channel in ((1, 8), (2, 10)):
      for k in kings[1]:
        if any(_chebyshev(t, k) == distance for t in attacks[0][q]):
          out[channel, q, k] = True
      for k in kings[0]:
        if any(_chebyshev(t, k) == distance for t in attacks[1][q]):
          out[channel + 1, q, k] = True
  for s in range(64):
    if board[s] is not None:
      for k in kings[0] + kings[1]:
        if _chebyshev(s, k) == 1:
          out[12, s, k] = True
  before = _attack_sets(_board(previous))
  for side in range(2):
    for i in range(64):
      for j in attacks[side][i] - before[side][i]:
        out[13 + side, i, j] = True
      for j in before[side][i] - attacks[side][i]:
        out[15 + side, i, j] = True
  out[17:] = out[:17].transpose(0, 2, 1)
  return out


# --------------------------------------------------------------------------- reference 2: the kernel, transcribed
def _step(board, row_step, column_step):
  shift = row_step * 8 + column_step
  moved = (board << shift) & MASK64 if shift >= 0 else board >> -shift
  if column_step == 1:
    moved &= ~0x0101010101010101
  elif column_step == 2:
    moved &= ~0x0303030303030303
  elif column_step == -1:
    moved &= 0x7F7F7F7F7F7F7F7F
  elif column_step == -2:
    moved &= 0x3F3F3F3F3F3F3F3F
  return moved & MASK64


def _king_moves(board):
  moved = _step(board, 1, 0) | _step(board, -1, 0) | _step(board, 0, 1) | _step(board, 0, -1)
  return moved | _step(board, 1, 1) | _step(board, 1, -1) | _step(board, -1, 1) | _step(board, -1, -1)


def _knight_moves(board):
  moved = _step(board, 1, 2) | _step(board, 2, 1) | _step(board, -1, 2) | _step(board, -2, 1)
  return moved | _step(board, 1, -2) | _step(board, 2, -1) | _step(board, -1, -2) | _step(board, -2, -1)


def _ring2_moves(board):
  moved = _step(board, 2, -2) | _step(board, 2, -1) | _step(board, 2, 0) | _step(board, 2, 1) | _step(board, 2, 2)
  moved |= _step(board, -2, -2) | _step(board, -2, -1) | _step(board, -2, 0) | _step(board, -2, 1)
  moved |= _step(board, -2, 2) | _step(board, -1, -2) | _step(board, 0, -2) | _step(board, 1, -2)
  return moved | _step(board, -1, 2) | _step(board, 0, 2) | _step(board, 1, 2)


def _reverse(board):
  return int(f"{board:064b}"[::-1], 2)


def _rays(square):
  bit = 1 << square
  below = bit - 1
  above = ~(bit | below) & MASK64
  column, row = square & 7, square >> 3
  file = (0x0101010101010101 << column) & MASK64
  rank = (0xFF << (square & 56)) & MASK64
  main = (0x80402010 << 32) | 0x08040201
  anti = (0x01020408 << 32) | 0x10204080
  skew = 8 * column - 8 * row
  diagonal = ((main >> max(skew, 0)) << max(-skew, 0)) & MASK64
  skew = 56 - 8 * column - 8 * row
  antidiagonal = ((anti >> max(skew, 0)) << max(-skew, 0)) & MASK64
  return (diagonal & above, antidiagonal & above, antidiagonal & below, diagonal & below,
          file & above, file & below, rank & above, rank & below)


def _scan(ray, occupied, positive):
  blockers = ray & occupied
  if positive:
    upto = (blockers ^ (blockers - 1)) & MASK64
    first = blockers & upto
    rest = blockers ^ first
    upto2 = (rest ^ (rest - 1)) & MASK64
  else:
    reversed_blockers = _reverse(blockers)
    reversed_upto = (reversed_blockers ^ (reversed_blockers - 1)) & MASK64
    upto = _reverse(reversed_upto)
    first = blockers & upto
    reversed_rest = reversed_blockers & ~reversed_upto & MASK64
    upto2 = _reverse((reversed_rest ^ (reversed_rest - 1)) & MASK64)
  return ray & upto, first, ray & upto2 & ~upto & MASK64


_POSITIVE = (True, True, False, False, True, False, True, False)


def _attack_row(source, planes, diagonal, orthogonal, forward):
  pawn, knight, bishop, rook, queen, king = planes
  row = 0
  if source & pawn:
    row |= _step(source, forward, -1) | _step(source, forward, 1)
  if source & knight:
    row |= _knight_moves(source)
  if source & king:
    row |= _king_moves(source)
  if source & (bishop | queen):
    row |= diagonal
  if source & (rook | queen):
    row |= orthogonal
  return row


def _frame_rows(boards, square):
  occupied = 0
  for board in boards:
    occupied |= board
  rays = _rays(square)
  reach = [_scan(rays[d], occupied, _POSITIVE[d])[0] for d in range(8)]
  diagonal, orthogonal = reach[0] | reach[1] | reach[2] | reach[3], reach[4] | reach[5] | reach[6] | reach[7]
  source = 1 << square
  return (_attack_row(source, boards[0:6], diagonal, orthogonal, 1),
          _attack_row(source, boards[6:12], diagonal, orthogonal, -1))


def reference_kernel_port(pieces, previous):
  """bool [12, 64] x2 -> bool [34, 64, 64], following the Triton row kernel statement by statement."""
  weights = [1 << s for s in range(64)]
  boards = [sum(w for w, bit in zip(weights, plane) if bit) for plane in pieces]
  before = [sum(w for w, bit in zip(weights, plane) if bit) for plane in previous]
  us, them = boards[0:6], boards[6:12]
  occupied_us = us[0] | us[1] | us[2] | us[3] | us[4] | us[5]
  occupied_them = them[0] | them[1] | them[2] | them[3] | them[4] | them[5]
  occupied = occupied_us | occupied_them
  diagonal_us, diagonal_them = us[2] | us[4], them[2] | them[4]
  orthogonal_us, orthogonal_them = us[3] | us[4], them[3] | them[4]
  king_us, king_them = us[5], them[5]
  rows = np.zeros((17, 64), dtype=object)
  for square in range(64):
    source = 1 << square
    here = (source & occupied) != 0
    rays = _rays(square)
    scans = [_scan(rays[d], occupied, _POSITIVE[d]) for d in range(8)]
    reach, first, xray = [s[0] for s in scans], [s[1] for s in scans], [s[2] for s in scans]
    diagonal, orthogonal = reach[0] | reach[1] | reach[2] | reach[3], reach[4] | reach[5] | reach[6] | reach[7]
    xray_diagonal, xray_orthogonal = xray[0] | xray[1] | xray[2] | xray[3], xray[4] | xray[5] | xray[6] | xray[7]
    attacks_us = _attack_row(source, us, diagonal, orthogonal, 1)
    attacks_them = _attack_row(source, them, diagonal, orthogonal, -1)
    xray_us = (xray_diagonal if source & diagonal_us else 0) | (xray_orthogonal if source & orthogonal_us else 0)
    xray_them = (xray_diagonal if source & diagonal_them else 0) | (xray_orthogonal if source & orthogonal_them else 0)

    def pin(toward, back, sliders):
      return first[toward] if here and (first[back] & sliders) else 0

    pin_us = pin(0, 3, diagonal_us) | pin(3, 0, diagonal_us) | pin(1, 2, diagonal_us) | pin(2, 1, diagonal_us)
    pin_us |= pin(4, 5, orthogonal_us) | pin(5, 4, orthogonal_us) | pin(6, 7, orthogonal_us) | pin(7, 6, orthogonal_us)
    pin_them = pin(0, 3, diagonal_them) | pin(3, 0, diagonal_them) | pin(1, 2, diagonal_them) | pin(2, 1, diagonal_them)
    pin_them |= (pin(4, 5, orthogonal_them) | pin(5, 4, orthogonal_them) | pin(6, 7, orthogonal_them)
                 | pin(7, 6, orthogonal_them))
    previous_us, previous_them = _frame_rows(before, square)
    rows[:, square] = [
      attacks_us, attacks_them, attacks_us & occupied_us, attacks_them & occupied_them, xray_us, xray_them,
      pin_us, pin_them,
      _king_moves(attacks_us) & king_them, _king_moves(attacks_them) & king_us,
      _ring2_moves(attacks_us) & king_them, _ring2_moves(attacks_them) & king_us,
      (_king_moves(source) & (king_us | king_them)) if here else 0,
      attacks_us & ~previous_us & MASK64, attacks_them & ~previous_them & MASK64,
      previous_us & ~attacks_us & MASK64, previous_them & ~attacks_them & MASK64,
    ]
  out = np.zeros((34, 64, 64), dtype=bool)
  for channel in range(17):
    for square in range(64):
      row = int(rows[channel, square])
      for target in range(64):
        if (row >> target) & 1:
          out[channel, square, target] = True
  out[17:] = out[:17].transpose(0, 2, 1)
  return out


# --------------------------------------------------------------------------- comparisons
def _planes_from_fens(fens):
  masks, values = encode_fens(fens)
  planes = expand_planes(masks, values)
  return planes[:, 0:12] > 0.5, planes[:, 13:25] > 0.5


def _compare(label, pieces, previous, module, check_definitions=True):
  got = module(torch.from_numpy(pieces), None if previous is None else torch.from_numpy(previous)).numpy()
  empty = np.zeros_like(pieces[0])
  mismatched = np.zeros((2, 34), dtype=np.int64)
  bad_positions = 0
  for index in range(len(pieces)):
    before = empty if previous is None else previous[index]
    port = reference_kernel_port(pieces[index], before)
    difference = got[index] != port
    mismatched[0] += difference.sum((1, 2))
    wrong = bool(difference.any())
    if check_definitions:
      definition = reference_definitions(pieces[index], before)
      difference = got[index] != definition
      mismatched[1] += difference.sum((1, 2))
      wrong = wrong or bool(difference.any())
    bad_positions += int(wrong)
  density = got.reshape(len(pieces), -1).sum(1)
  per_channel = got.sum((0, 2, 3))[:BASE_CHANNELS]
  print(f"{label}: {len(pieces)} positions, mismatched cells vs kernel port {int(mismatched[0].sum())}, vs definitions "
        f"{int(mismatched[1].sum()) if check_definitions else 'n/a'}, positions with a mismatch {bad_positions}; "
        f"set bits per position median {int(np.median(density))} max {int(density.max())}")
  print("   set bits per base channel: " + ", ".join(f"{name} {int(count)}" for name, count in zip(CHANNEL_NAMES, per_channel)))
  assert bad_positions == 0, f"{label}: channel mismatches vs port {mismatched[0].tolist()} vs definitions {mismatched[1].tolist()}"
  return got


def _sq(name):
  return (ord(name[0]) - ord("a")) + 8 * (ord(name[1]) - ord("1"))


def edge_cases(module):
  A, D, X, P, R1O, R1U, R2O, BE, G_ST, L_ST = 0, 2, 4, 6, 8, 9, 10, 12, 13, 15

  def run(fen, previous_fen=None):
    pieces, history = _planes_from_fens([fen])
    if previous_fen is not None:
      history, _ = _planes_from_fens([previous_fen])
    got = _compare(f"edge case {fen!r}", pieces, history, module)
    return got[0]

  # 1. Pin: opp rook e8, our knight e4, our king e1 (white to move, so squares are the usual ones).
  e = run("4r2k/8/8/8/4N3/8/8/4K3 w - - 0 1")
  assert e[P + 1, _sq("e4"), _sq("e1")] and e[P + 1].sum() == 1, "pinray_opp [e4, e1] only"
  assert e[A + 1, _sq("e8"), _sq("e4")] and not e[A + 1, _sq("e8"), _sq("e3")], "the ray stops at its first blocker, inclusive"
  assert [bool(e[X + 1, _sq("e8"), _sq(s)]) for s in ("e5", "e4", "e3", "e2", "e1")] == [False, False, True, True, True]
  assert e[17 + P + 1, _sq("e1"), _sq("e4")], "transpose channel"
  assert e[P].sum() == 0 and e[D + 1, _sq("e8"), _sq("h8")] and e[D + 1].sum() == 1, "the rook defends its own king only"
  # 2. X-ray through two blockers: our rook a1, pieces on a3, a5, a7 -> xray = a4, a5 only; pinray [a3, a5].
  e = run("7k/p7/8/n7/8/P7/8/R6K w - - 0 1")
  assert [bool(e[X, _sq("a1"), _sq(s)]) for s in ("a2", "a3", "a4", "a5", "a6", "a7")] == [False, False, True, True, False, False]
  assert e[P, _sq("a3"), _sq("a5")] and e[P].sum() == 1
  assert e[D, _sq("a1"), _sq("a3")] and e[D, _sq("a1"), _sq("h1")], "defends: own pawn a3 and own king h1"
  # 3. Adjacent kings (not a legal position; the definitions still apply), plus ring distances.
  e = run("8/8/8/3kK3/8/8/8/8 w - - 0 1")
  assert e[BE, _sq("e5"), _sq("d5")] and e[BE, _sq("d5"), _sq("e5")] and e[BE].sum() == 2
  assert e[R1O, _sq("e5"), _sq("d5")] and e[R1U, _sq("d5"), _sq("e5")]
  assert e[R2O, _sq("e5"), _sq("d5")], "our king attacks f4/f5/f6, which are at Chebyshev 2 from d5"
  # 4. Pawns on the promotion rank and on the edge files; black to move flips the frame.
  e = run("8/P6k/8/8/8/8/7p/K7 w - - 0 1")
  assert e[A, _sq("a7")].sum() == 1 and e[A, _sq("a7"), _sq("b8")]
  assert e[A + 1, _sq("h2")].sum() == 1 and e[A + 1, _sq("h2"), _sq("g1")]
  e = run("8/P6k/8/8/8/8/7p/K7 b - - 0 1")
  # black to move: the black pawn h2 is OURS and sits on h7 of the flipped frame, attacking g8
  assert e[A, _sq("h7")].sum() == 1 and e[A, _sq("h7"), _sq("g8")]
  assert e[A + 1, _sq("a2")].sum() == 1 and e[A + 1, _sq("a2"), _sq("b1")]
  e = run("P6k/8/8/8/8/8/8/K6p w - - 0 1")
  assert e[A, _sq("a8")].sum() == 0 and e[A + 1, _sq("h1")].sum() == 0, "a pawn on the last rank attacks nothing"
  # 5. Start position: lc0 leaves the history empty, so gained = all attacks and lost = nothing.
  e = run("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
  assert (e[G_ST] == e[A]).all() and (e[G_ST + 1] == e[A + 1]).all() and e[L_ST].sum() == 0 and e[L_ST + 1].sum() == 0
  # 6. A real previous position: 1. e4 just played, seen by Black (frame flipped: the white pawn is THEIRS, on e5).
  e = run("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
  assert e[G_ST].sum() == 0 and e[L_ST].sum() == 0, "FEN-only history = copies of the position: nothing gained or lost"
  current, _ = _planes_from_fens(["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"])
  before, _ = _planes_from_fens(["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"])
  e = _compare("edge case 1.e4 with its real previous position", current, before, module)[0]
  opp_gain, opp_loss = e[G_ST + 1], e[L_ST + 1]
  assert opp_gain[_sq("e5"), _sq("d4")] and opp_gain[_sq("e5"), _sq("f4")], "the pawn now attacks from e5 (flipped e4)"
  assert opp_loss[_sq("e7"), _sq("d6")] and opp_loss[_sq("e7"), _sq("f6")], "and no longer from e7 (flipped e2)"
  assert opp_gain[_sq("d8"), _sq("h4")] and opp_gain[_sq("f8"), _sq("a3")], "queen and bishop diagonals opened"
  print("edge cases: all explicit expectations hold")


def cost(module, pieces, previous):
  """Time per 1,024 positions on this CPU and the size of what the module allocates."""
  count = 1024
  repeat = -(-count // len(pieces))
  p = torch.from_numpy(np.concatenate([pieces] * repeat)[:count])
  q = torch.from_numpy(np.concatenate([previous] * repeat)[:count])
  module(p[:64], q[:64])
  best = 1e9
  for _ in range(3):
    started = time.perf_counter()
    out = module(p, q)
    best = min(best, time.perf_counter() - started)
  print(f"cost: {best * 1e3:.1f} ms per {count} positions on CPU ({torch.get_num_threads()} threads); output bool "
        f"{tuple(out.shape)} = {out.numel() / count / 1024:.0f} KB per position ({out.numel() / 2**20:.0f} MB per {count}); "
        f"as float32 {out.numel() * 4 / count / 1024:.0f} KB, as bf16 {out.numel() * 2 / count / 1024:.0f} KB per position")
  buffers = sum(b.numel() * b.element_size() for b in module.buffers())
  print(f"cost: geometry buffers {buffers / 1024:.0f} KB; transient peak (float [B, 64, 8, 7] ray tensors, the 17 bool channels before stacking, two float [B, 64, 64] attack copies) is about "
        f"{(2 * 64 * 56 * 4 + 8 * 64 * 65 + 2 * 4096 * 4 + 34 * 4096) / 1024:.0f} KB per position including the output")


def main():
  path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FENS
  count = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
  torch.manual_seed(0)
  module = BT6Relations()
  assert OPPOSITE_DIRECTION == tuple(SLIDER_DIRECTIONS.index((-r, -c)) for r, c in SLIDER_DIRECTIONS)
  edge_cases(module)
  fens = read_fens(path, count)
  pieces, history = _planes_from_fens(fens)
  started = time.perf_counter()
  _compare("FEN-only history (engine encoding)", pieces, history, module)
  # previous := another position of the file (shifted by one): arbitrary but valid boards, dense gained / lost
  _compare("previous position = the next FEN's position", pieces, np.roll(pieces, -1, axis=0), module)
  _compare("no history (previous = None)", pieces[:512], None, module)
  print(f"references took {time.perf_counter() - started:.0f} s")
  cost(module, pieces, history)
  print("test_bt6_relations: OK")


if __name__ == "__main__":
  main()
