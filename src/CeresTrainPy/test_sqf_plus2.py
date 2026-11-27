"""Reference tests of SquareFeaturesRay2 (square_features_plus.py) and position_outcome.py against python-chess
(2026-11-27b). Same positions as test_sqf_plus.py (TEST_FENS, random playouts, real puzzle FENs; frame of the side to
move).

  python test_sqf_plus2.py [num puzzle FENs per jsonl file]

  1. ray2 rows == the second piece met walking each ray with python-chess piece_at, on every square
  2. the six position facts == python-chess: attackers() for en prise / wins (the analysis rule, run_D.py),
     is_check(), legal_moves + gives_check() for has-check (castling / e.p. / under-promotions left out),
     flight squares as klabels.py (king removed). Also reported: agreement of has-check with the analysis label
     (all legal moves, castling / e.p. included; not a test).
"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess
import numpy as np
import torch

from fry_testlib import frame, piece_codes
from square_features import DIRECTIONS
from square_features_plus import SquareFeaturesPlus
from position_outcome import PositionOutcome, BASES
from test_sqf_plus import positions

VAL = [0, 1, 3, 3, 5, 9, 99]


def ray_second(fb, s, d):
  df, dr = DIRECTIONS[d]
  f, r, k, seen = s % 8 + df, s // 8 + dr, 1, 0
  while 0 <= f < 8 and 0 <= r < 8:
    p = fb.piece_at(r * 8 + f)
    if p is not None:
      seen += 1
      if seen == 2:
        return 1 + (p.piece_type + (0 if p.color == chess.WHITE else 6) - 1) * 7 + (k - 1)
    f, r, k = f + df, r + dr, k + 1
  return 0


def ref_facts(fb):
  def side(colour):
    n = 0
    for s, p in fb.piece_map().items():
      if p.color != colour or p.piece_type == chess.KING:
        continue
      att = fb.attackers(not colour, s)
      dfd = fb.attackers(colour, s)
      if not att:
        continue
      cheapest = min(VAL[fb.piece_type_at(x)] for x in att)
      if cheapest < VAL[p.piece_type] or not dfd:
        n += 1
    return min(n, 3)
  en_prise, wins = side(chess.WHITE), side(chess.BLACK)
  in_check = int(fb.is_check())
  has_check = 0
  for mv in fb.legal_moves:
    if fb.is_castling(mv) or fb.is_en_passant(mv) or mv.promotion not in (None, chess.QUEEN):
      continue
    if fb.gives_check(mv):
      has_check = 1
      break
  legal_check = int(any(fb.gives_check(m) for m in fb.legal_moves))
  def flights(colour):
    k = fb.king(colour)
    b2 = fb.copy(stack=False)
    b2.remove_piece_at(k)
    return sum(1 for s in chess.SquareSet(chess.BB_KING_ATTACKS[k])
               if fb.color_at(s) != colour and not b2.is_attacked_by(not colour, s))
  loc = [en_prise, wins, in_check, has_check, flights(chess.WHITE), flights(chess.BLACK)]
  return [b + v for b, v in zip(BASES, loc)], legal_check


def main():
  n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 300
  boards = positions(n_per)
  frames = [frame(b) for b in boards]
  legal_pos = [i for i, fb in enumerate(frames) if not fb.was_into_check()]
  print(f'{len(boards)} positions ({sum(b.turn == chess.BLACK for b in boards)} Black to move); '
        f'{len(frames) - len(legal_pos)} illegal (side not to move in check) left out of the position facts')
  codes = torch.from_numpy(np.stack([piece_codes(fb) for fb in frames]))

  m = SquareFeaturesPlus(8, fry=False, ray2=True)
  rows = m.rows(codes).numpy()
  bad = 0
  for i, fb in enumerate(frames):
    ref = np.array([[m.base_r2 + 85 * d + ray_second(fb, s, d) for d in range(8)] for s in range(64)])
    bad += int(not (rows[i] == ref).all())
  print(f'  ray2: {m.num_slots} slots, {m.num_rows} rows ({len(np.unique(rows))} used); module == reference on '
        f'{len(frames) - bad}/{len(frames)}')
  assert bad == 0 and float(m(codes).detach().abs().max()) == 0.0

  po = PositionOutcome(8, 16, True, True)
  got = po.rows(codes[legal_pos]).numpy()
  names = ['en_prise', 'wins', 'in_check', 'has_check', 'flights_own', 'flights_their']
  bad_slot = np.zeros(6, dtype=np.int64)
  dist = np.zeros((6, 10), dtype=np.int64)
  pseudo_vs_legal = 0
  for j, i in enumerate(legal_pos):
    ref, legal_check = ref_facts(frames[i])
    for k in range(6):
      bad_slot[k] += int(got[j, k] != ref[k])
      dist[k, ref[k] - BASES[k]] += 1
    pseudo_vs_legal += int((ref[3] - BASES[3]) != legal_check)
  n = len(legal_pos)
  for k in range(6):
    vals = {v: int(c) for v, c in enumerate(dist[k]) if c}
    print(f'  {names[k]:13s}: module == python-chess on {n - bad_slot[k]}/{n}   (values {vals})')
  print(f'  has_check (legal moves without castling / e.p.) vs the analysis label (all legal moves): differ in '
        f'{pseudo_vs_legal}/{n} positions')
  sq = po.square_add(po.rows(codes[:4]))
  vh = po.value_add(po.rows(codes[:4]))
  assert tuple(sq.shape) == (4, 1, 8) and tuple(vh.shape) == (4, 16) and float(sq.abs().max()) == 0 and float(vh.abs().max()) == 0
  assert bad_slot.sum() == 0
  print('RAY2 / POSITION OUTCOME OK')


if __name__ == '__main__':
  main()
