"""Reference tests of square_features_plus.py and gives_check.py against python-chess (2026-11-27).

  python test_sqf_plus.py [num puzzle FENs per jsonl file]

Positions: fry_testlib.TEST_FENS + game positions of a few random playouts + real puzzle FENs from the labelling jsonl
of the puzzle corpus (both colours to move). Every board is put in the frame of the side to move (python-chess
mirror for Black: ranks flipped, colours swapped), exactly as our record stores it.

The references below are written from the slot definitions with python-chess calls (attackers, is_attacked_by,
gives_check / checkers), NOT from the module code. Checks:
  1. every enabled slot family: module rows == reference rows on every square of every position
  2. Fry without neighbours == Fry's 23 rows with columns 4..11 removed; zero-init table => output exactly 0
  3. gives-check index of every legal move (castling, en passant and under-promotions excluded, see gives_check.py)
     == python-chess (direct: the moved piece is a checker; discovered: another piece is a checker)
"""
import glob
import os
import random
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess
import numpy as np
import torch

from fry_testlib import TEST_FENS, frame, piece_codes, square_feature_rows
from square_features_plus import SquareFeaturesPlus, FRY_KEEP_NO_NBR, LINE_PAIRS
from square_features import DIRECTIONS
from gives_check import GivesCheck


def positions(n_per_file):
  boards = [chess.Board(f) for f in TEST_FENS]
  rng = random.Random(1127)
  for g in range(40):                       # random playouts: varied middlegame / endgame material
    b = chess.Board()
    for ply in range(rng.randint(10, 120)):
      mv = list(b.legal_moves)
      if not mv:
        break
      b.push(rng.choice(mv))
      if ply % 7 == 3:
        boards.append(b.copy(stack=False))
  for p in sorted(glob.glob('/mnt/d/c1_640_34_2*_aug/labeled*_aug.jsonl')):
    k = 0
    with open(p, encoding='utf-8') as fh:
      for i, line in enumerate(fh):
        if i % 97 == 0:
          boards.append(chess.Board(line.split('"FEN": "')[1].split('"')[0]))
          k += 1
          if k >= n_per_file:
            break
  return boards


def ray_first(fb, s, d):
  df, dr = DIRECTIONS[d]
  f, r = s % 8 + df, s // 8 + dr
  while 0 <= f < 8 and 0 <= r < 8:
    p = fb.piece_at(r * 8 + f)
    if p is not None:
      return p.piece_type + (0 if p.color == chess.WHITE else 6)
    f, r = f + df, r + dr
  return 0


def ref_rows(fb, m):
  codes = piece_codes(fb)
  out = np.zeros((64, m.num_slots), dtype=np.int64)
  fry = square_feature_rows(codes) if m.fry else None
  for s in range(64):
    row = []
    if m.fry:
      row += list(fry[s][FRY_KEEP_NO_NBR] if m.drop_nbr else fry[s])
    if m.line_pairs:
      for L, (a, b) in enumerate(LINE_PAIRS):
        row.append(m.base_lp + L * 169 + 13 * ray_first(fb, s, a) + ray_first(fb, s, b))
    wa, ba = fb.attackers(chess.WHITE, s), fb.attackers(chess.BLACK, s)
    if m.cheap:
      co = min([fb.piece_type_at(x) for x in wa], default=0)
      ct = min([fb.piece_type_at(x) for x in ba], default=0)
      row.append(m.base_ca + codes[s] * 49 + co * 7 + ct)
    if m.counts:
      row.append(m.base_ac + min(len(wa), 3) * 4 + min(len(ba), 3))
    if m.king:
      vals = []
      for colour, base_c, base_r in ((chess.WHITE, m.base_ks, m.base_ks + 20), (chess.BLACK, m.base_ks + 10, m.base_ks + 24)):
        k = fb.king(colour)
        nb = chess.Board(fb.fen())
        nb.remove_piece_at(k)
        ring = [x for x in chess.SquareSet(chess.BB_KING_ATTACKS[k])]
        def flag(x):
          if fb.color_at(x) == colour:
            return 2
          return 1 if nb.is_attacked_by(not colour, x) else 0
        safe = sum(1 for x in ring if flag(x) == 0)
        vals.append((base_c + (1 + safe if s == k else 0), base_r + (1 + flag(s) if s in ring else 0)))
      row += [vals[0][0], vals[1][0], vals[0][1], vals[1][1]]
    out[s] = row
  return out


def ref_gives_check(fb):
  """[(from, to, index)] for the legal moves of a frame board (castling / e.p. / under-promotions left out)."""
  res = []
  if fb.was_into_check():        # illegal position: the side NOT to move is in check (one artificial TEST_FEN)
    return res
  for mv in fb.legal_moves:
    if fb.is_castling(mv) or fb.is_en_passant(mv) or (mv.promotion not in (None, chess.QUEEN)):
      continue
    b = fb.copy(stack=False)
    b.push(mv)
    ch = b.checkers()
    direct = mv.to_square in ch
    disc = any(x != mv.to_square for x in ch)
    res.append((mv.from_square, mv.to_square, int(direct) + 2 * int(disc)))
  return res


def main():
  n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 300
  boards = positions(n_per)
  frames = [frame(b) for b in boards]
  n_black = sum(b.turn == chess.BLACK for b in boards)
  print(f'{len(boards)} positions ({n_black} Black to move)')
  codes = torch.from_numpy(np.stack([piece_codes(fb) for fb in frames]))

  arms = {'fry_nonbr': dict(fry=True, drop_neighbours=True),
          'lp': dict(fry=False, line_pairs=True), 'ca': dict(fry=False, cheapest_attackers=True),
          'ac': dict(fry=False, attack_counts=True), 'ks': dict(fry=False, king_safety=True),
          'sqfplus': dict(fry=True, line_pairs=True, cheapest_attackers=True, attack_counts=True)}
  for name, kw in arms.items():
    m = SquareFeaturesPlus(8, **kw)
    rows = m.rows(codes).numpy()
    assert rows.max() < m.num_rows and rows.min() >= 0
    bad = sum(int(not (rows[i] == ref_rows(fb, m)).all()) for i, fb in enumerate(frames))
    used = len(np.unique(rows))
    out = m(codes)
    assert float(out.detach().abs().max()) == 0.0
    print(f'  {name:10s}: {m.num_slots} slots, {m.num_rows} rows ({used} used here); module == python-chess reference '
          f'on {len(frames) - bad}/{len(frames)} positions')
    assert bad == 0, name

  gc = GivesCheck()
  tot = bad = 0
  counts = np.zeros(4, dtype=np.int64)
  for i, fb in enumerate(frames):
    ref = ref_gives_check(fb)
    if not ref:
      continue
    fr = torch.tensor([[r[0] for r in ref]])
    to = torch.tensor([[r[1] for r in ref]])
    got = gc(codes[i:i + 1], fr, to)[0].tolist()
    exp = [r[2] for r in ref]
    for g_, e_ in zip(got, exp):
      counts[e_] += 1
    nb = sum(int(g_ != e_) for g_, e_ in zip(got, exp))
    if nb and bad < 5:
      print('   gives-check mismatch', fb.fen(), [(chess.Move(r[0], r[1]).uci(), g_, r[2]) for r, g_ in zip(ref, got) if g_ != r[2]][:4])
    bad += nb
    tot += len(ref)
  print(f'  gives-check: {tot - bad}/{tot} legal moves agree (reference: none {counts[0]}, direct {counts[1]}, '
        f'discovered {counts[2]}, double {counts[3]})')
  assert bad == 0
  print('SQF PLUS / GIVES CHECK OK')


if __name__ == '__main__':
  main()
