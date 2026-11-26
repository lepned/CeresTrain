"""Test-side helpers for the Fry blocks (python-chess; not imported by the trainer).

Everything here is written from the description in FRY_BLOCKS_NOTES.md / the review of the net, and is the
INDEPENDENT side of the tests: a board-based encoder of the 70 tokens, a per-square reference of the 23 square
features, the 1968-move table of Fry's own policy output, our square record from a board, and the legal-move index
pairs (our 1858 index, Fry's 1968 id).

Frame everywhere: the side to move; a Black-to-move position is `board.mirror()` (colours swapped, ranks mirrored).
"""

from __future__ import annotations

import chess
import numpy as np

from lc0_moves_1858 import MOVES_1858

DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1))   # (file step, rank step)
KNIGHT = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_INDEX_1858 = {m: i for i, m in enumerate(MOVES_1858)}


def frame(board: chess.Board) -> chess.Board:
  """The position as the side to move sees it (White to move)."""
  return board if board.turn == chess.WHITE else board.mirror()


def piece_codes(fb: chess.Board) -> np.ndarray:
  """64 codes of a frame board: 0 empty, 1..6 our P N B R Q K, 7..12 theirs."""
  out = np.zeros(64, dtype=np.int64)
  for s, p in fb.piece_map().items():
    out[s] = p.piece_type + (0 if p.color == chess.WHITE else 6)
  return out


def fry_tokens(board: chess.Board) -> np.ndarray:
  """The 70 token ids of a position (colour = the real side to move, e.p. only when a capture is legal)."""
  fb = frame(board)
  t = np.zeros(70, dtype=np.int64)
  t[:64] = piece_codes(fb)
  t[64] = 14 if board.turn == chess.WHITE else 13
  t[65] = 16 if fb.has_kingside_castling_rights(chess.WHITE) else 15
  t[66] = 16 if fb.has_queenside_castling_rights(chess.WHITE) else 15
  t[67] = 16 if fb.has_kingside_castling_rights(chess.BLACK) else 15
  t[68] = 16 if fb.has_queenside_castling_rights(chess.BLACK) else 15
  t[69] = 17 + (fb.ep_square if (fb.ep_square is not None and fb.has_legal_en_passant()) else 0)
  return t


def record_from_board(board: chess.Board, history: bool = True) -> np.ndarray:
  """Our square record [64, 137] of a position (TPGSquareRecord layout, floats as the loaders hand them over):
  current board copied into the 8 history slots, castling bytes 112..115, rule-50 byte 116, the e.p. byte 118 on
  the square of the pawn that just made a double step (whenever the position has an e.p. square, capturable or
  not -- the convention of the DirectFromV6 loader), rank / file one-hots 121..136."""
  fb = frame(board)
  codes = piece_codes(fb)
  rec = np.zeros((64, 137), dtype=np.float32)
  for h in range(8 if history else 1):
    rec[np.arange(64), h * 13 + codes] = 1.0
  rec[:, 112] = fb.has_kingside_castling_rights(chess.WHITE)
  rec[:, 113] = fb.has_queenside_castling_rights(chess.WHITE)
  rec[:, 114] = fb.has_kingside_castling_rights(chess.BLACK)
  rec[:, 115] = fb.has_queenside_castling_rights(chess.BLACK)
  rec[:, 116] = min(fb.halfmove_clock, 100) / 50.0
  if fb.ep_square is not None:
    rec[fb.ep_square - 8, 118] = 1.0
  rec[np.arange(64), 121 + np.arange(64) // 8] = 1.0
  rec[np.arange(64), 129 + np.arange(64) % 8] = 1.0
  return rec


def square_feature_rows(codes: np.ndarray) -> np.ndarray:
  """Reference (per square, plain loops) of the 23 table rows: [64, 23]."""
  def at(f, r):
    return int(codes[r * 8 + f]) if (0 <= f < 8 and 0 <= r < 8) else None
  def first(code):
    w = np.nonzero(codes == code)[0]
    return int(w[0]) if len(w) else None
  anchors = [first(6), first(12), first(5), first(11)]
  rows = np.zeros((64, 23), dtype=np.int64)
  for s in range(64):
    f0, r0 = s % 8, s // 8
    for k, a in enumerate(anchors):
      rows[s, k] = 226 * k + (225 if a is None else (r0 - a // 8 + 7) * 15 + (f0 - a % 8 + 7))
    for d, (df, dr) in enumerate(DIRECTIONS):
      c = at(f0 + df, r0 + dr)
      rows[s, 4 + d] = 904 + 14 * d + (13 if c is None else c)
      local = 0
      for dist in range(1, 8):
        c = at(f0 + df * dist, r0 + dr * dist)
        if c is None:
          break
        if c != 0:
          local = 1 + (c - 1) * 7 + (dist - 1)
          break
      rows[s, 12 + d] = 1016 + 85 * d + local
    ours = sum(1 for df, dr in KNIGHT if at(f0 + df, r0 + dr) == 2)
    theirs = sum(1 for df, dr in KNIGHT if at(f0 + df, r0 + dr) == 8)
    rows[s, 20] = 1696 + min(ours, 2) * 3 + min(theirs, 2)
    for slot, dr in ((21, 1), (22, -1)):
      local = 0
      for dist in range(1, 8):
        c = at(f0, r0 + dr * dist)
        if c is None:
          break
        if c in (1, 7):
          local = 1 + (7 if c == 7 else 0) + (dist - 1)
          break
      rows[s, slot] = (1705 if slot == 21 else 1720) + local
  return rows


def fry_move_table():
  """Fry's 1968 policy ids as (from, to, promotion letter or '') in the side-to-move frame:
  0..1791 from-to pairs (from ascending, to ascending) of queen and knight moves on an empty board; 1792..1879
  promotions rank 7 -> 8 (from file a..h, to file ascending, pieces N R B Q); 1880..1967 the same rank 2 -> 1."""
  moves = []
  for f in range(64):
    ff, fr = f % 8, f // 8
    for t in range(64):
      if t == f:
        continue
      df, dr = t % 8 - ff, t // 8 - fr
      if df == 0 or dr == 0 or abs(df) == abs(dr) or (abs(df), abs(dr)) in ((1, 2), (2, 1)):
        moves.append((f, t, ''))
  assert len(moves) == 1792
  for r_from, r_to in ((6, 7), (1, 0)):
    for ff in range(8):
      for tf in (ff - 1, ff, ff + 1):
        if 0 <= tf < 8:
          for piece in 'nrbq':
            moves.append((r_from * 8 + ff, r_to * 8 + tf, piece))
  assert len(moves) == 1968
  return moves


_FRY_INDEX = None


def legal_move_indices(board: chess.Board):
  """For every legal move of the position: (uci in the frame, our 1858 index, Fry's 1968 id).
  Ours: castling = king takes rook, knight promotion = the plain move. Fry's: castling = the king's two-square move,
  every promotion (queen included) = a promotion id."""
  global _FRY_INDEX
  if _FRY_INDEX is None:
    _FRY_INDEX = {m: i for i, m in enumerate(fry_move_table())}
  fb = frame(board)
  out = []
  for mv in fb.legal_moves:
    f, t = mv.from_square, mv.to_square
    promo = chess.piece_symbol(mv.promotion) if mv.promotion else ''
    if fb.is_castling(mv):
      rook = 7 if t > f else 0
      ours = _INDEX_1858[chess.SQUARE_NAMES[f] + chess.SQUARE_NAMES[rook]]
    else:
      ours = _INDEX_1858[chess.SQUARE_NAMES[f] + chess.SQUARE_NAMES[t] + ('' if promo in ('', 'n') else promo)]
    out.append((mv.uci(), ours, _FRY_INDEX[(f, t, promo)]))
  return out


TEST_FENS = [
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
  'r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4',
  'rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3',           # e.p. legal, White
  'rnbqkbnr/pppp1ppp/8/8/3Pp3/5N2/PPP1PPPP/RNBQKB1R b KQkq d3 0 3',          # e.p. legal, Black
  '8/P6k/8/8/8/8/7p/K7 w - - 0 60',                                          # promotions, White
  '8/P6k/8/8/8/8/1p6/K1N5 b - - 0 60',                                       # promotions with captures, Black
  'r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 8',    # both castlings, White
  'r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 6 8',    # both castlings, Black
  '4k3/8/8/8/8/8/8/4K2R w K - 0 1',
  '2q1k3/8/1Q6/8/3Q4/8/5q2/4K3 w - - 0 1',                                   # several queens
  '4k3/8/8/2n5/8/4N3/8/4K3 b - - 0 1',                                       # no queens
  '8/8/8/3k4/2pP4/8/8/3K4 b - d3 0 1',                                       # e.p. legal in an ending
  '8/8/8/8/k1pP3R/8/8/3K4 b - d3 0 1',                                       # e.p. would expose the king: NOT legal
  '6k1/p3b2p/1p1pP3/2p3P1/1Pnp3B/P6P/3Q3K/8 w - - 0 38',
  '2r1kb2/5p1p/p3p3/1p2P1Q1/1Pq5/P1p3P1/2P4P/2KR1R2 w - - 0 28',
]
