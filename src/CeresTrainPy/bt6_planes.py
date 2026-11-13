"""Lc0 classical 112-plane input for positions given by FEN only, as plane masks + values (the lc0ex input form).

A Python port of mycel's `Lc0Net.EncodeMasks` (src/Mycel/Lc0Net.cs) for a board without move history, i.e. lc0's
INPUT_CLASSICAL_112_PLANE with history fill "fen_only":

* planes 13h .. 13h + 12 (h = 0..7 plies back): our P N B R Q K, their P N B R Q K, repetition. Element s = square s
  (a1 = 0) in the frame of the side to move: with Black to move the board is mirrored (rank flip, s ^ 56) and the
  colours swapped. ALL eight history slots are in the frame of the side to move NOW.
* A FEN has no history. From the standard start position the slots 1..7 stay empty; from any other position they
  are copies of the position, with the en-passant pawn put back on its starting square (the double push that made
  the e.p. square did not happen yet in the copies). The repetition planes are 0.
* 104 / 105: we can castle queen / king side; 106 / 107: they can; 108: all ones with Black to move;
  109: the 50-move counter in plies (raw, every square); 110: zeros; 111: ones.

Plane form: mask bit s = square s is set, value = what the set squares hold (1 for every plane but 109).
Like the engine, an e.p. square is kept only when a pawn of the side to move can (pseudo-legally) capture on it.
Castling rights are read from the standard KQkq letters only (no Chess960 rook-file letters).
"""

from __future__ import annotations

import numpy as np

PLANES = 112
HISTORY = 8
PLANES_PER_BOARD = 13
AUX_BASE = 104
EMPTY = 12
_PIECES = "PNBRQKpnbrqk"  # engine piece code = colour * 6 + type, white = 0
_START = ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "w", "KQkq", "-")
_ALL = 0xFFFFFFFFFFFFFFFF


def parse_fen(fen: str):
  """Return (squares[64] piece codes with a1 = 0, side (0 white), castle bits, ep square or 64, halfmove)."""
  parts = fen.split()
  squares = [EMPTY] * 64
  rank, file = 7, 0
  for ch in parts[0]:
    if ch == "/":
      rank, file = rank - 1, 0
    elif ch.isdigit():
      file += int(ch)
    else:
      squares[8 * rank + file] = _PIECES.index(ch)
      file += 1
  side = 0 if len(parts) < 2 or parts[1] == "w" else 1
  castle = 0  # bit colour * 2 = king side, colour * 2 + 1 = queen side
  for ch, bit in zip("KQkq", (1, 2, 4, 8)):
    if len(parts) > 2 and ch in parts[2]:
      castle |= bit
  ep = 64
  if len(parts) > 3 and parts[3] != "-" and len(parts[3]) == 2:
    ep_square = (ord(parts[3][0]) - ord("a")) + 8 * (ord(parts[3][1]) - ord("1"))
    if 0 <= ep_square < 64 and (ep_square >> 3) == (5 if side == 0 else 2) and squares[ep_square] == EMPTY:
      pushed = ep_square - 8 if side == 0 else ep_square + 8
      capturer_row = ep_square - 8 if side == 0 else ep_square + 8  # same rank as the pushed pawn
      file = ep_square & 7
      capturers = [capturer_row + step for step in (-1, 1) if 0 <= file + step < 8]
      if squares[pushed] == (side ^ 1) * 6 and any(squares[s] == side * 6 for s in capturers):
        ep = ep_square
  halfmove = 0
  if len(parts) > 4:
    try:
      halfmove = int(parts[4])
    except ValueError:
      halfmove = 0
  return squares, side, castle, ep, halfmove


def _is_start(fen: str) -> bool:
  parts = fen.split()
  return tuple(parts[:4]) == _START if len(parts) >= 4 else False


def fen_to_masks(fen: str):
  """One FEN -> (masks uint64 [112], values float32 [112])."""
  squares, side, castle, ep, halfmove = parse_fen(fen)
  masks = [0] * PLANES
  values = np.ones(PLANES, dtype=np.float32)
  flip = side == 1
  us, them = side, side ^ 1
  history = [squares]
  if not _is_start(fen):
    older = list(squares)
    if ep != 64:
      # the side not to move just pushed a pawn two squares, past the e.p. square: put it back
      pusher = side ^ 1
      step = 8 if pusher == 0 else -8
      at, origin = ep + step, ep - step
      if older[at] == pusher * 6 and older[origin] == EMPTY:
        older[origin], older[at] = older[at], EMPTY
    history += [older] * (HISTORY - 1)
  for h, position in enumerate(history):
    base = h * PLANES_PER_BOARD
    for s in range(64):
      piece = position[s ^ 56 if flip else s]
      if piece == EMPTY:
        continue
      colour, kind = divmod(piece, 6)
      masks[base + (0 if colour == us else 6) + kind] |= 1 << s
    # repetition plane (base + 12): never set for a FEN without history
  if (castle >> (us * 2 + 1)) & 1:
    masks[AUX_BASE + 0] = _ALL
  if (castle >> (us * 2)) & 1:
    masks[AUX_BASE + 1] = _ALL
  if (castle >> (them * 2 + 1)) & 1:
    masks[AUX_BASE + 2] = _ALL
  if (castle >> (them * 2)) & 1:
    masks[AUX_BASE + 3] = _ALL
  if flip:
    masks[AUX_BASE + 4] = _ALL
  masks[AUX_BASE + 5] = _ALL
  values[AUX_BASE + 5] = halfmove
  masks[AUX_BASE + 7] = _ALL
  return np.array(masks, dtype=np.uint64), values


def encode_fens(fens):
  """FENs -> (masks uint64 [N, 112], values float32 [N, 112])."""
  encoded = [fen_to_masks(fen) for fen in fens]
  return np.stack([m for m, _ in encoded]), np.stack([v for _, v in encoded])


def expand_planes(masks: np.ndarray, values: np.ndarray) -> np.ndarray:
  """(masks [N, 112], values [N, 112]) -> float32 planes [N, 112, 64] (the kernel `expand_planes`)."""
  bits = (masks[:, :, None] >> np.arange(64, dtype=np.uint64)[None, None, :]) & np.uint64(1)
  return bits.astype(np.float32) * values[:, :, None].astype(np.float32)


def read_fens(path: str, count: int | None = None):
  with open(path, "r", encoding="utf-8") as handle:
    fens = [line.strip() for line in handle if line.strip()]
  return fens if count is None else fens[:count]
