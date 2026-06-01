#!/usr/bin/env python3
"""End-to-end validation of the 4 NEW V3-extended aux channels.

For each position in the freshly-upgraded V3-ext shard:
  1. Decode 12 piece bitboards from the V2 piece one-hot (first 137 bytes/sq)
  2. Compute oracle aux[0..3] via python-chess (the slow, obviously-correct way)
  3. Compare to the C#-baked aux[0..3] in the shard
  4. Assert byte-identical match

Channels checked (all are properties of the PIECE on the square — side-agnostic):
  [3] mobility        — pseudo-legal move count of piece on sq (captures-only for pawns,
                        matches C# AddMobility which uses attack-tables ∩ ~own)
                        encoding: scaled = raw * 100 / 27, capped at 100
  [4] defender_count  — friendly attackers of the piece on sq (excluding self)
                        encoding: count * 100 / 8
  [5] is_pinned       — 0 / 100 boolean (pinned to friendly king by opp slider)
  [6] is_threatened   — 0 / 100 boolean (attacked by opp piece of STRICTLY lower value;
                        king-attacked → also threatened)

Channel 0/1/2 (attackers) are already validated by AugFeatSanity Phase 2 — we skip them.

Usage:
  python3 validate_v3ext_aux_bytes.py <v3ext.zst> [num_positions]
"""

import sys
import os
import struct
import zstandard
import numpy as np
import chess

# Layout constants (must match Ceres TPGRecord.cs at HEAD)
HEADER_BYTES   = 610
SQ_STRIDE      = 141
V3EXT_BYTES    = HEADER_BYTES + 64 * SQ_STRIDE   # 9634

# TPG one-hot piece classes 1..12 → python-chess (PieceType, color)
TPG_TO_CHESS = {
  1:  (chess.PAWN,   chess.WHITE),  2:  (chess.KNIGHT, chess.WHITE),
  3:  (chess.BISHOP, chess.WHITE),  4:  (chess.ROOK,   chess.WHITE),
  5:  (chess.QUEEN,  chess.WHITE),  6:  (chess.KING,   chess.WHITE),
  7:  (chess.PAWN,   chess.BLACK),  8:  (chess.KNIGHT, chess.BLACK),
  9:  (chess.BISHOP, chess.BLACK),  10: (chess.ROOK,   chess.BLACK),
  11: (chess.QUEEN,  chess.BLACK),  12: (chess.KING,   chess.BLACK),
}

# Piece values for is_threatened (matching C# ComputeIsThreatenedForColor).
# Pawn=1, N=3, B=3, R=5, Q=9, K=100 (treated as ∞ — king is threatened by anything attacking).
PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9,  chess.KING: 100}


def decode_v3ext_position_to_board(rec: bytes) -> chess.Board:
  """Decode piece placement from a 9826-byte V3-ext record into a chess.Board.

  Note: TPG records are us-to-move oriented. We treat "our" pieces (cls 1..6) as
  WHITE and "opp" (cls 7..12) as BLACK. The extended features are color-symmetric
  so this role assignment doesn't affect oracle correctness — it matches what
  ComputeExtendedFromTpgSquareBytes does on the C# side.
  """
  board = chess.Board.empty()
  for sq in range(64):
    sq_off = HEADER_BYTES + sq * SQ_STRIDE
    # Find the piece class (slot 1..12 with byte > 50)
    for cls in range(1, 13):
      if rec[sq_off + cls] > 50:
        ptype, color = TPG_TO_CHESS[cls]
        board.set_piece_at(sq, chess.Piece(ptype, color))
        break
  return board


def oracle_aux_bytes(board: chess.Board) -> np.ndarray:
  """Compute aux[0..3] for all 64 squares. Returns (64, 4) uint8 array."""
  out = np.zeros((64, 4), dtype=np.uint8)

  # Build per-color attacker masks once (defender count uses same-color attackers).
  white_attackers_of = [board.attackers_mask(chess.WHITE, sq) for sq in range(64)]
  black_attackers_of = [board.attackers_mask(chess.BLACK, sq) for sq in range(64)]

  # Build per-piece-type opp attacker bitboards (for is_threatened).
  attackers_by_type_color = {}
  for color in (chess.WHITE, chess.BLACK):
    for ptype in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
      attackers_by_type_color[(color, ptype)] = board.pieces_mask(ptype, color)

  # Find kings for pin detection.
  white_king_sq = board.king(chess.WHITE)
  black_king_sq = board.king(chess.BLACK)

  for sq in range(64):
    piece = board.piece_at(sq)
    if piece is None:
      continue  # empty square — all aux bytes zero

    color = piece.color
    ptype = piece.piece_type

    # ---- [3] mobility (captures-only for pawns, attacks ∩ ~own for everything else) ----
    # Match C# AddMobility: uses ATTACK pattern (not push), ANDed with ~own-color pieces.
    # python-chess attacks_mask returns the attack squares; & ~own gives capture-mobility.
    own_mask = board.occupied_co[color]
    attacks = board.attacks_mask(sq)
    raw_mob = bin(attacks & ~own_mask).count('1')
    scaled = raw_mob * 100 // 27
    if scaled > 100:
      scaled = 100
    out[sq, 0] = scaled  # → aux[3]

    # ---- [4] defender_count (friendly attackers of this sq, excluding self) ----
    defs_mask = white_attackers_of[sq] if color == chess.WHITE else black_attackers_of[sq]
    # attackers_mask does NOT include the piece on the sq itself, so no self-subtract needed.
    raw_def = bin(defs_mask).count('1')
    out[sq, 1] = raw_def * 100 // 8  # → aux[4]

    # ---- [5] is_pinned (piece is pinned to its own king by an opp slider) ----
    # python-chess has board.is_pinned(color, sq) — uses exactly this definition.
    is_pinned = board.is_pinned(color, sq)
    out[sq, 2] = 100 if is_pinned else 0  # → aux[5]

    # ---- [6] is_threatened (attacked by opp piece of strictly lower value) ----
    # Special case: KING is "threatened" if attacked at all (no opp value < king-value=100,
    # but strict-less is still true since K=100 > anything).
    opp = not color
    opp_attackers_mask = (black_attackers_of[sq] if color == chess.WHITE else white_attackers_of[sq])
    threatened = False
    if opp_attackers_mask:
      my_val = PIECE_VAL[ptype]
      # For each opp piece type, check if its attackers-of-sq intersects with the opp pieces of that type
      # AND has strictly lower value.
      for opp_ptype, opp_val in PIECE_VAL.items():
        if opp_val < my_val:
          opp_pieces_of_type = attackers_by_type_color[(opp, opp_ptype)]
          if opp_attackers_mask & opp_pieces_of_type:
            threatened = True
            break
    out[sq, 3] = 100 if threatened else 0  # → aux[6]

  return out


def main():
  if len(sys.argv) < 2:
    print("usage: validate_v3ext_aux_bytes.py <v3ext.zst> [num_positions]")
    sys.exit(2)

  path = sys.argv[1]
  n_positions = int(sys.argv[2]) if len(sys.argv) > 2 else 500

  print(f"Validating up to {n_positions} positions from {path}")
  print(f"Comparing oracle aux[0..3] (mobility, defender, is_pinned, is_threatened) to baked bytes.")

  with open(path, 'rb') as f:
    dctx = zstandard.ZstdDecompressor()
    raw = dctx.stream_reader(f).read(n_positions * V3EXT_BYTES)

  actual_n = len(raw) // V3EXT_BYTES
  print(f"Decompressed {len(raw):,} bytes = {actual_n:,} records of {V3EXT_BYTES} bytes")

  mismatches_per_ch = [0, 0, 0, 0]
  ch_names = ['[0] mobility', '[1] defender_count', '[2] is_pinned', '[3] is_threatened']
  total_pieces = 0
  total_sq = 0
  first_mismatch = None

  for rec_idx in range(actual_n):
    rec = raw[rec_idx * V3EXT_BYTES : (rec_idx + 1) * V3EXT_BYTES]
    board = decode_v3ext_position_to_board(rec)
    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
      # Pin detection requires kings; corrupted/test position — skip
      continue

    expected = oracle_aux_bytes(board)  # (64, 4)

    # Extract baked aux[0..3] from each square slot
    actual = np.zeros((64, 4), dtype=np.uint8)
    for sq in range(64):
      sq_off = HEADER_BYTES + sq * SQ_STRIDE
      actual[sq, 0] = rec[sq_off + 137 + 0]   # mobility
      actual[sq, 1] = rec[sq_off + 137 + 1]   # defender_count
      actual[sq, 2] = rec[sq_off + 137 + 2]   # is_pinned
      actual[sq, 3] = rec[sq_off + 137 + 3]   # is_threatened

    diff_mask = (expected != actual)  # (64, 4)
    if diff_mask.any():
      for ch in range(4):
        mismatches_per_ch[ch] += int(diff_mask[:, ch].sum())
      if first_mismatch is None:
        # Find the first square+channel that differs
        for sq in range(64):
          for ch in range(4):
            if diff_mask[sq, ch]:
              first_mismatch = (rec_idx, sq, ch_names[ch], int(expected[sq, ch]), int(actual[sq, ch]), board.fen())
              break
          if first_mismatch:
            break

    total_pieces += sum(1 for sq in range(64) if board.piece_at(sq) is not None)
    total_sq += 64

  total_mismatches = sum(mismatches_per_ch)
  total_comparisons = total_sq * 4
  print()
  print(f"Records compared:      {actual_n:,}")
  print(f"Pieces seen:           {total_pieces:,}")
  print(f"Total byte comparisons: {total_comparisons:,}")
  print(f"Mismatches per channel:")
  for ch, name in enumerate(ch_names):
    pct = (mismatches_per_ch[ch] / total_sq * 100) if total_sq else 0
    print(f"  [{ch+3}] {name:<16s}: {mismatches_per_ch[ch]:>8,} ({pct:.4f}% of squares)")
  print(f"Total mismatches:      {total_mismatches:,}")

  if total_mismatches == 0:
    print()
    print("PASS — all 4 new aux channels match Python oracle byte-for-byte.")
    sys.exit(0)
  else:
    print()
    print(f"FAIL — mismatches detected.")
    if first_mismatch:
      rec_idx, sq, ch, exp, act, fen = first_mismatch
      print(f"  First mismatch: record {rec_idx}, sq {sq}, channel {ch}")
      print(f"    expected={exp}  actual={act}")
      print(f"    FEN: {fen}")
    sys.exit(1)


if __name__ == '__main__':
  main()
