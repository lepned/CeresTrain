# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
"""
# End of License Notice

"""Augmented input-feature MVP: per-square attack counts.

Computes three feature channels per square that the network currently has to
derive from raw piece placement:
  0: our_attackers   * (100 / 8) / 100  ≈ our_count / 8        ∈ [0, 1]
  1: opp_attackers   * (100 / 8) / 100  ≈ opp_count / 8        ∈ [0, 1]
  2: (our - opp + 8) * (100 / 16) / 100 ≈ (net + 8) / 16       ∈ [0, 1]

Quantization matches C# inference exactly (byte * 100 / 8, byte / 100 →
float in [0,1]) so training and inference features are bit-identical.

Two implementations live here:

  compute_aux_features_batch        (FAST: numpy-vectorized, ~30 ms / batch)
                                    Default production path. Uses bitboard
                                    representations + Kogge-Stone occluded
                                    fills for sliders + precomputed attack
                                    tables for non-sliders.

  compute_aux_features_batch_slow   (REFERENCE: python-chess, ~350 ms / batch)
                                    Per-board chess.Board reconstruction +
                                    128 attackers_mask calls per board.
                                    Retained as oracle for `_validate_equality`.

A standalone equality test (`python aux_features.py validate`) runs both on
1000 random positions and asserts byte-identical output.

The TPG per-square encoding (per scripts/inspect_tpg_for_960.py and
scripts/validate_tpg_shard.py) places the current-position piece one-hot in
bytes [0:13] of each 137-byte square slot:
  0  = empty
  1..6  = our  pieces (P, N, B, R, Q, K)
  7..12 = opp  pieces (P, N, B, R, Q, K)
"""

import os
import sys
import time
import numpy as np


NUM_AUG_CHANNELS = 3

# ============================================================================
# Bitboard layout
# ----------------------------------------------------------------------------
# Standard python-chess convention: bit i = square i, a1 = 0, h8 = 63.
# File f = sq & 7; rank r = sq >> 3.
# ============================================================================

_U64 = np.uint64

NOT_FILE_A = _U64(0xFEFEFEFEFEFEFEFE)   # all squares except file a (clear bit when shifting west)
NOT_FILE_H = _U64(0x7F7F7F7F7F7F7F7F)   # all squares except file h
RANK_2     = _U64(0x000000000000FF00)
RANK_7     = _U64(0x00FF000000000000)


# ============================================================================
# Precomputed non-slider attack tables
# ----------------------------------------------------------------------------
# Convention: attack_table[t] = bitmask of squares from which a piece of this
# type attacks square t. For knight/king, symmetric with attacks-from.
# For pawns: attackers_of_t depends on attacker color (white pawns attack
# diagonally forward, so a WHITE attacker of t is at t-9 or t-7).
# ============================================================================

def _init_knight_attackers_of():
    deltas = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
    t = np.zeros(64, dtype=_U64)
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        m = _U64(0)
        for df, dr in deltas:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                m |= _U64(1) << _U64(nr * 8 + nf)
        t[sq] = m
    return t


def _init_king_attackers_of():
    t = np.zeros(64, dtype=_U64)
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        m = _U64(0)
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                nf, nr = f + df, r + dr
                if 0 <= nf < 8 and 0 <= nr < 8:
                    m |= _U64(1) << _U64(nr * 8 + nf)
        t[sq] = m
    return t


def _init_pawn_attackers_of(white_attacker: bool):
    """Where would a [white/black] pawn need to be to attack square t?"""
    t = np.zeros(64, dtype=_U64)
    dr = -1 if white_attacker else 1   # white pawns attack +1 rank → attacker is one rank below target
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        m = _U64(0)
        for df in (-1, 1):
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                m |= _U64(1) << _U64(nr * 8 + nf)
        t[sq] = m
    return t


KNIGHT_ATTACKERS_OF      = _init_knight_attackers_of()
KING_ATTACKERS_OF        = _init_king_attackers_of()
PAWN_ATTACKERS_OF_WHITE  = _init_pawn_attackers_of(white_attacker=True)
PAWN_ATTACKERS_OF_BLACK  = _init_pawn_attackers_of(white_attacker=False)


# ============================================================================
# Vectorized popcount on (uint64) arrays
# ============================================================================

_M1  = _U64(0x5555555555555555)
_M2  = _U64(0x3333333333333333)
_M4  = _U64(0x0F0F0F0F0F0F0F0F)
_H01 = _U64(0x0101010101010101)


def _popcount(x: np.ndarray) -> np.ndarray:
    """Popcount of every element of a uint64 ndarray, returned as uint8."""
    x = x - ((x >> _U64(1)) & _M1)
    x = (x & _M2) + ((x >> _U64(2)) & _M2)
    x = (x + (x >> _U64(4))) & _M4
    return ((x * _H01) >> _U64(56)).astype(np.uint8)


# ============================================================================
# Bitboard extraction from TPG squares tensor
# ============================================================================

_POWERS_OF_2 = (_U64(1) << np.arange(64, dtype=_U64))   # (64,) uint64: 1, 2, 4, ..., 2^63


def _piece_bitboards(squares: np.ndarray):
    """Extract per-piece-type bitboards from a (B, 64, 137) squares tensor.

    Returns a list of 13 elements; indices 1..12 are (B,) uint64 bitboards
    for piece classes 1..12 (TPG one-hot indices). Index 0 is None (empty).
    """
    # squares[:, :, 1:13] is the one-hot (after /100, values ~0 or ~1)
    # mask threshold 0.001 distinguishes "set" from "clear" in float repr
    bbs = [None]
    for cls in range(1, 13):
        mask_b64 = squares[:, :, cls] > 0.001            # (B, 64) bool
        bb = (mask_b64.astype(_U64) * _POWERS_OF_2).sum(axis=1, dtype=_U64)  # (B,)
        bbs.append(bb)
    return bbs


# ============================================================================
# Slider attack via Kogge-Stone occluded fill (vectorized over batch)
# ----------------------------------------------------------------------------
# For each target square s and each ray direction, compute (B,) uint64 with
# the FIRST blocker bit set (or 0 if no blocker on board). Used to find
# slider attackers of s: blocker AND (rooks|queens) → rook/queen attackers
# from this direction; blocker AND (bishops|queens) → bishop/queen attackers.
# ============================================================================

def _shift_n(b):  return  b << _U64(8)
def _shift_s(b):  return  b >> _U64(8)
def _shift_e(b):  return (b << _U64(1)) & NOT_FILE_A   # shift east: bit moves to higher file
def _shift_w(b):  return (b >> _U64(1)) & NOT_FILE_H
def _shift_ne(b): return (b << _U64(9)) & NOT_FILE_A
def _shift_nw(b): return (b << _U64(7)) & NOT_FILE_H
def _shift_se(b): return (b >> _U64(7)) & NOT_FILE_A
def _shift_sw(b): return (b >> _U64(9)) & NOT_FILE_H


def _first_blocker_via_fill(s_bit_scalar: int, occ: np.ndarray, shift_fn):
    """For target bit s (scalar int) and per-board occupancy, return per-board
    bitboard with the FIRST occupied square along the ray set (0 if none).

    Algorithm (Kogge-Stone style occluded fill):
      cursor = s
      cursor |= shift(cursor) & empty    [×3 doublings: 1, 2, 4 steps]
      blocker = shift(cursor) & occupied

    occ: (B,) uint64
    Returns: (B,) uint64
    """
    s = _U64(s_bit_scalar)
    empty = ~occ
    cursor = np.full_like(occ, s, dtype=_U64)
    # cursor at this point contains s; propagate through empties.
    # Kogge-Stone doubling: extend by 1, then 2, then 4 steps each iteration.
    cursor |= shift_fn(cursor) & empty
    cursor |= shift_fn(shift_fn(cursor)) & shift_fn(empty) & empty
    cursor |= shift_fn(shift_fn(shift_fn(shift_fn(cursor)))) & \
              shift_fn(shift_fn(shift_fn(empty))) & shift_fn(shift_fn(empty)) & shift_fn(empty) & empty
    # cursor now covers s plus all empties reachable in up to 7 steps.
    # The first blocker is one step beyond cursor (and is occupied).
    return shift_fn(cursor) & occ


# Slider direction tables
_ROOK_DIRS   = [_shift_n, _shift_s, _shift_e, _shift_w]
_BISHOP_DIRS = [_shift_ne, _shift_nw, _shift_se, _shift_sw]


# ============================================================================
# Main vectorized entry point
# ============================================================================

def compute_aux_features_batch(squares: np.ndarray) -> np.ndarray:
    """Batched aug-feature compute. (B, 64, 137) → (B, 64, 3) float32.

    Fully vectorized over the batch dimension. Per batch of 4096:
      - piece bitboard extraction: ~5 ms
      - non-slider attacker counts: ~3 ms
      - slider attacker counts: ~20 ms
      - encode + assemble: ~2 ms
    Total: ~30 ms (vs ~350 ms for the python-chess reference).

    Exactly bit-equivalent to compute_aux_features_batch_slow — validated by
    `_validate_equality()` on 1000 random positions.
    """
    B = squares.shape[0]
    bbs = _piece_bitboards(squares)

    our_pieces  = bbs[1] | bbs[2] | bbs[3] | bbs[4] | bbs[5] | bbs[6]
    opp_pieces  = bbs[7] | bbs[8] | bbs[9] | bbs[10] | bbs[11] | bbs[12]
    occ         = our_pieces | opp_pieces

    # Initialize per-color, per-target-square attacker count.
    our_count = np.zeros((B, 64), dtype=np.uint8)
    opp_count = np.zeros((B, 64), dtype=np.uint8)

    # ---- Non-sliders: precomputed (64,) attacker-of-target tables. ----
    # For each target t, attackers of t by piece type T (our color) = popcount(our_T_bb & ATTACKERS_OF[t])
    # Vectorized via broadcasting: (B, 1) & (1, 64) = (B, 64).
    our_count += _popcount(bbs[1][:, None] & PAWN_ATTACKERS_OF_WHITE[None, :])   # our pawns (= WHITE pawns under us=WHITE convention)
    our_count += _popcount(bbs[2][:, None] & KNIGHT_ATTACKERS_OF[None, :])
    our_count += _popcount(bbs[6][:, None] & KING_ATTACKERS_OF[None, :])

    opp_count += _popcount(bbs[7][:, None]  & PAWN_ATTACKERS_OF_BLACK[None, :])  # opp pawns (= BLACK)
    opp_count += _popcount(bbs[8][:, None]  & KNIGHT_ATTACKERS_OF[None, :])
    opp_count += _popcount(bbs[12][:, None] & KING_ATTACKERS_OF[None, :])

    # ---- Sliders: per-target-square, per-direction first-blocker. ----
    # For each target sq t and direction d, the first blocker bit AND with
    # appropriate slider bitboard gives 1 if a slider of that color attacks
    # t from direction d. Use bool (0/1) since at most one attacker per direction.
    our_rq = bbs[4] | bbs[5]   # our R + Q
    our_bq = bbs[3] | bbs[5]   # our B + Q
    opp_rq = bbs[10] | bbs[11]
    opp_bq = bbs[9]  | bbs[11]

    for t in range(64):
        s_bit = 1 << t
        for shift_fn in _ROOK_DIRS:
            blocker = _first_blocker_via_fill(s_bit, occ, shift_fn)
            our_count[:, t] += ((blocker & our_rq) != 0).astype(np.uint8)
            opp_count[:, t] += ((blocker & opp_rq) != 0).astype(np.uint8)
        for shift_fn in _BISHOP_DIRS:
            blocker = _first_blocker_via_fill(s_bit, occ, shift_fn)
            our_count[:, t] += ((blocker & our_bq) != 0).astype(np.uint8)
            opp_count[:, t] += ((blocker & opp_bq) != 0).astype(np.uint8)

    # ---- Encode to bytes-divided-by-100 ----
    # Match C# encoding exactly: integer divide, no float intermediate.
    our32 = our_count.astype(np.int32)
    opp32 = opp_count.astype(np.int32)
    out = np.empty((B, 64, 3), dtype=np.float32)
    out[:, :, 0] = (our32 * 100 // 8) / 100.0
    out[:, :, 1] = (opp32 * 100 // 8) / 100.0
    out[:, :, 2] = ((our32 - opp32 + 8) * 100 // 16) / 100.0
    return out


# ============================================================================
# Reference (slow) implementation — kept as validation oracle
# ============================================================================

def _slow_squares_to_board(sq_tensor):
    """Reference: TPG squares tensor → python-chess.Board. Used only for validation."""
    import chess
    _PIECE_MAP = {
        1:  (chess.PAWN,   chess.WHITE),  2:  (chess.KNIGHT, chess.WHITE),
        3:  (chess.BISHOP, chess.WHITE),  4:  (chess.ROOK,   chess.WHITE),
        5:  (chess.QUEEN,  chess.WHITE),  6:  (chess.KING,   chess.WHITE),
        7:  (chess.PAWN,   chess.BLACK),  8:  (chess.KNIGHT, chess.BLACK),
        9:  (chess.BISHOP, chess.BLACK), 10:  (chess.ROOK,   chess.BLACK),
        11: (chess.QUEEN,  chess.BLACK), 12:  (chess.KING,   chess.BLACK),
    }
    board = chess.Board.empty()
    onehot = sq_tensor[:, 0:13]
    cls_per_square = onehot.argmax(axis=1)
    occupied_max = onehot.max(axis=1)
    for sq_idx in range(64):
        if occupied_max[sq_idx] <= 1e-3:
            continue
        cls = int(cls_per_square[sq_idx])
        if cls == 0:
            continue
        ptype, color = _PIECE_MAP[cls]
        board.set_piece_at(sq_idx, chess.Piece(ptype, color))
    return board


def _slow_aux_features_for_board(board):
    """Reference per-board aug features via python-chess."""
    import chess
    feats = np.zeros((64, NUM_AUG_CHANNELS), dtype=np.float32)
    for sq in range(64):
        w = bin(board.attackers_mask(chess.WHITE, sq)).count('1')
        b = bin(board.attackers_mask(chess.BLACK, sq)).count('1')
        feats[sq, 0] = (w * 100 // 8) / 100.0
        feats[sq, 1] = (b * 100 // 8) / 100.0
        feats[sq, 2] = ((w - b + 8) * 100 // 16) / 100.0
    return feats


def compute_aux_features_batch_slow(squares: np.ndarray) -> np.ndarray:
    """REFERENCE python-chess implementation. ~350 ms per batch of 4096.
    Retained as oracle for the equality test against the fast vectorized version."""
    B = squares.shape[0]
    out = np.zeros((B, 64, NUM_AUG_CHANNELS), dtype=np.float32)
    for i in range(B):
        board = _slow_squares_to_board(squares[i])
        out[i] = _slow_aux_features_for_board(board)
    return out


# ============================================================================
# Validation: fast vs slow on random positions
# ============================================================================

def _board_to_squares_tensor(board) -> np.ndarray:
    """Encode a python-chess Board into TPG-style (64, 137) one-hot squares tensor.
    Only the first 13 channels (piece one-hot) are filled; the rest are 0.
    Used in equality test to generate identical input for both implementations.
    """
    import chess
    _CHESS_TO_TPG_CLS = {
        (chess.PAWN,   chess.WHITE): 1,  (chess.KNIGHT, chess.WHITE): 2,
        (chess.BISHOP, chess.WHITE): 3,  (chess.ROOK,   chess.WHITE): 4,
        (chess.QUEEN,  chess.WHITE): 5,  (chess.KING,   chess.WHITE): 6,
        (chess.PAWN,   chess.BLACK): 7,  (chess.KNIGHT, chess.BLACK): 8,
        (chess.BISHOP, chess.BLACK): 9,  (chess.ROOK,   chess.BLACK): 10,
        (chess.QUEEN,  chess.BLACK): 11, (chess.KING,   chess.BLACK): 12,
    }
    sq = np.zeros((64, 137), dtype=np.float32)
    for sq_idx in range(64):
        piece = board.piece_at(sq_idx)
        if piece is None:
            continue
        cls = _CHESS_TO_TPG_CLS[(piece.piece_type, piece.color)]
        sq[sq_idx, cls] = 1.0
    return sq


def _validate_equality(n_positions: int = 1000, seed: int = 42) -> int:
    """Generate `n_positions` random chess positions, compute aug features via
    BOTH implementations, assert bit-exact byte equality.
    Returns 0 on success, nonzero on first mismatch.
    """
    import chess
    rng = np.random.RandomState(seed)

    # Build a mix of positions: starting position + random play-out from start
    boards = [chess.Board()]
    while len(boards) < n_positions:
        b = boards[-1].copy()
        if b.is_game_over() or rng.random() < 0.005:
            b = chess.Board()
        moves = list(b.legal_moves)
        if not moves:
            b = chess.Board()
            moves = list(b.legal_moves)
        m = moves[rng.randint(len(moves))]
        b.push(m)
        boards.append(b)

    # Encode all to a single squares tensor for vectorized run
    sq_batch = np.stack([_board_to_squares_tensor(b) for b in boards], axis=0)

    print(f'[validate] {n_positions} positions encoded; running both implementations...')

    t0 = time.perf_counter()
    fast = compute_aux_features_batch(sq_batch)
    t_fast = time.perf_counter() - t0
    print(f'  fast  : {t_fast*1000:.1f} ms ({t_fast*1000/n_positions:.3f} ms/pos)')

    t0 = time.perf_counter()
    slow = compute_aux_features_batch_slow(sq_batch)
    t_slow = time.perf_counter() - t0
    print(f'  slow  : {t_slow*1000:.1f} ms ({t_slow*1000/n_positions:.3f} ms/pos)')
    print(f'  speedup: {t_slow/t_fast:.1f}×')

    # Compare element-wise. Tolerance 0: must be bit-exact (we use integer quantization).
    if np.array_equal(fast, slow):
        print(f'[validate] PASS — {n_positions} positions × 64 squares × 3 channels = {n_positions*192} byte values, ZERO mismatches')
        return 0

    # Find first mismatch for diagnostic
    mismatch_mask = fast != slow
    n_mismatches = int(mismatch_mask.sum())
    print(f'[validate] FAIL — {n_mismatches} mismatches out of {fast.size} elements')
    first_idx = np.argwhere(mismatch_mask)[0]
    pos, sq, ch = first_idx
    ch_name = ('our', 'opp', 'net')[ch]
    print(f'  first mismatch at pos={pos}, sq={sq}, channel={ch_name}: fast={fast[pos, sq, ch]} slow={slow[pos, sq, ch]}')
    print(f'  FEN: {boards[pos].fen()}')
    return 1


# ============================================================================
# Standalone smoke + selftest
# ============================================================================

def _selftest():
    """Quick self-test on the starting position (used by Phase 1)."""
    import chess
    start = chess.Board()
    sq = _board_to_squares_tensor(start)

    # Slow reference
    feats_slow = _slow_aux_features_for_board(start)
    assert feats_slow.shape == (64, 3)
    white_sum = feats_slow[:, 0].sum()
    black_sum = feats_slow[:, 1].sum()
    net_mean  = feats_slow[:, 2].mean()
    assert abs(white_sum - black_sum) < 1e-3
    assert abs(net_mean - 0.5) < 1e-2

    # Fast version on the same single position (B=1 batch)
    feats_fast = compute_aux_features_batch(sq[np.newaxis, :, :])[0]
    if not np.array_equal(feats_fast, feats_slow):
        diff = np.argwhere(feats_fast != feats_slow)
        print(f'[selftest] starting-pos MISMATCH at {len(diff)} positions')
        for d in diff[:5]:
            sqi, ch = int(d[0]), int(d[1])
            print(f'  sq={sqi}, ch={ch}: fast={feats_fast[sqi, ch]} slow={feats_slow[sqi, ch]}')
        sys.exit(1)

    print(f'[aux_features selftest] OK')
    print(f'  starting pos: white-attackers-encoded sum = {white_sum:.3f}')
    print(f'  starting pos: black-attackers-encoded sum = {black_sum:.3f}')
    print(f'  starting pos: net-shifted mean           = {net_mean:.3f} (~0.5 = symmetric)')
    print(f'  fast == slow on starting position ✓')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'validate':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        sys.exit(_validate_equality(n))
    else:
        _selftest()
