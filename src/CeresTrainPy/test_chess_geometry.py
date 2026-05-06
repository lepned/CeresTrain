"""Sanity checks for chess_geometry.STATIC_GEOMETRY_FEATURES and PIECE_PSEUDO_ATTACK.

Run from CeresTrainPy as:
    python test_chess_geometry.py

Verifies that the precomputed tables match expected chess relationships on
hand-picked square pairs. Doesn't catch every bug (especially around
side-of-board pawn semantics) but catches gross mistakes in the indexing.
"""
import torch
from chess_geometry import (
    STATIC_GEOMETRY_FEATURES, PIECE_PSEUDO_ATTACK, NUM_STATIC_FEATURES,
    PieceRelationBias,
)


def _sq(file: int, rank: int) -> int:
    return rank * 8 + file


def main():
    print("STATIC_GEOMETRY_FEATURES shape:", STATIC_GEOMETRY_FEATURES.shape)
    print("PIECE_PSEUDO_ATTACK shape:", PIECE_PSEUDO_ATTACK.shape)
    print()

    # Test static features.
    a1 = _sq(0, 0); h1 = _sq(7, 0); a8 = _sq(0, 7); e4 = _sq(4, 3); d5 = _sq(3, 4)
    e5 = _sq(4, 4); h8 = _sq(7, 7); g8 = _sq(6, 7); a4 = _sq(0, 3)

    # Same square
    assert STATIC_GEOMETRY_FEATURES[0, e4, e4] == 1.0
    assert STATIC_GEOMETRY_FEATURES[0, e4, e5] == 0.0
    # Same file (e-file)
    assert STATIC_GEOMETRY_FEATURES[1, e4, e5] == 1.0
    # Same rank (rank 4 is the 4th rank = rank index 3)
    assert STATIC_GEOMETRY_FEATURES[2, a4, e4] == 1.0
    # Same diagonal (a1-h8)
    assert STATIC_GEOMETRY_FEATURES[3, a1, h8] == 1.0
    # Same anti-diagonal (h1-a8)
    assert STATIC_GEOMETRY_FEATURES[4, h1, a8] == 1.0
    # King distance (a1-h1 = 7) normalized
    assert abs(STATIC_GEOMETRY_FEATURES[5, a1, h1] - 1.0) < 1e-6
    print("OK: static-geometry features pass spot checks.")

    # Test piece pseudo-attacks.
    WP, WN, WB, WR, WQ, WK = 1, 2, 3, 4, 5, 6
    BP, BN, BB, BR, BQ, BK = 7, 8, 9, 10, 11, 12

    # Knight on b1 (sq 1) attacks a3 (sq 16), c3 (sq 18), d2 (sq 11)
    assert PIECE_PSEUDO_ATTACK[WN, _sq(1, 0), _sq(0, 2)] == 1.0  # b1 -> a3
    assert PIECE_PSEUDO_ATTACK[WN, _sq(1, 0), _sq(2, 2)] == 1.0  # b1 -> c3
    assert PIECE_PSEUDO_ATTACK[WN, _sq(1, 0), _sq(3, 1)] == 1.0  # b1 -> d2
    # Knight does not attack diagonals
    assert PIECE_PSEUDO_ATTACK[WN, _sq(1, 0), _sq(0, 1)] == 0.0  # b1 -> a2 (no)
    print("OK: knight pseudo-attacks pass spot checks.")

    # Bishop on a1 attacks the entire long diagonal
    for r in range(1, 8):
        assert PIECE_PSEUDO_ATTACK[WB, a1, _sq(r, r)] == 1.0
    # Bishop on a1 does NOT attack along ranks/files (pseudo)
    assert PIECE_PSEUDO_ATTACK[WB, a1, _sq(7, 0)] == 0.0  # h1
    assert PIECE_PSEUDO_ATTACK[WB, a1, _sq(0, 7)] == 0.0  # a8
    print("OK: bishop pseudo-attacks pass spot checks.")

    # Rook on e4 attacks all of e-file and rank 4
    for r in range(8):
        if r != 3:
            assert PIECE_PSEUDO_ATTACK[WR, e4, _sq(4, r)] == 1.0
    for f in range(8):
        if f != 4:
            assert PIECE_PSEUDO_ATTACK[WR, e4, _sq(f, 3)] == 1.0
    # Rook does NOT attack diagonals
    assert PIECE_PSEUDO_ATTACK[WR, e4, _sq(5, 4)] == 0.0  # f5
    print("OK: rook pseudo-attacks pass spot checks.")

    # Queen = rook + bishop union
    assert PIECE_PSEUDO_ATTACK[WQ, e4, _sq(5, 4)] == 1.0  # f5 (diagonal)
    assert PIECE_PSEUDO_ATTACK[WQ, e4, _sq(4, 7)] == 1.0  # e8 (file)
    print("OK: queen pseudo-attacks pass spot checks.")

    # White pawn on e4 attacks d5 and f5
    assert PIECE_PSEUDO_ATTACK[WP, e4, _sq(3, 4)] == 1.0  # d5
    assert PIECE_PSEUDO_ATTACK[WP, e4, _sq(5, 4)] == 1.0  # f5
    # White pawn does NOT attack the push square (e5)
    assert PIECE_PSEUDO_ATTACK[WP, e4, _sq(4, 4)] == 0.0
    # Black pawn on e5 attacks d4 and f4
    assert PIECE_PSEUDO_ATTACK[BP, e5, _sq(3, 3)] == 1.0  # d4
    assert PIECE_PSEUDO_ATTACK[BP, e5, _sq(5, 3)] == 1.0  # f4
    print("OK: pawn pseudo-attacks pass spot checks.")

    # Empty (channel 0) attacks nothing
    assert PIECE_PSEUDO_ATTACK[0].sum() == 0.0
    print("OK: empty piece type attacks nothing.")

    # Test PieceRelationBias module produces correct-shaped bias.
    NUM_HEADS = 8
    mod = PieceRelationBias(num_heads=NUM_HEADS)
    B = 3
    pt = torch.zeros(B, 64, 13)
    # Place a knight on b1 in batch 0
    pt[0, _sq(1, 0), WN] = 1.0
    bias = mod(pt)
    assert bias.shape == (B, NUM_HEADS, 64, 64), f"bad shape: {bias.shape}"
    # Bias should not be identically zero (the projection has randomness)
    assert bias.abs().sum() > 0
    print(f"OK: PieceRelationBias output shape = {tuple(bias.shape)}")
    print("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
