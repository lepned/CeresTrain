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
    PieceRelationBias, VisibilityChannels,
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

    # ---- VisibilityChannels check/flight families (2026-08 tactical program) ----
    vc = VisibilityChannels(families=('vis', 'check', 'flight'))
    CHECK = vc.family_slices['check']
    FLIGHT = vc.family_slices['flight']

    def _board(*pieces):
        """Build a [64, 13] one-hot with a correct EMPTY channel (channel 0
        set everywhere a piece is not) from (square, type) pairs."""
        pt_one = torch.zeros(64, 13)
        pt_one[:, 0] = 1.0
        for s, t in pieces:
            pt_one[s, 0] = 0.0
            pt_one[s, t] = 1.0
        return pt_one

    def _E(pt_one):
        return vc(pt_one.unsqueeze(0))[0]  # [64, 64, C]

    a2 = _sq(0, 1); a3 = _sq(0, 2); a4v = _sq(0, 3); a8v = _sq(0, 7)
    b1 = _sq(1, 0); c3 = _sq(2, 2); d5v = _sq(3, 4); d6 = _sq(3, 5)
    e5v = _sq(4, 4); h1v = _sq(7, 0); g7 = _sq(6, 6); g8v = _sq(6, 7)
    h7 = _sq(7, 6); h8v = _sq(7, 7)

    # check: stm rook a1, opp king a8 -> Ra8 delivers check trivially, but so
    # does any arrival square still on the a-file/8th-rank sightline. Rh1 checks
    # along rank... no: from h1 the rook attacks a1 (rook there) not a8. From b1
    # no check. From a2..a7 -> check (a-file clear above).
    pt1 = _board((a1, WR), (a8v, BK))
    E1 = _E(pt1)[:, :, CHECK]  # [64, 64, 4]: stm_out, opp_out, stm_in, opp_in
    assert E1[a1, a2, 0] == 1.0, "Ra2+ missed"
    assert E1[a1, _sq(0, 6), 0] == 1.0, "Ra7+ missed"
    assert E1[a1, _sq(1, 0), 0] == 0.0, "Rb1 is not check"
    assert E1[a1, h1v, 0] == 0.0, "Rh1 is not check"
    # in-channel = transpose of out-channel
    assert E1[a2, a1, 2] == 1.0

    # check blocker-awareness: own pawn a4 blocks — arrival at a2/a3 no longer
    # sees a8 (pawn in the way), so no check edge there.
    pt2 = _board((a1, WR), (a8v, BK), (a4v, WP))
    E2 = _E(pt2)[:, :, CHECK]
    assert E2[a1, a2, 0] == 0.0, "blocked Ra2 wrongly flagged as check"
    assert E2[a1, a3, 0] == 0.0, "blocked Ra3 wrongly flagged as check"

    # knight check: stm knight b1, opp king d5 -> Nc3 attacks d5.
    pt3 = _board((b1, WN), (d5v, BK))
    E3 = _E(pt3)[:, :, CHECK]
    assert E3[b1, c3, 0] == 1.0, "Nc3+ missed"
    assert E3[b1, _sq(0, 2), 0] == 0.0, "Na3 is not check"

    # pawn check via PUSH arrival: stm pawn e4, opp king d6 -> e5 attacks d6.
    pt4 = _board((e4, WP), (d6, BK))
    E4 = _E(pt4)[:, :, CHECK]
    assert E4[e4, e5v, 0] == 1.0, "e5+ (push arrival) missed"

    # opp-side check channel: opp rook h8, stm king a1 -> Rh1+ (arrival h1
    # attacks a1 along rank 1).
    pt5 = _board((h8v, BR), (a1, WK))
    E5 = _E(pt5)[:, :, CHECK]
    assert E5[h8v, h1v, 1] == 1.0, "opp Rh1+ missed"
    assert E5[h8v, g8v, 1] == 0.0, "opp Rg8 is not check"

    # flight: opp king h8, stm rook h1. Ring = {g7, g8, h7}; h7 covered by the
    # rook (h-file clear), g7/g8 free. stm flight channel is query-broadcast.
    pt6 = _board((h8v, BK), (h1v, WR))
    E6 = _E(pt6)[:, :, FLIGHT]
    assert E6[0, g8v, 0] == 1.0 and E6[40, g8v, 0] == 1.0, "free g8 missed"
    assert E6[0, g7, 0] == 1.0, "free g7 missed"
    assert E6[0, h7, 0] == 0.0, "covered h7 wrongly free"
    assert E6[0, e4, 0] == 0.0, "non-ring square flagged"
    # own piece blocks flight: opp pawn g7 -> g7 no longer free.
    pt7 = _board((h8v, BK), (h1v, WR), (g7, BP))
    E7 = _E(pt7)[:, :, FLIGHT]
    assert E7[0, g7, 0] == 0.0, "own-occupied g7 wrongly free"
    # pawn PUSH square must NOT deny a flight square (attack semantics):
    # stm pawn g6 covers f7/h7 (diagonals) but NOT g7.
    pt8 = _board((h8v, BK), (_sq(6, 5), WP))
    E8 = _E(pt8)[:, :, FLIGHT]
    assert E8[0, g7, 0] == 1.0, "pawn push square wrongly denies g7"
    assert E8[0, h7, 0] == 0.0, "pawn diagonal h7 wrongly free"

    print("OK: VisibilityChannels check/flight families pass positional checks.")

    # king-shadow (2026-08-17 review fix): stm Ra8, opp Ke8. The rook's ray
    # extends THROUGH the king, so f8 (behind the king) is covered, not a
    # phantom escape square; e7/f7 (off the ray) stay free.
    e8 = _sq(4, 7); f8 = _sq(5, 7); d8 = _sq(3, 7); e7 = _sq(4, 6); f7 = _sq(5, 6)
    pt9 = _board((a8, WR), (e8, BK))
    E9 = _E(pt9)[:, :, FLIGHT]
    assert E9[0, f8, 0] == 0.0, "king-shadow: f8 behind checked king wrongly free"
    assert E9[0, d8, 0] == 0.0, "d8 (directly attacked) wrongly free"
    assert E9[0, e7, 0] == 1.0 and E9[0, f7, 0] == 1.0, "off-ray escape squares missed"

    # promotion check (auto-queen, same review round): stm pawn e7, opp Kg8.
    # e8=Q attacks g8 along rank 8 -> check edge e7->e8. A mid-board pawn
    # arrival still uses pawn semantics (no false queen edges).
    pt10 = _board((e7, WP), (g8, BK))
    E10 = _E(pt10)[:, :, CHECK]
    assert E10[e7, e8, 0] == 1.0, "e8=Q+ promotion check missed"
    pt11 = _board((e4, WP), (g8, BK))
    E11 = _E(pt11)[:, :, CHECK]
    assert E11[e4, e5v, 0] == 0.0, "mid-board pawn arrival wrongly checks like a queen"

    print("OK: king-shadow flight + promotion-check fixes pass.")
    print("ALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
