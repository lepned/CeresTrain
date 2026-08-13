# Correctness gate for chess_geometry.VisibilityChannels (channel builder v2).
#
# Compares every channel (vis / xray / pinray, stm+opp, outgoing) against a
# python-chess oracle on random-playout positions, verifies the incoming
# channels are exact transposes, and asserts three doc-anchored edges from the
# Kovax program's worked position B (VISIBILITY_PROGRAM.md sec. 14.3):
#   Qh4->h8 xray (through the h7 pawn), h7->h8 pinray, Bb3->g8 visibility
#   onto an EMPTY square (the no-target-mask property).
#
# Run (WSL venv, CPU):
#   /home/lep/cerestrain-env/bin/python3 scripts/validate_visibility_channels.py [num_positions]
#
# Exit 0 = all positions match exactly; nonzero = mismatches printed.

import os
import random
import sys

import chess
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src', 'CeresTrainPy'))
from chess_geometry import VisibilityChannels

ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISH_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def board_to_onehot(board: chess.Board) -> torch.Tensor:
    """[64, 13] one-hot, white -> channels 1..6, black -> 7..12, 0 = empty.
    Square index = rank*8 + file (python-chess convention, matches builder)."""
    oh = torch.zeros(64, 13)
    for sq in range(64):
        pc = board.piece_at(sq)
        if pc is None:
            oh[sq, 0] = 1.0
        else:
            oh[sq, pc.piece_type + (0 if pc.color == chess.WHITE else 6)] = 1.0
    return oh


def oracle_edges(board: chess.Board, color: bool):
    """Return (vis, xray, pinray) as sets of (from_sq, to_sq) for `color`.
    vis:    piece reach, blocker-exact for sliders, NO target-side mask,
            pawn = capture diagonals + single push square (no double push).
    xray:   slider ray squares with strictly-between occupancy == exactly 1.
    pinray: (blocker, target) for each xray edge's single blocker."""
    vis, xray, pinray = set(), set(), set()
    for sq in range(64):
        pc = board.piece_at(sq)
        if pc is None or pc.color != color:
            continue
        t = pc.piece_type
        if t in (chess.KNIGHT, chess.KING):
            for to in board.attacks(sq):
                vis.add((sq, to))
        elif t == chess.PAWN:
            for to in board.attacks(sq):
                vis.add((sq, to))
            push = sq + 8 if color == chess.WHITE else sq - 8
            if 0 <= push < 64:
                vis.add((sq, push))
        else:  # B / R / Q sliders
            dirs = (ROOK_DIRS if t == chess.ROOK else
                    BISH_DIRS if t == chess.BISHOP else ROOK_DIRS + BISH_DIRS)
            for df, dr in dirs:
                f, r = sq % 8 + df, sq // 8 + dr
                cnt, first_blocker = 0, None
                while 0 <= f < 8 and 0 <= r < 8:
                    to = r * 8 + f
                    if cnt == 0:
                        vis.add((sq, to))
                    elif cnt == 1:
                        xray.add((sq, to))
                        pinray.add((first_blocker, to))
                    if board.piece_at(to) is not None:
                        cnt += 1
                        if cnt == 1:
                            first_blocker = to
                        elif cnt >= 2:
                            break
                    f, r = f + df, r + dr
    return vis, xray, pinray


def channel_to_set(ch: torch.Tensor):
    """[64, 64] {0,1} tensor -> set of (q, k) edges."""
    idx = (ch > 0.5).nonzero(as_tuple=False)
    return {(int(q), int(k)) for q, k in idx}


def check_position(module, board, label):
    oh = board_to_onehot(board).unsqueeze(0)
    E = module(oh)[0]  # [64, 64, 12]
    fails = []
    # Channel layout per family f (0=vis, 1=xray, 2=pinray):
    #   4f+0 stm_out, 4f+1 opp_out, 4f+2 stm_in, 4f+3 opp_in
    # In this validator "stm" channels (one-hot 1..6) = WHITE pieces.
    for side_i, color in ((0, chess.WHITE), (1, chess.BLACK)):
        o_vis, o_xray, o_pin = oracle_edges(board, color)
        for f, (fam, want) in enumerate((('vis', o_vis), ('xray', o_xray), ('pinray', o_pin))):
            got = channel_to_set(E[:, :, 4 * f + side_i])
            if got != want:
                fails.append(f'{label} {fam} side={side_i}: '
                             f'missing={sorted(want - got)[:5]} extra={sorted(got - want)[:5]} '
                             f'(want {len(want)}, got {len(got)})')
            # Incoming channel must be the exact transpose.
            got_in = channel_to_set(E[:, :, 4 * f + 2 + side_i])
            if got_in != {(k, q) for q, k in got}:
                fails.append(f'{label} {fam} side={side_i}: IN channel is not the transpose')
    return fails


def main():
    n_positions = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    random.seed(42)
    module = VisibilityChannels()
    assert module.num_channels == 12

    all_fails = []

    # --- Doc-anchored asserts: worked position B, VISIBILITY_PROGRAM.md 14.3 ---
    board_b = chess.Board('2b1qr1k/1pp1n2p/3p1p2/4pNp1/4P2Q/rB1PPR2/P1P3PP/R5K1 w - - 0 1')
    E = module(board_to_onehot(board_b).unsqueeze(0))[0]
    H4, H7, H8, B3, G8, F5, G7 = (chess.H4, chess.H7, chess.H8, chess.B3,
                                  chess.G8, chess.F5, chess.G7)
    anchors = (
        ('Qh4->h8 white xray', E[H4, H8, 4 * 1 + 0]),
        ('h7->h8 white pinray', E[H7, H8, 4 * 2 + 0]),
        ('Bb3->g8 white vis onto EMPTY square', E[B3, G8, 4 * 0 + 0]),
        ('Nf5->g7 white vis', E[F5, G7, 4 * 0 + 0]),
    )
    for name, v in anchors:
        if v.item() != 1.0:
            all_fails.append(f'ANCHOR FAILED: {name} = {v.item()}')
    all_fails += check_position(module, board_b, 'worked-position-B')

    # --- Random playout positions (includes startpos) ---
    checked = 1
    all_fails += check_position(module, chess.Board(), 'startpos')
    while checked < n_positions and len(all_fails) < 40:
        board = chess.Board()
        ply_target = random.randint(4, 120)
        for _ in range(ply_target):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(random.choice(moves))
        checked += 1
        all_fails += check_position(module, board, f'playout-{checked}(ply~{board.ply()})')

    if all_fails:
        print(f'FAIL: {len(all_fails)} mismatches over {checked} positions')
        for f in all_fails[:40]:
            print(' ', f)
        sys.exit(1)
    print(f'OK: {checked} positions, all 12 channels exact vs python-chess oracle; '
          f'4 doc anchors hold')


if __name__ == '__main__':
    main()
