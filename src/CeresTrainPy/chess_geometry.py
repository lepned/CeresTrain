# License Notice
"""
Chess-specific structural priors for transformer attention.

Provides:
  STATIC_GEOMETRY_FEATURES   [F_static, 64, 64] tensor of position-independent
                             square-pair features (king-distance, knight-distance,
                             same-file/rank/diagonal, etc.).
  PIECE_PSEUDO_ATTACK        [13, 64, 64] tensor: PIECE_PSEUDO_ATTACK[t, i, j]
                             is 1.0 iff a piece of one-hot type t standing on
                             square i pseudo-attacks square j (ignoring blockers
                             for sliding pieces). Indexed by the same 13-channel
                             one-hot used by TPGSquareRecord:
                             0 = empty, 1..6 = WP/WN/WB/WR/WQ/WK,
                             7..12 = BP/BN/BB/BR/BQ/BK.

  PieceRelationBias          nn.Module that combines the above into a per-head
                             attention bias of shape [B, num_heads, 64, 64]
                             from a per-position one-hot piece-type tensor of
                             shape [B, 64, 13].

Squares are indexed 0..63. Square i has file = i % 8, rank = i // 8 (rank 0 is
the side-to-move side per Ceres convention; the encoding is from the perspective
of the side to move, so "white" pawns advance from rank 1 toward rank 7).

This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
"""

import math
import torch
from torch import nn


def _file_of(sq: int) -> int: return sq % 8
def _rank_of(sq: int) -> int: return sq // 8


def _build_static_geometry_features() -> torch.Tensor:
    """Return [F_static=8, 64, 64] tensor of position-independent square-pair features.

    Channels:
      0: same square (i == j)
      1: same file
      2: same rank
      3: same diagonal (file_delta == rank_delta)
      4: same anti-diagonal (file_delta == -rank_delta)
      5: king (Chebyshev) distance, normalized to [0, 1]
      6: Manhattan distance, normalized to [0, 1]
      7: knight-move distance bucket, normalized to [0, 1]
         (0 = same square, 1 = one knight move away, 2 = two moves, 3 = three or more
          or unreachable). The bucket is computed via BFS over the knight graph and
          capped at 6 to bound the value.
    """
    F = 8
    out = torch.zeros(F, 64, 64, dtype=torch.float32)

    # Knight BFS distance for every (i, j).
    KNIGHT_DELTAS = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                     (1, 2), (1, -2), (-1, 2), (-1, -2)]
    knight_dist = [[None] * 64 for _ in range(64)]
    for src in range(64):
        # BFS from src.
        knight_dist[src][src] = 0
        frontier = [src]
        d = 0
        while frontier:
            nxt = []
            d += 1
            for sq in frontier:
                fr, rk = _file_of(sq), _rank_of(sq)
                for df, dr in KNIGHT_DELTAS:
                    nf, nr = fr + df, rk + dr
                    if 0 <= nf < 8 and 0 <= nr < 8:
                        nsq = nr * 8 + nf
                        if knight_dist[src][nsq] is None:
                            knight_dist[src][nsq] = d
                            nxt.append(nsq)
            frontier = nxt
        # Fill any unreached (shouldn't happen on 8x8, but guard).
        for j in range(64):
            if knight_dist[src][j] is None:
                knight_dist[src][j] = 6

    KNIGHT_CAP = 6
    for i in range(64):
        fi, ri = _file_of(i), _rank_of(i)
        for j in range(64):
            fj, rj = _file_of(j), _rank_of(j)
            df, dr = fj - fi, rj - ri
            out[0, i, j] = 1.0 if i == j else 0.0
            out[1, i, j] = 1.0 if df == 0 else 0.0
            out[2, i, j] = 1.0 if dr == 0 else 0.0
            out[3, i, j] = 1.0 if (df == dr and df != 0) else 0.0
            out[4, i, j] = 1.0 if (df == -dr and df != 0) else 0.0
            out[5, i, j] = max(abs(df), abs(dr)) / 7.0
            out[6, i, j] = (abs(df) + abs(dr)) / 14.0
            out[7, i, j] = min(knight_dist[i][j], KNIGHT_CAP) / KNIGHT_CAP
    return out


def _build_piece_pseudo_attack() -> torch.Tensor:
    """Return [13, 64, 64] tensor where entry [t, i, j] = 1 iff a piece of one-hot
    type t at square i pseudo-attacks square j (ignoring blockers). The one-hot
    indexing matches TPGSquareRecord:
      0 = empty
      1..6 = WP, WN, WB, WR, WQ, WK
      7..12 = BP, BN, BB, BR, BQ, BK

    Pawn attacks are diagonal capture squares only (pawns don't "attack" their
    push targets in chess). Pseudo-attacks for sliding pieces (B, R, Q) ignore
    blockers — bishop on h1 "attacks" all of a8 even when a piece is on d4. This
    is a deliberate simplification for tractability; the bias mechanism uses
    these as a soft prior rather than ground truth.
    """
    out = torch.zeros(13, 64, 64, dtype=torch.float32)

    KNIGHT_DELTAS = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                     (1, 2), (1, -2), (-1, 2), (-1, -2)]
    KING_DELTAS = [(df, dr) for df in (-1, 0, 1) for dr in (-1, 0, 1) if (df, dr) != (0, 0)]
    ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    QUEEN_DIRS = ROOK_DIRS + BISHOP_DIRS

    def _slide(i, dirs):
        attacks = []
        fi, ri = _file_of(i), _rank_of(i)
        for df, dr in dirs:
            nf, nr = fi + df, ri + dr
            while 0 <= nf < 8 and 0 <= nr < 8:
                attacks.append(nr * 8 + nf)
                nf += df
                nr += dr
        return attacks

    # Per-piece-type pseudo-attack patterns from each square i.
    for i in range(64):
        fi, ri = _file_of(i), _rank_of(i)

        # Knight (white index 2, black index 8).
        for df, dr in KNIGHT_DELTAS:
            nf, nr = fi + df, ri + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                j = nr * 8 + nf
                out[2, i, j] = 1.0
                out[8, i, j] = 1.0

        # Bishop (3, 9).
        for j in _slide(i, BISHOP_DIRS):
            out[3, i, j] = 1.0
            out[9, i, j] = 1.0

        # Rook (4, 10).
        for j in _slide(i, ROOK_DIRS):
            out[4, i, j] = 1.0
            out[10, i, j] = 1.0

        # Queen (5, 11).
        for j in _slide(i, QUEEN_DIRS):
            out[5, i, j] = 1.0
            out[11, i, j] = 1.0

        # King (6, 12).
        for df, dr in KING_DELTAS:
            nf, nr = fi + df, ri + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                j = nr * 8 + nf
                out[6, i, j] = 1.0
                out[12, i, j] = 1.0

        # Pawns: white (1) attacks forward-diagonal (rank+1, file±1);
        #        black (7) attacks backward-diagonal (rank-1, file±1).
        # In the side-to-move-perspective encoding, "white" = side to move,
        # advancing toward higher ranks.
        for df in (-1, 1):
            nf = fi + df
            if 0 <= nf < 8:
                # White pawn attacks at rank+1.
                if ri + 1 < 8:
                    j = (ri + 1) * 8 + nf
                    out[1, i, j] = 1.0
                # Black pawn attacks at rank-1.
                if ri - 1 >= 0:
                    j = (ri - 1) * 8 + nf
                    out[7, i, j] = 1.0
    return out


# Module-level constants computed once at import.
STATIC_GEOMETRY_FEATURES = _build_static_geometry_features()  # [8, 64, 64]
PIECE_PSEUDO_ATTACK = _build_piece_pseudo_attack()             # [13, 64, 64]
NUM_STATIC_FEATURES = STATIC_GEOMETRY_FEATURES.shape[0]


class PieceRelationBias(nn.Module):
    """Compute a per-head attention bias of shape [B, num_heads, 64, 64] from
    the per-position 13-channel piece-type one-hot encoding.

    Features per square pair (i, j):
      - Static geometry features: same-square / same-file / same-rank /
        same-diagonal / same-anti-diagonal / king-distance / Manhattan-distance /
        knight-distance (8 channels total, position-independent).
      - Dynamic piece-attack features (4 channels):
        * white pseudo-attack from i to j (sum over white piece types)
        * black pseudo-attack from i to j (sum over black piece types)
        * white piece on i (any white piece type present)
        * black piece on i (any black piece type present)

    Total feature dim F = 12. Projected to num_heads via a single learned
    Linear without bias. The same projection is applied at every attention
    layer; only one PieceRelationBias module is constructed per network and the
    same per-batch bias tensor is reused across all encoder layers (computed
    once per forward pass).
    """

    def __init__(self, num_heads: int):
        super().__init__()
        # Precompute the static-feature projection as a single per-head bias of
        # shape [num_heads, 64, 64] at init time, then store it as a TRAINABLE
        # parameter (initialised from the projection of static features). This
        # keeps the static contribution as a learnable per-head bias without
        # any per-forward computation. Dynamic per-position contribution is
        # computed via a single small fused matmul without slicing.
        # Build precomputed initial weights for the static-only bias.
        with torch.no_grad():
            # static_features: [F_static, 64, 64] -> apply Linear(F_static -> num_heads)
            tmp_proj = nn.Linear(NUM_STATIC_FEATURES, num_heads, bias=False)
            nn.init.normal_(tmp_proj.weight, mean=0.0, std=0.01)
            sf = STATIC_GEOMETRY_FEATURES.permute(1, 2, 0).contiguous()  # [64, 64, F_static]
            init_static_bias = tmp_proj(sf).permute(2, 0, 1).contiguous()  # [num_heads, 64, 64]
        self.static_bias = nn.Parameter(init_static_bias)

        # Dynamic projection: maps the 4 dynamic channels to a per-head bias.
        # The dynamic bias is added to the static bias at forward time.
        self.dyn_proj = nn.Linear(4, num_heads, bias=False)
        nn.init.normal_(self.dyn_proj.weight, mean=0.0, std=0.01)

        # Register precomputed constants as non-trainable buffers.
        self.register_buffer('piece_pseudo_attack', PIECE_PSEUDO_ATTACK, persistent=False)
        # Pre-permute pa_w/pa_b to the [64, 6, 64] layout we need for matmul, so
        # the permute happens once at init and the buffer is contiguous in the
        # exported graph (TRT-friendly).
        with torch.no_grad():
            pa_w = PIECE_PSEUDO_ATTACK[1:7].permute(1, 0, 2).contiguous()   # [64, 6, 64]
            pa_b = PIECE_PSEUDO_ATTACK[7:13].permute(1, 0, 2).contiguous()  # [64, 6, 64]
        self.register_buffer('pa_w_per_i', pa_w, persistent=False)
        self.register_buffer('pa_b_per_i', pa_b, persistent=False)
        self.num_heads = num_heads

    def forward(self, piece_type_onehot: torch.Tensor) -> torch.Tensor:
        """piece_type_onehot: [B, 64, 13]. Returns [B, num_heads, 64, 64].

        Output = static_bias (learned per-head, position-independent) +
                 dyn_proj([attack_w, attack_b, white_at_i, black_at_i])

        Implementation choices for TRT-friendliness:
          - The static contribution is a precomputed per-head bias parameter,
            no per-forward matmul over static features.
          - Dynamic contribution uses pre-permuted [64, 6, 64] buffers
            (registered at init) and explicit unsqueeze + matmul without
            relying on broadcast-tricky einsum.
        """
        dtype = piece_type_onehot.dtype

        pt_w = piece_type_onehot[:, :, 1:7].contiguous()    # [B, 64, 6]
        pt_b = piece_type_onehot[:, :, 7:13].contiguous()   # [B, 64, 6]

        pa_w = self.pa_w_per_i.to(dtype)                    # [64, 6, 64]
        pa_b = self.pa_b_per_i.to(dtype)                    # [64, 6, 64]

        # attack_white[b, i, j] = sum_t pt_w[b, i, t] * pa_w[i, t, j].
        # Use matmul: [B, 64, 1, 6] @ [1, 64, 6, 64] -> [B, 64, 1, 64] -> squeeze.
        attack_white = torch.matmul(pt_w.unsqueeze(2), pa_w.unsqueeze(0)).squeeze(2)
        attack_black = torch.matmul(pt_b.unsqueeze(2), pa_b.unsqueeze(0)).squeeze(2)

        # Piece presence at i (sum over 6 piece-type channels), broadcast to (i, j).
        # Avoid creating a runtime ones tensor — use repeat to materialise the j-axis.
        white_at_i = pt_w.sum(dim=2).unsqueeze(2).repeat(1, 1, 64)   # [B, 64, 64]
        black_at_i = pt_b.sum(dim=2).unsqueeze(2).repeat(1, 1, 64)   # [B, 64, 64]

        # Stack the 4 dynamic channels and project: [B, 4, 64, 64] -> [B, num_heads, 64, 64].
        # Via per-(i, j) Linear: fold spatial then unfold.
        dyn = torch.stack([attack_white, attack_black, white_at_i, black_at_i], dim=3)  # [B, 64, 64, 4]
        dyn_bias = self.dyn_proj(dyn)                                                    # [B, 64, 64, num_heads]
        dyn_bias = dyn_bias.permute(0, 3, 1, 2).contiguous()                             # [B, num_heads, 64, 64]

        # Static bias is shape [num_heads, 64, 64] — broadcast-add to [B, ...].
        bias = dyn_bias + self.static_bias.unsqueeze(0).to(dtype)
        return bias
