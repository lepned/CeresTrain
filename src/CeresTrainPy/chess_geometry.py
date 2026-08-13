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


def _build_ray_tables():
    """Constants for blocker-aware ray attention:
      ROOK_LINE  [64, 64]: i, j distinct and on the same rank or file
      BISH_LINE  [64, 64]: i, j distinct and on the same diagonal/anti-diagonal
      BETWEEN_T  [64, 4096]: BETWEEN_T[k, i*64+j] = 1 iff square k lies STRICTLY
                 between i and j on a rook or bishop line (0 for non-colinear pairs).
    """
    rook = torch.zeros(64, 64)
    bish = torch.zeros(64, 64)
    between_t = torch.zeros(64, 64 * 64)
    for i in range(64):
        fi, ri = i % 8, i // 8
        for j in range(64):
            if i == j:
                continue
            fj, rj = j % 8, j // 8
            df, dr = fj - fi, rj - ri
            on_rook = (df == 0) or (dr == 0)
            on_bish = abs(df) == abs(dr)
            if not (on_rook or on_bish):
                continue
            if on_rook:
                rook[i, j] = 1.0
            else:
                bish[i, j] = 1.0
            sf = (df > 0) - (df < 0)
            sr = (dr > 0) - (dr < 0)
            f, r = fi + sf, ri + sr
            while (f, r) != (fj, rj):
                between_t[r * 8 + f, i * 64 + j] = 1.0
                f, r = f + sf, r + sr
    return rook, bish, between_t


class RayAttentionBias(nn.Module):
    """Blocker-aware sliding-piece attention bias, computed IN-GRAPH from the
    13-channel piece one-hot (no new inputs -> no inference-side changes).

    Six type-resolved dynamic channels per square pair (i, j):
      w/b rook-line REAL attack   (R or Q on i, colinear, line clear)
      w/b bishop-line REAL attack (B or Q on i, colinear, line clear)
      w/b X-RAY                   (matching slider on i, colinear, EXACTLY ONE
                                   blocker between -> the pin/skewer/discovery channel)

    Blocker resolution is exact via integer counting: occ @ BETWEEN_T gives the
    number of occupied squares strictly between each pair, so relu(1 - n) is a
    0/1 'line clear' indicator and relu(1 - |n - 1|) is a 0/1 'exactly one
    blocker' indicator. One [B,64]x[64,4096] matmul + elementwise ops; the bias
    adds to attention logits like PieceRelationBias (fused-MHA-friendly).
    """

    def __init__(self, num_heads: int):
        super().__init__()
        rook, bish, between_t = _build_ray_tables()
        self.register_buffer('ray_rook_line', rook, persistent=False)
        self.register_buffer('ray_bish_line', bish, persistent=False)
        self.register_buffer('ray_between_t', between_t, persistent=False)
        self.ray_proj = nn.Linear(6, num_heads, bias=False)
        nn.init.normal_(self.ray_proj.weight, mean=0.0, std=0.01)
        self.num_heads = num_heads

    def forward(self, piece_type_onehot: torch.Tensor) -> torch.Tensor:
        """piece_type_onehot: [B, 64, 13] -> bias [B, num_heads, 64, 64]."""
        dtype = piece_type_onehot.dtype
        pt = piece_type_onehot
        occ = 1.0 - pt[:, :, 0]                                    # [B, 64]
        blocked = torch.matmul(occ, self.ray_between_t.to(dtype))  # [B, 4096] integer counts
        clear = torch.relu(1.0 - blocked).reshape(-1, 64, 64)      # exactly 0 blockers
        one_bl = torch.relu(1.0 - torch.abs(blocked - 1.0)).reshape(-1, 64, 64)  # exactly 1

        rook_line = self.ray_rook_line.to(dtype)                   # [64, 64]
        bish_line = self.ray_bish_line.to(dtype)
        # sliders on i (one-hot channels: 1..6 = WP WN WB WR WQ WK, 7..12 = black)
        w_rook = (pt[:, :, 4] + pt[:, :, 5]).unsqueeze(2)          # [B, 64, 1] (R + Q)
        w_bish = (pt[:, :, 3] + pt[:, :, 5]).unsqueeze(2)          # (B + Q)
        b_rook = (pt[:, :, 10] + pt[:, :, 11]).unsqueeze(2)
        b_bish = (pt[:, :, 9] + pt[:, :, 11]).unsqueeze(2)

        ch = torch.stack([
            w_rook * rook_line * clear,
            w_bish * bish_line * clear,
            b_rook * rook_line * clear,
            b_bish * bish_line * clear,
            (w_rook * rook_line + w_bish * bish_line) * one_bl,
            (b_rook * rook_line + b_bish * bish_line) * one_bl,
        ], dim=3)                                                  # [B, 64, 64, 6]
        bias = self.ray_proj(ch).permute(0, 3, 1, 2).contiguous()  # [B, H, 64, 64]
        return bias


def _build_pawn_visibility_tables():
    """Pawn VISIBILITY tables [64, 64] per side: the two capture diagonals PLUS
    the single push square directly ahead. Visibility deliberately includes the
    non-attacking push square (Kovax program: escape-square coverage needs
    edges onto empty squares; the pawn's reach includes the square it can move
    to). Double-push is excluded — it is blocker-conditional and rare enough
    that the (unconditional-indicator) channel would be wrong more often than
    informative. Channels 1..6 advance toward +rank, 7..12 toward -rank,
    matching PIECE_PSEUDO_ATTACK.

    ⚠ Any FUTURE channel needing true ATTACK semantics (e.g. king-ring
    coverage) must use PIECE_PSEUDO_ATTACK's pawn diagonals, NOT these tables:
    a pawn's visibility includes the push square it does not attack.
    """
    w = PIECE_PSEUDO_ATTACK[1].clone()
    b = PIECE_PSEUDO_ATTACK[7].clone()
    for i in range(64):
        f, r = i % 8, i // 8
        if r + 1 < 8:
            w[i, (r + 1) * 8 + f] = 1.0
        if r - 1 >= 0:
            b[i, (r - 1) * 8 + f] = 1.0
    return w, b


class VisibilityChannels(nn.Module):
    """Channel builder v2 for the visibility / attack-graph edge-bias program.

    Builds pairwise {0,1} edge indicator channels E[b, q, k, c] ONCE per
    forward from the 13-channel piece one-hot; the parent net projects them to
    per-head attention-logit biases (content-free form A) or content-gated
    variants. Spec follows the Kovax visibility program (VISIBILITY_PROGRAM.md):

    Families (each emitted as stm/opp x outgoing/incoming = 4 channels):
      vis    — VISIBILITY, not attacks: a piece's reach with NO target-side
               mask (edge onto empty / friendly / enemy square alike). All
               piece types: sliders blocker-exact (stop at first blocker,
               inclusive), knight/king leaper reach, pawn diagonals + push
               square. No-target-mask is deliberate: king-escape coverage
               needs edges onto empty squares.
      xray   — slider ray continued through EXACTLY ONE blocker (the
               pin/skewer/discovery geometry). Target may be any square on
               the ray with strictly-between occupancy == 1.
      pinray — edge FROM THE BLOCKER to the x-ray target: the pinned piece
               reads what it is pinned against in one hop. Strongest channel
               per edge in the source program despite 0.08% density.

    Families found NOISE / refuted there and deliberately absent: `defends`
    (derivable in ~1 hop), raw mobility row-sums. Selection rule: only add a
    channel the net cannot derive in ~1 layer.

    Channel order: for each family in `families`: [stm_out, opp_out, stm_in,
    opp_in], where *_in = transpose of *_out (incoming edges carry more weight
    than outgoing in the ungated form, per the source program's K-side
    asymmetry finding). "stm" = one-hot channels 1..6, "opp" = 7..12.

    All ops are matmul / elementwise over constant buffers (TRT-friendly, no
    einsum). Integer blocker counting via BETWEEN_T as in RayAttentionBias.
    """

    FAMILY_ORDER = ('vis', 'xray', 'pinray')

    def __init__(self, families=('vis', 'xray', 'pinray')):
        super().__init__()
        for f in families:
            assert f in self.FAMILY_ORDER, f'unknown visibility family: {f}'
        # Keep canonical order regardless of the order given.
        self.families = tuple(f for f in self.FAMILY_ORDER if f in families)
        assert self.families, 'VisibilityChannels needs at least one family'
        self.num_channels = 4 * len(self.families)
        # Per-family channel-index slices (for structure logging: per-family
        # weight RMS is the program's structure-first readout).
        self.family_slices = {f: slice(4 * i, 4 * i + 4)
                              for i, f in enumerate(self.families)}

        rook, bish, between_t = _build_ray_tables()
        self.register_buffer('vc_rook_line', rook, persistent=False)
        self.register_buffer('vc_bish_line', bish, persistent=False)
        self.register_buffer('vc_between_t', between_t, persistent=False)
        if 'pinray' in self.families:
            # [t, s, blocker] layout: constant right-operand of a broadcast
            # batched matmul with batch dims (B, t) — keeps the DYNAMIC batch
            # dim leading throughout (export/TRT-friendly; batch-last matmuls
            # force materialized transposes of the dynamic axis).
            bt3 = between_t.reshape(64, 64, 64).permute(2, 1, 0).contiguous()
            self.register_buffer('vc_between_stb', bt3, persistent=False)
        knight = PIECE_PSEUDO_ATTACK[2].clone()
        king = PIECE_PSEUDO_ATTACK[6].clone()
        pawn_w, pawn_b = _build_pawn_visibility_tables()
        self.register_buffer('vc_knight', knight, persistent=False)
        self.register_buffer('vc_king', king, persistent=False)
        self.register_buffer('vc_pawn_stm', pawn_w, persistent=False)
        self.register_buffer('vc_pawn_opp', pawn_b, persistent=False)

    def _side_channels(self, pt, base, clear, one_bl, pawn_vis, dtype):
        """Channels for one side. pt: [B, 64, 13]; base: 1 for stm (one-hot
        channels 1..6) or 7 for opp. Returns dict family -> [B, 64, 64]."""
        rook_line = self.vc_rook_line.to(dtype)
        bish_line = self.vc_bish_line.to(dtype)
        # Piece presence at source square i: [B, 64, 1].
        p_pawn = pt[:, :, base + 0].unsqueeze(2)
        p_knight = pt[:, :, base + 1].unsqueeze(2)
        p_bish = pt[:, :, base + 2].unsqueeze(2)
        p_rook = pt[:, :, base + 3].unsqueeze(2)
        p_queen = pt[:, :, base + 4].unsqueeze(2)
        p_king = pt[:, :, base + 5].unsqueeze(2)

        rq = p_rook + p_queen
        bq = p_bish + p_queen

        out = {}
        if 'vis' in self.families:
            # Sliders: line clear (target itself unmasked — first blocker IS a
            # visible square). Leapers + pawn: constant reach tables. Terms are
            # disjoint per source square (one piece per square; rook/bishop
            # lines are disjoint square-pair sets), so the sum stays {0,1}.
            out['vis'] = (rq * rook_line * clear
                          + bq * bish_line * clear
                          + p_knight * self.vc_knight.to(dtype)
                          + p_king * self.vc_king.to(dtype)
                          + p_pawn * pawn_vis.to(dtype))
        xray = None
        if 'xray' in self.families or 'pinray' in self.families:
            xray = (rq * rook_line + bq * bish_line) * one_bl
        if 'xray' in self.families:
            out['xray'] = xray
        if 'pinray' in self.families:
            # pinray[b, t] = occ[b] * OR_s( between(s, t)[b] AND xray[s, t] ):
            # b is THE single blocker of some x-ray onto t. Broadcast batched
            # matmul, batch dims (B, t), dynamic batch dim kept leading:
            # [B, t, 1, s] @ [t, s, blocker] -> [B, t, 1, blocker].
            occ = 1.0 - pt[:, :, 0]                                    # [B, 64]
            xr_t = xray.permute(0, 2, 1).unsqueeze(2)                  # [B, t, 1, s]
            pin = torch.matmul(xr_t, self.vc_between_stb.to(dtype))    # [B, t, 1, blocker]
            pin = pin.squeeze(2).permute(0, 2, 1)                      # [B, blocker, t]
            # clamp: two sliders x-raying the same target through the same
            # blocker square would sum to 2; keep {0,1} indicator semantics.
            out['pinray'] = occ.unsqueeze(2) * pin.clamp(max=1.0)
        return out

    def forward(self, piece_type_onehot: torch.Tensor) -> torch.Tensor:
        """piece_type_onehot: [B, 64, 13] -> E [B, 64, 64, num_channels]."""
        pt = piece_type_onehot
        dtype = pt.dtype
        occ = 1.0 - pt[:, :, 0]                                       # [B, 64]
        blocked = torch.matmul(occ, self.vc_between_t.to(dtype))      # [B, 4096]
        clear = torch.relu(1.0 - blocked).reshape(-1, 64, 64)
        one_bl = torch.relu(1.0 - torch.abs(blocked - 1.0)).reshape(-1, 64, 64)

        stm = self._side_channels(pt, 1, clear, one_bl, self.vc_pawn_stm, dtype)
        opp = self._side_channels(pt, 7, clear, one_bl, self.vc_pawn_opp, dtype)

        chans = []
        for fam in self.families:
            chans.append(stm[fam])
            chans.append(opp[fam])
            chans.append(stm[fam].transpose(1, 2))
            chans.append(opp[fam].transpose(1, 2))
        return torch.stack(chans, dim=3)                # [B, 64, 64, C]


def build_ray_context_tables(from_idx, to_idx):
  """Constant per-move square-pooling operators for the ray-context policy term.

  R  [1858, 64]: for collinear (queen-line) moves, the squares strictly BETWEEN
     from and to plus the collinear extension BEHIND the from-square (pin/skewer/
     x-ray geometry along the move's own line). All-zero for knight moves.
  R2 [1858, 64]: the rays through the from-square NOT collinear with the move
     direction — the rays actually VACATED by the move (discovery geometry; a
     move never vacates its own line). For knights: all 8 rays.

  Rows are mean-normalized; empty rows stay zero (clamp avoids 0/0 for the
  knight/adjacent-move cases flagged in the campaign review).
  Squares use lc0 indexing: a1=0 .. h8=63 (rank*8 + file).
  """
  DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
  n = len(from_idx)
  R = torch.zeros(n, 64)
  R2 = torch.zeros(n, 64)
  for m in range(n):
    fr, ff = divmod(from_idx[m], 8)
    tr, tf = divmod(to_idx[m], 8)
    dr, df = tr - fr, tf - ff
    coll = None
    if dr == 0 and df != 0:
      coll = (0, 1 if df > 0 else -1)
    elif df == 0 and dr != 0:
      coll = (1 if dr > 0 else -1, 0)
    elif dr != 0 and abs(dr) == abs(df):
      coll = (1 if dr > 0 else -1, 1 if df > 0 else -1)
    if coll is not None:
      sr, sf = coll
      r_, f_ = fr + sr, ff + sf
      while (r_, f_) != (tr, tf):
        R[m, r_ * 8 + f_] = 1.0
        r_ += sr; f_ += sf
      r_, f_ = fr - sr, ff - sf
      while 0 <= r_ < 8 and 0 <= f_ < 8:
        R[m, r_ * 8 + f_] = 1.0
        r_ -= sr; f_ -= sf
    for sr, sf in DIRS:
      if coll is not None and (sr, sf) in (coll, (-coll[0], -coll[1])):
        continue
      r_, f_ = fr + sr, ff + sf
      while 0 <= r_ < 8 and 0 <= f_ < 8:
        R2[m, r_ * 8 + f_] = 1.0
        r_ += sr; f_ += sf
  R = R / R.sum(dim=1, keepdim=True).clamp(min=1.0)
  R2 = R2 / R2.sum(dim=1, keepdim=True).clamp(min=1.0)
  return R, R2
