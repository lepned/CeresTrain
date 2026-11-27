"""Gives-check index of a move token (NetDef MoveTokenGivesCheck, 2026-11-27): 0 no check, 1 direct check only,
2 discovered check only, 3 both (double check). Computed in-graph from the 64 piece codes of the side-to-move frame
(0 empty, 1..6 ours, 7..12 theirs) and the token's from / to squares; the decoder adds a zero-initialised row of a
[4, dm] table (row 0 masked out) to the token's input embedding.

Rules (pseudo-legal moves; the move-token candidates are a superset of the legal moves):
  direct     the moved piece on `to` attacks their king with the board after the move (from vacated): knight jump,
             pawn diagonal (upward), or a bishop / rook / queen line with nothing in between. A pawn reaching the last
             rank is scored as a QUEEN (the token carries all four promotion slots; queen is the common case).
  discovered `from` is the first piece on a ray from their king, the second piece on that ray is our bishop / rook /
             queen of the matching line type, and `to` is not between their king and that slider.
Not represented (scored 0 / left as computed): castling tokens (king -> own rook; scored 0), the discovered check of
an en-passant capture through the removed pawn, under-promotions (scored as a queen).

Ops: constant [64,64] tables gathered by (king, square), a 7-step scan along the 8 rays from their king, comparisons.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from square_features import DIRECTIONS, _step_table


def _tables():
  dirt = torch.full((64, 64), 8, dtype=torch.long)       # direction index from a to b (DIRECTIONS), 8 = not on a line
  dist = torch.zeros(64, 64, dtype=torch.long)           # steps along that line
  for a in range(64):
    fa, ra = a % 8, a // 8
    for d, (df, dr) in enumerate(DIRECTIONS):
      for k in range(1, 8):
        f, r = fa + df * k, ra + dr * k
        if not (0 <= f < 8 and 0 <= r < 8):
          break
        dirt[a, r * 8 + f] = d
        dist[a, r * 8 + f] = k
  kn = torch.zeros(64, 64, dtype=torch.bool)
  pawn = torch.zeros(64, 64, dtype=torch.bool)           # pawn of ours on a attacks b
  for a in range(64):
    fa, ra = a % 8, a // 8
    for df, dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
      f, r = fa + df, ra + dr
      if 0 <= f < 8 and 0 <= r < 8:
        kn[a, r * 8 + f] = True
    for df in (-1, 1):
      f, r = fa + df, ra + 1
      if 0 <= f < 8 and 0 <= r < 8:
        pawn[a, r * 8 + f] = True
  return dirt, dist, kn, pawn


class GivesCheck(nn.Module):
  """(codes [B,64] int64, fr [B,M], to [B,M]) -> [B,M] int64 in 0..3. No parameters."""

  def __init__(self):
    super().__init__()
    dirt, dist, kn, pawn = _tables()
    self.register_buffer('dirt', dirt.reshape(-1), persistent=False)
    self.register_buffer('dist', dist.reshape(-1), persistent=False)
    self.register_buffer('kn', kn.reshape(-1), persistent=False)
    self.register_buffer('pawn', pawn.reshape(-1), persistent=False)
    self.register_buffer('ray_idx', _step_table(DIRECTIONS, 7), persistent=False)         # [64,8,7], 64 = off board
    self.register_buffer('is_orth', torch.tensor([d < 4 for d in range(8)] + [False]), persistent=False)   # [9]

  def forward(self, codes, fr, to):
    B = codes.shape[0]
    dev = codes.device
    sq = torch.arange(64, device=dev)
    K = torch.where(codes == 12, sq, torch.full_like(codes, 64)).amin(dim=1)              # [B] their king; 64 absent
    present = K < 64
    Kc = K.clamp(max=63)
    ext = torch.cat([codes, torch.zeros(B, 1, dtype=codes.dtype, device=dev)], dim=1)
    rsq = self.ray_idx.index_select(0, Kc).reshape(B, 56)                                 # squares along the 8 rays
    rc = torch.gather(ext, 1, rsq).reshape(B, 8, 7)                                       # codes (0 off board)
    # first and second piece on each ray from their king: code, distance
    seen = torch.zeros(B, 8, dtype=codes.dtype, device=dev)
    c1 = torch.zeros_like(seen); d1 = torch.zeros_like(seen)
    c2 = torch.zeros_like(seen); d2 = torch.zeros_like(seen)
    for k in range(7):
      c = rc[:, :, k]
      occ = (c > 0).to(codes.dtype)
      h1 = occ * (seen == 0).to(codes.dtype)
      h2 = occ * (seen == 1).to(codes.dtype)
      c1 = c1 + h1 * c; d1 = d1 + h1 * (k + 1)
      c2 = c2 + h2 * c; d2 = d2 + h2 * (k + 1)
      seen = seen + occ
    pad = torch.zeros(B, 1, dtype=codes.dtype, device=dev)
    c1, d1, c2, d2 = (torch.cat([t, pad], dim=1) for t in (c1, d1, c2, d2))               # [B,9]; column 8 = no line
    d1 = torch.where(d1 == 0, torch.full_like(d1, 8), d1)                                 # nothing on the ray
    d2 = torch.where(d2 == 0, torch.full_like(d2, 8), d2)

    Kb = Kc.unsqueeze(1)
    dir_to = self.dirt.index_select(0, (Kb * 64 + to).reshape(-1)).reshape(fr.shape)     # [B,M] 0..8
    dist_to = self.dist.index_select(0, (Kb * 64 + to).reshape(-1)).reshape(fr.shape)
    dir_fr = self.dirt.index_select(0, (Kb * 64 + fr).reshape(-1)).reshape(fr.shape)
    dist_fr = self.dist.index_select(0, (Kb * 64 + fr).reshape(-1)).reshape(fr.shape)
    mover = torch.gather(codes, 1, fr)
    target = torch.gather(codes, 1, to)
    mover = torch.where((mover == 1) & (to >= 56), torch.full_like(mover, 5), mover)     # promotion scored as queen

    # direct check
    knight = (mover == 2) & self.kn.index_select(0, (to * 64 + Kb).reshape(-1)).reshape(fr.shape)
    pawn = (mover == 1) & self.pawn.index_select(0, (to * 64 + Kb).reshape(-1)).reshape(fr.shape)
    orth_to = self.is_orth.index_select(0, dir_to.reshape(-1)).reshape(fr.shape)
    line_ok = (dir_to < 8) & (((mover == 4) & orth_to) | ((mover == 3) & ~orth_to) | (mover == 5))
    g = lambda t, idx: torch.gather(t, 1, idx)
    fr_first = (dir_fr == dir_to) & (dist_fr == g(d1, dir_to)) & (dir_fr < 8)            # from is the first piece on that ray
    eff = torch.where(fr_first, g(d2, dir_to), g(d1, dir_to))
    slider = line_ok & (dist_to <= eff)
    direct = knight | pawn | slider

    # discovered check
    on_first = (dir_fr < 8) & (dist_fr == g(d1, dir_fr))
    s_code = g(c2, dir_fr)
    orth_fr = self.is_orth.index_select(0, dir_fr.reshape(-1)).reshape(fr.shape)
    s_ok = ((s_code == 5) | ((s_code == 4) & orth_fr) | ((s_code == 3) & ~orth_fr)) & (dir_fr < 8)
    still_blocks = (dir_to == dir_fr) & (dist_to < g(d2, dir_fr))
    disc = on_first & s_ok & ~still_blocks

    castle = (target >= 1) & (target <= 6)                                                 # king -> own rook token
    ok = present.unsqueeze(1) & ~castle
    return (direct & ok).to(torch.long) + 2 * (disc & ok).to(torch.long)
