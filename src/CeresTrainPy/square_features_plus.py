"""Extended square features (2026-11-27, puzzle-only ladder): Fry's slots and four new slot families, each behind its
own NetDef key, all rows of ONE zero-initialised [rows, ModelDim] table named `table` (same name / class as
square_features.SquareFeatures, so the optimiser partition and weight-decay rules apply unchanged).

Keys (ceres_net builds this module instead of SquareFeatures when any of the four new keys or the neighbour drop is set):
  SquareFeatures                 Fry's 23 slots (square_features.py)
  SquareFeaturesDropNeighbours   Fry's slots without the 8 neighbour slots (15 slots; needs SquareFeatures)
  SquareFeaturesLinePairs        4 slots, 169 rows each: for each line through the square (directions 0/1, 2/3, 4/7,
                                 5/6), 13 * first code one way + first code the other way (0 = nothing up to the edge)
  SquareFeaturesCheapestAttackers 1 slot, 637 rows: occupant (13) x cheapest of our direct attackers (7: 0 none,
                                 1..6 P N B R Q K) x cheapest of theirs (7)
  SquareFeaturesAttackCounts     1 slot, 16 rows: min(our direct attackers, 3) * 4 + min(theirs, 3)
  SquareFeaturesKingSafety       4 slots: (a) on our king's square 1 + number of its safe flight squares (0..8),
                                 0 elsewhere (10 rows); (b) the same for their king; (c) on the 8 squares around our
                                 king 1 + flag (0 free, 1 attacked by them, 2 occupied by our own piece; own piece wins),
                                 0 elsewhere (4 rows); (d) the same for their king (attacked by us, occupied by theirs).
                                 "Attacked" is computed with that king removed from the board (a slider checking along a
                                 line also covers the square behind the king), so safe = not own-occupied and not
                                 attacked = a legal king move.
  SquareFeaturesRay2             8 slots, 85 rows each (2026-11-27b): the SECOND piece met on the ray in direction d and
                                 its distance from the square, encoded as Fry's ray slot: 0 none, else
                                 1 + (code - 1) * 7 + (distance - 1)

"Direct attackers" = python-chess `board.attackers(colour, square)`: sliders up to the first blocker, knights,
pawns (ours attack upward in the frame of the side to move), kings; pins ignored, no x-ray / battery.

Everything is integer gathers on constant index tables, comparisons, sums and a min (ReduceMin) over 16 candidates;
the first-piece scans are the 7-step unrolled scan of square_features.py. No ArgMax / CumSum / loops over squares.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from square_features import (SquareFeatures, NUM_ROWS as FRY_ROWS, DIRECTIONS, _exporting)

LINE_PAIRS = ((0, 1), (2, 3), (4, 7), (5, 6))
FRY_KEEP_NO_NBR = [0, 1, 2, 3] + list(range(12, 23))       # Fry slot columns without the neighbour slots 4..11
ROWS_LP = 169
ROWS_CA = 13 * 7 * 7
ROWS_AC = 16
ROWS_KS = (10, 10, 4, 4)
ROWS_RAY2 = 85


class SquareFeaturesPlus(SquareFeatures):
  """codes [B, 64] int64 -> [B, 64, width]: the sum of the square's table rows over the enabled slot families."""

  def __init__(self, width: int, fry: bool = True, drop_neighbours: bool = False, line_pairs: bool = False,
               cheapest_attackers: bool = False, attack_counts: bool = False, king_safety: bool = False,
               ray2: bool = False) -> None:
    super().__init__(width)
    assert fry or not drop_neighbours, 'SquareFeaturesDropNeighbours needs SquareFeatures'
    self.fry, self.drop_nbr = bool(fry), bool(drop_neighbours)
    self.line_pairs, self.cheap, self.counts, self.king = bool(line_pairs), bool(cheapest_attackers), bool(attack_counts), bool(king_safety)
    self.ray2 = bool(ray2)
    assert self.fry or self.line_pairs or self.cheap or self.counts or self.king or self.ray2, 'no slot family enabled'
    base = FRY_ROWS if self.fry else 0            # Fry's rows keep their own index space (unused neighbour rows stay 0)
    self.base_lp = base; base += 4 * ROWS_LP if self.line_pairs else 0
    self.base_ca = base; base += ROWS_CA if self.cheap else 0
    self.base_ac = base; base += ROWS_AC if self.counts else 0
    self.base_ks = base; base += sum(ROWS_KS) if self.king else 0
    self.base_r2 = base; base += 8 * ROWS_RAY2 if self.ray2 else 0
    self.num_rows = base
    self.num_slots = ((15 if self.drop_nbr else 23) if self.fry else 0) + 4 * self.line_pairs + self.cheap \
                     + self.counts + 4 * self.king + 8 * self.ray2
    self.table = nn.Parameter(torch.zeros(self.num_rows, width))
    self.register_buffer('is_orth', torch.tensor([d < 4 for d in range(8)]), persistent=False)          # [8]
    self.register_buffer('pawn_ours_dir', torch.tensor([d in (6, 7) for d in range(8)]), persistent=False)
    self.register_buffer('pawn_theirs_dir', torch.tensor([d in (4, 5) for d in range(8)]), persistent=False)
    adj = torch.zeros(64, 64, dtype=torch.bool)
    for a in range(64):
      for b in range(64):
        adj[a, b] = a != b and max(abs(a % 8 - b % 8), abs(a // 8 - b // 8)) == 1
    self.register_buffer('adj', adj, persistent=False)
    self.register_buffer('fry_keep', torch.tensor(FRY_KEEP_NO_NBR), persistent=False)
    self.register_buffer('r2_base', self.base_r2 + ROWS_RAY2 * torch.arange(8), persistent=False)

  # ---------------------------------------------------------------------------------------------------- helpers
  def _scan(self, codes: torch.Tensor):
    """First piece on each of the 8 rays of every square: code [B,64,8] (0 none) and distance [B,64,8] (0 none)."""
    B = codes.shape[0]
    ext = torch.cat([codes, torch.zeros(B, 1, dtype=codes.dtype, device=codes.device)], dim=1)
    ray = ext.index_select(1, self.ray_idx.reshape(-1)).reshape(B, 64, 8, 7)
    free = torch.ones_like(ray[:, :, :, 0], dtype=torch.bool)
    first = torch.zeros_like(ray[:, :, :, 0])
    dist = torch.zeros_like(ray[:, :, :, 0])
    for k in range(7):
      c = ray[:, :, :, k]
      hit = free & (c > 0)
      first = first + hit.to(c.dtype) * c
      dist = dist + hit.to(c.dtype) * (k + 1)
      free = free & ~(c > 0)
    return ext, first, dist

  def _attackers(self, ext, first, dist):
    """[B,64,16] code of each direct attacker candidate (0 = not an attacker): 8 rays then 8 knight squares."""
    B = first.shape[0]
    F, D = first, dist
    orth = self.is_orth.view(1, 1, 8)
    rq = (F == 4) | (F == 5) | (F == 10) | (F == 11)
    bq = (F == 3) | (F == 5) | (F == 9) | (F == 11)
    slider = (orth & rq) | (~orth & bq)
    king = (D == 1) & ((F == 6) | (F == 12))
    pawn = (D == 1) & (((F == 1) & self.pawn_ours_dir.view(1, 1, 8)) | ((F == 7) & self.pawn_theirs_dir.view(1, 1, 8)))
    ray_att = torch.where(slider | king | pawn, F, torch.zeros_like(F))
    kn = ext.index_select(1, self.knight_idx.reshape(-1)).reshape(B, 64, 8)
    kn_att = torch.where((kn == 2) | (kn == 8), kn, torch.zeros_like(kn))
    return torch.cat([ray_att, kn_att], dim=-1)

  @staticmethod
  def _sides(A):
    ours = (A >= 1) & (A <= 6)
    theirs = A >= 7
    return ours, theirs

  def _king_pos(self, codes, code):
    sq = self.sq_rank * 8 + self.sq_file
    return torch.where(codes == code, sq, torch.full_like(codes, 64)).amin(dim=1)       # [B]; 64 = absent

  def _king_rows(self, codes, king_code, own_lo, own_hi, attacker_is_theirs: bool, base_count, base_ring):
    """Flight count row on the king's square and ring flag rows for one king."""
    B = codes.shape[0]
    removed = torch.where(codes == king_code, torch.zeros_like(codes), codes)
    ext, F, D = self._scan(removed)
    ours, theirs = self._sides(self._attackers(ext, F, D))
    attacked = (theirs if attacker_is_theirs else ours).any(dim=-1)                       # [B,64] by the other side
    own = (codes >= own_lo) & (codes <= own_hi)
    kpos = self._king_pos(codes, king_code)                                               # [B]
    present = kpos < 64
    ring = self.adj.index_select(0, kpos.clamp(max=63)) & present.unsqueeze(1)            # [B,64]
    safe = ring & ~own & ~attacked
    count = safe.sum(dim=1)                                                               # [B] 0..8
    is_k = codes == king_code
    row_count = base_count + torch.where(is_k, 1 + count.unsqueeze(1), torch.zeros_like(codes))
    flag = torch.where(own, torch.full_like(codes, 2), attacked.to(codes.dtype))
    row_ring = base_ring + torch.where(ring, 1 + flag, torch.zeros_like(codes))
    return row_count, row_ring

  # ---------------------------------------------------------------------------------------------------- rows
  def rows(self, codes: torch.Tensor) -> torch.Tensor:
    out = []
    if self.fry:
      r = super().rows(codes)                                                             # [B,64,23]
      out.append(r.index_select(2, self.fry_keep) if self.drop_nbr else r)
    if self.line_pairs or self.cheap or self.counts:
      ext, F, D = self._scan(codes)
      if self.line_pairs:
        for L, (a, b) in enumerate(LINE_PAIRS):
          out.append((self.base_lp + L * ROWS_LP + 13 * F[:, :, a] + F[:, :, b]).unsqueeze(-1))
      if self.cheap or self.counts:
        A = self._attackers(ext, F, D)
        ours, theirs = self._sides(A)
        if self.cheap:
          seven = torch.full_like(A, 7)
          co = torch.where(ours, A, seven).amin(dim=-1)
          ct = torch.where(theirs, A - 6, seven).amin(dim=-1)
          co = torch.where(co == 7, torch.zeros_like(co), co)
          ct = torch.where(ct == 7, torch.zeros_like(ct), ct)
          out.append((self.base_ca + codes * 49 + co * 7 + ct).unsqueeze(-1))
        if self.counts:
          no = ours.sum(dim=-1).clamp(max=3)
          nt = theirs.sum(dim=-1).clamp(max=3)
          out.append((self.base_ac + no * 4 + nt).unsqueeze(-1))
    if self.king:
      b = self.base_ks
      oc, orr = self._king_rows(codes, 6, 1, 6, True, b, b + 20)
      tc, trr = self._king_rows(codes, 12, 7, 12, False, b + 10, b + 24)
      out += [oc.unsqueeze(-1), tc.unsqueeze(-1), orr.unsqueeze(-1), trr.unsqueeze(-1)]
    if self.ray2:
      out.append(self.r2_base + self._second(codes))
    return torch.cat(out, dim=-1)                                                         # [B,64,num_slots]

  def _second(self, codes):
    """Second piece on each ray of every square, Fry's ray encoding [B,64,8]: 0 none, else 1 + (code-1)*7 + (dist-1)."""
    B = codes.shape[0]
    ext = torch.cat([codes, torch.zeros(B, 1, dtype=codes.dtype, device=codes.device)], dim=1)
    ray = ext.index_select(1, self.ray_idx.reshape(-1)).reshape(B, 64, 8, 7)
    seen = torch.zeros_like(ray[:, :, :, 0])
    second = torch.zeros_like(ray[:, :, :, 0])
    for k in range(7):
      c = ray[:, :, :, k]
      occ = (c > 0).to(c.dtype)
      second = second + occ * (seen == 1).to(c.dtype) * (1 + (c - 1) * 7 + k)
      seen = seen + occ
    return second

  def forward(self, codes: torch.Tensor) -> torch.Tensor:
    rows = self.rows(codes)
    B, S = rows.shape[0], rows.shape[2]
    if _exporting():
      return self.table.index_select(0, rows.reshape(-1)).reshape(B, 64, S, self.width).sum(dim=2)
    return torch.nn.functional.embedding_bag(rows.reshape(B * 64, S), self.table, mode='sum').reshape(B, 64, self.width)
