"""Position-level outcome facts (2026-11-27b; NetDef PositionOutcomeFeatures / PositionOutcomeToValueHead).

Six facts of the whole position from the side to move's view, each one row of ONE zero-initialised table, rows summed:
    slot  rows  fact
    0     4     min(3, our non-king pieces EN PRISE): attacked by them and (undefended or their cheapest attacker worth
                less than the piece)                                                  values P 1, N 3, B 3, R 5, Q 9
    1     4     min(3, their non-king pieces we can WIN by the same rule): attacked by us and (undefended or our cheapest
                attacker worth less)
    2     2     we are in check
    3     2     we have a LEGAL checking move (castling and en passant left out; a promotion scored as a queen).
                Legality in-graph: king moves only to squares not attacked with the king removed; other pieces: not
                off their pin line, and when in check only onto the checker's square or the squares between it and
                our king (single check), none in double check.
    4     9     safe flight squares of our king (0..8; that king removed for the attack test = legal king moves)
    5     9     the same for their king
"Attacked" = direct attackers as python-chess attackers() (square_features_plus.py), the same rule as the analysis
labels (sqf_analysis/run_D.py, run_D2.py, klabels.py); has-check is the analysis label except castling / e.p. moves.

NOT included: "a check that can be neither captured nor blocked". Exact computation needs, for every candidate checking
move, the board after that move and the opponent's LEGAL replies on it (captures of the checker, interpositions, with
pins and king safety re-evaluated in the new position): a second ply per move, i.e. [B, 4096] boards each with its own
attack scan. That is not a gather-and-compare feature of the current board, so it is left out.

Two consumers, each its own switch and its own zero-init table:
  PositionOutcomeFeatures     `table` [30, ModelDim]: the summed rows added to EVERY square's embedding
  PositionOutcomeToValueHead  `vh_table` [30, value-head hidden]: the summed rows added to the value heads' first-layer
                              pre-activation (the `_v_inject` path that the WDL head and the HL-Gauss head both read)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from square_features import DIRECTIONS, KNIGHT_JUMPS, _step_table, _exporting
from square_features_plus import SquareFeaturesPlus
from gives_check import GivesCheck

NUM_ROWS = 30
NUM_SLOTS = 6
BASES = (0, 4, 8, 10, 12, 21)
PIECE_VALUE = (0, 1, 3, 3, 5, 9, 99)


class PositionOutcome(nn.Module):
  """codes [B, 64] -> rows [B, 6]; square_add(rows) [B, 1, D]; value_add(rows) [B, Hv]."""

  # the attack-scan helpers of SquareFeaturesPlus (they read only the buffers registered below)
  _scan = SquareFeaturesPlus._scan
  _attackers = SquareFeaturesPlus._attackers
  _sides = staticmethod(SquareFeaturesPlus._sides)
  _king_pos = SquareFeaturesPlus._king_pos

  def __init__(self, width: int, value_width: int, to_squares: bool, to_value_head: bool) -> None:
    super().__init__()
    assert to_squares or to_value_head
    self.table = nn.Parameter(torch.zeros(NUM_ROWS, width)) if to_squares else None
    self.vh_table = nn.Parameter(torch.zeros(NUM_ROWS, value_width)) if to_value_head else None
    sq = torch.arange(64)
    self.register_buffer('sq_rank', sq // 8, persistent=False)
    self.register_buffer('sq_file', sq % 8, persistent=False)
    ray_idx = _step_table(DIRECTIONS, 7)
    self.register_buffer('ray_idx', ray_idx, persistent=False)
    self.register_buffer('knight_idx', _step_table(KNIGHT_JUMPS, 1)[:, :, 0], persistent=False)
    self.register_buffer('is_orth', torch.tensor([d < 4 for d in range(8)]), persistent=False)
    self.register_buffer('pawn_ours_dir', torch.tensor([d in (6, 7) for d in range(8)]), persistent=False)
    self.register_buffer('pawn_theirs_dir', torch.tensor([d in (4, 5) for d in range(8)]), persistent=False)
    adj = torch.zeros(64, 64, dtype=torch.bool)
    for a in range(64):
      for b in range(64):
        adj[a, b] = a != b and max(abs(a % 8 - b % 8), abs(a // 8 - b // 8)) == 1
    self.register_buffer('adj', adj, persistent=False)
    self.register_buffer('value', torch.tensor(PIECE_VALUE), persistent=False)
    self.register_buffer('bases', torch.tensor(BASES), persistent=False)
    # pseudo-legal move tables: ray one-hot [64, 56, 64] (target square of step k on ray d), knight / pawn tables
    oh = torch.zeros(64, 56, 64)
    for f in range(64):
      for j, t in enumerate(ray_idx[f].reshape(-1).tolist()):
        if t < 64:
          oh[f, j, t] = 1.0
    self.register_buffer('ray_oh', oh, persistent=False)
    kn = torch.zeros(64, 64); push = torch.zeros(64, 64); dbl = torch.zeros(64, 64); pcap = torch.zeros(64, 64)
    for f in range(64):
      fa, ra = f % 8, f // 8
      for df, dr in KNIGHT_JUMPS:
        if 0 <= fa + df < 8 and 0 <= ra + dr < 8:
          kn[f, (ra + dr) * 8 + fa + df] = 1.0
      if ra < 7:
        push[f, f + 8] = 1.0
        for df in (-1, 1):
          if 0 <= fa + df < 8:
            pcap[f, f + 8 + df] = 1.0
      if ra == 1:
        dbl[f, f + 16] = 1.0
    for n, t in (('kn', kn), ('push', push), ('dbl', dbl), ('pcap', pcap)):
      self.register_buffer(n, t, persistent=False)
    mid = torch.arange(64); mid = torch.where(mid < 56, mid + 8, mid)
    self.register_buffer('push_mid', mid, persistent=False)
    self.register_buffer('kstep', torch.tensor([k == 0 for d in range(8) for k in range(7)]), persistent=False)   # [56]
    ar = torch.arange(4096)
    self.register_buffer('pair_from', ar // 64, persistent=False)
    self.register_buffer('pair_to', ar % 64, persistent=False)
    self.gc = GivesCheck()

  # ------------------------------------------------------------------------------------------------ facts
  def _flights(self, codes, king_code, own_lo, own_hi, attacker_is_theirs):
    """(safe flight count [B], attacked-by-the-other-side map with this king removed [B,64])."""
    removed = torch.where(codes == king_code, torch.zeros_like(codes), codes)
    ext, F, D = self._scan(removed)
    ours, theirs = self._sides(self._attackers(ext, F, D))
    attacked = (theirs if attacker_is_theirs else ours).any(dim=-1)
    own = (codes >= own_lo) & (codes <= own_hi)
    kpos = self._king_pos(codes, king_code)
    ring = self.adj.index_select(0, kpos.clamp(max=63)) & (kpos < 64).unsqueeze(1)
    return (ring & ~own & ~attacked).sum(dim=1), attacked

  def legal(self, codes, P, A, F, D, attacked_noK):
    """[B, 4096] bool: the pseudo-legal moves P restricted to legal ones (pins, check evasion, king safety)."""
    B = codes.shape[0]
    K = self._king_pos(codes, 6).clamp(max=63)                                             # [B]
    ar = torch.arange(64, device=codes.device)
    raysq = self.ray_idx.index_select(0, K).reshape(B, 8, 7)                               # squares on the rays from K
    FK = torch.gather(F, 1, K.view(B, 1, 1).expand(-1, 1, 8)).squeeze(1)                    # [B,8] first piece code
    DK = torch.gather(D, 1, K.view(B, 1, 1).expand(-1, 1, 8)).squeeze(1)                    # [B,8] its distance
    AK = torch.gather(A, 1, K.view(B, 1, 1).expand(-1, 1, 16)).squeeze(1)                   # [B,16] attackers of K
    chk_ray = AK[:, :8] >= 7                                                               # [B,8] their checker on ray d
    chk_kn = AK[:, 8:] >= 7                                                                # [B,8]
    n_chk = chk_ray.sum(1) + chk_kn.sum(1)                                                 # [B]
    kstep = torch.arange(1, 8, device=codes.device).view(1, 1, 7)
    seg = chk_ray.unsqueeze(-1) & (kstep <= DK.unsqueeze(-1))                               # checker square and between
    resolve = ((raysq.unsqueeze(-1) == ar.view(1, 1, 1, 64)) & seg.unsqueeze(-1)).flatten(1, 2).any(1)       # [B,64]
    knsq = self.knight_idx.index_select(0, K)                                              # [B,8]
    resolve = resolve | ((knsq.unsqueeze(-1) == ar.view(1, 1, 64)) & chk_kn.unsqueeze(-1)).any(1)
    # pins: first piece from K on ray d is ours (not the king), the second is their slider of the line's type
    ext = torch.cat([codes, torch.zeros(B, 1, dtype=codes.dtype, device=codes.device)], dim=1)
    rc = torch.gather(ext, 1, raysq.reshape(B, 56)).reshape(B, 8, 7)
    seen = torch.zeros(B, 8, dtype=codes.dtype, device=codes.device)
    c2 = torch.zeros_like(seen)
    for k in range(7):
      c = rc[:, :, k]
      occ = (c > 0).to(codes.dtype)
      c2 = c2 + occ * (seen == 1).to(codes.dtype) * c
      seen = seen + occ
    orth = self.is_orth.view(1, 8)
    pin = ((FK >= 1) & (FK <= 5)) & ((orth & ((c2 == 10) | (c2 == 11))) | (~orth & ((c2 == 9) | (c2 == 11))))   # [B,8]
    pin_sq = torch.gather(raysq.reshape(B, 56), 1, (torch.arange(8, device=codes.device) * 7).view(1, 8) + (DK - 1).clamp(min=0))
    pin_dir = ((pin_sq.unsqueeze(-1) == ar.view(1, 1, 64)) & pin.unsqueeze(-1)).to(codes.dtype)
    pin_dir = (pin_dir * (torch.arange(8, device=codes.device).view(1, 8, 1) + 1)).sum(1)  # [B,64] 0 = not pinned
    fr = self.pair_from.unsqueeze(0).expand(B, -1)
    to = self.pair_to.unsqueeze(0).expand(B, -1)
    dir_kt = self.gc.dirt.index_select(0, (K.unsqueeze(1) * 64 + to).reshape(-1)).reshape(B, 4096) + 1   # 1..8, 9 none
    pd = torch.gather(pin_dir, 1, fr)
    pin_ok = (pd == 0) | (pd == dir_kt)
    is_king = torch.gather(codes, 1, fr) == 6
    chk_ok = (n_chk == 0).unsqueeze(1) | ((n_chk == 1).unsqueeze(1) & torch.gather(resolve, 1, to))
    king_ok = ~torch.gather(attacked_noK, 1, to)
    # plain logic instead of a boolean Where (onnxruntime's CPU provider has no Where for BOOL values)
    return P & ((is_king & king_ok) | (~is_king & pin_ok & chk_ok))

  def pseudo_legal(self, codes, ext, F, D):
    """[B, 4096] bool: from*64 + to of every pseudo-legal move of the side to move (castling / e.p. not included)."""
    B = codes.shape[0]
    dt = torch.float32
    reach = ((torch.arange(1, 8, device=codes.device).view(1, 1, 1, 7) <= D.unsqueeze(-1)) | (D == 0).unsqueeze(-1))   # [B,64,8,7]
    T = codes.unsqueeze(-1)                                                                # [B,64,1]
    orth = self.is_orth.view(1, 1, 8)
    slide = (((T == 4) | (T == 5)) & orth) | (((T == 3) | (T == 5)) & ~orth)               # [B,64,8]
    king = (T == 6).unsqueeze(-1) & self.kstep.view(1, 1, 8, 7)                            # [B,64,8,7]
    M = (reach & (slide.unsqueeze(-1) | king)).reshape(B, 64, 56).to(dt)
    att = torch.einsum('bfk,fkt->bft', M, self.ray_oh.to(dt))                              # [B,64,64]
    att = att + (T == 2).to(dt) * self.kn.to(dt).unsqueeze(0)
    own = ((codes >= 1) & (codes <= 6)).to(dt)
    theirs = (codes >= 7).to(dt)
    empty = (codes == 0).to(dt)
    pawn = (T == 1).to(dt)                                                                 # [B,64,1]
    mid_empty = torch.gather(empty, 1, self.push_mid.unsqueeze(0).expand(B, -1)).unsqueeze(-1)
    pw = pawn * (self.push.to(dt) * empty.unsqueeze(1) + self.dbl.to(dt) * mid_empty * empty.unsqueeze(1)
                 + self.pcap.to(dt) * theirs.unsqueeze(1))
    moves = att * (1.0 - own).unsqueeze(1) + pw
    return (moves > 0.5).reshape(B, 4096)

  def rows(self, codes: torch.Tensor) -> torch.Tensor:
    B = codes.shape[0]
    ext, F, D = self._scan(codes)
    A = self._attackers(ext, F, D)
    ours, theirs = self._sides(A)
    seven = torch.full_like(A, 7)
    co = torch.where(ours, A, seven).amin(dim=-1)
    ct = torch.where(theirs, A - 6, seven).amin(dim=-1)
    n_o, n_t = ours.sum(dim=-1), theirs.sum(dim=-1)
    co = torch.where(co == 7, torch.zeros_like(co), co)
    ct = torch.where(ct == 7, torch.zeros_like(ct), ct)
    occ_type = torch.where(codes > 6, codes - 6, codes)
    occ_val = self.value.index_select(0, occ_type.reshape(-1)).reshape(B, 64)
    v_co = self.value.index_select(0, co.reshape(-1)).reshape(B, 64)
    v_ct = self.value.index_select(0, ct.reshape(-1)).reshape(B, 64)
    our_pc = (codes >= 1) & (codes <= 5)
    their_pc = (codes >= 7) & (codes <= 11)
    en_prise = (our_pc & (n_t > 0) & ((v_ct < occ_val) | (n_o == 0))).sum(dim=1).clamp(max=3)
    wins = (their_pc & (n_o > 0) & ((v_co < occ_val) | (n_t == 0))).sum(dim=1).clamp(max=3)
    kpos = self._king_pos(codes, 6)
    in_check = (torch.gather(n_t, 1, kpos.clamp(max=63).unsqueeze(1)).squeeze(1) > 0) & (kpos < 64)
    fl_o, att_o = self._flights(codes, 6, 1, 6, True)
    fl_t, _ = self._flights(codes, 12, 7, 12, False)
    L = self.legal(codes, self.pseudo_legal(codes, ext, F, D), A, F, D, att_o)
    fr = self.pair_from.unsqueeze(0).expand(B, -1)
    to = self.pair_to.unsqueeze(0).expand(B, -1)
    has_check = (L & (self.gc(codes, fr, to) > 0)).any(dim=1)
    loc = torch.stack([en_prise, wins, in_check.to(codes.dtype), has_check.to(codes.dtype), fl_o, fl_t], dim=1)
    return self.bases.unsqueeze(0) + loc                                                  # [B, 6]

  def _sum(self, tab, rows):
    B = rows.shape[0]
    return tab.index_select(0, rows.reshape(-1)).reshape(B, NUM_SLOTS, tab.shape[1]).sum(dim=1)

  def square_add(self, rows):
    return self._sum(self.table, rows).unsqueeze(1)                                        # [B, 1, D]

  def value_add(self, rows):
    return self._sum(self.vh_table, rows)                                                  # [B, Hv]
