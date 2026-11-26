"""Policy and value heads in the form of W. Gong's "Fry" chess transformer (config PolicyHeadFry / ValueHeadFry;
2026-11-26), on OUR outputs: the policy in the lc0 / Ceres 1858-move layout, the value as HL-Gauss bucket logits.

Policy (y = the 64 square states after the trunk-end norm):

    F = y @ Wf + bf        T = y @ Wt + bt                      (width -> width each, no hidden layer, no activation)
    L[f, t] = F[f] . T[t] / sqrt(width)                         64 x 64, float32
    logit[m] = L[from(m), to(m)]                                for every move m of our 1858
    logit[m] += (y[to(m)] @ Wp + bp)[piece(m)]                  promotions only; Wp width -> 4, order N, B, R, Q

Our 1858 layout has 66 explicit promotion moves (suffix q / r / b); a KNIGHT promotion is the plain from-to move of
the pawn step. The 22 plain moves from rank 7 to rank 8 with |file step| <= 1 therefore take the knight term when
one of OUR PAWNS stands on the from square (read from the piece codes), and no term otherwise (a queen, rook, king
or bishop making that step). With that, the logit of every legal move equals what Fry's own 1968-move head gives
for the same weights. Castling: our training targets use king-takes-rook (e1h1 / e1a1), Fry's weights were trained
with the king's two-square move (e1g1 / e1c1); `king_castling` (config PolicyHeadFryKingCastling, for loading his
weights) makes e1h1 / e1a1 read L[e1, g1] / L[e1, c1] while our king stands on e1. Default off: a net trained on our
targets learns L[e1, h1] directly.

Value: z = GELU(y @ Wd + bd) per token (width -> 16, ALL tokens: 70 with FryInput, else 64); h = GELU(flatten(z) @ W0
+ b0) (tokens * 16 -> hidden, default width); bucket logits = h @ W1 + b1. The buckets are our HL-Gauss buckets
(equal width over q in [-1, 1], which is Fry's expected score s = (q + 1) / 2 over [0, 1] bucket for bucket).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from lc0_moves_1858 import MOVES_1858, FROM_1858, TO_1858

PROMO_ORDER = 'nbrq'      # output order of the promotion projection (Fry's: 0 N, 1 B, 2 R, 3 Q)


def _policy_tables():
  n = len(MOVES_1858)
  pair = torch.tensor([f * 64 + t for f, t in zip(FROM_1858, TO_1858)], dtype=torch.int64)
  promo_idx = torch.zeros(n, dtype=torch.int64)
  explicit = torch.zeros(n)
  knight = torch.zeros(n)
  for i, (m, f, t) in enumerate(zip(MOVES_1858, FROM_1858, TO_1858)):
    if len(m) == 5:
      promo_idx[i] = t * 4 + PROMO_ORDER.index(m[4])
      explicit[i] = 1.0
    elif f // 8 == 6 and t // 8 == 7 and abs(f % 8 - t % 8) <= 1:
      promo_idx[i] = t * 4 + PROMO_ORDER.index('n')
      knight[i] = 1.0
  assert int(explicit.sum()) == 66 and int(knight.sum()) == 22, (explicit.sum(), knight.sum())
  # king-castling remap: index of e1h1 reads the pair (e1, g1), e1a1 reads (e1, c1)
  alt = pair.clone()
  castle = torch.zeros(n, dtype=torch.bool)
  for kxr, king2 in (('e1h1', 'e1g1'), ('e1a1', 'e1c1')):
    i, j = MOVES_1858.index(kxr), MOVES_1858.index(king2)
    alt[i] = pair[j]
    castle[i] = True
  return pair, promo_idx, explicit, knight, alt, castle


class FryPolicyHead(nn.Module):
  """[B, 64, width] square states + piece codes [B, 64] -> [B, 1858] logits (float32)."""

  def __init__(self, width: int, king_castling: bool = False) -> None:
    super().__init__()
    self.width = width
    self.king_castling = bool(king_castling)
    self.from_proj = nn.Linear(width, width)
    self.to_proj = nn.Linear(width, width)
    self.promo_proj = nn.Linear(width, 4)
    pair, promo_idx, explicit, knight, alt, castle = _policy_tables()
    self.register_buffer('pair_idx', pair, persistent=False)
    self.register_buffer('promo_idx', promo_idx, persistent=False)
    self.register_buffer('promo_explicit', explicit, persistent=False)
    self.register_buffer('promo_knight', knight, persistent=False)
    self.register_buffer('from_idx', torch.tensor(FROM_1858, dtype=torch.int64), persistent=False)
    self.register_buffer('alt_pair_idx', alt, persistent=False)
    self.register_buffer('castle_mask', castle, persistent=False)

  def forward(self, y: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    B = y.shape[0]
    f = self.from_proj(y).float()
    t = self.to_proj(y).float()
    p = self.promo_proj(y).float()
    with torch.autocast(device_type=y.device.type, enabled=False):
      L = torch.matmul(f, t.transpose(1, 2)).reshape(B, 64 * 64) * (1.0 / math.sqrt(self.width))
      logits = L.index_select(1, self.pair_idx)
      if self.king_castling:
        king_home = (codes[:, 4] == 6).unsqueeze(1) & self.castle_mask.unsqueeze(0)
        logits = torch.where(king_home, L.index_select(1, self.alt_pair_idx), logits)
      pawn_from = (codes == 1).to(logits.dtype).index_select(1, self.from_idx)                  # [B, 1858]
      term = p.reshape(B, 64 * 4).index_select(1, self.promo_idx)
      return logits + term * (self.promo_explicit + self.promo_knight * pawn_from)


class FryValueHead(nn.Module):
  """[B, tokens, width] token states -> [B, buckets] HL-Gauss logits."""

  def __init__(self, width: int, tokens: int, buckets: int, squeeze: int = 16, hidden: int = 0) -> None:
    super().__init__()
    self.tokens = tokens
    self.squeeze = nn.Linear(width, squeeze)
    self.fc = nn.Linear(tokens * squeeze, hidden or width)
    self.out = nn.Linear(hidden or width, buckets)

  def forward(self, y: torch.Tensor, inject: torch.Tensor = None) -> torch.Tensor:
    z = torch.nn.functional.gelu(self.squeeze(y))
    h = self.fc(z.reshape(z.shape[0], -1))
    if inject is not None:
      h = h + inject
    return self.out(torch.nn.functional.gelu(h))
