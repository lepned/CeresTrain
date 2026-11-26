"""The 70-token input (config FryInput; 2026-11-26), after W. Gong's "Fry" chess transformer.

64 square tokens + 6 metadata tokens, built inside forward from our square record:

    token    meaning                               id
    0..63    piece on the square                   0 empty, 1..6 our P N B R Q K, 7..12 their P N B R Q K
    64       colour to move                        13 Black, 14 White
    65       we may castle king side               15 no, 16 yes
    66       we may castle queen side              15 / 16
    67       they may castle king side             15 / 16
    68       they may castle queen side            15 / 16
    69       en passant                            17 none, else 17 + target square (40..47 in our frame)

    x0[b, t] = tok_embedding[id[b, t]] + pos_embedding[t]          (81 ids, 70 positions; no norm, no dense layer)

Frame: the side to move, square = rank * 8 + file with a1 = 0 -- the frame and token order of our record (token t of
a training record IS square t: checked on real TPG shards and on the DirectFromV6 loader, see FRY_BLOCKS_NOTES.md).

What our record cannot give exactly:
  * colour (token 64): the record is side-to-move relative and carries no colour. `tokens_from_record` writes the
    constant COLOUR_FROM_RECORD (14, "White"); a net trained on our records never sees the other id.
  * en passant (token 69): Fry sets it only when an en-passant capture is LEGAL. The record flags the double-pushed
    pawn (byte 118 on its square, 32 + file); here the token is set when one of our pawns stands beside it
    (pseudo-legal). The two differ only when the capture would expose our king.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from square_features import codes_from_record

NUM_SQUARES = 64
NUM_META = 6
NUM_TOKENS = NUM_SQUARES + NUM_META
VOCAB = 81
TOK_BLACK, TOK_WHITE = 13, 14
TOK_NO, TOK_YES = 15, 16
TOK_EP_NONE = 17
COLOUR_FROM_RECORD = TOK_WHITE
REC_CASTLE = 112          # record bytes 112..115: CanOO, CanOOO, OpponentCanOO, OpponentCanOOO
REC_EN_PASSANT = 118      # record byte 118: set on the square of the pawn that just made a double step


def tokens_from_record(squares: torch.Tensor, codes: torch.Tensor = None) -> torch.Tensor:
  """Our record [B, 64, >= 137] -> the 70 token ids [B, 70] int64 (see the module text for colour and e.p.)."""
  if codes is None:
    codes = codes_from_record(squares)
  B = codes.shape[0]
  colour = torch.full((B, 1), COLOUR_FROM_RECORD, dtype=torch.int64, device=codes.device)
  castle = TOK_NO + squares[:, 0, REC_CASTLE:REC_CASTLE + 4].round().to(torch.int64)             # [B, 4]
  pushed = squares[:, 32:40, REC_EN_PASSANT] > 0.5                                               # [B, 8] by file
  ours = (codes[:, 32:40] == 1)                                                                  # our pawns on that rank
  zero = torch.zeros(B, 1, dtype=torch.bool, device=codes.device)
  beside = torch.cat([ours[:, 1:], zero], dim=1) | torch.cat([zero, ours[:, :-1]], dim=1)
  target = 40 + torch.arange(8, device=codes.device, dtype=torch.int64)
  ep = TOK_EP_NONE + ((pushed & beside).to(torch.int64) * target).sum(dim=1, keepdim=True)
  return torch.cat([codes, colour, castle, ep], dim=1)


class FryInput(nn.Module):
  """Token + position embedding of the 70 tokens. `forward(tokens)` -> [B, 70, width]."""

  def __init__(self, width: int) -> None:
    super().__init__()
    self.width = width
    self.tok_embedding = nn.Embedding(VOCAB, width)
    self.pos_embedding = nn.Parameter(torch.randn(NUM_TOKENS, width))

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    return self.tok_embedding(tokens) + self.pos_embedding
