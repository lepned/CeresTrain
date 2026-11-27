"""Per-square attention templates (config AttnTemplates; 2026-11-20).

A small bank of learned 64 x 64 logit tables shared by the whole trunk, mixed PER QUERY SQUARE and per head from the
square's own state (after W. Gong's "Fry" chess transformer, gilliamwong.github.io/chess-transformer-site):

    c[b, i, h, t] = mix(xn[b, i])                       one Linear (in_dim -> heads * T) per site, with bias
    bias[b, h, i, j] = sum_t c[b, i, h, t] * table[t, i, j]

and `bias` is added to the attention logits before the softmax, on the same path as the shared smolgen bias
(piece_relation_bias: MatMul -> Add -> softmax). Unlike smolgen (one mix per head from a whole-board summary, the
same for every query square) each query square chooses its own mix, and there are T (8) tables instead of a
64 x 64 table per smolgen code direction. No whole-board GEMM: one in_dim -> heads * T GEMM and a T-term contraction.

The table starts at ZERO: the bias is exactly 0 at step 0 and the net is its template-free self in function
(the mixer has an ordinary Linear init, so the table gets a gradient from the first step; the mixer's gradient is 0
until the table moves -- the usual one-side-zero pairing).

Sites (config AttnTemplateScope), wired in nbt_layer.NestedBottleneckLayer:
    'layer'  one mixer per inner layer, on that layer's own pre-attention norm output (inner width)   [Fry's form]
    'block'  one mixer per NBT block, on the normalised block input (outer width); its bias goes to every inner
             layer of the block (the arrangement of NBTSharedSmolgen)

Optimiser classes (wd_partition.py / train.py): the table is a raw zero-init logit table -> no weight decay, AdamW
in every Muon scope; the mixer kernels are the table's coupling partner (their gradient is ~0 while the table is
~0, which Muon's fixed-spectral-norm update would turn into full-size noise) -> AdamW as well, ordinary decay.

Piece-dependent templates (config TemplatesPieceQuery / TemplatesPieceKey, 2026-11-27; plain encoder trunk only):
the template map used for query square i becomes

    TemplatesPieceQuery   table[t, i, j] + piece_query[t, code(i), i, j]
    TemplatesPieceKey     table[t, i, j] + piece_key[t, code(j), i, j]

with code(s) the piece on s (0 empty, 1..6 our P N B R Q K, 7..12 theirs, side-to-move frame; square_features.
codes_from_record). piece_* are FULL [T, 13, 64, 64] tables, shared by every layer like `table` (8 templates:
425,984 parameters each), ZERO-initialised: the bias equals the control's bit for bit at step 0 (the piece term is
a separate c @ 0 = +-0 added to the unchanged control product), and they get a gradient from the first step (c is
non-zero at init). The position's own maps D_pos[b, i, t, j] (query and key terms summed) are gathered ONCE per
forward (AttnTemplateBank.position_tables: one Gather of [13 * 64] rows on code * 64 + square, no loop) and every
layer adds c @ D_pos to its bias. Optimiser class as `table` (raw zero-init logit tables: no_decay, AdamW; they are
4-D so Muon never takes them).
"""

from __future__ import annotations

import torch
import torch.nn as nn


NUM_PIECE_CODES = 13     # 0 empty, 1..6 ours, 7..12 theirs (square_features.codes_from_record)


class AttnTemplateBank(nn.Module):
  """The T shared tables: table[t, i, j] = logit template t for query square i attending key square j.
  Optionally the piece-dependent parts piece_query / piece_key [T, 13, 64, 64] (see the module docstring)."""

  def __init__(self, count: int, num_tokens: int = 64, piece_query: bool = False, piece_key: bool = False) -> None:
    super().__init__()
    assert count >= 1, count
    self.count = count
    self.num_tokens = num_tokens
    self.table = nn.Parameter(torch.zeros(count, num_tokens, num_tokens))
    self.piece_query = (nn.Parameter(torch.zeros(count, NUM_PIECE_CODES, num_tokens, num_tokens))
                        if piece_query else None)
    self.piece_key = (nn.Parameter(torch.zeros(count, NUM_PIECE_CODES, num_tokens, num_tokens))
                      if piece_key else None)
    if piece_query or piece_key:
      self.register_buffer('sq_index', torch.arange(num_tokens), persistent=False)

  @property
  def has_piece(self) -> bool:
    return self.piece_query is not None or self.piece_key is not None

  def position_tables(self, codes: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Piece codes [B, 64] int64 -> the position's piece maps D_pos [B, i, T, j] in `dtype`:
         D_pos[b, i, t, j] = piece_query[t, code[b, i], i, j] + piece_key[t, code[b, j], i, j]   (present parts only)
    One Gather per part on row code * 64 + square of the table laid out as [(code, square), ...]."""
    assert self.has_piece
    B, n, T = codes.shape[0], self.num_tokens, self.count
    rows = (codes * n + self.sq_index).reshape(-1)                                  # [B * 64]
    out = None
    if self.piece_query is not None:
      tq = self.piece_query.to(dtype).permute(1, 2, 0, 3).reshape(NUM_PIECE_CODES * n, T * n)   # [(k, i), (t, j)]
      out = tq.index_select(0, rows).reshape(B, n, T, n)                                         # [b, i, t, j]
    if self.piece_key is not None:
      tk = self.piece_key.to(dtype).permute(1, 3, 2, 0).reshape(NUM_PIECE_CODES * n, n * T)     # [(k, j), (i, t)]
      g = tk.index_select(0, rows).reshape(B, n, n, T).permute(0, 2, 3, 1)                       # [b, j, i, t] -> [b, i, t, j]
      out = g if out is None else out + g
    return out.contiguous()


class _BankRef:
  """Holds the bank without registering it as a submodule (it is registered once, at the net root)."""

  def __init__(self, bank: AttnTemplateBank) -> None:
    self.bank = bank


class AttnTemplateMixer(nn.Module):
  """One site's mixing weights: normalised state [B, 64, in_dim] -> logit bias [B, heads, 64, 64]."""

  def __init__(self, in_dim: int, heads: int, bank: AttnTemplateBank) -> None:
    super().__init__()
    self.heads = heads
    self.count = bank.count
    self.num_tokens = bank.num_tokens
    self.mix = nn.Linear(in_dim, heads * bank.count)
    self._ref = _BankRef(bank)

  @property
  def bank(self) -> AttnTemplateBank:
    return self._ref.bank

  def coefficients(self, xn: torch.Tensor) -> torch.Tensor:
    """[B, 64, in_dim] -> [B, 64, heads, T] (query square, head, template)."""
    return self.mix(xn).reshape(xn.shape[0], self.num_tokens, self.heads, self.count)

  def forward(self, xn: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
    """pos: the position's piece maps [B, i, T, j] (AttnTemplateBank.position_tables), or None (plain templates)."""
    c = self.coefficients(xn)                                     # [B, i, h, t]
    table = self.bank.table.to(c.dtype).transpose(0, 1)           # [i, t, j]
    # per query square i one [h, t] @ [t, j] product, broadcast over the batch: [B, i, h, j] -> [B, h, i, j]
    bias = torch.matmul(c, table)
    if pos is not None:
      # piece-dependent part as a SEPARATE product added to the unchanged control product (bit-identical at D = 0)
      bias = bias + torch.matmul(c, pos.to(c.dtype))
    return bias.permute(0, 2, 1, 3)
