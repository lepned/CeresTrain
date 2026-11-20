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
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttnTemplateBank(nn.Module):
  """The T shared tables: table[t, i, j] = logit template t for query square i attending key square j."""

  def __init__(self, count: int, num_tokens: int = 64) -> None:
    super().__init__()
    assert count >= 1, count
    self.count = count
    self.num_tokens = num_tokens
    self.table = nn.Parameter(torch.zeros(count, num_tokens, num_tokens))


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

  def forward(self, xn: torch.Tensor) -> torch.Tensor:
    c = self.coefficients(xn)                                     # [B, i, h, t]
    table = self.bank.table.to(c.dtype).transpose(0, 1)           # [i, t, j]
    # per query square i one [h, t] @ [t, j] product, broadcast over the batch: [B, i, h, j] -> [B, h, i, j]
    return torch.matmul(c, table).permute(0, 2, 1, 3)
