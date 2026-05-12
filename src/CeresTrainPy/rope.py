# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""True RoPE (Rotary Position Embedding) for chess board attention.

Per Su et al. "RoFormer" (2021). Encodes position by rotating Q and K vectors
in pairs by an angle proportional to position index. Zero learnable parameters,
stays on the fast scaled_dot_product_attention path (no bias addition needed),
position info is intrinsic to the rotated Q/K.

For chess: 2D RoPE with file (0-7) and rank (0-7) halves of d_head. First half
of d_head rotates by file index, second half rotates by rank index. Squares
are indexed in canonical 0-63 order: file = idx % 8, rank = idx // 8.
"""
import os
import torch
from torch import Tensor


def precompute_rope_freqs(d_head: int, base: float = None):
  """If base is None, reads from ROPE_BASE env var (default 1000.0).

  Default base=1000 was chosen via 256-12 SwiGLU pre-norm 3M ablation:
    base=10000: Pol 1995  (RoFormer default — wastes ~half the freq dims at
                           chess's 0-7 position range, those dims rotate <0.01
                           rad over the whole board so carry no signal)
    base=1000:  Pol 2009  (+14 Pol — goldilocks, most freq dims active)
    base=100:   Pol 1981  (-14 Pol — too aggressive, fastest dims alias
                           before reaching position 7)
  Override via env var ROPE_BASE for ablations."""
  if base is None:
    base = float(os.environ.get('ROPE_BASE', '1000.0'))
  # Returns:
  #   cos_table, sin_table: each shape (64, d_head). Indexed by square 0-63.
  #   Designed to multiply Q/K of shape (B, num_heads, 64, d_head) via broadcast.
  assert d_head % 2 == 0, f"d_head={d_head} must be even"
  d_half = d_head // 2  # bytes per axis (file + rank)
  assert d_half % 2 == 0, f"d_half={d_half} must be even (we rotate pairs)"

  # Frequencies: standard RoPE schedule, base ** (-2i/d) for i in 0..d/2
  freqs = 1.0 / (base ** (torch.arange(0, d_half, 2).float() / d_half))  # (d_half/2,)

  # Square indices → (file, rank) in canonical 0-63 order
  squares = torch.arange(64)
  files = (squares % 8).float()  # (64,)
  ranks = (squares // 8).float()  # (64,)

  # angles: (64, d_half/2)
  angles_file = files[:, None] * freqs[None, :]
  angles_rank = ranks[:, None] * freqs[None, :]

  # cos/sin tables, expanded to (64, d_half) by interleaving each cos/sin
  # twice consecutively so the rotate-half trick lines up
  cos_file = torch.cos(angles_file).repeat_interleave(2, dim=-1)  # (64, d_half)
  sin_file = torch.sin(angles_file).repeat_interleave(2, dim=-1)
  cos_rank = torch.cos(angles_rank).repeat_interleave(2, dim=-1)
  sin_rank = torch.sin(angles_rank).repeat_interleave(2, dim=-1)

  cos_table = torch.cat([cos_file, cos_rank], dim=-1)  # (64, d_head)
  sin_table = torch.cat([sin_file, sin_rank], dim=-1)
  return cos_table, sin_table


def apply_rope(x: Tensor, cos_table: Tensor, sin_table: Tensor) -> Tensor:
  """Apply RoPE to Q or K.

  Args:
    x: (B, num_heads, 64, d_head)
    cos_table, sin_table: (64, d_head) — broadcasts over B and heads
  Returns rotated tensor of same shape as x.

  Standard "rotate half" formulation:
    For pairs (x[2i], x[2i+1]):
      out[2i]   = x[2i]   * cos - x[2i+1] * sin
      out[2i+1] = x[2i+1] * cos + x[2i]   * sin
    Equivalent to: x*cos_interleaved + rotate_half(x)*sin_interleaved
    where rotate_half swaps adjacent pairs and negates the first element of each.
  """
  # rotate_half: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
  x_pairs = x.reshape(*x.shape[:-1], -1, 2)  # (..., d_head/2, 2)
  x1, x2 = x_pairs.unbind(-1)
  x_rot = torch.stack([-x2, x1], dim=-1).reshape(*x.shape)
  return x * cos_table + x_rot * sin_table
