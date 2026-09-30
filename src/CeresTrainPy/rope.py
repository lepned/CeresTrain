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
import math
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
    base=100:   Pol 1981  (-14 Pol; NB the fastest pair is 1 rad/square at EVERY
                           base -- base only moves the slow end of the spectrum,
                           and the +-14 differences sit at the seed-noise floor)
  Override via env var ROPE_BASE for ablations. NB not stored in the checkpoint
  (the tables are non-persistent buffers): a re-export must run with the same base."""
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


class LearnableRope2D(torch.nn.Module):
  """Learnable 2D RoPE (KataGo model_pytorch.py: learnable_rope / compute_learnable_rope_cos_sin,
  the form their trained transformer nets use; config RoPELearnable).

  Every head and every dim pair p owns a frequency vector (omega_x, omega_y); the pair on
  square (file, rank) is rotated by angle = omega_x * file + omega_y * rank. Unlike the fixed
  2D RoPE above (first half of d_head = file, second half = rank, one spectrum shared by all
  heads and layers) a pair here is a plane wave in ANY board direction, so a head can express
  diagonal or knight geometry in q.k directly, and each layer learns its own spectrum.
  q.k stays a function of the offset (dfile, drank) only.

  Init (config RoPELearnableInit, RoPELearnableFreqMin/Max; DEFAULT = 'stratified' [1/16, pi/2],
  user decision 2026-09-30 after the 8x8 assessment below):
    'loguniform': each component log-uniform in [min, max] rad/square with a random sign
                  (KataGo: [1/50, 1], calibrated on 19x19; on 8x8 the median |omega| is 0.34
                  and no pair exceeds 1 rad/square, see the 09-30 assessment).
    'stratified': per head the magnitudes are geometric from max down to min across the pairs
                  (every head gets the full multi-scale bank, like the fixed spectrum) and
                  the direction is uniform on the circle (diagonals/knight directions covered).
  Drawn from a private generator seeded by the layer index, so a learnable arm stays
  init-paired with a fixed-RoPE control (repo convention: no global-RNG draws in modules).

  Training: the tables are rebuilt from `freqs` on every forward, in fp32; the rotation then
  runs at the precision of the FIXED path (fp32 tables x bf16 Q promote to fp32) so the two
  variants are numerically comparable. `freqs` is 3-D (H, d_head/2, 2) => AdamW under every
  Muon scope (ndim != 2). Weight decay: wd_partition puts it in no_decay, and train.py gives
  it Muon wd scale 0 unconditionally (KataGo: ~zero WD); AdamW-family optimizers honour
  no_decay by themselves.

  Export: call `bake()` on the export copy (save_model does) -- the onnx exporter only
  constant-folds tensors up to 8192 elements, so above ModelDim 256 an unbaked graph keeps
  Cos/Sin(Add(Mul, Mul)) on FP16 angles of up to ~14 rad (24x the rotation error)."""
  # 1/16 rad/square = 0,44 rad across the board (no wasted, position-free pairs); pi/2 = adjacent
  # offsets 90 degrees apart (sharp local geometry) without reaching the pure parity pair at pi.
  # KataGo's own init is loguniform [1/50, 1] (calibrated on 19x19).
  DEFAULT_MIN_FREQ = 1.0 / 16.0
  DEFAULT_MAX_FREQ = math.pi / 2
  INITS = ('loguniform', 'stratified')

  def __init__(self, num_heads: int, d_head: int, layer_num: int = 0, init: str = 'stratified',
               freq_min: float = DEFAULT_MIN_FREQ, freq_max: float = DEFAULT_MAX_FREQ):
    super().__init__()
    assert d_head % 2 == 0, f"d_head={d_head} must be even (rotation acts on pairs)"
    assert init in self.INITS, f"RoPELearnableInit must be one of {self.INITS} (was {init!r})"
    assert 0 < freq_min < freq_max, f"RoPELearnableFreqMin/Max must satisfy 0 < min < max (was {freq_min}, {freq_max})"
    self.num_heads = num_heads
    self.d_head = d_head
    self.init = init
    n_pairs = d_head // 2
    g = torch.Generator().manual_seed(0x0E0F + int(layer_num or 0))
    log_lo, log_hi = math.log(freq_min), math.log(freq_max)
    if init == 'loguniform':
      mag = torch.exp(torch.empty(num_heads, n_pairs, 2).uniform_(log_lo, log_hi, generator=g))
      sign = (torch.randint(0, 2, (num_heads, n_pairs, 2), generator=g) * 2 - 1).float()
      freqs = mag * sign
    else:
      # |omega| geometric max -> min over the pairs (identical bank in every head), direction
      # uniform on the circle per head and pair.
      mags = torch.exp(torch.linspace(log_hi, log_lo, n_pairs))                       # (P,)
      theta = torch.empty(num_heads, n_pairs).uniform_(0.0, 2.0 * math.pi, generator=g)  # (H, P)
      freqs = torch.stack([mags * torch.cos(theta), mags * torch.sin(theta)], dim=-1)  # (H, P, 2)
    self.freqs = torch.nn.Parameter(freqs)          # (H, P, 2) = (omega_x, omega_y)
    squares = torch.arange(64)
    # Board coordinates, canonical 0-63 order (file = idx % 8, rank = idx // 8), as the
    # fixed tables above. Any constant offset cancels in q.k.
    self.register_buffer('files', (squares % 8).float(), persistent=False)
    self.register_buffer('ranks', (squares // 8).float(), persistent=False)
    self._baked = False

  def tables(self):
    """(cos, sin), each (H, 64, d_head), pairs interleaved for apply_rope; fp32."""
    f = self.freqs.float()
    angles = (self.files[None, :, None] * f[:, None, :, 0]
              + self.ranks[None, :, None] * f[:, None, :, 1])          # (H, 64, P)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)                # (H, 64, d_head)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin

  def bake(self):
    """Freeze the tables as fp32 buffers for export (see class docstring). Only on a COPY of
    the training model: a baked module no longer reads `freqs`."""
    with torch.no_grad():
      cos, sin = self.tables()
    self.register_buffer('baked_cos', cos.detach().clone(), persistent=False)
    self.register_buffer('baked_sin', sin.detach().clone(), persistent=False)
    self._baked = True

  def forward(self):
    if self._baked:
      return self.baked_cos, self.baked_sin
    return self.tables()


def bake_learnable_rope(model) -> int:
  """Bake every LearnableRope2D in `model` (an EXPORT COPY). Returns the count."""
  n = 0
  for m in model.modules():
    if isinstance(m, LearnableRope2D) and not m._baked:
      m.bake(); n += 1
  return n
