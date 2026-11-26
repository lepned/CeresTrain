"""Square features (config SquareFeatures; 2026-11-26): per square, the sum of 23 rows of ONE learned table.

After the input of W. Gong's "Fry" chess transformer. Each of the 64 squares looks up 23 rows of a zero-initialised
[1735, width] table and adds their sum to its embedding; the rows are computed inside forward from the 64 piece codes
of the current board in the side-to-move frame (square = rank * 8 + file, a1 = 0; code 0 empty, 1..6 our P N B R Q K,
7..12 their P N B R Q K -- the order of bytes 0..12 of our square record, so `codes_from_record` is a plain read).

    slot    base   rows   local index
    0       0      226    offset of the square from OUR KING: (rank_s - rank_k + 7) * 15 + (file_s - file_k + 7); 225 absent
    1       226    226    the same for THEIR KING
    2       452    226    the same for OUR QUEEN (the one on the lowest square index); 225 if none
    3       678    226    the same for THEIR QUEEN
    4..11   904    14 x 8 code (0..12; 13 = off the board) of the neighbour in direction d
    12..19  1016   85 x 8 first piece on the ray in direction d: 0 none, else 1 + (code - 1) * 7 + (distance - 1)
    20      1696   9      min(our knights attacking the square, 2) * 3 + min(their knights attacking it, 2)
    21      1705   15     nearest PAWN up the file (other pieces skipped): 0 none, else 1 + (7 if theirs) + (distance - 1)
    22      1720   15     the same down the file

Directions d = 0..7 as (file step, rank step): (0,+1) (0,-1) (+1,0) (-1,0) (+1,+1) (-1,+1) (+1,-1) (-1,-1).

The index computation is integer gathers and comparisons on constant index tables (no ArgMax, no CumSum, no loop over
squares): the "first piece on a ray" is a 7-step unrolled scan. The table starts at ZERO, so the module adds exactly 0
at step 0 and a net is its feature-free self in function until the table moves.

Optimiser classes (wd_partition.py / train.py): the table is a lookup table -> AdamW in every Muon scope, no weight
decay in our default partition (Fry's own partition decays it: Opt key WeightDecayScheme 'fry').
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_ROWS = 1735
NUM_SLOTS = 23
OFF_BOARD = 13                      # the "code" of a square beyond the edge (neighbour slots)
DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1))   # (file step, rank step)
KNIGHT_JUMPS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
BASE_KING_QUEEN = (0, 226, 452, 678)     # our king, their king, our queen, their queen
BASE_NEIGHBOUR = 904
BASE_RAY = 1016
BASE_KNIGHT = 1696
BASE_PAWN_UP = 1705
BASE_PAWN_DOWN = 1720


def codes_from_record(squares: torch.Tensor) -> torch.Tensor:
  """Our square record [B, 64, >= 13] (bytes 0..12 = one-hot of the piece on the square, current board) ->
  piece codes [B, 64] int64. A weighted sum instead of argmax: the one-hot is exactly 0 / 1 in any float type."""
  w = torch.arange(13, device=squares.device, dtype=squares.dtype)
  return (squares[:, :, 0:13] * w).sum(dim=-1).round().to(torch.int64)


def _step_table(steps, max_len: int) -> torch.Tensor:
  """[64, len(steps), max_len] square reached from s by k + 1 steps of each (file, rank) step; 64 = off the board."""
  t = torch.full((64, len(steps), max_len), 64, dtype=torch.int64)
  for s in range(64):
    f0, r0 = s % 8, s // 8
    for d, (df, dr) in enumerate(steps):
      for k in range(max_len):
        f, r = f0 + df * (k + 1), r0 + dr * (k + 1)
        if not (0 <= f < 8 and 0 <= r < 8):
          break
        t[s, d, k] = r * 8 + f
  return t


def _exporting() -> bool:
  # TorchScript tracing, the legacy ONNX exporter, and torch.export (the dynamo ONNX exporter)
  return torch.onnx.is_in_onnx_export() or torch.jit.is_tracing() or torch.compiler.is_exporting()


class SquareFeatures(nn.Module):
  """codes [B, 64] int64 -> [B, 64, width]: the sum of the square's 23 table rows."""

  def __init__(self, width: int) -> None:
    super().__init__()
    self.width = width
    self.table = nn.Parameter(torch.zeros(NUM_ROWS, width))
    sq = torch.arange(64)
    self.register_buffer('sq_rank', sq // 8, persistent=False)
    self.register_buffer('sq_file', sq % 8, persistent=False)
    self.register_buffer('ray_idx', _step_table(DIRECTIONS, 7), persistent=False)                 # [64, 8, 7]
    self.register_buffer('knight_idx', _step_table(KNIGHT_JUMPS, 1)[:, :, 0], persistent=False)   # [64, 8]
    self.register_buffer('nbr_base', BASE_NEIGHBOUR + 14 * torch.arange(8), persistent=False)
    self.register_buffer('ray_base', BASE_RAY + 85 * torch.arange(8), persistent=False)

  def _offset_rows(self, codes: torch.Tensor, code: int, base: int) -> torch.Tensor:
    """Rows of the offset of every square from the (lowest-index) piece with `code`; base + 225 if there is none."""
    sq = self.sq_rank * 8 + self.sq_file                                              # arange(64) on the device
    pos = torch.where(codes == code, sq, torch.full_like(codes, 64)).amin(dim=1, keepdim=True)   # [B, 1]; 64 = absent
    pr = torch.div(pos, 8, rounding_mode='floor')
    pf = pos - pr * 8
    local = (self.sq_rank - pr + 7) * 15 + (self.sq_file - pf + 7)                    # [B, 64]
    return base + torch.where(pos < 64, local, torch.full_like(local, 225))

  def rows(self, codes: torch.Tensor) -> torch.Tensor:
    """codes [B, 64] int64 (0..12) -> table rows [B, 64, 23] int64."""
    B = codes.shape[0]
    ext = torch.cat([codes, torch.zeros(B, 1, dtype=codes.dtype, device=codes.device)], dim=1)   # square 64 = off board
    ray = ext.index_select(1, self.ray_idx.reshape(-1)).reshape(B, 64, 8, 7)                      # codes along each ray; 0 off board
    on_board = (self.ray_idx < 64)                                                    # [64, 8, 7]

    out = [self._offset_rows(codes, 6, BASE_KING_QUEEN[0]), self._offset_rows(codes, 12, BASE_KING_QUEEN[1]),
           self._offset_rows(codes, 5, BASE_KING_QUEEN[2]), self._offset_rows(codes, 11, BASE_KING_QUEEN[3])]

    # neighbours: the first step of each ray; off the board -> code 13
    nbr = torch.where(on_board[:, :, 0], ray[:, :, :, 0], torch.full_like(ray[:, :, :, 0], OFF_BOARD))
    out.append(nbr + self.nbr_base)                                                   # [B, 64, 8]

    # first piece on each ray, and the nearest pawn up / down the file: a 7-step scan (no ArgMax / CumSum)
    free = torch.ones_like(ray[:, :, :, 0], dtype=torch.bool)                         # nothing met so far
    first = torch.zeros_like(ray[:, :, :, 0])
    pfree = torch.ones_like(ray[:, :, 0:2, 0], dtype=torch.bool)
    pawn = torch.zeros_like(ray[:, :, 0:2, 0])
    for k in range(7):
      c = ray[:, :, :, k]
      occ = c > 0
      hit = free & occ
      first = first + hit.to(c.dtype) * (1 + (c - 1) * 7 + k)
      free = free & ~occ
      cp = c[:, :, 0:2]
      is_p = (cp == 1) | (cp == 7)
      phit = pfree & is_p
      pawn = pawn + phit.to(c.dtype) * (1 + (cp == 7).to(c.dtype) * 7 + k)
      pfree = pfree & ~is_p
    out.append(first + self.ray_base)                                                 # [B, 64, 8]

    kn = ext.index_select(1, self.knight_idx.reshape(-1)).reshape(B, 64, 8)
    ours = (kn == 2).sum(dim=-1).clamp(max=2)
    theirs = (kn == 8).sum(dim=-1).clamp(max=2)
    out.append(BASE_KNIGHT + ours * 3 + theirs)
    out.append(BASE_PAWN_UP + pawn[:, :, 0])
    out.append(BASE_PAWN_DOWN + pawn[:, :, 1])
    return torch.cat([o.unsqueeze(-1) if o.dim() == 2 else o for o in out], dim=-1)   # [B, 64, 23]

  def forward(self, codes: torch.Tensor) -> torch.Tensor:
    rows = self.rows(codes)
    B = rows.shape[0]
    if _exporting():
      # export graph: Gather + ReduceSum (embedding_bag exports as a loop)
      return self.table.index_select(0, rows.reshape(-1)).reshape(B, 64, NUM_SLOTS, self.width).sum(dim=2)
    # training / eval: summed lookup without materialising [B, 64, 23, width]
    return torch.nn.functional.embedding_bag(rows.reshape(B * 64, NUM_SLOTS), self.table, mode='sum').reshape(B, 64, self.width)
