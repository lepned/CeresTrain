"""BT6 relation tensor: the 34 attack-graph channels E[B, 34, 64, 64] of Lc0's "BT6" (EGT2) nets.

Exactly what the lc0ex kernel `prologue_egt` builds from the lc0 input planes (lczero_triton/bt4/kernels/
prologue_egt.py), vectorised in torch so it can run inside training on the GPU.

Conventions
-----------
* Squares are numbered s = 8 * row + column in lc0 plane coordinates: row 0 is the back rank of the side to move.
* `pieces` is bool [B, 12, 64]: planes 0-5 = side to move P, N, B, R, Q, K, planes 6-11 = opponent P, N, B, R, Q, K
  (lc0 planes 0-11 thresholded at > 0.5).
* `previous` is the same for the previous position, already in the CURRENT side's view (lc0 planes 13-24, no flip):
  "stm" in the previous frame is still the side that is to move NOW. An all-zero `previous` (missing history) gives
  gained = all attacks, lost = nothing.
* E[b, c, i, j]: i = source (attention query) square, j = target (attention key) square.

Channels (17 base channels, then their 17 transposes: E[17 + c, i, j] = E[c, j, i])
    0/1   attacks stm/opp   pawn diagonals (stm toward +row, opp toward -row), knight, king, slider rays up to and
                            INCLUDING the first blocker; any target (empty, enemy or friendly); no en passant
    2/3   defends stm/opp   attacks whose target holds a piece of the same side
    4/5   xray stm/opp      sliders only: squares strictly beyond the first blocker up to and including the second
    6/7   pinray stm/opp    [b, t]: a slider of that side hits the occupied square b first along direction d, and t
                            is the first occupied square beyond b along the same d (b, t of any colour)
    8     kingring1_opp     [q, k]: k holds an opp king and the stm attacks of q touch a square at Chebyshev
                            distance exactly 1 from k
    9     kingring1_own     the same with opp attacks and the stm king
    10/11 kingring2 opp/own the same with Chebyshev distance exactly 2
    12    blockescape       [s, k]: s occupied, k holds a king (either side) at Chebyshev distance 1 from s
    13/14 gained stm/opp    attacks & ~previous attacks
    15/16 lost stm/opp      previous attacks & ~attacks
"""

from __future__ import annotations

import torch
import torch.nn as nn

BASE_CHANNELS = 17
RELATION_CHANNELS = 34
CHANNEL_NAMES = (
  "attacks_stm", "attacks_opp", "defends_stm", "defends_opp", "xray_stm", "xray_opp", "pinray_stm", "pinray_opp",
  "kingring1_opp", "kingring1_own", "kingring2_opp", "kingring2_own", "blockescape",
  "gained_stm", "gained_opp", "lost_stm", "lost_opp",
)

# Slider directions as (row step, column step), in the kernel's order: NE, NW, SE, SW (diagonal), N, S, E, W.
SLIDER_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))
# Index of the opposite direction of each entry of SLIDER_DIRECTIONS.
OPPOSITE_DIRECTION = (3, 2, 1, 0, 5, 4, 7, 6)
_PAD = 64  # index used for "off the board" while the ray tables are built


def _leaper_table(steps) -> torch.Tensor:
  """bool [64, 64]: table[i, j] = j is reached from i by one of the (row, column) steps."""
  table = torch.zeros(64, 64, dtype=torch.bool)
  for square in range(64):
    row, column = divmod(square, 8)
    for row_step, column_step in steps:
      r, c = row + row_step, column + column_step
      if 0 <= r < 8 and 0 <= c < 8:
        table[square, 8 * r + c] = True
  return table


def _ray_index() -> torch.Tensor:
  """long [64, 8, 7]: the k-th square from i along direction d, or 64 past the edge of the board."""
  index = torch.full((64, 8, 7), _PAD, dtype=torch.long)
  for square in range(64):
    row, column = divmod(square, 8)
    for d, (row_step, column_step) in enumerate(SLIDER_DIRECTIONS):
      for k in range(7):
        r, c = row + (k + 1) * row_step, column + (k + 1) * column_step
        if 0 <= r < 8 and 0 <= c < 8:
          index[square, d, k] = 8 * r + c
  return index


class BT6Relations(nn.Module):
  """Vectorised E[B, 34, 64, 64] (bool) from the piece planes of the current and the previous position.

  No parameters; the geometry tables are non-persistent buffers, so the module follows `.to(device)` and adds
  nothing to a state dict. The result carries no gradient (BT6 trains with E under stop_gradient).

  Formulation: constant one-hot tables and float matmuls only -- no scatter, no cumulative sum, no integer indexing
  with data -- so the same code runs in training, exports to ONNX with ordinary operators (MatMul, Less, And, Or,
  Concat) and has no data-dependent shape:
    along  = occupancy @ G            the board read along the 8 x 7 ray squares of every source square
    before = along @ L                number of blockers strictly before each ray square (L strictly triangular)
    reach  = (before == 0) -> back to target squares by a [64, n, 64] one-hot matmul per source square
  All counts are small exact integers in float32 (autocast is switched off inside)."""

  def __init__(self) -> None:
    super().__init__()
    king = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0)]
    knight = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
    ring2 = [(r, c) for r in range(-2, 3) for c in range(-2, 3) if max(abs(r), abs(c)) == 2]
    self.register_buffer("pawn_up", _leaper_table([(1, -1), (1, 1)]), persistent=False)
    self.register_buffer("pawn_down", _leaper_table([(-1, -1), (-1, 1)]), persistent=False)
    self.register_buffer("knight", _leaper_table(knight), persistent=False)
    self.register_buffer("king", _leaper_table(king), persistent=False)
    # float copies of the two ring tables for the king-ring dilation (a [B, 64, 64] x [64, 64] matmul)
    self.register_buffer("king_f", _leaper_table(king).float(), persistent=False)
    self.register_buffer("ring2_f", _leaper_table(ring2).float(), persistent=False)
    rays = _ray_index()                                              # [64, 8, 7], 64 = off the board
    one_hot = torch.zeros(64, 56, 65)
    one_hot.scatter_(2, rays.reshape(64, 56, 1), 1.0)
    one_hot = one_hot[:, :, :64].contiguous()                        # [source, ray slot, square]; off-board rows are 0
    # gather table: along[b, i * 56 + k] = sum_s board[b, s] * G[s, i * 56 + k]
    self.register_buffer("gather_table", one_hot.permute(2, 0, 1).reshape(64, 64 * 56).contiguous(), persistent=False)
    # back-to-target tables per source square: [64, n, 64]
    self.register_buffer("to_target_diag", one_hot.view(64, 8, 7, 64)[:, :4].reshape(64, 28, 64).contiguous(), persistent=False)
    self.register_buffer("to_target_orth", one_hot.view(64, 8, 7, 64)[:, 4:].reshape(64, 28, 64).contiguous(), persistent=False)
    self.register_buffer("to_target_all", one_hot, persistent=False)
    self.register_buffer("strictly_before", torch.triu(torch.ones(7, 7), diagonal=1), persistent=False)
    # permutation matrix that swaps every direction with its opposite, [8, 8]
    opposite = torch.zeros(8, 8)
    for d, o in enumerate(OPPOSITE_DIRECTION):
      opposite[o, d] = 1.0
    self.register_buffer("opposite", opposite, persistent=False)

  # ------------------------------------------------------------------ helpers (float32, exact small integers)
  def _along(self, board_f: torch.Tensor) -> torch.Tensor:
    """float [B, 64] -> float [B, 64, 8, 7]: the board read along every ray of every square (0 past the edge)."""
    return torch.matmul(board_f, self.gather_table).view(board_f.shape[0], 64, 8, 7)

  @staticmethod
  def _to_targets(table: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Ray-ordered float [B, 64, n] -> target-ordered bool [B, 64, 64] through the one-hot [64, n, 64] table. The
    rays of one square never overlap, so every target receives at most one ray slot."""
    return torch.matmul(values.unsqueeze(2), table).squeeze(2) > 0.5

  def _reach(self, occupied_f: torch.Tensor):
    """Slider geometry: (blockers along the rays, ray squares up to and including the first blocker, ray squares
    strictly beyond it up to and including the second), float [B, 64, 8, 7] each."""
    along = self._along(occupied_f)
    before = torch.matmul(along, self.strictly_before)               # blockers strictly before each ray square
    open_ray = (before < 0.5).float()
    beyond = ((before > 0.5) & (before < 1.5)).float()
    return along, open_ray, beyond

  def _attacks(self, side: torch.Tensor, reach_diag: torch.Tensor, reach_orth: torch.Tensor, pawn_table: torch.Tensor):
    """Attack channel of one side. `side` is bool [B, 6, 64] (P, N, B, R, Q, K)."""
    pawn, knight, bishop, rook, queen, king = side.unbind(1)
    attacks = pawn[:, :, None] & pawn_table
    attacks = attacks | (knight[:, :, None] & self.knight)
    attacks = attacks | (king[:, :, None] & self.king)
    attacks = attacks | ((bishop | queen)[:, :, None] & reach_diag)
    return attacks | ((rook | queen)[:, :, None] & reach_orth)

  def _frame_attacks(self, pieces: torch.Tensor):
    """Channels 0/1 of one frame (used for the previous position): (attacks stm, attacks opp)."""
    batch = pieces.shape[0]
    _, open_ray, _ = self._reach(pieces.any(1).float())
    reach_diag = self._to_targets(self.to_target_diag, open_ray[:, :, :4].reshape(batch, 64, 28))
    reach_orth = self._to_targets(self.to_target_orth, open_ray[:, :, 4:].reshape(batch, 64, 28))
    return (self._attacks(pieces[:, 0:6], reach_diag, reach_orth, self.pawn_up),
            self._attacks(pieces[:, 6:12], reach_diag, reach_orth, self.pawn_down))

  def _pins(self, here: torch.Tensor, first: torch.Tensor, diagonal: torch.Tensor, orthogonal: torch.Tensor):
    """Pinray channel of one side. `first` is the first-blocker mask along the rays, float [B, 64, 8, 7]; `diagonal`
    / `orthogonal` are that side's sliders, bool [B, 64]. Square b gets its first blocker along d when b is occupied
    and the first blocker of b in the OPPOSITE direction is a slider of this side that rides the line."""
    batch = here.shape[0]
    sliders = torch.cat((self._along(diagonal.float())[:, :, :4], self._along(orthogonal.float())[:, :, 4:]), dim=2)
    first_is_slider = (first * sliders).sum(-1)                              # [B, 64, 8]: along d from b (0 or 1)
    behind = torch.matmul(first_is_slider, self.opposite)                    # the slider sits behind b
    pinned = here.float()[:, :, None] * behind                               # [B, 64, 8]
    return self._to_targets(self.to_target_all, (first * pinned[..., None]).reshape(batch, 64, 56))

  # ------------------------------------------------------------------ forward
  @torch.no_grad()
  def forward(self, pieces: torch.Tensor, previous: torch.Tensor | None = None) -> torch.Tensor:
    """pieces, previous: bool [B, 12, 64] (previous None = missing history). Returns bool [B, 34, 64, 64]."""
    with torch.autocast(device_type=pieces.device.type, enabled=False):
      pieces = pieces.bool()
      batch = pieces.shape[0]
      us, them = pieces[:, 0:6], pieces[:, 6:12]
      occupied_us, occupied_them = us.any(1), them.any(1)
      occupied = occupied_us | occupied_them
      diagonal_us, orthogonal_us = us[:, 2] | us[:, 4], us[:, 3] | us[:, 4]
      diagonal_them, orthogonal_them = them[:, 2] | them[:, 4], them[:, 3] | them[:, 4]
      king_us, king_them = us[:, 5], them[:, 5]

      along, open_ray, beyond = self._reach(occupied.float())
      reach_diag = self._to_targets(self.to_target_diag, open_ray[:, :, :4].reshape(batch, 64, 28))
      reach_orth = self._to_targets(self.to_target_orth, open_ray[:, :, 4:].reshape(batch, 64, 28))
      xray_diag = self._to_targets(self.to_target_diag, beyond[:, :, :4].reshape(batch, 64, 28))
      xray_orth = self._to_targets(self.to_target_orth, beyond[:, :, 4:].reshape(batch, 64, 28))
      first = along * open_ray       # the first blocker itself, float [B, 64, 8, 7]

      attacks_us = self._attacks(us, reach_diag, reach_orth, self.pawn_up)
      attacks_them = self._attacks(them, reach_diag, reach_orth, self.pawn_down)
      attacks_us_f, attacks_them_f = attacks_us.float(), attacks_them.float()
      channels = [
        attacks_us,
        attacks_them,
        attacks_us & occupied_us[:, None, :],
        attacks_them & occupied_them[:, None, :],
        (diagonal_us[:, :, None] & xray_diag) | (orthogonal_us[:, :, None] & xray_orth),
        (diagonal_them[:, :, None] & xray_diag) | (orthogonal_them[:, :, None] & xray_orth),
        self._pins(occupied, first, diagonal_us, orthogonal_us),
        self._pins(occupied, first, diagonal_them, orthogonal_them),
        # King rings: dilate the attack rows by the ring table (exact small integer counts), keep the king columns.
        (torch.matmul(attacks_us_f, self.king_f) > 0.5) & king_them[:, None, :],
        (torch.matmul(attacks_them_f, self.king_f) > 0.5) & king_us[:, None, :],
        (torch.matmul(attacks_us_f, self.ring2_f) > 0.5) & king_them[:, None, :],
        (torch.matmul(attacks_them_f, self.ring2_f) > 0.5) & king_us[:, None, :],
        occupied[:, :, None] & self.king & (king_us | king_them)[:, None, :],
      ]
      if previous is None:
        nothing = torch.zeros_like(attacks_us)
        channels += [attacks_us, attacks_them, nothing, nothing]
      else:
        previous_us, previous_them = self._frame_attacks(previous.bool())
        channels += [attacks_us & ~previous_us, attacks_them & ~previous_them,
                     previous_us & ~attacks_us, previous_them & ~attacks_them]
      base = torch.stack(channels, dim=1)                              # [B, 17, 64, 64]
      return torch.cat((base, base.transpose(-1, -2)), dim=1)


def relative_index() -> torch.Tensor:
  """long [64, 64]: relidx[i, j] = 15 * (row_i - row_j + 7) + (col_i - col_j + 7), 225 bins."""
  squares = torch.arange(64)
  row, column = squares // 8, squares % 8
  return 15 * (row[:, None] - row[None, :] + 7) + (column[:, None] - column[None, :] + 7)
