"""Piece tokens (2026-11-07, config PieceTokens / PieceTokenDecompressDim; NBT trunks only).

The trunk runs on K (32) of the 64 squares instead of all of them: every piece always gets a token (32 is the most
pieces a board can hold), the free slots go to squares that were occupied on the previous board (the last move's
from-square), then to the remaining empty squares by index. Square identity needs nothing new: the rank/file one-hot
bytes [121:137] are part of every square's 137 input features. The trunk is permutation-equivariant (the shared
smolgen is scattered back to the 64-square layout and its bias gathered at the tokens, nbt_layer.py), so token order
is irrelevant and no sort is needed.

After the trunk, PieceTokenDecompress rebuilds [B, 64, D] for the heads and the move-token decoder (which reads
flow[from] and flow[to], and the to-square is usually empty): every square queries the K trunk outputs with one
narrow cross-attention on top of its post-embedding state; the token squares then take their trunk output.
Everything is topk / gather / scatter / matmul / softmax at fixed K (TRT-static), like the move-token decoder.
"""
import math
import torch

from rms_norm import make_norm
from nbt_layer import gather_tokens, scatter_tokens

# Selection score steps (fp32): occupied now 1.0, occupied on the previous board 0.5, square index tie-break
# 2e-3 per square (max 0.126 < 0.5; > the fp16 ulp at 1.5, same lesson as dual_plane.py's slot selection).
_W_NOW, _W_PREV, _TIE = 1.0, 0.5, 2e-3


def select_piece_tokens(squares: torch.Tensor, k: int) -> torch.Tensor:
  """squares [B, 64, >=26] raw input (bytes 0..12 = current board one-hot with 0 = empty, 13..25 = previous board)
  -> [B, k] square indices: all pieces first, then the previous board's pieces, then empty squares."""
  occ_now = 1.0 - squares[:, :, 0].float()
  occ_prev = 1.0 - squares[:, :, 13].float()
  tie = torch.arange(64, device=squares.device, dtype=torch.float32) * _TIE
  return torch.topk(occ_now * _W_NOW + occ_prev * _W_PREV + tie, k=k, dim=1).indices


class PieceTokenDecompress(torch.nn.Module):
  """Cross-attention from the 64 squares (post-embedding state) to the K trunk tokens, d wide, then up to D.
  Zero-init output: at step 0 a non-token square keeps its post-embedding state."""
  def __init__(self, model_dim: int, d: int, num_heads: int, norm_type: str, eps: float):
    super().__init__()
    assert d % num_heads == 0, f'PieceTokenDecompress: width {d} not divisible by {num_heads} heads'
    self.num_heads = num_heads
    self.d_k = d // num_heads
    self.norm_q = make_norm(norm_type, model_dim, eps=eps)
    self.norm_kv = make_norm(norm_type, model_dim, eps=eps)
    self.q = torch.nn.Linear(model_dim, d, bias=False)
    self.kv = torch.nn.Linear(model_dim, 2 * d, bias=False)
    self.out = torch.nn.Linear(d, model_dim, bias=False)
    torch.nn.init.zeros_(self.out.weight)

  def forward(self, flow64: torch.Tensor, tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """flow64 [B, 64, D] post-embedding; tokens [B, K, D] trunk output; idx [B, K] -> [B, 64, D]."""
    B, T, _ = flow64.shape
    K = tokens.shape[1]
    H, dk = self.num_heads, self.d_k
    q = self.q(self.norm_q(flow64)).reshape(B, T, H, dk).transpose(1, 2)                 # [B, H, 64, dk]
    k, v = self.kv(self.norm_kv(tokens)).reshape(B, K, H, 2 * dk).transpose(1, 2).chunk(2, dim=-1)
    att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(dk), dim=-1)    # [B, H, 64, K]
    o = torch.matmul(att, v).transpose(1, 2).reshape(B, T, H * dk)
    return scatter_tokens(flow64 + self.out(o), idx, tokens)
