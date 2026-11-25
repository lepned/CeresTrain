"""BT6's relation / pair / edge term reads on one of OUR encoder layers (config BT6NativeLayers; bt6_hybrid.py).

The NBT-BT6 hybrid replaces the inner layers by bt6_net.BT6Layer (LayerNorm, QKVG output gate, sigmoid GLU). This
module is the other arrangement: the inner layers stay our own EncoderLayer / DotProductAttention (RMSNorm, SwiGLU
FFN, shared smolgen bias), and the layers at the positions in BT6TermsInnerLayers get BT6's term reads on their
attention logits -- nothing else of the layer changes. With Q, K [B, heads, 64, hd] the layer's own projections and
`bias` every per-head bias the layer already adds (shared smolgen, ...):

    static = Q.K / sqrt(hd) + (Q.Pq + C1).E + (K.Pk).E + C2.Z0          (relation terms, pair-stream read)
    L      = t_node * static + t_edge * (We.e) + bias                    (edge read; our bias, unscaled)
    H      = L * (1 + Wm.e)                                              (door)
    A      = softmax_j(H) * 2 sigmoid(Wg.e + bg)                         (post-softmax gate)
    out    = A @ V

which is bt6_net.BT6Layer's attention (forward_reference, term by term) without the output gate. At initialisation
t_node = 1, t_edge = 0 and every table but We is zero: door 0, gate 1, L = Q.K / sqrt(hd) + bias -- the layer is
bit-for-bit our plain layer in function, and the terms grow from there.

Parameter names end in the BT6 reader suffixes (.t_node, .C1, .Pk, ...) under `...inner.<j>.attention.bt6t`, so the
trainer's name rules for the hybrid apply unchanged: never weight-decayed, never under Muon. QK-clip
(DotProductAttention.apply_qk_clip) scales Pq / Pk with the head's Q / K rows, as BT6Layer.apply_qk_clip does.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from bt6_net import BASE_RELATION_CHANNELS, BT6Config, RelationContext, _make_reader_parameters, attention_core, stacked_reads


class BT6NativeTerms(nn.Module):
  """The ten reader tensors of one BT6 layer, for one DotProductAttention with cfg.heads heads of cfg.head_dim."""

  def __init__(self, cfg: BT6Config) -> None:
    super().__init__()
    self.cfg = cfg
    _make_reader_parameters(self, cfg)       # t_node, t_edge, C1, C2, We, Wm, Wg, bg (We is the only random draw)
    self.Ue = None                           # BT6EdgeValues: edge message projection (enable_edge_values)
    self.Pk = nn.Parameter(torch.zeros(cfg.heads, cfg.relation_channels, cfg.head_dim))   # (k_j . Pk[h, c]) E[c, i, j]
    self.Pq = nn.Parameter(torch.zeros(cfg.heads, cfg.relation_channels, cfg.head_dim))   # (q_i . Pq[h, c]) E[c, i, j]

  def logits(self, Q: torch.Tensor, K: torch.Tensor, ctx: RelationContext, e: torch.Tensor, read_state: torch.Tensor):
    """Q, K [B, heads, 64, hd] -> (L without our bias [B, heads, 64, 64], door, gate logits). L and door are in the
    dtype of `read_state` (the assembly dtype the trunk chose: the autocast dtype or float32), the gate logits as
    stacked_reads returns them (float32 unless cfg.pair_autocast)."""
    cfg = self.cfg
    batch, heads, hd = Q.shape[0], cfg.heads, cfg.head_dim
    pd = read_state.dtype
    t = self.t_node.float()
    Qp, Kp = Q.to(pd), K.to(pd)
    # relation terms: per (b, h) the [64, channels] coefficient rows, then per (b, square) a matmul with E
    #   query side -> [b, i, h, j];  key side (channel halves rolled: E[17 + c, i, j] = E[c, j, i]) -> [b, j, h, i]
    Pq = (self.Pq.float() * t[:, None, None]).to(pd).transpose(1, 2)                                       # [h, hd, c]
    Pk = (self.Pk.float() * t[:, None, None]).roll(BASE_RELATION_CHANNELS, 1).to(pd).transpose(1, 2)
    rows = torch.matmul(Qp, Pq[None]) + (self.C1.float() * t[:, None]).to(pd)[None, :, None, :]            # [b, h, 64, c]
    keys = torch.matmul(Kp, Pk[None])
    Ei = ctx.Ei(pd).view(batch * 64, -1, 64)
    rel_q = torch.bmm(rows.permute(0, 2, 1, 3).reshape(batch * 64, heads, -1), Ei).view(batch, 64, heads, 64)
    rel_k = torch.bmm(keys.permute(0, 2, 1, 3).reshape(batch * 64, heads, -1), Ei).view(batch, 64, heads, 64)
    # edge read + pair-stream read (already scaled by t_edge / t_node), door, gate logits: one matmul on [e | Z0]
    read, door, gate_logits = stacked_reads([self], cfg, e, read_state)[0]
    scale = (t / math.sqrt(hd)).to(pd).view(1, heads, 1, 1)
    L = torch.matmul(Qp, Kp.transpose(-1, -2)) * scale + read
    L = L + rel_q.permute(0, 2, 1, 3) + rel_k.permute(0, 2, 3, 1)
    return L, door, gate_logits

  def enable_edge_values(self) -> None:
    """BT6EdgeValues (2026-11-25): the edge state also carries CONTENT, not only the weights. Per head h and query
    square i the edge message m[h, i] = sum_j A[h, i, j] e[:, i, j] (the attention-weighted edge state of the pairs
    it reads, C = edge channels) is projected to the head width and added to A @ V:
        out[h, i] = sum_j A[h, i, j] v[h, j] + m[h, i] Ue[h]          Ue [heads, C, head_dim], zero at init
    so the layer is unchanged at initialisation. Without it the relations steer only where a square looks (logits,
    door, gate); "j attacks i" can reach i only through the choice of j. Ue is a reader table (no weight decay,
    no Muon) and follows QK-clip like nothing else (it is on the value side)."""
    self.Ue = nn.Parameter(torch.zeros(self.cfg.heads, self.cfg.edge_channels, self.cfg.head_dim))

  def attend(self, Q, K, V, bias, ctx: RelationContext, e: torch.Tensor, read_state: torch.Tensor):
    """The whole attention of a term layer: (A @ V [B, heads, 64, hd] in V's dtype, H [B, heads, 64, 64]); with
    BT6EdgeValues the edge message m Ue is added to A @ V."""
    L, door, gate_logits = self.logits(Q, K, ctx, e, read_state)
    if bias is not None:
      L = L + bias.to(L.dtype)
    cfg = self.cfg
    low = cfg.pair_autocast and L.dtype in (torch.bfloat16, torch.float16)
    if self.Ue is None:
      return attention_core(L, door, gate_logits, V, cfg.gate_cap, None, cfg.output_gate_scale, low=low)
    edge = ctx.edge_channels_last(e, V.dtype)                      # channels-last [B, 64 i, 64 j, C], one per state
    out, H, m = attention_core(L, door, gate_logits, V, cfg.gate_cap, None, cfg.output_gate_scale, low=low, edge=edge)
    return out + torch.matmul(m, self.Ue.to(m.dtype)[None]), H

  @torch.no_grad()
  def scale_heads(self, s: torch.Tensor) -> None:
    """QK-clip: the q/k-dependent relation terms shrink with the head's Q / K rows (s = sqrt(gamma) per head)."""
    self.Pq.mul_(s.to(self.Pq.dtype)[:, None, None])
    self.Pk.mul_(s.to(self.Pk.dtype)[:, None, None])
