# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""EGT edge stream in a TRT-servable form (2026-10-07, NetDef EGTEdgeStream; NBT trunks only).

Kovax' EGT (edge_stream.py in his lczero-training fork) keeps a learned edge state e [64, 64, d_e] that every block reads
inside attention and that a few "sites" update. Its original read (multiplicative premult on the logits and a door AFTER
the softmax) breaks TensorRT's fused attention; measured on an untrained 1920 replica that costs ~0.56x serving speed. This
module is the rewritten form measured at ~0.78x (b256) / ~0.92x (b16) with trtexec (C:/Dev/Chess/CeresTrain/egt_trt):

  state   e0 = rmsnorm(E . P_in + T_off[offset(i, j)])     E = VisibilityChannels [B,64,64,C], channels-last [B,64,64,d_e]
  read    (once per NBT block, shared by its inner layers; e is constant between sites)
            bias[h,i,j] = t_edge_h * (W_e . e_ij)_h + log(2 sigmoid((W_g . e_ij)_h + b_h))      -> additive (fusable)
            q_i *= (1 + (M_q . mean_j e_ij)_h) * t_node_h ;  k_j *= 1 + (M_k . mean_i e_ij)_h       -> separable premult
            o_i *= 2 sigmoid((M_r . mean_j e_ij)_h)                                               -> door as a row scale
  site    (after the listed blocks) e += O_e . concat_h(raw logits of the block's last inner layer)    (readback)
          e += triplet (Ag form, path einsums: i -> k -> j, gated softmax)                            (fp16-friendly)
          e += FFN_x2(rmsnorm(e));  e = rmsnorm(e)

Function-class differences vs Kovax' form (UNTESTED in training): the pairwise post-softmax door becomes a log-door inside
the softmax plus a per-row scale; the pairwise premult becomes separable (row x column) scales.

Step-0 identity: every reader of e is zero-init (W_e, W_g, b, M_q, M_k, M_r, O_e, triplet out-projection, FFN out), with
t_node = t_edge = 1, so bias = log(2 sigmoid(0)) = 0, all scales = 1 and the sites leave e untouched: the trunk computes
exactly what it computes without the stream. One trainable zero per path (Kovax' rule). Built in a forked RNG under a fixed
seed by the caller, so the rest of the net initializes bit-identically with and without the stream.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms_last(x, eps=1e-6):
  xf = x.float()
  return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)


def _offset_index():
  sq = torch.arange(64)
  r, f = sq // 8, sq % 8
  return (r[:, None] - r[None, :] + 7) * 15 + (f[:, None] - f[None, :] + 7)      # [64, 64] in 0..224


class EGTEdgeRead(nn.Module):
  def __init__(self, d_e: int, heads: int):
    super().__init__()
    self.t_node = nn.Parameter(torch.ones(heads))
    self.t_edge = nn.Parameter(torch.ones(heads))
    self.w_e = nn.Parameter(torch.zeros(d_e, heads))
    self.w_g = nn.Parameter(torch.zeros(d_e, heads))
    self.b_g = nn.Parameter(torch.zeros(heads))
    self.m_q = nn.Parameter(torch.zeros(d_e, heads))
    self.m_k = nn.Parameter(torch.zeros(d_e, heads))
    self.m_r = nn.Parameter(torch.zeros(d_e, heads))

  def forward(self, e):
    """e [B,64,64,d_e] -> (bias [B,H,64,64], (qs [B,H,64], ks [B,H,64], rs [B,H,64]))."""
    dt = e.dtype
    bias = (self.t_edge.to(dt)[None, :, None, None] * torch.einsum('bijc,ch->bhij', e, self.w_e.to(dt))
            + F.logsigmoid(torch.einsum('bijc,ch->bhij', e, self.w_g.to(dt)) + self.b_g.to(dt)[None, :, None, None])
            + math.log(2.0))
    rowm, colm = e.mean(dim=2), e.mean(dim=1)                                     # [B,64,d_e]: over j / over i
    qs = (1.0 + torch.einsum('bic,ch->bhi', rowm, self.m_q.to(dt))) * self.t_node.to(dt)[None, :, None]
    ks = 1.0 + torch.einsum('bjc,ch->bhj', colm, self.m_k.to(dt))
    rs = 2.0 * torch.sigmoid(torch.einsum('bic,ch->bhi', rowm, self.m_r.to(dt)))
    return bias, (qs, ks, rs)


class EGTEdgeSite(nn.Module):
  def __init__(self, d_e: int, heads: int, tri_heads: int, ffn_mult: int):
    super().__init__()
    assert d_e % tri_heads == 0, 'EGTEdgeDim must be divisible by EGTEdgeTripletHeads'
    self.d_e, self.th = d_e, tri_heads
    self.o_e = nn.Parameter(torch.zeros(heads, d_e))                              # readback (zero)
    self.tri_v = nn.Parameter(torch.randn(d_e, 2 * d_e) / math.sqrt(d_e))
    self.tri_eg = nn.Parameter(torch.randn(d_e, 4 * tri_heads) / math.sqrt(d_e))
    self.tri_eg_b = nn.Parameter(torch.zeros(4 * tri_heads))
    self.tri_o = nn.Parameter(torch.zeros(2 * d_e, d_e))                          # triplet out (zero)
    self.ffn_in = nn.Parameter(torch.randn(d_e, ffn_mult * d_e) / math.sqrt(d_e))
    self.ffn_in_b = nn.Parameter(torch.zeros(ffn_mult * d_e))
    self.ffn_out = nn.Parameter(torch.zeros(ffn_mult * d_e, d_e))                 # FFN out (zero)

  def forward(self, e, logits):
    """e [B,64,64,d_e], logits [B,H,64,64] (pre-softmax, the block's last inner layer) -> e'."""
    B, dt, de, th = e.shape[0], e.dtype, self.d_e, self.th
    e = e + torch.einsum('bhij,hc->bijc', logits.to(dt), self.o_e.to(dt))
    n = _rms_last(e)
    wv, weg, beg = self.tri_v.to(dt), self.tri_eg.to(dt), self.tri_eg_b.to(dt)
    # splits on the WEIGHTS (constant-folded at export), never on activations
    v_in = (n @ wv[:, :de]).reshape(B, 64, 64, th, de // th)
    v_out = (n @ wv[:, de:]).reshape(B, 64, 64, th, de // th)
    a_in = torch.softmax(n @ weg[:, 0:th] + beg[0:th], dim=2) * torch.sigmoid(n @ weg[:, th:2 * th] + beg[th:2 * th])
    a_out = torch.softmax(n @ weg[:, 2 * th:3 * th] + beg[2 * th:3 * th], dim=1) * torch.sigmoid(n @ weg[:, 3 * th:] + beg[3 * th:])
    va_in = torch.einsum('bikh,bkjhd->bijhd', a_in, v_in).reshape(B, 64, 64, de)     # path i -> k -> j
    va_out = torch.einsum('bkih,bjkhd->bijhd', a_out, v_out).reshape(B, 64, 64, de)
    wo = self.tri_o.to(dt)
    e = e + va_in @ wo[:de] + va_out @ wo[de:]
    h = torch.relu(_rms_last(e) @ self.ffn_in.to(dt) + self.ffn_in_b.to(dt))
    e = e + h @ self.ffn_out.to(dt)
    return _rms_last(e)


class EGTEdgeStream(nn.Module):
  def __init__(self, num_channels: int, num_blocks: int, heads: int, d_e: int, sites, tri_heads: int, ffn_mult: int):
    super().__init__()
    self.sites = tuple(int(s) for s in sites)
    # a site after the LAST block would produce an update nobody reads: its params never get a gradient (breaks DDP
    # without static_graph and wastes the triplet) -- refused (review 2026-10-07)
    assert all(0 <= s < num_blocks - 1 for s in self.sites),       f'EGTEdgeSites {self.sites} must lie in 0..{num_blocks - 2} (a site after the last block is never read)'
    self.p_in = nn.Parameter(torch.randn(num_channels, d_e) / math.sqrt(num_channels))
    self.t_off = nn.Parameter(torch.randn(225, d_e) * 0.5)
    self.register_buffer('off_idx', _offset_index(), persistent=False)
    self.reads = nn.ModuleList([EGTEdgeRead(d_e, heads) for _ in range(num_blocks)])
    self.site_mods = nn.ModuleDict({str(s): EGTEdgeSite(d_e, heads, tri_heads, ffn_mult) for s in self.sites})

  def init_state(self, E):
    """E [B,64,64,C] visibility channels -> e [B,64,64,d_e]."""
    dt = E.dtype
    return _rms_last(E @ self.p_in.to(dt) + self.t_off[self.off_idx].to(dt)[None])

  def read(self, block: int, e):
    return self.reads[block](e)

  def is_site(self, block: int) -> bool:
    return block in self.sites

  def update(self, block: int, e, logits):
    return self.site_mods[str(block)](e, logits)
