# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""MOVE-TOKEN DECODER (design B, 2026-09-02) — an action-centric read-out.

After the square trunk encodes the board, one token per candidate MOVE
(pseudo-legal from-to pair, built in-graph) is refined by a small decoder
(self-attention among moves, cross-attention to the 64 squares, FFN) and read
out directly: the policy is one logit per move token, scattered into the
1858-way vector; the value heads get a zero-init inject of the pooled move state.

Evidence (this campaign): the dual-plane knockout showed the net moved its whole
policy function into a from-to bilinear decode; a second token set with
cross-talk (the piece plane) is worth 20-40 Elo; everything that adds RELATIONS to
square attention is a substitute — so spend compute in MOVE space instead.

Candidate construction (a SUPERSET of legal moves is what matters — a missing
legal move is un-representable, a spurious one is masked by the loss/serving):
  cand[from,to] = vis_stm_out[from,to] * (1 - own_occ[to])       (blocker-exact reach,
                                                                    own-target masked)
                + double pawn push (from on rank 1, from+8 and from+16 empty)   <- reviewer
                  finding: `vis` deliberately omits it; 3.4 % of legal moves.
                + castling as KING-TAKES-ROOK pairs (the TPG/Ceres encoding: e1h1, e1a1;
                  FRC-compatible): king square -> every own rook on the king's rank
                  (castling RIGHTS are not in the input; a phantom pair is harmless).
  Selection: TopK over the 4096 pairs of  cand + tie/4096*0.5  (tie < the 0/1 gate
  by construction — the fp16 tie-break lesson from dual_plane.py), M = MoveTokenMax.
  The decoder is permutation-equivariant and the scatter uses gathered indices, so
  only the selected SET matters (order flips under fp16 are harmless).

Promotions: 22 pairs carry 4 policy indices (bare = knight / rook lift, q, r, b).
The token head emits 4 logits per token and a constant [1858] promotion-slot
buffer selects the right one, so underpromotions are representable.

Absent moves (no token) get a floor of -30: exp(-30) ~ 1e-13 at serving, and a
missing legal move costs at most 30*p in the CE instead of destroying the logged
policy loss (a -1e4 floor would).

Everything is topk / gather / scatter / matmul / softmax (TRT-static, fixed M).
No per-sample weight matmuls (the dual_plane rel_gains 46x-slowdown class).
"""

import torch
from torch import nn
from rms_norm import make_norm
from chess_geometry import VisibilityChannels
from lc0_moves_1858 import MOVES_1858, FROM_1858, TO_1858

MT_FLOOR = -30.0


def _build_move_tables():
  """Constant tables: flat from*64+to per policy index, promotion slot per index
  (0 = bare/knight, 1 = q, 2 = r, 3 = b), and the [4096] from/to lookup."""
  pair_flat = [f * 64 + t for f, t in zip(FROM_1858, TO_1858)]
  slot = []
  for m in MOVES_1858:
    slot.append({'q': 1, 'r': 2, 'b': 3}.get(m[4], 0) if len(m) == 5 else 0)
  pf = torch.tensor(pair_flat, dtype=torch.long)
  sl = torch.tensor(slot, dtype=torch.long)
  return pf, sl, pf * 4 + sl


class MoveTokenBlock(nn.Module):
  """Pre-norm block: masked self-attention among move tokens, cross-attention
  from moves to the 64 square states, SwiGLU FFN. Plain matmul/softmax."""

  def __init__(self, dm: int, s_dim: int, heads: int, ffn_mult: int, norm_type: str):
    super().__init__()
    assert dm % heads == 0
    self.h, self.dk = heads, dm // heads
    self.ln1 = make_norm(norm_type, dm)
    self.qkv = nn.Linear(dm, 3 * dm, bias=False)
    self.proj = nn.Linear(dm, dm, bias=False)
    self.ln2 = make_norm(norm_type, dm)
    self.ln_s = make_norm(norm_type, s_dim)
    self.xq = nn.Linear(dm, dm, bias=False)
    self.xkv = nn.Linear(s_dim, 2 * dm, bias=False)
    self.xproj = nn.Linear(dm, dm, bias=False)
    self.ln3 = make_norm(norm_type, dm)
    self.ffn_in = nn.Linear(dm, 2 * ffn_mult * dm, bias=False)     # SwiGLU: gate | up
    self.ffn_out = nn.Linear(ffn_mult * dm, dm, bias=False)

  def _attn(self, q, k, v, key_bias=None):
    B, Tq = q.shape[0], q.shape[1]
    Tk = k.shape[1]
    q = q.reshape(B, Tq, self.h, self.dk).transpose(1, 2)
    k = k.reshape(B, Tk, self.h, self.dk).transpose(1, 2)
    v = v.reshape(B, Tk, self.h, self.dk).transpose(1, 2)
    s = torch.matmul(q, k.transpose(2, 3)) * (self.dk ** -0.5)
    if key_bias is not None:
      s = s + key_bias.to(s.dtype)                     # [B,1,1,Tk]: -1e4 on padded keys
    a = torch.softmax(s, dim=-1)
    return torch.matmul(a, v).transpose(1, 2).reshape(B, Tq, self.h * self.dk)

  def forward(self, x, s_flow_n, key_bias):
    h = self.ln1(x)
    q, k, v = self.qkv(h).chunk(3, dim=-1)
    x = x + self.proj(self._attn(q, k, v, key_bias))
    h = self.ln2(x)
    k2, v2 = self.xkv(s_flow_n).chunk(2, dim=-1)
    x = x + self.xproj(self._attn(self.xq(h), k2, v2))
    h = self.ln3(x)
    g, u = self.ffn_in(h).chunk(2, dim=-1)
    x = x + self.ffn_out(torch.nn.functional.silu(g) * u)
    return x


class MoveTokenDecoder(nn.Module):
  def __init__(self, s_dim: int, norm_type: str, dm: int = 160, layers: int = 3,
               heads: int = 4, ffn_mult: int = 2, max_tokens: int = 128,
               value_inject_dim: int = 0, value2: bool = False, pol_bias: bool = True):
    super().__init__()
    self.dm, self.M = dm, max_tokens
    # Candidate source: private visibility module ('vis' family only). Its
    # construction runs a random probe -> fork_rng keeps the global stream
    # untouched (bit-pairing with the control; the edge-aux lesson).
    with torch.random.fork_rng(devices=[]):
      self.vis = VisibilityChannels(families=('vis',))
    pf, sl, pfs = _build_move_tables()
    self.register_buffer('mv_pair_flat', pf, persistent=False)      # [1858] from*64+to
    self.register_buffer('mv_pair_slot', pfs, persistent=False)     # [1858] (from*64+to)*4+slot
    ar = torch.arange(4096, dtype=torch.long)
    self.register_buffer('pair_from', ar // 64, persistent=False)   # [4096]
    self.register_buffer('pair_to', ar % 64, persistent=False)
    # Deterministic tie-break: kept as an INT buffer and turned into fp32 at use
    # time (review B3: a float buffer is downcast under BFloat16Pure and ~4000 of
    # the 4096 tie levels collapse; fp32 ulp at 1.0 is 1.2e-7, so a 1e-4 step is
    # safe there). Max 0.5 < the 0/1 candidate gate.
    self.register_buffer('tie_rank', ar, persistent=False)
    # CASTLING = KING-TAKES-ROOK in the TPG/Ceres move encoding (e1h1 / e1a1, FRC-
    # style), NOT e1g1/e1c1 — measured on real T91 data 2026-09-02: 100 % of the
    # 1.6 % missing-target cases were king(ch 6) -> own rook(ch 4). Candidate
    # pairs: king square -> every own rook on the king's rank (superset; covers
    # Chess960). Constant same-rank table, Mul+Add only.
    sr = torch.zeros(64, 64)
    for k in range(64):
      for r in range(64):
        if k // 8 == r // 8:
          sr[k, r] = 1.0
    self.register_buffer('same_rank', sr, persistent=False)
    # Double pawn push table: from on rank 1 (squares 8..15) -> from+16.
    dbl = torch.zeros(64, 64)
    for f in range(8, 16):
      dbl[f, f + 16] = 1.0
    self.register_buffer('dbl_push', dbl, persistent=False)
    self.register_buffer('rank1', (ar[:64] // 8 == 1).float(), persistent=False)   # [64]
    mid = torch.arange(64, dtype=torch.long); mid = torch.where(mid < 48, mid + 8, mid)
    self.register_buffer('push_mid', mid, persistent=False)         # from -> from+8 (clamped)
    # Castling pairs (stm-relative: king e1=4, rooks h1=7 / a1=0).
    self.register_buffer('castle_pair', torch.tensor([4 * 64 + 6, 4 * 64 + 2], dtype=torch.long),
                         persistent=False)
    # Token init: concat(flow[from], flow[to], 4 vis channels of the pair).
    self.w_in = nn.Linear(2 * s_dim + 4, dm)
    self.blocks = nn.ModuleList([MoveTokenBlock(dm, s_dim, heads, ffn_mult, norm_type)
                                 for _ in range(layers)])
    self.out_ln = make_norm(norm_type, dm)
    # Policy read: 4 logits per token (promotion slots). SMALL fixed-key init,
    # not zero: with both readers (pol, value inject) at zero the decoder body
    # receives no gradient at all at step 0 — the product-rule cascade this
    # campaign diagnosed (caught by test_move_tokens.py). The magnitude
    # trajectory of pol.weight remains the does-the-net-want-this diagnostic.
    self.pol = nn.Linear(dm, 4, bias=False)
    with torch.no_grad():
      self.pol.weight.uniform_(-0.02, 0.02, generator=torch.Generator().manual_seed(0x0B0B))
    # Per-move bias table (1858). pol_bias=False keeps it as a frozen zero BUFFER so the
    # graph/diagnostics are unchanged and the token features must carry the whole logit.
    if pol_bias:
      self.mt_pol_bias = nn.Parameter(torch.zeros(1858))
    else:
      self.register_buffer('mt_pol_bias', torch.zeros(1858), persistent=False)
    # Value inject (zero-init, separate per head — the dp_value_inject pattern).
    self.value_inject_dim = value_inject_dim
    if value_inject_dim > 0:
      self.v_inject = nn.Linear(2 * dm, value_inject_dim, bias=False)
      nn.init.zeros_(self.v_inject.weight)
      if value2:
        self.v2_inject = nn.Linear(2 * dm, value_inject_dim, bias=False)
        nn.init.zeros_(self.v2_inject.weight)

  def candidates(self, squares13):
    """squares13 [B,64,13] one-hot -> (cand [B,4096] in {0,1}, E [B,64,64,4])."""
    E = self.vis(squares13)                                   # [B,64,64,4]: stm_out, opp_out, stm_in, opp_in
    dt = E.dtype
    own = squares13[:, :, 1:7].sum(dim=2).clamp(max=1.0).to(dt)             # [B,64]
    empty = squares13[:, :, 0].to(dt)
    cand = E[..., 0] * (1.0 - own).unsqueeze(1)                             # own-target masked
    pawn = squares13[:, :, 1].to(dt)
    mid_empty = torch.gather(empty, 1, self.push_mid.unsqueeze(0).expand(empty.shape[0], -1))
    dbl_from = pawn * self.rank1.to(dt).unsqueeze(0) * mid_empty            # [B,64]
    cand = cand + dbl_from.unsqueeze(2) * self.dbl_push.to(dt).unsqueeze(0) * empty.unsqueeze(1)
    # Castling as king-takes-rook: cand[k, r] += king[k] * rook[r] * same_rank[k, r].
    king = squares13[:, :, 6].to(dt)                                        # [B,64]
    rook = squares13[:, :, 4].to(dt)
    cand = cand + king.unsqueeze(2) * rook.unsqueeze(1) * self.same_rank.to(dt).unsqueeze(0)
    return cand.reshape(-1, 4096).clamp(max=1.0), E

  def forward(self, squares13, flow):
    """flow [B,64,S] (post trunk norm). Returns (policy_add [B,1858], pooled [B,2dm],
    stats dict, sel [B,M], valid [B,M])."""
    B, S = flow.shape[0], flow.shape[2]
    cand, E = self.candidates(squares13)
    score = cand.float() + (self.tie_rank.float() * (0.5 / 4096.0)).unsqueeze(0)
    sel = torch.topk(score, self.M, dim=1).indices                          # [B,M]
    valid = torch.gather(cand, 1, sel) > 0.5                                 # [B,M]
    fr = torch.gather(self.pair_from.unsqueeze(0).expand(B, -1), 1, sel)     # [B,M]
    to = torch.gather(self.pair_to.unsqueeze(0).expand(B, -1), 1, sel)
    f_from = torch.gather(flow, 1, fr.unsqueeze(-1).expand(-1, -1, S))
    f_to = torch.gather(flow, 1, to.unsqueeze(-1).expand(-1, -1, S))
    e_pair = torch.gather(E.reshape(B, 4096, 4), 1, sel.unsqueeze(-1).expand(-1, -1, 4)).to(flow.dtype)
    x = self.w_in(torch.cat([f_from, f_to, e_pair], dim=-1))                # [B,M,dm]
    key_bias = ((~valid).to(flow.dtype) * -1e4).reshape(B, 1, 1, self.M)
    for blk in self.blocks:
      x = blk(x, blk.ln_s(flow), key_bias)      # each block re-norms the squares under ITS norm (review B1)
    xo = self.out_ln(x)
    # Policy: 4 slot logits per token -> [B,4096,4] buffer at floor -> flat index_select.
    tok = self.pol(xo)                                                       # [B,M,4]
    tok = torch.where(valid.unsqueeze(-1), tok, torch.full_like(tok, MT_FLOOR))
    buf = torch.full((B, 4096, 4), MT_FLOOR, device=tok.device, dtype=tok.dtype)
    buf = buf.scatter(1, sel.unsqueeze(-1).expand(-1, -1, 4), tok)
    pol = buf.reshape(B, 16384).index_select(1, self.mv_pair_slot) + self.mt_pol_bias.to(tok.dtype)
    # Pools over VALID tokens (mean + max).
    w = valid.to(xo.dtype).unsqueeze(-1)
    n_valid = w.sum(dim=1)                                                   # [B,1]
    mean_pool = (xo * w).sum(dim=1) / n_valid.clamp_min(1.0)
    # Max-pool guarded for the no-candidate case (mate/stalemate): without the
    # guard it would be ~-1e4 in every channel and feed the value head (review B2).
    max_pool = torch.where(n_valid > 0, (xo + (w - 1.0) * 1e4).amax(dim=1), torch.zeros_like(mean_pool))
    pooled = torch.cat([mean_pool, max_pool], dim=-1)
    n_cand = cand.float().sum(dim=1)
    stats = {'mt_count_mean': n_cand.mean(), 'mt_count_max': n_cand.amax(),
             'mt_trunc_rate': (n_cand > self.M).float().mean()}
    return pol, pooled, stats, sel, valid
