# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""HL-Gauss CHILD-Q DISTRIBUTION head (2026-10-07; port of Kovax' `childq_dist_head.py` + `ChildQDistLoss`).

Per stored child move of a v8 record, a K-bin histogram over that child's q (root side-to-move frame) on the atom grid
linspace(-1, 1, K). Supervised by an HL-Gauss target; gives the trunk a per-legal-move value signal instead of one number
per position. Training-only: the stash is filled under self.training and never enters the export graph.

Head (reads the TRUNK output [B, 64, C], gradient flows into the trunk = Kovax 'attach: true'):
  t = act(tokens(x))                      own embedding, never shared with a served head
  q = Wq t, k = Wk t                      [B, 64, d]
  logit[from,to,b] = sum_d q[from,d] * w_bins[d,b] * k[to,d] / sqrt(d)
  promotion p in {q,r,b}: + k[to] @ (w_promo[:, p] + w_promo[:, 3])   (knight promotion = the plain from-to entry, lc0)
  + mean_sq(t) @ pos_w + pos_b            position bias
w_bins, w_promo, pos_w, pos_b are ZERO-INIT => uniform histogram at step 0. The caller builds the module under a fixed
seed in a forked RNG, so every pre-existing parameter is bit-identical with and without the head.

Only the stored children are scored (gather from/to per child_idx), not the full [64, 64, K] tensor.

Loss per position (Kovax LF:2953-3017): c_k = n_k / (n_k + kappa) over valid children, CE(softmax(logit_k), target_k),
sum_k c_k * CE_k / max(sum_k c_k, 1e-6), then mean over the positions that have any valid child.
  abs target: t = clip(q, -1, 1)
  gap target: t = clip(q - max_valid q, -2, 0) + 1
Target = HL-Gauss on the atom grid (outer cells run to +-inf), sigma = sigma_ratio * spacing, blended toward two-hot as
|t| -> 1 (alpha = clip((1 - |t|) / taper, 0, 1)), renormalised.
"""
import math
import torch
import torch.nn as nn

from lc0_moves_1858 import FROM_1858, TO_1858, MOVES_1858

# lc0 1858 encoding: the plain from-to entry on a 7th->8th rank pawn move IS the knight promotion; q/r/b carry the
# suffix (MOVES_1858 'a7a8q', 'a7a8r', 'a7a8b'). Same convention as lc0's attention policy (q/r/b = base + offset).
_PROMO_CODE = {'q': 0, 'r': 1, 'b': 2}


def move_tables_1858():
  """(from_sq, to_sq, promo) int64 [1858]; promo = 0/1/2 for a q/r/b promotion, -1 otherwise (incl. knight promotion)."""
  frm = torch.tensor(FROM_1858, dtype=torch.int64)
  to = torch.tensor(TO_1858, dtype=torch.int64)
  promo = torch.tensor([_PROMO_CODE.get(m[4], -1) if len(m) == 5 else -1 for m in MOVES_1858], dtype=torch.int64)
  return frm, to, promo


def hlgauss_atom_target(t, num_bins: int, sigma_ratio: float = 0.75, edge_taper: float = 0.1875):
  """[..., K] HL-Gauss target on the atom grid linspace(-1,1,K) for scalar targets t (any shape)."""
  t = t.float()
  spacing = 2.0 / (num_bins - 1)
  sigma = sigma_ratio * spacing
  support = torch.linspace(-1.0, 1.0, num_bins, device=t.device)
  mid = (support[:-1] + support[1:]) / 2.0
  cdf_mid = torch.special.ndtr((mid - t.unsqueeze(-1)) / sigma)
  cdf = torch.cat([torch.zeros_like(cdf_mid[..., :1]), cdf_mid, torch.ones_like(cdf_mid[..., :1])], dim=-1)
  target = cdf[..., 1:] - cdf[..., :-1]
  if edge_taper > 0.0:
    pos = ((t + 1.0) / spacing).clamp(0.0, float(num_bins - 1))
    lo = pos.floor().long().clamp(0, num_bins - 2)
    frac = (pos - lo.float()).unsqueeze(-1)
    two_hot = (torch.nn.functional.one_hot(lo, num_bins).float() * (1.0 - frac)
               + torch.nn.functional.one_hot(lo + 1, num_bins).float() * frac)
    alpha = ((1.0 - t.abs()) / edge_taper).clamp(0.0, 1.0).unsqueeze(-1)
    target = alpha * target + (1.0 - alpha) * two_hot
  return target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class ChildQDistHead(nn.Module):
  def __init__(self, in_dim: int, embed_dim: int, d_model: int, num_bins: int, activation, position_bias: bool = True):
    super().__init__()
    if num_bins < 3 or num_bins % 2 == 0:
      raise ValueError(f'ChildQDistBins must be odd and >= 3 (got {num_bins}): with K even, q = 0 falls on a cell boundary')
    self.num_bins = num_bins
    self.d_model = d_model
    self.activation = activation
    self.tokens = nn.Linear(in_dim, embed_dim)
    self.q = nn.Linear(embed_dim, d_model)
    self.k = nn.Linear(embed_dim, d_model)
    self.w_bins = nn.Parameter(torch.zeros(d_model, num_bins))
    self.w_promo = nn.Parameter(torch.zeros(d_model, 4, num_bins))
    self.pos_w = nn.Parameter(torch.zeros(embed_dim, num_bins)) if position_bias else None
    self.pos_b = nn.Parameter(torch.zeros(num_bins)) if position_bias else None

  def forward(self, x):
    """trunk [B, 64, C] -> stash (q [B,64,d], k [B,64,d], pos [B,K] or None), float32."""
    t = self.activation(self.tokens(x))
    q = self.q(t).float()
    k = self.k(t).float()
    with torch.autocast(device_type=x.device.type, enabled=False):
      pos = (t.float().mean(dim=1) @ self.pos_w.float() + self.pos_b.float()) if self.pos_w is not None else None
    return q, k, pos

  def child_logits(self, stash, child_idx, tables):
    """[B, S, K] logits for the moves in child_idx [B, S] (1858 space, -1 = none; those rows are garbage, mask them)."""
    with torch.autocast(device_type=stash[0].device.type, enabled=False):
      return self._child_logits_fp32(stash, child_idx, tables)

  def _child_logits_fp32(self, stash, child_idx, tables):
    q, k, pos = stash
    frm_t, to_t, promo_t = tables
    idx = child_idx.clamp_min(0)
    frm, to, promo = frm_t[idx], to_t[idx], promo_t[idx]                       # [B, S]
    qf = torch.gather(q, 1, frm.unsqueeze(-1).expand(-1, -1, q.shape[-1]))     # [B, S, d]
    kt = torch.gather(k, 1, to.unsqueeze(-1).expand(-1, -1, k.shape[-1]))      # [B, S, d]
    logits = ((qf * kt) @ self.w_bins.float()) / math.sqrt(self.d_model)               # [B, S, K]
    # q/r/b promotions: + k[to] @ (w_promo[:, p] + w_promo[:, 3])
    w_off = (self.w_promo[:, :3, :] + self.w_promo[:, 3:4, :]).float()                   # [d, 3, K]
    off = torch.einsum('bsd,dpk->bspk', kt, w_off)                              # [B, S, 3, K]
    off = torch.gather(off, 2, promo.clamp_min(0)[..., None, None].expand(-1, -1, 1, self.num_bins)).squeeze(2)
    logits = logits + torch.where((promo >= 0).unsqueeze(-1), off, torch.zeros_like(off))
    if pos is not None:
      logits = logits + pos.unsqueeze(1)
    return logits

  def participation(self, stash):
    """0 * sum over every stash tensor (DDP static graph: batches without a child table must still touch the head)."""
    q, k, pos = stash
    # the readout tables are only reached through child_logits, so touch them explicitly (review 2026-10-07: without
    # this w_bins/w_promo get no grad on child-less batches and DDP static_graph hangs on a mixed cv4+T91 run)
    s = q.sum() + k.sum() + (pos.sum() if pos is not None else 0.0) + self.w_bins.sum() + self.w_promo.sum()
    return 0.0 * s


def childq_dist_loss(logits, child_idx, child_q, child_n, gap: bool, kappa: float = 2.0,
                     sigma_ratio: float = 0.75, edge_taper: float = 0.1875):
  """Per-position visit-weighted HL-Gauss CE over the valid children, averaged over positions with any valid child.
  Returns (loss, diag) where loss is CE MINUS target entropy (a true KL, same gradient as CE) per the file convention."""
  valid = child_idx >= 0
  K = logits.shape[-1]
  q = torch.where(valid, child_q.float(), torch.zeros_like(child_q, dtype=torch.float32))
  if gap:
    q0 = torch.where(valid, q, torch.full_like(q, -float('inf'))).max(dim=1, keepdim=True).values
    q0 = torch.where(torch.isfinite(q0), q0, torch.zeros_like(q0))
    tq = (q - q0).clamp(-2.0, 0.0) + 1.0
  else:
    tq = q.clamp(-1.0, 1.0)
  with torch.no_grad():
    target = hlgauss_atom_target(tq, K, sigma_ratio, edge_taper)               # [B, S, K]
  lp = torch.log_softmax(logits.float(), dim=-1)
  tc = target.clamp_min(1e-12)
  kl = (target * (tc.log() - lp)).sum(dim=-1)                                  # [B, S]
  n = child_n.float().clamp_min(0.0)
  c = torch.where(valid, n / (n + kappa), torch.zeros_like(n))
  c_sum = c.sum(dim=1)
  per_pos = (c * kl).sum(dim=1) / c_sum.clamp_min(1e-6)
  has = (c_sum > 0).float()
  n_has = has.sum().clamp_min(1.0)
  # rows without a valid child contribute exactly 0 (per_pos is 0 there); no host sync
  loss = (per_pos * has).sum() / n_has
  with torch.no_grad():
    support = torch.linspace(-1.0, 1.0, K, device=logits.device)
    p = lp.exp()
    mean = (p * support).sum(-1)
    var = ((p * support * support).sum(-1) - mean * mean).clamp_min(0.0)
    cs = c_sum.clamp_min(1e-6)
    diag = {'mae': (((c * (mean - tq).abs()).sum(1) / cs) * has).sum() / n_has,
            'sd': (((c * var.sqrt()).sum(1) / cs) * has).sum() / n_has,
            'rows': has.mean()}
  return loss, diag
