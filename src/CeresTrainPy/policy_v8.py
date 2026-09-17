# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""Policy-loss reshaping from the LC0 v8 child table (2026-09-17). Three data-side mechanisms, all default-off,
all loss/target-only (the serving graph is untouched). The child table arrives per batch as child_idx [B,S] (1858-space
move index, -1 = unused), child_q / child_prior [B,S] (root frame; prior = the generating net's own policy), child_n [B,S]
(raw root visits). Batches without a child table (the puzzle secondary of a mixed run) are left exactly as before.

  qgap_weights      "only-move" weighting: rows whose best and second-best searched child differ a lot in q are the
                    decisive ones; weight = 1 + lam * clip(q_best - q_2nd, 0, cap), renormalised to mean 1 per batch so
                    the loss SCALE (effective policy LR) is unchanged and only the gradient is redistributed.
  surprise_weights  weight by how much the search overturned the net: KL(search || prior) over the visited children,
                    divided by a reference (the corpus median, ~0.07 on cv4), clipped to [lo, hi], mean-1 per batch.
  q_improved_target Gumbel-style completed-Q sharpening of the policy target: target' ∝ target * exp(beta * (q_i - q_best))
                    for visited children (moves without a child keep their mass), renormalised over the legal moves.
                    KLD against the plain search target worsens by construction; read puzzles, not KLD.
"""
import torch


def _live(child_idx, child_n):
  return (child_idx >= 0) & (child_n > 0)


def child_q_gap(child_idx, child_q, child_n, cap: float = 0.5):
  """[B]: q_best - q_second over the visited children, 0 where fewer than two are visited, clipped to [0, cap]."""
  live = _live(child_idx, child_n)
  q = torch.where(live, child_q.float(), torch.full_like(child_q, -2.0, dtype=torch.float32))
  top2 = q.topk(2, dim=1).values
  gap = (top2[:, 0] - top2[:, 1]).clamp(0.0, cap)
  return torch.where(live.sum(dim=1) >= 2, gap, torch.zeros_like(gap))


def qgap_weights(child_idx, child_q, child_n, lam: float, cap: float = 0.5):
  gap = child_q_gap(child_idx, child_q, child_n, cap)
  w = 1.0 + lam * gap
  w = w / w.mean().clamp_min(1e-6)
  diag = {'pw_qgap_mean': gap.mean(), 'pw_qgap_frac_gt_010': (gap > 0.10).float().mean(),
          'pw_qgap_w_max': w.max(), 'pw_qgap_w_frac_gt_15': (w > 1.5).float().mean()}
  return w.detach(), diag


def search_vs_prior_kl(policy_target, child_idx, child_n, child_prior):
  """[B]: KL(search policy || generating net's prior) over the visited children (both renormalised on that set).
  Rows without usable children return NaN (caller decides the weight)."""
  live = _live(child_idx, child_n) & (child_prior > 0)
  p = torch.gather(policy_target.float(), 1, child_idx.clamp_min(0)) * live
  pr = child_prior.float() * live
  ps, prs = p.sum(dim=1, keepdim=True), pr.sum(dim=1, keepdim=True)
  ok = (ps.squeeze(1) > 0) & (prs.squeeze(1) > 0)
  p = p / ps.clamp_min(1e-9)
  pr = pr / prs.clamp_min(1e-9)
  term = torch.where(p > 0, p * (p.clamp_min(1e-12).log() - pr.clamp_min(1e-12).log()), torch.zeros_like(p))
  kl = term.sum(dim=1)
  return torch.where(ok, kl, torch.full_like(kl, float('nan')))


def surprise_weights(policy_target, child_idx, child_n, child_prior, ref_kl: float, lo: float = 0.5, hi: float = 4.0):
  kl = search_vs_prior_kl(policy_target, child_idx, child_n, child_prior)
  w = torch.where(torch.isnan(kl), torch.ones_like(kl), (kl / ref_kl).clamp(lo, hi))
  w = w / w.mean().clamp_min(1e-6)
  klv = kl[~torch.isnan(kl)]
  diag = {'pw_surprise_kl_mean': klv.mean() if klv.numel() else torch.zeros((), device=kl.device),
          'pw_surprise_kl_median': klv.median() if klv.numel() else torch.zeros((), device=kl.device),
          'pw_srp_w_max': w.max(), 'pw_srp_w_frac_gt_15': (w > 1.5).float().mean()}
  return w.detach(), diag


def q_improved_target(policy_target, child_idx, child_q, child_n, beta: float):
  """target' ∝ target * exp(beta * (q_i - q_best)) on visited children; other legal moves keep their mass."""
  live = _live(child_idx, child_n)
  q = torch.where(live, child_q.float(), torch.full_like(child_q, -2.0, dtype=torch.float32))
  qmax = q.max(dim=1, keepdim=True).values
  lw = torch.where(live, beta * (child_q.float() - qmax), torch.zeros_like(q))          # <= 0, 0 where dead
  add = torch.zeros(policy_target.shape, device=policy_target.device, dtype=torch.float32)
  # child_idx is unique per row in 1858 space (no promotions collide there); dead slots add exactly 0
  add.scatter_add_(1, child_idx.clamp_min(0), lw)
  t = policy_target.float()
  legal = t > 0
  t2 = torch.where(legal, t * add.exp(), torch.zeros_like(t))
  t2 = t2 / t2.sum(dim=1, keepdim=True).clamp_min(1e-9)
  with torch.no_grad():
    top_changed = (t2.argmax(dim=1) != t.argmax(dim=1)).float().mean()
    mass_moved = 0.5 * (t2 - t / t.sum(dim=1, keepdim=True).clamp_min(1e-9)).abs().sum(dim=1).mean()
    diag = {'pw_qpol_top1_changed': top_changed, 'pw_qpol_mass_moved': mass_moved}
  return t2, diag        # fp32 on purpose: policy_loss masks legality by target > 0; fp16 could underflow a legal move to 0
