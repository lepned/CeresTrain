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
  grill_target      Grill et al. (2020) "MCTS as regularized policy optimization": on the visited children the target is
                    y* = argmax_y sum y*q - lam*KL(mu || y) = lam*mu / (alpha - q), mu = the generating net's prior renormalised
                    on the visited set, lam = c*sqrt(N)/(N + |A|) (N = root visits of the visited set, |A| = legal moves),
                    alpha by bisection so that sum y* = 1. The visited children keep their combined target mass, which y*
                    redistributes; unvisited legal moves keep their mass. Same math as Ceres' RegularizedPolicyOptimum
                    (ReverseKL).
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


def grill_y_star(mu, q, lam, live, iters: int = 60):
  """[B,S] y* = lam*mu/(alpha-q) on live slots (0 elsewhere), alpha per row by bisection so that sum y* = 1.
  mu must sum to 1 over the live slots of each row; lam [B] > 0. Bracket: alpha in (max(q + lam*mu), max(q) + lam]
  (at the lower end the largest term alone is >= 1, at the upper end every term is <= mu).
  Solved in float64 (review 2026-10-02): in float32 a best-q move with a tiny prior can converge alpha onto q itself,
  and the clamp below would then hand that move ~all the mass -- a finite, plausible-looking, wrong target."""
  out_dtype = mu.dtype
  mu, q, lam = mu.double(), q.double(), lam.double()
  lam_ = lam.unsqueeze(1)
  neg = torch.full_like(q, -1e9)
  lo = torch.where(live, q + lam_ * mu, neg).max(dim=1, keepdim=True).values
  hi = torch.where(live, q, neg).max(dim=1, keepdim=True).values + lam_
  for _ in range(iters):
    mid = 0.5 * (lo + hi)
    s = torch.where(live, lam_ * mu / (mid - q).clamp_min(1e-300), torch.zeros_like(q)).sum(dim=1, keepdim=True)
    too_big = s > 1.0                    # sum decreases in alpha: too much mass => alpha must grow
    lo = torch.where(too_big, mid, lo)
    hi = torch.where(too_big, hi, mid)
  alpha = 0.5 * (lo + hi)
  y = torch.where(live, lam_ * mu / (alpha - q).clamp_min(1e-300), torch.zeros_like(q))
  return (y / y.sum(dim=1, keepdim=True).clamp_min(1e-300)).to(out_dtype)


def grill_target(policy_target, child_idx, child_q, child_n, child_prior, c: float, iters: int = 60):
  """Grill RPO target (see module doc). Rows with fewer than two visited children, or no usable prior, keep the plain target."""
  t = policy_target.float()
  legal = t > 0
  live = _live(child_idx, child_n) & (child_prior > 0)
  n_live = live.sum(dim=1)
  q = torch.where(live, child_q.float(), torch.zeros_like(child_q, dtype=torch.float32))
  mu = torch.where(live, child_prior.float(), torch.zeros_like(q))
  mu_sum = mu.sum(dim=1, keepdim=True)
  ok = (n_live >= 2) & (mu_sum.squeeze(1) > 0)
  mu = mu / mu_sum.clamp_min(1e-12)
  n_vis = torch.where(live, child_n.float(), torch.zeros_like(q)).sum(dim=1)
  n_act = legal.sum(dim=1).float().clamp_min(1.0)
  lam = c * n_vis.sqrt() / (n_vis + n_act)
  ok = ok & (lam > 0)
  y = grill_y_star(mu, q, lam.clamp_min(1e-9), live & ok.unsqueeze(1), iters)
  idx = child_idx.clamp_min(0)
  t_vis = torch.gather(t, 1, idx) * live                                   # current target mass on each visited child
  m_vis = t_vis.sum(dim=1, keepdim=True)
  delta = torch.where(live & ok.unsqueeze(1), m_vis * y - t_vis, torch.zeros_like(q))
  t2 = t.clone()
  t2.scatter_add_(1, idx, delta)                                           # child_idx is unique per row; dead slots add 0
  t2 = torch.where(legal, t2.clamp_min(1e-7), torch.zeros_like(t2))        # policy_loss masks legality by target > 0
  t2 = t2 / t2.sum(dim=1, keepdim=True).clamp_min(1e-9)
  with torch.no_grad():
    tn = t / t.sum(dim=1, keepdim=True).clamp_min(1e-9)
    diag = {'pw_grill_top1_changed': (t2.argmax(dim=1) != t.argmax(dim=1)).float().mean(),
            'pw_grill_mass_moved': 0.5 * (t2 - tn).abs().sum(dim=1).mean(),
            'pw_grill_lambda_mean': lam[ok].mean() if ok.any() else torch.zeros((), device=t.device),
            'pw_grill_rows_used': ok.float().mean()}
  return t2, diag


def grill_completed_target(policy_target, stored_idx, stored_prior, child_idx, child_q, child_visits, root_q,
                           c: float = 2.5, beta: float = 0.25, n0: float = 1.0, iters: int = 60,
                           min_visited: int = 1, nonfinite_v_fallback: bool = False):
  """Kovax's Grill target (2026-10-02), Gumbel-style completed Q + a visit blend, over EVERY stored move:
    N = sum n_a (n_a = child_visits on visited slots, 0 elsewhere), |A| = number of stored moves,
    v_hat = sum_vis pi*q / sum_vis pi  (falls back to v = root_q when nothing is visited),
    v_mix = (v + N*v_hat) / (1 + N),  q_hat_a = (n_a*q_a + n0*v_mix) / (n_a + n0)  (unvisited: v_mix),
    pi_hat = lam*pi / (alpha - q_hat) with lam = c*sqrt(N)/(N + |A|)  (grill_y_star),
    Y = (1 - beta)*pi_hat + beta*n_a/N,  renormalised.
  Slots are aligned: stored_idx/stored_prior cover every stored slot, child_idx/child_q the visited ones (-1 elsewhere).
  Rows with N == 0 or no usable prior keep the plain target. Legal moves outside the stored set get a 1e-7 floor
  (policy_loss masks legality by target > 0).
  min_visited / nonfinite_v_fallback (Kovax' fallbacks, used by the Grill aux head): rows with fewer than min_visited
  visited children, or (when set) a non-finite v (root_q argument; the aux head passes orig_q, NaN when unknown), also
  keep the plain target. Defaults reproduce the original behaviour."""
  t = policy_target.float()
  legal = t > 0
  stored = stored_idx >= 0
  vis = (child_idx >= 0) & stored
  pi = torch.where(stored, stored_prior.float(), torch.zeros_like(stored_prior, dtype=torch.float32))
  n = torch.where(vis, child_visits.float(), torch.zeros_like(pi))
  q = torch.where(vis, child_q.float(), torch.zeros_like(pi))
  N = n.sum(dim=1, keepdim=True)
  n_act = stored.sum(dim=1, keepdim=True).float()
  v_raw = root_q.float().reshape(-1, 1)
  v_ok = torch.isfinite(v_raw).squeeze(1)
  v = torch.where(torch.isfinite(v_raw), v_raw, torch.zeros_like(v_raw))
  pv = torch.where(vis & (n > 0), pi, torch.zeros_like(pi))
  v_hat = torch.where(pv.sum(1, keepdim=True) > 0, (pv * q).sum(1, keepdim=True) / pv.sum(1, keepdim=True).clamp_min(1e-12), v)
  v_mix = (v + N * v_hat) / (1.0 + N)
  q_hat = torch.where(stored, (n * q + n0 * v_mix) / (n + n0).clamp_min(1e-9), torch.zeros_like(pi))
  pi_sum = pi.sum(dim=1, keepdim=True)
  ok = (N.squeeze(1) > 0) & (pi_sum.squeeze(1) > 0)
  if min_visited > 1:
    ok = ok & ((vis & (n > 0)).sum(dim=1) >= min_visited)
  if nonfinite_v_fallback:
    ok = ok & v_ok
  mu = pi / pi_sum.clamp_min(1e-12)
  lam = (c * N.sqrt() / (N + n_act).clamp_min(1.0)).squeeze(1)
  live = stored & (mu > 0) & ok.unsqueeze(1)
  y = grill_y_star(mu, q_hat, lam.clamp_min(1e-9), live, iters)
  Y = (1.0 - beta) * y + beta * n / N.clamp_min(1.0)
  Y = torch.where(stored & ok.unsqueeze(1), Y, torch.zeros_like(Y))
  dense = torch.zeros_like(t)
  dense.scatter_add_(1, stored_idx.clamp_min(0), Y)                         # stored_idx unique per row; masked slots add 0
  dense = torch.where(legal, dense.clamp_min(1e-7), torch.zeros_like(dense))
  dense = dense / dense.sum(dim=1, keepdim=True).clamp_min(1e-9)
  tn = t / t.sum(dim=1, keepdim=True).clamp_min(1e-9)
  t2 = torch.where(ok.unsqueeze(1), dense, tn)
  with torch.no_grad():
    ent = lambda p: -(p.clamp_min(1e-12).log() * p).sum(1).mean()
    diag = {'pw_grill_top1_changed': (t2.argmax(dim=1) != t.argmax(dim=1)).float().mean(),
            'pw_grill_mass_moved': 0.5 * (t2 - tn).abs().sum(dim=1).mean(),
            'pw_grill_lambda_mean': lam[ok].mean() if ok.any() else torch.zeros((), device=t.device),
            'pw_grill_rows_used': ok.float().mean(),
            'pw_grill_entropy_plain': ent(tn), 'pw_grill_entropy_target': ent(t2)}
  return t2, diag
