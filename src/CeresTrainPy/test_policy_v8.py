"""Contract test for the v8 child-table policy reshaping (policy_v8.py, 2026-09-17).

    python test_policy_v8.py     (from src/CeresTrainPy, CPU)

1. only-move q-gap weights: gap = q_best - q_2nd over VISITED children, clipped; rows with < 2 visited children get gap 0;
   weights are mean-1; lam scales them.
2. search-surprise weights: KL(search || prior) over the visited children, both renormalised on that set; identical
   distributions give KL 0 -> weight lo; a row without children gets weight 1 before normalisation; clipping holds.
3. completed-Q target: best-q child keeps its mass ratio, worse children shrink by exp(beta*dq), moves without a child
   keep their mass, illegal moves stay 0, rows sum to 1, beta -> 0 reproduces the input.
4. losses.policy_loss(row_weights=ones) equals the unweighted loss; the logged number is the unweighted CE.
"""
import os, sys, math
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_v8 import child_q_gap, qgap_weights, search_vs_prior_kl, surprise_weights, q_improved_target


def _table():
  # 3 rows x 4 slots; row 2 has a single visited child, row 0 a clear only-move
  ci = torch.tensor([[10, 20, 30, -1], [40, 50, 60, 70], [80, -1, -1, -1]])
  cq = torch.tensor([[0.8, 0.2, -0.5, 0.0], [0.1, 0.05, 0.0, -0.3], [0.3, 0.0, 0.0, 0.0]])
  cn = torch.tensor([[500, 200, 50, 0], [300, 250, 100, 20], [40, 0, 0, 0]])
  return ci, cq, cn


def qgap():
  ci, cq, cn = _table()
  gap = child_q_gap(ci, cq, cn)
  assert torch.allclose(gap, torch.tensor([0.5, 0.05, 0.0]), atol=1e-6), gap      # 0.6 clipped to 0.5; 0.05; single child -> 0
  w, d = qgap_weights(ci, cq, cn, lam=2.0)
  raw = torch.tensor([2.0, 1.1, 1.0])
  assert torch.allclose(w, raw / raw.mean(), atol=1e-6) and abs(float(w.mean()) - 1) < 1e-6, w
  print(f'  qgap OK: gaps {gap.tolist()}, weights {[round(x, 3) for x in w.tolist()]}, frac>0.10 {float(d["pw_qgap_frac_gt_010"]):.2f}')


def surprise():
  ci, cq, cn = _table()
  t = torch.zeros(3, 1858)
  t[0, [10, 20, 30]] = torch.tensor([0.6, 0.3, 0.1]); t[1, [40, 50, 60, 70]] = torch.tensor([0.4, 0.3, 0.2, 0.1]); t[2, 80] = 1.0
  prior = torch.zeros(3, 4)
  prior[0] = torch.tensor([0.6, 0.3, 0.1, 0.0])         # identical -> KL 0
  prior[1] = torch.tensor([0.1, 0.2, 0.3, 0.4])         # reversed -> large KL
  prior[2] = torch.tensor([0.5, 0.0, 0.0, 0.0])         # single child -> KL 0 after renormalisation
  kl = search_vs_prior_kl(t, ci, cn, prior)
  assert abs(float(kl[0])) < 1e-6 and float(kl[1]) > 0.3 and abs(float(kl[2])) < 1e-6, kl
  w, d = surprise_weights(t, ci, cn, prior, ref_kl=0.07)
  raw = torch.tensor([0.5, min(4.0, float(kl[1]) / 0.07), 0.5])
  assert torch.allclose(w, raw / raw.mean(), atol=1e-5), (w, raw)
  # a row with no usable child at all -> weight 1 (before normalisation)
  ci2 = ci.clone(); ci2[2] = -1
  kl2 = search_vs_prior_kl(t, ci2, cn, prior); assert torch.isnan(kl2[2])
  w2, _ = surprise_weights(t, ci2, cn, prior, ref_kl=0.07)
  raw2 = torch.tensor([0.5, min(4.0, float(kl[1]) / 0.07), 1.0]); assert torch.allclose(w2, raw2 / raw2.mean(), atol=1e-5)
  print(f'  surprise OK: KL {[round(x, 3) for x in kl.tolist()]}, weights {[round(x, 3) for x in w.tolist()]} (clip 0.5..4, mean 1)')


def qpol():
  ci, cq, cn = _table()
  t = torch.zeros(3, 1858, dtype=torch.float16)
  t[0, [10, 20, 30, 99]] = torch.tensor([0.5, 0.3, 0.1, 0.1]).half()     # 99 = legal move without a child
  t[1, [40, 50, 60, 70]] = torch.tensor([0.4, 0.3, 0.2, 0.1]).half()
  t[2, 80] = 1.0
  t2, d = q_improved_target(t, ci, cq, cn, beta=2.0)
  t2 = t2.float()
  assert t2.dtype == torch.float32 and torch.allclose(t2.sum(1), torch.ones(3), atol=1e-3)
  assert float(t2[0, 5]) == 0.0, 'illegal moves stay 0'
  # row 0: child 10 (q .8) keeps factor 1, child 20 (q .2) factor exp(-1.2), child 30 (q -.5) exp(-2.6), move 99 factor 1
  raw = torch.tensor([0.5, 0.3 * math.exp(-1.2), 0.1 * math.exp(-2.6), 0.1]); raw = raw / raw.sum()
  assert torch.allclose(t2[0, [10, 20, 30, 99]], raw, atol=2e-3), (t2[0, [10, 20, 30, 99]], raw)
  # row 2: single child -> unchanged
  assert abs(float(t2[2, 80]) - 1.0) < 1e-6
  t0, _ = q_improved_target(t, ci, cq, cn, beta=0.0)
  assert torch.allclose(t0.float(), t.float() / t.float().sum(1, keepdim=True), atol=1e-3), 'beta 0 must reproduce the input'
  print(f'  completed-Q target OK: row0 {[round(x, 3) for x in t2[0, [10, 20, 30, 99]].tolist()]}, top1 changed {float(d["pw_qpol_top1_changed"]):.2f}, '
        f'mass moved {float(d["pw_qpol_mass_moved"]):.3f}')


def weighted_ce():
  from losses import LossCalculator
  import torch.nn as nn
  lc = LossCalculator(nn.Linear(4, 4))
  torch.manual_seed(1)
  out = torch.randn(3, 1858)
  t = torch.zeros(3, 1858); t[:, :5] = torch.softmax(torch.randn(3, 5), dim=1)
  l0 = lc.policy_loss(t, out, True, False, 1.0)
  l1 = lc.policy_loss(t, out, True, False, 1.0, row_weights=torch.ones(3))
  assert torch.allclose(l0, l1, atol=1e-5), (l0, l1)
  w = torch.tensor([2.0, 0.5, 0.5])
  l2 = lc.policy_loss(t, out, True, False, 1.0, row_weights=w)
  assert not torch.allclose(l0, l2, atol=1e-4) and torch.allclose(lc._last_policy_log_loss, l0, atol=1e-5), 'logged CE must stay unweighted'
  print(f'  weighted CE OK: ones == unweighted ({float(l0):.4f}); weights change the returned loss ({float(l2):.4f}) but not the logged one')


if __name__ == '__main__':
  qgap(); surprise(); qpol(); weighted_ce()
  print('ALL OK')
