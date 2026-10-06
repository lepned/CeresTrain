"""Contract test for the Grill RPO policy target (policy_v8.grill_y_star / grill_target, 2026-10-02).

    python test_policy_v8_grill.py            (CPU, synthetic)

1. y* matches an independent float64 bisection of sum lam*mu/(alpha-q) = 1, sums to 1, and is positive.
2. Limits: lam -> large gives mu; lam -> small concentrates on the best q.
3. grill_target: unvisited legal moves keep their mass, the visited set keeps its combined mass, illegal moves stay 0,
   legal moves stay > 0, rows with fewer than two visited children are unchanged, and dead slots never write index 0.
"""
import numpy as np
import torch
from policy_v8 import grill_y_star, grill_target, grill_completed_target


def ref_y(mu, q, lam):
  lo, hi = max(q + lam * mu), max(q) + lam
  for _ in range(200):
    mid = 0.5 * (lo + hi)
    if (lam * mu / (mid - q)).sum() > 1.0:
      lo = mid
    else:
      hi = mid
  a = 0.5 * (lo + hi)
  y = lam * mu / (a - q)
  return y / y.sum()


def test_solver():
  g = torch.Generator().manual_seed(1)
  B, S = 64, 12
  mu = torch.rand(B, S, generator=g) + 0.01
  mu = mu / mu.sum(1, keepdim=True)
  q = torch.rand(B, S, generator=g) * 2 - 1
  lam = torch.rand(B, generator=g) * 0.5 + 0.01
  live = torch.ones(B, S, dtype=torch.bool)
  y = grill_y_star(mu, q, lam, live)
  for b in range(B):
    r = ref_y(mu[b].double().numpy(), q[b].double().numpy(), float(lam[b]))
    assert np.allclose(y[b].double().numpy(), r, atol=2e-5), (b, np.abs(y[b].numpy() - r).max())
  assert torch.allclose(y.sum(1), torch.ones(B), atol=1e-5) and (y > 0).all()
  y_big = grill_y_star(mu, q, torch.full((B,), 1e4), live)
  assert torch.allclose(y_big, mu, atol=1e-3), 'lam -> inf must give mu'
  y_small = grill_y_star(mu, q, torch.full((B,), 1e-4), live)
  assert (y_small.argmax(1) == q.argmax(1)).all() and (y_small.max(1).values > 0.9).all(), 'lam -> 0 must pick argmax q'
  print('OK solver: matches float64 reference, limits mu / argmax-q hold')


def test_target():
  B, M, S = 3, 1858, 6
  t = torch.zeros(B, M)
  legal = torch.tensor([3, 10, 25, 40, 77, 300, 901])
  t[:, legal] = torch.tensor([0.30, 0.25, 0.20, 0.10, 0.05, 0.05, 0.05])
  child_idx = torch.full((B, S), -1, dtype=torch.long)
  child_idx[:, :4] = torch.tensor([3, 10, 25, 40])
  child_n = torch.zeros(B, S, dtype=torch.long); child_n[:, :4] = torch.tensor([300, 250, 200, 100])
  child_q = torch.zeros(B, S); child_q[:, :4] = torch.tensor([0.10, 0.30, 0.00, -0.20])
  child_prior = torch.zeros(B, S); child_prior[:, :4] = torch.tensor([0.4, 0.2, 0.2, 0.1])
  child_n[2, 1:] = 0                                              # row 2: only one visited child -> unchanged
  t2, diag = grill_target(t, child_idx, child_q, child_n, child_prior, c=2.0)
  tn = t / t.sum(1, keepdim=True)
  assert torch.allclose(t2.sum(1), torch.ones(B), atol=1e-5)
  assert (t2[t == 0] == 0).all() and (t2[t > 0] > 0).all(), 'legality must be preserved exactly'
  vis = torch.tensor([3, 10, 25, 40]); unvis = torch.tensor([77, 300, 901])
  for b in range(2):
    assert torch.allclose(t2[b, unvis], tn[b, unvis], atol=1e-5), 'unvisited legal moves keep their mass'
    assert abs(float(t2[b, vis].sum() - tn[b, vis].sum())) < 1e-5, 'visited set keeps its combined mass'
    assert int(t2[b, vis].argmax()) == 1, 'the clearly best-q child (idx 10) should gain the top spot'
  assert torch.allclose(t2[2], tn[2], atol=1e-6), 'row with < 2 visited children must be unchanged'
  assert float(t2[0, 0]) == 0.0, 'dead slots (child_idx = -1) must not write index 0'
  print('OK target: mass bookkeeping, legality, fallbacks; diag', {k: round(float(v), 4) for k, v in diag.items()})


def test_completed():
  B, M, S = 3, 1858, 6
  t = torch.zeros(B, M)
  legal = torch.tensor([3, 10, 25, 40, 77, 300])
  t[:, legal] = torch.tensor([0.30, 0.25, 0.20, 0.10, 0.10, 0.05])
  stored_idx = torch.full((B, S), -1, dtype=torch.long); stored_idx[:, :5] = torch.tensor([3, 10, 25, 40, 77])   # 300 legal, not stored
  stored_prior = torch.zeros(B, S); stored_prior[:, :5] = torch.tensor([0.3, 0.2, 0.2, 0.2, 0.1])
  child_idx = torch.full((B, S), -1, dtype=torch.long); child_idx[:, :4] = stored_idx[:, :4]                    # 77 stored, unvisited
  n = torch.zeros(B, S); n[:, :4] = torch.tensor([300., 250., 200., 50.])
  q = torch.zeros(B, S); q[:, :4] = torch.tensor([0.10, 0.30, 0.00, -0.20])
  root_q = torch.full((B, 1), 0.12)
  n[2] = 0; child_idx[2] = -1                                                                                       # row 2: nothing visited
  t2, diag = grill_completed_target(t, stored_idx, stored_prior, child_idx, q, n, root_q, c=2.5, beta=0.25, n0=1.0)
  tn = t / t.sum(1, keepdim=True)
  assert torch.allclose(t2.sum(1), torch.ones(B), atol=1e-5)
  assert (t2[t == 0] == 0).all() and (t2[t > 0] > 0).all(), 'legality must be preserved exactly'
  assert float(t2[0, 300]) < 1e-5, 'legal but not stored -> floor only'
  assert int(t2[0].argmax()) == 10, 'clearly best completed q (idx 10) should lead'
  assert float(t2[0, 77]) > 1e-4, 'unvisited stored move gets mass via q_hat = v_mix and its prior'
  assert torch.allclose(t2[2], tn[2], atol=1e-6), 'nothing visited (N == 0) -> plain target'
  y_b1, _ = grill_completed_target(t, stored_idx, stored_prior, child_idx, q, n, root_q, beta=1.0)
  nn = n[0, :4] / n[0, :4].sum()
  assert torch.allclose(y_b1[0, torch.tensor([3, 10, 25, 40])], nn, atol=1e-5), 'beta = 1 must give the visit distribution'
  print('OK completed: legality, floor, completed-Q for unvisited, N == 0 fallback, beta = 1 -> visits')


def test_tiny_prior():
  # review 2026-10-02: best-q move with a tiny prior; a float32 solve converged alpha onto q and gave it ~all the mass
  mu = torch.tensor([[1e-7, 0.5 - 0.5e-7, 0.5 - 0.5e-7]])
  q = torch.tensor([[1.0, 0.0, -0.1]])
  y = grill_y_star(mu, q, torch.tensor([0.05]), torch.ones(1, 3, dtype=torch.bool))
  r = ref_y(mu[0].double().numpy(), q[0].double().numpy(), 0.05)
  assert np.allclose(y[0].double().numpy(), r, atol=1e-4), (y, r)
  print('OK tiny prior: float64 solve matches the reference', [round(float(v), 4) for v in y[0]])


def test_aux_fallbacks():
  # 2026-10-07 Grill aux head (Kovax' fallbacks): min_visited and a non-finite v keep the plain target;
  # the defaults must reproduce the original function exactly.
  B, M, S = 3, 1858, 6
  t = torch.zeros(B, M)
  legal = torch.tensor([3, 10, 25, 40, 77])
  t[:, legal] = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.10])
  stored_idx = torch.full((B, S), -1, dtype=torch.long); stored_idx[:, :5] = legal
  stored_prior = torch.zeros(B, S); stored_prior[:, :5] = torch.tensor([0.3, 0.2, 0.2, 0.2, 0.1])
  child_idx = stored_idx.clone()
  n = torch.zeros(B, S); n[:, :5] = torch.tensor([300., 250., 200., 50., 10.])
  q = torch.zeros(B, S); q[:, :5] = torch.tensor([0.10, 0.30, 0.00, -0.20, -0.5])
  n[1, 1:] = 0; child_idx[1, 1:] = -1                                     # row 1: one visited child
  v = torch.tensor([[0.12], [0.12], [float('nan')]])                     # row 2: unknown v
  tn = t / t.sum(1, keepdim=True)
  base, _ = grill_completed_target(t, stored_idx, stored_prior, child_idx, q, n, torch.nan_to_num(v))
  same, _ = grill_completed_target(t, stored_idx, stored_prior, child_idx, q, n, torch.nan_to_num(v), min_visited=1,
                                   nonfinite_v_fallback=False)
  assert torch.equal(base, same), 'defaults must reproduce the original target'
  aux, d = grill_completed_target(t, stored_idx, stored_prior, child_idx, q, n, v, n0=2.0, min_visited=2,
                                  nonfinite_v_fallback=True)
  assert torch.isfinite(aux).all()
  assert not torch.allclose(aux[0], tn[0], atol=1e-4), 'row 0 is a normal Grill row'
  assert torch.allclose(aux[1], tn[1], atol=1e-6), 'one visited child < min_visited -> plain target'
  assert torch.allclose(aux[2], tn[2], atol=1e-6), 'NaN v -> plain target'
  assert abs(float(d['pw_grill_rows_used']) - 1 / 3) < 1e-6
  # mutation guard: without the NaN fallback the NaN row must NOT silently become plain
  nofb, _ = grill_completed_target(t, stored_idx, stored_prior, child_idx, q, n, v, n0=2.0, min_visited=2)
  assert not torch.allclose(nofb[2], tn[2], atol=1e-4), 'NaN v is zero-filled when the fallback is off'
  print('OK aux fallbacks: defaults unchanged, min_visited and NaN-v rows keep the plain target')


if __name__ == '__main__':
  test_aux_fallbacks()
  test_tiny_prior()
  test_solver()
  test_target()
  test_completed()
