"""move_token_value_order_loss: the v8 child-q target, and that the old target is intact.

The control arm of the phase-2 A/B runs the SAME function with `child=None`. If that path
shifted even slightly, the experiment would be measuring a code change as well as a target
change — so the first test here is a bit-exact comparison against a standalone copy of the
pre-change implementation.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
from move_tokens import move_token_value_order_loss, _children_to_tokens

M, SCRATCH = 8, 4096


def _reference_visit_mass(u, sel, valid, policy_target, mv_pair_flat, topk, min_mass=0.0):
  """Verbatim copy of the implementation as it stood before the child-q change."""
  B = u.shape[0]
  t = policy_target.float()
  pair_mass = torch.zeros(B, 4096, device=t.device, dtype=t.dtype).index_add_(1, mv_pair_flat, t)
  tok_t = torch.gather(pair_mass, 1, sel) * valid.to(t.dtype)
  s_all = u.float().masked_fill(~valid | (tok_t <= 0), float('-inf'))
  order = torch.argsort(tok_t, dim=1, descending=True)
  s = torch.gather(s_all, 1, order)
  suf = torch.flip(torch.logcumsumexp(torch.flip(s, dims=[1]), dim=1), dims=[1])
  K = min(int(topk), s.shape[1])
  s_k, suf_k = s[:, :K], suf[:, :K]
  ok = (torch.gather(tok_t, 1, order)[:, :K] > float(min_mass)) & torch.isfinite(s_k)
  terms = torch.where(ok, suf_k - s_k, torch.zeros_like(s_k))
  return terms.sum(dim=1).mean()


def _fixture(seed=0, B=6):
  g = torch.Generator().manual_seed(seed)
  u = torch.randn(B, M, generator=g)
  # Each token owns a distinct from-to pair.
  sel = torch.arange(M).unsqueeze(0).expand(B, M).contiguous()
  valid = torch.ones(B, M, dtype=torch.bool)
  valid[:, -2:] = False                       # two dead tokens per row
  mv_pair_flat = torch.arange(1858) % 4096
  pt = torch.zeros(B, 1858)
  pt[:, :M] = torch.rand(B, M, generator=g)
  pt = pt / pt.sum(1, keepdim=True)
  return u, sel, valid, pt, mv_pair_flat


def control_arm_unchanged():
  for seed in range(5):
    u, sel, valid, pt, mvp = _fixture(seed)
    for mm in (0.0, 0.01):
      got, _ = move_token_value_order_loss(u, sel, valid, pt, mvp, topk=5, min_mass=mm)
      want = _reference_visit_mass(u, sel, valid, pt, mvp, topk=5, min_mass=mm)
      assert torch.equal(got, want), f'control path changed (seed {seed}, min_mass {mm}): {got} vs {want}'
  print('  control arm OK: child=None is bit-identical to the pre-change implementation')


def child_q_ranks_by_value():
  """Visit order and value order must be able to DISAGREE, or the arm tests nothing."""
  u, sel, valid, pt, mvp = _fixture(1)
  B = u.shape[0]
  S = 16
  ci = torch.full((B, S), -1, dtype=torch.int64)
  cq = torch.zeros(B, S)
  cn = torch.zeros(B, S, dtype=torch.int64)
  # Child k covers pair k (token k). Visits DESCENDING (the writer's contract), values
  # deliberately ASCENDING, so the two targets rank the tokens in opposite orders.
  for k in range(M - 2):
    ci[:, k] = k
    cn[:, k] = 100 - 10 * k
    cq[:, k] = -0.9 + 0.2 * k
  loss_q, diag_q = move_token_value_order_loss(u, sel, valid, pt, mvp, topk=5, child=(ci, cq, cn))
  loss_v, diag_v = move_token_value_order_loss(u, sel, valid, pt, mvp, topk=5)
  assert not torch.equal(loss_q, loss_v), 'child-q target gave the same loss as visit mass'
  assert diag_q['mt_vord_targets_per_row'] == float(M - 2)
  print(f'  child-q OK: visit and value orders disagree -> different loss '
        f'({loss_v.item():.4f} vs {loss_q.item():.4f}), {int(diag_q["mt_vord_targets_per_row"])} targets/row')


def promotion_tiebreak():
  """A pair carrying several moves must take q from the MOST-VISITED one."""
  B, S = 2, 8
  sel = torch.zeros(B, 1, dtype=torch.int64)          # one token, pair 0
  valid = torch.ones(B, 1, dtype=torch.bool)
  u = torch.zeros(B, 1)
  mv_pair_flat = torch.zeros(1858, dtype=torch.int64)  # every move -> pair 0
  ci = torch.full((B, S), -1, dtype=torch.int64)
  cq = torch.zeros(B, S)
  cn = torch.zeros(B, S, dtype=torch.int64)
  # Slots arrive n_raw-descending: slot 0 is the most visited, and its q must win.
  ci[:, 0], cn[:, 0], cq[:, 0] = 10, 90, 0.75
  ci[:, 1], cn[:, 1], cq[:, 1] = 11, 40, -0.60
  ci[:, 2], cn[:, 2], cq[:, 2] = 12, 5, 0.99     # highest q but nearly unvisited
  _, diag = move_token_value_order_loss(u, sel, valid, torch.zeros(B, 1858), mv_pair_flat,
                                        topk=1, child=(ci, cq, cn))
  assert diag['mt_vord_targets_per_row'] == 1.0
  # Assert the VALUE that survives the tiebreak, not just that the loss is finite. The
  # earlier version only checked finiteness, so it passed for any winner -- including the
  # arbitrary ones the old CUDA scatter could produce. Two pairs, each over-subscribed,
  # with the correct answer different from both the lowest-q and the highest-q child.
  mv2 = torch.zeros(1858, dtype=torch.int64)
  mv2[20], mv2[21] = 1, 1                        # moves 20, 21 -> pair 1
  ci2 = torch.full((B, S), -1, dtype=torch.int64)
  cq2 = torch.zeros(B, S)
  cn2 = torch.zeros(B, S, dtype=torch.int64)
  ci2[:, 0], cn2[:, 0], cq2[:, 0] = 10, 90, 0.75   # pair 0 winner
  ci2[:, 1], cn2[:, 1], cq2[:, 1] = 11, 40, -0.60
  ci2[:, 2], cn2[:, 2], cq2[:, 2] = 12, 5, 0.99    # highest q, nearly unvisited -> must lose
  ci2[:, 3], cn2[:, 3], cq2[:, 3] = 21, 70, -0.25  # pair 1 winner (NOT the lowest slot of the two)
  ci2[:, 4], cn2[:, 4], cq2[:, 4] = 20, 30, 0.40
  sel2 = torch.tensor([[0, 1]] * B)
  valid2 = torch.ones(B, 2, dtype=torch.bool)
  tok_n, tok_q = _children_to_tokens(sel2, valid2, mv2, ci2, cn2, cq2)
  assert torch.allclose(tok_n, torch.tensor([[90.0, 70.0]] * B)), tok_n
  assert torch.allclose(tok_q, torch.tensor([[0.75, -0.25]] * B)), tok_q

  # A token whose pair has no child at all must come back dead, not borrow a neighbour's.
  sel3 = torch.tensor([[0, 7]] * B)
  n3, q3 = _children_to_tokens(sel3, valid2, mv2, ci2, cn2, cq2)
  assert torch.allclose(n3[:, 1], torch.zeros(B)), n3
  print('  promotion tiebreak OK: most-visited child wins per pair '
        '(q 0.75/-0.25, n 90/70), empty pair stays dead')


def unvisited_excluded():
  """q is the -32768 sentinel where n_raw == 0; those moves must not become targets."""
  u, sel, valid, pt, mvp = _fixture(2)
  B, S = u.shape[0], 16
  ci = torch.full((B, S), -1, dtype=torch.int64)
  cq = torch.zeros(B, S)
  cn = torch.zeros(B, S, dtype=torch.int64)
  ci[:, 0], cn[:, 0], cq[:, 0] = 0, 50, 0.4         # visited
  ci[:, 1], cn[:, 1], cq[:, 1] = 1, 0, -1.0         # present but UNVISITED
  _, diag = move_token_value_order_loss(u, sel, valid, pt, mvp, topk=5, child=(ci, cq, cn))
  assert diag['mt_vord_targets_per_row'] == 1.0, \
      f'unvisited child counted as a target ({diag["mt_vord_targets_per_row"]})'
  print('  unvisited excluded OK: only the visited child becomes a ranked target')


def gradient_flows():
  u, sel, valid, pt, mvp = _fixture(3)
  B, S = u.shape[0], 16
  ci = torch.full((B, S), -1, dtype=torch.int64)
  cq = torch.zeros(B, S)
  cn = torch.zeros(B, S, dtype=torch.int64)
  for k in range(M - 2):
    ci[:, k], cn[:, k], cq[:, k] = k, 100 - 10 * k, 0.5 - 0.2 * k
  u = u.clone().requires_grad_(True)
  loss, _ = move_token_value_order_loss(u, sel, valid, pt, mvp, topk=5, child=(ci, cq, cn))
  loss.backward()
  assert u.grad is not None and torch.isfinite(u.grad).all() and u.grad.abs().sum() > 0
  # Dead tokens must receive no gradient.
  assert u.grad[:, -2:].abs().max() == 0, 'masked tokens received gradient'
  print('  gradient OK: finite, non-zero, and none reaches the masked tokens')


if __name__ == '__main__':
  control_arm_unchanged()
  child_q_ranks_by_value()
  promotion_tiebreak()
  unvisited_excluded()
  gradient_flows()
  print('ALL OK')
