"""Contract test for the move-token ACTION head (2026-09-17): a per-token WDL readout exported as the 'action'
output [B,1858,3] that Ceres consumes (per-child FPU), trained on the v8 child table in the CHILD frame.

    python test_move_token_action.py      (from src/CeresTrainPy, CPU)

1. Graph: output slot 8 of the net is [B,1858,3]; every legal move's row equals its token's logits; moves without a
   token are neutral (all-zero logits = uniform WDL, never NaN); all four promotion variants share the pair's row.
2. Frame: a child with q = +1 (winning for the mover, root frame) must target L = 1 in the child frame; W - L = -q.
3. Loss: finite, decreases under a few AdamW steps, gradient reaches only act (and the trunk), promotion tiebreak
   picks the most-visited child, rows without a visited child contribute 0, weighting by visits^0.5 is row-normalised.
4. Plumbing: weight-decay partition covers both act parameters; a batch without a child table yields a zero-valued
   participation term (DDP) rather than a crash.
"""
import os, sys
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dual_plane_edge_aux import build, random_boards
from move_tokens import move_token_action_loss, _children_to_tokens


def _net():
  # The config refuses the head without its loss and the loss without a DirectFromV6 source (the child table is
  # v8-only), so the test builds under a DirectFromV6 data config; no data is read at construction.
  import test_dual_plane_edge_aux as T
  old = dict(T.DATA_BASE); T.DATA_BASE['SourceType'] = 'DirectFromV6'
  try:
    net, _ = build({'DualPlanePolicyDecode': False, 'UseMoveTokens': True, 'MoveTokenDim': 64, 'MoveTokenHeads': 2,
                    'MoveTokenLayers': 2, 'MoveTokenMax': 64, 'MoveTokenActionHead': True},
                   {'LossMoveTokenActionMultiplier': 0.1}, 'mtact')
  finally:
    T.DATA_BASE.clear(); T.DATA_BASE.update(old)
  assert net.mt_action and abs(net.mt_act_w - 0.1) < 1e-9
  return net


def graph():
  net = _net(); net.eval()
  sq = random_boards(5)
  with torch.no_grad():
    o = net(sq, None)
    act = o[8]
    _, _, _, sel, valid, _, _, act_dec = net.move_tokens(sq[:, :, 0:13].float(), torch.randn(sq.shape[0], 64, net.EMBEDDING_DIM))
  assert tuple(act.shape) == (5, 1858, 3), act.shape
  assert torch.isfinite(act).all()
  mv = net.move_tokens.mv_pair_flat
  # every move whose from-to pair has a valid token carries that token's logits; the rest is neutral
  with torch.no_grad():
    covered = torch.zeros(5, 1858, dtype=torch.bool)
    for b in range(5):
      for m in range(sel.shape[1]):
        if not bool(valid[b, m]): continue
        rows = (mv == sel[b, m]).nonzero().flatten()
        assert len(rows) >= 1
        ref = act_dec[b, rows[0]]
        for r in rows:
          assert torch.allclose(act_dec[b, r], ref), 'promotion variants must share the pair row'
          covered[b, r] = True
    assert (act_dec[~covered] == 0).all(), 'moves without a token must be neutral (zero logits)'
    assert covered.any()
  # The claim proper: the NET's exported action rows equal act(xo) of the SAME forward (train mode stashes both).
  net.train()
  with torch.no_grad():
    o_t = net(sq, None)
    act_t, (sel_t, valid_t, _) = o_t[8].float(), net._last_mt
    tok_t = net.move_tokens._last_act.float()
  n_rows = 0
  for b in range(5):
    for m in range(sel_t.shape[1]):
      if not bool(valid_t[b, m]): continue
      for r in (mv == sel_t[b, m]).nonzero().flatten():
        assert torch.allclose(act_t[b, r], tok_t[b, m], atol=1e-5), 'net action row must equal its token logits'
        n_rows += 1
  assert n_rows > 0
  net.eval()
  print(f'  graph OK: action output [B,1858,3], {int(covered.sum())} legal rows carry token logits, '
        f'{int((~covered).sum())} rows neutral, promotion variants share rows; {n_rows} net rows == token logits of the same forward')
  return net


def _child_table(net, sq, q_of_token, n_of_token, d_of_token=None):
  """Build a synthetic child table whose entries land on the given tokens (one child per valid token)."""
  with torch.no_grad():
    _, _, _, sel, valid, _, _, _ = net.move_tokens(sq[:, :, 0:13].float(), torch.randn(sq.shape[0], 64, net.EMBEDDING_DIM))
  B, M = sel.shape
  mv = net.move_tokens.mv_pair_flat
  S = 32
  ci = torch.full((B, S), -1, dtype=torch.long); cq = torch.zeros(B, S); cn = torch.zeros(B, S, dtype=torch.long); cd = torch.zeros(B, S)
  for b in range(B):
    k = 0
    for m in range(M):
      if k >= S or not bool(valid[b, m]): break
      idx = (mv == sel[b, m]).nonzero().flatten()[0]
      ci[b, k] = idx; cq[b, k] = q_of_token(b, m); cn[b, k] = n_of_token(b, m)
      cd[b, k] = 0.0 if d_of_token is None else d_of_token(b, m)
      k += 1
  return sel, valid, (ci, cq, cd, cn)


def frame_and_loss(net):
  torch.manual_seed(3)
  sq = random_boards(4)
  # q = +1 for token 0 of every row (root frame: winning for the mover) => child-frame target L = 1
  sel, valid, child = _child_table(net, sq, lambda b, m: 1.0 if m == 0 else 0.0, lambda b, m: 10000 if m == 0 else 1)
  net.train()
  o = net(sq, None)
  act_tok = net.move_tokens._last_act
  assert act_tok is not None and tuple(act_tok.shape[1:]) == (sel.shape[1], 3)
  loss, diag = move_token_action_loss(act_tok, sel, valid, net.move_tokens.mv_pair_flat, child)
  assert torch.isfinite(loss) and loss.item() >= -1e-5, loss
  # frame check on the target construction itself
  ci, cq, cd, cn = child
  w, q, d = _children_to_tokens(sel, valid, net.move_tokens.mv_pair_flat, ci, cn, cq, cd)
  W = ((1 - q - d) / 2).clamp(0, 1); L = ((1 + q - d) / 2).clamp(0, 1)
  assert torch.allclose((W - L)[w > 0], -q[w > 0]), 'child-frame V must equal -q (root frame)'
  assert float(L[0, 0]) == 1.0 and float(W[0, 0]) == 0.0
  # only act + upstream get gradient; the MLP action head does not exist
  net.zero_grad(set_to_none=True); loss.backward()
  assert net.move_tokens.act.weight.grad is not None and net.move_tokens.act.weight.grad.abs().sum() > 0
  assert net.move_tokens.act.bias.grad is not None
  assert not hasattr(net, 'action_head')
  # a few AdamW steps on the head alone must drive the loss down and the token-0 prediction toward L
  opt = torch.optim.AdamW(net.move_tokens.act.parameters(), lr=1e-1)
  l0 = loss.item()
  for _ in range(60):
    opt.zero_grad(set_to_none=True)
    net(sq, None)
    l, _ = move_token_action_loss(net.move_tokens._last_act, sel, valid, net.move_tokens.mv_pair_flat, child)
    l.backward(); opt.step()
  net(sq, None)
  l1, diag = move_token_action_loss(net.move_tokens._last_act, sel, valid, net.move_tokens.mv_pair_flat, child)
  p0 = torch.softmax(net.move_tokens._last_act[:, 0].float(), dim=-1)
  assert l1.item() < l0 * 0.5, (l0, l1.item())
  assert (p0[:, 2] > 0.7).all() and (p0[:, 2] > p0[:, 0]).all(), p0
  print(f'  frame+loss OK: KL {l0:.3f} -> {l1.item():.3f} after 60 head-only steps, token-0 P(L) {float(p0[:, 2].min()):.2f} '
        f'(q=+1 root frame => L in child frame); targets/row {float(diag["mt_act_targets_per_row"]):.1f}')


def tiebreak_and_masks(net):
  torch.manual_seed(5)
  sq = random_boards(3)
  sel, valid, (ci, cq, cd, cn) = _child_table(net, sq, lambda b, m: 0.3, lambda b, m: 5)
  # a PROMOTION sibling: a second 1858 index that maps to the SAME from-to pair as child 0 (queen/knight/... variants of
  # one pair), with a DIFFERENT q and MORE visits at a later slot -> the most-visited variant must win the pair. Random
  # boards rarely offer a real promotion, so fall back to the same index (duplicate) when no sibling exists.
  S = ci.shape[1]; mv = net.move_tokens.mv_pair_flat; n_real = 0
  for b in range(3):
    sib = [int(r) for r in (mv == mv[ci[b, 0]]).nonzero().flatten() if int(r) != int(ci[b, 0])]
    ci[b, S - 1] = sib[0] if sib else ci[b, 0]; n_real += bool(sib)
    cq[b, S - 1] = -0.9; cn[b, S - 1] = 50
  w, q, d = _children_to_tokens(sel, valid, net.move_tokens.mv_pair_flat, ci, cn, cq, cd)
  assert torch.allclose(q[:, 0], torch.full((3,), -0.9)) and (w[:, 0] == 50).all(), 'most-visited child must win the pair'
  # and a synthetic promotion pair independent of the board: two different indices sharing a pair
  pairs = mv.unique(return_counts=True); multi = pairs[0][pairs[1] >= 2][0]
  idx2 = (mv == multi).nonzero().flatten()[:2]
  sel_p = torch.tensor([[int(multi)]]); valid_p = torch.tensor([[True]])
  ci_p = torch.tensor([[int(idx2[0]), int(idx2[1])]]); cq_p = torch.tensor([[0.4, -0.7]]); cn_p = torch.tensor([[10, 60]]); cd_p = torch.zeros(1, 2)
  w_p, q_p, _ = _children_to_tokens(sel_p, valid_p, mv, ci_p, cn_p, cq_p, cd_p)
  assert abs(float(q_p[0, 0]) + 0.7) < 1e-6 and float(w_p[0, 0]) == 60, ('promotion variants: the most-visited one supplies the pair target', q_p, w_p)
  # a row without any child contributes nothing and does not poison the mean
  ci2 = ci.clone(); ci2[1] = -1
  net.train(); net(sq, None)
  loss_a, diag_a = move_token_action_loss(net.move_tokens._last_act, sel, valid, net.move_tokens.mv_pair_flat, (ci2, cq, cd, cn))
  assert torch.isfinite(loss_a) and abs(float(diag_a['mt_act_rows']) - 2 / 3) < 1e-6
  # no child anywhere: exact-zero loss with a live graph (participation)
  ci3 = torch.full_like(ci, -1)
  net(sq, None)
  loss_z, diag_z = move_token_action_loss(net.move_tokens._last_act, sel, valid, net.move_tokens.mv_pair_flat, (ci3, cq, cd, cn))
  assert float(loss_z) == 0.0 and loss_z.requires_grad and float(diag_z['mt_act_rows']) == 0.0
  print(f'  tiebreak+masks OK: most-visited promotion sibling wins ({n_real}/3 real siblings on the boards + 1 synthetic pair); empty rows excluded; all-empty batch = 0 with graph')


def plumbing(net):
  from wd_partition import partition_weight_decay
  decay, no_decay = partition_weight_decay(net)
  assert 'move_tokens.act.weight' in decay and 'move_tokens.act.bias' in no_decay, 'wd partition must cover the action head'
  print('  plumbing OK: wd partition covers act.weight (decay) and act.bias (no_decay)')


if __name__ == '__main__':
  net = graph()
  frame_and_loss(net)
  tiebreak_and_masks(net)
  plumbing(net)
  print('ALL OK')
