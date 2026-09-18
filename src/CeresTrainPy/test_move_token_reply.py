"""Contract test for REPLY SUPERVISION of the opponent keys (move_tokens.py, 2026-09-18).

    python test_move_token_reply.py [/path/to/v8.tar]     (from src/CeresTrainPy, CPU)

1. Frame: LC0 move indices are side-to-move framed, so a reply recorded in the child position sits on the ROOT board
   mirrored (square ^ 56). mirror_pair is an involution and maps e2e4 <-> e7e5-style pairs.
2. Synthetic: with opponent tokens on, a child table whose reply is a valid opponent token gives coverage 1, the CE and
   the reply-q Huber are finite, gradients reach the last block's qkv and rq_head only through the reply path, and a
   few steps on the head-0 scores drive top-1 accuracy up.
3. Masks: a reply outside the Mo opponent tokens is excluded (coverage < 1, no NaN); an all-dead table gives 0 with graph.
4. Optional real corpus: the fraction of recorded replies that land on a root-board opponent candidate is high WITH the
   mirror and low WITHOUT it (this pins the frame on data, not on reasoning).
"""
import os, sys
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dual_plane_edge_aux import build, random_boards
from move_tokens import mirror_pair, reply_targets, move_token_reply_loss

MO = 24


def _net(reply=True):
  import test_dual_plane_edge_aux as T
  old = dict(T.DATA_BASE); T.DATA_BASE['SourceType'] = 'DirectFromV6'
  try:
    net, _ = build({'DualPlanePolicyDecode': False, 'UseMoveTokens': True, 'MoveTokenDim': 64, 'MoveTokenHeads': 2,
                    'MoveTokenLayers': 2, 'MoveTokenMax': 64, 'MoveTokenOppMax': MO, 'MoveTokenReplySup': reply},
                   {'LossMoveTokenReplyMultiplier': 0.1, 'LossMoveTokenReplyQMultiplier': 0.1} if reply else {}, 'mtrep' if reply else 'mtopp')
  finally:
    T.DATA_BASE.clear(); T.DATA_BASE.update(old)
  return net


def frame():
  p = torch.tensor([12 * 64 + 28, 0])                       # e2e4 (12->28) and a1a1
  m = mirror_pair(p)
  assert int(m[0]) == 52 * 64 + 36, m                        # e7e5 (52->36)
  assert torch.equal(mirror_pair(m), p), 'mirror must be an involution'
  print('  frame OK: e2e4 <-> e7e5 under the rank mirror, involution holds')


def head0_equality(net, sq, rl):
  """The supervised logits must equal head 0 of the last block's OWN attention scores over [own keys | opponent keys]."""
  dec = net.move_tokens; blk = dec.blocks[-1]; cap = {}
  def pre(mod, args, kwargs):
    cap['x'] = args[0]; cap['key_bias'] = args[2]; cap['x_opp'] = kwargs.get('x_opp'); cap['opp_bias'] = kwargs.get('opp_bias')
  h = blk.register_forward_pre_hook(pre, with_kwargs=True)
  try:
    net.train(); net(sq, None)
  finally:
    h.remove()
  rl2 = net.move_tokens._last_reply[0]
  dm = cap['x'].shape[-1]; dk = blk.dk; B = cap['x'].shape[0]
  q, k, v = blk.qkv(blk.ln1(cap['x'])).chunk(3, dim=-1)
  ko, vo = torch.nn.functional.linear(blk.ln1(cap['x_opp']), blk.qkv.weight[dm:]).chunk(2, dim=-1)
  k = torch.cat([k, ko], dim=1)
  qh = q.reshape(B, -1, blk.h, dk).transpose(1, 2)[:, 0]
  kh = k.reshape(B, -1, blk.h, dk).transpose(1, 2)[:, 0]
  s = torch.matmul(qh, kh.transpose(1, 2)) * (dk ** -0.5) + torch.cat([cap['key_bias'], cap['opp_bias']], dim=-1).reshape(B, 1, -1)
  assert torch.allclose(rl2, s.float(), atol=1e-5), float((rl2 - s).abs().max())
  print('  head-0 equality OK: supervised logits == the last block\'s head-0 scores over [own | opponent] keys')


def _forward(net, sq):
  net.train()
  net(sq, None)
  rl, rqp, sel_o, valid_o = net.move_tokens._last_reply
  sel, valid, _ = net._last_mt
  return rl, rqp, sel, valid, sel_o, valid_o


def synthetic():
  torch.manual_seed(7)
  net = _net(); sq = random_boards(4)
  rl, rqp, sel, valid, sel_o, valid_o = _forward(net, sq)
  B, M = sel.shape; assert rl.shape == (B, M, M + MO) and rqp.shape == (B, M)
  mv = net.move_tokens.mv_pair_flat
  # child table: one child per valid own token; its reply = the FIRST valid opponent token, expressed in the child frame
  S = 40
  ci = torch.full((B, S), -1, dtype=torch.long); cn = torch.zeros(B, S, dtype=torch.long)
  crep = torch.full((B, S), -1, dtype=torch.long); crq = torch.full((B, S), -2.0)
  want = torch.full((B, M), -1, dtype=torch.long)
  for b in range(B):
    j = int(valid_o[b].nonzero().flatten()[0]); opp_pair = int(sel_o[b, j])
    child_pair = int(mirror_pair(torch.tensor(opp_pair)))
    rep_idx = int((mv == child_pair).nonzero().flatten()[0])
    k = 0
    for m in range(M):
      if k >= S or not bool(valid[b, m]): break
      ci[b, k] = int((mv == sel[b, m]).nonzero().flatten()[0]); cn[b, k] = 10 + m
      crep[b, k] = rep_idx; crq[b, k] = 0.3; want[b, m] = M + j; k += 1
  child = (ci, cn, crep, crq)
  tgt, rq, w, live, has = reply_targets(sel, valid, sel_o, valid_o, mv, child)
  assert torch.equal(tgt[live], want[live]) and bool(has[live].all()), 'every live token must map to the first opp token (joint index M+j)'
  head0_equality(net, sq, rl)
  ce, rql, d = move_token_reply_loss(rl, rqp, sel, valid, sel_o, valid_o, mv, child)
  assert torch.isfinite(ce) and torch.isfinite(rql) and abs(float(d['mt_rep_coverage']) - 1.0) < 1e-6
  net.zero_grad(set_to_none=True); (ce + rql).backward()
  last = net.move_tokens.blocks[-1]
  assert last.qkv.weight.grad is not None and last.qkv.weight.grad.abs().sum() > 0
  assert net.move_tokens.rq_head.weight.grad is not None and net.move_tokens.rq_head.weight.grad.abs().sum() > 0
  assert net.move_tokens.blocks[0].qkv.weight.grad is not None, 'earlier blocks participate through x_last_in'
  # a few AdamW steps on the decoder must raise top-1 and lower the losses
  opt = torch.optim.AdamW(net.move_tokens.parameters(), lr=3e-3)
  t0 = float(d['mt_rep_top1']); c0 = float(ce)
  for _ in range(30):
    opt.zero_grad(set_to_none=True)
    rl, rqp, sel, valid, sel_o, valid_o = _forward(net, sq)
    ce, rql, d = move_token_reply_loss(rl, rqp, sel, valid, sel_o, valid_o, mv, child)
    (ce + rql).backward(); opt.step()
  assert float(ce) < c0 * 0.5 and float(d['mt_rep_top1']) > 0.9, (c0, float(ce), float(d['mt_rep_top1']))
  assert float(d['mt_rep_joint_mass']) > 0.5, 'the joint softmax must actually put mass on the reply'
  print(f'  synthetic OK: coverage 1.0, CE {c0:.3f} -> {float(ce):.3f}, top-1 {t0:.2f} -> {float(d["mt_rep_top1"]):.2f}, '
        f'rq MAE {float(d["mt_rep_rq_mae"]):.3f}; grads reach last-block qkv + rq_head')
  # masks: a reply that is NOT an opponent candidate (own move index) is excluded; all-dead table = 0 with graph
  crep2 = ci.clone()
  _, _, _, live2, has2 = reply_targets(sel, valid, sel_o, valid_o, mv, (ci, cn, crep2, crq))
  assert bool(live2.any()) and float(has2.float().sum()) < float(live2.float().sum()), 'own-move indices must mostly miss the opponent tokens'
  ce3, rql3, d3 = move_token_reply_loss(rl, rqp, sel, valid, sel_o, valid_o, mv, (torch.full_like(ci, -1), cn, crep, crq))
  assert float(ce3) == 0.0 and float(rql3) == 0.0 and ce3.requires_grad and float(d3['mt_rep_coverage']) == 0.0
  print('  masks OK: replies outside the opponent tokens are excluded, all-dead batch = 0 with graph')
  # a net WITHOUT reply supervision must not stash anything
  n2 = _net(reply=False); n2.train(); n2(sq, None)
  assert getattr(n2.move_tokens, '_last_reply', None) is None
  print('  off-switch OK')


def real(path, ngames=40):
  """Coverage of recorded replies among the root-board opponent candidates, mirrored vs unmirrored."""
  import numpy as np, gzip, tarfile
  from test_v8_dataset import _bare
  d = _bare(); d.skip_count = 1
  net = _net(); dec = net.move_tokens; dec.eval()
  hit_m = hit_u = tot = 0
  with tarfile.open(path) as tf:
    g = 0
    for m in tf:
      if not m.isfile() or not m.name.endswith('.gz'): continue
      recs = d._decode_chunk(gzip.decompress(tf.extractfile(m).read()))
      if recs is None: continue
      out = d._records_to_arrays(recs)
      squares, v8x = out[9], out[14]
      sq = torch.tensor(np.asarray(squares)).float()
      for r in range(len(sq)):
        s13 = sq[r:r + 1, :, 0:13]
        with torch.no_grad():
          _, E1 = dec.candidates(s13)
          cand_o = dec.candidates_opp(s13, E1)[0] > 0.5                       # [4096] bool, root frame
        ci, crep = v8x.child_idx[r], v8x.child_reply[r]
        for k in range(len(ci)):
          if ci[k] < 0 or crep[k] < 0: continue
          pair = int(dec.mv_pair_flat[int(crep[k])])
          hit_m += bool(cand_o[int(mirror_pair(torch.tensor(pair)))]); hit_u += bool(cand_o[pair]); tot += 1
      g += 1
      if g >= ngames: break
  assert tot > 0, 'no replies decoded'
  print(f'  real OK: {tot} recorded replies; on a root-board opponent candidate: mirrored {hit_m / tot:.1%}, unmirrored {hit_u / tot:.1%}')
  assert hit_m / tot > 0.7 and hit_m > 3 * hit_u, 'the mirrored frame must dominate'


if __name__ == '__main__':
  frame(); synthetic()
  if len(sys.argv) >= 2:
    real(sys.argv[1])
  print('ALL OK')
