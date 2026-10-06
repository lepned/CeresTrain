"""Contract tests for the child-q distribution heads (childq_dist.py) + Grill aux head + LossSTValueMultiplier (2026-10-07).

Run from src/CeresTrainPy:

    python test_childq_dist.py            (CPU, synthetic)

1. HL-Gauss atom target vs an independent float64 re-implementation of Kovax' _build_target (atom grid, +-inf outer
   cells, edge taper toward two-hot); sums to 1; mean-preserving at the edges (two-hot).
2. child_logits (gather over stored children) == the full [64,64,K] construction Kovax uses, including the three
   q/r/b promotion offsets and the position bias, for random NONZERO parameters.
3. childq_dist_loss == a float64 per-position / per-child reference loop (abs and gap targets, visit weights, rows
   without valid children excluded); mutation guards: wrong gap sign and dropped visit weights are caught.
4. Net level: the ctrl net and the cq + grill net share EVERY parameter outside cq_*/grill_head bit for bit (forked
   fixed-seed init); training forward + loss is finite, the stash is consumed, gradients reach the trunk; child-less
   batches give a zero participation term; eval forward has no stash and equals the control; the weight-decay
   partition covers the new raw parameters.
5. Config: the retired CERES_STVALUE_WEIGHT env raises; LossSTValueMultiplier reaches the net; cq / grill aux on a
   non-v8 SourceType raise.
"""
import os, sys, math, json, tempfile, shutil

os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from childq_dist import hlgauss_atom_target, ChildQDistHead, childq_dist_loss, move_tables_1858
from lc0_moves_1858 import MOVES_1858


# ---------------------------------------------------------------------------------------------------------------- 1
def ref_target(t, K, sr=0.75, taper=0.1875):
  from math import erf, sqrt, floor
  ndtr = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))
  sp = 2.0 / (K - 1); sig = sr * sp
  sup = [-1.0 + i * sp for i in range(K)]
  cdf = [0.0] + [ndtr(((sup[i] + sup[i + 1]) / 2 - t) / sig) for i in range(K - 1)] + [1.0]
  tg = np.array([cdf[i + 1] - cdf[i] for i in range(K)])
  if taper > 0:
    pos = min(max((t + 1.0) / sp, 0.0), K - 1.0)
    lo = min(max(int(floor(pos)), 0), K - 2); fr = pos - lo
    th = np.zeros(K); th[lo] += 1 - fr; th[lo + 1] += fr
    a = min(max((1.0 - abs(t)) / taper, 0.0), 1.0)
    tg = a * tg + (1 - a) * th
  return tg / tg.sum()


def test_target():
  K = 33
  ts = torch.tensor([-1.0, -0.999, -0.95, -0.5, -0.03, 0.0, 0.0312, 0.4, 0.83, 0.9, 1.0])
  got = hlgauss_atom_target(ts, K).double().numpy()
  for i, t in enumerate(ts.tolist()):
    r = ref_target(t, K)
    assert np.allclose(got[i], r, atol=1e-6), (t, np.abs(got[i] - r).max())
  assert np.allclose(got.sum(1), 1.0, atol=1e-6)
  sup = np.linspace(-1, 1, K)
  for i, t in enumerate(ts.tolist()):
    if abs(t) >= 1 - 1e-9:
      assert abs((got[i] * sup).sum() - t) < 1e-6, 'two-hot at the edge must be mean-preserving'
  print('OK target: matches float64 reference of the atom-grid HL-Gauss + edge taper; edges mean-preserving')


# ---------------------------------------------------------------------------------------------------------------- 2
def full_logits(head, stash):
  """Kovax' construction: [64,64,K] base + 8x24 promo block, mapped to 1858 by move string. [B, 1858, K]."""
  q, k, pos = stash
  B = q.shape[0]; K = head.num_bins
  base = torch.einsum('nid,dc,njd->nijc', q, head.w_bins, k) / math.sqrt(head.d_model)       # [B,64,64,K]
  off = torch.einsum('btd,dpk->bptk', k, head.w_promo)                                        # [B,4,64,K]
  off = off[:, :3] + off[:, 3:4]                                                              # [B,3,64,K]
  sq = lambda s: (ord(s[0]) - 97) + 8 * (int(s[1]) - 1)
  out = torch.zeros(B, 1858, K)
  for i, m in enumerate(MOVES_1858):
    f, t = sq(m[:2]), sq(m[2:4])
    v = base[:, f, t]
    if len(m) == 5:
      v = v + off[:, 'qrb'.index(m[4]), t]       # lc0 1858: plain entry = knight promotion
    out[:, i] = v
  if pos is not None:
    out = out + pos.unsqueeze(1)
  return out


def test_gather_vs_full():
  torch.manual_seed(0)
  B, S, K = 3, 40, 33
  head = ChildQDistHead(16, 24, 8, K, nn.Mish(), position_bias=True)
  with torch.no_grad():
    for p in (head.w_bins, head.w_promo, head.pos_w, head.pos_b):
      p.normal_()
  x = torch.randn(B, 64, 16)
  stash = head(x)
  tables = move_tables_1858()
  promo_moves = [i for i, m in enumerate(MOVES_1858) if len(m) == 5]
  assert len(promo_moves) == 66
  ci = torch.stack([torch.randperm(1858)[:S] for _ in range(B)])
  ci[0, :6] = torch.tensor(promo_moves[:6])                                                   # make sure promos are hit
  ci[1, -5:] = -1
  got = head.child_logits(stash, ci, tables)
  ref = full_logits(head, stash)
  ref_g = torch.gather(ref, 1, ci.clamp_min(0).unsqueeze(-1).expand(-1, -1, K))
  live = (ci >= 0).unsqueeze(-1)
  assert torch.allclose(got * live, ref_g * live, atol=1e-4), (got - ref_g).abs().max()
  # zero-init: uniform logits whatever the trunk
  h0 = ChildQDistHead(16, 24, 8, K, nn.Mish())
  lg0 = h0.child_logits(h0(x), ci, tables)
  assert torch.equal(lg0, torch.zeros_like(lg0)), 'zero-init bins must give exactly uniform logits at step 0'
  print('OK gather path == full 64x64(+promo, +pos) construction; zero-init => uniform')


# ---------------------------------------------------------------------------------------------------------------- 3
def ref_loss(logits, ci, cq, cn, gap, kappa=2.0):
  B, S, K = logits.shape
  per = []
  for b in range(B):
    val = [s for s in range(S) if ci[b, s] >= 0]
    if not val:
      continue
    q0 = max(float(cq[b, s]) for s in val)
    num = den = 0.0
    for s in val:
      q = float(cq[b, s])
      t = (min(max(q - q0, -2.0), 0.0) + 1.0) if gap else min(max(q, -1.0), 1.0)
      tg = ref_target(t, K)
      lp = logits[b, s].double().numpy(); lp = lp - np.log(np.exp(lp - lp.max()).sum()) - lp.max()
      kl = float((tg * (np.log(np.clip(tg, 1e-12, None)) - lp)).sum())
      c = float(cn[b, s]) / (float(cn[b, s]) + kappa)
      num += c * kl; den += c
    per.append(num / max(den, 1e-6))
  return sum(per) / max(len(per), 1)


def test_loss():
  torch.manual_seed(1)
  B, S, K = 5, 12, 33
  logits = torch.randn(B, S, K)
  ci = torch.randint(0, 1858, (B, S)); ci[2] = -1; ci[3, 7:] = -1                            # row 2 has no child
  cq = torch.rand(B, S) * 2 - 1
  cn = torch.randint(0, 400, (B, S))
  for gap in (False, True):
    got, d = childq_dist_loss(logits, ci, cq, cn, gap=gap)
    ref = ref_loss(logits, ci, cq, cn, gap)
    assert abs(float(got) - ref) < 1e-4, (gap, float(got), ref)
    assert abs(float(d['rows']) - 0.8) < 1e-6
  # mutation guards
  wrong_gap, _ = childq_dist_loss(logits, ci, -cq, cn, gap=True)
  assert abs(float(wrong_gap) - ref_loss(logits, ci, cq, cn, True)) > 1e-3, 'a gap-sign error must change the loss'
  flat, _ = childq_dist_loss(logits, ci, cq, torch.full_like(cn, 10 ** 6), gap=False)
  assert abs(float(flat) - ref_loss(logits, ci, cq, cn, False)) > 1e-4, 'dropped visit weights must change the loss'
  # gradient only through logits; nothing at all from rows without children
  lg = logits.clone().requires_grad_(True)
  l, _ = childq_dist_loss(lg, ci, cq, cn, gap=False)
  l.backward()
  assert lg.grad[2].abs().sum() == 0 and lg.grad[3, 7:].abs().sum() == 0, 'masked slots must get no gradient'
  print('OK loss: abs/gap match the float64 reference loop; mutation guards fire; masked slots get no gradient')


# ---------------------------------------------------------------------------------------------------------------- 4
NET_BASE = {
  "ModelDim": 64, "NumLayers": 2, "NumHeads": 4, "PreNorm": False, "NormType": "RMSNorm",
  "FFNMultiplier": 2, "FFNActivationType": "Mish", "HeadsActivationType": "Mish",
  "NonLinearAttention": False, "SoftCapCutoff": 100,
  "SmolgenDimPerSquare": 0, "SmolgenDim": 0, "SmolgenToHeadDivisor": 1, "SmolgenActivationType": "Swish",
  "UseRPE": False, "UseRPE_V": False, "UseRoPE": False,
}
OPT_BASE = {
  "NumTrainingPositions": 1000, "BatchSizeForwardPass": 4, "BatchSizeBackwardPass": 4,
  "Optimizer": "Muon", "LearningRateBase": 1e-4, "LossValueMultiplier": 1.0,
  "LossValue2Multiplier": 0.0, "LossPolicyMultiplier": 1, "LossMLHMultiplier": 0,
  "LossUNCMultiplier": 0, "LossUncertaintyPolicyMultiplier": 0, "LossValueDMultiplier": 0,
  "TorchSeed": 777, "TPGV3": 0, "AuxFeaturesPerSquare": 0,
}
DATA_V6 = {"SourceType": "DirectFromV6", "TrainingFilesDirectory": "/none", "FractionQ": 1, "WDLLabelSmoothing": 0,
           "V6SkipCount": 1}
DATA_GEN = {"SourceType": "DirectFromPositionGenerator", "TrainingFilesDirectory": "/none", "FractionQ": 1,
            "WDLLabelSmoothing": 0}
EXEC_BASE = {"ID": "cq_test", "DeviceType": "cpu", "DeviceIDs": [0], "DataType": "Float32",
             "UseHistory": True, "DropoutRate": 0, "EngineType": "CSharpViaTorchscript"}
CQ_ON = {"ChildQDistAbsWeight": 1.0, "ChildQDistGapWeight": 1.0, "ChildQDistDim": 16, "GrillAuxHeadWeight": 1.0}


def build(opt_over, tag, data=DATA_V6):
  from config import Configuration
  from ceres_net import CeresNet
  d = tempfile.mkdtemp(prefix='cq_')
  try:
    for suf, obj in (('net', NET_BASE), ('opt', {**OPT_BASE, **opt_over}), ('data', data),
                     ('exec', {**EXEC_BASE, 'ID': tag}), ('monitoring', {})):
      with open(os.path.join(d, f'{tag}_ceres_{suf}.json'), 'w') as f:
        json.dump(obj, f)
    cfg = Configuration(d, tag)
    torch.manual_seed(int(OPT_BASE['TorchSeed']))
    m = CeresNet(None, cfg,
                 policy_loss_weight=cfg.Opt_LossPolicyMultiplier, value_loss_weight=cfg.Opt_LossValueMultiplier,
                 moves_left_loss_weight=cfg.Opt_LossMLHMultiplier, unc_loss_weight=cfg.Opt_LossUNCMultiplier,
                 value2_loss_weight=cfg.Opt_LossValue2Multiplier, q_deviation_loss_weight=cfg.Opt_LossQDeviationMultiplier,
                 value_diff_loss_weight=cfg.Opt_LossValueDMultiplier, value2_diff_loss_weight=cfg.Opt_LossValue2DMultiplier,
                 action_loss_weight=cfg.Opt_LossActionMultiplier, uncertainty_policy_weight=cfg.Opt_LossUncertaintyPolicyMultiplier,
                 action_uncertainty_loss_weight=cfg.Opt_LossActionUncertaintyMultiplier, q_ratio=cfg.Data_FractionQ)
    return m, cfg
  finally:
    shutil.rmtree(d, ignore_errors=True)


def fake_batch(B, child=True, seed=5):
  from config import NUM_INPUT_BYTES_PER_SQUARE
  g = torch.Generator().manual_seed(seed)
  sq = torch.zeros(B, 64, NUM_INPUT_BYTES_PER_SQUARE)
  sq[:, :, 0] = 1.0
  S = 10
  pol = torch.zeros(B, 1858)
  ci = torch.stack([torch.randperm(1858, generator=g)[:S] for _ in range(B)])
  for b in range(B):
    w = torch.rand(S, generator=g) + 0.05
    pol[b, ci[b]] = w / w.sum()
  wdl = torch.tensor([[0.4, 0.3, 0.3]]).expand(B, 3).clone()
  batch = {'policies': pol, 'wdl_deblundered': wdl, 'wdl_q': wdl, 'wdl_nondeblundered': wdl,
           'mlh': torch.zeros(B, 1), 'unc': torch.zeros(B, 1), 'uncertainty_policy': torch.zeros(B, 1),
           'q_deviation_lower': torch.zeros(B, 1), 'q_deviation_upper': torch.zeros(B, 1)}
  if child:
    n = torch.randint(5, 300, (B, S), generator=g)
    batch.update({'child_idx': ci, 'child_q': torch.rand(B, S, generator=g) * 2 - 1, 'child_n': n,
                  'child_ndef': n, 'stored_idx': ci.clone(),
                  'stored_prior': torch.rand(B, S, generator=g) + 0.01,
                  'root_q': torch.zeros(B, 1), 'orig_q': torch.full((B, 1), 0.1)})
  return sq, batch


def run_loss(m, batch, sq):
  from losses import LossCalculator
  lc = LossCalculator(nn.Linear(4, 4))
  outs = m(sq, None)
  (policy_out, value_out, mlh_out, unc_out, value2_out, qdl, qdu, unc_pol, action_out, _) = outs[:10]
  return m.compute_loss(lc, batch, policy_out, value_out, mlh_out, unc_out, value2_out, qdl, qdu, unc_pol,
                        None, None, None, action_out, None, 0, 0, 0, False)


def test_net():
  from wd_partition import partition_weight_decay
  ctrl, _ = build({}, 'ctrl')
  cq, _ = build(CQ_ON, 'cq')
  sd_c, sd_q = ctrl.state_dict(), cq.state_dict()
  extra = sorted(set(sd_q) - set(sd_c))
  assert extra and all(k.startswith(('cq_abs_head.', 'cq_gap_head.', 'grill_head.')) for k in extra), extra
  for k in sd_c:
    assert torch.equal(sd_c[k], sd_q[k]), f'bit-pairing broken at {k}'
  assert cq.cq_abs_head.w_bins.abs().sum() == 0 and cq.cq_gap_head.w_promo.abs().sum() == 0
  dec, nodec = partition_weight_decay(cq)
  assert 'cq_abs_head.w_bins' in nodec and 'cq_gap_head.pos_w' in nodec and 'cq_abs_head.tokens.weight' in dec
  print(f'  bit-pairing OK ({len(extra)} new params, all cq_*/grill_head); wd partition covers them')

  B = 4
  sq, batch = fake_batch(B)
  cq.train(); cq.zero_grad(set_to_none=True)
  loss = run_loss(cq, batch, sq)
  assert torch.isfinite(loss), loss
  assert cq._last_cq_abs is None and cq._last_cq_gap is None and getattr(cq, '_last_grill_out', None) is None, \
      'stash must be consumed'
  loss.backward()
  for n_ in ('cq_abs_head.w_bins', 'cq_gap_head.w_bins', 'grill_head.fcFinal.weight'):
    g = dict(cq.named_parameters())[n_].grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f'{n_} got no gradient'
  # trunk gradient through the cq head alone: zero the other heads' weights by using a loss on the cq stash only
  cq.zero_grad(set_to_none=True)
  with torch.no_grad():
    cq.cq_abs_head.w_bins.normal_()      # nonzero bins so the trunk is reachable
  _ = cq(sq, None)
  from childq_dist import childq_dist_loss as _cql
  lg = cq.cq_abs_head.child_logits(cq._last_cq_abs, batch['child_idx'], (cq.cq_from, cq.cq_to, cq.cq_promo))
  cq._last_cq_abs = cq._last_cq_gap = cq._last_grill_out = None
  l, _ = _cql(lg, batch['child_idx'], batch['child_q'], batch['child_n'], gap=False)
  l.backward()
  trunk = [p for n_, p in cq.named_parameters() if 'transformer_layer' in n_ and p.grad is not None and p.grad.abs().sum() > 0]
  assert trunk, 'cq loss must reach the trunk (attach)'
  print(f'  training loss {float(loss):.4f} finite; stash consumed; heads + trunk receive gradient')

  # child-less batch (TPG secondary): participation only, still finite and backward-able
  sq2, batch2 = fake_batch(B, child=False)
  cq.zero_grad(set_to_none=True)
  loss2 = run_loss(cq, batch2, sq2)
  loss_c = run_loss(ctrl.train(), batch2, sq2)
  assert torch.isfinite(loss2) and abs(float(loss2) - float(loss_c)) < 1e-4, (float(loss2), float(loss_c))
  loss2.backward()
  # DDP static_graph needs EVERY new param in the graph on child-less batches (review 2026-10-07 finding 1)
  _missing = [n_ for n_, p in cq.named_parameters()
              if n_.startswith(('cq_abs_head.', 'cq_gap_head.', 'grill_head.')) and p.grad is None]
  assert not _missing, f'params without grad on a child-less batch (DDP static graph would hang): {_missing}'
  print('  child-less batch: participation only (loss equals the control)')

  cq.eval(); ctrl.eval()
  with torch.no_grad():
    oq = cq(sq, None); oc = ctrl(sq, None)
  assert getattr(cq, '_last_cq_abs', None) is None and getattr(cq, '_last_grill_out', None) is None, 'eval must not stash'
  for i, (a, c) in enumerate(zip(oq, oc)):
    if torch.is_tensor(a):
      assert torch.equal(a, c), f'eval output {i} differs from the control'
  print('  eval: no stash, outputs bit-identical to the control')
  print('OK net level')


# ---------------------------------------------------------------------------------------------------------------- 5
def test_config():
  os.environ['CERES_STVALUE_WEIGHT'] = '0.5'
  try:
    build({}, 'env')
    raise SystemExit('FAIL: CERES_STVALUE_WEIGHT did not raise')
  except ValueError as e:
    assert 'retired' in str(e)
  finally:
    del os.environ['CERES_STVALUE_WEIGHT']
  m, _ = build({'LossSTValueMultiplier': 0.5}, 'st')
  assert m.stvalue_weight == 0.5 and hasattr(m, 'stvalue_head')
  for over in ({'ChildQDistAbsWeight': 1.0}, {'GrillAuxHeadWeight': 1.0}):
    try:
      build(over, 'rej', data=DATA_GEN)
      raise SystemExit(f'FAIL: {over} on a non-v8 source did not raise')
    except ValueError:
      pass
  for over in ({'ChildQDistAbsWeight': 1.0, 'ChildQDistBins': 32}, {'GrillAuxHeadWeight': 1.0, 'GrillAuxValue': 'best'}):
    try:
      build(over, 'rej2')
      raise SystemExit(f'FAIL: {over} did not raise')
    except ValueError:
      pass
  print('OK config: env retired, LossSTValueMultiplier wired, non-v8 / bad values rejected')


if __name__ == '__main__':
  test_target()
  test_gather_vs_full()
  test_loss()
  test_net()
  test_config()
