"""Smoke + contract tests for the dual-plane EDGE-AUX supervision (boelge 13 / P1,
2026-09-02, "Edge Stream Diagnosis").

Run from src/CeresTrainPy:

    python test_dual_plane_edge_aux.py

Contracts checked (tiny 64-dim, 2-layer net, CPU, random legal-ish boards):
  1. construction: withhold shrinks the plane's channel count; aux head shapes;
     loud rejections for the silent-no-op configs (detach without aux, rel-aux
     without withhold, aux without a learned edge state).
  2. bit-pairing: with the same TorchSeed, the aux-on net and the withheld
     control share EVERY parameter except dp_eaux_* (fixed-key init draws
     nothing from the global RNG stream).
  3. training forward + loss: finite losses, pi target sums to 1 with the null
     bucket, rel targets are {0,1}; gradients reach the plane's edge-update
     weights (eu_out) with detach=False and do NOT with detach=True (probe).
  4. eval forward: no stash, outputs bit-identical to the withheld control
     (training-only head; serving graph untouched).
"""
import os, sys, json, tempfile, shutil

os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NET_BASE = {
  "ModelDim": 64, "NumLayers": 2, "NumHeads": 4, "PreNorm": False, "NormType": "RMSNorm",
  "FFNMultiplier": 2, "FFNActivationType": "Mish", "HeadsActivationType": "Mish",
  "NonLinearAttention": False, "SoftCapCutoff": 100,
  "SmolgenDimPerSquare": 0, "SmolgenDim": 0, "SmolgenToHeadDivisor": 1, "SmolgenActivationType": "Swish",
  "UseRPE": False, "UseRPE_V": False, "UseRoPE": False,
  "UseVisEdgeBias": False, "VisEdgeFamilies": "vis,xray,pinray,check,flight", "VisEdgeGates": "",
  "UseDualPlane": True, "DualPlanePolicyDecode": True, "DualPlaneRelDegrees": True,
  "DualPlaneLayers": 2, "DualPlaneSoftMinHeads": 1, "DualPlaneDim": 64,
  "DualPlaneEdgeUpdate": True, "DualPlaneEdgeToTrunk": True, "DualPlaneEdgeToTrunkMask": True,
}
OPT_BASE = {
  "NumTrainingPositions": 1000, "BatchSizeForwardPass": 4, "BatchSizeBackwardPass": 4,
  "Optimizer": "Muon", "LearningRateBase": 1e-4, "LossValueMultiplier": 1.0,
  "LossValue2Multiplier": 0.0, "LossPolicyMultiplier": 1, "LossMLHMultiplier": 0,
  "LossUNCMultiplier": 0, "LossUncertaintyPolicyMultiplier": 0, "LossValueDMultiplier": 0,
  "TorchSeed": 777, "TPGV3": 1, "AuxFeaturesPerSquare": 0,
}
DATA_BASE = {"SourceType": "DirectFromPositionGenerator", "TrainingFilesDirectory": "/none",
             "FractionQ": 1, "WDLLabelSmoothing": 0}
EXEC_BASE = {"ID": "eaux_test", "DeviceType": "cpu", "DeviceIDs": [0], "DataType": "Float32",
             "UseHistory": True, "DropoutRate": 0, "EngineType": "CSharpViaTorchscript"}


def build(net_over, opt_over, tag):
  from config import Configuration
  from ceres_net import CeresNet
  d = tempfile.mkdtemp(prefix='eaux_')
  try:
    for suf, obj in (('net', {**NET_BASE, **net_over}), ('opt', {**OPT_BASE, **opt_over}),
                     ('data', DATA_BASE), ('exec', {**EXEC_BASE, 'ID': tag}), ('monitoring', {})):
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


def random_boards(B, seed=3):
  """[B,64,137] float: 13-ch piece one-hot per square (kings guaranteed), rest zero."""
  from config import NUM_INPUT_BYTES_PER_SQUARE
  g = torch.Generator().manual_seed(seed)
  sq = torch.zeros(B, 64, NUM_INPUT_BYTES_PER_SQUARE)
  for b in range(B):
    perm = torch.randperm(64, generator=g)
    n_pieces = int(torch.randint(8, 30, (1,), generator=g))
    sq[b, :, 0] = 1.0
    sq[b, perm[0], 0] = 0; sq[b, perm[0], 6] = 1.0        # own king
    sq[b, perm[1], 0] = 0; sq[b, perm[1], 12] = 1.0       # enemy king
    for i in range(2, n_pieces):
      side = int(torch.randint(0, 2, (1,), generator=g))
      pt = int(torch.randint(1, 6, (1,), generator=g))      # P,N,B,R,Q (1..5 / 7..11)
      sq[b, perm[i], 0] = 0
      sq[b, perm[i], pt + (6 if side else 0)] = 1.0
  return sq


def fake_batch(B, sq, seed=5):
  """Policy target: mass on random moves whose from-square holds an own piece
  (so some are piece->piece pairs = captures)."""
  from lc0_moves_1858 import FROM_1858, TO_1858
  g = torch.Generator().manual_seed(seed)
  own = sq[:, :, 1:7].sum(-1) > 0                          # [B,64]
  F = torch.tensor(FROM_1858); T = torch.tensor(TO_1858)
  pol = torch.zeros(B, 1858)
  for b in range(B):
    cand = torch.nonzero(own[b][F]).reshape(-1)
    pick = cand[torch.randperm(len(cand), generator=g)[:12]]
    w = torch.rand(len(pick), generator=g)
    pol[b, pick] = w / w.sum()
  wdl = torch.tensor([[0.4, 0.3, 0.3]]).expand(B, 3).clone()
  return {'policies': pol, 'wdl_deblundered': wdl, 'wdl_q': wdl, 'wdl_nondeblundered': wdl,
          'mlh': torch.zeros(B, 1), 'unc': torch.zeros(B, 1), 'uncertainty_policy': torch.zeros(B, 1),
          'q_deviation_lower': torch.zeros(B, 1), 'q_deviation_upper': torch.zeros(B, 1)}


def run_loss(m, batch, sq):
  from losses import LossCalculator
  lc = LossCalculator(nn.Linear(4, 4))
  outs = m(sq, None)
  (policy_out, value_out, mlh_out, unc_out, value2_out, qdl, qdu, unc_pol, action_out, _) = outs[:10]
  return m.compute_loss(lc, batch, policy_out, value_out, mlh_out, unc_out, value2_out, qdl, qdu, unc_pol,
                        None, None, None, action_out, None, 0, 0, 0, False)


def main():
  B = 4
  sq = random_boards(B)
  batch = fake_batch(B, sq)

  # --- 1. loud rejections -------------------------------------------------
  for name, net_over, opt_over in (
      ('detach-without-aux', {'DualPlaneEdgeAuxDetach': True}, {}),
      ('rel-without-withhold', {}, {'LossDualPlaneEdgeRelMultiplier': 1.0}),
      ('aux-without-learned-edges', {'DualPlaneEdgeUpdate': False, 'DualPlaneEdgeToTrunk': False,
                                     'DualPlaneEdgeToTrunkMask': False, 'DualPlaneEdgeAuxWithhold': 'xray'},
       {'LossDualPlaneEdgeRelMultiplier': 1.0}),
      ('withhold-unknown', {'DualPlaneEdgeAuxWithhold': 'defends'}, {})):
    try:
      build(net_over, opt_over, 'rej')
      raise SystemExit(f'FAIL: {name} did not raise')
    except (ValueError, AssertionError) as e:
      print(f'  rejection OK ({name}): {str(e)[:90]}')

  # --- 2. construction + bit-pairing --------------------------------------
  WH = 'xray,pinray'
  ctrl, _ = build({'DualPlaneEdgeAuxWithhold': WH}, {}, 'ctrl')
  aux, _ = build({'DualPlaneEdgeAuxWithhold': WH},
                 {'LossDualPlaneEdgePiMultiplier': 0.05, 'LossDualPlaneEdgeRelMultiplier': 2.0}, 'aux')
  probe, _ = build({'DualPlaneEdgeAuxWithhold': WH, 'DualPlaneEdgeAuxDetach': True},
                   {'LossDualPlaneEdgePiMultiplier': 0.05, 'LossDualPlaneEdgeRelMultiplier': 2.0}, 'probe')
  assert ctrl.dual_plane.blocks[0].rel_proj.weight.shape[1] == 12, 'withhold 2 of 5 families -> 12 plane channels'
  assert aux.dp_eaux_w.shape == (1 + 8, 12), aux.dp_eaux_w.shape
  assert aux.dp_eaux_w.abs().sum() > 0, 'edge-aux head must be NONZERO-init'
  sd_c, sd_a = ctrl.state_dict(), aux.state_dict()
  extra = sorted(set(sd_a) - set(sd_c))
  assert extra == ['dp_eaux_b', 'dp_eaux_w'], extra
  for k in sd_c:
    assert torch.equal(sd_c[k], sd_a[k]), f'bit-pairing broken at {k}'
  print('  construction + bit-pairing OK (aux vs withheld control identical outside dp_eaux_*)')

  # --- 2b. optimizer-build contract: the weight-decay partition must cover the
  # raw nn.Parameters (bench bug 09-02: ea4fe1a died at train.py's partition
  # assert before step 0 because no branch matched dp_eaux_*).
  from wd_partition import partition_weight_decay
  for m, tag in ((ctrl, 'ctrl'), (aux, 'aux'), (probe, 'probe')):
    dec, nodec = partition_weight_decay(m)          # asserts completeness itself
    if tag != 'ctrl':
      assert 'dp_eaux_w' in nodec and 'dp_eaux_b' in nodec, (tag, 'dp_eaux_* must be no_decay')
  print('  weight-decay partition OK (dp_eaux_* covered, no_decay)')

  # --- 3. training forward + loss + gradient routing -----------------------
  for m, tag, want_plane_grad in ((aux, 'supervised', True), (probe, 'detached probe', False)):
    m.train(); m.zero_grad(set_to_none=True)
    loss = run_loss(m, batch, sq)
    assert torch.isfinite(loss), loss
    assert m._last_dp_eaux is None, 'stash must be consumed by compute_loss'
    loss.backward()
    g_head = m.dp_eaux_w.grad
    assert g_head is not None and torch.isfinite(g_head).all() and g_head.abs().sum() > 0, f'{tag}: head got no grad'
    g_eu = m.dual_plane.blocks[-1].eu_out.weight.grad
    has = g_eu is not None and g_eu.abs().sum() > 0
    # eu_out is zero-init; its grad arrives from the readers (aux head, e2t, rel_proj...).
    # With the aux on, eu_out receives a gradient through the aux head immediately; the
    # detached probe must not add any (other zero-init readers give 0 at step 0).
    assert has == want_plane_grad, f'{tag}: eu_out grad present={has}, expected {want_plane_grad}'
    print(f'  {tag}: loss {float(loss):.4f}, head grad ok, plane grad present={has}')

  # target sanity via the stash (re-run forward only)
  aux.train()
  _ = aux(sq, None)
  lg, sel, occ, tgt = aux._last_dp_eaux
  aux._last_dp_eaux = None
  assert lg.shape == (B, 32, 32, 9) and tgt.shape == (B, 32, 32, 8), (lg.shape, tgt.shape)
  assert set(torch.unique(tgt).tolist()) <= {0.0, 1.0}, 'rel targets must be binary'
  pairm = occ.float().unsqueeze(2) * occ.float().unsqueeze(1)
  assert (tgt * (1 - pairm.unsqueeze(-1))).sum() >= 0   # off-piece entries are simply masked in the loss
  print(f'  targets OK: rel positive rate on piece pairs = '
        f'{float((tgt * pairm.unsqueeze(-1)).sum() / (pairm.sum() * 8)):.4f}')

  # --- 4. eval forward: no stash, identical to control ---------------------
  aux.eval(); ctrl.eval()
  with torch.no_grad():
    oa = aux(sq, None); oc = ctrl(sq, None)
  assert getattr(aux, '_last_dp_eaux', None) is None, 'eval must not stash'
  for i, (a, c) in enumerate(zip(oa, oc)):
    if torch.is_tensor(a):
      assert torch.equal(a, c), f'eval output {i} differs from the withheld control'
  print('  eval OK: no stash, outputs bit-identical to the withheld control')

  # --- 5. other code paths: shared-E (UseVisEdgeBias) + monolithic (no e2t) ---
  for tag, net_over, opt_over in (
      ('shared-E + withhold', {'UseVisEdgeBias': True, 'DualPlaneEdgeAuxWithhold': WH},
       {'LossDualPlaneEdgePiMultiplier': 0.05, 'LossDualPlaneEdgeRelMultiplier': 2.0}),
      ('monolithic path, pi-only', {'DualPlaneEdgeToTrunk': False, 'DualPlaneEdgeToTrunkMask': False},
       {'LossDualPlaneEdgePiMultiplier': 0.05}),
      ('triplet-only learned edges', {'DualPlaneEdgeUpdate': False, 'DualPlaneEdgeToTrunk': False,
                                      'DualPlaneEdgeToTrunkMask': False, 'DualPlaneTripletAttention': True,
                                      'DualPlaneTripletHeads': 2, 'DualPlaneEdgeAuxWithhold': 'pinray'},
       {'LossDualPlaneEdgeRelMultiplier': 1.0})):
    m, _ = build(net_over, opt_over, 'path')
    m.train(); m.zero_grad(set_to_none=True)
    loss = run_loss(m, batch, sq)
    assert torch.isfinite(loss), (tag, loss)
    loss.backward()
    assert m.dp_eaux_w.grad is not None and m.dp_eaux_w.grad.abs().sum() > 0, tag
    if 'shared' in tag:
      assert m.dual_plane.blocks[0].rel_proj.weight.shape[1] == 12, 'shared-E withhold must select 12 plane channels'
    m.eval()
    with torch.no_grad():
      _ = m(sq, None)
    assert getattr(m, '_last_dp_eaux', None) is None
    print(f'  path OK ({tag}): loss {float(loss.detach()):.4f}')

  # --- 6. wave-3 mechanisms: ET-form triplet + nonzero reader init -----------
  et, _ = build({'DualPlaneEdgeAuxWithhold': WH, 'DualPlaneTripletAttention': 1, 'DualPlaneTripletHeads': 2,
                 'DualPlaneTripletForm': 'et'},
                {'LossDualPlaneEdgePiMultiplier': 0.05, 'LossDualPlaneEdgeRelMultiplier': 2.0}, 'et')
  assert hasattr(et.dual_plane.blocks[-1], 'ta_v2') and not hasattr(et.dual_plane.blocks[0], 'ta_v2')
  et.train(); et.zero_grad(set_to_none=True)
  loss = run_loss(et, batch, sq); assert torch.isfinite(loss); loss.backward()
  assert et.dual_plane.blocks[-1].ta_v2.weight.grad is not None, 'ET composed value must receive gradient'
  et.eval()
  with torch.no_grad(): _ = et(sq, None)
  print(f'  ET-form triplet OK: loss {float(loss.detach()):.4f}')
  try:
    build({'DualPlaneTripletForm': 'et'}, {}, 'rej'); raise SystemExit('FAIL: TripletForm without TripletAttention not rejected')
  except ValueError as e:
    print(f'  rejection OK (form-without-triplet): {str(e)[:70]}')
  ri, _ = build({'DualPlaneEdgeAuxWithhold': WH, 'DualPlaneReaderInit': 0.02},
                {'LossDualPlaneEdgePiMultiplier': 0.05, 'LossDualPlaneEdgeRelMultiplier': 2.0}, 'ri')
  sd_r = ri.state_dict()
  readers = [k for k in sd_r if any(t in k for t in ('rel_proj', 'eu_out', 'eu_deg', 'e2t_proj'))]
  assert readers and all(sd_r[k].abs().sum() > 0 for k in readers), 'readers must be nonzero'
  assert not torch.equal(sd_r['dual_plane.blocks.0.rel_proj.weight'], sd_r['dual_plane.blocks.1.rel_proj.weight']), 'blocks must differ'
  for k in sd_a:
    if k in sd_r and k not in readers:
      assert torch.equal(sd_a[k], sd_r[k]), f'reader-init broke pairing at {k}'
  ri.eval(); aux.eval()
  with torch.no_grad():
    o_r = ri(sq, None)[0]; o_a = aux(sq, None)[0]
  assert not torch.equal(o_r, o_a), 'reader init must change step-0 outputs (by design)'
  print(f'  reader-init OK: {len(readers)} reader tensors nonzero, all other tensors bit-paired with aux')
  print('ALL OK')


if __name__ == '__main__':
  main()
