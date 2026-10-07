"""Contract tests for the EGT edge stream in TRT form (egt_edge.py, NetDef EGTEdgeStream; 2026-10-07).

Run from src/CeresTrainPy:

    python test_egt_edge.py            (CPU, tiny NBT net)

1. Bit-pairing: with the same TorchSeed the EGT net and the control share EVERY parameter outside egt.* bit for bit
   (forked fixed-seed init); the weight-decay partition covers the new raw parameters.
2. Step-0 identity: all readers zero-init => eval outputs bit-identical to the control.
3. Live path: with the zero readers perturbed, outputs change, the training loss is finite and gradients reach the read,
   readback, triplet, FFN and init tables as well as the trunk.
4. Readback logits: softmax(NBT._raw_logits(...)) == the attention probabilities the layer actually used (same bias and
   q/k scales), so the readback reads exactly the scores the block computed.
5. Config: EGTEdgeStream without NBT is rejected.
"""
import os, sys, json, tempfile, shutil

os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NET_BASE = {
  "ModelDim": 64, "NumLayers": 3, "NumHeads": 4, "PreNorm": False, "NormType": "RMSNorm",
  "FFNMultiplier": 2, "FFNActivationType": "SwiGLU", "HeadsActivationType": "Mish",
  "NonLinearAttention": False, "SoftCapCutoff": 0,
  "SmolgenDimPerSquare": 0, "SmolgenDim": 0, "SmolgenToHeadDivisor": 1, "SmolgenActivationType": "Swish",
  "UseRPE": False, "UseRPE_V": False, "UseRoPE": False,
  "NBTInnerLayers": 2, "NBTWidthDivisor": 2, "NBTInnerHeads": 4, "NBTInnerPreNorm": True,
  "NBTProjActivation": "Swish", "NBTProjNorm": "Norm",
}
EGT_ON = {"EGTEdgeStream": True, "EGTEdgeSites": [0, 1]}   # 3 blocks: a site after the last block is refused
OPT_BASE = {
  "NumTrainingPositions": 1000, "BatchSizeForwardPass": 4, "BatchSizeBackwardPass": 4,
  "Optimizer": "Muon", "LearningRateBase": 1e-4, "LossValueMultiplier": 1.0,
  "LossValue2Multiplier": 0.0, "LossPolicyMultiplier": 1, "LossMLHMultiplier": 0,
  "LossUNCMultiplier": 0, "LossUncertaintyPolicyMultiplier": 0, "LossValueDMultiplier": 0,
  "TorchSeed": 777, "TPGV3": 0, "AuxFeaturesPerSquare": 0,
}
DATA_BASE = {"SourceType": "DirectFromPositionGenerator", "TrainingFilesDirectory": "/none", "FractionQ": 1,
             "WDLLabelSmoothing": 0}
EXEC_BASE = {"ID": "egt_test", "DeviceType": "cpu", "DeviceIDs": [0], "DataType": "Float32",
             "UseHistory": True, "DropoutRate": 0, "EngineType": "CSharpViaTorchscript"}


def build(net_over, tag):
  from config import Configuration
  from ceres_net import CeresNet
  d = tempfile.mkdtemp(prefix='egt_')
  try:
    for suf, obj in (('net', {**NET_BASE, **net_over}), ('opt', OPT_BASE), ('data', DATA_BASE),
                     ('exec', {**EXEC_BASE, 'ID': tag}), ('monitoring', {})):
      with open(os.path.join(d, f'{tag}_ceres_{suf}.json'), 'w') as f:
        json.dump(obj, f)
    cfg = Configuration(d, tag)
    torch.manual_seed(int(OPT_BASE['TorchSeed']))
    return CeresNet(None, cfg,
                    policy_loss_weight=cfg.Opt_LossPolicyMultiplier, value_loss_weight=cfg.Opt_LossValueMultiplier,
                    moves_left_loss_weight=cfg.Opt_LossMLHMultiplier, unc_loss_weight=cfg.Opt_LossUNCMultiplier,
                    value2_loss_weight=cfg.Opt_LossValue2Multiplier, q_deviation_loss_weight=cfg.Opt_LossQDeviationMultiplier,
                    value_diff_loss_weight=cfg.Opt_LossValueDMultiplier, value2_diff_loss_weight=cfg.Opt_LossValue2DMultiplier,
                    action_loss_weight=cfg.Opt_LossActionMultiplier, uncertainty_policy_weight=cfg.Opt_LossUncertaintyPolicyMultiplier,
                    action_uncertainty_loss_weight=cfg.Opt_LossActionUncertaintyMultiplier, q_ratio=cfg.Data_FractionQ)
  finally:
    shutil.rmtree(d, ignore_errors=True)


def boards(B, seed=3):
  from config import NUM_INPUT_BYTES_PER_SQUARE
  g = torch.Generator().manual_seed(seed)
  sq = torch.zeros(B, 64, NUM_INPUT_BYTES_PER_SQUARE)
  for b in range(B):
    perm = torch.randperm(64, generator=g)
    sq[b, :, 0] = 1.0
    sq[b, perm[0], 0] = 0; sq[b, perm[0], 6] = 1.0
    sq[b, perm[1], 0] = 0; sq[b, perm[1], 12] = 1.0
    for i in range(2, 20):
      side = int(torch.randint(0, 2, (1,), generator=g)); pt = int(torch.randint(1, 6, (1,), generator=g))
      sq[b, perm[i], 0] = 0; sq[b, perm[i], pt + (6 if side else 0)] = 1.0
  return sq


def batch_for(B, seed=5):
  g = torch.Generator().manual_seed(seed)
  pol = torch.zeros(B, 1858)
  for b in range(B):
    idx = torch.randperm(1858, generator=g)[:10]
    w = torch.rand(10, generator=g) + 0.05
    pol[b, idx] = w / w.sum()
  wdl = torch.tensor([[0.4, 0.3, 0.3]]).expand(B, 3).clone()
  return {'policies': pol, 'wdl_deblundered': wdl, 'wdl_q': wdl, 'wdl_nondeblundered': wdl,
          'mlh': torch.zeros(B, 1), 'unc': torch.zeros(B, 1), 'uncertainty_policy': torch.zeros(B, 1),
          'q_deviation_lower': torch.zeros(B, 1), 'q_deviation_upper': torch.zeros(B, 1)}


def run_loss(m, batch, sq):
  from losses import LossCalculator
  outs = m(sq, None)
  (policy_out, value_out, mlh_out, unc_out, value2_out, qdl, qdu, unc_pol, action_out, _) = outs[:10]
  return m.compute_loss(LossCalculator(nn.Linear(4, 4)), batch, policy_out, value_out, mlh_out, unc_out, value2_out,
                        qdl, qdu, unc_pol, None, None, None, action_out, None, 0, 0, 0, False)


def main():
  from wd_partition import partition_weight_decay
  B = 4
  sq, batch = boards(B), batch_for(B)

  # 1 + 2
  ctrl = build({}, 'ctrl')
  egt = build(EGT_ON, 'egt')
  sd_c, sd_e = ctrl.state_dict(), egt.state_dict()
  extra = sorted(set(sd_e) - set(sd_c))
  assert extra and all(k.startswith('egt.') for k in extra), extra
  assert not (set(sd_c) - set(sd_e))
  for k in sd_c:
    assert torch.equal(sd_c[k], sd_e[k]), f'bit-pairing broken at {k}'
  # NBT up projections are zero-init (every block is an identity at step 0): give BOTH nets the same random ups so the
  # trunk actually depends on its attention, otherwise no edge-stream effect could ever be visible.
  with torch.no_grad():
    g0 = torch.Generator().manual_seed(7)
    for (nc, pc), (ne, pe) in zip(ctrl.named_parameters(), [(n, p) for n, p in egt.named_parameters() if not n.startswith('egt.')]):
      assert nc == ne
      if nc.endswith('.up.weight'):
        w = torch.randn(pc.shape, generator=g0) * 0.05
        pc.copy_(w); pe.copy_(w)
  dec, nodec = partition_weight_decay(egt)
  assert all(k in nodec for k in extra if k in dict(egt.named_parameters()))
  ctrl.eval(); egt.eval()
  with torch.no_grad():
    oc, oe = ctrl(sq, None), egt(sq, None)
  for i, (a, b) in enumerate(zip(oc, oe)):
    if torch.is_tensor(a):
      assert torch.equal(a, b), f'step-0 output {i} differs from the control'
  print(f'OK pairing ({len(extra)} new egt.* tensors) + wd partition + step-0 eval identity')

  # 3 live path
  with torch.no_grad():
    g = torch.Generator().manual_seed(11)
    for n, p in egt.named_parameters():
      if n.startswith('egt.') and p.abs().sum() == 0:
        p.copy_(torch.randn(p.shape, generator=g) * 0.2)
  with torch.no_grad():
    oe2 = egt(sq, None)
  assert not torch.allclose(oe2[0], oc[0]), 'live edge stream must change the policy output'
  egt.train(); egt.zero_grad(set_to_none=True)
  loss = run_loss(egt, batch, sq)
  assert torch.isfinite(loss), loss
  loss.backward()
  P = dict(egt.named_parameters())
  for n in ('egt.reads.0.w_e', 'egt.reads.1.w_g', 'egt.reads.2.m_q', 'egt.reads.2.m_k', 'egt.reads.1.m_r',
            'egt.site_mods.0.o_e', 'egt.site_mods.0.tri_o', 'egt.site_mods.0.tri_v', 'egt.site_mods.0.ffn_out',
            'egt.p_in', 'egt.t_off'):
    assert P[n].grad is not None and torch.isfinite(P[n].grad).all() and P[n].grad.abs().sum() > 0, f'{n}: no gradient'
  trunk = [n for n, p in egt.named_parameters() if 'transformer_layer' in n and p.grad is not None and p.grad.abs().sum() > 0]
  assert trunk
  for n, p in egt.named_parameters():
    if n.startswith('egt.'):
      assert p.grad is not None, f'{n}: no gradient (every site must be read by a later block)'
  print(f'OK live path: loss {float(loss):.4f}, gradients reach read/readback/triplet/FFN/init tables and the trunk')

  # 4 readback logits == the scores the layer used (also with softcap and per-head logit temperature, review 10-07)
  check_readback(egt, sq, 'softcap 0')
  for tag, over in (('softcap 100 + head temp', {'SoftCapCutoff': 100, 'UseHeadLogitTemp': True}),
                    ('softcap 5 (cap active)', {'SoftCapCutoff': 5})):
    m = build({**EGT_ON, **over}, 'rb')
    with torch.no_grad():
      g1 = torch.Generator().manual_seed(13)
      for n, p in m.named_parameters():
        if (n.startswith('egt.') and p.abs().sum() == 0) or n.endswith('.up.weight') or n.endswith('head_logit_temp'):
          p.copy_(torch.randn(p.shape, generator=g1) * (3.0 if 'w_e' in n else 0.3))
    check_readback(m, sq, tag)

  # 5 config
  try:
    build({**EGT_ON, 'NBTInnerLayers': 0}, 'rej')
    raise SystemExit('FAIL: EGTEdgeStream without NBT accepted')
  except ValueError:
    pass
  try:
    build({**EGT_ON, 'EGTEdgeSites': [0, 2]}, 'rej2')
    raise SystemExit('FAIL: a site after the last block accepted')
  except AssertionError:
    pass
  print('OK config rejections (no NBT, site after the last block)')

  # 6 triplet index oracle (broadcast sums, independent of the einsum strings)
  check_triplet_oracle()
  print('ALL OK')


def check_triplet_oracle():
  from egt_edge import EGTEdgeSite, _rms_last
  torch.manual_seed(2)
  site = EGTEdgeSite(16, 4, 4, 2)
  with torch.no_grad():
    site.tri_o.copy_(torch.randn(site.tri_o.shape) * 0.3)
    site.ffn_out.zero_(); site.o_e.zero_()
  e = _rms_last(torch.randn(2, 64, 64, 16))
  got = site(e, torch.zeros(2, 4, 64, 64))
  # oracle: a_in[i,k,h] (softmax over k), out_in[i,j,h,d] = sum_k a_in[i,k,h] v_in[k,j,h,d]   (path i -> k -> j)
  #         a_out[k,i,h] (softmax over k), out_out[i,j,h,d] = sum_k a_out[k,i,h] v_out[j,k,h,d]
  n = _rms_last(e); de, th = 16, 4
  wv, weg, beg = site.tri_v, site.tri_eg, site.tri_eg_b
  v_in = (n @ wv[:, :de]).reshape(2, 64, 64, th, de // th)
  v_out = (n @ wv[:, de:]).reshape(2, 64, 64, th, de // th)
  lg = n @ weg + beg
  a_in = torch.softmax(lg[..., 0:th], dim=2) * torch.sigmoid(lg[..., th:2 * th])
  a_out = torch.softmax(lg[..., 2 * th:3 * th], dim=1) * torch.sigmoid(lg[..., 3 * th:])
  oi = (a_in[:, :, :, None, :, None] * v_in[:, None, :, :, :, :]).sum(dim=2)                 # [b,i,j,h,d] over k
  oo = (a_out.permute(0, 2, 1, 3)[:, :, :, None, :, None] * v_out.permute(0, 2, 1, 3, 4)[:, None]).sum(dim=2)
  e2 = e + oi.reshape(2, 64, 64, de) @ site.tri_o[:de] + oo.reshape(2, 64, 64, de) @ site.tri_o[de:]
  want = _rms_last(e2)
  assert torch.allclose(got, want, atol=1e-4), float((got - want).abs().max())
  print(f'OK triplet oracle (path form, broadcast sums): max diff {float((got - want).abs().max()):.1e}')


def check_readback(egt, sq, tag):
  from nbt_layer import NestedBottleneckLayer
  egt.eval()
  blk = egt.transformer_layer[0]
  last_att = blk.inner[-1].attention
  captured = {}
  orig = last_att.sdp_and_smol_or_rpe
  def spy(*a, **k):
    H, A = orig(*a, **k)
    captured['A'] = A
    return H, A
  last_att.sdp_and_smol_or_rpe = spy
  captured_logits = {}
  orig_raw = NestedBottleneckLayer._raw_logits
  def raw_spy(layer, h, prb, sc):
    s = orig_raw(layer, h, prb, sc)
    captured_logits.setdefault('s', s)        # block 0 is the first site; block 2 comes later
    return s
  NestedBottleneckLayer._raw_logits = staticmethod(raw_spy)
  try:
    with torch.no_grad():
      egt(sq, None)
  finally:
    NestedBottleneckLayer._raw_logits = staticmethod(orig_raw)
    last_att.sdp_and_smol_or_rpe = orig
  A_used = captured['A'].float()
  A_raw = torch.softmax(captured_logits['s'].float(), dim=-1)
  assert torch.allclose(A_used, A_raw, atol=1e-5), (tag, float((A_used - A_raw).abs().max()))
  print(f'OK readback logits [{tag}]: softmax(raw logits) == attention probabilities used '
        f'(max diff {float((A_used - A_raw).abs().max()):.2e})')


if __name__ == '__main__':
  main()
