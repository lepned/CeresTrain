"""Tests for NBT hyper-connections (hyper_conn.py, NBTHyperConn), value residual (NBTValueResidual) and head mixing
(NBTHeadMix), 2026-11-08.

  python test_hc_vrhm.py                      # block-level tests
  python test_hc_vrhm.py <cfg_dir> <cfg_id>   # + real-net tests on a small copy of an NBT config (CPU)
"""
import sys
import torch

from hyper_conn import sinkhorn, HyperConnection, HyperConnReadout
from test_piece_tokens import _squares


def test_sinkhorn():
  for n in (2, 4, 8):
    P = sinkhorn(torch.randn(n, n), 20)
    assert torch.allclose(P.sum(-1), torch.ones(n), atol=1e-6) and torch.allclose(P.sum(-2), torch.ones(n), atol=1e-2)
  P = sinkhorn(5.0 * torch.eye(4), 20)
  assert float(P.diagonal().min()) > 0.97


def test_hyper_conn_identity():
  # identical streams + one-hot read + write-to-all + doubly-stochastic mix == the plain residual, for any branch
  torch.manual_seed(0)
  B, T, n, D = 2, 5, 4, 16
  x = torch.randn(B, T, D)
  for dyn in ('none', 'prepost'):
    hcs = [HyperConnection(n, D, l, dyn, 20, 'RMSNorm') for l in range(6)]
    ro = HyperConnReadout(n, D)
    s, plain = x.repeat(1, 1, n), x
    for hc in hcs:
      a, b = hc.coeffs(s)
      h_in = hc.read(s, a)
      assert torch.allclose(h_in, plain, atol=1e-6)
      br = torch.tanh(h_in @ torch.randn(D, D))
      s, plain = hc.write(s, br, b), plain + br
    assert torch.allclose(ro(s), plain, atol=1e-5), dyn


def _real(cfg_dir, cfg_id):
  import json, os, shutil
  from config_bootstrap import bootstrap_env_from_config
  bootstrap_env_from_config(os.path.dirname(cfg_dir), cfg_id, quiet=True, configs_dir=cfg_dir)
  from config import Configuration
  from ceres_net import CeresNet
  from wd_partition import partition_weight_decay
  tag = 'zz_hc_vrhm'
  base = json.load(open(f'{cfg_dir}/{cfg_id}_ceres_net.json'))
  base.update(ModelDim=384, NumLayers=4, NumHeads=4, NBTInnerHeads=4, NBTInnerLayers=2, NBTWidthDivisor=3, EGTEdgeStream=False)
  def build(**over):
    net = dict(base); net.update(over)
    json.dump(net, open(f'{cfg_dir}/{tag}_ceres_net.json', 'w'), indent=1)
    cfg = Configuration(cfg_dir, tag)
    torch.manual_seed(0)
    return CeresNet(None, cfg, policy_loss_weight=1, value_loss_weight=1, moves_left_loss_weight=0, unc_loss_weight=0,
                    value2_loss_weight=0, q_deviation_loss_weight=0, value_diff_loss_weight=0, value2_diff_loss_weight=0,
                    action_loss_weight=0, uncertainty_policy_weight=0, action_uncertainty_loss_weight=0, q_ratio=1).eval()
  def outs_equal(a_out, r_out, what, atol=1e-4):
    for a, r in zip(a_out, r_out):
      if isinstance(r, torch.Tensor):
        assert a.shape == r.shape and torch.allclose(a, r, atol=atol, rtol=1e-4), (what, float((a - r).abs().max()))
  ARMS = (('HC 4 static', dict(NBTHyperConn=4)),
          ('HC 4 prepost', dict(NBTHyperConn=4, NBTHyperConnDynamic='prepost')),
          ('HC 2', dict(NBTHyperConn=2)),
          ('HC inner 2 static', dict(NBTHyperConnInner=2)),
          ('HC inner 2 prepost', dict(NBTHyperConnInner=2, NBTHyperConnInnerDynamic='prepost')),
          ('HC inner 2 + outer 2 + VR', dict(NBTHyperConnInner=2, NBTHyperConn=2, NBTValueResidual=True)),
          ('HC 4 + MoD 32', dict(NBTHyperConn=4, NBTMoDCapacity=32)),
          ('value residual', dict(NBTValueResidual=True)),
          ('head mix all', dict(NBTHeadMix=True)),
          ('head mix group 2', dict(NBTHeadMix=True, NBTHeadMixGroup=2)),
          ('VR + head mix (arm VH)', dict(NBTValueResidual=True, NBTHeadMix=True)),
          ('HC 4 + VR + HM + registers 8', dict(NBTHyperConn=4, NBTHyperConnDynamic='prepost', NBTValueResidual=True,
                                                NBTHeadMix=True, RegisterTokens=8, RegisterTokensToValue=True)))
  try:
    for p in ('data', 'exec', 'monitoring', 'opt'):
      shutil.copy(f'{cfg_dir}/{cfg_id}_ceres_{p}.json', f'{cfg_dir}/{tag}_ceres_{p}.json')
    sq = _squares(3)
    base_net = build()
    bsd0 = {k: v.clone() for k, v in base_net.state_dict().items()}
    with torch.no_grad():
      # non-trivial trunk: random up projections (zero-init otherwise makes every block an identity)
      for n, p in base_net.named_parameters():
        if n.endswith('.up.weight'):
          p.normal_(0, 0.02)
      ref = base_net(sq, None)
      reg_ref = None
      for name, over in ARMS:
        net = build(**over)
        sd = net.state_dict()
        new = [k for k in sd if k not in bsd0]
        if 'MoD' in name:
          # the MoD router changes the function (and draws from the RNG): compare against MoD alone, same weights
          mnet = build(NBTMoDCapacity=32)
          with torch.no_grad():
            for n_, p_ in mnet.named_parameters():
              if n_.endswith('.up.weight'):
                p_.normal_(0, 0.02)
          net.load_state_dict(mnet.state_dict(), strict=False)
          outs_equal(net(sq, None), mnet(sq, None), name)
          print(f'  {name}: == MoD-only net with the same weights')
          continue
        for k, v in bsd0.items():
          assert torch.equal(v, sd[k]), f'{name}: init differs from the base net at {k}'
        missing, unexpected = net.load_state_dict(base_net.state_dict(), strict=False)
        assert not unexpected and sorted(missing) == sorted(new), (name, missing, unexpected)
        out = net(sq, None)
        if 'registers' in name:
          # registers change the function; compare against the registers-only net with the same weights
          rnet = build(RegisterTokens=8, RegisterTokensToValue=True)
          rnet.load_state_dict({k: v for k, v in net.state_dict().items() if k in rnet.state_dict()}, strict=True)
          outs_equal(out, rnet(sq, None), name)
        else:
          outs_equal(out, ref, name)
        decay, no_decay = partition_weight_decay(net)
        assert all(k in no_decay for k in new if not k.startswith('register_')), (name, [k for k in new if k in decay])
        print(f'  {name}: init bit-identical, == base at init with base weights, {len(new)} new tensors '
              f'({sum(sd[k].numel() for k in new):,} params), wd partition OK')
    # head-mix math: the wide-head Q~K~^T equals sum_h' M[g,h'] Q_h' K_h'^T within each group
    hm = build(NBTHeadMix=True, NBTHeadMixGroup=2)
    att = hm.transformer_layer[0].inner[0].attention
    with torch.no_grad():
      att.head_mix.copy_(torch.randn_like(att.head_mix))
    Q, K = torch.randn(2, 4, 64, 32), torch.randn(2, 4, 64, 32)
    Qm, Km = att._head_mix_qk(Q, K)
    S = Q @ K.transpose(-1, -2)
    for g in range(4):
      grp = g // 2
      want = sum(att.head_mix[grp, g % 2, j] * S[:, grp * 2 + j] for j in range(2))
      assert torch.allclose(Qm[:, g] @ Km[:, g].transpose(-1, -2), want, atol=1e-4), g
    print('  head mix: wide-head scores == group-mixed logits')
    # QK-clip sees the UNMIXED per-head logits (apply_qk_clip can only shrink a head's own Q/K rows)
    att.qk_clip_monitor = True
    att.train()
    att._last_max_logit = None
    hm.train()
    hm(sq, None)
    assert att._last_max_logit is not None and att._last_max_logit.shape == (4,)
    att.qk_clip_monitor = False
    hm.eval()
    # export bake: H_res as a constant buffer reproduces the Sinkhorn matrix
    hcn = build(NBTHyperConn=4)
    m0 = hcn.hyper_conn[0].res_matrix()
    hcn.hyper_conn[0].bake_res_matrix_()
    assert torch.equal(hcn.hyper_conn[0].res_matrix(), m0) and 'res_baked' not in hcn.state_dict()
    print('  QK-clip monitor stashes unmixed per-head max; H_res bake OK')
    # gradients reach every new parameter; the HC streams break symmetry (beta grads differ across streams)
    full = build(NBTHyperConn=4, NBTHyperConnDynamic='prepost', NBTValueResidual=True, NBTHeadMix=True)
    with torch.no_grad():
      for n, p in full.named_parameters():
        if n.endswith('.up.weight'):
          p.normal_(0, 0.02)
    full.train()
    out = full(sq, None)
    sum(o.float().square().mean() for o in out if isinstance(o, torch.Tensor) and o.is_floating_point()).backward()
    named = dict(full.named_parameters())
    for k in ('hyper_conn.1.beta', 'hyper_conn.1.alpha', 'hyper_conn.1.res_logits', 'hyper_conn.1.dyn_w',
              'hyper_conn_out.alpha', 'transformer_layer.1.inner.0.attention.vr_lambda',
              'transformer_layer.1.inner.0.attention.head_mix'):
      g = named[k].grad
      assert g is not None and float(g.abs().sum()) > 0, f'no gradient into {k}'
    gb = named['hyper_conn.1.beta'].grad
    assert float((gb - gb.mean()).abs().max()) > 0, 'HC beta gradients identical across streams (no symmetry breaking)'
    assert 'transformer_layer.0.inner.0.attention.vr_lambda' not in named, 'first inner layer must not have a lambda'
    print('  gradients reach HC / VR / HM params; HC beta gradients differ across streams')
    # inner hyper-connections: gradients into every inner-HC param, beta symmetry broken, bake reproduces H_res
    inn = build(NBTHyperConnInner=2, NBTHyperConnInnerDynamic='prepost')
    with torch.no_grad():
      for n, p in inn.named_parameters():
        if n.endswith('.up.weight'):
          p.normal_(0, 0.02)
        # identical streams at init give H_res an exactly zero gradient (any doubly-stochastic mix of equal rows is a
        # no-op); one optimizer step on beta separates them -- mimic it so the res_logits gradient check is meaningful
        if '.hyper_conn_inner.' in n and n.endswith('.beta'):
          p.add_(torch.randn_like(p) * 0.1)
    inn.train()
    out = inn(sq, None)
    sum(o.float().square().mean() for o in out if isinstance(o, torch.Tensor) and o.is_floating_point()).backward()
    named = dict(inn.named_parameters())
    for k in ('transformer_layer.1.hyper_conn_inner.1.beta', 'transformer_layer.1.hyper_conn_inner.1.alpha',
              'transformer_layer.1.hyper_conn_inner.1.res_logits', 'transformer_layer.1.hyper_conn_inner.1.dyn_w',
              'transformer_layer.1.hyper_conn_inner_out.alpha'):
      g = named[k].grad
      assert g is not None and float(g.abs().sum()) > 0, f'no gradient into {k}'
    gb = named['transformer_layer.1.hyper_conn_inner.0.beta'].grad
    assert float((gb - gb.mean()).abs().max()) > 0, 'inner HC beta gradients identical across streams'
    inn.eval()
    blk = inn.transformer_layer[1].hyper_conn_inner[0]
    m0 = blk.res_matrix()
    blk.bake_res_matrix_()
    assert torch.equal(blk.res_matrix(), m0)
    print('  inner HC: gradients reach every param, beta grads differ across streams, H_res bake OK')
    for over in (dict(NBTHyperConn=3), dict(NBTHyperConnDynamic='prepost'), dict(NBTHyperConnSinkhornIters=5),
                 dict(NBTHyperConn=4, PieceTokens=32), dict(NBTValueResidual=True, PieceTokens=32),
                 dict(NBTHyperConnInner=3), dict(NBTHyperConnInnerDynamic='prepost'), dict(NBTHyperConnInner=4), dict(NBTHyperConnInner=2, NBTMoDCapacity=32),
                 dict(NBTHyperConnInner=2, NBTInnerPreNorm=False),
                 dict(NBTValueResidual=True, NBTMoDCapacity=32), dict(NBTHeadMix=True, NBTHeadMixGroup=3),
                 dict(NBTHeadMixGroup=2), dict(NBTHeadMix=True, NBTHeadMixGroup=16), dict(NBTHeadMix=True, EGTEdgeStream=True),
                 dict(NBTHeadMix=True, NBTInnerHeads=8, NBTWidthDivisor=1, NBTHeadMixGroup=0)):
      refused = False
      try:
        build(**over)
      except (ValueError, AssertionError):
        refused = True
      assert refused, f'must be refused: {over}'
    print('  guards OK')
  finally:
    for p in ('data', 'exec', 'monitoring', 'opt', 'net'):
      fn = f'{cfg_dir}/{tag}_ceres_{p}.json'
      if os.path.exists(fn):
        os.remove(fn)


if __name__ == '__main__':
  for name, fn in list(globals().items()):
    if name.startswith('test_') and callable(fn):
      fn()
      print(f'{name} OK')
  if len(sys.argv) == 3:
    _real(sys.argv[1], sys.argv[2])
