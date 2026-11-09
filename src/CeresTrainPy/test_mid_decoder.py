"""Tests for the mid-depth decoder policy (opt MidDecoderPolicyWeight / MidDecoderPolicyBlocks), 2026-11-09.

  python test_mid_decoder.py <cfg_dir> <cfg_id>   # real-net tests on a small copy of an NBT + move-token config (CPU)
"""
import json, os, shutil, sys
import torch

from test_piece_tokens import _squares


def main(cfg_dir, cfg_id):
  from config_bootstrap import bootstrap_env_from_config
  bootstrap_env_from_config(os.path.dirname(cfg_dir), cfg_id, quiet=True, configs_dir=cfg_dir)
  from config import Configuration
  from ceres_net import CeresNet
  tag = 'zz_middec'
  base_net = json.load(open(f'{cfg_dir}/{cfg_id}_ceres_net.json'))
  base_net.update(ModelDim=384, NumLayers=4, NumHeads=4, NBTInnerHeads=4, NBTInnerLayers=2, NBTWidthDivisor=3,
                  EGTEdgeStream=False)
  base_opt = json.load(open(f'{cfg_dir}/{cfg_id}_ceres_opt.json'))
  for p in ('data', 'exec', 'monitoring'):
    shutil.copy(f'{cfg_dir}/{cfg_id}_ceres_{p}.json', f'{cfg_dir}/{tag}_ceres_{p}.json')

  def cfg(net_over=None, opt_over=None):
    net = dict(base_net); net.update(net_over or {})
    opt = dict(base_opt); opt.update(opt_over or {})
    json.dump(net, open(f'{cfg_dir}/{tag}_ceres_net.json', 'w'), indent=1)
    json.dump(opt, open(f'{cfg_dir}/{tag}_ceres_opt.json', 'w'), indent=1)
    return Configuration(cfg_dir, tag)

  def build(c):
    torch.manual_seed(0)
    return CeresNet(None, c, policy_loss_weight=1, value_loss_weight=1, moves_left_loss_weight=0, unc_loss_weight=0,
                    value2_loss_weight=0, q_deviation_loss_weight=0, value_diff_loss_weight=0, value2_diff_loss_weight=0,
                    action_loss_weight=0, uncertainty_policy_weight=0, action_uncertainty_loss_weight=0, q_ratio=1)

  try:
    sq = _squares(4)
    base = build(cfg())
    mid = build(cfg(opt_over=dict(MidDecoderPolicyWeight=0.05, MidDecoderPolicyBlocks=[0, 2])))
    # 1) the only new parameters are the ones-init per-depth gains (warm-start aux keys); base weights load into mid
    extra = set(mid.state_dict()) - set(base.state_dict())
    assert extra == {'mid_dec_gain.0', 'mid_dec_gain.1'} and not (set(base.state_dict()) - set(mid.state_dict())), extra
    res = mid.load_state_dict(base.state_dict(), strict=False)
    assert set(res.missing_keys) == extra and not res.unexpected_keys
    assert all(torch.equal(p, torch.ones_like(p)) for p in mid.mid_dec_gain)
    from wd_partition import partition_weight_decay
    _wd = partition_weight_decay(mid)
    _nd = _wd[1] if isinstance(_wd, tuple) else _wd
    assert all(any(k.endswith(f'mid_dec_gain.{j}') for k in _nd) for j in (0, 1)), 'mid_dec_gain must be no-decay'
    # 2) eval: no stash, outputs identical to the base net (export graph unchanged)
    base.eval(); mid.eval()
    with torch.no_grad():
      ob, om = base(sq, None), mid(sq, None)
    for a, b in zip(ob, om):
      if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), 'eval outputs differ'
    assert getattr(mid, '_last_mid_dec_policy', None) is None, 'eval forward must not stash'
    print('  eval identical, no stash OK')
    # 3) train: stash shape, and it equals the decoder run on the trunk-end-normed state after the block
    mid.train()
    states = {}
    hooks = [blk.register_forward_hook(lambda m, i, o, k=k: states.__setitem__(k, (o[0] if isinstance(o, tuple) else o)))
             for k, blk in enumerate(mid.transformer_layer)]
    out = mid(sq, None)
    for h in hooks:
      h.remove()
    stash = mid._last_mid_dec_policy
    assert stash is not None and stash.shape == (sq.shape[0], 2, 1858), stash.shape
    with torch.no_grad():
      for j, b in enumerate([0, 2]):
        ref = mid.move_tokens(sq[:, :, 0:13].to(states[b].dtype), mid.trunk_end_norm(states[b]))[0]
        assert torch.allclose(stash[:, j], ref, atol=1e-5), f'mid logits != decoder(norm(state after block {b}))'
    print('  train stash shape + equality with decoder(norm(state)) OK')
    # 4) gradient of the mid CE reaches blocks <= max(listed) and NOT later blocks; main policy out unaffected by stash
    mid.zero_grad()
    legal = torch.ones_like(stash[:, 0]).bool()
    tgt = torch.softmax(torch.randn(sq.shape[0], 1858), -1)
    loss = -(tgt.unsqueeze(1) * torch.log_softmax(stash.float(), -1)).sum(-1).mean()
    loss.backward()
    g = lambda blk: sum(float(p.grad.abs().sum()) for p in blk.parameters() if p.grad is not None)
    assert g(mid.transformer_layer[0]) > 0 and g(mid.transformer_layer[2]) > 0, 'no grad into listed blocks'
    assert g(mid.transformer_layer[3]) == 0, 'grad leaked into a block after the last listed block'
    assert sum(float(p.grad.abs().sum()) for p in mid.move_tokens.parameters() if p.grad is not None) > 0
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in mid.mid_dec_gain), 'gain gets no grad'
    with torch.no_grad():
      mid.mid_dec_gain[0].mul_(2.0)
    s2 = mid(sq, None) and mid._last_mid_dec_policy
    assert not torch.allclose(s2[:, 0], stash[:, 0].detach()) and torch.allclose(s2[:, 1], stash[:, 1].detach(), atol=1e-5), \
        'per-depth gain not applied to its own depth only'
    print('  gradient routing OK (blocks 0..2 yes, block 3 no, decoder yes)')
    # 5) guards
    for over, what in ((dict(MidDecoderPolicyWeight=0.05), 'no blocks'),
                       (dict(MidDecoderPolicyWeight=0.05, MidDecoderPolicyBlocks=[3]), 'last block'),
                       (dict(MidDecoderPolicyWeight=0.05, MidDecoderPolicyBlocks=[1, 1]), 'duplicate'),
                       (dict(MidDecoderPolicyBlocks=[1]), 'blocks without weight')):
      try:
        cfg(opt_over=over)
      except ValueError:
        continue
      raise AssertionError(f'guard missing: {what}')
    try:
      cfg(net_over=dict(UseMoveTokens=False), opt_over=dict(MidDecoderPolicyWeight=0.05, MidDecoderPolicyBlocks=[1]))
      raise AssertionError('guard missing: no move tokens')
    except ValueError:
      pass
    print('  guards OK')
  finally:
    for p in ('data', 'exec', 'monitoring', 'net', 'opt'):
      f = f'{cfg_dir}/{tag}_ceres_{p}.json'
      if os.path.exists(f):
        os.remove(f)
  print('test_mid_decoder OK')


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2])
