"""Tests for NBTReRead (second down projection per NBT block, nbt_layer.py), 2026-11-09.

  python test_nbt_reread.py <cfg_dir> <cfg_id>   # real-net tests on a small copy of an NBT config (CPU)
"""
import copy, json, os, shutil, sys
import torch

from test_piece_tokens import _squares


def main(cfg_dir, cfg_id):
  from config_bootstrap import bootstrap_env_from_config
  bootstrap_env_from_config(os.path.dirname(cfg_dir), cfg_id, quiet=True, configs_dir=cfg_dir)
  from config import Configuration
  from ceres_net import CeresNet
  from wd_partition import partition_weight_decay
  tag = 'zz_reread'
  base_net = json.load(open(f'{cfg_dir}/{cfg_id}_ceres_net.json'))
  base_net.update(ModelDim=384, NumLayers=4, NumHeads=4, NBTInnerHeads=4, NBTInnerLayers=3, NBTWidthDivisor=3,
                  EGTEdgeStream=False)
  for p in ('data', 'exec', 'monitoring', 'opt'):
    shutil.copy(f'{cfg_dir}/{cfg_id}_ceres_{p}.json', f'{cfg_dir}/{tag}_ceres_{p}.json')

  def cfg(**over):
    net = dict(base_net); net.update(over)
    json.dump(net, open(f'{cfg_dir}/{tag}_ceres_net.json', 'w'), indent=1)
    return Configuration(cfg_dir, tag)

  def build(**over):
    c = cfg(**over)
    torch.manual_seed(0)
    return CeresNet(None, c, policy_loss_weight=1, value_loss_weight=1, moves_left_loss_weight=0, unc_loss_weight=0,
                    value2_loss_weight=0, q_deviation_loss_weight=0, value_diff_loss_weight=0, value2_diff_loss_weight=0,
                    action_loss_weight=0, uncertainty_policy_weight=0, action_uncertainty_loss_weight=0, q_ratio=1)

  def outs(net, sq):
    with torch.no_grad():
      return [o for o in net.eval()(sq, None) if isinstance(o, torch.Tensor)]

  def same(a, b, atol=0.0):
    return all((torch.equal(x, y) if atol == 0 else torch.allclose(x, y, atol=atol)) for x, y in zip(a, b))

  try:
    sq = _squares(4)
    for k in (1, 2):
      for extra, what in (({}, 'plain'), (dict(NBTHyperConnInner=2), 'inner HC n=2'), (dict(NBTProjActivationShift=True), 'proj shift'),
                          (dict(NBTMoDCapacity=32), 'MoD 32'), (dict(NBTCarry=True), 'carry')):
        base = build(**extra)
        # the up projections are zero-init (no gradient reaches a block's interior at a fresh init): open them,
        # as after some training, so the identity / gradient checks are not trivially satisfied
        with torch.no_grad():
          g = torch.Generator().manual_seed(1)
          for b in base.transformer_layer:
            b.up.weight.copy_(0.02 * torch.randn(b.up.weight.shape, generator=g))
        rr = build(NBTReRead=k, **extra)
        # 1) only the new keys differ; base weights load; step 0 is bit-identical (gate 0)
        new = set(rr.state_dict()) - set(base.state_dict())
        assert new and all(n.endswith('.down2.weight') or n.endswith('.reread_gate') for n in new), new
        assert not (set(base.state_dict()) - set(rr.state_dict()))
        res = rr.load_state_dict(base.state_dict(), strict=False)
        assert set(res.missing_keys) == new and not res.unexpected_keys
        assert same(outs(base, sq), outs(rr, sq)), f'reread={k} {what}: step 0 not identical'
        # 2) gradients: gate gets grad at 0, down2 gets none at 0 and some once the gate is open
        rr.train(); rr.zero_grad()
        rr(sq, None)[0].float().pow(2).mean().backward()
        blk = rr.transformer_layer[1]
        assert blk.reread_gate.grad is not None and float(blk.reread_gate.grad.abs()) > 0, 'gate gets no grad'
        assert blk.down2.weight.grad is None or float(blk.down2.weight.grad.abs().sum()) == 0
        with torch.no_grad():
          for b in rr.transformer_layer:
            b.reread_gate.fill_(0.5)
        rr.zero_grad(); rr(sq, None)[0].float().pow(2).mean().backward()
        assert float(blk.down2.weight.grad.abs().sum()) > 0, 'down2 gets no grad with the gate open'
        o_open = outs(rr, sq)
        assert not same(outs(base, sq), o_open, atol=1e-6), 'open gate changes nothing'
        # 3) export fold of the projection-activation shift keeps the output (down2 folded too)
        if extra.get('NBTProjActivationShift'):
          ex = rr   # fold in place (last use of rr; deepcopy trips over stashed non-leaf tensors)
          assert all(b.fold_activation_shift_() for b in ex.transformer_layer)
          assert same(o_open, outs(ex, sq), atol=1e-4), 'shift fold changes the output with re-read'
        # 4) export fold of the re-read gate into down2 keeps the output; a second fold is a no-op
        assert all(b.fold_reread_gate_() for b in rr.transformer_layer)
        assert not any(b.fold_reread_gate_() for b in rr.transformer_layer), 'second gate fold must be a no-op'
        assert same(o_open, outs(rr, sq), atol=1e-4), 'gate fold changes the output'
        print(f'  reread={k} {what}: identity at init, gradients, open gate OK')
    # 4) optimizer partition: gate no-decay, down2 decayed like down
    rr = build(NBTReRead=1)
    dec, nod = partition_weight_decay(rr)
    assert all(any(n.endswith(f'transformer_layer.{i}.reread_gate') for n in nod) for i in range(4))
    assert all(any(n.endswith(f'transformer_layer.{i}.down2.weight') for n in dec) for i in range(4))
    print('  weight-decay partition OK')
    # 5) guards
    for over, what in ((dict(NBTReRead=3), 'index == NBTInnerLayers'), (dict(NBTReRead=1, NBTInnerLayers=1), 'one inner layer')):
      try:
        cfg(**over)
      except ValueError:
        continue
      raise AssertionError(f'guard missing: {what}')
    print('  guards OK')
  finally:
    for p in ('data', 'exec', 'monitoring', 'net', 'opt'):
      f = f'{cfg_dir}/{tag}_ceres_{p}.json'
      if os.path.exists(f):
        os.remove(f)
  print('test_nbt_reread OK')


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2])
