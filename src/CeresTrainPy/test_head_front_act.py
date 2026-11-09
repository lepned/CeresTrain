"""Tests for HeadFrontActivation (activations inside the shared head front-end, ceres_net.py), 2026-11-09.

  python test_head_front_act.py <cfg_dir> <cfg_id>   # real-net tests on a small copy of an NBT move-token config (CPU)
"""
import json, os, shutil, sys
import torch

from test_piece_tokens import _squares

MARKER = 'head_front_act_marker'


def main(cfg_dir, cfg_id):
  from config_bootstrap import bootstrap_env_from_config
  bootstrap_env_from_config(os.path.dirname(cfg_dir), cfg_id, quiet=True, configs_dir=cfg_dir)
  from config import Configuration
  from ceres_net import CeresNet
  tag = 'zz_headfront'
  with open(f'{cfg_dir}/{cfg_id}_ceres_net.json') as f:
    base_net = json.load(f)
  base_net.update(ModelDim=384, NumLayers=3, NumHeads=4, NBTInnerHeads=4, NBTInnerLayers=3, NBTWidthDivisor=3,
                  EGTEdgeStream=False, MoveTokenEdgeInput=False)

  def build(**over):
    net = dict(base_net); net.update(over)
    with open(f'{cfg_dir}/{tag}_ceres_net.json', 'w') as f:
      json.dump(net, f, indent=1)
    c = Configuration(cfg_dir, tag)
    torch.manual_seed(0)
    return CeresNet(None, c, policy_loss_weight=1, value_loss_weight=1, moves_left_loss_weight=0, unc_loss_weight=0,
                    value2_loss_weight=0, q_deviation_loss_weight=0, value_diff_loss_weight=0, value2_diff_loss_weight=0,
                    action_loss_weight=0, uncertainty_policy_weight=0, action_uncertainty_loss_weight=0, q_ratio=1)

  try:
    for p in ('data', 'exec', 'opt'):
      shutil.copy(f'{cfg_dir}/{cfg_id}_ceres_{p}.json', f'{cfg_dir}/{tag}_ceres_{p}.json')
    sq = _squares(4)
    base, act = build(HeadFrontActivation=False), build(HeadFrontActivation=True)
    assert not base.head_front_act and act.head_front_act
    # 1) no new parameters and identical init (the key draws nothing from the RNG); the only new state-dict key is
    #    the marker buffer, so a strict load across the toggle fails in both directions
    sb, sa = base.state_dict(), act.state_dict()
    assert set(sa) - set(sb) == {MARKER} and not (set(sb) - set(sa)), 'unexpected state-dict keys'
    assert all(torch.equal(sb[k], sa[k]) for k in sb), 'initial weights differ'
    assert [n for n, _ in base.named_parameters()] == [n for n, _ in act.named_parameters()]
    for dst, src in ((base, sa), (act, sb)):
      try:
        dst.load_state_dict(src, strict=True)
        raise AssertionError('strict load across the HeadFrontActivation toggle did not fail')
      except RuntimeError:
        pass
    # 2) flag off is the pre-change expression, flag on is act(shared(act(premap(x)).flatten))
    x = torch.randn(2, 64, base.EMBEDDING_DIM)
    assert torch.equal(base._head_front(x), base.headSharedLinear(base.headPremap(x).reshape(-1, 64 * base.HEAD_PREMAP_PER_SQUARE)))
    assert torch.equal(act._head_front(x), act.Activation(act.headSharedLinear(act.Activation(act.headPremap(x)).reshape(2, -1))))
    # 3) policy comes from the move-token decoder (reads the trunk, not the shared vector): unchanged; value changes
    assert base.use_move_tokens, 'test config must use move tokens'
    with torch.no_grad():
      ob, oa = base.eval()(sq, None), act.eval()(sq, None)
    assert torch.isfinite(oa[1]).all() and torch.equal(ob[0], oa[0]), 'policy changed'
    assert not torch.allclose(ob[1], oa[1]), 'value did not change'
    # 4) gradients reach both front-end linears through the activations
    act.train()
    act(sq, None)[1].float().pow(2).sum().backward()
    for lin in (act.headPremap, act.headSharedLinear):
      assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0, 'no gradient into the head front-end'
    # 5) guards: no-op / bypass combinations are refused
    for over in (dict(HeadsActivationType='None'),):
      try:
        build(HeadFrontActivation=True, **over)
        raise AssertionError(f'guard did not fire for {over}')
      except ValueError:
        pass
    print('test_head_front_act: OK')
  finally:
    for p in ('data', 'exec', 'opt', 'net'):
      f = f'{cfg_dir}/{tag}_ceres_{p}.json'
      if os.path.exists(f):
        os.remove(f)


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2])
