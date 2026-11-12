"""Tests for the move-token soft-policy readout (LossMoveTokenSoftPolicyMultiplier; move_tokens.py / ceres_net.py), 2026-11-12.

  python test_mt_soft_policy.py <cfg_dir> <cfg_id>   # real-net tests on a small copy of an NBT move-token config (CPU)
"""
import json, os, shutil, sys
import torch

from test_piece_tokens import _squares


def main(cfg_dir, cfg_id):
  from config_bootstrap import bootstrap_env_from_config
  bootstrap_env_from_config(os.path.dirname(cfg_dir), cfg_id, quiet=True, configs_dir=cfg_dir)
  from config import Configuration
  from ceres_net import CeresNet
  from move_tokens import MT_FLOOR
  tag = 'zz_mtsoftpol'
  with open(f'{cfg_dir}/{cfg_id}_ceres_net.json') as f:
    base_net = json.load(f)
  base_net.update(ModelDim=384, NumLayers=3, NumHeads=4, NBTInnerHeads=4, NBTInnerLayers=3, NBTWidthDivisor=3,
                  EGTEdgeStream=False, MoveTokenEdgeInput=False)
  with open(f'{cfg_dir}/{cfg_id}_ceres_opt.json') as f:
    base_opt = json.load(f)

  def build(**opt_over):
    with open(f'{cfg_dir}/{tag}_ceres_net.json', 'w') as f:
      json.dump(base_net, f, indent=1)
    opt = dict(base_opt); opt.update(opt_over)
    with open(f'{cfg_dir}/{tag}_ceres_opt.json', 'w') as f:
      json.dump(opt, f, indent=1)
    c = Configuration(cfg_dir, tag)
    torch.manual_seed(0)
    return CeresNet(None, c, policy_loss_weight=1, value_loss_weight=1, moves_left_loss_weight=0, unc_loss_weight=0,
                    value2_loss_weight=0, q_deviation_loss_weight=0, value_diff_loss_weight=0, value2_diff_loss_weight=0,
                    action_loss_weight=0, uncertainty_policy_weight=0, action_uncertainty_loss_weight=0, q_ratio=1)

  try:
    for p in ('data', 'exec'):
      shutil.copy(f'{cfg_dir}/{cfg_id}_ceres_{p}.json', f'{cfg_dir}/{tag}_ceres_{p}.json')
    sq = _squares(4)
    base, soft = build(), build(LossMoveTokenSoftPolicyMultiplier=8, MoveTokenSoftPolicyTemp=4)
    assert base.use_move_tokens and not base.move_tokens.soft_policy and soft.move_tokens.soft_policy
    assert soft.mt_spol_w == 8 and soft.mt_spol_temp == 4
    # 0) the soft target: the per-row floor (>= 2 entries at the row minimum) is dropped, the rest is p^(1/T)
    from ceres_net import soft_policy_target
    raw = torch.zeros(5, 1858)
    raw[0, :6] = torch.tensor([0.90, 0.06, 0.02, 0.0067, 0.0067, 0.0067])          # TPG-style: floor = one-visit unit
    raw[1, :4] = torch.tensor([0.70, 0.2995, 0.0005, 0.0005]).half().float()         # V6-style fixed floor (fp16 values)
    raw[2, :3] = torch.tensor([0.60, 0.30, 0.10])                                   # all visited: unique minimum is kept
    raw[3, :2] = torch.tensor([0.50, 0.50])                                         # all equal: nothing dropped
    st = soft_policy_target(raw, 4.0)                                               # row 4: no legal move (value-only row)
    assert torch.isfinite(st).all() and torch.allclose(st[:4].sum(-1), torch.ones(4), atol=1e-5) and st[4].sum() == 0
    assert (st[0, 3:6] == 0).all() and (st[0, :3] > 0).all() and (st[1, 2:4] == 0).all() and (st[1, :2] > 0).all()
    assert (st[2, :3] > 0).all() and torch.allclose(st[3, :2], torch.tensor([0.5, 0.5]))
    exp0 = torch.tensor([0.90, 0.06, 0.02]).pow(0.25); assert torch.allclose(st[0, :3], exp0 / exp0.sum(), atol=1e-6)
    assert torch.equal(soft_policy_target(raw.half(), 4.0) > 0, st > 0), 'fp16 targets select a different floor'
    # 1) the only new state is the readout + its bias; every other init is bit-identical (RNG state restored)
    sb, ss = base.state_dict(), soft.state_dict()
    new = {'move_tokens.spol.weight', 'move_tokens.mt_spol_bias'}
    assert set(ss) - set(sb) == new and not (set(sb) - set(ss)), f'unexpected state-dict keys: {set(ss) ^ set(sb)}'
    assert all(torch.equal(sb[k], ss[k]) for k in sb), 'initial weights differ'
    assert tuple(ss['move_tokens.spol.weight'].shape) == (4, soft.move_tokens.pol.in_features)
    # 2) eval graph unchanged: identical outputs, nothing stashed
    with torch.no_grad():
      ob, os_ = base.eval()(sq, None), soft.eval()(sq, None)
    assert all(torch.equal(a, b) for a, b in zip(ob, os_) if torch.is_tensor(a)), 'eval outputs changed'
    assert soft.move_tokens._last_spol is None
    # 3) training forward stashes [B,1858] logits with the SAME legal/absent layout as the policy output
    soft.train()
    out = soft(sq, None)
    sp = soft.move_tokens._last_spol
    assert sp is not None and sp.shape == out[0].shape == (4, 1858)
    assert torch.equal(sp.float() < MT_FLOOR / 2, out[0].float() < MT_FLOOR / 2), 'absent-move layout differs from pol'
    # 4) the loss: gradient reaches the readout, its bias, the decoder body and the trunk; and with T=1 and no
    #    main policy readout
    present = (out[0].float() > MT_FLOOR / 2)
    tgt = torch.rand(4, 1858) * present
    tgt[:, :] = torch.where(present, tgt.clamp(min=0.3), torch.zeros_like(tgt))     # a shared floor on ~30 % of the moves
    tgt = tgt / tgt.sum(-1, keepdim=True)
    legal = tgt > 0
    t = soft_policy_target(tgt, soft.mt_spol_temp)
    assert ((t > 0) <= legal).all() and (t > 0).sum() < legal.sum(), 'floor entries were not dropped'
    logp = torch.log_softmax(torch.where(legal, sp.float(), torch.full_like(sp.float(), -1e9)), dim=-1)
    ref = -(t * logp).sum(-1).mean() + (t * torch.log(t.clamp(min=1e-30))).sum(-1).mean()   # CE minus target entropy
    ref.backward()
    mt = soft.move_tokens
    for nm, prm in (('spol', mt.spol.weight), ('bias', mt.mt_spol_bias), ('decoder body', mt.w_in.weight)):
      assert prm.grad is not None and prm.grad.abs().sum() > 0, f'no gradient into {nm}'
    # trunk: at init everything behind a block's zero-init `up` has zero gradient, so check the trunk as a whole
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in soft.named_parameters() if 'transformer_layer' in n), 'no gradient into the trunk'
    assert mt.pol.weight.grad is None or mt.pol.weight.grad.abs().sum() == 0, 'soft loss leaked into the main readout'
    # 4b) an eval forward clears a stash that the loss did not consume
    assert soft.move_tokens._last_spol is not None
    with torch.no_grad():
      soft.eval()(sq, None)
    assert soft.move_tokens._last_spol is None, 'stale soft-policy stash survived an eval forward'
    # 5) guards
    for over in (dict(MoveTokenSoftPolicyTemp=4),
                 dict(LossMoveTokenSoftPolicyMultiplier=8, MoveTokenSoftPolicyTemp=0.5),
                 dict(LossMoveTokenSoftPolicyMultiplier=8, MoveTokenSoftPolicyTemp=0),
                 dict(LossMoveTokenSoftPolicyMultiplier=-1)):
      try:
        build(**over)
        raise AssertionError(f'guard did not fire for {over}')
      except ValueError:
        pass
    print('test_mt_soft_policy: OK')
  finally:
    for p in ('data', 'exec', 'opt', 'net'):
      f = f'{cfg_dir}/{tag}_ceres_{p}.json'
      if os.path.exists(f):
        os.remove(f)


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2])
