"""Contract test for growing the move-token decoder at resume (decoder_grow.py, 2026-09-16).

    python test_decoder_grow.py [/path/to/ckpt /path/to/configs <grown_config_id>]   (from src/CeresTrainPy, CPU)

1. Synthetic: a 2-block decoder net's weights loaded into a 4-block net: the two new blocks' keys are the only
   missing ones; WITHOUT zeroing the outputs differ; WITH zeroing policy/value match to fp32 rounding; every
   parameter of the new blocks receives gradient; a partially-fresh block is refused.
2. Optional real checkpoint: a 4-block checkpoint into a 6-block config (the server switch scenario).
"""
import os, sys
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dual_plane_edge_aux import build, random_boards, fake_batch, run_loss
from decoder_grow import zero_init_fresh_decoder_blocks, NOOP_FRESH


def _outputs(net, sq):
  net.eval()
  with torch.no_grad():
    o = net(sq, None)
  return o[0].float(), o[1].float()


def synthetic():
  base = {'DualPlanePolicyDecode': False, 'UseMoveTokens': True, 'MoveTokenDim': 64, 'MoveTokenHeads': 2, 'MoveTokenMax': 64}
  small, _ = build(dict(base, MoveTokenLayers=2), {}, 'grow2')
  with torch.no_grad():
    for p in small.parameters():
      p.add_(torch.randn_like(p) * 0.02)
  sd = small.state_dict()
  sq = random_boards(6)
  p0, v0 = _outputs(small, sq)
  big, _ = build(dict(base, MoveTokenLayers=4), {}, 'grow4')
  res = big.load_state_dict(sd, strict=False)
  assert not res.unexpected_keys, res.unexpected_keys
  assert res.missing_keys and all(k.startswith('move_tokens.blocks.2.') or k.startswith('move_tokens.blocks.3.') for k in res.missing_keys), res.missing_keys[:4]
  p1, _ = _outputs(big, sq)
  d_raw = float((p1 - p0).abs().max())
  assert d_raw > 1e-3, f'fresh random blocks should change the output (got {d_raw:.2e})'
  grown = zero_init_fresh_decoder_blocks(big, res.missing_keys)
  assert grown == [2, 3], grown
  p2, v2 = _outputs(big, sq)
  d_pol = float((p2 - p0).abs().max()); d_val = float((v2 - v0).abs().max())
  assert d_pol < 1e-4 and d_val < 1e-5, (d_pol, d_val)
  big.train(); big.zero_grad(set_to_none=True)
  loss = run_loss(big, fake_batch(6, sq), sq); loss.backward()
  # Product rule at step 0: only the zeroed output projections have a non-zero gradient; everything upstream of
  # them in the new block has an exactly-zero (but PRESENT) gradient and wakes up once the projections move.
  # DDP needs every parameter to participate (grad tensor exists), not to be non-zero.
  for i in (2, 3):
    for n, p in big.move_tokens.blocks[i].named_parameters():
      assert p.grad is not None, f'new block {i} param {n} does not participate (DDP would abort)'
    for n in ('proj', 'xproj', 'ffn_out'):
      g = getattr(big.move_tokens.blocks[i], n).weight.grad
      assert g is not None and g.abs().sum() > 0, f'new block {i} {n} gets no gradient at step 0'
  # from-base warm start (every block fresh) is NOT growth: nothing zeroed, random init kept
  fresh_all, _ = build(dict(base, MoveTokenLayers=2), {}, 'grow_all')
  all_keys = [k for k in fresh_all.state_dict() if k.startswith('move_tokens.blocks.')]
  assert zero_init_fresh_decoder_blocks(fresh_all, all_keys) == []
  assert float(fresh_all.move_tokens.blocks[0].proj.weight.detach().abs().sum()) > 0, 'from-base warm start must keep its random init'
  # an existing block that only gains no-op keys (post-move / rel_w) is tolerated
  assert zero_init_fresh_decoder_blocks(big, ['move_tokens.blocks.0.rel_w']) == []
  # refusal: a partially fresh block
  try:
    zero_init_fresh_decoder_blocks(big, [k for k in res.missing_keys if k.startswith('move_tokens.blocks.2.proj')])
    raise AssertionError('partially-fresh block must be refused')
  except RuntimeError:
    pass
  print(f'  synthetic OK: 2 -> 4 blocks, {len(res.missing_keys)} fresh keys; raw diff {d_raw:.2e} -> zeroed diff policy {d_pol:.2e} / value {d_val:.2e}; '
        f'new blocks get gradient; partial-block refusal OK')


def real(ckpt_path, cfg_dir, cfg_id):
  from config import Configuration
  from ceres_net import CeresNet
  def make(cfg):
    return CeresNet(None, cfg,
                    policy_loss_weight=cfg.Opt_LossPolicyMultiplier, value_loss_weight=cfg.Opt_LossValueMultiplier,
                    moves_left_loss_weight=cfg.Opt_LossMLHMultiplier, unc_loss_weight=cfg.Opt_LossUNCMultiplier,
                    value2_loss_weight=cfg.Opt_LossValue2Multiplier, q_deviation_loss_weight=cfg.Opt_LossQDeviationMultiplier,
                    value_diff_loss_weight=cfg.Opt_LossValueDMultiplier, value2_diff_loss_weight=cfg.Opt_LossValue2DMultiplier,
                    action_loss_weight=cfg.Opt_LossActionMultiplier, uncertainty_policy_weight=cfg.Opt_LossUncertaintyPolicyMultiplier,
                    action_uncertainty_loss_weight=cfg.Opt_LossActionUncertaintyMultiplier, q_ratio=cfg.Data_FractionQ)
  sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model']
  n_ckpt = 1 + max(int(k.split('.')[2]) for k in sd if k.startswith('move_tokens.blocks.'))
  big = make(Configuration(cfg_dir, cfg_id))
  n_cfg = len(big.move_tokens.blocks)
  assert n_cfg > n_ckpt, (n_cfg, n_ckpt)
  res = big.load_state_dict(sd, strict=False)
  assert not res.unexpected_keys and all(k.startswith('move_tokens.blocks.') for k in res.missing_keys), (res.unexpected_keys, res.missing_keys[:3])
  # reference = the same config with the checkpoint's block count (built by trimming the grown net's blocks)
  import copy
  ref = copy.deepcopy(big)
  ref.move_tokens.blocks = torch.nn.ModuleList(list(ref.move_tokens.blocks)[:n_ckpt])
  # The reference may carry a post-move branch the checkpoint lacks (switched on at the same resume): that branch
  # is an exact no-op (pm_proj zero-init), so a non-strict load whose only missing keys are post-move keys is the
  # checkpoint's function.
  rres = ref.load_state_dict(sd, strict=False)
  assert not rres.unexpected_keys and all(k.split('.')[3] in NOOP_FRESH for k in rres.missing_keys), rres.missing_keys[:3]
  sq = random_boards(8, seed=11)
  p0, v0 = _outputs(ref, sq)
  p1, _ = _outputs(big, sq)
  d_raw = float((p1 - p0).abs().max())
  grown = zero_init_fresh_decoder_blocks(big, res.missing_keys)
  p2, v2 = _outputs(big, sq)
  d_pol = float((p2 - p0).abs().max()); d_val = float((v2 - v0).abs().max())
  assert grown == list(range(n_ckpt, n_cfg)) and d_pol < 1e-3 and d_val < 1e-4, (grown, d_pol, d_val)
  print(f'  real checkpoint OK ({os.path.basename(ckpt_path)}): {n_ckpt} -> {n_cfg} blocks, grown {grown}; raw diff {d_raw:.2e} -> '
        f'zeroed diff policy {d_pol:.2e} / value {d_val:.2e} (policy range {float(p0.abs().max()):.1f})')


if __name__ == '__main__':
  synthetic()
  if len(sys.argv) >= 4:
    real(sys.argv[1], sys.argv[2], sys.argv[3])
  print('ALL OK')
