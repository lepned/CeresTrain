"""Contract test for the gated-attention-output warm-start fold (gate_fold.py, 2026-09-14).

    python test_gate_fold.py [/path/to/ckpt /path/to/configs <config_id>]   (from src/CeresTrainPy, CPU)

1. Synthetic: control net (gate off) vs gated net (gate on, bias 4) loaded from the control's weights:
   WITHOUT the fold the outputs differ (the test bites); WITH the fold policy/value match to fp32 rounding;
   the gate weight receives gradient; refusal when the gate is already trained.
2. Optional real checkpoint: same check with a real (config, checkpoint) pair — the exact resume scenario.
"""
import os, sys
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dual_plane_edge_aux import build, random_boards, fake_batch, run_loss
from gate_fold import fold_gate_on_warm_start

GATE_ENV = 'CERES_GATED_ATTENTION_OUTPUT'


def _outputs(net, sq):
  net.eval()
  with torch.no_grad():
    o = net(sq, None)
  return o[0].float(), o[1].float()


def _load_gated_from(sd, net_kwargs):
  os.environ[GATE_ENV] = '1'
  try:
    g, _ = build(net_kwargs, {}, 'gate1')
  finally:
    os.environ[GATE_ENV] = '0'
  res = g.load_state_dict(sd, strict=False)
  assert not res.unexpected_keys, res.unexpected_keys
  assert res.missing_keys and all('.attn_out_gate.' in k for k in res.missing_keys), res.missing_keys
  return g, res.missing_keys


def synthetic():
  base = {'DualPlanePolicyDecode': False, 'UseMoveTokens': True, 'MoveTokenDim': 64,
          'MoveTokenLayers': 2, 'MoveTokenHeads': 2, 'MoveTokenMax': 64}
  os.environ[GATE_ENV] = '0'
  ctrl, _ = build(base, {}, 'gate0')
  # give the control non-trivial weights (fresh nets have zero-init couplings that would hide differences)
  with torch.no_grad():
    for p in ctrl.parameters():
      p.add_(torch.randn_like(p) * 0.02)
  sd = ctrl.state_dict()
  sq = random_boards(6)
  p0, v0 = _outputs(ctrl, sq)

  g, missing = _load_gated_from(sd, base)
  p1, v1 = _outputs(g, sq)
  d_nofold = float((p1 - p0).abs().max())
  assert d_nofold > 1e-3, f'without the fold the gated net should differ from the control (got {d_nofold:.2e})'
  n, g0 = fold_gate_on_warm_start(g)
  assert n == len(g.transformer_layer) and abs(g0 - 1 / (1 + torch.exp(torch.tensor(-4.0)).item())) < 1e-6, (n, g0)
  p2, v2 = _outputs(g, sq)
  d_pol = float((p2 - p0).abs().max()); d_val = float((v2 - v0).abs().max())
  assert d_pol < 1e-4 and d_val < 1e-5, (d_pol, d_val)
  # gate trains: gradient reaches every gate weight
  g.train(); g.zero_grad(set_to_none=True)
  loss = run_loss(g, fake_batch(6, sq), sq); loss.backward()
  for lyr in g.transformer_layer:
    w = lyr.attention.attn_out_gate.weight
    assert w.grad is not None and w.grad.abs().sum() > 0, 'gate weight gets no gradient'
  # refusal: a trained (non-zero) gate must not be folded
  with torch.no_grad():
    g.transformer_layer[0].attention.attn_out_gate.weight.fill_(0.01)
  try:
    fold_gate_on_warm_start(g); raise AssertionError('fold must refuse a non-zero gate weight')
  except RuntimeError:
    pass
  print(f'  synthetic OK: {len(missing)} fresh gate tensors; no-fold diff {d_nofold:.2e} -> folded diff policy {d_pol:.2e} / value {d_val:.2e}; '
        f'{n} layers folded at g0={g0:.4f}; gate gets gradient; trained-gate refusal OK')


def real(ckpt_path, cfg_dir, cfg_id):
  from config import Configuration
  from ceres_net import CeresNet
  def make():
    cfg = Configuration(cfg_dir, cfg_id)
    return CeresNet(None, cfg,
                    policy_loss_weight=cfg.Opt_LossPolicyMultiplier, value_loss_weight=cfg.Opt_LossValueMultiplier,
                    moves_left_loss_weight=cfg.Opt_LossMLHMultiplier, unc_loss_weight=cfg.Opt_LossUNCMultiplier,
                    value2_loss_weight=cfg.Opt_LossValue2Multiplier, q_deviation_loss_weight=cfg.Opt_LossQDeviationMultiplier,
                    value_diff_loss_weight=cfg.Opt_LossValueDMultiplier, value2_diff_loss_weight=cfg.Opt_LossValue2DMultiplier,
                    action_loss_weight=cfg.Opt_LossActionMultiplier, uncertainty_policy_weight=cfg.Opt_LossUncertaintyPolicyMultiplier,
                    action_uncertainty_loss_weight=cfg.Opt_LossActionUncertaintyMultiplier, q_ratio=cfg.Data_FractionQ)
  sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)['model']
  os.environ[GATE_ENV] = '0'
  ctrl = make(); ctrl.load_state_dict(sd, strict=True)
  os.environ[GATE_ENV] = '1'
  try:
    g = make()
  finally:
    os.environ[GATE_ENV] = '0'
  res = g.load_state_dict(sd, strict=False)
  assert not res.unexpected_keys and all('.attn_out_gate.' in k for k in res.missing_keys), (res.unexpected_keys, res.missing_keys[:3])
  sq = random_boards(8, seed=11)
  p0, v0 = _outputs(ctrl, sq)
  p1, _ = _outputs(g, sq)
  d_nofold = float((p1 - p0).abs().max())
  n, g0 = fold_gate_on_warm_start(g)
  p2, v2 = _outputs(g, sq)
  d_pol = float((p2 - p0).abs().max()); d_val = float((v2 - v0).abs().max())
  assert d_pol < 1e-3 and d_val < 1e-4, (d_pol, d_val)
  print(f'  real checkpoint OK ({os.path.basename(ckpt_path)}): {len(res.missing_keys)} fresh gate tensors, {n} layers folded at g0={g0:.4f}; '
        f'no-fold diff {d_nofold:.2e} -> folded diff policy {d_pol:.2e} / value {d_val:.2e} (policy range {float(p0.abs().max()):.1f})')


if __name__ == '__main__':
  synthetic()
  if len(sys.argv) >= 4:
    real(sys.argv[1], sys.argv[2], sys.argv[3])
  print('ALL OK')
