"""Depth probes via config (opt DepthProbeValueWeight / DepthProbePolicyWeight; 2026-10-07).

    python test_depth_probes.py            (CPU, tiny NBT net; reuses test_egt_edge's builders)

Value-only probes (policy weight 0): training loss finite, the shared value probe gets a gradient and so does the trunk
through it; eval outputs are bit-identical to the control (training-only); the retired env var raises.
"""
import os, sys
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_egt_edge as T


def build(opt_over, tag):
  saved = dict(T.OPT_BASE)
  T.OPT_BASE.update(opt_over)
  try:
    return T.build({}, tag)
  finally:
    T.OPT_BASE.clear(); T.OPT_BASE.update(saved)


def main():
  sq, batch = T.boards(4), T.batch_for(4)
  ctrl = build({}, 'ctrl')
  m = build({'DepthProbeValueWeight': 0.25}, 'dp')
  assert m.depth_probes_enabled and m.depth_probe_value_weight == 0.25 and m.depth_probe_policy_weight == 0.0
  sd_c = ctrl.state_dict()
  with torch.no_grad():                                   # same trunk weights (probe params are extra)
    for k, v in m.state_dict().items():
      if k in sd_c:
        v.copy_(sd_c[k])
    for n_, p in list(m.named_parameters()) + list(ctrl.named_parameters()):
      if n_.endswith('.up.weight'):
        torch.manual_seed(abs(hash(n_)) % 1000); p.copy_(torch.randn(p.shape) * 0.05)
  m.train(); m.zero_grad(set_to_none=True)
  loss = T.run_loss(m, batch, sq)
  assert torch.isfinite(loss)
  loss.backward()
  g = m.depth_probe_value.weight.grad
  assert g is not None and g.abs().sum() > 0, 'value probe got no gradient'
  assert any(p.grad is not None and p.grad.abs().sum() > 0 for n_, p in m.named_parameters() if 'transformer_layer' in n_)
  ctrl.eval(); m.eval()
  with torch.no_grad():
    for a, b in zip(ctrl(sq, None), m(sq, None)):
      if torch.is_tensor(a):
        assert torch.equal(a, b), 'depth probes must not change the eval outputs'
  os.environ['CERES_DEPTH_PROBES'] = '1'
  try:
    build({}, 'env'); raise SystemExit('FAIL: retired env accepted')
  except ValueError as e:
    assert 'retired' in str(e)
  finally:
    del os.environ['CERES_DEPTH_PROBES']
  print(f'OK value-only depth probes: loss {float(loss):.4f}, probe + trunk gradients, eval unchanged, env retired')
  print('ALL OK')


if __name__ == '__main__':
  main()
