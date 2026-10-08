"""Contract test for the headwise G1 attention-output gate (config GatedAttentionOutputHeadwise, 2026-10-08).

    python test_gated_attn_headwise.py            (CPU, tiny NBT net; no corpus)

1. Shape: attn_out_gate is Linear(d_model -> num_heads) per attention layer.
2. Step 0: zero weight + bias b scales every head output by sigmoid(b) -- identical to the elementwise gate at the same
   bias (both are the same constant), and different from the ungated net.
3. Live gate: gradients reach attn_out_gate.weight/bias.
4. Headwise without GatedAttentionOutput is refused (would be a silent no-op).
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_egt_edge as T

ENV = ('CERES_GATED_ATTENTION_OUTPUT', 'CERES_GATED_ATTENTION_OUTPUT_HEADWISE', 'CERES_GATED_ATTENTION_OUTPUT_BIAS')


def build_with(gate, headwise, bias='4.0', tag='hg'):
  for k in ENV:
    os.environ.pop(k, None)
  if gate:
    os.environ['CERES_GATED_ATTENTION_OUTPUT'] = '1'
    os.environ['CERES_GATED_ATTENTION_OUTPUT_BIAS'] = bias
  if headwise:
    os.environ['CERES_GATED_ATTENTION_OUTPUT_HEADWISE'] = '1'
  try:
    return T.build({}, tag)
  finally:
    for k in ENV:
      os.environ.pop(k, None)


def gates_of(m):
  return [mod.attn_out_gate for mod in m.modules() if getattr(mod, 'use_gated_attn_out', False)]


def main():
  sq = T.boards(4)
  ctrl = build_with(False, False, tag='ctrl')
  hw = build_with(True, True, tag='hw')
  ew = build_with(True, False, tag='ew')

  # 1 shape
  g_hw, g_ew = gates_of(hw), gates_of(ew)
  assert g_hw and len(g_hw) == len(g_ew), (len(g_hw), len(g_ew))
  for g in g_hw:
    assert g.out_features == T.NET_BASE['NBTInnerHeads'], g
  print(f'OK shape: {len(g_hw)} headwise gates [{g_hw[0].in_features} -> {g_hw[0].out_features}]')

  # 2 step 0: give all three nets the same non-zero NBT up projections (zero-init otherwise -> attention invisible)
  with torch.no_grad():
    gen = torch.Generator().manual_seed(21)
    ups = {n: torch.randn(p.shape, generator=gen) * 0.2 for n, p in ctrl.named_parameters() if n.endswith('.up.weight')}
    P = dict(ctrl.named_parameters())
    for n, w in ups.items():
      P[n].copy_(w)
    # nn.Linear construction draws from the RNG (gate sizes differ), so pair every shared tensor with the control
    sd = ctrl.state_dict()
    for m in (hw, ew):
      missing, unexpected = m.load_state_dict(sd, strict=False)
      assert not unexpected and all('attn_out_gate' in k for k in missing), (missing, unexpected)
    for m in (ctrl, hw, ew):
      m.eval()
    oc, oh, oe = ctrl(sq, None)[0], hw(sq, None)[0], ew(sq, None)[0]
  assert torch.allclose(oh, oe, atol=1e-5), float((oh - oe).abs().max())
  assert not torch.allclose(oh, oc, atol=1e-5), 'gate at sigmoid(4) must differ from the ungated net'
  print(f'OK step 0: headwise == elementwise at bias 4 (max diff {float((oh - oe).abs().max()):.1e}), != ungated')

  # 3 live gradients
  with torch.no_grad():
    gen = torch.Generator().manual_seed(22)
    for g in g_hw:
      g.weight.copy_(torch.randn(g.weight.shape, generator=gen) * 0.3)
  hw.train(); hw.zero_grad(set_to_none=True)
  loss = T.run_loss(hw, T.batch_for(4), sq)
  loss.backward()
  for g in g_hw:
    assert g.weight.grad is not None and g.weight.grad.abs().sum() > 0, 'no gradient to headwise gate weight'
    assert g.bias.grad is not None and g.bias.grad.abs().sum() > 0, 'no gradient to headwise gate bias'
  print(f'OK live: loss {loss.item():.4f}, gradients reach every headwise gate')

  # 4 refusal
  try:
    build_with(False, True, tag='rej')
    raise SystemExit('FAIL: headwise without GatedAttentionOutput accepted')
  except ValueError:
    pass
  print('OK headwise without GatedAttentionOutput refused')
  print('ALL OK')


if __name__ == '__main__':
  main()
