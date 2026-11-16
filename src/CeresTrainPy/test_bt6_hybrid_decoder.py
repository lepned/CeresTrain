"""NBT-BT6 hybrid WITH the move-token decoder and a configurable GLU width (config UseMoveTokens + BT6Hybrid,
BT6FFNMultiplier). Small nets, CPU.

  (a) defaults unchanged: BT6FFNMultiplier absent == 1 (same parameter names and shapes as before), the hybrid
      without the decoder still builds BT6's policy head
  (b) decoder variant: no BT6 policy / soft-policy head and no flat policy MLP; the decoder owns the policy (1858
      logits) and its inject reaches the value heads; GLU hidden = BT6FFNMultiplier x inner width
  (c) the real loss path (policy + value + value-order): finite, and EVERY trainable parameter receives a gradient
      (no unused parameter: DDP requirement)
  (d) config: SoftPolicyWeight / LossMoveTokenAuxMLPMultiplier / a fractional GLU width are refused; the kernel-
      builder carrier export refuses the decoder variant
  (e) state_dict save / reload gives identical outputs

  python test_bt6_hybrid_decoder.py
"""
import torch

from test_bt6_hybrid import batch_for, build, randomise, records, run_loss

NET_DEC = {'NumLayers': 3, 'NBTInnerLayers': 3, 'BT6EdgeSites': [0, 1], 'BT6TermsInnerLayers': [0],
           'BT6Embedding': 'linear', 'BT6FFNMultiplier': 2,
           'BT6SharedSmolgen': True, 'BT6SharedSmolgenDim': 48, 'BT6SmolgenPerSquareDim': 8, 'BT6SmolgenHeadDim': 16,
           'UseMoveTokens': True, 'MoveTokenDim': 32, 'MoveTokenLayers': 2, 'MoveTokenHeads': 4, 'MoveTokenFFNMult': 2,
           'MoveTokenMax': 96, 'MoveTokenPostMove': True}
OPT_DEC = {'SoftPolicyWeight': 0, 'LossMoveTokenValueOrderMultiplier': 0.15, 'MoveTokenValueOrderTopK': 5,
           'MuonAdamWScope': 'all-non-trunk+decoder'}


def shapes(net):
  return {k: tuple(v.shape) for k, v in net.state_dict().items()}


def test_defaults_unchanged():
  base = build({'BT6TermsInnerLayers': [0]})
  one = build({'BT6TermsInnerLayers': [0], 'BT6FFNMultiplier': 1})
  assert shapes(base) == shapes(one)
  assert all(torch.equal(a, b) for a, b in zip(base.state_dict().values(), one.state_dict().values()))
  assert base.bt6h.policy is not None and base.policy_head is None and not base.use_move_tokens
  print('test_defaults_unchanged: OK')


def test_decoder_variant():
  net = build(NET_DEC, OPT_DEC)
  randomise(net)
  with torch.no_grad():                                   # live smolgen bias and decoder value inject
    net.smolgenPrepLayer.weight.normal_(0, 0.05)
    for name, p in net.named_parameters():
      if name.startswith('move_tokens.') and ('inject' in name) and float(p.abs().max()) == 0:
        p.normal_(0, 0.05)
  assert net.bt6h.policy is None and net.bt6h.soft_policy is None and net.policy_head is None and net.use_move_tokens
  mid = net.transformer_layer[0].inner[0].cfg.d_model
  for blk in net.transformer_layer:
    for layer in blk.inner:
      assert layer.ffn.down.in_features == 2 * mid, (layer.ffn.down.in_features, mid)
  assert [l.terms for l in net.transformer_layer[0].inner] == [True, False, False]
  names = [n for n, _ in net.named_parameters()]
  assert not any(n.startswith(('bt6h.policy', 'bt6h.soft_policy', 'policy_head')) for n in names)
  assert any(n.startswith('move_tokens.') for n in names) and any(n.startswith('bt6h.sites') for n in names)
  sq = records(8)
  # ---- forward
  net.eval()
  with torch.no_grad():
    out = net(sq, None)
  policy, value = out[0], out[1]
  assert policy.shape == (8, 1858) and value.shape == (8, 3) and torch.isfinite(policy).all() and torch.isfinite(value).all()
  # the decoder's inject reaches the value heads: zeroing it changes the value
  saved = {k: v.clone() for k, v in net.state_dict().items()}
  with torch.no_grad():
    for name, p in net.named_parameters():
      if name.startswith('move_tokens.') and 'inject' in name:
        p.zero_()
    value0 = net(sq, None)[1]
  assert float((value0 - value).abs().max()) > 1e-6, 'decoder value inject is dead'
  net.load_state_dict(saved)
  # ---- (c) loss path and gradient participation
  net.train()
  loss = run_loss(net, batch_for(8), sq)
  assert torch.isfinite(loss)
  loss.backward()
  missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
  assert not missing, f'parameters without gradient: {missing[:12]} ({len(missing)})'
  # ---- (e) save / reload
  net.eval()
  twin = build(NET_DEC, OPT_DEC)
  twin.load_state_dict(net.state_dict(), strict=True)
  twin.eval()
  with torch.no_grad():
    a, b = net(sq, None), twin(sq, None)
  assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
  n_total = sum(p.numel() for p in net.parameters())
  n_dec = sum(p.numel() for n, p in net.named_parameters() if n.startswith('move_tokens.'))
  print(f'test_decoder_variant: OK  ({n_total:,} params, decoder {n_dec:,}; decoder owns the policy, inject live, '
        f'GLU hidden {2 * mid}, every parameter has a gradient)')
  return net


def test_refusals_and_export(net):
  for net_over, opt_over, what in (
      (NET_DEC, {**OPT_DEC, 'SoftPolicyWeight': 2.0}, 'SoftPolicyWeight'),
      (NET_DEC, {**OPT_DEC, 'LossMoveTokenAuxMLPMultiplier': 0.1}, 'LossMoveTokenAuxMLPMultiplier'),
      ({**NET_DEC, 'BT6FFNMultiplier': 1.3}, OPT_DEC, 'BT6FFNMultiplier'),
      ({'MoveTokenDim': 32}, None, 'UseMoveTokens is off')):
    try:
      build(net_over, opt_over)
    except ValueError as e:
      assert what in str(e), (what, str(e))
    else:
      raise AssertionError(f'{what} was not refused')
  from bt6_hybrid import export_served_hybrid
  try:
    export_served_hybrid(net)
  except NotImplementedError:
    pass
  else:
    raise AssertionError('export_served_hybrid accepted the decoder variant')
  print('test_refusals_and_export: OK')


if __name__ == '__main__':
  test_defaults_unchanged()
  test_refusals_and_export(test_decoder_variant())
