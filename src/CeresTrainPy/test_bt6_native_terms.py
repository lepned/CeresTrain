"""BT6's term reads on OUR OWN encoder layers (config BT6NativeLayers; bt6_native_terms.py). Small nets, CPU.

  (a) the term assembly equals bt6_net.BT6Layer.forward_reference term by term: with the same q / k and the same
      reader tensors, L (1 + door) == the reference layer's H
  (b) identity at initialisation: a fresh net computes the same outputs with the term reads attached and with the
      term layers run as plain layers (terms detached)
  (c) the real loss path with every table live: finite, every trainable parameter has a gradient; reader tensors
      are in the no-decay class and out of Muon by the hybrid's name rules
  (d) QK-clip: the monitor sees the post-door logits; a clip scales the head's Q / K rows AND Pq / Pk, so the
      q/k-dependent part of the logits shrinks by exactly gamma
  (e) with the move-token decoder (UseMoveTokens): builds, the decoder owns the policy, loss finite
  (f) config refusals; state_dict save / reload identical

  python test_bt6_native_terms.py
"""
import math

import torch

from test_bt6_hybrid import batch_for, build, randomise, records, run_loss
import bt6_net
from bt6_native_terms import BT6NativeTerms

NATIVE = {'BT6NativeLayers': True, 'NumLayers': 3, 'NBTInnerLayers': 3, 'BT6EdgeSites': [0, 1],
          'BT6TermsInnerLayers': [0], 'BT6Embedding': 'linear', 'FFNMultiplier': 2,
          'NBTSharedSmolgen': True, 'SmolgenDimPerSquare': 8, 'SmolgenDim': 32, 'SmolgenToHeadDivisor': 2,
          'NBTProjActivation': 'Swish'}
DECODER = {'UseMoveTokens': True, 'MoveTokenDim': 32, 'MoveTokenLayers': 2, 'MoveTokenHeads': 4, 'MoveTokenFFNMult': 2,
           'MoveTokenMax': 96, 'MoveTokenPostMove': True}
OPT_DEC = {'SoftPolicyWeight': 0, 'LossMoveTokenValueOrderMultiplier': 0.15, 'MoveTokenValueOrderTopK': 5,
           'MuonAdamWScope': 'all-non-trunk+decoder'}


def term_modules(net):
  return [(n, m) for n, m in net.named_modules() if isinstance(m, BT6NativeTerms)]


def live(net, seed=11):
  """Non-trivial reader tables, sites and smolgen (randomise covers the names ending in the reader suffixes)."""
  randomise(net, seed=seed)
  with torch.no_grad():
    if getattr(net, 'smolgenPrepLayer', None) is not None:
      net.smolgenPrepLayer.weight.normal_(0, 0.05)


def test_term_assembly():
  torch.manual_seed(5)
  cfg = bt6_net.BT6Config(d_model=64, layers=1, heads=4, ffn_hidden=64, edge_sites=(), site_hidden=32, policy_width=32,
                          error_head=False)
  layer = bt6_net.BT6Layer(cfg)
  prologue = bt6_net.BT6Prologue(cfg)
  terms = BT6NativeTerms(cfg)
  g = torch.Generator().manual_seed(7)
  with torch.no_grad():
    for name in bt6_net.READER_NAMES + ('Pk', 'Pq'):
      p = getattr(layer, name)
      p.copy_(torch.randn(p.shape, generator=g) * 0.3 + (0.7 if name == 't_node' else 0.0))
      getattr(terms, name).copy_(p)
  from bt6_relations import BT6Relations
  from bt6_hybrid import pieces_from_squares
  sq = records(6)
  ctx, S, e = prologue.context(BT6Relations()(*pieces_from_squares(sq)))
  e = e + torch.randn(e.shape, generator=g) * 0.5        # a non-trivial edge state
  x = torch.randn(6, 64, 64, generator=g)
  _, H_ref = layer.forward_reference(x, ctx, S, e, prologue)
  y = layer.ln1(x)
  Q = layer.q(y).view(6, 64, 4, 16).permute(0, 2, 1, 3)
  K = layer.k(y).view(6, 64, 4, 16).permute(0, 2, 1, 3)
  L, door, _ = terms.logits(Q, K, ctx, e, ctx.read_state(e, torch.float32))
  H = L * (1 + door)
  err = float((H - H_ref).abs().max() / H_ref.abs().max())
  assert err < 1e-4, err                                                      # float32, different contraction order
  print(f'(a) term assembly == BT6Layer.forward_reference: max relative difference {err:.1e}')


def test_identity_and_training():
  net = build(NATIVE)
  T = term_modules(net)
  assert len(T) == 3 and all('.inner.0.attention.bt6t' in n for n, _ in T), [n for n, _ in T]
  from encoder_layer import EncoderLayer
  assert all(isinstance(l, EncoderLayer) for blk in net.transformer_layer for l in blk.inner)
  assert net.bt6h.policy is not None and net.policy_head is None
  sq = records(8)
  # ---- (b) identity at initialisation (smolgen made live first: the bias path must agree too)
  with torch.no_grad():
    net.smolgenPrepLayer.weight.normal_(0, 0.05)
  net.eval()
  with torch.no_grad():
    with_terms = net(sq, None)
    for blk in net.transformer_layer:
      blk.inner[0].attention.bt6t = None
    plain = net(sq, None)
    for (_, m), blk in zip(T, net.transformer_layer):
      blk.inner[0].attention.bt6t = m
  d_pol = float((with_terms[0] - plain[0]).abs().max())
  d_val = float((with_terms[1] - plain[1]).abs().max())
  assert d_pol < 1e-4 and d_val < 1e-5, (d_pol, d_val)
  # ---- live tables change the function
  live(net)
  with torch.no_grad():
    moved = net(sq, None)
  assert float((moved[1] - plain[1]).abs().max()) > 1e-4
  # ---- (c) loss path, gradient participation, optimiser name rules
  net.train()
  loss = run_loss(net, batch_for(8), sq)
  assert torch.isfinite(loss)
  loss.backward()
  missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
  assert not missing, missing[:10]
  from bt6_hybrid import bt6_muon_excluded, is_bt6_layer_reader
  from wd_partition import partition_weight_decay
  decay, no_decay = partition_weight_decay(net)
  readers = [n for n, _ in net.named_parameters() if '.bt6t.' in n]
  assert len(readers) == 30 and all(is_bt6_layer_reader(n) and bt6_muon_excluded(n) for n in readers)
  assert all(n in no_decay and n not in decay for n in readers)
  # ---- (f) save / reload
  net.eval()
  twin = build(NATIVE)
  twin.load_state_dict(net.state_dict(), strict=True)
  twin.eval()
  with torch.no_grad():
    a, b = net(sq, None), twin(sq, None)
  assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
  print(f'(b) identity at init: policy {d_pol:.1e}, value {d_val:.1e};  (c) loss finite, all parameters have a gradient, '
        f'{len(readers)} reader tensors no-decay and out of Muon;  (f) save / reload identical')
  return net


def test_qk_clip(net):
  sq = records(8)
  att = net.transformer_layer[1].inner[0].attention
  rec = {}
  orig = att.bt6t.logits

  def capture(Q, K, ctx, e, rs):
    out = orig(Q, K, ctx, e, rs)
    rec['L'], rec['door'], rec['args'] = out[0].detach().clone(), out[1].detach().clone(), (Q.detach(), K.detach(), ctx, e.detach(), rs.detach())
    return out

  att.bt6t.logits = capture
  att.qk_clip_monitor = True
  att._last_max_logit = None
  net.train()
  with torch.no_grad():
    net(sq, None)
  att.bt6t.logits = orig
  m = att._last_max_logit.clone()
  assert m is not None and m.shape == (att.num_heads,)
  # the static (not q/k-dependent) part of L: the same call with Q = K = 0
  Q, K, ctx, e, rs = rec['args']
  with torch.no_grad():
    static = orig(torch.zeros_like(Q), torch.zeros_like(K), ctx, e, rs)[0]
    tau = float(m.median())
    pq0 = att.bt6t.Pq.clone()
    clipped = att.apply_qk_clip(tau)
    assert 0 < clipped < att.num_heads and att._last_max_logit is None
    gamma = (tau / m).clamp(max=1.0)
    s = gamma.sqrt()
    assert torch.allclose(att.bt6t.Pq, pq0 * s[:, None, None])
    # Q and K of the clipped layer scale by sqrt(gamma) per head (its input is unchanged)
    L1 = orig(Q * s.view(1, -1, 1, 1), K * s.view(1, -1, 1, 1), ctx, e, rs)[0]
  want = gamma.view(1, -1, 1, 1) * (rec['L'] - static) + static
  err = float((L1 - want).abs().max() / want.abs().max())
  assert err < 1e-5, err
  att.qk_clip_monitor = False
  print(f'(d) QK-clip: {clipped} of {att.num_heads} heads clipped; Pq / Pk scaled; q/k part of the logits x gamma (rel {err:.1e})')


def test_decoder_and_refusals():
  net = build({**NATIVE, **DECODER}, OPT_DEC)
  live(net)
  assert net.bt6h.policy is None and net.use_move_tokens and len(term_modules(net)) == 3
  sq = records(8)
  net.train()
  loss = run_loss(net, batch_for(8), sq)
  assert torch.isfinite(loss)
  loss.backward()
  missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
  assert not missing, missing[:10]
  net.eval()
  with torch.no_grad():
    out = net(sq, None)
  assert out[0].shape == (8, 1858) and torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all()
  for over, what in (({'BT6SharedSmolgen': True}, 'NBTSharedSmolgen'), ({'BT6FFNMultiplier': 2}, 'BT6FFNMultiplier'),
                     ({'BT6CheckpointBlocks': True}, 'not wired'), ({'SoftCapCutoff': 50}, 'plain QKV'),
                     ({'NBTSharedSmolgen': False}, 'per-layer smolgen')):
    try:
      build({**NATIVE, **over})
    except (ValueError, AssertionError) as e:
      assert what in str(e), (what, str(e))
    else:
      raise AssertionError(f'{over} was not refused')
  from bt6_hybrid import export_served_hybrid
  try:
    export_served_hybrid(net)
  except NotImplementedError:
    pass
  else:
    raise AssertionError('export_served_hybrid accepted a BT6NativeLayers net')
  n_total = sum(p.numel() for p in net.parameters())
  print(f'(e) with the move-token decoder: {n_total:,} params, decoder owns the policy, loss finite, all parameters have '
        f'a gradient; refusals OK')


if __name__ == '__main__':
  test_term_assembly()
  test_qk_clip(test_identity_and_training())
  test_decoder_and_refusals()
  print('test_bt6_native_terms: OK')
