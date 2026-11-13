"""Tests of the NBT-BT6 hybrid's CONFIG LEVERS (config.py: BT6TermsInnerLayers, BT6ShareReadersPerBlock,
BT6StackReadsPerBlock, BT6PairDtype). All levers default off; test_bt6_hybrid.py covers the exact-BT6 net.

  python test_bt6_hybrid_levers.py            all
  python test_bt6_hybrid_levers.py terms      only the BT6TermsInnerLayers test
"""
import io
import sys

import numpy as np
import torch

from test_bt6_hybrid import batch_for, build, randomise, records, run_loss

READER_SUFFIXES = ('.t_node', '.t_edge', '.C1', '.Pk', '.Pq', '.C2', '.We', '.Wm', '.Wg', '.bg')


def rel(a, b):
  return float((a - b).abs().max() / (b.abs().max() + 1e-12))


def served_check(net, sq):
  """The whole forward against the kernel-form forward on the export (FP16-representable dense weights first)."""
  from bt6_hybrid import export_served_hybrid, forward_served_hybrid, is_bt6_layer_reader, served_meta
  from bt6_net import served_to_torch
  with torch.no_grad():
    for name, p in net.named_parameters():
      if not (is_bt6_layer_reader(name) or name.startswith(('bt6h.prologue', 'bt6h.sites'))):
        p.copy_(p.half().float())
    out = net(sq, None)
  exported = export_served_hybrid(net)
  ref = forward_served_hybrid(served_to_torch(exported), sq, served_meta(net))
  worst = 0.0
  for name, tensor in {'policy': out[0], 'value': out[1], 'mlh': out[2], 'unc': out[3], 'value2': out[4]}.items():
    diff, scale = float((tensor.double() - ref[name].double()).abs().max()), max(1.0, float(tensor.abs().max()))
    worst = max(worst, diff / scale)
    assert diff < 2e-4 * scale, (name, diff, scale)
  return worst, exported


def steps(over, count=12, autocast=False):
  m = build({**over, 'BT6CheckpointBlocks': True}).train()
  sq16, batch16 = records(16), batch_for(16)
  optimizer = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.0)
  losses = []
  for _ in range(count):
    optimizer.zero_grad()
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
      loss = run_loss(m, batch16, sq16)
    assert torch.isfinite(loss), loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    optimizer.step()
    losses.append(float(loss.detach()))
  assert all(np.isfinite(losses)), losses
  return losses


# --------------------------------------------------------------------------- 1. BT6TermsInnerLayers
def test_terms_lever():
  """BT6TermsInnerLayers [0]: inner layer 0 of every block is the BT6 layer (bit-equal to a stand-alone bt6_net
  layer); the other inner layers are plain and equal a bt6_net layer whose term parameters are all zero with
  t_node = 1 (no relation / pair / edge-read term, door 0, post-softmax gate 2 sigmoid(0 + 0) = 1); they hold no
  reader parameters; a site reads q.k / sqrt(hd) of the block's last (plain) layer; export, optimiser groups,
  checkpointing, save / reload and a few optimiser steps."""
  from bt6_hybrid import bt6_muon_excluded, is_bt6_layer_reader, parameter_group_table
  from bt6_net import BT6Layer
  from wd_partition import partition_weight_decay
  over = {'NumLayers': 3, 'NBTInnerLayers': 3, 'BT6EdgeSites': [0, 1], 'BT6TermsInnerLayers': [0]}
  net = build(over).eval()
  randomise(net)
  sq = records(6)
  cfg = net.bt6h.layer_config
  for block in net.transformer_layer:
    assert [layer.terms for layer in block.inner] == [True, False, False]
    assert sum(('.' + n).endswith(READER_SUFFIXES) for n, _ in block.inner[0].named_parameters()) == 10
    for layer in block.inner[1:]:
      assert not any(('.' + n).endswith(READER_SUFFIXES) for n, _ in layer.named_parameters()), \
          'a plain layer must hold no reader parameters'
  captured, hooks = [], []
  for b, block in enumerate(net.transformer_layer):
    for j, layer in enumerate(block.inner):
      hooks.append(layer.register_forward_hook(lambda m, i, o, key=('layer', b, j): captured.append((key, m, i, o))))
  for key, site in net.bt6h.sites.items():
    hooks.append(site.register_forward_hook(lambda m, i, o, key=('site', int(key)): captured.append((key, m, i, o))))
  with torch.no_grad():
    net(sq, None)
  for h in hooks:
    h.remove()
  worst_plain, last_H, n_term, n_sites = 0.0, {}, 0, 0
  for key, module, inputs, output in captured:
    if key[0] == 'site':
      assert torch.equal(inputs[1], last_H[key[1]].float()), 'the site must read H of the last inner layer of its block'
      n_sites += 1
      continue
    x, E, S, e, prologue = inputs[:5]
    if module.terms:
      twin = BT6Layer(cfg)
      twin.load_state_dict(module.state_dict())
      with torch.no_grad():
        x2, H2 = twin(*inputs)
      assert torch.equal(x2, output[0]) and torch.equal(H2, output[1]), key
      n_term += 1
    else:
      # a full BT6 layer with ZERO term parameters and t_node = 1, in the literal einsum form and in the fast form
      twin = BT6Layer(cfg)
      report = twin.load_state_dict(module.state_dict(), strict=False)
      assert sorted(report.missing_keys) == sorted(n[1:] for n in READER_SUFFIXES) and not report.unexpected_keys
      with torch.no_grad():
        for n, p in twin.named_parameters():
          if ('.' + n).endswith(READER_SUFFIXES):
            p.fill_(1.0 if n == 't_node' else 0.0)
        x2, H2 = twin.forward_reference(x, E, S, e, prologue)
        x3, H3 = twin(x, E, S, e, prologue)
      assert rel(output[0], x2) < 2e-5 and rel(output[0], x3) < 2e-5, (key, rel(output[0], x2))
      # the training form of the plain layer (fused attention primitive) == its eval form
      module.train()
      with torch.no_grad():
        x4, H4 = module(*inputs)
      module.eval()
      assert rel(x4, output[0]) < 2e-5 and (H4 is None) == (output[1] is None), key
      assert H4 is None or rel(H4, output[1]) < 2e-5
      worst_plain = max(worst_plain, rel(output[0], x2))
      if output[1] is not None:
        assert key[2] == 2 and rel(output[1], H2) < 2e-5 and rel(output[1], H3) < 2e-5, key
        worst_plain = max(worst_plain, rel(output[1], H2))
      else:
        assert key[2] == 1, 'only the last inner layer computes H'
    last_H[key[1]] = output[1]
  assert n_term == 3 and n_sites == 2
  # whole forward vs the kernel-form forward on the export (plain layers exported as zero tables)
  worst, exported = served_check(net, sq)
  for b in range(3):
    for j in (1, 2):
      g = f'/encoder{b * 3 + j}/mha/egt'
      assert all(not exported[f'{g}/{n}'].any() for n in ('attack', 'key', 'query', 'scaled_coefficients', 'constant_bias',
                                                           'edge_read/w', 'door/w', 'gate/w', 'gate/b'))
      assert np.allclose(exported[f'{g}/qk_scale'], 1.0 / np.sqrt(cfg.head_dim))
  # optimiser groups, gradient participation, checkpoint on / off, save / reload, steps
  decay, no_decay = partition_weight_decay(net)
  readers = [n for n, _ in net.named_parameters() if is_bt6_layer_reader(n)]
  assert len(readers) == 10 * 3 and all(n in no_decay and bt6_muon_excluded(n) for n in readers)
  table = parameter_group_table(net, 'all-non-trunk')
  n_muon = sum(r[3] for r in table if r[0] == 'Muon')
  n_adamw = sum(r[3] for r in table if r[0] == 'AdamW')
  sqs, batch = records(4), batch_for(4)
  results = {}
  for flag in (False, True):
    m = build({**over, 'BT6CheckpointBlocks': flag}).train()
    randomise(m)
    outs = m(sqs, None)
    m._last_sp_out = None
    m.zero_grad()
    loss = run_loss(m, batch, sqs)
    loss.backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0.0]
    assert not dead, f'parameters without (or with an all-zero) gradient: {dead}'
    results[flag] = ([o.detach().clone() for o in outs[:5]], {n: p.grad.detach().clone() for n, p in m.named_parameters()})
  assert all(torch.equal(a, b) for a, b in zip(results[False][0], results[True][0]))
  worst_grad = max(rel(results[True][1][n], results[False][1][n]) for n in results[False][1])
  assert worst_grad < 1e-4, worst_grad
  buffer = io.BytesIO()
  torch.save(net.state_dict(), buffer)
  buffer.seek(0)
  twin_net = build(over).eval()
  twin_net.load_state_dict(torch.load(buffer))
  with torch.no_grad():
    assert all(torch.equal(a, b) for a, b in zip(net(sq, None)[:5], twin_net(sq, None)[:5]))
  full = build({'NumLayers': 3, 'NBTInnerLayers': 3, 'BT6EdgeSites': [0, 1]})
  try:
    full.load_state_dict(net.state_dict())
    raise AssertionError('a [0]-terms checkpoint must not load silently into an all-terms net')
  except RuntimeError:
    pass
  losses = steps(over)
  assert losses[-1] < 0.8 * losses[0], losses
  count = lambda mod: sum(p.numel() for p in mod.parameters())
  print(f'L1. BT6TermsInnerLayers [0] (3 blocks x 3): layer 0 of each block == stand-alone bt6_net layer (bitwise); the 6 '
        f'plain layers == a bt6_net layer with zero term parameters and t_node 1 (literal and fast form; output and '
        f'site-read H, max relative {worst_plain:.1e}) and hold no reader parameters; the {n_sites} sites read H of the plain '
        f'last layer; export (zero tables for plain layers) vs kernel-form forward {worst:.2e}; {len(readers)} reader '
        f'tensors in no_decay / out of Muon; groups {n_muon} Muon / {n_adamw} AdamW tensors; all parameters get a '
        f'non-zero gradient; checkpoint on / off: forward bit-identical, gradients {worst_grad:.1e}; save / reload '
        f'identical, refused by an all-terms net; 12 steps loss {losses[0]:.3f} -> {losses[-1]:.3f}; parameters '
        f'{count(net):,} vs all-terms {count(full):,}')


# --------------------------------------------------------------------------- 2. shared / stacked readers
def test_share_and_stack():
  from bt6_hybrid import is_bt6_layer_reader
  from wd_partition import partition_weight_decay
  base = {'NumLayers': 3, 'NBTInnerLayers': 3, 'BT6EdgeSites': [0, 1]}
  sq = records(6)
  # stacked reads: same parameters, same function
  a = build(base).eval()
  randomise(a)
  b = build({**base, 'BT6StackReadsPerBlock': True}).eval()
  b.load_state_dict(a.state_dict())
  with torch.no_grad():
    d_stack = max(rel(u, w) for u, w in zip(b(sq, None)[:5], a(sq, None)[:5]))
  assert d_stack < 2e-5, d_stack
  # shared readers: one reader set per block; equals an all-own net given the block's tables in every layer
  s = build({**base, 'BT6ShareReadersPerBlock': True}).eval()
  randomise(s)
  names = [n for n, _ in s.named_parameters() if is_bt6_layer_reader(n)]
  assert len(names) == 3 * (8 + 3 * 2), len(names)        # 8 shared per block + Pk, Pq per layer
  decay, no_decay = partition_weight_decay(s)
  assert all(n in no_decay for n in names)
  state = dict(a.state_dict())
  for key, value in s.state_dict().items():
    if '.bt6_readers.' in key:
      for j in range(3):
        target = key.replace('.bt6_readers.', f'.inner.{j}.')
        assert target in state, target
        state[target] = value
    else:
      state[key] = value
  a.load_state_dict(state)
  with torch.no_grad():
    d_share = max(rel(u, w) for u, w in zip(s(sq, None)[:5], a(sq, None)[:5]))
  assert d_share < 2e-5, d_share
  worst, _ = served_check(s, sq)
  m = build({**base, 'BT6ShareReadersPerBlock': True}).train()
  randomise(m)
  loss = run_loss(m, batch_for(4), records(4))
  loss.backward()
  dead = [n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0.0]
  assert not dead, dead
  losses = steps({**base, 'BT6ShareReadersPerBlock': True})
  assert losses[-1] < 0.8 * losses[0], losses
  count = lambda mod: sum(p.numel() for p in mod.parameters())
  print(f'L2. BT6StackReadsPerBlock: same state_dict, outputs equal to the per-layer form ({d_stack:.1e}); '
        f'BT6ShareReadersPerBlock: {len(names)} reader tensors (8 per block + Pk, Pq per layer), equal to an all-own net '
        f'with the block tables copied into every layer ({d_share:.1e}), export vs kernel form {worst:.2e}, all '
        f'parameters get a gradient, 12 steps loss {losses[0]:.3f} -> {losses[-1]:.3f}; parameters {count(s):,} vs {count(a):,}')


# --------------------------------------------------------------------------- 3. pair dtype
def test_pair_dtype():
  base = {'NumLayers': 3, 'NBTInnerLayers': 3, 'BT6EdgeSites': [0, 1]}
  sq = records(6)
  a = build(base).eval()
  randomise(a)
  b = build({**base, 'BT6PairDtype': 'autocast'}).eval()
  b.load_state_dict(a.state_dict())
  with torch.no_grad():
    exact = max(rel(u, w) for u, w in zip(b(sq, None)[:5], a(sq, None)[:5]))      # no autocast: identical function
    with torch.autocast('cpu', dtype=torch.bfloat16):
      low = max(rel(u.float(), w.float()) for u, w in zip(b(sq, None)[:5], a(sq, None)[:5]))
  assert exact == 0.0, exact
  l32 = steps(base, autocast=True)
  l16 = steps({**base, 'BT6PairDtype': 'autocast'}, autocast=True)
  assert l16[-1] < 0.8 * l16[0], l16
  print(f'L3. BT6PairDtype autocast: without autocast bit-identical to float32; under bf16 autocast the outputs move by '
        f'{low:.3f} relative to the float32 pair tensors; 12 steps under autocast: loss {l16[0]:.3f} -> {l16[-1]:.3f} '
        f'(float32 pair tensors: {l32[0]:.3f} -> {l32[-1]:.3f}), all finite')


if __name__ == '__main__':
  mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
  test_terms_lever()
  if mode == 'all':
    test_share_and_stack()
    test_pair_dtype()
  print('test_bt6_hybrid_levers: OK')
