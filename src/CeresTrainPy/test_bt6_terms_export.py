"""The kernel-builder carrier export of the two "terms" variants with the move-token decoder (bt6_terms_export.py).
Small nets, CPU, every zero-initialised table made non-trivial.

  (a) variant "bt" (BT6 layers, BT6SharedSmolgen on the term-free layers, GLU x2, decoder):
      forward_served_terms(export_served_terms(net)) reproduces the net -- policy (all 1858 outputs), value, value2
  (b) variant "nt" (our own layers with BT6 term reads, NBTSharedSmolgen on all layers, decoder): the same
  (c) the comparison is not vacuous: dropping the relation / edge tables, the smolgen bias or the decoder's value
      inject from the served tensors moves the outputs by far more than the agreement
  (d) carrier round trip: bt6_carrier.write_carrier / read_carrier returns the same names, shapes, dtypes and bytes
  (e) the decoder's tokens: tokens_from_squares gives the trainer's own selection (valid tokens = candidates)
  (f) variant nt with edge sites WITHOUT the triplet branch (BT6Triplet false): no triplet tensor in the export
  (g) (f) plus MoveTokenEdgeInput (the decoder's tokens read the final BT6 edge state through w_edge, made live):
      /mt/w_edge/w is exported, and dropping it moves the outputs

  python test_bt6_terms_export.py
"""
import os
import tempfile

import numpy as np
import torch

from test_bt6_hybrid import build, randomise, records
from test_bt6_hybrid_decoder import NET_DEC, OPT_DEC
from test_bt6_native_terms import DECODER, NATIVE
from bt6_net import served_to_torch
from bt6_terms_export import (export_served_terms, forward_served_terms, served_terms_meta, tokens_from_squares)


def live(net, seed=11):
  """Every zero-initialised tensor of the net non-trivial: reader tables, gates, sites, up projections (randomise),
  the shared smolgen prep layer, the decoder's zero-init projections / injects / per-move bias, zero biases, and
  spread norm gains (a wrong gain vector must be visible)."""
  randomise(net, seed=seed)
  g = torch.Generator().manual_seed(seed + 1)
  with torch.no_grad():
    net.smolgenPrepLayer.weight.normal_(0, 0.05, generator=g)
    net.smolgenPrepLayer.bias.normal_(0, 0.05, generator=g)
    for name, p in net.named_parameters():
      zero = float(p.abs().max()) == 0.0
      if name.startswith('move_tokens.') and zero:
        p.normal_(0, 0.5 if name.endswith('mt_pol_bias') else 0.05, generator=g)
      elif name.endswith('.scale') and p.dim() == 1 and float((p - 1).abs().max()) == 0.0:
        p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
      elif zero and p.dim() == 1 and p.numel() > 1 and name.endswith('.bias'):
        p.normal_(0, 0.05, generator=g)
    dead = [n for n, p in net.named_parameters() if p.numel() > 1 and float(p.abs().max()) == 0.0]
  return dead


def run(net, sq):
  net.eval()
  with torch.no_grad():
    out = net(sq, None)
  return out[0].float(), out[1].float(), out[4].float()


def served(net, sq, mutate=None):
  tensors = export_served_terms(net)
  t = served_to_torch(tensors)
  if mutate is not None:
    mutate(t)
  meta = served_terms_meta(net)
  tokens = tokens_from_squares(net, sq)
  out = forward_served_terms(t, sq, meta, tokens)
  return out, tensors, tokens


def check_variant(tag, net_over, opt_over, triplet=True, edge_input=False):
  net = build(net_over, opt_over)
  dead = live(net)
  assert not dead, f'{tag}: tensors still zero after live(): {dead}'
  assert bool(net.bt6h.layer_config.triplet) == triplet and (net.move_tokens.w_edge is not None) == edge_input
  sq = records(12)
  policy, value, value2 = run(net, sq)
  out, tensors, tokens = served(net, sq)
  assert any('/triplet/' in name for name in tensors) == triplet, f'{tag}: triplet tensors in the export'
  assert ('/mt/w_edge/w' in tensors) == edge_input, f'{tag}: /mt/w_edge/w in the export'
  present = policy > -15.0
  d_pol = float((out['policy'] - policy).abs().max())
  d_val = float((out['value'] - value).abs().max())
  d_val2 = float((out['value2'] - value2).abs().max())
  # the served tensors are FP16: agreement at FP16 weight rounding, not at float32 noise
  assert d_pol < 3e-2 and d_val < 1e-2 and d_val2 < 1e-2, (tag, d_pol, d_val, d_val2)
  assert bool(((out['policy'] > -15.0) == present).all()), f'{tag}: present-move pattern differs'
  top = (out['policy'].argmax(-1) == policy.argmax(-1)).float().mean()
  # (e) tokens
  mt = net.move_tokens
  cand, _ = mt.candidates(sq[:, :, 0:13].float())
  n_cand = cand.sum(dim=1).long()
  n_valid = (tokens[..., 2] & 1).sum(dim=1).long()
  assert torch.equal(torch.minimum(n_cand, torch.full_like(n_cand, mt.M)), n_valid)
  # (c) controls
  def drop_tables(t):
    for name in t:
      if '/mha/egt/' in name and not name.endswith('qk_scale'):
        t[name] = torch.zeros_like(t[name])
  def drop_smolgen(t):
    t['/smolgen/prep/w'] = torch.zeros_like(t['/smolgen/prep/w'])
    t['/smolgen/prep/b'] = torch.zeros_like(t['/smolgen/prep/b'])
  def drop_inject(t):
    t['/mt/v_inject/w'] = torch.zeros_like(t['/mt/v_inject/w'])
  def drop_sites(t):
    for name in t:
      if '/edge_site/readback' in name:
        t[name] = torch.zeros_like(t[name])
  def drop_edge_input(t):
    t['/mt/w_edge/w'] = torch.zeros_like(t['/mt/w_edge/w'])
  controls = {}
  for name, mutate in (('tables', drop_tables), ('smolgen', drop_smolgen), ('inject', drop_inject), ('sites', drop_sites),
                       *((('edge input', drop_edge_input),) if edge_input else ())):
    c, _, _ = served(net, sq, mutate)
    controls[name] = max(float((c['value'] - value).abs().max()), float((c['policy'] - policy)[present].abs().max()))
    assert controls[name] > 5 * max(d_pol, d_val), (tag, name, controls[name], d_pol, d_val)
  # (d) carrier round trip
  from bt6_carrier import read_carrier, write_carrier
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'c.pb.gz')
    write_carrier(path, tensors)
    back, _ = read_carrier(path)
  assert set(back) == set(tensors)
  for name, array in tensors.items():
    assert back[name].dtype == array.dtype and back[name].shape == array.shape and back[name].tobytes() == array.tobytes(), name
  f32 = sum(1 for a in tensors.values() if a.dtype == np.float32)
  print(f'{tag}: {len(tensors)} tensors ({f32} FP32 tables); served forward vs the net: policy max {d_pol:.2e} '
        f'(top-1 equal {top:.2f}), value {d_val:.2e}, value2 {d_val2:.2e}; controls (output shift when dropped): '
        + ', '.join(f'{k} {v:.2e}' for k, v in controls.items()) + '; carrier round trip identical')
  return net


def test_refused():
  net = build({'BT6TermsInnerLayers': [0]})          # the original hybrid: BT6 policy head, no decoder
  try:
    export_served_terms(net)
  except NotImplementedError:
    pass
  else:
    raise AssertionError('export_served_terms accepted the original hybrid')
  print('refusal of the original hybrid: OK')


if __name__ == '__main__':
  check_variant('(a) variant bt', NET_DEC, OPT_DEC)
  check_variant('(b) variant nt', {**NATIVE, **DECODER}, OPT_DEC)
  check_variant('(f) variant nt, no triplet', {**NATIVE, **DECODER, 'BT6Triplet': False}, OPT_DEC, triplet=False)
  check_variant('(g) variant nt, no triplet, edge input',
                {**NATIVE, **DECODER, 'BT6Triplet': False, 'MoveTokenEdgeInput': True}, OPT_DEC,
                triplet=False, edge_input=True)
  test_refused()
  print('test_bt6_terms_export: OK')
