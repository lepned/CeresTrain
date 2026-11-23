"""lc0ex carrier export of FLAT-trunk nets (bt6_terms_export.export_served_flat; 2026-11-23). Small nets, CPU.

The kernel-form forward on the exported tensors (forward_served_terms, variant "flat") must reproduce the PyTorch net
with every zero-initialised tensor made live:
  (a) flat + QK-norm + AttnTemplates + decoder, no BT6 (the fry12 form), history on
  (b) the same with InputHistoryKeep 0 (folded into the embedding)
  (c) flat + BT6 term reads on every layer + sites + MoveTokenEdgeInput (the fry12bt6 form)
  (d) flat + BT6 term reads on a subset of layers, one site, no QK-norm
  (e) refusals: BT6 + InputHistoryKeep
  plus controls (a part zeroed in the carrier must change the served output) and a carrier round trip.

  python test_bt6_flat_export.py
"""
import torch

import test_bt6_hybrid as _h
from test_bt6_hybrid import build, records
from bt6_net import served_to_torch
from bt6_terms_export import export_served_terms, forward_served_terms, served_terms_meta, tokens_from_squares

_h.NET_SMALL = {k: v for k, v in _h.NET_SMALL.items() if not k.startswith(('NBT', 'BT6'))}
_h.OPT_BASE = {**_h.OPT_BASE, 'SoftPolicyWeight': 0}

DECODER = {'UseMoveTokens': True, 'MoveTokenDim': 32, 'MoveTokenLayers': 2, 'MoveTokenHeads': 4, 'MoveTokenFFNMult': 2,
           'MoveTokenMax': 96, 'MoveTokenPostMove': True}
OPT_DEC = {'SoftPolicyWeight': 0, 'LossMoveTokenValueOrderMultiplier': 0.15, 'MoveTokenValueOrderTopK': 5,
           'MuonAdamWScope': 'all-non-trunk+decoder'}
FRY = {'ModelDim': 64, 'NumLayers': 4, 'NumHeads': 4, 'PreNorm': True, 'NBTInnerLayers': 0, 'UseQKNorm': True,
       'AttnTemplates': 4, 'AttnTemplateScope': 'layer', 'FFNMultiplier': 2, 'SmolgenDimPerSquare': 0, 'SmolgenDim': 0,
       **DECODER}
BT6 = {'BT6Hybrid': True, 'BT6NativeLayers': True, 'BT6EdgeSites': [0, 2], 'BT6Embedding': 'linear',
       'BT6Triplet': False, 'BT6SiteHidden': 64, 'BT6PolicyWidth': 64, 'MoveTokenEdgeInput': True}


def live_flat(net, seed=7):
  """Every zero-initialised tensor non-trivial, every unit gain spread (a wrong gain must be visible)."""
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, p in net.named_parameters():
      if float(p.abs().max()) == 0.0:
        p.copy_(torch.randn(p.shape, generator=g) * (0.5 if name.endswith(('mt_pol_bias', '.table')) else 0.05))
      elif name.endswith('.scale') and p.dim() == 1 and float((p - 1).abs().max()) == 0.0:
        p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
      elif name.endswith('t_node'):
        p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
    dead = [n for n, p in net.named_parameters() if p.numel() > 1 and float(p.abs().max()) == 0.0]
  return dead


def served(net, sq, mutate=None):
  tensors = export_served_terms(net)
  t = served_to_torch(tensors)
  if mutate is not None:
    mutate(t)
  meta = served_terms_meta(net)
  tokens = tokens_from_squares(net, sq)
  return forward_served_terms(t, sq, meta, tokens), tensors


def check(tag, over, controls=()):
  net = build(over, OPT_DEC)
  assert not live_flat(net)
  net.eval()
  sq = records(12)
  with torch.no_grad():
    ret = net(sq, None)
  policy, value, value2 = ret[0].float(), ret[1].float(), ret[4].float()
  out, tensors = served(net, sq)
  live = policy > -15
  dp = float((out['policy'] - policy)[live].abs().max())
  dv = float((out['value'] - value).abs().max())
  dv2 = float((out['value2'] - value2).abs().max()) if 'value2' in out else 0.0
  assert dp < 5e-4 and dv < 5e-4 and dv2 < 5e-4, (tag, dp, dv, dv2)
  shifts = []
  for name, pick in controls:
    def zero(t, pick=pick):
      for k in [k for k in t if pick(k)]:
        t[k] = torch.zeros_like(t[k])
    o2, _ = served(net, sq, zero)
    s = float((o2['value'] - out['value']).abs().max()) + float((o2['policy'] - out['policy'])[live].abs().max())
    assert s > 1e-3, f'{tag}: control {name} did not change the served output'
    shifts.append(f'{name} {s:.1e}')
  print(f'{tag}: {len(tensors)} tensors; served vs net policy {dp:.1e} value {dv:.1e} value2 {dv2:.1e}; controls: '
        + ', '.join(shifts))
  return net, tensors


def test_fry():
  check('(a) fry form (QK-norm, templates, no BT6)', FRY, controls=(
    ('templates', lambda k: k == '/tpl/table'), ('qk_norm gains', lambda k: '/qk_norm/' in k)))


def test_fry_h0():
  net, tensors = check('(b) fry form, InputHistoryKeep 0 (folded)', {**FRY, 'InputHistoryKeep': 0})
  w = tensors['/embed/linear/w']
  assert abs(w[13:112]).max() == 0.0 and abs(w[:13]).max() > 0, 'history channels not folded into board 0'


def test_fry_bt6():
  check('(c) fry form + BT6 all layers, sites [0, 2], edge input', {**FRY, **BT6}, controls=(
    ('relation tables', lambda k: k.endswith(('/egt/query', '/egt/key'))), ('sites', lambda k: '/edge_site/' in k),
    ('edge input', lambda k: k == '/mt/w_edge/w'), ('templates', lambda k: k == '/tpl/table')))


def test_bt6_subset():
  check('(d) BT6 on layers [1, 3], site [0], no QK-norm', {**FRY, **BT6, 'UseQKNorm': False, 'BT6TermsLayers': [1, 3],
                                                           'BT6EdgeSites': [0]})


def test_refusal():
  net = build({**FRY, **BT6, 'BT6TermsLayers': [1, 3], 'BT6EdgeSites': [0], 'InputHistoryKeep': 0}, OPT_DEC)
  try:
    export_served_terms(net)
  except NotImplementedError as e:
    assert 'InputHistoryKeep' in str(e)
  else:
    raise AssertionError('BT6 + InputHistoryKeep not refused')
  print('(e) BT6 + InputHistoryKeep refused')


if __name__ == '__main__':
  test_fry()
  test_fry_h0()
  test_fry_bt6()
  test_bt6_subset()
  test_refusal()
  print('ALL OK')
