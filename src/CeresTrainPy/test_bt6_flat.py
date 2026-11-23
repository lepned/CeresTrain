"""BT6 term reads + edge sites on a FLAT trunk (plain encoder, NBTInnerLayers 0; config BT6TermsLayers). Small nets, CPU.

  (a) identity at initialisation: the flat net with the BT6 parts == the same flat net without them (shared
      parameters copied), with QK-norm on and off, AttnTemplates live
  (b) live reader tables / sites change the function; the loss path is finite and every trainable parameter gets a
      gradient (reader tensors, sites, prologue); readers are no-decay and out of Muon by the name rules
  (c) the term layers read the layer's Q / K after QK-norm; QK-clip stays inert on QK-norm layers (Pq / Pk untouched)
  (d) the move-token decoder with MoveTokenEdgeInput reads the edge state of the last site
  (e) config refusals (no silent no-ops)
  (f) ONNX export (CERES_ONNX_OPSET, default 18 as save_model) == PyTorch in onnxruntime, TensorRT gate fix applies
  (g) torch.compile forward + backward == eager

  python test_bt6_flat.py
"""
import os
import tempfile

import torch

import test_bt6_hybrid as _h
from test_bt6_hybrid import batch_for, build, make_config, randomise, records, run_loss

# the helpers merge NET_SMALL (an NBT net) under every override: give them a flat base instead (no NBT keys)
_h.NET_SMALL = {k: v for k, v in _h.NET_SMALL.items() if not k.startswith(('NBT', 'BT6'))}
_h.OPT_BASE = {**_h.OPT_BASE, 'SoftPolicyWeight': 0}

FLAT = {'ModelDim': 64, 'NumLayers': 4, 'NumHeads': 4, 'PreNorm': True, 'NBTInnerLayers': 0, 'BT6Hybrid': True,
        'BT6PolicyWidth': 64,
        'UseQKNorm': True, 'AttnTemplates': 4, 'AttnTemplateScope': 'layer', 'FFNMultiplier': 2,
        'BT6NativeLayers': True, 'BT6EdgeSites': [0, 2], 'BT6Embedding': 'linear', 'BT6Triplet': False,
        'BT6SiteHidden': 16}
DECODER = {'UseMoveTokens': True, 'MoveTokenDim': 32, 'MoveTokenLayers': 2, 'MoveTokenHeads': 4, 'MoveTokenFFNMult': 2,
           'MoveTokenMax': 96, 'MoveTokenEdgeInput': True}
OPT_DEC = {'SoftPolicyWeight': 0, 'MuonAdamWScope': 'all-non-trunk'}


def base_of(over):
  """The same net without the BT6 parts (every BT6 key removed)."""
  return {k: v for k, v in over.items() if not k.startswith('BT6')}


def live_templates(net, seed=4):
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for n, p in net.named_parameters():
      if n.endswith('attn_templates.table'):
        p.copy_(torch.randn(p.shape, generator=g) * 0.5)


def check_identity(over, opt=None, tag=''):
  # with the move-token decoder (no edge input) both nets have the same heads: the BT6 net differs only in the trunk
  over = {**over, **{k: v for k, v in DECODER.items() if k != 'MoveTokenEdgeInput'}}
  opt = {**OPT_DEC, **(opt or {})}
  net = build(over, opt)
  base = build(base_of(over), opt)
  terms = [n for n, _ in net.named_parameters() if '.attention.bt6t.' in n]
  assert len(terms) == 10 * len(net.bt6_terms_layers), len(terms)
  live_templates(base)
  missing, unexpected = net.load_state_dict(base.state_dict(), strict=False)
  # the plain net also carries the flat policy MLP, unused with the decoder; the hybrid does not build it
  assert all(n.startswith('policy_head.') for n in unexpected), unexpected
  assert all('.bt6t.' in n or n.startswith('bt6h.') for n in missing), [n for n in missing if '.bt6t.' not in n][:5]
  net.eval(); base.eval()
  sq = records(8)
  with torch.no_grad():
    a, b = net(sq, None), base(sq, None)
  d_pol = float((a[0] - b[0]).abs().max())
  d_val = float((a[1] - b[1]).abs().max())
  assert d_pol < 1e-4 and d_val < 1e-5, (d_pol, d_val)
  print(f'(a) identity at init{tag}: policy {d_pol:.1e}, value {d_val:.1e}')
  return net, sq


def test_identity_qknorm_on_off():
  check_identity(FLAT, tag=' (QK-norm on)')
  check_identity({**FLAT, 'UseQKNorm': False}, tag=' (QK-norm off)')
  check_identity({**FLAT, 'BT6TermsLayers': [1, 3]}, tag=' (terms on layers 1, 3)')
  post = {k: v for k, v in FLAT.items() if k != 'AttnTemplateScope'} | {'PreNorm': False, 'AttnTemplates': 0}
  check_identity(post, tag=' (post-norm, no templates)')


def test_compile():
  # torch.compile (as train.py runs it): forward + backward of the flat + BT6 net with the decoder edge input
  net = build({**FLAT, **DECODER}, OPT_DEC)
  randomise(net)
  net.train()
  sq = records(4)
  eager = run_loss(net, batch_for(4), sq)
  cnet = torch.compile(net)
  loss = run_loss(cnet, batch_for(4), sq)
  loss.backward()
  assert torch.isfinite(loss) and abs(float(loss) - float(eager)) < 1e-4, (float(loss), float(eager))
  print(f'(g) torch.compile forward + backward: loss {float(loss):.5f} == eager {float(eager):.5f}')


def test_live_and_gradients():
  net = build(FLAT)
  assert net.bt6_flat and net.bt6_terms_layers == (0, 1, 2, 3) and net.bt6h.site_blocks == (0, 2)
  sq = records(8)
  net.eval()
  with torch.no_grad():
    before = net(sq, None)
  randomise(net)
  live_templates(net)
  with torch.no_grad():
    after = net(sq, None)
  assert float((after[1] - before[1]).abs().max()) > 1e-4
  net.train()
  loss = run_loss(net, batch_for(8), sq)
  assert torch.isfinite(loss)
  loss.backward()
  missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
  assert not missing, missing[:10]
  # the site after layer 2 is read by layer 3: its parameters must get a gradient (checked above) and be non-zero
  from bt6_hybrid import bt6_muon_excluded, is_bt6_layer_reader
  from wd_partition import partition_weight_decay
  decay, no_decay = partition_weight_decay(net)
  readers = [n for n, _ in net.named_parameters() if '.bt6t.' in n]
  assert len(readers) == 40 and all(is_bt6_layer_reader(n) and bt6_muon_excluded(n) for n in readers)
  assert all(n in no_decay and n not in decay for n in readers)
  site = [n for n, _ in net.named_parameters() if n.startswith('bt6h.sites')]
  assert site and all(bt6_muon_excluded(n) for n in site)
  print(f'(b) live tables change the function; loss {float(loss):.4f}; all {sum(1 for _ in net.parameters())} '
        f'parameters have gradients; {len(readers)} readers no-decay / out of Muon')


def test_qknorm_read_and_clip():
  net = build(FLAT)
  randomise(net)
  att = net.transformer_layer[1].attention
  assert att.use_qk_norm and att.bt6t is not None
  seen = {}
  orig = att.bt6t.attend

  def spy(Q, K, *a, **k):
    seen['Q'], seen['K'] = Q.detach().clone(), K.detach().clone()
    return orig(Q, K, *a, **k)
  att.bt6t.attend = spy
  net.eval()
  sq = records(4)
  with torch.no_grad():
    net(sq, None)
  att.bt6t.attend = orig
  # the term read saw qLN(Q), kLN(K): per-row rms == 1 (unit norm gains at init; randomise leaves qLN / kLN alone)
  for t in (seen['Q'], seen['K']):
    rms = t.float().pow(2).mean(-1).sqrt()
    assert float((rms - 1).abs().max()) < 1e-3, float((rms - 1).abs().max())
  # QK-clip: inert on QK-norm layers (Pq / Pk untouched)
  att._last_max_logit = torch.full((att.num_heads,), 1e4)
  pq, pk = att.bt6t.Pq.detach().clone(), att.bt6t.Pk.detach().clone()
  assert att.apply_qk_clip(1.0) == 0
  assert torch.equal(pq, att.bt6t.Pq) and torch.equal(pk, att.bt6t.Pk)
  # without QK-norm the clip scales Pq / Pk with the head's rows (unchanged NBT behaviour)
  net2 = build({**FLAT, 'UseQKNorm': False})
  randomise(net2)
  att2 = net2.transformer_layer[1].attention
  att2._last_max_logit = torch.full((att2.num_heads,), 4.0)
  pq2 = att2.bt6t.Pq.detach().clone()
  assert att2.apply_qk_clip(1.0) == att2.num_heads
  assert torch.allclose(att2.bt6t.Pq, pq2 * 0.5)
  print('(c) term reads see QK-normed Q / K; QK-clip inert with QK-norm, scales Pq / Pk without it')


def test_decoder_edge_input():
  net = build({**FLAT, **DECODER}, OPT_DEC)
  assert net.move_tokens is not None and net.bt6h.policy is None
  sq = records(6)
  net.eval()
  with torch.no_grad():
    a = net(sq, None)
  # the edge state after the last site reaches the decoder: perturbing that site's output changes the policy
  site = net.bt6h.sites['2']
  randomise(net)
  with torch.no_grad():
    b = net(sq, None)
    for p in site.parameters():
      p.add_(torch.randn_like(p) * 0.3)
    c = net(sq, None)
  assert float((c[0] - b[0]).abs().max()) > 1e-5
  net.train()
  loss = run_loss(net, batch_for(6), sq)
  assert torch.isfinite(loss)
  loss.backward()
  missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
  assert not missing, missing[:10]
  print(f'(d) decoder + MoveTokenEdgeInput on the flat trunk: loss {float(loss):.4f}, all parameters have gradients')


def test_refusals():
  bad = [
    ({**FLAT, 'BT6NativeLayers': False}, 'BT6NativeLayers'),
    ({**{k: v for k, v in FLAT.items() if k != 'AttnTemplateScope'}, 'BT6NativeLayers': False, 'AttnTemplates': 0},
     'needs BT6NativeLayers'),
    ({**FLAT, 'BT6TermsInnerLayers': [0]}, 'use BT6TermsLayers'),
    ({**FLAT, 'BT6TermsLayers': [0, 4]}, 'distinct layer indices'),
    ({**FLAT, 'BT6TermsLayers': [1, 1]}, 'distinct layer indices'),
    ({**FLAT, 'BT6EdgeSites': [3]}, 'BT6EdgeSites'),
    ({**FLAT, 'BT6CheckpointBlocks': True}, 'NBT-block levers'),
    ({**FLAT, 'BT6TermsLayers': []}, 'is empty'),
    ({**FLAT, 'BT6TermsLayers': [0, 1, 2]}, 'no term layer after the last site'),
    ({**FLAT, 'BT6ShareReadersPerBlock': True}, 'NBT-block levers'),
    ({**base_of(FLAT), 'BT6TermsLayers': [0]}, 'BT6Hybrid is off'),
  ]
  for over, msg in bad:
    try:
      make_config(over)
    except ValueError as e:
      assert msg in str(e), (msg, str(e))
    else:
      raise AssertionError(f'not refused: {msg}')
  # an NBT trunk refuses BT6TermsLayers
  try:
    make_config({'BT6Hybrid': True, 'BT6NativeLayers': True, 'NumLayers': 3, 'NBTInnerLayers': 3, 'BT6EdgeSites': [0],
                 'BT6TermsLayers': [0], 'BT6Embedding': 'linear', 'PreNorm': False})
  except ValueError as e:
    assert 'flat trunk' in str(e), str(e)
  else:
    raise AssertionError('NBT + BT6TermsLayers not refused')
  # review 11-23: bias options the BT6 loops do not pass, phase FiLM, smolgen message on flat
  bad2 = [
    ({**FLAT, 'UseVisEdgeBias': True}, 'not wired for the BT6 trunk'),
    ({**FLAT, 'UseRayAttentionBias': True}, 'not wired for the BT6 trunk'),
    ({**FLAT, 'SmolgenDimPerSquare': 8, 'SmolgenDim': 64}, 'a flat trunk has no smolgen option'),
  ]
  for over, msg in bad2:
    try:
      make_config(over)
    except ValueError as e:
      assert msg in str(e), (msg, str(e))
    else:
      raise AssertionError(f'not refused: {msg}')
  os.environ['CERES_PHASE_FILM'] = '1'
  try:
    make_config(FLAT)
  except ValueError as e:
    assert 'CERES_PHASE_FILM' in str(e), str(e)
  else:
    raise AssertionError('CERES_PHASE_FILM not refused')
  finally:
    del os.environ['CERES_PHASE_FILM']
  # resume: flat reader names are switch parameters (BT6TermsLayers widened), checked against their init values
  from resume_new_params import is_switch_parameter, check_identity
  cfg = make_config(FLAT)
  assert is_switch_parameter('transformer_layer.3.attention.bt6t.Pq', cfg)
  assert is_switch_parameter('transformer_layer.3.attention.bt6t.t_node', cfg)
  assert not is_switch_parameter('transformer_layer.3.attention.qkv.weight', cfg)
  net = build(FLAT)
  keys = [k for k in net.state_dict() if '.attention.bt6t.' in k]
  counts = check_identity(net, keys)
  assert counts and list(counts.values())[0] == len(keys), counts
  print(f'(e) {len(bad) + len(bad2) + 2} config refusals; flat reader names recognised on resume ({len(keys)} keys)')


def test_onnx_export():
  import onnxruntime as ort
  net = build({**FLAT, **DECODER}, OPT_DEC)
  randomise(net)
  live_templates(net)
  net.eval()
  sq = records(4).float()
  with torch.no_grad():
    ref = net(sq, None)
  d = tempfile.mkdtemp(prefix='bt6flat_')
  path = os.path.join(d, 'flat.onnx')

  class Wrap(torch.nn.Module):
    def __init__(s, m):
      super().__init__(); s.m = m
    def forward(s, x):
      o = s.m(x, None)
      return o[0], o[1]
  torch.onnx.export(Wrap(net), (sq[:2],), path, do_constant_folding=True, export_params=True,
                    opset_version=int(os.environ.get('CERES_ONNX_OPSET', '18')),
                    input_names=['squares'], output_names=['policy', 'value'],
                    dynamic_axes={n: {0: 'batch_size'} for n in ('squares', 'policy', 'value')})
  import copy
  import onnx
  from trt_gate_fix import apply_gate_fix, find_gated_attentions
  m = onnx.load(path)
  k = len(find_gated_attentions(m.graph))
  m2 = copy.deepcopy(m)
  assert k == len(net.bt6_terms_layers) and apply_gate_fix(m2) == k, k    # one gated attention per term layer
  for mm in (m, m2):
    sess = ort.InferenceSession(mm.SerializeToString(), providers=['CPUExecutionProvider'])
    pol, val = sess.run(None, {'squares': sq.numpy()})
    dp = float((torch.from_numpy(pol) - ref[0]).abs().max())
    dv = float((torch.from_numpy(val) - ref[1]).abs().max())
    assert dp < 1e-3 and dv < 1e-4, (dp, dv)
  print(f'(f) {k} gated attentions rewritten by the TensorRT gate fix, same outputs; ONNX opset {os.environ.get("CERES_ONNX_OPSET", "18")} == PyTorch: policy {dp:.1e}, value {dv:.1e}')


if __name__ == '__main__':
  torch.set_num_threads(4)
  test_identity_qknorm_on_off()
  test_live_and_gradients()
  test_qknorm_read_and_clip()
  test_decoder_edge_input()
  test_refusals()
  test_onnx_export()
  test_compile()
  print('ALL OK')
