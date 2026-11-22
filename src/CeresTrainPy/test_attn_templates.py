"""Per-square attention templates (config AttnTemplates / AttnTemplateScope) and NBTInputSkip. Small nets, CPU.

  (a) the mixer computes  bias[b, h, i, j] = sum_t c[b, i, h, t] * table[t, i, j]
  (b) one NBT block: the template bias reaches every inner attention AND the edge-site readback (want_logits)
  (c) the big run's layout (our layers + BT6 term reads, sites without triplet, decoder with edge input), smolgen
      replaced by templates, both scopes, with NBTInputSkip:
        identity at initialisation against the same weights in a net built without the two options;
        live tables change the function; loss finite; every parameter has a (non-zero) gradient;
        optimiser classes: tables and skip scalars no-decay, mixers decayed, all of them out of Muon by name;
        deepcopy keeps ONE bank; state_dict save / reload identical; the lc0ex carrier export refuses
  (d) ONNX (opset 18, fp32) == PyTorch in onnxruntime, and the TensorRT-11 gate fix still applies
  (e) a plain NBT net (no BT6): templates + input skip through the ordinary trunk loop, identity at init
  (f) config refusals
  (g) a PLAIN encoder trunk (NBTInnerLayers 0, PreNorm): one mixer per encoder layer + per-layer input skip, with and
      without per-layer smolgen

  python test_attn_templates.py
"""
import copy
import math
import os
import tempfile

import numpy as np
import torch

import test_bt6_hybrid as H
from test_bt6_hybrid import batch_for, build, randomise, records, run_loss
from test_bt6_hybrid_decoder import OPT_DEC
from test_bt6_native_terms import DECODER, NATIVE
from attn_templates import AttnTemplateBank, AttnTemplateMixer

BIG = {**NATIVE, **DECODER, 'BT6Triplet': False, 'MoveTokenEdgeInput': True,
       'NBTSharedSmolgen': False, 'SmolgenDimPerSquare': 0, 'SmolgenDim': 0}


def live(net, seed=11):
  """Every zero-initialised tensor of the new options (and the hybrid's reader tables / sites) non-trivial."""
  randomise(net, seed=seed)
  g = torch.Generator().manual_seed(seed + 1)
  with torch.no_grad():
    if getattr(net, 'attn_templates', None) is not None:
      net.attn_templates.table.normal_(0, 0.5, generator=g)
    if getattr(net, 'input_skip_a', None) is not None:
      net.input_skip_a.copy_(1.0 + 0.2 * torch.randn(net.input_skip_a.shape, generator=g))
      net.input_skip_b.copy_(0.3 * torch.randn(net.input_skip_b.shape, generator=g))


def test_mixer_formula():
  torch.manual_seed(3)
  bank = AttnTemplateBank(8)
  with torch.no_grad():
    bank.table.normal_()
  mixer = AttnTemplateMixer(48, 4, bank)
  x = torch.randn(5, 64, 48)
  got = mixer(x)
  c = mixer.mix(x).reshape(5, 64, 4, 8)
  want = torch.einsum('biht,tij->bhij', c, bank.table)
  err = float((got - want).abs().max())
  assert got.shape == (5, 4, 64, 64) and err < 1e-5, err
  assert not any(isinstance(m, AttnTemplateBank) for m in mixer.modules()), 'the bank must not be a submodule of a mixer'
  print(f'(a) mixer == einsum reference: max diff {err:.1e}')


def test_block_bias_and_readback():
  """One plain NBT block with ONE inner layer: the logits an edge site would read back move by exactly the template
  bias, and the block output moves with it (the bias reaches the attention)."""
  from test_nbt_layer import _block
  torch.manual_seed(1)
  blk = _block(num_inner=1, pre_norm=True)
  with torch.no_grad():
    blk.up.weight.normal_(0, 0.05)
  heads, mid, d = blk.inner_heads, blk.mid_dim, blk.model_dim
  x = torch.randn(3, 64, d)
  base_out, base_logits = blk(x, want_logits=True)
  for block_scope in (False, True):
    bank = AttnTemplateBank(8)
    blk.tmix = torch.nn.ModuleList([AttnTemplateMixer(d if block_scope else mid, heads, bank)])
    blk.tmix_block = block_scope
    out0, logits0 = blk(x, want_logits=True)
    assert torch.equal(out0, base_out) and torch.equal(logits0, base_logits), 'zero tables must be an exact identity'
    with torch.no_grad():
      bank.table.normal_(0, 0.5)
    out1, logits1 = blk(x, want_logits=True)
    h0 = blk.norm_in(x)
    h = blk.down(blk._shift(blk.act_in(h0)))
    tb = blk.tmix[0](h0 if block_scope else blk.inner[0].ln1(h))
    # the helper's layers have a soft cap of 100 after the bias (cap * tanh(s / cap)): compare the uncapped scores
    cap = blk.inner[0].attention.softcap_cutoff
    raw = (lambda s: cap * torch.atanh(s / cap)) if cap > 0 else (lambda s: s)
    err = float((raw(logits1) - (raw(base_logits) + tb)).abs().max())
    assert err < 1e-4, err
    assert float((out1 - base_out).abs().max()) > 1e-4
    print(f"(b) one block, scope {'block' if block_scope else 'layer'}: readback logits == plain + template bias "
          f'({err:.1e}); output changes')
  blk.tmix = None


def _reference_without(net, over):
  """The same weights in a net built WITHOUT AttnTemplates / NBTInputSkip (every key of it exists in `net`)."""
  ref = build({k: v for k, v in over.items() if k not in ('AttnTemplates', 'AttnTemplateScope', 'NBTInputSkip')}, OPT_DEC)
  state = net.state_dict()
  ref.load_state_dict({k: state[k] for k in ref.state_dict()}, strict=True)
  return ref


def test_big_layout(scope, skip):
  over = {**BIG, 'AttnTemplates': 8, 'AttnTemplateScope': scope, 'NBTInputSkip': skip}
  net = build(over, OPT_DEC)
  blocks, inner = len(net.transformer_layer), len(net.transformer_layer[0].inner)
  mixers = [m for m in net.modules() if isinstance(m, AttnTemplateMixer)]
  assert len(mixers) == (blocks if scope == 'block' else blocks * inner), len(mixers)
  assert all(m.bank is net.attn_templates for m in mixers)
  assert net.smolgenPrepLayer is None and all(b.shared_smolgen is None for b in net.transformer_layer)
  assert (getattr(net, 'input_skip_a', None) is not None) == skip
  sq = records(8)
  # ---- identity at initialisation (reader tables / sites made live first, tables and skip scalars left at init)
  randomise(net)
  ref = _reference_without(net, over)
  net.eval(); ref.eval()
  with torch.no_grad():
    a, b = net(sq, None), ref(sq, None)
  d_pol, d_val = float((a[0] - b[0]).abs().max()), float((a[1] - b[1]).abs().max())
  assert d_pol < 1e-5 and d_val < 1e-6, (d_pol, d_val)
  # ---- live: the function changes
  live(net)
  with torch.no_grad():
    moved = net(sq, None)
  assert float((moved[1] - a[1]).abs().max()) > 1e-4
  # ---- loss path and gradient participation
  net.train()
  loss = run_loss(net, batch_for(8), sq)
  assert torch.isfinite(loss)
  loss.backward()
  new = [(n, p) for n, p in net.named_parameters() if '.tmix.' in n or n.startswith(('attn_templates', 'input_skip'))]
  missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
  zero = [n for n, p in new if float(p.grad.abs().max()) == 0.0]
  assert not missing and not zero, (missing[:10], zero[:10])
  # ---- optimiser classes (wd_partition.py; train.py's Muon predicates exclude '.tmix.' and every non-2-D tensor)
  from wd_partition import partition_weight_decay
  decay, no_decay = partition_weight_decay(net)
  for n, p in new:
    if n.endswith('.mix.weight'):
      assert n in decay and 'transformer_layer' in n and '.tmix.' in n, n
    else:
      assert n in no_decay, n
      assert n.endswith('.mix.bias') or p.ndim != 2, (n, p.ndim)     # never a Muon matrix
  # ---- deepcopy (EMA / export copies) keeps one bank; save / reload
  twin = copy.deepcopy(net)
  assert all(m.bank is twin.attn_templates for m in twin.modules() if isinstance(m, AttnTemplateMixer))
  assert twin.attn_templates is not net.attn_templates
  fresh = build(over, OPT_DEC)
  fresh.load_state_dict(net.state_dict(), strict=True)
  net.eval(); fresh.eval()
  with torch.no_grad():
    x, y = net(sq, None), fresh(sq, None)
  assert torch.equal(x[0], y[0]) and torch.equal(x[1], y[1])
  # ---- no lc0ex front end yet: the carrier export must refuse, not silently drop the bias
  from bt6_terms_export import export_served_terms
  try:
    export_served_terms(net)
  except NotImplementedError as e:
    assert 'AttnTemplates' in str(e), str(e)
  else:
    raise AssertionError('export_served_terms accepted a net with AttnTemplates')
  n_new = sum(p.numel() for _, p in new)
  print(f"(c) big-run layout, scope {scope}, input skip {skip}: {len(mixers)} mixers, {n_new:,} new params; identity at "
        f'init (policy {d_pol:.1e}, value {d_val:.1e}); loss finite, all gradients live; optimiser classes OK; '
        f'deepcopy keeps one bank; save / reload identical; carrier export refuses')
  return net


def test_onnx(net):
  import onnx
  import onnxruntime as ort
  from trt_gate_fix import apply_gate_fix, find_gated_attentions
  net.eval()

  class SquaresOnly(torch.nn.Module):
    def __init__(self, inner):
      super().__init__(); self.inner = inner
    def forward(self, squares):
      out = self.inner(squares, None)
      return out[0], out[1], out[4]

  sq = records(8)
  with torch.no_grad():
    want = SquaresOnly(net)(sq)
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'net.onnx')
    torch.onnx.export(SquaresOnly(net), (sq[:2],), path, do_constant_folding=True, export_params=True, opset_version=18,
                      input_names=['squares'], output_names=['policy', 'value', 'value2'],
                      dynamic_axes={n: {0: 'batch_size'} for n in ('squares', 'policy', 'value', 'value2')})
    m = onnx.load(path)
  k = len(find_gated_attentions(m.graph))
  m2 = copy.deepcopy(m)
  assert k > 0 and apply_gate_fix(m2) == k
  onnx.checker.check_model(m2)
  so = ort.SessionOptions(); so.log_severity_level = 3
  for tag, mm in (('as exported', m), ('gate-fixed', m2)):
    got = ort.InferenceSession(mm.SerializeToString(), so, providers=['CPUExecutionProvider']).run(
      None, {'squares': sq.numpy().astype(np.float32)})
    for name, ya, yb in zip(('policy', 'value', 'value2'), got, want):
      ya, yb = ya.astype(np.float64), yb.numpy().astype(np.float64)
      if name == 'policy':
        keep = yb > -1e3
        assert (keep == (ya > -1e3)).all()
        ya, yb = ya[keep], yb[keep]
      err = np.abs(ya - yb).max() / (np.abs(yb).max() + 1e-12)
      assert err < 1e-4, (tag, name, err)
  print(f'(d) ONNX opset 18 fp32 == PyTorch in onnxruntime (batch 8 from a batch-2 trace), {k} gated attentions '
        f'rewritten by the TensorRT gate fix with the same outputs')


def test_plain_nbt():
  """No BT6: the ordinary trunk loop. Identity at init against the same weights without the options."""
  saved_net, saved_opt = H.NET_SMALL, H.OPT_BASE
  try:
    H.NET_SMALL = {k: v for k, v in saved_net.items() if not k.startswith('BT6')}
    H.OPT_BASE = {**saved_opt, 'SoftPolicyWeight': 0}
    over = {'NumLayers': 3, 'NBTInnerLayers': 2}
    net = build({**over, 'AttnTemplates': 4, 'NBTInputSkip': True})
    ref = build(over)
    with torch.no_grad():
      for blk in net.transformer_layer:
        blk.up.weight.normal_(0, 0.05)
    state = net.state_dict()
    ref.load_state_dict({k: state[k] for k in ref.state_dict()}, strict=True)
    sq = records(6)
    net.eval(); ref.eval()
    with torch.no_grad():
      a, b = net(sq, None), ref(sq, None)
    d_pol, d_val = float((a[0] - b[0]).abs().max()), float((a[1] - b[1]).abs().max())
    assert d_pol < 1e-5 and d_val < 1e-6, (d_pol, d_val)
    live(net)
    with torch.no_grad():
      moved = net(sq, None)
    assert float((moved[1] - a[1]).abs().max()) > 1e-5
    net.train()
    loss = run_loss(net, batch_for(6), sq)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and float(p.grad.abs().max()) > 0 for n, p in net.named_parameters()
               if '.tmix.' in n or n.startswith(('attn_templates', 'input_skip')))
    from wd_partition import partition_weight_decay
    partition_weight_decay(net)
    print(f'(e) plain NBT net: identity at init (policy {d_pol:.1e}, value {d_val:.1e}), live tables and skip scalars '
          f'change the function, gradients live, partition complete')
  finally:
    H.NET_SMALL, H.OPT_BASE = saved_net, saved_opt


def test_plain_encoder():
  """PLAIN encoder trunk (NBTInnerLayers 0, PreNorm; Fry's own form): one mixer per encoder layer on its pre-attention
  norm, + per-layer input skip; with and without per-layer smolgen. Identity at init against the same weights without
  the options; the bias is exactly the mixer output on the layer's ln1; gradients, optimiser classes, ONNX."""
  import onnxruntime as ort
  from encoder_layer import EncoderLayer
  from wd_partition import partition_weight_decay
  saved_net, saved_opt = H.NET_SMALL, H.OPT_BASE
  try:
    H.NET_SMALL = {**{k: v for k, v in saved_net.items() if not k.startswith(('BT6', 'NBT'))},
                   'NBTInnerLayers': 0, 'PreNorm': True}
    H.OPT_BASE = {**saved_opt, 'SoftPolicyWeight': 0}
    for smol in (False, True):
      over = {'NumLayers': 3, **({'SmolgenDimPerSquare': 8, 'SmolgenDim': 64} if smol else {})}
      full = {**over, 'AttnTemplates': 8, 'NBTInputSkip': True}
      net = build(full)
      layers = list(net.transformer_layer)
      assert all(isinstance(l, EncoderLayer) and isinstance(l.tmix, AttnTemplateMixer) and l.tmix.bank is net.attn_templates
                 for l in layers), 'one mixer per plain encoder layer, all on the root bank'
      ref = build(over)
      state = net.state_dict()
      ref.load_state_dict({k: state[k] for k in ref.state_dict()}, strict=True)
      sq = records(6)
      net.eval(); ref.eval()
      with torch.no_grad():
        a, b = net(sq, None), ref(sq, None)
      d_pol, d_val = float((a[0] - b[0]).abs().max()), float((a[1] - b[1]).abs().max())
      assert d_pol < 1e-5 and d_val < 1e-6, (d_pol, d_val)
      # the bias the layer adds == its mixer on its own ln1 output (hook the attention's incoming bias)
      live(net)
      seen = {}
      def grab(mod, args, kwargs):
        seen['prb'], seen['x'] = kwargs.get('piece_relation_bias'), args[3]
      hk = layers[1].attention.register_forward_pre_hook(grab, with_kwargs=True)
      with torch.no_grad():
        moved = net(sq, None)
      hk.remove()
      with torch.no_grad():
        want = layers[1].tmix(layers[1].ln1(seen['x']))
      assert seen['prb'] is not None and float((seen['prb'] - want).abs().max()) < 1e-6
      assert float((moved[1] - a[1]).abs().max()) > 1e-5
      net.train()
      loss = run_loss(net, batch_for(6), sq)
      assert torch.isfinite(loss)
      loss.backward()
      new = [(n, p) for n, p in net.named_parameters() if '.tmix.' in n or n.startswith(('attn_templates', 'input_skip'))]
      assert len(new) == 2 * len(layers) + 3, [n for n, _ in new]
      zero = [n for n, p in new if p.grad is None or float(p.grad.abs().max()) == 0.0]
      assert not zero, zero
      decay, no_decay = partition_weight_decay(net)
      for n, p in new:
        assert (n in decay) == n.endswith('.mix.weight'), n
      # ONNX (opset 18, fp32) == PyTorch
      net.eval()
      class SquaresOnly(torch.nn.Module):
        def __init__(self, inner):
          super().__init__(); self.inner = inner
        def forward(self, squares):
          out = self.inner(squares, None)
          return out[0], out[1], out[4]
      with torch.no_grad():
        want = SquaresOnly(net)(sq)
      with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'net.onnx')
        torch.onnx.export(SquaresOnly(net), (sq[:2],), path, do_constant_folding=True, export_params=True, opset_version=18,
                          input_names=['squares'], output_names=['policy', 'value', 'value2'],
                          dynamic_axes={n: {0: 'batch_size'} for n in ('squares', 'policy', 'value', 'value2')})
        so = ort.SessionOptions(); so.log_severity_level = 3
        got = ort.InferenceSession(path, so, providers=['CPUExecutionProvider']).run(None, {'squares': sq.numpy().astype(np.float32)})
      for name, ya, yb in zip(('policy', 'value', 'value2'), got, want):
        ya, yb = ya.astype(np.float64), yb.numpy().astype(np.float64)
        if name == 'policy':
          k = yb > -1e3
          assert (k == (ya > -1e3)).all()
          ya, yb = ya[k], yb[k]
        err = np.abs(ya - yb).max() / (np.abs(yb).max() + 1e-12)
        assert err < 1e-4, (name, err)
      print(f'(g) plain encoder trunk{" + per-layer smolgen" if smol else ""}: {len(layers)} mixers, identity at init '
            f'(policy {d_pol:.1e}, value {d_val:.1e}); bias == mixer(ln1 x); gradients live; classes OK; ONNX == PyTorch')
    for bad, what in (({'NumLayers': 2, 'AttnTemplates': 8, 'PreNorm': False}, 'PreNorm'),
                      ({'NumLayers': 2, 'AttnTemplates': 8, 'AttnTemplateScope': 'block'}, "'layer'")):
      try:
        build(bad)
      except ValueError as e:
        assert what in str(e), (what, str(e))
      else:
        raise AssertionError(f'{bad} was not refused')
    print('(g) plain encoder refusals OK (post-norm, block scope)')
  finally:
    H.NET_SMALL, H.OPT_BASE = saved_net, saved_opt


def test_refusals():
  base = {**BIG, 'AttnTemplates': 8}
  for over, what in (({**BIG, 'AttnTemplateScope': 'block'}, 'silent no-op'),
                     ({**base, 'AttnTemplateScope': 'head'}, "'layer' or 'block'"),
                     ({**base, 'NBTInnerPreNorm': False}, 'NBTInnerPreNorm'),
                     ({**base, 'BT6NativeLayers': False, 'FFNMultiplier': 1}, 'BT6'),
                     ({**BIG, 'NBTInputSkip': True, 'BT6NativeLayers': False}, 'BT6')):
    try:
      build(over, OPT_DEC)
    except (ValueError, AssertionError) as e:
      assert what in str(e), (what, str(e))
    else:
      raise AssertionError(f'{ {k: over[k] for k in over if k not in BIG or over[k] != BIG[k]} } was not refused')
  print('(f) config refusals OK')


if __name__ == '__main__':
  test_mixer_formula()
  test_block_bias_and_readback()
  net = None
  for scope in ('layer', 'block'):
    for skip in (False, True):
      net = test_big_layout(scope, skip)
  test_onnx(test_big_layout('layer', True))
  test_plain_nbt()
  test_plain_encoder()
  test_refusals()
  print('test_attn_templates: all OK')
