"""ValueHLGaussServe (2026-11-23; Fry's value head, variant B): the HL-Gauss head is the served Q, the WDL head the
served draw probability. Small flat nets, CPU.

  (a) served value: softmax(value) is a valid WDL, W - L == E[HL-Gauss Q], D == min(WDL head's D, 1 - |Q|);
      value2 exports the same served WDL
  (b) training: the WDL CE reads the WDL head's own logits; gradients reach the HL-Gauss head and the WDL head
  (c) config refusals
  (d) ONNX export == PyTorch (onnxruntime)
  (e) lc0ex carrier export: the kernel-form forward reproduces the served value (flat and NBT "nt" nets); zeroing the
      HL-Gauss head moves it
  (f) FP16 export with the fp32 'value_tail' island: finite, == PyTorch, with the draw clamp exercised
  (g) the grad-norm probe pass (compute_loss_or_gradnorm twice on one forward) keeps the WDL-head stash for the real pass

  python test_hlg_serve.py
"""
import os
import tempfile

import numpy as np
import torch

import test_bt6_hybrid as _h
from test_bt6_hybrid import batch_for, build, make_config, records, run_loss

_h.NET_SMALL = {k: v for k, v in _h.NET_SMALL.items() if not k.startswith(('NBT', 'BT6'))}
_h.OPT_BASE = {**_h.OPT_BASE, 'SoftPolicyWeight': 0}

from test_bt6_flat_export import FRY, BT6, OPT_DEC, live_flat, served  # noqa: E402

NET = {**FRY, **BT6, 'ValueHLGaussServe': True}
OPT = {**OPT_DEC, 'HLGaussWeight': 0.5, 'HLGaussBuckets': 128, 'LossValueMultiplier': 0.1, 'LossValue2Multiplier': 0}


def sharpen(net):
  """Large |Q| (HL-Gauss logits spread) and a large WDL draw logit, so that the D clamp is exercised."""
  with torch.no_grad():
    net.hlg_head.fcFinal.weight.mul_(60.0)
    net.hlg_head.fcFinal.bias.copy_(torch.linspace(-6.0, 6.0, net.hlg_buckets))
    net.value_head.fcFinal.bias.copy_(torch.tensor([0.0, 3.0, 0.0]))


def test_served_value():
  net = build(NET, OPT)
  assert net.hlg_serve and net.hlg_buckets == 128 and not hasattr(net, 'value2_head')
  assert not live_flat(net)
  sharpen(net)
  net.eval()
  sq = records(12)
  captured = {}
  hook = net.hlg_head.register_forward_hook(lambda m, i, o: captured.__setitem__('hlg', o))
  hook2 = net.value_head.register_forward_hook(lambda m, i, o: captured.__setitem__('wdl', o))
  with torch.no_grad():
    ret = net(sq, None)
  hook.remove(); hook2.remove()
  p = torch.softmax(ret[1].float(), dim=-1)
  assert float((p.sum(-1) - 1).abs().max()) < 1e-5 and float(p.min()) >= 0
  q = (torch.softmax(captured['hlg'].float(), -1) * net.hlg_centers).sum(-1)
  d = torch.minimum(torch.softmax(captured['wdl'].float(), -1)[:, 1], 1 - q.abs())
  assert float((p[:, 0] - p[:, 2] - q).abs().max()) < 1e-5, float((p[:, 0] - p[:, 2] - q).abs().max())
  assert float((p[:, 1] - d).abs().max()) < 1e-5
  clamped = int((torch.softmax(captured['wdl'].float(), -1)[:, 1] > 1 - q.abs()).sum())
  assert clamped > 0, 'the D clamp must be exercised'
  assert ret[4] is ret[1], 'value2 must export the served WDL'
  print(f'(a) served WDL valid; W-L == E[Q] (|Q| up to {float(q.abs().max()):.3f}, D clamped on {clamped}/12), D == min(D_wdl, 1-|Q|); value2 == value')


def test_training_gradients():
  net = build(NET, OPT)
  live_flat(net)
  net.train()
  sq = records(8).float()
  loss = run_loss(net, batch_for(8), sq)
  loss.backward()
  for head in ('hlg_head', 'value_head'):
    g = sum(float(p.grad.abs().sum()) for p in getattr(net, head).parameters() if p.grad is not None)
    assert g > 0, head
  assert getattr(net, '_last_value_raw', None) is None and getattr(net, '_last_hlg_out', None) is None, 'stashes consumed'
  print(f'(b) training loss {float(loss):.4f}; gradients reach hlg_head and value_head; stashes consumed')


def test_refusals():
  bad = [
    (NET, {**OPT, 'HLGaussWeight': 0}, 'needs HLGaussWeight > 0'),
    (NET, {**OPT, 'LossValue2Multiplier': 0.04}, 'needs LossValue2Multiplier 0'),
    (NET, {**OPT, 'LossValueMultiplier': 0}, 'needs LossValueMultiplier > 0'),
    ({**NET, 'MoveTokenExpectedValue': True}, OPT, 'MoveTokenExpectedValue'),
    (NET, {**OPT, 'HLGaussSigmaScale': 0}, 'needs HLGaussSigmaScale > 0'),
    (NET, {**OPT, 'HLGaussBuckets': 96}, 'power of two'),
    (NET, {**OPT, 'KLAnchorValueWeight': 0.1}, 'KLAnchorValueWeight'),
    (NET, {**OPT, 'MirrorConsistencyWeight': 0.1}, 'MirrorConsistencyWeight'),
  ]
  for net_over, opt_over, msg in bad:
    try:
      make_config(net_over, opt_over)
    except ValueError as e:
      assert msg in str(e), (msg, str(e))
    else:
      raise AssertionError(f'not refused: {msg}')
  print(f'(c) {len(bad)} config refusals')


def test_onnx():
  import onnxruntime as ort
  net = build(NET, OPT)
  live_flat(net)
  net.eval()
  sq = records(4).float()
  with torch.no_grad():
    ref = net(sq, None)
  path = os.path.join(tempfile.mkdtemp(prefix='hlg_'), 'hlg.onnx')

  class Wrap(torch.nn.Module):
    def __init__(s, m):
      super().__init__(); s.m = m
    def forward(s, x):
      o = s.m(x, None)
      return o[0], o[1], o[4]
  torch.onnx.export(Wrap(net), (sq[:2],), path, do_constant_folding=True, export_params=True, opset_version=18,
                    input_names=['squares'], output_names=['policy', 'value', 'value2'],
                    dynamic_axes={n: {0: 'batch_size'} for n in ('squares', 'policy', 'value', 'value2')})
  sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
  pol, val, val2 = sess.run(None, {'squares': sq.numpy()})
  dv = float((torch.from_numpy(val) - ref[1]).abs().max())
  dv2 = float((torch.from_numpy(val2) - ref[1]).abs().max())
  assert dv < 1e-4 and dv2 < 1e-4, (dv, dv2)
  print(f'(d) ONNX opset 18 == PyTorch: value {dv:.1e}, value2 {dv2:.1e}')


def test_carrier():
  net = build(NET, OPT)
  live_flat(net)
  sharpen(net)
  net.eval()
  sq = records(12)
  with torch.no_grad():
    ret = net(sq, None)
  out, tensors = served(net, sq)
  assert '/head/hlg/fc/w' in tensors and '/head/hlg/centers' in tensors
  # compared as probabilities: the carrier stores the (sharpened) head weights in FP16
  dv = float((torch.softmax(out['value'].float(), -1) - torch.softmax(ret[1].float(), -1)).abs().max())
  assert dv < 2e-3, dv
  def zero(t):
    t['/head/hlg/out/w'] = torch.zeros_like(t['/head/hlg/out/w'])
  o2, _ = served(net, sq, zero)
  shift = float((o2['value'] - out['value']).abs().max())
  assert shift > 1e-2, shift
  print(f'(e) carrier served WDL == net (probabilities): {dv:.1e}; zeroing the HL-Gauss head moves it by {shift:.3f}')


NBT = {'PreNorm': False, 'NBTWidthDivisor': 2, 'NBTInnerHeads': 4, 'NBTInnerPreNorm': True, 'NBTProjNorm': 'Norm',
       'BT6Hybrid': True, 'BT6PolicyWidth': 64, 'BT6SiteHidden': 32, 'BT6Triplet': False,
       'NumLayers': 3, 'NBTInnerLayers': 3, 'BT6NativeLayers': True, 'BT6EdgeSites': [0, 1], 'BT6TermsInnerLayers': [0],
       'BT6Embedding': 'linear', 'FFNMultiplier': 2, 'NBTSharedSmolgen': True, 'SmolgenDimPerSquare': 8, 'SmolgenDim': 32,
       'SmolgenToHeadDivisor': 2, 'NBTProjActivation': 'Swish',
       'UseMoveTokens': True, 'MoveTokenDim': 32, 'MoveTokenLayers': 2, 'MoveTokenHeads': 4, 'MoveTokenFFNMult': 2,
       'MoveTokenMax': 96, 'MoveTokenPostMove': True, 'ValueHLGaussServe': True}


def test_carrier_nbt():
  from test_bt6_terms_export import live
  net = build(NBT, OPT)
  assert net.hlg_serve and not net.bt6_flat
  assert not live(net)
  sharpen(net)
  net.eval()
  sq = records(12)
  with torch.no_grad():
    ret = net(sq, None)
  out, tensors = served(net, sq)
  assert '/head/hlg/fc/w' in tensors and 'value2' in out
  dv = float((torch.softmax(out['value'].float(), -1) - torch.softmax(ret[1].float(), -1)).abs().max())
  dv2 = float((torch.softmax(out['value2'].float(), -1) - torch.softmax(ret[4].float(), -1)).abs().max())
  assert dv < 2e-3 and dv2 < 2e-3, (dv, dv2)
  print(f'(e) NBT nt carrier served WDL == net: value {dv:.1e}, value2 {dv2:.1e}')


def test_fp16_value_tail():
  import onnx
  import onnxruntime as ort
  from onnxconverter_common.float16 import convert_float_to_float16
  from fp32_islands import apply_fp32_islands
  net = build(NET, OPT)
  live_flat(net)
  sharpen(net)
  with torch.no_grad():   # decisive positions: |Q| near 1, D clamped to ~0 -> the log clamp is hit
    net.hlg_head.fcFinal.bias.copy_(torch.linspace(-40.0, 40.0, net.hlg_buckets))
  net.eval()
  sq = records(8).float()
  with torch.no_grad():
    ref = net(sq, None)
  path = os.path.join(tempfile.mkdtemp(prefix='hlg16_'), 'hlg.onnx')

  class Wrap(torch.nn.Module):
    def __init__(s, m):
      super().__init__(); s.m = m
    def forward(s, x):
      o = s.m(x, None)
      return o[0], o[1], o[4]
  torch.onnx.export(Wrap(net), (sq[:2],), path, do_constant_folding=True, export_params=True, opset_version=18,
                    input_names=['squares'], output_names=['policy', 'value', 'value2'],
                    dynamic_axes={n: {0: 'batch_size'} for n in ('squares', 'policy', 'value', 'value2')})
  from onnxconverter_common.float16 import DEFAULT_OP_BLOCK_LIST
  # the conversion and the Cast reconcile exactly as save_model does them
  m16 = convert_float_to_float16(onnx.load(path), keep_io_types=False, min_positive_val=1e-10, max_finite_val=65504.0,
                                 op_block_list=list(DEFAULT_OP_BLOCK_LIST) + ['ReduceLogSumExp'])
  vt = {vi.name: vi.type.tensor_type.elem_type for vi in list(m16.graph.value_info) + list(m16.graph.output)}
  for n in m16.graph.node:
    if n.op_type == 'Cast' and n.output:
      for a in n.attribute:
        if a.name == 'to' and a.i in (1, 10) and vt.get(n.output[0]) in (1, 10) and a.i != vt[n.output[0]]:
          a.i = vt[n.output[0]]
  st = apply_fp32_islands(m16, ('norms', 'value_tail'))
  assert st['value_tail'] > 0
  sess = ort.InferenceSession(m16.SerializeToString(), providers=['CPUExecutionProvider'])
  _, val, val2 = sess.run(None, {'squares': sq.numpy().astype(np.float16)})
  val = torch.from_numpy(val.astype(np.float32))
  assert torch.isfinite(val).all() and torch.isfinite(torch.from_numpy(val2.astype(np.float32))).all()
  pr = torch.softmax(ref[1].float(), -1)
  dv = float((torch.softmax(val, -1) - pr).abs().max())
  assert float(pr.min()) < 1e-4, 'the probability clamp must be hit (L or W ~ 0 at |Q| -> 1 - 1/buckets)'
  assert dv < 5e-3, dv
  print(f'(f) fp16 + value_tail island ({st["value_tail"]} nodes fp32): finite, min logit {float(val.min()):.1f}, '
        f'WDL == PyTorch {dv:.1e} (min prob {float(pr.min()):.1e}, max |Q| {float((pr[:, 0] - pr[:, 2]).abs().max()):.4f})')


def test_gradnorm_probe_keeps_stash():
  from losses import LossCalculator
  net = build(NET, OPT)
  live_flat(net)
  net.train()
  sq = records(4).float()
  b = batch_for(4)
  outs = net(sq, None)
  (po, vo, mo, uo, v2, qdl, qdu, up, ao, _) = outs[:10]
  lc = LossCalculator(torch.nn.Linear(4, 4))
  args = (lc, b, po, vo, mo, uo, v2, qdl, qdu, up, None, None, None, ao, None, 0, 0, 0, False)
  net.compute_loss_or_gradnorm(*args, gradient_norm_logging_mode=True)
  assert net._last_value_raw is not None, 'the probe pass must not consume the stash'
  net.compute_loss_or_gradnorm(*args, gradient_norm_logging_mode=False)
  assert net._last_value_raw is None
  print('(g) grad-norm probe pass leaves the WDL-head stash for the real pass')


if __name__ == '__main__':
  torch.set_num_threads(4)
  test_served_value()
  test_training_gradients()
  test_refusals()
  test_onnx()
  test_carrier()
  test_carrier_nbt()
  test_fp16_value_tail()
  test_gradnorm_probe_keeps_stash()
  print('ALL OK')
