# FP32 islands (fp32_islands.py): structural RMSNorm-chain / Softmax detection and Cast insertion.
#   python3 test_fp32_islands.py              unit tests on a small exported pre-norm NBT net
#   python3 test_fp32_islands.py <fp16.onnx>  + report on an existing exported net
import os
import sys
import tempfile

import numpy as np
import onnx
import torch

from encoder_layer import EncoderLayer
from fp32_islands import apply_fp32_islands, find_rmsnorm_chains, find_softmax_nodes, inspect_precision
from nbt_layer import NestedBottleneckLayer
from rms_norm import make_norm


class _Net(torch.nn.Module):
  """2 NBT blocks (pre-norm inner layers, RMSNorm projections) on a 128-wide stream + trunk-end norm."""
  def __init__(self, big_residual_scale):
    super().__init__()
    self.scale = big_residual_scale
    mk = lambda j: EncoderLayer('T', 64, 64, 4, 64, 128, True, 0, False, 2, ffn_activation_type='SwiGLU',
                                norm_type='RMSNorm', layerNum=j, pre_norm=True)
    self.blocks = torch.nn.ModuleList([NestedBottleneckLayer(128, 64, mk, 2, 'RMSNorm', 1e-6, proj_activation='Swish')
                                       for _ in range(2)])
    for b in self.blocks:
      torch.nn.init.normal_(b.up.weight, 0, 0.05)
    self.end = make_norm('RMSNorm', 128, eps=1e-6)
    self.head = torch.nn.Linear(128, 3)

  def forward(self, x):
    x = x * self.scale                      # a large residual stream: x*x overflows fp16 past |x| > 256
    for b in self.blocks:
      x = b(x)
    return self.head(self.end(x)).mean(dim=1)


def _export(net, x, path):
  net.eval()
  torch.onnx.export(net, (x,), path, opset_version=18, do_constant_folding=True, export_params=True,
                    input_names=['x'], output_names=['y'])


def _to_fp16(src, dst):
  from onnxconverter_common.float16 import convert_float_to_float16
  m = onnx.load(src)
  m16 = convert_float_to_float16(m, keep_io_types=False, min_positive_val=1e-10, max_finite_val=65504.0)
  onnx.save(m16, dst)


def _providers():
  # ORT's CPU EP executes most fp16 ops by upcasting to fp32, so an fp16-overflow test needs a GPU EP.
  import onnxruntime as ort
  if 'CUDAExecutionProvider' in ort.get_available_providers():
    dev = int(os.environ.get('CERES_TEST_GPU', '1'))   # GPU 1 = the free one on this box
    return [('CUDAExecutionProvider', {'device_id': dev}), 'CPUExecutionProvider'], True
  return ['CPUExecutionProvider'], False


_GPU_USED = [False]


def _run(path, x16):
  import onnxruntime as ort
  prov, _ = _providers()
  so = ort.SessionOptions(); so.log_severity_level = 3
  s = ort.InferenceSession(path, so, providers=prov)
  _GPU_USED[0] = s.get_providers()[0] == 'CUDAExecutionProvider'   # the EP may fail to load (missing cuDNN) and fall back
  inp = s.get_inputs()[0]
  feed = x16.astype(np.float16) if 'float16' in inp.type else x16.astype(np.float32)
  return s.run(None, {inp.name: feed})[0].astype(np.float32)


def _torch_fp16_overflow_demo(net, x):
  """The graph-level overflow needs a real fp16 EP; without one, show the same effect in torch on the
  same net: .half() everywhere vs fp16 with the norms kept in fp32 (what the islands do)."""
  dev = 'cuda:%s' % os.environ.get('CERES_TEST_GPU', '1') if torch.cuda.is_available() else 'cpu'
  n = net.to(dev)
  with torch.no_grad():
    ref = n(x.to(dev)).float().cpu()
    out16 = n.half()(x.to(dev).half()).float().cpu()
  n.float()
  return float((out16 - ref).abs().max()) if torch.isfinite(out16).all() else float('inf')


def test_detection_and_overflow_protection():
  torch.manual_seed(0)
  d = tempfile.mkdtemp()
  net = _Net(big_residual_scale=400.0)
  x = torch.randn(2, 64, 128)
  f32, f16, isl, both = (os.path.join(d, n) for n in ('n32.onnx', 'n16.onnx', 'n16_isl.onnx', 'n16_both.onnx'))
  _export(net, x, f32)
  _to_fp16(f32, f16)
  m = onnx.load(f16)
  chains = find_rmsnorm_chains(m.graph)
  # 2 blocks x (norm_in + 2 inner x (ln1 + ln2) + norm_out) + trunk end = 2*6 + 1 = 13 RMSNorms
  assert len(chains) == 13, f'expected 13 RMSNorm chains, found {len(chains)}: {[len(c) for c in chains]}'
  assert all(4 <= len(c) <= 8 for c in chains), [len(c) for c in chains]
  assert len(find_softmax_nodes(m.graph)) == 4
  rep16 = inspect_precision(m)
  assert rep16['norm_chains_fp32'] == 0 and rep16['softmax_fp32'] == 0
  # islands = norms
  st = apply_fp32_islands(m, {'norms'})
  onnx.checker.check_model(m)
  onnx.shape_inference.infer_shapes(m, strict_mode=True)   # upcast constants must keep value_info consistent
  onnx.save(m, isl)
  rep = inspect_precision(m)
  assert rep['norm_chains'] == 13 and rep['norm_chains_fp32'] == 13, rep
  assert rep['attention_patterns'] == 4 and rep['attention_patterns_with_blockers'] == 0, rep
  assert st['inits_upcast'] >= 1, st   # dynamo dedups identical constants (all-ones gains, eps) in an untrained net
  # numerics: plain fp16 overflows in x*x (inf/nan or garbage); the island version tracks fp32.
  ref = _run(f32, x.numpy())
  out16 = _run(f16, x.numpy())
  outi = _run(isl, x.numpy())
  err16 = float(np.nanmax(np.abs(out16 - ref))) if np.isfinite(out16).all() else float('inf')
  erri = float(np.max(np.abs(outi - ref)))
  gpu = _GPU_USED[0]
  if gpu:
    assert not np.isfinite(out16).all() or err16 > 10 * erri, f'fp16 {err16} vs islands {erri} (overflow expected on a real fp16 EP)'
  else:
    # ORT fell back to CPU (fp16 ops upcast internally): demonstrate the overflow in torch instead.
    e16 = _torch_fp16_overflow_demo(net, x)
    assert e16 == float('inf') or e16 > 10 * erri, f'pure-fp16 net should be far off (x*x overflow): err {e16} vs islands {erri}'
    err16 = e16
  assert np.isfinite(outi).all() and erri < 2e-2 * (1 + float(np.abs(ref).max())), f'island output drifts from fp32: {erri}'
  # islands = norms + softmax: softmax now fp32 and flagged as a fusion blocker
  m2 = onnx.load(f16)
  apply_fp32_islands(m2, {'norms', 'softmax'})
  onnx.checker.check_model(m2)
  onnx.shape_inference.infer_shapes(m2, strict_mode=True)
  rep2 = inspect_precision(m2)
  assert rep2['softmax_fp32'] == 4 and rep2['attention_patterns_with_blockers'] == 4, rep2
  onnx.save(m2, both)
  outb = _run(both, x.numpy())
  assert float(np.max(np.abs(outb - ref))) < 2e-2 * (1 + float(np.abs(ref).max()))
  # idempotent: applying again adds nothing
  st2 = apply_fp32_islands(m, {'norms'})
  assert st2['casts_in'] == 0 and st2['casts_out'] == 0, st2
  print(f'  detection+overflow OK ({"GPU" if gpu else "CPU-only, overflow not testable"}): 13 chains, fp16 err {err16:.3g} vs islands {erri:.3g}, casts in/out {st["casts_in"]}/{st["casts_out"]}')


def test_robustness():
  # Unnamed / duplicate node names, a Constant shared by island and non-island consumers, a chain that
  # never applies to x (incomplete -> not protected, reported), and the no-op refusal.
  from onnx import helper
  x = helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT16, [1, 4, 8])
  y = helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT16, [1, 4, 8])
  z = helper.make_tensor_value_info('z', onnx.TensorProto.FLOAT16, [1, 4, 1])
  two = helper.make_node('Constant', [], ['two'], value=helper.make_tensor('t2', onnx.TensorProto.FLOAT16, [], [2.0]))
  eps = helper.make_node('Constant', [], ['eps'], value=helper.make_tensor('te', onnx.TensorProto.FLOAT16, [], [1e-6]))
  ax = helper.make_node('Constant', [], ['ax'], value=helper.make_tensor('ta', onnx.TensorProto.INT64, [1], [-1]))
  nodes = [two, eps, ax,
           helper.make_node('Pow', ['x', 'two'], ['sq']),                       # unnamed
           helper.make_node('ReduceMean', ['sq', 'ax'], ['mean'], keepdims=1, name='dup'),
           helper.make_node('Add', ['mean', 'eps'], ['me'], name='dup'),       # duplicate name
           helper.make_node('Sqrt', ['me'], ['s']),
           helper.make_node('Reciprocal', ['s'], ['r']),
           helper.make_node('Mul', ['x', 'r'], ['nrm']),
           helper.make_node('Mul', ['nrm', 'two'], ['y']),                      # 'two' shared with the island (twin)
           # a second, INCOMPLETE chain: mean of squares used as a feature, never applied to x
           helper.make_node('Pow', ['x', 'two'], ['sq2']),
           helper.make_node('ReduceMean', ['sq2', 'ax'], ['z'], keepdims=1)]
  m = helper.make_model(helper.make_graph(nodes, 'g', [x], [y, z]), opset_imports=[helper.make_opsetid('', 18)])
  m.ir_version = 9
  chains, inc = find_rmsnorm_chains(m.graph, return_incomplete=True)
  assert len(chains) == 1 and len(inc) == 1, (chains, inc)
  st = apply_fp32_islands(m, {'norms'})
  onnx.checker.check_model(m)
  onnx.shape_inference.infer_shapes(m, strict_mode=True)
  names = [n.name for n in m.graph.node]
  assert len(names) == len(set(names)) and all(names), 'all nodes uniquely named'
  rep = inspect_precision(m)
  assert rep['norm_chains'] == 1 and rep['norm_chains_fp32'] == 1 and rep['norm_chains_incomplete'] == 1, rep
  assert st['inits_upcast'] >= 1 and sum(1 for n in m.graph.node if n.op_type == 'Constant') >= 4, 'shared constant gets ONE fp32 twin'
  refused = False
  try:
    apply_fp32_islands(helper.make_model(helper.make_graph([helper.make_node('Relu', ['x'], ['y'])], 'g2', [x], [y])), {'norms'})
  except RuntimeError:
    refused = True
  assert refused, 'norms on a graph without RMSNorm chains must raise, not silently no-op'
  print('  robustness OK: unnamed/duplicate names, shared Constant twin, incomplete chain excluded, no-op refused')


def report(path):
  m = onnx.load(path, load_external_data=False)
  for k, v in inspect_precision(m).items():
    print(f'  {k:34s} {v}')


if __name__ == '__main__':
  test_detection_and_overflow_protection()
  test_robustness()
  if len(sys.argv) >= 2:
    report(sys.argv[1])
  print('test_fp32_islands: OK')
