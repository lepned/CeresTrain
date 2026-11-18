# TensorRT-11 gate fix (trt_gate_fix.py): the rewritten graph computes the same as the original in onnxruntime.
#   python3 test_trt_gate_fix.py               unit tests on small hand-built attention graphs + a BT6 term-read export
#   python3 test_trt_gate_fix.py <fp16.onnx>   + rewrite an existing exported net and compare it in onnxruntime
#                                                (random inputs in the trained range; the trainer's squares input)
import copy
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from trt_gate_fix import apply_gate_fix, find_gated_attentions

B, NH, T, HD = 3, 4, 64, 16


def _graph(dtype, gated=True, island=False, opset=18, softmax_axis=-1):
  """y = (softmax(q k^T / sqrt(hd) + bias) [* 2 sigmoid(z)]) @ v, inputs q k v [B,NH,T,HD], bias z [B,NH,T,T]."""
  t = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
  ins = [helper.make_tensor_value_info(n, t, [B, NH, T, HD]) for n in ('q', 'k', 'v')]
  ins += [helper.make_tensor_value_info(n, t, [B, NH, T, T]) for n in ('bias', 'z')]
  inits = [numpy_helper.from_array(np.array(1 / np.sqrt(HD), dtype=dtype), 'scale'),
           numpy_helper.from_array(np.array(2.0, dtype=dtype), 'two')]
  nodes = [helper.make_node('Transpose', ['k'], ['kt'], perm=[0, 1, 3, 2]),
           helper.make_node('MatMul', ['q', 'kt'], ['qk']),
           helper.make_node('Mul', ['qk', 'scale'], ['qks']),
           helper.make_node('Add', ['qks', 'bias'], ['h'])]
  sm_in = 'h'
  if island:
    nodes.append(helper.make_node('Cast', ['h'], ['h32'], to=TensorProto.FLOAT))
    sm_in = 'h32'
  sm_kw = {} if softmax_axis is None else {'axis': softmax_axis}
  nodes.append(helper.make_node('Softmax', [sm_in], ['p'], **sm_kw))
  p = 'p'
  if island:
    nodes.append(helper.make_node('Cast', ['p'], ['p16'], to=t))
    p = 'p16'
  if gated:
    nodes += [helper.make_node('Sigmoid', ['z'], ['sg']),
              helper.make_node('Mul', ['sg', 'two'], ['g']),
              helper.make_node('Mul', [p, 'g'], ['a'])]
    p = 'a'
  nodes.append(helper.make_node('MatMul', [p, 'v'], ['y']))
  g = helper.make_graph(nodes, 'att', ins, [helper.make_tensor_value_info('y', t, [B, NH, T, HD])], inits)
  m = helper.make_model(g, opset_imports=[helper.make_opsetid('', opset)])
  m.ir_version = 8
  return m


def _run(m, feeds):
  so = ort.SessionOptions()
  so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL   # compare the graphs, not ORT's fusions
  return ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider']).run(None, feeds)


def _feeds(dtype, rng, gate_zero=False):
  f = {n: rng.standard_normal((B, NH, T, HD)).astype(dtype) for n in ('q', 'k', 'v')}
  f['bias'] = (2 * rng.standard_normal((B, NH, T, T))).astype(dtype)
  z = (3 * rng.standard_normal((B, NH, T, T))).astype(dtype)
  if gate_zero:   # sigmoid underflows to exactly 0 for some keys (fp16: z < -17)
    z[:, :, :, ::7] = -30
  f['z'] = z
  return f


def test_equivalence():
  rng = np.random.default_rng(1)
  for dtype, tol in ((np.float32, 2e-5), (np.float16, 2e-2)):
    for island in ((False, True) if dtype == np.float16 else (False,)):
      for gate_zero in (False, True):
        m = _graph(dtype, island=island)
        f = _feeds(dtype, rng, gate_zero)
        ref = _run(m, f)[0].astype(np.float64)
        m2 = copy.deepcopy(m)
        assert apply_gate_fix(m2) == 1
        onnx.checker.check_model(m2)
        onnx.shape_inference.infer_shapes(m2, strict_mode=True)
        assert not find_gated_attentions(m2.graph), 'the rewritten graph must not contain the pattern any more'
        out = _run(m2, f)[0].astype(np.float64)
        assert np.isfinite(out).all()
        err = np.abs(out - ref).max() / (np.abs(ref).max() + 1e-12)
        print(f'  {np.dtype(dtype).name} island {island} gate zeros {gate_zero}: max rel err {err:.2e}')
        assert err < tol, err


def test_untouched_and_refusals():
  m = _graph(np.float16, gated=False)              # plain attention: nothing to do, graph unchanged
  before = m.SerializeToString()
  assert apply_gate_fix(m) == 0 and m.SerializeToString() == before
  m = _graph(np.float32, opset=12, softmax_axis=None)   # opset < 13: default softmax axis 1 -> refuse
  before = m.SerializeToString()
  try:
    apply_gate_fix(m)
    raise AssertionError('expected a refusal for a softmax over axis 1')
  except ValueError:
    pass
  assert m.SerializeToString() == before, 'a refusal must leave the graph untouched'
  m = _graph(np.float32, opset=12, softmax_axis=3)      # explicit last axis at opset 12: fine
  assert apply_gate_fix(m) == 1


def test_bt6_term_export():
  """The real exporter path: a small net with the big run's layout (our layers with BT6 term reads, edge sites
  without the triplet branch, move-token decoder with edge input; every zero table made live) exported by torch
  (opset 18, fp32 and fp16): it has gated attentions, and the rewrite runs the same in onnxruntime."""
  import os, tempfile
  import torch
  from test_bt6_hybrid import build, records
  from test_bt6_hybrid_decoder import OPT_DEC
  from test_bt6_native_terms import DECODER, NATIVE
  from test_bt6_terms_export import live
  net = build({**NATIVE, **DECODER, 'BT6Triplet': False, 'MoveTokenEdgeInput': True}, OPT_DEC)
  assert not live(net)
  net.eval()

  class SquaresOnly(torch.nn.Module):
    def __init__(self, inner):
      super().__init__(); self.inner = inner
    def forward(self, squares):
      out = self.inner(squares, None)
      return out[0], out[1], out[4]

  sq = records(8)
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'net.onnx')
    torch.onnx.export(SquaresOnly(net), (sq[:2],), path, do_constant_folding=True, export_params=True, opset_version=18,
                      input_names=['squares'], output_names=['policy', 'value', 'value2'],
                      dynamic_axes={n: {0: 'batch_size'} for n in ('squares', 'policy', 'value', 'value2')})
    m32 = onnx.load(path)
  from onnxconverter_common.float16 import convert_float_to_float16
  m16 = convert_float_to_float16(copy.deepcopy(m32), keep_io_types=False, min_positive_val=1e-10, max_finite_val=65504.0)
  # the exporter's Cast reconciliation (save_model.py, ONNX_FP16_CAST_RECONCILED): the converter leaves the 'to' of
  # internal fp32 upcasts at FLOAT while the declared output type became FLOAT16
  m16 = onnx.shape_inference.infer_shapes(m16)
  declared = {vi.name: vi.type.tensor_type.elem_type for vi in list(m16.graph.value_info) + list(m16.graph.output)}
  for n in m16.graph.node:
    if n.op_type == 'Cast':
      for att in n.attribute:
        if att.name == 'to' and att.i in (1, 10) and declared.get(n.output[0]) in (1, 10) and att.i != declared[n.output[0]]:
          att.i = declared[n.output[0]]
  for tag, m, x, tol in (('fp32', m32, sq.numpy().astype(np.float32), 1e-4), ('fp16', m16, sq.numpy().astype(np.float16), 2e-2)):
    k = len(find_gated_attentions(m.graph))
    assert k > 0, f'{tag}: no gated attention found in the exported term-read net'
    m2 = copy.deepcopy(m)
    assert apply_gate_fix(m2) == k
    onnx.checker.check_model(m2)
    assert not find_gated_attentions(m2.graph)
    so = ort.SessionOptions(); so.log_severity_level = 3
    run = lambda mm: ort.InferenceSession(mm.SerializeToString(), so, providers=['CPUExecutionProvider']).run(None, {'squares': x})
    a, b = run(m), run(m2)
    for name, ya, yb in zip(('policy', 'value', 'value2'), a, b):
      ya, yb = ya.astype(np.float64), yb.astype(np.float64)
      if name == 'policy':
        keep = ya > -1e3                                # the masked (illegal / absent) moves sit at a large negative floor
        assert (keep == (yb > -1e3)).all()
        ya, yb = ya[keep], yb[keep]
      err = np.abs(ya - yb).max() / (np.abs(ya).max() + 1e-12)
      print(f'  BT6 term-read export {tag}: {k} gated attentions; {name} max rel diff {err:.2e}')
      assert err < tol, (tag, name, err)


def compare_file(path, n=16):
  m = onnx.load(path)
  k = len(find_gated_attentions(m.graph))
  m2 = copy.deepcopy(m)
  assert apply_gate_fix(m2) == k
  inp = m.graph.input[0]
  shape = [n] + [d.dim_value for d in inp.type.tensor_type.shape.dim[1:]]
  rng = np.random.default_rng(2)
  x = (rng.integers(0, 2, size=shape) * rng.random(size=shape)).astype(np.float16)
  so = ort.SessionOptions(); so.log_severity_level = 3
  run = lambda mm: ort.InferenceSession(mm.SerializeToString(), so, providers=['CPUExecutionProvider']).run(None, {inp.name: x})
  a, b = run(m), run(m2)
  print(f'{path}: {k} gated attentions rewritten')
  for o, ya, yb in zip(m.graph.output, a, b):
    ya, yb = ya.astype(np.float64), yb.astype(np.float64)
    print(f'  {o.name:22s} max |diff| {np.abs(ya - yb).max():.3e}  (max |y| {np.abs(ya).max():.3g})')


if __name__ == '__main__':
  test_equivalence()
  test_untouched_and_refusals()
  test_bt6_term_export()
  print('ALL OK')
  if len(sys.argv) > 1:
    compare_file(sys.argv[1])
