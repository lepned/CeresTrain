"""TensorRT-11 fix for gated attentions in an exported ONNX graph (2026-11-18).

Why: the BT6 term-read attentions (bt6_native_terms.py, A = softmax(H) * 2 sigmoid(Wg e + bg)) export as
    Softmax(H) -> Mul(gate) -> MatMul(V).
TensorRT 11.3 (strongly typed, the Ceres / Mycel serving builds) fuses that chain into one attention kernel and
computes it WRONG: on prod_1920_12_nbt3d3_smAnt135notripei_t80t91_8B @100M the value W-L error against PyTorch was
0.088 mean / 0.82 max and the puzzle readings fell 60-117 points. TensorRT 10.16 and onnxruntime run the same file
correctly; with the gate removed TensorRT 11.3 matches onnxruntime through the whole net; FP32 Casts around the
Softmax do not help (TensorRT absorbs them). Marking any tensor on the Softmax -> Mul -> MatMul path as an output
also fixes it, i.e. it is that fusion.

The rewrite keeps the math and gives TensorRT the pattern it fuses correctly (bias before the softmax, like smolgen):
    softmax(H) * g @ V  ==  s * (softmax(H + log g) @ V),     s = sum_k softmax(H)_k * g_k   (per query row)
since softmax(H)_k g_k = exp(H_k + log g_k) / sum_j exp(H_j) = softmax(H + log g)_k * s. log g is clamped at -60, so
g = 0 gives weight 0 without -inf in the softmax input. The old Softmax and Mul stay (they compute s); the old
MatMul is replaced. A Softmax inside an fp32 island (fp32_islands.py 'softmax', Cast between Softmax and Mul) is
handled: the new softmax runs in the same dtype as the old one.

After the rewrite on the 100M net: TensorRT 11.3 top-1 64/64, value W-L max error 0.001; puzzles equal to the lc0ex
and onnxruntime readings; serving speed unchanged.

Usage:
  from trt_gate_fix import apply_gate_fix
  n = apply_gate_fix(model)                        # in place, returns the number of rewritten attentions
  python3 trt_gate_fix.py in.onnx out.onnx         # file -> file (an already exported net)
"""

import collections
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

LOG_GATE_FLOOR = -60.0
_NP = {TensorProto.FLOAT16: np.float16, TensorProto.FLOAT: np.float32, TensorProto.DOUBLE: np.float64}


def find_gated_attentions(graph):
  """[(softmax, cast_or_None, mul, matmul, gate_name)] for every Softmax(H) [-> Cast] -> Mul(., gate) -> MatMul(., V)
  where each intermediate has exactly that one consumer and the product is the MatMul's LEFT operand."""
  cons = collections.defaultdict(list)
  for n in graph.node:
    for i in n.input:
      cons[i].append(n)
  found = []
  for sm in graph.node:
    if sm.op_type != 'Softmax':
      continue
    s_out, cast = sm.output[0], None
    c = cons[s_out]
    if len(c) == 1 and c[0].op_type == 'Cast':
      cast = c[0]
      s_out = cast.output[0]
      c = cons[s_out]
    if len(c) != 1 or c[0].op_type != 'Mul':
      continue
    mul = c[0]
    if list(mul.input).count(s_out) != 1:
      continue
    gate = mul.input[1] if mul.input[0] == s_out else mul.input[0]
    c2 = cons[mul.output[0]]
    if len(c2) != 1 or c2[0].op_type != 'MatMul' or c2[0].input[0] != mul.output[0]:
      continue
    found.append((sm, cast, mul, c2[0], gate))
  return found


def apply_gate_fix(model, tag='trtgate'):
  """Rewrite every gated attention of `model` in place (see the module docstring). Returns the count.
  Raises (graph untouched) when a matched softmax does not normalise over the last axis or a dtype is unknown."""
  graph = model.graph
  matches = find_gated_attentions(graph)
  if not matches:
    return 0
  inferred = onnx.shape_inference.infer_shapes(model)   # types / ranks of the intermediates (lookup only)
  vis = list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)
  dtype = {vi.name: vi.type.tensor_type.elem_type for vi in vis}
  rank = {vi.name: len(vi.type.tensor_type.shape.dim) for vi in vis if vi.type.tensor_type.HasField('shape')}
  for init in graph.initializer:
    dtype.setdefault(init.name, init.data_type)
  opset = next((o.version for o in model.opset_import if o.domain in ('', 'ai.onnx')), 0)

  plans = []
  for k, (sm, cast, mul, mm, gate) in enumerate(matches):
    h = sm.input[0]
    ax = [a.i for a in sm.attribute if a.name == 'axis']
    axis = ax[0] if ax else (-1 if opset >= 13 else 1)
    r = rank.get(h)
    if not (axis == -1 or (r is not None and axis == r - 1)):
      raise ValueError(f'trt_gate_fix: Softmax {sm.name!r} normalises over axis {axis} (rank {r}, opset {opset}), '
                       f'not the last one')
    th, tg, tv = dtype.get(h), dtype.get(gate), dtype.get(mul.output[0])
    if th not in _NP or tg not in _NP or tv not in _NP:
      raise ValueError(f'trt_gate_fix: unknown dtype around Softmax {sm.name!r} (H {th}, gate {tg}, product {tv})')
    plans.append((k, sm, mul, mm, gate, h, th, tg, tv))

  replace = {}
  for k, sm, mul, mm, gate, h, th, tg, tv in plans:
    p = f'{tag}{k}_'
    floor = p + 'floor'
    axes = p + 'axes'
    graph.initializer.append(numpy_helper.from_array(np.array(LOG_GATE_FLOOR, dtype=_NP[tg]), floor))
    graph.initializer.append(numpy_helper.from_array(np.array([-1], dtype=np.int64), axes))
    nodes = [helper.make_node('Log', [gate], [p + 'logg'], name=p + 'log'),
             helper.make_node('Max', [p + 'logg', floor], [p + 'loggc'], name=p + 'floor_max')]
    logg = p + 'loggc'
    if tg != th:   # gate and H in different dtypes (softmax fp32 island): bring log g to H's dtype
      nodes.append(helper.make_node('Cast', [logg], [p + 'loggh'], name=p + 'cast_logg', to=th))
      logg = p + 'loggh'
    nodes += [helper.make_node('Add', [h, logg], [p + 'h2'], name=p + 'add'),
              helper.make_node('Softmax', [p + 'h2'], [p + 'p2'], name=p + 'softmax', axis=-1)]
    p2 = p + 'p2'
    if th != tv:   # back to the dtype the old product had (the old island's exit Cast)
      nodes.append(helper.make_node('Cast', [p2], [p + 'p2v'], name=p + 'cast_p2', to=tv))
      p2 = p + 'p2v'
    nodes += [helper.make_node('MatMul', [p2, mm.input[1]], [p + 'o2'], name=p + 'matmul'),
              helper.make_node('ReduceSum', [mul.output[0], axes], [p + 's'], name=p + 'rowsum', keepdims=1),
              helper.make_node('Mul', [p + 'o2', p + 's'], [mm.output[0]], name=p + 'scale')]
    replace[mm.output[0]] = nodes   # keyed by output name (protobuf wrappers have no stable identity)

  out = []
  for n in graph.node:
    out.extend(replace[n.output[0]] if n.op_type == 'MatMul' and n.output and n.output[0] in replace else [n])   # the new subgraph takes the old MatMul's place: H, gate, V and the
  del graph.node[:]                       # product are all produced before it, so the order stays topological
  graph.node.extend(out)
  return len(plans)


def main():
  if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(2)
  m = onnx.load(sys.argv[1])
  n = apply_gate_fix(m)
  onnx.checker.check_model(m)
  onnx.save(m, sys.argv[2])
  print(f'trt_gate_fix: {n} gated attention(s) rewritten: {sys.argv[1]} -> {sys.argv[2]}')


if __name__ == '__main__':
  main()
