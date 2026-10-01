# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

"""FP32 islands in an FP16 ONNX graph: precision control that lives IN THE GRAPH.

Why (2026-10-01, branch trt11-strongly-typed): TensorRT 11 removes weak typing (BuilderFlag::kFP16,
ILayer::setPrecision/setOutputType, kOBEY_PRECISION_CONSTRAINTS). Ceres' TensorRTNative wrapper used
those to keep the RMSNorm chains and the softmaxes of our FP16 nets in FP32 (hard-coded defaults
FP32PostAttentionNorm=1, FP32Softmax=1, FP32AllNorms=3 plus two automatic structural markers; mapping
2026-10-01). In a STRONGLY-TYPED build the precision of every layer is exactly the dtype in the ONNX,
so the same protection must be expressed as Cast(fp32) ... Cast(fp16) pairs around the protected
subgraphs. This module finds those subgraphs structurally (dynamo node names carry no meaning) and
inserts the Casts. Per island this is what the weak-typed engine did: fp16 -> fp32 is exact, the
constants inside (eps, exponent, norm gains) are the fp16 values upcast, fp32 -> fp16 rounds once at
the island exit. The DEFAULT SET differs from the wrapper's: softmax is off (see below).

Islands:
  'norms'   every RMSNorm chain  Pow(x,2)|Mul(x,x) -> ReduceMean -> Add(eps) -> Sqrt -> (Reciprocal|Div)
            -> Mul(x*r) [-> Mul(gain)]  (dynamo opset-18 decomposition; Pow(.,-0.5) accepted), plus fused
            RMSNormalization / LayerNormalization nodes (opset 23). A chain is only protected when it is
            COMPLETE (reaches the Mul/Div that applies the inverse RMS to x); a head-only island would exit
            through mean(x*x)+eps -- the largest value -- in fp16. This is the island that matters for a
            pre-norm / NBT trunk: the residual stream is not normalized between blocks and x*x overflows
            fp16 (65504) once |x| > 256.
  'softmax' every Softmax. OFF by default: it bought nothing measurable (tf3s @25M and 1536 @900M gates identical
            with and without it), and Casts around the Softmax risk breaking the MatMul -> (Add) -> Softmax -> MatMul
            pattern that TensorRT fuses into one MHA kernel (_gemm_mha_v2). Measured on TRT 10.15 / Ada the fusion
            survived (trtexec: 8 fused MHA with or without the island, 589 kernels in every variant, i.e. the Casts
            are absorbed into neighbouring kernels), but that is a tactic-level property that may differ per TRT
            version and GPU; inspect_precision() counts a Cast inside the pattern as a blocker to stay conservative.
Not expressible as Casts: tactic-level accumulation precision; the weak build's freedom to pick fp32
for unmarked layers (a strongly-typed build runs everything else in fp16).

Scope: the FP16 export only. The INT8 QDQ export (scripts/qdq_export.py default path) is fp32-internal
with INT8 GEMM inputs and is already fully typed; measured 2026-10-01 it is both more accurate and faster
than an fp16-internal QDQ graph with islands, so it needs nothing from this module.

Validation 2026-10-01 (TRT 10.15 strongly typed on a 4090, paired rg2600 n6000 puzzle gate, 20 s EPS):
  tf3s 512x10 NBT @25M: weak-typed = none = norms = norms+softmax = 2323/2756/2418; EPS all within noise.
  1536x12 NBT @900M:    weak-typed = none = norms = norms+softmax = 2550/2978/2696; EPS all within noise.
  (TRT 10.15 itself also forces Reduce/Pow of a recognized post-attention norm into fp32 in a strongly
  typed build; the islands make that protection explicit and pattern-independent.)

Usage:
  from fp32_islands import apply_fp32_islands, inspect_precision
  st = apply_fp32_islands(model, islands={'norms'})         # in place, returns counts; raises on a no-op
  python3 fp32_islands.py in.onnx out.onnx --islands norms   # file -> file (checker + strict shape inference)
  python3 fp32_islands.py in.onnx --check                    # report only (islands, attention purity, QDQ)
"""

import argparse
from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F32, F16 = TensorProto.FLOAT, TensorProto.FLOAT16
ELEMENTWISE_IN_NORM = ('Add', 'Sqrt', 'Reciprocal', 'Div', 'Mul', 'Sub')
NORM_OPS = ('RMSNormalization', 'LayerNormalization', 'SimplifiedLayerNormalization', 'SkipSimplifiedLayerNormalization')
ISLAND_KINDS = ('norms', 'softmax')


def _maps(graph):
  prod = {}
  cons = defaultdict(list)
  for n in graph.node:
    for o in n.output:
      prod[o] = n
    for i in n.input:
      if i:
        cons[i].append(n)
  inits = {t.name: t for t in graph.initializer}
  return prod, cons, inits


def _const_value(name, prod, inits):
  """Scalar value of an initializer / Constant node output, else None."""
  if name in inits:
    a = numpy_helper.to_array(inits[name])
    return float(a.reshape(-1)[0]) if a.size == 1 else None
  n = prod.get(name)
  if n is not None and n.op_type == 'Constant':
    for att in n.attribute:
      if att.name == 'value':
        a = numpy_helper.to_array(att.t)
        return float(a.reshape(-1)[0]) if a.size == 1 else None
  return None


def _is_const(name, prod, inits):
  return name in inits or (prod.get(name) is not None and prod[name].op_type == 'Constant')


def _cast_to(node):
  return next((a.i for a in node.attribute if a.name == 'to'), None) if node.op_type == 'Cast' else None


def _ensure_unique_names(graph, tag):
  """Node names are the keys below: give unnamed nodes a name and de-duplicate."""
  seen = set()
  for i, n in enumerate(graph.node):
    base = n.name or f'{tag}_node{i}'
    name = base
    k = 0
    while name in seen:
      k += 1
      name = f'{base}_{k}'
    n.name = name
    seen.add(name)


def find_rmsnorm_chains(graph, return_incomplete=False):
  """Returns a list of node-name lists, one per COMPLETE RMSNorm chain (decomposed or fused).

  A decomposed chain is anchored on ReduceMean(square(x)) and walked forward through the small
  elementwise tail; it is complete when the walk reaches the node that applies the result to the base
  x (Mul(x, r) or Div(x, s)); the optional gain Mul (other input = non-scalar constant) is appended.
  Incomplete chains are returned separately with return_incomplete=True (never protected).
  Node names are the chain keys, so unnamed / duplicate names are fixed first (in place)."""
  _ensure_unique_names(graph, 'fp32isl')
  prod, cons, inits = _maps(graph)
  chains, incomplete = [], []
  seen = set()
  for n in graph.node:
    if n.op_type in NORM_OPS:
      chains.append([n.name])
      continue
    if n.op_type != 'ReduceMean' or not n.input:
      continue
    sq = prod.get(n.input[0])
    if sq is None:
      continue
    is_square = (sq.op_type == 'Pow' and len(sq.input) > 1 and _const_value(sq.input[1], prod, inits) == 2.0) or \
                (sq.op_type == 'Mul' and len(sq.input) == 2 and sq.input[0] == sq.input[1])
    if not is_square or sq.name in seen:
      continue
    base = sq.input[0]
    chain = [sq.name, n.name]
    seen.add(sq.name)
    frontier = [n]
    depth = 0
    applied = None            # the Mul/Div that applies the inverse RMS to x
    while frontier and depth < 7 and applied is None:
      nxt = []
      for cur in frontier:
        for out in cur.output:
          for c in cons.get(out, []):
            if c.name in chain:
              continue
            ok = c.op_type in ELEMENTWISE_IN_NORM or \
                 (c.op_type == 'Pow' and len(c.input) > 1 and _const_value(c.input[1], prod, inits) == -0.5)
            if not ok:
              continue
            chain.append(c.name)
            if c.op_type in ('Mul', 'Div') and base in c.input:
              applied = c
              break
            nxt.append(c)
          if applied is not None:
            break
        if applied is not None:
          break
      frontier = nxt
      depth += 1
    if applied is None:
      incomplete.append(chain)
      continue
    for c in cons.get(applied.output[0], []):      # optional gain
      if c.op_type == 'Mul' and c.name not in chain:
        consts = [i for i in c.input if _is_const(i, prod, inits)]
        if consts and _const_value(consts[0], prod, inits) is None:
          chain.append(c.name)
          break
    chains.append(chain)
  return (chains, incomplete) if return_incomplete else chains


def find_softmax_nodes(graph):
  return [[n.name] for n in graph.node if n.op_type == 'Softmax']


def insert_fp32_islands(model, node_names, tag='fp32isl'):
  """Rewrites `model` in place so that every node named in `node_names` computes in FP32.
  Returns dict(casts_in, casts_out, inits_upcast, nodes)."""
  graph = model.graph
  _ensure_unique_names(graph, tag)
  island = {n.name for n in graph.node if n.name in node_names and n.op_type != 'Cast'}
  if not island:
    return dict(casts_in=0, casts_out=0, inits_upcast=0, nodes=0)
  prod, cons, inits = _maps(graph)
  node_by_name = {n.name: n for n in graph.node}
  graph_inputs = {i.name for i in graph.input}
  graph_outputs = {o.name for o in graph.output}
  vinfo = {vi.name: vi for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}
  vtypes = {k: v.type.tensor_type.elem_type for k, v in vinfo.items()}
  cast_in_cache = {}     # fp16 tensor -> fp32 Cast output (one Cast per tensor)
  twin_cache = {}        # shared fp16 constant -> fp32 twin name
  n_in = n_out = n_init = 0
  pending = []           # (new node, name of the producer it must follow; None = graph start)

  def set_vinfo_f32(name):
    vi = vinfo.get(name)
    if vi is not None and vi.type.tensor_type.elem_type == F16:
      vi.type.tensor_type.elem_type = F32

  def fp32_version(tensor):
    nonlocal n_in
    if tensor not in cast_in_cache:
      out = f'{tensor}_{tag}_f32'
      p = prod.get(tensor)
      pending.append((helper.make_node('Cast', [tensor], [out], to=F32, name=f'{tensor}_{tag}_castf32'),
                      p.name if p is not None else None))
      cast_in_cache[tensor] = out
      n_in += 1
    return cast_in_cache[tensor]

  for name in sorted(island):
    node = node_by_name[name]
    for k, inp in enumerate(node.input):
      if not inp:
        continue
      if inp in inits:
        t = inits[inp]
        if t.data_type != F16:
          continue
        shared = any(c.name not in island for c in cons.get(inp, []))
        if shared:
          twin = twin_cache.get(inp)
          if twin is None:
            twin = f'{inp}_{tag}_f32'
            graph.initializer.append(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), twin))
            inits[twin] = graph.initializer[-1]
            twin_cache[inp] = twin
            n_init += 1
          node.input[k] = twin
        else:
          t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), inp))
          set_vinfo_f32(inp)   # dynamo exports carry a value_info entry per initializer: keep it consistent
          n_init += 1
        continue
      p = prod.get(inp)
      if p is not None and p.op_type == 'Constant' and p.name not in island:
        att = next((a for a in p.attribute if a.name == 'value'), None)
        if att is None or att.t.data_type != F16:
          continue
        arr = numpy_helper.to_array(att.t).astype(np.float32)
        shared = any(c.name not in island for c in cons.get(inp, []))
        if shared:
          twin = twin_cache.get(inp)
          if twin is None:
            twin = f'{inp}_{tag}_f32'
            pending.append((helper.make_node('Constant', [], [twin], name=f'{p.name}_{tag}_f32',
                                             value=numpy_helper.from_array(arr, twin)), None))
            twin_cache[inp] = twin
            n_init += 1
          node.input[k] = twin
        else:
          att.t.CopyFrom(numpy_helper.from_array(arr, att.t.name))
          set_vinfo_f32(inp)
          n_init += 1
        continue
      if p is not None and p.name in island:
        continue   # island-internal edge, stays fp32
      if inp in graph_inputs or p is not None:
        declared = vtypes.get(inp)
        if declared is not None and declared != F16:
          continue   # already fp32, or an int/bool tensor (axes, masks): never cast those
        if p is not None and _cast_to(p) == F32:
          continue   # already an fp32 view (idempotence: a previous islands pass)
        node.input[k] = fp32_version(inp)
  # Outputs leaving the island: rename the island output and cast back to fp16 for outside consumers.
  cast_out_nodes = []
  for name in sorted(island):
    node = node_by_name[name]
    for k, out in enumerate(node.output):
      outside = [c for c in cons.get(out, []) if c.name not in island]
      if not outside and out not in graph_outputs:
        set_vinfo_f32(out)   # island-internal tensor
        continue
      if out not in graph_outputs and all(_cast_to(c) == F16 for c in outside):
        continue   # already exits through an fp16 Cast (idempotence)
      inner = f'{out}_{tag}_f32'
      node.output[k] = inner
      for c in cons.get(out, []):
        if c.name in island:
          for j, ci in enumerate(c.input):
            if ci == out:
              c.input[j] = inner
      cast_out_nodes.append((helper.make_node('Cast', [inner], [out], to=F16, name=f'{out}_{tag}_castf16'), name))
      graph.value_info.append(helper.make_tensor_value_info(inner, F32, None))   # the fp16 name keeps its own entry
      n_out += 1
  # Insert: input Casts right after the producer of the tensor they cast (graph inputs / Constant twins at
  # the graph start), output Casts right after their island node.
  insert_after = defaultdict(list)
  at_start = []
  for cnode, after in pending:
    (insert_after[after] if after is not None else at_start).append(cnode)
  for cnode, after in cast_out_nodes:
    insert_after[after].append(cnode)
  new_list = list(at_start)
  for n in list(graph.node):
    new_list.append(n)
    new_list.extend(insert_after.get(n.name, []))
  del graph.node[:]
  graph.node.extend(new_list)
  return dict(casts_in=n_in, casts_out=n_out, inits_upcast=n_init, nodes=len(island))


def apply_fp32_islands(model, islands=('norms',), tag='fp32isl', require_chains=True):
  """Insert the requested islands. Raises if 'norms' is requested but the graph has no RMSNorm chain
  (e.g. after ORT's quant_pre_process rewrote it): a silent no-op must not look like protection."""
  islands = {str(x).strip().lower() for x in islands} - {'none', ''}
  bad = islands - set(ISLAND_KINDS)
  assert not bad, f'unknown fp32 island kind(s) {sorted(bad)}; known: {ISLAND_KINDS}'
  names = set()
  chains, incomplete = [], []
  if 'norms' in islands:
    chains, incomplete = find_rmsnorm_chains(model.graph, return_incomplete=True)
    for ch in chains:
      names.update(ch)
    if incomplete:
      print(f'[fp32_islands] WARNING: {len(incomplete)} RMSNorm chain(s) end before the x*rsqrt apply and are NOT '
            f'protected (first: {incomplete[0]})', flush=True)
    if require_chains and not chains:
      raise RuntimeError('fp32_islands: norms requested but no RMSNorm chain found in the graph')
  if 'softmax' in islands:
    for ch in find_softmax_nodes(model.graph):
      names.update(ch)
  stats = insert_fp32_islands(model, names, tag)
  stats['chains'] = len(chains)
  stats['chains_incomplete'] = len(incomplete)
  stats['softmax'] = len(find_softmax_nodes(model.graph)) if 'softmax' in islands else 0
  return stats


def inspect_precision(model):
  """Report: RMSNorm chains and how many compute in fp32 (incomplete ones never count); softmaxes and
  their precision; attention patterns MatMul -> ... -> Softmax -> ... -> MatMul containing a Cast, Q/DQ
  or Tanh (softcap) = fusion blockers; QDQ placement (activation*activation MatMuls quantized?)."""
  try:
    model = onnx.shape_inference.infer_shapes(model)
  except Exception:
    pass
  g = model.graph
  prod, cons, inits = _maps(g)
  vtypes = {vi.name: vi.type.tensor_type.elem_type for vi in list(g.value_info) + list(g.output)}

  def out_type(n):
    return vtypes.get(n.output[0], None) if n.output else None

  chains, incomplete = find_rmsnorm_chains(g, return_incomplete=True)
  node_by_name = {n.name: n for n in g.node}
  chains_f32 = sum(1 for ch in chains if all(out_type(node_by_name[x]) == F32 for x in ch if x in node_by_name))
  sms = [n for n in g.node if n.op_type == 'Softmax']
  sm_f32 = sum(1 for n in sms if out_type(n) == F32)
  blockers = patterns = 0
  for sm in sms:
    path = []
    cur = sm
    for _ in range(8):
      p = prod.get(cur.input[0]) if cur.input else None
      if p is None:
        break
      path.append(p)
      if p.op_type == 'MatMul':
        break
      cur = p
    fwd = list(cons.get(sm.output[0], []))
    for c in list(fwd):
      for _ in range(4):
        if c.op_type == 'MatMul':
          break
        nxt = cons.get(c.output[0], []) if c.output else []
        if not nxt:
          break
        c = nxt[0]
        fwd.append(c)
    if any(p.op_type == 'MatMul' for p in path) and any(c.op_type == 'MatMul' for c in fwd):
      patterns += 1
      if any(n.op_type in ('Cast', 'QuantizeLinear', 'DequantizeLinear', 'Tanh') for n in path + fwd + [sm]):
        blockers += 1
  qdq = [n for n in g.node if n.op_type in ('QuantizeLinear', 'DequantizeLinear')]

  def is_weight(name):
    # initializer, or DQ(initializer) / DQ(Q(initializer)) as quantize_static rewrites weights
    if name in inits:
      return True
    p = prod.get(name)
    while p is not None and p.op_type in ('DequantizeLinear', 'QuantizeLinear'):
      if p.input[0] in inits:
        return True
      p = prod.get(p.input[0])
    return False

  act_mm_q = sum(1 for n in g.node if n.op_type == 'MatMul' and not any(is_weight(i) for i in n.input)
                 and any(prod.get(i) is not None and prod[i].op_type == 'DequantizeLinear' for i in n.input))
  return dict(norm_chains=len(chains), norm_chains_fp32=chains_f32, norm_chains_incomplete=len(incomplete),
              softmax=len(sms), softmax_fp32=sm_f32,
              attention_patterns=patterns, attention_patterns_with_blockers=blockers,
              qdq_nodes=len(qdq), act_matmuls_quantized=act_mm_q,
              fp32_initializers=sum(1 for t in g.initializer if t.data_type == F32),
              casts=sum(1 for n in g.node if n.op_type == 'Cast'), nodes=len(g.node))


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('src')
  ap.add_argument('dst', nargs='?')
  ap.add_argument('--islands', default='norms', help="comma list of %s or 'none'" % ','.join(ISLAND_KINDS))
  ap.add_argument('--check', action='store_true', help='report only')
  a = ap.parse_args()
  m = onnx.load(a.src)
  if a.check:
    for k, v in inspect_precision(m).items():
      print(f'  {k:34s} {v}')
    return
  isl = [s.strip() for s in a.islands.split(',') if s.strip()]
  st = apply_fp32_islands(m, isl)
  onnx.checker.check_model(m)
  onnx.shape_inference.infer_shapes(m, strict_mode=True)
  dst = a.dst or a.src.replace('.onnx', '.isl.onnx')
  onnx.save(m, dst)
  print(f'[fp32_islands] {a.src} -> {dst}: {st}')
  for k, v in inspect_precision(m).items():
    print(f'  {k:34s} {v}')


if __name__ == '__main__':
  main()
