"""Port the INT8 quantization of a served QDQ ONNX (qdq_export.py + qdq_to_fp16.py, trunk GEMM allowlist) onto a
BT6 TensorRT-plugin ONNX of the SAME net (bt6_trt_plugin_onnx.py + bt6_trt_plugin_onnx_r2.py output).

    python bt6_trt_plugin_port_qdq.py <plugin.onnx> <int8f16isl.onnx> <out.onnx> [--exclude KxN,...] [--no-output-qdq]

--exclude      weight shapes left in FP16 (e.g. 1536x768: the FFN down projection, whose input has outlier channels)
--no-output-qdq  skip the GEMM output fake-quant (the GEMM writes FP16; less error, one quantize step fewer)

The two graphs come from the same torch export, so the GEMM nodes keep their names (node_linear_2, node_MatMul_1476,
...). For every MatMul of the INT8 graph whose weight is a DequantizeLinear of an INT8 initializer, the plugin graph's
MatMul of the same name gets exactly that quantization:
  weight      the INT8 initializer, its scale / zero point and the DequantizeLinear (attributes copied)
  input       QuantizeLinear -> DequantizeLinear with the INT8 graph's activation scales, on whatever tensor feeds the
              MatMul in the plugin graph (for W_h: the BT6TermsAttention output instead of the exported head merge)
  output      the INT8 graph's output QuantizeLinear -> DequantizeLinear, when it has one
Everything else (plugins, FP32 islands, FP16 parts) is the plugin graph's. Quantize nodes on one tensor are shared.
"""

from __future__ import annotations

import sys
from collections import defaultdict

import onnx
from onnx import helper


def main(argv) -> int:
  import argparse
  ap = argparse.ArgumentParser()
  ap.add_argument("plug"); ap.add_argument("qdq"); ap.add_argument("out")
  ap.add_argument("--exclude", default=""); ap.add_argument("--no-output-qdq", action="store_true")
  a = ap.parse_args(argv[1:])
  plug_path, q_path, out_path = a.plug, a.qdq, a.out
  exclude = {tuple(int(v) for v in x.split("x")) for x in a.exclude.split(",") if x}
  m = onnx.load(plug_path)
  q = onnx.load(q_path)
  g, qg = m.graph, q.graph
  qprod = {o: n for n in qg.node for o in n.output}
  qcons = defaultdict(list)
  for n in qg.node:
    for i in n.input:
      qcons[i].append(n)
  qinit = {t.name: t for t in qg.initializer}
  pnodes = {n.name: n for n in g.node}
  have_init = {t.name for t in g.initializer}
  new_inits, new_nodes = [], []

  def take_init(name):
    if name and name not in have_init:
      new_inits.append(qinit[name])
      have_init.add(name)

  def clone(node, inputs, outputs, name):
    n = helper.make_node(node.op_type, inputs, outputs, name=name)
    n.attribute.extend(node.attribute)
    return n

  qdq_cache = {}

  def qdq(tensor, qnode, dqnode, tag):
    """Q -> DQ on `tensor` with the scales of the INT8 graph's (qnode, dqnode); shared per (tensor, scale)."""
    key = (tensor, qnode.input[1])
    if key in qdq_cache:
      return qdq_cache[key]
    for i in list(qnode.input[1:]) + list(dqnode.input[1:]):
      take_init(i)
    qo, do = f"{tensor}__q8_{tag}", f"{tensor}__dq8_{tag}"
    new_nodes.append(clone(qnode, [tensor] + list(qnode.input[1:]), [qo], f"{tensor}__Q8_{tag}"))
    new_nodes.append(clone(dqnode, [qo] + list(dqnode.input[1:]), [do], f"{tensor}__DQ8_{tag}"))
    qdq_cache[key] = do
    return do

  ported, missing = 0, []
  pcons = defaultdict(list)
  for n in g.node:
    for i in n.input:
      pcons[i].append(n)
  for mm in qg.node:
    if mm.op_type != "MatMul":
      continue
    wdq = qprod.get(mm.input[1])
    if wdq is None or wdq.op_type != "DequantizeLinear" or wdq.input[0] not in qinit:
      continue
    if tuple(qinit[wdq.input[0]].dims) in exclude:
      continue
    pm = pnodes.get(mm.name)
    if pm is None or pm.op_type != "MatMul":
      missing.append(mm.name)
      continue
    # weight
    for i in wdq.input:
      take_init(i)
    wout = f"{mm.name}__w8"
    new_nodes.append(clone(wdq, list(wdq.input), [wout], f"{mm.name}__WDQ8"))
    pm.input[1] = wout
    # input activation
    adq = qprod[mm.input[0]]
    aq = qprod[adq.input[0]]
    assert adq.op_type == "DequantizeLinear" and aq.op_type == "QuantizeLinear", (mm.name, adq.op_type, aq.op_type)
    pm.input[0] = qdq(pm.input[0], aq, adq, "in")
    # output
    oq = [] if a.no_output_qdq else [c for c in qcons[mm.output[0]] if c.op_type == "QuantizeLinear"]
    if oq:
      assert len(qcons[mm.output[0]]) == 1, mm.name
      odq = qcons[oq[0].output[0]][0]
      assert odq.op_type == "DequantizeLinear"
      out = pm.output[0]
      users = list(pcons[out])
      dq_out = qdq(out, oq[0], odq, "out")
      for u in users:
        for k, i in enumerate(u.input):
          if i == out:
            u.input[k] = dq_out
      for o in g.output:
        assert o.name != out, "a quantized GEMM output is a graph output"
    ported += 1
  # insert the new nodes before their first consumer (stable topological order)
  all_nodes = list(g.node) + new_nodes
  produced = {o: n for n in all_nodes for o in n.output}
  avail = {i.name for i in g.input} | have_init | {""}
  ordered, placed = [], set()
  for n0 in all_nodes:
    stack = [(n0, False)]
    while stack:
      cur, ready = stack.pop()
      if id(cur) in placed:
        continue
      if ready:
        placed.add(id(cur)); ordered.append(cur); continue
      stack.append((cur, True))
      for i in cur.input:
        if i not in avail and i in produced and id(produced[i]) not in placed:
          stack.append((produced[i], False))
  del g.node[:]
  g.node.extend(ordered)
  g.initializer.extend(new_inits)
  used = {i for n in g.node for i in n.input}
  keep = [t for t in g.initializer if t.name in used]
  del g.initializer[:]
  g.initializer.extend(keep)
  del g.value_info[:]
  onnx.save(m, out_path)
  print(f"{ported} GEMMs quantized ({len(missing)} INT8 GEMMs without a twin: {missing[:8]}), "
        f"{len(new_nodes)} Q/DQ nodes -> {out_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
