"""Round 3: INT8 FFN down with dynamic per-token activation scales, on an INT8 BT6 plugin ONNX whose FFN down is still
FP16 and whose FFN gate / up GEMMs are NOT merged (bt6_trt_plugin_port_qdq.py --exclude 1536x768, then
bt6_trt_plugin_merge.py --coef only: two separate INT8 GEMMs let TensorRT write their outputs as INT8 directly).

    python bt6_trt_plugin_glu_r3.py <in.onnx> <out.onnx> [--graph] [--layers 0-15] [--control ...]

Per FFN layer, the INT8 graph has  gate GEMM -> Q(sg) -> DQ -> + b -> SiLU ;  up GEMM -> Q(su) -> DQ ;  m = SiLU * up ;
FFN down MatMul(m, W FP16). This becomes
  plugin (default)  BT6GluQuant(Q(gate), Q(up), b; gate_scale sg, up_scale su) -> codes INT8, rmax FP16 [B, 64, 1]
  --graph           the same in plain ONNX ops: r = max(ReduceMax(|m|), 1e-3), Q/DQ(m / r, 1/127)  (TensorRT / Myelin
                    runs it as a two-pass kernel)
  then              DQ(codes, 1/127) @ DQ(W INT8, per output channel, symmetric) -> * rmax -> + bias (as before)
--control noscale:  CONTROL (must fail): the GEMM output is not multiplied back by rmax (per-token scale dropped).
--control nobias:   CONTROL (must fail): the plugin gets a zero gate bias.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import onnx
from onnx import helper, numpy_helper

from bt6_trt_plugin_smooth_r3 import parse_layers, toposort_and_prune


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("onnx_in"); ap.add_argument("onnx_out")
  ap.add_argument("--graph", action="store_true"); ap.add_argument("--layers", default="0-15")
  ap.add_argument("--control", default="", choices=["", "noscale", "nobias"])
  a = ap.parse_args(argv)
  layers = parse_layers(a.layers)
  m = onnx.load(a.onnx_in)
  g = m.graph
  inits = {t.name: t for t in g.initializer}
  arr = lambda name: numpy_helper.to_array(inits[name])
  prod = {o: n for n in g.node for o in n.output}
  cons = defaultdict(list)
  for n in g.node:
    for i in n.input:
      cons[i].append(n)
  downs = [n for n in g.node if n.op_type == "MatMul" and n.input[1] in inits and list(inits[n.input[1]].dims) == [1536, 768]]
  assert len(downs) == 16, len(downs)
  new_inits, new_nodes, dead = [], [], set()

  def add_init(array, name):
    t = numpy_helper.from_array(array, name); new_inits.append(t); return name

  def qdq_of(t):
    """t = DQ(Q(x)) -> (Q node, DQ node)."""
    dq = prod[t]; assert dq.op_type == "DequantizeLinear", (t, dq.op_type)
    q = prod[dq.input[0]]; assert q.op_type == "QuantizeLinear", (t, q.op_type)
    sq, sd = float(arr(q.input[1])), float(arr(dq.input[1]))
    assert sq == sd and not arr(q.input[2]).any(), (t, sq, sd)
    return q, dq, sq

  done = 0
  for l, mm in enumerate(downs):
    if l not in layers:
      continue
    L = f"r3g{l}"
    mid = prod[mm.input[0]]
    assert mid.op_type == "Mul", mid.op_type
    k = 1 if prod[mid.input[1]].op_type == "DequantizeLinear" else 0     # which factor is the up DQ output
    upt, silu = mid.input[k], prod[mid.input[1 - k]]
    assert silu.op_type == "Mul" and prod[upt].op_type == "DequantizeLinear", (silu.op_type, prod[upt].op_type)
    gin = [x for x in silu.input if prod[x].op_type == "Add"][0]          # g + b (Sigmoid's input too)
    add = prod[gin]
    gdq_t, bias = (add.input[0], add.input[1]) if add.input[1] in inits else (add.input[1], add.input[0])
    qg, _, sg = qdq_of(gdq_t)
    qu, _, su = qdq_of(upt)
    W = arr(mm.input[1]).astype(np.float32)
    w_scale = (np.maximum(np.abs(W).max(axis=0), 1e-8) / 127.0).astype(np.float16)
    q = np.clip(np.round(W / w_scale.astype(np.float32)), -127, 127).astype(np.int8)
    inv127 = add_init(np.array(1.0 / 127.0, np.float16), f"{L}_inv127")
    zp = add_init(np.array(0, np.int8), f"{L}_zp")
    if a.graph:
      ab, r0 = f"{L}_abs", f"{L}_rmax0"
      new_nodes.append(helper.make_node("Abs", [mm.input[0]], [ab], name=f"{L}_Abs"))
      new_nodes.append(helper.make_node("ReduceMax", [ab, add_init(np.array([-1], np.int64), f"{L}_ax")], [r0], keepdims=1, name=f"{L}_RowMax"))
      r = f"{L}_rmax"
      new_nodes.append(helper.make_node("Max", [r0, add_init(np.array(1e-3, np.float16), f"{L}_eps")], [r], name=f"{L}_Eps"))
      new_nodes.append(helper.make_node("Div", [mm.input[0], r], [f"{L}_xn"], name=f"{L}_RowNorm"))
      new_nodes.append(helper.make_node("QuantizeLinear", [f"{L}_xn", inv127, zp], [f"{L}_codes"], name=f"{L}_Q8"))
    else:
      b_in = bias
      if a.control == "nobias":
        b_in = add_init(np.zeros(1536, np.float16), f"{L}_zero_bias")
      r = f"{L}_rmax"
      node = helper.make_node("BT6GluQuant", [qg.output[0], qu.output[0], b_in], [f"{L}_codes", r], name=f"{L}_GluQuant",
                              domain="trt.plugins")
      node.attribute.extend([helper.make_attribute("plugin_version", "1"), helper.make_attribute("plugin_namespace", ""),
                             helper.make_attribute("gate_scale", float(np.float32(sg))),
                             helper.make_attribute("up_scale", float(np.float32(su)))])
      new_nodes.append(node)
      dead.add(mid.name)                                                    # the SiLU * up chain dies (pruned below)
    new_nodes.append(helper.make_node("DequantizeLinear", [f"{L}_codes", inv127, zp], [f"{L}_dq"], name=f"{L}_DQ8"))
    new_nodes.append(helper.make_node("DequantizeLinear", [add_init(q, f"{L}_wq"), add_init(w_scale, f"{L}_ws"),
                                                           add_init(np.zeros(768, np.int8), f"{L}_wz")],
                                      [f"{L}_w"], name=f"{L}_WDQ8", axis=1))
    y = mm.output[0]
    mm.input[0], mm.input[1] = f"{L}_dq", f"{L}_w"
    if a.control != "noscale":
      mm.output[0] = f"{L}_yn"
      new_nodes.append(helper.make_node("Mul", [f"{L}_yn", r], [y], name=f"{L}_RowScale"))
    done += 1
  keep = [n for n in g.node if n.name not in dead]
  del g.node[:]
  g.node.extend(keep + new_nodes)
  # prune dead nodes (the replaced SiLU / up DQ chains), then order
  produced = {o: n for n in g.node for o in n.output}
  live, work = set(), [o.name for o in g.output]
  while work:
    t = work.pop()
    if t in live:
      continue
    live.add(t)
    if t in produced:
      work.extend(produced[t].input)
  keep = [n for n in g.node if any(o in live for o in n.output)]
  del g.node[:]
  g.node.extend(keep)
  toposort_and_prune(g, new_inits)
  if not any(o.domain == "trt.plugins" for o in m.opset_import):
    m.opset_import.append(helper.make_opsetid("trt.plugins", 1))
  onnx.save(m, a.onnx_out)
  print(f"{done} FFN layers: INT8 FFN down, per-token scales via {'graph ops' if a.graph else 'BT6GluQuant'}"
        f"{' CONTROL ' + a.control if a.control else ''}: {len(g.node)} nodes -> {a.onnx_out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
