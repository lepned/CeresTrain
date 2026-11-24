"""Horizontal GEMM merges on a BT6 TensorRT-plugin ONNX (FP16, or INT8 after bt6_trt_plugin_port_qdq.py).

    python bt6_trt_plugin_merge.py <in.onnx> <out.onnx> [--coef] [--ffn]

--coef  per term layer: the template mixer GEMM (x @ Wm + b, tmix.mix) joins the layer's qkv GEMM (x @ Wq, same x):
        one GEMM x @ [Wq | Wm] -> [B, 64, 2304 + 192], read by BT6TermsAttention v2 with fused_coef = 1 (input 1 = the
        mixer bias). With an INT8 qkv GEMM (weight = DequantizeLinear of INT8, per output channel), Wm is quantized
        per output channel (symmetric, max |w| / 127) and the qkv output fake-quant is dropped (the plugin reads the
        GEMM's FP16 output).
--ffn   every pair of GEMMs on the same input with weights of the same kind (initializers, or DequantizeLinear of
        INT8 initializers along axis 1) and the FFN shape (768 -> 1536 twice, SwiGLU gate / up): one GEMM on the
        concatenated weight, then Split back into the two original output tensors.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("onnx_in"); ap.add_argument("onnx_out")
  ap.add_argument("--coef", action="store_true"); ap.add_argument("--ffn", action="store_true")
  a = ap.parse_args(argv)
  m = onnx.load(a.onnx_in)
  g = m.graph
  inits = {t.name: t for t in g.initializer}
  producer = {o: n for n in g.node for o in n.output}
  consumers = defaultdict(list)
  for n in g.node:
    for i in n.input:
      consumers[i].append(n)
  arr = lambda name: numpy_helper.to_array(inits[name])
  new_inits, new_nodes, dead = [], [], set()

  def add_init(array, name):
    t = numpy_helper.from_array(array, name)
    new_inits.append(t); inits[name] = t
    return name

  def weight_kind(mm):
    """('init', W) or ('q8', (q, scale, zp, dq node)) for a MatMul's weight input, else None."""
    w = mm.input[1]
    if w in inits:
      return "init", arr(w)
    dq = producer.get(w)
    if dq is not None and dq.op_type == "Transpose" and dq.input[0] in inits and len(arr(dq.input[0]).shape) == 2:
      return "init", arr(dq.input[0]).T.copy()        # an exported weight transpose that was not folded
    if dq is not None and dq.op_type == "DequantizeLinear" and dq.input[0] in inits:
      axis = [x.i for x in dq.attribute if x.name == "axis"]
      if axis == [1]:
        return "q8", (arr(dq.input[0]), arr(dq.input[1]), arr(dq.input[2]), dq)
    return None

  merged_coef, merged_ffn = 0, 0
  if a.coef:
    for n in [x for x in g.node if x.op_type == "BT6TermsAttention"]:
      fused = [x for x in n.attribute if x.name == "fused_coef"]
      if fused and fused[0].i == 1:
        continue
      # the qkv GEMM (possibly behind an output Q -> DQ) and the mixer Add(MatMul, bias)
      qkv_in = n.input[0]
      p = producer[qkv_in]
      drop_out_qdq = None
      if p.op_type == "DequantizeLinear":
        q = producer[p.input[0]]
        assert q.op_type == "QuantizeLinear"
        drop_out_qdq = (q, p)
        p = producer[q.input[0]]
      assert p.op_type == "MatMul", (n.name, p.op_type)
      mq = p
      add = producer[n.input[1]]
      assert add.op_type == "Add" and add.input[1] in inits, (n.name, add.op_type)
      mm = producer[add.input[0]]
      assert mm.op_type == "MatMul" and mm.input[1] in inits, (n.name, mm.op_type)
      kq = weight_kind(mq)
      wm = arr(mm.input[1]).astype(np.float32)
      # same activation: x, or Q -> DQ of x
      xq, xm = mq.input[0], mm.input[0]
      if xq != xm:
        dq = producer.get(xq); q = producer.get(dq.input[0]) if dq is not None else None
        assert dq is not None and dq.op_type == "DequantizeLinear" and q.op_type == "QuantizeLinear" and q.input[0] == xm, n.name
      L = n.name.rsplit("_", 1)[1]
      out = f"bt6m_qkvc_{L}"
      if kq[0] == "init":
        W = np.concatenate((kq[1], wm.astype(kq[1].dtype)), axis=1)
        new_nodes.append(helper.make_node("MatMul", [xq, add_init(W, f"bt6m_wqkvc_{L}")], [out], name=f"bt6m_qkvc_mm_{L}"))
      else:
        qw, sc, zp, dqn = kq[1]
        assert not np.any(zp), "symmetric weights expected"
        sm = np.maximum(np.abs(wm).max(axis=0), 1e-8) / 127.0
        qm = np.clip(np.round(wm / sm), -127, 127).astype(np.int8)
        Q = np.concatenate((qw, qm), axis=1)
        S = np.concatenate((sc.astype(np.float32), sm)).astype(sc.dtype)
        Z = np.zeros(Q.shape[1], zp.dtype)
        wdq = f"bt6m_wqkvc_{L}__dq"
        node = helper.make_node("DequantizeLinear", [add_init(Q, f"bt6m_wqkvc_{L}_q"), add_init(S, f"bt6m_wqkvc_{L}_s"),
                                                     add_init(Z, f"bt6m_wqkvc_{L}_z")], [wdq], name=f"bt6m_wqkvc_dq_{L}")
        node.attribute.extend(dqn.attribute)
        new_nodes.append(node)
        new_nodes.append(helper.make_node("MatMul", [xq, wdq], [out], name=f"bt6m_qkvc_mm_{L}"))
      n.input[0] = out
      n.input[1] = add.input[1]
      for x in list(n.attribute):
        if x.name == "fused_coef":
          n.attribute.remove(x)
      n.attribute.append(helper.make_attribute("fused_coef", 1))
      merged_coef += 1

  if a.ffn:
    by_input = defaultdict(list)
    for n in g.node:
      if n.op_type == "MatMul" and weight_kind(n) is not None:
        by_input[n.input[0]].append(n)
    for x, mms in by_input.items():
      if len(mms) != 2:
        continue
      k0, k1 = weight_kind(mms[0]), weight_kind(mms[1])
      if k0[0] != k1[0]:
        continue
      sh = (k0[1].shape if k0[0] == "init" else k0[1][0].shape)
      sh1 = (k1[1].shape if k1[0] == "init" else k1[1][0].shape)
      if not (sh == sh1 == (768, 1536)):
        continue
      tag = mms[0].name
      out = f"{tag}__ffn2"
      if k0[0] == "init":
        W = np.concatenate((k0[1], k1[1]), axis=1)
        new_nodes.append(helper.make_node("MatMul", [x, add_init(W, f"{tag}__ffn2_w")], [out], name=f"{tag}__FFN2"))
      else:
        Q = np.concatenate((k0[1][0], k1[1][0]), axis=1)
        S = np.concatenate((k0[1][1], k1[1][1]))
        Z = np.concatenate((k0[1][2], k1[1][2]))
        wdq = f"{tag}__ffn2_wdq"
        node = helper.make_node("DequantizeLinear", [add_init(Q, f"{tag}__ffn2_q"), add_init(S, f"{tag}__ffn2_s"),
                                                     add_init(Z, f"{tag}__ffn2_z")], [wdq], name=f"{tag}__FFN2_DQ")
        node.attribute.extend(k0[1][3].attribute)
        new_nodes.append(node)
        new_nodes.append(helper.make_node("MatMul", [x, wdq], [out], name=f"{tag}__FFN2"))
      split = add_init(np.array([1536, 1536], np.int64), f"{tag}__ffn2_split")
      new_nodes.append(helper.make_node("Split", [out, split], [mms[0].output[0], mms[1].output[0]], axis=-1,
                                        name=f"{tag}__FFN2_split"))
      dead.update({mms[0].name, mms[1].name})
      merged_ffn += 1

  nodes = [n for n in g.node if n.name not in dead] + new_nodes
  produced = {o: n for n in nodes for o in n.output}
  avail = {i.name for i in g.input} | set(inits) | {""}
  live, work = set(), [o.name for o in g.output]
  while work:
    t = work.pop()
    if t in live:
      continue
    live.add(t)
    if t in produced:
      work.extend(produced[t].input)
  keep = [n for n in nodes if any(o in live for o in n.output)]
  ordered, placed = [], set()
  for n0 in keep:
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
  used = {i for n in ordered for i in n.input}
  allinit = [t for t in list(g.initializer) + new_inits if t.name in used]
  del g.initializer[:]
  g.initializer.extend(allinit)
  del g.value_info[:]
  onnx.save(m, a.onnx_out)
  print(f"coef merges {merged_coef}, ffn merges {merged_ffn}: {len(ordered)} nodes -> {a.onnx_out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
