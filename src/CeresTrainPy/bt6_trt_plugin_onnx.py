"""Rewrite a served FP16 ONNX of a flat BT6-terms trunk (BT6NativeLayers + BT6TermsLayers on a flat encoder, QK-norm,
AttnTemplates scope 'layer') so that every term layer's attention is ONE custom node, BT6TermsAttention, which the
TensorRT plugin library trt_bt6_plugin (bt6_trt_plugin.dll / libbt6_trt_plugin.so) implements.

    python bt6_trt_plugin_onnx.py <served.onnx> <params.npz> <out.onnx> [--layers 0,1,...]

params.npz: bt6_trt_plugin_params.py (FP32 parameters from the .ts / checkpoint; the folds below are BT6's served
folds, bt6_terms_export.py, computed in float64).

What is replaced, per term layer L (scope inner.transformer_layer.L):
  everything from the qkv GEMM output (attention.qkv) and the template mixer output (tmix.mix) up to the input of the
  out-projection (attention.W_h): the head split, QK-norm, the relation terms (Pq / Pk / C1 against E), the stacked
  edge / pair / door / gate reads, the logit assembly, door, softmax, post-softmax gate, A @ V and the head merge.
  Nodes still needed elsewhere (the q.k and template bias the edge sites read after layers 1, 3, 5) stay; the rest is
  pruned as dead code.
Shared inputs built once (TensorRT ops):
  edges  INT32 [B, 64, 64, 2] = Cast(MatMul(Transpose(Cast(E, f32)), W)), W[c] = 2^c (c < 17) | 2^(c - 17): exact
  norm   S [B, 64, 64] FP16 (the prologue's rsqrt(mean_d Z^2 + eps), as exported)
  state  Transpose(Reshape(e, [B, 16, 64, 64]), [0, 2, 3, 1]) per edge-state version (e0 and after every site)
Per-layer tables (initializers): tab16 FP16 / tab32 FP32, layout in trt_bt6_plugin/src/bt6_attn_kernels.cu.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PLUGIN = "BT6TermsAttention"
DOMAIN = "trt.plugins"
HEADS, HD, TEMPLATES, CH, PAD = 24, 32, 8, 34, 40


def scope(node) -> str:
  for p in node.metadata_props:
    if p.key == "pkg.torch.onnx.name_scopes":
      return p.value
  return ""


def fold_tables(P: dict, layer: int):
  """(tab16 FP16, tab32 FP32) of one term layer: bt6_terms_export.py's folds in float64."""
  g = f"transformer_layer.{layer}.attention"
  f = lambda n: np.asarray(P[n], dtype=np.float64)
  t, te = f(f"{g}.bt6t.t_node"), f(f"{g}.bt6t.t_edge")
  C1, C2 = f(f"{g}.bt6t.C1"), f(f"{g}.bt6t.C2")
  We, Wm, Wg, bg = f(f"{g}.bt6t.We"), f(f"{g}.bt6t.Wm"), f(f"{g}.bt6t.Wg"), f(f"{g}.bt6t.bg")
  Pq, Pk = f(f"{g}.bt6t.Pq"), f(f"{g}.bt6t.Pk")
  D, T = f("bt6h.prologue.D"), f("bt6h.prologue.T")
  relidx = np.asarray(P["bt6h.prologue.relidx"]).astype(np.int64)
  offsets = T.T[:, relidx]                                                   # [P, 64, 64]
  qk_scale = t / math.sqrt(HD)
  attack = t[:, None] * C1
  query = t[:, None, None] * Pq
  key = t[:, None, None] * Pk
  scaled = t[:, None] * (C2 @ D.T)
  cbias = t[:, None, None] * np.einsum("ht,trc->hrc", C2, offsets)
  read = te[:, None] * We
  heads = t.shape[0]
  assert heads == HEADS and Pq.shape == (HEADS, CH, HD), (heads, Pq.shape)
  qpad = np.zeros((HEADS, PAD, HD)); qpad[:, :CH] = query
  kpad = np.zeros((HEADS, PAD, HD)); kpad[:, :CH] = key
  rdg = np.concatenate((read, Wm, Wg), axis=1)                               # [H, 48]
  tab16 = np.concatenate((qpad.ravel(), kpad.ravel(), rdg.ravel(), cbias.ravel())).astype(np.float16)
  qn, kn = f(f"{g}.qLN.scale"), f(f"{g}.kLN.scale")
  tab32 = np.concatenate((qk_scale, attack.ravel(), scaled.ravel(), bg, qn, kn)).astype(np.float32)
  return tab16, tab32


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("onnx_in"); ap.add_argument("params"); ap.add_argument("onnx_out")
  ap.add_argument("--layers", default="", help="comma list of term layers to replace (default: all 16)")
  a = ap.parse_args(argv)
  P = dict(np.load(a.params))
  m = onnx.load(a.onnx_in)
  g = m.graph
  nodes = list(g.node)
  producer = {o: n for n in nodes for o in n.output}
  consumers = defaultdict(list)
  for n in nodes:
    for i in n.input:
      consumers[i].append(n)
  layers = [int(x) for x in a.layers.split(",")] if a.layers else list(range(16))

  def find(pred):
    hits = [n for n in nodes if pred(n)]
    assert len(hits) == 1, [h.name for h in hits]
    return hits[0]

  def only_consumer(name, op):
    hits = [c for c in consumers[name] if c.op_type == op]
    assert len(hits) >= 1, (name, op, [c.op_type for c in consumers[name]])
    return hits[0]

  # ---- shared: E [B, 34, 64, 64] (input of the prologue einsum with D) and S (rsqrt chain after it)
  ein = find(lambda n: n.op_type == "Einsum" and "inner.bt6h.prologue.D" in n.input)
  E = ein.input[0]
  z = only_consumer(ein.output[0], "Add").output[0]
  x = only_consumer(z, "Pow").output[0]
  x = only_consumer(x, "ReduceMean").output[0]
  x = only_consumer(x, "Add").output[0]
  x = only_consumer(x, "Sqrt").output[0]
  S = only_consumer(x, "Reciprocal").output[0]

  new_nodes, new_inits = [], []
  W = np.zeros((CH, 2), np.float32)
  for c in range(CH):
    W[c, 0 if c < 17 else 1] = float(2 ** (c if c < 17 else c - 17))
  new_inits.append(numpy_helper.from_array(W, "bt6p_bitweights"))
  new_nodes += [
    helper.make_node("Cast", [E], ["bt6p_E32"], to=TensorProto.FLOAT, name="bt6p_cast_E"),
    helper.make_node("Transpose", ["bt6p_E32"], ["bt6p_Et"], perm=[0, 2, 3, 1], name="bt6p_transpose_E"),
    helper.make_node("MatMul", ["bt6p_Et", "bt6p_bitweights"], ["bt6p_bits"], name="bt6p_pack_E"),
    helper.make_node("Cast", ["bt6p_bits"], ["bt6p_edges"], to=TensorProto.INT32, name="bt6p_edges_i32"),
  ]
  tpl = np.asarray(P["attn_templates.table"], np.float32).astype(np.float16)
  assert tpl.shape == (TEMPLATES, 64, 64), tpl.shape
  new_inits.append(numpy_helper.from_array(tpl, "bt6p_tpl"))
  new_inits.append(numpy_helper.from_array(np.array([-1, 16, 64, 64], np.int64), "bt6p_state_shape"))
  states = {}

  def state_of(e):
    if e not in states:
      k = len(states)
      new_nodes.append(helper.make_node("Reshape", [e, "bt6p_state_shape"], [f"bt6p_e4_{k}"], name=f"bt6p_state_reshape_{k}"))
      new_nodes.append(helper.make_node("Transpose", [f"bt6p_e4_{k}"], [f"bt6p_state_{k}"], perm=[0, 2, 3, 1],
                                        name=f"bt6p_state_cm_{k}"))
      states[e] = f"bt6p_state_{k}"
    return states[e]

  rewire = {}   # W_h MatMul node name -> plugin output
  for L in layers:
    s = f"inner.transformer_layer.{L}"
    in_layer = lambda n, sub: f"'{s}.{sub}'" in scope(n)
    qkv = find(lambda n: n.op_type == "MatMul" and in_layer(n, "attention.qkv")).output[0]
    coef = find(lambda n: n.op_type == "Add" and in_layer(n, "tmix.mix")).output[0]
    wh = find(lambda n: n.op_type == "MatMul" and in_layer(n, "attention.W_h"))
    gate_mm = [n for n in nodes if n.op_type == "MatMul" and f"'{s}.attention'" in scope(n) and "baddbmm" in scope(n)]
    assert len(gate_mm) == 1, (L, [n.name for n in gate_mm])
    e = gate_mm[0].input[1]
    tab16, tab32 = fold_tables(P, L)
    new_inits.append(numpy_helper.from_array(tab16, f"bt6p_tab16_{L}"))
    new_inits.append(numpy_helper.from_array(tab32, f"bt6p_tab32_{L}"))
    out = f"bt6p_attn_{L}"
    new_nodes.append(helper.make_node(
      PLUGIN, [qkv, coef, "bt6p_edges", S, state_of(e), f"bt6p_tab16_{L}", f"bt6p_tab32_{L}", "bt6p_tpl"], [out],
      name=f"bt6p_attention_{L}", domain=DOMAIN, heads=HEADS, head_dim=HD, templates=TEMPLATES,
      plugin_version="1", plugin_namespace=""))
    rewire[wh.name] = out
  for n in nodes:
    if n.name in rewire:
      n.input[0] = rewire[n.name]

  # ---- insert the new nodes before their first consumer (topological order), prune dead code
  all_nodes = nodes + new_nodes
  produced = {o: n for n in all_nodes for o in n.output}
  avail = {i.name for i in g.input} | {t.name for t in g.initializer} | {t.name for t in new_inits} | {""}
  ordered, placed = [], set()

  def place(n):
    if id(n) in placed:
      return
    stack = [(n, False)]
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

  live = set()
  work = [o.name for o in g.output]
  while work:
    t = work.pop()
    if t in live:
      continue
    live.add(t)
    if t in produced:
      work.extend(produced[t].input)
  keep = [n for n in all_nodes if any(o in live for o in n.output)]
  for n in keep:
    place(n)
  removed = len(all_nodes) - len(ordered)
  del g.node[:]
  g.node.extend(ordered)
  used = {i for n in ordered for i in n.input}
  inits = [t for t in list(g.initializer) + new_inits if t.name in used]
  del g.initializer[:]
  g.initializer.extend(inits)
  # value_info of removed tensors would confuse the parser: drop it (TensorRT infers shapes itself)
  del g.value_info[:]
  if not any(o.domain == DOMAIN for o in m.opset_import):
    m.opset_import.append(helper.make_opsetid(DOMAIN, 1))
  onnx.save(m, a.onnx_out)
  print(f"{len(layers)} plugin layers, {len(states)} state versions, {removed} nodes pruned, "
        f"{len(ordered)} nodes -> {a.onnx_out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
