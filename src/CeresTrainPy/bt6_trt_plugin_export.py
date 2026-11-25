"""BT6 TensorRT plugin export: rewrites the served ONNX of a flat BT6-terms net with the move-token decoder (for example
768x16 flat BT6 HL-Gauss) into the plugin ONNX that the trt_bt6_plugin library (bt6_trt_plugin.dll /
libbt6_trt_plugin.so, plugin domain 'trt.plugins') runs. make_plugin_onnx.py chains the steps; each step is one
function here that takes an onnx.ModelProto, changes it in place and returns a one-line summary.

Steps (round = the plugin round the step came with):
  dump_params        (needs torch)  FP32 parameters the plugin tables are folded from, .ts / checkpoint -> .npz
  terms_attention    round 1: every term layer's attention (qkv output + template mixer output up to the W_h input)
                     becomes one BT6TermsAttention v1 node
  prologue_sites     round 2: BT6Prologue, raw logits from the attention plugin (v2, export_h), channels-last edge
                     state after the sites, every edge site as one BT6EdgeSite node
  merge_gemms        coef: template mixer GEMM joins the qkv GEMM (fused_coef = 1); ffn: SwiGLU gate / up GEMMs merged
  port_qdq           the INT8 quantization of a served QDQ ONNX (qdq_export.py + qdq_to_fp16.py) on the plugin graph
  glu_quant          round 3: INT8 FFN down with per-token scales from BT6GluQuant
  attn_int8_out      round 3: BT6TermsAttention v3 writes the INT8 W_h input itself
  decoder            round 4: decoder attentions as BT6DecAttn, read-out (out_ln, policy, pools) as BT6DecHead
FP16 chain: terms_attention -> prologue_sites -> merge_gemms(coef, ffn) -> decoder
INT8 chain: terms_attention -> prologue_sites -> port_qdq(exclude 1536x768) -> merge_gemms(coef) -> glu_quant ->
            attn_int8_out -> decoder

CLI (one step on files, for debugging and the gate controls):
    python bt6_trt_plugin_export.py params <net.ts | ckpt> <out.npz>
    python bt6_trt_plugin_export.py terms|sites <in.onnx> <params.npz> <out.onnx>
    python bt6_trt_plugin_export.py merge <in.onnx> <out.onnx> [--coef] [--ffn]
    python bt6_trt_plugin_export.py port_qdq <in.onnx> <int8f16isl.onnx> <out.onnx> [--exclude KxN,...]
    python bt6_trt_plugin_export.py glu|i8out|decoder <in.onnx> <out.onnx> [--control ...]
Every graph pattern is found from the graph's structure and asserted, so a graph that differs fails loudly instead of
being rewritten wrongly. Export code only, not on the training path.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DOMAIN = "trt.plugins"
HEADS, HD, TEMPLATES, CH, PAD = 24, 32, 8, 34, 40


# ===================================================================================================== shared helpers

def _scope(node) -> str:
  for p in node.metadata_props:
    if p.key == "pkg.torch.onnx.name_scopes":
      return p.value
  return ""


def _graph_index(nodes):
  producer = {o: n for n in nodes for o in n.output}
  consumers = defaultdict(list)
  for n in nodes:
    for i in n.input:
      consumers[i].append(n)
  return producer, consumers


def _plugin_opset(m):
  if not any(o.domain == DOMAIN for o in m.opset_import):
    m.opset_import.append(helper.make_opsetid(DOMAIN, 1))


def _finish(g, nodes, new_inits, prune=True) -> int:
  """Graph = `nodes` (dead ones pruned unless prune=False) in topological order, each node placed after its
  producers; initializers = the used ones of the graph's plus `new_inits`; value_info dropped (removed tensors would
  confuse the parser, TensorRT infers shapes itself). Returns the number of nodes removed."""
  if prune:
    produced = {o: n for n in nodes for o in n.output}
    live, work = set(), [o.name for o in g.output]
    while work:
      t = work.pop()
      if t in live:
        continue
      live.add(t)
      if t in produced:
        work.extend(produced[t].input)
    keep = [n for n in nodes if any(o in live for o in n.output)]
  else:
    keep = list(nodes)
  produced = {o: n for n in keep for o in n.output}
  avail = {i.name for i in g.input} | {t.name for t in g.initializer} | {t.name for t in new_inits} | {""}
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
  removed = len(nodes) - len(ordered)
  del g.node[:]
  g.node.extend(ordered)
  used = {i for n in ordered for i in n.input}
  inits = [t for t in list(g.initializer) + list(new_inits) if t.name in used]
  del g.initializer[:]
  g.initializer.extend(inits)
  del g.value_info[:]
  return removed


# ===================================================================================================== parameters

PARAM_KEEP = (".bt6t.", ".qLN.", ".kLN.", "bt6h.prologue.", "attn_templates.")


def dump_params(src: str, out: str) -> str:
  """Every parameter / buffer whose name contains one of PARAM_KEEP (leading 'inner.' / 'module.' / '_orig_mod.'
  stripped), from a TorchScript .ts or a checkpoint state dict, to an .npz (FP32 values as trained)."""
  import torch
  if src.endswith(".ts"):
    m = torch.jit.load(src, map_location="cpu")
    items = list(m.named_parameters()) + list(m.named_buffers())
  else:
    sd = torch.load(src, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    items = list(sd.items())
  kept = {}
  for name, value in items:
    if any(k in name for k in PARAM_KEEP):
      for p in ("_orig_mod.", "module.", "inner."):
        if name.startswith(p):
          name = name[len(p):]
      kept[name] = value.detach().cpu().numpy()
  if "bt6h.prologue.relidx" not in kept:
    from bt6_net import relative_index
    kept["bt6h.prologue.relidx"] = relative_index().numpy()
  np.savez(out, **kept)
  return f"{len(kept)} tensors -> {out}"


# ===================================================================================================== round 1

def fold_tables(P: dict, layer: int):
  """(tab16 FP16, tab32 FP32) of one term layer: bt6_terms_export.py's served folds, in float64. Layout in
  trt_bt6_plugin/src/bt6_attn_kernels.cu."""
  g = f"transformer_layer.{layer}.attention"
  if f"{g}.bt6t.Ue" in P:
    raise NotImplementedError(f"layer {layer}: BT6EdgeValues (edge message Ue) is not in the TensorRT plugin yet; "
                              "serve the plain ONNX")
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
  assert t.shape[0] == HEADS and Pq.shape == (HEADS, CH, HD), (t.shape, Pq.shape)
  qpad = np.zeros((HEADS, PAD, HD)); qpad[:, :CH] = query
  kpad = np.zeros((HEADS, PAD, HD)); kpad[:, :CH] = key
  rdg = np.concatenate((read, Wm, Wg), axis=1)                               # [H, 48]
  tab16 = np.concatenate((qpad.ravel(), kpad.ravel(), rdg.ravel(), cbias.ravel())).astype(np.float16)
  qn, kn = f(f"{g}.qLN.scale"), f(f"{g}.kLN.scale")
  tab32 = np.concatenate((qk_scale, attack.ravel(), scaled.ravel(), bg, qn, kn)).astype(np.float32)
  return tab16, tab32


def terms_attention(m, P: dict) -> str:
  """Round 1, on the served FP16 ONNX. Per term layer L (scope inner.transformer_layer.L) everything from the qkv
  GEMM output (attention.qkv) and the template mixer output (tmix.mix) up to the input of the out-projection
  (attention.W_h) becomes one BT6TermsAttention v1 node: head split, QK-norm, relation terms (Pq / Pk / C1 against E),
  the stacked edge / pair / door / gate reads, logit assembly, door, softmax, post-softmax gate, A @ V, head merge.
  Nodes still needed elsewhere (the q.k and template bias the edge sites read) stay; the rest is pruned.
  Shared inputs, built once with TensorRT ops:
    edges  INT32 [B, 64, 64, 2] = Cast(MatMul(Transpose(Cast(E, f32)), W)), W[c] = 2^c (c < 17) | 2^(c - 17): exact
    norm   S [B, 64, 64] FP16 (the prologue's rsqrt(mean_d Z^2 + eps), as exported)
    state  Transpose(Reshape(e, [B, 16, 64, 64]), [0, 2, 3, 1]) per edge-state version (e0 and after every site)"""
  g = m.graph
  nodes = list(g.node)
  producer, consumers = _graph_index(nodes)

  def find(pred):
    hits = [n for n in nodes if pred(n)]
    assert len(hits) == 1, [h.name for h in hits]
    return hits[0]

  def only_consumer(name, op):
    hits = [c for c in consumers[name] if c.op_type == op]
    assert len(hits) >= 1, (name, op, [c.op_type for c in consumers[name]])
    return hits[0]

  # E [B, 34, 64, 64] (input of the prologue einsum with D) and S (rsqrt chain after it)
  ein = find(lambda n: n.op_type == "Einsum" and "inner.bt6h.prologue.D" in n.input)
  E = ein.input[0]
  x = only_consumer(ein.output[0], "Add").output[0]
  for op in ("Pow", "ReduceMean", "Add", "Sqrt"):
    x = only_consumer(x, op).output[0]
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
  layers = list(range(16))
  for L in layers:
    s = f"inner.transformer_layer.{L}"
    in_layer = lambda n, sub: f"'{s}.{sub}'" in _scope(n)
    qkv = find(lambda n: n.op_type == "MatMul" and in_layer(n, "attention.qkv")).output[0]
    coef = find(lambda n: n.op_type == "Add" and in_layer(n, "tmix.mix")).output[0]
    wh = find(lambda n: n.op_type == "MatMul" and in_layer(n, "attention.W_h"))
    gate_mm = [n for n in nodes if n.op_type == "MatMul" and f"'{s}.attention'" in _scope(n) and "baddbmm" in _scope(n)]
    assert len(gate_mm) == 1, (L, [n.name for n in gate_mm])
    e = gate_mm[0].input[1]
    tab16, tab32 = fold_tables(P, L)
    new_inits.append(numpy_helper.from_array(tab16, f"bt6p_tab16_{L}"))
    new_inits.append(numpy_helper.from_array(tab32, f"bt6p_tab32_{L}"))
    out = f"bt6p_attn_{L}"
    new_nodes.append(helper.make_node(
      "BT6TermsAttention", [qkv, coef, "bt6p_edges", S, state_of(e), f"bt6p_tab16_{L}", f"bt6p_tab32_{L}", "bt6p_tpl"],
      [out], name=f"bt6p_attention_{L}", domain=DOMAIN, heads=HEADS, head_dim=HD, templates=TEMPLATES,
      plugin_version="1", plugin_namespace=""))
    rewire[wh.name] = out
  for n in nodes:
    if n.name in rewire:
      n.input[0] = rewire[n.name]
  removed = _finish(g, nodes + new_nodes, new_inits)
  _plugin_opset(m)
  return f"{len(layers)} BT6TermsAttention layers, {len(states)} state versions, {removed} nodes pruned, {len(g.node)} nodes"


# ===================================================================================================== round 2

def prologue_sites(m, P: dict) -> str:
  """Round 2, on the round-1 graph:
    prologue  the whole prologue (record -> piece boards -> 34 relation channels E -> S, e0, and round 1's INT32 bit
              packing of E) becomes one BT6Prologue node: squares + tab FP32 [8288] (D | Pin | T | Toff) ->
              (edges INT32 [B,64,64,2], S FP16 [B,64,64], e0 FP16 [B,64,64,16] channels-last). e0 channels-last is the
              first state input of the attention nodes and edge site 1's channels-last input.
    state_cm  the edge state after a site stays channels-last: attention nodes, the next site and the decoder's edge
              input read the site's channels-last result instead of Transpose -> Reshape -> Transpose copies.
    export_h  the attention node of every layer whose raw logits an edge site reads (q.k / sqrt(hd) + template bias)
              gets export_h = 1 and its second output replaces that Add; the layer's QK-norm / q.k nodes die.
    sites     every edge site (readback, rms, edge FFN with the reverse edge, rms) becomes one BT6EdgeSite node on the
              channels-last state and the plugin's raw logits; tab FP32 [3520] from the site's (FP16) initializers.
  All attention nodes become version 2 (export_h 0 / 1)."""
  g = m.graph
  nodes = list(g.node)
  producer, consumers = _graph_index(nodes)
  attn = sorted([n for n in nodes if n.op_type == "BT6TermsAttention"], key=lambda n: int(n.name.rsplit("_", 1)[1]))
  assert len(attn) == 16, len(attn)
  new_nodes, new_inits = [], []

  def replace_uses(old, new):
    for c in list(consumers[old]):
      for k, i in enumerate(c.input):
        if i == old:
          c.input[k] = new
      consumers[new].append(c)
    consumers[old] = []

  # edge-state tensors per state version: state input k = Transpose(Reshape(Reshape(e_k))), e_k channels-first
  state_inputs = sorted({n.input[4] for n in attn}, key=lambda s: int(s.rsplit("_", 1)[1]))
  e_first = []
  for s in state_inputs:
    t = producer[s]; r = producer[t.input[0]]; v = producer[r.input[0]]
    assert t.op_type == "Transpose" and r.op_type == "Reshape" and v.op_type == "Reshape", (s, t.op_type, r.op_type, v.op_type)
    e_first.append(v.input[0])
  # channels-last twin of every e_k: the site computes channels-last, then Transpose [0, 3, 1, 2]
  e_last = [None]
  for e in e_first[1:]:
    t = producer[e]
    assert t.op_type == "Transpose" and list(t.attribute[0].ints) == [0, 3, 1, 2], (e, t.op_type)
    e_last.append(t.input[0])

  # ---- prologue
  edges, S = attn[0].input[2], attn[0].input[3]
  assert all(n.input[2] == edges and n.input[3] == S for n in attn)
  tab = np.concatenate([np.asarray(P[k], np.float64).ravel() for k in
                        ("bt6h.prologue.D", "bt6h.prologue.Pin", "bt6h.prologue.T", "bt6h.prologue.Toff")])
  assert tab.size == 8288, tab.size
  assert np.array_equal(np.asarray(P["bt6h.prologue.relidx"]),
                        15 * (np.arange(64)[:, None] // 8 - np.arange(64)[None, :] // 8 + 7)
                        + (np.arange(64)[:, None] % 8 - np.arange(64)[None, :] % 8 + 7))
  new_inits.append(numpy_helper.from_array(tab.astype(np.float32), "bt6q_tab"))
  new_nodes.append(helper.make_node("BT6Prologue", ["squares", "bt6q_tab"], ["bt6q_edges", "bt6q_S", "bt6q_e0"],
                                    name="bt6q_prologue", domain=DOMAIN, plugin_version="1", plugin_namespace=""))
  for n in attn:
    n.input[2] = "bt6q_edges"
    n.input[3] = "bt6q_S"
  for c in list(consumers[e_first[0]]):
    if c.op_type == "Transpose" and list(c.attribute[0].ints) == [0, 2, 3, 1]:
      replace_uses(c.output[0], "bt6q_e0")
  for n in attn:
    if n.input[4] == state_inputs[0]:
      n.input[4] = "bt6q_e0"

  # ---- channels-last state after the sites
  for k in range(1, len(e_first)):
    ef, el = e_first[k], e_last[k]
    for n in attn:
      if n.input[4] == state_inputs[k]:
        n.input[4] = el
    for c in list(consumers[ef]):
      if c.op_type == "Transpose" and list(c.attribute[0].ints) == [0, 2, 3, 1]:
        replace_uses(c.output[0], el)

  # ---- raw logits from the plugin (site layers)
  exported = []
  for L, n in enumerate(attn):
    export = 0
    s = f"'inner.transformer_layer.{L}'"
    adds = [x for x in nodes if x.op_type == "Add" and s in _scope(x)
            and any("bt6h.sites" in _scope(c) for c in consumers[x.output[0]])]
    if adds:
      assert len(adds) == 1, (L, [x.name for x in adds])
      hraw = f"bt6p_hraw_{L}"
      replace_uses(adds[0].output[0], hraw)
      n.output.append(hraw)
      export = 1
      exported.append(L)
    for at in list(n.attribute):
      if at.name in ("plugin_version", "export_h"):
        n.attribute.remove(at)
    n.attribute.extend([helper.make_attribute("plugin_version", "2"), helper.make_attribute("export_h", export)])

  # ---- edge sites as BT6EdgeSite nodes
  inits = {t.name: t for t in g.initializer}
  W = lambda name: numpy_helper.to_array(inits[name]).astype(np.float64)

  def site_scope(x):
    parts = _scope(x).split(", ")
    return parts[-2] if len(parts) >= 2 else ""

  # numeric order of the site index ('inner.bt6h.sites.13'): a text sort puts 11, 13 before 3 and wires the sites to
  # the wrong state versions (a cycle) as soon as a site index has two digits
  site_scopes = sorted({site_scope(x) for x in nodes if site_scope(x).startswith("'inner.bt6h.sites.")},
                       key=lambda sc: int(sc.strip("'").rsplit(".", 1)[1]))
  for k, sc in enumerate(site_scopes, start=1):
    sn = [x for x in nodes if site_scope(x) == sc]
    mms = [x for x in sn if x.op_type == "MatMul"]
    assert len(mms) == 3, (sc, [x.name for x in mms])
    rb, f1, f2 = mms                                                     # readback, ffn in [n | n^T], ffn out
    add0 = [x for x in sn if x.op_type == "Add" and x.input[1] == rb.output[0]]
    assert len(add0) == 1, sc
    e_in = add0[0].input[0]
    tr = producer[rb.input[0]]
    assert tr.op_type == "Transpose", (sc, tr.op_type)
    hraw = tr.input[0]
    assert hraw.startswith("bt6p_hraw_"), (sc, hraw)
    bias = [x for x in sn if x.op_type == "Add" and x.input[0] == f1.output[0]]
    assert len(bias) == 1 and bias[0].input[1] in inits, sc
    R, F1, F2, b1 = W(rb.input[1]), W(f1.input[1]), W(f2.input[1]), W(bias[0].input[1])
    assert R.shape == (24, 16) and F1.shape == (32, 64) and F2.shape == (64, 16) and b1.shape == (64,), sc
    tab = np.concatenate((R.T.ravel(), F1[:16].T.ravel(), F1[16:].T.ravel(), b1, F2.T.ravel())).astype(np.float32)
    assert tab.size == 3520
    out = e_last[k]
    new_inits.append(numpy_helper.from_array(tab, f"bt6s_tab_{k}"))
    new_nodes.append(helper.make_node("BT6EdgeSite", [e_in, hraw, f"bt6s_tab_{k}"], [f"bt6s_state_{k}"],
                                      name=f"bt6s_site_{k}", domain=DOMAIN, plugin_version="1", plugin_namespace=""))
    for x in nodes + new_nodes[:-1]:
      for q, i in enumerate(x.input):
        if i == out:
          x.input[q] = f"bt6s_state_{k}"
    e_last[k] = f"bt6s_state_{k}"

  removed = _finish(g, nodes + new_nodes, new_inits)
  return (f"BT6Prologue, raw logits from layers {exported}, {len(site_scopes)} BT6EdgeSite: {removed} nodes pruned, "
          f"{len(g.node)} nodes")


# ===================================================================================================== GEMM merges

def merge_gemms(m, coef: bool, ffn: bool) -> str:
  """Horizontal GEMM merges (FP16 graph, or INT8 after port_qdq).
    coef  per term layer the template mixer GEMM (x @ Wm + b, tmix.mix) joins the qkv GEMM (same x): one GEMM
          x @ [Wq | Wm] -> [B, 64, 2304 + 192], read by BT6TermsAttention with fused_coef = 1 (input 1 = the mixer
          bias). With an INT8 qkv GEMM (DequantizeLinear of INT8, per output channel) Wm is quantized per output
          channel (symmetric, max |w| / 127) and the qkv output fake-quant is dropped.
    ffn   every pair of GEMMs on the same input with weights of the same kind and the FFN shape (768 -> 1536 twice,
          SwiGLU gate / up): one GEMM on the concatenated weight, then Split back into the two output tensors."""
  g = m.graph
  inits = {t.name: t for t in g.initializer}
  producer, _ = _graph_index(g.node)
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
      if [x.i for x in dq.attribute if x.name == "axis"] == [1]:
        return "q8", (arr(dq.input[0]), arr(dq.input[1]), arr(dq.input[2]), dq)
    return None

  merged_coef, merged_ffn = 0, 0
  if coef:
    for n in [x for x in g.node if x.op_type == "BT6TermsAttention"]:
      fused = [x for x in n.attribute if x.name == "fused_coef"]
      if fused and fused[0].i == 1:
        continue
      p = producer[n.input[0]]                         # the qkv GEMM, possibly behind an output Q -> DQ
      if p.op_type == "DequantizeLinear":
        q = producer[p.input[0]]
        assert q.op_type == "QuantizeLinear"
        p = producer[q.input[0]]
      assert p.op_type == "MatMul", (n.name, p.op_type)
      mq = p
      add = producer[n.input[1]]                       # the mixer Add(MatMul, bias)
      assert add.op_type == "Add" and add.input[1] in inits, (n.name, add.op_type)
      mm = producer[add.input[0]]
      assert mm.op_type == "MatMul" and mm.input[1] in inits, (n.name, mm.op_type)
      kq = weight_kind(mq)
      wm = arr(mm.input[1]).astype(np.float32)
      xq, xm = mq.input[0], mm.input[0]                # same activation: x, or Q -> DQ of x
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

  if ffn:
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

  _finish(g, [n for n in g.node if n.name not in dead] + new_nodes, new_inits)
  return f"coef merges {merged_coef}, ffn merges {merged_ffn}: {len(g.node)} nodes"


# ===================================================================================================== INT8 port

def port_qdq(m, q, exclude: str = "") -> str:
  """The INT8 quantization of the served QDQ ONNX `q` of the SAME net (qdq_export.py + qdq_to_fp16.py, trunk GEMM
  allowlist) onto the plugin graph `m`. Both come from the same torch export, so GEMM nodes keep their names. For every
  MatMul of `q` whose weight is a DequantizeLinear of an INT8 initializer (weight shape not in `exclude`, "KxN,..."),
  the plugin graph's MatMul of the same name gets that quantization: the INT8 weight with its DequantizeLinear, Q -> DQ
  with the same activation scales on whatever tensor feeds it in the plugin graph, and the output Q -> DQ when `q` has
  one. Quantize nodes on one tensor are shared."""
  ex = {tuple(int(v) for v in x.split("x")) for x in exclude.split(",") if x}
  g, qg = m.graph, q.graph
  qprod, qcons = _graph_index(qg.node)
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
  _, pcons = _graph_index(g.node)
  for mm in qg.node:
    if mm.op_type != "MatMul":
      continue
    wdq = qprod.get(mm.input[1])
    if wdq is None or wdq.op_type != "DequantizeLinear" or wdq.input[0] not in qinit:
      continue
    if tuple(qinit[wdq.input[0]].dims) in ex:
      continue
    pm = pnodes.get(mm.name)
    if pm is None or pm.op_type != "MatMul":
      missing.append(mm.name)
      continue
    for i in wdq.input:
      take_init(i)
    wout = f"{mm.name}__w8"
    new_nodes.append(clone(wdq, list(wdq.input), [wout], f"{mm.name}__WDQ8"))
    pm.input[1] = wout
    adq = qprod[mm.input[0]]
    aq = qprod[adq.input[0]]
    assert adq.op_type == "DequantizeLinear" and aq.op_type == "QuantizeLinear", (mm.name, adq.op_type, aq.op_type)
    pm.input[0] = qdq(pm.input[0], aq, adq, "in")
    oq = [c for c in qcons[mm.output[0]] if c.op_type == "QuantizeLinear"]
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
      assert all(o.name != out for o in g.output), "a quantized GEMM output is a graph output"
    ported += 1
  _finish(g, list(g.node) + new_nodes, new_inits, prune=False)
  return f"{ported} GEMMs quantized ({len(missing)} INT8 GEMMs without a twin: {missing[:8]}), {len(new_nodes)} Q/DQ nodes"


# ===================================================================================================== round 3

def glu_quant(m, control: str = "") -> str:
  """Round 3: INT8 FFN down with dynamic per-token activation scales, on an INT8 graph whose FFN down is still FP16
  and whose gate / up GEMMs are NOT merged (port_qdq exclude 1536x768, merge_gemms coef only: two separate INT8 GEMMs
  let TensorRT write their outputs as INT8). Per FFN layer
    gate GEMM -> Q(sg) -> DQ -> + b -> SiLU ;  up GEMM -> Q(su) -> DQ ;  m = SiLU * up ;  FFN down MatMul(m, W FP16)
  becomes BT6GluQuant(Q(gate), Q(up), b; gate_scale sg, up_scale su) -> codes INT8, rmax FP16 [B, 64, 1], then
  DQ(codes, 1/127) @ DQ(W INT8, per output channel, symmetric) -> * rmax -> + bias (as before).
  control 'noscale' (must fail the gate): the GEMM output is not multiplied back by rmax; 'nobias': zero gate bias."""
  assert control in ("", "noscale", "nobias"), control
  g = m.graph
  inits = {t.name: t for t in g.initializer}
  arr = lambda name: numpy_helper.to_array(inits[name])
  prod, _ = _graph_index(g.node)
  downs = [n for n in g.node if n.op_type == "MatMul" and n.input[1] in inits and list(inits[n.input[1]].dims) == [1536, 768]]
  assert len(downs) == 16, len(downs)
  new_inits, new_nodes, dead = [], [], set()

  def add_init(array, name):
    new_inits.append(numpy_helper.from_array(array, name)); return name

  def qdq_of(t):
    """t = DQ(Q(x)) -> (Q node, scale)."""
    dq = prod[t]; assert dq.op_type == "DequantizeLinear", (t, dq.op_type)
    q = prod[dq.input[0]]; assert q.op_type == "QuantizeLinear", (t, q.op_type)
    sq, sd = float(arr(q.input[1])), float(arr(dq.input[1]))
    assert sq == sd and not arr(q.input[2]).any(), (t, sq, sd)
    return q, sq

  for l, mm in enumerate(downs):
    L = f"r3g{l}"
    mid = prod[mm.input[0]]
    assert mid.op_type == "Mul", mid.op_type
    k = 1 if prod[mid.input[1]].op_type == "DequantizeLinear" else 0     # which factor is the up DQ output
    upt, silu = mid.input[k], prod[mid.input[1 - k]]
    assert silu.op_type == "Mul" and prod[upt].op_type == "DequantizeLinear", (silu.op_type, prod[upt].op_type)
    add = prod[[x for x in silu.input if prod[x].op_type == "Add"][0]]    # g + b (Sigmoid's input too)
    gdq_t, bias = (add.input[0], add.input[1]) if add.input[1] in inits else (add.input[1], add.input[0])
    qg, sg = qdq_of(gdq_t)
    qu, su = qdq_of(upt)
    W = arr(mm.input[1]).astype(np.float32)
    w_scale = (np.maximum(np.abs(W).max(axis=0), 1e-8) / 127.0).astype(np.float16)
    q = np.clip(np.round(W / w_scale.astype(np.float32)), -127, 127).astype(np.int8)
    inv127 = add_init(np.array(1.0 / 127.0, np.float16), f"{L}_inv127")
    zp = add_init(np.array(0, np.int8), f"{L}_zp")
    b_in = bias if control != "nobias" else add_init(np.zeros(1536, np.float16), f"{L}_zero_bias")
    r = f"{L}_rmax"
    node = helper.make_node("BT6GluQuant", [qg.output[0], qu.output[0], b_in], [f"{L}_codes", r], name=f"{L}_GluQuant",
                            domain=DOMAIN)
    node.attribute.extend([helper.make_attribute("plugin_version", "1"), helper.make_attribute("plugin_namespace", ""),
                           helper.make_attribute("gate_scale", float(np.float32(sg))),
                           helper.make_attribute("up_scale", float(np.float32(su)))])
    new_nodes.append(node)
    dead.add(mid.name)                                                    # the SiLU * up chain dies (pruned)
    new_nodes.append(helper.make_node("DequantizeLinear", [f"{L}_codes", inv127, zp], [f"{L}_dq"], name=f"{L}_DQ8"))
    new_nodes.append(helper.make_node("DequantizeLinear", [add_init(q, f"{L}_wq"), add_init(w_scale, f"{L}_ws"),
                                                           add_init(np.zeros(768, np.int8), f"{L}_wz")],
                                      [f"{L}_w"], name=f"{L}_WDQ8", axis=1))
    y = mm.output[0]
    mm.input[0], mm.input[1] = f"{L}_dq", f"{L}_w"
    if control != "noscale":
      mm.output[0] = f"{L}_yn"
      new_nodes.append(helper.make_node("Mul", [f"{L}_yn", r], [y], name=f"{L}_RowScale"))
  _finish(g, [n for n in g.node if n.name not in dead] + new_nodes, new_inits)
  _plugin_opset(m)
  return f"{len(downs)} FFN layers: INT8 FFN down with BT6GluQuant{' CONTROL ' + control if control else ''}: {len(g.node)} nodes"


def attn_int8_out(m, control_scale: float = 1.0) -> str:
  """Round 3: every attention output that feeds only QuantizeLinear(s) -> DequantizeLinear(s) -> W_h becomes the
  INT8 codes of BT6TermsAttention v3 (out_int8 = 1, out_scale = s); the QuantizeLinear goes. control_scale K
  (must fail the gate): the plugin quantizes with K * s while the DequantizeLinear keeps s."""
  g = m.graph
  inits = {t.name: t for t in g.initializer}
  _, cons = _graph_index(g.node)
  dead = set()
  for n in g.node:
    if n.op_type != "BT6TermsAttention":
      continue
    users = cons[n.output[0]]
    if len(users) != 1 or users[0].op_type != "QuantizeLinear":
      continue
    q = users[0]
    dqs = cons[q.output[0]]
    assert len(dqs) == 1 and dqs[0].op_type == "DequantizeLinear", n.name
    dq = dqs[0]
    sq = float(numpy_helper.to_array(inits[q.input[1]]))
    sd = float(numpy_helper.to_array(inits[dq.input[1]]))
    assert sq == sd and not numpy_helper.to_array(inits[q.input[2]]).any(), (n.name, sq, sd)
    for x in list(n.attribute):
      if x.name in ("plugin_version", "out_int8", "out_scale"):
        n.attribute.remove(x)
    n.attribute.extend([helper.make_attribute("plugin_version", "3"), helper.make_attribute("out_int8", 1),
                        helper.make_attribute("out_scale", float(np.float32(sq * control_scale)))])
    dq.input[0] = n.output[0]
    dead.add(q.name)
  keep = [n for n in g.node if n.name not in dead]
  del g.node[:]
  g.node.extend(keep)
  del g.value_info[:]
  return f"{len(dead)} attention plugins write INT8 (v3){' CONTROL scale x%g' % control_scale if control_scale != 1 else ''}"


# ===================================================================================================== round 4

def decoder(m, control: str = "") -> str:
  """Round 4: the move-token decoder's attentions and its policy / pool read-out as plugins (the decoder region is
  FP16 in both chains). Per decoder block (MoveTokenBlock with post_move) three attentions between TensorRT GEMMs:
    self   Split(qkv GEMM) -> Reshape/Transpose -> q k^T * s + key bias -> Softmax -> @ v -> Transpose/Reshape -> proj
    cross  xq GEMM, k / v = the block's columns of the shared square K|V GEMM [B, 64, 4096] (Split 8 x 512 -> Split 2)
    pm     pm_q GEMM, k / v from the same K|V GEMM, + the one-hot to / from edits with GatherND(pm_dk / pm_dv, mover)
  each become one BT6DecAttn node (mode 0 / 1 / 2) whose output is the tensor the projection GEMM reads. The post-move
  indices (from, to, mover) are cast once to one INT32 [B, 80, 3] tensor shared by the four pm nodes.
  out_ln + pol (256 -> 4) + where(valid) + scatter + index_select + per-move bias -> `policy`, and the masked mean /
  max pools -> the v_inject input, become one BT6DecHead node.
  Controls (must fail the gate): 'nobias' self attention without the key bias; 'noedit' zero post-move dk / dv;
  'kvoff' every cross attention reads block 0's K|V columns; 'headnomask' BT6DecHead with an all-valid key bias."""
  assert control in ("", "nobias", "noedit", "kvoff", "headnomask"), control
  g = m.graph
  inits = {t.name: t for t in g.initializer}
  arr = lambda name: numpy_helper.to_array(inits[name])
  prod, cons = _graph_index(g.node)
  new_inits, new_nodes = [], []

  def add_init(array, name):
    new_inits.append(numpy_helper.from_array(array, name)); return name

  def is_init(t):
    return t in inits

  def up_view(t):
    """Walk up through Transpose / Reshape to the tensor they view."""
    while t in prod and prod[t].op_type in ("Transpose", "Reshape"):
      t = prod[t].input[0]
    return t

  def plugin(op, ins, outs, name, **attrs):
    n = helper.make_node(op, ins, outs, name=name, domain=DOMAIN)
    n.attribute.extend([helper.make_attribute("plugin_version", "1"), helper.make_attribute("plugin_namespace", "")])
    for k, v in attrs.items():
      n.attribute.append(helper.make_attribute(k, v))
    new_nodes.append(n)
    return n

  # ---- the shared square K|V GEMM: Split into 8 x 512 (xkv 0..3 | pm_kv 0..3)
  kvsplit = [n for n in g.node if n.op_type == "Split" and len(n.output) == 8]
  assert len(kvsplit) == 1, len(kvsplit)
  kvsplit = kvsplit[0]
  assert arr(kvsplit.input[1]).tolist() == [512] * 8
  kv = kvsplit.input[0]
  kv_part = {o: j for j, o in enumerate(kvsplit.output)}

  # ---- body softmaxes (the ones feeding a MatMul), in topological order: per block self, cross, pm
  sms = [n for n in g.node if n.op_type == "Softmax" and any(c.op_type == "MatMul" for c in cons[n.output[0]])]
  assert len(sms) == 12, len(sms)
  scale = float(np.float32(32 ** -0.5))
  att = []
  for si, sm in enumerate(sms):
    blk, kind = si // 3, si % 3
    # score MatMul: walk up through Mul / Add (non-initializer operands)
    frontier, mm = [sm.input[0]], None
    while frontier and mm is None:
      p = prod[frontier.pop(0)]
      if p.op_type == "MatMul" and not any(is_init(x) for x in p.input):
        mm = p; break
      if p.op_type not in ("Mul", "Add"):
        continue
      frontier.extend(x for x in p.input if not is_init(x))
    qs, ks = up_view(mm.input[0]), up_view(mm.input[1])
    # output: forward to the Reshape whose consumer is the [256, 256] projection
    o, frontier, seen = None, [sm.output[0]], set()
    while frontier and o is None:
      t = frontier.pop(0)
      for c in cons[t]:
        if c.name in seen:
          continue
        seen.add(c.name)
        if c.op_type == "Reshape" and any(cc.op_type == "MatMul" and cc.input[1] in inits and
                                         list(inits[cc.input[1]].dims) == [256, 256] for cc in cons[c.output[0]]):
          o = c.output[0]; break
        frontier.extend(c.output)
    assert o is not None, sm.name
    rec = dict(block=blk, kind=kind, o=o, sm=sm.name)
    if kind == 0:
      sp = prod[qs]
      assert sp.op_type == "Split" and len(sp.output) == 3 and list(sp.output).index(qs) == 0, (sm.name, sp.op_type)
      assert ks == sp.output[1], (ks, sp.output)
      rec["qkv"] = sp.input[0]
      W = prod[sp.input[0]]
      assert W.op_type == "MatMul" and list(inits[W.input[1]].dims) == [256, 768], W.name
      adds = [prod[x] for x in sm.input if prod[x].op_type == "Add"]       # Add(scaled scores, Reshape(bias [B, 80]))
      assert len(adds) == 1, sm.name
      kb = [x for x in adds[0].input if prod[x].op_type == "Reshape"]
      assert len(kb) == 1, sm.name
      rec["kbias"] = prod[kb[0]].input[0]
    else:
      assert prod[qs].op_type == "MatMul" and list(inits[prod[qs].input[1]].dims) == [256, 256], (sm.name, prod[qs].op_type)
      rec["q"] = qs
      sp2 = prod[ks]
      assert sp2.op_type == "Split" and len(sp2.output) == 2 and sp2.output[0] == ks, sm.name
      j = kv_part[sp2.input[0]]
      assert j == (blk if kind == 1 else 4 + blk), (sm.name, j)
      rec["k_off"], rec["v_off"] = 512 * j, 512 * j + 256
      if kind == 2:
        # GatherND tables (dk in the score cone, dv in the output cone), dk / dv row 0 == the from-square constants
        def cone(start, depth):
          out, fr, vis = [], [(s, 0) for s in start], set()
          while fr:
            t, d = fr.pop()
            if t in vis or t not in prod or d > depth:
              continue
            vis.add(t)
            p = prod[t]
            out.append(p)
            if t in (qs, ks):
              continue
            fr.extend((x, d + 1) for x in p.input if not is_init(x))
          return out
        sc = cone([sm.input[0]], 12)
        scn = {p.name for p in sc}
        oc = [p for p in cone([o], 12) if p.name not in scn]
        gk = [p for p in sc if p.op_type == "GatherND"]
        gv = [p for p in oc if p.op_type == "GatherND"]
        assert len(gk) == 1 and len(gv) == 1, (sm.name, len(gk), len(gv))
        dk, dv = arr(gk[0].input[0]).astype(np.float32), arr(gv[0].input[0]).astype(np.float32)
        assert dk.shape == (7, 256) and dv.shape == (7, 256)
        fk = [x for p in sc if p.op_type == "Mul" for x in p.input if is_init(x) and list(inits[x].dims) == [1, 8, 1, 32]]
        fv = [x for p in oc if p.op_type == "Mul" for x in p.input if is_init(x) and list(inits[x].dims) == [1, 8, 1, 32]]
        assert len(fk) == 1 and len(fv) == 1, (sm.name, fk, fv)
        assert np.array_equal(arr(fk[0]).astype(np.float32).ravel(), dk[0]), sm.name
        assert np.array_equal(arr(fv[0]).astype(np.float32).ravel(), dv[0]), sm.name
        rec["tab"] = np.stack([dk, dv]).astype(np.float16)
        mover = prod[gk[0].input[1]]
        assert mover.op_type == "Unsqueeze", mover.op_type
        rec["mover"] = mover.input[0]
        assert prod[gv[0].input[1]].input[0] == rec["mover"]
        # to / from: the Equal whose one-hot multiplies the dk[mover] term is 'to'
        eqs = [p for p in sc if p.op_type == "Equal"]
        assert len(eqs) == 2, (sm.name, len(eqs))
        roles = {}
        for e in eqs:
          idx = [x for x in e.input if not is_init(x)][0]
          idx = prod[idx].input[0] if prod[idx].op_type == "Unsqueeze" else idx
          t, mulnode = e.output[0], None                                   # the Mul consuming this one-hot
          for _ in range(4):
            cs = [c for c in cons[t] if c.name in scn]
            if not cs:
              break
            if cs[0].op_type == "Mul":
              mulnode = cs[0]; break
            t = cs[0].output[0]
          assert mulnode is not None, (sm.name, e.name)
          other = [x for x in mulnode.input if x != t][0]
          roles["to" if any(p.op_type == "GatherND" for p in cone([other], 6)) else "from"] = idx
        assert set(roles) == {"to", "from"}, (sm.name, roles)
        rec["to"], rec["from"] = roles["to"], roles["from"]
    att.append(rec)

  # ---- post-move index tensor (shared): Cast(from, to, mover) -> INT32 [B, 80, 3]
  pms = [r for r in att if r["kind"] == 2]
  for r in pms[1:]:
    assert (r["from"], r["to"], r["mover"]) == (pms[0]["from"], pms[0]["to"], pms[0]["mover"])
  parts = []
  for nm, t in (("from", pms[0]["from"]), ("to", pms[0]["to"]), ("mover", pms[0]["mover"])):
    new_nodes.append(helper.make_node("Cast", [t], [f"r4_{nm}_i32"], to=TensorProto.INT32, name=f"r4_cast_{nm}"))
    new_nodes.append(helper.make_node("Unsqueeze", [f"r4_{nm}_i32", add_init(np.array([-1], np.int64), f"r4_ax_{nm}")],
                                      [f"r4_{nm}_u"], name=f"r4_unsq_{nm}"))
    parts.append(f"r4_{nm}_u")
  idx_name = "r4_pm_idx"
  new_nodes.append(helper.make_node("Concat", parts, [idx_name], axis=-1, name="r4_pm_idx_cat"))

  for r in att:
    nm = f"r4_dec{r['block']}_{['self', 'cross', 'pm'][r['kind']]}"
    prod[r["o"]].output[0] = r["o"] + "_r4dead"   # the old producer's output is renamed (pruned below)
    prod.pop(r["o"])
    if r["kind"] == 0:
      kb = r["kbias"]
      if control == "nobias":
        kb = nm + "_zero_kbias"
        new_nodes.append(helper.make_node("Mul", [r["kbias"], add_init(np.array(0, np.float16), nm + "_zero")], [kb],
                                          name=nm + "_zero_kbias_mul"))
      plugin("BT6DecAttn", [r["qkv"], kb], [r["o"]], nm, mode=0, k_off=256, v_off=512, scale=scale)
    elif r["kind"] == 1:
      ko, vo = (0, 256) if control == "kvoff" else (r["k_off"], r["v_off"])
      plugin("BT6DecAttn", [r["q"], kv], [r["o"]], nm, mode=1, k_off=ko, v_off=vo, scale=scale)
    else:
      tab = r["tab"] if control != "noedit" else np.zeros_like(r["tab"])
      plugin("BT6DecAttn", [r["q"], kv, idx_name, add_init(tab, nm + "_tab")], [r["o"]], nm, mode=2,
             k_off=r["k_off"], v_off=r["v_off"], scale=scale)

  # ---- read-out. out_ln: Mul(Mul(Cast32(x), rsqrt), out_ln.scale) -> Cast16 = xo
  sc_name = [t for t in inits if t.endswith("move_tokens.out_ln.scale")]
  assert len(sc_name) == 1, sc_name
  mul_s = [n for n in g.node if n.op_type == "Mul" and sc_name[0] in n.input]
  assert len(mul_s) == 1
  mul_r = prod[[x for x in mul_s[0].input if x != sc_name[0]][0]]
  assert mul_r.op_type == "Mul"
  x32 = [x for x in mul_r.input if prod[x].op_type == "Cast"][0]
  x16 = prod[x32].input[0]
  rs = prod[[x for x in mul_r.input if x != x32][0]]
  assert rs.op_type == "Reciprocal"
  sq = prod[rs.input[0]]; ad = prod[sq.input[0]]
  assert sq.op_type == "Sqrt" and ad.op_type == "Add"
  eps = float(arr([x for x in ad.input if is_init(x)][0]))
  # policy output: Add(Gather(Reshape(Scatter), slot), mt_pol_bias)
  pol = prod["policy"]
  assert pol.op_type == "Add"
  gat = prod[[x for x in pol.input if not is_init(x)][0]]
  pbias = arr([x for x in pol.input if is_init(x)][0]).astype(np.float32)
  assert gat.op_type == "Gather" and pbias.shape == (1858,)
  slot = arr(gat.input[1]).astype(np.int32)
  scat = prod[prod[gat.input[0]].input[0]]
  assert scat.op_type == "ScatterElements", scat.op_type
  where = prod[scat.input[2]]
  assert where.op_type == "Where"
  tokmm = prod[where.input[1]]
  assert tokmm.op_type == "MatMul" and list(inits[tokmm.input[1]].dims) == [256, 4]
  Wp = arr(tokmm.input[1]).astype(np.float32).T                     # [4, 256]
  topk = [n for n in g.node if n.op_type == "TopK"]
  assert len(topk) == 1
  sel = topk[0].output[1]
  # pooled: the Concat feeding v_inject
  vinj = [n for n in g.node if n.op_type == "Gemm" and n.input[1].endswith("move_tokens.v_inject.weight")]
  assert len(vinj) == 1
  cat = prod[vinj[0].input[0]]
  assert cat.op_type == "Concat" and len(cat.input) == 2
  kbias = [r["kbias"] for r in att if r["kind"] == 0][0]
  if control == "headnomask":
    new_nodes.append(helper.make_node("Mul", [kbias, add_init(np.array(0, np.float16), "r4_head_zero")], ["r4_head_kb0"],
                                      name="r4_head_zero_kbias"))
    kbias = "r4_head_kb0"
  tab = np.concatenate([arr(sc_name[0]).astype(np.float32), Wp.ravel(), pbias]).astype(np.float32)
  assert tab.shape == (256 + 4 * 256 + 1858,)
  new_nodes.append(helper.make_node("Cast", [sel], ["r4_sel_i32"], to=TensorProto.INT32, name="r4_cast_sel"))
  pooled = cat.output[0]
  pol.output[0] = "policy_r4dead"
  cat.output[0] = pooled + "_r4dead"
  plugin("BT6DecHead", [x16, kbias, "r4_sel_i32", add_init(tab, "r4_head_tab"), add_init(slot, "r4_head_slot")],
         ["policy", pooled], "r4_dec_head", eps=float(np.float32(eps)))

  _finish(g, list(g.node) + new_nodes, new_inits)
  _plugin_opset(m)
  return (f"{len(att)} BT6DecAttn nodes, BT6DecHead (eps {eps:.4g}){' CONTROL ' + control if control else ''}: "
          f"{len(g.node)} nodes")


# ===================================================================================================== CLI

def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description="one BT6 plugin export step on files (make_plugin_onnx.py runs the chain)")
  sub = ap.add_subparsers(dest="step", required=True)
  p = sub.add_parser("params"); p.add_argument("net"); p.add_argument("out")
  for name in ("terms", "sites"):
    p = sub.add_parser(name); p.add_argument("onnx_in"); p.add_argument("params"); p.add_argument("onnx_out")
  p = sub.add_parser("merge"); p.add_argument("onnx_in"); p.add_argument("onnx_out")
  p.add_argument("--coef", action="store_true"); p.add_argument("--ffn", action="store_true")
  p = sub.add_parser("port_qdq"); p.add_argument("onnx_in"); p.add_argument("qdq"); p.add_argument("onnx_out")
  p.add_argument("--exclude", default="")
  p = sub.add_parser("glu"); p.add_argument("onnx_in"); p.add_argument("onnx_out")
  p.add_argument("--control", default="", choices=["", "noscale", "nobias"])
  p = sub.add_parser("i8out"); p.add_argument("onnx_in"); p.add_argument("onnx_out")
  p.add_argument("--control-scale", type=float, default=1.0)
  p = sub.add_parser("decoder"); p.add_argument("onnx_in"); p.add_argument("onnx_out")
  p.add_argument("--control", default="", choices=["", "nobias", "noedit", "kvoff", "headnomask"])
  a = ap.parse_args(argv)
  if a.step == "params":
    print(dump_params(a.net, a.out)); return 0
  m = onnx.load(a.onnx_in)
  if a.step in ("terms", "sites"):
    P = dict(np.load(a.params))
    msg = (terms_attention if a.step == "terms" else prologue_sites)(m, P)
  elif a.step == "merge":
    msg = merge_gemms(m, a.coef, a.ffn)
  elif a.step == "port_qdq":
    msg = port_qdq(m, onnx.load(a.qdq), a.exclude)
  elif a.step == "glu":
    msg = glu_quant(m, a.control)
  elif a.step == "i8out":
    msg = attn_int8_out(m, a.control_scale)
  else:
    msg = decoder(m, a.control)
  onnx.save(m, a.onnx_out)
  print(f"{msg} -> {a.onnx_out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
