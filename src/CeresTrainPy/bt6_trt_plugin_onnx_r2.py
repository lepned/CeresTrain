"""Round 2 of the BT6 TensorRT-plugin rewrite: applied to the OUTPUT of bt6_trt_plugin_onnx.py (round 1, one
BT6TermsAttention v1 node per term layer).

    python bt6_trt_plugin_onnx_r2.py <round1.onnx> <params.npz> <out.onnx> [--no-prologue] [--no-export-h]
                                     [--no-state-cm] [--site]

Steps (each optional, all on by default):
  prologue   the whole prologue (record -> piece boards -> 34 relation channels E -> S, e0, and round 1's INT32 bit
             packing of E) becomes ONE BT6Prologue node: squares + tab FP32 [8288] (D | Pin | T | Toff, FP32 trainer
             values from params.npz) -> (edges INT32 [B,64,64,2], S FP16 [B,64,64], e0 FP16 [B,64,64,16] cell-major).
             e0 cell-major is both the first state input of the attention nodes and edge site 1's channels-last
             input (the exported Transpose [0, 2, 3, 1] of e0).
  export_h   the attention node of every layer whose raw logits an edge site reads (q.k / sqrt(hd) + template bias,
             the exported graph's Add feeding the site's readback) gets BT6TermsAttention v2 with export_h = 1 and
             its second output replaces that Add; the QK-norm / q.k / template-bias nodes of the layer die.
  state_cm   the edge state after a site stays channels-last: the attention nodes and the next site (and the
             decoder's edge input) read the site's channels-last result directly instead of Transpose -> Reshape ->
             Transpose copies.
  site       (--site, off by default) every edge site (readback, rms, edge FFN with the reverse edge, rms) becomes
             ONE BT6EdgeSite node on the channels-last state and the plugin's raw logits; tab FP32 [3520] from the
             site's (FP16) initializers.
All attention nodes become version 2 (export_h 0 / 1). Dead nodes and initializers are pruned.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DOMAIN = "trt.plugins"


def scope(node) -> str:
  for p in node.metadata_props:
    if p.key == "pkg.torch.onnx.name_scopes":
      return p.value
  return ""


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("onnx_in"); ap.add_argument("params"); ap.add_argument("onnx_out")
  ap.add_argument("--no-prologue", action="store_true")
  ap.add_argument("--no-export-h", action="store_true")
  ap.add_argument("--no-state-cm", action="store_true")
  ap.add_argument("--site", action="store_true", help="edge sites as BT6EdgeSite plugin nodes (needs every other step)")
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
  attn = sorted([n for n in nodes if n.op_type == "BT6TermsAttention"], key=lambda n: int(n.name.rsplit("_", 1)[1]))
  assert len(attn) == 16, len(attn)
  new_nodes, new_inits = [], []

  def replace_uses(old, new, only=None):
    for c in list(consumers[old]):
      if only is not None and c not in only:
        continue
      for k, i in enumerate(c.input):
        if i == old:
          c.input[k] = new
      consumers[new].append(c)
    consumers[old] = [c for c in consumers[old] if only is not None and c not in only]

  # ---------------------------------------------------------------- edge-state tensors per state version
  # state input k of the attention nodes = Transpose(Reshape(Reshape(e_k))); e_k [B, 16, 64, 64] channels-first
  state_inputs = sorted({n.input[4] for n in attn}, key=lambda s: int(s.rsplit("_", 1)[1]))
  e_first = []   # channels-first e_k
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

  # ---------------------------------------------------------------- prologue
  if not a.no_prologue:
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
    # e0's channels-last consumers (site 1's Transpose [0, 2, 3, 1]) and the state-0 input read the plugin's e0
    e0 = e_first[0]
    for c in list(consumers[e0]):
      if c.op_type == "Transpose" and list(c.attribute[0].ints) == [0, 2, 3, 1]:
        replace_uses(c.output[0], "bt6q_e0")
    for n in attn:
      if n.input[4] == state_inputs[0]:
        n.input[4] = "bt6q_e0"

  # ---------------------------------------------------------------- channels-last state after the sites
  if not a.no_state_cm:
    for k in range(1, len(e_first)):
      ef, el = e_first[k], e_last[k]
      for n in attn:
        if n.input[4] == state_inputs[k]:
          n.input[4] = el
      for c in list(consumers[ef]):
        if c.op_type == "Transpose" and list(c.attribute[0].ints) == [0, 2, 3, 1]:
          replace_uses(c.output[0], el)

  # ---------------------------------------------------------------- raw logits from the plugin (site layers)
  exported = []
  for L, n in enumerate(attn):
    export = 0
    if not a.no_export_h:
      s = f"'inner.transformer_layer.{L}'"
      adds = [x for x in nodes if x.op_type == "Add" and s in scope(x)
              and any("bt6h.sites" in scope(c) for c in consumers[x.output[0]])]
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

  # ---------------------------------------------------------------- edge sites as BT6EdgeSite nodes
  sites_done = []
  if a.site:
    assert not a.no_export_h and not a.no_state_cm and not a.no_prologue, "--site needs every other step"
    inits = {t.name: t for t in g.initializer}
    W = lambda name: numpy_helper.to_array(inits[name]).astype(np.float64)
    cur = defaultdict(list)
    for x in nodes:
      for i in x.input:
        cur[i].append(x)
    def sc_of(x):
      parts = scope(x).split(", ")
      return parts[-2] if len(parts) >= 2 else ""
    site_scopes = sorted({sc_of(x) for x in nodes if sc_of(x).startswith("'inner.bt6h.sites.")})
    for k, sc in enumerate(site_scopes, start=1):
      sn = [x for x in nodes if sc_of(x) == sc]
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
      # the following site's input / later readers were rewired to `out` by state_cm; e_last now names the plugin's
      e_last[k] = f"bt6s_state_{k}"
      sites_done.append(sc)

  # ---------------------------------------------------------------- order + prune (as round 1)
  all_nodes = nodes + new_nodes
  produced = {o: n for n in all_nodes for o in n.output}
  avail = {i.name for i in g.input} | {t.name for t in g.initializer} | {t.name for t in new_inits} | {""}
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
  removed = len(all_nodes) - len(ordered)
  del g.node[:]
  g.node.extend(ordered)
  used = {i for n in ordered for i in n.input}
  inits = [t for t in list(g.initializer) + new_inits if t.name in used]
  del g.initializer[:]
  g.initializer.extend(inits)
  del g.value_info[:]
  onnx.save(m, a.onnx_out)
  print(f"prologue {'on' if not a.no_prologue else 'off'}, raw-logit export on layers {exported}, "
        f"state_cm {'on' if not a.no_state_cm else 'off'}, sites {len(sites_done)}: {removed} nodes pruned, {len(ordered)} nodes -> {a.onnx_out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
