"""Round 4: the move-token decoder's attentions and its policy / pool read-out as TensorRT plugins, on a BT6 plugin
ONNX of round 2 (FP16 final, h768_r2_sitem.onnx) or round 3 (INT8 final, h768_r3_int8.onnx). The decoder region is
the same in both (FP16).

    python bt6_trt_plugin_dec_r4.py <in.onnx> <out.onnx> [--no-attn] [--no-head] [--blocks 0-3] [--control ...]

Per decoder block (MoveTokenBlock with post_move) the exported graph has three attentions between TensorRT GEMMs:
  self   Split(qkv GEMM) -> Reshape/Transpose -> q k^T * s + key bias -> Softmax -> @ v -> Transpose/Reshape -> proj
  cross  xq GEMM, k / v = the block's columns of the shared square K|V GEMM [B, 64, 4096] (Split 8 x 512 -> Split 2)
  pm     pm_q GEMM, k / v from the same K|V GEMM, + the one-hot to / from edits with GatherND(pm_dk / pm_dv, mover)
Each becomes one BT6DecAttn node (mode 0 / 1 / 2) whose output is the tensor the projection GEMM reads; the reshapes,
transposes, the two batched GEMMs and the Myelin softmax kernels go. The post-move indices (from, to, mover; int64 in the
graph) are cast once to one INT32 [B, 80, 3] tensor shared by the four pm nodes.

--head (default on): out_ln + pol (256 -> 4) + where(valid) + scatter + index_select + per-move bias -> `policy`, and the
masked mean / max pools -> the v_inject input, as one BT6DecHead node.

Every pattern is found from the graph's structure (Softmax nodes in topological order, then up / down through the
reshapes) and asserted, so a graph that differs fails loudly instead of being rewritten wrongly.

Controls (must fail the gate):
  --control nobias    self attention without the key bias (the -1e4 mask over invalid tokens)
  --control noedit    post-move attention with zero dk / dv tables (the post-move edits dropped)
  --control kvoff     cross attention of every block reads block 0's K|V columns (wrong block offset)
  --control headnomask BT6DecHead with an all-valid key bias (pools and policy over invalid tokens too)
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from bt6_trt_plugin_smooth_r3 import parse_layers, toposort_and_prune


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("onnx_in"); ap.add_argument("onnx_out")
  ap.add_argument("--no-attn", action="store_true"); ap.add_argument("--no-head", action="store_true")
  ap.add_argument("--blocks", default="0-3")
  ap.add_argument("--modes", default="0,1,2", help="attention modes to replace (0 self, 1 cross, 2 pm)")
  ap.add_argument("--control", default="", choices=["", "nobias", "noedit", "kvoff", "headnomask"])
  a = ap.parse_args(argv)
  blocks = parse_layers(a.blocks)
  modes = {int(x) for x in a.modes.split(",") if x != ""}
  m = onnx.load(a.onnx_in)
  g = m.graph
  inits = {t.name: t for t in g.initializer}
  arr = lambda name: numpy_helper.to_array(inits[name])
  prod = {o: n for n in g.node for o in n.output}
  cons = defaultdict(list)
  for n in g.node:
    for i in n.input:
      cons[i].append(n)
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
    n = helper.make_node(op, ins, outs, name=name, domain="trt.plugins")
    n.attribute.extend([helper.make_attribute("plugin_version", "1"), helper.make_attribute("plugin_namespace", "")])
    for k, v in attrs.items():
      n.attribute.append(helper.make_attribute(k, v))
    new_nodes.append(n)
    return n

  # ---- the shared square K|V GEMM: Split into 8 x 512 (xkv 0..3 | pm_kv 0..3)
  kvsplit = [n for n in g.node if n.op_type == "Split" and len(n.output) == 8]
  assert len(kvsplit) == 1, len(kvsplit)
  kvsplit = kvsplit[0]
  sizes = arr(kvsplit.input[1]).tolist()
  assert sizes == [512] * 8, sizes
  kv = kvsplit.input[0]
  kv_part = {o: j for j, o in enumerate(kvsplit.output)}

  # ---- body softmaxes (the ones feeding a MatMul), in topological order: per block self, cross, pm
  sms = [n for n in g.node if n.op_type == "Softmax" and any(c.op_type in ("MatMul", "Mul") for c in cons[n.output[0]])
         and any(c.op_type == "MatMul" for c in cons[n.output[0]])]
  assert len(sms) == 12, len(sms)
  scale = float(np.float32(32 ** -0.5))
  att = []
  for si, sm in enumerate(sms):
    blk, kind = si // 3, si % 3
    # score MatMul: walk up through Mul / Add (non-initializer operands)
    chain, frontier, mm = set(), [sm.input[0]], None
    while frontier and mm is None:
      t = frontier.pop(0)
      p = prod[t]
      if p.op_type == "MatMul" and not any(is_init(x) for x in p.input):
        mm = p; break
      if p.op_type not in ("Mul", "Add"):
        continue
      chain.add(p.name)
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
      # additive key bias: Add(scaled scores, Reshape(bias [B, 80]))
      adds = [prod[x] for x in sm.input if prod[x].op_type == "Add"]
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
          out, fr = [], [(s, 0) for s in start]
          vis = set()
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
          # the Mul consuming this one-hot (through Cast / Unsqueeze)
          t, mulnode = e.output[0], None
          for _ in range(4):
            cs = [c for c in cons[t] if c.name in scn]
            if not cs:
              break
            c = cs[0]
            if c.op_type == "Mul":
              mulnode = c; break
            t = c.output[0]
          assert mulnode is not None, (sm.name, e.name)
          other = [x for x in mulnode.input if x != t][0]
          sub = cone([other], 6)
          roles["to" if any(p.op_type == "GatherND" for p in sub) else "from"] = idx
        assert set(roles) == {"to", "from"}, (sm.name, roles)
        rec["to"], rec["from"] = roles["to"], roles["from"]
    att.append(rec)

  # ---- post-move index tensor (shared): Cast(from, to, mover) -> INT32 [B, 80, 3]
  pms = [r for r in att if r["kind"] == 2]
  for r in pms[1:]:
    assert (r["from"], r["to"], r["mover"]) == (pms[0]["from"], pms[0]["to"], pms[0]["mover"])
  idx_name = None
  if not a.no_attn and 2 in modes:
    parts = []
    for nm, t in (("from", pms[0]["from"]), ("to", pms[0]["to"]), ("mover", pms[0]["mover"])):
      new_nodes.append(helper.make_node("Cast", [t], [f"r4_{nm}_i32"], to=TensorProto.INT32, name=f"r4_cast_{nm}"))
      new_nodes.append(helper.make_node("Unsqueeze", [f"r4_{nm}_i32", add_init(np.array([-1], np.int64), f"r4_ax_{nm}")],
                                        [f"r4_{nm}_u"], name=f"r4_unsq_{nm}"))
      parts.append(f"r4_{nm}_u")
    idx_name = "r4_pm_idx"
    new_nodes.append(helper.make_node("Concat", parts, [idx_name], axis=-1, name="r4_pm_idx_cat"))

  done = []
  if not a.no_attn:
    for r in att:
      if r["block"] not in blocks or r["kind"] not in modes:
        continue
      nm = f"r4_dec{r['block']}_{['self', 'cross', 'pm'][r['kind']]}"
      dead = prod[r["o"]]
      dead.output[0] = r["o"] + "_r4dead"   # the old producer's output is renamed (pruned below)
      prod.pop(r["o"])
      if r["kind"] == 0:
        kb = r["kbias"]
        if a.control == "nobias":
          kb = nm + "_zero_kbias"
          new_nodes.append(helper.make_node("Mul", [r["kbias"], add_init(np.array(0, np.float16), nm + "_zero")], [kb],
                                            name=nm + "_zero_kbias_mul"))
        plugin("BT6DecAttn", [r["qkv"], kb], [r["o"]], nm, mode=0, k_off=256, v_off=512, scale=scale)
      elif r["kind"] == 1:
        ko, vo = r["k_off"], r["v_off"]
        if a.control == "kvoff":
          ko, vo = 0, 256
        plugin("BT6DecAttn", [r["q"], kv], [r["o"]], nm, mode=1, k_off=ko, v_off=vo, scale=scale)
      else:
        tab = r["tab"] if a.control != "noedit" else np.zeros_like(r["tab"])
        plugin("BT6DecAttn", [r["q"], kv, idx_name, add_init(tab, nm + "_tab")], [r["o"]], nm, mode=2,
               k_off=r["k_off"], v_off=r["v_off"], scale=scale)
      done.append(nm)

  head = ""
  if not a.no_head:
    # out_ln: Mul(Mul(Cast32(x), rsqrt), out_ln.scale) -> Cast16 = xo
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
    if a.control == "headnomask":
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
    head = f", BT6DecHead (eps {eps:.4g})"

  g.node.extend(new_nodes)
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
  print(f"{len(done)} BT6DecAttn nodes{head}{' CONTROL ' + a.control if a.control else ''}: {len(g.node)} nodes -> {a.onnx_out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
