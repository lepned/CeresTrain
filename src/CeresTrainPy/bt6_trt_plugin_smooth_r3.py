"""Round 3: INT8 FFN down (1536 -> 768) with SmoothQuant-style smoothing on a BT6 TensorRT-plugin INT8 ONNX in which
FFN down is still FP16 (bt6_trt_plugin_port_qdq.py --exclude 1536x768, then bt6_trt_plugin_merge.py --coef --ffn).

    python bt6_trt_plugin_smooth_r3.py <in.onnx> <out.onnx> --calib A.npz [--alpha 0.5] [--emit-s s.npz]
                                       [--act B.npz --pct 99.9999] [--layers 0-15] [--no-smooth]

Per FFN layer l (FFN down = the l-th MatMul with a [1536, 768] initializer weight; its input is m = SiLU(gate) * up):
  s_j   = amax(m_j)^alpha / amax(W_j,:)^(1 - alpha)      per input channel j (A.npz: <layer>/amax from calib_r3.py)
  graph m' = m * (1 / s)          one Mul by an FP16 constant, fused by TensorRT into the SiLU * up kernel
        Q -> DQ on m'             per tensor, symmetric; scale = (percentile of |m'| from B.npz, or max_j amax_j / s_j) / 127
        W' = diag(s) W            INT8 per output channel (axis 1, symmetric, max |W'_:,k| / 127), DequantizeLinear
        MatMul(DQ(m'), DQ(W'))    the bias Add and the residual stay as they are (FP16 output, no output Q/DQ)
m' @ W' = m @ W exactly in real arithmetic; only the quantization grid moves (outlier channels of m shrink, the matching
weight rows grow). --no-smooth: s = 1 (the naive INT8 FFN down, as the rejected round-2 variant 6b).
--emit-s writes <layer>/s (for the second calibration pass, calib_r3.py --smooth, which gives B.npz).
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def parse_layers(spec: str) -> set:
  out = set()
  for part in spec.split(","):
    if "-" in part:
      lo, hi = part.split("-"); out.update(range(int(lo), int(hi) + 1))
    elif part:
      out.add(int(part))
  return out


def toposort_and_prune(g, extra_inits):
  nodes = list(g.node)
  produced = {o: n for n in nodes for o in n.output}
  inits = {t.name for t in g.initializer} | {t.name for t in extra_inits}
  avail = {i.name for i in g.input} | inits | {""}
  ordered, placed = [], set()
  for n0 in nodes:
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
  allinit = [t for t in list(g.initializer) + list(extra_inits) if t.name in used]
  del g.initializer[:]
  g.initializer.extend(allinit)
  del g.value_info[:]


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("onnx_in"); ap.add_argument("onnx_out")
  ap.add_argument("--calib", required=True); ap.add_argument("--alpha", type=float, default=0.5)
  ap.add_argument("--emit-s", default=""); ap.add_argument("--act", default=""); ap.add_argument("--pct", type=float, default=99.9999)
  ap.add_argument("--layers", default="0-15"); ap.add_argument("--no-smooth", action="store_true")
  ap.add_argument("--per-token", action="store_true",
                  help="dynamic per-row activation scale: r = max_j |m'_j| per token, Q/DQ of m' / r with scale 1/127, "
                       "GEMM, output * r (all graph ops; the act percentile is not used)")
  ap.add_argument("--clip", type=float, default=1.0, help="--per-token: the row scale is clip * row max (<1 clips)")
  a = ap.parse_args(argv)
  layers = parse_layers(a.layers)
  cal = np.load(a.calib)
  act = np.load(a.act) if a.act else None
  m = onnx.load(a.onnx_in)
  g = m.graph
  inits = {t.name: t for t in g.initializer}
  arr = lambda name: numpy_helper.to_array(inits[name])
  downs = [n for n in g.node if n.op_type == "MatMul" and n.input[1] in inits and list(inits[n.input[1]].dims) == [1536, 768]]
  assert len(downs) == 16, len(downs)
  new_inits, new_nodes, emit = [], [], {}

  def add_init(array, name):
    t = numpy_helper.from_array(array, name); new_inits.append(t); return name

  report = []
  for l, mm in enumerate(downs):
    W = arr(mm.input[1]).astype(np.float32)                 # [1536 in, 768 out]
    ax = np.maximum(cal[f"layer{l}/amax"].astype(np.float32), 1e-6)
    aw = np.maximum(np.abs(W).max(axis=1), 1e-8)
    if a.no_smooth:
      s = np.ones(1536, np.float32)
    else:
      s = (ax ** a.alpha) / (aw ** (1.0 - a.alpha))
      s = np.clip(s, 1e-4, 1e4).astype(np.float32)
    s16 = s.astype(np.float16).astype(np.float32)         # the graph multiplies by the FP16 1/s; fold the same s
    inv16 = (1.0 / s16).astype(np.float16)
    s_eff = 1.0 / inv16.astype(np.float32)
    emit[f"layer{l}/s"] = s_eff
    if l not in layers:
      continue
    if act is not None:
      top = act[f"layer{l}/top"].astype(np.float64); n = int(act[f"layer{l}/n"][0])
      rank = int(np.floor((1.0 - a.pct / 100.0) * n))           # number of values above the percentile
      assert rank < top.size, (l, rank, top.size)
      amax_t = float(top[-1 - rank])
    else:
      amax_t = float((ax / s_eff).max())
    a_scale = np.float16(amax_t / 127.0)
    Wq = s_eff[:, None] * W
    w_scale = np.maximum(np.abs(Wq).max(axis=0), 1e-8) / 127.0
    w_scale16 = w_scale.astype(np.float16)
    q = np.clip(np.round(Wq / w_scale16.astype(np.float32)), -127, 127).astype(np.int8)
    L = f"r3d{l}"
    x = mm.input[0]
    xs = x
    if not a.no_smooth:
      xs = f"{L}_ms"
      new_nodes.append(helper.make_node("Mul", [x, add_init(inv16, f"{L}_inv_s")], [xs], name=f"{L}_smooth"))
    if a.per_token:
      ab, r0, r = f"{L}_abs", f"{L}_rmax", f"{L}_r"
      new_nodes.append(helper.make_node("Abs", [xs], [ab], name=f"{L}_Abs"))
      new_nodes.append(helper.make_node("ReduceMax", [ab, add_init(np.array([-1], np.int64), f"{L}_ax")], [r0],
                                        keepdims=1, name=f"{L}_RowMax"))
      new_nodes.append(helper.make_node("Max", [r0, add_init(np.array(1e-3, np.float16), f"{L}_eps")], [f"{L}_r1"], name=f"{L}_Eps"))
      if a.clip != 1.0:
        new_nodes.append(helper.make_node("Mul", [f"{L}_r1", add_init(np.array(a.clip, np.float16), f"{L}_clip")], [r], name=f"{L}_Clip"))
      else:
        r = f"{L}_r1"
      new_nodes.append(helper.make_node("Div", [xs, r], [f"{L}_xn"], name=f"{L}_RowNorm"))
      xs = f"{L}_xn"
      a_scale = np.float16(1.0 / 127.0)
      y = mm.output[0]
      mm.output[0] = f"{L}_yn"
      new_nodes.append(helper.make_node("Mul", [f"{L}_yn", r], [y], name=f"{L}_RowScale"))
      amax_t = float("nan")
    sc = add_init(np.array(a_scale, np.float16), f"{L}_a_scale")
    zp = add_init(np.array(0, np.int8), f"{L}_a_zp")
    new_nodes.append(helper.make_node("QuantizeLinear", [xs, sc, zp], [f"{L}_q8"], name=f"{L}_Q8_in"))
    new_nodes.append(helper.make_node("DequantizeLinear", [f"{L}_q8", sc, zp], [f"{L}_dq8"], name=f"{L}_DQ8_in"))
    new_nodes.append(helper.make_node("DequantizeLinear", [add_init(q, f"{L}_wq"), add_init(w_scale16, f"{L}_ws"),
                                                           add_init(np.zeros(768, np.int8), f"{L}_wz")],
                                      [f"{L}_w"], name=f"{L}_WDQ8", axis=1))
    mm.input[0] = f"{L}_dq8"
    mm.input[1] = f"{L}_w"
    # weight quantization error relative to the FP32 weight (in the original basis)
    werr = np.abs(q.astype(np.float32) * w_scale16.astype(np.float32)[None, :] / s_eff[:, None] - W).mean() / np.abs(W).mean()
    report.append(f"L{l}: s {s_eff.min():.2g}..{s_eff.max():.2g}, act amax {amax_t:.2f} (raw {ax.max():.0f}), w rel err {werr:.4f}")
  if a.emit_s:
    np.savez(a.emit_s, **emit)
    print("wrote", a.emit_s)
  g.node.extend(new_nodes)
  toposort_and_prune(g, new_inits)
  onnx.save(m, a.onnx_out)
  print("\n".join(report))
  print(f"{len(report)} FFN down GEMMs INT8 ({'no smoothing' if a.no_smooth else f'alpha {a.alpha}'}, "
        f"act {'per token, clip %g' % a.clip if a.per_token else 'pct %g' % a.pct if act is not None else 'amax'}) -> {a.onnx_out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
