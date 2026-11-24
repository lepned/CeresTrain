"""Round 3: the BT6TermsAttention plugin writes the INT8 W_h input itself (plugin version 3, out_int8 = 1).

    python bt6_trt_plugin_i8out_r3.py <in.onnx> <out.onnx> [--control-scale K]

In an INT8 plugin graph (bt6_trt_plugin_port_qdq.py) every attention output feeds only QuantizeLinear(scale s) ->
DequantizeLinear(s) -> W_h MatMul. Per such layer the plugin node becomes version 3 with out_int8 = 1, out_scale = s
(FP32 attribute), its output is the INT8 codes, and the QuantizeLinear goes: the DequantizeLinear reads the plugin
output directly. Saves the separate quantize kernel and halves the plugin's output bytes.
--control-scale K: CONTROL (must fail the gate): the plugin quantizes with K * s while the DequantizeLinear keeps s.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main(argv=None) -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("onnx_in"); ap.add_argument("onnx_out")
  ap.add_argument("--control-scale", type=float, default=1.0)
  a = ap.parse_args(argv)
  m = onnx.load(a.onnx_in)
  g = m.graph
  inits = {t.name: t for t in g.initializer}
  cons = defaultdict(list)
  for n in g.node:
    for i in n.input:
      cons[i].append(n)
  dead, done = set(), 0
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
                        helper.make_attribute("out_scale", float(np.float32(sq * a.control_scale)))])
    dq.input[0] = n.output[0]
    dead.add(q.name)
    done += 1
  keep = [n for n in g.node if n.name not in dead]
  del g.node[:]
  g.node.extend(keep)
  del g.value_info[:]
  onnx.save(m, a.onnx_out)
  print(f"{done} attention plugins write INT8 (v3){' CONTROL scale x%g' % a.control_scale if a.control_scale != 1 else ''} -> {a.onnx_out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
