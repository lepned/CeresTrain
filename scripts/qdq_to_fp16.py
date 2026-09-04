#!/usr/bin/env python3
"""
Post-pass for qdq_export.py outputs: convert the FP32-internal QDQ graph to FP16
internals (opset 19, fp16 Q/DQ scales, fp16 IO kept) with NVIDIA ModelOpt's
autocast converter. Without this every NON-quantized op (excluded MatMuls, Gemm
heads, norms, softmax, SwiGLU) runs in FP32 inside the strongly-typed TRT engine:
measured 09-04 on prod 640x12: same accuracy, 4-6 % lower latency; and excluded
MatMuls become free instead of 25-30 % slower.

Run with the ModelOpt venv (NOT the training venv):
  /home/lep/modelopt-env/bin/python scripts/qdq_to_fp16.py <net>.qdq.fp16io.onnx [<out>.onnx]
Recipe that produced the best 500M read (int8f16_dec):
  qdq_export.py ... --precision int8 --method percentile --calib_batches 8 \
      --exclude_regex 'node_(linear_1(2[4-9]|[3-5][0-9]|6[0-3])|MatMul_1664)' --no_verify --out X.int8_dec.onnx
  qdq_to_fp16.py X.int8_dec.onnx X.int8f16_dec.onnx
(the regex = the move-token decoder + heads of the dynamo export; check node names per net)
"""
import sys, os
import onnx
from modelopt.onnx.autocast.convert import convert_to_f16

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + '.f16.onnx'
m = convert_to_f16(onnx.load(src), low_precision_type='fp16', keep_io_types=True)
onnx.save(m, dst)
g = m.graph
print(f'[qdq_to_fp16] {src} -> {dst}: opset {[(o.domain, o.version) for o in m.opset_import]}, '
      f'fp16 inits {sum(1 for t in g.initializer if t.data_type == 10)}, '
      f'fp32 inits {sum(1 for t in g.initializer if t.data_type == 1)}, '
      f'Cast nodes {sum(1 for n in g.node if n.op_type == "Cast")}')
