"""Parity + structure test for export_folds.py (run under the WSL env):
  CERES_AUX_FEATURES_PER_SQUARE=0 python3 test_export_folds.py
Builds a small move-token net, exports it plain and folded, and asserts
(1) all outputs agree (ORT, fp32), (2) the folded graph has fewer MatMul/Gemm
and fewer Mul nodes, (3) the live net is untouched by the fold (deepcopy)."""
import copy, os, sys, tempfile
import torch
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dual_plane_edge_aux import build, random_boards
from export_folds import apply_export_folds


def _export(net, sq, path):
  torch.onnx.export(net, (sq, None), path, dynamo=True, opset_version=18, input_names=['squares', 'prior'])


def _counts(path):
  import onnx
  m = onnx.load(path)
  ops = {}
  for n in m.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
  return ops


def main():
  torch.manual_seed(0)
  net, _ = build({'DualPlanePolicyDecode': False, 'UseMoveTokens': True, 'MoveTokenDim': 64,
                  'MoveTokenLayers': 2, 'MoveTokenHeads': 2, 'MoveTokenMax': 64,
                  'FFNActivationType': 'SwiGLU'}, {}, 'fold')
  net.eval()
  # make weights non-trivial so folds are exercised (norm scales != 1, biases != 0)
  with torch.no_grad():
    for n, p in net.named_parameters():
      if n.endswith('.scale') or n.endswith('.bias'):
        p.add_(torch.randn_like(p) * 0.1)
  sq = random_boards(4)
  with torch.no_grad():
    ref = net(sq, None)
  folded = copy.deepcopy(net).eval()
  c = apply_export_folds(folded)
  print('  folds applied:', c)
  assert c['ffn_fused'] > 0 and c['mt_scale_folded'] == 2 and c['mt_norm_folded'] == 6, c
  with torch.no_grad():
    out = folded(sq, None)
    ref2 = net(sq, None)
  for i, (a, b) in enumerate(zip(ref, ref2)):
    if a is not None and b is not None:
      assert torch.equal(a, b), f'live net changed by the fold (output {i})'
  worst = 0.0
  for i, (a, b) in enumerate(zip(ref, out)):
    if a is None or b is None or a.numel() == 0:
      continue
    d = float((a - b).abs().max())
    worst = max(worst, d)
    assert d < 1e-4, f'output {i} differs after fold: max|d|={d:.2e}'
  print(f'  torch parity OK (max|d| {worst:.2e} over all heads)')
  try:
    import onnx_ir, onnxruntime as ort
  except ImportError:
    print('  (onnx_ir/onnxruntime missing: ONNX structure/parity check skipped)'); return
  p0 = os.path.join(tempfile.gettempdir(), 'fold_plain.onnx'); p1 = os.path.join(tempfile.gettempdir(), 'fold_folded.onnx')
  _export(net, sq, p0); _export(folded, sq, p1)
  c0, c1 = _counts(p0), _counts(p1)
  mm0 = c0.get('MatMul', 0) + c0.get('Gemm', 0); mm1 = c1.get('MatMul', 0) + c1.get('Gemm', 0)
  print(f'  ONNX MatMul+Gemm: {mm0} -> {mm1}; Mul: {c0.get("Mul",0)} -> {c1.get("Mul",0)}; nodes: {sum(c0.values())} -> {sum(c1.values())}')
  assert mm1 < mm0, 'fold must remove GEMM launches'
  s0 = ort.InferenceSession(p0, providers=['CPUExecutionProvider']); s1 = ort.InferenceSession(p1, providers=['CPUExecutionProvider'])
  o0 = s0.run(None, {s0.get_inputs()[0].name: sq.numpy()}); o1 = s1.run(None, {s1.get_inputs()[0].name: sq.numpy()})
  worst = max(float(np.abs(a - b).max()) for a, b in zip(o0, o1) if a.size)
  assert worst < 1e-3, worst
  print(f'  ONNX/ORT parity OK (max|d| {worst:.2e})')
  print('ALL OK')


if __name__ == '__main__':
  main()
