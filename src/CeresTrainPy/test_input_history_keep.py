"""InputHistoryKeep (config.py / ceres_net.py): the net sees only `keep` real history boards. Small nets, CPU.

  (a) keep k in {0, 1}, the big run's layout with templates: net(x) == the same weights without the key fed the
      reduced input (boards k+1..7 := board k, repetition flags 104+k+1..111 := 0); boards > k and their flags do
      not change the output; board 1 does change it at keep 1; state_dict keys identical to the net without the key
  (b) loss finite, gradients live; fp16 cast keeps the index buffer integral
  (c) ONNX (opset 18, fp32) == PyTorch in onnxruntime, reduction inside the graph
  (d) refusals

  python test_input_history_keep.py
"""
import copy
import os
import tempfile

import numpy as np
import torch

from test_bt6_hybrid import batch_for, build, records, run_loss
from test_bt6_hybrid_decoder import OPT_DEC
from test_attn_templates import BIG, live

OVER = {**BIG, 'AttnTemplates': 8}


def full_history(count, seed=7):
  """Records whose 8 history boards are all different real boards, with random repetition flags."""
  sq = records(count).clone()
  g = torch.Generator().manual_seed(seed)
  for h in range(2, 8):
    sq[:, :, h * 13:(h + 1) * 13] = torch.roll(sq[:, :, 0:13], shifts=h, dims=0)
  sq[:, :, 104:112] = (torch.rand(count, 64, 8, generator=g) < 0.3).to(sq.dtype)
  return sq


def reduce(x, keep):
  y = x.clone()
  for h in range(keep + 1, 8):
    y[:, :, h * 13:(h + 1) * 13] = x[:, :, keep * 13:keep * 13 + 13]
  y[:, :, 104 + keep + 1:112] = 0
  return y


def pair(keep):
  net = build({**OVER, 'InputHistoryKeep': keep}, OPT_DEC)
  live(net)
  ref = build(OVER, OPT_DEC)
  assert set(net.state_dict()) == set(ref.state_dict()), 'InputHistoryKeep must not change the checkpoint keys'
  ref.load_state_dict(net.state_dict(), strict=True)
  net.eval(); ref.eval()
  return net, ref


def test_equivalence(keep):
  net, ref = pair(keep)
  sq = full_history(8)
  with torch.no_grad():
    a = net(sq, None)
    b = ref(reduce(sq, keep), None)
    c = ref(sq, None)
    other = sq.clone()
    for h in range(keep + 1, 8):
      other[:, :, h * 13:(h + 1) * 13] = torch.roll(sq[:, :, 0:13], shifts=h + 3, dims=0)
    other[:, :, 104 + keep + 1:112] = 1 - other[:, :, 104 + keep + 1:112]
    d = net(other, None)
  for i in (0, 1, 4):
    assert torch.equal(a[i], b[i]), (keep, i, float((a[i] - b[i]).abs().max()))
    assert torch.equal(a[i], d[i]), (keep, i)
  assert float((a[1] - c[1]).abs().max()) > 1e-5, 'history beyond keep must have mattered to the full-history net'
  if keep == 1:
    with torch.no_grad():
      e = ref(reduce(sq, 0), None)
    assert float((a[1] - e[1]).abs().max()) > 1e-5, 'board 1 must reach the net at keep 1'
  print(f'(a) keep {keep}: == reference on the reduced input (bitwise), boards {keep + 1}..7 and their flags ignored'
        + (', board 1 used' if keep == 1 else '') + '; checkpoint keys unchanged')
  return net


def test_train_and_cast(net):
  net.train()
  sq = full_history(8)
  loss = run_loss(net, batch_for(8), sq)
  assert torch.isfinite(loss)
  loss.backward()
  missing = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
  assert not missing, missing[:10]
  half = copy.deepcopy(net).half()
  assert half.hist_idx.dtype == torch.long and half.hist_mask.dtype == torch.float16
  print('(b) loss finite, all gradients present; .half() keeps hist_idx long')


def test_onnx(net, keep):
  import onnx
  import onnxruntime as ort
  net.eval()

  class SquaresOnly(torch.nn.Module):
    def __init__(self, inner):
      super().__init__(); self.inner = inner
    def forward(self, squares):
      out = self.inner(squares, None)
      return out[0], out[1], out[4]

  sq = full_history(8)
  with torch.no_grad():
    want = SquaresOnly(net)(sq)
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'net.onnx')
    torch.onnx.export(SquaresOnly(net), (sq[:2],), path, do_constant_folding=True, export_params=True, opset_version=18,
                      input_names=['squares'], output_names=['policy', 'value', 'value2'],
                      dynamic_axes={n: {0: 'batch_size'} for n in ('squares', 'policy', 'value', 'value2')})
    m = onnx.load(path)
  assert any(n.op_type == 'Gather' and 'squares' in n.input for n in m.graph.node), 'history gather not at the graph input'
  so = ort.SessionOptions(); so.log_severity_level = 3
  got = ort.InferenceSession(m.SerializeToString(), so, providers=['CPUExecutionProvider']).run(
    None, {'squares': sq.numpy().astype(np.float32)})
  for name, ya, yb in zip(('policy', 'value', 'value2'), got, want):
    ya, yb = ya.astype(np.float64), yb.numpy().astype(np.float64)
    if name == 'policy':
      k = yb > -1e3
      assert (k == (ya > -1e3)).all()
      ya, yb = ya[k], yb[k]
    err = np.abs(ya - yb).max() / (np.abs(yb).max() + 1e-12)
    assert err < 1e-4, (name, err)
  print(f'(c) keep {keep}: ONNX == PyTorch in onnxruntime, Gather on the squares input')


def test_refusals():
  for v in (-2, 7):
    try:
      build({**OVER, 'InputHistoryKeep': v}, OPT_DEC)
    except ValueError as e:
      assert 'InputHistoryKeep' in str(e)
    else:
      raise AssertionError(f'InputHistoryKeep {v} was not refused')
  print('(d) refusals OK')


if __name__ == '__main__':
  for keep in (0, 1):
    net = test_equivalence(keep)
    test_onnx(net, keep)
  test_train_and_cast(net)
  test_refusals()
  print('test_input_history_keep: all OK')
