"""BT6EdgeValues: the edge message in the BT6 attention core (bt6_net.attention_core edge=..., BT6NativeTerms.Ue).

    python test_bt6_edge_values.py

(a) the memory-saving autograd function equals plain autograd on the same statements (outputs and every gradient,
    float64, with and without the output gate, with and without a gradient on H);
(b) the edge message is what it says: m[b, h, i, c] = sum_j A[b, h, i, j] e[b, i, j, c];
(c) BT6NativeTerms with Ue = 0 is bit-for-bit the layer without BT6EdgeValues, and Ue gets a gradient;
(d) flat-trunk CeresNet (test_bt6_flat's nets, decoder with edge input): with BT6EdgeValues == without at init; live Ue
    changes the function; every parameter gets a gradient; Ue is a reader (no weight decay, out of Muon);
(e) ONNX export == PyTorch in onnxruntime, with and without the TensorRT gate fix; torch.compile == eager.
"""

from __future__ import annotations

import torch

from bt6_net import BT6Config, RelationContext, attention_core, _core_forward


def _rand(*shape, requires_grad=True):
  return torch.randn(*shape, dtype=torch.float64, requires_grad=requires_grad)


def test_core_matches_plain_autograd():
  torch.manual_seed(0)
  B, Hh, hd, C = 2, 3, 8, 5
  for with_gate in (False, True):
    for want_H in (False, True):
      L, door, gl = _rand(B, Hh, 64, 64), (torch.randn(B, Hh, 64, 64, dtype=torch.float64) * 0.1).requires_grad_(), _rand(B, Hh, 64, 64)
      v, e = _rand(B, Hh, 64, hd), _rand(B, 64, 64, C)
      ol = _rand(B, Hh, 64, hd) if with_gate else None
      ins = [L, door, gl, v, e] + ([ol] if with_gate else [])
      for t in ins:
        t.retain_grad()
      w_out, w_H, w_m = torch.randn(B, Hh, 64, hd, dtype=torch.float64), torch.randn(B, Hh, 64, 64, dtype=torch.float64), \
        torch.randn(B, Hh, 64, C, dtype=torch.float64)
      grads = []
      for custom in (True, False):
        for t in ins:
          t.grad = None
        out, H, m = attention_core(L, door, gl, v, False, ol, 2.0, custom=custom, edge=e)
        loss = (out * w_out).sum() + (m * w_m).sum() + ((H * w_H).sum() if want_H else 0.0)
        loss.backward()
        grads.append(([out.detach(), H.detach(), m.detach()], [t.grad.clone() for t in ins]))
      for a, b in zip(grads[0][0] + grads[0][1], grads[1][0] + grads[1][1]):
        assert torch.allclose(a, b, atol=1e-10, rtol=1e-9), (with_gate, want_H, (a - b).abs().max())
  print("(a) custom autograd == plain autograd (out, H, m and all gradients)")


def test_message_definition():
  torch.manual_seed(1)
  B, Hh, hd, C = 2, 3, 8, 5
  L, door, gl = (torch.randn(B, Hh, 64, 64, dtype=torch.float64) for _ in range(3))
  v, e = torch.randn(B, Hh, 64, hd, dtype=torch.float64), torch.randn(B, 64, 64, C, dtype=torch.float64)
  _, H, gate, _, m = _core_forward(L, door, gl, v, False, None, 2.0, False, e)
  A = torch.softmax(H, -1) * gate
  ref = torch.einsum("bhij,bijc->bhic", A, e)
  assert torch.allclose(m, ref, atol=1e-12), (m - ref).abs().max()
  print("(b) m = sum_j A[., i, j] e[., i, j, :]")


def test_native_terms_identity():
  from bt6_native_terms import BT6NativeTerms
  torch.manual_seed(2)
  cfg = BT6Config(heads=4, d_model=64)
  base = BT6NativeTerms(cfg).double()
  ev = BT6NativeTerms(cfg).double()
  ev.load_state_dict(base.state_dict())
  ev.enable_edge_values()
  ev.double()
  with torch.no_grad():                                  # non-identity readers, so the test is not trivial
    for mod in (base, ev):
      torch.manual_seed(3)
      for p in (mod.C1, mod.Wm, mod.Wg, mod.Pq, mod.Pk):
        p.copy_(torch.randn_like(p) * 0.1)
  B, hd = 2, cfg.head_dim
  Q, K, V = (torch.randn(B, cfg.heads, 64, hd, dtype=torch.float64) for _ in range(3))
  E = (torch.rand(B, 34, 64, 64) < 0.05).float()
  S = torch.rand(B, 64, 64, dtype=torch.float64) + 0.5
  Z0 = torch.randn(B, cfg.pair_channels, 64, 64, dtype=torch.float64)
  e = torch.randn(B, cfg.edge_channels, 64, 64, dtype=torch.float64, requires_grad=True)
  ctx = RelationContext(E.double(), S, Z0)
  rs = ctx.read_state(e, torch.float64)
  o1, H1 = base.attend(Q, K, V, None, ctx, e, rs)
  o2, H2 = ev.attend(Q, K, V, None, ctx, e, rs)
  assert torch.equal(o1, o2) and torch.equal(H1, H2), ((o1 - o2).abs().max(), (H1 - H2).abs().max())
  o2.square().sum().backward()
  assert ev.Ue.grad is not None and ev.Ue.grad.abs().max() > 0
  print("(c) Ue = 0 is bit-for-bit the plain term layer; Ue gets a gradient")



def test_flat_net():
  import test_bt6_flat as F
  from test_bt6_hybrid import batch_for, build, randomise, records, run_loss
  from bt6_hybrid import bt6_muon_excluded, is_bt6_layer_reader
  from wd_partition import partition_weight_decay
  over = {**F.FLAT, **F.DECODER}
  net = build({**over, 'BT6EdgeValues': True}, F.OPT_DEC)
  plain = build(over, F.OPT_DEC)
  ue = [n for n, _ in net.named_parameters() if n.endswith('.bt6t.Ue')]
  assert len(ue) == len(net.bt6_terms_layers), ue
  randomise(plain); F.live_templates(plain)
  missing, unexpected = net.load_state_dict(plain.state_dict(), strict=False)
  assert sorted(missing) == sorted(ue) and not unexpected, (missing, unexpected)
  net.eval(); plain.eval()
  sq = records(8)
  with torch.no_grad():
    a, b = net(sq, None), plain(sq, None)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    g = torch.Generator().manual_seed(5)
    for n, p in net.named_parameters():
      if n.endswith('.bt6t.Ue'):
        p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    c = net(sq, None)
  dv = float((c[1] - b[1]).abs().max())
  assert dv > 1e-4, dv
  net.train()
  loss = run_loss(net, batch_for(8), sq)
  loss.backward()
  nograd = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]
  assert not nograd and torch.isfinite(loss), nograd[:10]
  assert all(float(dict(net.named_parameters())[n].grad.abs().max()) > 0 for n in ue)
  decay, no_decay = partition_weight_decay(net)
  assert all(is_bt6_layer_reader(n) and bt6_muon_excluded(n) and n in no_decay and n not in decay for n in ue)
  print(f'(d) flat net: identity at init (exact), live Ue changes value by {dv:.1e}, loss {float(loss):.4f}, '
        f'all parameters have gradients, {len(ue)} Ue no-decay / out of Muon')
  return net


def test_onnx_and_compile(net):
  import copy, os, tempfile
  import onnx
  import onnxruntime as ort
  from test_bt6_hybrid import batch_for, records, run_loss
  from trt_gate_fix import apply_gate_fix, find_gated_attentions
  net.eval()
  sq = records(4).float()
  with torch.no_grad():
    ref = net(sq, None)

  class Wrap(torch.nn.Module):
    def __init__(s, m):
      super().__init__(); s.m = m
    def forward(s, x):
      o = s.m(x, None)
      return o[0], o[1]
  path = os.path.join(tempfile.mkdtemp(prefix='bt6ev_'), 'ev.onnx')
  torch.onnx.export(Wrap(net), (sq[:2],), path, do_constant_folding=True, export_params=True,
                    opset_version=int(os.environ.get('CERES_ONNX_OPSET', '18')), input_names=['squares'],
                    output_names=['policy', 'value'], dynamic_axes={n: {0: 'batch_size'} for n in ('squares', 'policy', 'value')})
  m = onnx.load(path)
  m2 = copy.deepcopy(m)
  k = len(find_gated_attentions(m.graph))
  fixed = apply_gate_fix(m2)
  for tag, mm in (('plain', m), ('gate fix', m2)):
    sess = ort.InferenceSession(mm.SerializeToString(), providers=['CPUExecutionProvider'])
    pol, val = sess.run(None, {'squares': sq.numpy()})
    dp = float((torch.from_numpy(pol) - ref[0]).abs().max()); dv = float((torch.from_numpy(val) - ref[1]).abs().max())
    assert dp < 1e-3 and dv < 1e-4, (tag, dp, dv)
  print(f'(e) ONNX == PyTorch (policy {dp:.1e}, value {dv:.1e}); gate fix: {k} gated attentions found, {fixed} rewritten, same outputs')
  net.train()
  eager = run_loss(net, batch_for(4), sq)
  cnet = torch.compile(net)
  loss = run_loss(cnet, batch_for(4), sq)
  loss.backward()
  assert torch.isfinite(loss) and abs(float(loss) - float(eager)) < 1e-4, (float(loss), float(eager))
  print(f'(e) torch.compile forward + backward: loss {float(loss):.5f} == eager {float(eager):.5f}')


def test_core_m_only_and_cap():
  """Gradient through m alone (d_out None in backward) and with the gate cap, custom == plain autograd."""
  torch.manual_seed(4)
  B, Hh, hd, C = 2, 3, 8, 5
  for cap in (False, True):
    L, gl = _rand(B, Hh, 64, 64), _rand(B, Hh, 64, 64)
    door = (torch.randn(B, Hh, 64, 64, dtype=torch.float64) * 0.1).requires_grad_()
    v, e = _rand(B, Hh, 64, hd), _rand(B, 64, 64, C)
    w = torch.randn(B, Hh, 64, C, dtype=torch.float64)
    res = []
    for custom in (True, False):
      for t in (L, door, gl, v, e):
        t.grad = None
      out, H, m = attention_core(L, door, gl, v, cap, None, 2.0, custom=custom, edge=e)
      (m * w).sum().backward()
      res.append([t.grad.clone() if t.grad is not None else torch.zeros_like(t) for t in (L, door, gl, e)] + [v.grad])
    for a, b in zip(res[0][:4], res[1][:4]):
      assert torch.allclose(a, b, atol=1e-10, rtol=1e-9), (cap, (a - b).abs().max())
    assert res[0][4] is None or float(res[0][4].abs().max()) == 0.0
  print("(a2) loss on m alone and with the gate cap: custom == plain autograd")

if __name__ == "__main__":
  test_core_matches_plain_autograd()
  test_core_m_only_and_cap()
  test_message_definition()
  test_native_terms_identity()
  torch.set_num_threads(4)
  test_onnx_and_compile(test_flat_net())
  print("ALL OK")
