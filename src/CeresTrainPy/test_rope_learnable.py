# Learnable 2D RoPE (rope.LearnableRope2D, config RoPELearnable).
#   python3 test_rope_learnable.py                      unit tests
#   python3 test_rope_learnable.py <CFG_DIR> <CFG_ID>   + a real CeresNet built from a config with RoPELearnable
import math
import os
import sys
import tempfile

import torch

from encoder_layer import EncoderLayer
from rope import LearnableRope2D, apply_rope, bake_learnable_rope, precompute_rope_freqs
from wd_partition import partition_weight_decay


def _layer(heads=4, d=64, learnable=True, diff=0, layer_num=0, **rope_kw):
  return EncoderLayer('T', 64, 64, 4, d, 2 * d, True, 100, False, heads, ffn_activation_type='SwiGLU',
                      norm_type='RMSNorm', layerNum=layer_num, use_rope=True, rope_learnable=learnable,
                      use_diff_attention=diff, **rope_kw)


def test_init_paired_with_fixed_rope():
  # The frequencies come from a private generator: every OTHER weight must be bit-identical to a
  # fixed-RoPE layer built under the same global seed (else a learnable arm is not init-paired).
  torch.manual_seed(7); fixed = _layer(learnable=False)
  torch.manual_seed(7); learn = _layer(learnable=True)
  fixed_sd = fixed.state_dict()
  for n, p in learn.state_dict().items():
    if 'rope' in n: continue
    assert torch.equal(p, fixed_sd[n]), f'{n} differs: learnable RoPE consumed the global RNG'
  # Different layers get different frequencies; the same layer is reproducible.
  a, b, a2 = _layer(layer_num=0), _layer(layer_num=1), _layer(layer_num=0)
  assert not torch.equal(a.attention.rope.freqs, b.attention.rope.freqs)
  assert torch.equal(a.attention.rope.freqs, a2.attention.rope.freqs)


def test_stratified_init():
  r = LearnableRope2D(4, 32, init='stratified', freq_min=1 / 16, freq_max=1.5)
  mag = r.freqs.norm(dim=-1)                                  # (H, P)
  assert torch.allclose(mag[0], mag[1]), 'same magnitude bank in every head'
  assert abs(float(mag[0, 0]) - 1.5) < 1e-5 and abs(float(mag[0, -1]) - 1 / 16) < 1e-6, 'geometric max -> min'
  ratios = mag[0, 1:] / mag[0, :-1]
  assert torch.allclose(ratios, ratios[0], atol=1e-5), 'geometric spacing'
  theta = torch.atan2(r.freqs[..., 1], r.freqs[..., 0])
  assert float((theta > 0).float().mean()) > 0.3 and float((theta < 0).float().mean()) > 0.3, 'directions cover the circle'
  # Both inits are offset-only rotations, so the relative-position property holds for either.


def test_diff_attention_combination():
  # Learnable RoPE is allowed with the single-Q-tensor Diff modes (3 = half-dim); fixed RoPE is not.
  torch.manual_seed(3)
  layer = _layer(learnable=True, diff=3)
  y = layer(torch.randn(2, 64, 64))
  assert y.shape == (2, 64, 64) and torch.isfinite(y).all()
  refused = False
  try:
    _layer(learnable=False, diff=3)
  except AssertionError:
    refused = True
  assert refused, 'fixed RoPE + Diff must stay refused'


def test_bake_and_wide_export():
  # Above ModelDim 256 the exporter does not fold Cos/Sin (8192-element limit); bake() must leave
  # only constant tables in the graph, and baked == unbaked numerically.
  import copy
  import onnx
  torch.manual_seed(4)
  layer = _layer(heads=16, d=1024).eval()
  x = torch.randn(1, 64, 1024)
  baked = copy.deepcopy(layer)
  assert bake_learnable_rope(baked) == 1 and baked.attention.rope._baked
  with torch.no_grad():
    ref, out = layer(x), baked(x)
  assert torch.equal(ref, out), 'baking must not change the function'
  assert 'attention.rope.baked_cos' not in baked.state_dict(), 'baked tables are non-persistent'
  fn = os.path.join(tempfile.mkdtemp(), 'rope_layer_1024.onnx')
  torch.onnx.export(baked, (x,), fn, opset_version=18, do_constant_folding=True, export_params=True,
                    input_names=['x'], output_names=['y'])
  ops = {n.op_type for n in onnx.load(fn).graph.node}
  assert 'Cos' not in ops and 'Sin' not in ops, f'tables not baked: {sorted(ops)}'
  import onnxruntime as ort
  sess = ort.InferenceSession(fn, providers=['CPUExecutionProvider'])
  got = torch.from_numpy(sess.run(None, {'x': x.numpy()})[0])
  assert torch.allclose(got, ref, atol=1e-4), float((got - ref).abs().max())


def test_init_and_shapes():
  torch.manual_seed(0)
  r = LearnableRope2D(4, 32)
  assert r.freqs.shape == (4, 16, 2)
  assert r.init == 'stratified', 'default init'
  mag = r.freqs.norm(dim=-1)
  assert abs(float(mag.max()) - math.pi / 2) < 1e-5 and abs(float(mag.min()) - 1 / 16) < 1e-6, 'default range [1/16, pi/2] rad/square'
  assert float((r.freqs < 0).float().mean()) > 0.2, 'random directions'
  k = LearnableRope2D(4, 32, init='loguniform', freq_min=1 / 50, freq_max=1.0)
  a = k.freqs.abs()
  assert float(a.min()) >= 1 / 50 - 1e-6 and float(a.max()) <= 1.0 + 1e-6, 'KataGo init range [1/50, 1]'
  cos, sin = r()
  assert cos.shape == (4, 64, 32) and sin.shape == (4, 64, 32)
  assert torch.allclose(cos[:, :, 0::2], cos[:, :, 1::2]), 'pairs interleaved (cos repeated per pair)'
  assert torch.allclose(cos ** 2 + sin ** 2, torch.ones_like(cos), atol=1e-6)


def test_relative_position_property():
  # q.k after rotation depends only on the board offset (dfile, drank), for any (omega_x, omega_y).
  torch.manual_seed(1)
  r = LearnableRope2D(3, 16)
  cos, sin = r()
  q = torch.randn(1, 3, 64, 16); k = torch.randn(1, 3, 64, 16)
  # Use the SAME q/k vector on every square so only the rotation differs.
  q = q[:, :, :1].expand(-1, -1, 64, -1).contiguous(); k = k[:, :, :1].expand(-1, -1, 64, -1).contiguous()
  qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
  s = torch.einsum('bhqd,bhkd->bhqk', qr, kr)[0]           # (H, 64, 64)
  # squares a=(f,r) and b: compare (a, b) with (a+shift, b+shift) for on-board shifts.
  def sq(f, rk): return rk * 8 + f
  for (fa, ra, fb, rb, df, dr) in [(1, 1, 3, 2, 2, 3), (0, 0, 7, 7, 0, 0), (2, 5, 6, 1, 1, -1), (4, 4, 1, 6, -1, 1)]:
    a, b = s[:, sq(fa, ra), sq(fb, rb)], s[:, sq(fa + df, ra + dr), sq(fb + df, rb + dr)]
    assert torch.allclose(a, b, atol=1e-4), f'not offset-only: {a} vs {b}'
  # And it is NOT translation-free (position does matter): different offsets differ.
  assert not torch.allclose(s[:, sq(1, 1), sq(3, 2)], s[:, sq(1, 1), sq(2, 3)], atol=1e-3)


def test_fixed_rope_is_special_case():
  # Fixed 2D RoPE = learnable one with (omega_x, 0) on the first half of the pairs and (0, omega_y) on the second.
  d = 16
  cos_f, sin_f = precompute_rope_freqs(d, base=1000.0)
  r = LearnableRope2D(1, d)
  freqs = 1.0 / (1000.0 ** (torch.arange(0, d // 2, 2).float() / (d // 2)))   # (d/4,)
  with torch.no_grad():
    r.freqs.zero_()
    r.freqs[0, :d // 4, 0] = freqs      # file pairs
    r.freqs[0, d // 4:, 1] = freqs      # rank pairs
  cos_l, sin_l = r()
  assert torch.allclose(cos_l[0], cos_f, atol=1e-6) and torch.allclose(sin_l[0], sin_f, atol=1e-6)


def test_layer_grad_partition_and_export():
  torch.manual_seed(2)
  layer = _layer()
  x = torch.randn(2, 64, 64)
  layer(x).square().sum().backward()
  g = layer.attention.rope.freqs.grad
  assert g is not None and float(g.abs().sum()) > 0, 'frequencies must receive gradient'
  assert layer.attention.rope.freqs.ndim == 3, '3-D => AdamW under every Muon scope (ndim != 2)'
  # weight-decay partition: the frequencies are rates, not a weight matrix -> no_decay.
  m = torch.nn.Module(); m.transformer_layer = torch.nn.Sequential(layer)
  decay, no_decay = partition_weight_decay(m)
  assert 'transformer_layer.0.attention.rope.freqs' in no_decay, 'rope.freqs must be in no_decay'
  assert not any(n.endswith('rope.freqs') for n in decay)
  # The fixed variant keeps zero RoPE parameters and its non-persistent buffers.
  fixed = _layer(learnable=False)
  assert not any('rope' in n for n, _ in fixed.named_parameters())
  assert 'attention.rope_cos' not in fixed.state_dict(), 'fixed tables stay non-persistent'
  # ONNX export the way save_model does it (dynamo exporter, opset 18, constant folding): the
  # tables are Cos/Sin of the parameter and fold to constants; ORT must reproduce the layer.
  layer.eval()
  fn = os.path.join(tempfile.mkdtemp(), 'rope_layer.onnx')
  torch.onnx.export(layer, (x,), fn, opset_version=18, do_constant_folding=True, export_params=True,
                    input_names=['x'], output_names=['y'])
  import onnxruntime as ort
  sess = ort.InferenceSession(fn, providers=['CPUExecutionProvider'])
  with torch.no_grad():
    ref = layer(x)
  out = torch.from_numpy(sess.run(None, {'x': x.numpy()})[0])
  assert torch.allclose(out, ref, atol=1e-4), float((out - ref).abs().max())


def real(cfg_dir, cfg_id):
  from config import Configuration
  from ceres_net import CeresNet
  cfg = Configuration(cfg_dir, cfg_id)
  assert cfg.NetDef_UseRoPE and cfg.NetDef_RoPELearnable
  net = CeresNet(None, cfg, policy_loss_weight=1, value_loss_weight=1, moves_left_loss_weight=0, unc_loss_weight=0,
                 value2_loss_weight=0, q_deviation_loss_weight=0, value_diff_loss_weight=0, value2_diff_loss_weight=0,
                 action_loss_weight=0, uncertainty_policy_weight=0, action_uncertainty_loss_weight=0, q_ratio=1)
  ropes = [(n, p) for n, p in net.named_parameters() if n.endswith('rope.freqs')]
  n_att = sum(1 for n, _ in net.named_modules() if n.startswith('transformer_layer') and n.endswith('.attention'))
  assert len(ropes) == n_att, f'{len(ropes)} rope tables for {n_att} trunk attentions'
  decay, no_decay = partition_weight_decay(net)
  assert all(n in no_decay for n, _ in ropes)
  print(f'  real net OK: {len(ropes)} learnable RoPE tables of shape {tuple(ropes[0][1].shape)} '
        f'({sum(p.numel() for _, p in ropes):,} params), all in no_decay')


if __name__ == '__main__':
  test_init_and_shapes()
  test_relative_position_property()
  test_fixed_rope_is_special_case()
  test_layer_grad_partition_and_export()
  test_init_paired_with_fixed_rope()
  test_stratified_init()
  test_diff_attention_combination()
  test_bake_and_wide_export()
  if len(sys.argv) >= 3:
    real(sys.argv[1], sys.argv[2])
  print('test_rope_learnable: OK')
