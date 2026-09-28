# Nested bottleneck trunk (nbt_layer.py).
#   python3 test_nbt_layer.py                       block-level tests
#   python3 test_nbt_layer.py <CFG_DIR> <CFG_ID>    + a real CeresNet built from an NBT config
import sys

import torch

from encoder_layer import EncoderLayer
from nbt_layer import NestedBottleneckLayer


def _make(hidden, i, heads=4, pre_norm=False):
  return EncoderLayer('T', 64, 64, 4, hidden, 2 * hidden, True, 100, False, heads,
                      ffn_activation_type='SwiGLU', norm_type='RMSNorm', layerNum=i, pre_norm=pre_norm)


def _block(num_inner=2, seed=0, heads=4, pre_norm=False, proj_activation='None'):
  torch.manual_seed(seed)
  return NestedBottleneckLayer(128, 64, lambda j: _make(64, j, heads, pre_norm), num_inner, 'RMSNorm', 1e-6,
                               proj_activation=proj_activation)


def test_katago_style_options():
  # 2 heads at 64 (dim 32), pre-norm inner layers, Swish (= SiLU) before both projections.
  blk = _block(2, heads=2, pre_norm=True, proj_activation='Swish')
  assert all(l.num_attention_heads == 2 and l.pre_norm for l in blk.inner)
  assert not isinstance(blk.act_in, torch.nn.Identity) and not isinstance(blk.act_out, torch.nn.Identity)
  x = torch.randn(3, 64, 128)
  assert torch.equal(blk(x), x), 'still an exact identity at init'
  with torch.no_grad():
    blk.up.weight.normal_(0, 0.05)
    h = blk.down(torch.nn.functional.silu(blk.norm_in(x)))
    for l in blk.inner:
      h = l(h)
    ref = x + blk.up(torch.nn.functional.silu(blk.norm_out(h)))
    assert torch.allclose(blk(x), ref, atol=1e-6), 'activation must sit between norm and projection'
  # Every projection-side parameter must be on the gradient path once up is nonzero.
  blk(x).square().sum().backward()
  for name in ('down.weight', 'norm_in.scale', 'norm_out.scale'):
    g = dict(blk.named_parameters())[name].grad
    assert g is not None and float(g.abs().sum()) > 0, f'{name} gets no gradient'
  # A per-head bias built for another head count is refused, not broadcast.
  rejected = None
  try:
    blk(x, piece_relation_bias=torch.zeros(3, 4, 64, 64))
  except AssertionError as e:
    rejected = str(e)
  assert rejected is not None, 'head-count mismatch must be rejected'
  assert 'NBTInnerHeads' in rejected, rejected


def test_affine_proj_norm_and_fixup_init():
  from rms_norm import ChannelAffine, NORM_MODULE_TYPES
  torch.manual_seed(2)
  ref = NestedBottleneckLayer(128, 64, lambda j: _make(64, j), 2, 'RMSNorm', 1e-6)
  torch.manual_seed(2)
  blk = NestedBottleneckLayer(128, 64, lambda j: _make(64, j), 2, 'RMSNorm', 1e-6,
                              proj_norm='Affine', down_init_scale=0.5, block_index=3, proj_activation='Swish')
  assert isinstance(blk.norm_in, ChannelAffine) and isinstance(blk.norm_out, ChannelAffine)
  assert ChannelAffine in NORM_MODULE_TYPES, 'wd_partition must see the affine gains as norm gains'
  # KataGo fixscale constants: 1/sqrt(block+1) before down, 1/sqrt(K+1) before up.
  assert abs(blk.norm_in.fixed_scale - 0.5) < 1e-9 and abs(blk.norm_out.fixed_scale - 3 ** -0.5) < 1e-9
  # Same RNG stream: the down projection is the reference one scaled by 0.5.
  assert torch.allclose(blk.down.weight, ref.down.weight * 0.5)
  x = torch.randn(2, 64, 128)
  assert torch.equal(blk(x), x), 'still an exact identity at init'
  silu = torch.nn.functional.silu
  with torch.no_grad():
    blk.up.weight.normal_(0, 0.05)
    blk.norm_in.scale.fill_(2.0); blk.norm_in.bias.fill_(0.1)
    h = blk.down(silu(x * (2.0 * 0.5) + 0.1))
    for l in blk.inner:
      h = l(h)
    ref_out = x + blk.up(silu(h * (blk.norm_out.scale * 3 ** -0.5) + blk.norm_out.bias))
  assert torch.allclose(blk(x), ref_out, atol=1e-6), 'affine must be const*gamma*x+beta with no normalization'


def test_identity_at_init_and_grad():
  blk = _block(3)
  x = torch.randn(5, 64, 128, requires_grad=True)
  y = blk(x)
  assert y.shape == x.shape
  assert torch.equal(y, x), 'zero-init up projection must make the block an exact identity'
  assert len(blk.inner) == 3 and all(l.attention.d_model == 64 for l in blk.inner)
  with torch.no_grad():
    blk.up.weight.normal_(0, 0.02)
  blk(x).square().sum().backward()
  g = blk.inner[0].attention.qkv.weight.grad
  assert g is not None and float(g.abs().sum()) > 0, 'inner layers must be on the gradient path'


def test_piece_relation_bias_reaches_inner_layers():
  blk = _block()
  with torch.no_grad():
    blk.up.weight.normal_(0, 0.05)
  blk.eval()
  x = torch.randn(2, 64, 128)
  bias = torch.randn(2, 4, 64, 64) * 3.0
  with torch.no_grad():
    y0 = blk(x)
    y1 = blk(x, piece_relation_bias=bias)
    # The same bias fed to the inner layers by hand must reproduce the block exactly.
    h = blk.down(blk.norm_in(x))
    for l in blk.inner:
      h = l(h, piece_relation_bias=bias)
    y_ref = x + blk.up(blk.norm_out(h))
  assert not torch.allclose(y0, y1), 'piece_relation_bias must change the block output'
  assert torch.allclose(y1, y_ref, atol=1e-6), 'bias must go to every inner layer'


def test_film_on_branch_output():
  blk = _block()
  with torch.no_grad():
    blk.up.weight.normal_(0, 0.05)
  blk.eval()
  x = torch.randn(2, 64, 128)
  gamma = torch.randn(2, 1, 128) * 0.1
  beta = torch.randn(2, 1, 128) * 0.1
  with torch.no_grad():
    branch = blk(x) - x
    y = blk(x, film=(gamma, beta))
  assert torch.allclose(y, x + branch * (1.0 + gamma) + beta, atol=1e-6)


def real(cfg_dir, cfg_id):
  from config import Configuration
  from ceres_net import CeresNet
  from gate_fold import fold_gate_on_warm_start
  cfg = Configuration(cfg_dir, cfg_id)
  k, n_blocks = cfg.NetDef_NBTInnerLayers, cfg.NetDef_NumLayers // cfg.NetDef_LoopCount
  assert k > 0, 'config must enable NBTInnerLayers'
  net = CeresNet(None, cfg, policy_loss_weight=1, value_loss_weight=1, moves_left_loss_weight=0, unc_loss_weight=0,
                 value2_loss_weight=0, q_deviation_loss_weight=0, value_diff_loss_weight=0, value2_diff_loss_weight=0,
                 action_loss_weight=0, uncertainty_policy_weight=0, action_uncertainty_loss_weight=0, q_ratio=1)
  inner = [l for b in net.transformer_layer for l in b.inner]
  assert [l.layerNum for l in inner] == list(range(n_blocks * k)), [l.layerNum for l in inner]
  assert all(l.numLayers == cfg.NetDef_NumLayers * k for l in inner)
  assert all(l.attention.d_model == cfg.NetDef_NBTMidDim for l in inner)
  assert all(l.num_attention_heads == cfg.NetDef_NBTInnerHeads and l.pre_norm == cfg.NetDef_NBTInnerPreNorm for l in inner)
  assert net.trunk_end_norm is not None, 'an NBT trunk needs the trunk-end norm'
  assert all(float(b.up.weight.abs().max()) == 0.0 for b in net.transformer_layer)
  n_fold, _ = fold_gate_on_warm_start(net)   # walks .inner; 0 when the gate is off
  print(f'  real net OK: {n_blocks} blocks x {k} inner at {cfg.NetDef_NBTMidDim}, layerNum 0..{n_blocks * k - 1}, '
        f'numLayers {cfg.NetDef_NumLayers * k}, gate fold visited ({n_fold} gated)')
  _config_inheritance(cfg_dir, cfg_id)


def _config_inheritance(cfg_dir, cfg_id):
  # NBTInnerPreNorm absent => inherits the trunk PreNorm (review 2026-09-28 finding 1).
  import json, os, shutil
  from config import Configuration
  tag = 'zz_nbt_inherit'
  for p in ('data', 'exec', 'monitoring', 'opt'):
    shutil.copy(f'{cfg_dir}/{cfg_id}_ceres_{p}.json', f'{cfg_dir}/{tag}_ceres_{p}.json')
  net = json.load(open(f'{cfg_dir}/{cfg_id}_ceres_net.json'))
  net.pop('NBTInnerPreNorm', None)
  try:
    for pre in (False, True):
      net['PreNorm'] = pre
      json.dump(net, open(f'{cfg_dir}/{tag}_ceres_net.json', 'w'), indent=1)
      assert Configuration(cfg_dir, tag).NetDef_NBTInnerPreNorm == pre, f'must inherit PreNorm={pre}'
    net['PreNorm'] = True; net['NBTInnerPreNorm'] = False
    json.dump(net, open(f'{cfg_dir}/{tag}_ceres_net.json', 'w'), indent=1)
    assert Configuration(cfg_dir, tag).NetDef_NBTInnerPreNorm is False, 'explicit value must win'
  finally:
    for p in ('data', 'exec', 'monitoring', 'opt', 'net'):
      os.remove(f'{cfg_dir}/{tag}_ceres_{p}.json')
  print('  config OK: NBTInnerPreNorm inherits PreNorm when absent')


if __name__ == '__main__':
  test_identity_at_init_and_grad()
  test_piece_relation_bias_reaches_inner_layers()
  test_film_on_branch_output()
  test_katago_style_options()
  test_affine_proj_norm_and_fixup_init()
  if len(sys.argv) >= 3:
    real(sys.argv[1], sys.argv[2])
  print('test_nbt_layer: OK')
