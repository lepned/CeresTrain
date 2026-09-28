# Nested bottleneck trunk (nbt_layer.py).
#   python3 test_nbt_layer.py                       block-level tests
#   python3 test_nbt_layer.py <CFG_DIR> <CFG_ID>    + a real CeresNet built from an NBT config
import sys

import torch

from encoder_layer import EncoderLayer
from nbt_layer import NestedBottleneckLayer


def _make(hidden, i):
  return EncoderLayer('T', 64, 64, 4, hidden, 2 * hidden, True, 100, False, 4,
                      ffn_activation_type='SwiGLU', norm_type='RMSNorm', layerNum=i)


def _block(num_inner=2, seed=0):
  torch.manual_seed(seed)
  return NestedBottleneckLayer(128, 64, lambda j: _make(64, j), num_inner, 'RMSNorm', 1e-6)


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
  assert net.trunk_end_norm is not None, 'an NBT trunk needs the trunk-end norm'
  assert all(float(b.up.weight.abs().max()) == 0.0 for b in net.transformer_layer)
  n_fold, _ = fold_gate_on_warm_start(net)   # walks .inner; 0 when the gate is off
  print(f'  real net OK: {n_blocks} blocks x {k} inner at {cfg.NetDef_NBTMidDim}, layerNum 0..{n_blocks * k - 1}, '
        f'numLayers {cfg.NetDef_NumLayers * k}, gate fold visited ({n_fold} gated)')


if __name__ == '__main__':
  test_identity_at_init_and_grad()
  test_piece_relation_bias_reaches_inner_layers()
  test_film_on_branch_output()
  if len(sys.argv) >= 3:
    real(sys.argv[1], sys.argv[2])
  print('test_nbt_layer: OK')
