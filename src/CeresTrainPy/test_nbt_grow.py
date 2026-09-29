"""Contract test for growing an NBT trunk at resume (nbt_grow.py, config TrunkGrowInsertAfter, 2026-09-29).

    python test_nbt_grow.py   (from src/CeresTrainPy, CPU)

1. A 3-block NBT net (every weight perturbed, up projections non-zero) grown to 5 blocks: only the fresh blocks'
   keys are missing, outputs are bit-identical, and perturbing a fresh up projection makes them differ.
2. A wrong block count and the identity-breaking configs (Affine proj norm, Diff++) are refused.
"""
import os, sys
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dual_plane_edge_aux import build, random_boards
from nbt_grow import grow_trunk_state_dict

NBT = {'NBTInnerLayers': 2, 'NBTWidthDivisor': 2, 'NBTInnerPreNorm': True, 'NBTProjActivation': 'Swish'}
RESUME = {'CheckpointResumeFromFileName': 'unused_by_this_test'}


def _outputs(net, sq):
  net.eval()
  with torch.no_grad():
    o = net(sq, None)
  return [t.float() for t in o if torch.is_tensor(t) and t.is_floating_point()]


def main():
  small, _ = build(dict(NBT, NumLayers=3), {}, 'nbtg3')
  with torch.no_grad():
    for p in small.parameters():
      p.add_(torch.randn_like(p) * 0.02)
  big, cfg = build(dict(NBT, NumLayers=5), dict(RESUME, TrunkGrowInsertAfter=[-1, 1]), 'nbtg5')
  sd, o2n, fresh = grow_trunk_state_dict(small.state_dict(), len(big.transformer_layer), cfg.Opt_TrunkGrowInsertAfter)
  assert o2n == {0: 1, 1: 2, 2: 4} and fresh == [0, 3], (o2n, fresh)
  res = big.load_state_dict(sd, strict=False)
  pref = tuple(f'transformer_layer.{i}.' for i in fresh)
  assert not res.unexpected_keys and res.missing_keys and all(k.startswith(pref) for k in res.missing_keys), res
  sq = random_boards(8)
  a, b = _outputs(small, sq), _outputs(big, sq)
  assert all(torch.equal(x, y) for x, y in zip(a, b)), [float((x - y).abs().max()) for x, y in zip(a, b)]
  with torch.no_grad():
    big.transformer_layer[fresh[0]].up.weight.normal_(0, 0.05)
  assert any(not torch.equal(x, y) for x, y in zip(a, _outputs(big, sq))), 'perturbed fresh block changed nothing'
  print('OK identity: 3 -> 5 blocks bit-identical, fresh', fresh)

  try:
    grow_trunk_state_dict(small.state_dict(), 6, [1, 1])
    raise AssertionError('wrong block count accepted')
  except ValueError:
    pass
  for net_over in ({'NBTProjNorm': 'Affine'}, {'UseDiffAttention': 3}):
    try:
      build(dict(NBT, NumLayers=5, **net_over), dict(RESUME, TrunkGrowInsertAfter=[-1, 1]), 'nbtgbad')
      raise AssertionError(f'{net_over} accepted')
    except ValueError:
      pass
  print('OK guards: wrong count, Affine, Diff++ refused')


if __name__ == '__main__':
  main()
