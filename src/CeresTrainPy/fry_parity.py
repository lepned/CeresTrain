"""Fry (fry-final-117) in OUR trainer: the config that describes it, and the map from the released checkpoint's
tensors onto our parameters (2026-11-26). Used by test_fry_parity.py; not imported by the trainer.

The weights are GPL-3.0 material published by the net's author: load them locally for parity tests, do not
redistribute converted files. Nothing here is copied from the author's code: the map is written from the tensor
names and shapes of the checkpoint and from the description of the forward pass in FRY_BLOCKS_NOTES.md.
"""

from __future__ import annotations

import numpy as np
import torch

LAYERS = 28
QK_DIMS = [8] * 2 + [16] * 5 + [24] * 21          # per-head query / key width by layer

# The net: 768 wide, 28 layers, 24 query heads on 12 key/value heads, 70 tokens, 8 templates, SwiGLU hidden 992.
FRY_NET = {
  'ModelDim': 768, 'NumLayers': LAYERS, 'NumHeads': 24,
  'PreNorm': True, 'NormType': 'RMSNorm', 'NormEps': 1e-5,
  'UseQKV': True, 'UseQKNorm': True, 'AttentionMultiplier': 1,
  'FFNMultiplier': 992 / 768, 'FFNActivationType': 'SwiGLU', 'FFNNoBias': True,
  'HeadsActivationType': 'Mish', 'HeadWidthMultiplier': 4,
  'SmolgenDimPerSquare': 0, 'SmolgenDim': 0, 'SmolgenToHeadDivisor': 1, 'SmolgenActivationType': 'Swish',
  'UseRPE': False, 'UseRPE_V': False, 'UseRoPE': False, 'NonLinearAttention': False, 'SoftCapCutoff': 0,
  'NBTInnerLayers': 0,
  'AttnTemplates': 8, 'AttnTemplateScope': 'layer',
  'NBTInputSkip': True, 'InputSkipLayer0': True,
  'FryInput': True, 'SquareFeatures': True,
  'AttnKVHeads': 12, 'AttnQKDims': QK_DIMS, 'AttnValueDim': 32, 'AttnQKVBias': True,
  'PolicyHeadFry': True, 'PolicyHeadFryKingCastling': True,
  'ValueHeadFry': True, 'ValueHLGaussServe': True, 'ValueHLGaussServeDraw': 'none',
}
# The losses he trains (policy 1, HL-Gauss value 0.5, nothing else) and his optimizer split.
FRY_OPT = {
  'NumTrainingPositions': 4_000_000_000, 'BatchSizeForwardPass': 1026, 'BatchSizeBackwardPass': 4104,
  'Optimizer': 'Muon', 'MuonAdamWScope': 'fry', 'MuonMomentum': 0.95, 'Beta1': 0.9, 'Beta2': 0.999,
  'WeightDecayScheme': 'fry', 'WeightDecayFromSchedule': 0.17, 'WeightDecay': None,
  'LearningRateBase': 5e-4, 'GradientClipLevel': 1.0,
  'LossPolicyMultiplier': 1, 'LossValueMultiplier': 0, 'LossValue2Multiplier': 0, 'LossMLHMultiplier': 0,
  'LossUNCMultiplier': 0, 'LossQDeviationMultiplier': 0, 'LossUncertaintyPolicyMultiplier': 0,
  'LossValueDMultiplier': 0, 'LossValue2DMultiplier': 0, 'LossActionMultiplier': 0,
  'HLGaussWeight': 0.5, 'HLGaussBuckets': 128, 'HLGaussSigmaScale': 0.75,
  'TorchSeed': 777, 'TPGV3': 0, 'AuxFeaturesPerSquare': 0,
}


def name_map():
  """{checkpoint tensor name: (our parameter name, index or None)}; index selects one element of a per-layer vector."""
  m = {
    'embed.weight': ('fry_input.tok_embedding.weight', None),
    'position_embed': ('fry_input.pos_embedding', None),
    'square_features.table': ('square_features.table', None),
    'templates': ('attn_templates.table', None),
    'ln.weight': ('trunk_end_norm.scale', None),
    'from_projection.weight': ('fry_policy.from_proj.weight', None), 'from_projection.bias': ('fry_policy.from_proj.bias', None),
    'to_projection.weight': ('fry_policy.to_proj.weight', None), 'to_projection.bias': ('fry_policy.to_proj.bias', None),
    'promotion_projection.weight': ('fry_policy.promo_proj.weight', None),
    'promotion_projection.bias': ('fry_policy.promo_proj.bias', None),
    'board_down_project.0.weight': ('hlg_head.squeeze.weight', None), 'board_down_project.0.bias': ('hlg_head.squeeze.bias', None),
    'eval_head.0.weight': ('hlg_head.fc.weight', None), 'eval_head.0.bias': ('hlg_head.fc.bias', None),
    'eval_head.2.weight': ('hlg_head.out.weight', None), 'eval_head.2.bias': ('hlg_head.out.bias', None),
  }
  for i in range(LAYERS):
    e, t = f'encoder.{i}.', f'transformer_layer.{i}.'
    m[e + 'residual_scale'] = ('input_skip_a', i)
    m[e + 'input_scale'] = ('input_skip_b', i)
    for his, ours in (('Q', 'attention.q_proj'), ('K', 'attention.k_proj'), ('V', 'attention.v_proj'),
                      ('combine_heads', 'attention.W_h'), ('template_mix', 'tmix.mix')):
      m[e + his + '.weight'] = (t + ours + '.weight', None)
      m[e + his + '.bias'] = (t + ours + '.bias', None)
    m[e + 'q_norm.weight'] = (t + 'attention.qLN.scale', None)
    m[e + 'k_norm.weight'] = (t + 'attention.kLN.scale', None)
    m[e + 'ln1.weight'] = (t + 'ln1.scale', None)
    m[e + 'ln2.weight'] = (t + 'ln2.scale', None)
    # SwiGLU: out = down(SiLU(gate(x)) * up(x)); ours: linear2(SiLU(linear1(x)) * linear3(x))
    m[e + 'mlp.gate_projection.weight'] = (t + 'mlp.linear1.weight', None)
    m[e + 'mlp.up_projection.weight'] = (t + 'mlp.linear3.weight', None)
    m[e + 'mlp.down_projection.weight'] = (t + 'mlp.linear2.weight', None)
  return m


# integer tables of his own 1968-move output: not parameters of our net (ours gathers to the 1858 layout)
CHECKPOINT_BUFFERS = ('from_squares', 'to_squares', 'promotion_move_ids', 'promotion_to_squares', 'promotion_pieces')


def load_fry_checkpoint(net: torch.nn.Module, tensors) -> dict:
  """Copy the checkpoint's tensors (a dict / npz of numpy arrays) onto `net` built from FRY_NET.
  Returns {'loaded': n tensors, 'loaded_elements': n, 'ours_untouched': [...], 'theirs_unused': [...]}:
  every checkpoint tensor must be consumed (except the integer move tables) and every parameter of our trunk, Fry
  heads and embedding must be written; what stays untouched is listed (our heads that Fry does not have)."""
  params = dict(net.named_parameters())
  nm = name_map()
  written = set()
  elements = 0
  unused = []
  with torch.no_grad():
    for key in tensors.keys() if hasattr(tensors, 'keys') else tensors.files:
      if key in CHECKPOINT_BUFFERS:
        continue
      if key not in nm:
        unused.append(key)
        continue
      ours, idx = nm[key]
      src = torch.from_numpy(np.asarray(tensors[key])).float()
      dst = params[ours]
      if idx is None:
        assert tuple(dst.shape) == tuple(src.shape), (key, ours, tuple(src.shape), tuple(dst.shape))
        dst.copy_(src)
      else:
        assert src.numel() == 1, (key, src.shape)
        dst[idx] = src.reshape(())
      written.add(ours)
      elements += src.numel()
  untouched = sorted(n for n in params if n not in written)
  return {'loaded': len(written), 'loaded_elements': elements, 'ours_untouched': untouched, 'theirs_unused': unused}


def build_fry_net(net_over=None, opt_over=None):
  """Our CeresNet built from FRY_NET / FRY_OPT (float32, CPU), through the tests' config helper."""
  import test_bt6_hybrid as H
  saved = H.NET_SMALL, H.OPT_BASE
  try:
    H.NET_SMALL = {**FRY_NET, **(net_over or {})}
    H.OPT_BASE = {**FRY_OPT, **(opt_over or {})}
    return H.build()
  finally:
    H.NET_SMALL, H.OPT_BASE = saved
