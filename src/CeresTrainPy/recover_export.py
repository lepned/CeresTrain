# Standalone post-hoc export.
#
# Reconstructs CeresNet from a saved checkpoint and runs the same save_model
# path that train.py runs at end-of-training. Use when training completed but
# the in-process ONNX export failed (e.g. opset-conversion crash leaving only
# .ts + .onnx.data on disk).
#
# Usage:
#   python3 recover_export.py <TRAINING_ID> <OUTPUTS_DIR> <NUM_POS>
# Example:
#   python3 recover_export.py c2_512_25_swiglu_rope_base1000_PRE_b4096_10M /mnt/c/Dev/Chess/CeresTrain 10000384

import os, sys, torch

# Bridge the run config into os.environ BEFORE importing config/ceres_net:
# those modules read data-format settings (aux width, shard format) at IMPORT
# time, and rebuilding a net from a checkpoint under different values silently
# produces a differently-shaped net. Same mapping train.py uses.
from config_bootstrap import bootstrap_env_from_config
bootstrap_env_from_config(sys.argv[2], sys.argv[1])

from config import Configuration, NUM_INPUT_BYTES_PER_SQUARE
from ceres_net import CeresNet
from save_model import save_model

TRAINING_ID = sys.argv[1]
OUTPUTS_DIR = sys.argv[2]
NUM_POS     = sys.argv[3]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Configuration('.', os.path.join(OUTPUTS_DIR, "configs", TRAINING_ID))
NAME = os.environ.get('CERES_HOST_PREFIX', 'lepdev') + '_' + TRAINING_ID

model = CeresNet(None, config,
                 policy_loss_weight=config.Opt_LossPolicyMultiplier,
                 value_loss_weight=config.Opt_LossValueMultiplier,
                 moves_left_loss_weight=config.Opt_LossMLHMultiplier,
                 unc_loss_weight=config.Opt_LossUNCMultiplier,
                 value2_loss_weight=config.Opt_LossValue2Multiplier,
                 q_deviation_loss_weight=config.Opt_LossQDeviationMultiplier,
                 value_diff_loss_weight=config.Opt_LossValueDMultiplier,
                 value2_diff_loss_weight=config.Opt_LossValue2DMultiplier,
                 action_loss_weight=config.Opt_LossActionMultiplier,
                 uncertainty_policy_weight=config.Opt_LossUncertaintyPolicyMultiplier,
                 action_uncertainty_loss_weight=config.Opt_LossActionUncertaintyMultiplier,
                 q_ratio=config.Data_FractionQ).to(device)

CKPT = os.path.join(OUTPUTS_DIR, 'nets', 'ckpt_' + NAME + '_' + NUM_POS)
print('INFO: LOADING_CHECKPOINT', CKPT)
loaded = torch.load(CKPT, map_location=device, weights_only=False)

# AUX-WIDTH GUARD: strict=False below would SILENTLY skip a mismatched embedding
# and re-export a net with a randomly-initialized input layer. Detect the width
# disagreement up front and fail with the exact env var to set. (With the new
# default CERES_AUX_FEATURES_PER_SQUARE=4, V3 nets re-export with no env var;
# re-exporting a legacy 137-channel net needs CERES_AUX_FEATURES_PER_SQUARE=0.)
_ckpt_emb_w = loaded['model'].get('embedding_layer.weight', None)
if _ckpt_emb_w is not None:
  _ckpt_in = _ckpt_emb_w.shape[1]
  _model_in = model.embedding_layer.weight.shape[1]
  if _ckpt_in != _model_in:
    _prior = config.NetDef_PriorStateDim
    _ckpt_aux = _ckpt_in - NUM_INPUT_BYTES_PER_SQUARE - _prior
    raise ValueError(
      f"Aux-feature width mismatch: checkpoint embedding expects {_ckpt_in} input "
      f"features/square ({_ckpt_aux} aux) but the rebuilt model has {_model_in}. "
      f"Set CERES_AUX_FEATURES_PER_SQUARE={_ckpt_aux} and re-run. (checkpoint: {CKPT})")

# VIS-EDGE GUARD: same silent-skip class as the aux-width guard above. If the
# checkpoint carries trained vis edge-bias weights but the rebuilt model has
# none, strict=False would discard them with only a WARN and export a net
# missing an attention bias the trunk co-adapted to. The knobs are config-only
# (NetDef UseVisEdgeBias/VisEdgeGates), so this firing means the _ceres_net.json
# no longer matches what the checkpoint was trained with.
_ckpt_vis = [k for k in loaded['model']
             if k.startswith('vis_edge_proj.') or '.attack_gate_' in k]
_model_vis = [k for k in model.state_dict()
              if k.startswith('vis_edge_proj.') or '.attack_gate_' in k]
if _ckpt_vis and not _model_vis:
  raise ValueError(
    f'Checkpoint contains {len(_ckpt_vis)} vis edge-bias tensors but the rebuilt model has none. '
    f'Set "UseVisEdgeBias": true (and matching VisEdgeFamilies/VisEdgeGates) in the '
    f'_ceres_net.json config before re-exporting. (checkpoint: {CKPT})')
# Same guard for the ray attention bias (trained under CERES_RAY_ATTENTION_BIAS
# or "UseRayAttentionBias": a ray-trained checkpoint re-exported without the
# flag would silently drop ray_proj and export a net missing its bias.
_ckpt_ray = [k for k in loaded['model'] if k.startswith('ray_bias_')]
_model_ray = [k for k in model.state_dict() if k.startswith('ray_bias_')]
if _ckpt_ray and not _model_ray:
  raise ValueError(
    f'Checkpoint contains {len(_ckpt_ray)} ray-bias tensors but the rebuilt model has none. '
    f'Set "UseRayAttentionBias": true in the _ceres_net.json config before re-exporting. '
    f'(checkpoint: {CKPT})')
# Same guard for the graph-route heads and the tactic refiner (2026-08 tactical
# program; config NetDef UseGraphRouteHeads / RefinerIters). These are WORSE to
# silently drop than the bias-only cases above: the refiner residual feeds
# EVERY head (flow = flow + refiner delta), so exporting without it serves a
# functionally corrupted net, not just one missing an attention bias.
_ckpt_gr = [k for k in loaded['model'] if '.graph_route_' in k]
_model_gr = [k for k in model.state_dict() if '.graph_route_' in k]
if _ckpt_gr and not _model_gr:
  raise ValueError(
    f'Checkpoint contains {len(_ckpt_gr)} graph-route tensors but the rebuilt model has none. '
    f'Set "UseGraphRouteHeads": true (and UseVisEdgeBias + matching VisEdgeFamilies) in the '
    f'_ceres_net.json config before re-exporting. (checkpoint: {CKPT})')
# Same guard for the value min/max pool injectors (config NetDef
# ValueHeadMinMaxPool): zero-init at start but TRAINED weights are a real
# contribution to the value heads — silently dropping them exports a net whose
# value family lost a co-adapted input.
_ckpt_vp = [k for k in loaded['model'] if '_pool_inject' in k]
_model_vp = [k for k in model.state_dict() if '_pool_inject' in k]
if _ckpt_vp and not _model_vp:
  raise ValueError(
    f'Checkpoint contains {len(_ckpt_vp)} value-pool-inject tensors but the rebuilt model '
    f'has none. Set "ValueHeadMinMaxPool": true in the _ceres_net.json config before '
    f're-exporting. (checkpoint: {CKPT})')
# Same guard for soft-min heads (config NetDef SoftMinHeads). ARCH key, worse
# than the drop cases above in a different way: the params are per-layer tau
# scalars, but disabling the key also flips the first k heads' aggregation
# back to weighted mean — the rebuilt net computes a different FUNCTION even
# where weights match, so a mismatch here means the config is simply wrong.
_ckpt_sm = [k for k in loaded['model'] if '.softmin_log_tau' in k]
_model_sm = [k for k in model.state_dict() if '.softmin_log_tau' in k]
if bool(_ckpt_sm) != bool(_model_sm):
  raise ValueError(
    f'SoftMinHeads mismatch: checkpoint has {len(_ckpt_sm)} softmin_log_tau tensors, '
    f'rebuilt model has {len(_model_sm)}. Set "SoftMinHeads" in the _ceres_net.json '
    f'config to match the training run before re-exporting. (checkpoint: {CKPT})')
# Guard for the dual-plane P-plane (config NetDef UseDualPlane).
_DP_SUBSTR = ('dp_value', 'dp_pol_', 'dpva_', 'dpv_', 'dpe_w', 'dpd_', 'dpcv_', 'dpc_', 'dpch_', 'dp_surv')
_ckpt_dp = [k for k in loaded['model'] if k.startswith('dual_plane.') or any(t in k for t in _DP_SUBSTR)]
_model_dp = [k for k in model.state_dict() if k.startswith('dual_plane.') or any(t in k for t in _DP_SUBSTR)]
if sorted(_ckpt_dp) != sorted(_model_dp):   # SET equality: equal-count family swaps (e.g. dpd_* out, dpv_* in) must also refuse (review 2026-08-25b finding 5)
  raise ValueError(
    f'UseDualPlane mismatch: checkpoint has {len(_ckpt_dp)} dual-plane tensors, '
    f'rebuilt model has {len(_model_dp)}. Set "UseDualPlane" in the _ceres_net.json '
    f'config to match the training run before re-exporting. (checkpoint: {CKPT})')
# Guard for spectral PE (config NetDef UseSpectralPE).
_ckpt_sp = [k for k in loaded['model'] if 'spe_proj' in k]
_model_sp = [k for k in model.state_dict() if 'spe_proj' in k]
if bool(_ckpt_sp) != bool(_model_sp):
  raise ValueError(
    f'UseSpectralPE mismatch: checkpoint has {len(_ckpt_sp)} spe_proj tensors, '
    f'rebuilt model has {len(_model_sp)}. Set "UseSpectralPE" in the _ceres_net.json '
    f'config to match the training run before re-exporting. (checkpoint: {CKPT})')
# Guard for king-distance channels (config NetDef UseKingDistChannels).
_ckpt_kd = [k for k in loaded['model'] if 'kdist_proj' in k]
_model_kd = [k for k in model.state_dict() if 'kdist_proj' in k]
if bool(_ckpt_kd) != bool(_model_kd):
  raise ValueError(
    f'UseKingDistChannels mismatch: checkpoint has {len(_ckpt_kd)} kdist_proj tensors, '
    f'rebuilt model has {len(_model_kd)}. Set "UseKingDistChannels" in the _ceres_net.json '
    f'config to match the training run before re-exporting. (checkpoint: {CKPT})')
# Guard for the tactical codebook (config NetDef UseTacticalCodebook):
# dropping the key amputates a trained residual contributor = corrupt net.
_ckpt_cb = [k for k in loaded['model'] if k.startswith('cbk_')]
_model_cb = [k for k in model.state_dict() if k.startswith('cbk_')]
if bool(_ckpt_cb) != bool(_model_cb):
  raise ValueError(
    f'UseTacticalCodebook mismatch: checkpoint has {len(_ckpt_cb)} cbk_* tensors, '
    f'rebuilt model has {len(_model_cb)}. Set "UseTacticalCodebook" in the _ceres_net.json '
    f'config to match the training run before re-exporting. (checkpoint: {CKPT})')
# Guard for per-head logit temperature (config NetDef UseHeadLogitTemp):
# dropping the key silently rebuilds with temp=1 while the checkpoint trained
# with learned temps — a different function everywhere.
_ckpt_ht = [k for k in loaded['model'] if '.head_logit_temp' in k]
_model_ht = [k for k in model.state_dict() if '.head_logit_temp' in k]
if bool(_ckpt_ht) != bool(_model_ht):
  raise ValueError(
    f'UseHeadLogitTemp mismatch: checkpoint has {len(_ckpt_ht)} head_logit_temp tensors, '
    f'rebuilt model has {len(_model_ht)}. Set "UseHeadLogitTemp" in the _ceres_net.json '
    f'config to match the training run before re-exporting. (checkpoint: {CKPT})')
# Identical guard for the signed-tau dual (config NetDef SoftMaxAggHeads).
_ckpt_sx = [k for k in loaded['model'] if '.softmax_log_tau' in k]
_model_sx = [k for k in model.state_dict() if '.softmax_log_tau' in k]
if bool(_ckpt_sx) != bool(_model_sx):
  raise ValueError(
    f'SoftMaxAggHeads mismatch: checkpoint has {len(_ckpt_sx)} softmax_log_tau tensors, '
    f'rebuilt model has {len(_model_sx)}. Set "SoftMaxAggHeads" in the _ceres_net.json '
    f'config to match the training run before re-exporting. (checkpoint: {CKPT})')
_ckpt_rf = [k for k in loaded['model'] if k.startswith('tactical_refiner.')]
_model_rf = [k for k in model.state_dict() if k.startswith('tactical_refiner.')]
if _ckpt_rf and not _model_rf:
  raise ValueError(
    f'Checkpoint contains {len(_ckpt_rf)} tactic-refiner tensors but the rebuilt model has none. '
    f'Set "RefinerIters" (and matching RefinerDim/Heads/FFNMult) in the _ceres_net.json '
    f'config before re-exporting. (checkpoint: {CKPT})')

missing, unexpected = model.load_state_dict(loaded['model'], strict=False)
if missing:    print('WARN: missing keys (count={}): {}'.format(len(missing), missing[:5]))
if unexpected: print('WARN: unexpected keys (count={}): {}'.format(len(unexpected), unexpected[:5]))

state = {'optimizer': None}
save_model(NAME, OUTPUTS_DIR, config, model, state, NUM_POS, True)
print('INFO: RECOVER_EXPORT_DONE', NAME, NUM_POS)
