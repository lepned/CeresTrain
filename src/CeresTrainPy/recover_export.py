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

from config import Configuration, NUM_INPUT_BYTES_PER_SQUARE
from ceres_net import CeresNet
from save_model import save_model

TRAINING_ID = sys.argv[1]
OUTPUTS_DIR = sys.argv[2]
NUM_POS     = sys.argv[3]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Configuration('.', os.path.join(OUTPUTS_DIR, "configs", TRAINING_ID))
NAME = 'lepdev_' + TRAINING_ID

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

missing, unexpected = model.load_state_dict(loaded['model'], strict=False)
if missing:    print('WARN: missing keys (count={}): {}'.format(len(missing), missing[:5]))
if unexpected: print('WARN: unexpected keys (count={}): {}'.format(len(unexpected), unexpected[:5]))

state = {'optimizer': None}
save_model(NAME, OUTPUTS_DIR, config, model, state, NUM_POS, True)
print('INFO: RECOVER_EXPORT_DONE', NAME, NUM_POS)
