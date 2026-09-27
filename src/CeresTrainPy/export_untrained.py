# Export a RANDOMLY INITIALIZED CeresNet through the same save_model path as training.
#
# For serving-speed measurements of candidate architectures: TensorRT speed does not
# depend on the weight values, so an untrained export times exactly like a trained one
# of the same shape (FP16; INT8 calibration needs real weights).
#
# Usage (run from src/CeresTrainPy, like recover_export.py):
#   python3 export_untrained.py <TRAINING_ID> <OUTPUTS_DIR> [<TAG>]
# Writes <OUTPUTS_DIR>/nets/<host>_<TRAINING_ID>_<TAG>[m80].onnx (TAG default 'untrained').
# Set CERES_MT_EXPORT_MAX=80 to match the production move-token export.

import os, sys, torch

from config_bootstrap import bootstrap_env_from_config
bootstrap_env_from_config(sys.argv[2], sys.argv[1])

from config import Configuration
from ceres_net import CeresNet
from save_model import save_model

TRAINING_ID = sys.argv[1]
OUTPUTS_DIR = sys.argv[2]
TAG = sys.argv[3] if len(sys.argv) > 3 else 'untrained'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(777)

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
model.eval()
print(f'INFO: UNTRAINED {TRAINING_ID}: {sum(p.numel() for p in model.parameters()):,} params')

state = {'optimizer': None}
save_model(NAME, OUTPUTS_DIR, config, model, state, TAG, True)
print('INFO: EXPORT_UNTRAINED_DONE', NAME, TAG)
