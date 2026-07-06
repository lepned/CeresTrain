# Variant of recover_export.py that uses socket.gethostname() for the NAME
# prefix, matching what train.py does. This makes the recovery work without
# needing the 'lepdev_' hardcoded prefix.
#
# Usage:
#   python3 recover_export_hostname.py <TRAINING_ID> <OUTPUTS_DIR> <NUM_POS>

import os, sys, socket, torch

from config import Configuration
from ceres_net import CeresNet
from save_model import save_model

TRAINING_ID = sys.argv[1]
OUTPUTS_DIR = sys.argv[2]
NUM_POS     = sys.argv[3]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Configuration('.', os.path.join(OUTPUTS_DIR, "configs", TRAINING_ID))
# Mirror train.py:127 — hostname + config id
NAME = socket.gethostname() + "_" + TRAINING_ID
print(f'INFO: USING_NAME={NAME}')

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
missing, unexpected = model.load_state_dict(loaded['model'], strict=False)
if missing:    print('WARN: missing keys (count={}): {}'.format(len(missing), missing[:5]))
if unexpected: print('WARN: unexpected keys (count={}): {}'.format(len(unexpected), unexpected[:5]))

# Set the model to eval before export to ensure stable output
# (avoids any train-mode-dependent layer behavior leaking into the trace).
model.eval()

state = {'optimizer': None}
save_model(NAME, OUTPUTS_DIR, config, model, state, NUM_POS, True)
print('INFO: RECOVER_EXPORT_DONE', NAME, NUM_POS)
