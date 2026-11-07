# Piece-token conversion test (2026-11-07): export a trained 64-square NBT checkpoint as a PIECE-TOKEN net, untrained
# conversion (only the zero-init pt_decompress is new), to measure how much the trunk loses when it sees 32 squares.
#
# Usage (WSL, CPU):
#   python3 pt_convert_export.py <OUTPUTS_DIR> <TRAINING_ID> <CKPT_PATH> <NUM_POS> <TAG> [net-json overrides as k=v ...]
#   e.g. ... /mnt/d/speedtest/ptconv prod_1920_12_nbt3d3_smA_t80t91_8B /mnt/d/.../ckpt_..._2300078592 2300078592 pt32 PieceTokens=32
# Writes <OUTPUTS_DIR>/configs/<TRAINING_ID>_<TAG>_ceres_*.json (copies + net overrides) and exports
# <OUTPUTS_DIR>/nets/lepdev_<TRAINING_ID>_<TAG>_<NUM_POS>*.onnx through the normal save_model path.
import json, os, shutil, sys

OUT, TID, CKPT, NUM_POS, TAG = sys.argv[1:6]
OVERRIDES = {}
for kv in sys.argv[6:]:
  k, v = kv.split('=', 1)
  OVERRIDES[k] = json.loads(v)
NEW_ID = f'{TID}_{TAG}'
cfg_dir = os.path.join(OUT, 'configs')
for part in ('data', 'exec', 'monitoring', 'opt', 'net'):
  src, dst = f'{cfg_dir}/{TID}_ceres_{part}.json', f'{cfg_dir}/{NEW_ID}_ceres_{part}.json'
  if part == 'net':
    d = json.load(open(src)); d.update(OVERRIDES)
    json.dump(d, open(dst, 'w'), indent=1)
  else:
    shutil.copy(src, dst)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'CeresTrainPy'))
from config_bootstrap import bootstrap_env_from_config
bootstrap_env_from_config(OUT, NEW_ID)
import torch
from config import Configuration
from ceres_net import CeresNet
from save_model import save_model

config = Configuration('.', os.path.join(cfg_dir, NEW_ID))
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
                 q_ratio=config.Data_FractionQ)
loaded = torch.load(CKPT, map_location='cpu', weights_only=False)
missing, unexpected = model.load_state_dict(loaded['model'], strict=False)
assert not unexpected, f'unexpected keys: {unexpected[:8]}'
assert all(k.startswith('pt_decompress.') for k in missing), f'missing non-decompression keys: {[k for k in missing if not k.startswith("pt_decompress.")][:8]}'
print(f'INFO: PT_CONVERT {NEW_ID}: loaded {CKPT}, new (zero-init/fresh) keys: {missing}')
del loaded
NAME = 'lepdev_' + NEW_ID
save_model(NAME, OUT, config, model, {'optimizer': None}, NUM_POS, True)
print('INFO: PT_CONVERT_DONE', NAME, NUM_POS)
