"""
Standalone helper: re-exports ONNX from an existing checkpoint by calling
save_model.save_model(). Mirrors the end-of-training export path exactly —
picks up the current (fixed) FP16 conversion logic.

Usage:
    python3 reconvert_onnx.py <config_id> <outputs_dir> <ckpt_name>

Example:
    python3 reconvert_onnx.py puzzle_py_smoke /mnt/c/Dev/Chess/CeresTrain ckpt_lepdev_puzzle_py_smoke_20000768
"""

import os
import sys
import socket
import torch

if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(1)

CONFIG_ID = sys.argv[1]
OUTPUTS_DIR = sys.argv[2]

# See recover_export.py: the config->env bridge must run BEFORE importing
# config/ceres_net (they read data-format settings at import time).
from config_bootstrap import bootstrap_env_from_config
bootstrap_env_from_config(OUTPUTS_DIR, CONFIG_ID)

from config import Configuration
from ceres_net import CeresNet
from save_model import save_model
CKPT_NAME = sys.argv[3]

config = Configuration('.', os.path.join(OUTPUTS_DIR, "configs", CONFIG_ID))
NAME = socket.gethostname() + "_" + os.path.basename(CONFIG_ID)

# Plain-PyTorch model construction; pass writer=None (this script never logs).
# Loss weights MUST come from the config (bugfunn 2026-08-28, runde 2): head
# construction is gated on weight > 0, and eval-forward aliases missing heads
# (value2 -> value1, mlh -> unc) — hardcoded zeros silently exported a net
# whose value2 output was a COPY of value1 for every dual-value production run.
model = CeresNet(
    None, config,
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
    q_ratio=config.Data_FractionQ,
)

ckpt_path = os.path.join(OUTPUTS_DIR, "nets", CKPT_NAME)
print(f"Loading checkpoint: {ckpt_path}")
state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
if isinstance(state, dict) and 'state_dict' in state:
    sd = state['state_dict']
elif isinstance(state, dict) and 'model' in state:
    sd = state['model']
else:
    sd = state

sd_clean = {}
for k, v in sd.items():
    nk = k
    for prefix in ('_forward_module._orig_mod.', '_orig_mod.', 'model.'):
        if nk.startswith(prefix):
            nk = nk[len(prefix):]
    sd_clean[nk] = v

# Universell noekkelsett-vakt (som recover_export.py): enhver differanse
# mellom checkpoint og gjenoppbygget modell = korrupt eksport — hard stopp.
_ckpt_keys = set(sd_clean.keys())
_model_keys = set(model.state_dict().keys())
if _ckpt_keys != _model_keys:
    _only_ckpt = sorted(_ckpt_keys - _model_keys)
    _only_model = sorted(_model_keys - _ckpt_keys)
    raise ValueError(
        'State-dict key-set mismatch - export would be CORRUPT. '
        f'Keys only in checkpoint ({len(_only_ckpt)}): {_only_ckpt[:8]} ; '
        f'keys only in rebuilt model ({len(_only_model)}): {_only_model[:8]} ; '
        'align the _ceres_net.json config (and env-gated features) with the '
        f'training run before re-exporting. (checkpoint: {ckpt_path})')
model.load_state_dict(sd_clean, strict=True)
print("Loaded (strict, key-sets identical).")

model = model.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
model.eval()

num_pos_str = CKPT_NAME.split('_')[-1]
print(f"Calling save_model (will write ONNX with fixed FP16 conversion)")
save_model(NAME, OUTPUTS_DIR, config, model, state={}, num_pos=num_pos_str, save_all_formats=True)
print("DONE")
