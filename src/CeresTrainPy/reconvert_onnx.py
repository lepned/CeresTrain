"""
Standalone helper: re-exports ONNX from an existing Lightning checkpoint
by calling save_model.save_model(). Mirrors the end-of-training export path
exactly — picks up the current (fixed) FP16 conversion logic.

Usage:
    python reconvert_onnx.py <config_id> <outputs_dir> <ckpt_name>

Example:
    python reconvert_onnx.py puzzle_py_smoke /mnt/c/Dev/Chess/CeresTrain ckpt_lepdev_puzzle_py_smoke_20000768
"""

import os
import sys
import socket
import torch
import lightning as pl

from config import Configuration
from ceres_net import CeresNet
from save_model import save_model

if len(sys.argv) != 4:
    print(__doc__)
    sys.exit(1)

CONFIG_ID = sys.argv[1]
OUTPUTS_DIR = sys.argv[2]
CKPT_NAME = sys.argv[3]

config = Configuration('.', os.path.join(OUTPUTS_DIR, "configs", CONFIG_ID))
NAME = socket.gethostname() + "_" + os.path.basename(CONFIG_ID)

fabric = pl.Fabric(
    accelerator=config.Exec_DeviceType.lower(),
    devices=[0] if isinstance(config.Exec_DeviceIDs, list) else config.Exec_DeviceIDs,
    precision='bf16-mixed' if config.Exec_DataType == 'BFloat16' else '32-true',
)
fabric.launch()

model = CeresNet(
    fabric, config,
    1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5,
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

missing, unexpected = model.load_state_dict(sd_clean, strict=False)
print(f"Loaded. missing={len(missing)} unexpected={len(unexpected)}")
if missing[:3]:
    print(f"  Sample missing: {missing[:3]}")
if unexpected[:3]:
    print(f"  Sample unexpected: {unexpected[:3]}")

# NOTE: do NOT fabric.setup(model) — that wraps it in _FabricModule which
# breaks torch.export's input-count inference (sees (x,y) as 1 tuple element).
# save_model() expects an unwrapped model_nocompile anyway.
model = model.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
model.eval()

num_pos_str = CKPT_NAME.split('_')[-1]
print(f"Calling save_model (will write ONNX with fixed FP16 conversion)")
save_model(NAME, OUTPUTS_DIR, config, fabric, model, state={}, num_pos=num_pos_str, save_all_formats=True)
print("DONE")
