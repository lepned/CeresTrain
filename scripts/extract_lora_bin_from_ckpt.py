"""Post-hoc LoRA bin extractor.

Reads a Lightning fabric checkpoint (`ckpt_*` file produced by save_checkpoint)
and writes a `.lora_<num_pos>.bin` matching what save_model would have emitted.
Used to recover artifacts when save_model.py's bin-save gate was too narrow
(pre-fix: only fired on Opt_LoRARankDivisor>0, missing env-var-only LoRA paths).

Usage:
  V8_BASE_CKPT=/path/to/ckpt_NAME_500224  V8_BIN_OUT=/path/to/lepdev_NAME.lora_500224.bin  python3 extract_lora_bin_from_ckpt.py
"""

import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'CeresTrainPy'))
from lora import serialize_lora_to_binary  # noqa: E402

CKPT = os.environ['V8_BASE_CKPT']
OUT  = os.environ['V8_BIN_OUT']

state = torch.load(CKPT, map_location='cpu', weights_only=False)['model']
state = {k.replace('_forward_module._orig_mod.', ''): v for k, v in state.items()}

lora_matrices = {}
for k in list(state.keys()):
    if not k.endswith('.lora_A'):
        continue
    layer = k[:-len('.lora_A')]
    bk = f'{layer}.lora_B'
    ak = f'{layer}.lora_alpha'
    if bk not in state or ak not in state:
        raise ValueError(f'Missing lora_B or lora_alpha for {layer}')
    A = state[k].detach().to(torch.float32).cpu().numpy()
    B = state[bk].detach().to(torch.float32).cpu().numpy()
    alpha = state[ak].detach().to(torch.float32).cpu().numpy()
    alpha_list = [float(alpha)] if alpha.size == 1 else alpha.tolist()
    lora_matrices[layer] = {
        'alpha': alpha_list,
        'A': A.tolist(),
        'B': B.tolist(),
    }

print(f'[extract] found {len(lora_matrices)} LoRA layers in ckpt')
serialize_lora_to_binary(OUT, lora_matrices)
