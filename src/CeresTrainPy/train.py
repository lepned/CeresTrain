# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

import os
import fnmatch
import sys
import socket
import datetime
import math
import contextlib
import numpy as np
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torchinfo import summary
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from rms_norm import RMSNorm
from derf_norm import DerfNorm
from dyt_norm import DyTNorm
from losses import LossCalculator
from tpg_dataset import TPGDataset, TPGMixedDataset
from config import Configuration
import lora
from config import NUM_TOKENS_INPUT, NUM_TOKENS_NET, NUM_INPUT_BYTES_PER_SQUARE, TOTAL_INPUT_FEATURES_PER_SQUARE
from utils import calc_flops

from ceres_net import CeresNet
from soft_moe_batched_dual import SoftMoEBatchedDual
from multi_expert import MultiExpertLayer
from save_model import save_model, save_checkpoint

from AdEMAMix import AdEMAMix
from AdEMAMixShampoo import AdEMAMixDistributedShampoo
from soap import SOAP
from muon import Muon


def _grad_norm(model, norm_type: float = 2.0) -> Dict[str, float]:
    """Per-parameter and total gradient norms. Plain-PyTorch replacement for
    lightning.pytorch.utilities.grad_norm. Returns a dict shaped like
    {'grad_<n>_norm/<param-name>': float, 'grad_<n>_norm_total': float}.
    Match Lightning's float-formatted keys (e.g. 'grad_2.0_norm_total')."""
    nt = float(norm_type)
    norms: Dict[str, float] = {}
    total = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n = p.grad.detach().data.norm(nt).item()
        norms[f'grad_{nt}_norm/{name}'] = n
        total += n ** nt
    norms[f'grad_{nt}_norm_total'] = total ** (1.0 / nt) if total > 0 else 0.0
    return norms


def _move_batch_to_device(batch, device):
    """Recursively move any tensor leaves in a batch (dict / list / tuple) to
    device using non_blocking transfers (requires pin_memory=True). Replaces
    Lightning's setup_dataloaders auto-move."""
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_batch_to_device(v, device) for v in batch)
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch

print(torch.__version__)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True) # efficient seems faster than flash for short sequences


TRAINING_ID = sys.argv[1]
OUTPUTS_DIR = sys.argv[2]

# make sure any required subdirectories exist
os.makedirs(os.path.join(OUTPUTS_DIR, "nets"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_DIR, "tblogs"), exist_ok=True)

config = Configuration('.', os.path.join(OUTPUTS_DIR, "configs", TRAINING_ID))
TPG_TRAIN_DIR = config.Data_TrainingFilesDirectory 


#TODO: would be better to use asserts but they are not captured by the remote process executor
if TPG_TRAIN_DIR is None:
  print('ERROR: TrainingFilesDirectory is null')
  exit(1)
elif not os.path.isdir(TPG_TRAIN_DIR): 
  print('ERROR: TrainingFilesDirectory does not exist:', TPG_TRAIN_DIR)
  exit(1)
elif not os.listdir(TPG_TRAIN_DIR):
  print(f"ERROR: The directory TrainingFilesDirectory ('{TPG_TRAIN_DIR}') is empty.")
  exit(1)


def print_model_trainable_details(model):
  num_params = 0
  num_layers = 0
  print("Model details (trainable parameters only):\n")
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(f"Layer: {name} | Size: {param.size()} | Total parameters: {param.numel()}")
      num_params+= param.numel()
      num_layers = num_layers + 1
  print()
  print("INFO: NUM_PARAMETERS", str(num_params))


NAME = socket.gethostname() + "_" + os.path.basename(TRAINING_ID)

accelerator = config.Exec_DeviceType.lower()
devices = config.Exec_DeviceIDs if not config.Exec_ExportOnly else config.Exec_DeviceIDs[0]

BATCH_SIZE = config.Opt_BatchSizeBackwardPass

# PreNorm now supported (encoder_layer.py forward branches on pre_norm flag,
# ceres_net.py adds a trunk-end norm after the stack). Both flag values legal.
assert config.Exec_DataType == 'BFloat16' or config.Exec_DataType == 'BFloat16Pure', 'Only BFloat16 or BFloat16Pure training supported'
assert config.Opt_LoRARankDivisor == 0 or config.Opt_CheckpointResumeFromFileName is not None, 'LoRA requires Opt_CheckpointResumeFromFileName resume'

MAX_POSITIONS = config.Opt_NumTrainingPositions

if config.NetDef_TrainOn4BoardSequences:
  BOARDS_PER_BATCH = 4
else:
  BOARDS_PER_BATCH = 1  

LR = config.Opt_LearningRateBase
WEIGHT_DECAY = config.Opt_WeightDecay

num_pos = 0

time_last_status_update = datetime.datetime.now()
time_last_save = datetime.datetime.now()
time_start = datetime.datetime.now()
time_last_save_permanent = datetime.datetime.now()
time_last_save_transient = datetime.datetime.now()

def get_most_extreme_weight_value(model):
  extreme_value = 0.0
  for param in model.parameters():
    if param.requires_grad:
      param_max = abs(param.max().item())
      param_min = abs(param.min().item())
      if param_max > extreme_value:
        extreme_value = param_max
      if param_min > extreme_value:
        extreme_value = param_min
  return extreme_value

# Storage for previous parameters for calc_weight_update_ratio
previous_params = {}

def calc_weight_update_ratio(model, logger):
  global previous_params

  total_update_norm = 0
  total_weight_norm = 0

  with torch.no_grad():
    for name, param in model.named_parameters():
      if param.requires_grad:
        if name in previous_params:
          delta = param - previous_params[name]
          update_norm = torch.norm(delta).item()
          weight_norm = torch.norm(param).item()

          total_update_norm += update_norm
          total_weight_norm += weight_norm

        previous_params[name] = param.clone().detach()

  if total_weight_norm > 0:
    return total_update_norm / total_weight_norm
  else:
    return 0


last_logged_pos_num = 0

def on_before_optimizer_step(writer, model, optimizer, pos_num):
    global last_logged_pos_num

    step = pos_num // BATCH_SIZE

    # Log only periodically.
    LOG_EVERY_N_POSITIONS = 100000
    positions_since_logged = pos_num - last_logged_pos_num
    if positions_since_logged < LOG_EVERY_N_POSITIONS:
      return
    else:
      last_logged_pos_num = pos_num

    # log ratio of average absolute weight update to average absolute weight
    # note that this does retain an extra copy of the model parameters and increase GPU memory usage
    weight_ratio = calc_weight_update_ratio(model, writer)
    writer.add_scalar("update_weight_ratio", weight_ratio, pos_num)

    norms = _grad_norm(model, norm_type=2)

    # update_magnitude is an approximate measure of effective magnitude of weight updates which
    # depends multiplicatively upon the size of the gradients and the current learning rate
    update_magnitude = norms['grad_2.0_norm_total'] * optimizer.param_groups[0]['lr']
    writer.add_scalar("update_magnitude", update_magnitude, pos_num)

    for k, v in norms.items():
      writer.add_scalar(k, v, pos_num)
    writer.add_scalar("max_abs_weight", get_most_extreme_weight_value(model), pos_num)

    LOG_GRAD_HISTOGRAMS = False
    if LOG_GRAD_HISTOGRAMS:
      for k, v in model.named_parameters():
        if v.grad is not None:
          writer.add_histogram(tag=k, values=v.grad, global_step=pos_num)


def Train():
  global num_pos
  global fraction_complete

  print("**** STARTING ", NAME)
  

  if config.Exec_UseFP8:
    raise NotImplementedError(
        "Exec_UseFP8 was previously supported via Lightning Fabric's TransformerEnginePrecision; "
        "the plain-PyTorch path does not wire transformer-engine directly. "
        "Use Exec_DataType='BFloat16' instead.")

  # Plain-PyTorch device + tensorboard setup. Replaces Lightning Fabric.
  device = torch.device(f"{accelerator}:{devices[0]}" if accelerator != 'cpu' else 'cpu')
  writer = SummaryWriter(os.path.join(OUTPUTS_DIR, 'tblogs', NAME))
  # bf16-mixed: model weights are fp32, forward runs under autocast
  # bf16-pure : model weights are bf16, no autocast needed
  USE_AUTOCAST = (config.Exec_DataType == 'BFloat16')


  # NOTE: these very small values for MLH and UNC are best because
  #       they enhance training stability and don't negatively affect policy/value
  #       but produce MLH/UNC outputs which are not significantly less accurate
  #       than if were at higher loss weight.
  model = CeresNet(writer, config, policy_loss_weight=config.Opt_LossPolicyMultiplier,
                   value_loss_weight= config.Opt_LossValueMultiplier, 
                   moves_left_loss_weight= config.Opt_LossMLHMultiplier, 
                   unc_loss_weight= config.Opt_LossUNCMultiplier,
                   value2_loss_weight= config.Opt_LossValue2Multiplier,
                   q_deviation_loss_weight= config.Opt_LossQDeviationMultiplier,
                   value_diff_loss_weight = config.Opt_LossValueDMultiplier,
                   value2_diff_loss_weight = config.Opt_LossValue2DMultiplier,
                   action_loss_weight = config.Opt_LossActionMultiplier,
                   uncertainty_policy_weight = config.Opt_LossUncertaintyPolicyMultiplier,
                   action_uncertainty_loss_weight = config.Opt_LossActionUncertaintyMultiplier,
                   q_ratio=config.Data_FractionQ)


  # LoRA can be active via several env-var paths even when head
  # Opt_LoRARankDivisor is 0:
  #   - body: CERES_LORA_ATTN_RANK_DIV / CERES_LORA_FFN_RANK_DIV / CERES_LORA_TRANSFORMER_RANK_DIV
  #   - head front-end (headPremap + headSharedLinear): CERES_LORA_HEADFRONT_RANK_DIV
  #   - smolgen (sm1/sm2/sm3 + smolgenPrepLayer): CERES_LORA_SMOLGEN_RANK_DIV
  # In any of these cases we must freeze all non-LoRA params, otherwise the
  # entire orig net (~255M params) becomes trainable and a stage-1 run
  # effectively does full fine-tune + LoRA (caused a system blackout once).
  _body_attn_div_init  = int(os.environ.get('CERES_LORA_ATTN_RANK_DIV', '0') or 0)
  _body_ffn_div_init   = int(os.environ.get('CERES_LORA_FFN_RANK_DIV', '0') or 0)
  _body_legacy_init    = int(os.environ.get('CERES_LORA_TRANSFORMER_RANK_DIV', '0') or 0)
  _headfront_div_init  = int(os.environ.get('CERES_LORA_HEADFRONT_RANK_DIV', '0') or 0)
  _smolgen_div_init    = int(os.environ.get('CERES_LORA_SMOLGEN_RANK_DIV', '0') or 0)
  _gtab_active_init    = int(os.environ.get('CERES_GTAB', '0') or 0) > 0
  _tsb_active_init     = bool(getattr(config, 'NetDef_TSB_Enabled', False))
  _body_lora_active_init = (_body_attn_div_init > 0 or _body_ffn_div_init > 0 or _body_legacy_init > 0
                            or _headfront_div_init > 0 or _smolgen_div_init > 0
                            or _gtab_active_init or _tsb_active_init)

  if config.Opt_LoRARankDivisor > 0 or _body_lora_active_init:
    # Freeze all parameters except:
    #   - LoRA (head LoRA via Opt_LoRARankDivisor and/or env-var LoRA)
    #   - GTAB tactical adapter and gate (when CERES_GTAB=1)
    #   - TSB tactical FFN and gate (when NetDef_TSB_Enabled=true)
    for name, param in model.named_parameters():
      keep_trainable = ("lora_A" in name or "lora_B" in name or "lora_alpha" in name
                        or "tactical_adapter" in name or "tactical_gate" in name
                        or "tactical_ffn" in name or ".tsb." in name)
      if not keep_trainable:
        param.requires_grad = False
   
  # Possibly compile model (as recommended by Lightning docs, comile should appear before fabric.setup).
  # N.B. when debugging, may be helpful to disable this line (otherwise breakpoints relating to graph evaluation will not be hit).
  model_nocompile = model
  if config.Opt_PyTorchCompileMode is not None and not config.Exec_ExportOnly:
    # mode choices: default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs    
    model = torch.compile(model, mode=config.Opt_PyTorchCompileMode, dynamic=False)
  
  # carefully set weight decay to apply only to appropriate subset of parameters
  # based on code from: https://github.com/karpathy/minGPT
  whitelist_weight_modules = (torch.nn.Linear, SoftMoEBatchedDual, MultiExpertLayer)
  blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RMSNorm, DerfNorm, DyTNorm)

  decay = set()
  no_decay = set()

  for mn, m in model.named_modules():
      for pn, p in m.named_parameters():
          fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
          if pn.endswith('bias'):
              no_decay.add(fpn)
          elif "rpe" in fpn:
              decay.add(fpn)
          elif "lora" in fpn:
              no_decay.add(fpn)
          elif "transformer_layer" in fpn:
              decay.add(fpn)           
          elif "rpe_factor" in fpn:
              pass
          elif "alphas" in fpn: # for Denseformer
              decay.add(fpn)
          elif ".mem_" in fpn:
              decay.add(fpn)
          elif "mlp.linear" in fpn:
              decay.add(fpn)
          elif "qkv" in fpn:
              decay.add(fpn)
          elif "embedding" in fpn:
              no_decay.add(fpn)
          elif isinstance(m, blacklist_weight_modules):
              no_decay.add(fpn)
          elif isinstance(m, whitelist_weight_modules):
              decay.add(fpn)

  
  param_dict = {pn: p for pn, p in model.named_parameters()}
  inter_params = decay & no_decay
  union_params = decay | no_decay
  assert len(inter_params) == 0, "parameters %s appear in both decay/no_decay sets" % (str(inter_params), )
  assert len(param_dict.keys() - union_params) == 0, "parameters %s were not fully partitioned into decay/no_decay sets" \
                                              % (str(param_dict.keys() - union_params), ) 
        
  optim_groups = [
      {"params": [param_dict[pn] for pn in sorted(list(decay))  if "rpe_factor" not in pn], "weight_decay": WEIGHT_DECAY},
      {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if "rpe_factor" not in pn], "weight_decay": 0.0},
  ]

  if config.Opt_LoRARankDivisor > 0:
    # LoRA parameters are not in the saved model, so the above optim_groups is incomplete (won't work)
    # Therefore disable use of optim_groups (apply weight decay to all parameters).
    # TODO: Consider if this needs to be improved (though it's probaby harmless).
    optim_groups = model.parameters()


  def num_warmup_positions():
    # Warmup is 5% of positions (but not more than 100mm).
    # Note that some sources (e.g. the SOAP paper) suggest long warmups (up to 25% of training data) are beneficial.
    return int(min(100_000_000, 0.05 * config.Opt_NumTrainingPositions))


  STEPS_AdEMAMix_WARMUP = (num_warmup_positions() // 2) // config.Opt_BatchSizeBackwardPass

  # Loss and optimizer
  if config.Opt_Optimizer == 'SGD':
    optimizer = optim.SGD(optim_groups, lr=LR*0, momentum=config.Opt_Beta1, weight_decay=WEIGHT_DECAY)
  elif config.Opt_Optimizer == 'NAdamW':
    optimizer = optim.NAdam(optim_groups, lr=LR, weight_decay=WEIGHT_DECAY, betas=(config.Opt_Beta1, config.Opt_Beta2), decoupled_weight_decay=True)
  elif config.Opt_Optimizer == 'AdamW':
    optimizer = optim.AdamW(optim_groups, lr=LR, weight_decay=WEIGHT_DECAY, betas=(config.Opt_Beta1, config.Opt_Beta2), fused=True)
  elif config.Opt_Optimizer == 'SOAP':
    PRECONDITION_FREQUENCY = 30 # typically small batch sizes used suggest less frequent updating is required
    optimizer = SOAP(optim_groups, lr=LR, weight_decay=WEIGHT_DECAY, betas=(config.Opt_Beta1, config.Opt_Beta2, config.Opt_Beta3), \
                     max_precond_size=999999, precondition_frequency=PRECONDITION_FREQUENCY)
  elif config.Opt_Optimizer == 'Muon':
    muon_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embedding" not in n and "transformer_layer" in n and p.requires_grad] # 2D parameters can use Muon
    adamw_params =[p for n, p in model.named_parameters() if (p.ndim < 2 or "embedding" in n or not "transformer_layer" in n) and p.requires_grad] # non-2D should not use Muon
    optimizer = Muon(lr=LR, wd=WEIGHT_DECAY, momentum=config.Opt_Beta1, adamw_betas=(config.Opt_Beta1, config.Opt_Beta2), muon_params=muon_params, adamw_params=adamw_params)
  elif config.Opt_Optimizer == 'AdEMAMix':
    optimizer = AdEMAMix(optim_groups, lr=LR, weight_decay=WEIGHT_DECAY, betas=(config.Opt_Beta1, config.Opt_Beta2, config.Opt_Beta3), alpha=config.Opt_Alpha, T_alpha_beta3= STEPS_AdEMAMix_WARMUP)
  elif config.Opt_Optimizer == 'AdEMAMixShampoo':
    optimizer = AdEMAMixDistributedShampoo(optim_groups, lr=LR, weight_decay=WEIGHT_DECAY, betas=(config.Opt_Beta1, config.Opt_Beta2, config.Opt_Beta3), alpha=config.Opt_Alpha, T_alpha_beta3= STEPS_AdEMAMix_WARMUP)
  elif config.Opt_Optimizer == 'AdamW8bit':
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(optim_groups, lr=LR, weight_decay=WEIGHT_DECAY, betas=(config.Opt_Beta1, config.Opt_Beta2))    
  else:
    raise ValueError("Unsupported optimizer: ", config.Opt_Optimizer)

  fraction_complete = 0


 
  """
  Lambda which determines current learning rate (as a fraction of the maximum).
  """
  def lr_lambda(epoch : int):
    global fraction_complete
    global num_pos
   
    # After warmup phase, the LR is held constant until some fraction of training is complete
    # and thereafter ramps down using a truncated consine decay, terminating around 0.10
    FRAC_START_DECAY = config.Opt_LRBeginDecayAtFractionComplete
    MIN_LR = 0.05
    WARMUP_POS = num_warmup_positions()

    if num_pos < WARMUP_POS:
      return (float(num_pos) / float(WARMUP_POS))**0.5 # inverse square root warmup
    elif fraction_complete < FRAC_START_DECAY:
      return 1.0
    elif fraction_complete > 1:
      return MIN_LR # shouldn't happen
    else:
      # linear deacay to MIN_LR
      slope = (MIN_LR - 1.0) / (1.0 - FRAC_START_DECAY)
      return 1.0 + slope * (fraction_complete - FRAC_START_DECAY)

  scheduler = LambdaLR(optimizer, lr_lambda)

  state = {"model": model, "optimizer": optimizer, "num_pos" : num_pos}


  # Sample code if needed to load from a torchscript model
  if False:
    torchscript_model = torch.jit.load("/mnt/deve/cout/nets/ckpt_DGX_C_256_12_8_6_4bn_B1_2024_vl01_sf_final.ts")
    with torch.no_grad():
      for pytorch_param, torchscript_param in zip(model_nocompile.parameters(), torchscript_model.parameters()):
         pytorch_param.data.copy_(torchscript_param.data)
      # save_model(NAME, OUTPUTS_DIR, config, fabric, model_nocompile, state, "postconvert", True)
    del torchscript_model

       
  # Move model to device. (Lightning's fabric.setup did this implicitly along with
  # DDP wrapping; we are single-process so just .to(device).)
  model = model.to(device)
  if config.Exec_DataType == 'BFloat16Pure':
    model = model.to(torch.bfloat16)

  # Possibly dump summary of model layers.
  DUMP_SUMMARY = False # *** WARNING *** Inexplicably enabling this causes much worse loses (already seen at 5mm pos).
                       # Therefore this should only be enabled to capture the summary, not to include training.
  if DUMP_SUMMARY:
    SUMMARY_DTYPE = torch.float16 # summarize as if float16 because this is the likely target inference type
    SUMMARY_COL_NAMES_TO_SHOW = ("input_size", "output_size", "num_params", "params_percent", "mult_adds", "trainable",)
    model_for_summary = model_nocompile.to(SUMMARY_DTYPE)
    model_stats = summary(model_for_summary,
                          input_data=[torch.rand((256, NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE), dtype=SUMMARY_DTYPE, device=model_for_summary.device),
                                      torch.rand((256, NUM_TOKENS_INPUT, 4), dtype=SUMMARY_DTYPE, device=model_for_summary.device)],
                          dtypes=(SUMMARY_DTYPE, SUMMARY_DTYPE),
                          verbose=2, col_names = SUMMARY_COL_NAMES_TO_SHOW)
    print(model_stats)
    exit(0) # See warning comment above.

  batch_size_forward = config.Opt_BatchSizeForwardPass

  def worker_init_fn(worker_id):
    dataset.set_worker_id(worker_id)

  # Use two concurrent dataset workers (if more than one training data file is available).
  # Override via CERES_NUM_DATASET_WORKERS env var — useful when DataLoader CPU work
  # (zstd decompression + TPG parsing) is the bottleneck. Note: V3 aux features are baked
  # into the TPG record and read directly, so CERES_AUX_FEATURES_PER_SQUARE adds no recompute.
  count_zst_files = len(fnmatch.filter(os.listdir(TPG_TRAIN_DIR), '*.zst'))
  _DEFAULT_NUM_DATASET_WORKERS = 0 if sys.platform.startswith("win") else 1
  NUM_DATASET_WORKERS = int(os.environ.get('CERES_NUM_DATASET_WORKERS', _DEFAULT_NUM_DATASET_WORKERS))
  if NUM_DATASET_WORKERS != _DEFAULT_NUM_DATASET_WORKERS:
    print(f'[train] NUM_DATASET_WORKERS override: {_DEFAULT_NUM_DATASET_WORKERS} -> {NUM_DATASET_WORKERS} (via CERES_NUM_DATASET_WORKERS)')
  PREFETCH_FACTOR = None if NUM_DATASET_WORKERS == 0 else 4 # to keep GPU busy
 
  world_size = len(devices)
  rank = 0 if world_size == 1 else dist.get_rank()
  primary_dataset = TPGDataset(TPG_TRAIN_DIR, batch_size_forward // world_size, config.Data_WDLLabelSmoothing,
                               rank, world_size, NUM_DATASET_WORKERS,
                               BOARDS_PER_BATCH, config.Data_NumTPGFilesToSkip, config.Exec_TestFlag)

  # Optional secondary corpus (e.g. puzzle TPG mixed with T80 self-play).
  # Triggered when both Data_TrainingFilesDirectory2 is set AND Data_RatioSet1ToSet2 > 0.
  secondary_dataset = None
  if (getattr(config, 'Data_TrainingFilesDirectory2', None)
      and int(getattr(config, 'Data_RatioSet1ToSet2', 0) or 0) > 0):
    secondary_dataset = TPGDataset(config.Data_TrainingFilesDirectory2,
                                   batch_size_forward // world_size,
                                   config.Data_WDLLabelSmoothing,
                                   rank, world_size, NUM_DATASET_WORKERS,
                                   BOARDS_PER_BATCH, 0, config.Exec_TestFlag)
    print(f'[mixed-dataset] primary={TPG_TRAIN_DIR}')
    print(f'[mixed-dataset] secondary={config.Data_TrainingFilesDirectory2}')
    print(f'[mixed-dataset] ratio = {config.Data_RatioSet1ToSet2}:1 (primary:secondary batches)')

  dataset = TPGMixedDataset(primary_dataset, secondary_dataset,
                            int(getattr(config, 'Data_RatioSet1ToSet2', 0) or 0))

  dataloader = DataLoader(dataset, batch_size=None, pin_memory=True, num_workers=NUM_DATASET_WORKERS, worker_init_fn=worker_init_fn, prefetch_factor=PREFETCH_FACTOR)
  # NOTE: previously wrapped with fabric.setup_dataloaders to auto-move batches to
  # device. We now move batches explicitly inside the training loop with
  # _move_batch_to_device() — avoids Lightning's recursive _apply_to_collection_slow
  # walk which intermittently wedged at CUDA-sync points.

  config.pretty_print()
  print_model_trainable_details(model)


  NUM_POS_TO_SKIP = 0
  
  COMPUTE_FLOPS = False # WARNING: This is disabled because it causes dramatically higher VRAM usage on GPU 0, use only to generate stats.
  FLOPS_CALCULATED = False
  
  if config.Opt_CheckpointResumeFromFileName is not None:
    loaded = torch.load(config.Opt_CheckpointResumeFromFileName, map_location=device)
   
    # name adjustment sometimes needed for reload
    # loaded["model"] = {f'_orig_mod.{key}': value for key, value in loaded["model"].items()}

    # LoRA / GTAB wrapping introduces extra params not present in orig ckpt.
    # If ANY env-var LoRA or GTAB path is active, use strict=False remap path
    # even when head LoRA (Opt_LoRARankDivisor) is 0.
    _body_attn_div  = int(os.environ.get('CERES_LORA_ATTN_RANK_DIV', '0') or 0)
    _body_ffn_div   = int(os.environ.get('CERES_LORA_FFN_RANK_DIV', '0') or 0)
    _body_legacy    = int(os.environ.get('CERES_LORA_TRANSFORMER_RANK_DIV', '0') or 0)
    _headfront_div  = int(os.environ.get('CERES_LORA_HEADFRONT_RANK_DIV', '0') or 0)
    _smolgen_div    = int(os.environ.get('CERES_LORA_SMOLGEN_RANK_DIV', '0') or 0)
    _gtab_active    = int(os.environ.get('CERES_GTAB', '0') or 0) > 0
    _tsb_active     = bool(getattr(config, 'NetDef_TSB_Enabled', False))
    _body_lora_active = (_body_attn_div > 0 or _body_ffn_div > 0 or _body_legacy > 0
                         or _headfront_div > 0 or _smolgen_div > 0 or _gtab_active
                         or _tsb_active)

    # Load into model_nocompile (the un-wrapped nn.Module). When torch.compile
    # is enabled, `model` is the OptimizedModule wrapper whose state_dict keys
    # are prefixed with `_orig_mod.`, but the saved checkpoint comes from
    # model_nocompile.state_dict() which has un-prefixed keys. Loading into
    # model_nocompile sidesteps the prefix mismatch and updates the underlying
    # parameters that `model` shares.
    if config.Opt_LoRARankDivisor == 0 and not _body_lora_active:
      # load checkpoint parameters, expect all to match (strict = True)
      model_nocompile.load_state_dict(loaded["model"], strict = True)
    else:
      # Rebuild new state dictionary.
      # Mostly copy over parameters from the checkpoint with same name,
      # except if the current model has original_layer
      # (indicating now subsumed within original_layer within LoRA layer)
      # then map to the original name as saved in the pre-LoRA checkpoint.
      new_state_dict = {}

      for name, param in model_nocompile.state_dict().items():
        if "lora_" in name:
          pass # not expected to be found in checkpoint, can start out empty
        elif "tactical_adapter" in name or "tactical_gate" in name:
          pass # GTAB modules are new — not in orig ckpt; keep their init values
        elif "tactical_ffn" in name or ".tsb." in name:
          pass # TSB modules are new — not in orig ckpt; keep their init values
        else:
          # Map to the original name (before it was subsumed within original_layer)
          name_in_checkpoint = name.replace("original_layer.", "") if "original_layer" in name else name
          new_state_dict[name] = loaded["model"][name_in_checkpoint]

      # Load updated state dict
      model_nocompile.load_state_dict(new_state_dict, strict=False)

    # PiSSA re-initialization (if enabled). MUST run after base weights are loaded.
    # Vanilla LoRA init (lora_B=0) is already done inside LoRALinear.__init__; PiSSA
    # overwrites it now using the SVD of the just-loaded base weight, and subtracts
    # the rank-r approximation from the base so the model output at init is unchanged.
    if config.Opt_LoRARankDivisor != 0 and lora.LORA_USE_PISSA:
      lora.apply_pissa_to_model(model)


    # Check all layers for zero parameters
    if config.Opt_CheckpointResumeFromFileName is not None:
      for name, param in model.named_parameters():
        if param.requires_grad:  # Check only trainable parameters
          if torch.all(param == 0):  # Check if all elements in the tensor are zero
            print(f"Note: layer {name} has all zero values. This is expected only for LoRA layers.")

    # Unified optimizer-resume path:
    #   - If the loaded optimizer dict's param_groups match the current optimizer, a
    #     normal load_state_dict works.
    #   - If not (e.g. resuming from a reconstructed checkpoint whose optimizer state
    #     was built on a different model, or LoRA), substitute current param_groups
    #     and load only the 'state' portion (fresh if empty).
    loaded_optimizer_state = loaded["optimizer"]
    current_param_groups = optimizer.param_groups
    loaded_param_groups = loaded_optimizer_state.get("param_groups", [])

    groups_match = (len(current_param_groups) == len(loaded_param_groups)
                    and all(len(cg["params"]) == len(lg["params"])
                            for cg, lg in zip(current_param_groups, loaded_param_groups)))
    if not groups_match:
      print(f"[checkpoint-resume] optimizer param_groups mismatch "
            f"(current={len(current_param_groups)} vs loaded={len(loaded_param_groups)}) — "
            f"substituting current groups, starting optimizer state fresh")
      loaded_optimizer_state["param_groups"] = current_param_groups
      loaded_optimizer_state["state"] = {}
    optimizer.load_state_dict(loaded_optimizer_state)
    
    
    num_pos = int(loaded["num_pos"]) # N.B. be sure to use a multiple of the batch size
    print("INFO: LOAD_CHECKPOINT", config.Opt_CheckpointResumeFromFileName, num_pos)

    # NUM_POS_TO_SKIP = num_pos # enable this line if want to skip training data already seen (but slow)
    del loaded

  # ----------------------------------------------------------------------
  # KL-anchor reference model (optional, RLHF-style fine-tuning regularizer).
  # When config.Opt_KLAnchorRefCheckpoint is set and at least one beta > 0,
  # we load a frozen vanilla CeresNet from the reference checkpoint. Its forward
  # outputs are used as soft targets for KL regularization terms added to the
  # per-batch loss. Reference is NOT compiled, NOT wrapped by fabric.setup,
  # NOT under DDP — pure eval-mode bf16 per rank with no autograd graph.
  # ----------------------------------------------------------------------
  ref_model = None
  kl_active = (config.Opt_KLAnchorRefCheckpoint is not None
               and config.Opt_KLAnchorRefCheckpoint != ""
               and (float(config.Opt_KLAnchorPolicyWeight) > 0.0
                    or float(config.Opt_KLAnchorValueWeight) > 0.0))
  if kl_active:
    print(f"INFO: KL_ANCHOR_REF {config.Opt_KLAnchorRefCheckpoint} "
          f"beta_pol={config.Opt_KLAnchorPolicyWeight} beta_val={config.Opt_KLAnchorValueWeight}")
    # Build a vanilla CeresNet for the reference. Temporarily clear LoRA/GTAB
    # env vars so the reference is constructed without adapters even if the
    # trainable model uses them. Reference must match the saved ckpt's arch.
    _saved_env_keys = ('CERES_LORA_ATTN_RANK_DIV', 'CERES_LORA_FFN_RANK_DIV',
                       'CERES_LORA_TRANSFORMER_RANK_DIV', 'CERES_LORA_HEADFRONT_RANK_DIV',
                       'CERES_LORA_SMOLGEN_RANK_DIV', 'CERES_GTAB',
                       'CERES_LORA_LAYER_MIN', 'CERES_LORA_LAYER_MAX')
    _saved_env = {k: os.environ.get(k, None) for k in _saved_env_keys}
    _saved_lora_div = config.Opt_LoRARankDivisor
    try:
      for k in _saved_env_keys:
        os.environ.pop(k, None)
      config.Opt_LoRARankDivisor = 0
      # Pass the same loss weights as the trainable model so architecture matches
      # the saved checkpoint exactly (heads are conditionally constructed based on
      # whether their loss weight is > 0). Reference is never used for loss compute,
      # but its forward must produce the same shapes as the trainable's forward.
      ref_model = CeresNet(None, config,  # writer=None: ref model never logs
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
    finally:
      config.Opt_LoRARankDivisor = _saved_lora_div
      for k, v in _saved_env.items():
        if v is not None:
          os.environ[k] = v
    ref_loaded = torch.load(config.Opt_KLAnchorRefCheckpoint, map_location=device)
    # strict=False: the saved checkpoint may contain extra heads (e.g. value2_head)
    # that the current config doesn't enable — those keys are unused by the reference
    # forward (only policy_out and value_out matter for KL) and can be discarded.
    # Missing keys would still indicate a real mismatch — log them for visibility.
    ref_load_result = ref_model.load_state_dict(ref_loaded["model"], strict=False)
    if (ref_load_result.missing_keys or ref_load_result.unexpected_keys):
      print(f"INFO: KL_ANCHOR_REF_LOAD missing={len(ref_load_result.missing_keys)} "
            f"unexpected={len(ref_load_result.unexpected_keys)}")
      if ref_load_result.missing_keys:
        print(f"  missing: {ref_load_result.missing_keys[:5]}{'...' if len(ref_load_result.missing_keys) > 5 else ''}")
      if ref_load_result.unexpected_keys:
        print(f"  unexpected: {ref_load_result.unexpected_keys[:5]}{'...' if len(ref_load_result.unexpected_keys) > 5 else ''}")
    del ref_loaded
    ref_model = ref_model.to(device).to(torch.bfloat16)
    ref_model.eval()
    for p in ref_model.parameters():
      p.requires_grad_(False)
    print("INFO: KL_ANCHOR_REF_LOADED")

  # compute batch sizes
  batch_size_opt = config.Opt_BatchSizeBackwardPass
  assert batch_size_opt >= batch_size_forward and batch_size_opt % batch_size_forward == 0, 'data batch size must be be multiple of optimization batch size'
  num_batches_gradient_accumulate = batch_size_opt // batch_size_forward
  batch_accumulation_counter = 0
  last_save_model_pos = 0

  loss_calc = LossCalculator(model)

  model.train()

  wdl_reverse = torch.tensor([2, 1, 0]) # for reversing perspective on WDL
  

  # Train Network
  for batch_idx, (batch) in enumerate(dataloader):
    if (num_pos >= MAX_POSITIONS and not config.Exec_ExportOnly):
        break

    # Move the freshly-fetched batch to GPU. Replaces Lightning's recursive
    # auto-move; with pin_memory=True these are true async DMA transfers.
    batch = _move_batch_to_device(batch, device)

    fraction_complete = num_pos / MAX_POSITIONS
    model.train()

    # Periodically log statistics
    show_losses = (num_pos % (1024 * 64) == 0)

    is_accumulating = ((batch_accumulation_counter + 1) % num_batches_gradient_accumulate) != 0
    # Single-GPU: no DDP sync skipping needed. Autocast handles bf16-mixed precision
    # (Fabric did this implicitly via precision='bf16-mixed').
    _amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_AUTOCAST else contextlib.nullcontext()
    with _amp_ctx:
      this_lr = scheduler.get_last_lr()[0]

      if config.Exec_ExportOnly:
        assert config.Opt_CheckpointResumeFromFileName is not None, "ExportOnly specified but no checkpoint file specified"
        print("Exporting to files with postexport suffix....")
        save_model(NAME, OUTPUTS_DIR, config, model_nocompile, state, "postexport", True)
        print("INFO: EXIT_STATUS", "SUCCESS")
        exit(3)

      if COMPUTE_FLOPS and not FLOPS_CALCULATED and torch.cuda.is_available():
        calc_flops(model_nocompile.to(torch.float), batch[0], loss_calc, optimizer, num_pos, config.Opt_BatchSizeForwardPass, calc_backward=False)
        calc_flops(model_nocompile.to(torch.float), batch[0], loss_calc, optimizer, num_pos, config.Opt_BatchSizeForwardPass, calc_backward=True)
        optimizer.zero_grad()
        FLOPS_CALCULATED = True

        
      if BOARDS_PER_BATCH == 1:
        batch = batch[0]
        num_processing_now = batch['squares'].shape[0]
        policy_out, value_out, moves_left_out, unc_out, value2_out, q_deviation_lower, q_deviation_upper, uncertainty_policy_out, _, _, _ = model(batch['squares'], None)
        loss = model.compute_loss(loss_calc, batch, policy_out, value_out, moves_left_out, unc_out,
                                  value2_out, q_deviation_lower, q_deviation_upper, uncertainty_policy_out, 
                                  None, None, 
                                  None, None,
                                  None,
                                  0, num_pos, this_lr, show_losses)

      else:
        assert BOARDS_PER_BATCH == 4

        # Weights for the action loss terms.
        # The training data has 2 positions which are always optimal (or nearly optimal) moves
        # for every 1 which more evenly distributed over possible moves (of all quality).
        # To compensate for this non-representative training data distribution,
        # we give less weight to the over-sampled best continuation moves.
        # Note the logic below references Value and not Value2 as the target for the action head
        # It was found that Value2 is too noisy to make a good target, using it yields approx -50Elo       
        LOSS_WEIGHT_ACTION_BEST_CONTINUATION = 0.15
        LOSS_WEIGHT_ACTION_RANDOM_CONTINUATION = 1.0
        
        num_processing_now = batch[0]['squares'].shape[0] * BOARDS_PER_BATCH
        
        #Board 1
        sub_batch = batch[0]
        policy_out1, value_out1, moves_left_out1, unc_out1, value2_out1,  q_deviation_lower1, q_deviation_upper1, uncertainty_policy_out1, action_out1, state_out1, action_uncertainty_out1 = model(sub_batch['squares'], None)
        loss1 = model.compute_loss(loss_calc, sub_batch, policy_out1, value_out1, moves_left_out1, unc_out1,
                                   value2_out1, q_deviation_lower1, q_deviation_upper1, uncertainty_policy_out1, 

                                   None, None, 
                                   None, None, 
                                   action_uncertainty_out1,
                                   
                                   0, num_pos, this_lr, show_losses)
        
        # Board 2
        sub_batch = batch[1]
        policy_out2, value_out2, moves_left_out2, unc_out2, value2_out2, q_deviation_lower2, q_deviation_upper2, uncertainty_policy_out2, action_out2, state_out2, action_uncertainty_out2 = model(sub_batch['squares'], state_out1)

        if config.Opt_LossActionMultiplier > 0:
          action2_played_move_indices = sub_batch['policy_index_in_parent'].to(dtype=torch.int)
          extracted_action1_out = action_out1[torch.arange(0, action_out1.size(0)), action2_played_move_indices.squeeze(-1)]
          extracted_action1_out = extracted_action1_out[:, wdl_reverse]
        else:
          extracted_action1_out = None
          
        loss2 = model.compute_loss(loss_calc, sub_batch, policy_out2, value_out2, moves_left_out2, unc_out2,
                                   value2_out2, q_deviation_lower2, q_deviation_upper2, uncertainty_policy_out2, 

                                   value_out1[:, wdl_reverse], value2_out1[:, wdl_reverse], # prior value outputs for value differencing
                                   value_out2.detach(), extracted_action1_out,  # action target/output from previous board
                                   action_uncertainty_out2,
                                   
                                   LOSS_WEIGHT_ACTION_BEST_CONTINUATION, num_pos, this_lr, show_losses)
        
        # Board 3
        sub_batch = batch[2]
        policy_out3, value_out3, moves_left_out3, unc_out3, value2_out3, q_deviation_lower3, q_deviation_upper3, uncertainty_policy_out3, action_out3, _, action_uncertainty_out3 = model(sub_batch['squares'], state_out2)

        if config.Opt_LossActionMultiplier > 0:
          action3_played_move_indices = sub_batch['policy_index_in_parent'].to(dtype=torch.int)
          extracted_action2_out = action_out2[torch.arange(0, action_out2.size(0)), action3_played_move_indices.squeeze(-1)]
          extracted_action2_out = extracted_action2_out[:, wdl_reverse]
        else:
          extracted_action2_out = None

        loss3 = model.compute_loss(loss_calc, sub_batch, policy_out3, value_out3, moves_left_out3, unc_out3,
                                   value2_out3, q_deviation_lower3, q_deviation_upper3, uncertainty_policy_out3,

                                   value_out2[:, wdl_reverse], value2_out2[:, wdl_reverse], # prior value outputs for value differencing
                                   value_out3.detach(), extracted_action2_out, # action target/output from previous board
                                   action_uncertainty_out3,

                                   LOSS_WEIGHT_ACTION_BEST_CONTINUATION, num_pos, this_lr, show_losses)

        # Board 4 (only used if action loss is enabled)
        if config.Opt_LossActionMultiplier > 0:
          sub_batch = batch[3]
          policy_out4, value_out4, moves_left_out4, unc_out4, value2_out4, q_deviation_lower4, q_deviation_upper4, uncertainty_policy_out4, action_out4, _, action_uncertainty_out4 = model(sub_batch['squares'], state_out1)


          action4_played_move_indices = sub_batch['policy_index_in_parent'].to(dtype=torch.int)
          extracted_action1_other_out = action_out1[torch.arange(0, action_out1.size(0)), action4_played_move_indices.squeeze(-1)]
          extracted_action1_other_out = extracted_action1_other_out[:, wdl_reverse]
          
          loss4 = model.compute_loss(loss_calc, sub_batch, None, None, None, None,
                                     None, None, None, None,

                                     None, None,
                                     value_out4.detach(), extracted_action1_other_out, # action target/output from previous board
                                     action_uncertainty_out4,
                                     
                                     LOSS_WEIGHT_ACTION_RANDOM_CONTINUATION, num_pos, this_lr, show_losses)

        if config.Opt_LossActionMultiplier > 0:
          loss = (loss1 + loss2 + loss3 + loss4) / 3 # although there are 4 loss terms, the last one is typically very small so we only divide by 3
        else:
          loss = (loss1 + loss2 + loss3) / 3 # only 3 boards used

      # GTAB gate-sparsity regularizer: penalize unnecessary gate firing.
      # The gate's last value was cached in the forward pass. Adding mean(g)
      # to the loss encourages the gate to stay near 0 unless the puzzle loss
      # gain from firing exceeds the sparsity cost. Default lambda 0.01.
      if getattr(model, 'use_gtab', False) and getattr(model, '_last_gate_value', None) is not None:
        _gtab_lambda = float(os.environ.get('CERES_GTAB_GATE_LAMBDA', '0.01') or 0.01)
        loss = loss + _gtab_lambda * model._last_gate_value.mean()

      # TSB gate-sparsity regularizer: penalize unnecessary tactical-branch firing.
      # Each transformer block has a per-block scalar gate; the net stacks them
      # into _last_tsb_gates of shape [num_layers, B, 1, 1]. Mean across layers and
      # batch encourages each block's gate to stay near 0 unless the puzzle loss
      # gain from firing exceeds the sparsity cost. Default lambda 0.01.
      if getattr(model, 'use_tsb', False) and getattr(model, '_last_tsb_gates', None) is not None:
        _tsb_lambda = float(os.environ.get('CERES_TSB_GATE_LAMBDA', '0.01') or 0.01)
        loss = loss + _tsb_lambda * model._last_tsb_gates.mean()

      # KL-divergence anchor: pull student outputs toward frozen reference outputs.
      # For BOARDS_PER_BATCH==4 we anchor only on board 1 (canonical learned target;
      # 4x cheaper, and boards 2-4 are sequence-conditioned so a single anchor suffices).
      # Policy KL is computed on legal moves only (illegal-move logits are arbitrary
      # and would dominate the divergence). Both terms upcast to float32 for numerical
      # stability — bf16 KL is too noisy.
      if kl_active:
        if BOARDS_PER_BATCH == 1:
          _anchor_squares = batch['squares']
          _anchor_policies = batch['policies']
          _student_pol = policy_out
          _student_val = value_out
        else:
          _anchor_squares = batch[0]['squares']
          _anchor_policies = batch[0]['policies']
          _student_pol = policy_out1
          _student_val = value_out1

        with torch.no_grad():
          # Reference is bf16 but the dataloader-side input is float32
          # (the trainable model is auto-cast by fabric; the reference is not).
          _ref_input = _anchor_squares.to(torch.bfloat16)
          _ref_pol, _ref_val, *_ = ref_model(_ref_input, None)

        _beta_pol = float(config.Opt_KLAnchorPolicyWeight)
        _beta_val = float(config.Opt_KLAnchorValueWeight)
        _kl_pol_val = None
        _kl_val_val = None

        if _beta_pol > 0.0:
          _legal_mask = (_anchor_policies > 0)
          _NEG = -1e4
          _sp_masked = torch.where(_legal_mask, _student_pol, torch.full_like(_student_pol, _NEG))
          _rp_masked = torch.where(_legal_mask, _ref_pol, torch.full_like(_ref_pol, _NEG))
          _log_sp = F.log_softmax(_sp_masked.float(), dim=-1)
          _log_rp = F.log_softmax(_rp_masked.float(), dim=-1)
          _kl_pol = (_log_sp.exp() * (_log_sp - _log_rp)).sum(-1).mean()
          loss = loss + _beta_pol * _kl_pol
          _kl_pol_val = _kl_pol.detach().item()

        if _beta_val > 0.0:
          _log_sv = F.log_softmax(_student_val.float(), dim=-1)
          _log_rv = F.log_softmax(_ref_val.float(), dim=-1)
          _kl_val = (_log_sv.exp() * (_log_sv - _log_rv)).sum(-1).mean()
          loss = loss + _beta_val * _kl_val
          _kl_val_val = _kl_val.detach().item()

        if show_losses:
          print(f"KL_ANCHOR pol={_kl_pol_val if _kl_pol_val is not None else 'off'} "
                f"val={_kl_val_val if _kl_val_val is not None else 'off'}")

    # Backward outside the autocast context (standard practice; bf16-mixed needs no
    # gradient scaling unlike fp16, so plain loss.backward() is correct).
    loss.backward()

    if not is_accumulating:
      if config.Opt_GradientClipLevel > 0:
        # NOTE: we deliberately do NOT pass error_if_nonfinite=True here. Lightning
        # Fabric's clip_gradients defaulted that to True, which forced a CUDA-sync
        # NaN/Inf check on every step and intermittently wedged on slow CUDA syncs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.Opt_GradientClipLevel)
      scheduler.step()

#      GRAD_NORM_LOG_FREQUENCY = 200
#      if (num_pos // BATCH_SIZE) % GRAD_NORM_LOG_FREQUENCY == GRAD_NORM_LOG_FREQUENCY - 1:
      on_before_optimizer_step(writer, model, optimizer, num_pos)

      optimizer.step()
      optimizer.zero_grad()

    batch_accumulation_counter = batch_accumulation_counter + 1

    # update number of positions processed (single-GPU; world_size=1)
    num_pos = num_pos + num_processing_now
    num_batches = num_pos // BATCH_SIZE

    # Emit checkpoint when specified interval has passed since last save.
    # Previously also gated by `num_batches % (CheckpointFreq // BATCH_SIZE) == 0`,
    # which silently doubled the effective interval when CheckpointFreq wasn't
    # divisible by BATCH_SIZE (e.g. 100M / 2048 → modulo only matches at exact
    # multiples of 48828 batches, which combined with the diff-threshold made the
    # first checkpoint fire at ~200M instead of ~100M). The modulo check was
    # only ever needed for cross-rank synchronization in multi-GPU runs; this is
    # single-GPU, so the diff-threshold alone is correct and safer.
    if config.Opt_CheckpointFrequencyNumPositions > 0 and (num_pos - last_save_model_pos >= config.Opt_CheckpointFrequencyNumPositions):
      save_checkpoint(NAME, OUTPUTS_DIR, config, model_nocompile, state, str(num_pos))
      save_model(NAME, OUTPUTS_DIR, config, model_nocompile, state, str(num_pos), True)
      last_save_model_pos = num_pos

    current_time = datetime.datetime.now()

    global time_start
    global time_last_status_update
    global time_last_save_transient

    time_since_start = (current_time - time_start).seconds
    time_since_status_update = (current_time - time_last_status_update).seconds
    time_since_save_transient = (current_time - time_last_save_transient).seconds

    STATUS_UPDATE_INTERVAL = 10 # log output to console every 10 seconds
    should_show_status = (time_since_status_update > STATUS_UPDATE_INTERVAL) or (num_pos >= MAX_POSITIONS)
  
    # save output artifacts (except checkpoint file) every 120 (or 30 if LoRA) minutes (with label "last")    
    SAVE_LAST_INTERVAL = 120 * 60 if config.Opt_LoRARankDivisor == 0 else 30 * 60
    should_save_transient = time_since_save_transient > SAVE_LAST_INTERVAL
    if should_save_transient:
      save_model(NAME, OUTPUTS_DIR, config, model_nocompile, state, "last", True)
      time_last_save_transient  = datetime.datetime.now()

    if should_show_status:
      # Note that this code executes only for primary worker (if multi-GPU),
      # and the statistics are collected over the recent training history only for that worker.
      # Although incomplete, the resulting statistics should nevertheless be reasonably accurate.
      total_loss =  (config.Opt_LossPolicyMultiplier * loss_calc.LAST_POLICY_LOSS
                    + config.Opt_LossValueMultiplier * loss_calc.LAST_VALUE_LOSS
                    + config.Opt_LossValue2Multiplier * loss_calc.LAST_VALUE2_LOSS
                    + config.Opt_LossMLHMultiplier * loss_calc.LAST_MLH_LOSS
                    + config.Opt_LossUNCMultiplier * loss_calc.LAST_UNC_LOSS
                    + config.Opt_LossQDeviationMultiplier * loss_calc.LAST_Q_DEVIATION_LOWER_LOSS       
                    + config.Opt_LossQDeviationMultiplier * loss_calc.LAST_Q_DEVIATION_UPPER_LOSS       
                    + config.Opt_LossUncertaintyPolicyMultiplier * loss_calc.LAST_UNCERTAINTY_POLICY_LOSS
                     
                    + config.Opt_LossValueDMultiplier * loss_calc.LAST_VALUE_DIFF_LOSS
                    + config.Opt_LossValue2DMultiplier * loss_calc.LAST_VALUE2_DIFF_LOSS
                     
                    + config.Opt_LossActionMultiplier * loss_calc.LAST_ACTION_LOSS)

        
      # Note that this output line is parsed by the C# class CeresTrainProgressLoggingLine
      print("TRAIN:", num_pos, ",", 
            total_loss, ",", 
            loss_calc.LAST_VALUE_LOSS if config.Opt_LossValueMultiplier > 0 else 0, ",", 
            loss_calc.LAST_POLICY_LOSS if config.Opt_LossPolicyMultiplier > 0 else 0, ",", 
            loss_calc.LAST_VALUE_ACC if config.Opt_LossValueMultiplier > 0 else 0, ",", 
            loss_calc.LAST_POLICY_ACC if config.Opt_LossPolicyMultiplier > 0 else 0, ",", 
            loss_calc.LAST_MLH_LOSS if config.Opt_LossMLHMultiplier > 0 else 0, ",",  
            loss_calc.LAST_UNC_LOSS if config.Opt_LossUNCMultiplier > 0 else 0, ",", 
            loss_calc.LAST_VALUE2_LOSS if config.Opt_LossValue2Multiplier > 0 else 0, ",", 
            loss_calc.LAST_Q_DEVIATION_LOWER_LOSS if config.Opt_LossQDeviationMultiplier > 0 else 0, ",", 
            loss_calc.LAST_Q_DEVIATION_UPPER_LOSS if config.Opt_LossQDeviationMultiplier > 0 else 0, ",", 
            loss_calc.LAST_UNCERTAINTY_POLICY_LOSS if config.Opt_LossUncertaintyPolicyMultiplier > 0 else 0, ",", 

            loss_calc.LAST_VALUE_DIFF_LOSS if config.Opt_LossValueDMultiplier > 0 else 0, ",", 
            loss_calc.LAST_VALUE2_DIFF_LOSS if config.Opt_LossValue2DMultiplier > 0 else 0, ",", 

            loss_calc.LAST_ACTION_LOSS if config.Opt_LossActionMultiplier > 0 else 0, ",",
            loss_calc.LAST_ACTION_UNCERTAINTY_LOSS if config.Opt_LossActionUncertaintyMultiplier > 0 else 0, ",",
              
            scheduler.get_last_lr()[0], flush=True)
      loss_calc.reset_counters()
      time_last_status_update = datetime.datetime.now()

  # final save and convert to Torchscript
  save_checkpoint(NAME, OUTPUTS_DIR, config, model_nocompile, state, str(num_pos))
  save_model(NAME, OUTPUTS_DIR, config, model_nocompile, state, str(num_pos), True)

  writer.flush()
  writer.close()
  print("INFO: EXIT_STATUS", "SUCCESS")

Train()

