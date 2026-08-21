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
import re
import sys
import socket
import datetime
import math
import contextlib
import json as _bootstrap_json

# ---- Config -> env bootstrap (BEFORE the heavy imports) -------------------
# The data-format and survival-sidecar settings are consumed at IMPORT time by
# module-level reads (config.py, tpg_dataset.py, losses.py), so plain
# Configuration fields would arrive too late. The bridge lives in
# config_bootstrap so the standalone checkpoint tools (recover_export.py,
# reconvert_onnx.py) apply the IDENTICAL mapping — they rebuild nets from
# checkpoints these same settings shape, and a tool that disagrees with the
# training run re-exports a differently-shaped net (the 2026-08-07 incident).
from config_bootstrap import bootstrap_env_from_config
if len(sys.argv) > 2:
  bootstrap_env_from_config(sys.argv[2], sys.argv[1])
# ---------------------------------------------------------------------------

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
from torch.nn.parallel import DistributedDataParallel as DDP

from rms_norm import RMSNorm
from derf_norm import DerfNorm
from dyt_norm import DyTNorm
from losses import LossCalculator
from tpg_dataset import TPGDataset, TPGMixedDataset, TPG_TARGET_SIDECAR_MODE
from config import Configuration, split_roots
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
else:
  # ONLY DirectFromV6 accepts MULTIPLE ';'-separated roots (v6_dataset.py);
  # a multi-part value under any other SourceType would pass per-part
  # validation here and then die deep in dataset init on the raw joined
  # string (review 2026-08-21 finding 8). Empty/separator-only values are
  # rejected too (they used to skip the loop and fail late).
  _dir_parts = split_roots(TPG_TRAIN_DIR)
  if not _dir_parts:
    print(f"ERROR: TrainingFilesDirectory ('{TPG_TRAIN_DIR}') names no directories.")
    exit(1)
  if len(_dir_parts) > 1 and str(getattr(config, 'Data_SourceType', '') or '') != 'DirectFromV6':
    print(f"ERROR: multiple ';'-separated TrainingFilesDirectory roots are only "
          f"supported with SourceType DirectFromV6 (got {config.Data_SourceType!r}).")
    exit(1)
  for _dir_part in _dir_parts:
    if not os.path.isdir(_dir_part):
      print('ERROR: TrainingFilesDirectory does not exist:', _dir_part)
      exit(1)
    if next(os.scandir(_dir_part), None) is None:
      print(f"ERROR: The directory TrainingFilesDirectory ('{_dir_part}') is empty.")
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

# ---------------------------------------------------------------------------
# Distributed (multi-GPU) setup — OPT-IN via torchrun. When NOT launched under
# torchrun, WORLD_SIZE is unset/1 and everything below runs EXACTLY as before:
# one process, rank 0, world_size 1 (the long-standing single-GPU path).
#
#   Real multi-GPU :  torchrun --nproc_per_node=4 train.py <ID> <OUT>
#   Single-GPU sim :  CERES_DDP_BACKEND=gloo torchrun --nproc_per_node=2 train.py <ID> <OUT>
#                     (NCCL wants one GPU per rank and hangs if ranks share a
#                      device, so the simulation uses the gloo CPU backend and
#                      maps every rank onto cuda:0. Validates all DDP control
#                      flow — grad all-reduce, no_sync accumulation, rank-0 save
#                      gating, data sharding, global position counting — but
#                      gives no speedup and needs N model copies to fit in VRAM.)
#
# Optimizer note: the Muon variant in muon.py updates each param from its local
# .grad; DDP all-reduces (averages) grads across ranks BEFORE optimizer.step(),
# so Muon sees the global-mean gradient — no double reduction (its
# `import torch.distributed` is unused). AdEMAMixShampoo DOES call all_reduce
# itself — review separately before running Shampoo under DDP.
# ---------------------------------------------------------------------------
def _setup_distributed():
  ws = int(os.environ.get('WORLD_SIZE', '1') or 1)
  if ws <= 1 or config.Exec_ExportOnly:
    return False, 0, 0, 1
  rank = int(os.environ.get('RANK', '0') or 0)
  local_rank = int(os.environ.get('LOCAL_RANK', '0') or 0)
  backend = os.environ.get('CERES_DDP_BACKEND', 'nccl')
  dist.init_process_group(backend=backend, init_method='env://')
  if torch.cuda.is_available():
    gpu = local_rank if local_rank < torch.cuda.device_count() else 0
    torch.cuda.set_device(gpu)
  return True, rank, local_rank, ws

IS_DISTRIBUTED, RANK, LOCAL_RANK, WORLD_SIZE = _setup_distributed()
IS_MASTER = (RANK == 0)
# GPU ordinal this rank trains on. Normally one GPU per rank (cuda:local_rank);
# in the single-GPU simulation (more ranks than GPUs) every rank shares cuda:0.
DDP_LOCAL_GPU = (LOCAL_RANK if (IS_DISTRIBUTED and torch.cuda.is_available()
                                and LOCAL_RANK < torch.cuda.device_count()) else 0)
if IS_DISTRIBUTED:
  print(f"[ddp] rank={RANK}/{WORLD_SIZE} local_rank={LOCAL_RANK} gpu=cuda:{DDP_LOCAL_GPU} "
        f"backend={os.environ.get('CERES_DDP_BACKEND','nccl')} master={IS_MASTER}", flush=True)


class _NoOpWriter:
  """Stand-in for SummaryWriter on non-master ranks: silently drops all logging
  so only rank 0 writes tensorboard + console output. Prevents 4× duplicate
  TRAIN lines that the C# CeresTrainProgressLoggingLine parser would choke on."""
  def add_scalar(self, *a, **k): pass
  def add_histogram(self, *a, **k): pass
  def add_text(self, *a, **k): pass
  def flush(self): pass
  def close(self): pass

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

    # Only rank 0 logs grad/weight diagnostics. (calc_weight_update_ratio also
    # retains a full extra copy of the params — no reason to pay that on 3 ranks.)
    if not IS_MASTER:
      return

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

  # Optional run seed (config Opt 'TorchSeed'; env CERES_TORCH_SEED as fallback).
  # Unset = previous behaviour: no seeding at all, so weight init, dropout
  # masks, hard-replay/mirror row sampling and the file-mirror coin flips all
  # differ between runs — the ±2-3 Elo "seed noise" measured on same-config
  # arms. Setting it makes two arms start from IDENTICAL weights and take the
  # same stochastic decisions, so a gate delta is attributable to the change
  # under test rather than to init luck. Pair with ShuffleSeed (data order) for
  # a fully controlled A/B.
  #
  # Scope note: this is reproducibility BETWEEN RUNS, not bit-exact
  # determinism — that additionally needs torch.use_deterministic_algorithms
  # and disabling cuDNN autotuning, which costs throughput and is deliberately
  # not done here.
  #
  # DDP: every rank seeds IDENTICALLY. Weight init must match across ranks
  # (DDP broadcasts parameters at construction anyway, so this only makes the
  # pre-broadcast state agree), and data divergence is already handled by the
  # rank-partitioned file slices, not by RNG.
  _torch_seed = getattr(config, 'Opt_TorchSeed', None)
  if _torch_seed in (None, ''):
    _torch_seed = os.environ.get('CERES_TORCH_SEED')
  if _torch_seed not in (None, ''):
    _torch_seed = int(_torch_seed)
    import random as _py_random
    torch.manual_seed(_torch_seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(_torch_seed)
    np.random.seed(_torch_seed & 0xFFFFFFFF)
    _py_random.seed(_torch_seed)
    print(f'[train] TORCH SEED set to {_torch_seed} (weight init + dropout + sampling; '
          f'identical on every rank). Data order is governed separately by ShuffleSeed.',
          flush=True)

  if config.Exec_UseFP8:
    raise NotImplementedError(
        "Exec_UseFP8 was previously supported via Lightning Fabric's TransformerEnginePrecision; "
        "the plain-PyTorch path does not wire transformer-engine directly. "
        "Use Exec_DataType='BFloat16' instead.")

  # Plain-PyTorch device + tensorboard setup. Replaces Lightning Fabric.
  if IS_DISTRIBUTED:
    device = torch.device(f"cuda:{DDP_LOCAL_GPU}")
  else:
    # ExportOnly collapses `devices` to a bare int (line ~131, Fabric-era convention).
    _dev0 = devices[0] if isinstance(devices, (list, tuple)) else devices
    device = torch.device(f"{accelerator}:{_dev0}" if accelerator != 'cpu' else 'cpu')
  # Only rank 0 writes tensorboard/console; other ranks get a silent no-op writer.
  writer = SummaryWriter(os.path.join(OUTPUTS_DIR, 'tblogs', NAME)) if IS_MASTER else _NoOpWriter()
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
    #   - private value front-end in 'inject' mode (value_premap + the zero-init
    #     injectors). These are NEW modules with no base weights to protect, and
    #     they contribute exactly zero at step 0, so they train at full rank
    #     rather than through an adapter — rank-limiting them would throttle the
    #     very pathway the run is testing. See config ValueHeadChannelsMode.
    for name, param in model.named_parameters():
      keep_trainable = ("lora_A" in name or "lora_B" in name or "lora_alpha" in name
                        or "tactical_adapter" in name or "tactical_gate" in name
                        or "tactical_ffn" in name or ".tsb." in name
                        or "value_premap" in name or "_priv_inject" in name
                        # Vis edge-bias modules (NetDef UseVisEdgeBias/VisEdgeGates):
                        # NEW zero-init modules with no base weights to protect —
                        # same full-rank rationale as the 'inject' front-end above.
                        # Without this they freeze at zero and the feature is a
                        # silent no-op in every LoRA/GTAB/TSB run.
                        or "vis_edge_proj" in name or "attack_gate_" in name
                        # Graph-route heads + tactic refiner (2026-08 tactical
                        # program): same NEW-zero-init-module rationale.
                        or "graph_route" in name or "tactical_refiner" in name
                        # Value min/max pool injectors (NetDef ValueHeadMinMaxPool):
                        # same NEW-zero-init-module rationale.
                        or "_pool_inject" in name
                        # Dual-plane P-plane + decode/aux/attention couplings and
                        # the 2026-08-20/21 input/pattern modules (kdist/spectral
                        # PE/codebook): same NEW-module rationale — review finding
                        # 2026-08-20 #5 (they froze at zero under LoRA/GTAB/TSB).
                        or "dual_plane" in name or "dp_value" in name
                        or "dp_pol_" in name or "dpva_" in name or "dp_surv" in name
                        or "dpv_" in name or "dpe_w" in name or "dpd_" in name
                        or "kdist_proj" in name or "spe_proj" in name or "cbk_" in name)
      if not keep_trainable:
        param.requires_grad = False
   
  # Per-head QK-clip (config 'QKClipTau', see config.py): arm the per-module
  # max-logit monitors BEFORE torch.compile so the training-only stash branch
  # specializes into the compiled graph. Module refs kept for the post-step clip.
  QK_CLIP_TAU = float(getattr(config, 'Opt_QKClipTau', 0) or 0)
  _qk_clip_mods = []
  _qk_clip_last_report = [-10**18]  # mutable so the train loop can throttle QKCLIP prints
  _qk_clip_sync = None              # cross-rank stash reduction (DDP only); None = single-GPU
  if QK_CLIP_TAU > 0:
    from dot_product_attention import DotProductAttention
    for _mod in model.modules():
      if isinstance(_mod, DotProductAttention):
        _mod.qk_clip_monitor = True
        _qk_clip_mods.append(_mod)
    if not _qk_clip_mods:
      raise ValueError('QKClipTau set but no DotProductAttention modules found')
    if IS_DISTRIBUTED:
      # Each rank observes the per-head max logit over ITS OWN microbatch. Clipping on
      # those independently would rescale different heads by different factors on each
      # rank and silently diverge the replicas — which is why this combination used to
      # be refused outright. Reducing the stashes with MAX before the clip removes the
      # problem and is also the semantically correct statistic: the max over the global
      # batch is exactly what a single-GPU run of the same batch would have seen.
      #
      # Deadlock-safety: the buffer is sized from the module list (identical on every
      # rank, fixed for the whole run), so all ranks always enter the collective with
      # the same shape regardless of which stashes happen to be populated. Missing
      # stashes contribute 0, which yields gamma=1 => no clip. The reduced values are
      # written back UNCONDITIONALLY, so every rank clips from byte-identical input
      # even in the edge case where a stash exists on some ranks but not others.
      _qk_clip_sizes = [int(_m.num_heads) for _m in _qk_clip_mods]
      _qk_clip_dev = torch.device('cpu') if dist.get_backend() == 'gloo' else device
      _qk_clip_buf = torch.zeros(sum(_qk_clip_sizes), dtype=torch.float32, device=_qk_clip_dev)

      def _qk_clip_sync():
        _qk_clip_buf.zero_()
        _off = 0
        for _m, _n in zip(_qk_clip_mods, _qk_clip_sizes):
          _v = getattr(_m, '_last_max_logit', None)
          if _v is not None:
            _qk_clip_buf[_off:_off + _n] = _v.detach().float().to(_qk_clip_dev)
          _off += _n
        dist.all_reduce(_qk_clip_buf, op=dist.ReduceOp.MAX)
        _off = 0
        for _m, _n in zip(_qk_clip_mods, _qk_clip_sizes):
          _m._last_max_logit = _qk_clip_buf[_off:_off + _n].to(next(_m.parameters()).device)
          _off += _n
    print(f'[train] QK-CLIP armed: tau={QK_CLIP_TAU}, {len(_qk_clip_mods)} attention modules '
          f'(per-head weight rescale after optimizer step)'
          + (f'; DDP: stashes MAX-reduced across {WORLD_SIZE} ranks so every replica clips identically'
             if IS_DISTRIBUTED else ''), flush=True)

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
          elif "vda_query" in fpn: # depth-attention pseudo-query (bare 1-D vector, bias-like)
              no_decay.add(fpn)
          elif "rc_btype" in fpn or "rc_u" in fpn or "rc_v" in fpn or "rc_w" in fpn: # ray-context bare vectors (bias-like)
              no_decay.add(fpn)
          elif "rc_W" in fpn: # ray-context projections (plain Linear weights)
              decay.add(fpn)
          elif fpn.endswith('softmin_log_tau') or fpn.endswith('softmax_log_tau') \
                or fpn.endswith('head_logit_temp'):
              # Log-scale mechanism params (trunk soft-agg taus, per-head logit
              # temps, P-plane taus): bias-like 1-D log params. MUST precede the
              # 'transformer_layer' catch-all — review finding 2026-08-20 #6:
              # weight decay was pulling these toward their 1.0 no-op inits,
              # regularizing away the very mechanisms under test.
              no_decay.add(fpn)
          elif "dual_plane" in fpn and "log_tau" in fpn:
              # P-plane soft-min temperatures: bias-like 1-D log params.
              no_decay.add(fpn)
          elif "cbk_keys" in fpn or "cbk_vals" in fpn:
              # Tactical-codebook motif tables: embedding-like raw matrices
              # (row = motif), not projection weights — follow the embedding
              # convention (no decay; also keeps them out of Muon's
              # orthogonalization, which targets true weight matrices).
              no_decay.add(fpn)
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
    _muon_scope = getattr(config, 'Opt_MuonAdamWScope', 'all-non-trunk')
    if _muon_scope == 'final-only':
      # dje-style partition: Muon drives ALL hidden 2-D matrices (trunk, headPremap,
      # headSharedLinear, each Head's hidden fc, smolgen prep); the internal AdamW
      # gets only final output layers (fcFinal, single-Linear aux heads), embeddings,
      # LoRA adapters and 1-D params (norms/biases).
      def _use_muon(n, p):
        if p.ndim != 2: return False              # Muon handles exactly-2-D matrices (its ctor asserts); norms/biases and any >=3-D exotic go AdamW
        if 'embedding' in n: return False         # lookup-table-like: AdamW
        if 'cbk_keys' in n or 'cbk_vals' in n: return False  # codebook motif tables: embedding-like rows, AdamW
        if 'fcFinal' in n: return False           # each Head's final output layer: AdamW
        if 'placement_value_head' in n or 'survival_head' in n or 'stvalue_head' in n or 'dp_surv_head' in n: return False  # single-Linear aux heads ARE final layers
        if 'lora' in n.lower(): return False      # low-rank adapters: orthogonalized updates unsuitable
        return True
    elif _muon_scope == 'all-non-trunk':
      # Legacy partition: Muon only for 2-D trunk params; everything else AdamW.
      # ndim == 2 (not >= 2): Muon's ctor asserts exactly-2-D, so any >=3-D
      # trunk param (e.g. the [H, C, d_k] vis edge-bias gate tensors) must go
      # to the internal AdamW group — same rule the 'final-only' scope applies.
      def _use_muon(n, p):
        return p.ndim == 2 and 'embedding' not in n and 'transformer_layer' in n
    elif _muon_scope == 'ffn-only':
      # Kovax-partisjon (2026-08-20, "Nadam for Attention and muon for FFN",
      # AdamW substituted for NAdam by design — the load-bearing choice is
      # taking ATTENTION matrices out of Muon's orthogonalization, which
      # suits square FFN GEMMs but may fight the spectral sensitivity of
      # QK^T products): Muon = 2-D FFN linears only; attention qkv/proj and
      # smolgen go to the internal AdamW group at base lr.
      def _use_muon(n, p):
        return (p.ndim == 2 and 'embedding' not in n and 'transformer_layer' in n
                and ('mlp.linear' in n or 'tactical_ffn' in n))
    else:
      raise ValueError(f"Unsupported MuonAdamWScope: {_muon_scope!r} (use 'all-non-trunk', 'final-only' or 'ffn-only')")
    muon_params  = [p for n, p in model.named_parameters() if p.requires_grad and _use_muon(n, p)]
    adamw_params = [p for n, p in model.named_parameters() if p.requires_grad and not _use_muon(n, p)]
    print(f"[train] Muon partition scope={_muon_scope}: {len(muon_params)} muon / {len(adamw_params)} adamw params", flush=True)
    # Split-LR: separate rate for the internal-AdamW group (heads/embeddings/norms/biases)
    # while the Muon trunk keeps LearningRateBase. One knob was silently shared by two
    # optimizers with different natural scales; the fast trunk rate is needed by the trunk
    # (all-cold 20M arm collapsed) while AdamW@2e-4 is the prod-validated value rate.
    # Ratio-preserved through the LR schedule. CONFIG-ONLY by design ('LearningRateBaseHeads'
    # in the opt json; absent/null = legacy single-rate) — no env override, so the config
    # artifact always tells the truth about what trained.
    if os.environ.get('CERES_MUON_HEADS_LR'):
      raise ValueError('CERES_MUON_HEADS_LR was removed by design - set LearningRateBaseHeads '
                       'in the _ceres_opt.json config instead (config-only, self-documenting).')
    _heads_lr = getattr(config, 'Opt_LearningRateBaseHeads', None)
    if _heads_lr is not None:
      _heads_lr = float(_heads_lr)
    if _heads_lr is not None:
      print(f'[train] Muon SPLIT-LR: trunk (Muon 2-D) lr={LR}, heads/embeddings/norms (internal AdamW) lr={_heads_lr} '
            f'(ratio {_heads_lr / LR:.4g}, schedule-proportional); {len(muon_params)} muon / {len(adamw_params)} adamw params')
    # Per-head Muon (Kimi K3-style, config 'MuonPerHeadAttention'): orthogonalize each
    # attention head's projection block independently instead of the full fused matrix,
    # so gradient correlations across heads can't couple through one global NS step.
    # Block layout is taken from how each weight is RESHAPED at use in
    # DotProductAttention.forward, not guessed from shape:
    #   qkv  linear path:    out rows are [H, qkv_mult*d_k*m] head-major -> per-head
    #                        AND per-projection blocks (nb = qkv_mult*H, axis 0)
    #   qkv  nonlinear path: out rows are [qkv_mult, d_model*m] — no head structure
    #                        yet -> per-projection only (nb = qkv_mult, axis 0)
    #   q2/k2/v2/q2b:        out rows head-major -> per-head (nb = H, axis 0)
    #   W_h:                 input cols are concatenated head outputs -> per-head
    #                        along columns (nb = H, axis 1)
    # LoRA-wrapped layers are skipped (their base weights are frozen/adapted, and
    # LoRA params never run under Muon anyway).
    _phm_specs = {}
    if getattr(config, 'Opt_MuonPerHeadAttention', False):
      from dot_product_attention import DotProductAttention
      _muon_ids = set(id(p) for p in muon_params)
      def _plain_weight(sub):
        return sub.weight if isinstance(sub, torch.nn.Linear) else None
      for _mod in model.modules():
        if not isinstance(_mod, DotProductAttention):
          continue
        _H = _mod.num_heads
        _w = _plain_weight(_mod.qkv)
        if _w is not None and id(_w) in _muon_ids:
          _nb = _mod.qkv_multiplier * (1 if _mod.use_nonlinear_attention else _H)
          if _nb > 1 and _w.shape[0] % _nb == 0:
            _phm_specs[_w] = (0, _nb)
        for _sname in ('q2', 'k2', 'v2', 'q2b'):
          _s = getattr(_mod, _sname, None)
          _w = _plain_weight(_s) if _s is not None else None
          if _w is not None and id(_w) in _muon_ids and _w.shape[0] % _H == 0:
            _phm_specs[_w] = (0, _H)
        _w = _plain_weight(_mod.W_h)
        if _w is not None and id(_w) in _muon_ids and _w.shape[1] % _H == 0:
          _phm_specs[_w] = (1, _H)
      if not _phm_specs:
        raise ValueError('MuonPerHeadAttention=true but no eligible attention matrices found '
                         '(LoRA-wrapped model, or attention params not under Muon scope?)')
      print(f'[train] Muon PER-HEAD attention: {len(_phm_specs)} attention matrices head-split', flush=True)
    # FAMILY LR RATIOS (split-LR program 2026-08-20, Kovax-style): per-param
    # multiplicative ratios applied on top of the group lr in BOTH Muon and
    # AdamW branches. Unlike LearningRateBaseHeads (whole-internal-AdamW-group
    # knob, also hits embeddings/norms/taus), these target NAME FAMILIES
    # regardless of partition membership:
    #   LearningRateHeadsRatio     — output-head family (Kovax runs 1/3)
    #   LearningRateCouplingsRatio — dual-plane zero-init couplings (plan H2)
    _HEAD_FAMILY = ('policy_head.', 'value_head.', 'value2_head.', 'unc_head.',
                    'mlh_head.', 'qdev_upper.', 'qdev_lower.', 'headPremap.',
                    'headSharedLinear.', 'unc_policy.')
    _COUPLING_FAMILY = ('dual_plane.', 'dp_value_inject.', 'dp_value2_inject.',
                        'dp_pol_q.', 'dp_pol_p.', 'dpva_', 'dp_surv_head.')
    _heads_ratio = getattr(config, 'Opt_LearningRateHeadsRatio', None)
    _coup_ratio = getattr(config, 'Opt_LearningRateCouplingsRatio', None)
    assert not (_heads_ratio is not None and _heads_lr is not None), \
        'LearningRateHeadsRatio and LearningRateBaseHeads are mutually exclusive (different group semantics)'
    _lr_ratios = {}
    if _heads_ratio is not None or _coup_ratio is not None:
      _n_h = _n_c = 0
      for _pn, _pp in model.named_parameters():
        if _heads_ratio is not None and any(f in _pn for f in _HEAD_FAMILY):
          _lr_ratios[_pp] = float(_heads_ratio); _n_h += 1
        elif _coup_ratio is not None and any(f in _pn for f in _COUPLING_FAMILY):
          _lr_ratios[_pp] = float(_coup_ratio); _n_c += 1
      # Membership dump (phase-0 smoke contract): grep-able, one line per family.
      print(f'[train] FAMILY-LR: heads ratio={_heads_ratio} ({_n_h} params), '
            f'couplings ratio={_coup_ratio} ({_n_c} params); '
            f'ratios ride the schedule multiplicatively', flush=True)
      if _heads_ratio is not None and _n_h == 0:
        raise ValueError('LearningRateHeadsRatio set but no head-family params matched')
      if _coup_ratio is not None and _n_c == 0:
        raise ValueError('LearningRateCouplingsRatio set but no coupling-family params matched (UseDualPlane off?)')
    # MuonMomentum decouples the Muon SGD-momentum from the internal-AdamW
    # beta1 (see config.py) — reference combo is momentum 0.95 / adam beta1 0.9.
    _muon_mom = config.Opt_MuonMomentum if getattr(config, 'Opt_MuonMomentum', None) is not None else config.Opt_Beta1
    _muon_aeps = config.Opt_MuonAdamWEps if getattr(config, 'Opt_MuonAdamWEps', None) is not None else 1e-8
    optimizer = Muon(lr=LR, wd=WEIGHT_DECAY, momentum=_muon_mom, adamw_betas=(config.Opt_Beta1, config.Opt_Beta2), adamw_eps=_muon_aeps, muon_params=muon_params, adamw_params=adamw_params, adamw_lr=_heads_lr, head_split_specs=_phm_specs or None, lr_ratios=_lr_ratios or None)
    if getattr(config, 'Opt_MuonMomentum', None) is not None or getattr(config, 'Opt_MuonAdamWEps', None) is not None:
      print(f'[train] Muon decoupled: momentum={_muon_mom} (adamw beta1={config.Opt_Beta1}), adamw_eps={_muon_aeps}')
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
    MIN_LR = float(getattr(config, 'Opt_LRMinFactor', 0.05) or 0.05)
    WARMUP_POS = num_warmup_positions()

    if num_pos < WARMUP_POS:
      return (float(num_pos) / float(WARMUP_POS))**0.5 # inverse square root warmup
    elif fraction_complete < FRAC_START_DECAY:
      return 1.0
    elif fraction_complete > 1:
      return MIN_LR # shouldn't happen
    elif getattr(config, 'Opt_LRDecayShape', 'linear') == 'cosine':
      # half-cosine decay to MIN_LR (reference no-plateau shape when
      # FRAC_START_DECAY == 0: warmup then cosine over the entire run)
      _prog = (fraction_complete - FRAC_START_DECAY) / (1.0 - FRAC_START_DECAY)
      return MIN_LR + (1.0 - MIN_LR) * 0.5 * (1.0 + math.cos(math.pi * _prog))
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

       
  # Move model to device, then (multi-GPU only) wrap in DistributedDataParallel.
  # The optimizer above was built from these same parameter tensors; DDP does NOT
  # replace them (it only registers gradient-sync hooks), so the optimizer stays
  # valid and optimizer.step() updates the same tensors DDP fills .grad on.
  model = model.to(device)
  if config.Exec_DataType == 'BFloat16Pure':
    model = model.to(torch.bfloat16)
  DDP_STATIC_GRAPH = False   # set below when DDP is active; read by the train loop
  if IS_DISTRIBUTED:
    # Two DDP modes, chosen by the data layout:
    #
    #  * BOARDS_PER_BATCH==1 (single forward per backward): default mode with
    #    find_unused_parameters=True. CeresNet has conditionally-used heads, and an
    #    output that never reaches the loss would otherwise trip DDP's
    #    "parameter ... did not receive grad" error. Set CERES_DDP_FIND_UNUSED=0 once
    #    a config is verified to use every parameter (drops a per-step graph walk).
    #
    #  * BOARDS_PER_BATCH==4 (model() called 3-4× before ONE backward): the default
    #    reducer mishandles multiple forwards per backward ("marked ready twice").
    #    static_graph=True is the supported path for that — it also natively handles
    #    unused params and is faster, at the cost of assuming the used-parameter set
    #    is identical every iteration (true here). Auto-enabled for 4-board; force
    #    on/off with CERES_DDP_STATIC_GRAPH=1/0. static_graph and
    #    find_unused_parameters are mutually exclusive, so static_graph wins.
    _static_graph = int(os.environ.get('CERES_DDP_STATIC_GRAPH',
                                       '1' if BOARDS_PER_BATCH > 1 else '0') or 0) > 0
    DDP_STATIC_GRAPH = _static_graph
    _find_unused = (int(os.environ.get('CERES_DDP_FIND_UNUSED', '1') or 1) > 0
                    and not _static_graph)
    # Gradient bucket size. DDP coalesces gradients into buckets and fires one
    # all-reduce per bucket as it fills; PyTorch's 25 MB default is tuned for
    # fast homogeneous interconnects. On a box whose ranks are NOT all on one
    # NVLink island (e.g. two NVLink pairs bridged by PCIe), cross-island
    # traffic is latency-dominated and fewer/larger transfers win. Raise it
    # there; there is no benefit on a single fast fabric, so the default is
    # unchanged.
    _bucket_mb = float(getattr(config, 'Opt_DDPBucketCapMB', 0) or 0)
    _ddp_kwargs = {'bucket_cap_mb': _bucket_mb} if _bucket_mb > 0 else {}
    model = DDP(model, device_ids=[DDP_LOCAL_GPU], output_device=DDP_LOCAL_GPU,
                find_unused_parameters=_find_unused, gradient_as_bucket_view=True,
                static_graph=_static_graph, **_ddp_kwargs)
    # bf16 gradient compression (config Opt_DDPBF16Compress). Halves the bytes
    # on the wire by casting each bucket to bf16 for the all-reduce and back to
    # fp32 afterwards — close to a 2x win when the run is communication-bound,
    # and near-free in quality here because the gradients were produced under
    # bf16 autocast in the first place (the fp32 .grad buffers hold values that
    # already passed through bf16 matmuls). Off by default: on a fast fabric it
    # buys nothing and the cast is pure overhead.
    if bool(getattr(config, 'Opt_DDPBF16Compress', False)):
      from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as _ddp_hooks
      model.register_comm_hook(state=None, hook=_ddp_hooks.bf16_compress_hook)
      print('[ddp] bf16 gradient compression ENABLED (all-reduce payload halved)', flush=True)
    print(f"[ddp] model wrapped in DistributedDataParallel "
          f"(static_graph={_static_graph}, find_unused_parameters={_find_unused}, "
          f"bucket_cap_mb={_bucket_mb if _bucket_mb > 0 else 'default'})", flush=True)
  # Custom CeresNet methods/attributes (compute_loss, use_gtab, _last_gate_value,
  # _last_tsb_gates) are NOT exposed through the DDP wrapper — DDP only forwards
  # forward(). Route those calls through the raw module (model_nocompile shares the
  # same parameter tensors); the DDP-wrapped `model` is used only for forward() so
  # its gradient-sync hooks fire. In single-GPU mode `core is model_nocompile` too.
  core = model_nocompile

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
  # (The old count_zst_files here was a write-only dead variable that fully
  # listed every root at startup — deleted, review 2026-08-21 finding 13.)
  _DEFAULT_NUM_DATASET_WORKERS = 0 if sys.platform.startswith("win") else 1
  NUM_DATASET_WORKERS = int(os.environ.get('CERES_NUM_DATASET_WORKERS', _DEFAULT_NUM_DATASET_WORKERS))
  if NUM_DATASET_WORKERS != _DEFAULT_NUM_DATASET_WORKERS:
    print(f'[train] NUM_DATASET_WORKERS override: {_DEFAULT_NUM_DATASET_WORKERS} -> {NUM_DATASET_WORKERS} (via CERES_NUM_DATASET_WORKERS)')
  PREFETCH_FACTOR = None if NUM_DATASET_WORKERS == 0 else 4 # to keep GPU busy
 
  # world_size/rank come from torchrun (single-GPU: 1 and 0). Each rank reads a
  # disjoint file-shard (tpg_dataset slices files by rank/world_size) and a
  # 1/world_size slice of the GLOBAL forward batch, so the effective optimization
  # batch — and therefore the LR schedule — is identical to a single-GPU run.
  # Requires batch_size_forward divisible by world_size and >= world_size files.
  world_size = WORLD_SIZE
  rank = RANK
  # Per-stream file-mirror override (default: both streams inherit the global
  # CERES_FILE_MIRROR_AUG). E.g. mirror only the puzzle secondary in a mixed run:
  #   CERES_FILE_MIRROR_AUG_PRIMARY=0 CERES_FILE_MIRROR_AUG_SECONDARY=0.5
  def _opt_env_float0(name):
    v = os.environ.get(name)
    return float(v) if v not in (None, '') else None
  _MIRROR_PRIMARY = _opt_env_float0('CERES_FILE_MIRROR_AUG_PRIMARY')
  _MIRROR_SECONDARY = _opt_env_float0('CERES_FILE_MIRROR_AUG_SECONDARY')

  # Primary source: TPG shards (default) or direct LC0 v6 .gz chunks
  # (Data config "SourceType": "DirectFromV6" — zero-storage path for
  # training straight from pre-rescored LC0 data; see v6_dataset.py).
  _IS_V6_SOURCE = str(getattr(config, 'Data_SourceType', '') or '') == 'DirectFromV6'
  if _IS_V6_SOURCE:
    # q-deviation targets do not exist in v6 records; the loader yields zeros,
    # so a nonzero loss weight would silently train the head toward zero.
    assert float(getattr(config, 'Opt_LossQDeviationMultiplier', 0) or 0) == 0, \
        'DirectFromV6 requires LossQDeviationMultiplier=0 (no q-deviation data in v6 records)'
    from v6_dataset import V6ChunkDataset
    primary_dataset = V6ChunkDataset(TPG_TRAIN_DIR, batch_size_forward // world_size,
                                     config.Data_WDLLabelSmoothing,
                                     rank, world_size, NUM_DATASET_WORKERS,
                                     BOARDS_PER_BATCH, config.Data_NumTPGFilesToSkip,
                                     config.Exec_TestFlag,
                                     file_mirror_prob=_MIRROR_PRIMARY,
                                     skip_count=getattr(config, 'Data_V6SkipCount', None),
                                     shuffle_pool=getattr(config, 'Data_V6ShufflePool', None),
                                     max_resultq_delta=getattr(config, 'Data_V6MaxResultQDelta', None))
  else:
    primary_dataset = TPGDataset(TPG_TRAIN_DIR, batch_size_forward // world_size, config.Data_WDLLabelSmoothing,
                                 rank, world_size, NUM_DATASET_WORKERS,
                                 BOARDS_PER_BATCH, config.Data_NumTPGFilesToSkip, config.Exec_TestFlag,
                                 file_mirror_prob=_MIRROR_PRIMARY)

  # Optional secondary corpus (e.g. puzzle TPG mixed with T80 self-play).
  # Triggered when both Data_TrainingFilesDirectory2 is set AND Data_RatioSet1ToSet2 > 0.
  secondary_dataset = None
  if (getattr(config, 'Data_TrainingFilesDirectory2', None)
      and int(getattr(config, 'Data_RatioSet1ToSet2', 0) or 0) > 0):
    # Secondary corpus may use a different shard record format than the primary
    # (e.g. V3 1.6B-position primary + V2 survival-labeled secondary):
    # CERES_TPG_SQUARE_BYTES2 overrides for the secondary only (default = primary's).
    _sec_square_bytes = os.environ.get('CERES_TPG_SQUARE_BYTES2')
    _sec_square_bytes = int(_sec_square_bytes) if _sec_square_bytes not in (None, '') else None
    if _sec_square_bytes is not None:
      print(f'[mixed-dataset] secondary shard format override: {_sec_square_bytes} bytes/square')
    # Strict sidecar mode ('required'/=1) is an assertion about the PRIMARY corpus;
    # a sidecar-less secondary (e.g. puzzle TPG) is intended, so demote it to 'auto'
    # there instead of dying on its first batch. Secondaries WITH sidecars still
    # supply survival targets under 'auto'.
    _sec_sidecar_mode = 'auto' if TPG_TARGET_SIDECAR_MODE == 'required' else None
    secondary_dataset = TPGDataset(config.Data_TrainingFilesDirectory2,
                                   batch_size_forward // world_size,
                                   config.Data_WDLLabelSmoothing,
                                   rank, world_size, NUM_DATASET_WORKERS,
                                   BOARDS_PER_BATCH, 0, config.Exec_TestFlag,
                                   square_bytes=_sec_square_bytes,
                                   sidecar_mode=_sec_sidecar_mode,
                                   file_mirror_prob=_MIRROR_SECONDARY)
    print(f'[mixed-dataset] primary={TPG_TRAIN_DIR}')
    print(f'[mixed-dataset] secondary={config.Data_TrainingFilesDirectory2}')
    print(f'[mixed-dataset] ratio = {config.Data_RatioSet1ToSet2}:1 (primary:secondary batches)')

  # Per-stream loss routing: optional loss-weight OVERRIDES applied only to batches
  # from the SECONDARY corpus (tagged by TPGMixedDataset). Unset = secondary batches
  # use the same weights as primary (legacy behavior). Example: a puzzle secondary
  # whose degenerate value labels must not reach the value heads:
  #   CERES_SECONDARY_LOSS_VALUE_MULT=0 CERES_SECONDARY_LOSS_VALUE2_MULT=0 CERES_SECONDARY_LOSS_AUX_MULT=0
  def _opt_env_float(name):
    v = os.environ.get(name)
    return float(v) if v not in (None, '') else None
  SECONDARY_POLICY_MULT    = _opt_env_float('CERES_SECONDARY_LOSS_POLICY_MULT')
  SECONDARY_VALUE_MULT     = _opt_env_float('CERES_SECONDARY_LOSS_VALUE_MULT')
  SECONDARY_VALUE2_MULT    = _opt_env_float('CERES_SECONDARY_LOSS_VALUE2_MULT')
  SECONDARY_AUX_MULT       = _opt_env_float('CERES_SECONDARY_LOSS_AUX_MULT')       # unc + qdev + unc_policy + mlh together
  SECONDARY_PLACEMENT_MULT = _opt_env_float('CERES_SECONDARY_LOSS_PLACEMENT_MULT')
  SECONDARY_SURVIVAL_MULT  = _opt_env_float('CERES_SECONDARY_LOSS_SURVIVAL_MULT')
  SECONDARY_STVALUE_MULT   = _opt_env_float('CERES_SECONDARY_LOSS_STVALUE_MULT')

  # Single source of truth: attr name on `core` -> override value. Save/apply/restore
  # all iterate this dict, so adding a weight cannot desynchronize the three steps.
  SECONDARY_WEIGHT_OVERRIDES = {}
  if SECONDARY_POLICY_MULT    is not None: SECONDARY_WEIGHT_OVERRIDES['policy_loss_weight']  = SECONDARY_POLICY_MULT
  if SECONDARY_VALUE_MULT     is not None: SECONDARY_WEIGHT_OVERRIDES['value_loss_weight']   = SECONDARY_VALUE_MULT
  # HL-Gauss trains on the same value labels, so it must follow the value
  # override (scaled by its own base weight — the MULT is a multiplier here):
  # otherwise a "value-off" secondary stream still leaks its degenerate value
  # labels into the trunk through the hlg head.
  if SECONDARY_VALUE_MULT is not None and float(getattr(config, 'Opt_HLGaussWeight', 0) or 0) > 0:
    SECONDARY_WEIGHT_OVERRIDES['hlg_weight'] = SECONDARY_VALUE_MULT * float(config.Opt_HLGaussWeight)
  # Optimistic-policy is policy-family: follow the policy override, same scaled-
  # multiplier convention as hlg above.
  if SECONDARY_POLICY_MULT is not None and float(getattr(config, 'Opt_OptimisticPolicyWeight', 0) or 0) > 0:
    SECONDARY_WEIGHT_OVERRIDES['opt_policy_weight'] = SECONDARY_POLICY_MULT * float(config.Opt_OptimisticPolicyWeight)
  # Soft-policy is policy-family too (same gap existed for it since the s5 era;
  # closed here for consistency with the hlg/opt review fixes). Base weight may
  # come from env fallback, so read the resolved value off the core module.
  if SECONDARY_POLICY_MULT is not None and float(getattr(core, 'soft_policy_weight', 0) or 0) > 0:
    SECONDARY_WEIGHT_OVERRIDES['soft_policy_weight'] = SECONDARY_POLICY_MULT * float(core.soft_policy_weight)
  if SECONDARY_VALUE2_MULT    is not None: SECONDARY_WEIGHT_OVERRIDES['value2_loss_weight']  = SECONDARY_VALUE2_MULT
  if SECONDARY_AUX_MULT is not None:
    for _aux_attr in ('unc_loss_weight', 'q_deviation_loss_weight', 'uncertainty_policy_weight', 'moves_left_loss_weight'):
      SECONDARY_WEIGHT_OVERRIDES[_aux_attr] = SECONDARY_AUX_MULT
  if SECONDARY_PLACEMENT_MULT is not None: SECONDARY_WEIGHT_OVERRIDES['placement_value_weight'] = SECONDARY_PLACEMENT_MULT
  if SECONDARY_SURVIVAL_MULT  is not None: SECONDARY_WEIGHT_OVERRIDES['survival_target_weight'] = SECONDARY_SURVIVAL_MULT
  if SECONDARY_STVALUE_MULT   is not None: SECONDARY_WEIGHT_OVERRIDES['stvalue_weight'] = SECONDARY_STVALUE_MULT

  if SECONDARY_WEIGHT_OVERRIDES:
    # Fail loudly on configurations where routing would be a silent no-op.
    if secondary_dataset is None:
      raise ValueError('CERES_SECONDARY_LOSS_*_MULT set but no secondary corpus is configured '
                       '(needs TrainingFilesDirectory2 and RatioSet1ToSet2 > 0)')
    if BOARDS_PER_BATCH != 1:
      raise NotImplementedError('per-stream loss routing (CERES_SECONDARY_LOSS_*_MULT) is only implemented for '
                                'BOARDS_PER_BATCH==1; the 4-board action path would silently ignore it')
    if SECONDARY_WEIGHT_OVERRIDES.get('placement_value_weight', 0) > 0 and core.placement_value_weight == 0:
      raise ValueError('CERES_SECONDARY_LOSS_PLACEMENT_MULT > 0 requires the placement head to be enabled '
                       '(CERES_PLACEMENT_VALUE_WEIGHT > 0), otherwise the head does not exist and the '
                       'override is a silent no-op')
    print(f'[mixed-dataset] SECONDARY loss overrides: {SECONDARY_WEIGHT_OVERRIDES} (heads not listed inherit primary weights)')

  # Mirror-consistency value regularizer (CERES_VALUE_MIRROR_CONS_WEIGHT, default
  # 0 = off): label-free variance reduction on value1. A fixed-size subset of each
  # batch (castling-rights-free records only — the file-mirror-aug eligibility
  # rule) is mirrored a<->h and forwarded again; a symmetric KL between the two
  # value distributions penalizes self-disagreement on positions that are exactly
  # value-symmetric. Subset size is FIXED (frac * batch, resampled with
  # replacement if few eligible) so torch.compile sees one extra static shape.
  # BOARDS_PER_BATCH==1 path only. Adds no parameters; export graph unchanged.
  # Config-first, env-fallback (see config.py note).
  MIRROR_CONS_WEIGHT = (float(getattr(config, 'Opt_MirrorConsWeight', 0) or 0)
                        or float(os.environ.get('CERES_VALUE_MIRROR_CONS_WEIGHT', '0') or 0))
  MIRROR_CONS_FRAC = (float(getattr(config, 'Opt_MirrorConsFraction', 0) or 0)
                      or float(os.environ.get('CERES_VALUE_MIRROR_CONS_FRAC', '0.25') or 0.25))
  _cfg_focal = float(getattr(config, 'Opt_ValueFocalGamma', 0) or 0)
  if _cfg_focal > 0:
    import losses as _losses_module
    # Re-assert the focal/provenance mutual exclusion here: the import-time
    # assert in losses.py only sees the env var, and this config path would
    # otherwise silently drop value1 provenance weighting.
    assert _losses_module.VALUE_PROV_WEIGHTS is None or _losses_module.VALUE_PROV_SCOPE == 'value2', \
      'ValueFocalGamma (config) and value1-scope provenance weighting are mutually exclusive'
    _losses_module.VALUE_FOCAL_GAMMA = _cfg_focal
    print(f'[train] value1 FOCAL gamma set from config: {_cfg_focal}')
  _mirror_perm64_t = None
  # Hysteresis controller state (see config.py): mode starts ACTIVE; smoothed
  # level tracks the term with momentum 0.98 (~50-batch horizon).
  MIRROR_PROBE_STEPS = int(getattr(config, 'Opt_MirrorConsProbeSteps', 0) or 0)
  MIRROR_AUTO_LOW = float(getattr(config, 'Opt_MirrorConsAutoLow', 0.003))
  MIRROR_AUTO_HIGH = float(getattr(config, 'Opt_MirrorConsAutoHigh', 0.008))
  _mirror_active = True
  _mirror_level = None
  _mirror_step = 0
  if MIRROR_CONS_WEIGHT > 0:
    if BOARDS_PER_BATCH != 1:
      raise NotImplementedError('CERES_VALUE_MIRROR_CONS_WEIGHT is only implemented for BOARDS_PER_BATCH==1')
    if WORLD_SIZE > 1:
      # Three separate blockers, all measured/analysed 2026-08-17 — a future
      # multi-GPU port has to clear every one, not just the first:
      #  1. The second forward through the DDP-wrapped model needs
      #     static_graph=True (the default reducer dies with "marked ready
      #     twice" on two forwards per backward).
      #  2. Whether the term runs at all is DATA-dependent per rank
      #     (_elig.numel() > 0 below), so ranks can disagree and desync into a
      #     hang; the decision would need an all-reduce (MIN) first.
      #  3. static_graph also requires a constant per-iteration graph, which
      #     the probe/hysteresis mode (MirrorConsProbeSteps, running the term
      #     only every Nth step) violates — under DDP the term must run every
      #     step or never, i.e. the thermostat that makes it cheap has to go.
      raise NotImplementedError(
        'CERES_VALUE_MIRROR_CONS_WEIGHT is single-GPU only: the second forward through the '
        'DDP-wrapped model needs static_graph, the run/skip decision is rank-divergent, and '
        'probe mode breaks static_graph\'s constant-graph requirement (see comment above)')
    import tpg_dataset as _tpgds
    _tpgds._ensure_mirror_tables()
    _mirror_perm64_t = torch.tensor(_tpgds._MIRROR_PERM64, dtype=torch.long)
    print(f'[train] VALUE MIRROR-CONSISTENCY enabled: w={MIRROR_CONS_WEIGHT}, frac={MIRROR_CONS_FRAC} '
          f'(castling-rights records exempt; value1 sym-KL)')

  # Hard-position replay buffer (config: HardReplayBufferSize/Fraction/MaxReuse/
  # MinKL/ReuseTarget/StartFraction —
  # see config.py). Flow per primary batch: (1) BEFORE forward, replace a fixed
  # number of rows with sampled buffer entries; (2) after the loss, measure
  # per-sample value1-KL on the NON-replayed rows and push the hardest ones
  # (top inject-count) into the buffer; (3) evict entries after MaxReuse
  # replays (plus FIFO when full). Secondary-stream batches are untouched.
  HARD_REPLAY_SIZE = int(getattr(config, 'Opt_HardReplayBufferSize', 0) or 0)
  HARD_REPLAY_FRAC = float(getattr(config, 'Opt_HardReplayFraction', 0.125))
  HARD_REPLAY_MAX_REUSE = int(getattr(config, 'Opt_HardReplayMaxReuse', 8))
  # Consecutive signature-mismatched draws before an entry is dropped. Named
  # rather than inline so it is not read as tied to MaxReuse (whose default
  # is also 8). Without it, entries whose signature never matches are
  # unreachable by BOTH eviction paths and accumulate until injection decays.
  HARD_REPLAY_MISS_LIMIT = 8
  # Absolute admission floor / reuse CAP / start delay (all see config.py).
  # All default to 0 = legacy relative top-k with fixed injection from step 0.
  HARD_REPLAY_MIN_KL = float(getattr(config, 'Opt_HardReplayMinKL', 0.0) or 0.0)
  HARD_REPLAY_REUSE_TARGET = float(getattr(config, 'Opt_HardReplayReuseTarget', 0.0) or 0.0)
  HARD_REPLAY_START_FRAC = float(getattr(config, 'Opt_HardReplayStartFraction', 0.0) or 0.0)
  # NO `or default` on any of these: it maps an EXPLICIT 0 (and '' / false on
  # the selector) back to the default before the validators can reject it —
  # the silent-arm-switch the validators exist to prevent. config.py already
  # guarantees the attributes exist with None-checked defaults.
  HARD_REPLAY_SELECTOR = str(getattr(config, 'Opt_HardReplaySelector', 'valuekl'))
  HARD_REPLAY_TOPN = int(getattr(config, 'Opt_HardReplayTopN', 3))
  HARD_REPLAY_TN_PBEST = float(getattr(config, 'Opt_HardReplayTopNPBest', 0.5))
  HARD_REPLAY_TN_MINABSQ = float(getattr(config, 'Opt_HardReplayTopNMinAbsQ', 0.4))
  HARD_REPLAY_TN_EXIT = float(getattr(config, 'Opt_HardReplayTopNExitMargin', 0.5))
  HARD_REPLAY_TN_KEYS = bool(getattr(config, 'Opt_HardReplayTopNKeysPresent', False))
  # EMAs of rows actually ADMITTED / INJECTED per micro-step. Injection is sized
  # from the intake EMA so reuse tends to the target; both are also the only
  # honest basis for the realised-reuse metric (a smoothed numerator over a raw
  # per-step denominator reads 64x high on a low-admission step). Per-rank and
  # local: no collectives, exactly like the buffer itself.
  _hr_intake_ema = None
  _hr_inject_ema = 0.0
  _hr_displaced_total = 0   # cumulative injected rows this rank (reporting only)
  _hard_replay_buf = []   # {'rows':…, 'uses': int, 'kl': float, 'miss': int, 'sig':…}
  if HARD_REPLAY_SIZE == 0 and (HARD_REPLAY_MIN_KL > 0 or HARD_REPLAY_REUSE_TARGET > 0
                                or HARD_REPLAY_START_FRAC > 0
                                or HARD_REPLAY_SELECTOR != 'valuekl'
                                or HARD_REPLAY_TN_KEYS):
    # Every validator and the banner live inside the SIZE>0 block, so without
    # this a control-shaped config carrying treatment knobs would run silently
    # as a plain control while presenting as a treatment arm.
    raise ValueError('HardReplayMinKL/ReuseTarget/StartFraction are set but '
                     'HardReplayBufferSize is 0 — replay is OFF. Remove the knobs '
                     'or set a buffer size.')
  if HARD_REPLAY_SIZE > 0:
    if BOARDS_PER_BATCH != 1:
      raise NotImplementedError('HardReplayBufferSize > 0 is only implemented for BOARDS_PER_BATCH==1')
    # Mean reuse is EXACTLY injected/admitted by flow conservation (every
    # injection increments one entry's 'uses'); MaxReuse does not reduce it and
    # buffer size cancels. ReuseTarget is therefore a CAP that binds only while
    # admissions are scarce — early on, when many rows clear the floor, realised
    # reuse is lower and variety is higher, which is the behaviour we want.
    if not (0.0 < HARD_REPLAY_FRAC <= 1.0):
      raise ValueError(f'HardReplayFraction must be in (0,1]; got {HARD_REPLAY_FRAC}. '
                       f'0 fills the buffer and never injects; >1 raises a shape error at the '
                       f'first injection.')
    if HARD_REPLAY_MIN_KL < 0 or not (0.0 <= HARD_REPLAY_START_FRAC < 1.0):
      raise ValueError(f'HardReplayMinKL must be >=0 (got {HARD_REPLAY_MIN_KL}) and '
                       f'HardReplayStartFraction in [0,1) (got {HARD_REPLAY_START_FRAC}).')
    if HARD_REPLAY_SELECTOR not in ('valuekl', 'topn'):
      raise ValueError(f'HardReplaySelector must be "valuekl" or "topn", got '
                       f'{HARD_REPLAY_SELECTOR!r} (typo/empty value would silently change the arm).')
    if HARD_REPLAY_SELECTOR == 'topn':
      if not (1 <= HARD_REPLAY_TOPN <= 10 and 0.0 < HARD_REPLAY_TN_PBEST < 1.0
              and 0.0 <= HARD_REPLAY_TN_MINABSQ < 1.0 and HARD_REPLAY_TN_EXIT >= 0.0):
        raise ValueError(f'topn selector knobs out of range: N={HARD_REPLAY_TOPN}, '
                         f'PBest={HARD_REPLAY_TN_PBEST}, MinAbsQ={HARD_REPLAY_TN_MINABSQ}, '
                         f'ExitMargin={HARD_REPLAY_TN_EXIT}')
      if HARD_REPLAY_MIN_KL > 0:
        raise ValueError('HardReplayMinKL is INERT under the topn selector (admission floor is '
                         'the margin at 0, exit at -ExitMargin). Remove it so the run log '
                         'cannot mislabel the arm.')
    if HARD_REPLAY_SELECTOR == 'valuekl' and HARD_REPLAY_TN_KEYS:
      raise ValueError('HardReplayTopN* knobs are set but HardReplaySelector is "valuekl" — '
                       'they would be silently inert and the A/B result would be attributed '
                       'to a topn treatment that never ran. Set the selector or remove them.')
    if (HARD_REPLAY_REUSE_TARGET > 0 and HARD_REPLAY_MIN_KL <= 0
        and HARD_REPLAY_SELECTOR == 'valuekl'):
      raise ValueError(
        'HardReplayReuseTarget > 0 requires HardReplayMinKL > 0 under the valuekl selector: '
        'with the legacy relative top-k intake, admissions always equal the injection quota, '
        'so realised reuse is pinned at ~1 and the configured target is a silent no-op. '
        '(The topn selector has an intrinsic absolute criterion and is exempt.)')
    if HARD_REPLAY_REUSE_TARGET > 0 and HARD_REPLAY_MAX_REUSE <= HARD_REPLAY_REUSE_TARGET:
      raise ValueError(
        f'HardReplayMaxReuse ({HARD_REPLAY_MAX_REUSE}) must be STRICTLY GREATER than '
        f'HardReplayReuseTarget ({HARD_REPLAY_REUSE_TARGET}): at equality buffer occupancy is '
        f'a neutral random walk; below it the buffer starves and injection silently falls '
        f'under HardReplayFraction. Use e.g. MaxReuse = 3x the target.')
    print(f'[train] HARD-REPLAY enabled: size={HARD_REPLAY_SIZE}, '
          f'selector={HARD_REPLAY_SELECTOR}'
          + (f' (N={HARD_REPLAY_TOPN}, pbest>{HARD_REPLAY_TN_PBEST}, '
             f'|W-L|>{HARD_REPLAY_TN_MINABSQ}, exit_margin={HARD_REPLAY_TN_EXIT})'
             if HARD_REPLAY_SELECTOR == 'topn' else
             f', min_kl={HARD_REPLAY_MIN_KL or "off (relative top-k)"}') +
          f', inject_cap={HARD_REPLAY_FRAC}, max_reuse={HARD_REPLAY_MAX_REUSE}, '
          f'reuse_target={HARD_REPLAY_REUSE_TARGET or "off (fixed injection)"}, '
          f'start_fraction={HARD_REPLAY_START_FRAC}')
    if WORLD_SIZE > 1:
      # DDP: the buffer is deliberately PER RANK, and that is the correct
      # semantics rather than a compromise — each rank owns a disjoint slice of
      # the corpus (see tpg_dataset partitioning), so its hard positions are its
      # own. Nothing here is collective: harvesting reads that rank's own loss
      # tensors, injection replaces rows in the batch BEFORE the forward, and
      # the batch shape the reducer sees is unchanged. Ranks may inject
      # different row counts without desyncing, because no rank enters a
      # collective the others skip.
      #
      # Two consequences for a config ported from a single-GPU recipe:
      #   * size is per rank, so N ranks hold N x HardReplayBufferSize in total
      #   * the fraction applies to the per-rank batch, so each rank churns its
      #     buffer faster while seeing fewer positions
      print(f'[train] HARD-REPLAY under DDP: buffer is per-rank ({WORLD_SIZE} x '
            f'{HARD_REPLAY_SIZE} total), fed from each rank own shard slice; ' 
            f'no collectives involved. Divide the size by {WORLD_SIZE} to match a '
            f'single-GPU recipe.', flush=True)

  # Policy/value gradient-conflict probe (config: GradConflictProbeSteps — see config.py).
  # Differentiates the two loss families separately every N optimizer steps and reports
  # the angle between their gradients on the parameters they SHARE. Head-private params
  # are excluded by construction: the value head receives no policy gradient and vice
  # versa, so a cosine over them would be undefined noise. The three groups that ARE
  # shared are reported separately because they answer different questions —
  # headfront is the shared bottleneck the private-value-head work is about, trunk is
  # the one no head-side change can fix.
  GC_PROBE_STEPS = int(getattr(config, 'Opt_GradConflictProbeSteps', 0) or 0)
  _gc_groups = {}
  _gc_opt_steps = 0
  if GC_PROBE_STEPS > 0:
    if WORLD_SIZE > 1:
      raise NotImplementedError('GradConflictProbeSteps > 0 is single-GPU only: the extra '
                                'per-family backward passes desync DDP\'s gradient reducer')
    if config.Opt_PyTorchCompileMode is not None:
      raise ValueError('GradConflictProbeSteps > 0 requires PyTorchCompileMode null: compiled '
                       'autograd rejects the retained graph the probe needs. To measure a '
                       'compiled production run, resume its checkpoint uncompiled for a few '
                       'million positions with the probe on.')
    for _n, _p in model_nocompile.named_parameters():
      if not _p.requires_grad:
        continue
      if 'transformer_layer' in _n:
        _gc_groups.setdefault('trunk', []).append((_n, _p))
      elif 'headPremap' in _n or 'headSharedLinear' in _n:
        _gc_groups.setdefault('headfront', []).append((_n, _p))
      elif 'embedding' in _n:
        _gc_groups.setdefault('emb', []).append((_n, _p))
    _gc_order = [g for g in ('trunk', 'headfront', 'emb') if g in _gc_groups]
    _gc_params = [p for g in _gc_order for (_, p) in _gc_groups[g]]
    _gc_slices = []
    _gc_at = 0
    for g in _gc_order:
      _gc_slices.append((g, _gc_at, _gc_at + len(_gc_groups[g])))
      _gc_at += len(_gc_groups[g])
    print(f'[train] GRAD-CONFLICT probe enabled: every {GC_PROBE_STEPS} optimizer steps, '
          f'groups ' + ', '.join(f'{g}={len(_gc_groups[g])}' for g in _gc_order)
          + ' (2 extra backwards per probe step; diagnostic only, training unaffected)', flush=True)

  # EMA / SWA shadow weights (config: EMAPeriodSteps/EMAMaxN — see config.py).
  # Shadow copies of all floating-point state tensors live on the model's
  # device; updated every EMAPeriodSteps optimizer steps with the lc0 capped
  # running average. Deliberately NOT persisted across resume: re-initialized
  # from the live weights at boot (n=0) and re-warmed within ~EMAMaxN periods.
  EMA_PERIOD = int(getattr(config, 'Opt_EMAPeriodSteps', 0) or 0)
  EMA_MAX_N = int(getattr(config, 'Opt_EMAMaxN', 10) or 10)
  _ema_sd = None
  _ema_n = 0
  _ema_opt_steps = 0
  # DDP: the shadow lives on RANK 0 ONLY. Every rank holds bit-identical weights
  # (DDP broadcasts params at construction, then all-reduces grads so each rank
  # applies the same update to the same starting point — the same invariant the
  # rank-0-only checkpoint save already relies on), so a per-rank shadow would be
  # WORLD_SIZE identical copies of which only rank 0's is ever exported. Skipping
  # it elsewhere is safe because the update below and the export swap perform NO
  # collective operations: rank-divergent code only breaks DDP when it makes one
  # rank enter a collective the others don't. Non-master ranks leave _ema_sd None,
  # which no-ops every downstream EMA block.
  if EMA_PERIOD > 0 and IS_MASTER:
    # Shadow kept in fp32 regardless of model precision: under BFloat16Pure a
    # bf16 shadow's ~8-bit mantissa would swallow the w/(n+1) increments late
    # in training, exactly when averaging matters most. copy_ casts on export.
    _ema_sd = {k: v.detach().clone().float() for k, v in model_nocompile.state_dict().items()
               if torch.is_floating_point(v)}
    print(f'[train] EMA weight averaging enabled: period={EMA_PERIOD} opt-steps, '
          f'max_n={EMA_MAX_N} ({len(_ema_sd)} tensors; dual export <ckpt>ema.onnx)')
    if WORLD_SIZE > 1:
      # The period counts OPTIMIZER STEPS, and each step consumes WORLD_SIZE times
      # more positions under DDP — so the same value averages over a WORLD_SIZE
      # times longer stretch of training than it did on one GPU. Divide
      # EMAPeriodSteps by the GPU count to reproduce a single-GPU-validated recipe.
      print(f'[train] EMA under DDP: shadow on rank 0 only; period is in optimizer '
            f'steps, so it now spans {WORLD_SIZE}x more positions than single-GPU '
            f'(divide EMAPeriodSteps by {WORLD_SIZE} to match a 1-GPU recipe)', flush=True)

  # Aux heads (placement/survival/stvalue/depth-probes) under DDP.
  #
  # Their losses are built from tensors STASHED on the module, not from the
  # forward() return value. DDP's default reducer discovers used parameters by
  # walking the autograd graph backwards from the returned outputs, so those
  # heads' params look "unused" — the reducer marks them ready immediately, then
  # the real backward produces gradients for them and it dies with
  # "Expected to mark a variable ready only once".
  #
  # static_graph=True is the supported path: it derives the used-parameter set
  # from the FIRST backward instead of from an output traversal, so stash-only
  # params are included and reduced correctly. Its precondition is that the set
  # is IDENTICAL every iteration — which is why compute_loss now emits
  # zero-weighted participation terms for the aux heads on batches whose targets
  # are absent (mixed-corpus sidecar 'auto' mode, e.g. a puzzle secondary stream
  # alongside a survival-labelled primary). Without those terms the set would
  # shrink on target-less batches and static_graph would fire
  # "Your training graph has changed in this iteration".
  # (oppp_head and opt_head are stash-only too — omitting them here made the
  # exact recipe-prescribed 4xA100 run die at step 1 with the raw reducer
  # error instead of this guard's message, review 2026-08-21 finding 1. The
  # action head's OUTPUT is returned from forward so its params are visible
  # to the reducer, but its loss still needs the participation term.)
  if (getattr(core, 'dp_surv_weight', 0) > 0
      or getattr(core, 'placement_value_weight', 0) > 0 or getattr(core, 'survival_target_weight', 0) > 0
      or getattr(core, 'stvalue_weight', 0) > 0 or getattr(core, 'depth_probes_enabled', False)
      or getattr(core, 'opp_policy_weight', 0) > 0
      or getattr(core, 'opt_policy_weight', 0) > 0
      # Remaining stash-only heads (2026-08-21 action review finding 2 — the
      # guard must cover EVERY head whose loss is built from a module stash,
      # not just the currently-planned recipe's):
      or getattr(core, 'soft_policy_weight', 0) > 0
      or getattr(core, 'hlg_weight', 0) > 0
      or getattr(core, 'value_contrast_weight', 0) > 0
      or getattr(core, 'use_value_depth_attention', False)
      or getattr(core, 'refiner_deep_sup_weight', 0) > 0) and WORLD_SIZE > 1:
    if not _static_graph:
      raise NotImplementedError(
        'placement/survival/stvalue/depth-probe/opp-policy/optimistic-policy aux heads '
        'under DDP require static_graph: the stashed aux output is invisible to DDP\'s '
        'default reducer. Re-launch with CERES_DDP_STATIC_GRAPH=1.')
    print(f'[ddp] stash-only aux heads enabled under DDP via '
          f'static_graph; compute_loss emits zero-weighted participation terms so the '
          f'used-parameter set stays constant on target-less batches', flush=True)

  # Survival head requires sidecar targets in (at least some of) the batches.
  if getattr(core, 'survival_target_weight', 0) > 0:
    _sidecar_mode = (os.environ.get('CERES_TPG_TARGET_SIDECAR', '0') or '0').strip().lower()
    if _IS_V6_SOURCE:
      # v6/v7 chunks carry no survival labels at all (review finding 12: the
      # sidecar file-listing below would either die on tar dirs or, worse,
      # let the head train with zero supervision under '=1').
      raise ValueError('CERES_SURVIVAL_TARGET_WEIGHT > 0 is unsupported with '
                       'SourceType DirectFromV6 (no survival labels in v6/v7 records)')
    if _sidecar_mode in ('0', ''):
      raise ValueError('CERES_SURVIVAL_TARGET_WEIGHT > 0 requires CERES_TPG_TARGET_SIDECAR=1 or auto '
                       '(and a corpus generated with gen-tpg --survival-horizon)')
    if _sidecar_mode == 'auto':
      # Fail loudly if NO dataset dir contains any sidecar at all — the head would
      # silently receive zero supervision for the whole run.
      _dirs = [TPG_TRAIN_DIR]
      if getattr(config, 'Data_TrainingFilesDirectory2', None):
        _dirs.append(config.Data_TrainingFilesDirectory2)
      _any_sidecar = any(f.endswith('.tgt.zst') for d in _dirs for f in os.listdir(d))
      if not _any_sidecar:
        raise ValueError(f'CERES_TPG_TARGET_SIDECAR=auto but no .tgt.zst sidecars found in any dataset dir: {_dirs}')

  # Provenance-weighted value loss (CERES_VALUE_PROV_WEIGHTS) needs the v7x sidecars
  # actually loaded, else the weighting is a silent whole-run no-op.
  if (os.environ.get('CERES_VALUE_PROV_WEIGHTS', '') or '').strip() and _IS_V6_SOURCE:
    # DirectFromV6 supplies v7x IN-BAND from v7 records (no sidecar files);
    # verify via the corpus diagnosis instead of listing .v7x.zst files.
    if 7 not in getattr(primary_dataset, '_diag_versions', set()):
      raise ValueError('CERES_VALUE_PROV_WEIGHTS set but the DirectFromV6 corpus is not '
                       'v7 (no z_provenance available in v6 records)')
  elif (os.environ.get('CERES_VALUE_PROV_WEIGHTS', '') or '').strip():
    _v7x_mode_pw = (os.environ.get('CERES_TPG_V7X_SIDECAR', '0') or '0').strip().lower()
    if _v7x_mode_pw in ('0', ''):
      raise ValueError('CERES_VALUE_PROV_WEIGHTS set but CERES_TPG_V7X_SIDECAR is off — '
                       'z_provenance would never reach the loss (set CERES_TPG_V7X_SIDECAR=1 or auto)')
    if _v7x_mode_pw == 'auto':
      _dirs_pw = [TPG_TRAIN_DIR]
      if getattr(config, 'Data_TrainingFilesDirectory2', None):
        _dirs_pw.append(config.Data_TrainingFilesDirectory2)
      if not any(f.endswith('.v7x.zst') for d in _dirs_pw for f in os.listdir(d)):
        raise ValueError('CERES_VALUE_PROV_WEIGHTS set with CERES_TPG_V7X_SIDECAR=auto but no '
                         f'.v7x.zst sidecars found in any dataset dir: {_dirs_pw}')

  # Short-term value head requires V7-extras sidecar targets (censored q_st/d_st).
  if getattr(core, 'stvalue_weight', 0) > 0 and _IS_V6_SOURCE:
    if 7 not in getattr(primary_dataset, '_diag_versions', set()):
      raise ValueError('CERES_STVALUE_WEIGHT > 0 but the DirectFromV6 corpus is not v7 '
                       '(no censored q_st/d_st in v6 records)')
  elif getattr(core, 'stvalue_weight', 0) > 0:
    _v7x_mode = (os.environ.get('CERES_TPG_V7X_SIDECAR', '0') or '0').strip().lower()
    if _v7x_mode in ('0', ''):
      raise ValueError('CERES_STVALUE_WEIGHT > 0 requires CERES_TPG_V7X_SIDECAR=1 or auto '
                       '(and a corpus generated with gen-tpg --v7-extras)')
    if _v7x_mode == 'auto':
      _dirs = [TPG_TRAIN_DIR]
      if getattr(config, 'Data_TrainingFilesDirectory2', None):
        _dirs.append(config.Data_TrainingFilesDirectory2)
      _any_v7x = any(f.endswith('.v7x.zst') for d in _dirs for f in os.listdir(d))
      if not _any_v7x:
        raise ValueError(f'CERES_TPG_V7X_SIDECAR=auto but no .v7x.zst sidecars found in any dataset dir: {_dirs}')

  # Opp-policy / action-played aux heads require v7 targets that only the
  # DirectFromV6 path supplies in-band (TPG .v7x sidecars carry only the
  # cens_q/cens_d/prov triple). Without this preflight a misconfigured run
  # would pay the full head forward every step with zero supervision and no
  # log line, review 2026-08-21 finding 4. A TPG SECONDARY in a mixed run is
  # fine (participation terms cover its batches); the PRIMARY must supply
  # the targets.
  for _aux_w, _aux_name, _pop_attr in (
      (getattr(core, 'opp_policy_weight', 0), 'LossOppPolicyMultiplier', '_diag_opp_populated'),
      (getattr(core, 'action_played_weight', 0), 'LossActionPlayedMultiplier', '_diag_action_populated')):
    if _aux_w > 0:
      if not _IS_V6_SOURCE:
        raise ValueError(f'{_aux_name} > 0 requires SourceType DirectFromV6 with a v7 corpus '
                         f'(TPG records/sidecars carry no opp/action targets)')
      if 7 not in getattr(primary_dataset, '_diag_versions', set()):
        raise ValueError(f'{_aux_name} > 0 but the DirectFromV6 corpus is not v7 '
                         f'(no OppPlayedIndex/QAfterPlayedMove in v6 records)')
      if BOARDS_PER_BATCH != 1:
        # 4-board mode calls compute_loss four times against the SAME batch
        # dict, so board N's aux output would be scored against board 1's
        # targets (action review finding 9). Unreachable today (DirectFromV6
        # forces 1-board) — this keeps it loud if that ever changes.
        raise ValueError(f'{_aux_name} > 0 is not supported with BOARDS_PER_BATCH={BOARDS_PER_BATCH}')
      # NaN q_after targets are masked per-record in the loader, so partial
      # population is SAFE (just proportionally less signal) — the gate's job
      # is only to catch a corpus whose ExtraV7 tail was never written.
      # Healthy corpora measure ~99% (per-game last records are the gap).
      _pop = getattr(primary_dataset, _pop_attr, None)
      if _pop is not None and _pop < 0.05:
        raise ValueError(f'{_aux_name} > 0 but only {_pop:.1%} of sampled v7 records carry a '
                         f'populated target — this corpus\'s ExtraV7 tail looks zero-filled; '
                         f'training would push the head toward garbage (finding 7)')
      if _pop is not None and _pop < 0.90:
        print(f'[train] WARNING: {_aux_name} target only {_pop:.1%} populated in the sampled '
              f'corpus (healthy v7 measures ~99%) — masked records train nothing, so the '
              f'effective aux data volume is reduced accordingly', flush=True)

  # Curriculum prologue (CERES_MIX_PROLOGUE_POSITIONS): serve ONLY secondary
  # (e.g. puzzle) batches for the first N positions, then switch to the normal
  # ratio cycle — single-run curriculum, no restart. Combined with
  # CERES_SECONDARY_LOSS_VALUE_MULT=0 etc., the prologue (and all later replay
  # batches) are automatically policy-only. On RESUME the already-trained
  # position count is parsed from the checkpoint filename's trailing number and
  # subtracted; if parsing fails the prologue is SKIPPED entirely (safe default:
  # resumes virtually always happen after the prologue, and replaying puzzle-only
  # batches mid-main-phase would be far worse than skipping).
  _PROLOGUE_POS = int(float(os.environ.get('CERES_MIX_PROLOGUE_POSITIONS', '0') or 0))
  _prologue_per_worker = 0
  if _PROLOGUE_POS > 0 and secondary_dataset is not None:
    _resume_pos = 0
    _resume_fn = getattr(config, 'Opt_CheckpointResumeFromFileName', None)
    if _resume_fn:
      _m = re.search(r'_(\d+)$', os.path.basename(_resume_fn))
      if _m:
        _resume_pos = int(_m.group(1))
      else:
        print(f'[mixed-dataset] WARNING: cannot parse position count from resume checkpoint '
              f'{_resume_fn!r} — SKIPPING prologue on this (re)start.')
        _resume_pos = _PROLOGUE_POS
    _remaining = max(0, _PROLOGUE_POS - _resume_pos)
    _workers_eff = max(1, NUM_DATASET_WORKERS)
    _prologue_per_worker = _remaining // (batch_size_forward * _workers_eff)
    print(f'[mixed-dataset] curriculum prologue: {_remaining} positions secondary-only '
          f'({_prologue_per_worker} batches/worker x {_workers_eff} workers x {world_size} ranks)')

  dataset = TPGMixedDataset(primary_dataset, secondary_dataset,
                            int(getattr(config, 'Data_RatioSet1ToSet2', 0) or 0),
                            prologue_batches_per_worker=_prologue_per_worker)

  dataloader = DataLoader(dataset, batch_size=None, pin_memory=True, num_workers=NUM_DATASET_WORKERS, worker_init_fn=worker_init_fn, prefetch_factor=PREFETCH_FACTOR)
  # NOTE: previously wrapped with fabric.setup_dataloaders to auto-move batches to
  # device. We now move batches explicitly inside the training loop with
  # _move_batch_to_device() — avoids Lightning's recursive _apply_to_collection_slow
  # walk which intermittently wedged at CUDA-sync points.

  if IS_MASTER:
    config.pretty_print()
    print_model_trainable_details(model)


  NUM_POS_TO_SKIP = 0
  
  COMPUTE_FLOPS = False # WARNING: This is disabled because it causes dramatically higher VRAM usage on GPU 0, use only to generate stats.
  FLOPS_CALCULATED = False
  
  if config.Opt_CheckpointResumeFromFileName is not None:
    loaded = torch.load(config.Opt_CheckpointResumeFromFileName, map_location=device)

    # AUX-WIDTH GUARD: the checkpoint's embedding layer encodes how many input
    # features (hence aux channels) it was trained with. If that disagrees with the
    # model we just built, the strict=False resume paths below would SILENTLY skip
    # the embedding and leave it randomly initialized (a garbage net, no error).
    # Fail loudly here instead, naming the exact env var to set. (Relevant now that
    # CERES_AUX_FEATURES_PER_SQUARE defaults to 4: resuming a legacy 137-channel net
    # requires CERES_AUX_FEATURES_PER_SQUARE=0.)
    _ckpt_emb_w = loaded["model"].get("embedding_layer.weight", None)
    if _ckpt_emb_w is not None:
      _ckpt_in = _ckpt_emb_w.shape[1]
      _model_in = model_nocompile.embedding_layer.weight.shape[1]
      if _ckpt_in != _model_in:
        _prior = config.NetDef_PriorStateDim
        _ckpt_aux = _ckpt_in - NUM_INPUT_BYTES_PER_SQUARE - _prior
        _model_aux = _model_in - NUM_INPUT_BYTES_PER_SQUARE - _prior
        raise ValueError(
          f"Aux-feature width mismatch on resume: checkpoint embedding expects "
          f"{_ckpt_in} input features/square ({_ckpt_aux} aux), but this run built a model "
          f"with {_model_in} ({_model_aux} aux). Set CERES_AUX_FEATURES_PER_SQUARE={_ckpt_aux} "
          f"to match the checkpoint, then re-run. "
          f"(checkpoint: {config.Opt_CheckpointResumeFromFileName})")

    # QAT checkpoints carry fake-quant buffers (`<module>._fq_act_range`) that the
    # freshly-built (pre-convert) model does not have yet — convert_to_fake_quant()
    # re-creates and re-calibrates them after this load. Strip them so the
    # strict=True resume path below doesn't reject them as unexpected keys. No-op
    # for non-QAT checkpoints (which have no such keys).
    _fq_keys = [k for k in loaded["model"] if "._fq_" in k]
    if _fq_keys:
      for k in _fq_keys:
        del loaded["model"][k]
      print(f"INFO: QAT_RESUME stripped {len(_fq_keys)} fake-quant buffer keys "
            f"from checkpoint (ranges will be re-calibrated)", flush=True)

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
    # Aux-head prefix registry — SINGLE SOURCE OF TRUTH for "params that can
    # legitimately exist on exactly one side of a resume" (2026-08-21 action
    # review finding 6: the LoRA remap kept a second, divergent hardcoded
    # list). Consumed by BOTH the non-LoRA reconciliation below and the LoRA
    # base-checkpoint remap.
    #
    # Config/env-gated heads, auxiliary/training-only — dropping or
    # fresh-initializing them never corrupts the served heads. (action_head is
    # exported but zero-impact on the other heads; fresh-init is exactly the
    # from-scratch state.)
    _AUX_HEAD_PREFIXES = ('placement_value_', 'survival_head.', 'stvalue_', 'vda_', 'phase_film', 'ray_bias_', 'depth_probe_', 'depth_ctl_', 'rc_', 'vc_head.', 'sp_head.', 'hlg_head.', 'opt_head.', 'oppp_head.', 'action_head.')
    # Private value front-end, 'inject' mode ONLY: new modules that can legitimately
    # exist on one side of a resume — they are zero-init, so the net is bit-identical
    # to the base at step 0 and fresh-initializing them is exactly right.
    #
    # Deliberately NOT extended to 'replace'. That mode rewires the value family's
    # INPUT, so a fresh-initialized value_premap would feed the trained value head a
    # random private vector. It is tempting to rely on 'replace' also resizing
    # value_head/value2/unc/hlg (load_state_dict rejects a size mismatch even with
    # strict=False), but the widths COINCIDE whenever 64*ValueHeadChannels ==
    # HEAD_IN_SIZE — and since HEAD_IN_SIZE = 64 * (HEAD_PREMAP_PER_SQUARE // 4),
    # at EMBEDDING_DIM 256 that collapses to simply ValueHeadChannels ==
    # HeadWidthMultiplier (at 384/mult 2 it is ValueHeadChannels == 3). Exactly the
    # small channel counts a C-ablation would sweep. With every shape matching, the
    # load would succeed and the corruption would be announced by nothing louder than
    # an INFO line. Leaving the prefixes out here routes that case into the strict
    # load / _missing_other check instead, which raises.
    if getattr(model_nocompile, 'value_priv_inject_mode', False):
      _AUX_HEAD_PREFIXES = _AUX_HEAD_PREFIXES + ('value_premap.', 'value_priv_inject.', 'value2_priv_inject.')
    # Vis edge-bias params (CERES_VIS_EDGE_BIAS/_GATES): the form-A projections
    # are top-level ('vis_edge_proj.') but the B/C gate params live NESTED in
    # each layer's attention (transformer_layer.N.attention.attack_gate_*), so
    # prefix matching can't express them — use a predicate. All are zero-init,
    # so fresh-initializing on resume reproduces the base net exactly (same
    # rationale as the 'inject' private-value front-end above).
    _AUX_HEAD_PREFIXES = _AUX_HEAD_PREFIXES + ('vis_edge_proj.',)
    # Dual-plane family + 2026-08-20/21 modules (review finding #5): all are
    # exact step-0 no-ops (zero-init couplings), so fresh-init on warm start
    # reproduces the base net — same contract as the inject front-end.
    _AUX_HEAD_PREFIXES = _AUX_HEAD_PREFIXES + (
        'dual_plane.', 'dp_value_inject.', 'dp_value2_inject.', 'dp_pol_q.',
        'dp_pol_p.', 'dpva_', 'dp_surv_head.', 'dpv_a.', 'dpv_b.', 'dpe_w.',
        'dpd_in.', 'dpd_out.', 'kdist_proj.', 'spe_proj.', 'cbk_')
    # Graph-route heads + tactic refiner (2026-08 tactical program): the
    # refiner is top-level ('tactical_refiner.'), the route params live
    # nested per attention layer (transformer_layer.N.attention.graph_route_*).
    # Both are exact step-0 no-ops (zero-init proj_out / tanh-zero gate), so
    # fresh-initializing on warm-start reproduces the base net exactly.
    _AUX_HEAD_PREFIXES = _AUX_HEAD_PREFIXES + ('tactical_refiner.',)
    # Value min/max pool injectors (NetDef ValueHeadMinMaxPool): top-level,
    # zero-init — fresh-initializing on warm start reproduces the base net.
    _AUX_HEAD_PREFIXES = _AUX_HEAD_PREFIXES + ('value_pool_inject.', 'value2_pool_inject.')
    def _is_aux_key(k):
      return k.startswith(_AUX_HEAD_PREFIXES) or '.attack_gate_' in k or '.graph_route_' in k

    if config.Opt_LoRARankDivisor == 0 and not _body_lora_active:
      # Placement value head etc. are config/env-gated, so their params can
      # exist on exactly one side of a resume. Handle both directions LOUDLY
      # here instead of dying in the strict load.
      _ckpt_model_sd = loaded["model"]
      _model_has_placement = any(_is_aux_key(k) for k in model_nocompile.state_dict())
      _ckpt_placement_keys = [k for k in _ckpt_model_sd if _is_aux_key(k)]
      _model_aux_keys = {k for k in model_nocompile.state_dict() if _is_aux_key(k)}
      _dropped = [k for k in _ckpt_placement_keys if k not in _model_aux_keys]
      if _dropped:
        print(f"INFO: AUX_HEAD checkpoint keys dropped (env var not set this run): {_dropped}")
        _ckpt_model_sd = {k: v for k, v in _ckpt_model_sd.items() if k not in _dropped}
      _fresh = [k for k in _model_aux_keys if k not in _ckpt_model_sd]
      if _fresh:
        print(f"INFO: AUX_HEAD newly enabled on resume; params start fresh-initialized: {sorted(_fresh)}")
        _pl_res = model_nocompile.load_state_dict(_ckpt_model_sd, strict=False)
        _missing_other = [k for k in _pl_res.missing_keys if not _is_aux_key(k)]
        if _missing_other or _pl_res.unexpected_keys:
          raise RuntimeError(f"Resume mismatch beyond aux heads: missing={_missing_other} unexpected={_pl_res.unexpected_keys}")
        # vda mode-3 -> mode-4 warm start: the aux value head sees exactly the
        # augmented input the checkpoint's value_head was trained on, so inherit
        # those weights instead of fresh-initializing (the SERVED value_head also
        # keeps them and re-adapts to the plain input).
        if getattr(model_nocompile, 'vda_mode', 0) == 4 and any(k.startswith('vda_aux_head.') for k in _fresh):
          _vh_sd = model_nocompile.value_head.state_dict()
          model_nocompile.vda_aux_head.load_state_dict(_vh_sd)
          print("INFO: VDA mode-4 warm start — vda_aux_head inherited value_head weights", flush=True)
      else:
        # load checkpoint parameters, expect all to match (strict = True)
        model_nocompile.load_state_dict(_ckpt_model_sd, strict = True)
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
        elif "value_premap" in name or "_priv_inject" in name:
          pass # private value front-end ('inject' mode) is new — not in orig ckpt.
               # Injectors are zero-init, so keeping their init values is exactly
               # what reproduces the base net at step 0.
        elif "vis_edge_proj" in name or "attack_gate_" in name:
          pass # vis edge-bias modules (CERES_VIS_EDGE_BIAS/_GATES) are new — not
               # in orig ckpt; zero-init, so keeping init values reproduces the
               # base net at step 0.
        elif "graph_route" in name or "tactical_refiner" in name:
          pass # graph-route heads + tactic refiner (2026-08 tactical program)
               # are new — not in orig ckpt; exact step-0 no-ops (zero-init
               # proj_out / tanh-zero gate), so keeping init values reproduces
               # the base net at step 0.
        elif "_pool_inject" in name:
          pass # value min/max pool injectors (NetDef ValueHeadMinMaxPool) are
               # new — not in orig ckpt; zero-init, so keeping init values
               # reproduces the base net at step 0.
        else:
          # Map to the original name (before it was subsumed within original_layer)
          name_in_checkpoint = name.replace("original_layer.", "") if "original_layer" in name else name
          if name_in_checkpoint in loaded["model"]:
            new_state_dict[name] = loaded["model"][name_in_checkpoint]
          elif _is_aux_key(name):
            # Aux head newer than the base checkpoint (review 2026-08-21
            # finding 9; action review finding 6: reuse the SAME registry as
            # the non-LoRA path, not a second hand-maintained list) — keep
            # init values, the head trains from scratch during the fine-tune.
            pass
          else:
            raise KeyError(f'LoRA base checkpoint is missing required weight {name_in_checkpoint!r} '
                           f'(model layer {name!r}) and it is not a known-new aux head')

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
    if config.Opt_Optimizer == 'Muon' and groups_match:
      # Positional-state guard: optimizer state_dicts key per-param state by POSITION
      # within the group, and the Muon group's order is muon_params + adamw_params.
      # If the partition changed vs the checkpoint (e.g. new MuonAdamWScope), that
      # order changes too, so a warm load would re-key moment buffers onto the WRONG
      # params — silent corruption wherever shapes coincide. Detect via per-position
      # use_muon flags; on mismatch start optimizer state fresh (model weights are
      # unaffected; Muon momentum and Adam moments rebuild within a few hundred steps).
      _ld_st = loaded_optimizer_state.get("state", {})
      _cur_flags = [bool(optimizer.state[p]["use_muon"])
                    for grp in optimizer.param_groups for p in grp["params"]]
      _partition_changed = any(
        (_ld_st.get(i, {}).get("use_muon") is not None)
        and (bool(_ld_st[i]["use_muon"]) != cur)
        for i, cur in enumerate(_cur_flags))
      if _partition_changed:
        print("[checkpoint-resume] Muon partition differs from checkpoint "
              "(MuonAdamWScope change?) — starting optimizer state fresh to avoid "
              "positional state misalignment", flush=True)
        loaded_optimizer_state["state"] = {}
    optimizer.load_state_dict(loaded_optimizer_state)

    # Re-assert construction-time Muon settings that load_state_dict clobbers
    # (or drops entirely on the fresh-state paths above):
    #   1) per-param use_muon flags — step() requires them; loaded positional flags
    #      are stale across partition changes, and the fresh-state paths drop them
    #      entirely (previously step() would have crashed with KeyError).
    #   2) adamw_lr_ratio — a NEW LearningRateBaseHeads must take effect on resume
    #      (the scheduler re-asserts 'lr' from base_lrs every step, but nothing
    #      else re-reads the ratio).
    #   4) momentum / adamw_betas / adamw_eps / wd — same class of problem: these
    #      live in the Muon param_groups, so load_state_dict replaces them with
    #      whatever the checkpoint carried whenever the groups match positionally
    #      (the mismatch path above already substitutes current_param_groups).
    #      Without this, changing MuonMomentum/MuonAdamWEps/Beta*/WeightDecay and
    #      resuming silently keeps the OLD values, while the construction-time
    #      print ~600 lines up claims the new ones. Config wins on resume.
    if config.Opt_Optimizer == 'Muon':
      for _p in muon_params:
        _st = optimizer.state[_p]
        if _st.get("use_muon") is False:  # moved AdamW->Muon: purge stale Adam moments
          for _k in ("step", "moment1", "moment2"):
            _st.pop(_k, None)
        _st["use_muon"] = True
        # 3) head_split specs — construction-time truth (MuonPerHeadAttention) wins
        #    over whatever the checkpoint carried, in both directions. Momentum
        #    buffers are full-matrix under both modes, so they stay warm.
        _spec = optimizer.head_split_specs.get(_p)
        if _spec is not None:
          _st["head_split"] = _spec
        else:
          _st.pop("head_split", None)
      for _p in adamw_params:
        _st = optimizer.state[_p]
        if _st.get("use_muon") is True:   # moved Muon->AdamW: purge stale momentum
          _st.pop("momentum_buffer", None)
        _st["use_muon"] = False
      _hl = getattr(config, 'Opt_LearningRateBaseHeads', None)
      _new_ratio = (float(_hl) / LR) if _hl is not None else 1.0
      _muon_hparams = {
        'adamw_lr_ratio': _new_ratio,
        'momentum': _muon_mom,
        'adamw_betas': (config.Opt_Beta1, config.Opt_Beta2),
        'adamw_eps': _muon_aeps,
        'wd': WEIGHT_DECAY,
      }
      for g in optimizer.param_groups:
        for _k, _v in _muon_hparams.items():
          if g.get(_k) != _v:
            print(f"[checkpoint-resume] re-asserting Muon {_k} "
                  f"{g.get(_k)} -> {_v}", flush=True)
            g[_k] = _v

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

  # Pass the raw module (LossCalculator just stores the ref for metadata; the
  # DDP wrapper would not expose CeresNet's custom attributes).
  loss_calc = LossCalculator(core)

  # INT8 Quantization-Aware Training (QAT). Env-gated (CERES_QAT_INT8) so no
  # config-schema / C# change. Swaps every nn.Linear -> FakeQuantLinear in place
  # (preserves Parameter identity, so the already-built optimizer is unaffected;
  # frozen-range mode adds no new params). Fake-quant is gated on training mode,
  # so save_model()'s eval() export stays clean FP16; the real INT8 QDQ is then
  # applied by scripts/qdq_export.py at deploy. Intended use: short KL-anchor
  # distillation fine-tune (Opt_KLAnchorRefCheckpoint = the FP teacher) of the
  # from_onnx flagship checkpoint, so the weights become INT8-robust. The
  # reference/teacher model is NOT converted (only model_nocompile).
  QAT_ENABLED = int(os.environ.get('CERES_QAT_INT8', '0') or 0) > 0
  QAT_CALIB_POS = int(os.environ.get('CERES_QAT_CALIB_POS', '200000') or 200000)
  # Calibrate for CERES_QAT_CALIB_POS positions measured FROM THE CURRENT num_pos
  # (0 on a fresh run, the checkpoint step on a resume). Without the resume offset
  # a resumed QAT run would freeze on step 0 (num_pos already >> calib_pos) with
  # uncalibrated (range=1.0) activation buffers.
  QAT_FREEZE_AT = num_pos + QAT_CALIB_POS
  if QAT_ENABLED:
    import fake_quant
    _qat_pct = float(os.environ.get('CERES_QAT_PERCENTILE', '99.999') or 99.999)
    _qat_excl = [s for s in os.environ.get('CERES_QAT_EXCLUDE', '').split(',') if s]
    _qat_wonly = int(os.environ.get('CERES_QAT_WEIGHTS_ONLY', '0') or 0) > 0
    _qat_wfrozen = int(os.environ.get('CERES_QAT_FROZEN_WSCALE', '0') or 0) > 0
    _nq, _ne = fake_quant.convert_to_fake_quant(
        model_nocompile, percentile=_qat_pct, exclude=_qat_excl,
        quant_weights=True, quant_acts=(not _qat_wonly),
        freeze_weight_scales=_qat_wfrozen)
    print(f"INFO: QAT_INT8 enabled — converted {_nq} nn.Linear "
          f"(excluded {_ne}); percentile={_qat_pct} weights_only={_qat_wonly} "
          f"frozen_wscale={_qat_wfrozen} "
          f"calib_pos={QAT_CALIB_POS} freeze_at_pos={QAT_FREEZE_AT}", flush=True)

  model.train()

  wdl_reverse = torch.tensor([2, 1, 0]) # for reversing perspective on WDL
  

  _last_show_losses_pos = 0
  # Train Network
  for batch_idx, (batch) in enumerate(dataloader):
    if (num_pos >= MAX_POSITIONS and not config.Exec_ExportOnly):
        break

    # Move the freshly-fetched batch to GPU. Replaces Lightning's recursive
    # auto-move; with pin_memory=True these are true async DMA transfers.
    batch = _move_batch_to_device(batch, device)

    fraction_complete = num_pos / MAX_POSITIONS
    model.train()

    # QAT: observe activation ranges for the first CERES_QAT_CALIB_POS positions,
    # then freeze them (cheap no-op after the one-time freeze).
    if QAT_ENABLED:
      fake_quant.freeze_if_ready(model_nocompile, num_pos, QAT_FREEZE_AT)

    # Periodically log statistics. Interval-passed, NOT modulo: since replayed
    # rows no longer count toward num_pos, the counter advances in uneven steps
    # once injection starts, and `num_pos % 65536 == 0` then simply never fires
    # again — every TB stat silently stopped for the rest of the run (observed
    # in smoke_rep5: zero hard_replay metrics after the 50% switch-on).
    show_losses = (num_pos - _last_show_losses_pos) >= (1024 * 64)
    if show_losses:
      _last_show_losses_pos = num_pos

    is_accumulating = ((batch_accumulation_counter + 1) % num_batches_gradient_accumulate) != 0

    # Arm the gradient-conflict probe for this micro-batch (see setup block). Measured
    # on the FIRST micro-batch of an accumulation group: one micro-batch is a fair
    # sample of the angle, and probing the last one would mean differentiating a graph
    # that the accumulated backward is about to consume. Must be set before the forward,
    # since compute_loss builds the per-family subtotals only when armed.
    if GC_PROBE_STEPS > 0:
      core._gc_probe_now = (batch_accumulation_counter % num_batches_gradient_accumulate == 0) \
                           and (_gc_opt_steps % GC_PROBE_STEPS == 0)
      core._gc_policy_loss = None
      core._gc_value_loss = None
    # DDP gradient-accumulation: suppress the cross-rank all-reduce on every micro-
    # step EXCEPT the last one of an accumulation window (this is exactly what
    # model.no_sync() toggles; setting the flag avoids re-indenting the forward
    # block). On the final, non-accumulating micro-step the flag is True, so the
    # backward all-reduces the summed gradients once. No-op in single-GPU mode.
    #
    # ⚠ NOT under static_graph: the no_sync path is explicitly unsupported there
    # (PyTorch docs), and the combination dies inside the reducer with
    # "expect_autograd_hooks_ INTERNAL ASSERT FAILED" on the first accumulating
    # backward — which every production config hits, since they accumulate
    # (512 forward / 4096 backward = 8 micro-steps). Leaving sync ON every
    # micro-step is mathematically IDENTICAL: each micro-step all-reduces (i.e.
    # averages across ranks) its own gradients and accumulates them into .grad,
    # and sum-over-microsteps of mean-over-ranks == mean-over-ranks of
    # sum-over-microsteps. The only cost is one all-reduce per micro-step
    # instead of one per optimizer step — bandwidth, not correctness.
    if IS_DISTRIBUTED and not DDP_STATIC_GRAPH:
      model.require_backward_grad_sync = (not is_accumulating)
    # Autocast handles bf16-mixed precision (Fabric did this implicitly via
    # precision='bf16-mixed').
    _amp_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if USE_AUTOCAST else contextlib.nullcontext()
    with _amp_ctx:
      this_lr = scheduler.get_last_lr()[0]

      if config.Exec_ExportOnly:
        assert config.Opt_CheckpointResumeFromFileName is not None, "ExportOnly specified but no checkpoint file specified"
        print("Exporting to files with postexport suffix....")
        # Export OUTSIDE autocast: tracing under autocast(bf16) bakes BF16-typed
        # ops into the ONNX graph (TRT's Mish importer rejects BF16); the regular
        # interval saves run outside the autocast region and are clean.
        with torch.amp.autocast('cuda', enabled=False):
          save_model(NAME, OUTPUTS_DIR, config, model_nocompile, state, "postexport", True)
        print("INFO: EXIT_STATUS", "SUCCESS")
        exit(3)

      if COMPUTE_FLOPS and not FLOPS_CALCULATED and torch.cuda.is_available():
        calc_flops(model_nocompile.to(torch.float), batch[0], loss_calc, optimizer, num_pos, config.Opt_BatchSizeForwardPass, calc_backward=False)
        calc_flops(model_nocompile.to(torch.float), batch[0], loss_calc, optimizer, num_pos, config.Opt_BatchSizeForwardPass, calc_backward=True)
        optimizer.zero_grad()
        FLOPS_CALCULATED = True

        
      # Reset per ITERATION at loop level, not inside the BOARDS_PER_BATCH==1
      # branch: Python does not scope by block, so a branch-local init would
      # leave the PREVIOUS step's value visible to the position count below,
      # and would be undefined entirely on the BOARDS_PER_BATCH>1 path where
      # hard replay is not implemented at all.
      _replay_entries = []
      if BOARDS_PER_BATCH == 1:
        batch = batch[0]
        is_secondary_batch = bool(batch.pop('is_secondary', False))
        num_processing_now = batch['squares'].shape[0]

        # Hard-replay INJECTION (see setup block): swap buffered hard rows into
        # this primary batch before the forward pass. Batched per KEY (one stack
        # + one H2D transfer per tensor key) instead of per row x key — the naive
        # loop cost ~1000 small dispatches/step.
        _replay_slots = None
        # Start delay: the hard set turns over early, so focusing before the net
        # has matured spends the budget on rows ordinary training resolves anyway
        # (see config.py for the measured turnover).
        _hr_on = (HARD_REPLAY_SIZE > 0 and fraction_complete >= HARD_REPLAY_START_FRAC)
        # Inject-EMA oppdateres FØR buffer-guarden: ellers fryser den på sin
        # siste verdi når bufferet tømmes, og de to diagnosene som skal avsløre
        # sult rapporterer helse nøyaktig ved sult.
        if _hr_on and not is_secondary_batch:
          _hr_inject_ema = 0.99 * _hr_inject_ema  # decays; re-added below on injection
        if _hr_on and not is_secondary_batch and _hard_replay_buf:
          _rb = batch['squares'].shape[0]
          _rcap = int(_rb * HARD_REPLAY_FRAC)
          if HARD_REPLAY_REUSE_TARGET > 0 and _hr_intake_ema is not None:
            # Target x realised admissions, CAPPED by HardReplayFraction. The cap
            # binds while admissions are plentiful (early: lower reuse, more
            # variety); the target binds once the floor makes them scarce.
            _rk = min(int(round(HARD_REPLAY_REUSE_TARGET * _hr_intake_ema)), _rcap)
          else:
            _rk = _rcap
          # Floor at 1 whenever the buffer is non-empty: at 0 the eviction paths
          # below never run and the buffer would freeze full of stale entries
          # while `hard_replay_buffer_fill` still looked healthy.
          _rk = min(max(_rk, 1), len(_hard_replay_buf))
          # Min-population gate: right after switch-on the buffer holds a
          # handful of entries, and _rk clamped to len(buf) would inject EVERY
          # entry into EVERY micro-batch of the accumulation window — one row
          # contributing 8x to a single optimizer step at full LR, while its
          # 'uses' burns toward MaxReuse in one window. Wait until the buffer
          # can cover a full injection quota (fills in a few steps).
          if len(_hard_replay_buf) < _rcap:
            _rk = 0
          # Key-signature matching: under sidecar 'auto' modes, batches from
          # different shards can carry different target-key sets — an entry may
          # only be injected into a batch with the SAME tensor keys (otherwise the
          # victim row's survival/censored/provenance targets would silently
          # label the replayed position).
          _bsig = frozenset(_k for _k, _v in batch.items() if torch.is_tensor(_v))
          # randperm, NOT randint: with-replacement sampling injects one entry
          # into several slots of the SAME micro-batch. randperm fixes only that
          # granularity — the same entry can still recur across the micro-batches
          # of one accumulation window (~5-6% of injected rows at steady state),
          # which the min-population gate above keeps bounded; a per-window draw
          # would remove it entirely if it ever proves to matter.
          _picks0 = torch.randperm(len(_hard_replay_buf))[:_rk].tolist()
          _sel = [(_bi, _hard_replay_buf[_bi]) for _bi in _picks0
                  if _hard_replay_buf[_bi]['sig'] == _bsig]
          if _sel:
            _picks = [_bi for _bi, _ in _sel]
            _entries = [_e for _, _e in _sel]
            _replay_slots = torch.randperm(_rb)[:len(_entries)]
            _replay_entries = _entries
            _slots_dev = _replay_slots.to(batch['squares'].device)
            for _k in _entries[0]['rows']:
              _stk = torch.stack([_e['rows'][_k] for _e in _entries])
              batch[_k][_slots_dev] = _stk.to(batch[_k].device, non_blocking=True)
            for _e in _entries:
              _e['uses'] += 1
              _e['miss'] = 0
            _hr_inject_ema += 0.01 * len(_entries)
            _hr_displaced_total += len(_entries)
          else:
            _picks = []
          # ONE combined descending eviction pass over every index touched this
          # step. Two separate passes would leave the second dereferencing indices
          # the first had already swap-removed (IndexError, and under DDP a hang
          # rather than a clean crash, since the other ranks stay in all-reduce).
          _missed = set(_picks0) - set(_picks)
          for _bi in _missed:
            _hard_replay_buf[_bi]['miss'] += 1
          _dead = {_bi for _bi in _picks0
                   if _hard_replay_buf[_bi]['uses'] >= HARD_REPLAY_MAX_REUSE
                   or _hard_replay_buf[_bi]['miss'] >= HARD_REPLAY_MISS_LIMIT}
          for _bi in _dead:
            _hard_replay_buf[_bi]['gone'] = True
          for _bi in sorted(_dead, reverse=True):
            _hard_replay_buf[_bi] = _hard_replay_buf[-1]
            _hard_replay_buf.pop()

        policy_out, value_out, moves_left_out, unc_out, value2_out, q_deviation_lower, q_deviation_upper, uncertainty_policy_out, _, _, _ = model(batch['squares'], None)

        # Per-stream loss routing: for secondary-corpus batches, temporarily swap the
        # loss weights on `core` (compute_loss reads self.*_loss_weight at call time;
        # the forward above already ran, so head construction/routing is unaffected).
        _use_overrides = is_secondary_batch and bool(SECONDARY_WEIGHT_OVERRIDES)
        if _use_overrides:
          _saved_w = {attr: getattr(core, attr) for attr in SECONDARY_WEIGHT_OVERRIDES}
          for attr, value in SECONDARY_WEIGHT_OVERRIDES.items():
            setattr(core, attr, value)
        try:
          loss = core.compute_loss(loss_calc, batch, policy_out, value_out, moves_left_out, unc_out,
                                    value2_out, q_deviation_lower, q_deviation_upper, uncertainty_policy_out,
                                    None, None,
                                    None, None,
                                    None,
                                    0, num_pos, this_lr, show_losses)
        finally:
          if _use_overrides:
            for attr, value in _saved_w.items():
              setattr(core, attr, value)

        # Mirror-consistency value regularizer (see setup block above).
        # PRIMARY batches only: secondary (puzzle) batches must neither receive
        # value-side mirror gradients (policy-only phases!) nor feed the
        # thermostat's level with a different distribution. Hysteresis: in
        # dormant mode the term runs (and enforces) only on the final
        # micro-batch of every MIRROR_PROBE_STEPS-th OPTIMIZER step (counter
        # deliberately matches the config doc's units under grad accumulation).
        _mirror_run_now = MIRROR_CONS_WEIGHT > 0 and not is_secondary_batch
        if MIRROR_PROBE_STEPS > 0:
          if not is_accumulating:
            _mirror_step += 1
          if _mirror_run_now and not _mirror_active:
            _mirror_run_now = (not is_accumulating) and (_mirror_step % MIRROR_PROBE_STEPS == 0)
        if _mirror_run_now:
          _sq = batch['squares']
          _elig = (_sq[:, 0, 112:116].sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
          if _elig.numel() > 0:
            _k = max(1, int(_sq.shape[0] * MIRROR_CONS_FRAC))
            _pick = _elig[torch.randint(_elig.numel(), (_k,), device=_elig.device)]
            if _mirror_perm64_t.device != _sq.device:
              _mirror_perm64_t = _mirror_perm64_t.to(_sq.device)
            _msq = _sq[_pick][:, _mirror_perm64_t, :]
            _msq[:, :, 129:137] = torch.flip(_msq[:, :, 129:137], dims=[-1])
            # The second forward re-stashes gate/TSB diagnostics with the
            # mirror-subset shapes; the gate-sparsity regularizers consume them
            # AFTER this block, so preserve the main-batch stashes across it.
            _sav_gate = getattr(core, '_last_gate_value', None)
            _sav_tsb = getattr(core, '_last_tsb_gates', None)
            _, _mv_out, _, _, _, _, _, _, _, _, _ = model(_msq, None)
            core._last_gate_value = _sav_gate
            core._last_tsb_gates = _sav_tsb
            _lp_o = torch.log_softmax(value_out[_pick].float(), dim=-1)
            _lp_m = torch.log_softmax(_mv_out.float(), dim=-1)
            _p_o, _p_m = _lp_o.exp(), _lp_m.exp()
            _mc_loss = 0.5 * ((_p_o * (_lp_o - _lp_m)).sum(-1) + (_p_m * (_lp_m - _lp_o)).sum(-1)).mean()
            loss = loss + MIRROR_CONS_WEIGHT * _mc_loss
            if MIRROR_PROBE_STEPS > 0:
              _mcv = float(_mc_loss.detach())
              # Active mode: slow smoothing over per-step measurements.
              # Dormant mode: probes are sparse — blend fast AND let a single
              # raw probe above High reactivate immediately (a 0.98-smoothed
              # level would need ~85 probes to cross the threshold).
              _mom = 0.98 if _mirror_active else 0.5
              _mirror_level = _mcv if _mirror_level is None else _mom * _mirror_level + (1 - _mom) * _mcv
              if _mirror_active and _mirror_level < MIRROR_AUTO_LOW:
                _mirror_active = False
                print(f'[train] mirror-consistency DORMANT (level {_mirror_level:.5f} < {MIRROR_AUTO_LOW}); probing every {MIRROR_PROBE_STEPS} steps')
              elif not _mirror_active and _mcv > MIRROR_AUTO_HIGH:
                _mirror_active = True
                print(f'[train] mirror-consistency REACTIVATED (probe {_mcv:.5f} > {MIRROR_AUTO_HIGH})')
            if show_losses:
              core._log("value_mirror_cons_loss", _mc_loss, step=num_pos)
              if MIRROR_PROBE_STEPS > 0 and _mirror_level is not None:
                core._log("value_mirror_cons_level", _mirror_level, step=num_pos)
                core._log("value_mirror_cons_active", 1.0 if _mirror_active else 0.0, step=num_pos)

        # Hard-replay INTAKE: score the fresh (non-replayed) rows with the
        # configured selector (valuekl: per-sample value1-KL; topn: top-N margin
        # with eligibility gates) and buffer the ones above the mode's floor.
        # CPU copies, ~40KB/row.
        # Gate INTAKE on the start delay too, not just injection: filling the
        # buffer from step 0 would stock it with rows an immature net found
        # hard, and 54.3% of those fall below the floor on their own by the
        # end (measured). Replay would then switch on against a stale set —
        # exactly what the delay exists to avoid. Fill costs ~40 steps.
        if _hr_on and not is_secondary_batch:
          with torch.no_grad():
            _vt = (batch['wdl_q'].float() * core.q_ratio
                   + batch['wdl_deblundered'].float() * (1.0 - core.q_ratio))
            _hce = torch.nn.functional.cross_entropy(value_out.float(), _vt, reduction='none')
            _htt = torch.clamp(_vt + 1e-6, min=1e-6)
            _hkl = _hce + (_htt * torch.log(_htt)).sum(dim=-1)
            if HARD_REPLAY_SELECTOR == 'topn':
              # Score = top-3 MARGIN in logit space: logit(3rd best legal move)
              # minus logit(target's best move). >0 = the clear best move is NOT
              # in the net's top-3 (admit); more negative = contained by a wider
              # margin (exit on re-score below -ExitMargin). Rows failing the
              # eligibility gates (no clear solution, or drawish so the target
              # argmax is search noise) are forced to -1e9: never admitted, and
              # replayed rows always pass the gates since their (buffered)
              # targets were eligible at admission. NOTE: _hkl above is still
              # computed first because the fresh/replay VALUE diagnostics below
              # remain KL-based in both modes; only selection switches scale.
              _tn_tgt = batch['policies'].float()
              # "Ranked moves" = search-VISITED moves (target > 0), a legality
              # proxy — same convention as losses.py and the offline calibration
              # (top3_crit.py); strictly conservative vs true legality. Changing
              # this to a real legal mask would silently invalidate the measured
              # pbest/absq calibration.
              _tn_pol = policy_out.float()
              _tn_pol.masked_fill_(_tn_tgt <= 0, -1e9)
              _tn_pmax, _tn_best = _tn_tgt.max(dim=-1)
              _tn_nth = _tn_pol.topk(HARD_REPLAY_TOPN, dim=-1).values[:, HARD_REPLAY_TOPN - 1]
              _tn_bestlogit = _tn_pol.gather(1, _tn_best.unsqueeze(1)).squeeze(1)
              # Clamp: an +inf margin (saturated logits on exactly the rows this
              # selector admits) would pass admission, pass the NaN guard, and
              # fail every floor comparison forever — an immortal entry pinning
              # the score metric at inf. Logit margins beyond +-100 carry no
              # information anyway.
              _tn_margin = torch.clamp(torch.nan_to_num(_tn_nth - _tn_bestlogit,
                                                        nan=-1e9, posinf=100.0, neginf=-1e9),
                                       min=-1e9, max=100.0)
              _wq = batch['wdl_q'].float()
              _tn_elig = ((_tn_pmax > HARD_REPLAY_TN_PBEST)
                          & ((_wq[:, 0] - _wq[:, 2]).abs() > HARD_REPLAY_TN_MINABSQ))
              _hsel = torch.where(_tn_elig, _tn_margin,
                                  torch.full_like(_tn_margin, -1e9))
            else:
              _hsel = _hkl

            # Separate fresh-vs-replayed diagnostics: batch metrics are
            # composition-shifted by the injected hard rows, so log each subset
            # on its own. The fresh numbers are the ones comparable to other
            # runs; the replay-vs-fresh gap shows what the revisits teach.
            if show_losses:
              _vmatch = (value_out.argmax(dim=-1) == _vt.argmax(dim=-1)).float() * 100.0
              if _replay_slots is not None:
                _repm = torch.zeros(_hkl.shape[0], dtype=torch.bool, device=_hkl.device)
                _repm[_replay_slots.to(_hkl.device)] = True
                core._log("value_kl_fresh", _hkl[~_repm].mean(), step=num_pos)
                core._log("value_kl_replay", _hkl[_repm].mean(), step=num_pos)
                core._log("value_acc_fresh", _vmatch[~_repm].mean(), step=num_pos)
                core._log("value_acc_replay", _vmatch[_repm].mean(), step=num_pos)
              else:
                core._log("value_kl_fresh", _hkl.mean(), step=num_pos)
                core._log("value_acc_fresh", _vmatch.mean(), step=num_pos)

            if _replay_slots is not None:
              # RE-SCORE the injected rows: 'kl' is otherwise an admission-time
              # snapshot, so at a reuse target of N most of the budget would go to
              # rows the net has since learned. Their current KL is already here.
              _rsl = _replay_slots.to(_hsel.device)
              _cur = _hsel[_rsl].detach().float().cpu().tolist()
              for _n, _e in enumerate(_replay_entries):
                # NaN fails every comparison, so a NaN kl would make the entry
                # immortal (never floor-dropped) and poison buffer_kl_mean for
                # the rest of the run. Force NaN to below the drop floor.
                _e['kl'] = _cur[_n] if _cur[_n] == _cur[_n] else -1e9
              if HARD_REPLAY_MIN_KL > 0 or HARD_REPLAY_SELECTOR == 'topn':
                # HYSTERESIS: drop well BELOW the admission floor, not at it.
                # Dropping at the same threshold churns entries out after ~1 use
                # (simulated realised reuse 5.2/3.1/2.0 against a target of 8 at
                # 30/50/80% re-score-below-floor rates) and starves the buffer.
                _dfloor = (-HARD_REPLAY_TN_EXIT if HARD_REPLAY_SELECTOR == 'topn'
                           else 0.5 * HARD_REPLAY_MIN_KL)
                # Only re-scored entries can have moved below the drop floor
                # (everything else was admitted at >= MinKL > 0.5*MinKL and its
                # stored kl has not changed since), so gate the O(buffer) scan
                # behind an O(_rk) test — it fires rarely, not every step.
                # 'gone' skips entries MaxReuse-evicted at injection this step:
                # their (collapsed) re-score would otherwise trip this gate and
                # walk the O(buffer) scan on steps where nothing can drop.
                if any(_e['kl'] <= _dfloor and not _e.get('gone')
                       for _e in _replay_entries):
                  _drop = sorted((_bi for _bi, _e in enumerate(_hard_replay_buf)
                                  if _e['kl'] <= _dfloor and _e['uses'] > 0), reverse=True)
                  for _bi in _drop:
                    _hard_replay_buf[_bi] = _hard_replay_buf[-1]
                    _hard_replay_buf.pop()
              _hsel[_rsl] = -1e9                            # replayed rows never re-enter
            # Intake quota stays at HardReplayFraction so the ABSOLUTE FLOOR does
            # the selecting. Sizing it as FRAC/ReuseTarget instead would cap
            # admissions at 1.56% while only ~1-3% of rows clear the floor, so the
            # rank quota would bind first and the floor would never remove a single
            # row — the arm would silently run legacy top-k at a smaller quota.
            _hnk = max(1, int(_hsel.shape[0] * HARD_REPLAY_FRAC))
            _hv, _hi = torch.topk(_hsel, min(_hnk, _hsel.shape[0]))
            _hfloor = (0.0 if HARD_REPLAY_SELECTOR == 'topn'
                       else (HARD_REPLAY_MIN_KL if HARD_REPLAY_MIN_KL > 0 else 0.0))
            _hkeep = _hv > _hfloor
            _hvk = _hv[_hkeep].cpu()
            _hi = _hi[_hkeep]
            # Updated on EVERY intake including empty ones, so the estimate decays
            # when the net stops failing instead of freezing at its last value.
            _adm = int(_hi.numel())
            # One-shot saturation warning (finding: if admissions fill the rank
            # quota persistently, ReuseTarget degenerates toward reuse ~1 — the
            # regime the quota-vs-floor design assumes is transient).
            global _hr_quota_warned
            if (_adm >= _hnk and HARD_REPLAY_REUSE_TARGET > 0
                and not globals().get('_hr_quota_warned')):
              _hr_quota_warned = True
              print(f'[train] HARD-REPLAY NOTE: admissions saturate the intake quota '
                    f'({_adm}/{_hnk}); while this persists, realised reuse sits near '
                    f'inject/admit rather than the ReuseTarget. Expected transiently '
                    f'after switch-on; investigate if hard_replay_reuse_realised stays low.')
            _hr_intake_ema = float(_adm) if _hr_intake_ema is None                              else 0.99 * _hr_intake_ema + 0.01 * _adm
            if _hi.numel() > 0:
              # One indexed gather + one D2H transfer per key; per-row entries
              # hold zero-copy CPU views into the shared intake tensors.
              # .clone() below: _v[_j] is a storage-sharing VIEW into this block,
              # so one surviving entry pins the entire block's host memory
              # (measured 128x amplification). The selective floor-drop is the
              # worst case for that, since it retires an arbitrary subset.
              _rows_cpu = {_k: _t[_hi].detach().to('cpu', copy=True)
                           for _k, _t in batch.items() if torch.is_tensor(_t)}
              _rsig = frozenset(_rows_cpu)
              for _j in range(_hi.numel()):
                _entry = {'uses': 0, 'kl': float(_hvk[_j]), 'sig': _rsig, 'miss': 0,
                          'rows': {_k: _v[_j].clone() for _k, _v in _rows_cpu.items()}}
                if len(_hard_replay_buf) >= HARD_REPLAY_SIZE:
                  # random replacement at cap (buffer churns; strict FIFO not needed)
                  _hard_replay_buf[int(torch.randint(len(_hard_replay_buf), (1,)))] = _entry
                else:
                  _hard_replay_buf.append(_entry)
            if show_losses:
              core._log("hard_replay_buffer_fill", float(len(_hard_replay_buf)), step=num_pos)
              # Realised volumes. Without these the design is blind: a floor that
              # admits nothing and a starved buffer look identical to a healthy run.
              # Reuse is EMA/EMA — a smoothed numerator over a raw per-step
              # denominator reads 8x/64x high on low-admission steps, which become
              # common exactly as the treatment self-limits.
              core._log("hard_replay_intake_rows", float(_adm), step=num_pos)
              core._log("hard_replay_intake_ema", float(_hr_intake_ema or 0.0), step=num_pos)
              core._log("hard_replay_inject_ema", float(_hr_inject_ema), step=num_pos)
              core._log("hard_replay_inject_frac_realised",
                        float(_hr_inject_ema) / max(_hkl.shape[0], 1), step=num_pos)
              # Unique-data displacement, reported instead of being subtracted
              # from num_pos (see the num_pos comment for why subtracting is
              # rank-divergent under DDP). x WORLD_SIZE approximates the global
              # figure since ranks draw comparable volumes.
              core._log("hard_replay_displaced_pos",
                        float(_hr_displaced_total * WORLD_SIZE), step=num_pos)
              core._log("hard_replay_reuse_realised",
                        float(_hr_inject_ema) / max(float(_hr_intake_ema or 0.0), 1e-6), step=num_pos)
              if _hard_replay_buf:
                # Buffer population stats (logging cadence only): how hard the
                # current population is, and typical reuse progress.
                core._log("hard_replay_buffer_score_mean" if HARD_REPLAY_SELECTOR == 'topn'
                          else "hard_replay_buffer_kl_mean",
                          sum(_e['kl'] for _e in _hard_replay_buf) / len(_hard_replay_buf), step=num_pos)
                core._log("hard_replay_buffer_uses_mean",
                          sum(_e['uses'] for _e in _hard_replay_buf) / len(_hard_replay_buf), step=num_pos)

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
        loss1 = core.compute_loss(loss_calc, sub_batch, policy_out1, value_out1, moves_left_out1, unc_out1,
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
          
        loss2 = core.compute_loss(loss_calc, sub_batch, policy_out2, value_out2, moves_left_out2, unc_out2,
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

        loss3 = core.compute_loss(loss_calc, sub_batch, policy_out3, value_out3, moves_left_out3, unc_out3,
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
          
          loss4 = core.compute_loss(loss_calc, sub_batch, None, None, None, None,
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
      if getattr(core, 'use_gtab', False) and getattr(core, '_last_gate_value', None) is not None:
        _gtab_lambda = float(os.environ.get('CERES_GTAB_GATE_LAMBDA', '0.01') or 0.01)
        loss = loss + _gtab_lambda * core._last_gate_value.mean()

      # TSB gate-sparsity regularizer: penalize unnecessary tactical-branch firing.
      # Each transformer block has a per-block scalar gate; the net stacks them
      # into _last_tsb_gates of shape [num_layers, B, 1, 1]. Mean across layers and
      # batch encourages each block's gate to stay near 0 unless the puzzle loss
      # gain from firing exceeds the sparsity cost. Default lambda 0.01.
      if getattr(core, 'use_tsb', False) and getattr(core, '_last_tsb_gates', None) is not None:
        _tsb_lambda = float(os.environ.get('CERES_TSB_GATE_LAMBDA', '0.01') or 0.01)
        loss = loss + _tsb_lambda * core._last_tsb_gates.mean()

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

    # GRADIENT-CONFLICT PROBE (see setup block). Runs BEFORE the real backward and uses
    # torch.autograd.grad, which returns gradients instead of accumulating into .grad —
    # so the optimizer sees exactly what it would have seen with the probe off. Both
    # calls retain the graph; loss.backward() below then consumes it as usual.
    # allow_unused because a family may not reach every shared parameter (missing ->
    # contributes zero to both the dot product and the norms, which is correct).
    if GC_PROBE_STEPS > 0 and getattr(core, '_gc_probe_now', False):
      core._gc_probe_now = False
      _gc_pl, _gc_vl = core._gc_policy_loss, core._gc_value_loss
      if torch.is_tensor(_gc_pl) and torch.is_tensor(_gc_vl) and _gc_pl.requires_grad and _gc_vl.requires_grad:
        _gp = torch.autograd.grad(_gc_pl, _gc_params, retain_graph=True, allow_unused=True)
        _gv = torch.autograd.grad(_gc_vl, _gc_params, retain_graph=True, allow_unused=True)
        _gc_msg = []
        for _gname, _lo, _hi in _gc_slices:
          _dot = _np = _nv = 0.0
          for _a, _b in zip(_gp[_lo:_hi], _gv[_lo:_hi]):
            if _a is None or _b is None:
              continue
            _af, _bf = _a.float(), _b.float()
            _dot += float((_af * _bf).sum())
            _np += float((_af * _af).sum())
            _nv += float((_bf * _bf).sum())
          _np, _nv = _np ** 0.5, _nv ** 0.5
          _cos = _dot / (_np * _nv) if _np > 0 and _nv > 0 else float('nan')
          _ratio = _nv / _np if _np > 0 else float('nan')
          _gc_msg.append(f'{_gname} cos {_cos:+.4f} |gv|/|gp| {_ratio:.4f}')
          if writer is not None:
            writer.add_scalar(f'gradconflict/cos_{_gname}', _cos, num_pos)
            writer.add_scalar(f'gradconflict/ratio_{_gname}', _ratio, num_pos)
        print(f'GRADCONFLICT: {num_pos} , ' + ' , '.join(_gc_msg), flush=True)
        del _gp, _gv
      core._gc_policy_loss = None
      core._gc_value_loss = None

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

      if GC_PROBE_STEPS > 0:
        _gc_opt_steps += 1   # counts optimizer steps, so the probe cadence is in the config's units

      # Per-head QK-clip: rescale Q/K rows of any head whose max logit exceeded
      # tau this step (weight-level; no-op once training is stable). Throttled
      # QKCLIP log line + TB scalar so clip activity is visible/greppable.
      if QK_CLIP_TAU > 0:
        if _qk_clip_sync is not None:
          # DDP: fold every rank's per-head maxima into one MAX before clipping, so
          # all replicas apply the identical rescale (see the setup block). Runs on
          # every rank — it is a collective.
          _qk_clip_sync()
        _clip_total = 0
        for _m in _qk_clip_mods:
          _clip_total += _m.apply_qk_clip(QK_CLIP_TAU)
        if _clip_total > 0:
          if num_pos - _qk_clip_last_report[0] > 1_000_000:
            _qk_clip_last_report[0] = num_pos
            print(f'QKCLIP: {num_pos} , {_clip_total} heads clipped (tau {QK_CLIP_TAU})', flush=True)
          if writer is not None:
            writer.add_scalar('qk_clip_heads', _clip_total, num_pos)

      # EMA shadow update (see setup block): lc0 capped running average.
      # Runs AFTER the QK-clip rescale so the shadow never integrates weight
      # states the live net was explicitly corrected away from.
      if _ema_sd is not None:
        _ema_opt_steps += 1
        if _ema_opt_steps % EMA_PERIOD == 0:
          with torch.no_grad():
            _msd = model_nocompile.state_dict()
            _a = _ema_n / (_ema_n + 1.0)
            _b = 1.0 / (_ema_n + 1.0)
            for _k, _ev in _ema_sd.items():
              _ev.mul_(_a).add_(_msd[_k], alpha=_b)
          _ema_n = min(_ema_n + 1, EMA_MAX_N)

    batch_accumulation_counter = batch_accumulation_counter + 1

    # Update GLOBAL positions processed. num_processing_now is this rank's share
    # (forward batch was split batch_size_forward // world_size), so multiply by
    # WORLD_SIZE to count positions across all ranks. WORLD_SIZE==1 single-GPU, so
    # this is unchanged there. Counting globally keeps MAX_POSITIONS, checkpoint
    # cadence and the LR-decay schedule (fraction_complete) on the same footing as
    # a single-GPU run rather than running world_size× too long.
    # num_pos counts GROSS positions, replayed rows included — deliberately.
    # Subtracting injected rows was tried (2026-08-19) and reverted the same
    # day: the subtraction is rank-LOCAL (each rank's buffer draws differ), so
    # under DDP the counters diverge, every rank derives a DIFFERENT LR from
    # its own num_pos, and ranks cross MAX_POSITIONS on different iterations —
    # an NCCL hang with no final checkpoint. It also does not buy a clean A/B:
    # matching on unique positions un-matches on optimizer steps, compute and
    # the EMA window. The unique-data displacement is intrinsic to replay and
    # is REPORTED instead (hard_replay_displaced_pos) rather than hidden in
    # the schedule counter.
    num_pos = num_pos + num_processing_now * WORLD_SIZE
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
      # Only rank 0 writes checkpoint/ONNX/TS — all ranks hold identical (DDP-synced)
      # weights, so rank 0's copy is authoritative; letting every rank write the same
      # paths would corrupt the files. last_save_model_pos is still advanced on every
      # rank so the cadence stays in lockstep.
      if IS_MASTER:
        save_checkpoint(NAME, OUTPUTS_DIR, config, model_nocompile, state, str(num_pos))
        save_model(NAME, OUTPUTS_DIR, config, model_nocompile, state, str(num_pos), True)
        # EMA dual export: temporarily swap the shadow weights into the live
        # module (state_dict tensors are shared with the compiled model, so
        # in-place copy + restore keeps training untouched), export with an
        # 'ema' label suffix, restore the raw weights.
        if _ema_sd is not None and _ema_n > 0:
          with torch.no_grad():
            _msd = model_nocompile.state_dict()
            _raw_backup = {_k: _msd[_k].detach().clone() for _k in _ema_sd}
            for _k, _ev in _ema_sd.items():
              _msd[_k].copy_(_ev)
          try:
            save_model(NAME, OUTPUTS_DIR, config, model_nocompile, state, str(num_pos) + 'ema', True)
          finally:
            with torch.no_grad():
              for _k, _bv in _raw_backup.items():
                _msd[_k].copy_(_bv)
      last_save_model_pos = num_pos

    current_time = datetime.datetime.now()

    global time_start
    global time_last_status_update
    global time_last_save_transient

    time_since_start = (current_time - time_start).seconds
    time_since_status_update = (current_time - time_last_status_update).seconds
    time_since_save_transient = (current_time - time_last_save_transient).seconds

    STATUS_UPDATE_INTERVAL = 10 # log output to console every 10 seconds
    # Only rank 0 prints the (parsed) TRAIN status lines — multiple ranks emitting
    # them would feed duplicate/interleaved lines to the C# log parser.
    should_show_status = ((time_since_status_update > STATUS_UPDATE_INTERVAL) or (num_pos >= MAX_POSITIONS)) and IS_MASTER

    # save output artifacts (except checkpoint file) every 120 (or 30 if LoRA) minutes (with label "last")
    SAVE_LAST_INTERVAL = 120 * 60 if config.Opt_LoRARankDivisor == 0 else 30 * 60
    should_save_transient = (time_since_save_transient > SAVE_LAST_INTERVAL) and IS_MASTER
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
      # Placement value head (aux) loss on its own line — the TRAIN line format is
      # parsed by C# (CeresTrainProgressLoggingLine.cs) and must not change shape.
      # Interval average via LossCalculator, same semantics as the TRAIN-line losses.
      if getattr(core, 'placement_value_weight', 0) > 0 and loss_calc.PENDING_COUNT > 0:
        print("PLACEV:", num_pos, ",", loss_calc.LAST_PLACEMENT_VALUE_LOSS, flush=True)
      if getattr(core, 'survival_target_weight', 0) > 0 and loss_calc.PENDING_COUNT > 0:
        print("SURV:", num_pos, ",", loss_calc.LAST_SURVIVAL_LOSS, ",", loss_calc.LAST_SURVIVAL_ACC, flush=True)
      if getattr(core, 'stvalue_weight', 0) > 0 and loss_calc.PENDING_COUNT > 0:
        print("STVAL:", num_pos, ",", loss_calc.LAST_STVALUE_LOSS, flush=True)
      loss_calc.reset_counters()
      time_last_status_update = datetime.datetime.now()

  # final save and convert to Torchscript (rank 0 only; weights are DDP-synced).
  # Skipped when the interval save already wrote this exact position count
  # (checkpoint frequency == run length) — the redundant re-save is wasted work
  # and, pre-2026-07-22, overwrote the .lora bin with an empty one post-merge.
  if IS_MASTER:
    if config.Opt_CheckpointFrequencyNumPositions > 0 and last_save_model_pos == num_pos:
      print(f"INFO: final save skipped (checkpoint already written at {num_pos})")
    else:
      save_checkpoint(NAME, OUTPUTS_DIR, config, model_nocompile, state, str(num_pos))
      save_model(NAME, OUTPUTS_DIR, config, model_nocompile, state, str(num_pos), True)
      # Final EMA export — without this, an off-cadence run end (or a run with
      # interval checkpointing disabled) would silently discard the freshest —
      # i.e. most valuable — averaged weights.
      if _ema_sd is not None and _ema_n > 0:
        with torch.no_grad():
          _msd = model_nocompile.state_dict()
          _raw_backup = {_k: _msd[_k].detach().clone() for _k in _ema_sd}
          for _k, _ev in _ema_sd.items():
            _msd[_k].copy_(_ev)
        try:
          save_model(NAME, OUTPUTS_DIR, config, model_nocompile, state, str(num_pos) + 'ema', True)
        finally:
          with torch.no_grad():
            for _k, _bv in _raw_backup.items():
              _msd[_k].copy_(_bv)

  writer.flush()
  writer.close()

  # Hold all ranks until rank 0 has finished writing the final artifacts, then
  # tear down the process group cleanly (avoids NCCL teardown warnings / a rank
  # exiting mid-collective).
  if IS_DISTRIBUTED:
    dist.barrier()
    dist.destroy_process_group()

  if IS_MASTER:
    print("INFO: EXIT_STATUS", "SUCCESS")

Train()

