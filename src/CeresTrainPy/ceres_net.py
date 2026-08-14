# License Notice

"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.

Ceres is free software distributed under the terms of the GNU General Public License v3.0.
You should have received a copy of the GNU General Public License along with CeresTrain.
If not, see <http://www.gnu.org/licenses/>.
"""

# End of License Notice

# NOTE: this module is derived from: https://github.com/Rocketknight1/minimal_lczero.

import os
import math
from multiprocessing import Value
from typing import Tuple, NamedTuple

import torch
import torch.nn as nn
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from activation_functions import to_activation
from losses import LossCalculator
from encoder_layer import EncoderLayer
from config import Configuration
from mlp2_layer import MLP2Layer
from rms_norm import RMSNorm, make_norm
from lora import LoRALinear
from utils import DWA
from tactical_adapter import TacticalAdapter, PositionGate, gtab_enabled
from chess_geometry import PieceRelationBias, RayAttentionBias, VisibilityChannels

from config import NUM_TOKENS_INPUT, NUM_TOKENS_NET, NUM_INPUT_BYTES_PER_SQUARE, TOTAL_INPUT_FEATURES_PER_SQUARE



class Head(nn.Module):
  def __init__(self, Activation, IN_SIZE, FC_SIZE, OUT_SIZE, lora_rank_divisor):
    super(Head, self).__init__()

    self.fc = nn.Linear(IN_SIZE, FC_SIZE)
    if lora_rank_divisor > 0:
      self.fc = LoRALinear(self.fc, lora_rank_divisor, True)

    self.fcActivation = Activation

    self.fcFinal = nn.Linear(FC_SIZE, OUT_SIZE)
    if lora_rank_divisor > 0:
      self.fcFinal = LoRALinear(self.fcFinal, lora_rank_divisor, True)

  def forward(self, flow, inject=None):
    flow = self.fc(flow)
    if inject is not None:
      # Additive pre-activation injection (ValueHeadChannelsMode='inject'):
      # equivalent to widening fc's input with extra columns, without changing
      # fc's shape. Callers pass None everywhere else.
      flow = flow + inject
    flow = self.fcActivation(flow)
    flow = self.fcFinal(flow)
    return flow


class CeresNet(nn.Module):
  def __init__(
    self,
    writer : SummaryWriter,
    config : Configuration,
    policy_loss_weight,
    value_loss_weight,
    moves_left_loss_weight,
    unc_loss_weight,
    value2_loss_weight,
    q_deviation_loss_weight,
    value_diff_loss_weight,
    value2_diff_loss_weight,
    action_loss_weight,
    uncertainty_policy_weight,
    action_uncertainty_loss_weight,
    q_ratio):
    """
    CeresNet is a transformer-architecture chess network built directly on PyTorch.
    `writer` is a torch.utils.tensorboard.SummaryWriter (or None) used for metric logging.
    """
    super().__init__()

    self.writer = writer
    self.config = config
     
    self.DROPOUT_RATE = config.Exec_DropoutRate
    self.EMBEDDING_DIM = config.NetDef_ModelDim
    self.NUM_LAYERS = config.NetDef_NumLayers
    # Looped transformer: NUM_DISTINCT_LAYERS modules constructed; the body
    # applies the full sequence LOOP_COUNT times so total depth = NUM_LAYERS.
    # Default LoopCount=1 → distinct == NUM_LAYERS (current behaviour).
    self.LOOP_COUNT = config.NetDef_LoopCount
    self.NUM_DISTINCT_LAYERS = self.NUM_LAYERS // self.LOOP_COUNT


    self.TRANSFORMER_OUT_DIM = self.EMBEDDING_DIM * NUM_TOKENS_NET

    self.NUM_HEADS = config.NetDef_NumHeads
    self.FFN_MULT = config.NetDef_FFNMultiplier
    self.DEEPNORM = config.NetDef_DeepNorm
    self.denseformer = config.NetDef_DenseFormer
    self.prior_state_dim = config.NetDef_PriorStateDim
    self.moves_left_loss_weight = moves_left_loss_weight
    self.q_deviation_loss_weight = q_deviation_loss_weight
    self.value2_loss_weight = value2_loss_weight
    self.uncertainty_policy_weight = uncertainty_policy_weight
    self.action_uncertainty_loss_weight = action_uncertainty_loss_weight

    
    self.Activation  = to_activation(config.NetDef_HeadsActivationType)
    self.test = config.Exec_TestFlag

    # When CERES_AUX_FEATURES_PER_SQUARE > 0, TOTAL_INPUT_FEATURES_PER_SQUARE
    # = NUM_INPUT_BYTES_PER_SQUARE + NUM_AUX_FEATURES_PER_SQUARE (e.g. 140 vs 137).
    self.embedding_layer = nn.Linear(TOTAL_INPUT_FEATURES_PER_SQUARE + self.prior_state_dim, self.EMBEDDING_DIM)
    self.embedding_norm = make_norm(config.NetDef_NormType, self.EMBEDDING_DIM, eps=1E-6)

    HEAD_MULT = config.NetDef_HeadWidthMultiplier

    HEAD_PREMAP_DIVISOR = 64
    self.HEAD_PREMAP_PER_SQUARE = (HEAD_MULT * self.EMBEDDING_DIM) // HEAD_PREMAP_DIVISOR
    self.headPremap = nn.Linear(self.EMBEDDING_DIM, self.HEAD_PREMAP_PER_SQUARE)

    # Shared head front-end width (config HeadSharedLinearDiv; 4 = the historical
    # hardcoded value, so every pre-existing net is unchanged). See config.py for why
    # this is adjustable: the private-value-head hypothesis, once the gradient probe
    # removed its crowding-out premise, is a claim about the width of THIS vector.
    HEAD_SHARED_LINEAR_DIV = int(getattr(config, 'NetDef_HeadSharedLinearDiv', 4) or 4)
    assert self.HEAD_PREMAP_PER_SQUARE % HEAD_SHARED_LINEAR_DIV == 0, \
        (f'HeadSharedLinearDiv={HEAD_SHARED_LINEAR_DIV} must divide HEAD_PREMAP_PER_SQUARE='
         f'{self.HEAD_PREMAP_PER_SQUARE} (= HeadWidthMultiplier*ModelDim/64) exactly — '
         f'otherwise the head input width silently truncates')
    self.HEAD_IN_SIZE = 64 * (self.HEAD_PREMAP_PER_SQUARE // HEAD_SHARED_LINEAR_DIV)
    if HEAD_SHARED_LINEAR_DIV != 4:
      print(f'[ceres_net] SHARED HEAD FRONT-END widened: div={HEAD_SHARED_LINEAR_DIV} '
            f'-> HEAD_IN_SIZE {self.HEAD_IN_SIZE} (default div 4 would give '
            f'{64 * (self.HEAD_PREMAP_PER_SQUARE // 4)})')
    self.headSharedLinear = nn.Linear(64 * self.HEAD_PREMAP_PER_SQUARE, self.HEAD_IN_SIZE)

    # PRIVATE VALUE FRONT-END (config ValueHeadChannels; see config.py): the
    # value family gets its own per-square projection D -> C, flattened to
    # 64*C, bypassing the policy-shared bottleneck entirely (lc0-style).
    self.value_head_channels = int(getattr(config, 'Opt_ValueHeadChannels', 0) or 0)
    self.value_head_channels_mode = str(getattr(config, 'Opt_ValueHeadChannelsMode', 'replace') or 'replace').lower()
    # 'replace' rewires the value family's INPUT (from-scratch only — head widths
    # change, so pre-flag checkpoints can't load). 'inject' keeps every head width
    # intact and adds a zero-init private projection into the value head's hidden
    # pre-activation instead: same function, but loadable from an existing net and
    # bit-identical to it at step 0. See config.py for the full rationale.
    self.value_priv_replace = self.value_head_channels > 0 and self.value_head_channels_mode == 'replace'
    self.value_priv_inject_mode = self.value_head_channels > 0 and self.value_head_channels_mode == 'inject'
    if self.value_head_channels > 0:
      # LoRA head-front adapters wrap only headPremap/headSharedLinear, which the
      # value family BYPASSES in 'replace' mode — a head-front fine-tune would
      # silently become policy-only. Fail loudly there. NOT a conflict in 'inject'
      # mode: the value heads keep reading the shared vector (fS_value IS
      # fS_others), so the adapters reach them exactly as they reach every other
      # head, and value_premap trains full-rank alongside. Since 'inject' exists
      # precisely to fine-tune an existing net, blocking it here would forbid the
      # combination the mode was built for.
      assert not self.value_priv_replace or int(os.environ.get('CERES_LORA_HEADFRONT_RANK_DIV', '0') or 0) == 0,         "CERES_LORA_HEADFRONT_RANK_DIV is incompatible with ValueHeadChannelsMode='replace' (value_premap has no adapter)"
      self.value_premap = nn.Linear(self.EMBEDDING_DIM, self.value_head_channels)
      self.VALUE_PRIV_SIZE = 64 * self.value_head_channels
    if self.value_priv_replace:
      self.VALUE_IN_SIZE = self.VALUE_PRIV_SIZE
      print(f'[ceres_net] PRIVATE VALUE FRONT-END (replace) enabled: {self.value_head_channels} ch/square '
            f'-> {self.VALUE_IN_SIZE}-dim private value input (shared front-end bypassed for value family)')
    else:
      self.VALUE_IN_SIZE = self.HEAD_IN_SIZE
    self.unc_self_error = bool(int(getattr(config, 'Opt_UncSelfError', 0) or 0))
    if self.unc_self_error:
      print("[ceres_net] UNC SELF-ERROR mode: unc trains toward the student's own "
            "(q_pred - q_target)^2 (detached); sigma consumers use sqrt(unc)")

    # Optional LoRA wrap of the head front-end (the per-position vector that
    # feeds every head). Gated by CERES_LORA_HEADFRONT_RANK_DIV — sits
    # downstream of body, upstream of all heads. Lets fine-tunes refine what
    # heads read without touching the body's shared feature manifold.
    _hf_rd = int(os.environ.get('CERES_LORA_HEADFRONT_RANK_DIV', '0') or 0)
    if _hf_rd > 0:
      self.headPremap = LoRALinear(self.headPremap, _hf_rd, True)
      self.headSharedLinear = LoRALinear(self.headSharedLinear, _hf_rd, True)

    # When restrict_pv flag is set, only policy_head and value_head get LoRA
    # adapters; all other heads use rank_div=0 (standard Linear, frozen base).
    # Aligns with "minimal-intervention" recipe: most labels match orig output,
    # so only policy + value heads need adaptable capacity.
    _lora_rd = config.Opt_LoRARankDivisor
    _restrict_pv = bool(getattr(config, 'Opt_LoRARestrictPolicyValueOnly', False))
    _restrict_v  = bool(getattr(config, 'Opt_LoRARestrictValueOnly', False))
    # When _restrict_v is set, ONLY the value head gets LoRA adapters; policy
    # and all other heads use rank_div=0. Combined with
    # CERES_LORA_TRANSFORMER_RANK_DIV=0 and LossPolicyMultiplier=0, this isolates
    # value-head fine-tuning: only value_head.* layers receive any gradient.
    _pol_rd   = 0 if _restrict_v else _lora_rd
    _val_rd   = _lora_rd
    _other_rd = 0 if (_restrict_pv or _restrict_v) else _lora_rd

    self.policy_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858, _pol_rd)

    if self.prior_state_dim > 0:
      self.state_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 64*self.prior_state_dim, _other_rd)

    if action_loss_weight > 0:
      self.action_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858 * 3, _other_rd)

    if action_uncertainty_loss_weight > 0:
      self.action_uncertainty_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858, _other_rd)

    self.value_head = Head(self.Activation, self.VALUE_IN_SIZE, 64 * HEAD_MULT, 3, _val_rd)
    # unc follows the value family: on the private path it reads the private
    # value features (audit finding #4 — error-family heads should enrich the
    # value representation, not the shared bottleneck).
    self.unc_head = Head(self.Activation, self.VALUE_IN_SIZE, 32 * HEAD_MULT, 1, _other_rd)

    if self.value2_loss_weight > 0:
      self.value2_head = Head(self.Activation, 2 + self.VALUE_IN_SIZE, 64 * HEAD_MULT, 3, _other_rd)

    if self.value_priv_inject_mode:
      # Zero-init private injectors (ValueHeadChannelsMode='inject'). Bias-free and
      # zero-weight => exactly no contribution at step 0, so the net reproduces the
      # base checkpoint bit-for-bit and the fine-tune starts where it should. Not
      # LoRA-wrapped: full rank, since they cannot damage anything from zero.
      self.value_priv_inject = nn.Linear(self.VALUE_PRIV_SIZE, 64 * HEAD_MULT, bias=False)
      nn.init.zeros_(self.value_priv_inject.weight)
      if self.value2_loss_weight > 0:
        self.value2_priv_inject = nn.Linear(self.VALUE_PRIV_SIZE, 64 * HEAD_MULT, bias=False)
        nn.init.zeros_(self.value2_priv_inject.weight)
      _n_inj = self.VALUE_PRIV_SIZE * 64 * HEAD_MULT * (2 if self.value2_loss_weight > 0 else 1)
      _n_inj += self.EMBEDDING_DIM * self.value_head_channels + self.value_head_channels
      print(f'[ceres_net] PRIVATE VALUE FRONT-END (inject) enabled: {self.value_head_channels} ch/square '
            f'-> {self.VALUE_PRIV_SIZE}-dim private vector added into value1'
            f'{"/value2" if self.value2_loss_weight > 0 else ""} hidden pre-activation '
            f'({_n_inj:,} new params, zero-init => base recovered exactly at step 0)')

    if self.uncertainty_policy_weight > 0:
      self.unc_policy = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1, _other_rd)

    if moves_left_loss_weight > 0:
      self.mlh_head = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1, _other_rd)

    if q_deviation_loss_weight > 0:
      self.qdev_upper = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1, _other_rd)
      self.qdev_lower = Head(self.Activation, self.HEAD_IN_SIZE, 32 * HEAD_MULT, 1, _other_rd)

    # Depth-attending value head (AttnRes-inspired, Kimi K3 / arXiv 2603.15031).
    # Env-gated: CERES_VALUE_DEPTH_ATTENTION=1 (default 0 = fully off, exactly current
    # behavior). Motivation: the value head reads only the FINAL trunk layer, whose
    # features are policy-shaped; value estimation may want earlier-depth features
    # (material/structure) that later layers abstract away. Mechanism: mean-pool each
    # depth state over squares -> [B, L+1, D] (post-embedding + each layer), RMSNorm,
    # then a learned pseudo-query attends over depth; the mixed vector is projected
    # into head space and ADDED to fS_value only (policy path untouched).
    # Zero-init query -> uniform depth weights at step 0; zero-init projection ->
    # exact no-op at step 0 (GTAB pattern). NOT training-only: it feeds value_out,
    # so it is part of the export graph (simple matmul/softmax ops, export-safe).
    # Modes: 0 = off, 1 = pooled (board mean-pooled per depth, context added to
    # fS_value after the head front-end), 2 = PER-SQUARE (each square attends over
    # its OWN depth trajectory; context added to the value-side flow BEFORE
    # headPremap, GTAB-style). Mode 2 preserves square-localized information
    # (e.g. a specific pawn's early-layer identity) that mode 1's pooling loses.
    # Mode 3 = BOTH pathways combined: they inject on opposite sides of the head
    # front-end bottleneck (premap compresses 256->16 per square), so they carry
    # non-nested information — per-square = spatially resolved but bottlenecked,
    # pooled = global but full-bandwidth post-bottleneck. All modes now collect
    # RAW references and compute at the tail (deferred pooling): identical math
    # (mean commutes) but no fusion-breaking taps in the trunk — measured worth
    # ~18% NPS at MCTS batch sizes.
    # Mode 4 = TRAINING-ONLY AUXILIARY: the served value head reads the plain
    # final-layer path (exact novda export graph — zero serving cost), while the
    # full mode-3 machinery + an auxiliary value head run only under
    # self.training with their own loss (CERES_VDA_AUX_WEIGHT, default 1.0).
    # Preserves the deep-supervision benefit (gradients reach every trunk layer)
    # without any inference footprint. Warm-startable from a mode-3 checkpoint:
    # vda_* transfer unchanged; vda_aux_head inherits value_head's weights
    # (train.py resume handles the copy).
    self.vda_mode = int(os.environ.get('CERES_VALUE_DEPTH_ATTENTION', '0') or 0)
    assert self.vda_mode in (0, 1, 2, 3, 4), f'CERES_VALUE_DEPTH_ATTENTION must be 0-4, got {self.vda_mode}'
    self.use_value_depth_attention = self.vda_mode > 0
    self.vda_aux_weight = float(os.environ.get('CERES_VDA_AUX_WEIGHT', '1.0') or 1.0)
    if self.use_value_depth_attention:
      _MODE_NAMES = {1: 'pooled', 2: 'per-square', 3: 'combined (per-square + pooled)',
                     4: 'training-only auxiliary (combined machinery, novda serving graph)'}
      self.vda_norm = make_norm(config.NetDef_NormType, self.EMBEDDING_DIM, eps=1E-6)
      self.vda_query = nn.Parameter(torch.zeros(self.EMBEDDING_DIM))
      _vda_out = self.HEAD_IN_SIZE if self.vda_mode == 1 else self.EMBEDDING_DIM
      self.vda_proj = nn.Linear(self.EMBEDDING_DIM, _vda_out)
      nn.init.zeros_(self.vda_proj.weight)
      nn.init.zeros_(self.vda_proj.bias)
      _n_params = self.EMBEDDING_DIM * _vda_out + 2 * self.EMBEDDING_DIM + _vda_out
      if self.vda_mode == 4:
        # Auxiliary value head (training-only): same structure as value_head, fed
        # by the mode-3-augmented front-end. Inherits value_head weights on
        # mode-3 warm-start (they were trained on exactly this augmented input).
        self.vda_aux_head = Head(self.Activation, self.HEAD_IN_SIZE, 64 * HEAD_MULT, 3, 0)
      if self.vda_mode in (3, 4):
        # Global/pooled branch: own query + own post-front-end projection.
        # 'vda_query' substring in the name keeps it in train.py's no_decay rule.
        self.vda_query_g = nn.Parameter(torch.zeros(self.EMBEDDING_DIM))
        self.vda_proj_g = nn.Linear(self.EMBEDDING_DIM, self.HEAD_IN_SIZE)
        nn.init.zeros_(self.vda_proj_g.weight)
        nn.init.zeros_(self.vda_proj_g.bias)
        _n_params += self.EMBEDDING_DIM * self.HEAD_IN_SIZE + self.EMBEDDING_DIM + self.HEAD_IN_SIZE
      print(f'[ceres_net] VALUE DEPTH ATTENTION enabled (mode {self.vda_mode} = '
            f'{_MODE_NAMES[self.vda_mode]}): pseudo-query over '
            f'{self.NUM_LAYERS + 1} depth states [{self.EMBEDDING_DIM}], deferred pooling '
            f'({_n_params} params, zero-init no-op)')

    # Policy depth attention (PDA, tier 1 = shared states). CERES_POLICY_DEPTH_ATTENTION=1
    # gives the POLICY path a per-square depth read mirroring vda mode 2's injection
    # (context added to the policy-side flow BEFORE the head front-end). Sharing:
    # reuses vda_norm and — when vda modes 2/3 already computed them — the normed
    # per-square depth states, so the state reads (the dominant serving cost of
    # depth attention) are paid once for both heads. Own pseudo-query + zero-init
    # projection: the policy path is bit-identical at init. Parameter names keep
    # the 'vda_' prefix deliberately: train.py's partition rule ('vda_query' ->
    # no_decay) and the aux-head resume prefixes ('vda_') then apply unchanged.
    self.pda_mode = int(os.environ.get('CERES_POLICY_DEPTH_ATTENTION', '0') or 0)
    assert self.pda_mode in (0, 1), f'CERES_POLICY_DEPTH_ATTENTION must be 0/1, got {self.pda_mode}'
    if self.pda_mode:
      if not self.use_value_depth_attention:
        self.vda_norm = make_norm(config.NetDef_NormType, self.EMBEDDING_DIM, eps=1E-6)
      self.vda_query_p = nn.Parameter(torch.zeros(self.EMBEDDING_DIM))
      self.vda_proj_p = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM)
      nn.init.zeros_(self.vda_proj_p.weight)
      nn.init.zeros_(self.vda_proj_p.bias)
      print(f'[ceres_net] POLICY DEPTH ATTENTION enabled (tier-1 shared-states'
            f'{" with vda" if self.use_value_depth_attention else ""}): per-square '
            f'pseudo-query over {self.NUM_LAYERS + 1} depth states [{self.EMBEDDING_DIM}], '
            f'zero-init no-op')

    # Depth probes (CERES_DEPTH_PROBES=1): per-depth deep supervision + in-process
    # control heads (adapted from the T1 vda+pda package, 2026-08 — its TB evidence:
    # ~2x early sample efficiency measured via a last-layer-only control head).
    # TRAINING-ONLY (self.training gated, stash pattern) -> export graph unchanged,
    # zero serving cost, no Ceres-side changes.
    #   depth_probe_policy: ONE weight-shared Linear D->1858 read from each pooled
    #     depth state (weight sharing keeps params ~= one small head and makes the
    #     per-depth loss curves directly comparable); depth_probe_value: shared
    #     Linear D->3 (WDL). Losses = mean-over-depths CE at weights
    #     CERES_DEPTH_PROBE_POLICY_WEIGHT / _VALUE_WEIGHT (default 0.05 each).
    #   depth_ctl_policy / depth_ctl_value: identical-shape heads reading ONLY the
    #     final state through a DETACHED input — pure measurement apparatus for
    #     paired in-process reads (they train themselves, never shape the trunk).
    self.depth_probes_enabled = int(os.environ.get('CERES_DEPTH_PROBES', '0') or 0) > 0
    if self.depth_probes_enabled:
      self.depth_probe_policy_weight = float(os.environ.get('CERES_DEPTH_PROBE_POLICY_WEIGHT', '0.05') or 0.05)
      self.depth_probe_value_weight = float(os.environ.get('CERES_DEPTH_PROBE_VALUE_WEIGHT', '0.05') or 0.05)
      self.depth_probe_norm = make_norm(config.NetDef_NormType, self.EMBEDDING_DIM, eps=1E-6)
      self.depth_probe_policy = nn.Linear(self.EMBEDDING_DIM, 1858)
      self.depth_probe_value = nn.Linear(self.EMBEDDING_DIM, 3)
      self.depth_ctl_policy = nn.Linear(self.EMBEDDING_DIM, 1858)
      self.depth_ctl_value = nn.Linear(self.EMBEDDING_DIM, 3)
      _n = (self.EMBEDDING_DIM + 1) * (1858 + 3) * 2 + 2 * self.EMBEDDING_DIM
      print(f'[ceres_net] DEPTH PROBES enabled: shared policy/value probes over '
            f'{self.NUM_LAYERS + 1} pooled depth states (w_p={self.depth_probe_policy_weight}, '
            f'w_v={self.depth_probe_value_weight}) + detached ctl heads ({_n} params, training-only)')

    # Ray-Context factored policy term (campaign idea 17 + reviewer fixes;
    # CERES_RAY_CONTEXT: 1 = from-to bilinear + per-move bias only (de-confound
    # arm), 2 = + ray terms). Added ZERO-INIT on top of the dense MLP policy
    # head (rc_WT/u/v/w/btype all start 0 => exact no-op at init, no in-dist
    # memorization regression risk). r_m pools LIVE post-norm square states over
    # the move's between+behind line (R: pin/skewer/x-ray); d_m pools the rays
    # VACATED by the move (R2: the reviewer's corrected discovery operator —
    # non-collinear rays through from, since a move never vacates its own line).
    # Constant-index Gathers + matmuls only: TRT/INT8-clean, no attention change.
    self.ray_context_mode = int(os.environ.get('CERES_RAY_CONTEXT', '0') or 0)
    if self.ray_context_mode > 0:
      self.rc_dh = int(os.environ.get('CERES_RAY_CONTEXT_DH', '64') or 64)
      # Memory-lean serving formulation (CERES_RAY_CONTEXT_CHUNKED=N, export-time
      # only): mathematically identical rewrite that avoids the naive path's
      # [B,1858,dh] intermediates (~243MB each at B=1024 fp16, several live at
      # once). Bilinear becomes F@T^T ([B,64,64]) + flat pair-gather; ray terms
      # become tiny [B,64,64] H-matrices with the single remaining [1858]-axis
      # matmul processed in N chunks (peak [B,1858/N,dh]). Set only in the
      # ExportOnly process — the trainer keeps the fused path.
      self.rc_chunks = int(os.environ.get('CERES_RAY_CONTEXT_CHUNKED', '0') or 0)
      from lc0_moves_1858 import FROM_1858, TO_1858
      self.register_buffer('rc_from', torch.tensor(FROM_1858, dtype=torch.long), persistent=False)
      self.register_buffer('rc_to', torch.tensor(TO_1858, dtype=torch.long), persistent=False)
      if self.rc_chunks > 0:
        _ft_flat = [f * 64 + t for f, t in zip(FROM_1858, TO_1858)]
        self.register_buffer('rc_ft_flat', torch.tensor(_ft_flat, dtype=torch.long), persistent=False)
      self.rc_WF = nn.Linear(self.EMBEDDING_DIM, self.rc_dh, bias=False)
      self.rc_WT = nn.Linear(self.EMBEDDING_DIM, self.rc_dh, bias=False)
      nn.init.zeros_(self.rc_WT.weight)
      self.rc_btype = nn.Parameter(torch.zeros(1858))
      if self.ray_context_mode >= 2:
        from chess_geometry import build_ray_context_tables
        _R, _R2 = build_ray_context_tables(FROM_1858, TO_1858)
        self.register_buffer('rc_R', _R, persistent=False)
        self.register_buffer('rc_R2', _R2, persistent=False)
        self.rc_WR = nn.Linear(self.EMBEDDING_DIM, self.rc_dh, bias=False)
        self.rc_WD = nn.Linear(self.EMBEDDING_DIM, self.rc_dh, bias=False)
        self.rc_u = nn.Parameter(torch.zeros(self.rc_dh))
        self.rc_v = nn.Parameter(torch.zeros(self.rc_dh))
        self.rc_w = nn.Parameter(torch.zeros(self.rc_dh))
      print(f'[ceres_net] RAY-CONTEXT policy term enabled: mode {self.ray_context_mode} '
            f'({"bilinear-only" if self.ray_context_mode == 1 else "bilinear+ray+discovery"}), '
            f'dh={self.rc_dh}, zero-init additive')

    # Soft-policy auxiliary head (KataGo "auxiliary soft policy target", via
    # lc0/Monroe: +10% convergence; CERES_SOFT_POLICY_WEIGHT > 0 enables).
    # An extra policy-shaped head trained on a TEMPERATURE-FLATTENED version of
    # the same policy target (p^(1/T) over legal moves, renormalized; T =
    # CERES_SOFT_POLICY_TEMP, default 4): dense gradient on the move-ordering
    # tail + anti-overconfidence regularization, inherited by the main policy
    # head through the shared trunk. Training-only; never in the export graph.
    # Config-first, env-fallback (same contract as mirror/focal — config.py).
    self.soft_policy_weight = (float(getattr(config, 'Opt_SoftPolicyWeight', 0) or 0)
                               or float(os.environ.get('CERES_SOFT_POLICY_WEIGHT', '0') or 0))
    if self.soft_policy_weight > 0:
      self.soft_policy_temp = (float(getattr(config, 'Opt_SoftPolicyTemp', 0) or 0)
                               or float(os.environ.get('CERES_SOFT_POLICY_TEMP', '4') or 4))
      self.sp_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858, 0)
      print(f'[ceres_net] SOFT-POLICY aux head enabled: T={self.soft_policy_temp}, '
            f'w={self.soft_policy_weight} (training-only)')

    # Value-discrimination mechanisms for sharp/decisive corpora (value head
    # collapses on puzzle-only data: near-uniform "decisive" labels destroy
    # calibration, gates degenerate to ~1400-1700). Both are training-only.
    #
    # CERES_VALUE_RANK_WEIGHT: in-batch pairwise RANKING loss on value1 —
    # calibration-free discrimination: positions whose targets differ must be
    # ORDERED correctly by E[V], regardless of absolute probabilities. Solution
    # lines alternate STM so puzzle batches contain abundant W/L contrast pairs.
    #
    # CERES_VALUE_CONTRAST_WEIGHT: aux per-move WDL head ("policy teaches
    # value"): the solution move inherits the record's (search-)WDL, every
    # other LEGAL move is labeled pure LOSS-for-STM (correct across win,
    # draw-defense, and defender-side records of solution-line expansion —
    # see the label-rule comment in compute_loss). Gives the trunk per-move
    # outcome contrast the plain value CE never provides; value1/value2 read
    # the shaped trunk. Head is training-only (never in the export graph).
    # NB: measured NEUTRAL on the puzzle value gate at 10M (but that gate
    # evaluates depth-1 child positions the corpus never trains on).
    self.value_rank_weight = float(os.environ.get('CERES_VALUE_RANK_WEIGHT', '0') or 0)
    if self.value_rank_weight > 0:
      print(f'[ceres_net] VALUE RANK loss enabled: in-batch pairwise ordering, w={self.value_rank_weight}')
    self.value_contrast_weight = float(os.environ.get('CERES_VALUE_CONTRAST_WEIGHT', '0') or 0)
    if self.value_contrast_weight > 0:
      self.vc_head = Head(self.Activation, self.HEAD_IN_SIZE, 64 * HEAD_MULT, 1858 * 3, 0)
      print(f'[ceres_net] VALUE CONTRAST aux head enabled: per-move WDL from puzzle '
            f'policy targets (training-only), w={self.value_contrast_weight}')

    # HL-Gauss categorical value head (config: HLGaussWeight/Buckets/SigmaScale;
    # 0 = off). Softmax-CE against a Gaussian histogram target over the scalar
    # q = w - l in [-1,1] (Farebrother et al. 2024, "Stop Regressing"): forces
    # FINE-GRAINED value resolution that the 3-class WDL CE never demands
    # (+0.3 vs +0.5 must land in different buckets). Reads fS_value like the
    # main value head; training-only stash head, never in the export graph.
    # sigma = SigmaScale * bucket_width smooths the target over neighboring
    # buckets (the "HL-Gauss" part — lc0's one-hot bucket variant is the
    # degenerate sigma->0 case). No new data: target derives from wdl_q blend.
    # Optimistic-policy aux head (config: OptimisticPolicyWeight/Strength/
    # Alpha — see config.py). Separate training-only policy head (the vanilla
    # policy head keeps its full CE): per-sample weight sigmoid((z-S)*A),
    # z = (target_q - q_pred)/(sigma + 1e-5), sigma from the unc head
    # (assumed |error|-scale; a scale mismatch only shifts effective S).
    # Both q_pred and sigma are DETACHED (lc0 propagate_value_gradients=false).
    self.opt_policy_weight = float(getattr(config, 'Opt_OptimisticPolicyWeight', 0) or 0)
    self.opt_serve_blend = float(getattr(config, 'Opt_OptimisticPolicyServeBlend', 0) or 0)
    assert self.opt_serve_blend == 0 or self.opt_policy_weight > 0, \
      'OptimisticPolicyServeBlend requires OptimisticPolicyWeight > 0 (the head must exist)'
    # Soft-policy serve blend: same mechanism for the KataGo soft-policy head.
    # The two compose as a three-way logit blend:
    #   policy = (1 - l_opt - l_soft)*vanilla + l_opt*optimistic + l_soft*soft
    self.soft_serve_blend = float(getattr(config, 'Opt_SoftPolicyServeBlend', 0) or 0)
    assert self.soft_serve_blend == 0 or self.soft_policy_weight > 0, \
      'SoftPolicyServeBlend requires SoftPolicyWeight/CERES_SOFT_POLICY_WEIGHT > 0 (the head must exist)'
    assert 0.0 <= self.opt_serve_blend and 0.0 <= self.soft_serve_blend \
        and self.opt_serve_blend + self.soft_serve_blend <= 1.0, \
      'OptimisticPolicyServeBlend + SoftPolicyServeBlend must lie in [0, 1]'
    if self.opt_policy_weight > 0:
      self.opt_strength = float(getattr(config, 'Opt_OptimisticPolicyStrength', 2.0))
      self.opt_alpha = float(getattr(config, 'Opt_OptimisticPolicyAlpha', 3.0))
      self.opt_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858, 0)
      _serve_txt = f', SERVE-BLEND lambda={self.opt_serve_blend} (in export graph!)' if self.opt_serve_blend > 0 else ' (training-only)'
      print(f'[ceres_net] OPTIMISTIC-POLICY aux head enabled: w={self.opt_policy_weight}, '
            f'strength={self.opt_strength}, alpha={self.opt_alpha}{_serve_txt}')

    self.hlg_weight = float(getattr(config, 'Opt_HLGaussWeight', 0) or 0)
    if self.hlg_weight > 0:
      self.hlg_buckets = int(getattr(config, 'Opt_HLGaussBuckets', 32) or 32)
      _hlg_sigma_scale = float(getattr(config, 'Opt_HLGaussSigmaScale', 0.75) or 0.75)
      self.hlg_sigma = _hlg_sigma_scale * (2.0 / self.hlg_buckets)
      self.hlg_head = Head(self.Activation, self.VALUE_IN_SIZE, 64 * HEAD_MULT, self.hlg_buckets, 0)
      self.register_buffer('hlg_edges', torch.linspace(-1.0, 1.0, self.hlg_buckets + 1), persistent=False)
      print(f'[ceres_net] HL-GAUSS categorical value head enabled: w={self.hlg_weight}, '
            f'buckets={self.hlg_buckets}, sigma={self.hlg_sigma:.4f} (training-only)')

    # Placement value head (AUXILIARY, training-only). Env-gated: CERES_PLACEMENT_VALUE_WEIGHT
    # (default 0 = fully off, exactly current behavior). Additive per-square WDL decomposition:
    # each square's trunk embedding contributes a 3-vector of WDL logits; the position value is
    # the SUM over the 64 contributions (+ learned bias). No cross-square mixing in the head, so
    # to fit the value target the trunk must localize onto each square's embedding what the piece
    # standing there contributes to the outcome — a learned, context-aware piece-square
    # decomposition. Trained against the same value_target as value1 (wdl_q blend).
    # The output is stashed on the module for the loss (GTAB gate pattern) and NEVER added to
    # the forward return tuple, so the ONNX/TorchScript export signature is unchanged.
    self.placement_value_weight = float(os.environ.get('CERES_PLACEMENT_VALUE_WEIGHT', '0') or 0)
    if self.placement_value_weight > 0:
      self.placement_value_head = nn.Linear(self.EMBEDDING_DIM, 3)
      self.placement_value_bias = nn.Parameter(torch.zeros(3))
      print(f'[ceres_net] PLACEMENT VALUE HEAD enabled: aux weight {self.placement_value_weight} '
            f'(additive per-square WDL decomposition over trunk flow [64 x {self.EMBEDDING_DIM}])')

    # K-ply survival head (AUXILIARY, training-only; SURVIVAL_TARGET_SPEC.md). Per-square
    # fate classification: class d in 1..K = piece captured d plies later, K+1 = survives;
    # class 0 (empty square) exists but is masked from the loss. Same export-safe stash
    # pattern as the placement head. Requires sidecar targets (CERES_TPG_TARGET_SIDECAR=1).
    self.survival_target_weight = float(os.environ.get('CERES_SURVIVAL_TARGET_WEIGHT', '0') or 0)
    self.survival_horizon = int(os.environ.get('CERES_SURVIVAL_HORIZON', '8') or 8)
    if self.survival_target_weight > 0:
      self.survival_head = nn.Linear(self.EMBEDDING_DIM, self.survival_horizon + 2)
      print(f'[ceres_net] SURVIVAL HEAD enabled: aux weight {self.survival_target_weight}, K={self.survival_horizon} '
            f'(per-square fate classification over trunk flow [64 x {self.EMBEDDING_DIM}])')

    # Short-term value head (AUXILIARY, training-only; V7_EXTRAS_SIDECAR_SPEC.md).
    # 3-logit WDL against the blunder-censored short-term EMA targets (censored q_st/d_st)
    # carried by V7-extras sidecars — a "what happens over the next few moves" value signal
    # that never blends across a detected blunder. Same additive per-square decomposition
    # and export-safe stash pattern as the placement head. Requires CERES_TPG_V7X_SIDECAR.
    self.stvalue_weight = float(os.environ.get('CERES_STVALUE_WEIGHT', '0') or 0)
    if self.stvalue_weight > 0:
      self.stvalue_head = nn.Linear(self.EMBEDDING_DIM, 3)
      self.stvalue_bias = nn.Parameter(torch.zeros(3))
      print(f'[ceres_net] SHORT-TERM VALUE HEAD enabled: aux weight {self.stvalue_weight} '
            f'(WDL vs censored q_st/d_st from .v7x sidecars)')



    if self.DEEPNORM:     
      self.alpha = math.pow(2 * self.NUM_LAYERS, 0.25)
    else:      
      self.alpha = 1

    SMOLGEN_PER_SQUARE_DIM = config.NetDef_SmolgenDimPerSquare
    SMOLGEN_INTERMEDIATE_DIM = config.NetDef_SmolgenDim

    ATTENTION_MULTIPLIER = config.NetDef_AttentionMultiplier

    if config.NetDef_SoftMoE_NumExperts > 0:
      assert config.NetDef_SoftMoE_MoEMode in ("None", "ReplaceLinear", "AddLinearSecondLayer", "ReplaceLinearSecondLayer"), 'implementation restriction: only AddLinearSecondLayer currently supported'
      assert config.NetDef_SoftMoE_NumSlotsPerExpert == 1
      assert config.NetDef_SoftMoE_UseBias == True
      assert config.NetDef_SoftMoE_UseNormalization == False
      assert config.NetDef_SoftMoE_OnlyForAlternatingLayers == True
      
    EPS = 1E-6
    
    if SMOLGEN_PER_SQUARE_DIM > 0 and SMOLGEN_INTERMEDIATE_DIM > 0:
      self.smolgenPrepLayer = nn.Linear(SMOLGEN_INTERMEDIATE_DIM // config.NetDef_SmolgenToHeadDivisor, NUM_TOKENS_NET * NUM_TOKENS_NET)
      # Optional LoRA wrap of the shared smolgen prep layer (gated by
      # CERES_LORA_SMOLGEN_RANK_DIV). Per-attention sm1/sm2/sm3 are wrapped
      # in dot_product_attention.py via the same env var.
      _sm_rd = int(os.environ.get('CERES_LORA_SMOLGEN_RANK_DIV', '0') or 0)
      if _sm_rd > 0:
        self.smolgenPrepLayer = LoRALinear(self.smolgenPrepLayer, _sm_rd, True)
    else:
      self.smolgenPrepLayer = None

    if config.NetDef_UseRPE or config.NetDef_UseRelBias:
      self.rpe_factor_shared = torch.nn.Parameter(make_rpe_map(), requires_grad=False)
    else:
      self.rpe_factor_shared = None

    # Visibility edge bias v2 — parsed HERE (before the transformer stack)
    # because the optional B/C content gates live inside each layer's attention
    # (they read that layer's per-head Q/K) and need the channel count at
    # EncoderLayer construction time. Module creation happens later, next to
    # the other bias modules. CONFIG-ONLY (NetDef fields, see config.py): these
    # knobs change the parameter tree, so they must be reconstructable from the
    # net config alone (resume, recover_export, serving).
    #   NetDef UseVisEdgeBias           enable channels + form-A per-layer bias
    #   NetDef VisEdgeFamilies          subset of vis,xray,pinray (default all)
    #   NetDef VisEdgeGates = q|k|qk    add content-gated forms B / C / B+C
    #   NetDef VisEdgeSharedProjection  one shared form-A projection (ablation)
    for _ev in ('CERES_VIS_EDGE_BIAS', 'CERES_VIS_EDGE_FAMILIES',
                'CERES_VIS_EDGE_GATES', 'CERES_VIS_EDGE_SHARED'):
      assert not os.environ.get(_ev), \
          f'{_ev} is retired — set the NetDef VisEdge* fields in the _ceres_net.json config instead'
    self.use_vis_edge_bias = bool(getattr(config, 'NetDef_UseVisEdgeBias', False))
    # Canonicalize families (lowercase, dedupe, canonical order) HERE so the
    # gate channel count below always agrees with VisibilityChannels'
    # num_channels (which applies the same canonicalization internally).
    _fams_raw = [f.strip().lower() for f in
                 str(getattr(config, 'NetDef_VisEdgeFamilies', 'vis,xray,pinray') or '').split(',') if f.strip()]
    for _f in _fams_raw:
      assert _f in VisibilityChannels.FAMILY_ORDER, \
          f'VisEdgeFamilies: unknown family {_f!r} (know: {VisibilityChannels.FAMILY_ORDER})'
    self._vis_edge_families = tuple(f for f in VisibilityChannels.FAMILY_ORDER if f in _fams_raw)
    if self.use_vis_edge_bias:
      assert self._vis_edge_families, 'VisEdgeFamilies resolved to an empty family list'
    self.vis_edge_shared = bool(getattr(config, 'NetDef_VisEdgeSharedProjection', False))
    # Accept the codebase-standard disable spellings for the gate field.
    _gm = str(getattr(config, 'NetDef_VisEdgeGates', '') or '').strip().lower()
    self.vis_edge_gate_mode = '' if _gm in ('', '0', 'off', 'none', 'false') else _gm
    if self.vis_edge_gate_mode:
      assert self.use_vis_edge_bias, 'VisEdgeGates requires UseVisEdgeBias=true'
      assert self.vis_edge_gate_mode in ('q', 'k', 'qk'), \
          f'VisEdgeGates must be q, k or qk (or empty/0/off to disable), got: {self.vis_edge_gate_mode}'
    _vis_gate_channels = (4 * len(self._vis_edge_families)
                          if (self.use_vis_edge_bias and self.vis_edge_gate_mode) else 0)

    num_tokens_q = NUM_TOKENS_NET
    num_tokens_kv = NUM_TOKENS_NET

    self.transformer_layer = torch.nn.Sequential(
       *[EncoderLayer('T', num_tokens_q, num_tokens_kv,
                      self.NUM_LAYERS, self.EMBEDDING_DIM,
                      self.FFN_MULT*self.EMBEDDING_DIM, 
                      config.NetDef_UseQKV,
                      config.NetDef_SoftCapCutoff,
                      config.NetDef_UseQKNorm,
                      self.NUM_HEADS,
                      ffn_activation_type = config.NetDef_FFNActivationType, 
                      norm_type = config.NetDef_NormType, layernorm_eps=EPS, 
                      use_global = config.NetDef_FFNUseGlobalEveryNLayers > 0 and (i % config.NetDef_FFNUseGlobalEveryNLayers) == config.NetDef_FFNUseGlobalEveryNLayers - 1,
                      attention_multiplier = ATTENTION_MULTIPLIER,
                      smoe_mode = config.NetDef_SoftMoE_MoEMode,
                      smoe_num_experts = config.NetDef_SoftMoE_NumExperts,
                      smoe_expert_input_dim = config.NetDef_SoftMoE_ExpertInputDim,
                      smolgen_per_square_dim = SMOLGEN_PER_SQUARE_DIM,
                      smolgen_intermediate_dim = SMOLGEN_INTERMEDIATE_DIM,
                      smolgen_head_divisor = config.NetDef_SmolgenToHeadDivisor,
                      smolgenPrepLayer = self.smolgenPrepLayer,
                      smolgen_activation_type = config.NetDef_SmolgenActivationType,
                      smolgen_delta_rank = config.NetDef_SmolgenDeltaRank,
                      alpha=self.alpha, layerNum=i, dropout_rate=self.DROPOUT_RATE,
                      use_rpe=config.NetDef_UseRPE, 
                      use_rpe_v=config.NetDef_UseRPE_V,
                      rpe_factor_shared=self.rpe_factor_shared,
                      use_rel_bias=config.NetDef_UseRelBias,
                      use_rope=config.NetDef_UseRoPE,
                      use_nonlinear_attention=config.NetDef_NonLinearAttention,
                      dual_attention_mode = config.NetDef_DualAttentionMode if not config.Exec_TestFlag else (config.NetDef_DualAttentionMode if i % 2 == 1 else 'None'),
                      test = config.Exec_TestFlag,
                      use_diff_attention = config.NetDef_UseDiffAttention,
                      tsb_enabled = getattr(config, 'NetDef_TSB_Enabled', False),
                      tsb_ffn_multiplier = getattr(config, 'NetDef_TSB_FFNMultiplier', 1),
                      tsb_gate_bias_init = getattr(config, 'NetDef_TSB_GateBiasInit', -4.0),
                      tsb_gate_mlp_hidden_divisor = getattr(config, 'NetDef_TSB_GateMLPHiddenDivisor', 8),
                      vis_gate_channels = _vis_gate_channels,
                      vis_gate_mode = self.vis_edge_gate_mode,
                      pre_norm = config.NetDef_PreNorm)
        for i in range(self.NUM_DISTINCT_LAYERS)])

    # Pre-norm trunks need a final norm AFTER the stack and before the heads,
    # because the residual stream is unnormalized at the stack output (each
    # block applies norm only to its sublayer input, never to the residual).
    # Without this, the head input's distribution depends on stack depth /
    # init scale and can blow up. Standard pattern in LLaMA, GPT-NeoX, etc.
    if config.NetDef_PreNorm:
      self.trunk_end_norm = make_norm(config.NetDef_NormType, self.EMBEDDING_DIM, eps=1E-6)
    else:
      self.trunk_end_norm = None

    self.policy_loss_weight = policy_loss_weight
    self.value_loss_weight = value_loss_weight
    self.moves_left_loss_weight = moves_left_loss_weight
    self.unc_loss_weight = unc_loss_weight
    self.value2_loss_weight = value2_loss_weight
    self.q_deviation_loss_weight = q_deviation_loss_weight
    self.value_diff_loss_weight = value_diff_loss_weight
    self.value2_diff_loss_weight = value2_diff_loss_weight
    self.action_loss_weight = action_loss_weight
    self.uncertainty_policy_weight = uncertainty_policy_weight
    self.action_uncertainty_loss_weight = action_uncertainty_loss_weight
    self.q_ratio = q_ratio

    if (self.denseformer):
      self.dwa_modules = torch.nn.ModuleList([DWA(n_alphas=i+2, depth=self.EMBEDDING_DIM) for i in range(self.NUM_LAYERS)])

    # Plan 3: piece-relation attention bias. One shared module computes a
    # per-position bias of shape [B, num_heads, 64, 64] from the piece-type
    # one-hot in `squares`; the same bias is fed to every encoder layer's
    # attention. Reuse-across-layers is intentional — piece relations are a
    # property of the position, not of the depth of reasoning.
    self.use_piece_relation_bias = config.NetDef_UsePieceRelationBias
    if self.use_piece_relation_bias:
      self.piece_relation_bias_module = PieceRelationBias(num_heads=self.NUM_HEADS)

    # Ray attention bias (CERES_RAY_ATTENTION_BIAS=1): blocker-aware sliding-piece
    # attack + x-ray channels as additive attention bias, computed in-graph from
    # the piece one-hots (see chess_geometry.RayAttentionBias). Motivation from the
    # 2026-08 mechanism decomposition: RPE's geometry-in-attention delivered the
    # OOD generalization (+126) but its Q·R wiring costs ~23% NPS (fused-MHA break);
    # this delivers RESOLVED geometry (pins/x-rays included, which RPE cannot see)
    # through the cheap additive-bias pattern. Composes additively with PRB.
    # RPE-from-embedding experiment (CERES_RPE_FROM_EMBEDDING=1, requires UseRPE):
    # RPE einsums read the post-embedding state through each layer's own qkv
    # weights instead of the layer's live Q/K. Zero new params; measures how much
    # of RPE's win requires LAYER-COMPUTED content (the discriminator between
    # cheap once-materialized geometry conditioning and per-layer schemes).
    self.rpe_from_embedding = int(os.environ.get('CERES_RPE_FROM_EMBEDDING', '0') or 0) > 0
    if self.rpe_from_embedding:
      print('[ceres_net] RPE-FROM-EMBEDDING experiment enabled: RPE terms read static '
            'post-embedding content (zero new params)')
    # ARCHITECTED serving graph for RPE-fromEmb (CERES_RPE_GENPHASE=1, requires
    # CERES_RPE_FROM_EMBEDDING=1): since the coupling content is STATIC per
    # position, each layer's QK-rpe contribution is computed ONCE in a generator
    # phase at forward start and delivered as a per-layer ADDITIVE score bias —
    # the attention kernels keep a plain fusable QK^T(+bias) and the in-attention
    # rpe einsums are skipped. Mathematically identical for the QK terms (bias is
    # pre-divided by sqrt(d_k) to match the entry point after score scaling);
    # rpe_v intentionally dropped (measured dead weight, pvsmoke8).
    self.rpe_genphase = int(os.environ.get('CERES_RPE_GENPHASE', '0') or 0) > 0
    if self.rpe_genphase:
      assert self.rpe_from_embedding, 'CERES_RPE_GENPHASE requires CERES_RPE_FROM_EMBEDDING=1'
      print('[ceres_net] RPE GENERATOR-PHASE serving graph enabled: per-layer QK-rpe '
            'biases precomputed from static embedding; in-attention einsums + rpe_v skipped')

    _ray_env = int(os.environ.get('CERES_RAY_ATTENTION_BIAS', '0') or 0) > 0
    self.use_ray_bias = bool(getattr(config, 'NetDef_UseRayAttentionBias', False)) or _ray_env
    if _ray_env:
      print('[ceres_net] DEPRECATED: CERES_RAY_ATTENTION_BIAS env var — set '
            '"UseRayAttentionBias": true in the _ceres_net.json config instead. '
            'The env fallback exists only for the in-flight prod 200M run and '
            'will become a hard error.')
    if self.use_ray_bias:
      self.ray_bias_module = RayAttentionBias(num_heads=self.NUM_HEADS)
      print(f'[ceres_net] RAY ATTENTION BIAS enabled: 6 blocker-aware slider/x-ray '
            f'channels -> per-head additive bias ({6 * self.NUM_HEADS} params)')

    # Visibility edge bias v2 (CERES_VIS_EDGE_BIAS=1): channel builder + form-A
    # content-free injection per the Kovax visibility program
    # (C:\Dev\Chess\Temp\VISIBILITY_PROGRAM.md). Pairwise {0,1} edge channels
    # (families vis/xray/pinray, each stm/opp x out/in) built ONCE per forward
    # by chess_geometry.VisibilityChannels and shared by all layers; each layer
    # projects them with its own zero-init Linear(C -> num_heads) (per-block
    # attack_w in the source program; zero-init => exact step-0 no-op).
    # CERES_VIS_EDGE_FAMILIES selects families (default all three);
    # CERES_VIS_EDGE_SHARED=1 collapses to one shared projection (ablation arm).
    # Composes additively with PRB / ray-bias / RPE via the piece_relation_bias
    # path. Env parsing happened before the transformer stack (see there); the
    # optional B/C content-gate parameters live inside each layer's attention.
    if self.use_vis_edge_bias:
      # Mutually exclusive with the ray bias: VisibilityChannels' slider-vis and
      # xray families re-express RayAttentionBias's channels exactly, so running
      # both double-parameterizes identical indicators and makes the
      # "ray-bias vs vis-bias" ablation uninterpretable.
      assert not self.use_ray_bias, \
          'UseVisEdgeBias and CERES_RAY_ATTENTION_BIAS are mutually exclusive ablation arms'
      self.vis_channels_module = VisibilityChannels(families=self._vis_edge_families)
      _n_proj = 1 if self.vis_edge_shared else self.NUM_DISTINCT_LAYERS
      _C = self.vis_channels_module.num_channels
      self.vis_edge_proj = torch.nn.ModuleList(
          [torch.nn.Linear(_C, self.NUM_HEADS, bias=False) for _ in range(_n_proj)])
      for _lin in self.vis_edge_proj:
        torch.nn.init.zeros_(_lin.weight)
      print(f'[ceres_net] VISIBILITY EDGE BIAS enabled: families={self.vis_channels_module.families} '
            f'({_C} channels), {"shared" if self.vis_edge_shared else "per-layer"} '
            f'zero-init projection ({_C * self.NUM_HEADS * _n_proj} params), '
            f'content gates: {self.vis_edge_gate_mode or "off"}')

    # Phase-FiLM (CERES_PHASE_FILM=1): phase-conditioned per-layer FFN modulation.
    # Rationale: a small net averages one circuit over opening/middlegame/endgame;
    # FiLM gives it phase-specialized sub-circuits at elementwise cost (MoE-without-
    # routing). Conditioning = the position's piece census (sum over squares of the
    # 13-channel one-hot, the same planes PieceRelationBias reads) + a material
    # scalar, normalized to O(1) -> tiny MLP -> per-DISTINCT-layer (gamma, beta) of
    # width D, applied to each layer's FFN output as out*(1+gamma)+beta (broadcast
    # over squares). Final projection zero-init => exact no-op at step 0.
    # 'phase_film' prefix is in train.py's aux-resume prefixes (ckpt-compatible in
    # both directions). Serving cost ~nil: MLP runs once per position (~0.4M MACs),
    # application is elementwise (same op class as the measured-free attention gate).
    self.use_phase_film = int(os.environ.get('CERES_PHASE_FILM', '0') or 0) > 0
    if self.use_phase_film:
      _film_hidden = 64
      self.phase_film_mlp = nn.Sequential(
        nn.Linear(14, _film_hidden),
        self.Activation,
        nn.Linear(_film_hidden, self.NUM_DISTINCT_LAYERS * 2 * self.EMBEDDING_DIM))
      nn.init.zeros_(self.phase_film_mlp[2].weight)
      nn.init.zeros_(self.phase_film_mlp[2].bias)
      # Material weights per one-hot channel: empty,P,N,B,R,Q,K (white then black).
      self.register_buffer('phase_film_matw',
                           torch.tensor([0., 1., 3., 3., 5., 9., 0., 1., 3., 3., 5., 9., 0.]),
                           persistent=False)
      _n = 14 * _film_hidden + _film_hidden + _film_hidden * self.NUM_DISTINCT_LAYERS * 2 * self.EMBEDDING_DIM + self.NUM_DISTINCT_LAYERS * 2 * self.EMBEDDING_DIM
      print(f'[ceres_net] PHASE-FILM enabled: piece-census -> per-layer FFN (gamma, beta) '
            f'[{self.NUM_DISTINCT_LAYERS} x 2 x {self.EMBEDDING_DIM}] ({_n} params, zero-init no-op)')

    # GTAB (Gated Tactical Adapter Branch) — optional parallel mini-transformer
    # that contributes additively to the post-body flow, gated by a learned
    # position classifier. Zero-init by construction: orig is recovered exactly
    # at training step 0. See tactical_adapter.py for details.
    self.use_gtab = gtab_enabled()
    assert self.value_head_channels == 0 or (self.vda_mode == 0 and not self.use_gtab),       'ValueHeadChannels (private value front-end) is incompatible with vda/gtab modes'
    self.gtab_value_only = self.use_gtab and (int(os.environ.get('CERES_GTAB_VALUE_ONLY', '0') or 0) > 0)
    if self.use_gtab:
      self.tactical_adapter = TacticalAdapter(in_dim=self.EMBEDDING_DIM)
      self.tactical_gate    = PositionGate(in_dim=self.EMBEDDING_DIM)
      # Buffer for last gate activation (for sparsity loss + diagnostics).
      self._last_gate_value = None

    # TSB (Tactical SwiGLU Bypass) — per-block parallel SwiGLU FFN + scalar gate.
    # Each EncoderLayer holds its own TSBSwiGLU; here we just track net-level
    # state. Gate values are collected after the encoder forward for the
    # gate-sparsity regularizer in train.py.
    self.use_tsb = bool(getattr(config, 'NetDef_TSB_Enabled', False))
    self._last_tsb_gates = None  # set in forward() when TSB is active


  def _rpe_genphase_biases(self, emb):
    """RPE-fromEmb architected graph (see __init__): per-layer QK-rpe score biases
    computed once from the static post-embedding state. Returns a list of
    [B, H, 64, 64] tensors, one per distinct layer."""
    biases = []
    # fp32 throughout the generator: keeps the exported graph free of BF16-typed
    # ops (TRT's Mish importer rejects BF16); the bias is cast to the score dtype
    # at the add site anyway.
    emb = emb.to(torch.float32)
    B = emb.shape[0]
    for layer in self.transformer_layer:
      att = layer.attention
      if att.use_nonlinear_attention:
        qkv_e = att.qkv(emb).reshape(B, -1, 3, att.d_model * att.attention_multiplier)
        qkv_e = torch.nn.functional.mish(att.qkvLN(qkv_e))
        _qe, _ke, _ = torch.unbind(qkv_e, dim=-2)
        Qe = att.q2(_qe).reshape(B, -1, att.num_heads, att.d_k * att.attention_multiplier).permute(0, 2, 1, 3)
        Ke = att.k2(_ke).reshape(B, -1, att.num_heads, att.d_k * att.attention_multiplier).permute(0, 2, 1, 3)
      else:
        qkv_e = att.qkv(emb).reshape(B, -1, att.num_heads, 3 * att.d_k * att.attention_multiplier).permute(0, 2, 1, 3)
        Qe, Ke, _ = qkv_e.chunk(3, dim=-1)
      if att.use_qk_norm:
        Qe = att.qLN(Qe)
        Ke = att.kLN(Ke)
      _d = att.d_k * att.attention_multiplier
      rpe_q = (att.rpe_q @ att.rpeFactorShared).reshape(_d, att.num_heads, 64, 64)
      rpe_k = (att.rpe_k @ att.rpeFactorShared).reshape(_d, att.num_heads, 64, 64)
      _bias = torch.einsum('bhqd,dhqk->bhqk', Qe, rpe_q) + torch.einsum('bhkd,dhqk->bhqk', Ke, rpe_k)
      biases.append(_bias / (att.d_k ** 0.5))
    return biases

  def _log(self, name, value, step):
    """Log a scalar metric to tensorboard. Replaces former fabric.log() call.
    Tolerates tensor or python-scalar values; no-op if writer is None."""
    if self.writer is None:
      return
    if isinstance(value, torch.Tensor):
      value = value.item()
    self.writer.add_scalar(name, value, step)


  def forward(self, squares: torch.Tensor, prior_state:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(squares, list):
      # when saving/restoring from ONNX the input will appear as a list instead of sequence of arguments
      squares = squares[0]

    flow = squares

    # Plan 3: chess-specific piece-relation attention bias. Compute once per
    # forward from the per-square 13-channel piece-type one-hot (current ply,
    # bytes 0..12 of each square's encoding per TPGSquareRecord layout).
    # Reused across all encoder layers to avoid re-projecting per layer.
    piece_relation_bias_tensor = None
    if self.use_piece_relation_bias:
      piece_type_curr = squares[:, :, 0:13]  # [B, 64, 13]
      piece_relation_bias_tensor = self.piece_relation_bias_module(piece_type_curr)
    if self.use_ray_bias:
      _ray_bias = self.ray_bias_module(squares[:, :, 0:13])
      piece_relation_bias_tensor = _ray_bias if piece_relation_bias_tensor is None \
          else piece_relation_bias_tensor + _ray_bias

    # Visibility edge channels (see __init__): built once per forward, shared
    # by all layers. Form-A biases are precomputed per distinct layer HERE
    # (the rpe_gen_biases pattern) rather than inside the layer loop, so
    # shared mode computes one tensor and LoopCount>1 never recomputes.
    vis_edge_E = None
    vis_edge_biases = None
    if self.use_vis_edge_bias:
      vis_edge_E = self.vis_channels_module(squares[:, :, 0:13])  # [B, 64, 64, C]
      if self.vis_edge_shared:
        _vb0 = self.vis_edge_proj[0](vis_edge_E).permute(0, 3, 1, 2)
        vis_edge_biases = [_vb0] * self.NUM_DISTINCT_LAYERS
      else:
        # All per-layer projections in ONE matmul (weights concatenated along
        # the output dim — constant-folded at export) + one permute, instead of
        # NUM_DISTINCT_LAYERS separate matmul+permute materializations.
        _W = torch.cat([_lin.weight for _lin in self.vis_edge_proj], dim=0)  # [L*H, C]
        _vb = torch.matmul(vis_edge_E, _W.transpose(0, 1))                   # [B,64,64,L*H]
        _vb = _vb.reshape(_vb.shape[0], 64, 64, self.NUM_DISTINCT_LAYERS,
                          self.NUM_HEADS).permute(0, 3, 4, 1, 2)             # [B,L,H,64,64]
        vis_edge_biases = list(_vb.unbind(1))

    # Phase-FiLM conditioning (see __init__): piece census + material scalar from
    # the same one-hot planes PRB reads, computed ONCE per forward. Normalization:
    # counts x0.1 (piece counts 0..8 -> ~O(1)), material x0.025 (0..78 -> ~O(1)).
    phase_film_tensor = None
    if self.use_phase_film:
      _pf_census = squares[:, :, 0:13].sum(dim=1)                              # [B, 13]
      _pf_mat = (_pf_census * self.phase_film_matw).sum(dim=-1, keepdim=True)  # [B, 1]
      _pf_feat = torch.cat((_pf_census * 0.1, _pf_mat * 0.025), dim=-1)        # [B, 14]
      phase_film_tensor = self.phase_film_mlp(_pf_feat.to(self.phase_film_mlp[0].weight.dtype)) \
          .reshape(-1, self.NUM_DISTINCT_LAYERS, 2, self.EMBEDDING_DIM)        # [B, L, 2, D]

    # save a copy of the qblunders (2 bytes) for later use in value 2 head (only)
    QBLUNDER_SLICE_BEGIN = 119
    QBLUNDER_SLICE_END = 121
    qblunders_negative_positive = squares[:, 0, QBLUNDER_SLICE_BEGIN:QBLUNDER_SLICE_END].clone().view(-1, 2)

    # insert zeros at the two qblunder slots in the main flow (masked out)
    # N.B. it is essential that this operation be done not using in-place operations so the 
    #      PyTorch compile and ONNX graph generation correctly captures in the computation graph
    flow = torch.cat((flow[:, :, :QBLUNDER_SLICE_BEGIN], torch.zeros_like(flow[:, :, QBLUNDER_SLICE_BEGIN:QBLUNDER_SLICE_END]), flow[:, :, QBLUNDER_SLICE_END:]), dim=2)

    # experimental (disabled) custom value head blending
    # condition = (flow[:, :, 109] <0.01) & (flow[:, :, 110] < 0.3) & (flow[:, :, 111] < 0.3)
    # flow[:, :, 107] = condition.bfloat16()
    # flow[:, :, 108] = 1 - flow[:, :, 107]

    # Embedding layer.
    flow_squares = flow.reshape(-1, NUM_TOKENS_INPUT, (NUM_TOKENS_INPUT * TOTAL_INPUT_FEATURES_PER_SQUARE) // NUM_TOKENS_INPUT)

    if self.prior_state_dim > 0:
      # Append prior state to the input if is available for this position.
      append_tensor = prior_state if prior_state is not None else torch.zeros(squares.shape[0], NUM_TOKENS_INPUT, self.prior_state_dim).to(flow.device).to(torch.bfloat16)
      append_tensor = append_tensor.reshape(squares.shape[0], NUM_TOKENS_INPUT, self.prior_state_dim)
      flow_squares = torch.cat((flow_squares, append_tensor), dim=-1)

    flow = self.embedding_layer(flow_squares)
    flow = self.embedding_norm(flow)

    # GTAB reads the post-embedding flow (independent of body) so the adapter
    # branch is structurally separate from any body distortion.
    flow_post_embed = flow if self.use_gtab else None

    if self.denseformer:
      all_previous_x = [flow]

    # Depth-attention state collection: RAW references for ALL modes (deferred
    # pooling — zero ops in the trunk, all compute at the tail; activations are
    # alive for backward anyway so collection is free).
    if self.use_value_depth_attention or self.pda_mode or (self.depth_probes_enabled and self.training):
      vda_states = [flow]  # post-embedding state
    rpe_src_tensor = flow if (self.rpe_from_embedding and not self.rpe_genphase) else None  # post-embedding, pre-layer-1
    rpe_gen_biases = self._rpe_genphase_biases(flow) if (self.rpe_from_embedding and self.rpe_genphase) else None

    # Main transformer body (stack of encoder layers).
    # Looped transformer: apply NUM_DISTINCT_LAYERS modules LOOP_COUNT times,
    # giving total effective depth = NUM_LAYERS. Default LoopCount=1 reduces
    # to the original "stack of NUM_LAYERS distinct layers" behaviour.
    # DenseFormer + LoopCount>1 is not supported (DWA expects a unique module
    # per effective layer position) — guard against it explicitly.
    if self.denseformer and self.LOOP_COUNT != 1:
      raise NotImplementedError("DenseFormer is not supported with LoopCount > 1.")
    for loop_iter in range(self.LOOP_COUNT):
      for i in range(self.NUM_DISTINCT_LAYERS):
        _prb_l = piece_relation_bias_tensor
        if rpe_gen_biases is not None:
          _prb_l = rpe_gen_biases[i] if _prb_l is None else _prb_l + rpe_gen_biases[i]
        if vis_edge_biases is not None:
          _vb = vis_edge_biases[i]  # [B, H, 64, 64], precomputed above
          _prb_l = _vb if _prb_l is None else _prb_l + _vb
        flow = self.transformer_layer[i](flow, piece_relation_bias=_prb_l,
                                         film=None if phase_film_tensor is None else
                                              (phase_film_tensor[:, i, 0].unsqueeze(1),
                                               phase_film_tensor[:, i, 1].unsqueeze(1)),
                                         rpe_src=rpe_src_tensor,
                                         rpe_precomputed=rpe_gen_biases is not None,
                                         vis_edge=vis_edge_E if self.vis_edge_gate_mode else None)
        if self.denseformer:
          eff_idx = loop_iter * self.NUM_DISTINCT_LAYERS + i
          all_previous_x.append(flow)
          flow = self.dwa_modules[eff_idx](all_previous_x)
        if self.use_value_depth_attention or self.pda_mode or (self.depth_probes_enabled and self.training):
          vda_states.append(flow)  # post-layer (post-DWA if denseformer)

    # Pre-norm: final norm before the heads (see __init__ comment).
    if self.trunk_end_norm is not None:
      flow = self.trunk_end_norm(flow)

    # TSB: collect per-block gate values across all layers for the gate-sparsity
    # regularizer in train.py. Each layer caches its last gate in _last_tsb_gate.
    if self.use_tsb:
      _gates = [layer._last_tsb_gate for layer in self.transformer_layer
                if getattr(layer, '_last_tsb_gate', None) is not None]
      if len(_gates) > 0:
        self._last_tsb_gates = torch.stack(_gates, dim=0)  # [num_layers, B, 1, 1]
      else:
        self._last_tsb_gates = None


    # GTAB residual add: tactical adapter contributes additively to body output,
    # gated by a learned position classifier. Adapter output is zero-init so
    # this is a no-op at training step 0 (orig recovered exactly).
    #
    # Two routing modes:
    #   - default (CERES_GTAB_VALUE_ONLY=0): adapter contributes to ALL heads
    #     via the shared `flow` tensor. Same as v100/v101/v102.
    #   - value-only (CERES_GTAB_VALUE_ONLY=1): adapter contributes only to
    #     value_head and value2_head; policy and other heads read orig body
    #     output. Two parallel passes through head front-end. Policy is
    #     bit-identical to orig (modulo head LoRA recalibration if any).
    if self.use_gtab:
      flow_aux = self.tactical_adapter(flow_post_embed)   # [B, 64, dim]
      g_x      = self.tactical_gate(flow_post_embed)      # [B, 1, 1]
      self._last_gate_value = g_x                          # cached for sparsity loss + diagnostics
      if self.gtab_value_only:
        # Compute two head front-end paths.
        flow_value  = flow + g_x * flow_aux                  # value-side: with adapter
        flow_others = flow                                   # other-heads: orig body output
        fS_value  = self.headSharedLinear(self.headPremap(flow_value).reshape(-1, 64 * self.HEAD_PREMAP_PER_SQUARE))
        fS_others = self.headSharedLinear(self.headPremap(flow_others).reshape(-1, 64 * self.HEAD_PREMAP_PER_SQUARE))
      else:
        flow = flow + g_x * flow_aux
        fS_others = self.headSharedLinear(self.headPremap(flow).reshape(-1, 64 * self.HEAD_PREMAP_PER_SQUARE))
        fS_value  = fS_others
    else:
      fS_others = self.headSharedLinear(self.headPremap(flow).reshape(-1, 64 * self.HEAD_PREMAP_PER_SQUARE))
      fS_value  = fS_others

    # PRIVATE VALUE FRONT-END (see __init__): the value family gets its own
    # per-square projection, bypassing the policy-shared bottleneck. gtab/vda
    # are asserted off in this mode, so `flow` is the correct source tensor.
    # 'replace' swaps the head input outright; 'inject' keeps the shared input
    # and adds the private features inside the head instead (see below).
    _v_inject = None
    _v2_inject = None
    if self.value_head_channels > 0:
      _priv_value = self.value_premap(flow).reshape(-1, self.VALUE_PRIV_SIZE)
      if self.value_priv_replace:
        fS_value = _priv_value
      else:
        _v_inject = self.value_priv_inject(_priv_value)
        if self.value2_loss_weight > 0:
          _v2_inject = self.value2_priv_inject(_priv_value)

    # Depth-attending value context (see __init__). Non-in-place adds create NEW
    # tensors, so fS_others (often the same object as fS_value) is untouched —
    # policy and all other heads are bit-identical to the baseline path.
    _hn_shared = None  # normed per-square depth states, shared by vda modes 2/3/4 and pda
    if self.vda_mode == 1:
      # Deferred pooling at the tail (identical math to pooling at collection —
      # mean over squares commutes). Mean-then-stack: avoids materializing a
      # [B, L+1, 64, D] tensor that stack-then-mean would keep for backward.
      _vda_pooled = torch.stack([_hs.mean(dim=1) for _hs in vda_states], dim=1)  # [B, L+1, D]
      _vda_h = self.vda_norm(_vda_pooled)
      _vda_scores = torch.matmul(_vda_h, self.vda_query) * (self.EMBEDDING_DIM ** -0.5)
      _vda_alpha = torch.softmax(_vda_scores, dim=1).unsqueeze(-1)         # [B, L+1, 1]
      fS_value = fS_value + self.vda_proj((_vda_alpha * _vda_h).sum(dim=1))
      if self.training:
        # Stash for diagnostics (same training-gated attribute-mutation pattern as
        # the placement head — keeps the dynamo/ONNX export path mutation-free).
        self._last_vda_alpha = _vda_alpha.detach()
    elif self.vda_mode == 4 and not self.training:
      pass  # serving graph is EXACTLY novda — no vda ops exported
    elif self.vda_mode in (2, 3, 4):
      # Per-square: each square attends over its OWN depth trajectory. Depth loop
      # (static unroll under compile) avoids materializing a [B, L+1, 64, D] stack.
      _hn = [self.vda_norm(_hs) for _hs in vda_states]                     # (L+1) x [B, 64, D]
      _hn_shared = _hn
      _vda_scores = torch.stack([torch.matmul(_h, self.vda_query) for _h in _hn],
                                dim=-1) * (self.EMBEDDING_DIM ** -0.5)     # [B, 64, L+1]
      _vda_alpha = torch.softmax(_vda_scores, dim=-1)                      # [B, 64, L+1]
      _ctx = _vda_alpha[..., 0:1] * _hn[0]
      for _i in range(1, len(_hn)):
        _ctx = _ctx + _vda_alpha[..., _i:_i + 1] * _hn[_i]                 # [B, 64, D]
      # Inject into the value-side flow BEFORE the head front-end (GTAB pattern);
      # zero-init vda_proj -> _flow_v == the branch flow -> fS_value recomputes to
      # bit-identical values at init. In gtab_value_only mode this recompute
      # replaces the fS_value computed above (redundant pass, rare mode, harmless).
      _flow_v = (flow_value if (self.use_gtab and self.gtab_value_only) else flow) + self.vda_proj(_ctx)
      _fS_aug = self.headSharedLinear(self.headPremap(_flow_v).reshape(-1, 64 * self.HEAD_PREMAP_PER_SQUARE))
      if self.vda_mode != 4:
        fS_value = _fS_aug   # modes 2/3: the served value head reads the augmented path
      if self.vda_mode in (3, 4):
        # Combined mode: ALSO the pooled/global branch (full-bandwidth add AFTER
        # the head front-end bottleneck). Reuses the per-square normed states —
        # pooling normed states differs from norming pooled states, but both are
        # valid parameterizations and this saves a second norm pass.
        _hg = torch.stack([_h.mean(dim=1) for _h in _hn], dim=1)           # [B, L+1, D] (mean-then-stack, no big transient)
        _g_scores = torch.matmul(_hg, self.vda_query_g) * (self.EMBEDDING_DIM ** -0.5)
        _g_alpha = torch.softmax(_g_scores, dim=1).unsqueeze(-1)           # [B, L+1, 1]
        _g_ctx = self.vda_proj_g((_g_alpha * _hg).sum(dim=1))
        if self.vda_mode == 4:
          _fS_aug = _fS_aug + _g_ctx
        else:
          fS_value = fS_value + _g_ctx
        if self.training:
          self._last_vda_alpha_g = _g_alpha.detach()
      if self.vda_mode == 4:
        # Auxiliary value head on the augmented path (training-only reach — this
        # whole branch is skipped in eval by the guard above). Stash for the aux
        # loss in compute_loss; served value_head reads the untouched fS_value.
        self._last_vda_aux_out = self.vda_aux_head(_fS_aug)
      if self.training:
        # Square-averaged profile keeps the compute_loss logging block shape-
        # compatible ([B, L+1, 1]); entropy there is entropy-of-mean, a coarser
        # diagnostic than per-square entropy but comparable across modes.
        self._last_vda_alpha = _vda_alpha.detach().mean(dim=1).unsqueeze(-1)

    # Policy depth attention (tier 1; see __init__). Reuses the vda-normed depth
    # states when the vda branch already computed them (state reads paid once);
    # otherwise norms them here (pda without vda, or vda mode 4 in eval). The
    # policy-side flow gets its own depth context and a separate front-end pass
    # produces fS_policy for the POLICY head only — mlh/unc/action/etc stay on
    # fS_others. Zero-init vda_proj_p => fS_policy == fS_others at init.
    fS_policy = fS_others
    if self.pda_mode:
      if _hn_shared is None:
        _hn_shared = [self.vda_norm(_hs) for _hs in vda_states]
      _pda_scores = torch.stack([torch.matmul(_h, self.vda_query_p) for _h in _hn_shared],
                                dim=-1) * (self.EMBEDDING_DIM ** -0.5)     # [B, 64, L+1]
      _pda_alpha = torch.softmax(_pda_scores, dim=-1)
      _pctx = _pda_alpha[..., 0:1] * _hn_shared[0]
      for _i in range(1, len(_hn_shared)):
        _pctx = _pctx + _pda_alpha[..., _i:_i + 1] * _hn_shared[_i]        # [B, 64, D]
      _flow_p = flow + self.vda_proj_p(_pctx)
      fS_policy = self.headSharedLinear(self.headPremap(_flow_p).reshape(-1, 64 * self.HEAD_PREMAP_PER_SQUARE))
      if self.training:
        self._last_pda_alpha = _pda_alpha.detach().mean(dim=1).unsqueeze(-1)

    # Depth probes (training-only; see __init__). Pooled per-depth states through
    # the SHARED probe heads; final state (detached) through the ctl heads. Stash
    # pattern — never part of the export signature.
    if self.depth_probes_enabled and self.training:
      _dp_pooled = torch.stack([_hs.mean(dim=1) for _hs in vda_states], dim=1)   # [B, L+1, D]
      _dp_n = self.depth_probe_norm(_dp_pooled)
      self._last_depth_probe_policy = self.depth_probe_policy(_dp_n)             # [B, L+1, 1858]
      self._last_depth_probe_value = self.depth_probe_value(_dp_n)               # [B, L+1, 3]
      _dp_fin = _dp_n[:, -1].detach()   # ctl heads: final state only, trunk NEVER shaped by them
      self._last_depth_ctl_policy = self.depth_ctl_policy(_dp_fin)               # [B, 1858]
      self._last_depth_ctl_value = self.depth_ctl_value(_dp_fin)                 # [B, 3]

    # Placement value head (aux, training-only; see __init__). Per-square WDL-logit
    # contributions summed over squares, stashed for compute_loss. Gated on
    # self.training so ONNX/TorchScript export (which runs under eval()) never
    # executes the attribute mutation — the PT2 dynamo export path rejects tensor
    # attribute mutation in forward, and the swallowed exception would otherwise
    # silently produce checkpoint-only runs with no .onnx.
    if (self.placement_value_weight > 0 or self.survival_target_weight > 0 or self.stvalue_weight > 0) and self.training:
      flow_aux_src = flow_value if (self.use_gtab and self.gtab_value_only) else flow
      if self.placement_value_weight > 0:
        pv_contrib = self.placement_value_head(flow_aux_src)                       # [B, 64, 3]
        self._last_placement_value_out = pv_contrib.sum(dim=1) + self.placement_value_bias  # [B, 3]
      if self.survival_target_weight > 0:
        self._last_survival_out = self.survival_head(flow_aux_src)                 # [B, 64, K+2]
      if self.stvalue_weight > 0:
        st_contrib = self.stvalue_head(flow_aux_src)                               # [B, 64, 3]
        self._last_stvalue_out = st_contrib.sum(dim=1) + self.stvalue_bias         # [B, 3]

    # Value-contrast aux head (see __init__): training-only stash, never in the
    # export graph (same gating pattern as the placement/survival aux heads).
    if self.value_contrast_weight > 0 and self.training:
      self._last_vc_out = self.vc_head(fS_others).reshape(-1, 1858, 3)

    # Soft-policy aux head (see __init__): reads the same features as the main
    # policy head; training-only stash.
    if self.soft_policy_weight > 0 and self.training:
      self._last_sp_out = self.sp_head(fS_policy)

    # HL-Gauss categorical value head (see __init__): training-only stash.
    if self.hlg_weight > 0 and self.training:
      self._last_hlg_out = self.hlg_head(fS_value)

    # Optimistic-policy aux head (see __init__): training-only stash.
    if self.opt_policy_weight > 0 and self.training:
      self._last_opt_out = self.opt_head(fS_policy)

    # Heads. Policy reads fS_policy (== fS_others unless pda); value reads fS_value (with adapter).
    policy_out = self.policy_head(fS_policy)

    # Policy SERVE blend (see __init__): eval/export mode only — three-way
    # logit-space mix of vanilla / optimistic / soft heads, applied BEFORE the
    # ray-context add so rc stays unscaled at any lambda. Training untouched.
    if (self.opt_serve_blend > 0 or self.soft_serve_blend > 0) and not self.training:
      _pol_blend = (1.0 - self.opt_serve_blend - self.soft_serve_blend) * policy_out
      if self.opt_serve_blend > 0:
        _pol_blend = _pol_blend + self.opt_serve_blend * self.opt_head(fS_policy)
      if self.soft_serve_blend > 0:
        _pol_blend = _pol_blend + self.soft_serve_blend * self.sp_head(fS_policy)
      policy_out = _pol_blend
    if self.ray_context_mode > 0:
      # Ray-context factored term (see __init__): every move's logit reads its
      # own from/to square states — and in mode 2 the live contents of its ray
      # (between+behind) and of the rays it vacates (discovery). `flow` here is
      # the post-trunk_end_norm [B, 64, D] square state the heads read.
      _rcF = self.rc_WF(flow)                                       # [B, 64, dh]
      _rcT = self.rc_WT(flow)
      if getattr(self, 'rc_chunks', 0) > 0:
        # Memory-lean serving formulation (see __init__). Identity:
        #   bilinear_m = <F[from_m], T[to_m]> = (F @ T^T)[from_m, to_m]
        #   sum_d F[from_m,d]*u_d*(R@G)[m,d] = (R @ (G @ (F*u)^T))[m, from_m]
        # so all cross terms reduce to [B,64,64] H-matrices; only the final
        # R/R2 matmul spans the 1858 axis, and it is processed in chunks.
        _FT = torch.matmul(_rcF, _rcT.transpose(1, 2))              # [B, 64, 64]
        _rc_add = _FT.reshape(-1, 4096)[:, self.rc_ft_flat] * (self.rc_dh ** -0.5) + self.rc_btype
        if self.ray_context_mode >= 2:
          _G_R = self.rc_WR(flow)                                   # [B, 64, dh]
          _G_D = self.rc_WD(flow)
          _H1 = torch.matmul(_G_R, (_rcF * self.rc_u).transpose(1, 2))  # [B, 64, 64] (s x from)
          _H2 = torch.matmul(_G_R, (_rcT * self.rc_v).transpose(1, 2))  # (s x to)
          _H3 = torch.matmul(_G_D, (_rcF * self.rc_w).transpose(1, 2))  # (s x from)
          _parts = []
          _csz = (1858 + self.rc_chunks - 1) // self.rc_chunks
          for _c0 in range(0, 1858, _csz):
            _c1 = min(_c0 + _csz, 1858)
            _Rc = self.rc_R[_c0:_c1].to(_rcF.dtype)                 # [C, 64]
            _R2c = self.rc_R2[_c0:_c1].to(_rcF.dtype)
            _n = _c1 - _c0
            _ar = torch.arange(_n, device=flow.device) * 64
            _if = _ar + self.rc_from[_c0:_c1]                       # flat [C] into [C*64]
            _it = _ar + self.rc_to[_c0:_c1]
            _t1 = torch.matmul(_Rc, _H1).reshape(-1, _n * 64)[:, _if]    # [B, C]
            _t2 = torch.matmul(_Rc, _H2).reshape(-1, _n * 64)[:, _it]
            _t3 = torch.matmul(_R2c, _H3).reshape(-1, _n * 64)[:, _if]
            _parts.append(_t1 + _t2 + _t3)
          _rc_add = _rc_add + torch.cat(_parts, dim=1)
      else:
        _Fm = _rcF[:, self.rc_from]                                 # [B, 1858, dh]
        _Tm = _rcT[:, self.rc_to]
        _rc_add = (_Fm * _Tm).sum(-1) * (self.rc_dh ** -0.5) + self.rc_btype
        if self.ray_context_mode >= 2:
          _r = torch.matmul(self.rc_R.to(_rcF.dtype), self.rc_WR(flow))   # [B, 1858, dh]
          _d = torch.matmul(self.rc_R2.to(_rcF.dtype), self.rc_WD(flow))
          _rc_add = _rc_add + (_Fm * _r * self.rc_u).sum(-1) \
                            + (_Tm * _r * self.rc_v).sum(-1) \
                            + (_Fm * _d * self.rc_w).sum(-1)
      policy_out = policy_out + _rc_add
    value_out = self.value_head(fS_value, _v_inject)
    value2_out = self.value2_head(torch.cat((fS_value, qblunders_negative_positive), -1), _v2_inject) if self.value2_loss_weight > 0 else value_out
    unc_out = self.unc_head(fS_value if self.value_priv_replace else fS_others)
    unc_policy_out = self.unc_policy(fS_others) if self.uncertainty_policy_weight > 0 else unc_out # unc_out is just a dummy so not None

    action_out             = self.action_head(fS_others).reshape(-1, 1858, 3) if self.action_loss_weight > 0 else unc_out
    action_uncertainty_out = self.action_uncertainty_head(fS_others) if self.action_uncertainty_loss_weight > 0 else unc_out
    state_out              = self.state_head(fS_others) if self.prior_state_dim > 0 else unc_out
    moves_left_out         = self.mlh_head(fS_others) if self.moves_left_loss_weight > 0 else unc_out
    q_deviation_lower_out = self.qdev_lower(fS_others) if self.q_deviation_loss_weight > 0 else unc_out
    q_deviation_upper_out = self.qdev_upper(fS_others) if self.q_deviation_loss_weight > 0 else unc_out

    ret = policy_out, value_out, moves_left_out, unc_out, value2_out, q_deviation_lower_out, q_deviation_upper_out, unc_policy_out, action_out, state_out, action_uncertainty_out

    return ret


  def compute_loss(self, loss_calc : LossCalculator, batch, policy_out, value_out, moves_left_out, unc_out,
                    value2_out, q_deviation_lower_out, q_deviation_upper_out, uncertainty_policy_out,
                    prior_value_out, prior_value2_out,
                    action_target, action_out, action_uncertainty_out,
                    multiplier_action_loss,
                    num_pos, last_lr, log_stats):

    # If we are logging statistics, optionally make two passes, the first of which
    # calculates and logs individual per-head gradient norms ("GRADNORM: <head> , raw , weighted"
    # lines). Controlled by CERES_LOG_GRAD_NORMS_EVERY = N: run the diagnostic on every Nth
    # stats interval (0/unset = off). N.B. only works with non-compiled models on a single GPU
    # (backward-per-head desyncs DDP allreduce; compiled autograd rejects the retained graph),
    # and each pass costs ~one extra backward per head — meant for short measurement runs.
    if not hasattr(self, '_gradnorm_log_every'):
      self._gradnorm_log_every = int(os.environ.get('CERES_LOG_GRAD_NORMS_EVERY', '0') or 0)
      self._gradnorm_log_count = 0
      if self._gradnorm_log_every > 0:
        print(f'[ceres_net] per-head gradient-norm logging every {self._gradnorm_log_every} stats intervals '
              f'(requires single GPU + PyTorchCompileMode off)')
    LOG_PER_LOSS_GRADIENT_NORMS = False
    if self._gradnorm_log_every > 0 and log_stats:
      self._gradnorm_log_count += 1
      if self._gradnorm_log_count % self._gradnorm_log_every == 0:
        LOG_PER_LOSS_GRADIENT_NORMS = True
    if LOG_PER_LOSS_GRADIENT_NORMS and log_stats:
      self.compute_loss_or_gradnorm(loss_calc, batch, policy_out, value_out, moves_left_out, unc_out,
                                    value2_out, q_deviation_lower_out, q_deviation_upper_out, uncertainty_policy_out,
                                    prior_value_out, prior_value2_out,
                                    action_target, action_out, action_uncertainty_out,
                                    multiplier_action_loss,
                                    num_pos, last_lr, log_stats, gradient_norm_logging_mode = True)
       
    return self.compute_loss_or_gradnorm (loss_calc, batch, policy_out, value_out, moves_left_out, unc_out,
                                          value2_out, q_deviation_lower_out, q_deviation_upper_out, uncertainty_policy_out,
                                          prior_value_out, prior_value2_out,
                                          action_target, action_out, action_uncertainty_out,
                                          multiplier_action_loss,
                                          num_pos, last_lr, log_stats, gradient_norm_logging_mode = False)


  def compute_loss_or_gradnorm(self, loss_calc : LossCalculator, batch, policy_out, value_out, moves_left_out, unc_out,
                               value2_out, q_deviation_lower_out, q_deviation_upper_out, uncertainty_policy_out,
                               prior_value_out, prior_value2_out,
                               action_target, action_out, action_uncertainty_out,
                               multiplier_action_loss,
                               num_pos, last_lr, log_stats, gradient_norm_logging_mode):
    policy_target = batch['policies']
    wdl_deblundered = batch['wdl_deblundered']
    wdl_q = batch['wdl_q']
    moves_left_target = batch['mlh']
    unc_target = batch['unc']
    wdl_nondeblundered = batch['wdl_nondeblundered']
    uncertainty_policy_target = batch['uncertainty_policy']
    q_deviation_lower_target = batch['q_deviation_lower']
    q_deviation_upper_target = batch['q_deviation_upper']
    
    #	Subtract entropy from cross entropy to insulate loss magnitude 
    #	from distributional shift and make the loss more interpretable 
    #	because it takes out the portion that is irreducible.
    SUBTRACT_ENTROPY = True

   
    # Note that the loss weights are passed into the loss calculation functions in loss_calc module.
    # But they are only used for informational purposes and NOT applied to the losses applied by these functions.
    # Instead, the loss weights are only applied in the weighted average calculation in the assignment to total_loss.
    # Therefore the values logged (e.g. to Tensorboard) are the raw (unweighted) losses 
    # which are invariant to the particular weights in use (to facilitate comparison across different training runs).

    # Value2 target = pure nondeblundered z (game result), per production dual-value recipe
    # (value1 = search-Q via q_ratio=FractionQ, value2 = raw outcome z). The softened
    # 0.70/0.15/0.15 blend below is the prior default, disabled for the prod recipe.
    #wdl_blend = (wdl_nondeblundered * 0.70 + wdl_deblundered * 0.15 + wdl_q * 0.15)
    wdl_blend = wdl_nondeblundered
    value_target = wdl_q * self.q_ratio + wdl_deblundered * (1 - self.q_ratio)

    # z-provenance (v7x sidecar) for optional per-record value-loss weighting
    # (CERES_VALUE_PROV_WEIGHTS); None when the batch carries no v7x keys.
    z_provenance = batch.get('z_provenance', None)

    p_loss = 0 if policy_out is None else loss_calc.policy_loss(policy_target, policy_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.policy_loss_weight)
    v_loss = 0 if value_out is None else loss_calc.value_loss(value_target, value_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.value_loss_weight, provenance=z_provenance)

    # Value RANK loss (see __init__): calibration-free in-batch ordering of
    # E[V] = p(w) - p(l) against the target margin. Only pairs whose targets
    # genuinely differ (|dt| > 0.2) participate; hinge margin 0.1.
    value_rank_loss = 0
    if self.value_rank_weight > 0 and value_out is not None and not gradient_norm_logging_mode:
      _vr_p = torch.softmax(value_out.float(), dim=-1)
      _vr_s = _vr_p[:, 0] - _vr_p[:, 2]
      _vr_t = (value_target[:, 0] - value_target[:, 2]).float()
      _vr_dt = _vr_t.unsqueeze(0) - _vr_t.unsqueeze(1)                       # [B, B]
      _vr_ds = _vr_s.unsqueeze(0) - _vr_s.unsqueeze(1)
      _vr_m = (_vr_dt.abs() > 0.2).float()
      value_rank_loss = (torch.relu(0.1 - _vr_ds * torch.sign(_vr_dt)) * _vr_m).sum() \
          / _vr_m.sum().clamp(min=1.0)

    # Value CONTRAST aux (see __init__): per-move WDL CE — solution move keeps
    # the record's WDL, every other legal move is labeled LOSS-for-STM.
    # The loss-vector label (not a flip of the record WDL) is correct across
    # all three record classes in solution-line-expanded puzzle data: winning
    # side deviates -> ~loss (sharp positions); drawish DEFENSE puzzles (~5%,
    # user obs.) -> failing the only defense loses (a flip of a draw would
    # wrongly stay a draw); defender-side records in the line -> every move
    # still loses (a flip would wrongly label them WINS). Non-solution moves
    # are the heuristic side of the label, so they get 1/4 weight.
    vc_loss = 0
    _vc = getattr(self, '_last_vc_out', None)
    if _vc is not None and not gradient_norm_logging_mode:
      self._last_vc_out = None
      _vc_legal = policy_target > 0                                          # [B, 1858]
      _vc_sol = policy_target.argmax(dim=-1)                                 # [B]
      _vc_lab = torch.zeros_like(value_target).unsqueeze(1).expand(-1, 1858, -1).clone()
      _vc_lab[:, :, 2] = 1.0                                                 # loss-for-STM everywhere...
      _vc_ar = torch.arange(_vc_lab.shape[0], device=_vc_lab.device)
      _vc_lab[_vc_ar, _vc_sol] = value_target                                # ...except the solution
      _vc_ce = -(_vc_lab * torch.log_softmax(_vc.float(), dim=-1)).sum(-1)   # [B, 1858]
      _vc_w = _vc_legal.float() * 0.25
      _vc_w[_vc_ar, _vc_sol] = 1.0
      vc_loss = (_vc_ce * _vc_w).sum() / _vc_w.sum().clamp(min=1.0)
    # HL-Gauss categorical value loss (see __init__): KL between the Gaussian
    # histogram of the scalar q-target and the bucket softmax. CE minus target
    # entropy, per the file convention, so the logged number is a true KL.
    hlg_loss = 0
    _hlg = getattr(self, '_last_hlg_out', None)
    if _hlg is not None and not gradient_norm_logging_mode:
      self._last_hlg_out = None
      _hq = (value_target[:, 0] - value_target[:, 2]).float().clamp(-1.0, 1.0)
      _hz = (self.hlg_edges.unsqueeze(0) - _hq.unsqueeze(1)) / self.hlg_sigma      # [B, N+1]
      _hcdf = torch.special.ndtr(_hz)
      _hp = _hcdf[:, 1:] - _hcdf[:, :-1]
      _hp = _hp / _hp.sum(dim=-1, keepdim=True).clamp_min(1e-9)                    # renormalize edge truncation
      _hlp = torch.log_softmax(_hlg.float(), dim=-1)
      _hpc = _hp.clamp_min(1e-9)
      hlg_loss = (-(_hp * _hlp).sum(-1) + (_hpc * _hpc.log()).sum(-1)).mean()

    # Optimistic-policy aux loss (see __init__): masked per-sample CE on the
    # aux head, weighted toward positions where value1 UNDERESTIMATES the
    # target in units of the unc head's predicted error. Weight-normalized so
    # the logged number stays a per-effective-sample CE.
    opt_loss = 0
    _opt = getattr(self, '_last_opt_out', None)
    if _opt is not None and not gradient_norm_logging_mode:
      self._last_opt_out = None
      if value_out is not None and unc_out is not None and policy_out is not None:
        with torch.no_grad():
          _op_p = torch.softmax(value_out.float(), dim=-1)
          _op_qpred = _op_p[:, 0] - _op_p[:, 2]
          _op_tq = (value_target[:, 0] - value_target[:, 2]).float()
          if getattr(self, 'unc_self_error', False):
            _op_sigma = unc_out.float().abs().reshape(-1).sqrt() + 1e-5   # unc predicts sigma^2
          else:
            _op_sigma = unc_out.float().abs().reshape(-1) + 1e-5
          _op_z = (_op_tq - _op_qpred) / _op_sigma
          _op_w = torch.sigmoid((_op_z - self.opt_strength) * self.opt_alpha)
        _op_legal = policy_target > 0
        _op_masked = torch.where(_op_legal, _opt.float(), torch.full_like(_opt.float(), -1e4))
        _op_lp = torch.log_softmax(_op_masked, dim=-1)
        _op_ce = -(policy_target.float() * _op_lp).sum(-1)
        # lc0 semantics: plain mean of w*CE — SELF-GATING (mean weight ~0.003
        # early, so the term is nearly silent until genuinely many positions
        # are undervalued). The earlier weight-NORMALIZED mean turned this
        # into a full-CE-scale term from step one (~3.0 weighted mass, 2x the
        # policy loss) and destabilized the s7 preflight at peak LR.
        opt_loss = (_op_w * _op_ce).mean()

    v2_loss = 0 if value2_out is None else loss_calc.value2_loss(wdl_blend, value2_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.value2_loss_weight, provenance=z_provenance)
    ml_loss = 0 if moves_left_out is None else loss_calc.moves_left_loss(moves_left_target, moves_left_out, gradient_norm_logging_mode, self.moves_left_loss_weight)
    # UNC SELF-ERROR mode (config UncSelfError; see config.py): train unc toward
    # the STUDENT's own realized squared value error (lc0 ValueErrorLoss
    # semantics) instead of the teacher-time |DeltaQVersusV| data field.
    if getattr(self, 'unc_self_error', False) and unc_out is not None and value_out is not None:
      with torch.no_grad():
        _ue_p = torch.softmax(value_out.float(), dim=-1)
        _ue_qp = _ue_p[:, 0] - _ue_p[:, 2]
        _ue_qt = (value_target[:, 0] - value_target[:, 2]).float()
        unc_target = ((_ue_qp - _ue_qt) ** 2).unsqueeze(-1)
    u_loss = 0 if unc_out is None else loss_calc.unc_loss(unc_target, unc_out, gradient_norm_logging_mode, self.unc_loss_weight)
    q_deviation_lower_loss = 0 if q_deviation_lower_out is None else loss_calc.q_deviation_lower_loss(q_deviation_lower_target, q_deviation_lower_out, gradient_norm_logging_mode, self.q_deviation_loss_weight)
    q_deviation_upper_loss = 0 if q_deviation_upper_out is None else loss_calc.q_deviation_upper_loss(q_deviation_upper_target, q_deviation_upper_out, gradient_norm_logging_mode, self.q_deviation_loss_weight)


    if self.config.NetDef_TrainOn4BoardSequences:
      # TO DO: probably the multiplier_action_loss should somehow be propagated into the gradient norms when these are calculated
      action_loss = multiplier_action_loss * loss_calc.action_loss(action_target, action_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.action_loss_weight)
      action_uncertainty_loss = multiplier_action_loss * self.action_uncertainty_loss_weight * loss_calc.action_unc_loss(torch.abs(action_target - action_out), action_uncertainty_out, gradient_norm_logging_mode, self.action_uncertainty_loss_weight)
      # We have two value scores and want them to be consistent modulo inversion (prior_board and this_board).
      # The value of this board is taken to be "more definitive" so it is the target (however this assumes policy was correct....)
      value_diff_loss = 0 if self.value_diff_loss_weight == 0 or prior_value_out == None else loss_calc.value_diff_loss(value_out, prior_value_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.value_diff_loss_weight)
      value2_diff_loss = 0 if self.value2_diff_loss_weight == 0 or prior_value2_out == None else loss_calc.value2_diff_loss(value2_out, prior_value2_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.value2_diff_loss_weight)
    else:
      action_loss = 0
      action_uncertainty_loss = 0
      value_diff_loss = 0
      value2_diff_loss = 0

    uncertainty_policy_loss = 0 if uncertainty_policy_out is None else loss_calc.uncertainty_policy_loss(uncertainty_policy_target, uncertainty_policy_out, gradient_norm_logging_mode, self.uncertainty_policy_weight)

    # Placement value head (aux): same CE-minus-entropy as value_loss, against the same
    # value_target, via LossCalculator (comparable magnitude + LAST_* interval averaging
    # + correct behavior in gradient_norm_logging_mode). Consumes the output stashed by
    # forward (never part of the export signature). Consume-and-clear so a stale stash
    # can never be re-used by a later loss call whose forward opted out (e.g. the 4-board
    # path's action-only board 4, which passes value_out=None).
    placement_loss = 0
    # vda mode-4 auxiliary value loss (training-only deep-supervision path).
    # Same CE-minus-entropy form as value_loss against the same target, weighted
    # by CERES_VDA_AUX_WEIGHT; consume-and-clear stash pattern. Not consumed on
    # the grad-norm diagnostic pass (it contributes no GRADNORM line, and this
    # keeps the loss intact for the real pass that follows).
    vda_aux_loss = 0
    _vda_aux = getattr(self, '_last_vda_aux_out', None)
    if _vda_aux is not None and value_out is not None and not gradient_norm_logging_mode:
      self._last_vda_aux_out = None
      _aux_entropy = loss_calc.entropy(value_target) if SUBTRACT_ENTROPY else 0.0
      vda_aux_loss = loss_calc.ce_loss.forward(_vda_aux.float(), value_target) - _aux_entropy

    # Depth probes (see __init__): mean-over-depths CE for the shared probes
    # (weighted into total) + ctl-head losses (detached input -> gradients reach
    # ONLY the ctl heads; added unweighted since they cannot touch the trunk).
    # Same consume-and-clear + skip-on-gradnorm-pass conventions as vda_aux.
    depth_probe_ploss = 0; depth_probe_vloss = 0; depth_ctl_ploss = 0; depth_ctl_vloss = 0
    _dpp = getattr(self, '_last_depth_probe_policy', None)
    if _dpp is not None and policy_out is not None and value_out is not None and not gradient_norm_logging_mode:
      self._last_depth_probe_policy = None
      _dpv = self._last_depth_probe_value; self._last_depth_probe_value = None
      _dcp = self._last_depth_ctl_policy; self._last_depth_ctl_policy = None
      _dcv = self._last_depth_ctl_value; self._last_depth_ctl_value = None
      _L1 = _dpp.shape[1]
      _legal = policy_target.greater(0)
      _p_ent = loss_calc.entropy(policy_target) if SUBTRACT_ENTROPY else 0.0
      _v_ent = loss_calc.entropy(value_target) if SUBTRACT_ENTROPY else 0.0
      _dpp_m = torch.where(_legal.unsqueeze(1), _dpp, torch.full_like(_dpp, loss_calc.MASK_POLICY_VALUE))
      _pt_rep = policy_target.unsqueeze(1).expand(-1, _L1, -1).reshape(-1, policy_target.shape[-1])
      depth_probe_ploss = loss_calc.ce_loss.forward(_dpp_m.reshape(-1, _dpp.shape[-1]).float(), _pt_rep) - _p_ent
      _vt_rep = value_target.unsqueeze(1).expand(-1, _L1, -1).reshape(-1, 3)
      depth_probe_vloss = loss_calc.ce_loss.forward(_dpv.reshape(-1, 3).float(), _vt_rep) - _v_ent
      _dcp_m = torch.where(_legal, _dcp, torch.full_like(_dcp, loss_calc.MASK_POLICY_VALUE))
      depth_ctl_ploss = loss_calc.ce_loss.forward(_dcp_m.float(), policy_target) - _p_ent
      depth_ctl_vloss = loss_calc.ce_loss.forward(_dcv.float(), value_target) - _v_ent
      if log_stats:
        # Per-depth curves — the core diagnostic: WHERE in the trunk does
        # policy/value information become linearly decodable, and how does the
        # profile move over training.
        with torch.no_grad():
          for _d in range(_L1):
            self._log(f"depth_probe_policy_d{_d:02d}",
                      loss_calc.ce_loss.forward(_dpp_m[:, _d].float(), policy_target) - _p_ent, step=num_pos)
            self._log(f"depth_probe_value_d{_d:02d}",
                      loss_calc.ce_loss.forward(_dpv[:, _d].float(), value_target) - _v_ent, step=num_pos)

    # Soft-policy aux CE (see __init__): target = p^(1/T) over legal moves,
    # renormalized. Same legality masking as the main policy loss; consume-and-
    # clear; skipped on the gradnorm pass.
    soft_policy_loss = 0
    _sp = getattr(self, '_last_sp_out', None)
    if _sp is not None and not gradient_norm_logging_mode:
      self._last_sp_out = None
      _sp_legal = policy_target > 0
      _sp_t = torch.where(_sp_legal, policy_target.float(), torch.zeros_like(policy_target, dtype=torch.float32))
      _sp_t = _sp_t.pow(1.0 / self.soft_policy_temp)
      _sp_t = _sp_t / _sp_t.sum(dim=-1, keepdim=True).clamp(min=1e-9)
      _sp_m = torch.where(_sp_legal, _sp, torch.full_like(_sp, loss_calc.MASK_POLICY_VALUE))
      soft_policy_loss = loss_calc.ce_loss.forward(_sp_m.float(), _sp_t) \
          - (loss_calc.entropy(_sp_t) if SUBTRACT_ENTROPY else 0.0)

    _pv_out = getattr(self, '_last_placement_value_out', None)
    if self.placement_value_weight > 0 and _pv_out is not None and value_out is not None:
      # Consume-and-clear only on the REAL pass: the grad-norm diagnostic pass
      # must not eat the stash the real pass needs (pre-existing bug, fixed
      # 2026-08-07 review round — same guard as the hlg/sp/vc consumers).
      if not gradient_norm_logging_mode:
        self._last_placement_value_out = None
      placement_loss = loss_calc.placement_value_loss(value_target, _pv_out.float(), SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.placement_value_weight)

    # K-ply survival aux head: per-square fate CE against sidecar targets (empty squares
    # masked inside loss_calc.survival_loss). Same consume-and-clear/value_out gating
    # as the placement head.
    survival_loss = 0
    _sv_out = getattr(self, '_last_survival_out', None)
    if self.survival_target_weight > 0 and _sv_out is not None and value_out is not None:
      if not gradient_norm_logging_mode:
        self._last_survival_out = None
      survival_target = batch.get('survival', None)
      # No 'survival' key = batch from a sidecar-less shard (CERES_TPG_TARGET_SIDECAR=auto
      # mixed-corpus mode): skip the loss for this batch. train.py validates at startup
      # that at least one dataset actually carries sidecars, so this cannot be a silent
      # full-run no-op misconfiguration.
      if survival_target is not None:
        survival_loss = loss_calc.survival_loss(survival_target, _sv_out, gradient_norm_logging_mode, self.survival_target_weight)

    # Short-term value aux head: CE against the WDL built from censored q_st/d_st
    # (V7-extras sidecar; STM-relative, matching TPG conventions), optionally weighted
    # per record by z-provenance. Missing keys = batch from a v7x-less shard
    # (CERES_TPG_V7X_SIDECAR=auto mixed-corpus mode): skip, as with survival.
    # Same consume-and-clear/value_out gating as the other aux heads.
    stvalue_loss = 0
    _st_out = getattr(self, '_last_stvalue_out', None)
    if self.stvalue_weight > 0 and _st_out is not None and value_out is not None:
      if not gradient_norm_logging_mode:
        self._last_stvalue_out = None
      _cens_q = batch.get('censored_q_st', None)
      if _cens_q is not None:
        stvalue_loss = loss_calc.stvalue_loss(_cens_q, batch['censored_d_st'], batch.get('z_provenance', None),
                                              _st_out.float(), SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.stvalue_weight)

    total_loss = (self.policy_loss_weight * p_loss
        + self.value_loss_weight * v_loss
        + self.value2_loss_weight * v2_loss
        + self.moves_left_loss_weight * ml_loss
        + self.unc_loss_weight * u_loss
        + self.q_deviation_loss_weight * q_deviation_lower_loss
        + self.q_deviation_loss_weight * q_deviation_upper_loss
        + self.value_diff_loss_weight * value_diff_loss
        + self.value2_diff_loss_weight * value2_diff_loss
        + self.action_loss_weight * action_loss
        + self.action_uncertainty_loss_weight * action_uncertainty_loss
        + self.uncertainty_policy_weight * uncertainty_policy_loss
        + self.placement_value_weight * placement_loss
        + self.survival_target_weight * survival_loss
        + self.stvalue_weight * stvalue_loss
        + (self.vda_aux_weight * vda_aux_loss if self.vda_mode == 4 else 0)
        + ((self.depth_probe_policy_weight * depth_probe_ploss
            + self.depth_probe_value_weight * depth_probe_vloss
            + depth_ctl_ploss + depth_ctl_vloss)
           if not isinstance(depth_probe_ploss, int) else 0)
        + (self.value_rank_weight * value_rank_loss if not isinstance(value_rank_loss, int) else 0)
        + (self.value_contrast_weight * vc_loss if not isinstance(vc_loss, int) else 0)
        + (self.hlg_weight * hlg_loss if not isinstance(hlg_loss, int) else 0)
        + (self.opt_policy_weight * opt_loss if not isinstance(opt_loss, int) else 0)
        + (self.soft_policy_weight * soft_policy_loss if not isinstance(soft_policy_loss, int) else 0))

    # POLICY/VALUE GRADIENT-CONFLICT PROBE (config GradConflictProbeSteps; the
    # measurement lives in train.py). When armed for this step, stash the two family
    # subtotals — WEIGHTED, i.e. as they actually enter total_loss — so the train loop
    # can differentiate them separately and measure the angle between the two gradients
    # on the parameters they SHARE. Nothing here alters total_loss; the subtotals are
    # additional views of terms already summed above.
    #
    # Family assignment follows what each head is predicting, not which tensor it reads:
    #   policy  <- policy, soft-policy, optimistic-policy, policy-uncertainty, action(+unc)
    #   value   <- value1/value2, unc, q-deviation, value-diff, stvalue, placement,
    #              vda aux, value-rank, value-contrast, HL-Gauss
    # Deliberately in NEITHER: moves-left and survival (predict neither W/D/L nor a move
    # distribution, so they would blur the very angle being measured), the gate/TSB
    # sparsity regularizers and mirror-consistency (added later, in train.py), and the
    # depth probes (deep supervision spans both families by construction).
    if getattr(self, '_gc_probe_now', False):
      self._gc_policy_loss = (self.policy_loss_weight * p_loss
          + self.uncertainty_policy_weight * uncertainty_policy_loss
          + self.action_loss_weight * action_loss
          + self.action_uncertainty_loss_weight * action_uncertainty_loss
          + (self.opt_policy_weight * opt_loss if not isinstance(opt_loss, int) else 0)
          + (self.soft_policy_weight * soft_policy_loss if not isinstance(soft_policy_loss, int) else 0))
      self._gc_value_loss = (self.value_loss_weight * v_loss
          + self.value2_loss_weight * v2_loss
          + self.unc_loss_weight * u_loss
          + self.q_deviation_loss_weight * q_deviation_lower_loss
          + self.q_deviation_loss_weight * q_deviation_upper_loss
          + self.value_diff_loss_weight * value_diff_loss
          + self.value2_diff_loss_weight * value2_diff_loss
          + self.stvalue_weight * stvalue_loss
          + self.placement_value_weight * placement_loss
          + (self.vda_aux_weight * vda_aux_loss if self.vda_mode == 4 else 0)
          + (self.value_rank_weight * value_rank_loss if not isinstance(value_rank_loss, int) else 0)
          + (self.value_contrast_weight * vc_loss if not isinstance(vc_loss, int) else 0)
          + (self.hlg_weight * hlg_loss if not isinstance(hlg_loss, int) else 0))

    if (log_stats):
      if not gradient_norm_logging_mode:
        stat_suffix = ""
        policy_accuracy = 0 if policy_out is None else loss_calc.calc_accuracy(policy_target, policy_out, True)
        value_accuracy = 0 if value_out is None else loss_calc.calc_accuracy(value_target, value_out, False)

        # Cheap logging-cadence diagnostics (computed only every ~10s, zero
        # steady-state training cost):
        # - policy_entropy: mean softmax entropy over LEGAL moves — the
        #   late-phase policy sprint that eats value is literally this curve
        #   falling (over-peaking watch; lc0 logs the same).
        # - value1_value2_kl: disagreement between the Q-trained and z-trained
        #   value heads — rising divergence signals target tension.
        with torch.no_grad():
          if policy_out is not None:
            _pe_legal = policy_target > 0
            _pe_masked = torch.where(_pe_legal, policy_out.float(), torch.full_like(policy_out.float(), -1e4))
            _pe_lp = torch.log_softmax(_pe_masked, dim=-1)
            _pe = -(_pe_lp.exp() * _pe_lp).sum(-1).mean()
            self._log("policy_entropy", _pe, step=num_pos)
          if value_out is not None and value2_out is not None and self.value2_loss_weight > 0:
            _v1_lp = torch.log_softmax(value_out.float(), dim=-1)
            _v2_lp = torch.log_softmax(value2_out.float(), dim=-1)
            _v12 = (_v1_lp.exp() * (_v1_lp - _v2_lp)).sum(-1).mean()
            self._log("value1_value2_kl", _v12, step=num_pos)
        self._log("pos_mm", num_pos // 1000000., step=num_pos)
        self._log("LR", last_lr, step=num_pos)
        self._log("total_loss", total_loss, step=num_pos)

        # Visibility edge bias: per-family projection-weight RMS. The source
        # program's structure-first readout — channel-weight RMS reproduces
        # 10-300x better across runs than short-horizon losses, so this is the
        # earliest reliable signal that a family is being used.
        if self.use_vis_edge_bias:
          with torch.no_grad():
            for _fam, _sl in self.vis_channels_module.family_slices.items():
              _w = torch.stack([_lin.weight[:, _sl] for _lin in self.vis_edge_proj])
              self._log("vis_w_rms_" + _fam, _w.pow(2).mean().sqrt(), step=num_pos)
            if self.vis_edge_gate_mode:
              # B/C gate energy per family (gate params live in each layer's
              # attention; channel dim is index 1 of [H, C, d]).
              for _gname, _tag in (('attack_gate_q', 'gateq'), ('attack_gate_k', 'gatek')):
                _gs = [getattr(_l.attention, _gname) for _l in self.transformer_layer]
                if _gs[0] is None:
                  continue
                for _fam, _sl in self.vis_channels_module.family_slices.items():
                  _w = torch.stack([_g[:, _sl, :] for _g in _gs])
                  self._log(f"vis_{_tag}_rms_{_fam}", _w.pow(2).mean().sqrt(), step=num_pos)

        # Log GPU (CUDA) statistics
        if torch.cuda.is_available():
          for gpu_num in range(torch.cuda.device_count()):
            # Note: we enumerate all devices on the host instead of only those used 
            #       for this training run (self.config.Exec_DeviceIDs) for two reasons:
            #       1. The torch.cuda numbering scheme (highest performance at lowest index)
            #          is potentially different from the what the application sees
            #          (unless the enviroment variable is set: CUDA_DEVICE_ORDER=PCI_BUS_ID).
            #       2. Potentially (for power and thermal reasons) it is useful to monitor all devices
            #          even if not used by this training run.                   
            try:
              self._log("gpu_temp_"+str(gpu_num), torch.cuda.temperature(gpu_num), step=num_pos)
              self._log("gpu_power_draw_"+str(gpu_num), torch.cuda.power_draw(gpu_num)/1000, step=num_pos)
              self._log("gpu_utilization_"+str(gpu_num), torch.cuda.utilization(gpu_num), step=num_pos)
              #self._log("gpu_memory_used_"+str(gpu_num), torch.cuda.memory_usage(gpu_num), step=num_pos)
              #self._log("gpu_clock_rate_"+str(gpu_num), torch.cuda.clock_rate(gpu_num), step=num_pos)
            except:
              pass # requires pynvml, may fail e.g. on Windows    
      else:
        stat_suffix = "_gnorm"

      if not gradient_norm_logging_mode:
        self._log("policy_acc" + stat_suffix,policy_accuracy,  step=num_pos)
        self._log("value_acc" + stat_suffix,value_accuracy,  step=num_pos)

      self._log("policy_loss" + stat_suffix, p_loss,  step=num_pos)
      self._log("value_loss" + stat_suffix, v_loss,  step=num_pos)
      self._log("value2_loss" + stat_suffix, v2_loss,  step=num_pos)
      if self.placement_value_weight > 0 and not isinstance(placement_loss, int):
        self._log("placement_value_loss" + stat_suffix, placement_loss, step=num_pos)
      if self.survival_target_weight > 0 and not isinstance(survival_loss, int):
        self._log("survival_loss" + stat_suffix, survival_loss, step=num_pos)
      if self.stvalue_weight > 0 and not isinstance(stvalue_loss, int):
        self._log("stvalue_loss" + stat_suffix, stvalue_loss, step=num_pos)
      if self.vda_mode == 4 and not isinstance(vda_aux_loss, int):
        self._log("vda_aux_value_loss" + stat_suffix, vda_aux_loss, step=num_pos)
      if not isinstance(depth_probe_ploss, int):
        self._log("depth_probe_policy_loss" + stat_suffix, depth_probe_ploss, step=num_pos)
        self._log("depth_probe_value_loss" + stat_suffix, depth_probe_vloss, step=num_pos)
        self._log("depth_ctl_policy_loss" + stat_suffix, depth_ctl_ploss, step=num_pos)
        self._log("depth_ctl_value_loss" + stat_suffix, depth_ctl_vloss, step=num_pos)
      if not isinstance(value_rank_loss, int):
        self._log("value_rank_loss" + stat_suffix, value_rank_loss, step=num_pos)
      if not isinstance(vc_loss, int):
        self._log("value_contrast_loss" + stat_suffix, vc_loss, step=num_pos)
      if not isinstance(hlg_loss, int):
        self._log("hlgauss_value_loss" + stat_suffix, hlg_loss, step=num_pos)
      if not isinstance(opt_loss, int):
        self._log("optimistic_policy_loss" + stat_suffix, opt_loss, step=num_pos)
      if not isinstance(soft_policy_loss, int):
        self._log("soft_policy_loss" + stat_suffix, soft_policy_loss, step=num_pos)
      if not gradient_norm_logging_mode:
        # Depth-attending value head diagnostics (see forward): WHICH depths does the
        # value head read? Logs batch-mean attention weight per depth state
        # (vda_alpha_d00 = post-embedding ... d<L> = final layer), plus per-sample
        # entropy (uniform over 11 states = ln 11 ~ 2.398 nats; falling entropy =
        # sharpening depth preference) and the expected depth index. Consume-and-clear,
        # same stash pattern as the placement head.
        _vda_a = getattr(self, '_last_vda_alpha', None)
        if self.use_value_depth_attention and _vda_a is not None:
          self._last_vda_alpha = None
          _a = _vda_a.float().squeeze(-1)                    # [B, L+1]
          _a_mean = _a.mean(dim=0)                           # [L+1]
          for _d in range(_a_mean.shape[0]):
            self._log(f"vda_alpha_d{_d:02d}", _a_mean[_d], step=num_pos)
          _ent = -(_a * (_a + 1e-9).log()).sum(dim=1).mean()
          _depth = (_a * torch.arange(_a.shape[1], device=_a.device, dtype=_a.dtype)).sum(dim=1).mean()
          self._log("vda_alpha_entropy", _ent, step=num_pos)
          self._log("vda_alpha_mean_depth", _depth, step=num_pos)
        # Mode 3: the pooled/global branch gets its own profile (vda_alpha_g_*).
        _vda_ag = getattr(self, '_last_vda_alpha_g', None)
        if _vda_ag is not None:
          self._last_vda_alpha_g = None
          _ag = _vda_ag.float().squeeze(-1)
          _ag_mean = _ag.mean(dim=0)
          for _d in range(_ag_mean.shape[0]):
            self._log(f"vda_alpha_g_d{_d:02d}", _ag_mean[_d], step=num_pos)
          self._log("vda_alpha_g_entropy", -(_ag * (_ag + 1e-9).log()).sum(dim=1).mean(), step=num_pos)
        # PDA: policy-side depth profile (pda_alpha_*), square-averaged like vda's.
        _pda_a = getattr(self, '_last_pda_alpha', None)
        if _pda_a is not None:
          self._last_pda_alpha = None
          _pa = _pda_a.float().squeeze(-1)                   # [B, L+1]
          _pa_mean = _pa.mean(dim=0)
          for _d in range(_pa_mean.shape[0]):
            self._log(f"pda_alpha_d{_d:02d}", _pa_mean[_d], step=num_pos)
          self._log("pda_alpha_entropy", -(_pa * (_pa + 1e-9).log()).sum(dim=1).mean(), step=num_pos)
          self._log("pda_alpha_mean_depth", (_pa * torch.arange(_pa.shape[1], device=_pa.device, dtype=_pa.dtype)).sum(dim=1).mean(), step=num_pos)
      self._log("moves_left_loss" + stat_suffix, ml_loss, step=num_pos)
      self._log("unc_loss" + stat_suffix, u_loss, step=num_pos)
      self._log("unc_policy_loss" + stat_suffix, uncertainty_policy_loss, step=num_pos)
      self._log("q_deviation_lower_loss" + stat_suffix, q_deviation_lower_loss, step=num_pos)
      self._log("q_deviation_upper_loss" + stat_suffix, q_deviation_upper_loss, step=num_pos)
      self._log("value_diff_loss" + stat_suffix, value_diff_loss, step=num_pos)
      self._log("value2_diff_loss" + stat_suffix, value2_diff_loss, step=num_pos)
      self._log("action_loss" + stat_suffix, action_loss, step=num_pos)
      self._log("action_uncertainty_loss" + stat_suffix, action_uncertainty_loss, step=num_pos)

    return total_loss


  
"""
Prepare static relative position (RPE) encoding map.
This RPE idea and initialization code taken from work of Daniel Monroe, see:
https://github.com/Ergodice/lczero-training/blob/a7271f25a1bd84e5e22bf924f7365cd003cb8d2f/tf/tfprocess.py
""" 
def make_rpe_map():
  # 15 * 15 in units for distance pairs to 64 * 64 pairs of squares
  # (rounded from 15 up to 16 to be a power of 2)
  out = torch.zeros((16*16, 64*64))
  for i in range(8):
    for j in range(8):
      for k in range(8):
        for l in range(8):
          out[15 * (i - k + 7) + (j - l + 7), 64 * (i * 8 + j) + k * 8 + l] = 1
  return out


