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

    # Played-move action training (2026-08-21): the v7 fields supply an exact
    # per-played-move WDL target (q_after_played == -next.best_q verified to
    # MAE 0.000; d from the next record) � the 4-board-sequence machinery is
    # not needed. The head itself is the SAME exported 'action' output the
    # Ceres TRT evaluator detects by name and MCTS consumes via
    # ActionWDLForMove; only the training signal differs.
    self.action_played_weight = float(getattr(config, 'Opt_LossActionPlayedMultiplier', 0) or 0)
    if self.action_played_weight > 0 and action_loss_weight > 0:
      # One head, two different target definitions (4-board sequence targets
      # vs v7 played-move targets) — action review finding 9.
      raise ValueError('LossActionPlayedMultiplier and LossActionMultiplier are mutually '
                       'exclusive (both would train action_head on conflicting targets)')
    if action_loss_weight > 0 or self.action_played_weight > 0:
      self.action_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858 * 3, _other_rd)
      if self.action_played_weight > 0:
        print(f'[ceres_net] ACTION head (played-move mode) enabled: w={self.action_played_weight} '
              f'(EXPORTED as output "action"; target = v7 q/d-after-played, invalid masked)')

    if action_uncertainty_loss_weight > 0:
      self.action_uncertainty_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858, _other_rd)

    # VALUE POOL CHANNELS (toolbox T1.2 "channels" variant, NetDef
    # ValueHeadPoolChannels): min/max extreme-square summaries of the trunk
    # concatenated as FIRST-CLASS INPUT COLUMNS to the value family's first
    # linear — default init, carrying signal from step 0 — instead of the
    # zero-init additive offer of ValueHeadMinMaxPool (which died on the cf3g
    # chassis, where the AND-heads already supply extreme info and the
    # optimizer must additionally DISCOVER a zero-init path). ARCH key: widens
    # value_head.fc/value2_head.fc, so warm starts across a toggle fail loudly
    # on the shape mismatch (no silent-drop guard needed).
    self.value_pool_channels = bool(getattr(config, 'NetDef_ValueHeadPoolChannels', False))
    assert not (self.value_pool_channels and getattr(config, 'NetDef_ValueHeadMinMaxPool', False)), \
        'ValueHeadPoolChannels and ValueHeadMinMaxPool are redundant together — pick one'
    _v_extra = 2 * self.EMBEDDING_DIM if self.value_pool_channels else 0
    if self.value_pool_channels:
      print(f'[ceres_net] VALUE POOL CHANNELS enabled: trunk min/max over squares '
            f'[B,{_v_extra}] concatenated into value1'
            f'{"/value2" if self.value2_loss_weight > 0 else ""} head input '
            f'(first linear widened by {_v_extra} columns, default init — active from step 0)')

    self.value_head = Head(self.Activation, self.VALUE_IN_SIZE + _v_extra, 64 * HEAD_MULT, 3, _val_rd)
    # unc follows the value family: on the private path it reads the private
    # value features (audit finding #4 — error-family heads should enrich the
    # value representation, not the shared bottleneck).
    self.unc_head = Head(self.Activation, self.VALUE_IN_SIZE, 32 * HEAD_MULT, 1, _other_rd)

    if self.value2_loss_weight > 0:
      self.value2_head = Head(self.Activation, 2 + self.VALUE_IN_SIZE + _v_extra, 64 * HEAD_MULT, 3, _other_rd)

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

    # VALUE MIN/MAX POOL SIDE-CHANNELS (2026-08 tactics toolbox T1.2, NetDef
    # ValueHeadMinMaxPool): worst-square/best-square summaries of the trunk
    # for the value family only. The value input is a 64:1 compressed
    # projection; extreme per-square facts (the one uncovered flight square,
    # the one hanging piece) die in that compression — the same AND/OR
    # quantifier gap the soft-min heads target, but at the head instead of in
    # attention. min/max over squares of the trunk flow [B,64,D] -> [B,2D],
    # fed through zero-init bias-free Linears into the value1/value2 hidden
    # pre-activation (the ValueHeadChannelsMode='inject' pattern): exact
    # step-0 no-op, composes additively with the private front-end inject.
    self.value_minmax_pool = bool(getattr(config, 'NetDef_ValueHeadMinMaxPool', False))
    if self.value_minmax_pool:
      self.value_pool_inject = nn.Linear(2 * self.EMBEDDING_DIM, 64 * HEAD_MULT, bias=False)
      nn.init.zeros_(self.value_pool_inject.weight)
      _n_vp = 2 * self.EMBEDDING_DIM * 64 * HEAD_MULT
      if self.value2_loss_weight > 0:
        self.value2_pool_inject = nn.Linear(2 * self.EMBEDDING_DIM, 64 * HEAD_MULT, bias=False)
        nn.init.zeros_(self.value2_pool_inject.weight)
        _n_vp *= 2
      print(f'[ceres_net] VALUE MIN/MAX POOL enabled: trunk amin/amax over squares '
            f'[B,{2 * self.EMBEDDING_DIM}] -> value1'
            f'{"/value2" if self.value2_loss_weight > 0 else ""} hidden pre-activation '
            f'({_n_vp:,} new params, zero-init => exact step-0 no-op)')

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
    # Config-foerst (2026-08-31), env-fallback for legacy — eksplisitt config-0
    # overstyrer en eksportert env-var (MirrorConsWeight-moensteret).
    _pda_cfg = getattr(config, 'NetDef_PolicyDepthAttention', None)
    self.pda_mode = (int(_pda_cfg) if _pda_cfg is not None
                     else int(os.environ.get('CERES_POLICY_DEPTH_ATTENTION', '0') or 0))
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
    # POLICY HEAD FORM (2026-09-02, config PolicyHeadForm): 'fromto' makes the
    # from-to bilinear (Lc0/BT4-style attention policy head, ray-context mode 1
    # machinery) the PRIMARY policy head and bypasses the 1858-way MLP head.
    # Motivation: the dual-plane knockout showed the net moved its whole policy
    # function into the mover-bilinear decode and let the MLP head decay to a
    # fixed prior — it prefers the from-to form when offered one. rc_WT is then
    # given a normal init (it is the head, not an add-on). Promotions share the
    # from-to score and differ only through the per-move bias rc_btype.
    self.policy_head_form = str(os.environ.get('CERES_POLICY_HEAD_FORM', 'mlp') or 'mlp').lower()
    assert self.policy_head_form in ('mlp', 'fromto'), f'PolicyHeadForm: {self.policy_head_form!r}'
    if self.policy_head_form == 'fromto' and self.ray_context_mode == 0:
      self.ray_context_mode = 1
    if self.policy_head_form == 'fromto':
      print('[ceres_net] POLICY HEAD FORM = fromto: from-to bilinear IS the policy head '
            f'(ray-context mode {self.ray_context_mode}); the MLP policy head is bypassed')
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
      if self.policy_head_form != 'fromto':
        nn.init.zeros_(self.rc_WT.weight)     # add-on form: exact step-0 no-op
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
    # POLICY MARGIN loss (I4, 2026-08-25 probe-trio): hinge paa logit-gapet
    # mellom target-argmax og de 4 sterkeste rivalene. Motivasjon: E1 maalte at
    # ~52 % av alle puzzle-feil er naer-bommer (gap < 1.0) og at flathet
    # anti-korrelerer med top-1; E2 at policy-targets FLATER naer matt.
    # Konfidensvekting med targetens egen top-1-masse gjor at skarpingen kun
    # presser der laereren selv er sikker (one-hot puzzle-records faar full
    # vekt, flate naer-matt game-records nesten ingen) - vi skarper aldri mot
    # en tilfeldig argmax av et flatt maal.
    self.policy_margin_weight = float(os.environ.get('CERES_POLICY_MARGIN_WEIGHT', '0') or 0)
    self.policy_margin_value = float(os.environ.get('CERES_POLICY_MARGIN_VALUE', '1.0') or 1.0)
    if self.policy_margin_weight > 0:
      print(f'[ceres_net] POLICY MARGIN loss enabled: hinge m={self.policy_margin_value} mot top-4 rivaler, '
            f'konfidensvektet med target-top1, w={self.policy_margin_weight}')
    # PLACKETT-LUCE policy ranking aux (Kovax, 2026-09-02): listwise ListMLE
    # likelihood of the TARGET'S move ORDER over its top-K moves,
    #   -log P(pi) = sum_{k<=K} [ logsumexp_{j>=k} s_(j) - s_(k) ]   (s ordered by target rank,
    # suffix over ALL legal moves). Unlike CE it does not fit the target's
    # sharpness; it only pushes the net's ORDERING toward the teacher's, so it can
    # sit beside CE (which stays the anchor) without the confidently-wrong-
    # sharpening failure mode. Config: PolicyPLWeight (0 = off), PolicyPLTopK.
    self.policy_pl_weight = float(os.environ.get('CERES_POLICY_PL_WEIGHT', '0') or 0)
    self.policy_pl_topk = int(os.environ.get('CERES_POLICY_PL_TOPK', '3') or 3)
    if self.policy_pl_weight > 0:
      print(f'[ceres_net] POLICY PLACKETT-LUCE ranking loss enabled: top-{self.policy_pl_topk} '
            f'ListMLE over legal moves in target order, w={self.policy_pl_weight} (CE stays the anchor)')
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
    # Opponent-policy aux head (Monroe/LC0 idea, +5% there; unlocked 2026-08-21
    # by v7 records carrying OppPlayedIndex — 99% populated in both Kovax
    # corpora). Predicts the OPPONENT'S REPLY move (their stm-relative 1858
    # frame) from our position — forces threat/response modeling, aimed at the
    # measured intermezzo/calculation weakness. Training-only (never exported);
    # target 'opp_played_idx' arrives only from DirectFromV6 v7 sources, and
    # the loss masks -1 (no reply / non-v7 batches contribute nothing).
    self.opp_policy_weight = float(getattr(config, 'Opt_LossOppPolicyMultiplier', 0) or 0)
    if self.opp_policy_weight > 0:
      self.oppp_head = Head(self.Activation, self.HEAD_IN_SIZE, 128 * HEAD_MULT, 1858, 0)
      print(f'[ceres_net] OPPONENT-POLICY aux head enabled: w={self.opp_policy_weight} '
            f'(training-only; target = v7 OppPlayedIndex, -1 masked)')

    self.hlg_weight = float(getattr(config, 'Opt_HLGaussWeight', 0) or 0)
    if self.hlg_weight > 0:
      self.hlg_buckets = int(getattr(config, 'Opt_HLGaussBuckets', 32) or 32)
      # Runde-2-funn: 'or 0.75' sluket eksplisitt 0 (one-hot-bucket-kontrakten
      # i config.py) tilbake til 0.75 — None-sjekk i stedet for falsy-sjekk.
      _hlg_ss = getattr(config, 'Opt_HLGaussSigmaScale', None)
      _hlg_sigma_scale = 0.75 if _hlg_ss is None else float(_hlg_ss)
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
    
    # SMOLBASIS (Kovax-idé 2026-08-31): erstatt hele smolgen-generatoren
    # (~8,4M params paa 256x10 = ~40 %% av nettet!) med K statiske laerte
    # 64x64-basistabeller per hode + en bitteliten per-lags koeffisient-
    # generator (~1/26 params). Rangproben maalte effektiv rang ~31 paa vaar
    # trente smolgen — dette er den FUNKSJONELLE testen av om hoyrangs-
    # innholdet betyr noe. Bank init randn*0.05, koeff zero-init i DPA
    # (rel_gains-laerdommen: signalvei fra steg en, no-op ved step 0).
    # SMBSTATIC (trinn 2+3, 2026-08-31): Kovax' sterkeste form — REN statisk
    # tabell per hode, delt over alle lag, ingen conditioner.
    #   mode 1 = fri 64x64 per hode (~33k)   mode 2 = 15x15-relativ (~1.8k)
    # Zero-init (tabellen ER outputen — direkte gradient, ingen rel_gains-felle).
    self.smol_static_mode = int(getattr(config, 'NetDef_SmolgenStaticMode', 0) or 0)
    assert self.smol_static_mode in (0, 1, 2)
    if self.smol_static_mode == 1:
      self.smol_static_bank = nn.Parameter(torch.zeros(self.NUM_HEADS, NUM_TOKENS_NET * NUM_TOKENS_NET))
    elif self.smol_static_mode == 2:
      self.smol_static_bank = nn.Parameter(torch.zeros(self.NUM_HEADS, 225))
      qf = torch.arange(64) % 8; qr = torch.arange(64) // 8
      _bins = ((qf[None, :] - qf[:, None]) + 7) * 15 + ((qr[None, :] - qr[:, None]) + 7)
      self.register_buffer('smol_rel_bins', _bins.reshape(-1).long(), persistent=False)
    if self.smol_static_mode > 0:
      print(f'[ceres_net] SMBSTATIC enabled (mode {self.smol_static_mode}): '
            f'{"fri 64x64" if self.smol_static_mode==1 else "15x15-RELATIV"} statisk tabell per hode, '
            f'delt over lag ({self.smol_static_bank.numel()/1e3:.1f}k params) — ERSTATTER smolgen, INGEN conditioner')
    self.smol_basis_k = int(getattr(config, 'NetDef_SmolgenStaticBasisK', 0) or 0)
    assert not (self.smol_basis_k > 0 and self.smol_static_mode > 0), 'velg EN smolgen-erstatning'
    if self.smol_static_mode > 0:
      self.smolgenPrepLayer = None
    elif self.smol_basis_k > 0:
      self.smol_basis_bank = nn.Parameter(
          torch.randn(self.NUM_HEADS, self.smol_basis_k, NUM_TOKENS_NET * NUM_TOKENS_NET) * 0.05)
      self.smolgenPrepLayer = None
      print(f'[ceres_net] SMOLBASIS enabled: {self.smol_basis_k} statiske basistabeller/hode '
            f'({self.NUM_HEADS}x{self.smol_basis_k}x4096 = '
            f'{self.NUM_HEADS*self.smol_basis_k*4096/1e3:.0f}k params, delt bank) '
            f'+ per-lags zero-init koeffisienter — ERSTATTER smolgen-generatoren')
    elif SMOLGEN_PER_SQUARE_DIM > 0 and SMOLGEN_INTERMEDIATE_DIM > 0:
      self.smolgenPrepLayer = nn.Linear(SMOLGEN_INTERMEDIATE_DIM // config.NetDef_SmolgenToHeadDivisor, NUM_TOKENS_NET * NUM_TOKENS_NET)
      # Optional LoRA wrap of the shared smolgen prep layer (gated by
      # CERES_LORA_SMOLGEN_RANK_DIV). Per-attention sm1/sm2/sm3 are wrapped
      # in dot_product_attention.py via the same env var.
      _sm_rd = int(os.environ.get('CERES_LORA_SMOLGEN_RANK_DIV', '0') or 0)
      if _sm_rd > 0:
        self.smolgenPrepLayer = LoRALinear(self.smolgenPrepLayer, _sm_rd, True)
    else:
      self.smolgenPrepLayer = None

    if config.NetDef_UseRPE:
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

    # Graph-route heads (2026-08 tactical program, see dot_product_attention):
    # per-head gated blend of softmax attention with exact row-stochastic
    # routing over the visibility edge channels. Consumes the SAME shared E
    # as the form-A projection and B/C gates, so it requires UseVisEdgeBias.
    self.use_graph_route = bool(getattr(config, 'NetDef_UseGraphRouteHeads', False))
    if self.use_graph_route:
      assert self.use_vis_edge_bias, 'UseGraphRouteHeads requires UseVisEdgeBias=true'
    _graph_route_channels = (4 * len(self._vis_edge_families)
                             if (self.use_vis_edge_bias and self.use_graph_route) else 0)

    # Soft-min ("AND-logic") value-aggregation heads (2026-08 tactical program,
    # see dot_product_attention). ARCH key, not a zero-init add-on: with
    # SoftMinHeads > 0 the first k heads of every layer aggregate V by an
    # attention-weighted soft minimum from step 0. Independent of the vis-edge
    # machinery (no E consumption, no extra channels).
    self.softmin_heads = int(getattr(config, 'NetDef_SoftMinHeads', 0) or 0)
    # Signed-tau dual (T1.1): SoftMaxAggHeads = m gives the NEXT m heads a
    # soft-MAX aggregation (same formula, tau = -exp(log_tau) init -1) for
    # existential/threat facts. Same ARCH-key semantics.
    self.softmax_agg_heads = int(getattr(config, 'NetDef_SoftMaxAggHeads', 0) or 0)
    # Per-head logit temperature (T4.1): learnable per-head sharpness on the
    # pre-softmax logits, init 1.0 = exact step-0 no-op. See dot_product_attention.
    self.use_head_logit_temp = bool(getattr(config, 'NetDef_UseHeadLogitTemp', False))

    # TACTICAL CODEBOOK (toolbox T3.3, NetDef UseTacticalCodebook): 256
    # learnable motif vectors read by one post-trunk cross-attention block
    # (64x256 linear attention — no new quadratic term). Explicit pattern
    # LIBRARY instead of patterns smeared through FFN weights: each square
    # matches its state against the codebook and reads back the matched
    # motif's contribution. Zero-init out-projection => exact step-0
    # bit-identity (headtemp/refiner contract class). Runs after the refiner,
    # before all heads, so every consumer reads the motif-enriched states.
    # KING-CENTRIC DISTANCE CHANNELS (toolbox T3.2, NetDef UseKingDistChannels):
    # all current geometry (RPE, rays, vis) is piece-relative; attack
    # evaluation is KING-relative (zones, ring distance). Per square: one-hot
    # Chebyshev-distance bucket (0..7) to own king and to enemy king (16
    # channels), computed in-graph as king-plane @ constant table (ray-
    # machinery pattern), then a zero-init Linear 16 -> D added to the
    # post-embedding flow => exact step-0 bit-identity. Known risk (weak
    # prior): smolgen already carries king context globally.
    self.use_king_dist = bool(getattr(config, 'NetDef_UseKingDistChannels', False))
    if self.use_king_dist:
      from chess_geometry import build_king_distance_onehot_table
      self.register_buffer('kdist_table', build_king_distance_onehot_table(), persistent=False)
      self.kdist_proj = nn.Linear(16, self.EMBEDDING_DIM, bias=False)
      nn.init.zeros_(self.kdist_proj.weight)
      print(f'[ceres_net] KING-DIST CHANNELS enabled: 2x8 Chebyshev buckets -> zero-init '
            f'Linear 16->{self.EMBEDDING_DIM} added post-embedding (exact step-0 no-op)')

    # DUAL-PLANE P-PLANE (dual_plane_concept.md, Stage A1 scope; NetDef
    # UseDualPlane): 32 occupancy-TopK piece tokens, relation-typed P<->P
    # attention (double-gathered VisibilityChannels E), optional soft-min
    # quantifier heads over PIECES, one zero-init cross-read of the final
    # square flow, masked mean+softmin pools -> zero-init injects into the
    # value1/value2 hidden pre-activation. S-plane untouched; policy
    # isolation provable. Requires UseVisEdgeBias (E is the relation source).
    self.use_dual_plane = bool(getattr(config, 'NetDef_UseDualPlane', False))
    # Loud rejection (review 2026-08-25 finding 3): candidate read-outs without the
    # plane would be a SILENT byte-identical control — the failure class the
    # GradScale guard below exists to prevent.
    if not self.use_dual_plane:
      # Generisk vakt (review 2026-08-25b finding 2): ENHVER satt DualPlane-
      # undernokkel uten planet er en stille byte-identisk kontroll. Feiler
      # hoyt for hele klassen i stedet for nokkel-for-nokkel.
      # Parametriske nokler har truthy DEFAULTS og er meningslose aa flagge;
      # kun feature-flagg (default 0/False) indikerer intensjon.
      _dp_param_keys = {'NetDef_DualPlanePolicyGradScale', 'NetDef_DualPlaneSoftMinHeads',
                        'NetDef_DualPlaneDim', 'NetDef_DualPlaneLayers', 'NetDef_DualPlaneSurvivalK',
                        # Hotfix 2026-09-01 (BlockRepeat-klassen, fanget av 320-kanari):
                        # TripletHeads har truthy DEFAULT (4) og fyrte guarden for ALLE
                        # no-plane-configs siden d2bafdb. Parametrisk, ikke intensjons-flagg.
                        'NetDef_DualPlaneTripletHeads',
                        # boelge 13: parametriske (default ''/0.0, men navngitte former er truthy)
                        'NetDef_DualPlaneTripletForm', 'NetDef_DualPlaneReaderInit'}
      _dp_set = [k for k, v in vars(config).items()
                 if k.startswith('NetDef_DualPlane') and k not in _dp_param_keys and bool(v)]
      _dp_set += [k for k in ('NetDef_MoveEdgeDecode', 'NetDef_MoveDegreeDecode')
                  if bool(getattr(config, k, False))]   # runde-3: konsumeres kun i plane-grenen
      if _dp_set:
        raise ValueError(f'DualPlane sub-flags set without UseDualPlane: true — silent no-op refused: {_dp_set}')
    # Loud rejection OUTSIDE the dual-plane branch (action review finding 5):
    # a grad-scale value with UseDualPlane off used to be a completely silent
    # no-op — no banner, no attr, arm runs as a byte-identical control.
    _dpgs_cfg = getattr(config, 'NetDef_DualPlanePolicyGradScale', None)
    if not self.use_dual_plane and _dpgs_cfg is not None and float(_dpgs_cfg) != 1.0:
      raise ValueError('DualPlanePolicyGradScale is set but NetDef_UseDualPlane is off — '
                       'the knob would be a silent no-op and the arm a hidden control')
    if self.use_dual_plane:
      from dual_plane import DualPlane
      # Same source-tensor rationale as the pool asserts (review finding #9):
      # the P-plane cross-reads plain `flow` and dpva queries pre-vda fS_value.
      assert int(os.environ.get('CERES_VALUE_DEPTH_ATTENTION', '0') or 0) == 0, \
          'UseDualPlane is incompatible with vda modes (P-plane reads plain flow/fS_value)'
      from tactical_adapter import gtab_enabled as _dp_gtab_enabled
      assert not _dp_gtab_enabled(), \
          'UseDualPlane is incompatible with GTAB (P-plane would read the non-adapter stream)'
      # Gjenbruk den kanoniske parsen (bugfunn 2026-08-28): re-parsing av raa
      # config-streng med '' som default divergerte fra 679-684 (case/default)
      # => feil _dp_rel_C og shape-krasj langt fra aarsaken.
      _dp_fams_all = self._vis_edge_families
      assert _dp_fams_all, 'UseDualPlane needs VisEdgeFamilies for its relation channels'
      # BOELGE 13 / P1 (2026-09-02): EDGE-AUX — par-supervisjon paa planets
      # kant-tilstand (se config.py for noekkel-semantikken). WITHHOLD fjerner
      # familier fra planets input (og dermed fra ALLE plan-side konsumenter:
      # rel_proj/e2t/degrees/move-edge-decode — alt som er dimensjonert av
      # _dp_rel_C) og gjoer dem til BCE-maal for den laerte kant-tilstanden.
      # S-planets egen vis_edge_E (UseVisEdgeBias) er uroert.
      _dp_wh_raw = [f.strip().lower() for f in
                    str(getattr(config, 'NetDef_DualPlaneEdgeAuxWithhold', '') or '').split(',') if f.strip()]
      for _f in _dp_wh_raw:
        assert _f in _dp_fams_all, \
            f'DualPlaneEdgeAuxWithhold: {_f!r} er ikke i VisEdgeFamilies {_dp_fams_all} (kan bare holde tilbake det som finnes)'
      self.dp_eaux_withhold = tuple(f for f in VisibilityChannels.FAMILY_ORDER if f in _dp_wh_raw)
      _dp_fams = tuple(f for f in _dp_fams_all if f not in self.dp_eaux_withhold)
      assert _dp_fams, 'DualPlaneEdgeAuxWithhold kan ikke holde tilbake ALLE familiene — planet trenger minst en relasjonskanal'
      _dp_rel_C = 4 * len(_dp_fams)
      self.dp_eaux_pi_w = float(getattr(config, 'Opt_LossDualPlaneEdgePiMultiplier', 0) or 0)
      self.dp_eaux_rel_w = float(getattr(config, 'Opt_LossDualPlaneEdgeRelMultiplier', 0) or 0)
      self.dp_eaux_detach = bool(getattr(config, 'NetDef_DualPlaneEdgeAuxDetach', False))
      self.dp_eaux_on = (self.dp_eaux_pi_w > 0) or (self.dp_eaux_rel_w > 0)
      if self.dp_eaux_detach and not self.dp_eaux_on:
        raise ValueError('DualPlaneEdgeAuxDetach uten LossDualPlaneEdgePi/RelMultiplier > 0 er en stille no-op (skjult kontroll)')
      if self.dp_eaux_rel_w > 0 and not self.dp_eaux_withhold:
        raise ValueError('LossDualPlaneEdgeRelMultiplier > 0 krever DualPlaneEdgeAuxWithhold (ellers er maalet en identitets-avlesning av inputen)')
      if self.dp_eaux_on and not (bool(getattr(config, 'NetDef_DualPlaneEdgeUpdate', False))
                                  or bool(getattr(config, 'NetDef_DualPlaneTripletAttention', False))):
        raise ValueError('Edge-aux krever en LAERT kant-tilstand (DualPlaneEdgeUpdate og/eller DualPlaneTripletAttention) — '
                         'uten den er kantene den statiske E og hodet en linear probe paa inputen')
      # Relation source: reuse the S-plane's vis_edge_E when UseVisEdgeBias is
      # on; on a BARE chassis (the A1 one-key-delta design: dp1 = nvc +
      # UseDualPlane) build a private VisibilityChannels and compute E solely
      # for the P-plane — the S-plane attention stays untouched either way.
      self.dp_private_vis = not bool(getattr(config, 'NetDef_UseVisEdgeBias', False))
      if self.dp_private_vis:
        self.dp_vis_module = VisibilityChannels(families=_dp_fams)
      if self.dp_eaux_withhold:
        # Shared-E case: the S-plane tensor carries ALL families in canonical
        # order, 4 channels each — select the plane's and the withheld subsets.
        # Private case: a second tiny VisibilityChannels builds the targets.
        _idx = {f: [4 * i + o for o in range(4)] for i, f in enumerate(_dp_fams_all)}
        self.register_buffer('dp_plane_ch_idx',
                             torch.tensor(sum((_idx[f] for f in _dp_fams), []), dtype=torch.long),
                             persistent=False)
        self.register_buffer('dp_eaux_ch_idx',
                             torch.tensor(sum((_idx[f] for f in self.dp_eaux_withhold), []), dtype=torch.long),
                             persistent=False)
        if self.dp_private_vis and self.dp_eaux_rel_w > 0:
          # fork_rng: VisibilityChannels runs a construction-time random probe
          # (the out/in swap identity), which would advance the global stream
          # and break bit-pairing with the withheld control (caught by
          # test_dual_plane_edge_aux.py on the very first run).
          with torch.random.fork_rng(devices=[]):
            self.dp_eaux_vis_module = VisibilityChannels(families=self.dp_eaux_withhold)
        if not self.dp_eaux_on:
          print(f'[ceres_net] DUAL-PLANE EDGE-AUX WITHHELD CONTROL: {self.dp_eaux_withhold} fjernet fra '
                f'planets input ({_dp_rel_C} kanaler igjen), INGEN kant-supervisjon (kontrollarmen)')
      if self.dp_eaux_on:
        # Readout: ONE linear head on the final [B,32,32,C] edge state ->
        # [pi-edge logit (1)] + [withheld relation logits (4 per family)].
        # NONZERO init (the whole point: a live gradient into the edges from
        # step 0 — the zero-init product-rule cascade is the diagnosed
        # bottleneck) drawn from a FIXED module-local key, as plain
        # nn.Parameters (an nn.Linear would draw from the global RNG stream and
        # shift every later init => the arm would not be bit-paired with its
        # control; Kovax-invarianten). Not part of the served graph.
        _T = (1 if self.dp_eaux_pi_w > 0 else 0) + (4 * len(self.dp_eaux_withhold) if self.dp_eaux_rel_w > 0 else 0)
        _g = torch.Generator().manual_seed(0x0E6E)
        _w = torch.empty(_T, _dp_rel_C)
        _w.uniform_(-(_dp_rel_C ** -0.5), _dp_rel_C ** -0.5, generator=_g)
        self.dp_eaux_w = nn.Parameter(_w)
        self.dp_eaux_b = nn.Parameter(torch.zeros(_T))
        if self.dp_eaux_pi_w > 0:
          from lc0_moves_1858 import FROM_1858 as _F7, TO_1858 as _T7
          self.register_buffer('dp_eaux_mv_ft',
                               torch.tensor(_F7, dtype=torch.long) * 64 + torch.tensor(_T7, dtype=torch.long),
                               persistent=False)   # [1858] from*64+to
        print(f'[ceres_net] DUAL-PLANE EDGE-AUX enabled: pi-edge w={self.dp_eaux_pi_w}, '
              f'rel w={self.dp_eaux_rel_w} on withheld {self.dp_eaux_withhold} '
              f'({_T} targets from {_dp_rel_C} edge channels, nonzero fixed-key init'
              f'{", DETACHED probe: no gradient into the plane" if self.dp_eaux_detach else ""}; training-only)')
      _dp_smh = int(getattr(config, 'NetDef_DualPlaneSoftMinHeads', 2) or 0)
      _dp_dim = int(getattr(config, 'NetDef_DualPlaneDim', 128) or 128)
      _dp_layers = int(getattr(config, 'NetDef_DualPlaneLayers', 2) or 2)
      _dp_il = bool(getattr(config, 'NetDef_DualPlaneInterleave', False))
      self.dual_plane = DualPlane(s_dim=self.EMBEDDING_DIM, rel_channels=_dp_rel_C,
                                  norm_type=config.NetDef_NormType,
                                  dp=_dp_dim, heads=max(4, _dp_dim // 32),
                                  layers=_dp_layers,
                                  softmin_heads=_dp_smh,
                                  interleave_cross=_dp_il,
                                  rel_degrees=bool(getattr(config, 'NetDef_DualPlaneRelDegrees', False)),
                                  rel_gains=bool(getattr(config, 'NetDef_DualPlaneRelGains', False)),
                                  rel_degrees2=bool(getattr(config, 'NetDef_DualPlaneRelDegrees2', False)),
                                  king_flight=bool(getattr(config, 'NetDef_DualPlaneKingFlight', False)),
                                  king_zone=bool(getattr(config, 'NetDef_DualPlaneKingZone', False)),
                                  edge_update=bool(getattr(config, 'NetDef_DualPlaneEdgeUpdate', False)),
                                  triplet_attention=getattr(config, 'NetDef_DualPlaneTripletAttention', False),
                                  triplet_heads=int(getattr(config, 'NetDef_DualPlaneTripletHeads', 4) or 4),
                                  triplet_form=(str(getattr(config, 'NetDef_DualPlaneTripletForm', '') or '') or 'tgt'),
                                  reader_init=float(getattr(config, 'NetDef_DualPlaneReaderInit', 0) or 0))
      self.dp_reader_init = float(getattr(config, 'NetDef_DualPlaneReaderInit', 0) or 0)
      if str(getattr(config, 'NetDef_DualPlaneTripletForm', '') or '') and \
          not getattr(config, 'NetDef_DualPlaneTripletAttention', False):
        raise ValueError('DualPlaneTripletForm er satt uten DualPlaneTripletAttention (stille no-op)')
      if self.dp_reader_init > 0:
        print(f'[ceres_net] DUAL-PLANE READER-INIT: edge readers (rel_proj/eu_out/eu_deg/ta_out/e2t_proj) '
              f'uniform(+-{self.dp_reader_init}) from fixed keys — NOT a step-0 no-op by design')
      # KANT->TRUNK (boelge 9, 2026-08-30): planets ferdig-utviklede kanter
      # loeftes til [B,H,64,64]-bias paa trunk-attention (via one-hot-scatter,
      # dense matmuls). Teorigrunnlag: Kovax' rute-modell + klipp-stillhets-
      # funnet (kantruten avlaster attention-logits — naa ogsaa i trunken).
      # Zero-init => eksakt step-0 no-op. Krever edge_update (levende kanter);
      # inkompatibel med DualPlaneInterleave (fase-splitten).
      self.dp_edge_to_trunk = bool(getattr(config, 'NetDef_DualPlaneEdgeToTrunk', False))
      self.dp_e2t_mask = bool(getattr(config, 'NetDef_DualPlaneEdgeToTrunkMask', False))
      if self.dp_e2t_mask and not self.dp_edge_to_trunk:
        raise ValueError('DualPlaneEdgeToTrunkMask krever DualPlaneEdgeToTrunk (ellers stille inert — review-funn 6)')
      if self.dp_edge_to_trunk:
        if not bool(getattr(config, 'NetDef_DualPlaneEdgeUpdate', False)):
          raise ValueError('DualPlaneEdgeToTrunk krever DualPlaneEdgeUpdate (levende kanter) — '
                           'uten den loeftes STATISKE kanter (skjult kontroll). ValueError per hus-konvensjon (-O).')
        if _dp_il:
          raise ValueError('DualPlaneEdgeToTrunk er inkompatibel med DualPlaneInterleave (fase-splitt)')
        self.e2t_proj = nn.Linear(_dp_rel_C, self.NUM_HEADS, bias=False)
        if self.dp_reader_init > 0:
          with torch.no_grad():
            self.e2t_proj.weight.uniform_(-self.dp_reader_init, self.dp_reader_init,
                                          generator=torch.Generator().manual_seed(0x0EA0 + 9999))
        else:
          nn.init.zeros_(self.e2t_proj.weight)
        print(f'[ceres_net] DUAL-PLANE EDGE-TO-TRUNK enabled: levende kanter -> zero-init '
              f'[B,{self.NUM_HEADS},64,64]-bias paa alle trunk-lag (fase-splittet plan)')
      if getattr(config, 'NetDef_DualPlaneEdgeUpdate', False):
        print('[ceres_net] DUAL-PLANE EDGE-UPDATE enabled: laert residual kant-oppdatering '
              'per P-blokk (E_ij <- E_ij + f(E_ij, x_i, x_j), zero-init) + degree-refresh '
              'fra levende kanter inn i tokenene (exact step-0 no-op)')
      if getattr(config, 'NetDef_DualPlaneRelDegrees2', False):
        print(f'[ceres_net] DUAL-PLANE REL-DEGREES-2 enabled: second-order coverage '
              f'(targets/attackers, 2x{_dp_rel_C} ch) -> zero-init token features '
              f'(exact step-0 no-op)')
      if getattr(config, 'NetDef_DualPlaneKingFlight', False):
        print(f'[ceres_net] DUAL-PLANE KING-FLIGHT enabled: 8-neighbor flight-zone '
              f'in-degree ({_dp_rel_C} ch) + occupancy into king tokens, '
              f'zero-init (exact step-0 no-op)')
      if getattr(config, 'NetDef_DualPlaneKingZone', False):
        print(f'[ceres_net] DUAL-PLANE KING-ZONE (kf2) enabled: per-piece coverage of '
              f'both king 3x3 zones ({_dp_rel_C} ch each) from e_rows, '
              f'zero-init (exact step-0 no-op)')
      # NOTE: no 'or default' here — 0.0 (full detach) is a legitimate setting
      # and the falsy-swallow would silently turn that arm into the control
      # (review 2026-08-21 finding 3).
      _dpgs = getattr(config, 'NetDef_DualPlanePolicyGradScale', 1.0)
      self.dp_policy_grad_scale = float(1.0 if _dpgs is None else _dpgs)
      if self.dp_policy_grad_scale != 1.0:
        # ValueError, not assert: house convention for config-combo guards
        # (asserts vanish under python -O) — action review finding 5.
        if not getattr(config, 'NetDef_DualPlanePolicyDecode', False):
          raise ValueError('DualPlanePolicyGradScale != 1.0 requires DualPlanePolicyDecode '
                           '(the scale is applied inside the policy-decode branch; without '
                           'decode the knob is a no-op)')
        print(f'[ceres_net] DUAL-PLANE POLICY-GRAD SCALE enabled: policy-decode gradients '
              f'into shared P-tokens x {self.dp_policy_grad_scale} (forward identity; '
              f'value gradients and decode weights unscaled)')
      if getattr(config, 'NetDef_DualPlaneRelDegrees', False):
        print(f'[ceres_net] DUAL-PLANE REL-DEGREES enabled: 2x{_dp_rel_C} degree channels -> '
              f'zero-init token features (exact step-0 no-op)')
      if getattr(config, 'NetDef_DualPlaneRelGains', False):
        print('[ceres_net] ⛔ WARNING: DualPlaneRelGains is TRT-UNSERVABLE — measured '
              '46x slower under TensorRT (547 vs 25,050 EPS, cached engine, both orders, '
              '2026-08-22). ORT shows the arithmetic is free, so this is a TRT '
              'compilation outcome: the per-sample W_eff turns one big constant-weight '
              'GEMM into B tiny dynamic-weight GEMMs per block. Fine for research runs '
              'that will never be served; see dual_plane.py for servable redesigns.',
              flush=True)
        print(f'[ceres_net] DUAL-PLANE REL-GAINS enabled: per-block masked-mean -> '
              f'per-head/channel gains on relation bias ({_dp_rel_C} ch), '
              f'zero-init (exact step-0 no-op)')
      # Stage A3 (NetDef DualPlanePolicyDecode): mover-bilinear policy term at
      # DECODE — logit[m] += q(S-state at to(m))^T p(piece token standing on
      # from(m)). Piece-selection composed with destination content, exactly
      # where the bilinear family won before (ray-context). Zero-init q side
      # => exact step-0 policy no-op; gathers on constant index tables only.
      self.dp_policy_decode = bool(getattr(config, 'NetDef_DualPlanePolicyDecode', False))
      if self.dp_policy_decode:
        from lc0_moves_1858 import FROM_1858, TO_1858
        _mv = torch.tensor(TO_1858, dtype=torch.long) * 64 + torch.tensor(FROM_1858, dtype=torch.long)
        self.register_buffer('dp_move_flat', _mv, persistent=False)   # [1858] to*64+from
        _DP_DQ = 64
        self.dp_pol_q = nn.Linear(self.EMBEDDING_DIM, _DP_DQ, bias=False)
        nn.init.zeros_(self.dp_pol_q.weight)
        self.dp_pol_p = nn.Linear(self.dual_plane.dp, _DP_DQ, bias=False)
        print(f'[ceres_net] DUAL-PLANE POLICY DECODE enabled: mover-bilinear dq={_DP_DQ}, '
              f'zero-init q-side (exact step-0 policy no-op)')
      # ATTACKER×VICTIM decode (catalogue #2, sac-rule-compliant: LEARNED pair
      # bilinear over piece TOKENS, no hand-coded value diff — sign-free, the
      # net decides when Qxh7 is brilliant): logit[m] += p_a(mover) · p_b(piece
      # on to(m)). Non-captures get exactly 0 (empty to-squares are unselected
      # slots / occ-masked). Zero-init b-side => step-0 policy no-op.
      self.dp_victim_decode = bool(getattr(config, 'NetDef_DualPlaneVictimDecode', False))
      # MOVE-EDGE decode (catalogue #3): the move's OWN relation-edge channels
      # (is this a check-edge? a pinray? vacates an x-ray?) gathered from the
      # already-computed E tensor straight into the move score. Zero-init.
      # Works on ANY dp chassis: with UseVisEdgeBias it reads the shared S-plane
      # E; on a bare chassis it reads the P-plane's PRIVATE E (computed anyway
      # for the relation biases — the gather is free either way).
      self.move_edge_decode = bool(getattr(config, 'NetDef_MoveEdgeDecode', False))
      if self.dp_victim_decode or self.move_edge_decode:
        assert self.dp_policy_decode, 'victim/edge decode extend DualPlanePolicyDecode'
        from lc0_moves_1858 import FROM_1858 as _F2, TO_1858 as _T2
        _mv_ft = torch.tensor(_F2, dtype=torch.long) * 64 + torch.tensor(_T2, dtype=torch.long)
        self.register_buffer('dp_move_flat_ft', _mv_ft, persistent=False)  # [1858] from*64+to
      if self.dp_victim_decode:
        _DPV_DQ = 64
        self.dpv_a = nn.Linear(self.dual_plane.dp, _DPV_DQ, bias=False)
        self.dpv_b = nn.Linear(self.dual_plane.dp, _DPV_DQ, bias=False)
        nn.init.zeros_(self.dpv_b.weight)
        print('[ceres_net] ATTACKER×VICTIM DECODE enabled: pair-bilinear dq=64, '
              'zero-init victim-side (exact step-0 no-op)')
      if self.move_edge_decode:
        self.dpe_w = nn.Linear(_dp_rel_C, 1, bias=False)
        nn.init.zeros_(self.dpe_w.weight)
        print(f'[ceres_net] MOVE-EDGE DECODE enabled: {_dp_rel_C} edge channels -> '
              'per-move scalar, zero-init (exact step-0 no-op)')
      # MOVE-DEGREE decode (catalogue idea B): per-move scalar from the
      # DESTINATION square's per-channel in-degree ("who hits where I land")
      # plus the FROM square's out-degree ("what I abandon by leaving").
      # Descriptive facts with LEARNED sign-free weights — sac-compliant: the
      # sacrificing queen knows h7 is defended and plays it anyway.
      self.move_degree_decode = bool(getattr(config, 'NetDef_MoveDegreeDecode', False))
      if self.move_degree_decode:
        assert self.dp_policy_decode, 'MoveDegreeDecode extends DualPlanePolicyDecode'
        from lc0_moves_1858 import FROM_1858 as _F3, TO_1858 as _T3
        self.register_buffer('dp_move_to', torch.tensor(_T3, dtype=torch.long), persistent=False)
        self.register_buffer('dp_move_from', torch.tensor(_F3, dtype=torch.long), persistent=False)
        self.dpd_in = nn.Linear(_dp_rel_C, 1, bias=False)
        self.dpd_out = nn.Linear(_dp_rel_C, 1, bias=False)
        nn.init.zeros_(self.dpd_in.weight)
        nn.init.zeros_(self.dpd_out.weight)
        print(f'[ceres_net] MOVE-DEGREE DECODE enabled: destination in-degree + origin '
              f'out-degree ({_dp_rel_C} ch each), zero-init (exact step-0 no-op)')
      # CANDIDATE-VALUE READ (N3, 2026-08-25): value = max-over-moves is the
      # ground truth of chess; today's value family never sees WHICH moves
      # exist. Top-K policy candidates (selection + weights fully DETACHED —
      # no gradient reaches the policy path) are summarized as (mover-token,
      # move-edge channels, destination in-degree, origin out-degree, detached
      # logit) features, softmax(T=2)-weighted, and injected zero-init into the
      # value1/value2 hidden pre-activation. K>policy-argmax deliberately:
      # value must see the threats policy does NOT choose.
      self.dp_cand_value = int(getattr(config, 'NetDef_DualPlaneCandidateValue', 0) or 0)
      if self.dp_cand_value > 0:
        if not self.dp_policy_decode:
          raise ValueError('DualPlaneCandidateValue extends DualPlanePolicyDecode (enable it)')
        if not hasattr(self, 'dp_move_to'):
          from lc0_moves_1858 import FROM_1858 as _F4, TO_1858 as _T4
          self.register_buffer('dp_move_to', torch.tensor(_T4, dtype=torch.long), persistent=False)
          self.register_buffer('dp_move_from', torch.tensor(_F4, dtype=torch.long), persistent=False)
        self.dpcv_embed = nn.Linear(self.dual_plane.dp + 3 * _dp_rel_C + 1, 64)
        self.dpcv_out = nn.Linear(64, 64 * HEAD_MULT, bias=False)
        nn.init.zeros_(self.dpcv_out.weight)
        if self.value2_loss_weight > 0:
          self.dpcv_out2 = nn.Linear(64, 64 * HEAD_MULT, bias=False)
          nn.init.zeros_(self.dpcv_out2.weight)
        print(f'[ceres_net] CANDIDATE-VALUE READ enabled: top-{self.dp_cand_value} detached '
              f'policy candidates (mover-token + edge + degrees) -> zero-init value1/value2 '
              f'injects (exact step-0 no-op; no gradient to policy)')
      # CANDIDATE ATTENTION RE-SCORE (N1, 2026-08-25): today every move logit is
      # scored in ISOLATION — the Qxh7 logit never sees what the Rxh7 logit
      # knows, and the measured 2700 failure mode is selection (792/2000 top-3
      # vs 90/2000 top-1). Top-K candidates (selection DETACHED) become move
      # tokens = (mover piece-token, to-square embed, the move's E channels,
      # detached logit) that attend over EACH OTHER + the 32 piece tokens in
      # one mini block; a zero-init scalar re-score is added back onto the K
      # logits. Architecture, not a hinge loss: the sbm tombstone (mass-stealing
      # margin objective) does not apply — CE and targets are untouched.
      self.dp_cand_attn = int(getattr(config, 'NetDef_DualPlaneCandidateAttention', 0) or 0)
      if self.dp_cand_attn > 0:
        if not self.dp_policy_decode:
          raise ValueError('DualPlaneCandidateAttention extends DualPlanePolicyDecode (enable it)')
        if not hasattr(self, 'dp_move_to'):
          from lc0_moves_1858 import FROM_1858 as _F5, TO_1858 as _T5
          self.register_buffer('dp_move_to', torch.tensor(_T5, dtype=torch.long), persistent=False)
          self.register_buffer('dp_move_from', torch.tensor(_F5, dtype=torch.long), persistent=False)
        _dc = 64
        self.dpc_embed = nn.Linear(2 * self.dual_plane.dp + _dp_rel_C + 1, _dc)
        self.dpc_piece = nn.Linear(self.dual_plane.dp, _dc, bias=False)   # brikke-tokens -> kv-rom
        self.dpc_q = nn.Linear(_dc, _dc, bias=False)
        self.dpc_k = nn.Linear(_dc, _dc, bias=False)
        self.dpc_v = nn.Linear(_dc, _dc, bias=False)
        self.dpc_out = nn.Linear(_dc, _dc, bias=False)
        self.dpc_score = nn.Linear(_dc, 1, bias=False)
        nn.init.zeros_(self.dpc_score.weight)
        print(f'[ceres_net] CANDIDATE-ATTENTION RE-SCORE enabled: top-{self.dp_cand_attn} detached '
              f'candidates as move tokens, mutual attention over candidates + piece tokens, '
              f'zero-init re-score onto the K logits (exact step-0 no-op)')
      # CHECK-CHAIN DECODE (N2, 2026-08-25): mate patterns are compositions of
      # the check and flight channels the net already computes 1-hop. CPU
      # prerequisite probe (3,000 mateIn2): the static composition "gives check
      # AND the enemy king has zero free flight squares" fires on 16.2% of
      # solution moves vs 1.0% of other legal moves (16.7x lift, near-zero
      # false positives) — a low-recall/high-precision feature. Four per-move
      # composed channels through a zero-init linear into the move score
      # (dpe_w cost class; NO move simulation — current-position geometry only,
      # same documented approximation class as the check channel itself).
      self.dp_check_chain = bool(getattr(config, 'NetDef_DualPlaneCheckChain', False))
      if self.dp_check_chain:
        if not self.dp_policy_decode:
          raise ValueError('DualPlaneCheckChain extends DualPlanePolicyDecode (enable it)')
        # Runde-3-fiks 2026-08-29 (init-rekkefoelge-bug): vis_channels_module
        # lages FOERST SENERE i __init__, saa delt-chassis-oppslaget (2026-08-25b
        # finding 1) virket aldri — utled layouten direkte fra den kanoniske
        # familie-parsen i stedet (samme konstruksjon som chess_geometry:
        # 4 kanaler per familie i _dp_fams-rekkefoelge).
        _fs = {f: slice(4 * i, 4 * i + 4) for i, f in enumerate(_dp_fams)}
        if 'check' not in _fs or 'flight' not in _fs:
          raise ValueError('DualPlaneCheckChain requires check+flight in VisEdgeFamilies')
        self.dp_ch_check = _fs['check'].start      # [stm_out, ...] -> stm_out forst
        self.dp_ch_flight = _fs['flight'].start
        if not hasattr(self, 'dp_move_to'):
          from lc0_moves_1858 import FROM_1858 as _F6, TO_1858 as _T6
          self.register_buffer('dp_move_to', torch.tensor(_T6, dtype=torch.long), persistent=False)
          self.register_buffer('dp_move_from', torch.tensor(_F6, dtype=torch.long), persistent=False)
        self.dpch_w = nn.Linear(4, 1, bias=False)
        nn.init.zeros_(self.dpch_w.weight)
        print('[ceres_net] CHECK-CHAIN DECODE enabled: 4 composed check/flight move '
              'channels -> zero-init move score (exact step-0 no-op)')
      self.dp_value_inject = nn.Linear(2 * self.dual_plane.dp, 64 * HEAD_MULT, bias=False)
      nn.init.zeros_(self.dp_value_inject.weight)
      if self.value2_loss_weight > 0:
        self.dp_value2_inject = nn.Linear(2 * self.dual_plane.dp, 64 * HEAD_MULT, bias=False)
        nn.init.zeros_(self.dp_value2_inject.weight)
      # GATEDE PLAN-INJECTS (A, 2026-08-25, arXiv 2505.06708-prinsippet paa
      # injectene): value-hodets lesing av planet er i dag REN LINEAER — men
      # 4x-ablasjonen maalte at planet betyr 4x mer i skarpe stillinger. Gaten
      # sigma(W_g * fS_value) gjor plan-avhengigheten INNHOLDSBETINGET paa
      # trunk-konteksten. Injectene er zero-init, saa sigma(x)*0 = 0 => eksakt
      # steg-0-no-op fra scratch UANSETT gate-init; bias +4 (sigma=0.982) gjor
      # den i tillegg naer-no-op ved innsetting oppaa TRENTE injects.
      self.dp_gated_injects = bool(getattr(config, 'NetDef_DualPlaneGatedInjects', False))
      if self.dp_gated_injects:
        self.dpgi_v = nn.Linear(self.VALUE_IN_SIZE, 64 * HEAD_MULT, bias=True)   # fS_value-bredden (jf. dpva)
        nn.init.zeros_(self.dpgi_v.weight)
        nn.init.constant_(self.dpgi_v.bias, 4.0)
        if self.value2_loss_weight > 0:
          self.dpgi_v2 = nn.Linear(self.VALUE_IN_SIZE, 64 * HEAD_MULT, bias=True)
          nn.init.zeros_(self.dpgi_v2.weight)
          nn.init.constant_(self.dpgi_v2.bias, 4.0)
        print('[ceres_net] GATED PLANE-INJECTS enabled: sigma(W_g*fS_value) x value1/value2 '
              'pool-injects (content-conditioned plane reliance; exact step-0 no-op from scratch)')
      # I2-ABLASJON (2026-08-26, offer-antimoensteret fra theme-matrisen):
      # CERES_DP_NO_POOL_INJECTS=1 kobler POOL-injectene ut av value-veien og
      # lar kun attention-lesinger (dpva/dpcv) staa igjen. Modulene beholdes
      # (state_dict uendret => export-vaktene upaavirket); de faar bare null
      # gradient. Kausal test: baerer poolingen materiell-prioret som skader
      # offer-cellene, mens attention-lesing slipper unna?
      self.dp_no_pool_injects = int(os.environ.get('CERES_DP_NO_POOL_INJECTS', '0') or 0) > 0
      if self.dp_no_pool_injects:
        print('[ceres_net] DP POOL-INJECTS DISABLED (I2-ablasjon): value leser planet kun via attention-veier')
      # PER-PIECE SURVIVAL AUX (training-only; value-grip hypothesis 2026-08-21):
      # predict each PIECE TOKEN's fate (captured at ply d / survives) against
      # the square-indexed survival sidecar targets gathered to piece slots.
      # Gives value-relevant threat state a DIRECT gradient grip on the piece
      # tokens (the decode lesson: grips work, offers get ignored). Amputated
      # at export (training-gated stash, placement/survival pattern).
      self.dp_surv_weight = float(getattr(config, 'NetDef_DualPlaneSurvivalAux', 0.0) or 0.0)
      if self.dp_surv_weight > 0:
        _dp_sk = int(getattr(config, 'NetDef_DualPlaneSurvivalK', 4) or 4)
        self.dp_surv_head = nn.Linear(self.dual_plane.dp, _dp_sk + 2)
        print(f'[ceres_net] DUAL-PLANE PER-PIECE SURVIVAL AUX enabled: weight {self.dp_surv_weight}, '
              f'K={_dp_sk} (fate CE on piece tokens vs sidecar targets gathered to slots)')
      # VALUE-ATTENTION READ (dpva): instead of only the pooled summary, the
      # value pathway asks the piece plane QUESTIONS — nq content-conditioned
      # queries (projected from fS_value) attend over the 32 piece tokens
      # (empty slots masked), and the attended answer enters value1/value2
      # via zero-init injects. Richer grip than one pooled offer; exact
      # step-0 no-op; plain matmul/softmax (serving graph, TRT-safe).
      self.dp_value_attn = int(getattr(config, 'NetDef_DualPlaneValueAttention', 0) or 0)
      if self.dp_value_attn > 0:
        _dpva_dk = 64
        self.dpva_q = nn.Linear(self.VALUE_IN_SIZE, self.dp_value_attn * _dpva_dk, bias=False)
        self.dpva_k = nn.Linear(self.dual_plane.dp, _dpva_dk, bias=False)
        self.dpva_v = nn.Linear(self.dual_plane.dp, _dpva_dk, bias=False)
        self.dpva_out = nn.Linear(self.dp_value_attn * _dpva_dk, 64 * HEAD_MULT, bias=False)
        nn.init.zeros_(self.dpva_out.weight)
        if self.value2_loss_weight > 0:
          self.dpva_out2 = nn.Linear(self.dp_value_attn * _dpva_dk, 64 * HEAD_MULT, bias=False)
          nn.init.zeros_(self.dpva_out2.weight)
        print(f'[ceres_net] DUAL-PLANE VALUE-ATTENTION enabled: {self.dp_value_attn} queries x dk={_dpva_dk} '
              f'over piece tokens -> zero-init value1/value2 injects (exact step-0 no-op)')
      _n_dp = sum(p.numel() for p in self.dual_plane.parameters())
      print(f'[ceres_net] DUAL-PLANE enabled: 32 piece tokens, dp={_dp_dim}, {_dp_layers} P-blocks'
            f'{" (interleaved cross)" if _dp_il else ""} ({_dp_rel_C} relation channels), '
            f'value injects{" + policy decode" if self.dp_policy_decode else ""} '
            f'({_n_dp:,} P-plane params, zero-init injects => exact step-0 no-op)')

    # MOVE-GRAPH SPECTRAL PE (toolbox T4.3, NetDef UseSpectralPE): Laplacian
    # eigenvector coordinates of the four elementary move graphs (N/K/R/B),
    # OCCUPANCY-GATED — each type's block contributes only on squares hosting
    # that type (queen gates both R and B blocks). Knight distance is not
    # Euclidean; these coordinates are native to how pieces actually travel.
    # Zero-init Linear 32 -> D post-embedding => exact step-0 bit-identity.
    self.use_spectral_pe = bool(getattr(config, 'NetDef_UseSpectralPE', False))
    if self.use_spectral_pe:
      from chess_geometry import build_spectral_pe_table
      # persistent=True (review finding #2): the table comes from eigh over
      # DEGENERATE eigenspaces, so the basis/sign is LAPACK-arbitrary — a
      # rebuild on another platform (WSL train -> Windows export) would pair
      # trained spe_proj weights with a rotated eigenbasis. Persisting stores
      # the training-time basis in the checkpoint.
      self.register_buffer('spe_table', build_spectral_pe_table(8), persistent=True)
      self.spe_proj = nn.Linear(32, self.EMBEDDING_DIM, bias=False)
      nn.init.zeros_(self.spe_proj.weight)
      print(f'[ceres_net] SPECTRAL PE enabled: 4 move-graphs x 8 eigenvectors, occupancy-gated, '
            f'zero-init Linear 32->{self.EMBEDDING_DIM} post-embedding (exact step-0 no-op)')

    self.use_tactical_codebook = bool(getattr(config, 'NetDef_UseTacticalCodebook', False))
    if self.use_tactical_codebook:
      _CBK_N, _CBK_DK = 256, 64
      self.cbk_norm = make_norm(config.NetDef_NormType, self.EMBEDDING_DIM, eps=1E-6)
      self.cbk_q = nn.Linear(self.EMBEDDING_DIM, _CBK_DK, bias=False)
      self.cbk_keys = nn.Parameter(torch.randn(_CBK_N, _CBK_DK) * 0.02)
      self.cbk_vals = nn.Parameter(torch.randn(_CBK_N, self.EMBEDDING_DIM) * 0.02)
      self.cbk_out = nn.Linear(self.EMBEDDING_DIM, self.EMBEDDING_DIM, bias=False)
      nn.init.zeros_(self.cbk_out.weight)
      _n_cbk = (self.EMBEDDING_DIM * _CBK_DK + _CBK_N * _CBK_DK
                + _CBK_N * self.EMBEDDING_DIM + self.EMBEDDING_DIM ** 2)
      print(f'[ceres_net] TACTICAL CODEBOOK enabled: {_CBK_N} motif vectors, dk={_CBK_DK}, '
            f'one post-trunk cross-attn block ({_n_cbk:,} params, zero-init out => exact step-0 no-op)')

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
                      smol_basis_k = getattr(self, 'smol_basis_k', 0),
                      smol_basis_bank = getattr(self, 'smol_basis_bank', None),
                      smol_static_mode = getattr(self, 'smol_static_mode', 0),
                      smol_static_bank = getattr(self, 'smol_static_bank', None),
                      smol_rel_bins = getattr(self, 'smol_rel_bins', None),
                      smolgen_activation_type = config.NetDef_SmolgenActivationType,
                      smolgen_delta_rank = config.NetDef_SmolgenDeltaRank,
                      alpha=self.alpha, layerNum=i, dropout_rate=self.DROPOUT_RATE,
                      use_rpe=config.NetDef_UseRPE, 
                      use_rpe_v=config.NetDef_UseRPE_V,
                      rpe_factor_shared=self.rpe_factor_shared,
                      use_rope=config.NetDef_UseRoPE,
                      use_nonlinear_attention=config.NetDef_NonLinearAttention,
                      test = config.Exec_TestFlag,
                      use_diff_attention = config.NetDef_UseDiffAttention,
                      tsb_enabled = getattr(config, 'NetDef_TSB_Enabled', False),
                      tsb_ffn_multiplier = getattr(config, 'NetDef_TSB_FFNMultiplier', 1),
                      tsb_gate_bias_init = getattr(config, 'NetDef_TSB_GateBiasInit', -4.0),
                      tsb_gate_mlp_hidden_divisor = getattr(config, 'NetDef_TSB_GateMLPHiddenDivisor', 8),
                      vis_gate_channels = _vis_gate_channels,
                      vis_gate_mode = self.vis_edge_gate_mode,
                      graph_route_channels = _graph_route_channels,
                      softmin_heads = self.softmin_heads,
                      softmax_agg_heads = self.softmax_agg_heads,
                      use_head_logit_temp = self.use_head_logit_temp,
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

    # Iterated tactic refiner (2026-08 tactical program, see tactical_refiner.py):
    # one small weight-shared block applied RefinerIters times to the trunk
    # output — serial calculation depth for forcing lines at a fraction of a
    # full layer per iteration. Zero-init output => exact no-op at step 0.
    # Part of the SERVING graph (all heads read the refined flow).
    # RefinerDeepSupWeight > 0 additionally supervises the intermediate
    # iterations with the policy target during training (vda deep-supervision
    # precedent); the aux logits go through the PLAIN shared head front-end +
    # base policy head (no ray-context/pda augmentation — gradient shaping
    # only, never served).
    self.refiner_iters = int(getattr(config, 'NetDef_RefinerIters', 0) or 0)
    self.refiner_deep_sup_weight = 0.0
    if self.refiner_iters > 0:
      from tactical_refiner import TacticalRefiner
      _rf_dim = int(getattr(config, 'NetDef_RefinerDim', 128) or 128)
      _rf_heads = int(getattr(config, 'NetDef_RefinerHeads', 4) or 4)
      _rf_ffn = int(getattr(config, 'NetDef_RefinerFFNMult', 2) or 2)
      self.refiner_deep_sup_weight = float(getattr(config, 'NetDef_RefinerDeepSupWeight', 0) or 0)
      self.tactical_refiner = TacticalRefiner(in_dim=self.EMBEDDING_DIM,
                                              inner_dim=_rf_dim, num_heads=_rf_heads,
                                              ffn_mult=_rf_ffn, iters=self.refiner_iters,
                                              norm_type=config.NetDef_NormType,
                                              softcap_cutoff=config.NetDef_SoftCapCutoff)
      _rf_params = sum(p.numel() for p in self.tactical_refiner.parameters())
      print(f'[ceres_net] TACTIC REFINER enabled: dim {_rf_dim} x {_rf_heads} heads, '
            f'ffn_mult {_rf_ffn}, {self.refiner_iters} weight-shared iterations '
            f'({_rf_params} params, zero-init no-op), deep-sup weight {self.refiner_deep_sup_weight}')

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

    # Config-only per the config-over-env rule; the transitional env fallback
    # (kept only for the 2026-08 prod ray6 200M run, now concluded) is retired.
    assert not os.environ.get('CERES_RAY_ATTENTION_BIAS'), \
        'CERES_RAY_ATTENTION_BIAS is retired — set "UseRayAttentionBias": true ' \
        'in the _ceres_net.json config instead'
    self.use_ray_bias = bool(getattr(config, 'NetDef_UseRayAttentionBias', False))
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
    # Same source-tensor rationale: the pool reads plain `flow`; under gtab
    # value-only the value family reads flow_value instead and the pool would
    # silently summarize the wrong stream.
    assert not self.value_minmax_pool or (self.vda_mode == 0 and not self.use_gtab), \
      'ValueHeadMinMaxPool is incompatible with vda/gtab modes (pool reads plain flow)'
    assert not self.value_pool_channels or (self.vda_mode == 0 and not self.use_gtab), \
      'ValueHeadPoolChannels is incompatible with vda/gtab modes (pool reads plain flow)'
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

    # MOVE-TOKEN DECODER (design B, 2026-09-02; see move_tokens.py). Constructed
    # LAST so every pre-existing parameter keeps its init (bit-pairing with the
    # control). Owns the policy: the MLP head is bypassed, so every other policy
    # owner is refused loudly (the plane's decode, fromto form, ray-context, and
    # the eval-only serve blends that would reassign policy_out at export).
    self.use_move_tokens = bool(getattr(config, 'NetDef_UseMoveTokens', False))
    self._last_mt = None
    if self.use_move_tokens:
      from move_tokens import MoveTokenDecoder
      if getattr(self, 'dp_policy_decode', False):
        raise ValueError('UseMoveTokens owns the policy: DualPlanePolicyDecode (and its candidate/'
                         'victim/edge/degree/check-chain decodes) must be off')
      if getattr(self, 'dp_cand_attn', 0) > 0 or getattr(self, 'dp_cand_value', 0) > 0:
        raise ValueError('UseMoveTokens: DualPlaneCandidateAttention/CandidateValue must be 0 '
                         '(they re-score policy_out and would read absent-move floors as candidates)')
      if self.policy_head_form != 'mlp' or self.ray_context_mode > 0:
        raise ValueError('UseMoveTokens is incompatible with PolicyHeadForm=fromto / RayContext '
                         '(two policy owners; ray-context would also add onto absent-move floors)')
      if self.opt_serve_blend > 0 or self.soft_serve_blend > 0:
        raise ValueError('UseMoveTokens: OptimisticPolicyServeBlend/SoftPolicyServeBlend must be 0 '
                         '(eval-only blend would leak MLP-head logits onto absent moves)')
      _mt_vi = bool(getattr(config, 'NetDef_MoveTokenValueInject', True))
      _mt_pb = bool(getattr(config, 'NetDef_MoveTokenPolBias', True))
      self.move_tokens = MoveTokenDecoder(
          s_dim=self.EMBEDDING_DIM, norm_type=config.NetDef_NormType,
          dm=int(getattr(config, 'NetDef_MoveTokenDim', 160) or 160),
          layers=int(getattr(config, 'NetDef_MoveTokenLayers', 3) or 3),
          heads=int(getattr(config, 'NetDef_MoveTokenHeads', 4) or 4),
          ffn_mult=int(getattr(config, 'NetDef_MoveTokenFFNMult', 2) or 2),
          max_tokens=int(getattr(config, 'NetDef_MoveTokenMax', 128) or 128),
          value_inject_dim=(64 * HEAD_MULT) if _mt_vi else 0,
          value2=self.value2_loss_weight > 0, pol_bias=_mt_pb)
      _n_mt = sum(p.numel() for p in self.move_tokens.parameters())
      print(f'[ceres_net] MOVE TOKENS enabled: M={self.move_tokens.M} candidate from-to tokens, '
            f'dm={self.move_tokens.dm}, {len(self.move_tokens.blocks)} decoder blocks '
            f'({_n_mt:,} params); policy = per-token 4-slot logits scattered to 1858 + per-move bias '
            f'(MLP policy head bypassed; its params stay in the ckpt, unused); '
            f'value inject {"on" if _mt_vi else "off"}; per-move bias {"on" if _mt_pb else "OFF (frozen zero)"}; '
            f'absent-move floor {-30.0}')


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


  def _dp_plane_E(self, vis_edge_E, squares13):
    """Boelge 13: planets relasjonskilde, med WITHHOLD-splitten.
    Returns (E_plane [B,64,64,C_plane], E_tgt [B,64,64,4W] or None).
    E_plane feeds every plane-side consumer (P-blocks, e2t, decodes); E_tgt is
    the withheld families, built only when the rel-aux loss will consume them
    (training) — never in eval/export, so the served graph is unchanged."""
    _need_tgt = self.training and getattr(self, 'dp_eaux_rel_w', 0) > 0
    if vis_edge_E is not None:
      if not getattr(self, 'dp_eaux_withhold', ()):
        return vis_edge_E, None
      _Ep = vis_edge_E.index_select(3, self.dp_plane_ch_idx)
      _Et = vis_edge_E.index_select(3, self.dp_eaux_ch_idx) if _need_tgt else None
      return _Ep, _Et
    _Ep = self.dp_vis_module(squares13)
    _Et = self.dp_eaux_vis_module(squares13) if _need_tgt else None
    return _Ep, _Et

  def _grad_scale_read(self, x):
    """Bit-eksakt grad-skala-lesing av delte plan-tokens (to review-funn bakt inn:
    2026-08-21 #12 ulp-eksakthet via x.detach()+(x-x.detach())*a; action-review #8
    inf-inf=NaN-fellen via isfinite-where). ENESTE definisjon — brukes av decode-
    kjeden og kandidat-attention (review 2026-08-25b finding 10)."""
    if self.dp_policy_grad_scale != 1.0 and self.training:
      _a = self.dp_policy_grad_scale
      _xd = x.detach()
      return torch.where(torch.isfinite(_xd), _xd + (x - _xd) * _a,
                         x * _a + _xd * (1.0 - _a))
    return x

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
    # Boelge 9: kjoer planets trunk-uavhengige fase FOER trunken og loeft de
    # levende kantene inn som attention-bias (komponerer additivt med PRB/ray;
    # konsumeres av lag-loekka via piece_relation_bias_tensor).
    _e2t_state = None
    _dp_E_tgt = None     # boelge 13: withheld-family targets [B,64,64,4W] (training + rel-aux only)
    if self.use_dual_plane and getattr(self, 'dp_edge_to_trunk', False):
      _e2t_E, _dp_E_tgt = self._dp_plane_E(vis_edge_E, squares[:, :, 0:13])
      # NB (review-funn 5): flow er enda raa squares her (fp32) — planet kjoerer
      # fase 1 i fp32-inputs under autocast, mot monolitt-stiens post-trunk-dtype;
      # ufarlig (autocast styrer matmuls), men A/B-delta mot kzedge inkluderer
      # denne mikro-presisjonsforskjellen.
      _e2t_x, _e2t_rel, _e2t_sel, _e2t_occ = self.dual_plane.run_pblocks(
          squares[:, :, 0:13].to(flow.dtype), _e2t_E)
      _e2t_state = (_e2t_x, _e2t_rel, _e2t_sel, _e2t_occ, _e2t_E)
      _S = torch.nn.functional.one_hot(_e2t_sel, 64).to(flow.dtype)          # [B,32,64]
      _bh = self.e2t_proj(_e2t_rel)                                          # [B,32,32,H]
      # To-matmul-form i stedet for 3-input-einsum (review boelge 9): TRTs
      # Einsum-lag er dokumentert maks 2 inputs — vaar stack viste seg aa
      # taale noden empirisk (gates+EPS kjoerte), men matmul-formen er
      # strengt portabel og raskere. Identisk matte: S^T @ bh @ S per hode.
      # NB (dokumentert design-avvik, review-funn 3): loeftet er UMASKERT —
      # tomme slots (skjevt hoyindeks-valgte felter) faar sine laerte
      # kant-verdier scattret til ekte feltpar naar proj trener seg vekk fra
      # null. Slik ble 1151-gevinsten MAALT; en both-empty-maskert variant er
      # flagget oppfoelgingsarm foer horisont-adopsjon.
      if getattr(self, 'dp_e2t_mask', False):
        # BOTH-EMPTY-MASK (flagget oppfoelgingsarm, review-funn 3): null kanter
        # der BEGGE slots er tomme (ren stoey paa skjevt valgte felter);
        # piece<->tomfelt beholdes (kontroll-/fluktdekning = mekanismens poeng).
        _occf = _e2t_occ.to(_bh.dtype)
        _pm = 1.0 - (1.0 - _occf).unsqueeze(2) * (1.0 - _occf).unsqueeze(1)  # [B,32,32]
        _bh = _bh * _pm.unsqueeze(-1)
      _bhp = _bh.permute(0, 3, 1, 2)                                         # [B,H,32,32]
      _b64 = torch.matmul(_S.transpose(1, 2).unsqueeze(1),
                          torch.matmul(_bhp, _S.unsqueeze(1)))               # [B,H,64,64]
      piece_relation_bias_tensor = _b64 if piece_relation_bias_tensor is None           else piece_relation_bias_tensor + _b64
    if self.use_vis_edge_bias:
      if self.vis_edge_shared:
        _vb0 = self.vis_edge_proj[0](vis_edge_E).permute(0, 3, 1, 2)
        vis_edge_biases = [_vb0] * self.NUM_DISTINCT_LAYERS
      elif (not self.training
            and all(type(_l) is torch.nn.Linear and _l.bias is None
                    for _l in self.vis_edge_proj)):
        # Export/eval serving graph: all per-layer projections in ONE matmul
        # (weights concatenated along the output dim — constant-folded at
        # export) + one permute, instead of NUM_DISTINCT_LAYERS separate
        # matmul+permute materializations. Guarded to plain bias-less Linear —
        # reading .weight directly bypasses module forwards, so any wrapper
        # (QAT FakeQuantLinear, LoRA, hooks, a future bias) must take the loop
        # below. Training also takes the loop: unbind would keep all L grads
        # buffered through trunk backward, and the loop's per-layer tensors
        # are independently mutable (the unbound views below all alias one
        # storage — never write them in place).
        _W = torch.cat([_lin.weight for _lin in self.vis_edge_proj], dim=0)  # [L*H, C]
        _vb = torch.matmul(vis_edge_E, _W.transpose(0, 1))                   # [B,64,64,L*H]
        _vb = _vb.unflatten(-1, (self.NUM_DISTINCT_LAYERS,
                                 self.NUM_HEADS)).permute(0, 3, 4, 1, 2)     # [B,L,H,64,64]
        vis_edge_biases = list(_vb.unbind(1))
      else:
        vis_edge_biases = [self.vis_edge_proj[i](vis_edge_E).permute(0, 3, 1, 2)
                           for i in range(self.NUM_DISTINCT_LAYERS)]

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
      append_tensor = prior_state if prior_state is not None else torch.zeros(squares.shape[0], NUM_TOKENS_INPUT, self.prior_state_dim).to(flow.device).to(flow.dtype)  # foelg compute-dtype, ikke hardkodet bf16 (bugfunn 2026-08-28)
      append_tensor = append_tensor.reshape(squares.shape[0], NUM_TOKENS_INPUT, self.prior_state_dim)
      flow_squares = torch.cat((flow_squares, append_tensor), dim=-1)

    flow = self.embedding_layer(flow_squares)
    flow = self.embedding_norm(flow)

    # King-centric distance channels (see __init__): king one-hot planes
    # (6 = STM king, 12 = opponent king, STM-relative encoding) against the
    # constant bucket table; zero-init projection added post-embedding.
    if self.use_king_dist:
      _kt = self.kdist_table.to(flow.dtype)
      _kd_own = torch.matmul(squares[:, :, 6].to(flow.dtype), _kt).reshape(-1, 64, 8)
      _kd_opp = torch.matmul(squares[:, :, 12].to(flow.dtype), _kt).reshape(-1, 64, 8)
      flow = flow + self.kdist_proj(torch.cat([_kd_own, _kd_opp], dim=-1))

    # Spectral PE (see __init__). Type-presence gates from the one-hot planes
    # (1..6 = P,N,B,R,Q,K white; 7..12 mirror): N block gated by knights,
    # K by kings, R by rooks+queens, B by bishops+queens.
    if self.use_spectral_pe:
      _sq = squares.to(flow.dtype)
      _g = torch.stack([
        _sq[:, :, 2] + _sq[:, :, 8],                       # knights
        _sq[:, :, 6] + _sq[:, :, 12],                      # kings
        _sq[:, :, 4] + _sq[:, :, 10] + _sq[:, :, 5] + _sq[:, :, 11],  # rooks + queens
        _sq[:, :, 3] + _sq[:, :, 9] + _sq[:, :, 5] + _sq[:, :, 11],   # bishops + queens
      ], dim=-1)                                            # [B, 64, 4]
      _pe = self.spe_table.to(flow.dtype).reshape(64, 4, 8)  # [64, 4, 8]
      _gated = (_g.unsqueeze(-1) * _pe.unsqueeze(0)).reshape(-1, 64, 32)
      flow = flow + self.spe_proj(_gated)

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
                                         vis_edge=vis_edge_E if (self.vis_edge_gate_mode or self.use_graph_route) else None)
        if self.denseformer:
          eff_idx = loop_iter * self.NUM_DISTINCT_LAYERS + i
          all_previous_x.append(flow)
          flow = self.dwa_modules[eff_idx](all_previous_x)
        if self.use_value_depth_attention or self.pda_mode or (self.depth_probes_enabled and self.training):
          vda_states.append(flow)  # post-layer (post-DWA if denseformer)

    # Pre-norm: final norm before the heads (see __init__ comment).
    if self.trunk_end_norm is not None:
      flow = self.trunk_end_norm(flow)

    # Iterated tactic refiner (see __init__): residual add of the weight-shared
    # iterated block. Runs BEFORE GTAB/vda/heads so every consumer reads the
    # refined square states. Deep supervision: intermediate iterations' policy
    # logits through the plain shared front-end + base policy head, stashed
    # for compute_loss (training-only attribute mutation — the established
    # export-safe stash pattern).
    if self.refiner_iters > 0:
      _rf_collect = self.training and self.refiner_deep_sup_weight > 0 and self.refiner_iters > 1
      _rf_final, _rf_inter = self.tactical_refiner(flow, collect_intermediate=_rf_collect)
      if _rf_collect and _rf_inter:
        _rf_aux = []
        for _rf_d in _rf_inter:
          _rf_fS = self.headSharedLinear(self.headPremap(flow + _rf_d)
                                         .reshape(-1, 64 * self.HEAD_PREMAP_PER_SQUARE))
          _rf_aux.append(self.policy_head(_rf_fS))
        self._last_refiner_policy = torch.stack(_rf_aux, dim=1)   # [B, T-1, 1858]
      flow = flow + _rf_final

    # Tactical codebook (see __init__): one cross-attn read of the motif
    # library, residual-added so every head sees the enriched states.
    # Plain matmul+softmax — TRT-safe, fp32 for the small attention math.
    if self.use_tactical_codebook:
      _cx = self.cbk_norm(flow).float()
      _cs = torch.matmul(self.cbk_q(_cx), self.cbk_keys.t()) * (self.cbk_keys.shape[1] ** -0.5)
      _ca = torch.softmax(_cs, dim=-1)                       # [B, 64, 256]
      flow = flow + self.cbk_out(torch.matmul(_ca, self.cbk_vals)).to(flow.dtype)

    # TSB: collect per-block gate values across all layers for the gate-sparsity
    # regularizer in train.py. Each layer caches its last gate in _last_tsb_gate.
    if self.use_tsb:
      _gates = [layer._last_tsb_gate for layer in self.transformer_layer
                if getattr(layer, '_last_tsb_gate', None) is not None]
      if len(_gates) > 0:
        self._last_tsb_gates = torch.stack(_gates, dim=0) if self.training else None  # training-gated (dynamo-eksport avviser attributt-mutasjon; bugfunn 2026-08-28)  # [num_layers, B, 1, 1]
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
      self._last_gate_value = g_x if self.training else None  # training-gated (bugfunn 2026-08-28)                          # cached for sparsity loss + diagnostics
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

    # DUAL-PLANE (see __init__): piece-plane summary into the value family's
    # hidden pre-activation. Reads the ONE-HOT slice of squares, the shared
    # vis_edge_E relation tensor (computed once per forward above) and the
    # FINAL square flow. Composes additively with the other injects.
    _dp_tokens = None
    _dp_E_dec = None
    _dp_deg_cache = None    # (indeg, outdeg) delt mellom dpd og cand-value (review finding 9)
    if self.use_dual_plane:
      if _e2t_state is not None:
        _e2t_x, _e2t_rel, _dp_sel, _dp_occ, _dp_E = _e2t_state
        _dp_E_dec = _dp_E
        _dp_rel_final = _e2t_rel
        _dp_pool, _dp_tokens, _dp_sel, _dp_occ = self.dual_plane.finish(_e2t_x, flow, _dp_sel, _dp_occ)
      else:
        _dp_E, _dp_E_tgt = self._dp_plane_E(vis_edge_E, squares[:, :, 0:13])
        _dp_E_dec = _dp_E   # kept for the move-edge decode (shared or private source)
        _dp_pool, _dp_tokens, _dp_sel, _dp_occ, _dp_rel_final = self.dual_plane(
            squares[:, :, 0:13].to(flow.dtype), _dp_E, flow)
      # EDGE-AUX stash (boelge 13, training-only; consumed in compute_loss):
      # readout on the FINAL edge state (the tensor e2t lifts), plus the
      # withheld-family targets double-gathered to the same slot pairs.
      if self.training and getattr(self, 'dp_eaux_on', False):
        _ea_rel = _dp_rel_final.detach() if self.dp_eaux_detach else _dp_rel_final
        _ea_logits = torch.nn.functional.linear(_ea_rel.float(), self.dp_eaux_w, self.dp_eaux_b)   # [B,32,32,T]
        _ea_tgt = None
        if _dp_E_tgt is not None:
          _Bt, _Ct = _dp_E_tgt.shape[0], _dp_E_tgt.shape[-1]
          _ea_rows = torch.gather(_dp_E_tgt, 1, _dp_sel.reshape(_Bt, 32, 1, 1).expand(-1, -1, 64, _Ct))
          _ea_tgt = torch.gather(_ea_rows, 2, _dp_sel.reshape(_Bt, 1, 32, 1).expand(-1, 32, -1, _Ct)).float()
        self._last_dp_eaux = (_ea_logits, _dp_sel, _dp_occ, _ea_tgt)
      if not getattr(self, 'dp_no_pool_injects', False):
        _dpv = self.dp_value_inject(_dp_pool)
        if getattr(self, 'dp_gated_injects', False):
          _dpv = _dpv * torch.sigmoid(self.dpgi_v(fS_value))
        _v_inject = _dpv if _v_inject is None else _v_inject + _dpv
        if self.value2_loss_weight > 0:
          _dpv2 = self.dp_value2_inject(_dp_pool)
          if getattr(self, 'dp_gated_injects', False):
            _dpv2 = _dpv2 * torch.sigmoid(self.dpgi_v2(fS_value))
          _v2_inject = _dpv2 if _v2_inject is None else _v2_inject + _dpv2
      # Per-piece survival aux stash (training-only; consumed in compute_loss).
      if self.training and getattr(self, 'dp_surv_weight', 0) > 0:
        self._last_dp_surv = (self.dp_surv_head(_dp_tokens), _dp_sel, _dp_occ)
      # Value-attention read (see __init__): fS_value-conditioned queries over
      # the piece tokens; empty slots masked as keys.
      if getattr(self, 'dp_value_attn', 0) > 0:
        _vq = self.dpva_q(fS_value).reshape(-1, self.dp_value_attn, 64)
        _vk = self.dpva_k(_dp_tokens)
        _vv = self.dpva_v(_dp_tokens)
        _vsc = torch.matmul(_vq, _vk.transpose(1, 2)) * (64 ** -0.5)
        _vsc = _vsc + ((_dp_occ.to(_vsc.dtype) - 1.0) * 1e4).unsqueeze(1)
        _vat = torch.matmul(torch.softmax(_vsc, dim=-1), _vv).reshape(-1, self.dp_value_attn * 64)
        _vai = self.dpva_out(_vat)
        _v_inject = _vai if _v_inject is None else _v_inject + _vai
        if self.value2_loss_weight > 0:
          _vai2 = self.dpva_out2(_vat)
          _v2_inject = _vai2 if _v2_inject is None else _v2_inject + _vai2

    # VALUE MIN/MAX POOL (see __init__): extreme-square summaries into the
    # value family's hidden pre-activation. Composes additively with the
    # private-front-end inject above; amin/amax export as ReduceMin/ReduceMax.
    if self.value_minmax_pool:
      # min/max(dim).values instead of amin/amax: the amin+amax+cat pattern
      # produced a pathological inductor kernel on CUDA (host-side TDR resets /
      # in-WSL infinite spin, 2026-08-19 — three vmp launches: two
      # cudaErrorUnknown crashes at step ~1, one 100%-CPU soft hang in
      # compile). torch.aminmax was tried first but has no autograd derivative
      # (torch 2.7). min/max lower to a different (values+indices) codegen
      # path; forward math is identical, backward routes grad to the arg
      # element instead of splitting among ties — fine for this purpose.
      _mn = flow.min(dim=1).values
      _mx = flow.max(dim=1).values
      _pool = torch.cat([_mn, _mx], dim=-1)
      _vp = self.value_pool_inject(_pool)
      _v_inject = _vp if _v_inject is None else _v_inject + _vp
      if self.value2_loss_weight > 0:
        _v2p = self.value2_pool_inject(_pool)
        _v2_inject = _v2p if _v2_inject is None else _v2_inject + _v2p

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

    # Opponent-policy aux head (see __init__): training-only stash.
    if self.opp_policy_weight > 0 and self.training:
      self._last_oppp_out = self.oppp_head(fS_policy)

    # Heads. Policy reads fS_policy (== fS_others unless pda); value reads fS_value (with adapter).
    if getattr(self, 'policy_head_form', 'mlp') == 'fromto':
      # MLP head bypassed (see __init__): the from-to bilinear below is the head.
      # Zeros keep every downstream additive term (plane decode, ray-context)
      # unchanged; the dead MLP branch is pruned from the export graph.
      policy_out = torch.zeros(fS_policy.shape[0], 1858, device=fS_policy.device, dtype=fS_policy.dtype)
    elif getattr(self, 'use_move_tokens', False):
      # MOVE TOKENS own the policy (see __init__ / move_tokens.py). `flow` is the
      # post-trunk-norm [B,64,D] square state; the decoder builds its candidate
      # set in-graph from the one-hot slice.
      _mt_pol, _mt_pool, _mt_stats, _mt_sel, _mt_valid = self.move_tokens(
          squares[:, :, 0:13].to(flow.dtype), flow)
      policy_out = _mt_pol.to(fS_policy.dtype)
      if self.move_tokens.value_inject_dim > 0:
        _mvi = self.move_tokens.v_inject(_mt_pool)
        _v_inject = _mvi if _v_inject is None else _v_inject + _mvi
        if self.value2_loss_weight > 0:
          _mvi2 = self.move_tokens.v2_inject(_mt_pool)
          _v2_inject = _mvi2 if _v2_inject is None else _v2_inject + _mvi2
      if self.training:
        self._last_mt = (_mt_sel, _mt_valid, _mt_stats)
    else:
      policy_out = self.policy_head(fS_policy)

    # Dual-plane mover-bilinear decode (Stage A3, see __init__): pair scores
    # destination-square x piece-slot, mapped to from-squares via the slot
    # one-hot, then flat-gathered per move on the constant to*64+from table.
    if self.use_dual_plane and self.dp_policy_decode and _dp_tokens is not None:
      # POLICY-GRAD SCALE (2026-08-21, server value-oscillation diagnosis):
      # the P-plane is a SMALL pool shared by the policy decode (strong,
      # committing gradients) and the value injects (weak) — policy
      # continuously reshapes the representation value reads, giving value
      # its peak-then-oscillate profile while baseline (trunk-only value)
      # stays sluggish-but-stable. This op is forward-IDENTITY but scales
      # the policy-decode gradients flowing INTO the shared tokens by alpha
      # (<1 damps policy's reshaping power; decode WEIGHTS still train at
      # full rate; value gradients untouched). Eval/export graphs are
      # unchanged (gated on self.training).
      _dpt = self._grad_scale_read(_dp_tokens)   # se _grad_scale_read (hoisted)
      _q = self.dp_pol_q(flow)                                   # [B, 64, dq] (zero-init => 0)
      _p = self.dp_pol_p(_dpt) * _dp_occ.unsqueeze(-1).to(_dp_tokens.dtype)  # [B, 32, dq]
      _pair = torch.matmul(_q, _p.transpose(1, 2))               # [B, 64to, 32slot]
      _sl1h = torch.nn.functional.one_hot(_dp_sel, 64).to(_pair.dtype)  # [B, 32, 64from]
      _tofrom = torch.matmul(_pair, _sl1h)                       # [B, 64to, 64from]
      _corr = _tofrom.reshape(-1, 4096).index_select(1, self.dp_move_flat)  # [B, 1858]
      policy_out = policy_out + _corr.to(policy_out.dtype)

      # Attacker×victim decode (see __init__): mover-token × to-square-token
      # bilinear, mapped slots -> (from, to) squares via the slot one-hot.
      # Empty to-squares contribute exactly 0 (unselected/occ-masked slots).
      if getattr(self, 'dp_victim_decode', False):
        _occm = _dp_occ.unsqueeze(-1).to(_dp_tokens.dtype)
        _pa = self.dpv_a(_dpt) * _occm                           # [B, 32, dq] (grad-scaled read)
        _pb = self.dpv_b(_dpt) * _occm
        _pp = torch.matmul(_pa, _pb.transpose(1, 2))             # [B, 32m, 32v]
        _ftsq = torch.matmul(_sl1h.transpose(1, 2),
                             torch.matmul(_pp.to(_sl1h.dtype), _sl1h))  # [B, 64from, 64to]
        _corr2 = _ftsq.reshape(-1, 4096).index_select(1, self.dp_move_flat_ft)
        policy_out = policy_out + _corr2.to(policy_out.dtype)

      # Move-edge decode (see __init__): gather the move's own (from, to)
      # relation-edge channels from the shared E tensor into the move score.
      if getattr(self, 'move_edge_decode', False) and _dp_E_dec is not None:
        _C_e = _dp_E_dec.shape[-1]
        _em = _dp_E_dec.reshape(-1, 4096, _C_e).index_select(1, self.dp_move_flat_ft)
        _corr3 = self.dpe_w(_em.float()).squeeze(-1)             # [B, 1858]
        policy_out = policy_out + _corr3.to(policy_out.dtype)

      # Move-degree decode (see __init__): square-level degree scores gathered
      # per move at the destination (in-degree) and origin (out-degree).
      if getattr(self, 'move_degree_decode', False) and _dp_E_dec is not None:
        _indeg = _dp_E_dec.sum(dim=1).float()  # sum i kilde-dtype, cast det LILLE resultatet (kf-v1-regelen)                    # [B, 64, C]  edges INTO j
        _outdeg = _dp_E_dec.sum(dim=2).float()                   # [B, 64, C]  edges FROM i (kf-v1-regelen, runde-4: ogsaa denne)
        _dp_deg_cache = (_indeg, _outdeg)
        _s_in = self.dpd_in(_indeg).squeeze(-1)                  # [B, 64]
        _s_out = self.dpd_out(_outdeg).squeeze(-1)               # [B, 64]
        _corr4 = (_s_in.index_select(1, self.dp_move_to)
                  + _s_out.index_select(1, self.dp_move_from))   # [B, 1858]
        policy_out = policy_out + _corr4.to(policy_out.dtype)

      # Check-chain decode (see __init__): per-move composed channels.
      if getattr(self, 'dp_check_chain', False) and _dp_E_dec is not None:
        _chk = _dp_E_dec[..., self.dp_ch_check].float()            # [B, 64, 64] vaare sjakk-edges
        _flc = _dp_E_dec[..., self.dp_ch_flight].float().amax(dim=1)  # [B, 64] frie fluktfelter (kolonne)
        _nfl = _flc.sum(dim=-1, keepdim=True)                      # [B, 1] antall frie fluktfelter
        _cmv = _chk.reshape(-1, 4096).index_select(1, self.dp_move_from * 64 + self.dp_move_to)  # [B, 1858]
        _f1 = _cmv                                                 # gir sjakk
        _f2 = _cmv * (_nfl == 0).float()                           # sjakk + matt-nett komplett (probens signal)
        _f3 = _cmv * _flc.index_select(1, self.dp_move_to)         # sjakk som LANDER paa fluktfelt
        _f4 = _cmv * (_nfl / 8.0)                                  # gradert aapenhet (forventet negativ vekt)
        _corr5 = self.dpch_w(torch.stack([_f1, _f2, _f3, _f4], dim=-1)).squeeze(-1)
        policy_out = policy_out + _corr5.to(policy_out.dtype)


    # Candidate selection source (review finding 5): UNBLENDED logits — the serve
    # blend below reassigns policy_out at eval, and selection must match training.
    _pol_cand_src = policy_out
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
      _pol_cand_src = _pol_cand_src + _rc_add   # selection sees rc (review finding 6)
    # --- Shared candidate machinery (review finding 10: ONE definition) -------
    # _cand_base(pol_src, K, tok_read): masked-topk selection (detached) +
    # per-candidate gathers shared by cand-attn and cand-value. tok_read chooses
    # the token view: grad-scaled _dpt for the POLICY-side consumer (finding 4),
    # raw _dp_tokens for the VALUE-side consumer (value grads are unscaled by
    # design). Also returns a validity mask: with K above the pseudo-legal count
    # the -1e4-masked garbage rows must not participate (finding 8).
    if (getattr(self, 'dp_cand_attn', 0) > 0 or getattr(self, 'dp_cand_value', 0) > 0)        and _dp_tokens is not None:
      _occb = _dp_occ.unsqueeze(-1).to(_dp_tokens.dtype)
      _sl1b = torch.nn.functional.one_hot(_dp_sel, 64).to(_dp_tokens.dtype)
      _dpnb = _dp_tokens.shape[-1]
      _Ceb = _dp_E_dec.shape[-1]
      _ourb = squares[:, :, 1:7].sum(-1).float()
      def _cand_base(pol_src, K, tok_read):
        with torch.no_grad():
          _fok = _ourb.index_select(1, self.dp_move_from)
          _cl = pol_src.float() + (_fok - 1.0) * 1e4
          _cw, _ci = torch.topk(_cl, K, dim=1)
          _val = (_cw > -5e3).float()                               # ekte kandidat?
          _cwf = (_cw / 10.0) * _val                                # logit-feature; garbage->0 (fp16-overflow ved serve, review 2026-08-25b finding 4)
        _cf = self.dp_move_from[_ci]
        _ct = self.dp_move_to[_ci]
        _tsq = torch.matmul(_sl1b.transpose(1, 2), tok_read * _occb)
        _mt = torch.gather(_tsq, 1, _cf.unsqueeze(-1).expand(-1, -1, _dpnb))
        _em = torch.gather(_dp_E_dec.reshape(-1, 4096, _Ceb), 1,
                           (_cf * 64 + _ct).unsqueeze(-1).expand(-1, -1, _Ceb))
        return _ci, _cw, _cwf, _val, _cf, _ct, _mt, _em

    # Candidate-attention re-score (see __init__). Placed AFTER the rc add so
    # selection sees every policy correction (finding 6), selecting on the
    # UNBLENDED source (finding 5); reads the GRAD-SCALED token view so
    # DualPlanePolicyGradScale keeps its promise on this path too (finding 4).
    if getattr(self, 'dp_cand_attn', 0) > 0 and _dp_tokens is not None:
      _Ka = self.dp_cand_attn
      _ci5, _cw5, _cwf5, _val5, _cf5, _ct5, _mt5, _em5 = _cand_base(_pol_cand_src, _Ka, _dpt)
      _feats5 = torch.cat([squares[:, :, 0:13].to(_dp_tokens.dtype),
                           self.dual_plane.filerank.to(_dp_tokens.dtype).unsqueeze(0)
                               .expand(squares.shape[0], 64, 16)], dim=-1)
      _tosq5 = self._grad_scale_read(self.dual_plane.embed(torch.gather(
          _feats5, 1, _ct5.unsqueeze(-1).expand(-1, -1, 29))))
      _cft5 = torch.cat([_mt5, _tosq5, _em5.to(_dp_tokens.dtype),
                         _cwf5.unsqueeze(-1).to(_dp_tokens.dtype)], dim=-1)
      _ctok = torch.nn.functional.mish(self.dpc_embed(_cft5))
      _pk5 = self.dpc_piece(_dpt)
      _kv_in = torch.cat([_ctok, _pk5], dim=1)
      _q5 = self.dpc_q(_ctok).reshape(-1, _Ka, 2, 32).transpose(1, 2)
      _k5 = self.dpc_k(_kv_in).reshape(-1, _Ka + 32, 2, 32).transpose(1, 2)
      _v5 = self.dpc_v(_kv_in).reshape(-1, _Ka + 32, 2, 32).transpose(1, 2)
      _sc5 = torch.matmul(_q5, _k5.transpose(-1, -2)) * (32 ** -0.5)
      _msk5 = torch.cat([_val5, _dp_occ.float()], dim=1)           # garbage-kandidater maskes som keys (finding 8)
      _sc5 = _sc5 + ((_msk5 - 1.0) * 1e4).reshape(-1, 1, 1, _Ka + 32).to(_sc5.dtype)
      _at5 = torch.matmul(torch.softmax(_sc5, dim=-1), _v5)
      _at5 = _at5.transpose(1, 2).reshape(-1, _Ka, 64)
      _ctok2 = _ctok + self.dpc_out(_at5)
      _rs5 = self.dpc_score(_ctok2).squeeze(-1) * _val5.to(_ctok2.dtype)  # garbage far 0 (finding 8)
      _cadd = torch.zeros_like(policy_out).scatter(1, _ci5, _rs5.to(policy_out.dtype))
      policy_out = policy_out + _cadd
      _pol_cand_src = _pol_cand_src + _cadd    # cand-value ser re-scoren

    # VALUE POOL CHANNELS (see __init__): concat the extreme-square summaries
    # onto the value family's head input. Separate variable — fS_value itself
    # is shared with unc/other heads and must keep its width. Same
    # min/max(dim).values formulation as the vmp block (aminmax has no
    # autograd derivative in torch 2.7; amin/amax+cat hit a pathological
    # inductor kernel, 2026-08-19).
    if self.value_pool_channels:
      _poolc = torch.cat([flow.min(dim=1).values, flow.max(dim=1).values], dim=-1)
      fS_value_v = torch.cat([fS_value, _poolc], dim=-1)
    else:
      fS_value_v = fS_value
    # CANDIDATE-VALUE READ (see __init__). Selection on the UNBLENDED candidate
    # source (post-rc, incl. the cand-attn re-score; review findings 5-6), via the
    # shared _cand_base helper (finding 10) with the RAW token view — value
    # gradients into the plane tokens are unscaled by design.
    if getattr(self, 'dp_cand_value', 0) > 0 and _dp_tokens is not None:
      _Kc = self.dp_cand_value
      _ci4, _cw4, _cwf4, _val4, _cf4, _ct4, _mt4, _em4 = _cand_base(_pol_cand_src, _Kc, _dp_tokens)
      with torch.no_grad():
        _cwt4 = torch.softmax(_cw4 / 2.0, dim=1)                  # T=2; garbage underflows til ~0
      if _dp_deg_cache is not None:                                # gjenbruk dpd-summene (finding 9)
        _ind4, _outd4 = _dp_deg_cache
      else:
        _ind4 = _dp_E_dec.sum(dim=1).float()  # kf-v1-regelen: ikke materialiser [B,64,64,C] i fp32
        _outd4 = _dp_E_dec.float().sum(dim=2)
      _Ce4 = _dp_E_dec.shape[-1]
      _into4 = torch.gather(_ind4, 1, _ct4.unsqueeze(-1).expand(-1, -1, _Ce4))
      _outf4 = torch.gather(_outd4, 1, _cf4.unsqueeze(-1).expand(-1, -1, _Ce4))
      _cfeat = torch.cat([_mt4.float(), _em4.float(), _into4, _outf4,
                          _cwf4.unsqueeze(-1)], dim=-1)
      _cemb = torch.nn.functional.mish(self.dpcv_embed(_cfeat.to(_dp_tokens.dtype)))
      _csum = (_cemb * _cwt4.unsqueeze(-1).to(_cemb.dtype)).sum(dim=1)
      _cvi = self.dpcv_out(_csum).to(fS_value.dtype)
      _v_inject = _cvi if _v_inject is None else _v_inject + _cvi
      if self.value2_loss_weight > 0:
        _cvi2 = self.dpcv_out2(_csum).to(fS_value.dtype)
        _v2_inject = _cvi2 if _v2_inject is None else _v2_inject + _cvi2
    value_out = self.value_head(fS_value_v, _v_inject)
    value2_out = self.value2_head(torch.cat((fS_value_v, qblunders_negative_positive), -1), _v2_inject) if self.value2_loss_weight > 0 else value_out
    unc_out = self.unc_head(fS_value if self.value_priv_replace else fS_others)
    unc_policy_out = self.unc_policy(fS_others) if self.uncertainty_policy_weight > 0 else unc_out # unc_out is just a dummy so not None

    _want_action = self.action_loss_weight > 0 or getattr(self, 'action_played_weight', 0) > 0
    # Export strip (set by save_model via CERES_EXPORT_STRIP_ACTION): in eval
    # mode emit the cheap unc alias instead, so the exported graph omits the
    # [B,1858,3] head (~9-12% TRT EPS) while training still uses it.
    if getattr(self, 'export_strip_action', False) and not self.training:
      _want_action = False
    action_out             = self.action_head(fS_others).reshape(-1, 1858, 3) if _want_action else unc_out
    if getattr(self, 'action_played_weight', 0) > 0 and self.training:
      self._last_actionp_out = action_out
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
      # Runde-3-fiks 2026-08-29: proben zero_grad()-er per hode — midt i et
      # akkumuleringsvindu visket den ut mikro-batch 1..k-1 og ga et stille
      # skjevt optimizer-steg paa hvert probet intervall. Snapshot/restore av
      # akkumulerte gradienter goer proben eksakt ikke-destruktiv (probe-modusen
      # er uansett dokumentert som kort maalemodus paa singel GPU).
      _acc_grads = {n: p.grad.detach().clone() for n, p in self.named_parameters()
                    if p.grad is not None}
      self.compute_loss_or_gradnorm(loss_calc, batch, policy_out, value_out, moves_left_out, unc_out,
                                    value2_out, q_deviation_lower_out, q_deviation_upper_out, uncertainty_policy_out,
                                    prior_value_out, prior_value2_out,
                                    action_target, action_out, action_uncertainty_out,
                                    multiplier_action_loss,
                                    num_pos, last_lr, log_stats, gradient_norm_logging_mode = True)
      if _acc_grads:
        for _n, _p in self.named_parameters():
          if _n in _acc_grads:
            _p.grad = _acc_grads[_n]
       
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

    # Policy MARGIN loss (see __init__): target-argmax skal staa margin-klar av rivalene.
    policy_margin_loss = 0
    if self.policy_margin_weight > 0 and policy_out is not None and not gradient_norm_logging_mode:
      _pm_t = policy_target.float()
      _pm_conf, _pm_best = _pm_t.max(dim=1)                       # [B] target-skarphet + solver-slot
      _pm_logits = policy_out.float()
      _pm_best_logit = _pm_logits.gather(1, _pm_best.unsqueeze(1)).squeeze(1)
      # Rivaler kun blant LOVLIGE trekk (bugfunn 2026-08-28): CE presser aldri
      # illegale logits ned (de er maskert i policy_loss), saa uten denne masken
      # kunne hinge-budsjettet lekke til aa presse utrente illegale logits i
      # stedet for aa skjerpe det lovlige gapet E1 motiverte tapet med.
      # target > 0 = samme lovlighets-proxy som policy_loss bruker.
      _pm_illegal = _pm_t <= 0
      _pm_rivals = _pm_logits.masked_fill(_pm_illegal, float('-inf'))                              .scatter(1, _pm_best.unsqueeze(1), float('-inf')).topk(4, dim=1).values
      # < 5 lovlige trekk => -inf-rivaler; hinge blir eksakt 0 der (relu av -inf-gap).
      _pm_rivals = _pm_rivals.clamp_min(-1e9)
      _pm_h = torch.relu(self.policy_margin_value - (_pm_best_logit.unsqueeze(1) - _pm_rivals)).mean(dim=1)
      policy_margin_loss = (_pm_conf.detach() * _pm_h).mean()

    # Policy PLACKETT-LUCE ranking loss (see __init__): ListMLE over the target's
    # top-K order. Legality proxy = target > 0 (same as policy_loss); illegal
    # moves get -inf so they never enter a suffix. Positions with fewer than K
    # legal moves contribute only their legal ranks. Ties inside the target's
    # top-K are broken by argsort order (stable enough: visit counts rarely tie).
    policy_pl_loss = 0
    _pl_log = {}
    if self.policy_pl_weight > 0 and policy_out is not None and not gradient_norm_logging_mode:
      _pl_t = policy_target.float()
      _pl_logits = policy_out.float().masked_fill(_pl_t <= 0, float('-inf'))
      _pl_order = torch.argsort(_pl_t, dim=1, descending=True)                     # target rank order
      _pl_s = torch.gather(_pl_logits, 1, _pl_order)                               # logits in target order
      # suffix logsumexp: lse_{j>=k} = flip(logcumsumexp(flip(s)))
      _pl_suf = torch.flip(torch.logcumsumexp(torch.flip(_pl_s, dims=[1]), dim=1), dims=[1])
      _K = self.policy_pl_topk
      _pl_terms = (_pl_suf[:, :_K] - _pl_s[:, :_K])                                # [B,K] >= 0
      _pl_valid = torch.gather(_pl_t, 1, _pl_order)[:, :_K] > 0                    # rank k has target mass
      _pl_terms = torch.where(_pl_valid, _pl_terms, torch.zeros_like(_pl_terms))
      policy_pl_loss = _pl_terms.sum(dim=1).mean()
      with torch.no_grad():
        # ordering diagnostics: does the net's top-1 / top-K set match the target's?
        _pl_pred = torch.topk(_pl_logits, _K, dim=1).indices
        _pl_log['policy_pl_top1'] = (_pl_pred[:, 0] == _pl_order[:, 0]).float().mean()
        _pl_tgt_set = _pl_order[:, :_K]
        _pl_hits = (_pl_pred.unsqueeze(2) == _pl_tgt_set.unsqueeze(1)).any(dim=2).float().sum(dim=1)
        _pl_log['policy_pl_topk_overlap'] = (_pl_hits / _K).mean()

    # Value CONTRAST aux (see __init__): per-move WDL CE — solution move keeps
    # the record's WDL, every other legal move is labeled LOSS-for-STM.
    # The loss-vector label (not a flip of the record WDL) is correct across
    # all three record classes in solution-line-expanded puzzle data: winning
    # side deviates -> ~loss (sharp positions); drawish DEFENSE puzzles (~5%,
    # user obs.) -> failing the only defense loses (a flip of a draw would
    # wrongly stay a draw); defender-side records in the line -> every move
    # still loses (a flip would wrongly label them WINS). Non-solution moves
    # are the heuristic side of the label, so they get 1/4 weight.
    # NB (runde-3, dokumentert avvik): vc-CE trekker IKKE fra target-entropien
    # slik naboene gjoer — logget verdi flyter derfor med korpusets remisrate.
    # Gradienten er upaavirket; les vc kun innen samme korpus.
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
    # Played-move action loss (see __init__): soft-CE between the exported
    # action head's WDL at the PLAYED slot and the v7-derived after-move WDL.
    # Invalid targets (-1: last ply, NO_MOVE, non-v7 batches) are masked; a
    # batch with zero valid targets contributes exactly 0.
    actp_loss = 0
    _actp_participation_only = False
    _actp = getattr(self, '_last_actionp_out', None)
    if _actp is not None and not gradient_norm_logging_mode:
      self._last_actionp_out = None
      _apidx = batch.get('action_played_idx', None)
      if _apidx is not None and not isinstance(_actp, int):
        _apidx = _apidx.reshape(-1).long()
        _aq = batch['action_q_after'].reshape(-1).float()
        _ad = batch['action_d_after'].reshape(-1).float()
        _avalid = (_apidx >= 0) & (_apidx < 1858)
        if bool(_avalid.any()):
          _aw = ((1.0 + _aq - _ad) * 0.5).clamp(0, 1)
          _al = ((1.0 - _aq - _ad) * 0.5).clamp(0, 1)
          _add = (1.0 - _aw - _al).clamp(0, 1)
          _tgt = torch.stack([_aw, _add, _al], dim=1)[_avalid]
          _rows = torch.arange(_actp.size(0), device=_actp.device)[_avalid]
          _pred = _actp[_rows, _apidx[_avalid]]
          actp_loss = torch.nn.functional.cross_entropy(_pred.float(), _tgt)
          if SUBTRACT_ENTROPY:
            # House convention (action review finding 11): subtract the soft
            # target's entropy so the logged number is a true KL — otherwise
            # the WDL target entropy (~0.9-1.0) dominates the scalar and a
            # d-distribution shift between corpora reads as a regression.
            actp_loss = actp_loss + (_tgt * _tgt.clamp_min(1e-12).log()).sum(-1).mean()
        else:
          # DDP participation term (see the survival template above): keeps
          # action_head in the backward's used-parameter set on all-invalid
          # batches; exact-zero gradient, excluded from logging.
          actp_loss = 0.0 * _actp.float().sum()
          _actp_participation_only = True
      else:
        # Batch without action targets (e.g. TPG secondary in a mixed run).
        actp_loss = 0.0 * _actp.float().sum()
        _actp_participation_only = True

    # Opponent-policy aux (see __init__): CE against the opponent's reply
    # move; -1 targets (no reply / batches without v7 data) are masked, and
    # a batch with zero valid targets contributes exactly 0.
    oppp_loss = 0
    _oppp_participation_only = False
    _oppp = getattr(self, '_last_oppp_out', None)
    if _oppp is not None and not gradient_norm_logging_mode:
      self._last_oppp_out = None
      _ot = batch.get('opp_played_idx', None)
      if _ot is not None:
        _ot = _ot.reshape(-1).long()
        _ovalid = (_ot >= 0) & (_ot < 1858)
        if bool(_ovalid.any()):
          oppp_loss = torch.nn.functional.cross_entropy(
              _oppp.float()[_ovalid], _ot[_ovalid])
        else:
          # DDP participation term (review 2026-08-21 finding 2): all--1 batch.
          oppp_loss = 0.0 * _oppp.float().sum()
          _oppp_participation_only = True
      else:
        # Batch without the key (TPG sidecar 3-tuple / mixed-run secondary):
        # zero-weighted read keeps oppp_head in the used-parameter set so
        # static_graph does not see a shrinking graph (finding 2).
        oppp_loss = 0.0 * _oppp.float().sum()
        _oppp_participation_only = True

    # MOVE-TOKEN diagnostics (training-only stash from forward): candidate count,
    # truncation rate, and the number that matters — how often the target's
    # argmax move has NO token (it would then sit at the -30 floor).
    _mt_log = {}
    _mt = getattr(self, '_last_mt', None)
    if _mt is not None and not gradient_norm_logging_mode:
      self._last_mt = None
      _mt_sel, _mt_valid, _mt_stats = _mt
      with torch.no_grad():
        _tgt_pair = self.move_tokens.mv_pair_flat[policy_target.argmax(dim=1)]          # [B]
        _hit = ((_mt_sel == _tgt_pair.unsqueeze(1)) & _mt_valid).any(dim=1).float()
        _mt_log['mt_missing_target_rate'] = 1.0 - _hit.mean()
        for _k, _v in _mt_stats.items():
          _mt_log[_k] = _v
        _mt_log['mt_pol_w_rms'] = self.move_tokens.pol.weight.float().pow(2).mean().sqrt()
        _mt_log['mt_pol_bias_rms'] = self.move_tokens.mt_pol_bias.float().pow(2).mean().sqrt()
        if self.move_tokens.value_inject_dim > 0:
          _mt_log['mt_vinject_rms'] = self.move_tokens.v_inject.weight.float().pow(2).mean().sqrt()

    # EDGE-AUX (boelge 13 / P1): par-supervisjon paa planets kant-tilstand.
    # Two targets on the final [B,32,32,C] edge state, both restricted to
    # PIECE-PIECE slot pairs (the plane's competence; empty slots masked):
    #   pi-edge: the search policy's mass on moves between two piece squares
    #            (captures; castling too under king-takes-rook encoding),
    #            scattered from the 1858 move list onto (from,to) and gathered
    #            to slot pairs, plus a NULL bucket = the remaining (quiet) mass.
    #            Legality-masked softmax-CE minus target entropy (house KL form).
    #   rel:     BCE against the withheld visibility families (xray/pinray/...)
    #            that were REMOVED from the plane's input — reachable only by
    #            composing the families it still sees (anti-substitution form).
    # Logged: losses, pi top-1, capture share, per-family BCE and separation
    # (mean sigmoid on positives minus negatives = the decodability probe).
    dpe_pi_loss = 0
    dpe_rel_loss = 0
    _dpe_log = {}
    _dpe = getattr(self, '_last_dp_eaux', None)
    if _dpe is not None and not gradient_norm_logging_mode:
      self._last_dp_eaux = None
      _ea_lg, _ea_sel, _ea_occ, _ea_tgt = _dpe
      _Be = _ea_lg.shape[0]
      _occf = _ea_occ.float()
      _pairm = _occf.unsqueeze(2) * _occf.unsqueeze(1)                       # [B,32,32] both real pieces
      _col = 0
      if self.dp_eaux_pi_w > 0:
        _pl = _ea_lg[..., 0]                                                 # [B,32,32]
        _col = 1
        _pt = policy_target.float()
        _flat = torch.zeros(_Be, 4096, device=_pt.device, dtype=_pt.dtype).index_add_(1, self.dp_eaux_mv_ft, _pt)
        _flatl = torch.zeros(_Be, 4096, device=_pt.device, dtype=_pt.dtype).index_add_(1, self.dp_eaux_mv_ft, (_pt > 0).float())
        _pidx = (_ea_sel.unsqueeze(2) * 64 + _ea_sel.unsqueeze(1)).reshape(_Be, 1024)   # from*64+to per slot pair
        _tm = torch.gather(_flat, 1, _pidx).reshape(_Be, 32, 32) * _pairm          # capture mass on piece pairs
        _lm = (torch.gather(_flatl, 1, _pidx).reshape(_Be, 32, 32) > 0) & (_pairm > 0)
        _null = (1.0 - _tm.sum(dim=(1, 2))).clamp_min(0.0)                        # quiet mass -> null bucket
        _lgm = torch.where(_lm, _pl, torch.full_like(_pl, -1e4)).reshape(_Be, 1024)
        _lgm = torch.cat([_lgm, torch.zeros(_Be, 1, device=_pl.device, dtype=_pl.dtype)], dim=1)   # null logit fixed at 0
        _tg = torch.cat([_tm.reshape(_Be, 1024), _null.unsqueeze(1)], dim=1)
        _tg = _tg / _tg.sum(dim=1, keepdim=True).clamp_min(1e-6)
        _lp = torch.log_softmax(_lgm, dim=1)
        _ce = -(_tg * _lp).sum(dim=1)
        _ent = -(_tg * _tg.clamp_min(1e-12).log()).sum(dim=1)
        dpe_pi_loss = (_ce - _ent).mean()
        with torch.no_grad():
          _dpe_log['dp_edge_pi_top1'] = (_lgm.argmax(dim=1) == _tg.argmax(dim=1)).float().mean()
          _dpe_log['dp_edge_pi_capshare'] = (1.0 - _null).mean()
      if self.dp_eaux_rel_w > 0 and _ea_tgt is not None:
        _rl = _ea_lg[..., _col:]                                             # [B,32,32,4W]
        _bce = torch.nn.functional.binary_cross_entropy_with_logits(_rl, _ea_tgt, reduction='none')
        _pm4 = _pairm.unsqueeze(-1)
        _den = _pm4.sum() * _rl.shape[-1]
        dpe_rel_loss = (_bce * _pm4).sum() / _den.clamp_min(1.0)
        with torch.no_grad():
          _sg = torch.sigmoid(_rl)
          for _fi, _fam in enumerate(self.dp_eaux_withhold):
            _sl = slice(4 * _fi, 4 * _fi + 4)
            _t4, _s4, _b4 = _ea_tgt[..., _sl], _sg[..., _sl], _bce[..., _sl]
            _pos = (_t4 > 0.5).float() * _pm4
            _neg = (_t4 <= 0.5).float() * _pm4
            _dpe_log['dp_edge_rel_' + _fam + '_bce'] = (_b4 * _pm4).sum() / (_pm4.sum() * 4).clamp_min(1.0)
            _dpe_log['dp_edge_rel_' + _fam + '_sep'] = ((_s4 * _pos).sum() / _pos.sum().clamp_min(1.0)
                                                        - (_s4 * _neg).sum() / _neg.sum().clamp_min(1.0))
            _dpe_log['dp_edge_rel_' + _fam + '_rate'] = _pos.sum() / _pm4.sum().clamp_min(1.0) / 4

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

    # Runde-3-fiks 2026-08-29: gate paa VEKT i tillegg til None — forward
    # substituerer dummy-aliaser (value2->value1, mlh/qdev/unc_policy->unc)
    # for eksport-signaturen, saa None-gating alene scoret aliasene som ekte
    # tap: plausible-men-falske TB-kurver, korrupte GRADNORM-attribusjoner
    # (full ekstra backward per alias) og bortkastet compute per steg.
    v2_loss = 0 if value2_out is None or value2_out is value_out else loss_calc.value2_loss(wdl_blend, value2_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.value2_loss_weight, provenance=z_provenance)
    ml_loss = 0 if moves_left_out is None or moves_left_out is unc_out else loss_calc.moves_left_loss(moves_left_target, moves_left_out, gradient_norm_logging_mode, self.moves_left_loss_weight)
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
    q_deviation_lower_loss = 0 if q_deviation_lower_out is None or q_deviation_lower_out is unc_out else loss_calc.q_deviation_lower_loss(q_deviation_lower_target, q_deviation_lower_out, gradient_norm_logging_mode, self.q_deviation_loss_weight)
    q_deviation_upper_loss = 0 if q_deviation_upper_out is None or q_deviation_upper_out is unc_out else loss_calc.q_deviation_upper_loss(q_deviation_upper_target, q_deviation_upper_out, gradient_norm_logging_mode, self.q_deviation_loss_weight)


    if self.config.NetDef_TrainOn4BoardSequences:
      # TO DO: probably the multiplier_action_loss should somehow be propagated into the gradient norms when these are calculated
      # Runde-2-funn: board 1 kalles med action_target/action_out=None
      # (train.py-call-siten) — ubetingede kall ville krasje paa F.softmax(None).
      action_loss = 0 if action_target is None or action_out is None else           multiplier_action_loss * loss_calc.action_loss(action_target, action_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.action_loss_weight)
      # Bugfunn 2026-08-28: (a) vekten ble anvendt HER OG i total_loss => vekt^2
      # (soesterlinjen action_loss anvender kun multiplier_action_loss); (b) target
      # |action_target - action_out| var ikke detached => huber-tapet dyttet
      # action-hodet mot unc-hodets prediksjon (samme lekkasje opt-tapet
      # eksplisitt detacher mot).
      action_uncertainty_loss = 0 if action_target is None or action_out is None or action_uncertainty_out is None else           multiplier_action_loss * loss_calc.action_unc_loss(torch.abs(action_target - action_out).detach(), action_uncertainty_out, gradient_norm_logging_mode, self.action_uncertainty_loss_weight)
      # We have two value scores and want them to be consistent modulo inversion (prior_board and this_board).
      # The value of this board is taken to be "more definitive" so it is the target (however this assumes policy was correct....)
      value_diff_loss = 0 if self.value_diff_loss_weight == 0 or prior_value_out == None else loss_calc.value_diff_loss(value_out, prior_value_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.value_diff_loss_weight)
      value2_diff_loss = 0 if self.value2_diff_loss_weight == 0 or prior_value2_out == None else loss_calc.value2_diff_loss(value2_out, prior_value2_out, SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.value2_diff_loss_weight)
    else:
      action_loss = 0
      action_uncertainty_loss = 0
      value_diff_loss = 0
      value2_diff_loss = 0

    uncertainty_policy_loss = 0 if uncertainty_policy_out is None or uncertainty_policy_out is unc_out else loss_calc.uncertainty_policy_loss(uncertainty_policy_target, uncertainty_policy_out, gradient_norm_logging_mode, self.uncertainty_policy_weight)

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

    # Refiner deep-supervision policy CE (see __init__/forward): mean CE over
    # the intermediate refiner iterations against the SAME policy target, with
    # the standard legality masking. Same consume-and-clear + skip-on-gradnorm
    # conventions as the other stashed aux losses (depth-probe pattern).
    refiner_ploss = 0
    _rfp = getattr(self, '_last_refiner_policy', None)
    if _rfp is not None and policy_out is not None and not gradient_norm_logging_mode:
      self._last_refiner_policy = None
      _rf_T = _rfp.shape[1]
      _rf_legal = policy_target.greater(0)
      _rf_ent = loss_calc.entropy(policy_target) if SUBTRACT_ENTROPY else 0.0
      _rfp_m = torch.where(_rf_legal.unsqueeze(1), _rfp,
                           torch.full_like(_rfp, loss_calc.MASK_POLICY_VALUE))
      _rf_pt = policy_target.unsqueeze(1).expand(-1, _rf_T, -1).reshape(-1, policy_target.shape[-1])
      refiner_ploss = loss_calc.ce_loss.forward(
          _rfp_m.reshape(-1, _rfp.shape[-1]).float(), _rf_pt) - _rf_ent

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
    _survival_participation_only = False
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
      else:
        # DDP static_graph participation term (see the aux-head block in train.py):
        # a zero-weighted read of the stashed output keeps survival_head in the
        # backward's used-parameter set on target-less batches, so the set stays
        # constant across iterations. Mathematically a no-op (exact zero gradient).
        # Flagged so the logging block below can still tell "no target this batch"
        # from a real 0.0 — otherwise every sidecar-less batch would emit a hard
        # zero sample and dilute the survival curve the gates are read from.
        survival_loss = 0.0 * _sv_out.float().sum()
        _survival_participation_only = True

    # Per-piece survival aux (dual-plane): fate CE on piece tokens, targets =
    # square survival sidecar gathered to the piece slots. Empty slots and
    # class-0 squares masked. Same consume-and-clear + sidecar-less-batch
    # zero-read pattern as the square survival head above.
    dp_surv_loss = 0
    _dp_surv_participation_only = False
    _dps = getattr(self, '_last_dp_surv', None)
    if _dps is not None and not gradient_norm_logging_mode:
      self._last_dp_surv = None   # runde-4: clear UBETINGET (vekt-0-batcher lot graf-stashen ligge)
    if getattr(self, 'dp_surv_weight', 0) > 0 and _dps is not None and value_out is not None:
      _dps_out, _dps_sel, _dps_occ = _dps
      _dps_tgt_sq = batch.get('survival', None)
      if _dps_tgt_sq is not None:
        _tgt_p = torch.gather(_dps_tgt_sq.to(_dps_sel.device).long(), 1, _dps_sel)   # [B, 32]
        # Sidecar horizon K_gen may exceed the head's DualPlaneSurvivalK (review
        # finding 2026-08-20 #1: raw class K_gen+1 overflows a K+2-logit CE ->
        # device assert). Clamp maps "captured at ply > K" and "survives" both
        # onto the head's own survives-beyond-K class — semantically exact for
        # fate-within-K classification.
        _tgt_p = _tgt_p.clamp(max=self.dp_surv_head.out_features - 1)
        _m = (_tgt_p > 0) & (_dps_occ > 0)
        if _m.any():
          dp_surv_loss = torch.nn.functional.cross_entropy(
              _dps_out[_m].float(), _tgt_p[_m])
        else:
          dp_surv_loss = 0.0 * _dps_out.float().sum()
          _dp_surv_participation_only = True
      else:
        dp_surv_loss = 0.0 * _dps_out.float().sum()
        _dp_surv_participation_only = True

    # Short-term value aux head: CE against the WDL built from censored q_st/d_st
    # (V7-extras sidecar; STM-relative, matching TPG conventions), optionally weighted
    # per record by z-provenance. Missing keys = batch from a v7x-less shard
    # (CERES_TPG_V7X_SIDECAR=auto mixed-corpus mode): skip, as with survival.
    # Same consume-and-clear/value_out gating as the other aux heads.
    stvalue_loss = 0
    _stvalue_participation_only = False
    _st_out = getattr(self, '_last_stvalue_out', None)
    if self.stvalue_weight > 0 and _st_out is not None and value_out is not None:
      if not gradient_norm_logging_mode:
        self._last_stvalue_out = None
      _cens_q = batch.get('censored_q_st', None)
      if _cens_q is not None:
        stvalue_loss = loss_calc.stvalue_loss(_cens_q, batch['censored_d_st'], batch.get('z_provenance', None),
                                              _st_out.float(), SUBTRACT_ENTROPY, gradient_norm_logging_mode, self.stvalue_weight)
      else:
        # DDP static_graph participation term — see the survival branch above.
        stvalue_loss = 0.0 * _st_out.float().sum()
        _stvalue_participation_only = True

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
        + (self.dp_surv_weight * dp_surv_loss if not isinstance(dp_surv_loss, int) else 0)
        + self.stvalue_weight * stvalue_loss
        + (self.vda_aux_weight * vda_aux_loss if self.vda_mode == 4 else 0)
        + ((self.depth_probe_policy_weight * depth_probe_ploss
            + self.depth_probe_value_weight * depth_probe_vloss
            + depth_ctl_ploss + depth_ctl_vloss)
           if not isinstance(depth_probe_ploss, int) else 0)
        + (self.value_rank_weight * value_rank_loss if not isinstance(value_rank_loss, int) else 0)
        + (self.policy_margin_weight * policy_margin_loss if not isinstance(policy_margin_loss, int) else 0)
        + (self.policy_pl_weight * policy_pl_loss if not isinstance(policy_pl_loss, int) else 0)
        + (self.value_contrast_weight * vc_loss if not isinstance(vc_loss, int) else 0)
        + (self.hlg_weight * hlg_loss if not isinstance(hlg_loss, int) else 0)
        + (self.opt_policy_weight * opt_loss if not isinstance(opt_loss, int) else 0)
        + (self.opp_policy_weight * oppp_loss if not isinstance(oppp_loss, int) else 0)
        + (self.action_played_weight * actp_loss if not isinstance(actp_loss, int) else 0)
        + (self.soft_policy_weight * soft_policy_loss if not isinstance(soft_policy_loss, int) else 0)
        + (self.refiner_deep_sup_weight * refiner_ploss if not isinstance(refiner_ploss, int) else 0)
        + (self.dp_eaux_pi_w * dpe_pi_loss if not isinstance(dpe_pi_loss, int) else 0)
        + (self.dp_eaux_rel_w * dpe_rel_loss if not isinstance(dpe_rel_loss, int) else 0))

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
          + (self.policy_margin_weight * policy_margin_loss if not isinstance(policy_margin_loss, int) else 0)
          + (self.policy_pl_weight * policy_pl_loss if not isinstance(policy_pl_loss, int) else 0)
          + self.uncertainty_policy_weight * uncertainty_policy_loss
          + self.action_loss_weight * action_loss
          + self.action_uncertainty_loss_weight * action_uncertainty_loss
          + (self.opt_policy_weight * opt_loss if not isinstance(opt_loss, int) else 0)
        + (self.opp_policy_weight * oppp_loss if not isinstance(oppp_loss, int) else 0)
        + (self.action_played_weight * actp_loss if not isinstance(actp_loss, int) else 0)
          + (self.soft_policy_weight * soft_policy_loss if not isinstance(soft_policy_loss, int) else 0)
          # pi-edge is a policy-target CE (rel-aux is neither family: kept out).
          + (self.dp_eaux_pi_w * dpe_pi_loss if not isinstance(dpe_pi_loss, int) else 0)
          # Refiner deep-sup is pure policy-target CE, so it belongs to the
          # policy family (unlike the depth probes, which span both).
          + (self.refiner_deep_sup_weight * refiner_ploss if not isinstance(refiner_ploss, int) else 0))
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
        # Base-rate-aware value diagnostics: value_acc alone is inflated by a
        # drawish corpus and is blind to compression. See losses.value_metrics.
        if value_out is not None:
          for _mk, _mv in loss_calc.value_metrics(value_target, value_out).items():
            self._log(_mk + stat_suffix, _mv, step=num_pos)

      # Runde-3/4: i VANLIG modus logges den rene CE-en (losses stasher den —
      # returverdien kan baere sibling-margin/only-move-reweighting); i
      # gnorm-modus ER p_loss selve gradientnormen og skal logges som foer.
      # Bar attributt-tilgang med vilje: rename skal feile hoyt, ikke stille
      # falle tilbake til den kontaminerte verdien.
      self._log("policy_loss" + stat_suffix,
                p_loss if gradient_norm_logging_mode else loss_calc._last_policy_log_loss,
                step=num_pos)
      self._log("value_loss" + stat_suffix, v_loss,  step=num_pos)
      if not isinstance(v2_loss, int):
        self._log("value2_loss" + stat_suffix, v2_loss,  step=num_pos)
      if self.placement_value_weight > 0 and not isinstance(placement_loss, int):
        self._log("placement_value_loss" + stat_suffix, placement_loss, step=num_pos)
      if self.survival_target_weight > 0 and not isinstance(survival_loss, int)           and not _survival_participation_only:
        self._log("survival_loss" + stat_suffix, survival_loss, step=num_pos)
      if getattr(self, 'dp_surv_weight', 0) > 0 and not isinstance(dp_surv_loss, int)           and not _dp_surv_participation_only:
        self._log("dp_survival_loss" + stat_suffix, dp_surv_loss, step=num_pos)
      if self.stvalue_weight > 0 and not isinstance(stvalue_loss, int)           and not _stvalue_participation_only:
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
      if not isinstance(policy_margin_loss, int):
        self._log("policy_margin_loss" + stat_suffix, policy_margin_loss, step=num_pos)
      if not isinstance(policy_pl_loss, int):
        self._log("policy_pl_loss" + stat_suffix, policy_pl_loss, step=num_pos)
        for _k, _v in _pl_log.items():
          self._log(_k + stat_suffix, _v, step=num_pos)
      if not isinstance(vc_loss, int):
        self._log("value_contrast_loss" + stat_suffix, vc_loss, step=num_pos)
      if not isinstance(hlg_loss, int):
        self._log("hlgauss_value_loss" + stat_suffix, hlg_loss, step=num_pos)
      if not isinstance(opt_loss, int):
        self._log("optimistic_policy_loss" + stat_suffix, opt_loss, step=num_pos)
      if not isinstance(oppp_loss, int) and not _oppp_participation_only:
        self._log("opp_policy_loss" + stat_suffix, oppp_loss, step=num_pos)
      if not isinstance(actp_loss, int) and not _actp_participation_only:
        self._log("action_played_loss" + stat_suffix, actp_loss, step=num_pos)
      for _k, _v in _mt_log.items():
        self._log(_k + stat_suffix, _v, step=num_pos)
      if not isinstance(dpe_pi_loss, int):
        self._log("dp_edge_pi_loss" + stat_suffix, dpe_pi_loss, step=num_pos)
      if not isinstance(dpe_rel_loss, int):
        self._log("dp_edge_rel_loss" + stat_suffix, dpe_rel_loss, step=num_pos)
      for _k, _v in _dpe_log.items():
        self._log(_k + stat_suffix, _v, step=num_pos)
      if not isinstance(soft_policy_loss, int):
        self._log("soft_policy_loss" + stat_suffix, soft_policy_loss, step=num_pos)
      if not isinstance(refiner_ploss, int):
        self._log("refiner_deepsup_policy_loss" + stat_suffix, refiner_ploss, step=num_pos)
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
      # Runde-3: avslaatte hoder (int 0) logges ikke — ingen null/garbage-kurver.
      if not isinstance(ml_loss, int):
        self._log("moves_left_loss" + stat_suffix, ml_loss, step=num_pos)
      if not isinstance(u_loss, int):
        self._log("unc_loss" + stat_suffix, u_loss, step=num_pos)
      if not isinstance(uncertainty_policy_loss, int):
        self._log("unc_policy_loss" + stat_suffix, uncertainty_policy_loss, step=num_pos)
      if not isinstance(q_deviation_lower_loss, int):
        self._log("q_deviation_lower_loss" + stat_suffix, q_deviation_lower_loss, step=num_pos)
      if not isinstance(q_deviation_upper_loss, int):
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


