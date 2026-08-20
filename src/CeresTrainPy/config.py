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
import json

"""
Global constants.
"""
NUM_TOKENS_INPUT = 64 # Raw input number of tokens
NUM_TOKENS_NET = 64 # Number of tokens used by net
NUM_INPUT_BYTES_PER_SQUARE = 137 # Raw input width per token

# Optional augmented input features computed in DataLoader from the per-square
# piece-one-hot information. V3 layout (post-2026-06-01 cleanup) exposes 4 channels —
# tactical-motif features that earned their place via ablation:
#   [0]=mobility            (pseudo-legal moves per piece)
#   [1]=defender_count      (same-color attackers of piece)
#   [2]=is_pinned           (boolean: piece pinned to own king by opp slider)
#   [3]=is_threatened       (boolean: attacked by strictly-lower-value opp piece)
# When > 0, the dataset's squares tensor is widened by this many extra channels per
# square and the embedding layer input dim grows correspondingly.
# DEFAULTS TO 4 (full V3). The TPG reader (tpg_dataset.py) is V3-only — every
# record carries 141 bytes/square = 137 base + these 4 baked aux channels — so a
# model that consumes them is the right default for all current data. Overriding
# to 0 builds a legacy 137-channel model that IGNORES the aux features (use only
# for deliberate no-aux ablations or to resume a pre-aux 137-channel net).
# Toggle via env var: CERES_AUX_FEATURES_PER_SQUARE=4 (full V3, default) / =0 (legacy)
# / =N to use the first N aux channels. Legacy CERES_AUG_FEATURES_PER_SQUARE also honored.
# This is the SINGLE SOURCE OF TRUTH — tpg_dataset.py imports this value rather than
# re-reading the env, so the model width and data width can never disagree.
NUM_AUX_FEATURES_PER_SQUARE = int(os.environ.get('CERES_AUX_FEATURES_PER_SQUARE',
                                  os.environ.get('CERES_AUG_FEATURES_PER_SQUARE', '4')) or 4)
TOTAL_INPUT_FEATURES_PER_SQUARE = NUM_INPUT_BYTES_PER_SQUARE + NUM_AUX_FEATURES_PER_SQUARE
if NUM_AUX_FEATURES_PER_SQUARE > 0:
  print(f'[config] AUX_FEATURES enabled: +{NUM_AUX_FEATURES_PER_SQUARE} channels per square, '
        f'total = {TOTAL_INPUT_FEATURES_PER_SQUARE} per square')


def read_config(file_path):
  """Reads a JSON configuration file and returns a dictionary."""
  with open(file_path, 'r') as file:
      return json.load(file)

class Configuration:
  """
    Initializes the Configuration object by loading various configuration settings from JSON files.

    This method reads multiple configuration files corresponding to different aspects of the training 
    process (execution, data handling, optimization, and network definition). Each configuration file 
    is identified by the provided 'id'. The method loads these settings into the object's attributes 
    for easy access throughout the program.

    Parameters:
        config_files_dir (str): The directory path where configuration files are stored.
        id (str): A unique identifier used to specify which configuration files to load. 
                    This ID is appended to each config file's base name.
  """
    
  def __init__(self, config_files_dir : str, id : str):
    self.id = id

    print('READING CONFIG FILES FOR TRAINING RUN", id, "FROM DIRECTORY', config_files_dir)
    config_data_file = os.path.join(config_files_dir, id + '_ceres_data.json')
    config_exec_file = os.path.join(config_files_dir, id + '_ceres_exec.json')
    config_opt_file = os.path.join(config_files_dir,  id + '_ceres_opt.json')
    config_net_def_file = os.path.join(config_files_dir, id + '_ceres_net.json')

    config_exec = read_config(config_exec_file)
    config_data = read_config(config_data_file)
    config_opt = read_config(config_opt_file)
    config_net_def = read_config(config_net_def_file)

    # Initialize class members from config_data
    self.Data_SourceType = config_data.get('SourceType', 'DirectFromPositionGenerator')
    self.Data_PositionGenerator = config_data.get('PositionGenerator', {})
    self.Data_TrainingFilesDirectory = config_data.get('TrainingFilesDirectory', None)
    self.Data_NumTPGFilesToSkip = config_data.get('NumTPGFilesToSkip', 0)
    self.Data_FractionQ = config_data.get('FractionQ', 0.0)
    self.Data_WDLLabelSmoothing = config_data.get('WDLLabelSmoothing', 0.0)
    # Optional secondary TPG dir for mixed-corpus training. When set, batches are
    # interleaved with the primary at the configured ratio (e.g. 19 → 19 primary
    # batches per 1 secondary). Default: None / 0 = single-source (legacy behaviour).
    self.Data_TrainingFilesDirectory2 = config_data.get('TrainingFilesDirectory2', None)
    self.Data_RatioSet1ToSet2 = int(config_data.get('RatioSet1ToSet2', 0))

    # Initialize class members from config_exec
    self.Exec_ID = config_exec.get('ID', 'TEST')
    self.Exec_DeviceType = config_exec.get('DeviceType', 'CUDA')
    self.Exec_DeviceIDs = config_exec.get('DeviceIDs', [0])
    self.Exec_DataType = config_exec.get('DataType', 'BFloat16')
    self.Exec_UseFP8 = config_exec.get('UseFP8', False)
    self.Exec_DropoutRate = config_exec.get('DropoutRate', 0)
    self.Exec_DropoutDuringInference = config_exec.get('DropoutDuringInference', False)
    self.Exec_EngineType = config_exec.get('EngineType', 0)
    self.Exec_SaveNetwork1FileName = config_exec.get('SaveNetwork1FileName', None)
    self.Exec_SaveNetwork2FileName = config_exec.get('SaveNetwork2FileName', None)
    self.Exec_ActivationMonitorDumpSkipCount = config_exec.get('ActivationMonitorDumpSkipCount', 0)
    self.Exec_SupplementaryStat = config_exec.get('SupplementaryStat', None)
    self.Exec_TrackFinalLayerIntrinsicDimensionality = config_exec.get('TrackFinalLayerIntrinsicDimensionality', False)
    self.Exec_MonitorActivationStats = config_exec.get('MonitorActivationStats', False)
    self.Exec_ExportOnly = config_exec.get('ExportOnly', False)
    self.Exec_TestFlag = config_exec.get('TestFlag', False)
    self.Exec_TestValue = config_exec.get('TestValue', 0)

    # Initialize class members from config_opt
    self.Opt_LoRARankDivisor = config_opt.get('LoRARankDivisor', 0)
    self.Opt_LoRARestrictPolicyValueOnly = config_opt.get('LoRARestrictPolicyValueOnly', False)
    self.Opt_LoRARestrictValueOnly = config_opt.get('LoRARestrictValueOnly', False)
    self.Opt_NumTrainingPositions = config_opt.get('NumTrainingPositions', 10_000_000)
    self.Opt_BatchSizeForwardPass = config_opt.get('BatchSizeForwardPass', 2048)
    self.Opt_BatchSizeBackwardPass = config_opt.get('BatchSizeBackwardPass', 2048)
    self.Opt_Optimizer = config_opt.get('Optimizer', 'AdamW')
    self.Opt_CheckpointResumeFromFileName = config_opt.get('CheckpointResumeFromFileName')
    self.Opt_CheckpointFrequencyNumPositions = config_opt.get('CheckpointFrequencyNumPositions', 200_000_000)
    self.Opt_PyTorchCompileMode = config_opt.get('PyTorchCompileMode', "max-autotune")
    self.Opt_WeightDecay = config_opt.get('WeightDecay', 0.01)
    self.Opt_LearningRateBase = config_opt.get('LearningRateBase', 0.0005)
    # LR-decay shape and floor (TRAINING_RECOMMENDATIONS.md §4). 'linear'
    # (default) = legacy behavior; 'cosine' = half-cosine from the decay start
    # to the floor — combined with LRBeginDecayAtFractionComplete 0.0 this is
    # the reference no-plateau shape (warmup, then cosine over the whole run).
    # LRMinFactor = floor as a fraction of peak (legacy hardcoded 0.05;
    # reference practice 0.10).
    self.Opt_LRDecayShape = (config_opt.get('LRDecayShape', 'linear') or 'linear').strip().lower()
    assert self.Opt_LRDecayShape in ('linear', 'cosine'), f"LRDecayShape must be 'linear' or 'cosine', got {self.Opt_LRDecayShape!r}"
    _lrmf = config_opt.get('LRMinFactor')   # (_cfg_num is defined further down)
    self.Opt_LRMinFactor = 0.05 if _lrmf is None else float(_lrmf)
    # Split-LR (Muon only): separate absolute rate for the internal-AdamW group
    # (heads/embeddings/norms/biases); the Muon trunk keeps LearningRateBase.
    # None/absent = legacy single-rate. CONFIG-ONLY (no env override by design).
    self.Opt_LearningRateBaseHeads = config_opt.get('LearningRateBaseHeads', None)
    # Family LR ratios (split-LR program 2026-08-20): multiplicative on base LR,
    # name-family targeted regardless of Muon/AdamW membership (Kovax runs 1/3
    # on heads). Mutually exclusive with LearningRateBaseHeads.
    self.Opt_LearningRateHeadsRatio = config_opt.get('LearningRateHeadsRatio', None)
    self.Opt_LearningRateCouplingsRatio = config_opt.get('LearningRateCouplingsRatio', None)
    # Muon momentum decoupling (Muon only): historically Beta1 fed BOTH the
    # Muon SGD-momentum and the internal-AdamW beta1, so the reference combo
    # (Muon momentum 0.95 + Adam beta1 0.9, Kovax live configs) was
    # inexpressible. MuonMomentum overrides the Muon side only; None/absent =
    # legacy coupling (momentum = Beta1). MuonAdamWEps exposes the internal
    # AdamW epsilon (legacy hardcoded 1e-8; reference uses 1e-7).
    _mm = config_opt.get('MuonMomentum')
    self.Opt_MuonMomentum = None if _mm is None else float(_mm)
    _mae = config_opt.get('MuonAdamWEps')
    self.Opt_MuonAdamWEps = None if _mae is None else float(_mae)
    # Muon partition scope (Muon only): which params the internal AdamW gets.
    #   'all-non-trunk' (legacy default): everything outside transformer_layer —
    #       whole heads (incl. hidden 2-D fc), embeddings, norms, biases.
    #   'final-only' (dje-style): only final output layers (fcFinal / single-Linear
    #       aux heads), embeddings, LoRA adapters and 1-D params; head hidden
    #       matrices train under Muon like the trunk.
    self.Opt_MuonAdamWScope = config_opt.get('MuonAdamWScope', 'all-non-trunk')
    # Hard-position replay buffer (0 = off). An online ring buffer of the
    # highest-value-KL batch rows (measured by the LIVE net each step); a
    # fraction of every subsequent batch is replaced with buffered hard rows so
    # difficult positions are revisited under evolving trunk states instead of
    # seen once. Buffer churns continuously (unlike a fixed replay set, which
    # is the known puzzle-overfit failure mode). CONFIG-ONLY by design.
    #   HardReplayBufferSize: max rows held (CPU memory ~40KB/row).
    #   HardReplayFraction:   share of each batch replaced by replayed rows.
    #   HardReplayMaxReuse:   evict an entry after this many replays.
    # NOTE on parsing: `x or default` is only used where 0 IS the off-switch;
    # parameters where an explicit 0 is a plausible intentional value use
    # None-checked parsing so user zeros are respected.
    def _cfg_num(key, default, cast=float):
      _v = config_opt.get(key)
      return cast(default if _v is None else _v)
    self.Opt_HardReplayBufferSize = int(config_opt.get('HardReplayBufferSize', 0) or 0)
    self.Opt_HardReplayFraction = _cfg_num('HardReplayFraction', 0.125)
    self.Opt_HardReplayMaxReuse = _cfg_num('HardReplayMaxReuse', 8, int)
    # ---- Selection / dosing (see train.py replay block) -------------------
    # HardReplayMinKL: ABSOLUTE admission floor on per-row value KL (0 = off,
    # legacy relative top-k). The legacy selector is a pure RANK: it always fills
    # its quota, so as the net improves "the hardest 12.5%" becomes progressively
    # less hard and the buffer fills with almost-understood rows. Measured
    # 2026-08-19 on T91: at a 12.5% quota the LOWEST admitted row had KL 0.124
    # against a corpus median of 0.012. Learnability rises monotonically with
    # ABSOLUTE KL and has no upper cutoff (improvement vs corpus baseline, top
    # band = most learnable):
    #     KL 0.00-0.05 (74.2% of rows) 0.6x | 0.05-0.15 1.8x | 0.15-0.40 2.1x
    #     KL 0.40-1.00 ( 2.8% of rows) 2.8x | 1.00-2.50 5.2x
    # 0.4 excludes everything at or below 2.1x and admits ~3.4% of the stream.
    # An absolute floor also makes the treatment SELF-LIMITING: admissions fall
    # as the net improves, instead of permanently displacing 12.5% of every batch
    # (the fixed displacement is the surviving explanation for why the 2026-08-18
    # arm generalised worse while fitting its own distribution fine).
    self.Opt_HardReplayMinKL = _cfg_num('HardReplayMinKL', 0.0)
    # HardReplayReuseTarget: desired mean reuse (0 = off, legacy fixed injection
    # at HardReplayFraction). Mean reuse is EXACTLY injected/admitted by flow
    # conservation, so injection is sized as target x realised admissions rather
    # than as a fixed batch share -- which both pins the dose and lets replay
    # VOLUME taper with the floor above. HardReplayFraction remains a hard cap on
    # injection. NOTE: buffer size does NOT affect reuse (it cancels); it only
    # sets residency/diversity. HardReplayMaxReuse must stay comfortably ABOVE
    # this target -- at MaxReuse == target occupancy is a neutral random walk and
    # below it the buffer starves and injection silently falls under the cap.
    self.Opt_HardReplayReuseTarget = _cfg_num('HardReplayReuseTarget', 0.0)
    # HardReplayStartFraction: hold replay OFF until this fraction of training is
    # done (0 = from step 0). Measured 2026-08-19 on the 1B ctrl checkpoints: the
    # hard set TURNS OVER early, so focusing on it early wastes the budget on rows
    # ordinary training resolves by itself. Of the rows hardest at 100M, 54.3% had
    # fallen below KL 0.4 by 900M (median 0.890 -> 0.354). Stability of the top-3%
    # set, measured against the 900M net: 45.7% @100M, 46.7% @300M, 64.9% @500M,
    # 76.4% @700M — the jump is between 300M and 500M, which is why 0.5 is the
    # natural cut rather than an arbitrary one.
    self.Opt_HardReplayStartFraction = _cfg_num('HardReplayStartFraction', 0.0)
    # HardReplaySelector: what makes a row "hard".
    #   'valuekl' (default): per-row value-KL vs the search target, admitted above
    #       HardReplayMinKL. Measured result (2026-08-19, two arms): this diet
    #       damages value calibration — the selected rows are value-extreme and
    #       8x oversampling shifts the value head's gradient composition.
    #   'topn': the position has a CLEAR solution the net misranks. Admission:
    #       target's best move prob > HardReplayTopNPBest (unambiguous answer)
    #       AND |W-L| > HardReplayTopNMinAbsQ (not a drawish position where many
    #       moves tie and the target argmax is search noise)
    #       AND the best move is NOT among the net's HardReplayTopN highest-
    #       ranked moves. "Ranked moves" means moves with SEARCH VISITS
    #       (target > 0) — a visited-moves proxy for legality, the same
    #       convention as losses.py and the offline calibration, and strictly
    #       conservative (can only under-admit) vs a true legal mask. Changing
    #       it to true legality would silently invalidate the calibration below.
    #       Measured on ctrl@600M (N=3, pbest 0.5, absq 0.4): 1.45% of the
    #       stream (dose 8 feasible), value-EASY (median value-KL 0.0023 vs
    #       0.0102 corpus — no value-diet damage), and ~70% of the population
    #       is STILL misranked at 1B under ordinary training: real headroom,
    #       unlike the value-hard set where 54% self-resolved.
    #   Internal score = top-N margin: logit(Nth best ranked move) minus
    #   logit(target best move). >0 = fails containment (admit); exit on
    #   re-score when contained by more than HardReplayTopNExitMargin
    #   (hysteresis, same role as the 0.5x floor drop in 'valuekl' mode).
    #   NOTE parse rule: None-checked (a user zero must reach the validator and
    #   fail loudly there, not be silently replaced by the default).
    _sel_raw = config_opt.get('HardReplaySelector')
    self.Opt_HardReplaySelector = 'valuekl' if _sel_raw is None else str(_sel_raw)
    self.Opt_HardReplayTopN = _cfg_num('HardReplayTopN', 3, int)
    self.Opt_HardReplayTopNPBest = _cfg_num('HardReplayTopNPBest', 0.5)
    self.Opt_HardReplayTopNMinAbsQ = _cfg_num('HardReplayTopNMinAbsQ', 0.4)
    self.Opt_HardReplayTopNExitMargin = _cfg_num('HardReplayTopNExitMargin', 0.5)
    # For the selector-mismatch guard in train.py: were any topn knobs
    # EXPLICITLY present in the json (as opposed to defaulted)?
    self.Opt_HardReplayTopNKeysPresent = any(
      k in config_opt for k in ('HardReplayTopN', 'HardReplayTopNPBest',
                                'HardReplayTopNMinAbsQ', 'HardReplayTopNExitMargin'))
    # EMA / SWA weight averaging (lc0-style capped running average; 0 = off).
    # Every EMAPeriodSteps optimizer steps: ema = ema*(n/(n+1)) + w*(1/(n+1)),
    # n = min(n+1, EMAMaxN) — i.e. an EMA whose momentum saturates at
    # 1 - 1/(EMAMaxN+1). Each checkpoint additionally exports the EMA weights
    # as <name>_<numpos>ema.onnx alongside the raw ones. lc0's TF reference
    # ships swa_steps=100 / swa_max_n=10 and PUBLISHES the averaged nets
    # ("-swa-" files); our post-hoc 3-ckpt averaging measured +34/+29 Elo value
    # at zero policy cost on collapse-phase endpoints (2026-08-06). EMA state
    # is NOT persisted across resume (re-warms from current weights, ~1k steps).
    self.Opt_EMAPeriodSteps = int(config_opt.get('EMAPeriodSteps', 0) or 0)
    self.Opt_EMAMaxN = _cfg_num('EMAMaxN', 10, int)
    # HL-Gauss categorical value head (0 = off): auxiliary softmax-CE over a
    # Gaussian histogram of q = w - l in [-1,1] — fine-grained value-resolution
    # signal (see ceres_net.py). Training-only head, stripped at export.
    self.Opt_HLGaussWeight = float(config_opt.get('HLGaussWeight', 0) or 0)
    self.Opt_HLGaussBuckets = int(config_opt.get('HLGaussBuckets', 32) or 32)
    self.Opt_HLGaussSigmaScale = _cfg_num('HLGaussSigmaScale', 0.75)  # explicit 0 = one-hot buckets (lc0 degenerate case)
    # Mirror-consistency value regularizer + focal value weighting: config is
    # authoritative; the CERES_VALUE_MIRROR_CONS_* / CERES_VALUE_FOCAL_GAMMA
    # env vars remain as FALLBACK (used only when the config field is 0/absent)
    # so launch scripts from the env era keep working across resumes.
    self.Opt_MirrorConsWeight = float(config_opt.get('MirrorConsistencyWeight', 0) or 0)
    self.Opt_MirrorConsFraction = float(config_opt.get('MirrorConsistencyFraction', 0) or 0)
    # Mirror-consistency HYSTERESIS controller ("symmetry thermostat"; 0 =
    # static always-on). When the smoothed mirror loss falls below AutoLow the
    # term drops to a probe every ProbeSteps optimizer steps (~0.1-0.5% cost);
    # if a probe's smoothed level exceeds AutoHigh (e.g. decay-phase sharpening
    # reintroduces asymmetry) full per-step enforcement resumes automatically.
    # Thresholds calibrated against s6 (2026-08-07): highest observed smoothed
    # mirror loss WITH enforcement active was 0.0035 — so High sits just below
    # that (drift alarm before reaching known-bad territory) and Low well
    # below the normal operating level.
    # PRIVATE VALUE FRONT-END (2026-08-09 comparative audit, finding #1;
    # 0 = legacy shared front-end). When > 0, the value family (value1,
    # value2, unc, hlg) reads its OWN per-square projection D -> C channels
    # (flattened 64*C), lc0-style, instead of the policy-shared doubly-
    # compressed headPremap/headSharedLinear vector. The shared front-end is
    # optimized predominantly by the policy gradient, so the legacy value
    # head drinks through policy's straw (typ. 256-dim shared vs lc0's 2048
    # private) — the suspected structural seat of the value deficit.
    # Reference C=32. Incompatible with vda/gtab modes (asserted).
    self.Opt_ValueHeadChannels = int(config_opt.get('ValueHeadChannels', 0) or 0)
    # How the private channels reach the value heads:
    #   'replace' (default, from-scratch): value_head/value2/unc/hlg read the
    #       private 64*C vector INSTEAD of the shared one. Changes head input
    #       widths, so a checkpoint that predates the flag cannot be loaded —
    #       correct for from-scratch, useless for fine-tuning an existing net.
    #   'inject' (fine-tune): head input widths are UNCHANGED. The private
    #       vector goes through its own zero-initialized projection that is
    #       ADDED to the value head's hidden pre-activation. Algebraically this
    #       is exactly concatenating the private features onto the head's input
    #       (fc([shared, priv]) == fc_shared(shared) + W_priv @ priv), but as a
    #       separate module it needs no weight surgery: every pre-existing key
    #       loads unchanged and the net is BIT-IDENTICAL to the base at step 0
    #       (zero-init), which is what a LoRA/fine-tune resume requires.
    #       Applies to value1 + value2 only (the heads the hypothesis is about);
    #       unc/hlg keep the shared input so a served net's aux-head calibration
    #       is not disturbed. The injectors are full-rank and NOT LoRA-wrapped —
    #       they start at exactly zero contribution, so they cannot damage the
    #       base net no matter how hard they train, and rank-limiting them would
    #       throttle the very pathway under test.
    self.Opt_ValueHeadChannelsMode = str(config_opt.get('ValueHeadChannelsMode', 'replace') or 'replace').lower()
    if self.Opt_ValueHeadChannelsMode not in ('replace', 'inject'):
      raise ValueError(f"Unsupported ValueHeadChannelsMode: {self.Opt_ValueHeadChannelsMode!r} (use 'replace' or 'inject')")
    # Uncertainty head SELF-ERROR mode (audit finding #2; 0 = legacy): train
    # unc toward the STUDENT's own realized squared value error
    # (q_pred - q_target)^2, detached — lc0 semantics — instead of the
    # teacher-time |DeltaQVersusV| data field. Self-calibrating; downstream
    # sigma consumers (optimistic policy) then use sqrt(unc). SERVING NOTE:
    # the exported ONNX 'unc' output changes UNITS in this mode (sigma^2, not
    # |dQ| sigma-scale) with no marker in the artifact — any serving-side
    # UncertaintyV consumer must sqrt() it. Document per-net when serving.
    self.Opt_UncSelfError = int(config_opt.get('UncSelfError', 0) or 0)
    # Optimistic-policy aux head (lc0 vdapda-branch port; 0 = off): a separate
    # training-only policy head whose CE is weighted sigmoid((z - Strength) *
    # Alpha) with z = (target_q - predicted_q) / (predicted_sigma + 1e-5) —
    # i.e. trained only where the value head UNDERESTIMATES the outcome,
    # in units of the net's own uncertainty (unc head). lc0 defaults 2.0/3.0.
    self.Opt_OptimisticPolicyWeight = float(config_opt.get('OptimisticPolicyWeight', 0) or 0)
    self.Opt_OptimisticPolicyStrength = _cfg_num('OptimisticPolicyStrength', 2.0)
    self.Opt_OptimisticPolicyAlpha = _cfg_num('OptimisticPolicyAlpha', 3.0)
    # Serve-time blend of the optimistic head into the exported policy output
    # (0 = pure vanilla, 1 = full swap — what lc0 ships). Logit-space blend
    # applied ONLY in eval/export mode, BEFORE any ray-context add (rc stays
    # unscaled). Training is untouched. Baked into the ONNX graph at export,
    # so per-lambda comparisons = re-export the same checkpoint with different
    # values (ExportOnly). Requires OptimisticPolicyWeight > 0 (head must
    # exist and be trained).
    self.Opt_OptimisticPolicyServeBlend = _cfg_num('OptimisticPolicyServeBlend', 0.0)
    # Data-format / survival-sidecar settings (TPGV3, AuxFeaturesPerSquare,
    # SquareBytes, SquareBytes2, AuxChannelIndices, ShuffleSeed,
    # TargetSidecar, V7XSidecar, SurvivalHorizon, SurvivalTargetWeight,
    # SurvivalLossBuckets, SurvivalCaptureWeight), the stream-routing keys
    # (SecondaryLoss*Mult, MixProloguePositions, FileMirrorAug, KeepDrawProb)
    # and the generic "Env" object are NOT parsed here: they are consumed at
    # import time by module-level reads, so train.py bridges them from this
    # JSON into os.environ BEFORE importing those modules (see the config->env
    # bootstrap at the top of train.py; JSON booleans/int-floats/lists are
    # normalized to the string forms the consumers parse). CERES_* env vars
    # remain as fallback for keys absent from the config.
    # CAVEAT: only train.py has the bootstrap — standalone tools
    # (recover_export*.py, reconvert_onnx.py, scripts/*) are still env-only
    # and need the CERES_* vars exported manually.
    # Soft-policy aux head (KataGo "auxiliary soft policy target"): config-first
    # with env fallback (CERES_SOFT_POLICY_WEIGHT/_TEMP). ServeBlend composes
    # with the optimistic blend as a three-way logit mix (sum of blends <= 1).
    self.Opt_SoftPolicyWeight = float(config_opt.get('SoftPolicyWeight', 0) or 0)
    self.Opt_SoftPolicyTemp = float(config_opt.get('SoftPolicyTemp', 0) or 0)  # 0 = env/default 4
    self.Opt_SoftPolicyServeBlend = _cfg_num('SoftPolicyServeBlend', 0.0)
    self.Opt_MirrorConsProbeSteps = int(config_opt.get('MirrorConsProbeSteps', 0) or 0)
    self.Opt_MirrorConsAutoLow = _cfg_num('MirrorConsAutoLow', 0.0015)
    self.Opt_MirrorConsAutoHigh = _cfg_num('MirrorConsAutoHigh', 0.003)
    self.Opt_ValueFocalGamma = float(config_opt.get('ValueFocalGamma', 0) or 0)
    # Run seed for weight init / dropout / sampling (see train.py). None = the
    # historical behaviour of no seeding at all. Pair with the ShuffleSeed key
    # (bridged to CERES_SHUFFLE_SEED, governs data order) when two arms must
    # differ only by the mechanism under test.
    self.Opt_TorchSeed = config_opt.get('TorchSeed', None)
    # DDP communication tuning — only matters when the run is comm-bound, which
    # in practice means ranks spanning more than one NVLink island (see
    # DDP_MULTI_GPU.md). Both default to the previous behaviour.
    #   DDPBucketCapMB   gradient bucket size in MB (0/absent = PyTorch's 25)
    #   DDPBF16Compress  send gradients as bf16 (halves the payload)
    self.Opt_DDPBucketCapMB = config_opt.get('DDPBucketCapMB', 0)
    self.Opt_DDPBF16Compress = bool(config_opt.get('DDPBF16Compress', False))
    # Per-head Muon (Muon only, Kimi K3-style): orthogonalize each attention head's
    # projection block independently (qkv per-head x per-projection on the linear
    # path / per-projection on the nonlinear path; q2/k2/v2/q2b per-head rows;
    # W_h per-head input columns) instead of one NS step over the full fused matrix.
    # False/absent = legacy full-matrix orthogonalization.
    self.Opt_MuonPerHeadAttention = bool(config_opt.get('MuonPerHeadAttention', False))
    # Per-head QK-clip (Kimi K2 MuonClip, kept in the K3 recipe): after each
    # optimizer step, any attention head whose max pre-softcap logit exceeded
    # this threshold gets its Q/K projection rows rescaled by sqrt(tau/max).
    # Weight-level -> zero inference cost, no export divergence, inert once
    # training is stable. 0/null/absent = off. Typical tau ~ SoftCapCutoff
    # territory (e.g. 100); ineffective (and skipped) under UseQKNorm.
    self.Opt_QKClipTau = float(config_opt.get('QKClipTau', 0) or 0)
    # POLICY/VALUE GRADIENT-CONFLICT PROBE (0/absent = off): every N optimizer steps,
    # differentiate the policy-family and value-family loss subtotals SEPARATELY and
    # report the cosine between the two gradients on the parameters they share, plus
    # the value/policy gradient-norm ratio. Measures the late-phase see-saw directly
    # instead of inferring it from gate tables, and — because the trunk, the shared
    # head front-end and the embedding are reported separately — says WHERE the two
    # objectives compete: a negative cosine confined to headPremap/headSharedLinear
    # is the shared-front-end bottleneck, a negative cosine in the trunk is not
    # fixable by any head-side change.
    #
    # Costs two extra backward passes on probe steps only (~2/N overhead; N=500 is
    # well under 1%), measured on the first micro-batch of the accumulation group.
    # Purely diagnostic: it uses torch.autograd.grad, which returns gradients instead of
    # accumulating into .grad — verified by snapshotting every parameter's .grad across
    # the probe block and finding it unchanged on every probe step. (A run-level A/B
    # cannot show this: two byte-identical runs of this trainer already diverge on their
    # own, so a probe-on/probe-off weight comparison measures only that noise.)
    #
    # READ IT BINNED. The per-step cosine is genuinely noisy — a 5-probe toy run spanned
    # -0.44 to +0.53 on the trunk — because each probe is one micro-batch. The signal
    # being hunted (does the angle go negative as the anneal proceeds?) is a trend over
    # tens of probes, so average over bins exactly as with the TRAIN-log diagnostics.
    #
    # Requires single GPU and PyTorchCompileMode off (same constraints as the older
    # per-head gradient-norm logging: extra backwards desync DDP's reducer, and
    # compiled autograd rejects the retained graph). To read a production run's LATE
    # phase, resume its checkpoint uncompiled for a few million positions with this on.
    self.Opt_GradConflictProbeSteps = int(config_opt.get('GradConflictProbeSteps', 0) or 0)
    self.Opt_LRBeginDecayAtFractionComplete = config_opt.get('LRBeginDecayAtFractionComplete', 0.25)
    self.Opt_Beta1 = config_opt.get('Beta1', 0.90)
    self.Opt_Beta2 = config_opt.get('Beta2', 0.98)
    self.Opt_Beta3 = config_opt.get('Beta3', 0.9999)
    self.Opt_Alpha = config_opt.get('Alpha', 5)
    self.Opt_GradientClipLevel = config_opt.get('GradientClipLevel', 1.0)
    self.Opt_LossValueMultiplier = config_opt.get('LossValueMultiplier', 0.5)
    self.Opt_LossValue2Multiplier = config_opt.get('LossValue2Multiplier', 0.0)
    self.Opt_LossValueDMultiplier = config_opt.get('LossValueDMultiplier', 0.0)
    self.Opt_LossValue2DMultiplier = config_opt.get('LossValue2DMultiplier', 0.0)
    self.Opt_LossUncertaintyPolicyMultiplier = config_opt.get('LossUncertaintyPolicyMultiplier', 0.0)
    self.Opt_LossActionMultiplier = config_opt.get('LossActionMultiplier', 0.0)
    self.Opt_LossActionUncertaintyMultiplier = config_opt.get('LossActionUncertaintyMultiplier', 0.0)
    self.Opt_LossQDeviationMultiplier = config_opt.get('LossQDeviationMultiplier', 0.0)    
    self.Opt_LossPolicyMultiplier = config_opt.get('LossPolicyMultiplier', 1)
    self.Opt_LossMLHMultiplier = config_opt.get('LossMLHMultiplier', 0)
    self.Opt_LossUNCMultiplier = config_opt.get('LossUNCMultiplier', 0)

    # KL-divergence anchor (RLHF-style reference-model regularization).
    # When KLAnchorRefCheckpoint is set and at least one weight is > 0,
    # train.py loads a frozen reference and adds beta * KL(student || ref) per batch.
    self.Opt_KLAnchorRefCheckpoint = config_opt.get('KLAnchorRefCheckpoint', None)
    self.Opt_KLAnchorPolicyWeight = config_opt.get('KLAnchorPolicyWeight', 0.0)
    self.Opt_KLAnchorValueWeight = config_opt.get('KLAnchorValueWeight', 0.0)

    self.Opt_TestValue = config_opt.get('TestValue', 0)

    # Initialize class members from config_net_def
    self.NetDef_TrainOn4BoardSequences = config_net_def.get('TrainOn4BoardSequences', False)
    if not self.NetDef_TrainOn4BoardSequences:
      self.Opt_LossActionMultiplier = 0
      self.Opt_LossActionUncertaintyMultiplier = 0
      self.Opt_LossValueDMultiplier = 0
      self.Opt_LossValue2DMultiplier = 0

    self.NetDef_ModelDim = config_net_def.get('ModelDim', 256)
    self.NetDef_NumLayers = config_net_def.get('NumLayers', 8)
    # Plan 3: per-position chess-specific attention bias (piece-attack patterns
    # + static square-pair geometry). When True, a single PieceRelationBias
    # module is constructed and its output is added to attention logits in
    # every encoder layer. Default False reduces to baseline behaviour.
    self.NetDef_UsePieceRelationBias = config_net_def.get('UsePieceRelationBias', False)
    # Visibility edge bias v2 (Kovax visibility program): pairwise edge channels
    # + per-layer form-A attention-logit bias, optional B/C content gates.
    # CONFIG-ONLY by design (no env override): these change the parameter tree,
    # so they must live in the net config for checkpoints to be reloadable and
    # re-exportable from config alone (same rationale as LearningRateBaseHeads).
    # Ray attention bias (the pre-VisEdge slider-only edge bias). Config field
    # added 2026-08-14 per the config-over-env rule; the legacy
    # CERES_RAY_ATTENTION_BIAS env var is RETIRED and asserts in ceres_net
    # (the transitional fallback ended with the 2026-08 prod ray6 200M run).
    self.NetDef_UseRayAttentionBias = config_net_def.get('UseRayAttentionBias', False)
    self.NetDef_UseVisEdgeBias = config_net_def.get('UseVisEdgeBias', False)
    self.NetDef_VisEdgeFamilies = config_net_def.get('VisEdgeFamilies', 'vis,xray,pinray')
    self.NetDef_VisEdgeGates = config_net_def.get('VisEdgeGates', '')
    self.NetDef_VisEdgeSharedProjection = config_net_def.get('VisEdgeSharedProjection', False)
    # Graph-route heads (2026-08 tactical program): gated exact routing over the
    # visibility edge channels; requires UseVisEdgeBias (see ceres_net).
    self.NetDef_UseGraphRouteHeads = config_net_def.get('UseGraphRouteHeads', False)
    # Soft-min ("AND-logic") value-aggregation heads (2026-08 tactical program):
    # first k heads/layer aggregate V by attention-weighted soft minimum.
    # ARCH key (changes aggregation from step 0), not a zero-init add-on. 0 = off.
    self.NetDef_SoftMinHeads = config_net_def.get('SoftMinHeads', 0)
    self.NetDef_SoftMaxAggHeads = config_net_def.get('SoftMaxAggHeads', 0)
    self.NetDef_UseHeadLogitTemp = config_net_def.get('UseHeadLogitTemp', False)
    self.NetDef_UseTacticalCodebook = config_net_def.get('UseTacticalCodebook', False)
    self.NetDef_UseKingDistChannels = config_net_def.get('UseKingDistChannels', False)
    self.NetDef_UseSpectralPE = config_net_def.get('UseSpectralPE', False)
    self.NetDef_UseDualPlane = config_net_def.get('UseDualPlane', False)
    self.NetDef_DualPlaneSoftMinHeads = config_net_def.get('DualPlaneSoftMinHeads', 2)
    self.NetDef_DualPlanePolicyDecode = config_net_def.get('DualPlanePolicyDecode', False)
    self.NetDef_DualPlaneDim = config_net_def.get('DualPlaneDim', 128)
    self.NetDef_DualPlaneLayers = config_net_def.get('DualPlaneLayers', 2)
    self.NetDef_DualPlaneInterleave = config_net_def.get('DualPlaneInterleave', False)
    self.NetDef_DualPlaneSurvivalAux = config_net_def.get('DualPlaneSurvivalAux', 0.0)
    self.NetDef_DualPlaneSurvivalK = config_net_def.get('DualPlaneSurvivalK', 4)
    self.NetDef_DualPlaneValueAttention = config_net_def.get('DualPlaneValueAttention', 0)
    self.NetDef_DualPlaneVictimDecode = config_net_def.get('DualPlaneVictimDecode', False)
    self.NetDef_MoveEdgeDecode = config_net_def.get('MoveEdgeDecode', False)
    self.NetDef_DualPlaneRelDegrees = config_net_def.get('DualPlaneRelDegrees', False)
    self.NetDef_MoveDegreeDecode = config_net_def.get('MoveDegreeDecode', False)
    self.NetDef_DualPlaneRelGains = config_net_def.get('DualPlaneRelGains', False)
    # Value min/max pool side-channels (2026-08 tactics toolbox T1.2): trunk
    # amin/amax over squares into the value family's hidden pre-activation.
    # Zero-init no-op add-on (the 'inject' class).
    self.NetDef_ValueHeadMinMaxPool = config_net_def.get('ValueHeadMinMaxPool', False)
    self.NetDef_ValueHeadPoolChannels = config_net_def.get('ValueHeadPoolChannels', False)
    # Iterated tactic refiner (see tactical_refiner.py): 0 = off.
    self.NetDef_RefinerIters = config_net_def.get('RefinerIters', 0)
    self.NetDef_RefinerDim = config_net_def.get('RefinerDim', 128)
    self.NetDef_RefinerHeads = config_net_def.get('RefinerHeads', 4)
    self.NetDef_RefinerFFNMult = config_net_def.get('RefinerFFNMult', 2)
    self.NetDef_RefinerDeepSupWeight = config_net_def.get('RefinerDeepSupWeight', 0.0)
    # Looped transformer: NumLayers represents the EFFECTIVE depth (number of
    # encoder layer applications). LoopCount controls weight tying — when >1,
    # NumLayers/LoopCount distinct EncoderLayer modules are constructed and each
    # is applied LoopCount times in sequence. LoopCount=1 (default) is the
    # standard untied behaviour. NumLayers must be divisible by LoopCount.
    self.NetDef_LoopCount = config_net_def.get('LoopCount', 1)
    if self.NetDef_NumLayers % self.NetDef_LoopCount != 0:
      raise ValueError(
        f"NumLayers ({self.NetDef_NumLayers}) must be divisible by LoopCount "
        f"({self.NetDef_LoopCount}) for looped transformer.")
    self.NetDef_NumHeads = config_net_def.get('NumHeads', 8)
    self.NetDef_UseQKV = config_net_def.get('UseQKV', True)
    self.NetDef_DualAttentionMode = config_net_def.get('DualAttentionMode', 'None')
    self.NetDef_PreNorm = config_net_def.get('PreNorm', False)
    self.NetDef_NormType = config_net_def.get('NormType', 'LayerNorm')
    self.NetDef_AttentionMultiplier = config_net_def.get('AttentionMultiplier', 1)
    self.NetDef_NonLinearAttention = config_net_def.get('NonLinearAttention', False)
    self.NetDef_FFNMultiplier = config_net_def.get('FFNMultiplier', 1)
    self.NetDef_FFNActivationType = config_net_def.get('FFNActivationType', 'ReLUSquared')
    self.NetDef_FFNUseGlobalEveryNLayers = config_net_def.get('FFNUseGlobalEveryNLayers', 0)
    self.NetDef_HeadsActivationType = config_net_def.get('HeadsActivationType', 'ReLU')
    self.NetDef_PriorStateDim = config_net_def.get('PriorStateDim', 64)  
    self.NetDef_DeepNorm = config_net_def.get('DeepNorm', False) 
    self.NetDef_DenseFormer = config_net_def.get('DenseFormer', False) 
    self.NetDef_SmolgenDimPerSquare = config_net_def.get('SmolgenDimPerSquare', 32)
    self.NetDef_SmolgenDim = config_net_def.get('SmolgenDim', 512)
    self.NetDef_SmolgenToHeadDivisor = config_net_def.get('SmolgenToHeadDivisor', 2)
    self.NetDef_SmolgenActivationType = config_net_def.get('SmolgenActivationType', 'None')
    # Variant A per-layer smolgen delta: per-encoder-layer low-rank zero-init
    # additive correction on top of the shared smolgenPrepLayer output. 0 = off
    # (baseline smolgen). > 0 = enable with that rank (typically 4 or 8). Only
    # active when smolgen itself is enabled (SmolgenDim and SmolgenDimPerSquare
    # both > 0). Module is zero-initialized so model is bit-identical to
    # baseline at training step 0.
    self.NetDef_SmolgenDeltaRank = config_net_def.get('SmolgenDeltaRank', 0)
    self.NetDef_HeadWidthMultiplier = config_net_def.get('HeadWidthMultiplier', 4)
    # Width of the SHARED head front-end: HEAD_IN_SIZE = 64 * (HEAD_PREMAP_PER_SQUARE //
    # HeadSharedLinearDiv). 4 = the long-standing hardcoded value, so absent/4 reproduces
    # every existing net exactly; 2 doubles the vector every head reads, 1 quadruples it.
    #
    # Why it is a knob now: the gradient-conflict probe showed the value family sends ~11x
    # MORE gradient magnitude through headPremap/headSharedLinear than the policy family
    # does, with a cosine of ~0 between them — so the shared front-end is not a site where
    # policy crowds value out, and the private-value-head hypothesis reduces to a claim
    # about WIDTH (256 dims at D=256/mult 4, vs lc0's 2048-dim private per-square vector).
    # Width is testable directly and symmetrically: if value gains as much from a wider
    # SHARED front-end as from a private one, "private" was never the operative word.
    # Note this widens the input of every head, so any gain must be read against the
    # parameters added (policy is the internal control: a front-end-specific effect should
    # move value more than policy).
    self.NetDef_HeadSharedLinearDiv = int(config_net_def.get('HeadSharedLinearDiv', 4) or 4)
    if self.NetDef_HeadSharedLinearDiv < 1:
      raise ValueError(f"HeadSharedLinearDiv must be >= 1, got {self.NetDef_HeadSharedLinearDiv}")
    self.NetDef_UseRPE = config_net_def.get('UseRPE', False)
    self.NetDef_UseRPE_V = config_net_def.get('UseRPE_V', True)
    self.NetDef_UseRelBias = config_net_def.get('UseRelBias', False)
    self.NetDef_UseRoPE = config_net_def.get('UseRoPE', False)
    # Differential Attention V2 (Microsoft Apr 2026): doubles Q heads (Q1, Q2
    # split), computes two attention maps, subtracts with per-token sigmoid(λ)
    # gate to cancel attention noise. When smolgen is on, bias is added to both
    # attention maps (Option A — both branches inherit same per-position prior;
    # differential cancels Q1-vs-Q2 noise on top). Untested with RoPE / softcap.
    self.NetDef_UseDiffAttention = config_net_def.get('UseDiffAttention', False)
    self.NetDef_UseQKNorm = config_net_def.get('UseQKNorm', False)
    self.NetDef_SoftCapCutoff = config_net_def.get('SoftCapCutoff', 100)

    self.NetDef_TestValue = config_net_def.get('TestValue', 0)

    # SoftMoEConfig is a nested structure, so it requires special handling
    soft_moe_config = config_net_def.get('SoftMoEConfig', {})
    self.NetDef_SoftMoE_MoEMode = soft_moe_config.get('MoEMode', 'None')
    self.NetDef_SoftMoE_OnlyForAlternatingLayers = soft_moe_config.get('OnlyForAlternatingLayers', True)
    self.NetDef_SoftMoE_NumExperts = soft_moe_config.get('NumExperts', 0)
    self.NetDef_SoftMoE_NumSlotsPerExpert = soft_moe_config.get('NumSlotsPerExpert', 0)
    self.NetDef_SoftMoE_UseNormalization = soft_moe_config.get('UseNormalization', False)
    self.NetDef_SoftMoE_UseBias = soft_moe_config.get('UseBias', True)
    # Fine-grained MoE: pre-projects ffn_dim -> ExpertInputDim before phi
    # routing + expert processing. Only meaningful in ReplaceLinearSecondLayer
    # mode. 0 disables (default — experts process full ffn_dim).
    self.NetDef_SoftMoE_ExpertInputDim = soft_moe_config.get('ExpertInputDim', 0)

    # TSB (Tactical SwiGLU Bypass) — nested config block.
    # Per-block parallel SwiGLU FFN + scalar gate added beside the orig FFN.
    # When Enabled=false (default), all TSB code paths are no-ops.
    tsb_config = config_net_def.get('TSBConfig', {})
    self.NetDef_TSB_Enabled = tsb_config.get('Enabled', False)
    self.NetDef_TSB_FFNMultiplier = tsb_config.get('FFNMultiplier', 1)
    self.NetDef_TSB_GateBiasInit = tsb_config.get('GateBiasInit', -4.0)
    self.NetDef_TSB_GateMLPHiddenDivisor = tsb_config.get('GateMLPHiddenDivisor', 8)


  def pretty_print(self):
    print("")
    print("CONFIGURATION OF TRAINING RUN:", self.id)
    attributes = vars(self)
    sorted_attributes = sorted(attributes.items(), key=lambda x: x[0])

    # Pretty print the sorted attributes
    for attr_name, attr_value in sorted_attributes:
      print(f"{attr_name}: {attr_value}")


def Test():
  dir = '.'
  config = Configuration(dir, 'test')

  config.pretty_print()
