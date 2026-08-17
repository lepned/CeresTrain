"""Config -> environment bootstrap, shared by train.py and the standalone tools.

A run's data-format and sidecar settings are consumed at IMPORT time by
module-level reads (config.py, tpg_dataset.py, losses.py), so plain
Configuration fields arrive too late to shape them. This module bridges them
from the run's `<ID>_ceres_opt.json` into os.environ, and MUST be called
before importing those modules.

Why it is shared: the same settings decide how a CHECKPOINT is interpreted
(input width, aux channels, shard format). A tool that rebuilds a net from a
checkpoint — recover_export.py, reconvert_onnx.py — must see the same values
the training run saw, or it silently rebuilds a differently-shaped net and
re-exports garbage. That was the 2026-08-07 export incident (an export shell
without CERES_AUX_FEATURES_PER_SQUARE=0 crashed on width mismatch), and it
stayed possible for every tool other than train.py until this module existed.

Precedence: a key PRESENT in the config overrides the environment (config is
authoritative); absent keys leave the environment untouched (env fallback for
legacy runs whose configs predate these keys).

Usage (before the heavy imports):

    from config_bootstrap import bootstrap_env_from_config
    bootstrap_env_from_config(OUTPUTS_DIR, TRAINING_ID)

    from config import Configuration        # now reads the bridged values
"""
import json
import os
import sys

# Friendly config key -> environment variable.
BOOTSTRAP_ENV_MAP = {
    # data format / sidecars
    'TPGV3': 'CERES_TPG_V3',
    'AuxFeaturesPerSquare': 'CERES_AUX_FEATURES_PER_SQUARE',
    # Per-corpus shard format. TPGV3 is the whole-run toggle; these two are the
    # explicit widths, and SquareBytes2 is what a MIXED run needs (e.g. a V3
    # 141-byte primary with a V2 137-byte secondary). NB a 137-byte corpus
    # carries no aux bytes, so any V2 dataset forces AuxFeaturesPerSquare=0
    # for the whole run.
    'SquareBytes': 'CERES_TPG_SQUARE_BYTES',
    'SquareBytes2': 'CERES_TPG_SQUARE_BYTES2',
    'AuxChannelIndices': 'CERES_AUX_CHANNEL_INDICES',
    'TargetSidecar': 'CERES_TPG_TARGET_SIDECAR',
    'V7XSidecar': 'CERES_TPG_V7X_SIDECAR',
    'SurvivalHorizon': 'CERES_SURVIVAL_HORIZON',
    'SurvivalTargetWeight': 'CERES_SURVIVAL_TARGET_WEIGHT',
    'SurvivalLossBuckets': 'CERES_SURVIVAL_LOSS_BUCKETS',
    'SurvivalCaptureWeight': 'CERES_SURVIVAL_CAPTURE_WEIGHT',
    # stream routing / data augmentation
    'SecondaryLossPolicyMult': 'CERES_SECONDARY_LOSS_POLICY_MULT',
    'SecondaryLossValueMult': 'CERES_SECONDARY_LOSS_VALUE_MULT',
    'SecondaryLossValue2Mult': 'CERES_SECONDARY_LOSS_VALUE2_MULT',
    'SecondaryLossAuxMult': 'CERES_SECONDARY_LOSS_AUX_MULT',
    'SecondaryLossPlacementMult': 'CERES_SECONDARY_LOSS_PLACEMENT_MULT',
    'SecondaryLossSurvivalMult': 'CERES_SECONDARY_LOSS_SURVIVAL_MULT',
    'SecondaryLossStvalueMult': 'CERES_SECONDARY_LOSS_STVALUE_MULT',
    'MixProloguePositions': 'CERES_MIX_PROLOGUE_POSITIONS',
    'FileMirrorAug': 'CERES_FILE_MIRROR_AUG',
    'KeepDrawProb': 'CERES_KEEP_DRAW_PROB',
}


def normalize(value):
    """Normalize JSON literals to the string forms the env consumers parse
    (2026-08-07 review): booleans -> '1'/'0' (consumers compare == '0' or call
    int(); str(True) would silently select the WRONG branch), integral floats
    -> int string (int('4.0') raises), lists -> comma-join (the natural JSON
    spelling of e.g. SurvivalLossBuckets [2,4] -> '2,4')."""
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (list, tuple)):
        return ','.join(normalize(v) for v in value)
    return str(value)


def bootstrap_env_from_config(outputs_dir, training_id, quiet=False, configs_dir=None):
    """Bridge <outputs_dir>/configs/<training_id>_ceres_opt.json into os.environ.

    configs_dir: pass the configs directory directly when a caller already has
    it (some tools take a config dir rather than an outputs dir); outputs_dir
    is then ignored.

    Returns the dict of variables actually applied ({} when there is no config
    or the bridge was skipped). Never raises for a missing/legacy config — an
    env-only run must keep working — but a malformed "Env" SECTION is a hard
    exit: the config exists and is wrong, and running on with silently dropped
    settings is worse than stopping.
    """
    applied = {}
    try:
        _dir = configs_dir if configs_dir is not None else os.path.join(outputs_dir, 'configs')
        path = os.path.join(_dir, str(training_id) + '_ceres_opt.json')
        if not os.path.isfile(path):
            if not quiet:
                print(f'[bootstrap] no opt-config at {path}; env-only run')
            return applied
        with open(path, encoding='utf-8') as f:
            cfg = json.load(f)
        # Collect EVERYTHING first, apply atomically afterwards — a malformed
        # "Env" section must not leave a half-bridged environment behind.
        pending = {}
        for key, env_name in BOOTSTRAP_ENV_MAP.items():
            if key in cfg and cfg[key] is not None:
                if cfg[key] == '':
                    print(f'[bootstrap] WARNING: {key} is an empty string; skipped '
                          f'(use null/omit to fall back to env)')
                    continue
                pending[env_name] = normalize(cfg[key])
        # Generic escape hatch: an "Env" dict in the config is bridged with the
        # same normalization (applied AFTER the friendly names, so Env wins on
        # collision). Lets ANY env-gated knob — arch flags, probes, one-off
        # switches — live in the run's config without mapping maintenance.
        env_section = cfg.get('Env')
        if env_section is not None:
            if not isinstance(env_section, dict):
                print(f'[bootstrap] ERROR: "Env" must be a JSON object of VAR: value, '
                      f'got {type(env_section).__name__}')
                sys.exit(1)
            for k, v in env_section.items():
                if v is None or v == '':
                    print(f'[bootstrap] WARNING: Env.{k} is null/empty; skipped')
                    continue
                pending[str(k)] = normalize(v)
        for k, v in pending.items():
            os.environ[k] = v
            if not quiet:
                print(f'[bootstrap] {k}={v} (from config)')
        applied = pending
    except SystemExit:
        raise
    except Exception as e:  # never let the bootstrap kill a legacy env-driven run
        print(f'[bootstrap] WARNING: config->env bridge failed ({e}); using env as-is')
    return applied
