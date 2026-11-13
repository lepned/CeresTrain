"""Resume INTO A NET WITH MORE SWITCHES through the real train.py path (opt CheckpointResumeAllowNewParameters;
resume_new_params.py), on the tiny hybrid net, CPU.

    python test_bt6_hybrid_resume.py <work dir> [training files dir] [stage ...]

Stages (default: all): base, normal, a, c, refuse, ddp.
  base     train.py from scratch: tiny net, BT6TermsInnerLayers [0], no smolgen; checkpoint at the end.
  normal   plain resume of that checkpoint (no new switch), with the allow key off and on: the resumed model and
           optimizer state equal the checkpoint bit for bit.
  a        resume into terms in ALL inner layers.          c   resume into BT6SharedSmolgen.
           For both: outputs of the resumed net before its first step vs the checkpointed net (float32 eval, and
           training mode under bf16 autocast), optimizer state of every existing parameter == the checkpoint's,
           new parameters without state, the run trains on and writes its own checkpoint, new parameters move.
  refuse   resumes that must fail: new parameters without the allow key; BT6ShareReadersPerBlock; a shape change;
           a switch turned off.
  ddp      transition a under torchrun, 2 ranks, gloo.
Every train.py run is a subprocess with CUDA hidden; its log is <work dir>/logs_test/<id>.log.
"""
import glob
import io
import json
import os
import subprocess
import sys

os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WORK = os.path.abspath(sys.argv[1])
DATA_DIR = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isalpha() else '/mnt/d/T80_v2_sub'
STAGES = [s for s in sys.argv[2:] if s.isalpha()] or ['base', 'normal', 'a', 'c', 'refuse', 'ddp']
BASE_POSITIONS, MORE_POSITIONS, BATCH = 1024, 256, 64

NET = {
  "TrainOn4BoardSequences": False, "LoRARankDivisor": 0, "UseQKV": True, "DualAttentionMode": "None",
  "PreNorm": False, "NormType": "RMSNorm", "AttentionMultiplier": 1, "FFNMultiplier": 1, "FFNActivationType": "SwiGLU",
  "FFNUseGlobalEveryNLayers": 0, "HeadsActivationType": "Mish", "PriorStateDim": 0, "NonLinearAttention": False,
  "SoftCapCutoff": 0, "UseQKNorm": False, "DeepNorm": False, "DenseFormer": False,
  "SmolgenDimPerSquare": 0, "SmolgenDim": 0, "SmolgenToHeadDivisor": 1, "SmolgenActivationType": "Swish",
  "UseRPE": False, "UseRPE_V": False, "UseRoPE": False, "UseRelBias": False, "HeadWidthMultiplier": 4,
  "UseVisEdgeBias": False, "RefinerIters": 0, "UseDiffAttention": False, "UseDualPlane": False,
  "UseMoveTokens": False, "EGTEdgeStream": False,
  "NBTInnerPreNorm": True, "NBTProjActivation": "None", "NBTProjNorm": "Norm", "NBTFixupInit": False,
  "NBTSharedSmolgen": False, "BT6Hybrid": True, "BT6Embedding": "bt6", "BT6CheckpointBlocks": False,
  # tiny: outer 128 / inner 64 (4 heads x 16) / 2 blocks x 3 inner layers, site after block 0, terms in layer 0 only
  "ModelDim": 128, "NumLayers": 2, "NumHeads": 4, "NBTInnerLayers": 3, "NBTWidthDivisor": 2, "NBTInnerHeads": 4,
  "BT6EdgeSites": [0], "BT6PolicyWidth": 64, "BT6SoftPolicyWidth": 32, "BT6SiteHidden": 32, "BT6EmbeddingDense": 16,
  "BT6TermsInnerLayers": [0],
}
OPT = {
  "LoRARankDivisor": 0, "Optimizer": "Muon", "CheckpointResumeFromFileName": None, "PyTorchCompileMode": None,
  "WeightDecay": 0.01, "MuonHonorNoDecay": True, "MuonHonorNoDecayScope": "all",
  "LearningRateBase": 0.002, "LRBeginDecayAtFractionComplete": 0.7,
  "Beta1": 0.9, "MuonMomentum": 0.9, "Beta2": 0.95, "Beta3": 0.9999, "Alpha": 5, "GradientClipLevel": 1.0,
  "EMAPeriodSteps": -1, "EMAMaxN": 10,
  "LossValueMultiplier": 1.0, "LossValue2Multiplier": 0.04, "LossPolicyMultiplier": 1, "LossMLHMultiplier": 0,
  "LossUNCMultiplier": 0.005, "LossQDeviationMultiplier": 0, "LossUncertaintyPolicyMultiplier": 0,
  "LossValueDMultiplier": 0, "LossValue2DMultiplier": 0, "LossActionMultiplier": 0, "LossActionUncertaintyMultiplier": 0,
  "SoftPolicyWeight": 1.0, "SoftPolicyTemp": 4.0,
  "TPGV3": 0, "AuxFeaturesPerSquare": 0, "TorchSeed": 777, "ShuffleSeed": 20261113,
  "MuonAdamWScope": "all-non-trunk", "SquareBytes": 137, "LearningRateHeadsRatio": 0.5,
  "DDPBF16Compress": False, "DDPManualGradSync": False, "MuonDistributed": False,
  "NumTrainingPositions": BASE_POSITIONS, "BatchSizeForwardPass": BATCH, "BatchSizeBackwardPass": BATCH,
  "CheckpointFrequencyNumPositions": 100000000,
}
DATA = {"SourceType": "DirectFromPositionGenerator", "TrainingFilesDirectory": DATA_DIR, "TrainingFilesDirectory2": None,
        "RatioSet1ToSet2": 0, "NumTPGFilesToSkip": 0, "FractionQ": 1, "WDLLabelSmoothing": 0, "SquareBytesUnsigned": True}
EXEC = {"DeviceType": "cpu", "DeviceIDs": [0], "DataType": "BFloat16", "UseFP8": False, "UseHistory": True,
        "RunInDocker": False, "DropoutRate": 0, "DropoutDuringInference": False, "EngineType": "CSharpViaTorchscript",
        "ExportOnly": False, "TestFlag": False, "TestValue": 0}
A_NET = {"BT6TermsInnerLayers": None}                                    # (a) terms in all inner layers
B_NET = {"BT6TermsInnerLayers": None, "BT6ShareReadersPerBlock": True}    # (b) + one reader set per block
C_NET = {"BT6SharedSmolgen": True, "BT6SharedSmolgenDim": 48, "BT6SmolgenPerSquareDim": 8, "BT6SmolgenHeadDim": 16}


def write_config(tag, net=None, opt=None, execution=None):
  os.makedirs(os.path.join(WORK, 'configs'), exist_ok=True)
  net_cfg = {k: v for k, v in {**NET, **(net or {})}.items() if not (k == 'BT6TermsInnerLayers' and v is None)}
  for suffix, obj in (('net', net_cfg), ('opt', {**OPT, **(opt or {})}), ('data', DATA),
                      ('exec', {**EXEC, **(execution or {}), 'ID': tag}), ('monitoring', {"CustomMonitoringIntervalSeconds": 600})):
    with open(os.path.join(WORK, 'configs', f'{tag}_ceres_{suffix}.json'), 'w') as handle:
      json.dump(obj, handle, indent=1)


def run_train(tag, dump=None, ranks=1, expect_fail=False):
  os.makedirs(os.path.join(WORK, 'logs_test'), exist_ok=True)
  env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'CERES_NUM_DATASET_WORKERS': '1', 'OMP_NUM_THREADS': '4'}
  env.pop('CERES_RESUME_DUMP', None)
  if dump:
    env['CERES_RESUME_DUMP'] = dump
  if ranks > 1:
    torchrun = os.path.join(os.path.dirname(sys.executable), 'torchrun')
    command = [torchrun, '--standalone', f'--nproc_per_node={ranks}', 'train.py', tag, WORK]
    env['CERES_DDP_STATIC_GRAPH'] = '1'
    env['CERES_DDP_BACKEND'] = 'gloo'
  else:
    command = [sys.executable, 'train.py', tag, WORK]
  log = os.path.join(WORK, 'logs_test', f'{tag}.log')
  with open(log, 'w') as handle:
    code = subprocess.call(command, cwd=HERE, env=env, stdout=handle, stderr=subprocess.STDOUT)
  text = open(log, errors='replace').read()
  ok = code == 0 and 'EXIT_STATUS SUCCESS' in text
  if expect_fail:
    assert not ok, f'{tag}: the run was expected to be refused but succeeded ({log})'
  else:
    assert ok, f'{tag}: train.py failed (exit {code}), see {log}:\n' + text[-3000:]
  return text


def checkpoint_of(tag, positions):
  found = [p for p in glob.glob(os.path.join(WORK, 'nets', f'ckpt_*_{tag}_{positions}')) if not p.endswith('.json')]
  assert len(found) == 1, (tag, positions, found)
  return found[0]


def build(tag):
  from bt6_hybrid_export_check import build_net
  return build_net(WORK, tag)[0]


def outputs(net, state, mode):
  from test_bt6_hybrid import records
  net.load_state_dict(state)
  sq = records(8)
  with torch.no_grad():
    if mode == 'eval':
      net.eval()
      out = net(sq, None)
    else:
      net.train()
      with torch.autocast('cpu', dtype=torch.bfloat16):
        out = net(sq, None)
      net._last_sp_out = None
  return [o.float() for o in out[:5]]


def rel(a, b):
  return float((a - b).abs().max() / (b.abs().max() + 1e-12))


def same_entry(a, b):
  if a.keys() != b.keys():
    return False
  return all(torch.equal(a[k], b[k]) if torch.is_tensor(a[k]) else a[k] == b[k] for k in a)


def compare_resume(tag, base_tag, ckpt_path, base_names):
  """The dump train.py wrote right after resuming `tag` (before its first step) against the checkpoint."""
  ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
  dump = torch.load(os.path.join(WORK, f'dump_{tag}.pt'), map_location='cpu', weights_only=False)
  ckpt_model = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
  new_keys = sorted(k for k in dump['model'] if k not in ckpt_model)
  assert all(torch.equal(dump['model'][k], v) for k, v in ckpt_model.items()), 'an existing tensor changed on resume'
  assert dump['num_pos'] == int(ckpt['num_pos']) == BASE_POSITIONS
  # optimizer: every existing parameter's state == the checkpoint's (by name); new parameters carry no moments
  old_state = {name: ckpt['optimizer']['state'].get(i, {}) for i, name in enumerate(base_names)}
  new_state = {name: dump['optimizer']['state'].get(i, {}) for i, name in enumerate(dump['optimizer_names'])}
  assert set(old_state) <= set(new_state)
  moments = ('momentum_buffer', 'moment1', 'moment2', 'step')
  kept = 0
  for name, entry in old_state.items():
    assert same_entry(entry, new_state[name]), f'optimizer state of {name} differs from the checkpoint'
    kept += any(k in entry for k in moments)
  fresh = [n for n in new_state if n not in old_state]
  assert all(not any(k in new_state[n] for k in moments) for n in fresh), 'a new parameter starts with moments'
  # function: the resumed net before its first step vs the checkpointed net
  base_net, new_net = build(base_tag), build(tag)
  diffs = {}
  for mode in ('eval', 'bf16-train'):
    a, b = outputs(base_net, ckpt_model, mode), outputs(new_net, dump['model'], mode)
    diffs[mode] = max(rel(u, w) for u, w in zip(b, a))
    diffs[mode + ' bitwise'] = all(torch.equal(u, w) for u, w in zip(a, b))
  return new_keys, kept, len(fresh), diffs, dump


def stage_base():
  write_config('rs_base')
  run_train('rs_base')
  path = checkpoint_of('rs_base', BASE_POSITIONS)
  assert os.path.exists(path + '.datastream.json')
  print(f'base: train.py trained the tiny [0] net for {BASE_POSITIONS} positions on CPU; checkpoint {os.path.basename(path)}')


def resume_opt(extra=None):
  return {'CheckpointResumeFromFileName': checkpoint_of('rs_base', BASE_POSITIONS),
          'NumTrainingPositions': BASE_POSITIONS + MORE_POSITIONS, **(extra or {})}


def base_names():
  write_config('rs_normal', opt=resume_opt())
  dump = os.path.join(WORK, 'dump_rs_normal.pt')
  if not os.path.exists(dump):
    run_train('rs_normal', dump=dump)
  return torch.load(dump, map_location='cpu', weights_only=False)['optimizer_names']


def stage_normal():
  ckpt_path = checkpoint_of('rs_base', BASE_POSITIONS)
  for tag, extra in (('rs_normal', None), ('rs_normal_allow', {'CheckpointResumeAllowNewParameters': True})):
    write_config(tag, opt=resume_opt(extra))
    text = run_train(tag, dump=os.path.join(WORK, f'dump_{tag}.pt'))
    names = torch.load(os.path.join(WORK, f'dump_{tag}.pt'), map_location='cpu', weights_only=False)['optimizer_names']
    new_keys, kept, fresh, diffs, dump = compare_resume(tag, 'rs_base', ckpt_path, names)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert not new_keys and fresh == 0 and diffs['eval bitwise'] and diffs['bf16-train bitwise']
    assert len(dump['optimizer']['state']) == len(ckpt['optimizer']['state'])
    assert all(same_entry(ckpt['optimizer']['state'][i], dump['optimizer']['state'][i]) for i in ckpt['optimizer']['state'])
    assert 'new parameters on resume' not in text
    checkpoint_of(tag, BASE_POSITIONS + MORE_POSITIONS)
    print(f'normal resume ({tag}): model and optimizer state after the resume == the checkpoint bit for bit '
          f'({len(ckpt["model"])} tensors, {kept} parameters with moments); outputs bit-identical; trained on to '
          f'{BASE_POSITIONS + MORE_POSITIONS}')
  # the two resumes (allow key off / on) end in the same weights if this machine's CPU training is reproducible
  a = torch.load(checkpoint_of('rs_normal', BASE_POSITIONS + MORE_POSITIONS), map_location='cpu', weights_only=False)['model']
  b = torch.load(checkpoint_of('rs_normal_allow', BASE_POSITIONS + MORE_POSITIONS), map_location='cpu', weights_only=False)['model']
  worst = max(rel(b[k].float(), a[k].float()) for k in a if a[k].is_floating_point())
  print(f'normal resume, allow key off vs on, after {MORE_POSITIONS} more positions: final weights max relative difference {worst:.1e}')


def stage_transition(tag, net_over, label, moved_suffix, ranks=1, opt_extra=None):
  ckpt_path = checkpoint_of('rs_base', BASE_POSITIONS)
  names = base_names()
  write_config(tag, net=net_over, opt=resume_opt({'CheckpointResumeAllowNewParameters': True, **(opt_extra or {})}))
  text = run_train(tag, dump=os.path.join(WORK, f'dump_{tag}.pt'), ranks=ranks)
  new_keys, kept, fresh, diffs, dump = compare_resume(tag, 'rs_base', ckpt_path, names)
  assert new_keys and fresh > 0 and kept > 50
  assert diffs['eval'] < 2e-5, diffs
  assert 'optimizer state KEPT' in text and 'CheckpointResumeAllowNewParameters:' in text
  final = torch.load(checkpoint_of(tag, BASE_POSITIONS + MORE_POSITIONS), map_location='cpu', weights_only=False)
  moved = [k for k in new_keys if k.endswith(moved_suffix) and not torch.equal(final['model'][k].cpu(), dump['model'][k])]
  assert moved, f'no new parameter ending in {moved_suffix} moved after the resume'
  assert int(final['num_pos']) == BASE_POSITIONS + MORE_POSITIONS
  assert all(torch.isfinite(v).all() for v in final['model'].values() if v.is_floating_point())
  print(f'{label} ({tag}{", 2 ranks gloo" if ranks > 1 else ""}): {len(new_keys)} new tensors; every existing tensor == checkpoint; '
        f'outputs before the first step vs the checkpointed net: float32 eval max relative {diffs["eval"]:.1e} (bitwise '
        f'{diffs["eval bitwise"]}), training mode under bf16 autocast {diffs["bf16-train"]:.1e} (bitwise '
        f'{diffs["bf16-train bitwise"]}); optimizer state of {kept} existing parameters with moments == checkpoint, {fresh} new '
        f'parameters without moments; trained on to {BASE_POSITIONS + MORE_POSITIONS}, all finite, {len(moved)} new '
        f'{moved_suffix} tensors moved')


def stage_refuse():
  cases = (
    ('rs_ref_noallow', A_NET, {}, 'Missing key(s)', 'new parameters WITHOUT the allow key (today\'s strict load)'),
    ('rs_ref_shared', B_NET, {'CheckpointResumeAllowNewParameters': True}, 'BT6ShareReadersPerBlock switched on at resume',
     'BT6ShareReadersPerBlock at resume'),
    ('rs_ref_shape', {'BT6SiteHidden': 48}, {'CheckpointResumeAllowNewParameters': True}, 'tensors changed shape',
     'a shape change (BT6SiteHidden 32 -> 48)'),
    ('rs_ref_soft', {}, {'CheckpointResumeAllowNewParameters': True, 'SoftPolicyWeight': 0}, 'have no place in this net',
     'a part turned OFF (soft-policy head removed)'),
    ('rs_ref_embed', {'BT6Embedding': 'linear'}, {'CheckpointResumeAllowNewParameters': True}, 'not parameters of a switch',
     'missing keys that no switch explains (embedding form changed)'),
  )
  for tag, net_over, opt_extra, needle, label in cases:
    write_config(tag, net=net_over, opt=resume_opt(opt_extra))
    text = run_train(tag, expect_fail=True)
    assert needle in text, f'{tag}: expected {needle!r} in the log'
    assert not glob.glob(os.path.join(WORK, 'nets', f'ckpt_*_{tag}_*')), f'{tag}: a refused run wrote a checkpoint'
    print(f'refused: {label} -> "{needle}"')


if __name__ == '__main__':
  if 'base' in STAGES:
    stage_base()
  if 'normal' in STAGES:
    stage_normal()
  if 'a' in STAGES:
    stage_transition('rs_a', A_NET, '(a) terms in all inner layers', '.C1')
  if 'c' in STAGES:
    stage_transition('rs_c', C_NET, '(c) shared smolgen', 'smolgenPrepLayer.weight')
  if 'refuse' in STAGES:
    stage_refuse()
  if 'ddp' in STAGES:
    # the production DDP settings: manual gradient sync, bf16 on the wire, distributed Muon
    stage_transition('rs_a_ddp', A_NET, '(a) terms in all inner layers', '.C1', ranks=2,
                     opt_extra={'DDPManualGradSync': True, 'DDPBF16Compress': True, 'MuonDistributed': True})
  if 'compile' in STAGES:
    # torch.compile 'default' on CPU (needs a C++ compiler for inductor); not part of the default stages
    stage_transition('rs_a_compile', A_NET, '(a) terms in all inner layers, PyTorchCompileMode default', '.C1',
                     opt_extra={'PyTorchCompileMode': 'default'})
    stage_transition('rs_c_compile', C_NET, '(c) shared smolgen, PyTorchCompileMode default', 'smolgenPrepLayer.weight',
                     opt_extra={'PyTorchCompileMode': 'default'})
  print('test_bt6_hybrid_resume: stages', STAGES, 'OK')
