"""Training smoke of the Fry blocks through the REAL train.py on CPU: a very small net with every new key on, a few
optimizer steps on real lc0 v6 chunks, then the trainer's own ONNX export, read back in onnxruntime.

  python test_fry_smoke.py <dir with loose lc0 v6 .gz chunks> [work dir]

What it shows: the config validates, the net builds, the loss computes on real batches, the optimizer partition is
the one asked for (the trainer's own log lines: Muon scope 'fry', WeightDecayScheme 'fry', the weight decay from
the schedule rule), the loss is finite at every logged step, a checkpoint and an ONNX file are written, and the ONNX
runs in onnxruntime (CPU) and agrees with the PyTorch net loaded from the checkpoint. It also lists the ONNX
operators and integer tensors a TensorRT build has to accept (nothing is built with TensorRT here).

CPU only: bf16 autocast, torch.compile, DDP and the GPU speed are NOT exercised.
"""
import collections
import json
import os
import re
import subprocess
import sys
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')

HERE = os.path.dirname(os.path.abspath(__file__))
ID = 'dev_64_4_fryall_t80_smoke'

NET = {
  'ModelDim': 64, 'NumLayers': 4, 'NumHeads': 4, 'PreNorm': True, 'NormType': 'RMSNorm', 'NormEps': 1e-5,
  'UseQKV': True, 'UseQKNorm': True, 'AttentionMultiplier': 1,
  'FFNMultiplier': 1.25, 'FFNActivationType': 'SwiGLU', 'FFNNoBias': True,
  'HeadsActivationType': 'Mish', 'HeadWidthMultiplier': 4,
  'SmolgenDimPerSquare': 0, 'SmolgenDim': 0, 'SmolgenToHeadDivisor': 4, 'SmolgenActivationType': 'Swish',
  'UseRPE': False, 'UseRPE_V': False, 'UseRoPE': False, 'NonLinearAttention': False, 'SoftCapCutoff': 0,
  'NBTInnerLayers': 0, 'InputHistoryKeep': 0,
  'AttnTemplates': 4, 'AttnTemplateScope': 'layer', 'NBTInputSkip': True, 'InputSkipLayer0': True,
  'FryInput': True, 'SquareFeatures': True,
  'AttnKVHeads': 2, 'AttnQKDims': [4, 8, 12, 12], 'AttnValueDim': 16, 'AttnQKVBias': True,
  'PolicyHeadFry': True, 'ValueHeadFry': True, 'ValueHLGaussServe': True,
}
OPT = {
  'NumTrainingPositions': 16384, 'BatchSizeForwardPass': 256, 'BatchSizeBackwardPass': 256,
  'Optimizer': 'Muon', 'MuonAdamWScope': 'fry', 'MuonMomentum': 0.95, 'Beta1': 0.9, 'Beta2': 0.999,
  'WeightDecayScheme': 'fry', 'WeightDecay': 0.01,     # (the schedule rule needs a real step count: unit-tested in test_fry_blocks.py)
  'LearningRateBase': 1e-3, 'LRBeginDecayAtFractionComplete': 1.0, 'LRWarmupPhaseMultiplier': 0.03, 'GradientClipLevel': 1.0,
  'CheckpointFrequencyNumPositions': 16384, 'CheckpointResumeFromFileName': None, 'PyTorchCompileMode': None,
  'LossPolicyMultiplier': 1, 'LossValueMultiplier': 0.1, 'LossValue2Multiplier': 0, 'LossMLHMultiplier': 0,
  'LossUNCMultiplier': 0, 'LossQDeviationMultiplier': 0, 'LossUncertaintyPolicyMultiplier': 0,
  'LossValueDMultiplier': 0, 'LossValue2DMultiplier': 0, 'LossActionMultiplier': 0, 'LossActionUncertaintyMultiplier': 0,
  'HLGaussWeight': 0.5, 'HLGaussBuckets': 128, 'EMAPeriodSteps': -1,
  'TorchSeed': 777, 'ShuffleSeed': 20261126, 'TPGV3': 0, 'AuxFeaturesPerSquare': 0, 'SquareBytes': 137,
}
EXEC = {'ID': ID, 'DeviceType': 'cpu', 'DeviceIDs': [0], 'DataType': 'BFloat16', 'UseFP8': False, 'UseHistory': True,
        'DropoutRate': 0, 'DropoutDuringInference': False, 'EngineType': 'CSharpViaTorchscript', 'ExportOnly': False,
        'TestFlag': False, 'TestValue': 0}
MON = {'CustomMonitoringIntervalSeconds': 600}


def write_configs(work, data_dir):
  cdir = os.path.join(work, 'configs')
  os.makedirs(cdir, exist_ok=True)
  data = {'SourceType': 'DirectFromV6', 'TrainingFilesDirectory': data_dir, 'FractionQ': 0.8, 'WDLLabelSmoothing': 0,
          'V6SkipCount': 4}
  for suf, obj in (('net', NET), ('opt', OPT), ('data', data), ('exec', EXEC), ('monitoring', MON)):
    with open(os.path.join(cdir, f'{ID}_ceres_{suf}.json'), 'w') as f:
      json.dump(obj, f, indent=2)


def main():
  data_dir = sys.argv[1]
  work = sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix='fry_smoke_')
  write_configs(work, data_dir)
  env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
  log_path = os.path.join(work, 'train_stdout.log')
  with open(log_path, 'w') as log:
    r = subprocess.run([sys.executable, 'train.py', ID, work], cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
  log = open(log_path, errors='replace').read()
  keep = [l for l in log.splitlines() if re.search(r"Muon partition|WeightDecayScheme|WeightDecayFromSchedule|wd partition|FRY |SQUARE FEATURES|"
                                                   r"GROUPED-QUERY|INPUT SKIP|ATTENTION TEMPLATES|NUM_PARAMETERS|HL-GAUSS", l)]
  print('\n'.join('   ' + l[:330] for l in keep))
  if r.returncode != 0:
    print(log[-6000:])
    raise SystemExit(f'train.py exited with {r.returncode} (log: {log_path})')
  train_lines = [l for l in log.splitlines() if l.startswith('TRAIN:')]
  print(f'   {len(train_lines)} TRAIN lines; first / last:')
  for l in train_lines[:1] + train_lines[-1:]:
    print('   ' + l[:330])
  assert train_lines, 'no TRAIN line in the log'
  first, last = (float(l.split(',')[1]) for l in (train_lines[0], train_lines[-1]))
  assert last < first, f'the loss did not fall: {first} -> {last}'
  print(f'   total loss {first:.4f} -> {last:.4f} over {len(train_lines)} logged steps')
  assert 'nan' not in ' '.join(train_lines).lower(), 'NaN in the training log'
  assert "Muon partition scope=fry" in log and "WeightDecayScheme 'fry'" in log
  nets = sorted(os.listdir(os.path.join(work, 'nets')))
  print('   files written:', nets)
  onnx_files = [n for n in nets if n.endswith('.onnx') and not n.endswith('ema.onnx')]
  assert onnx_files, 'no ONNX file was exported'
  check_onnx(os.path.join(work, 'nets', onnx_files[-1]), work, nets)
  print(f'SMOKE OK (work dir {work})')


def check_onnx(path, work, nets):
  import numpy as np
  import onnx
  import onnxruntime as ort
  model = onnx.load(path)
  ops = collections.Counter(n.op_type for n in model.graph.node)
  ins = [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim], onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type))
         for i in model.graph.input]
  outs = [(o.name, onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type)) for o in model.graph.output]
  print(f'   ONNX {os.path.basename(path)}: opset {model.opset_import[0].version}, {len(model.graph.node)} nodes; inputs {ins}')
  print(f'   outputs {outs}')
  print(f'   operators: {dict(sorted(ops.items()))}')
  # what TensorRT may dislike: integer tensors, index operators, casts to integers
  int_inits = [(i.name, onnx.TensorProto.DataType.Name(i.data_type), list(i.dims)) for i in model.graph.initializer
               if i.data_type in (onnx.TensorProto.INT64, onnx.TensorProto.INT32) and int(np.prod(i.dims) if len(i.dims) else 1) > 8]
  casts = collections.Counter(onnx.TensorProto.DataType.Name(a.i) for n in model.graph.node if n.op_type == 'Cast'
                              for a in n.attribute if a.name == 'to')
  watch = ('ArgMax', 'ArgMin', 'TopK', 'Gather', 'GatherElements', 'GatherND', 'ScatterND', 'ScatterElements', 'CumSum', 'Where',
           'Equal', 'Greater', 'Less', 'And', 'Or', 'Not', 'Mod', 'ReduceMin', 'ReduceMax', 'Round', 'Einsum', 'Expand', 'Tile', 'Pad',
           'Loop', 'If', 'NonZero', 'Erf')
  print(f'   for a TensorRT build -- operators to watch: { {k: ops[k] for k in watch if ops.get(k)} }')
  print(f'   casts by target type: {dict(casts)}; integer initializers with more than 8 elements: {len(int_inits)} '
        f'(e.g. {int_inits[:6]})')
  so = ort.SessionOptions()
  so.log_severity_level = 3
  sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
  # the same positions through onnxruntime and through the PyTorch net rebuilt from the checkpoint
  sys.path.insert(0, HERE)
  import chess
  import torch
  from fry_testlib import TEST_FENS, record_from_board, legal_move_indices
  rec = np.stack([record_from_board(chess.Board(f)) for f in TEST_FENS]).astype(np.float32)
  feed_name = sess.get_inputs()[0].name
  in_type = sess.get_inputs()[0].type
  feed = rec.astype(np.float16) if 'float16' in in_type else rec
  if len(sess.get_inputs()[0].shape) == 3 and sess.get_inputs()[0].shape[-1] not in (137, None, 'None') and isinstance(sess.get_inputs()[0].shape[-1], int):
    assert sess.get_inputs()[0].shape[-1] == 137, sess.get_inputs()[0].shape
  got = sess.run(None, {feed_name: feed})
  names = [o.name for o in sess.get_outputs()]
  print(f'   onnxruntime: ran {len(TEST_FENS)} positions; outputs {[(n, tuple(g.shape)) for n, g in zip(names, got)]}')
  pol = np.asarray(got[names.index('policy')], dtype=np.float64)
  val = np.asarray(got[names.index('value')], dtype=np.float64)
  assert np.isfinite(pol).all() and np.isfinite(val).all()
  wdl = np.exp(val) / np.exp(val).sum(-1, keepdims=True)
  print(f'   served WDL of the start position: {np.round(wdl[0], 4).tolist()} (sums to {wdl[0].sum():.4f})')
  ckpts = [n for n in nets if n.startswith('ckpt_') and '.' not in n]
  if ckpts:
    from config import Configuration
    from ceres_net import CeresNet
    import io
    sink, stdout = io.StringIO(), sys.stdout
    sys.stdout = sink
    try:
      cfg = Configuration('.', os.path.join(work, 'configs', ID))
      net = CeresNet(None, cfg,
                     policy_loss_weight=cfg.Opt_LossPolicyMultiplier, value_loss_weight=cfg.Opt_LossValueMultiplier,
                     moves_left_loss_weight=cfg.Opt_LossMLHMultiplier, unc_loss_weight=cfg.Opt_LossUNCMultiplier,
                     value2_loss_weight=cfg.Opt_LossValue2Multiplier, q_deviation_loss_weight=cfg.Opt_LossQDeviationMultiplier,
                     value_diff_loss_weight=cfg.Opt_LossValueDMultiplier, value2_diff_loss_weight=cfg.Opt_LossValue2DMultiplier,
                     action_loss_weight=cfg.Opt_LossActionMultiplier, uncertainty_policy_weight=cfg.Opt_LossUncertaintyPolicyMultiplier,
                     action_uncertainty_loss_weight=cfg.Opt_LossActionUncertaintyMultiplier, q_ratio=cfg.Data_FractionQ)
    finally:
      sys.stdout = stdout
    ck = torch.load(os.path.join(work, 'nets', ckpts[-1]), map_location='cpu', weights_only=False)
    sd = ck.get('model', ck.get('model_state_dict', ck)) if isinstance(ck, dict) else ck
    sd = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True)
    net.eval()
    with torch.no_grad():
      out = net(torch.from_numpy(rec), None)
    dp = 0.0
    for b, f in enumerate(TEST_FENS):
      idx = [o for _, o, _ in legal_move_indices(chess.Board(f))]
      dp = max(dp, float(np.abs(pol[b, idx] - out[0][b, idx].double().numpy()).max()))
    dv = float(np.abs(val - out[1].double().numpy()).max())
    print(f'   ONNX (onnxruntime) vs the PyTorch net from {ckpts[-1]}: max |policy logit diff| on legal moves {dp:.2e}, '
          f'max |value (log-WDL) diff| {dv:.2e}')
    assert dp < 5e-2 and dv < 5e-2, (dp, dv)


if __name__ == '__main__':
  main()
