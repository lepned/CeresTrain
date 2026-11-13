"""Checks of a trained BT6-hybrid net's files, on CPU (no TensorRT):

    python bt6_hybrid_export_check.py <outputs dir> <training id> <checkpoint file> [<trainer .onnx>] [<carrier out>]

1. checkpoint reload: the net rebuilt from the config + the trainer's checkpoint file; two rebuilds give identical
   outputs, and the TorchScript file the trainer wrote (if next to the checkpoint) matches;
2. ONNX: (a) a float32 export made here with the trainer's exporter settings (dynamo, opset from CERES_ONNX_OPSET /
   18, single `squares` input, the trainer's output names) run in onnxruntime against PyTorch; (b) the trainer's own
   .onnx (FP16-converted, with its FP32 islands) run in onnxruntime against PyTorch; (c) the op-type census of the
   graph and a list of the ops that are unusual for our nets or data-dependent;
3. serving carrier: export_served_hybrid -> forward_served_hybrid (kernel form) against PyTorch, and the carrier
   written to a .pb.gz file and read back.
"""

import collections
import io
import os
import sys

os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HEAD_OUTPUT_NAMES = ['policy', 'value', 'mlh', 'unc', 'value2', 'q_deviation_lower', 'q_deviation_upper',
                     'uncertainty_policy', 'action', 'prior_state_out', 'action_uncertainty']
# Op types our production exports (plain attention + smolgen + MLP heads) consist of; anything else is listed.
USUAL_OPS = {'MatMul', 'Gemm', 'Add', 'Sub', 'Mul', 'Div', 'Reshape', 'Transpose', 'Concat', 'Slice', 'Softmax', 'Sqrt',
             'Pow', 'ReduceMean', 'Reciprocal', 'Sigmoid', 'Tanh', 'Softplus', 'Mish', 'Relu', 'Cast', 'Constant',
             'Unsqueeze', 'Squeeze', 'Shape', 'Gather', 'Expand', 'Identity', 'Flatten', 'LayerNormalization', 'Erf',
             'ConstantOfShape', 'Einsum', 'Neg', 'Exp', 'Log', 'Split'}
DATA_DEPENDENT_OPS = {'NonZero', 'Loop', 'If', 'Scan', 'Compress', 'Unique', 'TopK', 'NonMaxSuppression', 'Where_dynamic'}


def build_net(outputs_dir, training_id):
  from config import Configuration
  from ceres_net import CeresNet
  cfg = Configuration('.', os.path.join(outputs_dir, 'configs', training_id))
  torch.manual_seed(1)
  stdout, sys.stdout = sys.stdout, io.StringIO()
  try:
    net = CeresNet(None, cfg,
                   policy_loss_weight=cfg.Opt_LossPolicyMultiplier, value_loss_weight=cfg.Opt_LossValueMultiplier,
                   moves_left_loss_weight=cfg.Opt_LossMLHMultiplier, unc_loss_weight=cfg.Opt_LossUNCMultiplier,
                   value2_loss_weight=cfg.Opt_LossValue2Multiplier, q_deviation_loss_weight=cfg.Opt_LossQDeviationMultiplier,
                   value_diff_loss_weight=cfg.Opt_LossValueDMultiplier, value2_diff_loss_weight=cfg.Opt_LossValue2DMultiplier,
                   action_loss_weight=cfg.Opt_LossActionMultiplier, uncertainty_policy_weight=cfg.Opt_LossUncertaintyPolicyMultiplier,
                   action_uncertainty_loss_weight=cfg.Opt_LossActionUncertaintyMultiplier, q_ratio=cfg.Data_FractionQ)
  finally:
    sys.stdout = stdout
  return net.eval(), cfg


def load_checkpoint(net, path):
  state = torch.load(path, map_location='cpu', weights_only=False)['model']
  state = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state.items()}
  missing, unexpected = net.load_state_dict(state, strict=False)
  return missing, unexpected


def worst(a, b):
  return max(float((x.double() - y.double()).abs().max() / max(1.0, float(y.abs().max()))) for x, y in zip(a, b))


def main():
  outputs_dir, training_id, checkpoint = sys.argv[1], sys.argv[2], sys.argv[3]
  trainer_onnx = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else None
  carrier_out = sys.argv[5] if len(sys.argv) > 5 else None
  from test_bt6_hybrid import records
  sq = records(8)

  # ---- 1. checkpoint reload
  net, cfg = build_net(outputs_dir, training_id)
  missing, unexpected = load_checkpoint(net, checkpoint)
  twin, _ = build_net(outputs_dir, training_id)
  load_checkpoint(twin, checkpoint)
  with torch.no_grad():
    out, out2 = net(sq, None), twin(sq, None)
  assert all(torch.equal(a, b) for a, b in zip(out, out2))
  print(f'1. checkpoint {os.path.basename(checkpoint)}: loaded into the rebuilt net (missing keys {list(missing)}, unexpected '
        f'{list(unexpected)}); two reloads give identical outputs on {len(sq)} real positions; policy {tuple(out[0].shape)}, '
        f'value {tuple(out[1].shape)}, value2 {tuple(out[4].shape)}')
  ts_path = checkpoint.replace('ckpt_', '') + '.ts'
  if os.path.exists(ts_path):
    try:
      ts = torch.jit.load(ts_path, map_location='cpu')
      with torch.no_grad():
        ts_out = ts(sq.to(next(ts.parameters()).dtype), torch.zeros(len(sq), 64, 0))
      print(f'   TorchScript {os.path.basename(ts_path)}: max relative diff vs the reloaded net '
            f'{worst([t.float() for t in ts_out[:5]], out[:5]):.2e} (first 5 outputs)')
    except Exception as error:  # noqa: BLE001 - report, do not hide
      print(f'   TorchScript {os.path.basename(ts_path)}: could not be run here ({type(error).__name__}: {str(error)[:200]})')

  # ---- 2. ONNX
  import onnx
  import onnxruntime as ort

  class SquaresOnly(torch.nn.Module):
    def __init__(self, inner):
      super().__init__()
      self.inner = inner

    def forward(self, squares):
      return self.inner(squares, torch.zeros(squares.shape[0], 64, 0, dtype=squares.dtype))

  opset = int(os.environ.get('CERES_ONNX_OPSET', '18'))
  fp32_path = os.path.join(os.path.dirname(checkpoint), 'bt6h_export_check_fp32.onnx')
  torch.onnx.export(SquaresOnly(net), (sq[:2],), fp32_path, do_constant_folding=True, export_params=True,
                    opset_version=opset, input_names=['squares'], output_names=HEAD_OUTPUT_NAMES,
                    dynamic_axes={'squares': {0: 'batch_size'}, **{n: {0: 'batch_size'} for n in HEAD_OUTPUT_NAMES}})
  session = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
  got = session.run(None, {'squares': sq.numpy()})
  names = [o.name for o in session.get_outputs()]
  diff = worst([torch.from_numpy(g.astype(np.float32)) for g in got[:5]], out[:5])
  print(f'2a. float32 ONNX (dynamo exporter, opset {opset}, batch {len(sq)} != the export batch 2): onnxruntime vs PyTorch '
        f'max relative diff {diff:.2e} over {names[:5]}; outputs {[(n, tuple(g.shape)) for n, g in zip(names, got)]}')
  model = onnx.load(fp32_path)
  census = collections.Counter(node.op_type for node in model.graph.node)
  print(f'2c. op census of the float32 graph ({sum(census.values())} nodes, opset {model.opset_import[0].version}): '
        + ', '.join(f'{k} {v}' for k, v in sorted(census.items(), key=lambda kv: -kv[1])))
  unusual = {k: v for k, v in census.items() if k not in USUAL_OPS}
  print('    op types not in our usual set: ' + (', '.join(f'{k} x{v}' for k, v in sorted(unusual.items())) or 'none'))
  dependent = {k: v for k, v in census.items() if k in DATA_DEPENDENT_OPS}
  print('    data-dependent-shape ops (NonZero / Loop / If / TopK / Unique / Compress): '
        + (', '.join(f'{k} x{v}' for k, v in dependent.items()) or 'none'))
  integer = collections.Counter()
  for node in model.graph.node:
    if node.op_type in ('ScatterElements', 'ScatterND', 'GatherElements', 'GatherND', 'CumSum', 'And', 'Or', 'Not', 'Xor',
                        'Greater', 'Less', 'GreaterOrEqual', 'LessOrEqual', 'Equal', 'Where', 'Pad', 'ReduceMax', 'ReduceSum',
                        'BitShift', 'Mod'):
      integer[node.op_type] += 1
  print('    integer / boolean / indexing ops (the relation channels and the 1858 gather): '
        + ', '.join(f'{k} x{v}' for k, v in sorted(integer.items())))
  if trainer_onnx:
    session16 = ort.InferenceSession(trainer_onnx, providers=['CPUExecutionProvider'])
    info = session16.get_inputs()[0]
    feed = sq.numpy().astype(np.float16 if 'float16' in info.type else np.float32)
    got16 = session16.run(None, {info.name: feed})
    diff16 = [float(np.abs(g.astype(np.float64) - o.numpy().astype(np.float64)).max()) for g, o in zip(got16[:5], out[:5])]
    finite = all(np.isfinite(g).all() for g in got16)
    top1 = float((got16[0].argmax(1) == out[0].numpy().argmax(1)).mean())
    census16 = collections.Counter(node.op_type for node in onnx.load(trainer_onnx).graph.node)
    print(f'2b. trainer ONNX {os.path.basename(trainer_onnx)} (input {info.type}, FP16-converted with the trainer\'s FP32 '
          f'islands; {sum(census16.values())} nodes, Cast x{census16.get("Cast", 0)}): onnxruntime runs, outputs finite {finite}; '
          f'max abs diff vs float32 PyTorch policy {diff16[0]:.4f}, value {diff16[1]:.4f}, mlh {diff16[2]:.4f}, unc {diff16[3]:.4f}, '
          f'value2 {diff16[4]:.4f}; policy top-1 agreement {100 * top1:.0f} % ({len(sq)} positions)')

  # ---- 3. serving carrier
  from bt6_carrier import read_carrier, write_carrier
  from bt6_hybrid import export_served_hybrid, forward_served_hybrid, served_meta
  from bt6_net import served_to_torch
  exported = export_served_hybrid(net)
  ref = forward_served_hybrid(served_to_torch(exported), sq, served_meta(net))
  pairs = [('policy', out[0]), ('value', out[1]), ('unc', out[3])] + ([('value2', out[4])] if 'value2' in ref else []) \
      + ([('mlh', out[2])] if 'mlh' in ref else [])
  diffs = {name: float((tensor.double() - ref[name].double()).abs().max()) for name, tensor in pairs}
  print(f'3. serving carrier: {len(exported)} tensors ({sum(a.size for a in exported.values()):,} elements); kernel-form '
        f'forward on the exported (FP16 dense / FP32 table) tensors vs the float32 net: max abs diff '
        + ', '.join(f'{k} {v:.4f}' for k, v in diffs.items()))
  if carrier_out:
    write_carrier(carrier_out, exported)
    back, _ = read_carrier(carrier_out)
    assert list(back) == list(exported) and all(np.array_equal(back[k], exported[k]) for k in exported)
    print(f'   written and read back: {carrier_out} ({os.path.getsize(carrier_out) / 2**20:.1f} MB)')


if __name__ == '__main__':
  main()
