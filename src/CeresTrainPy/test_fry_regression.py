"""Regression of the Fry blocks: with every new key OFF an existing config must build the same net and compute the
same numbers as the code before the blocks. CPU.

  python test_fry_regression.py dump <code dir> <config dir> <config id> <out.npz>
      (run with <code dir> = this tree and = a pristine copy of the base commit; imports come from <code dir>)
  python test_fry_regression.py compare <a.npz> <b.npz>
  python test_fry_regression.py all <base code dir> <config dir> <id> [<id> ...]
      dump with both trees (subprocesses) and compare, for every id

`dump` builds CeresNet from the config (TorchSeed), and stores: a SHA-256 of every parameter and buffer at
initialisation, the eval-mode outputs for fixed records, the training-mode loss and a SHA-256 of every gradient
after one backward. `compare` requires all of it to be bit-identical.
"""
import hashlib
import os
import subprocess
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
sys.dont_write_bytecode = True      # the base tree is read-only for us: no __pycache__ there


def _sha(t):
  return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def dump(code_dir, config_dir, config_id, out_path):
  sys.path.insert(0, code_dir)
  import io
  import numpy as np
  import torch
  torch.set_num_threads(8)
  torch.use_deterministic_algorithms(True, warn_only=True)
  from config import Configuration
  from ceres_net import CeresNet
  import test_bt6_hybrid as H          # records / batch helpers (same file in both trees)
  sink, stdout = io.StringIO(), sys.stdout
  sys.stdout = sink
  try:
    cfg = Configuration(config_dir, config_id)
    torch.manual_seed(int(cfg.Opt_TorchSeed) if getattr(cfg, 'Opt_TorchSeed', None) is not None else 777)
    net = CeresNet(None, cfg,
                   policy_loss_weight=cfg.Opt_LossPolicyMultiplier, value_loss_weight=cfg.Opt_LossValueMultiplier,
                   moves_left_loss_weight=cfg.Opt_LossMLHMultiplier, unc_loss_weight=cfg.Opt_LossUNCMultiplier,
                   value2_loss_weight=cfg.Opt_LossValue2Multiplier, q_deviation_loss_weight=cfg.Opt_LossQDeviationMultiplier,
                   value_diff_loss_weight=cfg.Opt_LossValueDMultiplier, value2_diff_loss_weight=cfg.Opt_LossValue2DMultiplier,
                   action_loss_weight=cfg.Opt_LossActionMultiplier, uncertainty_policy_weight=cfg.Opt_LossUncertaintyPolicyMultiplier,
                   action_uncertainty_loss_weight=cfg.Opt_LossActionUncertaintyMultiplier, q_ratio=cfg.Data_FractionQ)
  finally:
    sys.stdout = stdout
  res = {}
  names = []
  for n, p in list(net.named_parameters()) + list(net.named_buffers()):
    names.append(f'{n}|{tuple(p.shape)}|{_sha(p)}')
  res['state'] = np.array(names)
  # zero-initialised tables would hide a path: give every exactly-zero parameter a deterministic non-zero value
  g = torch.Generator().manual_seed(1234)
  with torch.no_grad():
    for n, p in net.named_parameters():
      if float(p.abs().max()) == 0.0:
        p.copy_(torch.randn(p.shape, generator=g) * 0.05)
  sq = H.records(6)
  net.eval()
  with torch.no_grad():
    out = net(sq, None)
  for i, o in enumerate(out):
    if torch.is_tensor(o):
      res[f'eval_out_{i}'] = o.float().numpy()
  net.train()
  loss = H.run_loss(net, H.batch_for(6), sq)
  res['loss'] = np.array([float(loss)], dtype=np.float64)
  loss.backward()
  res['grads'] = np.array([f'{n}|{"none" if p.grad is None else _sha(p.grad)}' for n, p in net.named_parameters()])
  # optimiser classes: the weight-decay partition
  from wd_partition import partition_weight_decay
  decay, no_decay = partition_weight_decay(net)
  res['decay'] = np.array(sorted(decay))
  res['no_decay'] = np.array(sorted(no_decay))
  np.savez(out_path, **res)
  print(f'dumped {config_id} from {code_dir}: {len(names)} tensors, {sum(p.numel() for p in net.parameters()):,} parameters, loss {float(loss):.6f}')


def compare(a_path, b_path):
  import numpy as np
  a, b = np.load(a_path), np.load(b_path)
  assert sorted(a.files) == sorted(b.files), (sorted(a.files), sorted(b.files))
  bad = []
  for k in a.files:
    x, y = a[k], b[k]
    if x.shape != y.shape or not (x == y).all():
      if x.dtype.kind in 'US':
        diff = sorted(set(x.tolist()) ^ set(y.tolist()))[:6]
        bad.append((k, f'{len(set(x.tolist()) ^ set(y.tolist()))} entries differ, e.g. {diff}'))
      else:
        bad.append((k, f'max |diff| {float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if x.shape == y.shape else "shape"}'))
  n_state, n_grad = len(a['state']), len(a['grads'])
  n_out = sum(1 for k in a.files if k.startswith('eval_out_'))
  if bad:
    print('DIFFERENT:', bad)
    return False
  print(f'   BIT-IDENTICAL: {n_state} parameters / buffers at initialisation, {n_out} eval outputs, the training loss '
        f'({float(a["loss"][0]):.6f}), {n_grad} gradients, the weight-decay partition ({len(a["decay"])} / {len(a["no_decay"])})')
  return True


def run_all(base_dir, config_dir, ids):
  import tempfile
  here = os.path.dirname(os.path.abspath(__file__))
  ok = True
  with tempfile.TemporaryDirectory() as d:
    for cid in ids:
      paths = []
      for tag, code in (('new', here), ('base', base_dir)):
        out = os.path.join(d, f'{cid}_{tag}.npz')
        r = subprocess.run([sys.executable, os.path.abspath(__file__), 'dump', code, config_dir, cid, out],
                           capture_output=True, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        tail = [l for l in r.stdout.splitlines() if l.startswith('dumped')]
        if r.returncode != 0 or not tail:
          print(r.stdout[-2000:], r.stderr[-4000:])
          raise SystemExit(f'dump failed for {cid} ({tag})')
        print('  ', tail[-1])
        paths.append(out)
      ok = compare(*paths) and ok
  print('REGRESSION OK' if ok else 'REGRESSION FAILED')
  return ok


if __name__ == '__main__':
  if sys.argv[1] == 'dump':
    dump(*sys.argv[2:6])
  elif sys.argv[1] == 'compare':
    sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
  elif sys.argv[1] == 'all':
    sys.exit(0 if run_all(sys.argv[2], sys.argv[3], sys.argv[4:]) else 1)
