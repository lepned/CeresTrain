"""Python-only TPG-record policy evaluator.

Loads a trained CeresTrain model checkpoint, iterates records from a TPG shard,
runs the model forward (computing aug features in-line when enabled), and reports
top-1/top-3 policy accuracy against the solver-move target embedded in the TPG
record. Used to compare model variants apples-to-apples on the same TPG-encoded
data, without needing the full EngineBattle→Ceres→TRT inference pipeline.

Usage:
  python tpg_eval.py <ckpt_path> <config_dir> <tag> <tpg_shard.zst> [--n 5000] [--aug 0|3]

The --aug flag controls whether the eval computes augmented features per-record
before forward pass. Must match how the model was trained (use --aug 3 for
augfeat-trained nets, --aug 0 for baseline).
"""
import argparse, os, sys, time
import numpy as np
import torch
import zstandard

# Make CeresTrainPy importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src', 'CeresTrainPy'))

BYTES_PER_POS = 9378
SIZE_SQUARE = 137
SQUARES_OFFSET = BYTES_PER_POS - 64 * SIZE_SQUARE  # 610
MAX_MOVES = 92


def parse_records(shard_path: str, n_records: int):
  """Yield decoded (squares, policy_target_dense) for up to n_records."""
  dctx = zstandard.ZstdDecompressor()
  with open(shard_path, 'rb') as f:
    data = dctx.stream_reader(f).read()
  total = len(data) // BYTES_PER_POS
  n = min(n_records, total)
  print(f'[tpg_eval] shard has {total:,} records; using {n:,}')

  arr = np.frombuffer(data[:n * BYTES_PER_POS], dtype=np.uint8).reshape(n, BYTES_PER_POS)

  # Offsets within record — re-derived from tpg_dataset.py:200-283 running sum:
  # 12 wdl_nondeblundered + 12 wdl_deblundered + 12 wdl_q + 4 played_q_suboptim
  # + 4+42 (skipped) + 4+2+2+2 (skipped reference-model)
  # + 4 KLDPolicy + 4 mlh + 4 uncertainty + 2 q_dev_lower + 2 q_dev_upper
  # + 2 policy_index_in_parent + 64+64 (skipped PlyBin arrays)
  # = 242 → policies_indices at 242, policies_values at 426
  # squares start at 610 (verified: SQUARES_OFFSET = BYTES_PER_POS - 64*137 = 610)
  POLICIES_INDICES_OFFSET = 242
  POLICIES_VALUES_OFFSET = 242 + MAX_MOVES * 2

  squares_bytes = arr[:, SQUARES_OFFSET:SQUARES_OFFSET + 64 * SIZE_SQUARE]
  squares = squares_bytes.reshape(n, 64, SIZE_SQUARE).astype(np.float32) / 100.0  # (n, 64, 137)

  pol_idx = np.ascontiguousarray(arr[:, POLICIES_INDICES_OFFSET:POLICIES_INDICES_OFFSET + MAX_MOVES * 2]) \
    .view(dtype=np.int16).reshape(n, MAX_MOVES).astype(np.int64)
  pol_val = np.ascontiguousarray(arr[:, POLICIES_VALUES_OFFSET:POLICIES_VALUES_OFFSET + MAX_MOVES * 2]) \
    .view(dtype=np.float16).reshape(n, MAX_MOVES).astype(np.float32)

  # Simpler approach than building dense target: identify each record's solver
  # move as the index with the highest probability in its sparse representation.
  # Then top-1 = model_argmax == solver_idx; top-3 = solver_idx ∈ model_top3.
  # This bypasses dense-target construction (some aug records have sums >> 1
  # which is benign per-record but breaks dense aggregation).
  # Filter out records with no meaningful policy mass (value-only entries).
  pol_val_max = pol_val.max(axis=1)
  has_policy = pol_val_max > 0.01    # require at least 1% mass on the top move
  solver_argpos = pol_val.argmax(axis=1)                       # (n,)
  solver_idx = pol_idx[np.arange(n), solver_argpos]            # (n,) — move-index 0..1857
  # Sanity: solver_idx in valid range
  valid_range = (solver_idx >= 0) & (solver_idx < 1858)
  keep = has_policy & valid_range
  print(f'[tpg_eval] {keep.sum():,}/{n:,} records have a solver-identifiable policy target')

  return squares[keep], solver_idx[keep].astype(np.int64)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('ckpt')
  ap.add_argument('config_dir')
  ap.add_argument('tag')
  ap.add_argument('shard')
  ap.add_argument('--n', type=int, default=5000)
  ap.add_argument('--aug', type=int, default=0, help='0 or 3 — must match training setup')
  ap.add_argument('--batch', type=int, default=512)
  args = ap.parse_args()

  # Set env var BEFORE importing config so TOTAL_INPUT_FEATURES_PER_SQUARE is correct
  os.environ['CERES_AUG_FEATURES_PER_SQUARE'] = str(args.aug)

  from config import Configuration, TOTAL_INPUT_FEATURES_PER_SQUARE
  print(f'[tpg_eval] TOTAL_INPUT_FEATURES_PER_SQUARE = {TOTAL_INPUT_FEATURES_PER_SQUARE}')
  from ceres_net import CeresNet

  cfg = Configuration(args.config_dir, args.tag)

  print(f'[tpg_eval] building model...')
  model = CeresNet(writer=None, config=cfg,
                   policy_loss_weight=1.0, value_loss_weight=1.0,
                   moves_left_loss_weight=cfg.Opt_LossMLHMultiplier,
                   unc_loss_weight=cfg.Opt_LossUNCMultiplier,
                   value2_loss_weight=cfg.Opt_LossValue2Multiplier,
                   q_deviation_loss_weight=cfg.Opt_LossQDeviationMultiplier,
                   value_diff_loss_weight=cfg.Opt_LossValueDMultiplier,
                   value2_diff_loss_weight=cfg.Opt_LossValue2DMultiplier,
                   action_loss_weight=cfg.Opt_LossActionMultiplier,
                   uncertainty_policy_weight=cfg.Opt_LossUncertaintyPolicyMultiplier,
                   action_uncertainty_loss_weight=cfg.Opt_LossActionUncertaintyMultiplier,
                   q_ratio=0.0)

  print(f'[tpg_eval] loading checkpoint from {args.ckpt}...')
  ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
  # strip the "_orig_mod." prefix that torch.compile adds
  sd = ckpt['model']
  sd_clean = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
  missing, unexpected = model.load_state_dict(sd_clean, strict=False)
  if missing:
    print(f'  WARN missing keys: {len(missing)} (first 3: {missing[:3]})')
  if unexpected:
    print(f'  WARN unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})')

  model = model.cuda().to(torch.bfloat16).eval()

  print(f'[tpg_eval] loading TPG data: {args.shard}')
  squares, solver_target = parse_records(args.shard, args.n)
  n = squares.shape[0]
  print(f'[tpg_eval] evaluating {n:,} records (batch={args.batch})...')

  # if aug=3, prep feature computer
  if args.aug > 0:
    from aug_features import compute_aug_features_batch

  top1 = 0
  top3 = 0
  solver_logprob_sum = 0.0   # avg log-prob the model assigns to the solver move (lower KLD proxy)

  t0 = time.perf_counter()
  with torch.no_grad():
    for off in range(0, n, args.batch):
      sq_batch = squares[off:off + args.batch]              # (B, 64, 137)
      solver_batch = solver_target[off:off + args.batch]    # (B,) — solver move-index

      if args.aug > 0:
        aug = compute_aug_features_batch(sq_batch)
        sq_batch_full = np.concatenate([sq_batch, aug], axis=-1)
      else:
        sq_batch_full = sq_batch

      x = torch.from_numpy(sq_batch_full).to(torch.bfloat16).cuda()
      prior_dim = max(cfg.NetDef_PriorStateDim, 4)
      prior = torch.zeros(x.shape[0], 64, prior_dim, dtype=torch.bfloat16, device='cuda')

      out = model(x, prior)
      pol_logits = out[0].float().cpu().numpy()              # (B, 1858)

      pred_top1 = pol_logits.argmax(axis=1)
      top1 += (pred_top1 == solver_batch).sum()

      pred_top3 = np.argpartition(pol_logits, -3, axis=1)[:, -3:]
      top3 += sum(solver_batch[i] in pred_top3[i] for i in range(len(solver_batch)))

      pol_prob = torch.softmax(out[0].float(), dim=1).cpu().numpy()  # (B, 1858)
      # log-prob the model assigns to the solver move (higher = better calibrated)
      solver_logprob_sum += float(np.log(np.clip(pol_prob[np.arange(len(solver_batch)), solver_batch], 1e-12, 1.0)).sum())

  dt = time.perf_counter() - t0
  print(f'[tpg_eval] done in {dt:.1f}s ({n/dt:.0f} pos/s)')
  print()
  print(f'  shard:    {os.path.basename(args.shard)}')
  print(f'  records:  {n:,}')
  print(f'  top-1:    {top1/n*100:.2f}%  ({top1}/{n})')
  print(f'  top-3:    {top3/n*100:.2f}%  ({top3}/{n})')
  print(f'  avg log-prob on solver: {solver_logprob_sum/n:.4f}  (higher = better; proxy for KLD)')


if __name__ == '__main__':
  main()
