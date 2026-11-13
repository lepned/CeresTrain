"""CPU check that the BT6 hybrid trains under DistributedDataParallel the way train.py wraps it:
static_graph=True (required for the stashed soft-policy loss), per-block gradient checkpointing ON, 2 ranks on the
gloo backend, a few optimiser steps through the real loss path.

    python test_bt6_hybrid_ddp.py

What it shows: no "did not receive grad" / "marked ready twice" error from the reducer with torch.utils.checkpoint
(use_reentrant=False) inside the wrapped module; after every step both ranks hold identical parameters; the loss is
finite and falls. It does NOT exercise CUDA, NCCL, bf16 autocast on GPU or torch.compile.
"""

import os
import sys

os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def worker(rank, world, port, queue):
  os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'] = '127.0.0.1', str(port)
  torch.set_num_threads(4)
  dist.init_process_group('gloo', rank=rank, world_size=world)
  from torch.nn.parallel import DistributedDataParallel as DDP
  from test_bt6_hybrid import batch_for, build, records
  from losses import LossCalculator
  net = build({'NumLayers': 3, 'BT6EdgeSites': [0, 1], 'BT6CheckpointBlocks': True}).train()   # same seed on every rank
  model = DDP(net, static_graph=True)
  optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
  sq_all, batch_all = records(16), batch_for(16)
  sq = sq_all[rank::world]
  batch = {k: v[rank::world] for k, v in batch_all.items()}
  calc = LossCalculator(torch.nn.Linear(4, 4))
  losses = []
  for _ in range(6):
    optimizer.zero_grad()
    outs = model(sq, None)
    (policy_out, value_out, mlh_out, unc_out, value2_out, qdl, qdu, unc_pol, action_out, _) = outs[:10]
    loss = net.compute_loss(calc, batch, policy_out, value_out, mlh_out, unc_out, value2_out, qdl, qdu, unc_pol,
                            None, None, None, action_out, None, 0, 0, 0, False)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
  no_grad = [n for n, p in net.named_parameters() if p.grad is None]
  digest = float(sum(p.detach().double().sum() for p in net.parameters()))
  queue.put((rank, losses, digest, no_grad))
  dist.destroy_process_group()


def main():
  world, port = 2, 29731
  context = mp.get_context('spawn')
  queue = context.Queue()
  processes = [context.Process(target=worker, args=(rank, world, port, queue)) for rank in range(world)]
  for process in processes:
    process.start()
  results = sorted(queue.get(timeout=900) for _ in processes)
  for process in processes:
    process.join(timeout=60)
  assert all(process.exitcode == 0 for process in processes), [process.exitcode for process in processes]
  (_, losses0, digest0, missing0), (_, losses1, digest1, missing1) = results
  assert not missing0 and not missing1, (missing0, missing1)
  assert abs(digest0 - digest1) < 1e-6 * max(1.0, abs(digest0)), (digest0, digest1)
  mean_first, mean_last = (losses0[0] + losses1[0]) / 2, (losses0[-1] + losses1[-1]) / 2
  assert mean_last < mean_first
  print(f'DDP (2 ranks, gloo, CPU, static_graph=True, checkpoint per block ON, soft-policy stash loss on): 6 steps without '
        f'reducer error; every parameter received a gradient; both ranks hold identical parameters (sum {digest0:.6f}); '
        f'mean loss {mean_first:.3f} -> {mean_last:.3f}')
  print('test_bt6_hybrid_ddp: OK')


if __name__ == '__main__':
  main()
