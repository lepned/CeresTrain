"""Contract test for distributed Muon (muon.py enable_distributed, opt MuonDistributed; 2026-10-07).

    python test_muon_distributed.py            (CPU, gloo, 3 spawned ranks; run under Linux/WSL)

Every rank holds the same parameters and the same gradients (as after the gradient all-reduce). After several steps the
distributed optimizer must produce EXACTLY the parameters and momentum buffers of the replicated single-process Muon,
including per-head (head_split) matrices, AdamW params and weight decay; owners are balanced.
"""
import os, sys, tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORLD, STEPS = 3, 4
SHAPES = [(64, 32), (32, 96), (128, 128), (48, 16), (16, 48), (96, 64), (24, 40)]    # Muon takes 2-D only


def make():
  torch.manual_seed(0)
  muon = [torch.nn.Parameter(torch.randn(*s) * 0.1) for s in SHAPES]
  adam = [torch.nn.Parameter(torch.randn(10) * 0.1)]
  return muon, adam


def grads(step):
  g = torch.Generator().manual_seed(500 + step)
  return [torch.randn(*s, generator=g) for s in SHAPES], [torch.randn(10, generator=g)]


def run_opt(distributed, rank=0, world=1):
  from muon import Muon
  muon, adam = make()
  opt = Muon(lr=0.02, wd=0.01, muon_params=muon, adamw_params=adam, momentum=0.95,
             head_split_specs={muon[2]: (0, 4), muon[5]: (1, 2)}, hyperball_params=[muon[0]])
  if distributed:
    opt.enable_distributed(rank, world)
  for s in range(STEPS):
    gm, ga = grads(s)
    for i, (p, g) in enumerate(zip(muon + adam, gm + ga)):
      p.grad = None if (s == 2 and i == 3) else g.clone()     # one param without grad on one step (all ranks)
    opt.step()
  out = {'params': [p.detach().clone() for p in muon + adam],
         'bufs': [opt.state[p]['momentum_buffer'].clone() for p in muon]}
  if distributed:
    out['owner'] = [opt._owner[id(p)] for p in muon]
  return out


def worker(rank, init_file, out_dir):
  dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=WORLD)
  torch.save(run_opt(True, rank, WORLD), os.path.join(out_dir, f'r{rank}.pt'))
  dist.destroy_process_group()


if __name__ == '__main__':
  ref = run_opt(False)
  d = tempfile.mkdtemp(prefix='md_')
  mp.spawn(worker, args=(os.path.join(d, 'init'), d), nprocs=WORLD, join=True)
  for r in range(WORLD):
    res = torch.load(os.path.join(d, f'r{r}.pt'))
    for i, (a, b) in enumerate(zip(ref['params'], res['params'])):
      assert torch.equal(a, b), f'rank {r} param {i} differs (max {float((a - b).abs().max()):.3e})'
    for i, (a, b) in enumerate(zip(ref['bufs'], res['bufs'])):
      assert torch.equal(a, b), f'rank {r} momentum {i} differs'
  owners = res['owner']
  for r in range(WORLD):
    assert torch.load(os.path.join(d, f'r{r}.pt'))['owner'] == owners, 'owner map must be identical on every rank'
  assert len(set(owners)) == WORLD, f'every rank should own some matrix: {owners}'
  print(f'OK distributed Muon: {WORLD} ranks x {STEPS} steps bit-identical to the replicated step '
        f'(params, momentum; head_split, hyperball, a None-grad step, AdamW, wd); owners identical on all ranks {owners}')
  print('ALL OK')
