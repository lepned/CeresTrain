"""Contract test for grad_sync.py (manual once-per-step gradient all-reduce, opt DDPManualGradSync; 2026-10-07).

    python test_grad_sync.py            (CPU, gloo, 3 spawned ranks)

Each rank accumulates 3 micro-steps on its OWN data, then all_reduce_grads averages once. Checked against a single-process
reference computed from all ranks' data:
  * every rank ends with grad == mean over ranks of (sum over its micro-steps)   (== what DDP produces)
  * a parameter used only on rank 0 gets grad_0 / world everywhere (absent ranks contribute zeros, as DDP)
  * a parameter used on NO rank keeps grad None (optimizer skips it, as before)
  * the bf16 wire variant matches within bf16 rounding
  * broadcast_model makes all ranks start from rank 0's weights
"""
import os, sys, tempfile
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORLD, MICRO = 3, 3


class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.a = nn.Linear(8, 16)
    self.b = nn.Linear(16, 4)
    self.only0 = nn.Linear(8, 4)       # used on rank 0 only
    self.never = nn.Linear(8, 4)       # used nowhere

  def forward(self, x, rank):
    y = self.b(torch.relu(self.a(x)))
    if rank == 0:
      y = y + self.only0(x)
    return y


def data(rank, micro):
  g = torch.Generator().manual_seed(1000 + 10 * rank + micro)
  return torch.randn(5, 8, generator=g), torch.randn(5, 4, generator=g)


def worker(rank, init_file, bf16, out_dir, bucket_elems=32 * 1024 * 1024, bf16_model=False):
  dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=WORLD)
  from grad_sync import broadcast_model, all_reduce_grads
  torch.manual_seed(100 + rank)                       # DIFFERENT init per rank -> broadcast must fix it
  m = Net()
  if bf16_model:
    m = m.to(torch.bfloat16)                          # BFloat16Pure: bf16 grads (the fp32-only bf16-wire cast is skipped)
  broadcast_model(m)
  for k in range(MICRO):
    x, t = data(rank, k)
    if bf16_model:
      x, t = x.to(torch.bfloat16), t.to(torch.bfloat16)
    ((m(x, rank) - t) ** 2).mean().backward()
  all_reduce_grads(list(m.parameters()), WORLD, bf16=bf16, bucket_elems=bucket_elems)
  torch.save({'params': {n: p.detach().clone() for n, p in m.named_parameters()},
              'grads': {n: (None if p.grad is None else p.grad.clone()) for n, p in m.named_parameters()}},
             os.path.join(out_dir, f'r{rank}.pt'))
  dist.destroy_process_group()


def reference(w0, bf16_model=False):
  m = Net()
  if bf16_model:
    m = m.to(torch.bfloat16)
  m.load_state_dict(w0)
  acc = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
  for r in range(WORLD):
    m.zero_grad(set_to_none=True)
    for k in range(MICRO):
      x, t = data(r, k)
      if bf16_model:
        x, t = x.to(torch.bfloat16), t.to(torch.bfloat16)
      ((m(x, r) - t) ** 2).mean().backward()
    for n, p in m.named_parameters():
      if p.grad is not None:
        acc[n] += p.grad
  return {n: v / WORLD for n, v in acc.items()}


def run(bf16, bucket_elems=32 * 1024 * 1024, bf16_model=False):
  d = tempfile.mkdtemp(prefix='gs_')
  init = os.path.join(d, 'init').replace('\\', '/')
  mp.spawn(worker, args=(init, bf16, d, bucket_elems, bf16_model), nprocs=WORLD, join=True)
  res = [torch.load(os.path.join(d, f'r{r}.pt')) for r in range(WORLD)]
  w0 = res[0]['params']
  for r in range(1, WORLD):
    for n in w0:
      assert torch.equal(w0[n], res[r]['params'][n]), f'broadcast: rank {r} {n} differs'
  ref = reference(w0, bf16_model)
  tol = 2e-2 if (bf16 or bf16_model) else 1e-6
  for r in range(WORLD):
    g = res[r]['grads']
    assert g['never.weight'] is None and g['never.bias'] is None, 'unused-everywhere params must keep grad None'
    for n in ('a.weight', 'a.bias', 'b.weight', 'b.bias', 'only0.weight', 'only0.bias'):
      err = float((g[n].float() - ref[n].float()).abs().max() / (ref[n].float().abs().max() + 1e-12))
      assert err < tol, (bf16, r, n, err)
  print(f'OK {"bf16" if bf16 else "fp32"} wire, bucket_elems {bucket_elems}, {"bf16" if bf16_model else "fp32"} model: {WORLD} ranks x {MICRO} micro-steps == reference mean of sums; '
        f'rank-0-only param averaged with zeros; unused stays None; broadcast exact')


if __name__ == '__main__':
  run(False)
  run(True)
  run(False, bucket_elems=50)              # many buckets (size split)
  run(True, bf16_model=True)               # bf16 grads with the bf16-wire flag (fp32-only cast guard skipped)
  print('ALL OK')
