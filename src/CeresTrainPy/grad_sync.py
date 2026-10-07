# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""Manual data-parallel gradient synchronisation (2026-10-07; opt DDPManualGradSync).

Why: under DDP static_graph (required by the stash-only aux heads) DDP cannot skip the all-reduce on accumulating
micro-steps (no_sync is unsupported there), so every micro-step all-reduces the full gradient. On the 4xA100 PCIe box
with 4 micro-steps per optimizer step that was 18-26 % of GPU time (torch.profiler, 10-07). Without the DDP wrapper,
gradients simply accumulate locally and are averaged ONCE per optimizer step here.

Mathematically identical to DDP: DDP averages each micro-step's gradients across ranks and accumulates;
sum over micro-steps of the cross-rank mean == the cross-rank mean of the locally accumulated sum.

Parameters whose .grad is None on SOME ranks get a zero gradient on those ranks (one tiny MAX all-reduce of a presence
mask agrees on the layout, so every rank packs identical buckets); parameters without a gradient on EVERY rank stay None,
so the optimizer skips them exactly as before.
"""
import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def broadcast_model(model, src: int = 0):
  """Make every rank start from rank src's parameters and buffers (what DDP does at construction)."""
  with torch.no_grad():
    for t in list(model.parameters()) + list(model.buffers()):
      dist.broadcast(t.data, src=src)


@torch.no_grad()
def all_reduce_grads(params, world_size: int, bf16: bool = False, bucket_elems: int = 32 * 1024 * 1024):
  """Average .grad of `params` (same list, same order on every rank) across ranks in place."""
  params = [p for p in params if p.requires_grad]
  if not params or world_size <= 1:
    return
  dev = params[0].device
  present = torch.tensor([p.grad is not None for p in params], dtype=torch.uint8, device=dev)
  dist.all_reduce(present, op=dist.ReduceOp.MAX)
  grads = []
  for p, m in zip(params, present.tolist()):
    if not m:
      continue
    if p.grad is None:
      p.grad = torch.zeros_like(p)
    grads.append(p.grad)
  # bucket by dtype, in order, ~bucket_elems elements each
  buckets, cur, cur_n, cur_dt = [], [], 0, None
  for g in grads:
    if cur and (g.dtype != cur_dt or cur_n + g.numel() > bucket_elems):
      buckets.append(cur); cur, cur_n = [], 0
    cur.append(g); cur_n += g.numel(); cur_dt = g.dtype
  if cur:
    buckets.append(cur)
  for b in buckets:
    flat = _flatten_dense_tensors(b)
    if bf16 and flat.dtype == torch.float32:
      wire = flat.to(torch.bfloat16)
      dist.all_reduce(wire)
      flat = wire.to(torch.float32)
    else:
      dist.all_reduce(flat)
    flat.div_(world_size)
    for g, s in zip(b, _unflatten_dense_tensors(flat, b)):
      g.copy_(s)
