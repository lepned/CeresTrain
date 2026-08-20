"""One-shot probe: distribution of policy-target sharpness (top1-top2 gap and
entropy) in a TPG corpus. Decides whether only-move CE weighting (toolbox T1.3)
has anything to grip on a given corpus: if gap is ~1 everywhere (one-hot
labels) the weighting is a uniform no-op there.

Usage: python tools/policy_gap_probe.py <tpg_dir> [num_batches]
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tpg_dataset import TPGDataset

def main():
    tpg_dir = sys.argv[1]
    num_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    BATCH = 1024
    sq_bytes = int(os.environ.get('PROBE_SQUARE_BYTES', '141'))
    ds = TPGDataset(tpg_dir, BATCH, 0.0, 0, 1, 1, 1, square_bytes=sq_bytes)
    ds.set_worker_id(0)  # outside a DataLoader worker: index % num_workers == None never matches
    gaps, ents, top1s = [], [], []
    for _ in range(num_batches):
        batch = ds[0][0]
        t = batch['policies'].float()
        t = t / t.sum(dim=1, keepdim=True).clamp_min(1e-9)
        top2 = t.topk(2, dim=1).values
        gaps.append(top2[:, 0] - top2[:, 1])
        top1s.append(top2[:, 0])
        tc = t.clamp_min(1e-9)
        ents.append(-(tc * tc.log()).sum(dim=1))
    g = torch.cat(gaps); e = torch.cat(ents); t1 = torch.cat(top1s)
    n = g.numel()
    print(f'n={n} positions from {tpg_dir}')
    qs = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95])
    print('gap  quantiles (5/25/50/75/95%):', [round(x, 4) for x in torch.quantile(g, qs).tolist()])
    print('top1 quantiles              :', [round(x, 4) for x in torch.quantile(t1, qs).tolist()])
    print('entropy quantiles (nats)    :', [round(x, 4) for x in torch.quantile(e, qs).tolist()])
    for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.95), (0.95, 1.01)]:
        frac = ((g >= lo) & (g < hi)).float().mean().item()
        print(f'  gap [{lo:.2f},{hi:.2f}): {100*frac:5.1f}%')
    lam = 2.0
    w = 1.0 + lam * g
    print(f'weight spread at lambda={lam}: mean {w.mean():.3f}, p5/p95 '
          f'{torch.quantile(w, torch.tensor([0.05, 0.95])).tolist()}')

if __name__ == '__main__':
    main()
