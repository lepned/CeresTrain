"""BT6EdgeSite.forward (channels-last GEMM form) == forward_reference (the literal einsum form): outputs and the
gradients of every parameter and of both inputs, for both triplet contractions, with / without the reverse edge and
the triplet branch. float64, so the comparison is free of float32 ordering noise.

  python test_bt6_site_fast.py
"""
import torch

import bt6_net


def test_site_fast():
  worst = 0.0
  cases = 0
  for contraction in ('path', 'ag'):
    for rev in (True, False):
      for triplet in (True, False):
        for heads in (4, 24):
          torch.manual_seed(3)
          cfg = bt6_net.BT6Config(d_model=heads * 8, layers=1, heads=heads, ffn_hidden=heads * 8, edge_sites=(),
                                  site_hidden=24, policy_width=16, error_head=False, rev_edge=rev,
                                  triplet=triplet, triplet_contraction=contraction)
          site = bt6_net.BT6EdgeSite(cfg).double()
          with torch.no_grad():
            for p in site.parameters():                        # the zero-initialised maps made live
              p.copy_(torch.randn_like(p) * 0.4)
          e = torch.randn(3, cfg.edge_channels, 64, 64, dtype=torch.float64, requires_grad=True)
          H = torch.randn(3, heads, 64, 64, dtype=torch.float64, requires_grad=True)
          # .float() inside the site would drop to float32: run both forms with the casts made no-ops
          real_float = torch.Tensor.float
          torch.Tensor.float = lambda t: t
          try:
            a = site(e, H)
            b = site.forward_reference(e, H)
            w = torch.randn_like(a)
            ga = torch.autograd.grad((a * w).sum(), [e, H] + list(site.parameters()))
            gb = torch.autograd.grad((b * w).sum(), [e, H] + list(site.parameters()))
          finally:
            torch.Tensor.float = real_float
          assert a.shape == b.shape == e.shape and a.is_contiguous()
          worst = max(worst, float((a - b).abs().max() / b.abs().max()))
          for x, y in zip(ga, gb):
            worst = max(worst, float((x - y).abs().max() / y.abs().max().clamp(min=1e-30)))
          cases += 1
  assert worst < 1e-10, worst
  print(f'test_site_fast: OK  ({cases} configurations; outputs and all gradients equal to relative {worst:.1e} in float64)')


if __name__ == '__main__':
  test_site_fast()
