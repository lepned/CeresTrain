"""Contract test for AttentionNorm 'softpick' (rectified softmax, arXiv 2504.20966; 2026-10-08).

    python test_softpick.py            (CPU, tiny NBT net with EGT; no corpus)

1. Oracle: Softpick == ReLU(e^x - 1) / (sum |e^x - 1| + eps) in float64, incl. large logits (stable form), all-negative
   rows (exact zeros), rows sum <= 1, fp16/bf16 inputs finite.
2. Full net (NBT + EGT + vis-edge bias) with softpick: every trunk attention uses Softpick, forward/backward finite,
   gradients reach the trunk; the output differs from the softmax net with identical weights.
3. EGT readback: the layer's normalizer applied to the raw readback logits == the attention probabilities used.
4. Refusals: unknown AttentionNorm; softpick + AttentionSinkLogit.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_egt_edge as T
from dot_product_attention import Softpick, DotProductAttention


def oracle():
  sp = Softpick(eps=1e-6)
  g = torch.Generator().manual_seed(1)
  x = torch.randn(3, 4, 64, 64, generator=g, dtype=torch.float64) * 3
  x[0, 0, 0] = -torch.rand(64, generator=g, dtype=torch.float64) - 0.1      # all-negative row
  x[0, 0, 1] = x[0, 0, 1] + 40.0                                           # large logits (naive e^x would be ~1e17+)
  ref = torch.relu(torch.expm1(x)) / (torch.expm1(x).abs().sum(-1, keepdim=True) + 1e-6)
  out = sp(x.float()).double()
  assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4), float((out - ref).abs().max())
  assert torch.equal(out[0, 0, 0], torch.zeros(64, dtype=torch.float64)), 'all-negative row must be exactly zero'
  s = out.sum(-1)
  assert (s <= 1 + 1e-5).all() and (out >= 0).all()
  for dt in (torch.float16, torch.bfloat16):
    o = sp((x * 2).to(dt))
    assert o.dtype == dt and torch.isfinite(o).all()
  print(f'OK oracle (max diff {float((out - ref).abs().max()):.1e}), all-negative row = 0, rows sum <= 1 '
        f'(min {float(s.min()):.3f}), fp16/bf16 finite')


def main():
  oracle()
  NET = {**T.EGT_ON, 'UseVisEdgeBias': True}
  T.EGT_ON = {**T.EGT_ON, 'UseVisEdgeBias': True, 'AttentionNorm': 'softpick'}
  sp_net = T.build(T.EGT_ON, 'sp')
  sm_net = T.build(NET, 'sm')
  atts = [m for m in sp_net.modules() if isinstance(m, DotProductAttention)]
  trunk_atts = [a for a in atts if isinstance(a.softmax, Softpick)]
  n_inner = T.NET_BASE['NumLayers'] * T.NET_BASE['NBTInnerLayers']
  assert len(trunk_atts) == n_inner, (len(trunk_atts), n_inner)
  print(f'OK {len(trunk_atts)} trunk attentions use Softpick ({len(atts)} DotProductAttention modules in total)')
  sq, batch = T.boards(4), T.batch_for(4)
  with torch.no_grad():
    gen = torch.Generator().manual_seed(31)
    for n, p in sm_net.named_parameters():
      if (n.startswith('egt.') or n.startswith('vis_edge_proj')) and p.abs().sum() == 0 or n.endswith('.up.weight'):
        p.copy_(torch.randn(p.shape, generator=gen) * 0.3)
    missing, unexpected = sp_net.load_state_dict(sm_net.state_dict(), strict=True), None
    sp_net.eval(); sm_net.eval()
    o_sp, o_sm = sp_net(sq, None)[0], sm_net(sq, None)[0]
  assert torch.isfinite(o_sp).all() and not torch.allclose(o_sp, o_sm, atol=1e-4)
  sp_net.train(); sp_net.zero_grad(set_to_none=True)
  loss = T.run_loss(sp_net, batch, sq)
  assert torch.isfinite(loss)
  loss.backward()
  P = dict(sp_net.named_parameters())
  for n in ('vis_edge_proj.0.weight', 'egt.reads.1.w_e'):
    assert P[n].grad is not None and P[n].grad.abs().sum() > 0, f'{n}: no gradient'
  tr = [n for n, p in sp_net.named_parameters() if 'transformer_layer' in n and p.grad is not None and p.grad.abs().sum() > 0]
  assert tr
  print(f'OK softpick net: loss {loss.item():.4f}, finite, differs from softmax net, grads reach vis bias / EGT / trunk')
  T.check_readback(sp_net, sq, 'softpick')
  for over, what in (({'AttentionNorm': 'sparsemax'}, 'unknown AttentionNorm'),):
    try:
      T.build(over, 'rej')
      raise SystemExit(f'FAIL: {what} accepted')
    except ValueError:
      pass
  os.environ['CERES_ATTENTION_SINK_LOGIT'] = '1'
  try:
    T.build({'AttentionNorm': 'softpick'}, 'rej2')
    raise SystemExit('FAIL: softpick + sink logit accepted')
  except AssertionError:
    pass
  finally:
    os.environ.pop('CERES_ATTENTION_SINK_LOGIT', None)
  print('OK refusals (unknown norm, softpick + sink logit)')
  print('ALL OK')


if __name__ == '__main__':
  main()
