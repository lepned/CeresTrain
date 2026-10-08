"""Contract test for MoveTokenEdgeInput (move-token decoder reads the final EGT edge state; 2026-10-08).

    python test_mt_edge_input.py            (CPU, tiny NBT net with EGT + move tokens; no corpus)

1. Step 0: zero-init w_edge -> outputs bit-identical to the same net without edge input (paired weights).
2. Live: with w_edge non-zero the policy changes; gradients reach w_edge, the EGT state (p_in) and the trunk.
3. Oracle: the token edge feature equals rms(e[from, to]) @ w_edge for the gathered pairs.
4. Eval fused path (export form) == training (unfused) path in eval mode.
5. Refusal: MoveTokenEdgeInput without EGTEdgeStream.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_egt_edge as T

MT = {'UseMoveTokens': True, 'MoveTokenDim': 32, 'MoveTokenLayers': 1, 'MoveTokenHeads': 2, 'MoveTokenMax': 64}


def main():
  sq, batch = T.boards(4), T.batch_for(4)
  base = T.build({**T.EGT_ON, **MT}, 'mt0')
  edge = T.build({**T.EGT_ON, **MT, 'MoveTokenEdgeInput': True}, 'mt1')
  assert edge.move_tokens.w_edge is not None and base.move_tokens.w_edge is None
  sd_b, sd_e = base.state_dict(), edge.state_dict()
  extra = sorted(set(sd_e) - set(sd_b))
  assert extra == ['move_tokens.w_edge'], extra
  for k in sd_b:
    assert torch.equal(sd_b[k], sd_e[k]), f'init pairing broken at {k} (w_edge must not draw from the RNG)'
  with torch.no_grad():
    g = torch.Generator().manual_seed(41)
    for n, p in base.named_parameters():
      if (n.startswith('egt.') and p.abs().sum() == 0) or n.endswith('.up.weight'):
        p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    edge.load_state_dict({**base.state_dict(), 'move_tokens.w_edge': edge.move_tokens.w_edge.detach().clone()})
    base.eval(); edge.eval()
    ob, oe = base(sq, None), edge(sq, None)
  for i, (a, b) in enumerate(zip(ob, oe)):
    if torch.is_tensor(a):
      assert torch.equal(a, b), f'step-0 output {i} differs'
  print(f'OK step 0: {len(extra)} new tensor (move_tokens.w_edge), init paired, outputs bit-identical')
  from wd_partition import partition_weight_decay
  decay, no_decay = partition_weight_decay(edge)[:2]
  assert 'move_tokens.w_edge' in no_decay, 'w_edge must be partitioned (no_decay)'
  print('OK wd partition: move_tokens.w_edge -> no_decay')

  # 3 oracle on the token feature
  mt = edge.move_tokens
  with torch.no_grad():
    mt.w_edge.copy_(torch.randn(mt.w_edge.shape, generator=g) * 0.5)
    e = torch.randn(2, 64, 64, mt.edge_dim, generator=g)
    sel = torch.randint(0, 4096, (2, 7), generator=g)
    got = mt._edge_tokens(e, sel, torch.float32)
    fr, to = sel // 64, sel % 64
    ref = torch.stack([e[b, fr[b], to[b]] for b in range(2)])
    ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = ref @ mt.w_edge
  assert torch.allclose(got, ref, atol=1e-5), float((got - ref).abs().max())
  print(f'OK oracle: token feature == rms(e[from, to]) @ w_edge (max diff {float((got - ref).abs().max()):.1e})')

  # 2 live
  with torch.no_grad():
    oe2 = edge(sq, None)
  assert not torch.allclose(oe2[0], ob[0]), 'live edge input must change the policy'
  edge.train(); edge.zero_grad(set_to_none=True)
  loss = T.run_loss(edge, batch, sq)
  assert torch.isfinite(loss)
  loss.backward()
  P = dict(edge.named_parameters())
  for n in ('move_tokens.w_edge', 'egt.p_in', 'egt.site_mods.0.ffn_out'):
    assert P[n].grad is not None and P[n].grad.abs().sum() > 0, f'{n}: no gradient'
  print(f'OK live: loss {loss.item():.4f}, gradients reach w_edge, the EGT state (p_in, site FFN) and the trunk')

  # 4 fused eval path vs unfused
  edge.eval()
  with torch.no_grad():
    mt.export_fused = True
    of = edge(sq, None)
    mt.export_fused = False
    ou = edge(sq, None)
  assert torch.allclose(of[0], ou[0], atol=1e-4), float((of[0] - ou[0]).abs().max())
  print(f'OK fused export path == unfused (policy max diff {float((of[0] - ou[0]).abs().max()):.1e})')

  # 4b opponent tokens (MoveTokenOppMax > 0): the sel_o edge path
  opp = T.build({**T.EGT_ON, **MT, 'MoveTokenEdgeInput': True, 'MoveTokenOppMax': 16}, 'mt2')
  with torch.no_grad():
    opp.move_tokens.w_edge.copy_(torch.randn(opp.move_tokens.w_edge.shape, generator=g) * 0.5)
  opp.train(); opp.zero_grad(set_to_none=True)
  l2 = T.run_loss(opp, batch, sq)
  l2.backward()
  assert torch.isfinite(l2) and opp.move_tokens.w_edge.grad.abs().sum() > 0
  print(f'OK opponent tokens: loss {l2.item():.4f}, w_edge gets gradient with MoveTokenOppMax 16')

  # 5 refusal
  try:
    T.build({**MT, 'MoveTokenEdgeInput': True}, 'rej')
    raise SystemExit('FAIL: MoveTokenEdgeInput without EGTEdgeStream accepted')
  except ValueError:
    pass
  print('OK refusal (no EGTEdgeStream)')
  print('ALL OK')


if __name__ == '__main__':
  main()
