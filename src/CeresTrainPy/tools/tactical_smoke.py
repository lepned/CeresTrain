"""Tactical-mechanism smoke test (2026-08 tactical program).

Covers the three mechanisms added for the puzzle-capacity campaign:
  1. VisibilityChannels check/flight families (chess_geometry.py)
  2. Graph-route heads (dot_product_attention.py, NetDef UseGraphRouteHeads)
  3. Iterated tactic refiner + deep supervision (tactical_refiner.py,
     NetDef RefinerIters / RefinerDeepSupWeight)

Verifies, on a tiny CPU net (pattern copied from tools/tsb_smoke.py):
  - step-0 EXACT equivalence: enabling all three mechanisms with shared
    weights copied from a baseline net leaves policy/value outputs
    bit-identical (zero-init / tanh-zero-gate invariants);
  - init invariants on the new parameters;
  - gradient flow reaches the mechanisms' first-phase parameters;
  - the refiner deep-supervision stash appears in train mode with the
    right shape and is absent in eval mode.

Run from CeresTrainPy:  python tools/tactical_smoke.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE
from ceres_net import CeresNet
from tools.tsb_smoke import _make_config, _make_net


def _make_tactical_config():
    cfg = _make_config(tsb_enabled=False)
    cfg.id = 'tactical_smoke'
    cfg.Exec_ID = 'tactical_smoke'
    # tsb_smoke's tiny net predates HeadSharedLinearDiv (default 4): with
    # ModelDim 64 x HeadWidthMultiplier 2 the premap width is 2 per square,
    # not divisible by 4.
    cfg.NetDef_HeadSharedLinearDiv = 1
    cfg.NetDef_SoftMoE_ExpertInputDim = 0
    cfg.NetDef_UseDiffAttention = False
    cfg.NetDef_UseRoPE = False
    # Smolgen on (the 256x10 target arch has it; exercises coexistence).
    cfg.NetDef_SmolgenDimPerSquare = 8
    cfg.NetDef_SmolgenDim = 32
    cfg.NetDef_SmolgenToHeadDivisor = 2
    cfg.NetDef_SmolgenActivationType = 'Swish'
    cfg.NetDef_SmolgenDeltaRank = 0
    # Vis edge-bias with ALL families incl. the new check/flight.
    cfg.NetDef_UseVisEdgeBias = True
    cfg.NetDef_VisEdgeFamilies = 'vis,xray,pinray,check,flight'
    cfg.NetDef_VisEdgeGates = 'qk'
    cfg.NetDef_VisEdgeSharedProjection = False
    # Graph-route heads.
    cfg.NetDef_UseGraphRouteHeads = True
    # Refiner with deep supervision.
    cfg.NetDef_RefinerIters = 3
    cfg.NetDef_RefinerDim = 32
    cfg.NetDef_RefinerHeads = 2
    cfg.NetDef_RefinerFFNMult = 2
    cfg.NetDef_RefinerDeepSupWeight = 0.25
    return cfg


def _make_baseline_config():
    cfg = _make_tactical_config()
    cfg.NetDef_UseVisEdgeBias = False
    cfg.NetDef_VisEdgeGates = ''
    cfg.NetDef_UseGraphRouteHeads = False
    cfg.NetDef_RefinerIters = 0
    cfg.NetDef_RefinerDeepSupWeight = 0.0
    return cfg


def main():
    torch.manual_seed(42)
    net_base = _make_net(_make_baseline_config()).eval()
    net_tact = _make_net(_make_tactical_config()).eval()

    # Copy shared weights from baseline into the tactical net.
    src = net_base.state_dict()
    dst = dict(net_tact.state_dict())
    copied = 0
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k].copy_(v); copied += 1
    net_tact.load_state_dict(dst, strict=False)
    print(f"copied {copied} shared tensors baseline -> tactical")

    # Init invariants.
    n_new = 0
    for n, p in net_tact.named_parameters():
        if 'vis_edge_proj' in n or 'attack_gate_' in n:
            assert torch.all(p == 0), f"{n} not zero-init"
            n_new += 1
        elif 'graph_route_gate' in n:
            assert torch.all(p == 0), f"{n} not zero-init"
            n_new += 1
        elif 'graph_route_w' in n:
            assert torch.all(p > 0), f"{n} not positive-init"
            n_new += 1
        elif 'tactical_refiner.proj_out' in n:
            assert torch.all(p == 0), f"{n} not zero-init"
            n_new += 1
        elif 'tactical_refiner' in n:
            n_new += 1
    assert n_new > 0
    print(f"init invariants verified over {n_new} new param tensors")

    # E channels sanity: check/flight present and check nonzero on a real-ish
    # board encoding is covered by test_chess_geometry; here just shape.
    E = net_tact.vis_channels_module(torch.nn.functional.one_hot(
        torch.zeros(2, 64, dtype=torch.long), 13).float())
    assert E.shape == (2, 64, 64, 20), f"expected 20 channels (5 families), got {E.shape}"
    print(f"E shape with 5 families: {tuple(E.shape)}")

    # Step-0 exact equivalence.
    torch.manual_seed(0)
    B = 2
    squares = torch.randn(B, NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE)
    with torch.no_grad():
        out_b = net_base(squares, None)
        out_t = net_tact(squares, None)
    assert torch.equal(out_b[0], out_t[0]), \
        f"policy differs at init: max {(out_b[0]-out_t[0]).abs().max():.3e}"
    assert torch.equal(out_b[1], out_t[1]), \
        f"value differs at init: max {(out_b[1]-out_t[1]).abs().max():.3e}"
    print("PASS: all-mechanisms forward is bit-identical to baseline at init")

    # Deep-sup stash: present in train mode with shape [B, iters-1, 1858];
    # absent in eval.
    net_tact.train()
    _ = net_tact(squares, None)
    stash = getattr(net_tact, '_last_refiner_policy', None)
    assert stash is not None and stash.shape == (B, 2, 1858), \
        f"bad deep-sup stash: {None if stash is None else stash.shape}"
    net_tact._last_refiner_policy = None
    net_tact.eval()
    with torch.no_grad():
        _ = net_tact(squares, None)
    assert getattr(net_tact, '_last_refiner_policy', None) is None, \
        "deep-sup stash must not be set in eval (export safety)"
    print("PASS: deep-sup stash train-only, shape [B, iters-1, 1858]")

    # Gradient flow into first-phase params (gate / zero-init projections).
    net_tact.train()
    net_tact.zero_grad(set_to_none=True)
    out = net_tact(squares, None)
    loss = out[0].float().pow(2).mean() + out[1].float().pow(2).mean() \
        + net_tact._last_refiner_policy.float().pow(2).mean()
    loss.backward()
    got = {'graph_route_gate': False, 'vis_edge_proj': False,
           'tactical_refiner.proj_out': False, 'attack_gate_': False}
    for n, p in net_tact.named_parameters():
        for key in got:
            if key in n and p.grad is not None and p.grad.abs().sum() > 0:
                got[key] = True
    for key, ok in got.items():
        assert ok, f"no gradient reached {key}"
    print(f"PASS: gradients reach {sorted(got)}")

    print("ALL TACTICAL SMOKE CHECKS PASSED.")


if __name__ == '__main__':
    main()
