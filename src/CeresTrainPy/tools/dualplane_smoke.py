"""Dual-plane P-plane smoke test (dual_plane_concept.md Stage A1/Stage 0).

The P-plane routes into the net ONLY through zero-init value injects (and its
cross-read of the square flow is itself zero-init), so contract class =
step-0 bit-identity for BOTH heads, plus vpc-style policy isolation:
  - off: no dual_plane/dp_value params.
  - on + shared weights: policy AND value bit-identical at init.
  - ISOLATION: perturbing dp_value_inject moves value, policy bit-identical.
  - slot semantics (module-level): a hand-built sparse position selects
    exactly the occupied squares into slots; adding a piece changes the pool.
  - gradient flow: value loss reaches P-plane qkv, rel_proj, softmin_log_tau,
    cross-read x_k, and the trunk.
  - export parity (torch.export; TopK/Gather path).

Run from CeresTrainPy:  python tools/dualplane_smoke.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE
from tools.tsb_smoke import _make_config, _make_net
from dual_plane import DualPlane


def _mk(on):
    cfg = _make_config(tsb_enabled=False)
    cfg.id = 'dualplane_smoke'
    cfg.Exec_ID = 'dualplane_smoke'
    cfg.NetDef_HeadSharedLinearDiv = 1
    cfg.NetDef_SoftMoE_ExpertInputDim = 0
    cfg.NetDef_UseDiffAttention = False
    cfg.NetDef_UseRoPE = False
    cfg.NetDef_UseVisEdgeBias = True
    cfg.NetDef_VisEdgeFamilies = 'vis,xray,pinray,check,flight'
    cfg.NetDef_VisEdgeGates = ''
    cfg.NetDef_UseDualPlane = on
    return cfg


def _module_checks():
    torch.manual_seed(3)
    dp = DualPlane(s_dim=32, rel_channels=20, norm_type='RMSNorm').eval()
    B = 2
    sq = torch.zeros(B, 64, 13)
    sq[:, :, 0] = 1.0                       # all empty
    for b, squares_set in enumerate(((4, 12, 60), (4, 12, 60, 27))):
        for s in squares_set:
            sq[b, s, 0] = 0.0
            sq[b, s, 6 if s != 60 else 12] = 1.0
    E = torch.zeros(B, 64, 64, 20)
    s_flow = torch.randn(B, 64, 32)
    with torch.no_grad():
        out, toks, sel, occ = dp(sq, E, s_flow)
    assert out.shape == (B, 256) and torch.isfinite(out).all()
    assert toks.shape == (B, 32, 128) and sel.shape == (B, 32) and occ.shape == (B, 32)
    assert occ[0].sum() == 3 and occ[1].sum() == 4, "slot occupancy miscounted"
    # The occupied slots must be exactly the piece squares.
    real0 = set(sel[0][occ[0] > 0].tolist())
    assert real0 == {4, 12, 60}, f"slots {real0} != pieces"
    # Adding a 4th piece (batch row 1) must change the pooled summary.
    assert not torch.allclose(out[0], out[1]), "pool insensitive to piece count"
    print("PASS: module semantics (slots = piece squares, pools respond, finite)")


def main():
    _module_checks()

    torch.manual_seed(42)
    net_base = _make_net(_mk(False)).eval()
    net_dp = _make_net(_mk(True)).eval()

    assert not [n for n, _ in net_base.named_parameters() if 'dual_plane' in n or 'dp_value' in n]
    inj = dict(net_dp.named_parameters())['dp_value_inject.weight']
    assert torch.all(inj == 0), "dp_value_inject must be zero-init"
    print("PASS: param accounting (zero-init inject)")

    src = net_base.state_dict()
    dst = dict(net_dp.state_dict())
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k].copy_(v)
    net_dp.load_state_dict(dst, strict=False)

    torch.manual_seed(0)
    squares = torch.rand(2, NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE)
    with torch.no_grad():
        out_b = net_base(squares, None)
        out_d = net_dp(squares, None)
    for i in (0, 1):
        assert torch.equal(out_b[i], out_d[i]), f"BIT-IDENTITY BROKEN at init (output {i})"
    print("PASS: step-0 bit-identity (policy AND value)")

    with torch.no_grad():
        inj.normal_(0, 0.05)
        out_p = net_dp(squares, None)
    assert torch.equal(out_b[0], out_p[0]), "ISOLATION BROKEN: policy moved"
    assert not torch.equal(out_b[1], out_p[1]), "perturbed inject did not move value"
    assert all(torch.isfinite(o).all() for o in out_p if torch.is_tensor(o)), "non-finite"
    print("PASS: isolation (value moves, policy bit-identical)")

    net_dp.train()
    net_dp.zero_grad(set_to_none=True)
    out = net_dp(squares, None)
    loss = out[1].float().pow(2).mean()
    loss.backward()
    for frag in ('dual_plane.blocks.0.qkv', 'dual_plane.blocks.0.rel_proj',
                 'dual_plane.blocks.0.softmin_log_tau', 'dual_plane.x_out'):
        got = any(frag in n and p.grad is not None and p.grad.abs().sum() > 0
                  for n, p in net_dp.named_parameters())
        assert got, f"no gradient reached {frag}"
    # x_k sits BEHIND the zero-init x_out at step 0 (grad is exactly zero by
    # the chain rule — that is the design, x_out opens the path). Verify the
    # path is live once x_out is nonzero:
    with torch.no_grad():
        net_dp.dual_plane.x_out.weight.normal_(0, 0.05)
    net_dp.zero_grad(set_to_none=True)
    out = net_dp(squares, None)
    out[1].float().pow(2).mean().backward()
    got_xk = any('dual_plane.x_k' in n and p.grad is not None and p.grad.abs().sum() > 0
                 for n, p in net_dp.named_parameters())
    assert got_xk, "x_k still gradient-dead after opening x_out"
    got_qkv = any(n.endswith('attention.qkv.weight') and p.grad is not None
                  and p.grad.abs().sum() > 0 for n, p in net_dp.named_parameters())
    assert got_qkv, "no gradient reached the trunk"
    print("PASS: gradients reach P-plane (qkv/rel/tau/cross) and trunk")

    net_dp.eval()
    with torch.no_grad():
        out_ref = net_dp(squares, None)
    ep = torch.export.export(net_dp, (squares, None))
    with torch.no_grad():
        out_e = ep.module()(squares, None)
    assert torch.allclose(out_e[0], out_ref[0], atol=1e-5), "export/eager policy parity broken"
    assert torch.allclose(out_e[1], out_ref[1], atol=1e-5), "export/eager value parity broken"
    print("PASS: torch.export capture + eager parity (TopK/Gather path)")

    # --- BARE-chassis variant (A1 one-key-delta): UseVisEdgeBias OFF, the
    # P-plane builds its own VisibilityChannels; also softmin_heads=0 (A2 arm).
    cfg_bare = _mk(True)
    cfg_bare.NetDef_UseVisEdgeBias = False
    cfg_bare.NetDef_DualPlaneSoftMinHeads = 0
    cfg_bare_off = _mk(False)
    cfg_bare_off.NetDef_UseVisEdgeBias = False
    torch.manual_seed(42)
    net_b0 = _make_net(cfg_bare_off).eval()
    net_b1 = _make_net(cfg_bare).eval()
    assert not any('softmin_log_tau' in n for n, _ in net_b1.named_parameters()
                   if 'dual_plane' in n), "softmin_heads=0 must create no tau params"
    src = net_b0.state_dict()
    dst = dict(net_b1.state_dict())
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k].copy_(v)
    net_b1.load_state_dict(dst, strict=False)
    with torch.no_grad():
        o0 = net_b0(squares, None)
        o1 = net_b1(squares, None)
    for i in (0, 1):
        assert torch.equal(o0[i], o1[i]), f"bare-chassis bit-identity broken (output {i})"
    with torch.no_grad():
        dict(net_b1.named_parameters())['dp_value_inject.weight'].normal_(0, 0.05)
        o2 = net_b1(squares, None)
    assert torch.equal(o0[0], o2[0]) and not torch.equal(o0[1], o2[1]), \
        "bare-chassis isolation broken"
    print("PASS: bare-chassis variant (private VisibilityChannels, softmin=0, isolation)")

    # --- A3 variant: mover-bilinear policy decode on the bare chassis.
    cfg_a3 = _mk(True)
    cfg_a3.NetDef_UseVisEdgeBias = False
    cfg_a3.NetDef_DualPlanePolicyDecode = True
    torch.manual_seed(42)
    net_a3 = _make_net(cfg_a3).eval()
    assert torch.all(dict(net_a3.named_parameters())['dp_pol_q.weight'] == 0)
    dst = dict(net_a3.state_dict())
    for k, v in net_b0.state_dict().items():
        if k in dst and dst[k].shape == v.shape:
            dst[k].copy_(v)
    net_a3.load_state_dict(dst, strict=False)
    with torch.no_grad():
        oa = net_a3(squares, None)
    for i in (0, 1):
        assert torch.equal(o0[i], oa[i]), f"A3 bit-identity broken at init (output {i})"
    with torch.no_grad():
        dict(net_a3.named_parameters())['dp_pol_q.weight'].normal_(0, 0.05)
        oa2 = net_a3(squares, None)
    assert not torch.equal(o0[0], oa2[0]), "perturbed dp_pol_q did not move policy"
    assert torch.isfinite(oa2[0]).all(), "non-finite policy with decode term"
    net_a3.eval()
    with torch.no_grad():
        oa_ref = net_a3(squares, None)
    ep3 = torch.export.export(net_a3, (squares, None))
    with torch.no_grad():
        oa_e = ep3.module()(squares, None)
    assert torch.allclose(oa_e[0], oa_ref[0], atol=1e-5), "A3 export/eager parity broken"
    print("PASS: A3 mover-bilinear decode (bit-identity at init, policy moves when opened, export parity)")

    # --- dp4 variant: dim 256, 3 layers, interleaved (weight-shared) cross.
    cfg4 = _mk(True)
    cfg4.NetDef_UseVisEdgeBias = False
    cfg4.NetDef_DualPlanePolicyDecode = True
    cfg4.NetDef_DualPlaneDim = 256
    cfg4.NetDef_DualPlaneLayers = 3
    cfg4.NetDef_DualPlaneInterleave = True
    torch.manual_seed(42)
    net4 = _make_net(cfg4).eval()
    n_blk = sum(1 for n, _ in net4.named_parameters() if n.endswith('.qkv.weight') and 'dual_plane' in n)
    assert n_blk == 3, f"expected 3 P-blocks, got {n_blk}"
    assert net4.dual_plane.dp == 256 and net4.dual_plane.interleave_cross
    dst = dict(net4.state_dict())
    for k, v in net_b0.state_dict().items():
        if k in dst and dst[k].shape == v.shape:
            dst[k].copy_(v)
    net4.load_state_dict(dst, strict=False)
    with torch.no_grad():
        o4 = net4(squares, None)
    for i in (0, 1):
        assert torch.equal(o0[i], o4[i]), f"dp4 bit-identity broken at init (output {i})"
    # Open the zero-init gates first — with them at zero the P-plane gradient
    # is EXACTLY zero by construction (that is the no-op guarantee, not a bug).
    with torch.no_grad():
        dict(net4.named_parameters())['dp_value_inject.weight'].normal_(0, 0.05)
        dict(net4.named_parameters())['dp_pol_q.weight'].normal_(0, 0.05)
    net4.train(); net4.zero_grad(set_to_none=True)
    out4 = net4(squares, None)
    (out4[1].float().pow(2).mean() + out4[0].float().pow(2).mean()).backward()
    got = any('dual_plane.blocks.2' in n and p.grad is not None and p.grad.abs().sum() > 0
              for n, p in net4.named_parameters())
    assert got, "no gradient reached P-block 3"
    net4.eval()
    with torch.no_grad():
        o4r = net4(squares, None)
    ep4 = torch.export.export(net4, (squares, None))
    with torch.no_grad():
        o4e = ep4.module()(squares, None)
    assert torch.allclose(o4e[0], o4r[0], atol=1e-5) and torch.allclose(o4e[1], o4r[1], atol=1e-5)
    print("PASS: dp4 variant (256-dim, 3 blocks, interleaved cross — bit-identity, grads, export)")

    print("ALL DUAL-PLANE SMOKE CHECKS PASSED.")


if __name__ == '__main__':
    main()
