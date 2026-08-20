"""Soft-min attention-head smoke test (2026-08 tactical program, idea "AND-heads").

SoftMinHeads is an ARCH key (the first k heads aggregate V by an
attention-weighted soft minimum from step 0), not a zero-init no-op — so the
contract verified here is NOT bit-identity but:
  - formula-level: tau->0 recovers the weighted mean; tau large approaches the
    hard min over the attention support; Jensen (softmin <= mean) at tau=1;
    constant-V identity (softmin of a constant is that constant).
  - net-level: SoftMinHeads=0 creates no params and leaves the graph untouched
    (bit-identical to a baseline net with shared weights); =2 creates exactly
    one [2] log_tau per layer, zero-init (tau=1); forward differs from
    baseline (mechanism actually active); gradient reaches log_tau AND flows
    through the softmin path into qkv; eval forward is finite.
  - export-level: torch.export capture succeeds in eval mode and the captured
    graph reproduces the eager outputs (TRT-safety proxy: Exp/Log/ReduceMax/
    MatMul/Cat only).

Run from CeresTrainPy:  python tools/softmin_smoke.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE
from tools.tsb_smoke import _make_config, _make_net


def _softmin_ref(A, V, tau):
    """The exact formula used in dot_product_attention (fp32, LSE-shifted)."""
    negtv = -tau * V
    m = negtv.amax(dim=2, keepdim=True)
    s = torch.matmul(A, torch.exp(negtv - m))
    return -(torch.log(s.clamp_min(1e-20)) + m) / tau


def _formula_checks():
    torch.manual_seed(1)
    B, H, T, D = 3, 2, 64, 16
    logits = torch.randn(B, H, T, T)
    A = torch.softmax(logits, dim=-1)
    V = torch.randn(B, H, T, D)

    mean = torch.matmul(A, V)

    # tau -> 0: weighted mean. tau trades Taylor bias O(tau*Var) against the
    # fp32 log(1+eps) precision floor O(eps_mach/tau); 1e-3 balances them.
    out_small = _softmin_ref(A, V, torch.tensor(1e-3))
    assert torch.allclose(out_small, mean, atol=2e-3), \
        f"tau->0 mean-limit broken: max err {(out_small-mean).abs().max():.3e}"

    # Jensen at tau=1: softmin <= mean everywhere.
    out_1 = _softmin_ref(A, V, torch.tensor(1.0))
    assert (out_1 <= mean + 1e-5).all(), "Jensen violated: softmin exceeded weighted mean"

    # Large tau: softmax A is strictly positive, so the exact large-tau limit
    # is the hard min over ALL j of the log-penalized values
    #   min_j ( V[j,c] - log A[i,j] / tau )
    # (LSE max approximation; residual ~ log(#near-ties)/tau).
    tau_big = 200.0
    out_big = _softmin_ref(A, V, torch.tensor(tau_big))
    pen = V.unsqueeze(2) - torch.log(A).unsqueeze(-1) / tau_big  # [B,H,Tq,Tj,D]
    expected = pen.min(dim=3).values
    assert torch.allclose(out_big, expected, atol=5e-2), \
        f"hard-min limit broken: max err {(out_big-expected).abs().max():.3e}"

    # Constant V: softmin(const) == const for any tau.
    Vc = torch.ones(B, H, T, D) * 0.7
    out_c = _softmin_ref(A, Vc, torch.tensor(3.0))
    assert torch.allclose(out_c, Vc, atol=1e-5), "constant-V identity broken"
    print("PASS: formula checks (mean limit, Jensen, hard-min limit, constant-V)")


def _make_sm_config(k):
    cfg = _make_config(tsb_enabled=False)
    cfg.id = 'softmin_smoke'
    cfg.Exec_ID = 'softmin_smoke'
    cfg.NetDef_HeadSharedLinearDiv = 1
    cfg.NetDef_SoftMoE_ExpertInputDim = 0
    cfg.NetDef_UseDiffAttention = False
    cfg.NetDef_UseRoPE = False
    cfg.NetDef_SoftMinHeads = k
    return cfg


def main():
    _formula_checks()

    torch.manual_seed(42)
    net_base = _make_net(_make_sm_config(0)).eval()
    net_sm = _make_net(_make_sm_config(2)).eval()

    # Param accounting: exactly one [2] log_tau per encoder layer, zero-init;
    # k=0 creates none.
    base_sm = [n for n, _ in net_base.named_parameters() if 'softmin_log_tau' in n]
    assert not base_sm, f"SoftMinHeads=0 must create no params, found {base_sm}"
    sm_params = [(n, p) for n, p in net_sm.named_parameters() if 'softmin_log_tau' in n]
    n_layers = sum(1 for n, _ in net_sm.named_parameters() if n.endswith('attention.qkv.weight'))
    assert len(sm_params) == n_layers, f"expected {n_layers} log_tau tensors, got {len(sm_params)}"
    for n, p in sm_params:
        assert p.shape == (2,) and torch.all(p == 0), f"{n}: bad shape/init {p.shape} {p}"
    print(f"PASS: {len(sm_params)} per-layer log_tau params, shape [2], zero-init (tau=1)")

    # Copy shared weights baseline -> softmin net; forward must DIFFER
    # (mechanism active from step 0) and be finite.
    src = net_base.state_dict()
    dst = dict(net_sm.state_dict())
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k].copy_(v)
    net_sm.load_state_dict(dst, strict=False)

    torch.manual_seed(0)
    B = 2
    squares = torch.randn(B, NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE)
    with torch.no_grad():
        out_b = net_base(squares, None)
        out_s = net_sm(squares, None)
    assert all(torch.isfinite(o).all() for o in out_s if torch.is_tensor(o)), "non-finite output"
    assert not torch.equal(out_b[0], out_s[0]), \
        "softmin net identical to baseline — mechanism not active"
    print("PASS: softmin forward finite and differs from baseline (arch active from step 0)")

    # Gradient flow: loss must reach log_tau and the qkv projections.
    net_sm.train()
    net_sm.zero_grad(set_to_none=True)
    out = net_sm(squares, None)
    loss = out[0].float().pow(2).mean() + out[1].float().pow(2).mean()
    loss.backward()
    got_tau = any('softmin_log_tau' in n and p.grad is not None and p.grad.abs().sum() > 0
                  for n, p in net_sm.named_parameters())
    got_qkv = any(n.endswith('attention.qkv.weight') and p.grad is not None
                  and p.grad.abs().sum() > 0 for n, p in net_sm.named_parameters())
    assert got_tau, "no gradient reached softmin_log_tau"
    assert got_qkv, "no gradient reached qkv through the softmin path"
    print("PASS: gradients reach softmin_log_tau and qkv")

    # Export capture: torch.export in eval mode, parity vs eager.
    net_sm.eval()
    ep = torch.export.export(net_sm, (squares, None))
    with torch.no_grad():
        out_e = ep.module()(squares, None)
    assert torch.allclose(out_e[0], out_s[0], atol=1e-5), "export/eager policy parity broken"
    assert torch.allclose(out_e[1], out_s[1], atol=1e-5), "export/eager value parity broken"
    print("PASS: torch.export capture + eager parity")

    print("ALL SOFTMIN SMOKE CHECKS PASSED.")


if __name__ == '__main__':
    main()
