"""Signed-tau soft-MAX attention-head smoke test (2026-08 tactical program, T1.1).

SoftMaxAggHeads is the dual of SoftMinHeads: the m heads AFTER the softmin
group aggregate V with tau = -exp(log_tau) (init -1), turning the soft-min
into an attention-weighted soft MAXIMUM ("exists a threat" instead of
"all squares covered"). Same ARCH-key semantics — contract mirrors
softmin_smoke.py:
  - formula-level: negative tau in the SAME LSE expression is the exact dual
    (softmaxagg(A,V,t) == -softmin(A,-V,t)); tau->0- recovers the weighted
    mean; large |tau| approaches the hard max; Jensen (softmax_agg >= mean);
    constant-V identity.
  - net-level: SoftMaxAggHeads=0 creates no params; k=2,m=2 creates exactly
    one [2] softmin_log_tau AND one [2] softmax_log_tau per layer, zero-init;
    forward differs from both baseline and softmin-only (mechanism active);
    gradient reaches softmax_log_tau AND qkv; eval forward finite.
  - export-level: torch.export capture + eager parity.

Run from CeresTrainPy:  python tools/softmax_agg_smoke.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE
from tools.tsb_smoke import _make_config, _make_net
from tools.softmin_smoke import _softmin_ref


def _formula_checks():
    torch.manual_seed(1)
    B, H, T, D = 3, 2, 64, 16
    logits = torch.randn(B, H, T, T)
    A = torch.softmax(logits, dim=-1)
    V = torch.randn(B, H, T, D)

    mean = torch.matmul(A, V)

    # Exact duality: the shared formula with tau=-t must equal the negated
    # softmin of -V at tau=+t. This is the whole implementation claim.
    t = torch.tensor(1.7)
    out_neg = _softmin_ref(A, V, -t)
    dual = -_softmin_ref(A, -V, t)
    assert torch.allclose(out_neg, dual, atol=1e-5), \
        f"signed-tau duality broken: max err {(out_neg-dual).abs().max():.3e}"

    # tau -> 0-: weighted mean (same Taylor/precision tradeoff as softmin).
    out_small = _softmin_ref(A, V, torch.tensor(-1e-3))
    assert torch.allclose(out_small, mean, atol=2e-3), \
        f"tau->0- mean-limit broken: max err {(out_small-mean).abs().max():.3e}"

    # Jensen at tau=-1: soft-max >= mean everywhere.
    out_1 = _softmin_ref(A, V, torch.tensor(-1.0))
    assert (out_1 >= mean - 1e-5).all(), "Jensen violated: soft-max below weighted mean"

    # Large |tau|: hard max over the log-penalized values
    #   max_j ( V[j,c] + log A[i,j] / t )
    t_big = 200.0
    out_big = _softmin_ref(A, V, torch.tensor(-t_big))
    pen = V.unsqueeze(2) + torch.log(A).unsqueeze(-1) / t_big  # [B,H,Tq,Tj,D]
    expected = pen.max(dim=3).values
    assert torch.allclose(out_big, expected, atol=5e-2), \
        f"hard-max limit broken: max err {(out_big-expected).abs().max():.3e}"

    # Constant V: identity for any tau sign.
    Vc = torch.ones(B, H, T, D) * 0.7
    out_c = _softmin_ref(A, Vc, torch.tensor(-3.0))
    assert torch.allclose(out_c, Vc, atol=1e-5), "constant-V identity broken"
    print("PASS: formula checks (duality, mean limit, Jensen, hard-max limit, constant-V)")


def _make_sx_config(k_min, m_max):
    cfg = _make_config(tsb_enabled=False)
    cfg.id = 'softmax_agg_smoke'
    cfg.Exec_ID = 'softmax_agg_smoke'
    cfg.NetDef_HeadSharedLinearDiv = 1
    cfg.NetDef_SoftMoE_ExpertInputDim = 0
    cfg.NetDef_UseDiffAttention = False
    cfg.NetDef_UseRoPE = False
    cfg.NetDef_SoftMinHeads = k_min
    cfg.NetDef_SoftMaxAggHeads = m_max
    return cfg


def main():
    _formula_checks()

    torch.manual_seed(42)
    net_base = _make_net(_make_sx_config(0, 0)).eval()
    net_sm = _make_net(_make_sx_config(2, 0)).eval()
    net_sx = _make_net(_make_sx_config(2, 2)).eval()

    # Param accounting.
    base_sx = [n for n, _ in net_base.named_parameters() if 'softmax_log_tau' in n]
    assert not base_sx, f"SoftMaxAggHeads=0 must create no params, found {base_sx}"
    sm_only_sx = [n for n, _ in net_sm.named_parameters() if 'softmax_log_tau' in n]
    assert not sm_only_sx, f"softmin-only net must have no softmax_log_tau, found {sm_only_sx}"
    sx_params = [(n, p) for n, p in net_sx.named_parameters() if 'softmax_log_tau' in n]
    smin_params = [(n, p) for n, p in net_sx.named_parameters() if 'softmin_log_tau' in n]
    n_layers = sum(1 for n, _ in net_sx.named_parameters() if n.endswith('attention.qkv.weight'))
    assert len(sx_params) == n_layers, f"expected {n_layers} softmax_log_tau tensors, got {len(sx_params)}"
    assert len(smin_params) == n_layers, f"expected {n_layers} softmin_log_tau tensors, got {len(smin_params)}"
    for n, p in sx_params + smin_params:
        assert p.shape == (2,) and torch.all(p == 0), f"{n}: bad shape/init {p.shape} {p}"
    print(f"PASS: per-layer param accounting ({len(smin_params)}x softmin + {len(sx_params)}x softmax log_tau, shape [2], zero-init)")

    # Shared-weight copies baseline -> both nets; sx forward must differ from
    # BOTH baseline and softmin-only, and be finite.
    src = net_base.state_dict()
    for net in (net_sm, net_sx):
        dst = dict(net.state_dict())
        for k, v in src.items():
            if k in dst and dst[k].shape == v.shape:
                dst[k].copy_(v)
        net.load_state_dict(dst, strict=False)

    torch.manual_seed(0)
    B = 2
    squares = torch.randn(B, NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE)
    with torch.no_grad():
        out_b = net_base(squares, None)
        out_m = net_sm(squares, None)
        out_x = net_sx(squares, None)
    assert all(torch.isfinite(o).all() for o in out_x if torch.is_tensor(o)), "non-finite output"
    assert not torch.equal(out_b[0], out_x[0]), \
        "signed-tau net identical to baseline — mechanism not active"
    assert not torch.equal(out_m[0], out_x[0]), \
        "signed-tau net identical to softmin-only — softmax group not active"
    print("PASS: forward finite, differs from baseline AND softmin-only (both groups active)")

    # Gradient flow through the softmax group.
    net_sx.train()
    net_sx.zero_grad(set_to_none=True)
    out = net_sx(squares, None)
    loss = out[0].float().pow(2).mean() + out[1].float().pow(2).mean()
    loss.backward()
    got_tau = any('softmax_log_tau' in n and p.grad is not None and p.grad.abs().sum() > 0
                  for n, p in net_sx.named_parameters())
    got_qkv = any(n.endswith('attention.qkv.weight') and p.grad is not None
                  and p.grad.abs().sum() > 0 for n, p in net_sx.named_parameters())
    assert got_tau, "no gradient reached softmax_log_tau"
    assert got_qkv, "no gradient reached qkv through the soft-agg path"
    print("PASS: gradients reach softmax_log_tau and qkv")

    # Export capture + parity.
    net_sx.eval()
    ep = torch.export.export(net_sx, (squares, None))
    with torch.no_grad():
        out_e = ep.module()(squares, None)
    assert torch.allclose(out_e[0], out_x[0], atol=1e-5), "export/eager policy parity broken"
    assert torch.allclose(out_e[1], out_x[1], atol=1e-5), "export/eager value parity broken"
    print("PASS: torch.export capture + eager parity")

    print("ALL SOFTMAX-AGG SMOKE CHECKS PASSED.")


if __name__ == '__main__':
    main()
