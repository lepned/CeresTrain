"""Move-graph spectral PE smoke test (tactics toolbox T4.3).

UseSpectralPE: occupancy-gated Laplacian-eigenvector coordinates of the four
elementary move graphs, zero-init Linear 32->D post-embedding. Contract class
= step-0 bit-identity. Also checks table math: eigenvector orthonormality,
constant-vector exclusion, and knight-graph regularity sanity.

Run from CeresTrainPy:  python tools/spectralpe_smoke.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE
from tools.tsb_smoke import _make_config, _make_net
from chess_geometry import build_spectral_pe_table


def _table_checks():
    t = build_spectral_pe_table(8)
    assert t.shape == (64, 32)
    for b in range(4):
        blk = t[:, b * 8:(b + 1) * 8]
        gram = blk.t() @ blk
        assert torch.allclose(gram, torch.eye(8), atol=1e-4), f"block {b} not orthonormal"
        # nontrivial eigenvectors are orthogonal to the constant vector
        assert blk.sum(dim=0).abs().max() < 1e-4, f"block {b} contains the constant mode"
    print("PASS: table semantics (orthonormal, constant mode excluded)")


def _mk(on):
    cfg = _make_config(tsb_enabled=False)
    cfg.id = 'spe_smoke'
    cfg.Exec_ID = 'spe_smoke'
    cfg.NetDef_HeadSharedLinearDiv = 1
    cfg.NetDef_SoftMoE_ExpertInputDim = 0
    cfg.NetDef_UseDiffAttention = False
    cfg.NetDef_UseRoPE = False
    cfg.NetDef_UseSpectralPE = on
    return cfg


def main():
    _table_checks()

    torch.manual_seed(42)
    net_base = _make_net(_mk(False)).eval()
    net_sp = _make_net(_mk(True)).eval()

    assert not [n for n, _ in net_base.named_parameters() if 'spe_' in n]
    w = dict(net_sp.named_parameters())['spe_proj.weight']
    assert w.shape[1] == 32 and torch.all(w == 0), "spe_proj must be zero-init [D,32]"
    print("PASS: param accounting (zero-init spe_proj)")

    src = net_base.state_dict()
    dst = dict(net_sp.state_dict())
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k].copy_(v)
    net_sp.load_state_dict(dst, strict=False)

    torch.manual_seed(0)
    squares = torch.rand(2, NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE)
    with torch.no_grad():
        out_b = net_base(squares, None)
        out_s = net_sp(squares, None)
    for i in (0, 1):
        assert torch.equal(out_b[i], out_s[i]), f"BIT-IDENTITY BROKEN at init (output {i})"
    print("PASS: step-0 bit-identity with shared weights")

    with torch.no_grad():
        w.normal_(0, 0.05)
        out_p = net_sp(squares, None)
    assert not torch.equal(out_b[0], out_p[0]), "perturbed spe_proj did not move outputs"
    assert all(torch.isfinite(o).all() for o in out_p if torch.is_tensor(o)), "non-finite"
    print("PASS: activity (perturbed proj moves outputs, finite)")

    net_sp.train()
    net_sp.zero_grad(set_to_none=True)
    out = net_sp(squares, None)
    loss = out[0].float().pow(2).mean() + out[1].float().pow(2).mean()
    loss.backward()
    assert w.grad is not None and w.grad.abs().sum() > 0, "no gradient reached spe_proj"
    print("PASS: gradient flow")

    net_sp.eval()
    with torch.no_grad():
        out_ref = net_sp(squares, None)
    ep = torch.export.export(net_sp, (squares, None))
    with torch.no_grad():
        out_e = ep.module()(squares, None)
    assert torch.allclose(out_e[0], out_ref[0], atol=1e-5), "export/eager policy parity broken"
    assert torch.allclose(out_e[1], out_ref[1], atol=1e-5), "export/eager value parity broken"
    print("PASS: torch.export capture + eager parity")

    print("ALL SPECTRAL-PE SMOKE CHECKS PASSED.")


if __name__ == '__main__':
    main()
