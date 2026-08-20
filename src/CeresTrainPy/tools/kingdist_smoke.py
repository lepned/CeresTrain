"""King-distance channels smoke test (tactics toolbox T3.2).

UseKingDistChannels: per-square one-hot Chebyshev-distance buckets to both
kings via constant-table matmul, zero-init Linear 16->D added post-embedding.
Contract class = step-0 bit-identity:
  - table semantics: hand-checked distances land in the right buckets.
  - off: no params; on + shared weights: bit-identical forward at init.
  - activity after perturbing kdist_proj; finite.
  - gradient flow to kdist_proj (and trunk unaffected claim is N/A — additive
    at embedding, everything downstream sees it).
  - export parity.

Run from CeresTrainPy:  python tools/kingdist_smoke.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE
from tools.tsb_smoke import _make_config, _make_net
from chess_geometry import build_king_distance_onehot_table


def _table_checks():
    t = build_king_distance_onehot_table()
    assert t.shape == (64, 512)
    cases = [
        (4, 63, 7),   # king e1, square h8: max(3, 7) = 7
        (4, 4, 0),    # same square
        (4, 12, 1),   # e1 -> e2
        (27, 36, 1),  # d4 -> e5 diagonal = 1 (Chebyshev)
        (0, 7, 7),    # a1 -> h1
        (0, 9, 1),    # a1 -> b2
    ]
    for k, i, d in cases:
        row = t[k, i * 8:(i + 1) * 8]
        assert row.sum() == 1.0 and row[d] == 1.0, f"king {k} sq {i}: expected bucket {d}, got {row.tolist()}"
    print("PASS: table semantics (Chebyshev buckets hand-checked)")


def _mk(on):
    cfg = _make_config(tsb_enabled=False)
    cfg.id = 'kingdist_smoke'
    cfg.Exec_ID = 'kingdist_smoke'
    cfg.NetDef_HeadSharedLinearDiv = 1
    cfg.NetDef_SoftMoE_ExpertInputDim = 0
    cfg.NetDef_UseDiffAttention = False
    cfg.NetDef_UseRoPE = False
    cfg.NetDef_UseKingDistChannels = on
    return cfg


def main():
    _table_checks()

    torch.manual_seed(42)
    net_base = _make_net(_mk(False)).eval()
    net_kd = _make_net(_mk(True)).eval()

    assert not [n for n, _ in net_base.named_parameters() if 'kdist' in n]
    kd_w = dict(net_kd.named_parameters())['kdist_proj.weight']
    assert kd_w.shape[1] == 16 and torch.all(kd_w == 0), "kdist_proj must be zero-init [D,16]"
    print("PASS: param accounting (zero-init kdist_proj)")

    src = net_base.state_dict()
    dst = dict(net_kd.state_dict())
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k].copy_(v)
    net_kd.load_state_dict(dst, strict=False)

    torch.manual_seed(0)
    squares = torch.rand(2, NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE)
    with torch.no_grad():
        out_b = net_base(squares, None)
        out_k = net_kd(squares, None)
    for i in (0, 1):
        assert torch.equal(out_b[i], out_k[i]), f"BIT-IDENTITY BROKEN at init (output {i})"
    print("PASS: step-0 bit-identity with shared weights")

    with torch.no_grad():
        kd_w.normal_(0, 0.05)
        out_p = net_kd(squares, None)
    assert not torch.equal(out_b[0], out_p[0]), "perturbed kdist_proj did not move outputs"
    assert all(torch.isfinite(o).all() for o in out_p if torch.is_tensor(o)), "non-finite"
    print("PASS: activity (perturbed proj moves outputs, finite)")

    net_kd.train()
    net_kd.zero_grad(set_to_none=True)
    out = net_kd(squares, None)
    loss = out[0].float().pow(2).mean() + out[1].float().pow(2).mean()
    loss.backward()
    assert kd_w.grad is not None and kd_w.grad.abs().sum() > 0, "no gradient reached kdist_proj"
    print("PASS: gradient flow")

    net_kd.eval()
    with torch.no_grad():
        out_ref = net_kd(squares, None)
    ep = torch.export.export(net_kd, (squares, None))
    with torch.no_grad():
        out_e = ep.module()(squares, None)
    assert torch.allclose(out_e[0], out_ref[0], atol=1e-5), "export/eager policy parity broken"
    assert torch.allclose(out_e[1], out_ref[1], atol=1e-5), "export/eager value parity broken"
    print("PASS: torch.export capture + eager parity")

    print("ALL KING-DIST SMOKE CHECKS PASSED.")


if __name__ == '__main__':
    main()
