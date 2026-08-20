"""Tactical codebook smoke test (tactics toolbox T3.3).

UseTacticalCodebook adds a 256-vector motif library read by one post-trunk
cross-attn block with ZERO-INIT out-projection — contract class = exact
step-0 bit-identity (headtemp/refiner pattern):
  - off: no cbk_* params, graph untouched.
  - on + shared weights: forward BIT-IDENTICAL to baseline at init.
  - param accounting: cbk_q/keys/vals/out with expected shapes, out zero-init.
  - activity: perturbing cbk_out moves outputs (mechanism wired), finite.
  - gradient flow: loss reaches codebook keys/vals AND the trunk.
  - export parity: torch.export capture reproduces eager outputs.

Run from CeresTrainPy:  python tools/codebook_smoke.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE
from tools.tsb_smoke import _make_config, _make_net


def _mk(on):
    cfg = _make_config(tsb_enabled=False)
    cfg.id = 'codebook_smoke'
    cfg.Exec_ID = 'codebook_smoke'
    cfg.NetDef_HeadSharedLinearDiv = 1
    cfg.NetDef_SoftMoE_ExpertInputDim = 0
    cfg.NetDef_UseDiffAttention = False
    cfg.NetDef_UseRoPE = False
    cfg.NetDef_UseTacticalCodebook = on
    return cfg


def main():
    torch.manual_seed(42)
    net_base = _make_net(_mk(False)).eval()
    net_cb = _make_net(_mk(True)).eval()

    base_cb = [n for n, _ in net_base.named_parameters() if n.startswith('cbk_')]
    assert not base_cb, f"off must create no params, found {base_cb}"
    D = net_cb.EMBEDDING_DIM
    sd = dict(net_cb.named_parameters())
    assert sd['cbk_keys'].shape == (256, 64), f"cbk_keys {sd['cbk_keys'].shape}"
    assert sd['cbk_vals'].shape == (256, D), f"cbk_vals {sd['cbk_vals'].shape}"
    assert sd['cbk_q.weight'].shape == (64, D), f"cbk_q {sd['cbk_q.weight'].shape}"
    assert sd['cbk_out.weight'].shape == (D, D) and torch.all(sd['cbk_out.weight'] == 0), \
        "cbk_out must be zero-init"
    print("PASS: param accounting (q/keys/vals/out, out zero-init)")

    src = net_base.state_dict()
    dst = dict(net_cb.state_dict())
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k].copy_(v)
    net_cb.load_state_dict(dst, strict=False)

    torch.manual_seed(0)
    squares = torch.randn(2, NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE)
    with torch.no_grad():
        out_b = net_base(squares, None)
        out_c = net_cb(squares, None)
    for i in (0, 1):
        assert torch.equal(out_b[i], out_c[i]), f"BIT-IDENTITY BROKEN at init (output {i})"
    print("PASS: step-0 bit-identity with shared weights (zero-init out)")

    with torch.no_grad():
        net_cb.cbk_out.weight.normal_(0, 0.05)
        out_p = net_cb(squares, None)
    assert not torch.equal(out_b[0], out_p[0]), "perturbed cbk_out did not move policy"
    assert all(torch.isfinite(o).all() for o in out_p if torch.is_tensor(o)), "non-finite output"
    print("PASS: activity (perturbed out-proj moves outputs, finite)")

    net_cb.train()
    net_cb.zero_grad(set_to_none=True)
    out = net_cb(squares, None)
    loss = out[0].float().pow(2).mean() + out[1].float().pow(2).mean()
    loss.backward()
    for pname in ('cbk_keys', 'cbk_vals', 'cbk_q.weight'):
        p = sd[pname]
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient reached {pname}"
    got_qkv = any(n.endswith('attention.qkv.weight') and p.grad is not None
                  and p.grad.abs().sum() > 0 for n, p in net_cb.named_parameters())
    assert got_qkv, "no gradient reached the trunk"
    print("PASS: gradients reach codebook and trunk")

    net_cb.eval()
    with torch.no_grad():
        out_ref = net_cb(squares, None)
    ep = torch.export.export(net_cb, (squares, None))
    with torch.no_grad():
        out_e = ep.module()(squares, None)
    assert torch.allclose(out_e[0], out_ref[0], atol=1e-5), "export/eager policy parity broken"
    assert torch.allclose(out_e[1], out_ref[1], atol=1e-5), "export/eager value parity broken"
    print("PASS: torch.export capture + eager parity")

    print("ALL CODEBOOK SMOKE CHECKS PASSED.")


if __name__ == '__main__':
    main()
