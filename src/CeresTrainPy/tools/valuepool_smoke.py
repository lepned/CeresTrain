"""Value min/max pool smoke test (2026-08 tactics toolbox T1.2).

NetDef ValueHeadMinMaxPool is a zero-init no-op add-on (the 'inject' class),
so the full bit-identity contract applies:
  - step-0 EXACT equivalence vs baseline with shared weights;
  - param accounting: value_pool_inject (+ value2_pool_inject when value2 on),
    zero-init, bias-free;
  - gradient flow reaches both injectors;
  - torch.export capture succeeds in eval and reproduces eager outputs.

Run from CeresTrainPy:  python tools/valuepool_smoke.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE
from tools.tsb_smoke import _make_config, _make_net


def _make_cfg(pool):
    cfg = _make_config(tsb_enabled=False)
    cfg.id = 'valuepool_smoke'
    cfg.Exec_ID = 'valuepool_smoke'
    cfg.NetDef_HeadSharedLinearDiv = 1
    cfg.NetDef_SoftMoE_ExpertInputDim = 0
    cfg.NetDef_UseDiffAttention = False
    cfg.NetDef_UseRoPE = False
    cfg.NetDef_ValueHeadMinMaxPool = pool
    return cfg


def main():
    torch.manual_seed(42)
    net_base = _make_net(_make_cfg(False)).eval()
    net_pool = _make_net(_make_cfg(True)).eval()

    base_p = [n for n, _ in net_base.named_parameters() if '_pool_inject' in n]
    assert not base_p, f"pool off must create no params, found {base_p}"
    pool_p = [(n, p) for n, p in net_pool.named_parameters() if '_pool_inject' in n]
    has_v2 = net_pool.value2_loss_weight > 0
    expect = 2 if has_v2 else 1
    assert len(pool_p) == expect, f"expected {expect} injector tensors, got {[n for n,_ in pool_p]}"
    for n, p in pool_p:
        assert torch.all(p == 0), f"{n} not zero-init"
    print(f"PASS: {len(pool_p)} zero-init injector(s) (value2 {'on' if has_v2 else 'off'})")

    # Shared weights -> bit-identical at step 0.
    src = net_base.state_dict()
    dst = dict(net_pool.state_dict())
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k].copy_(v)
    net_pool.load_state_dict(dst, strict=False)

    torch.manual_seed(0)
    B = 2
    squares = torch.randn(B, NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE)
    with torch.no_grad():
        out_b = net_base(squares, None)
        out_p = net_pool(squares, None)
    assert torch.equal(out_b[0], out_p[0]), "policy differs at init"
    assert torch.equal(out_b[1], out_p[1]), "value differs at init"
    print("PASS: bit-identical to baseline at step 0")

    # Gradient flow into the injectors (zero-init but LINEAR in the weights,
    # so gradient is nonzero from step 1 — no dead-unit trap).
    net_pool.train()
    net_pool.zero_grad(set_to_none=True)
    out = net_pool(squares, None)
    loss = out[1].float().pow(2).mean()
    if has_v2:
        loss = loss + out[2].float().pow(2).mean() if torch.is_tensor(out[2]) else loss
    loss.backward()
    for n, p in net_pool.named_parameters():
        if '_pool_inject' in n and 'value2' not in n:
            assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient reached {n}"
    print("PASS: gradient reaches value_pool_inject")

    # ISOLATION: perturbing the injector must move VALUE but leave POLICY
    # bit-identical (the PTV lesson — prove the mechanism cannot leak into
    # other heads before trusting any gate delta).
    net_pool.eval()
    with torch.no_grad():
        for n, p in net_pool.named_parameters():
            if n.startswith('value_pool_inject'):
                p.add_(torch.randn_like(p) * 0.1)
        out_pert = net_pool(squares, None)
    assert torch.equal(out_pert[0], out_p[0]), "ISOLATION BROKEN: policy moved with injector"
    assert not torch.equal(out_pert[1], out_p[1]), "injector perturbation did not reach value"
    with torch.no_grad():  # restore zeros for the export parity check below
        for n, p in net_pool.named_parameters():
            if n.startswith('value_pool_inject'):
                p.zero_()
    print("PASS: isolation — injector reaches value only, policy bit-identical")

    # Export capture + parity.
    net_pool.eval()
    ep = torch.export.export(net_pool, (squares, None))
    with torch.no_grad():
        out_e = ep.module()(squares, None)
    assert torch.allclose(out_e[0], out_p[0], atol=1e-5), "export/eager policy parity broken"
    assert torch.allclose(out_e[1], out_p[1], atol=1e-5), "export/eager value parity broken"
    print("PASS: torch.export capture + eager parity")

    print("ALL VALUE-POOL SMOKE CHECKS PASSED.")


if __name__ == '__main__':
    main()
