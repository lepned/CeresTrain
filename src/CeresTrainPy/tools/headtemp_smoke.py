"""Per-head logit temperature smoke test (tactics toolbox T4.1).

UseHeadLogitTemp multiplies each head's fully-assembled pre-softmax logits by
exp(log_temp), init log 0 = temp 1 — so unlike the soft-agg ARCH keys the
contract HERE IS step-0 bit-identity:
  - off: no params created, graph untouched.
  - on + shared weights: forward BIT-IDENTICAL to baseline (exact no-op init).
  - param accounting: one [num_heads] log_temp per layer, zero-init.
  - activity: perturbing one log_temp moves policy output (mechanism wired).
  - qkclip visibility: the _last_max_logit stash scales with the perturbed
    temp (monitor sees effective magnitudes — the headroom-warning fix).
  - gradient flow: loss reaches head_logit_temp.
  - export parity: torch.export capture reproduces eager outputs.

Run from CeresTrainPy:  python tools/headtemp_smoke.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE
from tools.tsb_smoke import _make_config, _make_net


def _mk(on):
    cfg = _make_config(tsb_enabled=False)
    cfg.id = 'headtemp_smoke'
    cfg.Exec_ID = 'headtemp_smoke'
    cfg.NetDef_HeadSharedLinearDiv = 1
    cfg.NetDef_SoftMoE_ExpertInputDim = 0
    cfg.NetDef_UseDiffAttention = False
    cfg.NetDef_UseRoPE = False
    cfg.NetDef_UseHeadLogitTemp = on
    return cfg


def main():
    torch.manual_seed(42)
    net_base = _make_net(_mk(False)).eval()
    net_ht = _make_net(_mk(True)).eval()

    base_ht = [n for n, _ in net_base.named_parameters() if 'head_logit_temp' in n]
    assert not base_ht, f"off must create no params, found {base_ht}"
    ht_params = [(n, p) for n, p in net_ht.named_parameters() if 'head_logit_temp' in n]
    n_layers = sum(1 for n, _ in net_ht.named_parameters() if n.endswith('attention.qkv.weight'))
    assert len(ht_params) == n_layers, f"expected {n_layers} log_temp tensors, got {len(ht_params)}"
    nh = net_ht.NUM_HEADS
    for n, p in ht_params:
        assert p.shape == (nh,) and torch.all(p == 0), f"{n}: bad shape/init {p.shape}"
    print(f"PASS: param accounting ({len(ht_params)} per-layer log_temp, shape [{nh}], zero-init)")

    # Shared weights -> bit-identity at init.
    src = net_base.state_dict()
    dst = dict(net_ht.state_dict())
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k].copy_(v)
    net_ht.load_state_dict(dst, strict=False)

    torch.manual_seed(0)
    squares = torch.randn(2, NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE)
    with torch.no_grad():
        out_b = net_base(squares, None)
        out_h = net_ht(squares, None)
    for i in (0, 1):
        assert torch.equal(out_b[i], out_h[i]), f"BIT-IDENTITY BROKEN at init (output {i})"
    print("PASS: step-0 bit-identity with shared weights (temp=1 exact no-op)")

    # Activity: perturb one layer's temps.
    with torch.no_grad():
        ht_params[0][1].fill_(0.7)
    with torch.no_grad():
        out_p = net_ht(squares, None)
    assert not torch.equal(out_b[0], out_p[0]), "perturbed temp did not move policy — not wired"
    assert all(torch.isfinite(o).all() for o in out_p if torch.is_tensor(o)), "non-finite output"
    print("PASS: activity (perturbed temp moves outputs, finite)")

    # QKClip monitor sees effective magnitudes: stash scales with temp.
    net_ht.train()
    first_attn = None
    for m in net_ht.modules():
        if isinstance(getattr(m, 'head_logit_temp', None), torch.nn.Parameter):
            first_attn = m
            break
    first_attn.qk_clip_monitor = True
    with torch.no_grad():
        _ = net_ht(squares, None)
        stash_hot = first_attn._last_max_logit.clone()
        first_attn.head_logit_temp.zero_()
        _ = net_ht(squares, None)
        stash_base = first_attn._last_max_logit.clone()
    ratio = (stash_hot / stash_base.clamp_min(1e-9)).mean().item()
    import math
    assert abs(ratio - math.exp(0.7)) < 0.15 * math.exp(0.7), \
        f"qkclip stash does not scale with temp (ratio {ratio:.3f}, expected ~{math.exp(0.7):.3f})"
    first_attn.qk_clip_monitor = False
    print(f"PASS: qkclip monitor sees temp-scaled logits (stash ratio {ratio:.3f} ≈ e^0.7)")

    # Gradient flow.
    net_ht.zero_grad(set_to_none=True)
    out = net_ht(squares, None)
    loss = out[0].float().pow(2).mean() + out[1].float().pow(2).mean()
    loss.backward()
    got = any('head_logit_temp' in n and p.grad is not None and p.grad.abs().sum() > 0
              for n, p in net_ht.named_parameters())
    assert got, "no gradient reached head_logit_temp"
    print("PASS: gradient flow")

    # Export parity (with the perturbed temps still in place).
    net_ht.eval()
    with torch.no_grad():
        out_e_ref = net_ht(squares, None)
    ep = torch.export.export(net_ht, (squares, None))
    with torch.no_grad():
        out_e = ep.module()(squares, None)
    assert torch.allclose(out_e[0], out_e_ref[0], atol=1e-5), "export/eager policy parity broken"
    assert torch.allclose(out_e[1], out_e_ref[1], atol=1e-5), "export/eager value parity broken"
    print("PASS: torch.export capture + eager parity")

    print("ALL HEAD-TEMP SMOKE CHECKS PASSED.")


if __name__ == '__main__':
    main()
