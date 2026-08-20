"""Value pool CHANNELS smoke test (tactics toolbox T1.2, "channels" variant).

NetDef ValueHeadPoolChannels concatenates trunk min/max-over-squares summaries
as first-class input columns to the value family's first linear (default init,
active from step 0) — unlike the zero-init inject of ValueHeadMinMaxPool.
ARCH key, so the contract is NOT step-0 bit-identity of value outputs:
  - off-identity: key off creates the exact baseline shapes and, with shared
    weights, a bit-identical forward.
  - shape accounting: key on widens value_head.fc (and value2_head.fc) input
    by exactly 2*EMBEDDING_DIM; no other parameter changes shape.
  - ISOLATION: with all shared weights copied from baseline, the POLICY output
    is bit-identical — the mechanism can only touch the value family.
  - activity: value output differs from baseline (channels carry signal from
    step 0) and is finite.
  - gradient flow: loss on value reaches the widened columns AND flows through
    the pool into the trunk (qkv).
  - export parity: torch.export capture reproduces eager outputs.

Run from CeresTrainPy:  python tools/valuepool_channels_smoke.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE
from tools.tsb_smoke import _make_config, _make_net


def _mk(pool_on):
    cfg = _make_config(tsb_enabled=False)
    cfg.id = 'vpc_smoke'
    cfg.Exec_ID = 'vpc_smoke'
    cfg.NetDef_HeadSharedLinearDiv = 1
    cfg.NetDef_SoftMoE_ExpertInputDim = 0
    cfg.NetDef_UseDiffAttention = False
    cfg.NetDef_UseRoPE = False
    cfg.NetDef_ValueHeadPoolChannels = pool_on
    return cfg


def main():
    torch.manual_seed(42)
    net_base = _make_net(_mk(False)).eval()
    net_vpc = _make_net(_mk(True)).eval()

    # --- shape accounting ---
    sd_b = net_base.state_dict()
    sd_v = net_vpc.state_dict()
    emb = net_base.EMBEDDING_DIM
    widened = []
    for k in sd_b:
        if sd_b[k].shape != sd_v[k].shape:
            widened.append((k, tuple(sd_b[k].shape), tuple(sd_v[k].shape)))
    for k, sb, sv in widened:
        assert 'value' in k and k.endswith('fc.weight'), f"unexpected shape change: {k} {sb}->{sv}"
        assert sv[1] - sb[1] == 2 * emb, f"{k}: widened by {sv[1]-sb[1]}, expected {2*emb}"
    names = [k for k, _, _ in widened]
    assert any('value_head' in k for k in names), "value_head.fc not widened"
    print(f"PASS: shape accounting ({len(widened)} widened first-linears: {names})")

    # --- copy shared weights; for widened fc, copy the original columns so the
    # baseline function is embedded in the vpc net (new columns keep their init).
    dst = dict(net_vpc.state_dict())
    for k, v in sd_b.items():
        if dst[k].shape == v.shape:
            dst[k].copy_(v)
        else:
            dst[k][:, :v.shape[1]].copy_(v)
    net_vpc.load_state_dict(dst, strict=False)

    torch.manual_seed(0)
    squares = torch.randn(2, NUM_TOKENS_INPUT, TOTAL_INPUT_FEATURES_PER_SQUARE)
    with torch.no_grad():
        out_b = net_base(squares, None)
        out_v = net_vpc(squares, None)

    # out[0]=policy, out[1]=value (project convention followed by sister smokes)
    assert torch.equal(out_b[0], out_v[0]), "ISOLATION BROKEN: policy differs"
    print("PASS: isolation (policy bit-identical with shared weights)")
    assert all(torch.isfinite(o).all() for o in out_v if torch.is_tensor(o)), "non-finite output"
    assert not torch.equal(out_b[1], out_v[1]), \
        "value identical to baseline — channels carry no signal (mechanism inert)"
    print("PASS: activity (value differs from step 0, finite)")

    # --- gradient flow ---
    net_vpc.train()
    net_vpc.zero_grad(set_to_none=True)
    out = net_vpc(squares, None)
    loss = out[1].float().pow(2).mean()
    loss.backward()
    fc_g = None
    for n, p in net_vpc.named_parameters():
        if 'value_head.fc.weight' in n:
            fc_g = p.grad
    assert fc_g is not None and fc_g[:, -2 * emb:].abs().sum() > 0, \
        "no gradient reached the new pool columns"
    got_qkv = any(n.endswith('attention.qkv.weight') and p.grad is not None
                  and p.grad.abs().sum() > 0 for n, p in net_vpc.named_parameters())
    assert got_qkv, "no gradient flowed through the pool into the trunk"
    print("PASS: gradients reach pool columns and trunk")

    # --- export parity ---
    net_vpc.eval()
    ep = torch.export.export(net_vpc, (squares, None))
    with torch.no_grad():
        out_e = ep.module()(squares, None)
    assert torch.allclose(out_e[0], out_v[0], atol=1e-5), "export/eager policy parity broken"
    assert torch.allclose(out_e[1], out_v[1], atol=1e-5), "export/eager value parity broken"
    print("PASS: torch.export capture + eager parity")

    print("ALL VALUE-POOL-CHANNELS SMOKE CHECKS PASSED.")


if __name__ == '__main__':
    main()
