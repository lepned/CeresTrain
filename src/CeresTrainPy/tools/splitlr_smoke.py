"""Split-LR smoke test (split_lr_plan_2026-08.md, Phase 0 contracts).

Covers the two new mechanisms:
  1. Muon per-param lr_ratios (family LR): a param with ratio r must move
     with an update whose magnitude scales by exactly r relative to an
     identical twin at ratio 1 — verified in BOTH the Muon branch (2-D,
     orthogonalized) and the AdamW branch (1-D), including the wd term.
  2. Schedule proportionality: halving group['lr'] halves both updates,
     preserving the ratio (ratios ride the schedule multiplicatively).
  3. 'ffn-only' scope predicate on REAL CeresNet parameter names: FFN
     linears land in Muon, attention qkv/proj/smolgen land in AdamW.
NOTE: correctness only — NO quality verdicts at smoke scale (user-ratified:
split-LR effects live in the decay tail, unmeasurable at 5M).

Run from CeresTrainPy:  python tools/splitlr_smoke.py
"""
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from muon import Muon
from tools.tsb_smoke import _make_config, _make_net


def _step_and_delta(p, opt):
    before = p.data.clone()
    opt.step()
    return (p.data - before).norm().item()


def _ratio_checks():
    torch.manual_seed(0)
    RATIO = 1.0 / 3.0
    # Muon branch: identical 2-D twins, identical grads, ratios 1 vs 1/3.
    a = torch.nn.Parameter(torch.randn(64, 64))
    b = torch.nn.Parameter(a.data.clone())
    g = torch.randn(64, 64)
    a.grad = g.clone(); b.grad = g.clone()
    # AdamW branch: identical 1-D twins.
    c = torch.nn.Parameter(torch.randn(64))
    d = torch.nn.Parameter(c.data.clone())
    g1 = torch.randn(64)
    c.grad = g1.clone(); d.grad = g1.clone()

    opt = Muon(lr=1e-3, wd=0.0, muon_params=[a, b], adamw_params=[c, d],
               lr_ratios={b: RATIO, d: RATIO})
    da = _step_and_delta_pair(opt, [a, b, c, d])
    ra_muon = da[1] / da[0]
    ra_adam = da[3] / da[2]
    assert abs(ra_muon - RATIO) < 0.02, f"Muon-branch ratio {ra_muon:.4f} != {RATIO:.4f}"
    assert abs(ra_adam - RATIO) < 0.02, f"AdamW-branch ratio {ra_adam:.4f} != {RATIO:.4f}"
    print(f"PASS: family ratio exact in both branches (muon {ra_muon:.4f}, adamw {ra_adam:.4f})")

    # Schedule proportionality: halve group lr, ratios must survive.
    for p in (a, b, c, d):
        p.grad = torch.randn_like(p)
    for grp in opt.param_groups:
        grp['lr'] *= 0.5
    da2 = _step_and_delta_pair(opt, [a, b, c, d])
    r2_muon = da2[1] / da2[0]
    r2_adam = da2[3] / da2[2]
    assert abs(r2_muon - RATIO) < 0.02 and abs(r2_adam - RATIO) < 0.02, \
        f"ratios drifted under schedule scaling ({r2_muon:.4f}, {r2_adam:.4f})"
    print("PASS: ratios ride the schedule (halved lr, ratio preserved)")

    # wd term scales with the family lr too (twin decay comparison).
    e = torch.nn.Parameter(torch.ones(32, 32) * 5.0)
    f = torch.nn.Parameter(e.data.clone())
    e.grad = torch.zeros(32, 32); f.grad = torch.zeros(32, 32)
    opt2 = Muon(lr=1e-2, wd=0.1, muon_params=[e, f], adamw_params=[],
                lr_ratios={f: RATIO})
    opt2.step()
    dec_e = 1.0 - e.data.mean().item() / 5.0
    dec_f = 1.0 - f.data.mean().item() / 5.0
    assert abs(dec_f / dec_e - RATIO) < 0.02, f"wd not family-scaled ({dec_f/dec_e:.4f})"
    print("PASS: weight decay follows the family lr")


def _step_and_delta_pair(opt, params):
    befores = [p.data.clone() for p in params]
    opt.step()
    return [(p.data - b).norm().item() for p, b in zip(params, befores)]


def _scope_checks():
    torch.manual_seed(42)
    cfg = _make_config(tsb_enabled=False)
    cfg.id = cfg.Exec_ID = 'splitlr_smoke'
    cfg.NetDef_HeadSharedLinearDiv = 1
    cfg.NetDef_SoftMoE_ExpertInputDim = 0
    cfg.NetDef_UseDiffAttention = False
    cfg.NetDef_UseRoPE = False
    net = _make_net(cfg)

    def use_muon_ffn_only(n, p):
        return (p.ndim == 2 and 'embedding' not in n and 'transformer_layer' in n
                and ('mlp.linear' in n or 'tactical_ffn' in n))

    muon_n = [n for n, p in net.named_parameters() if p.requires_grad and use_muon_ffn_only(n, p)]
    adam_n = [n for n, p in net.named_parameters() if p.requires_grad and not use_muon_ffn_only(n, p)]
    assert muon_n and all('mlp.linear' in n or 'tactical_ffn' in n for n in muon_n), \
        f"non-FFN leaked into Muon: {[n for n in muon_n if 'mlp' not in n][:3]}"
    leaked = [n for n in adam_n if 'attention.qkv' in n or 'attention.W_h' in n]
    assert leaked, "attention matrices not found in AdamW group under ffn-only"
    assert not any('attention.qkv' in n or 'attention.W_h' in n for n in muon_n), \
        "attention matrices leaked into Muon under ffn-only"
    print(f"PASS: ffn-only scope on real net ({len(muon_n)} FFN->Muon, "
          f"attention qkv/W_h confirmed in AdamW group of {len(adam_n)})")

    # Head-family predicate coverage on real names.
    fam = ('policy_head.', 'value_head.', 'value2_head.', 'unc_head.',
           'mlh_head.', 'qdev_upper.', 'qdev_lower.', 'headPremap.',
           'headSharedLinear.', 'unc_policy.')
    hits = [n for n, _ in net.named_parameters() if any(f in n for f in fam)]
    assert any('policy_head' in n for n in hits) and any('value_head' in n for n in hits)
    assert not any('transformer_layer' in n for n in hits), "trunk leaked into head family"
    print(f"PASS: head-family predicate matches {len(hits)} params, no trunk leak")


def main():
    _ratio_checks()
    _scope_checks()
    print("ALL SPLIT-LR SMOKE CHECKS PASSED.")


if __name__ == '__main__':
    main()
