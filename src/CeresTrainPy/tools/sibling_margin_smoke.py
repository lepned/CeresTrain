"""Sibling-margin policy-loss smoke test (tactics toolbox T1.4).

Pure loss-side knob (CERES_POLICY_SIBLING_MARGIN_WEIGHT): contracts are
loss-value contracts, mirroring onlymove_smoke.py:
  - off-identity: weight=0 returns EXACTLY the legacy mean-CE loss.
  - satisfied-margin no-op: when every target logit dominates its best wrong
    sibling by > margin, the hinge is 0 and the loss equals plain CE.
  - violated-margin direction: forced-and-wrong batch adds a positive term,
    and the gradient pushes the target logit UP and the best-wrong logit DOWN.
  - gap gating: near-uniform (quiet) targets contribute ~nothing even when
    their margin is violated.
  - illegal-move exclusion: a huge raw logit on an ILLEGAL move never becomes
    the best-wrong sibling (masking happens before the hinge).
  - logging separation: PENDING_POLICY_LOSS accumulates pure CE (unchanged by
    the margin term) so TRAIN lines stay cross-run comparable.

Run from CeresTrainPy:  python tools/sibling_margin_smoke.py
"""
import sys
import os
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import losses
from losses import LossCalculator


def _mk_sharp_targets(B, n=32):
    t = torch.zeros(B, 1858)
    for i in range(B):
        idx = torch.randperm(1858)[:n]
        probs = torch.full((n,), 0.03 / (n - 1))
        probs[0] = 0.97
        t[i, idx] = probs
    return t


def main():
    torch.manual_seed(11)
    B = 128
    calc = LossCalculator(None)
    losses.POLICY_SIBLING_MARGIN = 1.0

    t = _mk_sharp_targets(B)
    tgt_idx = t.argmax(dim=1)

    # --- off-identity ---
    out = torch.randn(B, 1858)
    losses.POLICY_SIBLING_MARGIN_WEIGHT = 0.0
    base = calc.policy_loss(t, out, False, False, 1.0)
    legal = t.greater(0)
    masked = torch.where(legal, out, torch.zeros_like(out).add_(calc.MASK_POLICY_VALUE))
    ref = F.cross_entropy(masked, t)
    assert torch.allclose(base, ref, atol=1e-6), "weight=0 path deviates from legacy CE"
    print("PASS: off-identity")

    # --- satisfied-margin no-op ---
    out_sat = torch.randn(B, 1858) * 0.1
    out_sat.scatter_(1, tgt_idx.unsqueeze(1), 10.0)  # target dominates by >> 1 nat
    losses.POLICY_SIBLING_MARGIN_WEIGHT = 0.5
    l_m = calc.policy_loss(t, out_sat, False, False, 1.0)
    losses.POLICY_SIBLING_MARGIN_WEIGHT = 0.0
    l_p = calc.policy_loss(t, out_sat, False, False, 1.0)
    assert torch.allclose(l_m, l_p, atol=1e-5), \
        f"satisfied margin must add nothing ({l_m.item():.6f} vs {l_p.item():.6f})"
    print("PASS: satisfied-margin no-op")

    # --- violated-margin direction + gradient signs ---
    out_v = (torch.randn(B, 1858) * 0.1).requires_grad_(True)
    losses.POLICY_SIBLING_MARGIN_WEIGHT = 0.5
    l_v = calc.policy_loss(t, out_v, False, False, 1.0)
    losses.POLICY_SIBLING_MARGIN_WEIGHT = 0.0
    l_v0 = calc.policy_loss(t, out_v, False, False, 1.0)
    assert l_v.item() > l_v0.item() + 1e-4, "violated margin must add a positive term"
    losses.POLICY_SIBLING_MARGIN_WEIGHT = 0.5
    l_v2 = calc.policy_loss(t, out_v, False, False, 1.0)
    l_v2.backward()
    g_t = out_v.grad.gather(1, tgt_idx.unsqueeze(1)).squeeze(1)
    assert (g_t < 0).float().mean() > 0.95, "gradient must push target logits up"
    print(f"PASS: violated-margin direction (+{l_v.item()-l_v0.item():.4f}) and gradient sign")

    # --- gap gating: quiet targets contribute ~nothing ---
    n = 32
    t_quiet = torch.zeros(B, 1858)
    for i in range(B):
        idx = torch.randperm(1858)[:n]
        probs = torch.rand(n) * 0.02 + 1.0
        t_quiet[i, idx] = probs / probs.sum()
    out_q = torch.randn(B, 1858) * 0.1  # margin certainly violated
    losses.POLICY_SIBLING_MARGIN_WEIGHT = 0.5
    l_q = calc.policy_loss(t_quiet, out_q, False, False, 1.0)
    losses.POLICY_SIBLING_MARGIN_WEIGHT = 0.0
    l_q0 = calc.policy_loss(t_quiet, out_q, False, False, 1.0)
    added = l_q.item() - l_q0.item()
    assert abs(added) < 0.02, f"quiet targets must be gap-gated to ~0, added {added:.4f}"
    print(f"PASS: gap gating (quiet batch adds only {added:.5f})")

    # --- illegal-move exclusion ---
    out_il = torch.randn(B, 1858) * 0.1
    illegal_idx = []
    for i in range(B):
        cand = (t[i] == 0).nonzero().squeeze(1)
        illegal_idx.append(cand[0].item())
    illegal_idx = torch.tensor(illegal_idx)
    out_il.scatter_(1, illegal_idx.unsqueeze(1), 50.0)  # huge logit on an ILLEGAL move
    out_il.scatter_(1, tgt_idx.unsqueeze(1), 10.0)      # target dominates all LEGAL moves
    losses.POLICY_SIBLING_MARGIN_WEIGHT = 0.5
    l_il = calc.policy_loss(t, out_il, False, False, 1.0)
    losses.POLICY_SIBLING_MARGIN_WEIGHT = 0.0
    l_il0 = calc.policy_loss(t, out_il, False, False, 1.0)
    assert torch.allclose(l_il, l_il0, atol=1e-5), \
        "illegal move leaked into the best-wrong sibling"
    print("PASS: illegal-move exclusion")

    # --- logging separation ---
    calc.reset_counters()
    losses.POLICY_SIBLING_MARGIN_WEIGHT = 0.5
    _ = calc.policy_loss(t, out_v.detach(), False, False, 1.0)
    logged_w = calc.PENDING_POLICY_LOSS
    calc.reset_counters()
    losses.POLICY_SIBLING_MARGIN_WEIGHT = 0.0
    _ = calc.policy_loss(t, out_v.detach(), False, False, 1.0)
    logged_p = calc.PENDING_POLICY_LOSS
    assert abs(logged_w - logged_p) < 1e-6, \
        f"logged policy loss must stay pure CE ({logged_w:.6f} vs {logged_p:.6f})"
    print("PASS: logged TRAIN policy loss unchanged (pure CE)")

    print("ALL SIBLING-MARGIN SMOKE CHECKS PASSED.")


if __name__ == '__main__':
    main()
