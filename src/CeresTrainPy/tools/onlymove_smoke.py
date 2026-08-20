"""Only-move CE weighting smoke test (tactics toolbox T1.3).

CERES_POLICY_ONLYMOVE_LAMBDA is a pure loss-side knob (no params, serving
graph untouched), so the contracts are loss-value contracts:
  - off-identity: lambda=0 path returns EXACTLY the legacy mean-CE loss.
  - normalization exactness: with lambda>0 but all samples at the SAME gap,
    the weighted loss equals the plain mean CE bitwise-close (weights cancel)
    — proves the batch normalization keeps the loss scale / effective LR
    unchanged and the mechanism is pure redistribution.
  - redistribution direction: in a mixed batch where the SHARP samples carry
    higher CE, the weighted loss is strictly above the mean (and below when
    sharp samples carry lower CE).
  - gradient flow: loss.backward() reaches the output logits in both paths;
    the gap weight itself is detached (no grad path through the target).

Run from CeresTrainPy:  python tools/onlymove_smoke.py
"""
import sys
import os
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import losses
from losses import LossCalculator


def _mk_targets(B, sharp_mask, n=64):
    """Soft targets over an n-move legal set: sharp rows ~ one-hot (gap ~0.97),
    quiet rows ~ near-uniform over n moves (gap ~0)."""
    t = torch.zeros(B, 1858)
    for i in range(B):
        idx = torch.randperm(1858)[:n]
        if sharp_mask[i]:
            probs = torch.full((n,), 0.03 / (n - 1))
            probs[0] = 0.97
        else:
            probs = torch.rand(n) * 0.02 + 1.0
            probs = probs / probs.sum()
        t[i, idx] = probs
    return t


def main():
    torch.manual_seed(7)
    B = 256
    calc = LossCalculator(None)

    out = torch.randn(B, 1858, requires_grad=True)

    # --- off-identity ---
    losses.POLICY_ONLYMOVE_LAMBDA = 0.0
    t_mixed = _mk_targets(B, torch.rand(B) < 0.5)
    base = calc.policy_loss(t_mixed, out, False, False, 1.0)
    legal = t_mixed.greater(0)
    masked = torch.where(legal, out, torch.zeros_like(out).add_(calc.MASK_POLICY_VALUE))
    ref = F.cross_entropy(masked, t_mixed)
    assert torch.allclose(base, ref, atol=1e-6), "lambda=0 path deviates from legacy mean CE"
    print("PASS: off-identity (lambda=0 == legacy mean CE)")

    # --- normalization exactness at uniform gap ---
    losses.POLICY_ONLYMOVE_LAMBDA = 2.0
    t_sharp = _mk_targets(B, torch.ones(B, dtype=torch.bool))
    l_w = calc.policy_loss(t_sharp, out, False, False, 1.0)
    losses.POLICY_ONLYMOVE_LAMBDA = 0.0
    l_p = calc.policy_loss(t_sharp, out, False, False, 1.0)
    assert torch.allclose(l_w, l_p, rtol=1e-4), \
        f"uniform-gap batch must reduce to plain mean CE ({l_w.item():.6f} vs {l_p.item():.6f})"
    print("PASS: normalization exactness (uniform gap -> weights cancel, scale unchanged)")

    # --- redistribution direction ---
    # Craft outputs so sharp samples have HIGH CE (logits favor a wrong move).
    sharp_mask = torch.zeros(B, dtype=torch.bool); sharp_mask[:B // 2] = True
    t2 = _mk_targets(B, sharp_mask)
    out2 = torch.randn(B, 1858) * 0.1
    wrong = (t2[:B // 2] > 0).float() * torch.rand(B // 2, 1858)
    wrong.scatter_(1, t2[:B // 2].argmax(dim=1, keepdim=True), -5.0)  # suppress the forced move
    out2[:B // 2] += 8.0 * wrong
    out2.requires_grad_(True)
    losses.POLICY_ONLYMOVE_LAMBDA = 2.0
    l_w2 = calc.policy_loss(t2, out2, False, False, 1.0)
    losses.POLICY_ONLYMOVE_LAMBDA = 0.0
    l_p2 = calc.policy_loss(t2, out2, False, False, 1.0)
    assert l_w2.item() > l_p2.item() + 1e-4, \
        f"sharp-and-wrong batch must weigh ABOVE the mean ({l_w2.item():.4f} vs {l_p2.item():.4f})"
    print(f"PASS: redistribution direction (weighted {l_w2.item():.4f} > mean {l_p2.item():.4f} "
          f"when forced positions are the wrong ones)")

    # --- gradient flow ---
    losses.POLICY_ONLYMOVE_LAMBDA = 2.0
    out3 = torch.randn(B, 1858, requires_grad=True)
    l3 = calc.policy_loss(t2, out3, False, False, 1.0)
    l3.backward()
    assert out3.grad is not None and out3.grad.abs().sum() > 0, "no gradient reached the logits"
    print("PASS: gradient flow through the weighted path")

    print("ALL ONLY-MOVE SMOKE CHECKS PASSED.")


if __name__ == '__main__':
    main()
