"""Unit test for PiSSA initialization in lora.py.

Critical invariant: after apply_pissa(), the LoRALinear forward output must
equal the original (unmodified) base layer's output to within numerical
precision. If this fails, training starts from a damaged state and any
positive results are meaningless.

Tests run on CPU with small dimensions; intended to be runnable while a
GPU training job is in flight.
"""
import sys
import os
import math
import torch
import torch.nn as nn

# Make the test runnable from anywhere; add the CeresTrainPy dir to sys.path.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import lora


def _make_layer(out_features, in_features, seed=0):
    """Create an nn.Linear with deterministic non-trivial weights."""
    torch.manual_seed(seed)
    layer = nn.Linear(in_features, out_features, bias=True)
    # Make weights distinct in magnitude so SVD is non-degenerate.
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight) * 0.5)
        layer.bias.copy_(torch.randn_like(layer.bias) * 0.1)
    return layer


def _max_abs_diff(a, b):
    return (a - b).abs().max().item()


def test_identity_at_init(out_dim=64, in_dim=128, rank_divisor=4, atol=1e-4):
    """After apply_pissa, output(x) must equal base_layer(x) within tolerance."""
    layer = _make_layer(out_dim, in_dim, seed=42)
    base_W = layer.weight.detach().clone()
    base_b = layer.bias.detach().clone()
    x = torch.randn(8, in_dim)
    expected = torch.nn.functional.linear(x, base_W, base_b)

    lora_layer = lora.LoRALinear(layer, rank_divisor=rank_divisor, enable_lora=True)
    # Sanity: vanilla init has lora_B = 0, so output = base
    out_before = lora_layer(x)
    diff_before = _max_abs_diff(out_before, expected)
    assert diff_before < 1e-6, f"vanilla init not identity: diff={diff_before}"

    # Apply PiSSA — must remain identity (within FP32 SVD precision)
    lora_layer.apply_pissa()
    out_after = lora_layer(x)
    diff_after = _max_abs_diff(out_after, expected)
    print(f"[identity] rank_divisor={rank_divisor} rank={lora_layer.rank}  "
          f"vanilla diff={diff_before:.2e}  pissa diff={diff_after:.2e}  (atol={atol})")
    assert diff_after < atol, f"PiSSA broke identity: diff={diff_after} > {atol}"


def test_post_pissa_invariants(out_dim=64, in_dim=128, rank_divisor=4):
    """After apply_pissa, lora_B is non-zero, alpha = sqrt(rank), base is modified."""
    layer = _make_layer(out_dim, in_dim, seed=43)
    base_W_before = layer.weight.detach().clone()
    lora_layer = lora.LoRALinear(layer, rank_divisor=rank_divisor, enable_lora=True)

    # Pre-PiSSA: lora_B is all zeros (vanilla init)
    assert lora_layer.lora_B.abs().max().item() < 1e-9, "lora_B should be zero before PiSSA"

    lora_layer.apply_pissa()

    # Post-PiSSA: lora_B is non-zero (PiSSA puts SVD components into it)
    b_max = lora_layer.lora_B.abs().max().item()
    assert b_max > 1e-3, f"lora_B should be non-zero after PiSSA, got max abs {b_max}"

    # Alpha = sqrt(rank)
    expected_alpha = math.sqrt(float(lora_layer.rank))
    actual_alpha = lora_layer.lora_alpha.item()
    assert abs(actual_alpha - expected_alpha) < 1e-5, \
        f"alpha mismatch: expected sqrt(rank)={expected_alpha}, got {actual_alpha}"

    # Base weight was modified (top-r approximation subtracted)
    base_diff = _max_abs_diff(layer.weight.detach(), base_W_before)
    assert base_diff > 1e-3, "base weight should have been modified by PiSSA"

    print(f"[invariants] rank={lora_layer.rank}  "
          f"|lora_B|max={b_max:.3f}  alpha={actual_alpha:.3f}  base|Δ|max={base_diff:.3f}")


def test_apply_pissa_to_model():
    """apply_pissa_to_model walks the model and PiSSA-initializes every LoRALinear."""
    class MultiLoRA(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = lora.LoRALinear(nn.Linear(64, 32), rank_divisor=4, enable_lora=True)
            self.fc2 = lora.LoRALinear(nn.Linear(32, 16), rank_divisor=4, enable_lora=True)
            self.bare = nn.Linear(16, 8)  # not LoRA — should be skipped
            self.fc3 = lora.LoRALinear(nn.Linear(8, 4), rank_divisor=2, enable_lora=False)  # disabled — skip
            self.fc4 = lora.LoRALinear(nn.Linear(4, 2), rank_divisor=2, enable_lora=True)

    torch.manual_seed(0)
    m = MultiLoRA()
    # Capture each enabled LoRALinear's input-output mapping pre-init for identity check
    test_inputs = {
        'fc1': torch.randn(2, 64),
        'fc2': torch.randn(2, 32),
        'fc4': torch.randn(2, 4),
    }
    expected_outputs = {k: getattr(m, k)(v) for k, v in test_inputs.items()}

    lora.apply_pissa_to_model(m)

    # All three enabled LoRALinear layers must remain identity
    for k, x in test_inputs.items():
        out = getattr(m, k)(x)
        diff = _max_abs_diff(out, expected_outputs[k])
        assert diff < 1e-3, f"{k}: identity broken (diff={diff:.2e})"

    # The disabled LoRA layer (fc3) must not have PiSSA applied
    assert m.fc3.lora_A is None, "fc3 should remain disabled (no LoRA params)"

    print("[apply_pissa_to_model] OK — 3 enabled layers PiSSA-initialized, 1 disabled skipped")


def test_min_rank_clamp():
    """When rank_divisor produces rank < MIN_RANK=4, code clamps to 4. PiSSA must still work."""
    # in_features=8, rank_divisor=8 → rank = max(MIN_RANK=4, 8//8=1) = 4
    layer = _make_layer(out_features=16, in_features=8, seed=99)
    x = torch.randn(2, 8)
    expected = layer(x).detach().clone()
    lora_layer = lora.LoRALinear(layer, rank_divisor=8, enable_lora=True)
    assert lora_layer.rank == 4, f"expected MIN_RANK=4 clamp, got rank={lora_layer.rank}"
    lora_layer.apply_pissa()
    out = lora_layer(x)
    diff = _max_abs_diff(out, expected)
    print(f"[min-rank] rank={lora_layer.rank}  diff={diff:.2e}")
    assert diff < 1e-3


def test_rectangular_layer():
    """Layer where out_features != in_features (typical for transformer FFN/heads)."""
    # FFN-shaped: in=640, out=1920 (matches CeresNet model)
    layer = _make_layer(out_features=1920, in_features=640, seed=7)
    x = torch.randn(4, 640)
    expected = layer(x).detach().clone()
    lora_layer = lora.LoRALinear(layer, rank_divisor=16, enable_lora=True)
    assert lora_layer.rank == 40, f"expected rank=40, got {lora_layer.rank}"
    lora_layer.apply_pissa()
    out = lora_layer(x)
    diff = _max_abs_diff(out, expected)
    print(f"[rectangular] in=640 out=1920 rank={lora_layer.rank}  diff={diff:.2e}")
    assert diff < 1e-2  # slightly looser due to bigger matrix multiply rounding


def test_dtype_preservation():
    """Original dtype is preserved after PiSSA (e.g., bfloat16)."""
    layer = nn.Linear(64, 128, bias=True)
    layer = layer.to(torch.bfloat16)
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))
        layer.bias.copy_(torch.randn_like(layer.bias))
    lora_layer = lora.LoRALinear(layer, rank_divisor=4, enable_lora=True)
    # Note: vanilla LoRA init creates lora_A/B as FP32 (inherits torch.zeros default).
    # PiSSA preserves the base layer's dtype on the base weight.
    lora_layer.apply_pissa()
    assert layer.weight.dtype == torch.bfloat16, "base weight dtype should be preserved"
    print(f"[dtype] base.weight stayed {layer.weight.dtype}; "
          f"lora_A.dtype={lora_layer.lora_A.dtype}, lora_B.dtype={lora_layer.lora_B.dtype}")


if __name__ == "__main__":
    print(f"=== PiSSA unit tests (LORA_USE_PISSA={lora.LORA_USE_PISSA}) ===")
    print(f"=== torch {torch.__version__} ===\n")

    test_identity_at_init(rank_divisor=4)
    test_identity_at_init(rank_divisor=8)
    test_identity_at_init(rank_divisor=16)
    test_post_pissa_invariants()
    test_min_rank_clamp()
    test_rectangular_layer()
    test_dtype_preservation()
    test_apply_pissa_to_model()

    print("\nAll PiSSA unit tests PASSED.")
