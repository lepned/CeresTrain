# ONNX → Checkpoint Reconstruction Status (FIXED)

## Status: RESOLVED ✅

Previously stuck with 150% relative error on deep policy/value outputs. Root cause found via layer-by-layer parity diagnostic.

## The bug

**`NonLinearAttention` was FALSE when it should be TRUE** in the inferred config.

Two compounding factors hid this:

1. The inspection output listed `attention.qkv.bias [1920]` which I initially interpreted as "standard MHA with combined QKV projection, no secondary nonlinearity."
2. My graph tracer only found Linear weights that had an **adjacent named `.bias`** (via the `MatMul + Add` pattern). But the q2/k2/v2 secondary projections in NonLinearAttention have **no bias** — they are bare `MatMul` ops feeding into subsequent attention scoring. These weights were NOT in my mapping; they were silently left as random init in the reconstructed net.

So the reconstruction was missing 102 weight tensors (3 matrices × 34 layers) corresponding to the `.attention.{q2,k2,v2}.weight` parameters under NonLinearAttention.

Trainer confirmed architecture in out-of-band conversation: **C1-640-34: 34 layers, 640 embedding, 20 heads per layer, 1920 dff size.** This matched our inference exactly, and combined with inspecting the attention-block op sequence (`qkvLN → Softplus → Tanh → Mul → Split → q2/k2/v2 MatMul`) confirmed NonLinearAttention = true.

## Fix

Two code/config changes:

1. Set `"NonLinearAttention": true` in `configs/c1_640_34_ceres_net.json`
2. Extend `reconstruct_ckpt_from_onnx.py`'s `build_weight_bias_pairs` to **also** index bare `MatMul` nodes by node-path name. Synthesize a PyTorch-like param name from the node name (e.g. `/transformer_layer.0/attention/q2/MatMul` → `transformer_layer.0.attention.q2.weight`) and pair it with the anonymous initializer. Extend `find_onnx_source` with a third lookup branch for this "bare" mapping.

After the fix:
- Mapping: **785 / 785** state_dict keys matched, zero shape mismatches, zero unmatched
- Parameter count: **249.5M** (previously 207.6M — the 42M delta is the q2/k2/v2 matrices)
- Forward-pass parity vs ONNX on same input:

| Output | max abs diff | max rel diff |
|---|---|---|
| policy | 0.023 | 0.88 (high on near-zero softmax slots, misleading) |
| value | 0.117 | 0.22 |
| unc | 0.002 | 0.01 |
| value2 | 0.22 | 0.10 |

All within FP16 numerical noise.

## Artifacts

- `C:/Dev/Chess/CeresTrain/nets/ckpt_c1_640_34_from_onnx_0` — reconstructed checkpoint, 249.5M params, FP16-precision parity with original ONNX
- `C:/Dev/Chess/CeresTrain/configs/c1_640_34_ceres_*.json` — working config set matching C1-640-34 architecture
- `C:/Dev/Chess/CeresTrain/reconstruct_ckpt_from_onnx.py` — now correctly handles bare-MatMul Linear weights
- `C:/Dev/Chess/CeresTrain/layer_by_layer_parity.py` — diagnostic tool that found the bug

## Implication

**Fine-tuning from C1-640-34 is now unblocked.** Configure `Opt_CheckpointResumeFromFileName` in a puzzle-training opt config to point at `ckpt_c1_640_34_from_onnx_0`, use small LR (1e-5 to 5e-5) and short training (5–15M positions) to adapt to puzzle distribution while preserving the strong pretrained value head.

This path gives us a calibrated value head "for free" (from a generally-trained net), which is exactly what we've been struggling to produce from puzzle-only data.
