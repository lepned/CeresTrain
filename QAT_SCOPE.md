# QAT Scope — closing the INT8 PTQ quality gap on C1-640-34

## Goal
PTQ ceiling (strongly-typed TRT, 99.999/64 percentile calib) is **−0.6pp policy / −0.8pp value**
vs FP16, giving "≈equal" tournament play at **+45% NPS**. QAT's job: recover that gap so the
speed becomes a strict strength win (puzzle ranking ≥ FP16 → tournament win, per the same-net
precision rule).

We are NOT chasing a new architecture — we fine-tune the *existing* flagship weights so they are
robust to the exact INT8 quantization we deploy.

## Why this is cheap and safe here (the key insight)
Do **distillation-based QAT**, not from-scratch training:
- **Student** = the flagship net with INT8 fake-quant inserted, initialized from the real weights.
- **Teacher** = the same flagship at full precision, frozen.
- Fine-tune the student on the TPG corpus so it reproduces the teacher's policy/value outputs
  *despite* the quantization rounding noise.

The teacher/student (KL-anchor) machinery **already exists** in `train.py:740-795`
(`Opt_KLAnchorRefCheckpoint`, `Opt_KLAnchorPolicyWeight`, `Opt_KLAnchorValueWeight`). This is the
same mechanism the LoRA KL05/10/30 work used. The net is already good → this is a short polish
fine-tune, not a training run.

## Confirmed enabling facts
- All quantizable ops are `nn.Linear` (ceres_net 6, mlp2_layer 10, dot_product_attention 13) →
  a fake-quant wrapper hooks every matmul automatically. No per-op surgery.
- Loadable flagship checkpoint exists: `C:/Dev/Chess/CeresTrain/nets/ckpt_c1_640_34_from_onnx_0`
  (785 tensors, 640×34, `{model, optimizer, num_pos}`, strict-load verified). Serves as BOTH
  student init and teacher.
- `train.py` resume path (622-672) load_state_dict(strict=True) + optimizer + step; has the
  aux-feature width guard. KL-anchor teacher loads via `torch.load` strict=False (795).
- Deployment is already solved: `qdq_export.py` (per-channel symmetric INT8 weights, per-tensor
  symmetric INT8 activations, percentile calib) → strongly-typed TRT build in
  `TensorRTWrapper.cpp`. QAT only changes *which weights* we export; the export+deploy path is
  unchanged.
- A dormant FP8/TransformerEngine QAT path exists in `mlp2_layer.py` (`use_te`) — evidence the
  trainer already tolerates quant-aware modules, but it's FP8-only/FFN-only, not reused here.

## Tooling decision (NEEDED — neither NVIDIA lib is installed)
`modelopt` and `pytorch_quantization` are absent; only `torch.ao.quantization` is present.

**Option A — Manual fake-quant module (RECOMMENDED).**
~120 lines: a `FakeQuantLinear` (or forward-pre-hook on `nn.Linear`) implementing the *exact*
qdq_export scheme — per-channel symmetric weight quant + per-tensor symmetric activation quant
with straight-through estimator. Seed activation ranges from the SAME percentile calibration we
already run, so QAT *starts exactly at the deployed PTQ operating point* and can only improve.
- Pros: zero new heavy deps; bit-exact match to deployment (no train/deploy skew); we already own
  and understand the math (proven in `surgery_activations.py`); trivial ONNX export (weights are
  just better — reuse `qdq_export.py` as-is).
- Cons: we write/maintain the fake-quant + calibration seeding (~half a day).

**Option B — `pip install nvidia-modelopt[torch]`.**
`mtq.quantize(model, INT8 cfg, forward_loop)` auto-inserts QDQ, calibrates, QAT-finetunes, exports
QDQ ONNX aligned to TRT.
- Pros: batteries-included, TRT-validated export.
- Cons: new heavy dep + install/version risk against torch 2.11/cu128; its default QDQ scheme may
  not match our per-tensor-activation / percentile choices → reintroduces the exact skew we just
  spent days eliminating; less control over the policy-vs-value opposite-percentile tension.

→ Recommend **A**. It reuses the entire pipeline we already validated and removes train/deploy skew
by construction. B is the fallback if manual STE proves fiddly.

## Implementation plan (Option A)
1. **`fake_quant.py`** (new): `FakeQuantLinear` wrapping `nn.Linear`.
   - Weights: per-output-channel symmetric INT8, scale = max|W_c|/127, STE round.
   - Activations: per-tensor symmetric INT8, range = calibrated (percentile 99.999), STE round,
     range registered as a buffer (frozen after calibration, or a learnable LSQ-style step — start
     frozen).
   - A module-swap helper that walks the model and replaces every `nn.Linear` (skip the final
     logits layer if puzzle shows head-sensitivity — the policy/value heads share the trunk, so
     keep heads quantized but watch them).
2. **Calibration pass**: run ~64 TPG positions, set each activation range from the percentile —
   identical to PTQ. This makes step-0 of QAT == current PTQ net (sanity: eval before any grad
   step should reproduce pol78.75/val90.85).
3. **train.py hook**: behind a `config.Opt_QAT` flag, after model build + resume, swap in
   fake-quant + calibrate. Keep everything else (WSD schedule, optimizer resume) intact.
4. **Distillation fine-tune**: set `Opt_KLAnchorRefCheckpoint = from_onnx_0`,
   `Opt_KLAnchorPolicyWeight`/`ValueWeight` > 0 (teacher = FP16). Loss = small task loss + KL to
   teacher. Low LR (≈ MIN_LR floor, ~1e-5..5e-5, no warmup spike — we're polishing). Short:
   **20–50M positions** (the net is already trained; we only adapt to quant noise).
5. **Export**: dump the QAT'd weights, run existing `qdq_export.py --method percentile
   --percentile 99.999 --byte_divisor 100` → strongly-typed TRT. No pipeline change.
6. **Validate**: winning-FEN sanity (value not collapsed) → rg2340 + 3-band EB puzzle suite.
   Success = pol ≥ 79.0 / val ≥ 90.85 (≥ FP16 floor) at +45% NPS. Per the same-net rule, that's a
   tournament win. Optional tournament confirm.

## Effort / compute
- Code: ~0.5–1 day (fake-quant module + train.py flag + calibration seeding).
- Compute: 20–50M-position distill fine-tune at b=4096 on the prod box — hours, not days
  (cf. full runs are 3–4B). Dev-box 5090 smoke at 5–10M first to prove the loop converges and
  step-0 reproduces PTQ.
- Iteration: the main knob is which layers stay quantized + activation-range learnability
  (frozen vs LSQ). Start all-quant + frozen ranges; only escalate if a band regresses.

## Risks / unknowns
- **Policy vs value share the trunk and prefer opposite percentiles** — PTQ can't satisfy both via
  one scale. QAT *should* dissolve this: the weights adapt so both heads tolerate the chosen scale,
  rather than us tuning the scale. This is the core bet; if a single global percentile still can't
  serve both, LSQ-learnable per-tensor steps are the escalation.
- STE + symmetric per-tensor activations must match TRT exactly or we get train/deploy skew —
  mitigated by reusing qdq_export's scheme verbatim and the step-0==PTQ sanity check.
- Final-logits layer sensitivity — keep an eye on the heads; can leave the two head projections in
  higher precision if needed (TRT strongly-typed honors mixed QDQ).
- Calibration OOM that capped PTQ at 64 positions does NOT bind QAT — QAT calibration is one-time
  range-seeding, ranges then adapt via gradients over the whole corpus.

## Deliverables checklist
- [ ] `src/CeresTrainPy/fake_quant.py` (FakeQuantLinear + swap + percentile calibrate)
- [ ] `train.py` `Opt_QAT` flag wiring (swap+calibrate after resume; reuse KL-anchor as teacher)
- [ ] config: `c1_640_34_ceres_qat.json` (resume=from_onnx_0, KL-anchor=from_onnx_0, low LR, 20–50M)
- [ ] dev-box 5M smoke: step-0 == PTQ, loss converges
- [ ] prod 20–50M distill run → export via qdq_export → strongly-typed TRT
- [ ] validate: FEN sanity + rg2340/3-band puzzle ≥ FP16 floor
