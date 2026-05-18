# Opset-23 fused-norm export recipe

**Audience:** Claude Code (or anyone) applying these changes on a *different* machine
than the one where they were authored. Self-contained runbook.

**Effect:** Re-exports an existing pre-norm + RMSNorm + RoPE network so the ONNX
contains one fused `RMSNormalization` op per RMSNorm (instead of a 6-op
`Pow/Mean/Add/Sqrt/Div/Mul` chain). TRT then applies its built-in FP32 fallback
to only the inner reduction. Measured win on the 10M-pos sibling
`c2_512_16_swiglu_rope_base1000_PRE_10M`: **NPS 20.1 K → 25.2 K (+25 %), EPS
18.0 K → 24.0 K (+33 %)** at 15 s movetime on startpos, RTX 5090. Expected to
transfer to the production `c1_512_16_v1_off_pt1` 200M-pos net (same arch).

The fix is split across two repos:

- **CeresTrain** — three Python files at export time (this repo)
- **Ceres** — one C++ source file in `TensorRTWrapper.cpp` (separate repo)

If only the CeresTrain changes are applied, TRT will still build correctly but
will mark zero FP32-norm layers (because the name-based matcher and the
structural marker both expect specific patterns). The wrapper change is
required for the FP32-overflow protection on **other** pre-norm nets — keep it
in mind when syncing.

---

## Background — why this is needed

Pre-norm trunks leave the residual stream un-normalized between blocks. After
many blocks, magnitudes grow large enough that `Pow(x, 2)` overflows in FP16
(FP16 max ≈ 65504, so `|x| > 256` overflows). PyTorch's exporter previously
emitted the explicit `Pow → ReduceMean → Add → Sqrt → Div → Mul` chain for our
custom `RMSNorm` class. TRT saw six separate FP16 ops per norm, and the only
mitigation was to force-cast all six to FP32 via the structural marker in the
C++ wrapper — ~365 FP32 layers for a 512×16 net.

PyTorch ≥ 2.4 has `torch.nn.functional.rms_norm`. PyTorch ≥ 2.10's dynamo ONNX
exporter at opset 23 (ONNX opset 23 added `RMSNormalization` as a first-class
op) emits a single fused node per norm, which TRT 10.15 handles natively with
internal FP32 reduce + FP16 mul. ~50 FP32 ops instead of ~365, ~44 % fewer
total TRT layers, and the bandwidth-bound EPS/NPS ratio climbs from 89 % to
95 %.

The same dynamo exporter also helpfully fuses `F.scaled_dot_product_attention`
into the opset-23 `Attention` op — but TRT 10.15's `Attention` plugin requires
the network to be built in strongly-typed mode, which the C++ wrapper does
not enable. The third file change below routes attention through the existing
explicit `MatMul → Softmax → MatMul` form so dynamo has nothing to fuse.

---

## File changes — CeresTrain repo

All three changes preserve `state_dict` keys → existing checkpoints load
unchanged. Math is identical to the previous implementations.

### 1. `src/CeresTrainPy/rms_norm.py`

Replace the `forward()` body of class `RMSNorm` with the PyTorch built-in:

```python
class RMSNorm(torch.nn.Module):
  def __init__(self, d_model : int, eps : float =1e-6):
    super().__init__()
    self.d_model = d_model
    self.eps = eps
    self.scale = torch.nn.Parameter(torch.ones(d_model))

  def forward(self, x : Tensor) -> Tensor:
    # Use the PyTorch built-in so ONNX export emits a single fused
    # LayerNormalization / RMSNormalization op (opset 17+) instead of the
    # decomposed Pow→Mean→Sqrt→Div→Mul chain. TRT recognises the fused op
    # and applies its internal FP32 fallback to only the reduction, leaving
    # the surrounding ops in FP16 — avoiding the 6-op-per-norm FP32 cost
    # that TensorRTWrapper's structural marker currently has to force.
    # Mathematically identical to the explicit form (same scale, same eps).
    return torch.nn.functional.rms_norm(x, (self.d_model,), self.scale, self.eps)
```

### 2. `src/CeresTrainPy/save_model.py`

Find the `torch.onnx.export(...)` call (around line 243 inside the
`if True:` legacy-ONNX-export block). Change `opset_version=18` to
`opset_version=23`:

```python
torch.onnx.export(_export_model,
                  _export_inputs,
                  SAVE_FULL_NAME,
                  do_constant_folding=True,
                  export_params=True,
                  opset_version=23,             # <-- was 18
                  input_names = _input_names,
                  output_names = head_output_names,
                  dynamic_axes=_output_axes_single)
```

Do **not** pass `dynamo=False`. PyTorch 2.10's default is `dynamo=True`, and
the dynamo path is what emits the fused `RMSNormalization`. The legacy path
explicitly rejects `aten::rms_norm` at opset 18 and won't emit fused norms
at any opset.

### 3. `src/CeresTrainPy/dot_product_attention.py`

In `DotProductAttention.forward()`, find the `else` branch under
`if self.use_smolgen:` and remove the `F.scaled_dot_product_attention` fast
path. Always route through `sdp_and_smol_or_rpe`:

```python
if self.use_smolgen:
  smolgen = self.calc_smolgen(x)
  H_cat, A = self.sdp_and_smol_or_rpe(Q, K, V, smolgen, piece_relation_bias=piece_relation_bias)
else:
  # Always route through the explicit Q·Kᵀ → softmax → ·V form. The previous
  # branch called torch.nn.functional.scaled_dot_product_attention, which
  # PyTorch ≥ 2.10's dynamo ONNX exporter auto-fuses into the opset-23
  # `Attention` op — and TRT 10.15's Attention plugin requires the network
  # to be built in strongly-typed mode, which the C++ wrapper does not use,
  # so engine build aborts with API Usage Error 3.
  # The explicit form is mathematically equivalent (no mask, no dropout),
  # exports cleanly to opset 23 as MatMul→Softmax→MatMul, and also gains
  # softcap support that the F.sdpa path was lacking.
  H_cat, A = self.sdp_and_smol_or_rpe(Q, K, V, None, piece_relation_bias=piece_relation_bias)
```

Training-side cost: a slight slowdown (5–15 % per step) because PyTorch's
F.sdpa with FlashAttention is faster than the explicit form for large batch
sizes. Math is identical for our case (no causal mask, no dropout). If a
training run is already in progress, **do not stop it for this change** —
the trained checkpoint is valid against either forward implementation, so
the change only needs to be in place at the next `recover_export.py`
invocation.

---

## Wrapper-side changes — Ceres repo

These are already shipped on `origin/main` of the Ceres repo (commit
`686bed7b TensorRTNative: FP32-mark RMSNorm chains structurally for
pre-norm nets`). A plain `git pull` in the Ceres tree picks them up
along with the **rebuilt Windows DLL** — no manual cpp compile is
required.

### After-pull workflow

Ceres.Chess.csproj already has a `CopyToOutputDirectory` /
`PreserveNewest` clause for `TensorRTWrapper.dll` (and the Linux
`.so` variants), so a normal Ceres rebuild auto-deploys the newer
DLL to `artifacts/release/net10.0/` (and `debug/` if you build
that config):

```powershell
git pull                              # picks up new cpp + new DLL
dotnet build -c Release Ceres.sln     # MSBuild copies the DLL to artifacts/
```

That's it — no manual copy required when going through `dotnet
build`. The artifact-dir DLLs are not git-tracked (PreserveNewest
makes them, build-time), so a pull alone wouldn't update them, but
the next rebuild will.

If you ever want to deploy a newer DLL *without* rebuilding the
managed code (rare — e.g. swapping wrappers between Ceres versions
for A/B), `Copy-Item` from `src\...\Native\TensorRTWrapper.dll` to
the artifact dir is the manual shortcut.

### When you would need to rebuild from cpp

The committed DLL is built against TRT 10.15.1.29 + CUDA 12.9 + VS
2022 MSVC v143. TRT 10.x exposes a stable ABI, so it loads against
any TRT 10.x runtime (10.10–10.15+ tested in this family). You only
need to rebuild if:

- the target machine has TRT 9.x or older (force-upgrade is easier);
- you want a Linux build — `linux-{arm64,x64}/libTensorRTWrapper.so`
  in the repo are pre-built from an earlier source revision and do
  **not** include the new structural FP32-norm marker. Rebuild via
  the `Makefile` in the same dir against your local TRT install;
- you want to tweak the marker further.

For a Windows rebuild, use the existing `build.cmd` in the same dir
after adjusting its `CUDA_ROOT` / `TENSORRT_ROOT` to local paths.
The script uses `cl /std:c++17 /O2 /MD /LD`, links `nvinfer_10.lib
nvonnxparser_10.lib cudart.lib`, and writes the output `.dll`
directly. After it succeeds, run the two `Copy-Item` lines above to
deploy.

### What the cpp changes look like

For reference (the diff is already on `origin/main`):

1. In `HasNormName()` (~line 322 of `TensorRTWrapper.cpp`), add the
   line `|| name.find("trunk_end_norm") != std::string::npos` to the
   substring list. The trainer's pre-norm trunk introduces a final
   norm by that name that the previous matcher missed.

2. Just before the existing `if (opts->fp32PostAttentionNorm ||
   ...)` block (~line 3092), a new structural FP32-norm marker walks
   forward from each `*.scale` initializer to its first
   kELEMENTWISE consumer, then BFS-back over Pow/ReduceMean/Add/
   Sqrt/Div compute layers and marks each FP32. Gated by
   `if (!opts->useBF16)` so it's a no-op under BF16 (which has the
   same 8-bit exponent as FP32 and doesn't overflow). An env-gated
   debug dump (`TRT_DUMP_LAYER_NAMES=1`) at the end of the same
   block enumerates the chains it finds — useful when the trainer's
   exporter output changes shape in the future.

To re-derive the block from scratch, search for the comment marker
`// -------- STRUCTURAL FP32 norm marker` in the source.

---

## Export environment

PyTorch 2.10's dynamo ONNX exporter needs three packages that are **not**
required by training:

- `onnxscript` ≥ 0.7
- `onnx` ≥ 1.21
- `onnxconverter-common` ≥ 1.16 (for the post-export FP16 cast in
  `save_model.py`)

On Windows, the Microsoft Store Python 3.10 has long-path support disabled by
default, and the `onnx` wheel contains test data with paths > 260 chars. Two
workarounds:

**A. Short install prefix (no admin needed):**

```powershell
python -m pip install --target D:\py-pkgs --no-user --upgrade onnxscript onnx onnxconverter-common
$env:PYTHONPATH = "D:\py-pkgs"
```

Then prefix every export command with `PYTHONPATH=D:\py-pkgs` (or set the
env var once in the shell).

**B. Enable long-path support (requires admin):**

```powershell
Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' `
                 -Name LongPathsEnabled -Value 1
```

Then `pip install onnxscript onnxconverter-common` works against the default
site-packages.

Also set `PYTHONIOENCODING=utf-8` before running the export — torch.onnx
prints emoji status markers (✅/❌) and the default Windows cp1252 codec
crashes on them.

---

## Re-export recipe

```bash
# 1. Sync the three trainer files above (rms_norm.py, save_model.py,
#    dot_product_attention.py). Sanity-check with git status that they
#    are the only modified files. Push origin if not already there.

# 2. In the Ceres repo: git pull, then dotnet build -c Release.
#    The wrapper change and rebuilt DLL are committed (686bed7b);
#    Ceres.Chess.csproj has a PreserveNewest copy rule that
#    auto-deploys the DLL to artifacts/release/net10.0/ on rebuild.
#    No cpp rebuild needed unless the target machine is on TRT 9.x.

# 3. Set up the export env (one-time):
python -m pip install --target D:\py-pkgs --no-user --upgrade onnxscript onnx onnxconverter-common

# 4. Re-export the final checkpoint:
cd src/CeresTrainPy
PYTHONIOENCODING=utf-8 PYTHONPATH=D:\py-pkgs python recover_export.py <TRAINING_ID> <OUTPUTS_DIR> <NUM_POS>

# Where the inputs are the standard recover_export.py args:
#   <TRAINING_ID>  bare training id (no leading "lepdev_" or hostname)
#   <OUTPUTS_DIR>  parent dir containing nets/ and configs/
#   <NUM_POS>      position count suffix on the checkpoint filename
#
# Example for the 200M c1_512_16 production net:
#   python recover_export.py c1_512_16_v1_off_pt1 F:/cout 200000512

# 5. Look for these lines in the export output:
#   [torch.onnx] Translate the graph into ONNX... ✅
#   INFO: ONNX_FILENAME ...
#   INFO: ONNX_FP16_CONVERSION_APPLIED ...
```

---

## Verification

After export, before deploying to Ceres, run this one-liner against the new
`.onnx`. Replace the path with the actual output:

```bash
PYTHONPATH=D:\py-pkgs python -c "
import onnx
from collections import Counter
m = onnx.load('<path-to-new.onnx>', load_external_data=False)
ops = Counter(n.op_type for n in m.graph.node)
print('RMSNormalization:', ops.get('RMSNormalization', 0))
print('Attention       :', ops.get('Attention', 0))
print('Pow             :', ops.get('Pow', 0))
print('opset           :', [(op.domain or 'ai.onnx', op.version) for op in m.opset_import])"
```

Expected output for a 16-layer pre-norm trunk:

```
RMSNormalization: 34    # 1 embedding + 2*16 per-block + 1 trunk-end
Attention       : 0     # must be zero — non-zero means F.sdpa fused into Attention
Pow             : 0     # must be zero — non-zero means RMSNorm was NOT fused
opset           : [('ai.onnx', 23)]
```

If `RMSNormalization` is missing or `Pow` is non-zero, the `rms_norm.py` or
`save_model.py` change did not take effect — re-check those files. If
`Attention` is non-zero, the `dot_product_attention.py` change did not take
effect — Ceres will fail to build the TRT engine with "API Usage Error 3"
("Attention can only be used with a strongly typed network").

Deploy the new `.onnx` to Ceres' network directory, clear the TRT engine
cache (`<NetDir>/trt_engines/<MachineName>/*.engine` matching the net's
filename pattern), and run a warmup `analyze` to trigger engine build. The
build log should contain:

```
[TensorRT] Auto-marked 34 compute layers FP32 across 34 norm chains (structural detection by scale-constant)
```

If you see `Auto-marked 365 ... across 50 norm chains` instead, the ONNX
still has the decomposed pattern — re-check the verification one-liner.

If the build aborts with `API Usage Error 3` mentioning `Attention can only
be used with a strongly typed network`, the `dot_product_attention.py`
change is missing.

---

## Expected speedup

Measured at 15 s movetime, startpos, RTX 5090, on the 10M-pos sibling
`c2_512_16_swiglu_rope_base1000_PRE_10M`:

| variant | NPS | EPS | EPS/NPS |
|---|---|---|---|
| FP16, decomposed RMSNorm, 365 FP32-marked layers | 20.1 K | 18.0 K | 89 % |
| **FP16, fused RMSNormalization, 34 FP32-marked layers** | **25.2 K** | **24.0 K** | **95 %** |

For the production `c1_512_16_v1_off_pt1` 200M-pos net the same architecture
holds; expect 24–26 K NPS up from the current 20.1 K. Bestmove/eval/WDL
output should match the previous net exactly (the math is identical;
only the graph shape changed).

---

## INT8 deployment validation

The fused-norm pipeline is INT8-friendly: a quantized engine preserves policy
and value with very high fidelity, opening up full-INT8 deployment (not the
partial-precision compromise existing Ceres INT8 nets need).

### Measured precision (smoke c2_640_34, 1 M puzzle pos, 1920 calibration positions)

| metric | result | acceptance |
|---|---|---|
| Policy top-1 agreement | 78.54 % | ≥75 % ✓ |
| Policy top-3 agreement (≥2 overlap) | 97.08 % | ≥90 % ✓ |
| Policy KL(FP16 ‖ INT8) mean | **0.00520** | <0.02 ✓ (4× tighter than threshold) |
| Value softmax L1 mean | 0.0295 | <0.05 ✓ |
| **Value WDL argmax agreement** | **100.00 %** | ≥99 % ✓ (never disagrees on W/D/L) |

The 22 % top-1 disagreement is between near-tied moves (expected quantization
signature, not failure). The 100 % WDL-argmax-preserved property is the
load-bearing finding for MCTS — eval direction never flips under INT8.

### Measured speed (WSL TRT 10.16, RTX 5090, batch=64)

| precision | latency/call | throughput | vs FP16 |
|---|---|---|---|
| FP16 | 7.43 ms | 8.62 K pos/sec | 1.00× |
| INT8 (calibrated) | 5.87 ms | 10.91 K pos/sec | **+26.6 %** |

Calibration quality affects accuracy (KLD), not speed — the same +27 % shows
up with naïve no-calibration trtexec runs. Projected Ceres-side INT8 NPS for
the c2_640_34 modern arch: **~12.3 K** (vs ~9.7 K FP16 measured, ~8.2 K for
the orig INT8 baseline — roughly +50 % over orig).

### Decision verdict thresholds

| precision band | what it means | action |
|---|---|---|
| **GREEN**: KLD < 0.05, top-1 delta < 1 pp, val argmax ≥ 99 % | Architecture is INT8-friendly. Full-INT8 deployment plausible. | Commit to production training; plan INT8 release path. |
| **AMBER**: KLD 0.05-0.20 | Borderline. Some layers will need FP16 kept (partial-INT8). | Train, plan partial-INT8 like existing orig/C3 pattern. |
| **RED**: KLD > 0.20 or garbage outputs | Some op is quantization-hostile. | Defer; investigate which layer and whether to reformulate. |

c2_640_34 lands solidly in the GREEN band.

### Validation tool — `scripts/int8_validate.py`

Builds FP16 + calibrated-INT8 engines, runs both on the same TPG inputs, and
reports the comparison table above. Use it on any new ONNX from this export
pipeline:

```bash
# WSL with cerestrain-env activated; cuda-python + tensorrt already in venv.
# polygraphy is NOT used here — it crashes (`_Map_base::at`) on opset-23
# RMSNormalization graphs. TRT Python API direct works.
source ~/cerestrain-env/bin/activate
python3 scripts/int8_validate.py \
  /mnt/c/Dev/Chess/CeresTrain/nets/<your_net>.onnx \
  /mnt/d/<your_tpg_dir> \
  --num_batches 30 --calib_batches 8
```

Outputs:
- `<your_net>.fp16.engine` and `<your_net>.int8.engine` saved next to the ONNX
- `<your_net>.int8.engine.calib.cache` (the entropy calibration cache — keep
  for reproducible rebuilds)
- The precision table and a GREEN/AMBER/RED verdict printed to stdout
- Per-call latency comparison

### Caveats and known issues

- **Polygraphy `convert --int8` is broken** for opset-23 + RMSNormalization on
  TRT 10.15-10.16: crashes inside the calibrator with `_Map_base::at` regardless
  of whether you provide a data loader. Use the TRT Python API directly (the
  script above does this).

- **TRT engines are platform-specific.** A Windows trtexec-built engine
  cannot load in WSL Linux TRT and vice versa. Build engines on the platform
  where they'll run. For Ceres-Windows production, build via the existing C++
  wrapper path (which calls TRT internally) — the script above is for
  validation/research in WSL only.

- **TRT 10.16 (pip) is stricter than 10.15 (Windows install)** about
  requiring an INT8 calibrator. 10.15 will build an INT8 engine with auto-
  derived default ranges; 10.16 errors with "Calibration failure occurred
  with no scaling factors detected" unless you provide a calibrator.

- **Output-boundary Dequantize ops stay FP16** ("Dequantize NNN [SCALE] has
  invalid precision Int8, ignored" warning during build is routine). The
  inner trunk is fully INT8.

- **Smoke-vs-production gap.** Activation-distribution shapes are largely
  arch-determined and don't change much between an undertrained smoke and a
  converged net, so the GREEN verdict here is meaningful for the production
  decision. But re-run this validation against a properly-trained net before
  shipping — sharp tactical positions in real play may stress the calibration
  ranges differently than puzzle-only data.

- **Production calibration data should be game positions**, not puzzle-only.
  This smoke used the puzzle TPG corpus the net was trained on; broader
  game-distribution calibration will give slightly different (probably
  tighter) ranges.

- **No Ceres-side INT8 loading path exists yet.** The wrapper has
  `TRT_LoadMultiProfileEngineFile` (cpp:3495) which can load a pre-built
  engine; wiring this into the engine-cache path so it picks up the INT8
  variant instead of building from ONNX is a small additional task (~50
  lines C# + minor naming convention). Defer this until a production-grade
  INT8 net exists.

---

## Rollback

The trainer-side changes are zero-impact on the training trajectory (math
identical). If for any reason the new export needs to be undone:

1. Revert the three trainer files to the prior versions.
2. Re-run `recover_export.py` — produces a decomposed-norm ONNX at opset 18.
3. Deploy that `.onnx`, clear the TRT engine cache.

The pre-existing structural marker in `TensorRTWrapper.cpp` will detect the
50 decomposed chains and apply 365 FP32 layers, matching the original
behaviour. Speed will drop back to ~20.1 K NPS.

The wrapper-side change (`HasNormName` addition + structural marker) is safe
to leave in place under any export style — it auto-detects what's in the
graph and marks the appropriate set.
