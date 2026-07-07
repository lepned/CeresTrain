"""Activation surgery: is the value break a SCALING problem (advanced PTQ /
SmoothQuant / AdaRound can fix it, NO retraining) or a BITS problem (signal at
the 8-bit noise floor -> needs QAT or FP16)?

Runs the FP32 model and the QDQ (quantization-simulating) model in ORT on the
SAME real positions (ORT executes Q/DQ as real ops -> simulates the rounding on
CPU, no TRT/Ceres needed) and:
  1. Confirms the value output collapses under simulated quantization (and by
     how much) vs FP32 — isolating the break to the quantization itself.
  2. Reads the per-tensor ACTIVATION magnitudes from the QDQ scales
     (scale*127 = max_abs the calibration saw) and reports the OUTLIER
     structure: if a few tensors/channels have huge magnitudes dominating the
     scale, the small value signal is crushed -> SmoothQuant/AdaRound territory
     (no retrain). If magnitudes are uniform yet value still dies -> the signal
     is genuinely sub-8-bit -> QAT.

Usage (WSL): CERES_AUX_FEATURES_PER_SQUARE=4 python3 surgery_activations.py <fp32_onnx> <qdq_onnx> <tpg_dir>
"""
import os, sys
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
sys.path.insert(0, '/mnt/c/Users/lepne/source/repos/CeresTrain/src/CeresTrainPy')
from tpg_dataset import TPGDataset

fp32_path, qdq_path, tpg = sys.argv[1], sys.argv[2], sys.argv[3]
N = 96

so = ort.SessionOptions(); so.log_severity_level = 3
sess_fp = ort.InferenceSession(fp32_path, so, providers=['CPUExecutionProvider'])
sess_q = ort.InferenceSession(qdq_path, so, providers=['CPUExecutionProvider'])
in_name = sess_fp.get_inputs()[0].name


def softmax(x):
    x = x.astype(np.float64); x = x - x.max(-1, keepdims=True)
    e = np.exp(x); return e / e.sum(-1, keepdims=True)


ds = TPGDataset(tpg, N, 0.0, 0, 1, 0, 1, 0, False)
sq = ds[0][0]['squares'].numpy().astype(np.float32)
vf = softmax(sess_fp.run(['value'], {in_name: sq})[0])
vq = softmax(sess_q.run(['value'], {in_name: sq})[0])

print('=== VALUE output: FP32 vs simulated-quant (ORT) ===')
print(f'FP32  value W spread across {N} positions: min {vf[:,0].min():.3f} max {vf[:,0].max():.3f} std {vf[:,0].std():.3f}')
print(f'QUANT value W spread across {N} positions: min {vq[:,0].min():.3f} max {vq[:,0].max():.3f} std {vq[:,0].std():.3f}')
print(f'  (FP32 std>>0 = value tracks position; QUANT std~0 = collapsed to constant prior)')
print(f'mean |FP32 W - QUANT W| over positions: {np.abs(vf[:,0]-vq[:,0]).mean():.3f}')
# how much of the FP32 value variation survives quantization?
if vf[:,0].std() > 1e-6:
    print(f'value signal RETAINED under quant: {100*vq[:,0].std()/vf[:,0].std():.1f}% of FP32 variation')

print('\n=== ACTIVATION magnitudes (from QDQ calibration scales; max_abs = scale*127) ===')
qm = onnx.load(qdq_path)
inits = {i.name: numpy_helper.to_array(i) for i in qm.graph.initializer}
init_names = set(inits)
act_max = []
for n in qm.graph.node:
    if n.op_type != 'QuantizeLinear':
        continue
    quantized = n.input[0]
    if quantized in init_names:
        continue  # weight, skip — we want ACTIVATIONS
    sc = inits.get(n.input[1])
    if sc is None:
        continue
    s = float(np.asarray(sc).reshape(-1)[0])
    act_max.append(s * 127.0)
act_max = np.array(sorted(act_max))
if len(act_max):
    med = np.median(act_max)
    print(f'#activation tensors quantized: {len(act_max)}')
    print(f'activation max_abs: min {act_max.min():.3f}  median {med:.3f}  p90 {np.percentile(act_max,90):.3f}  max {act_max.max():.3f}')
    print(f'OUTLIER ratio (max / median): {act_max.max()/max(med,1e-9):.1f}x  (>~10x => big outliers => SmoothQuant-fixable scaling problem)')
    print(f'top 5 activation magnitudes: {[round(x,2) for x in act_max[-5:]]}')
