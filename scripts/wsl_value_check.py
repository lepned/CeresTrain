"""WSL direct VALUE-CORRECTNESS check (not just INT8-vs-FP16 agreement).

Answers: does the INT8 engine produce CORRECT value (WDL) on real positions in
WSL, measured against the ground-truth TPG value label — for BOTH the FP16
(original onnx) reference and the INT8 (qdq.fp16io) engine.

If FP16 value-acc ~= INT8 value-acc and both are reasonable -> WSL INT8 value
WORKS -> the Ceres/Windows break is build/runtime/platform specific.
If INT8 value-acc is garbage in WSL too -> the quantization breaks value and
Ceres was correctly reporting it (we were chasing the wrong layer).

Usage (WSL): CERES_AUX_FEATURES_PER_SQUARE=4 python3 wsl_value_check.py <orig_onnx> <qdq_fp16io_onnx> <tpg_dir>
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/mnt/c/Users/lepne/source/repos/CeresTrain/src/CeresTrainPy')
from tpg_dataset import TPGDataset
import int8_validate as iv  # reuse Runner/build
import tensorrt as trt

orig, qdq, tpg = sys.argv[1], sys.argv[2], sys.argv[3]
BATCH = 64
NB = 30

# First: inspect one item to find the value label.
ds = TPGDataset(tpg, BATCH, 0.0, 0, 1, 0, 1, 0, False)
item = ds[0]
print('[inspect] item is', type(item).__name__, 'len', len(item) if hasattr(item, '__len__') else '?')
for i, p in enumerate(item):
    if isinstance(p, dict):
        print(f'  part{i} dict keys:', list(p.keys()))
    else:
        print(f'  part{i}', type(p).__name__, getattr(p, 'shape', None))


def build(onnx_path, out_path, is_qdq):
    if os.path.exists(out_path):
        os.remove(out_path)
    builder = trt.Builder(iv.LOG)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, iv.LOG)
    with open(onnx_path, 'rb') as f:
        assert parser.parse(f.read()), 'parse failed'
    feat = int(network.get_input(0).shape[-1])
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 33)
    cfg.set_flag(trt.BuilderFlag.FP16)
    if is_qdq:
        cfg.set_flag(trt.BuilderFlag.INT8)
    prof = builder.create_optimization_profile()
    prof.set_shape('squares', (1, 64, feat), (BATCH, 64, feat), (256, 64, feat))
    cfg.add_optimization_profile(prof)
    ser = builder.build_serialized_network(network, cfg)
    assert ser is not None, 'build failed'
    with open(out_path, 'wb') as f:
        f.write(bytes(ser))


base = os.path.splitext(orig)[0]
fp16p = base + '.vchk_fp16.engine'
int8p = base + '.vchk_int8.engine'
build(orig, fp16p, False)
build(qdq, int8p, True)
fp16 = iv.Runner(fp16p); int8 = iv.Runner(int8p)

# value label discovery: look for a 3-wide WDL target in item parts
def find_value_label(it):
    for p in it:
        if isinstance(p, dict):
            for k, v in p.items():
                arr = v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
                if arr.ndim == 2 and arr.shape[-1] == 3:
                    return k, None
    return None, None

vlabel_key, _ = find_value_label(item)
print('[inspect] value-label key guess:', vlabel_key)

fp16_corr = int8_corr = agree = n = 0
for _ in range(NB):
    it = ds[0]
    x = {'squares': it[0]['squares'].numpy()}
    # ground-truth value label
    lab = None
    for p in it:
        if isinstance(p, dict) and vlabel_key in p:
            lab = p[vlabel_key].numpy(); break
    if x['squares'].shape[0] != BATCH:
        continue
    of = fp16.prepare({'squares': x['squares'].astype(np.float16)}); fp16.infer(); fp16.copy_outputs(of)
    oi = int8.prepare({'squares': x['squares'].astype(np.float16)}); int8.infer(); int8.copy_outputs(oi)
    vk = 'value' if 'value' in of else 'value2'
    vf = of[vk].astype(np.float32).argmax(-1)
    vi = oi[vk].astype(np.float32).argmax(-1)
    agree += int((vf == vi).sum())
    if lab is not None:
        gt = lab.argmax(-1)
        fp16_corr += int((vf == gt).sum())
        int8_corr += int((vi == gt).sum())
    n += BATCH

print(f'\nPositions: {n}')
print(f'INT8 vs FP16 value-argmax agreement: {100*agree/n:.2f}%')
if vlabel_key:
    print(f'FP16 value accuracy vs GT label   : {100*fp16_corr/n:.2f}%')
    print(f'INT8 value accuracy vs GT label   : {100*int8_corr/n:.2f}%')
    print('VERDICT:', 'WSL INT8 value WORKS (Ceres/platform is the bug)' if int8_corr/max(n,1) > 0.25 else 'WSL INT8 value ALSO BROKEN (quantization breaks value)')
