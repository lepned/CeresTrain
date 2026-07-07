"""Show INT8 vs FP16 value on the CLEAREST-winning positions (interpretable
equivalent of a 'winning FEN' test) — does the WSL INT8 net correctly call a
clearly-won position as winning?

Uses dumped real positions (win_squares.npy + win_wdl.npy). Builds FP16 (orig
onnx) + INT8 (qdq.fp16io) engines, picks positions whose GT label is a
confident W or L, and prints the actual WDL distributions.

Usage (WSL): CERES_AUX_FEATURES_PER_SQUARE=4 python3 wsl_winning_check.py <orig_onnx> <qdq_onnx> <npy_dir>
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/mnt/c/Users/lepne/source/repos/CeresTrain/src/CeresTrainPy')
import int8_validate as iv
import tensorrt as trt

orig, qdq, npy = sys.argv[1], sys.argv[2], sys.argv[3]
sq = np.load(os.path.join(npy, 'win_squares.npy'))   # (N,64,141) fp16
wdl = np.load(os.path.join(npy, 'win_wdl.npy'))       # (N,3) fp32
N = sq.shape[0]


def build(onnx_path, out_path, is_qdq):
    if os.path.exists(out_path):
        os.remove(out_path)
    builder = trt.Builder(iv.LOG)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, iv.LOG)
    with open(onnx_path, 'rb') as f:
        assert parser.parse(f.read())
    feat = int(network.get_input(0).shape[-1])
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 33)
    cfg.set_flag(trt.BuilderFlag.FP16)
    if is_qdq:
        cfg.set_flag(trt.BuilderFlag.INT8)
    prof = builder.create_optimization_profile()
    prof.set_shape('squares', (1, 64, feat), (64, 64, feat), (256, 64, feat))
    cfg.add_optimization_profile(prof)
    ser = builder.build_serialized_network(network, cfg)
    assert ser is not None
    with open(out_path, 'wb') as f:
        f.write(bytes(ser))


def softmax(x):
    x = x.astype(np.float32); x = x - x.max(-1, keepdims=True)
    e = np.exp(x); return e / e.sum(-1, keepdims=True)


base = os.path.splitext(orig)[0]
fp16p, int8p = base + '.wc_fp16.engine', base + '.wc_int8.engine'
build(orig, fp16p, False); build(qdq, int8p, True)
fp16 = iv.Runner(fp16p); int8 = iv.Runner(int8p)

# Run all positions (batch 64) and collect value outputs.
B = 64
vf_all, vi_all = [], []
for i in range(0, N - N % B, B):
    x = {'squares': sq[i:i+B]}
    of = fp16.prepare(x); fp16.infer(); fp16.copy_outputs(of)
    oi = int8.prepare(x); int8.infer(); int8.copy_outputs(oi)
    vk = 'value' if 'value' in of else 'value2'
    vf_all.append(of[vk].astype(np.float32)); vi_all.append(oi[vk].astype(np.float32))
vf = softmax(np.concatenate(vf_all, 0)); vi = softmax(np.concatenate(vi_all, 0))
gt = softmax(wdl[:vf.shape[0]] * 1.0) if wdl.max() > 1.01 else wdl[:vf.shape[0]]

# clearest wins/losses by GT (W prob or L prob highest & confident)
conf = np.maximum(gt[:, 0], gt[:, 2])
order = np.argsort(-conf)
print(f'{"GT(W/D/L)":>22} | {"FP16(W/D/L)":>22} | {"INT8(W/D/L)":>22} | side')
agree_clear = nclear = 0
for idx in order[:14]:
    g, f, q = gt[idx], vf[idx], vi[idx]
    side = 'WIN ' if g[0] > g[2] else 'LOSS'
    print(f'  {g[0]:.2f}/{g[1]:.2f}/{g[2]:.2f}      |  {f[0]:.2f}/{f[1]:.2f}/{f[2]:.2f}      |  {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}      | {side}')
# overall: on the top-10% most-confident GT positions, INT8 argmax-correct rate
k = max(1, N // 10)
clear = order[:k]
int8_ok = (vi[clear].argmax(-1) == gt[clear].argmax(-1)).mean() * 100
fp16_ok = (vf[clear].argmax(-1) == gt[clear].argmax(-1)).mean() * 100
print(f'\nOn the {k} CLEAREST positions: FP16 value-correct {fp16_ok:.1f}% | INT8 value-correct {int8_ok:.1f}%')
