#!/usr/bin/env python3
"""
NVFP4 (E2M1, block-16, FP8 block scales) explicit-quantization export for a
CeresNet FP16 ONNX graph, in the exact ONNX form TensorRT's parser fuses into
FP4 GEMMs (the ModelOpt "double quantization" recipe):

  weights (offline, static):
      sw_f8 [K/16, N] = FP8(E4M3)(blockamax/6 / s_global),  s_global = amax_w / (6*448)
      DequantizeLinear(sw_f8, s_global)                         -> sw   [K/16, N]
      DequantizeLinear(w_f4 [K,N], sw, axis=0, block_size=16)   -> w_dq [K, N]
  activations (dynamic, in-engine):
      trt::TRT_FP4DynamicQuantize(a, s_ga, axis=-1, block_size=16, scale_type=FP8)
                                                                -> a_f4, sa_f8
      DequantizeLinear(sa_f8, s_ga)                             -> sa
      DequantizeLinear(a_f4, sa, axis=-1, block_size=16)        -> a_dq
      MatMul(a_dq, w_dq)

Only MatMuls with a 2-D initializer weight whose K and N are multiples of 16
are converted (the attention QK/AV act-act MatMuls and the tiny head GEMMs stay
FP16). Activation global scales come from an ORT calibration pass over TPG
positions (amax). Requires onnx>=1.18 (FLOAT4E2M1), ml_dtypes, TensorRT>=10.8
and a Blackwell GPU for the FP4 GEMM kernels.

usage (WSL, cerestrain venv):
  python3 scripts/fp4_export.py <net.onnx> <tpg_dir> --run_config <configs_dir> --run_id <id>
        [--batch 64] [--calib_batches 8] [--num_batches 30] [--min_n 128]
        [--exclude_pattern SUBSTR] [--scale_dtype fp16|fp32] [--no_verify]
Outputs next to the input:  <net>.nvfp4.onnx  (+ .fp16ref.engine / .nvfp4.engine when verifying)
"""
import argparse, os, sys
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# qdq_export does the run-config -> CERES_* env bridge at import time (it scans
# argv for --run_config/--run_id, same flag names as here) and pulls in
# TPGDataset + the TRT Runner/bench harness. Import it FIRST.
import qdq_export as qx
from tpg_dataset import TPGDataset

E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_MAX, FP8_MAX, BLOCK = 6.0, 448.0, 16


def to_e2m1_codes(x):
    """Round |x| to nearest E2M1 magnitude (ties to even code), return 4-bit codes."""
    a = np.abs(x).astype(np.float32)
    mid = (E2M1_VALUES[1:] + E2M1_VALUES[:-1]) / 2  # 0.25 0.75 1.25 1.75 2.5 3.5 5
    idx = np.searchsorted(mid, a, side='right').astype(np.int32)  # nearest-up on ties
    # ties-to-even: an exact midpoint landed on idx (odd/even?) -> prefer even code
    tie = np.isin(a, mid)
    idx = np.where(tie & (idx % 2 == 1), idx - 1, idx)
    idx = np.clip(idx, 0, 7)
    sign = (x < 0).astype(np.int32)
    return (sign << 3) | idx


def pack_fp4(codes):
    flat = codes.reshape(-1).astype(np.uint8)
    if flat.size % 2:
        flat = np.append(flat, np.uint8(0))
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)


def to_fp8_e4m3(x):
    import ml_dtypes
    return np.clip(x, 0, FP8_MAX).astype(ml_dtypes.float8_e4m3fn)


def quantize_weight_nvfp4(w):
    """w: fp32 [K,N] -> (w_f4 packed bytes, sw_f8 [K/16,N] bytes, s_global fp32, w_dq fp32)"""
    K, N = w.shape
    wb = w.reshape(K // BLOCK, BLOCK, N)
    amax_b = np.abs(wb).max(axis=1)                       # [K/16, N]
    amax_w = float(np.abs(w).max())
    s_global = amax_w / (FP4_MAX * FP8_MAX) if amax_w > 0 else 1.0
    s_b = amax_b / FP4_MAX / s_global
    s_b8 = to_fp8_e4m3(s_b)
    s_deq = s_b8.astype(np.float32) * s_global            # [K/16, N]
    s_safe = np.where(s_deq > 0, s_deq, 1.0)
    q = wb / s_safe[:, None, :]
    codes = to_e2m1_codes(q).reshape(K, N)
    w_dq = (E2M1_VALUES[codes & 7] * np.where(codes >> 3, -1.0, 1.0)).reshape(K // BLOCK, BLOCK, N) * s_deq[:, None, :]
    return pack_fp4(codes), s_b8.view(np.uint8).tobytes(), np.float32(s_global), w_dq.reshape(K, N)


def select_matmuls(g, min_n, exclude_pattern, only_dims=None):
    inits = {t.name: t for t in g.initializer}
    sel = []
    for n in g.node:
        if n.op_type != 'MatMul' or n.input[1] not in inits:
            continue
        t = inits[n.input[1]]
        if len(t.dims) != 2:
            continue
        K, N = t.dims
        if K % BLOCK or N % BLOCK or N < min_n:
            continue
        if only_dims and (K not in only_dims or N not in only_dims):
            continue
        if exclude_pattern and exclude_pattern in n.name:
            continue
        sel.append(n)
    return sel


def calibrate_amax(fp16_path, act_names, tpg_dir, batch, calib_batches):
    """Run the FP32 twin of the graph under ORT, return {tensor: amax}."""
    import onnxruntime as ort
    fp32_path = fp16_path + '.calib32.onnx'
    qx.fp16_to_fp32(fp16_path, fp32_path)
    m = onnx.load(fp32_path)
    existing = {o.name for o in m.graph.output}
    for a in act_names:
        if a not in existing:
            m.graph.output.append(helper.make_tensor_value_info(a, TensorProto.FLOAT, None))
    onnx.save(m, fp32_path)
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(fp32_path, so, providers=['CPUExecutionProvider'])
    ds = TPGDataset(tpg_dir, batch, 0.0, 0, 1, 0, 1, 0, False)
    amax = {a: 0.0 for a in act_names}
    names = list(act_names)
    for i in range(calib_batches):
        b = ds[0][0]['squares'].numpy().astype(np.float32)
        outs = sess.run(names, {'squares': b})
        for a, o in zip(names, outs):
            amax[a] = max(amax[a], float(np.abs(o).max()))
        print(f'[calib] batch {i+1}/{calib_batches}', flush=True)
    os.remove(fp32_path)
    return amax


def convert(fp16_path, out_path, tpg_dir, batch, calib_batches, min_n, exclude_pattern,
            scale_dtype, act_headroom, only_dims=None, weights_only=False):
    m = onnx.load(fp16_path)
    g = m.graph
    sel = select_matmuls(g, min_n, exclude_pattern, only_dims)
    act_names = []
    for n in sel:
        if n.input[0] not in act_names:
            act_names.append(n.input[0])
    print(f'[fp4] MatMuls selected: {len(sel)} / {sum(1 for n in g.node if n.op_type=="MatMul")}; '
          f'unique activation tensors: {len(act_names)}')
    amax = (calibrate_amax(fp16_path, act_names, tpg_dir, batch, calib_batches)
            if not weights_only else {a: 0.0 for a in act_names})

    sdt = TensorProto.FLOAT16 if scale_dtype == 'fp16' else TensorProto.FLOAT
    sdt_np = np.float16 if scale_dtype == 'fp16' else np.float32
    inits = {t.name: t for t in g.initializer}
    new_nodes_before = {}   # node index -> list of nodes to insert before it
    node_index = {id(n): i for i, n in enumerate(g.node)}
    act_done = {}
    n_w = 0
    wstats = []
    for n in sel:
        i = node_index[id(n)]
        pre = new_nodes_before.setdefault(i, [])
        # ---- weight ----
        t = inits[n.input[1]]
        w = numpy_helper.to_array(t).astype(np.float32)
        K, N = w.shape
        w_f4, sw_f8, s_g, w_dq = quantize_weight_nvfp4(w)
        wstats.append(float(np.abs(w - w_dq).mean() / (np.abs(w).mean() + 1e-12)))
        base = t.name
        g.initializer.remove(t)
        g.initializer.extend([
            helper.make_tensor(base + '_f4', TensorProto.FLOAT4E2M1, [K, N], w_f4.tobytes(), raw=True),
            helper.make_tensor(base + '_sf8', TensorProto.FLOAT8E4M3FN, [K // BLOCK, N], sw_f8, raw=True),
            numpy_helper.from_array(np.array(s_g, dtype=sdt_np), base + '_sg'),
        ])
        pre.append(helper.make_node('DequantizeLinear', [base + '_sf8', base + '_sg'], [base + '_sw'],
                                    name=base + '_dq_scale'))
        pre.append(helper.make_node('DequantizeLinear', [base + '_f4', base + '_sw'], [base + '_dq'],
                                    name=base + '_dq_w', axis=0, block_size=BLOCK))
        n.input[1] = base + '_dq'
        n_w += 1
        # ---- activation ----
        a = n.input[0]
        if weights_only:
            continue
        if a not in act_done:
            s_ga = amax[a] * act_headroom / (FP4_MAX * FP8_MAX) if amax[a] > 0 else 1.0
            an = a.replace('/', '_').replace(':', '_')
            g.initializer.extend([
                numpy_helper.from_array(np.array(s_ga, dtype=np.float32), an + '_sga32'),
                numpy_helper.from_array(np.array(s_ga, dtype=sdt_np), an + '_sga'),
            ])
            pre.append(helper.make_node('TRT_FP4DynamicQuantize', [a, an + '_sga32'],
                                        [an + '_f4', an + '_sf8'], name=an + '_dynq', domain='trt',
                                        axis=-1, block_size=BLOCK, scale_type=int(TensorProto.FLOAT8E4M3FN)))
            pre.append(helper.make_node('DequantizeLinear', [an + '_sf8', an + '_sga'], [an + '_sa'],
                                        name=an + '_dq_scale'))
            pre.append(helper.make_node('DequantizeLinear', [an + '_f4', an + '_sa'], [an + '_dq'],
                                        name=an + '_dq_a', axis=-1, block_size=BLOCK))
            act_done[a] = an + '_dq'
        n.input[0] = act_done[a]
    # splice
    nodes = list(g.node)
    out = []
    for i, nd in enumerate(nodes):
        out.extend(new_nodes_before.get(i, []))
        out.append(nd)
    del g.node[:]
    g.node.extend(out)
    # opsets: FLOAT4E2M1 + blocked DequantizeLinear need opset 23; trt custom domain
    for op in m.opset_import:
        if op.domain in ('', 'ai.onnx'):
            op.version = 23
    m.opset_import.append(helper.make_opsetid('trt', 1))
    m.ir_version = max(m.ir_version, 10)
    onnx.save(m, out_path)
    print(f'[fp4] weights quantized: {n_w}; mean rel |w-w_dq|/|w| = {np.mean(wstats):.4f} '
          f'(max {np.max(wstats):.4f}); activation dynq nodes: {len(act_done)}')
    print(f'[fp4] act amax range: {min(amax.values()):.3f} .. {max(amax.values()):.3f}')
    print(f'[fp4] -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('onnx')
    ap.add_argument('tpg_dir')
    ap.add_argument('--run_config', default=None)
    ap.add_argument('--run_id', default=None)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--calib_batches', type=int, default=8)
    ap.add_argument('--num_batches', type=int, default=30)
    ap.add_argument('--min_n', type=int, default=128, help='skip MatMuls with N below this')
    ap.add_argument('--exclude_pattern', default=None)
    ap.add_argument('--scale_dtype', choices=['fp16', 'fp32'], default='fp16')
    ap.add_argument('--act_headroom', type=float, default=1.0,
                    help='multiply calibrated activation amax (global scale) by this')
    ap.add_argument('--only_dims', default=None,
                    help='comma list; quantize only MatMuls whose K and N are both in this set '
                         '(e.g. 640,1920 = trunk attention/FFN only)')
    ap.add_argument('--weights_only', action='store_true',
                    help='FP4 weights only, activations stay FP16 (accuracy diagnosis; no FP4 GEMM speed)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--no_verify', action='store_true')
    ap.add_argument('--eval_only', default=None,
                    help='skip conversion; build+compare THIS quantized onnx against the fp16 <onnx>')
    ap.add_argument('--weak', action='store_true',
                    help='build the FP4 engine weakly typed (FP16+FP4 flags) instead of strongly typed')
    args = ap.parse_args()

    base = os.path.splitext(args.onnx)[0]
    if args.eval_only:
        out = args.eval_only
    else:
        out = args.out or (base + '.nvfp4.onnx')
        only = [int(x) for x in args.only_dims.split(',')] if args.only_dims else None
        convert(args.onnx, out, args.tpg_dir, args.batch, args.calib_batches, args.min_n,
                args.exclude_pattern, args.scale_dtype, args.act_headroom, only, args.weights_only)
    if args.no_verify:
        return

    import int8_validate as iv
    trt = iv.trt
    fp16ref = base + '.fp16ref.engine'
    fp4eng = os.path.splitext(out)[0] + '.engine'

    def build(onnx_path, out_path, fp4=False):
        if os.path.exists(out_path):
            print(f'[build] {out_path} exists, skip'); return
        builder = trt.Builder(iv.LOG)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        if fp4 and not args.weak:
            flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, iv.LOG)
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print('[parse-err]', parser.get_error(i))
                raise RuntimeError('parse failed')
        feat = int(network.get_input(0).shape[-1])
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 33)
        if not (fp4 and not args.weak):
            cfg.set_flag(trt.BuilderFlag.FP16)
            if fp4:
                cfg.set_flag(trt.BuilderFlag.FP4)
        prof = builder.create_optimization_profile()
        prof.set_shape('squares', (1, 64, feat), (args.batch, 64, feat), (256, 64, feat))
        cfg.add_optimization_profile(prof)
        ser = builder.build_serialized_network(network, cfg)
        if ser is None:
            raise RuntimeError('engine build failed')
        with open(out_path, 'wb') as f:
            f.write(bytes(ser))
        print(f'[build] saved {len(bytes(ser))//(1024*1024)} MB -> {out_path}')

    build(args.onnx, fp16ref)
    build(out, fp4eng, fp4=True)
    fp16 = iv.Runner(fp16ref); q = iv.Runner(fp4eng)
    ds = TPGDataset(args.tpg_dir, args.batch, 0.0, 0, 1, 0, 1, 0, False)
    print('\n=== NVFP4 vs FP16 precision ===')
    r = qx.compare_mixed(fp16, q, ds, args.batch, args.num_batches)
    print(f"Positions: {r['n']}")
    print(f"Policy top-1 agreement      : {r['top1_pct']:.2f}%")
    print(f"Policy top-3 agreement (>=2): {r['top3_pct']:.2f}%")
    print(f"Policy KL(FP16 || FP4) mean : {r['kl_mean']:.5f}")
    print(f"Value softmax L1 mean       : {r['val_l1_mean']:.4f}")
    print(f"Value WDL argmax agreement  : {r['val_argmax_pct']:.2f}%")
    print('\n=== Speed ===')
    raw = ds[0][0]['squares'].numpy()
    fp16.prepare({'squares': raw.astype(qx._in_dtype(fp16))})
    q.prepare({'squares': raw.astype(qx._in_dtype(q))})
    fp_ms, fp_thr = iv.bench('FP16', fp16, args.batch)
    q_ms, q_thr = iv.bench('FP4 ', q, args.batch)
    print(f'\nFP4 / FP16 latency speedup: {fp_ms/q_ms:.3f}x  ({(q_thr/fp_thr-1)*100:+.1f}% throughput)')


if __name__ == '__main__':
    main()
