"""QDQ (explicit) INT8 quantization prototype for CeresNet ONNX — Route A.

WHY THIS EXISTS
  The implicit IInt8EntropyCalibrator2 path (int8_validate.py) builds a BROKEN
  engine on decomposed-norm (opset-18) nets: TRT discards every scale
  ("Dequantize NNN [SCALE] has invalid precision Int8, ignored"), yielding NO
  speedup (~0.99x) and garbage policy/value. That is NOT an arch verdict — it is
  the deprecated implicit calibrator failing to place dynamic ranges across the
  decomposed norm's many ReduceMean/Pow/Sqrt/Div ops. TRT 10.1 itself says
  "superseded by explicit quantization". This script does explicit quantization:
  it inserts QuantizeLinear/DequantizeLinear (QDQ) nodes into the graph via ORT
  quantize_static so TRT honors the scales deterministically.

SELECTIVE BY CONSTRUCTION
  op_types_to_quantize=['MatMul'] -> only the GEMMs go INT8; norms / softmax /
  residual adds / value-head elementwise ops stay in FP precision (the
  precision-pinning lesson from the fused-RMSNorm FP16 overflow fix).

PIPELINE
  1. Lossless FP16 -> FP32 (ORT quantize_static requires FP32; served net is FP16).
  2. quant_pre_process (symbolic shape inference) for clean QDQ insertion.
  3. quantize_static (QDQ, QInt8 weights+activations, per-channel, Entropy calib
     on real TPG positions).
  4. Build a TRT engine from the QDQ onnx WITHOUT a calibrator (QDQ drives INT8),
     plus an FP16 reference, and report KLD / WDL-argmax / speedup vs FP16.

USAGE (WSL, cerestrain-env with tensorrt + cuda-python + onnxruntime):
  CERES_AUX_FEATURES_PER_SQUARE=4 python3 qdq_export.py \
      <fp16_onnx> <tpg_dir> [--calib_batches 16] [--num_batches 30] [--batch 64]

Outputs next to the input onnx:
  <net>.fp32.onnx   (lossless intermediate)
  <net>.qdq.onnx    (the QDQ INT8 graph — feed to Ceres ORT-TRT EP)
  <net>.qdq.engine + <net>.fp16ref.engine (for the comparison)
"""
import argparse, os, sys
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto

# Reuse the proven TRT Runner / compare / bench harness + TPGDataset wiring.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CERES_PY = '/mnt/c/Users/lepne/source/repos/CeresTrain/src/CeresTrainPy'
sys.path.insert(0, os.environ.get('CERES_PY_DIR', DEFAULT_CERES_PY))
from tpg_dataset import TPGDataset
from onnxruntime.quantization import (quantize_static, QuantType, QuantFormat,
                                      CalibrationDataReader, CalibrationMethod)


def fp16_to_fp32(in_path, out_path):
    """Lossless FP16->FP32 cast of a whole ONNX graph (FP16 subset of FP32)."""
    m = onnx.load(in_path)
    g = m.graph
    n_init = 0
    for init in g.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new = numpy_helper.from_array(arr, init.name)
            init.CopyFrom(new)
            n_init += 1
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        tt = vi.type.tensor_type
        if tt.elem_type == TensorProto.FLOAT16:
            tt.elem_type = TensorProto.FLOAT
    n_cast = n_const = 0
    for node in g.node:
        if node.op_type == 'Cast':
            for attr in node.attribute:
                if attr.name == 'to' and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
                    n_cast += 1
        # Convert ANY FP16 tensor-valued attribute (Constant 'value',
        # ConstantOfShape 'value', etc.) to FP32 — older exports embed FP16 here.
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT16:
                arr = numpy_helper.to_array(attr.t).astype(np.float32)
                attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))
                n_const += 1
            elif attr.type == onnx.AttributeProto.TENSORS:
                for t in attr.tensors:
                    if t.data_type == TensorProto.FLOAT16:
                        arr = numpy_helper.to_array(t).astype(np.float32)
                        t.CopyFrom(numpy_helper.from_array(arr, t.name))
                        n_const += 1
    onnx.save(m, out_path)
    print(f'[fp16->fp32] inits={n_init} cast_nodes={n_cast} tensor_attrs={n_const} -> {out_path}')


def restore_fp16_io(in_path, out_path):
    """Set graph IO back to FP16 (Ceres feeds/reads FP16) while leaving the
    internal FP32 + QDQ graph untouched, by inserting boundary Cast nodes:
      input:  FP16 graph input -> Cast(to FP32) -> (original consumers)
      output: (FP32 producer) -> Cast(to FP16) -> FP16 graph output
    """
    from onnx import helper
    m = onnx.load(in_path)
    g = m.graph
    n_in = n_out = 0
    for inp in g.input:
        tt = inp.type.tensor_type
        if tt.elem_type != TensorProto.FLOAT:
            continue
        orig = inp.name
        cast_out = orig + '_to_fp32'
        for node in g.node:
            for k, nm in enumerate(node.input):
                if nm == orig:
                    node.input[k] = cast_out
        g.node.insert(0, helper.make_node('Cast', [orig], [cast_out],
                                          to=TensorProto.FLOAT, name=orig + '_castfp32'))
        tt.elem_type = TensorProto.FLOAT16
        n_in += 1
    for out in g.output:
        tt = out.type.tensor_type
        if tt.elem_type != TensorProto.FLOAT:
            continue
        orig = out.name
        src = orig + '_pre_fp16'
        for node in g.node:
            for k, nm in enumerate(node.output):
                if nm == orig:
                    node.output[k] = src
        # Internal consumers of the output tensor (dynamo emits Identity alias
        # chains, e.g. on 'mlh') must keep reading the pre-Cast FP32 value —
        # otherwise they'd consume the FP16 Cast output (dtype break + topo
        # violation).
        for node in g.node:
            for k, nm in enumerate(node.input):
                if nm == orig:
                    node.input[k] = src
        g.node.append(helper.make_node('Cast', [src], [orig],
                                       to=TensorProto.FLOAT16, name=orig + '_castfp16'))
        tt.elem_type = TensorProto.FLOAT16
        n_out += 1
    onnx.save(m, out_path)
    print(f'[fp16-io] inserted {n_in} input + {n_out} output Casts -> {out_path}')


class TPGReader(CalibrationDataReader):
    """Feeds calibration positions in the format the net's input expects.
    input_name : the graph input tensor name ('squares' FP, or 'squares_byte' uint8).
    channels   : slice tpg 'squares'[:, :, :channels] (V3 141 -> V2 137).
    byte_div   : >0 -> emit uint8 = round(squares[:ch] * byte_div) (byte-input nets,
                 where the net does Cast->Div(byte_div); C1-640-34 uses 100).
    """
    def __init__(self, tpg_dir, batch, nbatches, input_name='squares', channels=None, byte_div=0):
        self.ds = TPGDataset(tpg_dir, batch, 0.0, 0, 1, 0, 1, 0, False)
        self.batch = batch
        self.n = nbatches
        self.i = 0
        self.input_name = input_name
        self.channels = channels
        self.byte_div = byte_div

    def _fmt(self, b):
        if self.channels is not None:
            b = b[:, :, :self.channels]
        if self.byte_div > 0:
            return np.clip(np.rint(b.astype(np.float32) * self.byte_div), 0, 255).astype(np.uint8)
        return b.astype(np.float32)

    def get_next(self):
        if self.i >= self.n:
            return None
        b = self.ds[0][0]['squares'].numpy()
        if b.shape[0] != self.batch:
            return None
        self.i += 1
        return {self.input_name: self._fmt(b)}

    def rewind(self):
        self.i = 0


def _in_dtype(runner):
    """Numpy dtype the engine expects for its input tensor (FP16 ref vs FP32 QDQ)."""
    import int8_validate as iv
    return np.dtype(iv.trt.nptype(runner.eng.get_tensor_dtype(runner.inames[0])))


def compare_mixed(fp16, qdq, ds, batch, num_batches):
    """Like int8_validate.compare but feeds each engine its own input dtype
    (the QDQ graph keeps FP32 IO; the FP16 reference keeps FP16 IO)."""
    import int8_validate as iv
    dt_f, dt_q = _in_dtype(fp16), _in_dtype(qdq)
    top1 = top3 = val_argmax = n = 0
    kl_sum = val_l1_sum = 0.0
    for _ in range(num_batches):
        b = ds[0][0]['squares'].numpy()
        if b.shape[0] != batch:
            continue
        of = fp16.prepare({'squares': b.astype(dt_f)}); fp16.infer(); fp16.copy_outputs(of)
        oi = qdq.prepare({'squares': b.astype(dt_q)}); qdq.infer(); qdq.copy_outputs(oi)
        pf = of['policy'].astype(np.float32); pi = oi['policy'].astype(np.float32)
        v_key = 'value' if 'value' in of else 'value2'
        vf = of[v_key].astype(np.float32); vi = oi[v_key].astype(np.float32)
        top1 += int((pf.argmax(-1) == pi.argmax(-1)).sum())
        t3f = np.argpartition(-pf, 3, -1)[:, :3]; t3i = np.argpartition(-pi, 3, -1)[:, :3]
        for k in range(batch):
            if len(set(t3f[k]) & set(t3i[k])) >= 2:
                top3 += 1
        kl_sum += iv.kl(iv.softmax(pf), iv.softmax(pi)) * batch
        val_l1_sum += float(np.abs(iv.softmax(vf) - iv.softmax(vi)).sum(-1).mean()) * batch
        val_argmax += int((vf.argmax(-1) == vi.argmax(-1)).sum())
        n += batch
    return dict(n=n, top1_pct=100*top1/n, top3_pct=100*top3/n, kl_mean=kl_sum/n,
                val_l1_mean=val_l1_sum/n, val_argmax_pct=100*val_argmax/n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('onnx', help='FP16-converted CeresNet ONNX (decomposed norm, opset 18)')
    ap.add_argument('tpg_dir')
    ap.add_argument('--calib_batches', type=int, default=8)
    ap.add_argument('--num_batches', type=int, default=30)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--method', choices=['minmax', 'entropy', 'percentile'], default='minmax',
                    help='minmax = scale to max (outliers dominate); percentile = clip '
                         'outliers at --percentile (recommended to reduce quality loss); '
                         'entropy = histograms (OOM-prone)')
    ap.add_argument('--percentile', type=float, default=99.999,
                    help='percentile for --method percentile (clip activation outliers; '
                         'lower = more aggressive clipping, e.g. 99.99)')
    ap.add_argument('--exclude_tail', type=int, default=0,
                    help='exclude the last N MatMul nodes from quantization (keep the '
                         'value-feeding late-trunk path FP16; value head is INT8-hostile)')
    ap.add_argument('--exclude_act_matmuls', action='store_true',
                    help='exclude activation*activation MatMuls (attention QK^T / A*V) '
                         'from quantization. QAT fake-quant only simulates the weight '
                         'GEMMs (nn.Linear), so quantizing these at export is '
                         'train/deploy skew; excluding them deploys exactly what QAT '
                         'trained against.')
    ap.add_argument('--out', default=None,
                    help='explicit path for the deployable fp16io onnx (default: '
                         '<base>.<tag>.fp16io.onnx). Use to keep A/B export variants '
                         'side by side without overwriting.')
    ap.add_argument('--train_ranges', default=None,
                    help='QAT checkpoint path: use its frozen _fq_act_range buffers '
                         'as the activation Q/DQ scales (scale=range/127) instead of '
                         'recalibrating. A QAT net co-adapts to the exact per-tensor '
                         'clip pattern; recalibration breaks it (KL 0.02 -> 0.32). '
                         'Modules are matched to MatMuls by fp16 weight-content hash '
                         '(dynamo anonymizes initializer names).')
    ap.add_argument('--precision', choices=['int8', 'fp8'], default='int8',
                    help='int8 = fixed-point (erases small value signal); fp8 = E4M3 '
                         'floating-point (preserves small magnitudes -> value should survive)')
    ap.add_argument('--byte_divisor', type=int, default=0,
                    help='byte-input nets (uint8 squares_byte that the net does '
                         'Cast->Div(N) on): emit calib as uint8 = round(squares*N). '
                         'C1-640-34 = 100. 0 = FP input (default).')
    ap.add_argument('--no_verify', action='store_true',
                    help='skip the standalone TRT build+compare (invalid on bad input / '
                         'slow on big nets); just emit the deployable QDQ onnx.')
    args = ap.parse_args()

    # Auto-detect the input tensor (name / channels) from the ONNX so this works
    # for both 'squares' FP16/141-ch (V3) and 'squares_byte' uint8/137-ch (V2) nets.
    _im = onnx.load(args.onnx)
    _in0 = _im.graph.input[0]
    in_name = _in0.name
    in_channels = _in0.type.tensor_type.shape.dim[-1].dim_value or None
    print(f'[input] name={in_name} channels={in_channels} byte_divisor={args.byte_divisor}')

    # FP8 (E4M3) is floating-point, symmetric. Round-1 FP8 used per-TENSOR for
    # weights too (one scale per whole weight matrix) and value collapsed even
    # strongly-typed; the working INT8 path is per-CHANNEL weights. Try
    # per-channel for FP8 as well (round 2).
    FP8 = args.precision == 'fp8'
    qtype = QuantType.QFLOAT8E4M3FN if FP8 else QuantType.QInt8
    per_channel = True
    tag = 'qdqfp8' if FP8 else 'qdq'

    base = os.path.splitext(args.onnx)[0]
    fp32_path = base + '.fp32.onnx'
    pre_path = base + '.fp32.pre.onnx'
    qdq_path = base + '.' + tag + '.onnx'

    # 1. lossless FP16 -> FP32
    fp16_to_fp32(args.onnx, fp32_path)

    # 2. shape inference / preprocessing for clean QDQ insertion.
    #    ORT symbolic shape inference asserts on this graph (an aux output type
    #    proto is neither tensor nor sequence), so skip it; fall back to the
    #    plain FP32 model if even the lite preprocess fails. quantize_static
    #    still runs onnx shape inference internally.
    quant_input = fp32_path
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(fp32_path, pre_path, skip_symbolic_shape=True)
        quant_input = pre_path
        print(f'[pre] quant_pre_process (skip_symbolic_shape) -> {pre_path}')
    except Exception as e:
        print(f'[pre] quant_pre_process skipped ({type(e).__name__}: {e}); using FP32 model directly')

    # 3. explicit QDQ static quantization (MatMul only -> norms/value-head stay FP)
    cmethod = {'minmax': CalibrationMethod.MinMax,
               'entropy': CalibrationMethod.Entropy,
               'percentile': CalibrationMethod.Percentile}[args.method]
    # smaller calibration batch keeps the activation-range collection in memory
    calib_batch = min(args.batch, 16)
    reader = TPGReader(args.tpg_dir, calib_batch, args.calib_batches,
                       input_name=in_name, channels=in_channels, byte_div=args.byte_divisor)
    # Exclude the last N MatMuls (the value-feeding late trunk) from quantization.
    nodes_to_exclude = []
    _m = onnx.load(quant_input) if (args.exclude_tail > 0 or args.exclude_act_matmuls) else None
    if args.exclude_tail > 0:
        _mm = [n.name for n in _m.graph.node if n.op_type == 'MatMul']
        nodes_to_exclude = _mm[-args.exclude_tail:]
        print(f'[qdq] excluding last {args.exclude_tail}/{len(_mm)} MatMuls from quant '
              f'(first excl {nodes_to_exclude[0]}, last {nodes_to_exclude[-1]})')
    if args.exclude_act_matmuls:
        # A MatMul input is "weight-like" if it is an initializer, or the output of
        # a constant-foldable chain (Transpose/Cast/Reshape/... of initializers).
        # MatMuls with NO weight-like input are the attention QK^T / A*V ops.
        const_like = {i.name for i in _m.graph.initializer}
        for n in _m.graph.node:
            if n.op_type in ('Transpose', 'Cast', 'Reshape', 'Unsqueeze', 'Squeeze',
                             'Identity', 'Constant') and \
               all((x in const_like) or (x == '') for x in n.input):
                const_like.update(n.output)
        _total = sum(1 for n in _m.graph.node if n.op_type == 'MatMul')
        act_mm = [n.name for n in _m.graph.node
                  if n.op_type == 'MatMul' and all(x not in const_like for x in n.input)]
        nodes_to_exclude += [n for n in act_mm if n not in nodes_to_exclude]
        print(f'[qdq] excluding {len(act_mm)}/{_total} activation*activation MatMuls '
              f'(attention QK^T/A*V) from quant')
    # FP8 in ORT requires Distribution calibration (histogram-based scale).
    if FP8:
        cmethod = CalibrationMethod.Distribution
    # TRT only supports SYMMETRIC quant (zero_point=0); FP8 is inherently
    # symmetric. ORT defaults activations to asymmetric for int8 -> TRT parse
    # fails 'Non-zero zero point'. Force symmetric on both.
    extra_opts = {'ActivationSymmetric': True, 'WeightSymmetric': True}
    if args.method == 'percentile':
        extra_opts['percentile'] = args.percentile  # clip activation outliers (ORT key)
    # --train_ranges: pin activation scales to the QAT ckpt's frozen ranges.
    if args.train_ranges:
        import hashlib
        import torch as _torch
        _sd = {k.replace('_forward_module._orig_mod.', ''): v
               for k, v in _torch.load(args.train_ranges, map_location='cpu',
                                       weights_only=False)['model'].items()}
        _ranges = {k[:-len('._fq_act_range')]: float(v.float().item())
                   for k, v in _sd.items() if k.endswith('_fq_act_range')}
        _whash = {}
        for _mod, _r in _ranges.items():
            _w = _sd.get(_mod + '.weight')
            if _w is None:
                continue
            _a = _w.to(_torch.float16).numpy()
            _whash[hashlib.sha1(_a.tobytes()).hexdigest()] = (_mod, _r)
            _whash[hashlib.sha1(_a.T.copy().tobytes()).hexdigest()] = (_mod, _r)
        if _m is None:
            _m = onnx.load(quant_input)
        _inits = {i.name: i for i in _m.graph.initializer}
        _const = set(_inits)
        _prod = {}
        for _n in _m.graph.node:
            for _o in _n.output:
                _prod[_o] = _n
            if _n.op_type in ('Transpose', 'Cast', 'Reshape', 'Unsqueeze',
                              'Squeeze', 'Identity', 'Constant') and \
               all((x in _const) or (x == '') for x in _n.input):
                _const.update(_n.output)
        def _w_init(name, hops=6):
            while name not in _inits and hops > 0:
                _p = _prod.get(name)
                if _p is None or not _p.input:
                    return None
                name = _p.input[0]; hops -= 1
            return _inits.get(name)
        _tens_ranges = {}   # act tensor -> list of ranges (multi-consumer tensors)
        _matched = _unmatched = 0
        for _n in _m.graph.node:
            if _n.op_type != 'MatMul':
                continue
            _ws = [x for x in _n.input if x in _const]
            _as = [x for x in _n.input if x not in _const]
            if not _ws or not _as:
                continue
            _init = _w_init(_ws[0])
            if _init is None:
                _unmatched += 1; continue
            _arr = numpy_helper.to_array(_init).astype(np.float16)
            _hit = _whash.get(hashlib.sha1(_arr.tobytes()).hexdigest())
            if _hit is None:
                _unmatched += 1; continue
            _tens_ranges.setdefault(_as[0], []).append(_hit[1])
            _matched += 1
        # ORT quirk: the validator accepts plain int/float but calc_quant_params
        # calls .squeeze() on the values — 0-d numpy arrays satisfy both.
        _overrides = {t: [{'scale': np.array(sum(rs) / len(rs) / 127.0, dtype=np.float32),
                           'zero_point': np.array(0, dtype=np.int8)}]
                      for t, rs in _tens_ranges.items()}
        extra_opts['TensorQuantOverrides'] = _overrides
        print(f'[qdq] train_ranges: {len(_ranges)} ckpt ranges; matched {_matched} '
              f'weight-MatMuls -> {len(_overrides)} tensor overrides; unmatched {_unmatched}')
    print(f'[qdq] calibrating: precision={args.precision} method={cmethod} '
          f'per_channel={per_channel} calib_batch={calib_batch} x {args.calib_batches} batches'
          + (f' percentile={args.percentile}' if args.method == 'percentile' else ''))
    quantize_static(
        quant_input, qdq_path, reader,
        quant_format=QuantFormat.QDQ,
        activation_type=qtype,
        weight_type=qtype,
        per_channel=per_channel,
        calibrate_method=cmethod,
        op_types_to_quantize=['MatMul'],
        nodes_to_exclude=nodes_to_exclude,
        extra_options=extra_opts,
    )
    print(f'[qdq] quantize_static ({args.precision}) -> {qdq_path}')
    # Count inserted QDQ nodes as a sanity check.
    qm = onnx.load(qdq_path)
    qn = sum(1 for n in qm.graph.node if n.op_type in ('QuantizeLinear', 'DequantizeLinear'))
    print(f'[qdq] inserted Q/DQ nodes: {qn}')

    # Restore FP16 IO so the artifact is Ceres-TensorRTNative-deployable
    # (Ceres feeds/reads FP16). This is the deployable net.
    qdq_deploy = args.out if args.out else (base + '.' + tag + '.fp16io.onnx')
    restore_fp16_io(qdq_path, qdq_deploy)
    qdq_path = qdq_deploy
    print(f'[deploy] {qdq_deploy}')

    if args.no_verify:
        print('[done] --no_verify: skipped standalone build/compare. '
              'Validate value via the winning-FEN test through Ceres (the only '
              'trustworthy check; TPGDataset-squares standalone = bad input).')
        return

    # 4. Build TRT engines (FP16 ref + QDQ) and compare. Reuse int8_validate harness.
    import int8_validate as iv
    trt = iv.trt
    fp16ref = base + '.fp16ref.engine'
    qdqeng = base + '.qdq.engine'

    def build(onnx_path, out_path, is_qdq=False):
        if os.path.exists(out_path):
            print(f'[build] {out_path} exists, skip'); return
        builder = trt.Builder(iv.LOG)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, iv.LOG)
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print('[parse-err]', parser.get_error(i))
                raise RuntimeError('parse failed')
        feat = int(network.get_input(0).shape[-1])
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 33)
        cfg.set_flag(trt.BuilderFlag.FP16)  # FP16 for non-quantized layers
        if is_qdq:
            # Without kINT8, TRT honors Q/DQ numerically but may not select INT8
            # GEMM tactics -> no speedup. Enable INT8 to allow INT8 kernels.
            cfg.set_flag(trt.BuilderFlag.INT8)
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
    build(qdq_path, qdqeng, is_qdq=True)

    fp16 = iv.Runner(fp16ref); qdq = iv.Runner(qdqeng)
    ds = TPGDataset(args.tpg_dir, args.batch, 0.0, 0, 1, 0, 1, 0, False)
    print('\n=== QDQ vs FP16 precision ===')
    r = compare_mixed(fp16, qdq, ds, args.batch, args.num_batches)
    print(f"Positions: {r['n']}")
    print(f"Policy top-1 agreement      : {r['top1_pct']:.2f}%")
    print(f"Policy top-3 agreement (>=2): {r['top3_pct']:.2f}%")
    print(f"Policy KL(FP16 || QDQ) mean : {r['kl_mean']:.5f}")
    print(f"Value softmax L1 mean       : {r['val_l1_mean']:.4f}")
    print(f"Value WDL argmax agreement  : {r['val_argmax_pct']:.2f}%")
    if r['kl_mean'] < 0.05 and r['val_argmax_pct'] > 99:
        print('VERDICT: GREEN — QDQ INT8 preserves policy+value.')
    elif r['kl_mean'] < 0.20:
        print('VERDICT: AMBER — borderline; consider weight-only or more exclusions.')
    else:
        print('VERDICT: RED — QDQ INT8 unstable.')

    print('\n=== Speed ===')
    raw = ds[0][0]['squares'].numpy()
    fp16.prepare({'squares': raw.astype(_in_dtype(fp16))})
    qdq.prepare({'squares': raw.astype(_in_dtype(qdq))})
    fp_ms, fp_thr = iv.bench('FP16', fp16, args.batch)
    q_ms, q_thr = iv.bench('QDQ ', qdq, args.batch)
    print(f'\nQDQ / FP16 latency speedup: {fp_ms/q_ms:.3f}x  ({(q_thr/fp_thr-1)*100:+.1f}% throughput)')


if __name__ == '__main__':
    main()
