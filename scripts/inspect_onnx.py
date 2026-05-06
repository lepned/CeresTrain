import sys, onnx, onnx.numpy_helper

path = sys.argv[1]
m = onnx.load(path)
g = m.graph
print(f"=== {path} ===")
print(f"IR version: {m.ir_version}")
print(f"Producer: {m.producer_name} {m.producer_version}")
print(f"Opsets: {[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]}")
print(f"\nInputs:")
for i in g.input:
    dims = [d.dim_value or d.dim_param or "?" for d in i.type.tensor_type.shape.dim]
    print(f"  {i.name}: {onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type)} {dims}")
print(f"\nOutputs:")
for o in g.output:
    dims = [d.dim_value or d.dim_param or "?" for d in o.type.tensor_type.shape.dim]
    print(f"  {o.name}: {onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type)} {dims}")
print(f"\nNodes: {len(g.node)}, Initializers: {len(g.initializer)}")
ops = {}
for n in g.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print(f"\nOp counts (top 40):")
for k, v in sorted(ops.items(), key=lambda x: -x[1])[:40]:
    print(f"  {k}: {v}")
print(f"Total unique ops: {len(ops)}")

total_params = 0
total_bytes = 0
dtype_counts = {}
biggest = []
for init in g.initializer:
    arr = onnx.numpy_helper.to_array(init)
    total_params += arr.size
    total_bytes += arr.nbytes
    dt = str(arr.dtype)
    dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
    biggest.append((arr.size, init.name, arr.shape, dt))
biggest.sort(reverse=True)
print(f"\nParameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"Init bytes: {total_bytes/1024/1024:.1f} MiB")
print(f"Init dtypes: {dtype_counts}")
print(f"\nBiggest 10 initializers:")
for sz, n, sh, dt in biggest[:10]:
    print(f"  {sz:>12,}  {dt:<10}  {sh}  {n}")
