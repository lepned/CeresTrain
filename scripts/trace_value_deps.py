"""Trace which MatMul nodes the VALUE output depends on vs the POLICY output,
to decide if selective INT8 quantization can preserve value:

  value_mm   = MatMuls that 'value' depends on (must stay FP16 to protect value)
  policy_mm  = MatMuls that 'policy' depends on
  policy_only = policy_mm - value_mm  (the ONLY MatMuls we can quantize without
               touching value's dependency path)

If policy_only is large -> selective INT8 viable (quantize policy_only, keep the
rest FP16). If policy_only is empty/tiny -> value & policy share the trunk ->
no value-preserving speedup (inherent tradeoff).

Usage: python3 trace_value_deps.py <onnx>
Prints the exclude-list (value_mm) so it can be fed to qdq_export nodes_to_exclude.
"""
import sys, onnx, json

m = onnx.load(sys.argv[1])
g = m.graph
producer = {}
for n in g.node:
    for o in n.output:
        producer[o] = n
init = {i.name for i in g.initializer}


def ancestor_matmuls(out_name):
    """All MatMul node names that are ancestors of the given graph output."""
    seen_nodes = set()
    mm = set()
    stack = [out_name]
    seen_t = set()
    while stack:
        t = stack.pop()
        if t in seen_t or t in init:
            continue
        seen_t.add(t)
        nd = producer.get(t)
        if nd is None or id(nd) in seen_nodes:
            continue
        seen_nodes.add(id(nd))
        if nd.op_type == 'MatMul':
            mm.add(nd.name)
        for inp in nd.input:
            stack.append(inp)
    return mm


outs = [o.name for o in g.output]
print('outputs:', outs)
value_mm = ancestor_matmuls('value') if 'value' in outs else set()
value2_mm = ancestor_matmuls('value2') if 'value2' in outs else set()
policy_mm = ancestor_matmuls('policy') if 'policy' in outs else set()
all_mm = {n.name for n in g.node if n.op_type == 'MatMul'}

val_all = value_mm | value2_mm
policy_only = policy_mm - val_all
print(f'\ntotal MatMuls            : {len(all_mm)}')
print(f'value (value+value2) deps: {len(val_all)}')
print(f'policy deps              : {len(policy_mm)}')
print(f'policy-ONLY (quantizable): {len(policy_only)}')
print(f'shared (value & policy)  : {len(policy_mm & val_all)}')

# what fraction of compute could be quantized if we protect value?
print(f'\n=> If we keep value-deps FP16, we can quantize {len(policy_only)}/{len(all_mm)} MatMuls')
# dump the exclude list (value deps) for qdq_export
with open(sys.argv[1] + '.value_exclude.json', 'w') as f:
    json.dump(sorted(val_all), f)
print(f'wrote value-exclude list ({len(val_all)} nodes) -> {sys.argv[1]}.value_exclude.json')
