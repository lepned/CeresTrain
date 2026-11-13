"""Reader and writer for lc0ex weight carriers (`*_lc0ex_carrier.pb.gz`), dependency-free (numpy only).

A carrier is a gzip-compressed Lc0 `Net` protobuf whose field 11 (`onnx_model`, sub-field 1 = `model`) holds an
ONNX ModelProto with initializers only. The lc0ex runtime indexes the initializers by NAME and requires, for every
persistent buffer a compiled program declares, an initializer with the same name, data type, shape and byte size.
Nothing else of the model is read, so this module walks the protobuf wire format directly.

    tensors, net = read_carrier(path)                # {name: numpy array}, the undecoded Net bytes
    write_carrier(path, tensors, template_net=net)   # every other Net field copied through byte for byte
"""

from __future__ import annotations

import gzip

import numpy as np

# ONNX TensorProto.DataType -> numpy
ONNX_DTYPES = {1: np.dtype("<f4"), 2: np.dtype("u1"), 3: np.dtype("i1"), 6: np.dtype("<i4"), 7: np.dtype("<i8"),
               10: np.dtype("<f2"), 11: np.dtype("<f8"), 13: np.dtype("<u8")}
NUMPY_TO_ONNX = {dtype: code for code, dtype in ONNX_DTYPES.items()}

_VARINT, _FIXED64, _LENGTH, _FIXED32 = 0, 1, 2, 5
_NET_ONNX_MODEL, _ONNX_MODEL_MODEL = 11, 1
_MODEL_IR_VERSION, _MODEL_OPSET, _MODEL_GRAPH = 1, 8, 7
_GRAPH_NAME, _GRAPH_INITIALIZER = 2, 5
_TENSOR_DIMS, _TENSOR_DATA_TYPE, _TENSOR_NAME, _TENSOR_RAW_DATA = 1, 2, 8, 9

# The non-ONNX fields of the BT6 flagship carrier's Net (magic, min_version, format) and the OnnxModel side fields
# (data type, input / output names), used when `write_carrier` gets no template.
_DEFAULT_NET_PREFIX = (b"\x0d\xc0\x01\x00\x00" + b"\x1a\x06\x08\x00\x10\x1f\x18\x00"
                       + b"\x22\x16\x12\x14\x08\x01\x10\x02\x18\x07\x20\x03\x28\x02\x30\x01\x38\x01\x40\x07\x48\x01\x50\x02")
_DEFAULT_ONNX_SUFFIX = (b"\x10\x01" + b"\x1a\x0d/input/planes" + b"\x2a\x0b/output/wdl" + b"\x32\x0e/output/policy"
                        + b"\x3a\x0b/output/mlh")


def _read_varint(buffer, offset):
  result, shift = 0, 0
  while True:
    byte = buffer[offset]
    offset += 1
    result |= (byte & 0x7F) << shift
    if not byte & 0x80:
      return result, offset
    shift += 7


def _fields(buffer):
  """Yield (field number, wire type, payload) of one protobuf message held in a memoryview."""
  offset, limit = 0, len(buffer)
  while offset < limit:
    key, offset = _read_varint(buffer, offset)
    number, wire = key >> 3, key & 7
    if wire == _VARINT:
      value, offset = _read_varint(buffer, offset)
      yield number, wire, value
    elif wire == _LENGTH:
      length, offset = _read_varint(buffer, offset)
      yield number, wire, buffer[offset:offset + length]
      offset += length
    elif wire == _FIXED64:
      yield number, wire, buffer[offset:offset + 8]
      offset += 8
    elif wire == _FIXED32:
      yield number, wire, buffer[offset:offset + 4]
      offset += 4
    else:
      raise ValueError(f"unsupported protobuf wire type {wire}")


def _varint(value: int) -> bytes:
  out = bytearray()
  while True:
    byte = value & 0x7F
    value >>= 7
    if value:
      out.append(byte | 0x80)
    else:
      out.append(byte)
      return bytes(out)


def _tag(number: int, wire: int) -> bytes:
  return _varint(number << 3 | wire)


def _length_header(number: int, size: int) -> bytes:
  return _tag(number, _LENGTH) + _varint(size)


def _reencode(number, wire, value) -> bytes:
  if wire == _VARINT:
    return _tag(number, wire) + _varint(int(value))
  if wire == _LENGTH:
    return _length_header(number, len(value)) + bytes(value)
  return _tag(number, wire) + bytes(value)


def _decode_tensor(payload):
  dims, data_type, name, raw = [], 0, None, None
  for number, wire, value in _fields(payload):
    if number == _TENSOR_DIMS and wire == _VARINT:
      dims.append(value)
    elif number == _TENSOR_DIMS and wire == _LENGTH:
      offset = 0
      while offset < len(value):
        dim, offset = _read_varint(value, offset)
        dims.append(dim)
    elif number == _TENSOR_DATA_TYPE and wire == _VARINT:
      data_type = value
    elif number == _TENSOR_NAME and wire == _LENGTH:
      name = bytes(value).decode("utf8")
    elif number == _TENSOR_RAW_DATA and wire == _LENGTH:
      raw = value
  return name, data_type, tuple(dims), raw


def extract_onnx_model(net):
  """The memoryview of the ONNX ModelProto inside a serialized Net."""
  for number, wire, value in _fields(memoryview(net)):
    if number == _NET_ONNX_MODEL and wire == _LENGTH:
      for inner, inner_wire, payload in _fields(value):
        if inner == _ONNX_MODEL_MODEL and inner_wire == _LENGTH:
          return payload
  raise ValueError("the network carries no onnx_model.model")


def read_carrier(path: str, keep=None):
  """Return ({name: array}, net bytes). Arrays are read-only views into the decompressed file, typed and shaped
  as declared. `keep(name) -> bool` filters the initializers (e.g. to skip the int8 copies)."""
  with gzip.open(path, "rb") as stream:
    net = stream.read()
  model = extract_onnx_model(net)
  graph = None
  for number, wire, value in _fields(model):
    if number == _MODEL_GRAPH and wire == _LENGTH:
      graph = value
      break
  if graph is None:
    raise ValueError("the ONNX model carries no graph")
  tensors = {}
  for number, wire, value in _fields(graph):
    if number != _GRAPH_INITIALIZER or wire != _LENGTH:
      continue
    name, data_type, dims, raw = _decode_tensor(value)
    if name is None or (keep is not None and not keep(name)):
      continue
    if data_type not in ONNX_DTYPES or raw is None:
      raise ValueError(f"{name}: unsupported initializer (data type {data_type}, raw_data {'missing' if raw is None else 'present'})")
    tensors[name] = np.frombuffer(raw, dtype=ONNX_DTYPES[data_type]).reshape(dims)
  return tensors, net


def _encode_tensor(name: str, array: np.ndarray) -> bytes:
  array = np.ascontiguousarray(array)
  dtype = array.dtype.newbyteorder("<") if array.dtype.byteorder == ">" else array.dtype
  code = NUMPY_TO_ONNX.get(np.dtype(dtype))
  if code is None:
    raise ValueError(f"{name}: numpy dtype {array.dtype} has no ONNX initializer type here")
  raw = array.astype(dtype, copy=False).tobytes()
  parts = [_tag(_TENSOR_DIMS, _VARINT) + _varint(int(dim)) for dim in array.shape]
  parts.append(_tag(_TENSOR_DATA_TYPE, _VARINT) + _varint(code))
  encoded_name = name.encode("utf8")
  parts.append(_length_header(_TENSOR_NAME, len(encoded_name)) + encoded_name)
  parts.append(_length_header(_TENSOR_RAW_DATA, len(raw)) + raw)
  return b"".join(parts)


def encode_initializer_model(tensors: dict, graph_name: str = "bt6_torch") -> bytes:
  """An ONNX ModelProto with `tensors` as initializers and no nodes (IR version 8, opset 17)."""
  encoded_name = graph_name.encode("utf8")
  graph_parts = [_length_header(_GRAPH_NAME, len(encoded_name)) + encoded_name]
  for name, array in tensors.items():
    tensor = _encode_tensor(name, array)
    graph_parts.append(_length_header(_GRAPH_INITIALIZER, len(tensor)) + tensor)
  graph = b"".join(graph_parts)
  opset = _length_header(1, 0) + _tag(2, _VARINT) + _varint(17)
  return b"".join([_tag(_MODEL_IR_VERSION, _VARINT) + _varint(8), _length_header(_MODEL_OPSET, len(opset)) + opset,
                   _length_header(_MODEL_GRAPH, len(graph)) + graph])


def write_carrier(path: str, tensors: dict, template_net=None, compresslevel: int = 1) -> None:
  """Write `tensors` ({name: numpy array}; dtype and shape are what the runtime checks) as a carrier.

  With `template_net` (the Net bytes `read_carrier` returned) every other field of the Net and of its OnnxModel is
  copied through unchanged; without it the BT6 flagship carrier's header fields are used."""
  model = encode_initializer_model(tensors)
  model_field = _length_header(_ONNX_MODEL_MODEL, len(model)) + model
  if template_net is None:
    onnx = model_field + _DEFAULT_ONNX_SUFFIX
    net = _DEFAULT_NET_PREFIX + _length_header(_NET_ONNX_MODEL, len(onnx)) + onnx
  else:
    parts, replaced = [], False
    for number, wire, value in _fields(memoryview(template_net)):
      if number == _NET_ONNX_MODEL and wire == _LENGTH:
        inner = [model_field if (n == _ONNX_MODEL_MODEL and w == _LENGTH) else _reencode(n, w, v)
                 for n, w, v in _fields(value)]
        onnx = b"".join(inner)
        parts.append(_length_header(_NET_ONNX_MODEL, len(onnx)) + onnx)
        replaced = True
      else:
        parts.append(_reencode(number, wire, value))
    if not replaced:
      raise ValueError("the template Net carries no onnx_model to replace")
    net = b"".join(parts)
  with gzip.open(path, "wb", compresslevel=compresslevel) as stream:
    stream.write(net)


def describe(tensors: dict) -> str:
  return "\n".join(f"{name}\t{array.dtype}\t{tuple(array.shape)}" for name, array in tensors.items())


__all__ = ["read_carrier", "write_carrier", "encode_initializer_model", "extract_onnx_model", "describe",
           "ONNX_DTYPES", "NUMPY_TO_ONNX"]
