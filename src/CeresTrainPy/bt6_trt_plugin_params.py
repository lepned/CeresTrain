"""Dump the FP32 parameters the BT6TermsAttention TensorRT plugin needs, from a net's TorchScript (.ts) or a
checkpoint state dict, to an .npz that bt6_trt_plugin_onnx.py folds into the plugin tables.

    python bt6_trt_plugin_params.py <net.ts | state_dict.pt> <out.npz>

Kept: every parameter / buffer whose name contains '.bt6t.', '.qLN.', '.kLN.', 'bt6h.prologue.' or 'attn_templates.'
(names with the leading 'inner.' / 'module.' / '_orig_mod.' stripped). Export code only, not on the training path.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

KEEP = (".bt6t.", ".qLN.", ".kLN.", "bt6h.prologue.", "attn_templates.")


def _strip(name: str) -> str:
  for p in ("_orig_mod.", "module.", "inner."):
    if name.startswith(p):
      name = name[len(p):]
  return name


def main(argv) -> int:
  src, out = argv[1], argv[2]
  if src.endswith(".ts"):
    m = torch.jit.load(src, map_location="cpu")
    items = list(m.named_parameters()) + list(m.named_buffers())
  else:
    sd = torch.load(src, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    items = list(sd.items())
  kept = {}
  for name, value in items:
    if any(k in name for k in KEEP):
      kept[_strip(name)] = value.detach().cpu().numpy()
  if "bt6h.prologue.relidx" not in kept:
    from bt6_net import relative_index
    kept["bt6h.prologue.relidx"] = relative_index().numpy()
  np.savez(out, **kept)
  print(f"{len(kept)} tensors -> {out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
