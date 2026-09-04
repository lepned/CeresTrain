"""
Export-time graph folds (2026-09-04 ideation T0-3). Function-identical rewrites of an
eval-mode CeresNet so the exported ONNX has fewer, larger kernels for TensorRT:

  (a) SwiGLU FFN gate|up fusion: linear1 (640->1920) and linear3 (640->1920) read the
      same input; one (640->3840) GEMM + one chunk replaces two GEMM launches and
      doubles the GEMM's N (better tensor-core tiling at the 64-token shapes).
  (b) Move-token decoder attention scale: q k^T * dk^-0.5 -> the dk^-0.5 is folded into
      the Q rows of `qkv` and into `xq` (both feed the block's shared `_attn`).
  (c) Move-token decoder pre-norm scales: RMSNorm(x) * scale -> Linear(W) becomes
      RMSNorm(x) -> Linear(W * scale) for ln1->qkv, ln2->xq, ln3->ffn_in (each norm
      output has exactly that one consumer). out_ln is NOT folded (it feeds the pools
      and the policy read). ln_s (square norm) is already shared in the fused export.

Applied to the export DEEPCOPY only (see save_model.py; config-driven via the NetDef key
"ExportFolds": none|mt|ffn|all). Exact in fp32 up to rounding (bf16 weights round once more);
parity is asserted by test_export_folds.py. Measured in EngineBattle on the 700M prod net:
mt +6-8 % EPS, ffn 0.96-1.00x, all 0.91-0.96x.
"""
import math
import torch


@torch.no_grad()
def apply_export_folds(core, ffn=True, mt_scale=True, mt_norms=True):
  from mlp2_layer import MLP2Layer
  from rms_norm import RMSNorm
  counts = {'ffn_fused': 0, 'mt_scale_folded': 0, 'mt_norm_folded': 0}
  if ffn:
    for m in core.modules():
      if not isinstance(m, MLP2Layer):
        continue
      if getattr(m, 'use_te', False) or getattr(m, 'activation_type', None) != 'SwiGLU':
        continue
      if getattr(m, 'ffn_softcap', None) is not None or getattr(m, 'linear13', None) is not None:
        continue
      l1, l3 = m.linear1, m.linear3
      if type(l1) is not torch.nn.Linear or type(l3) is not torch.nn.Linear:
        continue  # LoRA-wrapped or exotic: leave alone
      fused = torch.nn.Linear(l1.in_features, l1.out_features + l3.out_features,
                              bias=l1.bias is not None, device=l1.weight.device, dtype=l1.weight.dtype)
      fused.weight.copy_(torch.cat([l1.weight, l3.weight], dim=0))
      if l1.bias is not None:
        fused.bias.copy_(torch.cat([l1.bias, torch.zeros_like(l1.bias)], dim=0))
      m.linear13 = fused
      counts['ffn_fused'] += 1
  mt = getattr(core, 'move_tokens', None)
  if mt is not None and (mt_scale or mt_norms):
    dm = mt.dm
    for blk in mt.blocks:
      if mt_scale and not getattr(blk, '_attn_scale_folded', False):
        s = blk.dk ** -0.5
        blk.qkv.weight[:dm].mul_(s)     # Q rows of the fused qkv projection
        blk.xq.weight.mul_(s)           # cross-attention query
        blk._attn_scale_folded = True
        counts['mt_scale_folded'] += 1
      if mt_norms:
        for ln, lin in ((blk.ln1, blk.qkv), (blk.ln2, blk.xq), (blk.ln3, blk.ffn_in)):
          if isinstance(ln, RMSNorm) and not getattr(ln, '_scale_folded', False):
            lin.weight.mul_(ln.scale.to(lin.weight.dtype).unsqueeze(0))   # scale the input columns
            ln._scale_folded = True
            counts['mt_norm_folded'] += 1
  return counts
