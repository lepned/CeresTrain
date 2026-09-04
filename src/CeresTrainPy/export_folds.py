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


SUBNORMAL_LIMIT = 0.05    # max share of a folded weight allowed below fp16's smallest normal


def _subnormal_share(w):
  a = w.detach().half().float().abs()   # as the export will store it
  nz = a[a > 0]
  if nz.numel() == 0:
    return 0.0
  return float((nz < torch.finfo(torch.float16).tiny).float().mean())


@torch.no_grad()
def apply_export_folds(core, ffn=True, mt_scale=True, mt_norms=True, check_fp16=True):
  """check_fp16: refuse a fold that pushes weights into the fp16 subnormal range.

  Folding a scale < 1 into a weight shrinks it, and the exported graph is FP16, so weights can
  land below fp16's smallest normal (6.1e-5), lose mantissa bits, or be dropped by a kernel that
  flushes subnormals to zero. MEASURED 2026-09-05 on the exported 640x12 prod graph: the decoder
  tensors go from 0.03 % to 0.12-0.67 % subnormal (whole graph 0.057 % -> 0.066 %) — small enough
  to be safe, which is why the limit is 5 %: the guard is here to catch a net whose norm scales
  are small enough to make the fold destructive, not to flag the normal case. The residual effect
  of the fold in fp16 is ~1e-5 policy KL / 99.8 % top-1 agreement: mathematically exact, not
  bit-exact."""
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
      fused.weight.copy_(torch.cat([l1.weight, l3.weight], dim=0))   # pure concat: no arithmetic, no rounding
      if l1.bias is not None:
        fused.bias.copy_(torch.cat([l1.bias, torch.zeros_like(l1.bias)], dim=0))
      m.linear13 = fused
      counts['ffn_fused'] += 1
  mt = getattr(core, 'move_tokens', None)
  if mt is not None and (mt_scale or mt_norms):
    dm = mt.dm
    # Always evaluate in fp16: the fold runs while the weights are still fp32, but every export
    # of these nets is converted to fp16 afterwards, so the dtype here says nothing about the
    # dtype the folded weights will actually be stored in.
    _fp16 = check_fp16
    _before = {}
    if _fp16:
      for _i, _b in enumerate(mt.blocks):
        for _nm, _l in (('qkv', _b.qkv), ('xq', _b.xq), ('ffn_in', _b.ffn_in)):
          _before[(_i, _nm)] = _subnormal_share(_l.weight)
    for blk in mt.blocks:
      # Fold in FP32 and cast back: the weights are usually FP16/BF16, and multiplying
      # them in their own dtype adds a rounding step the unfolded graph never pays
      # (measured: 99.84 % -> ~100 % top-1 agreement on the 640x12 net).
      if mt_scale and not getattr(blk, '_attn_scale_folded', False):
        s = blk.dk ** -0.5
        w = blk.qkv.weight
        w[:dm].copy_((w[:dm].float() * s).to(w.dtype))    # Q rows of the fused qkv projection
        blk.xq.weight.copy_((blk.xq.weight.float() * s).to(blk.xq.weight.dtype))
        blk._attn_scale_folded = True
        counts['mt_scale_folded'] += 1
      if mt_norms:
        for ln, lin in ((blk.ln1, blk.qkv), (blk.ln2, blk.xq), (blk.ln3, blk.ffn_in)):
          if isinstance(ln, RMSNorm) and not getattr(ln, '_scale_folded', False):
            lw = lin.weight
            lw.copy_((lw.float() * ln.scale.float().unsqueeze(0)).to(lw.dtype))   # scale the input columns
            ln._scale_folded = True
            counts['mt_norm_folded'] += 1
    if _fp16:
      _bad = []
      for _i, _b in enumerate(mt.blocks):
        for _nm, _l in (('qkv', _b.qkv), ('xq', _b.xq), ('ffn_in', _b.ffn_in)):
          _a = _subnormal_share(_l.weight)
          if _a > SUBNORMAL_LIMIT and _a > 4 * _before[(_i, _nm)]:
            _bad.append(f'block{_i}.{_nm} {100*_before[(_i, _nm)]:.3f}% -> {100*_a:.3f}%')
      if _bad:
        raise ValueError(
            'ExportFolds would push fp16 weights subnormal (share below 6.1e-5 grew past '
            f'{100*SUBNORMAL_LIMIT:.1f}%): {"; ".join(_bad[:6])}'
            f'{" ..." if len(_bad) > 6 else ""}. The folded graph is then NOT equivalent — '
            'kernels that flush subnormals to zero drop those weights. Set ExportFolds=none '
            '(or export in fp32) for this net.')
  return counts
