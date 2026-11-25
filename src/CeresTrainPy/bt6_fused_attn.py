"""Fused Triton training kernel for the BT6 term-read attention of OUR native layers (config BT6FusedAttention;
bt6_native_terms.BT6NativeTerms.attend). One program per (position, head) holds the whole 64 x 64 tile in
registers: the logit assembly, the door, the softmax, the post-softmax gate and A @ V run in ONE kernel, so L, H,
the softmax output, A and the gate are never written to memory in the forward; the backward kernel recomputes them
and writes only the gradients of its plane inputs.

Fusion boundary (measured, batch 384 on the 5090). The linear parts stay batched matmuls on tensor cores -- exactly
the reference's own code, so their gradients are plain autograd:
  * relation terms (bt6_native_terms.BT6NativeTerms.logits): rel_q [B, 64(i), H, 64(j)] and rel_k [B, 64(j), H, 64(i)]
    from the per-head rows against E (an in-kernel loop over the 34 relation channels cost 0.34 ms of a 0.43 ms
    forward; in-kernel per-channel reductions in the backward ~20 ms per layer from register spills);
  * edge / pair read, door and gate logits (bt6_net.stacked_reads: ONE matmul for all heads against the read state;
    reading the read state per (position, head) inside the kernel was ~1.6 GB of traffic per layer).
The kernel then computes, in float32,

    L   = (Q.K) * t / sqrt(hd) + rel_q + rel_k^T + read + bias
    H   = L (1 + door);   A = softmax_j(H) * 2 sigmoid(gl);   out = A @ V      (A cast to V's dtype, as the reference)

and its backward writes dL (= the gradient of read, of bias, and -- in the two layouts -- of rel_q and rel_k),
d door and d gate logits, plus dQ, dK, dV and the gradient of the per-head scale t / sqrt(hd).

Numerics: everything inside the kernel is float32; the reference assembles L / door in the autocast dtype and runs
softmax and gate in float32 -- same class, slightly more precise. Supported: no gate cap, no pair_autocast (the
trainer's defaults), head_dim a power of two >= 16; `supported()` says when. Registered as torch.library custom ops
(forward op + backward op, register_autograd): torch.compile sees one opaque node, no graph break. CPU tensors and
ONNX export use the reference path (the caller decides). The H the reference returns is replaced by its per-(position,
head) maximum, which is all the caller reads of it (the QK-clip monitor's amax)."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from bt6_net import BASE_RELATION_CHANNELS, stacked_reads

T = 64            # tokens (squares)
NUM_WARPS = 4


@triton.jit
def _assemble(Q, K, SC, RQ, RK, RD, DR, GL, BIAS, b, h, sbb, sbh, srb, srh,
              H: tl.constexpr, HD: tl.constexpr, HAS_BIAS: tl.constexpr, IEEE: tl.constexpr):
  i = tl.arange(0, 64)
  d = tl.arange(0, HD)
  offs = i[:, None] * 64 + i[None, :]
  qkv = ((b * H + h) * 64) * HD
  q = tl.load(Q + qkv + i[:, None] * HD + d[None, :])
  k = tl.load(K + qkv + i[:, None] * HD + d[None, :])
  if IEEE:
    s = tl.dot(q, tl.trans(k), input_precision="ieee")
  else:
    s = tl.dot(q, tl.trans(k))
  # rel_q at [b, i, h, j]; rel_k at [b, j, h, i] (loaded as its [j, i] tile, transposed in registers)
  relo = (b * 64 * H + h) * 64 + i[:, None] * (H * 64) + i[None, :]
  rplane = b * srb + h * srh
  L = (s * tl.load(SC + h) + tl.load(RQ + relo).to(tl.float32) + tl.trans(tl.load(RK + relo).to(tl.float32))
       + tl.load(RD + rplane + offs).to(tl.float32))
  if HAS_BIAS:
    L += tl.load(BIAS + b * sbb + h * sbh + offs).to(tl.float32)
  door = tl.load(DR + rplane + offs).to(tl.float32)
  g = tl.load(GL + (b * H + h) * 4096 + offs).to(tl.float32)
  return q, k, s, L, door, g


@triton.jit
def _fwd_kernel(Q, K, V, SC, RQ, RK, RD, DR, GL, BIAS, O, HM, sbb, sbh, srb, srh,
                H: tl.constexpr, HD: tl.constexpr, HAS_BIAS: tl.constexpr, IEEE: tl.constexpr):
  pid = tl.program_id(0)
  b = pid // H
  h = pid % H
  q, k, s, L, door, g = _assemble(Q, K, SC, RQ, RK, RD, DR, GL, BIAS, b, h, sbb, sbh, srb, srh, H, HD, HAS_BIAS, IEEE)
  Hh = L * (1.0 + door)
  m = tl.max(Hh, axis=1)
  P = tl.exp(Hh - m[:, None])
  P = P / tl.sum(P, axis=1)[:, None]
  A = P * (2.0 * tl.sigmoid(g))
  i = tl.arange(0, 64)
  d = tl.arange(0, HD)
  qkv = ((b * H + h) * 64) * HD
  v = tl.load(V + qkv + i[:, None] * HD + d[None, :])
  if IEEE:
    o = tl.dot(A.to(v.dtype), v, input_precision="ieee")
  else:
    o = tl.dot(A.to(v.dtype), v)
  tl.store(O + qkv + i[:, None] * HD + d[None, :], o.to(O.dtype.element_ty))
  tl.store(HM + pid, tl.max(m, axis=0))


@triton.jit
def _bwd_kernel(Q, K, V, SC, RQ, RK, RD, DR, GL, BIAS, DO, DQ, DK, DV, DSC, DL, DLQ, DLK, DDOOR, DGL,
                sbb, sbh, srb, srh, H: tl.constexpr, HD: tl.constexpr, HAS_BIAS: tl.constexpr, IEEE: tl.constexpr):
  pid = tl.program_id(0)
  b = pid // H
  h = pid % H
  q, k, s, L, door, g = _assemble(Q, K, SC, RQ, RK, RD, DR, GL, BIAS, b, h, sbb, sbh, srb, srh, H, HD, HAS_BIAS, IEEE)
  Hh = L * (1.0 + door)
  m = tl.max(Hh, axis=1)
  P = tl.exp(Hh - m[:, None])
  P = P / tl.sum(P, axis=1)[:, None]
  gate = 2.0 * tl.sigmoid(g)
  A = P * gate
  i = tl.arange(0, 64)
  d = tl.arange(0, HD)
  offs = i[:, None] * 64 + i[None, :]
  qkv = ((b * H + h) * 64) * HD
  v = tl.load(V + qkv + i[:, None] * HD + d[None, :])
  do = tl.load(DO + qkv + i[:, None] * HD + d[None, :])
  if IEEE:
    dA = tl.dot(do, tl.trans(v), input_precision="ieee")
    dv = tl.dot(tl.trans(A.to(do.dtype)), do, input_precision="ieee")
  else:
    dA = tl.dot(do, tl.trans(v))
    dv = tl.dot(tl.trans(A.to(do.dtype)), do)
  tl.store(DV + qkv + i[:, None] * HD + d[None, :], dv.to(DV.dtype.element_ty))
  plane = pid * 4096
  tl.store(DGL + plane + offs, (dA * P * gate * (1.0 - 0.5 * gate)).to(DGL.dtype.element_ty))
  dP = dA * gate
  dH = P * (dP - tl.sum(dP * P, axis=1)[:, None])
  dL = dH * (1.0 + door)
  tl.store(DDOOR + plane + offs, (dH * L).to(DDOOR.dtype.element_ty))
  dLc = dL.to(DL.dtype.element_ty)
  tl.store(DL + plane + offs, dLc)
  relo = (b * 64 * H + h) * 64 + i[:, None] * (H * 64) + i[None, :]
  tl.store(DLQ + relo, dL.to(DLQ.dtype.element_ty))
  tl.store(DLK + relo, tl.trans(dL).to(DLK.dtype.element_ty))
  tl.store(DSC + pid, tl.sum(tl.sum(dL * s, axis=1), axis=0))
  dS = dL * tl.load(SC + h)
  if IEEE:
    dq = tl.dot(dS.to(k.dtype), k, input_precision="ieee")
    dk = tl.dot(tl.trans(dS.to(q.dtype)), q, input_precision="ieee")
  else:
    dq = tl.dot(dS.to(k.dtype), k)
    dk = tl.dot(tl.trans(dS.to(q.dtype)), q)
  tl.store(DQ + qkv + i[:, None] * HD + d[None, :], dq.to(DQ.dtype.element_ty))
  tl.store(DK + qkv + i[:, None] * HD + d[None, :], dk.to(DK.dtype.element_ty))


def _plane_strides(read, door):
  """Batch / head strides of the read and door planes [B, H, 4096] (views of one stacked_reads output: the same
  strides, each 4096-plane contiguous)."""
  assert read.stride() == door.stride() and read.stride(2) == 1, (read.stride(), door.stride())
  return read.stride(0), read.stride(1)


def _bias_args(bias, B, H):
  if bias is None:
    return None, 0, 0
  bb = bias.expand(B, H, T, T)
  if bb.stride(-1) != 1 or bb.stride(-2) != T:
    bb = bb.contiguous()
  return bb, bb.stride(0), bb.stride(1)


@torch.library.custom_op("ceres_bt6::fused_attn_fwd", mutates_args=())
def fused_attn_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, sc: torch.Tensor, relq: torch.Tensor,
                   relk: torch.Tensor, read: torch.Tensor, door: torch.Tensor, gl: torch.Tensor,
                   bias: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
  B, H, _, HD = Q.shape
  O = torch.empty_like(V)
  HM = torch.empty(B * H, device=Q.device, dtype=torch.float32)
  bb, sbb, sbh = _bias_args(bias, B, H)
  srb, srh = _plane_strides(read, door)
  _fwd_kernel[(B * H,)](Q, K, V, sc, relq, relk, read, door, gl, bb if bb is not None else Q, O, HM, sbb, sbh, srb, srh,
                        H=H, HD=HD, HAS_BIAS=bb is not None, IEEE=Q.dtype == torch.float32, num_warps=NUM_WARPS)
  return O, HM.view(B, H)


@fused_attn_fwd.register_fake
def _(Q, K, V, sc, relq, relk, read, door, gl, bias):
  return torch.empty_like(V), Q.new_empty((Q.shape[0], Q.shape[1]), dtype=torch.float32)


@torch.library.custom_op("ceres_bt6::fused_attn_bwd", mutates_args=())
def fused_attn_bwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, sc: torch.Tensor, relq: torch.Tensor,
                   relk: torch.Tensor, read: torch.Tensor, door: torch.Tensor, gl: torch.Tensor,
                   bias: torch.Tensor | None, dO: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                              torch.Tensor, torch.Tensor, torch.Tensor]:
  B, H, _, HD = Q.shape
  dO = dO.to(V.dtype).contiguous()
  dQ, dK, dV = torch.empty_like(Q), torch.empty_like(K), torch.empty_like(V)
  dsc = torch.empty(B * H, device=Q.device, dtype=torch.float32)
  dL = torch.empty(B, H, T * T, device=Q.device, dtype=read.dtype)
  dLq, dLk = torch.empty_like(relq), torch.empty_like(relk)
  ddoor = torch.empty(B, H, T * T, device=Q.device, dtype=door.dtype)
  dgl = torch.empty_like(gl)
  bb, sbb, sbh = _bias_args(bias, B, H)
  srb, srh = _plane_strides(read, door)
  _bwd_kernel[(B * H,)](Q, K, V, sc, relq, relk, read, door, gl, bb if bb is not None else Q, dO,
                        dQ, dK, dV, dsc, dL, dLq, dLk, ddoor, dgl, sbb, sbh, srb, srh,
                        H=H, HD=HD, HAS_BIAS=bb is not None, IEEE=Q.dtype == torch.float32, num_warps=NUM_WARPS)
  return dQ, dK, dV, dsc.view(B, H), dLq, dLk, dL, ddoor, dgl


@fused_attn_bwd.register_fake
def _(Q, K, V, sc, relq, relk, read, door, gl, bias, dO):
  B, H = Q.shape[0], Q.shape[1]
  return (torch.empty_like(Q), torch.empty_like(K), torch.empty_like(V), Q.new_empty((B, H), dtype=torch.float32),
          torch.empty_like(relq), torch.empty_like(relk), read.new_empty((B, H, T * T)), door.new_empty((B, H, T * T)),
          torch.empty_like(gl))


def _setup_context(ctx, inputs, output):
  ctx.save_for_backward(*inputs)
  ctx.has_bias = inputs[-1] is not None
  ctx.bias_shape = None if inputs[-1] is None else tuple(inputs[-1].shape)


def _backward(ctx, dO, dHM):
  Q, K, V, sc, relq, relk, read, door, gl, bias = ctx.saved_tensors
  B, H = Q.shape[0], Q.shape[1]
  dQ, dK, dV, dsc, dLq, dLk, dL, ddoor, dgl = fused_attn_bwd(Q, K, V, sc, relq, relk, read, door, gl,
                                                              bias if ctx.has_bias else None, dO)
  dbias = None
  if ctx.has_bias:
    db = dL.view(B, H, T, T)
    for dim, size in enumerate(ctx.bias_shape):
      if size == 1 and db.shape[dim] != 1:
        db = db.sum(dim=dim, keepdim=True)
    dbias = db.to(bias.dtype)
  return dQ, dK, dV, dsc.sum(0), dLq, dLk, dL, ddoor, dgl, dbias


fused_attn_fwd.register_autograd(_backward, setup_context=_setup_context)


# ----------------------------------------------------------------------------------------------------------------
def supported(cfg) -> bool:
  """The kernel covers the trainer's configuration: no gate cap, no pair_autocast, head_dim a power of two >= 16."""
  hd = int(cfg.head_dim)
  return (not cfg.gate_cap) and (not cfg.pair_autocast) and hd >= 16 and (hd & (hd - 1)) == 0


def fused_attend(terms, Q, K, V, bias, ctx, e, read_state):
  """BT6NativeTerms.attend on the fused kernel: (A @ V [B, heads, 64, hd] in V's dtype, Hmax [B, heads, 1, 1] float32
  = the per-(position, head) maximum of H, which is all the caller reads of H (the QK-clip monitor's amax)).
  The relation terms and the stacked reads are the reference's statements (BT6NativeTerms.logits, stacked_reads)."""
  cfg = terms.cfg
  B, heads, hd = Q.shape[0], cfg.heads, cfg.head_dim
  pd = read_state.dtype
  t = terms.t_node.float()
  Qp, Kp = Q.to(pd), K.to(pd)
  # relation terms, as BT6NativeTerms.logits: query side [b, i, h, j], key side [b, j, h, i]
  Pq = (terms.Pq.float() * t[:, None, None]).to(pd).transpose(1, 2)                                     # [h, hd, c]
  Pk = (terms.Pk.float() * t[:, None, None]).roll(BASE_RELATION_CHANNELS, 1).to(pd).transpose(1, 2)
  rows = torch.matmul(Qp, Pq[None]) + (terms.C1.float() * t[:, None]).to(pd)[None, :, None, :]          # [b, h, 64, c]
  keys = torch.matmul(Kp, Pk[None])
  Ei = ctx.Ei(pd).view(B * 64, -1, 64)
  relq = torch.bmm(rows.permute(0, 2, 1, 3).reshape(B * 64, heads, -1), Ei).view(B, 64, heads, 64)
  relk = torch.bmm(keys.permute(0, 2, 1, 3).reshape(B * 64, heads, -1), Ei).view(B, 64, heads, 64)
  read, door, gl = stacked_reads([terms], cfg, e, read_state)[0]      # [B, heads, 64, 64] each (one matmul)
  read = read.flatten(2)                                              # views: strided planes, no copy
  door = door.flatten(2)
  gl = gl.reshape(B, heads, 4096).contiguous()
  cdt = V.dtype
  sc = (t / math.sqrt(hd)).contiguous()
  out, HM = torch.ops.ceres_bt6.fused_attn_fwd(Q.to(cdt).contiguous(), K.to(cdt).contiguous(), V.contiguous(), sc,
                                               relq.contiguous(), relk.contiguous(), read, door, gl,
                                               None if bias is None else bias)
  return out, HM.view(B, heads, 1, 1)
