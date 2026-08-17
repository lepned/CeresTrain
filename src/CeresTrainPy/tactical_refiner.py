"""Iterated Tactic Refiner (2026-08 tactical program).

A SINGLE small pre-norm transformer block, weight-shared and applied T times
to the trunk output at a reduced inner dim — recurrence over depth rather than
new per-layer capacity. Rationale: tactics (mate-in-N, forcing sequences) need
SERIAL composition depth more than parallel width; a 10-layer trunk shares its
depth budget with everything else it does. Iterating one shared block adds
calculation steps at a fraction of a full layer's cost per step
(universal-transformer flavor).

Contract with ceres_net:
  - proj_out is ZERO-INIT: the refiner contributes exactly 0 at step 0, so the
    network is bit-identical to the non-refiner baseline at init (TSB/GTAB
    convention).
  - forward returns (delta_final, intermediate_deltas): intermediate deltas
    (iterations 1..T-1, through the SAME ln_out/proj_out) feed the optional
    deep-supervision policy loss (config RefinerDeepSupWeight; the vda
    deep-supervision surprise is the precedent). The final iteration is
    supervised by the main losses.
  - Attention is EXPLICIT matmul->softmax->matmul, never
    F.scaled_dot_product_attention: the refiner is part of the SERVING graph,
    and the dynamo ONNX exporter auto-fuses F.sdpa into the opset-23
    `Attention` op which TRT's plugin only accepts in strongly-typed builds
    (see the comment in dot_product_attention.forward).
  - All parameters live under the `tactical_refiner` name prefix (freeze-loop
    recognizability, same convention as GTAB's `tactical_` prefix).

This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
"""
import math

import torch
from torch import nn

from rms_norm import make_norm


class TacticalRefiner(nn.Module):
  def __init__(self, in_dim: int, inner_dim: int = 128, num_heads: int = 4,
               ffn_mult: int = 2, iters: int = 3, layernorm_eps: float = 1e-6,
               norm_type: str = 'LayerNorm', softcap_cutoff: float = 0):
    super().__init__()
    assert iters >= 1, 'RefinerIters must be >= 1 when the refiner is enabled'
    assert inner_dim % num_heads == 0, \
        f'refiner inner_dim {inner_dim} must divide num_heads {num_heads}'
    self.iters = iters
    self.inner_dim = inner_dim
    self.num_heads = num_heads
    self.head_dim = inner_dim // num_heads
    # Logit soft-cap (grok/gemma tanh form, same as DotProductAttention): the
    # refiner sits outside the trunk's stability machinery (QK-clip arms only
    # DotProductAttention modules; qk-norm is a trunk option), and its single
    # weight-shared qkv is applied T times, so unbounded logit growth would
    # compound across iterations — the s8 postmortem failure mode. Wired from
    # NetDef_SoftCapCutoff by ceres_net; 0 disables.
    self.softcap_cutoff = float(softcap_cutoff)

    self.proj_in = nn.Linear(in_dim, inner_dim, bias=False)

    # The one shared block (pre-norm attention + pre-norm GELU FFN).
    self.ln1 = make_norm(norm_type, inner_dim, eps=layernorm_eps)
    self.qkv = nn.Linear(inner_dim, 3 * inner_dim, bias=False)
    self.W_h = nn.Linear(inner_dim, inner_dim, bias=False)
    self.ln2 = make_norm(norm_type, inner_dim, eps=layernorm_eps)
    self.fc1 = nn.Linear(inner_dim, ffn_mult * inner_dim, bias=False)
    self.fc2 = nn.Linear(ffn_mult * inner_dim, inner_dim, bias=False)

    self.ln_out = make_norm(norm_type, inner_dim, eps=layernorm_eps)
    self.proj_out = nn.Linear(inner_dim, in_dim, bias=False)
    # Zero-init: exact no-op at training step 0 (see module docstring).
    nn.init.zeros_(self.proj_out.weight)

  def _step(self, h: torch.Tensor) -> torch.Tensor:
    B, T, D = h.shape
    qkv = self.qkv(self.ln1(h)).reshape(B, T, 3, self.num_heads, self.head_dim)
    q = qkv[:, :, 0].transpose(1, 2)                       # [B, H, T, dh]
    k = qkv[:, :, 1].transpose(1, 2)
    v = qkv[:, :, 2].transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
    if self.softcap_cutoff > 0:
      scores = self.softcap_cutoff * torch.tanh(scores / self.softcap_cutoff)
    a = torch.softmax(scores, dim=-1)
    o = torch.matmul(a, v).transpose(1, 2).reshape(B, T, D)
    h = h + self.W_h(o)
    h = h + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(h))))
    return h

  def forward(self, x: torch.Tensor, collect_intermediate: bool = False):
    """x: [B, 64, in_dim] trunk output. Returns (delta_final, inter) where
    delta_final is the residual to add to the trunk flow and inter is a list
    of iteration-1..T-1 deltas (empty/None unless collect_intermediate)."""
    h = self.proj_in(x)
    inter = [] if collect_intermediate else None
    for t in range(self.iters):
      h = self._step(h)
      if collect_intermediate and t < self.iters - 1:
        inter.append(self.proj_out(self.ln_out(h)))
    return self.proj_out(self.ln_out(h)), inter
