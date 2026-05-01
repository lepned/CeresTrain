"""Gated Tactical Adapter Branch (GTAB).

A small parallel transformer branch that runs alongside the orig body and
contributes additively to the head input, gated by a learned position
classifier. Designed so the orig is recovered exactly at initialization:
- adapter's output projection is zero-init -> adapter contributes 0
- gate's output bias is large negative -> sigmoid ~ 0
=> flow_out = flow_orig + g(x) * flow_aux ~ flow_orig at step 0

The adapter trains end-to-end with the puzzle loss; the gate sparsity loss
penalizes unnecessary firing so quiet positions stay close to orig.

Env-var controlled (analogous to LoRA env vars in body modules):
  CERES_GTAB                  enable (1=on)
  CERES_GTAB_INNER_DIM        adapter internal dim (default 256)
  CERES_GTAB_NUM_LAYERS       adapter depth (default 4)
  CERES_GTAB_NUM_HEADS        adapter attention heads (default 4)
  CERES_GTAB_FFN_MULT         adapter FFN multiplier (default 4)
  CERES_GTAB_GATE_INIT_BIAS   gate logit bias at init (default -4.0)

All adapter and gate parameters are named with prefix `tactical_` so train.py
can recognize them in the freeze loop.
"""
import os
import math
import torch
import torch.nn as nn

from rms_norm import RMSNorm


def _gtab_int(name, default):
    return int(os.environ.get(name, str(default)) or str(default))


def _gtab_float(name, default):
    return float(os.environ.get(name, str(default)) or str(default))


def gtab_enabled():
    return int(os.environ.get('CERES_GTAB', '0') or '0') > 0


class _AdapterLayer(nn.Module):
    """One pre-norm transformer block at the adapter's internal dim.

    Standard MHA + FFN, no smolgen / smoe / dual-attn — kept minimal so the
    adapter is a clean orthogonal capacity addition.
    """
    def __init__(self, dim, num_heads, ffn_mult, layernorm_eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f'adapter dim {dim} must divide num_heads {num_heads}'

        # pre-norm attention
        self.ln1 = nn.LayerNorm(dim, eps=layernorm_eps)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.W_h = nn.Linear(dim, dim, bias=False)

        # pre-norm FFN
        self.ln2 = nn.LayerNorm(dim, eps=layernorm_eps)
        ffn_inner = ffn_mult * dim
        self.fc1 = nn.Linear(dim, ffn_inner, bias=False)
        self.fc2 = nn.Linear(ffn_inner, dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, T, dim]
        B, T, D = x.shape

        # attention
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]   # each [B, T, H, Dh]
        q = q.transpose(1, 2)  # [B, H, T, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # [B, H, T, Dh]
        attn = attn.transpose(1, 2).reshape(B, T, D)
        x = x + self.W_h(attn)

        # ffn
        h = self.ln2(x)
        x = x + self.fc2(self.act(self.fc1(h)))
        return x


class TacticalAdapter(nn.Module):
    """Parallel mini-transformer branch.

    Reads post-embedding flow [B, 64, in_dim], projects to inner_dim, runs N
    encoder layers, projects back to in_dim. Output projection zero-init so
    the adapter contributes nothing at training step 0.
    """
    def __init__(self, in_dim, inner_dim=None, num_layers=None,
                 num_heads=None, ffn_mult=None, layernorm_eps=1e-6):
        super().__init__()
        if inner_dim   is None: inner_dim   = _gtab_int('CERES_GTAB_INNER_DIM', 256)
        if num_layers  is None: num_layers  = _gtab_int('CERES_GTAB_NUM_LAYERS', 4)
        if num_heads   is None: num_heads   = _gtab_int('CERES_GTAB_NUM_HEADS', 4)
        if ffn_mult    is None: ffn_mult    = _gtab_int('CERES_GTAB_FFN_MULT', 4)

        self.in_dim = in_dim
        self.inner_dim = inner_dim
        self.num_layers = num_layers

        self.proj_in = nn.Linear(in_dim, inner_dim, bias=False)
        self.layers = nn.ModuleList([
            _AdapterLayer(inner_dim, num_heads, ffn_mult, layernorm_eps)
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(inner_dim, eps=layernorm_eps)
        self.proj_out = nn.Linear(inner_dim, in_dim, bias=False)

        # CRITICAL: zero-init output projection so adapter contributes 0 at
        # training start. This is the architectural quiet-safety guarantee.
        nn.init.zeros_(self.proj_out.weight)

    def forward(self, x):
        # x: [B, 64, in_dim]
        h = self.proj_in(x)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_out(h)
        h = self.proj_out(h)   # zero at init -> all-zero output
        return h


class PositionGate(nn.Module):
    """Learned scalar position classifier in [0, 1].

    Pools the post-embedding flow over squares (mean over dim=1), passes
    through a small MLP, sigmoid. Output: [B, 1, 1] for broadcast multiply
    against the adapter's per-square output.

    Bias init at -4 -> sigmoid(-4) ~ 0.018 at start. Combined with the
    zero-init adapter output, the gate's near-zero initial value is for
    stability + interpretability rather than for orig-recovery (which the
    zero-init adapter already gives).
    """
    def __init__(self, in_dim, init_bias=None, hidden=64):
        super().__init__()
        if init_bias is None:
            init_bias = _gtab_float('CERES_GTAB_GATE_INIT_BIAS', -4.0)
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, 1, bias=True)
        # Bias the pre-sigmoid logit to a strongly negative value so gate ~ 0 at init.
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, float(init_bias))

    def forward(self, x):
        # x: [B, 64, in_dim] -> pool over squares -> [B, in_dim]
        pooled = x.mean(dim=1)
        h = torch.nn.functional.gelu(self.fc1(pooled))
        logit = self.fc2(h)            # [B, 1]
        g = torch.sigmoid(logit)        # [B, 1]
        return g.unsqueeze(-1)          # [B, 1, 1] for broadcast vs [B, 64, dim]
