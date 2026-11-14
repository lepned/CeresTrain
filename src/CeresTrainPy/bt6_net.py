"""Lc0 "BT6" (EGT2 flagship family) forward pass in PyTorch, exact against the lc0ex kernels.

Two implementations of the same function, deliberately independent of each other:

* `BT6Net` (nn.Module): the net in TRAINED-parameter form -- t_node, t_edge, C1, Pk, Pq, C2, We, Wm, Wg, bg per
  layer, D, T[225], Pin, Toff[225] in the prologue, separate gate / up projections in every GLU, separate q / k / v /
  output-gate projections. `export_served()` produces the served (folded) tensors with the lc0ex carrier's names,
  shapes and dtypes; `load_served()` inverts the folds to initialise the module from a carrier.
* `forward_served()`: the forward computed DIRECTLY from the served tensors, written after the kernels
  (prologue_egt, attention_egt, edge_site, triplet_site, policy_egt_bias, promotion_logits, policy_map, ...), so a
  comparison with the real compiled program does not depend on the fold inversion.

Shapes: B = batch, 64 squares s = 8 * row + column in lc0 plane coordinates (row 0 = back rank of the side to
move). Input `planes` is float [B, 112, 64] (lc0 classical planes; bt6_planes.expand_planes). Outputs: policy
logits [B, 1858], WDL logits [B, 3], moves left [B, 1], value error [B, 1] (sigmoid output, 0..1).

The forward, per position (pre-norm tower; LN = centred LayerNorm with scale and bias, eps 1e-3):
  prologue   E = BT6Relations(piece planes, previous piece planes)                         bool [34, 64, 64], no grad
             Z = D.E + T[relidx];  S = 1 / sqrt(mean_d Z^2 + 1e-6);  Z0 = S Z              pair stream, static
             e0 = rms_c(Pin.E + Toff[relidx])                                              edge-state seed [16, 64, 64]
  embedding  pos = flatten(piece planes over squares) W_pre + b  -> [64, dense]            one dense per POSITION
             x = LN0(mish([planes | pos] W + b)) * relu(gate) + position
             x = LN1(x + a_emb * GLU_emb(x)),   a_emb = (2 * layers) ** -0.25              post-norm, constant alpha
  layer      y = LN1(x);  q, k, v, g = four dense of y
             F = t_node[h] * (q.k / sqrt(hd) + C1.E + (q.Pq).E + (k.Pk).E + C2.Z0)
             H = (F + t_edge[h] * We.e) * (1 + Wm.e)                                       edge read, then the door
             A = softmax_j(H) * 2 sigmoid(Wg.e + bg)                                       post-softmax gate, no renorm
             x = x + ((A v) * 2 sigmoid(g)) W_out + b                                      output gate per channel
             x = x + GLU(LN2(x)),   GLU(z) = (sigmoid(z Wg) * (z Wu + bu)) Wd + bd
  edge site  (after the configured layers, reading that layer's H)
             e_hat = e + O.H;  triplet branch on rms(e_hat);  e = rms(e_hat + W2 relu(W1 n[i,j] + b1 + Wr n[j,i]))
  heads      on LN_final(x): attention policy with the pair bias u.Z0 and the promotion offsets, value (WDL
             logits), value error (sigmoid), moves left (relu).

Mish: the kernels use lc0's two-branch form, which is algebraically x * tanh(softplus(x)); F.mish is used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bt6_relations import BASE_CHANNELS as BASE_RELATION_CHANNELS
from bt6_relations import RELATION_CHANNELS, BT6Relations, relative_index
from lc0_moves_1858 import MOVES_1858

RELATIVE_BINS = 225
POLICY_RECORD = 64 * 64 + 8 * 24      # from * 64 + to, then 4096 + from_file * 24 + to_file * 3 + {q, r, b}
# The 34 relation bits as code groups of the served `/prologue/mix_codes` table: (first channel, bit count), and
# the row where each group's codes start (prologue_egt.CODE_GROUPS / CODE_GROUP_BASES).
CODE_GROUPS = ((0, 9), (9, 8), (17, 9), (26, 8))
CODE_GROUP_BASES = (0, 512, 768, 1280)


@dataclass
class BT6Config:
  """Sizes of a BT6-family net; the defaults are the BT6 flagship as served."""

  d_model: int = 1024
  layers: int = 24
  heads: int = 32
  ffn_hidden: int = 1024                 # GLU hidden width of the encoder layers
  embedding_dense: int = 512             # per-square width of the whole-board preprocess dense
  embedding_ffn_hidden: int = 2048       # GLU hidden width of the embedding FFN
  pair_channels: int = 16                # pair stream Z0
  edge_channels: int = 16                # edge state e (d_e)
  edge_sites: tuple = (3, 7, 11, 15, 19)  # update sites AFTER these layers (0-based)
  site_hidden: int = 64                  # edge FFN hidden width (ffn_mult 4)
  triplet: bool = True
  triplet_heads: int = 4
  triplet_dots: int = 4                  # triplet_heads * triplet_dots == edge_channels
  triplet_contraction: str = "path"      # "path" or "ag"
  rev_edge: bool = True                  # the edge FFN also reads the reverse edge n[j, i]
  gate_cap: bool = False                 # min(2 sigmoid(.), 1) on the post-softmax gate
  # Under autocast, run the logit ASSEMBLY (q.k, relation / pair terms, edge read, door product) in the autocast
  # dtype; softmax, gates and rms stay float32. False = float32 assembly. No effect without autocast.
  assembly_autocast: bool = True
  # Config lever (NOT BT6's training numerics; default off): under autocast also the door product's result, the
  # softmax and the post-softmax gate stay in the autocast dtype (BT6 trains them in float32). Needs
  # assembly_autocast. The output-gate sigmoid, the sites and the prologue stay float32.
  pair_autocast: bool = False
  output_gate: bool = True               # attention output gate 2 sigmoid(g(y))
  output_gate_scale: float = 2.0
  policy_width: int = 1024
  value_width: int = 128                 # per-square width of the value / error heads
  value_hidden: int = 128
  moves_width: int = 8
  moves_hidden: int = 128
  error_head: bool = True
  norm_eps: float = 1e-3
  rms_eps: float = 1e-6
  input_planes: int = 112
  relation_channels: int = field(default=RELATION_CHANNELS)

  def __post_init__(self) -> None:
    self.edge_sites = tuple(int(site) for site in self.edge_sites)
    if self.d_model % self.heads:
      raise ValueError("d_model must be a multiple of heads")
    if self.relation_channels != RELATION_CHANNELS:
      raise ValueError(f"BT6Relations provides {RELATION_CHANNELS} relation channels")
    if self.triplet and self.triplet_heads * self.triplet_dots != self.edge_channels:
      raise ValueError("triplet_heads * triplet_dots must equal edge_channels")
    if self.triplet_contraction not in ("path", "ag"):
      raise ValueError("triplet_contraction must be 'path' or 'ag'")
    if any(not 0 <= site < self.layers for site in self.edge_sites):
      raise ValueError("edge_sites must name layers of the tower")

  @property
  def head_dim(self) -> int:
    return self.d_model // self.heads

  @property
  def embedding_alpha(self) -> float:
    """Residual scale of the embedding FFN: a build-time constant, not a parameter."""
    return (2.0 * self.layers) ** -0.25

  @staticmethod
  def from_served(tensors: dict, **overrides) -> "BT6Config":
    """Infer the sizes from the shapes of a carrier's tensors. What the tensors cannot tell (gate cap, triplet
    contraction, output-gate scale) keeps its default unless overridden."""
    layers = 0
    while f"/encoder{layers}/ln1/scale" in tensors:
      layers += 1
    d_model = tensors["/attn_body/matmul/w"].shape[1]
    sites = tuple(i for i in range(layers) if f"/encoder{i}/edge_site/readback/w" in tensors)
    first_site = f"/encoder{sites[0]}/edge_site" if sites else None
    values = dict(
      d_model=d_model, layers=layers, heads=tensors["/encoder0/mha/egt/qk_scale"].shape[0],
      ffn_hidden=tensors["/encoder0/ffn/dense2/w"].shape[0],
      embedding_dense=tensors["/attn_body/matmul/w"].shape[0] - 112,
      embedding_ffn_hidden=tensors["/attn_body/ffn/dense2/w"].shape[0],
      pair_channels=tensors["/prologue/offsets"].shape[0], edge_channels=tensors["/prologue/edge_offsets"].shape[0],
      edge_sites=sites, output_gate="/encoder0/mha/qkvg/w" in tensors,
      policy_width=tensors["/policy/Q/w"].shape[0], value_width=tensors["/value/embedding/w"].shape[1],
      value_hidden=tensors["/value/dense1/w"].shape[1], moves_width=tensors["/moves/embedding/w"].shape[1],
      moves_hidden=tensors["/moves/dense1/w"].shape[1], error_head="/value_error/dense2/w" in tensors,
    )
    if first_site is not None:
      values.update(site_hidden=tensors[f"{first_site}/ffn/dense1/w"].shape[0],
                    triplet=f"{first_site}/triplet/value/w" in tensors,
                    rev_edge=f"{first_site}/ffn/dense1_rev/w" in tensors)
      if values["triplet"]:
        triplet_heads = tensors[f"{first_site}/triplet/gate/w"].shape[0] // 4
        values.update(triplet_heads=triplet_heads, triplet_dots=values["edge_channels"] // triplet_heads)
    values.update(overrides)
    return BT6Config(**values)


# ------------------------------------------------------------------------------------------ shared elementary pieces
def _fp32(x: torch.Tensor):
  """Context that switches autocast off on x's device: the float32 islands (relation einsums, logit assembly,
  softmax, gates, rms), as BT6 is trained ("BF16 compute with f32 softmax / gates / rms"). A no-op without autocast."""
  return torch.autocast(device_type=x.device.type, enabled=False)


def rms(x: torch.Tensor, dim: int, eps: float) -> torch.Tensor:
  """x / sqrt(mean x^2 + eps) over `dim` (no learned scale)."""
  return x * torch.rsqrt(x.pow(2).mean(dim, keepdim=True) + eps)


def policy_gather_index() -> torch.Tensor:
  """long [1858]: the position of every lc0 policy move in the 4,288-entry policy record of the attention policy
  head. Plain moves (knight promotions included) read from * 64 + to; q / r / b promotions read
  4096 + from_file * 24 + to_file * 3 + {q: 0, r: 1, b: 2}."""
  index = []
  for move in MOVES_1858:
    source = (ord(move[0]) - ord("a")) + 8 * (ord(move[1]) - ord("1"))
    target = (ord(move[2]) - ord("a")) + 8 * (ord(move[3]) - ord("1"))
    if len(move) == 5:
      index.append(4096 + (source & 7) * 24 + (target & 7) * 3 + "qrb".index(move[4]))
    else:
      index.append(source * 64 + target)
  return torch.tensor(index, dtype=torch.long)


def _promotion_record(logits: torch.Tensor, keys: torch.Tensor, promotion_weight: torch.Tensor) -> torch.Tensor:
  """[B, 64, 64] from-to logits + the keys [B, 64, width] -> the [B, 4288] policy record.

  off[f, 0..3] = K[56 + f] . W_promo[:, 0..3] for the eight last-rank keys; a pawn on file a of row 6 promoting on
  file f to piece c in {q, r, b} scores L[48 + a, 56 + f] + off[f, c] + off[f, 3] (kernel `promotion_logits`)."""
  batch = logits.shape[0]
  offsets = keys[:, 56:64] @ promotion_weight                        # [B, 8 (to file), 4]
  base = logits[:, 48:56, 56:64]                                     # [B, from file, to file]
  promotions = base[..., None] + offsets[:, None, :, 0:3] + offsets[:, None, :, 3:4]
  return torch.cat((logits.reshape(batch, 4096), promotions.reshape(batch, 192)), dim=1)


def policy_logits(queries, keys, promotion_weight, pair_bias, gather) -> torch.Tensor:
  """Attention-policy read-out shared by the main and auxiliary heads: (Q.K) / sqrt(width) + pair bias, promotion
  offsets (NOT divided), 1858 gather. `promotion_weight` is the nn.Linear weight [4, width]. Float32."""
  with _fp32(queries):
    queries, keys = queries.float(), keys.float()
    logits = torch.einsum("bid,bjd->bij", queries, keys) / math.sqrt(queries.shape[-1]) + pair_bias.float()
    record = _promotion_record(logits, keys, promotion_weight.float().T)
    return record[:, gather]


def _triplet(e_hat, value_w, gate_w, gate_b, out_w, heads: int, dots: int, contraction: str, eps: float):
  """The triplet branch of an edge site (kernel `triplet_site`): returns W_o (va_in | va_out), [B, S, 64, 64].

  All maps are [out, in] over the channel axis of [B, C, 64, 64] tensors."""
  batch, states = e_hat.shape[0], e_hat.shape[1]
  n = rms(e_hat, 1, eps)
  v = torch.einsum("oc,bcij->boij", value_w, n)                     # [B, 2S, 64, 64]
  eg = torch.einsum("oc,bcij->boij", gate_w, n) + gate_b[None, :, None, None]
  e_in, g_in, e_out, g_out = eg.split(heads, dim=1)                 # each [B, heads, 64, 64]
  a_in = torch.softmax(e_in, dim=-1) * torch.sigmoid(g_in)          # a_in[h, i, k]: softmax over the last axis
  a_out = torch.softmax(e_out, dim=-2) * torch.sigmoid(g_out)       # a_out[h, k, i]: softmax over the ROW axis
  v_in = v[:, :states].reshape(batch, heads, dots, 64, 64)          # channel = direction * S + head * dots + dot
  v_out = v[:, states:].reshape(batch, heads, dots, 64, 64)
  if contraction == "path":
    va_in = torch.einsum("bhik,bhdkj->bhdij", a_in, v_in)
    va_out = torch.einsum("bhki,bhdjk->bhdij", a_out, v_out)
  else:  # "ag"
    va_in = torch.einsum("bhik,bhdjk->bhdij", a_in, v_in)
    va_out = torch.einsum("bhki,bhdkj->bhdij", a_out, v_out)
  va = torch.cat((va_in.reshape(batch, states, 64, 64), va_out.reshape(batch, states, 64, 64)), dim=1)
  return torch.einsum("oc,bcij->boij", out_w, va)


# ====================================================================================================================
# 1. The forward straight from served tensors
# ====================================================================================================================
def served_to_torch(tensors: dict, device=None) -> dict:
  """Carrier arrays -> float32 torch tensors, skipping the integer (int8 / map) initializers."""
  out = {}
  for name, array in tensors.items():
    if isinstance(array, torch.Tensor):
      out[name] = array.float() if array.is_floating_point() else array
    elif array.dtype.kind == "f":
      out[name] = torch.from_numpy(np.ascontiguousarray(array).astype(np.float32)).to(device)
  return out


def _layer_norm(x, t, prefix, eps):
  return F.layer_norm(x, x.shape[-1:], t[f"{prefix}/scale"], t[f"{prefix}/bias"], eps)


def _dense(x, t, prefix, accumulate16: bool = False):
  """x W + b with the served [in, out] weight. `accumulate16` emulates the kernels' FP16 ACCUMULATOR: the GEMMs
  (CUTLASS and Triton `matmul`) add the products of 16 consecutive input channels per tensor-core instruction into
  an FP16 accumulator, i.e. the running sum is rounded to FP16 after every 16 terms; the bias is added in FP32."""
  weight = t[f"{prefix}/w"]
  if not accumulate16:
    return x @ weight + t[f"{prefix}/b"]
  total = torch.zeros(x.shape[:-1] + (weight.shape[1],), dtype=torch.float32, device=x.device)
  for start in range(0, weight.shape[0], 16):
    total = (total + x[..., start:start + 16] @ weight[start:start + 16]).half().float()
  return total + t[f"{prefix}/b"]


def _served_glu(x, t, prefix, r=lambda value: value, accumulate16: bool = False):
  """(sigmoid(x Wg) * (x Wu + bu)) Wd + bd from the packed [Wg | Wu] projection (gate columns first)."""
  packed = _dense(x, t, f"{prefix}/dense1", accumulate16)
  gate, up = packed.chunk(2, dim=-1)
  return _dense(r(torch.sigmoid(gate) * up), t, f"{prefix}/dense2", accumulate16)


def _identity(value):
  return value


def served_layer(t: dict, index: int, x, E, S, e, cfg: BT6Config, r=_identity, dense=None, glu=None, tap=None,
                 bias=None):
  """One encoder layer from the served tensors /encoder{index}/... (kernels: layer_norm, the packed [q|k|v|g] GEMM,
  attention_egt, the residual GEMMs, the GLU). x [B, 64, d], E float [B, 34, 64, 64], S [B, 64, 64],
  e [B, d_e, 64, 64] -> (x, H). `r` rounds what the FP16 program stores; `dense` / `glu` default to plain FP32."""
  dense = dense if dense is not None else (lambda value, prefix: _dense(value, t, prefix))
  glu = glu if glu is not None else (lambda value, prefix: _served_glu(value, t, prefix, r))
  tap = tap if tap is not None else (lambda name, value: value)
  batch, heads, hd, width = x.shape[0], cfg.heads, cfg.head_dim, cfg.d_model
  p = f"/encoder{index}"
  g = f"{p}/mha/egt"
  y = r(_layer_norm(x, t, f"{p}/ln1", cfg.norm_eps))
  packed = r(dense(y, f"{p}/mha/qkvg" if cfg.output_gate else f"{p}/mha/qkv"))
  e_read = r(e)                                                               # the attention reads an FP16 copy
  lanes = packed.view(batch, 64, 4 if cfg.output_gate else 3, heads, hd)
  q, k, v = lanes[:, :, 0], lanes[:, :, 1], lanes[:, :, 2]                    # [B, 64, heads, hd]
  scores = torch.einsum("bihd,bjhd->bhij", q, k) * t[f"{g}/qk_scale"].view(1, heads, 1, 1)
  row = torch.einsum("bihd,hcd->bhic", q, t[f"{g}/query"]) + t[f"{g}/attack"][None, :, None, :]
  column = torch.einsum("bjhd,hcd->bhjc", k, t[f"{g}/key"])
  scores = scores + torch.einsum("bhic,bcij->bhij", row, E) + torch.einsum("bhjc,bcij->bhij", column, E)
  scores = scores + S[:, None] * (torch.einsum("hc,bcij->bhij", t[f"{g}/scaled_coefficients"], E) + t[f"{g}/constant_bias"])
  if bias is not None:
    # per-position logit bias [B, heads, 64, 64] (NBT hybrid, shared smolgen; only given to term-free layers, whose
    # read and door are zero): H = q.k * qk_scale + bias
    scores = scores + bias
  read = torch.einsum("hc,bcij->bhij", t[f"{g}/edge_read/w"], e_read)
  door = torch.einsum("hc,bcij->bhij", t[f"{g}/door/w"], e_read)
  gate = torch.einsum("hc,bcij->bhij", t[f"{g}/gate/w"], e_read) + t[f"{g}/gate/b"].view(1, heads, 1, 1)
  H = (scores + read) * (1.0 + door)
  gate = 2.0 * torch.sigmoid(gate)
  if cfg.gate_cap:
    gate = gate.clamp(max=1.0)
  A = torch.softmax(H, dim=-1) * gate
  attended = torch.einsum("bhij,bjhd->bihd", A, v)
  if cfg.output_gate:
    attended = attended * (cfg.output_gate_scale * torch.sigmoid(lanes[:, :, 3]))
  x = tap(f"xa{index}", r(x + t[f"{p}/mha/alpha/w"] * dense(r(attended).reshape(batch, 64, width), f"{p}/mha/out")))
  x = r(x + t[f"{p}/ffn/alpha/w"] * glu(r(_layer_norm(x, t, f"{p}/ln2", cfg.norm_eps)), f"{p}/ffn"))
  return x, H


def served_site(t: dict, index: int, e, H, cfg: BT6Config):
  """The edge site /encoder{index}/edge_site/... (kernels edge_site + triplet_site): (e, H of that layer) -> e'."""
  s = f"/encoder{index}/edge_site"
  e_hat = e + torch.einsum("ch,bhij->bcij", t[f"{s}/readback/w"], H)
  if cfg.triplet:
    e_hat = e_hat + _triplet(e_hat, t[f"{s}/triplet/value/w"], t[f"{s}/triplet/gate/w"], t[f"{s}/triplet/gate/b"],
                             t[f"{s}/triplet/out/w"], cfg.triplet_heads, cfg.triplet_dots,
                             cfg.triplet_contraction, cfg.rms_eps)
  n = rms(e_hat, 1, cfg.rms_eps)
  hidden = torch.einsum("oc,bcij->boij", t[f"{s}/ffn/dense1/w"], n) + t[f"{s}/ffn/dense1/b"][None, :, None, None]
  if cfg.rev_edge:
    hidden = hidden + torch.einsum("oc,bcij->boij", t[f"{s}/ffn/dense1_rev/w"], n.transpose(-1, -2))
  return rms(e_hat + torch.einsum("oc,bcij->boij", t[f"{s}/ffn/dense2/w"], torch.relu(hidden)), 1, cfg.rms_eps)


def served_prologue(t: dict, E_bool, cfg: BT6Config, r=_identity):
  """(S [B, 64, 64], e0 [B, d_e, 64, 64]) from the packed relation bits through /prologue/mix_codes (prologue_egt)."""
  mixed = 0
  for (first, bits), base in zip(CODE_GROUPS, CODE_GROUP_BASES):
    weights = (2 ** torch.arange(bits, device=E_bool.device)).view(1, bits, 1, 1)
    codes = (E_bool[:, first:first + bits].long() * weights).sum(1)          # [B, 64, 64]
    mixed = mixed + t["/prologue/mix_codes"][base + codes]                    # [B, 64, 64, 32]
  pair = mixed[..., :cfg.pair_channels].permute(0, 3, 1, 2) + t["/prologue/offsets"]
  S = r(torch.rsqrt(pair.pow(2).mean(1) + cfg.rms_eps))                       # FP16 in the program
  state = mixed[..., cfg.pair_channels:].permute(0, 3, 1, 2) + t["/prologue/edge_offsets"]
  return S, rms(state, 1, cfg.rms_eps)


def served_policy(t: dict, x, E, S, r=_identity, dense=None):
  """Attention policy head from /policy/...: final-normed trunk [B, 64, D] -> logits [B, 1858]."""
  dense = dense if dense is not None else (lambda value, prefix: _dense(value, t, prefix))
  embedded = r(F.mish(dense(x, "/policy/embedding")))
  queries, keys = r(dense(embedded, "/policy/Q")), r(dense(embedded, "/policy/K"))
  logits = r(torch.einsum("bid,bjd->bij", queries, keys) * t["/policy/qk_scale"])
  logits = r(logits + S * (torch.einsum("c,bcij->bij", t["/policy/pair/scaled_coefficients"].view(-1), E)
                           + t["/policy/pair/constant_bias"].view(64, 64)))
  record = r(_promotion_record(logits, keys, t["/policy/promotion/w"]))
  return record[:, policy_gather_index().to(record.device)]


@torch.no_grad()
def forward_served(t: dict, planes: torch.Tensor, config: BT6Config, relations: BT6Relations | None = None,
                   taps: dict | None = None, round16: bool = False, accumulate16: bool = False,
                   force_stream: dict | None = None) -> dict:
  """The BT6 forward from served tensors `t` ({carrier name: float32 tensor}, see `served_to_torch`).

  `taps`, when given, is filled with intermediates (E, S, e0, x after the embedding, per-layer stream, H at sites).
  `round16` rounds every activation the FP16 program STORES in FP16 (GEMM / norm / attention outputs, S, the state
  copy the attention reads, the policy record) to FP16 and back: not bit-equal to the kernels (their accumulation
  order differs), but the same class of rounding, for telling FP16 noise from a systematic deviation.
  `accumulate16` additionally emulates the FP16 accumulators of the program's GEMMs (see `_dense`).
  `force_stream` ({tap name: [B, 64, D]}) REPLACES the residual stream at the named taps ("embedding_pre",
  "xa{i}" after the attention branch, "x{i}" after the FFN branch of layer i) after the tap was recorded: with the
  real program's stream as the forced value every layer is compared in isolation (teacher forcing)."""
  cfg = config
  r = (lambda value: value.half().float()) if round16 else _identity

  def dense(x, prefix):
    return _dense(x, t, prefix, accumulate16)

  def glu(x, prefix):
    return _served_glu(x, t, prefix, r, accumulate16)

  def tap(name, value):
    if taps is not None:
      taps[name] = value
    if force_stream is not None and name in force_stream:
      return force_stream[name].to(value.dtype)
    return value

  relations = relations if relations is not None else BT6Relations().to(planes.device)
  batch, width = planes.shape[0], cfg.d_model
  planes = planes.float()

  # ---- prologue (kernel prologue_egt): E, then S and e0 through the code-row table
  E_bool = relations(planes[:, 0:12] > 0.5, planes[:, 13:25] > 0.5)
  E = E_bool.float()
  S, e = served_prologue(t, E_bool, cfg, r)
  if taps is not None:
    taps.update(E=E_bool, S=S, e0=e)

  # ---- embedding
  per_square = planes.permute(0, 2, 1)                                        # [B, 64, 112]
  position = r(dense(per_square[:, :, :12].reshape(batch, 768), "/attn_body/preproc")).view(batch, 64, -1)
  x = r(dense(torch.cat((per_square, position), dim=-1), "/attn_body/matmul"))
  x = r(_layer_norm(F.mish(x), t, "/attn_body/ln0", cfg.norm_eps))
  x = r(x * t["/attn_body/mult_gate/w"] + t["/attn_body/position/w"].view(64, width))
  x = tap("embedding_pre", r(x + t["/attn_body/ffn/alpha/w"] * glu(x, "/attn_body/ffn")))
  x = r(_layer_norm(x, t, "/attn_body/ln1", cfg.norm_eps))
  if taps is not None:
    taps["embedding"] = x

  # ---- tower
  for index in range(cfg.layers):
    x, H = served_layer(t, index, x, E, S, e, cfg, r=r, dense=dense, glu=glu, tap=tap)
    if index in cfg.edge_sites:
      e = served_site(t, index, e, H, cfg)
      if taps is not None:
        taps[f"H{index}"], taps[f"e{index}"] = H, e
    x = tap(f"x{index}", x)
  x = r(_layer_norm(x, t, "/encoder/final_norm", cfg.norm_eps))

  # ---- heads
  out = {"policy": served_policy(t, x, E, S, r=r, dense=dense)}

  def dense_head(prefix):
    square = r(F.mish(dense(x, f"{prefix}/embedding"))).reshape(batch, -1)
    return dense(r(F.mish(dense(square, f"{prefix}/dense1"))), f"{prefix}/dense2")

  out["wdl"] = dense_head("/value")
  out["mlh"] = torch.relu(dense_head("/moves"))
  if cfg.error_head:
    out["err"] = torch.sigmoid(dense_head("/value_error"))
  return out


# ====================================================================================================================
# 2. The module in trained-parameter form
# ====================================================================================================================
def _linear_init(linear: nn.Linear, zero: bool = False) -> None:
  if zero:
    nn.init.zeros_(linear.weight)
  else:
    nn.init.xavier_normal_(linear.weight)
  if linear.bias is not None:
    nn.init.zeros_(linear.bias)


class BT6GLU(nn.Module):
  """Dauphin GLU FFN: (sigmoid(x Wg) * (x Wu + bu)) Wd + bd. The gate projection has NO bias."""

  def __init__(self, width: int, hidden: int) -> None:
    super().__init__()
    self.gate = nn.Linear(width, hidden, bias=False)
    self.up = nn.Linear(width, hidden)
    self.down = nn.Linear(hidden, width)
    for linear in (self.gate, self.up, self.down):
      _linear_init(linear)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down(torch.sigmoid(self.gate(x)) * self.up(x))


def _wide(dtype):
  """float32 for the half / single precision dtypes (the dtype of softmax and gates), else the dtype itself."""
  return torch.float32 if dtype in (torch.bfloat16, torch.float16, torch.float32) else dtype


def _core_forward(L, door, gate_logits, values, cap, out_logits, out_scale, low=False):
  """The statements of the attention core, out of place (differentiable by plain autograd). `low`: softmax and
  post-softmax gate in the dtype of L (the pair_autocast lever) instead of float32."""
  wide = L.dtype if low else _wide(L.dtype)
  H = torch.addcmul(L, L, door)                                              # (F + read) (1 + door)
  gate = torch.sigmoid(gate_logits.to(wide)) * 2.0
  if cap:
    gate = gate.clamp(max=1.0)
  A = torch.softmax(H, dim=-1, dtype=wide) * gate                            # float32; rows no longer sum to 1
  attended = torch.matmul(A.to(values.dtype), values)                        # A.v in the compute dtype
  if out_logits is None:
    return attended, H, gate, attended
  out_gate = (torch.sigmoid(out_logits.to(_wide(L.dtype))) * out_scale).to(attended.dtype)
  return attended * out_gate, H, gate, attended


class _AttentionCoreFunction(torch.autograd.Function):
  """door -> softmax -> post-softmax gate -> A.v -> output gate as ONE autograd node (see `attention_core`).

  forward saves only L, door, the gate values, v and (with an output gate) A.v and its logits; backward recomputes
  H and the softmax. Plain autograd on the same statements keeps H, the softmax output, the sigmoid output, the
  gate, A and their casts alive per layer."""

  @staticmethod
  def forward(ctx, L, door, gate_logits, values, out_logits, cap, out_scale, low=False):
    values = values.contiguous()
    out, H, gate, attended = _core_forward(L, door, gate_logits, values, cap, out_logits, out_scale, low)
    ctx.cap, ctx.out_scale = cap, out_scale
    # an output nobody differentiates arrives as None in backward (not a zero tensor): H is read only by the edge
    # sites, so in every other layer backward skips the [B, heads, 64, 64] cast-and-add of a zero gradient
    ctx.set_materialize_grads(False)
    if out_logits is None:
      ctx.save_for_backward(L, door, gate, values)
    else:
      ctx.save_for_backward(L, door, gate, values, attended, out_logits)
    return out, H

  @staticmethod
  def backward(ctx, d_out, d_H):
    saved = ctx.saved_tensors
    L, door, gate, values = saved[:4]
    wide = gate.dtype
    d_L = d_door = d_gate_logits = d_values = d_out_logits = None
    d_Hw = None
    if d_out is not None:
      if len(saved) > 4:
        attended, out_logits = saved[4], saved[5]
        wide_out = _wide(L.dtype)                                              # the output gate is float32 always
        sig = torch.sigmoid(out_logits.to(wide_out))
        if ctx.needs_input_grad[4]:
          # d(scale * sigmoid(z)) / dz = scale * sig * (1 - sig)
          d_out_logits = (d_out.to(wide_out) * attended.to(wide_out) * sig * (1.0 - sig) * ctx.out_scale).to(out_logits.dtype)
        d_out = d_out * sig.mul_(ctx.out_scale).to(d_out.dtype)
      d_out = d_out.to(values.dtype)
      P = torch.softmax(torch.addcmul(L, L, door), dim=-1, dtype=wide)
      d_A = torch.matmul(d_out, values.transpose(-1, -2)).to(wide)           # [B, heads, 64, 64]
      if ctx.needs_input_grad[3]:
        d_values = torch.matmul((P * gate).to(values.dtype).transpose(-1, -2), d_out)
      if ctx.needs_input_grad[2]:
        d_gate_logits = d_A * P * gate * (1.0 - 0.5 * gate)                  # d(2 sigmoid)/dz = g (1 - g / 2)
        if ctx.cap:
          d_gate_logits = d_gate_logits * (gate < 1.0)
      d_P = d_A.mul_(gate)
      d_Hw = P.mul_(d_P - (d_P * P).sum(dim=-1, keepdim=True))               # softmax backward (P reused in place)
    if d_H is not None:
      d_Hw = d_H.to(wide) if d_Hw is None else d_Hw.add_(d_H.to(wide))
    if d_Hw is not None:
      d_Hl = d_Hw.to(L.dtype)
      if ctx.needs_input_grad[0]:
        d_L = torch.addcmul(d_Hl, d_Hl, door)                                # dH (1 + door)
      if ctx.needs_input_grad[1]:
        d_door = d_Hl * L
    return d_L, d_door, d_gate_logits, d_values, d_out_logits, None, None, None


def attention_core(L, door, gate_logits, values, cap: bool = False, out_logits=None, out_scale: float = 2.0,
                   custom: bool | None = None, low: bool = False):
  """H = L (1 + door);  A = softmax_j(H) * 2 sigmoid(gate_logits) [capped at 1 if `cap`];
  out = (A @ values) * out_scale * sigmoid(out_logits)   (the output gate; skipped when out_logits is None).

  L, door [B, heads, 64, 64] in the assembly dtype; gate_logits float32; values, out_logits [B, heads, 64, head_dim]
  in the compute dtype. Returns (out [B, heads, 64, head_dim] in the compute dtype, H in the assembly dtype).
  The softmax and both sigmoids are float32; A is cast to the compute dtype for A.v and the gate product (as the
  original training code does). `custom` (default: when gradients are recorded) selects the memory-saving autograd
  function; otherwise the same statements run under plain autograd (export, eval, reference for the tests)."""
  if custom is None:
    custom = torch.is_grad_enabled() and (L.requires_grad or door.requires_grad or values.requires_grad)
  if custom:
    return _AttentionCoreFunction.apply(L, door, gate_logits, values, out_logits, cap, out_scale, low)
  out, H, _, _ = _core_forward(L, door, gate_logits, values, cap, out_logits, out_scale, low)
  return out, H


def _make_reader_parameters(module: nn.Module, cfg: "BT6Config") -> None:
  heads = cfg.heads
  module.t_node = nn.Parameter(torch.ones(heads))                           # temperature of every static term
  module.t_edge = nn.Parameter(torch.zeros(heads))                          # temperature of the edge read
  module.C1 = nn.Parameter(torch.zeros(heads, cfg.relation_channels))       # "attack": per-channel bias on E
  module.C2 = nn.Parameter(torch.zeros(heads, cfg.pair_channels))           # pair-stream read
  module.We = nn.Parameter(torch.randn(heads, cfg.edge_channels) / cfg.edge_channels)  # edge read, N(0, 1/16)
  module.Wm = nn.Parameter(torch.zeros(heads, cfg.edge_channels))           # door
  module.Wg = nn.Parameter(torch.zeros(heads, cfg.edge_channels))           # post-softmax gate
  module.bg = nn.Parameter(torch.zeros(heads))


READER_NAMES = ("t_node", "t_edge", "C1", "C2", "We", "Wm", "Wg", "bg")


class BT6Readers(nn.Module):
  """The reader parameters of a BT6 layer that do not touch q / k: scales t_node, t_edge and the tables C1 (relation
  bias), C2 (pair-stream read), We (edge read), Wm (door), Wg, bg (post-softmax gate). A BT6Layer owns these as its
  own parameters (BT6); with the BT6ShareReadersPerBlock lever ONE BT6Readers per NBT block is used by all its inner
  layers (Pk / Pq stay per layer)."""

  def __init__(self, cfg: "BT6Config") -> None:
    super().__init__()
    _make_reader_parameters(self, cfg)


def stacked_reads(sources, cfg: "BT6Config", e: torch.Tensor, read_state: torch.Tensor):
  """Everything of the logits that depends only on E, Z0, the edge state and the reader parameters, for the reader
  sets `sources` (BT6Layers or BT6Readers) in ONE matmul against read_state = [e | Z0]:
      read  = t_edge (We . e) + t_node (C2 . Z0)     door = Wm . e     gate logits = Wg . e + bg
  Returns per source (read, door, gate_logits), each [B, heads, 64, 64]; read / door in read_state's dtype, the gate
  logits float32 (the dtype of read_state with cfg.pair_autocast). One source = the per-layer form."""
  batch, heads, count = e.shape[0], cfg.heads, len(sources)
  pd = read_state.dtype
  low = cfg.pair_autocast and pd in (torch.bfloat16, torch.float16)
  zero = torch.zeros(heads, read_state.shape[1] - e.shape[1], dtype=torch.float32, device=e.device)
  rows = []
  for R in sources:
    rows.append(torch.cat((R.We.float() * R.t_edge.float()[:, None], R.C2.float() * R.t_node.float()[:, None]), dim=1))
    rows.append(torch.cat((R.Wm.float(), zero), dim=1))
    if low:
      rows.append(torch.cat((R.Wg.float(), zero), dim=1))
  groups = 3 if low else 2
  out = torch.bmm(torch.cat(rows, dim=0).to(pd)[None].expand(batch, -1, -1), read_state).view(batch, count, groups, heads, 64, 64)
  if low:
    gates = [out[:, i, 2] + R.bg.to(pd).view(1, heads, 1, 1) for i, R in enumerate(sources)]
  else:
    e32 = e.float().reshape(batch, -1, 4096)
    Wg = torch.cat([R.Wg.float() for R in sources], dim=0)
    bg = torch.cat([R.bg.float() for R in sources], dim=0)
    gate = torch.baddbmm(bg.view(1, -1, 1).expand(batch, -1, e32.shape[-1]), Wg[None].expand(batch, -1, -1), e32)
    gate = gate.view(batch, count, heads, 64, 64)
    gates = [gate[:, i] for i in range(count)]
  return [(out[:, i, 0], out[:, i, 1], gates[i]) for i in range(count)]


class RelationContext:
  """The relation data of one forward, shared by every layer: the relation tensor in the layouts the layers contract
  it in, the pair normaliser, and casts of the edge state. Built once (BT6Prologue.context); the layouts are created
  on first use and cached per dtype, so 24-30 layers do not each re-permute / re-cast a [B, 34, 64, 64] tensor.

    E          float32 [B, 34, 64, 64]  (c, i, j)                                  no gradient
    Ei(dtype)  [B, 64, 34, 64]  = E[b, c, i, j] at [b, i, c, j]: a batch of [34, 64] matrices per (b, i)
    Z0         float32 [B, P, 64, 64]   the static pair stream S * (D.E + T[relidx])   (gradient to D, T)
    read_state(e, dtype)  [B, d_e + P, 4096] = [e | Z0]: one matmul reads the edge state and the pair stream
  No lazily filled dictionaries and no id()-keyed caches: the object must be traceable by torch.compile and must
  give a checkpointed block the same tensors in its first pass and in its recompute.
    state(e, dtype)  the edge state [B, d_e, 4096] in `dtype` (cached per state version)
  Because E[17 + c, i, j] = E[c, j, i], the layout a KEY-side contraction needs, E[b, c, i, j] at [b, j, c, i], is
  Ei with its two channel halves swapped -- the layers roll their key tables by 17 channels instead of building it."""

  def __init__(self, E: torch.Tensor, S: torch.Tensor, Z0: torch.Tensor | None = None) -> None:
    self.E, self.S, self.Z0 = E, S, Z0      # Z0: the pair stream S * (D.E + T[relidx]), float32 [B, P, 64, 64]
    self._Ei, self._Ei_dtype = None, None

  def Ei(self, dtype):
    if self._Ei is None or self._Ei_dtype != dtype:
      self._Ei, self._Ei_dtype = self.E.permute(0, 2, 1, 3).to(dtype).contiguous(), dtype
    return self._Ei

  def read_state(self, e: torch.Tensor, dtype):
    """[e | Z0] as [B, d_e + P, 4096] in `dtype`: what the stacked edge-read / pair-read / door matmul contracts.
    Not cached: the caller builds it once per state version (outside a checkpointed block) and hands it to the
    layers that read that state."""
    batch = e.shape[0]
    return torch.cat((e.to(dtype).reshape(batch, -1, 4096), self.Z0.to(dtype).reshape(batch, -1, 4096)), dim=1)

  def prepare(self, dtype):
    """Build the relation layout for `dtype` NOW. Callers that run layers inside torch.utils.checkpoint must call
    this OUTSIDE the checkpointed function (a layout built during the first pass and reused in the recompute would
    make the two passes save different tensors)."""
    self.Ei(dtype)
    return self


class BT6Prologue(nn.Module):
  """Pair stream and edge-state seed: D [34, P], T [225, P], Pin [34, S], Toff [225, S] (all init N(0, 0.5))."""

  def __init__(self, cfg: BT6Config) -> None:
    super().__init__()
    self.eps = cfg.rms_eps
    self.D = nn.Parameter(torch.randn(cfg.relation_channels, cfg.pair_channels) * 0.5)
    self.T = nn.Parameter(torch.randn(RELATIVE_BINS, cfg.pair_channels) * 0.5)
    self.Pin = nn.Parameter(torch.randn(cfg.relation_channels, cfg.edge_channels) * 0.5)
    self.Toff = nn.Parameter(torch.randn(RELATIVE_BINS, cfg.edge_channels) * 0.5)
    self.register_buffer("relidx", relative_index(), persistent=False)

  def forward(self, E: torch.Tensor, return_pair_stream: bool = False):
    """E float [B, 34, 64, 64] -> (S [B, 64, 64], e0 [B, S, 64, 64]) and, on request, the pair stream Z0 = S * Z
    [B, P, 64, 64]. A reader sum_d c[d] Z0[d] equals S * ((c D^T).E + (c T^T)[relidx]) (`pair_read`, the form the
    served tables are folded to)."""
    with _fp32(E):
      E = E.float()
      Z = torch.einsum("bcij,cd->bdij", E, self.D.float()) + self.T.float()[self.relidx].permute(2, 0, 1)
      S = torch.rsqrt(Z.pow(2).mean(1) + self.eps)
      e0 = rms(torch.einsum("bcij,cd->bdij", E, self.Pin.float()) + self.Toff.float()[self.relidx].permute(2, 0, 1),
               1, self.eps)
      if return_pair_stream:
        return S, e0, Z * S[:, None]
    return S, e0

  def context(self, E: torch.Tensor):
    """E (bool or float [B, 34, 64, 64]) -> (RelationContext, S, e0): what one forward shares between its layers."""
    E = E.float()
    S, e0, Z0 = self.forward(E, return_pair_stream=True)
    return RelationContext(E, S, Z0), S, e0

  def pair_read(self, coefficients: torch.Tensor):
    """Reader coefficients c [H, P] -> (G [H, 34], K [H, 64, 64]) with sum_d c[h, d] Z0[d] = S * (G.E + K)."""
    coefficients = coefficients.float()
    return coefficients @ self.D.float().T, (coefficients @ self.T.float().T)[:, self.relidx]


class BT6Layer(nn.Module):
  """One pre-norm BT6 encoder layer with its own relation / pair / edge-read parameters.

  Config levers of the NBT hybrid (defaults = BT6): `terms=False` builds the layer WITHOUT any relation / pair /
  edge term and without their parameters (plain QKVG attention with the output gate, LayerNorm, sigmoid GLU:
  exactly a BT6 layer whose tables are zero, t_node = 1, post-softmax gate 2 sigmoid(0 + 0) = 1);
  `share_readers` makes the layer use a block's BT6Readers."""

  def __init__(self, cfg: BT6Config, terms: bool = True) -> None:
    super().__init__()
    self.cfg = cfg
    self.terms = bool(terms)
    object.__setattr__(self, "_shared", None)      # a block's BT6Readers (not a submodule of the layer)
    width, heads, hd, channels = cfg.d_model, cfg.heads, cfg.head_dim, cfg.relation_channels
    self.ln1 = nn.LayerNorm(width, eps=cfg.norm_eps)
    self.q, self.k, self.v = nn.Linear(width, width), nn.Linear(width, width), nn.Linear(width, width)
    self.g = nn.Linear(width, width) if cfg.output_gate else None          # attention output gate, zero init
    self.out = nn.Linear(width, width)
    if self.terms:
      # the only random draw among these is We, at the same place in the random stream as before the levers
      _make_reader_parameters(self, cfg)
      self.Pk = nn.Parameter(torch.zeros(heads, channels, hd))              # "key":   (k_j . Pk[h, c]) E[c, i, j]
      self.Pq = nn.Parameter(torch.zeros(heads, channels, hd))              # "query": (q_i . Pq[h, c]) E[c, i, j]
    self.ln2 = nn.LayerNorm(width, eps=cfg.norm_eps)
    self.ffn = BT6GLU(width, cfg.ffn_hidden)
    self.num_attention_heads = heads      # read by nbt_layer.NestedBottleneckLayer
    # "both residual scales are 1" (they are constants, 1.0 under pre-norm): a plain Python flag, so the forward
    # has no data-dependent branch (torch.compile). load_served / set_alphas keep it in step with the buffers.
    self._unit_alphas = True
    self.pre_norm = True
    # residual scales: constants (1.0 under pre-norm), kept as buffers so an exported carrier names them
    self.register_buffer("alpha_attention", torch.ones(1))
    self.register_buffer("alpha_ffn", torch.ones(1))
    for linear in (self.q, self.k, self.v, self.out):
      _linear_init(linear)
    if self.g is not None:
      _linear_init(self.g, zero=True)
    # QK-clip (config QKClipTau; armed by train.py before torch.compile): a plain Python flag, so the training-only
    # stash branch specializes into the compiled graph; nothing of it reaches eval / export.
    self.qk_clip_monitor = False
    self._last_max_logit = None

  # ------------------------------------------------------------------ QK-clip (MuonClip-style, weight level)
  @property
  def num_heads(self) -> int:
    return self.cfg.heads

  def _stash_max_logit(self, H: torch.Tensor) -> None:
    """Per-head max of the logits that enter the softmax (training only), max-accumulated over the micro-batches
    of one optimizer step; `apply_qk_clip` reads and resets it. Term layer: H = (q.k + relation terms + read)
    (1 + door); plain layer: q.k / sqrt(hd) (+ the shared-smolgen bias)."""
    mx = H.detach().amax(dim=(0, 2, 3)).float()
    prev = self._last_max_logit
    self._last_max_logit = mx if prev is None else torch.maximum(mx, prev)

  @torch.no_grad()
  def apply_qk_clip(self, tau: float) -> int:
    """After the optimizer step: every head whose max logit exceeded `tau` gets gamma = tau / max and
    the q and k projections (weights AND biases) of that head scaled by sqrt(gamma); in a term layer also Pq and
    Pk, so the three q/k-dependent logit terms q.k, q.Pq E and k.Pk E all shrink by exactly gamma (C1, the
    edge / pair-stream read and the door are not q/k-dependent and stay). Returns the number of heads clipped."""
    m = self._last_max_logit
    self._last_max_logit = None
    if m is None:
      return 0
    gamma = (tau / m.clamp(min=1e-6)).clamp(max=1.0)
    clipped = int((gamma < 0.9999).sum().item())
    if clipped == 0:
      return 0
    s = gamma.sqrt()
    hd = self.cfg.head_dim
    rows = s.repeat_interleave(hd).to(self.q.weight.dtype)                  # [heads * hd], head-major as the packing
    for lin in (self.q, self.k):
      lin.weight.mul_(rows[:, None])
      if lin.bias is not None:
        lin.bias.mul_(rows)
    if self.terms:
      self.Pq.mul_(s.to(self.Pq.dtype)[:, None, None])
      self.Pk.mul_(s.to(self.Pk.dtype)[:, None, None])
    return clipped

  # ------------------------------------------------------------------ config levers (hybrid)
  @property
  def readers(self):
    """Where t_node, t_edge, C1, C2, We, Wm, Wg, bg live: the layer itself, or the block's shared BT6Readers."""
    return self if self._shared is None else self._shared

  def share_readers(self, readers: "BT6Readers") -> None:
    """Drop the layer's own reader parameters and use `readers` (owned by the NBT block)."""
    assert self.terms
    for name in READER_NAMES:
      delattr(self, name)
    object.__setattr__(self, "_shared", readers)

  def reader_tensors(self) -> dict:
    """The ten reader tensors as BT6 has them per layer (what the export folds): own, shared, or -- for a layer
    without terms -- the zero tables that make a BT6 layer plain."""
    cfg = self.cfg
    if not self.terms:
      z = lambda *shape: torch.zeros(*shape, device=self.q.weight.device)
      return {"t_node": torch.ones(cfg.heads, device=self.q.weight.device), "t_edge": z(cfg.heads),
              "C1": z(cfg.heads, cfg.relation_channels), "C2": z(cfg.heads, cfg.pair_channels),
              "We": z(cfg.heads, cfg.edge_channels), "Wm": z(cfg.heads, cfg.edge_channels),
              "Wg": z(cfg.heads, cfg.edge_channels), "bg": z(cfg.heads),
              "Pk": z(cfg.heads, cfg.relation_channels, cfg.head_dim), "Pq": z(cfg.heads, cfg.relation_channels, cfg.head_dim)}
    R = self.readers
    return {**{name: getattr(R, name) for name in READER_NAMES}, "Pk": self.Pk, "Pq": self.Pq}

  def _forward_plain(self, x, want_H: bool, bias=None):
    """The layer without relation / pair / edge terms: softmax_j(H[i, j]) v_j with the output gate, where
        H[h, i, j] = q_i.k_j / sqrt(hd) + bias[h, i, j]
    `bias` [B, heads, 64, 64] (None = no bias) is the block's shared-smolgen logit bias (config BT6SharedSmolgen): it
    enters the logits before the softmax, in the dtype of q / k; the output gate is unchanged. H (computed only when
    asked for: an edge site reads it) INCLUDES the bias, as nbt_layer._raw_logits did for our earlier nets."""
    cfg = self.cfg
    batch, heads, hd = x.shape[0], cfg.heads, cfg.head_dim
    projections = [self.q, self.k, self.v] + ([self.g] if self.g is not None else [])
    packed = F.linear(self.ln1(x), torch.cat([m.weight for m in projections], dim=0), torch.cat([m.bias for m in projections], dim=0))
    lanes = packed.view(batch, 64, len(projections), heads, hd).permute(0, 2, 3, 1, 4)    # [B, lane, heads, 64, hd]
    H = None
    if self.training:
      # training: the fused attention primitive (softmax in the compute dtype, as every plain layer of this trainer)
      mask = None if bias is None else bias.to(lanes.dtype)       # a float attn_mask is ADDED to the logits
      attended = F.scaled_dot_product_attention(lanes[:, 0], lanes[:, 1], lanes[:, 2], attn_mask=mask)
      if want_H or self.qk_clip_monitor:
        H = torch.matmul(lanes[:, 0], lanes[:, 1].transpose(-1, -2)) / math.sqrt(hd)
        H = H if mask is None else H + mask
        if self.qk_clip_monitor:
          self._stash_max_logit(H)
        H = H if want_H else None
    else:
      # eval / export: the explicit statements with a float32 softmax (ONNX gets MatMul / Softmax, no Attention op)
      H = torch.matmul(lanes[:, 0], lanes[:, 1].transpose(-1, -2)) / math.sqrt(hd)
      H = H if bias is None else H + bias.to(H.dtype)
      attended = torch.matmul(torch.softmax(H.float(), dim=-1).to(lanes.dtype), lanes[:, 2])
      H = H if want_H else None
    if self.g is not None:
      attended = attended * (cfg.output_gate_scale * torch.sigmoid(lanes[:, 3].float())).to(attended.dtype)
    attended = attended.permute(0, 2, 1, 3).reshape(batch, 64, cfg.d_model)
    x = x + self.out(attended).to(x.dtype)
    x = x + self.ffn(self.ln2(x)).to(x.dtype)
    return x, H

  def forward(self, x, E, S, e, prologue: BT6Prologue, read_state: torch.Tensor | None = None, reads=None,
              want_H: bool = True, bias=None):
    """x [B, 64, D], E = RelationContext (or the float relation tensor [B, 34, 64, 64]), S [B, 64, 64],
    e [B, S, 64, 64] -> (x, H [B, heads, 64, 64] in the assembly dtype). `read_state` = ctx.read_state(e, dtype),
    when the caller has built it for all layers that read this state (else it is built here). `reads` = this
    layer's (read, door, gate_logits) from `stacked_reads`, when the block computed them for its layers.
    `want_H` matters only for a layer without terms (its H costs an extra matmul). `bias`: the block's
    shared-smolgen logit bias, used ONLY by a layer without terms (`_forward_plain`); a term layer ignores it.

    The same function as `forward_reference` (the literal einsum form), arranged for speed, close to the
    formulation of the original JAX training code (dense contractions, materialised pair stream, bf16 assembly):
      * t_node is folded into the small operands (q, Pq, C1, Pk, C2) instead of scaling a pairwise tensor;
      * the relation terms are two batched matmuls against the pre-permuted relation tensor of the context
        (query side: rows q.Pq + C1; key side: k.Pk with the channel halves rolled);
      * edge read, pair-stream read and door are ONE matmul of stacked weights against [e | Z0] (Z0 = the static
        pair stream S (D.E + T), materialised once per forward);
      * q, k, v, g are ONE packed projection with t_node / sqrt(hd) folded into the q rows of the weight;
      * q.k is added onto the read by baddbmm, the relation terms in place;
      * door -> softmax -> post-softmax gate -> A.v -> output gate is `attention_core`: in training a custom
        autograd function whose backward recomputes the softmax instead of keeping the pairwise tensors alive;
        A.v and the output-gate product run in the compute dtype (A cast from float32), as the JAX code does.
    Precision: softmax, both gates and (in the sites / prologue) every rms are float32 always. The logit ASSEMBLY
    (q.k, relation and pair terms, edge read, door product) runs in the autocast dtype when
    cfg.assembly_autocast is set and autocast is on (bf16 compute, as BT6 is trained), else in float32; H is cast
    to float32 before the softmax either way. Without autocast everything is float32."""
    if not self.terms:
      return self._forward_plain(x, want_H, bias)
    cfg = self.cfg
    R = self.readers
    batch, heads, hd = x.shape[0], cfg.heads, cfg.head_dim
    if isinstance(E, RelationContext):
      ctx = E
    else:                                    # a plain relation tensor: build the context here (slow path)
      with _fp32(x):
        G_all = torch.einsum("bcij,cd->bdij", E.float(), prologue.D.float()) + prologue.T.float()[prologue.relidx].permute(2, 0, 1)
        ctx = RelationContext(E.float(), S.float(), G_all * S.float()[:, None])
    lanes_n = 4 if self.g is not None else 3
    t = R.t_node.float()
    # One packed projection [q | k | v | g] of the normed input; the per-head scale t_node / sqrt(hd) of the logits
    # is folded into the q rows of the weight, so no [B, 64, D] tensor is scaled afterwards: q' = q t / sqrt(hd).
    q_scale = (t / math.sqrt(hd))[:, None].expand(heads, hd).reshape(-1)     # (no repeat_interleave: ONNX export)
    projections = [self.k, self.v] + ([self.g] if self.g is not None else [])
    weight = torch.cat([self.q.weight * q_scale[:, None]] + [m.weight for m in projections], dim=0)
    bias = torch.cat([self.q.bias * q_scale] + [m.bias for m in projections], dim=0)
    packed = F.linear(self.ln1(x), weight, bias)                              # [B, 64, lanes * D], compute dtype
    low = cfg.assembly_autocast and packed.dtype in (torch.bfloat16, torch.float16)
    pd = packed.dtype if low else torch.float32
    with _fp32(x):
      lanes = packed.view(batch, 64, lanes_n, heads, hd).permute(0, 2, 3, 1, 4)           # [B, lane, heads, 64, hd]
      qs = lanes[:, 0].to(pd).reshape(batch * heads, 64, hd)                  # q' (scaled)
      kb = lanes[:, 1].to(pd).reshape(batch * heads, 64, hd)
      # ---- relation terms: per (b, h) the [64, 34] coefficient rows, then per (b, square) a matmul with E
      #      query side -> [b, i, h, j];  key side (channel halves rolled, E[17 + c, i, j] = E[c, j, i]) -> [b, j, h, i]
      Ei = ctx.Ei(pd).view(batch * 64, -1, 64)
      Pq = (self.Pq.float() * math.sqrt(hd)).to(pd).transpose(1, 2)           # t (q.Pq) = q'.(sqrt(hd) Pq)
      Pk = (self.Pk.float() * t[:, None, None]).roll(BASE_RELATION_CHANNELS, 1).to(pd).transpose(1, 2)
      rows = torch.bmm(qs, Pq[None].expand(batch, -1, -1, -1).reshape(batch * heads, hd, -1)).view(batch, heads, 64, -1)
      rows = (rows + (R.C1.float() * t[:, None]).to(pd)[None, :, None, :]).permute(0, 2, 1, 3).reshape(batch * 64, heads, -1)
      keys = torch.bmm(kb, Pk[None].expand(batch, -1, -1, -1).reshape(batch * heads, hd, -1)).view(batch, heads, 64, -1)
      keys = keys.permute(0, 2, 1, 3).reshape(batch * 64, heads, -1)
      rel_q = torch.bmm(rows, Ei).view(batch, 64, heads, 64)
      rel_k = torch.bmm(keys, Ei).view(batch, 64, heads, 64)
      # ---- edge read + pair-stream read, the door and the gate logits: ONE batched matmul of the stacked reader
      #      tables against [e | Z0] (stacked_reads); the block may have computed them for all its layers
      if reads is None or reads[0].dtype != pd:
        if read_state is None or read_state.dtype != pd:
          read_state = ctx.read_state(e, pd)
        reads = stacked_reads([R], cfg, e, read_state)[0]
      read, door, gate_logits = reads
      # ---- F + read: q'.k onto the read (baddbmm), the two relation terms added in place
      Fr = torch.baddbmm(read.reshape(batch * heads, 64, 64), qs, kb.transpose(1, 2)).view(batch, heads, 64, 64)
      Fr = Fr.add_(rel_q.permute(0, 2, 1, 3)).add_(rel_k.permute(0, 2, 3, 1))
      # ---- door -> softmax -> gate -> A.v -> output gate (softmax and gates float32; pair_autocast lever: pd)
      attended, H = attention_core(Fr, door, gate_logits, lanes[:, 2],
                                   cfg.gate_cap, lanes[:, 3] if self.g is not None else None, cfg.output_gate_scale,
                                   low=cfg.pair_autocast and low)
      if self.training and self.qk_clip_monitor:
        self._stash_max_logit(H)
      attended = attended.permute(0, 2, 1, 3).reshape(batch, 64, cfg.d_model)
    if self._unit_alphas:                    # the constant residual scales are 1 under pre-norm: no multiply
      x = x + self.out(attended).to(x.dtype)
      x = x + self.ffn(self.ln2(x)).to(x.dtype)
    else:
      x = x + (self.alpha_attention.to(x.dtype) * self.out(attended)).to(x.dtype)
      x = x + (self.alpha_ffn.to(x.dtype) * self.ffn(self.ln2(x))).to(x.dtype)
    return x, H

  def forward_reference(self, x, E, S, e, prologue: BT6Prologue):
    """The literal einsum form of the layer in float32 (BT6_SPEC A3 term by term): the reference `forward` is tested
    against. E float [B, 34, 64, 64] or a RelationContext."""
    E = E.E if isinstance(E, RelationContext) else E
    cfg = self.cfg
    R = self.readers
    batch, heads, hd = x.shape[0], cfg.heads, cfg.head_dim
    y = self.ln1(x)
    q = self.q(y).view(batch, 64, heads, hd)
    k = self.k(y).view(batch, 64, heads, hd)
    v = self.v(y).view(batch, 64, heads, hd)
    gate_pre = self.g(y) if self.g is not None else None
    # float32 island: every pairwise [B, heads, 64, 64] tensor (identical arithmetic when the module runs in float32)
    with _fp32(x):
      q, k, v, E, S, e = q.float(), k.float(), v.float(), E.float(), S.float(), e.float()
      static = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(hd)
      row = torch.einsum("bihd,hcd->bhic", q, self.Pq.float()) + R.C1.float()[None, :, None, :]
      column = torch.einsum("bjhd,hcd->bhjc", k, self.Pk.float())
      static = static + torch.einsum("bhic,bcij->bhij", row, E) + torch.einsum("bhjc,bcij->bhij", column, E)
      G, K = prologue.pair_read(R.C2)
      static = static + S[:, None] * (torch.einsum("hc,bcij->bhij", G, E) + K)
      logits = R.t_node.float().view(1, heads, 1, 1) * static
      read = torch.einsum("hc,bcij->bhij", R.We.float(), e) * R.t_edge.float().view(1, heads, 1, 1)
      H = (logits + read) * (1.0 + torch.einsum("hc,bcij->bhij", R.Wm.float(), e))
      gate = 2.0 * torch.sigmoid(torch.einsum("hc,bcij->bhij", R.Wg.float(), e) + R.bg.float().view(1, heads, 1, 1))
      if cfg.gate_cap:
        gate = gate.clamp(max=1.0)
      A = torch.softmax(H, dim=-1) * gate                                   # rows no longer sum to 1
      attended = torch.einsum("bhij,bjhd->bihd", A, v).reshape(batch, 64, cfg.d_model)
      if gate_pre is not None:
        attended = attended * (cfg.output_gate_scale * torch.sigmoid(gate_pre.float()))
    x = x + (self.alpha_attention.to(x.dtype) * self.out(attended)).to(x.dtype)
    x = x + (self.alpha_ffn.to(x.dtype) * self.ffn(self.ln2(x))).to(x.dtype)
    return x, H


class BT6EdgeSite(nn.Module):
  """Edge-state update site: readback of the layer's logits, triplet branch, edge FFN with the reverse edge.

  All channel maps are stored [out, in] (as nn.Linear and as the carrier serves them)."""

  def __init__(self, cfg: BT6Config) -> None:
    super().__init__()
    self.cfg = cfg
    states, hidden = cfg.edge_channels, cfg.site_hidden
    self.readback = nn.Parameter(torch.zeros(states, cfg.heads))                      # O, zero init
    if cfg.triplet:
      self.triplet_value = nn.Parameter(nn.init.xavier_normal_(torch.empty(2 * states, states)))
      # logits and gates of both directions: e_in, g_in, e_out, g_out, `triplet_heads` channels each (BT6: 16)
      self.triplet_gate = nn.Parameter(nn.init.xavier_normal_(torch.empty(4 * cfg.triplet_heads, states)))
      self.triplet_gate_bias = nn.Parameter(torch.zeros(4 * cfg.triplet_heads))
      self.triplet_out = nn.Parameter(torch.zeros(states, 2 * states))                # Wo, zero init
    self.ffn_in = nn.Parameter(nn.init.xavier_normal_(torch.empty(hidden, states)))   # W1
    self.ffn_in_bias = nn.Parameter(torch.zeros(hidden))
    if cfg.rev_edge:
      self.ffn_rev = nn.Parameter(torch.zeros(hidden, states))                        # Wr, zero init
    self.ffn_out = nn.Parameter(torch.zeros(states, hidden))                          # W2, zero init

  def forward(self, e: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    cfg = self.cfg
    with _fp32(e):                                                           # the whole site is float32
      e, H = e.float(), H.float()
      e_hat = e + torch.einsum("ch,bhij->bcij", self.readback.float(), H)
      if cfg.triplet:
        e_hat = e_hat + _triplet(e_hat, self.triplet_value.float(), self.triplet_gate.float(),
                                 self.triplet_gate_bias.float(), self.triplet_out.float(),
                                 cfg.triplet_heads, cfg.triplet_dots, cfg.triplet_contraction, cfg.rms_eps)
      n = rms(e_hat, 1, cfg.rms_eps)
      hidden = torch.einsum("oc,bcij->boij", self.ffn_in.float(), n) + self.ffn_in_bias.float()[None, :, None, None]
      if cfg.rev_edge:
        hidden = hidden + torch.einsum("oc,bcij->boij", self.ffn_rev.float(), n.transpose(-1, -2))
      return rms(e_hat + torch.einsum("oc,bcij->boij", self.ffn_out.float(), torch.relu(hidden)), 1, cfg.rms_eps)


class BT6Embedding(nn.Module):
  def __init__(self, cfg: BT6Config) -> None:
    super().__init__()
    self.cfg = cfg
    width, dense = cfg.d_model, cfg.embedding_dense
    self.preprocess = nn.Linear(12 * 64, 64 * dense)        # whole-board dense: 768 -> 64 x dense
    self.project = nn.Linear(cfg.input_planes + dense, width)
    self.ln0 = nn.LayerNorm(width, eps=cfg.norm_eps)
    self.mult_gate = nn.Parameter(torch.ones(64, width))    # relu applied in the forward (and in the export)
    self.position = nn.Parameter(torch.zeros(64, width))
    self.ffn = BT6GLU(width, cfg.embedding_ffn_hidden)
    self.ln1 = nn.LayerNorm(width, eps=cfg.norm_eps)
    self.register_buffer("alpha", torch.tensor([cfg.embedding_alpha]))
    _linear_init(self.preprocess)
    _linear_init(self.project)

  def forward(self, planes: torch.Tensor) -> torch.Tensor:
    batch = planes.shape[0]
    per_square = planes.permute(0, 2, 1)                                          # [B, 64, 112]
    position = self.preprocess(per_square[:, :, :12].reshape(batch, 768)).view(batch, 64, -1)
    x = self.ln0(F.mish(self.project(torch.cat((per_square, position), dim=-1))))
    x = x * torch.relu(self.mult_gate) + self.position
    return self.ln1(x + self.alpha * self.ffn(x))                                 # post-norm here


class BT6PolicyHead(nn.Module):
  """Attention policy: L[i, j] = (Q_i . K_j + u . Z0[:, i, j]) / sqrt(width), promotion offsets, 1858 gather."""

  def __init__(self, cfg: BT6Config, trunk_width: int | None = None) -> None:
    super().__init__()
    self.cfg = cfg
    width = cfg.policy_width
    # trunk_width: the width of the state the head reads when it differs from the layers' d_model (NBT hybrid)
    self.embedding = nn.Linear(cfg.d_model if trunk_width is None else trunk_width, width)
    self.query = nn.Linear(width, width)
    self.key = nn.Linear(width, width)
    self.promotion = nn.Linear(width, 4, bias=False)
    self.mix = nn.Parameter(torch.zeros(cfg.pair_channels))                       # u: pair-stream read
    self.register_buffer("gather", policy_gather_index(), persistent=False)
    for linear in (self.embedding, self.query, self.key, self.promotion):
      _linear_init(linear)

  def pair_bias(self, E, S, prologue: BT6Prologue) -> torch.Tensor:
    """[B, 64, 64] from-to bias u . Z0 / sqrt(width) (also what a move-token decoder would gather per move)."""
    E = E.E if isinstance(E, RelationContext) else E
    with _fp32(E):
      G, K = prologue.pair_read(self.mix[None, :])
      return S.float() * (torch.einsum("c,bcij->bij", G[0], E.float()) + K[0]) / math.sqrt(self.cfg.policy_width)

  def embed(self, x: torch.Tensor) -> torch.Tensor:
    """The policy embedding mish(x W + b), [B, 64, width]: shared by auxiliary policy heads."""
    return F.mish(self.embedding(x))

  def forward(self, x, E, S, prologue: BT6Prologue, embedded: torch.Tensor | None = None) -> torch.Tensor:
    embedded = self.embed(x) if embedded is None else embedded
    queries, keys = self.query(embedded), self.key(embedded)
    return policy_logits(queries, keys, self.promotion.weight, self.pair_bias(E, S, prologue), self.gather)


class BT6DenseHead(nn.Module):
  """Per-square projection with mish, flatten, dense with mish, dense output (value / error / moves left)."""

  def __init__(self, trunk: int, square_width: int, hidden: int, outputs: int) -> None:
    super().__init__()
    self.embedding = nn.Linear(trunk, square_width)
    self.dense1 = nn.Linear(64 * square_width, hidden)
    self.dense2 = nn.Linear(hidden, outputs)
    for linear in (self.embedding, self.dense1, self.dense2):
      _linear_init(linear)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.dense2(F.mish(self.dense1(F.mish(self.embedding(x)).flatten(1))))


class BT6Net(nn.Module):
  """The full BT6 forward in trained-parameter form. `forward(planes)` returns a dict: policy [B, 1858] logits,
  wdl [B, 3] logits, mlh [B, 1], err [B, 1] (if configured); with `return_state=True` also the final edge state,
  S, E and the final-normed trunk (what a hybrid's decoder would read)."""

  def __init__(self, config: BT6Config | None = None) -> None:
    super().__init__()
    cfg = self.config = config if config is not None else BT6Config()
    self.relations = BT6Relations()
    self.prologue = BT6Prologue(cfg)
    self.embedding = BT6Embedding(cfg)
    self.layers = nn.ModuleList(BT6Layer(cfg) for _ in range(cfg.layers))
    self.sites = nn.ModuleDict({str(index): BT6EdgeSite(cfg) for index in cfg.edge_sites})
    self.final_norm = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
    self.policy = BT6PolicyHead(cfg)
    self.value = BT6DenseHead(cfg.d_model, cfg.value_width, cfg.value_hidden, 3)
    self.moves = BT6DenseHead(cfg.d_model, cfg.moves_width, cfg.moves_hidden, 1)
    self.value_error = BT6DenseHead(cfg.d_model, cfg.value_width, cfg.value_hidden, 1) if cfg.error_head else None

  def forward(self, planes: torch.Tensor, return_state: bool = False) -> dict:
    planes = planes.float()
    E, S, e = self.prologue.context(self.relations(planes[:, 0:12] > 0.5, planes[:, 13:25] > 0.5))   # E: no gradient
    x = self.embedding(planes)
    for index, layer in enumerate(self.layers):
      x, H = layer(x, E, S, e, self.prologue)
      if str(index) in self.sites:
        e = self.sites[str(index)](e, H)                                         # H is NOT detached
    x = self.final_norm(x)
    out = {"policy": self.policy(x, E, S, self.prologue), "wdl": self.value(x), "mlh": torch.relu(self.moves(x))}
    if self.value_error is not None:
      out["err"] = torch.sigmoid(self.value_error(x))
    if return_state:
      out.update(trunk=x, edge_state=e, S=S, E=E.E)
    return out

  # ------------------------------------------------------------------------------------------------ export
  @torch.no_grad()
  def export_served(self, parts=("embedding", "prologue", "layers", "heads", "policy")) -> dict:
    """The served tensors: {carrier name: numpy array} with the carrier's shapes and dtypes (FP16 dense weights,
    FP32 relation / edge tables), folded as lczero_triton/lab/carrier.py folds them (in float64):

      qk_scale = t_node / sqrt(hd);  attack, key, query = t_node * C1, Pk, Pq
      scaled_coefficients = t_node * (C2 D^T);  constant_bias = t_node * (C2 T^T)[relidx]
      edge_read/w = t_edge * We;  door/w = Wm;  gate/w = Wg;  gate/b = bg
      qkvg = [q | k | v | g] columns;  ffn/dense1 = [gate | up] columns, bias [0 | up bias]
      mult_gate = relu(gate);  offsets = T^T[:, relidx];  mix_codes = code-group sums of [D | Pin]
      /policy/pair/* = (u D^T, (u T^T)[relidx]) / sqrt(policy width)

    Not produced: the int8 / quantiser tensors and the D1 refit of the int8 artifact.

    `parts` selects sections (the NBT hybrid exports prologue + layers + policy through this method and adds
    its own tensors, bt6_hybrid.export_served_hybrid); the prologue tensors are always written."""
    cfg = self.config
    out: dict = {}

    def f64(tensor):
      return tensor.detach().double().cpu()

    def put(name, tensor, dtype=np.float16, shape=None):
      array = f64(tensor).numpy()
      if shape is not None:
        array = array.reshape(shape)
      with np.errstate(over="ignore"):
        converted = array.astype(dtype)
      if not np.isfinite(converted).all():
        raise ValueError(f"{name}: a value does not fit {np.dtype(dtype).name}")
      out[name] = np.ascontiguousarray(converted)

    def dense(prefix, linear):
      put(f"{prefix}/w", linear.weight.T)
      if linear.bias is not None:
        put(f"{prefix}/b", linear.bias)

    def norm(prefix, layer_norm):
      put(f"{prefix}/scale", layer_norm.weight)
      put(f"{prefix}/bias", layer_norm.bias)

    def glu(prefix, ffn, alpha):
      put(f"{prefix}/dense1/w", torch.cat((f64(ffn.gate.weight).T, f64(ffn.up.weight).T), dim=1))
      put(f"{prefix}/dense1/b", torch.cat((torch.zeros_like(f64(ffn.up.bias)), f64(ffn.up.bias))))
      dense(f"{prefix}/dense2", ffn.down)
      put(f"{prefix}/alpha/w", alpha)

    if "embedding" in parts:
      emb = self.embedding
      dense("/attn_body/preproc", emb.preprocess)
      dense("/attn_body/matmul", emb.project)
      norm("/attn_body/ln0", emb.ln0)
      put("/attn_body/mult_gate/w", torch.relu(emb.mult_gate))
      put("/attn_body/position/w", emb.position, shape=(1, 64, cfg.d_model))
      glu("/attn_body/ffn", emb.ffn, emb.alpha)
      norm("/attn_body/ln1", emb.ln1)

    pro = self.prologue
    D, T, Pin, Toff = f64(pro.D), f64(pro.T), f64(pro.Pin), f64(pro.Toff)
    relidx = pro.relidx.cpu()
    offsets = T.T[:, relidx]                                                 # [P, 64, 64]
    put("/prologue/channel_mix", D, np.float32)
    put("/prologue/offsets", offsets, np.float32)
    put("/prologue/edge_mix", Pin, np.float32)
    put("/prologue/edge_offsets", Toff.T[:, relidx], np.float32)
    mix = torch.cat((D, Pin), dim=1)
    rows = []
    for first, bits in CODE_GROUPS:
      codes = torch.arange(1 << bits)
      rows.append(((codes[:, None] >> torch.arange(bits)[None, :]) & 1).double() @ mix[first:first + bits])
    put("/prologue/mix_codes", torch.cat(rows), np.float32)
    put("/encoder/qk_scale", torch.tensor([1.0 / math.sqrt(cfg.head_dim)]))
    put("/policy/qk_scale", torch.tensor([1.0 / math.sqrt(cfg.policy_width)]))

    if "layers" in parts:
      for index, layer in enumerate(self.layers):
        p = f"/encoder{index}"
        projections = [layer.q, layer.k, layer.v] + ([layer.g] if layer.g is not None else [])
        name = "qkvg" if layer.g is not None else "qkv"
        put(f"{p}/mha/{name}/w", torch.cat([f64(linear.weight).T for linear in projections], dim=1))
        put(f"{p}/mha/{name}/b", torch.cat([f64(linear.bias) for linear in projections]))
        dense(f"{p}/mha/out", layer.out)
        put(f"{p}/mha/alpha/w", layer.alpha_attention)
        norm(f"{p}/ln1", layer.ln1)
        glu(f"{p}/ffn", layer.ffn, layer.alpha_ffn)
        norm(f"{p}/ln2", layer.ln2)
        # own tables; a block's shared readers written per layer; zero tables for a layer built without terms
        rt = layer.reader_tensors()
        t_node, t_edge, C2 = f64(rt["t_node"]), f64(rt["t_edge"]), f64(rt["C2"])
        g = f"{p}/mha/egt"
        put(f"{g}/qk_scale", t_node * (1.0 / math.sqrt(cfg.head_dim)), np.float32)
        put(f"{g}/attack", t_node[:, None] * f64(rt["C1"]), np.float32)
        put(f"{g}/key", t_node[:, None, None] * f64(rt["Pk"]), np.float32)
        put(f"{g}/query", t_node[:, None, None] * f64(rt["Pq"]), np.float32)
        put(f"{g}/scaled_coefficients", t_node[:, None] * (C2 @ D.T), np.float32)
        put(f"{g}/constant_bias", t_node[:, None, None] * torch.einsum("ht,trc->hrc", C2, offsets), np.float32)
        put(f"{g}/edge_read/w", t_edge[:, None] * f64(rt["We"]), np.float32)
        put(f"{g}/door/w", rt["Wm"], np.float32)
        put(f"{g}/gate/w", rt["Wg"], np.float32)
        put(f"{g}/gate/b", rt["bg"], np.float32)
        if str(index) in self.sites:
          site, s = self.sites[str(index)], f"{p}/edge_site"
          put(f"{s}/readback/w", site.readback, np.float32)
          if cfg.triplet:
            put(f"{s}/triplet/value/w", site.triplet_value, np.float32)
            put(f"{s}/triplet/gate/w", site.triplet_gate, np.float32)
            put(f"{s}/triplet/gate/b", site.triplet_gate_bias, np.float32)
            put(f"{s}/triplet/out/w", site.triplet_out, np.float32)
          put(f"{s}/ffn/dense1/w", site.ffn_in, np.float32)
          put(f"{s}/ffn/dense1/b", site.ffn_in_bias, np.float32)
          if cfg.rev_edge:
            put(f"{s}/ffn/dense1_rev/w", site.ffn_rev, np.float32)
          put(f"{s}/ffn/dense2/w", site.ffn_out, np.float32)

    if "heads" in parts:
      norm("/encoder/final_norm", self.final_norm)
      for prefix, head in (("/value", self.value), ("/moves", self.moves), ("/value_error", self.value_error)):
        if head is not None:
          dense(f"{prefix}/embedding", head.embedding)
          dense(f"{prefix}/dense1", head.dense1)
          dense(f"{prefix}/dense2", head.dense2)
    if "policy" in parts:
      pol = self.policy
      dense("/policy/embedding", pol.embedding)
      dense("/policy/Q", pol.query)
      dense("/policy/K", pol.key)
      put("/policy/promotion/w", pol.promotion.weight.T)
      u = f64(pol.mix)
      divisor = math.sqrt(cfg.policy_width)
      put("/policy/mix/coefficients", u)
      put("/policy/pair/scaled_coefficients", (u[None, :] @ D.T) / divisor)
      put("/policy/pair/constant_bias", torch.einsum("t,trc->rc", u, offsets)[None] / divisor)
      out["/policy/map/w"] = pol.gather.cpu().numpy().astype(np.int32)
    return out

  # ------------------------------------------------------------------------------------------------ import
  @torch.no_grad()
  def load_served(self, tensors: dict) -> dict:
    """Initialise the module from served tensors ({carrier name: numpy array}), inverting the folds in float64.

    Unique inversions: every dense / norm tensor (a transpose or a column split), t_node = qk_scale * sqrt(hd),
    C1 / Pk / Pq = table / t_node, Wm, Wg, bg, the site maps, D and Pin (from /prologue/channel_mix and edge_mix
    or, if the carrier lacks them, from the single-bit rows of mix_codes), T and Toff (one value per relative
    index bin; the served [P, 64, 64] tables must be constant per bin).

    NOT unique -- a consistent solution is chosen:
      * t_edge and We appear only as the product t_edge * We: t_edge := 1, We := edge_read/w.
      * a head with t_node == 0 has no recoverable C1 / Pk / Pq / C2: they are set to 0 (the served tables are 0).
      * mult_gate is served after relu: the parameter is set to the served (non-negative) value; where the served
        gate is exactly 0 the trained value may have been negative.
      * C2 (per layer) and u (policy) are served only through the folds c D^T [34] and (c T^T)[relidx] [225 bins]:
        solved by least squares over both (259 equations, 16 unknowns; exact when D, T have rank 16).

    Returns a report {check name: max abs deviation} of every consistency check made on the way."""
    cfg = self.config
    report: dict = {}

    def get(name):
      return torch.from_numpy(np.ascontiguousarray(tensors[name]).astype(np.float64))

    def set_(parameter, value):
      parameter.copy_(value.reshape(parameter.shape).to(parameter.dtype))

    def dense(prefix, linear):
      set_(linear.weight, get(f"{prefix}/w").T)
      if linear.bias is not None:
        set_(linear.bias, get(f"{prefix}/b"))

    def norm(prefix, layer_norm):
      set_(layer_norm.weight, get(f"{prefix}/scale"))
      set_(layer_norm.bias, get(f"{prefix}/bias"))

    def note(name, value):
      report[name] = max(report.get(name, 0.0), float(value))

    def glu(prefix, ffn, alpha):
      packed, bias = get(f"{prefix}/dense1/w"), get(f"{prefix}/dense1/b")
      hidden = packed.shape[1] // 2
      set_(ffn.gate.weight, packed[:, :hidden].T)
      set_(ffn.up.weight, packed[:, hidden:].T)
      set_(ffn.up.bias, bias[hidden:])
      note("glu gate-bias half (must be 0)", bias[:hidden].abs().max())
      dense(f"{prefix}/dense2", ffn.down)
      set_(alpha, get(f"{prefix}/alpha/w"))

    emb = self.embedding
    dense("/attn_body/preproc", emb.preprocess)
    dense("/attn_body/matmul", emb.project)
    norm("/attn_body/ln0", emb.ln0)
    gate = get("/attn_body/mult_gate/w")
    note("mult_gate negative part (must be 0)", (-gate).clamp(min=0).max())
    set_(emb.mult_gate, gate)
    set_(emb.position, get("/attn_body/position/w"))
    glu("/attn_body/ffn", emb.ffn, emb.alpha)
    note("embedding alpha - (2 layers)^-0.25", abs(float(emb.alpha) - cfg.embedding_alpha))
    norm("/attn_body/ln1", emb.ln1)

    # ---- prologue
    pro = self.prologue
    relidx = pro.relidx.cpu()
    flat = relidx.reshape(-1)
    representative = torch.zeros(RELATIVE_BINS, dtype=torch.long)
    representative[flat] = torch.arange(flat.numel())                         # one (i, j) cell per bin

    def bins(name):
      table = get(name)                                                        # [P, 64, 64]
      per_bin = table.reshape(table.shape[0], -1)[:, representative].T         # [225, P]
      note(f"{name} not a function of the relative index", (per_bin.T[:, relidx] - table).abs().max())
      return per_bin

    T, Toff = bins("/prologue/offsets"), bins("/prologue/edge_offsets")
    codes = get("/prologue/mix_codes")
    single = torch.cat([codes[base + (1 << torch.arange(bits))] for (_, bits), base in zip(CODE_GROUPS, CODE_GROUP_BASES)])
    if "/prologue/channel_mix" in tensors and "/prologue/edge_mix" in tensors:
      D, Pin = get("/prologue/channel_mix"), get("/prologue/edge_mix")
      note("mix_codes single-bit rows vs [channel_mix | edge_mix]", (single - torch.cat((D, Pin), dim=1)).abs().max())
    else:
      D, Pin = single[:, :cfg.pair_channels], single[:, cfg.pair_channels:]
    set_(pro.D, D)
    set_(pro.T, T)
    set_(pro.Pin, Pin)
    set_(pro.Toff, Toff)
    system = torch.cat((D, T), dim=0)                                          # [34 + 225, P]: c -> (c D^T | c T^T)

    def solve_reader(scaled, constant, label):
      """Rows c of [n, P] with c D^T = scaled [n, 34] and (c T^T)[relidx] = constant [n, 64, 64]."""
      per_bin = constant.reshape(constant.shape[0], -1)[:, representative]     # [n, 225]
      note(f"{label} constant_bias not a function of the relative index",
           (per_bin[:, relidx] - constant).abs().max())
      target = torch.cat((scaled, per_bin), dim=1)                             # [n, 259]
      solution = torch.linalg.lstsq(system, target.T).solution.T               # [n, P]
      note(f"{label} pair-fold least-squares residual", (solution @ system.T - target).abs().max())
      return solution

    # ---- tower
    hd_root = math.sqrt(cfg.head_dim)
    for index, layer in enumerate(self.layers):
      p = f"/encoder{index}"
      name = "qkvg" if layer.g is not None else "qkv"
      projections = [layer.q, layer.k, layer.v] + ([layer.g] if layer.g is not None else [])
      weights = get(f"{p}/mha/{name}/w").chunk(len(projections), dim=1)
      biases = get(f"{p}/mha/{name}/b").chunk(len(projections))
      for linear, weight, bias in zip(projections, weights, biases):
        set_(linear.weight, weight.T)
        set_(linear.bias, bias)
      dense(f"{p}/mha/out", layer.out)
      set_(layer.alpha_attention, get(f"{p}/mha/alpha/w"))
      norm(f"{p}/ln1", layer.ln1)
      glu(f"{p}/ffn", layer.ffn, layer.alpha_ffn)
      layer._unit_alphas = bool((layer.alpha_attention == 1).all() and (layer.alpha_ffn == 1).all())
      norm(f"{p}/ln2", layer.ln2)
      g = f"{p}/mha/egt"
      t_node = get(f"{g}/qk_scale") * hd_root
      alive = t_node != 0
      inverse = torch.where(alive, 1.0 / torch.where(alive, t_node, torch.ones_like(t_node)), torch.zeros_like(t_node))
      note("heads with t_node == 0", float((~alive).sum()))
      set_(layer.t_node, t_node)
      set_(layer.C1, get(f"{g}/attack") * inverse[:, None])
      set_(layer.Pk, get(f"{g}/key") * inverse[:, None, None])
      set_(layer.Pq, get(f"{g}/query") * inverse[:, None, None])
      set_(layer.C2, solve_reader(get(f"{g}/scaled_coefficients") * inverse[:, None],
                                  get(f"{g}/constant_bias") * inverse[:, None, None], "encoder"))
      layer.t_edge.fill_(1.0)
      set_(layer.We, get(f"{g}/edge_read/w"))
      set_(layer.Wm, get(f"{g}/door/w"))
      set_(layer.Wg, get(f"{g}/gate/w"))
      set_(layer.bg, get(f"{g}/gate/b"))
      if str(index) in self.sites:
        site, s = self.sites[str(index)], f"{p}/edge_site"
        set_(site.readback, get(f"{s}/readback/w"))
        if cfg.triplet:
          set_(site.triplet_value, get(f"{s}/triplet/value/w"))
          set_(site.triplet_gate, get(f"{s}/triplet/gate/w"))
          set_(site.triplet_gate_bias, get(f"{s}/triplet/gate/b"))
          set_(site.triplet_out, get(f"{s}/triplet/out/w"))
        set_(site.ffn_in, get(f"{s}/ffn/dense1/w"))
        set_(site.ffn_in_bias, get(f"{s}/ffn/dense1/b"))
        if cfg.rev_edge:
          set_(site.ffn_rev, get(f"{s}/ffn/dense1_rev/w"))
        set_(site.ffn_out, get(f"{s}/ffn/dense2/w"))

    # ---- heads
    norm("/encoder/final_norm", self.final_norm)
    for prefix, head in (("/value", self.value), ("/moves", self.moves), ("/value_error", self.value_error)):
      if head is not None:
        dense(f"{prefix}/embedding", head.embedding)
        dense(f"{prefix}/dense1", head.dense1)
        dense(f"{prefix}/dense2", head.dense2)
    pol = self.policy
    dense("/policy/embedding", pol.embedding)
    dense("/policy/Q", pol.query)
    dense("/policy/K", pol.key)
    set_(pol.promotion.weight, get("/policy/promotion/w").T)
    divisor = math.sqrt(cfg.policy_width)
    note("policy qk_scale - 1/sqrt(policy width)", abs(float(get("/policy/qk_scale")) - 1.0 / divisor))
    # u is served twice: as the (unread) FP16 vector /policy/mix/coefficients and through the two FP16 folds, which
    # the builder computed from the FLOAT32 u. Neither determines that float32 u exactly. The carried FP16 u is used
    # when present: it makes export -> load -> export a fixed point and reproduces /policy/mix/coefficients bit for
    # bit; its refold then differs from the served folds by at most one FP16 unit in some elements (reported).
    # (Measured on BT6: the least-squares u reproduces more fold elements -- 319 of 4,130 differ instead of 1,513 --
    # but changes 2 of the 16 carried values and is not a fixed point.)
    served = torch.cat((get("/policy/pair/scaled_coefficients").reshape(1, -1),
                        get("/policy/pair/constant_bias").reshape(1, -1)[:, representative]), dim=1)
    if "/policy/mix/coefficients" in tensors:
      u = get("/policy/mix/coefficients")
    else:
      u = solve_reader(get("/policy/pair/scaled_coefficients").reshape(1, -1) * divisor,
                       get("/policy/pair/constant_bias").reshape(1, 64, 64) * divisor, "policy")[0]
    refold = torch.cat(((u[None] @ D.T) / divisor, (u[None] @ T.T) / divisor), dim=1)
    note("policy pair folds vs refold of u (FP16 tensors)", (refold - served).abs().max())
    set_(pol.mix, u)
    if "/policy/map/w" in tensors:
      note("policy map differs from lc0's 1858 gather (entries)",
           float((torch.from_numpy(np.asarray(tensors["/policy/map/w"]).astype(np.int64)) != pol.gather.cpu()).sum()))
    return report


def count_parameters(module: nn.Module) -> int:
  return sum(parameter.numel() for parameter in module.parameters())
