"""NBT-BT6 hybrid: the BT6 parts (bt6_net.py, verified against the real BT6 program) wired for the Ceres trainer.

Net definition key `BT6Hybrid` (config.py). The trunk is our nested-bottleneck (NBT) trunk -- outer stream ModelDim,
blocks of NBTInnerLayers inner layers at ModelDim / NBTWidthDivisor -- where every inner layer is a `bt6_net.BT6Layer`
(LayerNorm eps 1e-3, QKVG with the 2 * sigmoid output gate, sigmoid GLU, its own t_node, t_edge, C1, Pk, Pq, C2, We,
Wm, Wg, bg). This module owns everything of BT6 that lives OUTSIDE the inner layers:

  relations   BT6Relations on OUR `squares` record: current pieces = bytes 1..12, previous position = bytes 14..25
  prologue    BT6Prologue: D, T[225], Pin, Toff[225] -> pair normaliser S and edge-state seed e0
  sites       BT6EdgeSite after the last inner layer of the blocks listed in BT6EdgeSites
  policy      BT6PolicyHead on the trunk-end state (embedding ModelDim -> width, Q, K, pair bias, promotions, 1858 map)
  soft_policy optional training-only BT6AuxPolicyHead sharing the policy embedding (own Q / K at its own width)
  embed       optional BT6RecordEmbedding: BT6's embedding form on our bytes

The inner layers themselves are created by `make_layer()` and live in ceres_net's `transformer_layer` (so the
trainer's name-based parameter rules see them as trunk parameters); `nbt_layer.NestedBottleneckLayer.forward_bt6`
runs them. Nothing here re-implements BT6 arithmetic: layer, site, prologue and policy read-out are the bt6_net
classes; test_bt6_hybrid.py checks the equivalence with bt6_net.BT6Net numerically.

History in the record. Our record fills a MISSING history position with a copy of the nearest newer board (C# TPG
writer and v6_dataset.py alike), where lc0's planes stay zero. So a position without real history has
previous == current and its gained / lost channels are EMPTY (BT6 itself, trained on zero planes, sees gained = all
attacks at ply 0). `relations_from_squares` additionally maps a previous position with no piece at all (a record
written without history bytes) to the same "nothing changed" case instead of "everything gained". A position with a
copied history is therefore indistinguishable, for these channels, from one where the last move changed no attack.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bt6_net import (BT6Config, BT6EdgeSite, BT6GLU, BT6Layer, BT6PolicyHead, BT6Prologue, CODE_GROUPS,
                     RelationContext, _fp32, _linear_init, policy_gather_index, policy_logits)
from bt6_relations import BT6Relations

# Names (suffixes of parameter names) of the raw relation / pair / edge reader tables and scales inside a BT6 inner
# layer: never weight-decayed, never under Muon (wd_partition.py, train.py). 2-D: C1, C2, We, Wm, Wg; 3-D: Pk, Pq.
BT6_LAYER_READER_SUFFIXES = (".t_node", ".t_edge", ".C1", ".Pk", ".Pq", ".C2", ".We", ".Wm", ".Wg", ".bg")
# Attribute name of the hybrid container on CeresNet, i.e. the prefix of its parameter names.
BT6_HYBRID_PREFIX = "bt6h."


def is_bt6_layer_reader(name: str) -> bool:
  """True for the reader tables / scales of a BT6 inner layer (name as in named_parameters, any wrapper prefix)."""
  return (".inner." in name or ".bt6_readers." in name) and name.endswith(BT6_LAYER_READER_SUFFIXES)


def is_bt6_output_gate(name: str) -> bool:
  """True for the zero-initialised attention output-gate projection of a BT6 inner layer (g.weight / g.bias)."""
  return ".inner." in name and (name.endswith(".g.weight") or name.endswith(".g.bias"))


def is_bt6_hybrid_param(name: str) -> bool:
  """True for every parameter of the hybrid container (prologue, sites, policy heads, record embedding)."""
  return name.replace("_orig_mod.", "").replace("module.", "").startswith(BT6_HYBRID_PREFIX)


def bt6_muon_excluded(name: str) -> bool:
  """Parameters of a hybrid net that never go under Muon, whatever the scope (train.py): the layer reader tables,
  the zero-init output-gate projection, and the whole bt6h container."""
  return is_bt6_layer_reader(name) or is_bt6_output_gate(name) or is_bt6_hybrid_param(name)


def bt6_is_ffn_kernel(name: str) -> bool:
  """The GLU kernels of a BT6 inner layer (ffn.gate / ffn.up / ffn.down weights): what the 'ffn-only' Muon scope
  (Kovax's "ffnmuon" partition) selects in a hybrid net."""
  return ".inner." in name and name.endswith((".ffn.gate.weight", ".ffn.up.weight", ".ffn.down.weight"))


def parameter_group_table(model, scope: str = "all-non-trunk"):
  """The optimiser groups of a HYBRID net under train.py's Muon scopes 'all-non-trunk' and 'ffn-only':
  rows (group, wd class, family, tensors, parameters). Mirrors train.py's predicates for the names that occur in a
  hybrid net (2-D trunk matrices inside transformer_layer, minus the BT6 exclusions); test_bt6_hybrid.py checks the
  Muon / AdamW tensor counts against what train.py itself prints in the smoke run."""
  from wd_partition import partition_weight_decay
  decay, _ = partition_weight_decay(model)

  def muon(name, p):
    base = (p.ndim == 2 and "embedding" not in name and "transformer_layer" in name and not bt6_muon_excluded(name))
    return base and (scope == "all-non-trunk" or bt6_is_ffn_kernel(name))

  def family(name):
    if is_bt6_layer_reader(name):
      return "inner layer: relation / pair / edge reader tables and scales"
    if is_bt6_output_gate(name):
      return "inner layer: output-gate projection g"
    if bt6_is_ffn_kernel(name):
      return "inner layer: GLU kernels (gate, up, down)"
    if ".shared_smolgen." in name or name.startswith("smolgenPrepLayer."):
      return "shared smolgen: per-block generators (sm1, sm2, sm3, norms) and the shared prep layer"
    if ".inner." in name:
      if name.endswith(("q.weight", "k.weight", "v.weight", "out.weight")):
        return "inner layer: attention kernels (q, k, v, out)"
      return "inner layer: biases and LayerNorm gains"
    if "transformer_layer" in name:
      return "NBT block: down / up projections" if name.endswith((".down.weight", ".up.weight")) else "NBT block: outer norms"
    if name.startswith("bt6h.prologue"):
      return "prologue tables (D, T, Pin, Toff)"
    if name.startswith("bt6h.sites"):
      return "edge sites"
    if name.startswith("bt6h.policy"):
      return "BT6 policy head"
    if name.startswith("bt6h.soft_policy"):
      return "soft-policy head (training-only)"
    if name.startswith("bt6h.embed"):
      return "record embedding (BT6 form)"
    if name.startswith(("embedding_layer", "embedding_norm")):
      return "embedding (Linear + norm)"
    return "value heads and shared head front end"

  rows: dict = {}
  for name, p in model.named_parameters():
    if not p.requires_grad:
      continue
    key = ("Muon" if muon(name, p) else "AdamW", "decay" if name in decay else "no_decay", family(name))
    entry = rows.setdefault(key, [0, 0])
    entry[0] += 1
    entry[1] += p.numel()
  return [(group, wd, fam, n, count) for (group, wd, fam), (n, count) in sorted(rows.items())]


def pieces_from_squares(squares: torch.Tensor):
  """Our record [B, 64, >= 26] -> (pieces, previous), bool [B, 12, 64] each, in BT6Relations' plane order
  (0-5 our P N B R Q K, 6-11 theirs). Bytes h * 13 + 0 are the EMPTY flag and are dropped. A previous position
  without any piece (no history bytes) is replaced by the current one: gained / lost empty."""
  pieces = squares[:, :, 1:13].transpose(1, 2) > 0.5
  previous = squares[:, :, 14:26].transpose(1, 2) > 0.5
  # ONNX-friendly form of "previous has no piece": the maximum over the 12 x 64 bytes
  missing = squares[:, :, 14:26].reshape(squares.shape[0], -1).amax(dim=1) <= 0.5
  # previous is all-False where `missing`, so OR-ing the current pieces in IS the replacement (no Where on bool:
  # onnxruntime has no boolean Where kernel)
  previous = previous | (missing[:, None, None] & pieces)
  return pieces, previous


class BT6RecordEmbedding(nn.Module):
  """BT6's embedding form on our record (bt6_net.BT6Embedding with the record in place of lc0's 112 planes):

      pos = flatten(12 current piece bytes over the 64 squares) W_pre + b      -> [64, dense]   (one dense per POSITION)
      x   = LN0(mish([record | pos] W + b)) * relu(gate[64, D]) + position[64, D]
      x   = LN1(x + a * GLU(x)),   a = (2 * inner layers of the trunk) ** -0.25   (a constant)

  `features` is the per-square record width the trainer feeds (137, +4 aux, + prior state)."""

  def __init__(self, features: int, width: int, dense: int, ffn_hidden: int, total_inner_layers: int,
               eps: float = 1e-3) -> None:
    super().__init__()
    self.dense = dense
    self.preprocess = nn.Linear(12 * 64, 64 * dense)
    self.project = nn.Linear(features + dense, width)
    self.ln0 = nn.LayerNorm(width, eps=eps)
    self.mult_gate = nn.Parameter(torch.ones(64, width))
    self.position = nn.Parameter(torch.zeros(64, width))
    self.ffn = BT6GLU(width, ffn_hidden)
    self.ln1 = nn.LayerNorm(width, eps=eps)
    self.register_buffer("alpha", torch.tensor([(2.0 * total_inner_layers) ** -0.25]))
    _linear_init(self.preprocess)
    _linear_init(self.project)

  def forward(self, record: torch.Tensor) -> torch.Tensor:
    """record [B, 64, features] (the trainer's `flow_squares`) -> [B, 64, width]."""
    batch = record.shape[0]
    position = self.preprocess(record[:, :, 1:13].reshape(batch, 768)).view(batch, 64, self.dense)
    x = self.ln0(F.mish(self.project(torch.cat((record, position.to(record.dtype)), dim=-1))))
    x = x * torch.relu(self.mult_gate).to(x.dtype) + self.position.to(x.dtype)
    return self.ln1(x + self.alpha.to(x.dtype) * self.ffn(x))


class BT6AuxPolicyHead(nn.Module):
  """A second attention policy read-out on the SHARED policy embedding (Kovax's arrangement for his auxiliary policy
  heads): own Q, K (embedding width -> width), promotion matrix and pair-stream read u. Training-only."""

  def __init__(self, embedding_width: int, width: int, pair_channels: int) -> None:
    super().__init__()
    self.width = width
    self.query = nn.Linear(embedding_width, width)
    self.key = nn.Linear(embedding_width, width)
    self.promotion = nn.Linear(width, 4, bias=False)
    self.mix = nn.Parameter(torch.zeros(pair_channels))
    self.register_buffer("gather", policy_gather_index(), persistent=False)
    for linear in (self.query, self.key, self.promotion):
      _linear_init(linear)

  def forward(self, embedded, E, S, prologue: BT6Prologue) -> torch.Tensor:
    E = E.E if isinstance(E, RelationContext) else E
    with _fp32(E):
      G, K = prologue.pair_read(self.mix[None, :])
      bias = S.float() * (torch.einsum("c,bcij->bij", G[0], E.float()) + K[0]) / math.sqrt(self.width)
    return policy_logits(self.query(embedded), self.key(embedded), self.promotion.weight, bias, self.gather)


class BT6Hybrid(nn.Module):
  """Everything of BT6 outside the inner layers, for one CeresNet (attribute `bt6h`)."""

  def __init__(self, model_dim: int, mid_dim: int, heads: int, num_blocks: int, inner_layers: int, sites,
               policy_width: int = 1024, soft_policy_width: int = 0, pair_channels: int = 16, edge_channels: int = 16,
               site_hidden: int = 64, triplet_heads: int = 4, rev_edge: bool = True, gate_cap: bool = False,
               output_gate_scale: float = 2.0, inner_norm_eps: float = 1e-3, assembly_autocast: bool = True,
               embedding: str = "linear",
               embedding_features: int = 137, embedding_dense: int = 512, embedding_ffn_hidden: int = 0,
               terms_inner_layers=None, share_readers: bool = False, stack_reads: bool = False,
               pair_autocast: bool = False) -> None:
    super().__init__()
    # Config levers (all off = BT6 inner layers): see config.py BT6TermsInnerLayers / BT6ShareReadersPerBlock /
    # BT6StackReadsPerBlock / BT6PairDtype.
    self.terms_inner_layers = (tuple(range(inner_layers)) if terms_inner_layers is None
                               else tuple(sorted(int(j) for j in terms_inner_layers)))
    self.share_readers, self.stack_reads = bool(share_readers), bool(stack_reads)
    self.model_dim, self.mid_dim, self.num_blocks, self.inner_layers = model_dim, mid_dim, num_blocks, inner_layers
    self.site_blocks = tuple(sorted(int(s) for s in sites))
    assert len(set(self.site_blocks)) == len(self.site_blocks), f"BT6EdgeSites {sites}: duplicate block"
    # a site after the LAST block would have no reader (the policy head reads only the static pair stream)
    assert all(0 <= s < num_blocks - 1 for s in self.site_blocks), \
        f"BT6EdgeSites {list(self.site_blocks)} must lie in 0..{num_blocks - 2}: a site after the last block is never read"
    assert edge_channels % triplet_heads == 0, "BT6EdgeDim must be divisible by BT6TripletHeads"
    # The layer config: a BT6Config whose d_model is the INNER width. `layers` is only used for bookkeeping here.
    self.layer_config = BT6Config(
      d_model=mid_dim, layers=num_blocks * inner_layers, heads=heads, ffn_hidden=mid_dim, pair_channels=pair_channels,
      edge_channels=edge_channels, edge_sites=(), site_hidden=site_hidden, triplet_heads=triplet_heads,
      triplet_dots=edge_channels // triplet_heads, rev_edge=rev_edge, gate_cap=gate_cap,
      output_gate_scale=output_gate_scale, policy_width=policy_width, norm_eps=inner_norm_eps, error_head=False,
      assembly_autocast=assembly_autocast, pair_autocast=pair_autocast)
    cfg = self.layer_config
    self.relations = BT6Relations()
    self.prologue = BT6Prologue(cfg)
    self.sites = nn.ModuleDict({str(block): BT6EdgeSite(cfg) for block in self.site_blocks})
    self.policy = BT6PolicyHead(cfg, trunk_width=model_dim)
    self.soft_policy = BT6AuxPolicyHead(policy_width, soft_policy_width, pair_channels) if soft_policy_width > 0 else None
    self.embed = None
    if embedding == "bt6":
      self.embed = BT6RecordEmbedding(embedding_features, model_dim, embedding_dense, embedding_ffn_hidden or model_dim,
                                      num_blocks * inner_layers, eps=inner_norm_eps)
    else:
      assert embedding == "linear", f"BT6Embedding must be 'bt6' or 'linear', got {embedding!r}"

  # ------------------------------------------------------------------ building blocks for ceres_net / nbt_layer
  def make_layer(self, position: int = 0) -> BT6Layer:
    """The BT6 inner layer at inner position `position` of a block (to be owned by the NBT block)."""
    return BT6Layer(self.layer_config, terms=position in self.terms_inner_layers)

  def finish_block(self, block) -> None:
    """Called once per NBT block after its inner layers exist: the block-level reader arrangement.
    block.bt6_readers: the shared BT6Readers of the block (BT6ShareReadersPerBlock) or None;
    block.bt6_stack: the block computes the E / Z0 / edge-state reads of its layers in one matmul."""
    from bt6_net import BT6Readers
    block.bt6_readers = None
    term_layers = [layer for layer in block.inner if layer.terms]
    if self.share_readers and term_layers:
      block.bt6_readers = BT6Readers(self.layer_config)
      for layer in term_layers:
        layer.share_readers(block.bt6_readers)
    block.bt6_stack = bool(term_layers) and (self.share_readers or self.stack_reads)

  @staticmethod
  def add_shared_smolgen(net, hidden_dim: int, per_square_dim: int, head_dim: int, activation: str, norm_type: str,
                         eps: float) -> int:
    """Config BT6SharedSmolgen: one nbt_layer.SharedSmolgen per NBT block (block.shared_smolgen) reading the
    normalised block input, all sharing ONE prep layer net.smolgenPrepLayer = Linear(head_dim -> 64 * 64) whose
    weight and bias are ZERO: the bias is exactly 0 until the prep layer has been trained, so a net loaded from a
    checkpoint without smolgen computes bit-identical outputs. The modules are built with the global RNG state
    saved and restored: every other parameter of the net gets the same initial value with the switch on or off.
    Returns the number of parameters added."""
    from nbt_layer import SharedSmolgen
    blocks = list(net.transformer_layer)
    heads = blocks[0].inner[0].cfg.heads
    rng = torch.get_rng_state()
    net.smolgenPrepLayer = nn.Linear(head_dim, 64 * 64)
    nn.init.zeros_(net.smolgenPrepLayer.weight)
    nn.init.zeros_(net.smolgenPrepLayer.bias)
    for block in blocks:
      assert any(not layer.terms for layer in block.inner), "BT6SharedSmolgen needs a term-free inner layer in the block"
      block.shared_smolgen = SharedSmolgen(block.down.in_features, heads, per_square_dim, hidden_dim, head_dim,
                                           net.smolgenPrepLayer, activation, norm_type, eps, 64)
    torch.set_rng_state(rng)
    return (sum(p.numel() for p in net.smolgenPrepLayer.parameters())
            + sum(p.numel() for block in blocks for p in block.shared_smolgen.parameters()))

  def start(self, squares: torch.Tensor):
    """squares [B, 64, >= 26] -> (RelationContext (its .E: float32 [B, 34, 64, 64], no gradient), S [B, 64, 64],
    e0 [B, d_e, 64, 64])."""
    pieces, previous = pieces_from_squares(squares)
    return self.prologue.context(self.relations(pieces, previous))

  def is_site(self, block: int) -> bool:
    return block in self.site_blocks

  def site(self, block: int, e: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    return self.sites[str(block)](e, H)

  def policy_forward(self, flow, E, S, want_soft: bool):
    """Trunk-end state [B, 64, ModelDim] -> (policy logits [B, 1858] float32, soft-policy logits or None)."""
    embedded = self.policy.embed(flow)
    policy = self.policy(flow, E, S, self.prologue, embedded=embedded)
    soft = self.soft_policy(embedded, E, S, self.prologue) if (want_soft and self.soft_policy is not None) else None
    return policy, soft

  def parameter_counts(self) -> dict:
    count = lambda module: sum(p.numel() for p in module.parameters()) if module is not None else 0
    return {"prologue": count(self.prologue), "edge sites": count(self.sites), "policy head": count(self.policy),
            "soft-policy head (training-only)": count(self.soft_policy), "record embedding": count(self.embed)}


# ====================================================================================================================
# Serving export: the carrier for the kernel builder (name plan in HYBRID_SERVING_SPEC.txt)
# ====================================================================================================================
@torch.no_grad()
def export_served_hybrid(net) -> dict:
  """Served tensors of a BT6-hybrid CeresNet: {carrier name: numpy array}, folded exactly as bt6_net.export_served
  folds a BT6 net (float64 folds; FP16 dense / norm tensors, FP32 relation / edge tables).

  Names. Inner layer j of block b is BT6 layer i = b * K + j and uses BT6's names /encoder{i}/...; a site after
  block b is /encoder{b * K + K - 1}/edge_site/...; prologue and policy keep BT6's names. New names:
    /nbt{b}/norm_in/{scale[,bias]}, /nbt{b}/down/w [D, d], /nbt{b}/norm_out/{scale[,bias]}, /nbt{b}/up/w [d, D]
    /trunk/final_norm/{scale[,bias]}
    /embed/...  (form "linear": /embed/linear/{w,b}, /embed/norm/scale; form "bt6": BT6's /attn_body/* names with the
                 record width in place of 112)
    /head/premap/{w,b}, /head/shared/{w,b}, /head/{value,value2,unc,mlh}/{fc,out}/{w,b}
  Training-only parts (soft-policy head, the flat policy MLP, loss-only heads) are not exported."""
  from bt6_net import BT6Net  # the per-part folds are BT6Net.export_served's; a shell BT6Net carries the parts

  hybrid = net.bt6h
  cfg = hybrid.layer_config
  blocks = list(net.transformer_layer)
  K = len(blocks[0].inner)
  shell = BT6Net.__new__(BT6Net)
  nn.Module.__init__(shell)
  shell.config = BT6Config(**{**cfg.__dict__, "layers": len(blocks) * K, "edge_sites": tuple(b * K + K - 1 for b in hybrid.site_blocks)})
  shell.relations = hybrid.relations
  shell.prologue = hybrid.prologue
  shell.layers = nn.ModuleList([layer for block in blocks for layer in block.inner])
  shell.sites = nn.ModuleDict({str(b * K + K - 1): hybrid.sites[str(b)] for b in hybrid.site_blocks})
  shell.policy = hybrid.policy
  out = shell.export_served(parts=("prologue", "layers", "policy"))

  def put(name, tensor, dtype=np.float16):
    array = tensor.detach().double().cpu().numpy().astype(dtype)
    if not np.isfinite(array).all():
      raise ValueError(f"{name}: a value does not fit {np.dtype(dtype).name}")
    out[name] = np.ascontiguousarray(array)

  def dense(prefix, linear):
    put(f"{prefix}/w", linear.weight.T)
    if linear.bias is not None:
      put(f"{prefix}/b", linear.bias)

  def norm(prefix, module):
    scale = getattr(module, "weight", None)
    scale = module.scale if scale is None else scale
    put(f"{prefix}/scale", scale)
    if getattr(module, "bias", None) is not None:
      put(f"{prefix}/bias", module.bias)

  for b, block in enumerate(blocks):
    norm(f"/nbt{b}/norm_in", block.norm_in)
    dense(f"/nbt{b}/down", block.down)
    norm(f"/nbt{b}/norm_out", block.norm_out)
    dense(f"/nbt{b}/up", block.up)
    smolgen = getattr(block, "shared_smolgen", None)
    if smolgen is not None:          # BT6SharedSmolgen: the block's generator (bias for its term-free layers)
      dense(f"/nbt{b}/smolgen/compress", smolgen.sm1)
      dense(f"/nbt{b}/smolgen/dense1", smolgen.sm2)
      norm(f"/nbt{b}/smolgen/ln1", smolgen.ln1)
      dense(f"/nbt{b}/smolgen/dense2", smolgen.sm3)
      norm(f"/nbt{b}/smolgen/ln2", smolgen.ln2)
  if getattr(blocks[0], "shared_smolgen", None) is not None:
    dense("/smolgen/prep", net.smolgenPrepLayer)            # shared by all blocks: [head_dim, 4096]
  norm("/trunk/final_norm", net.trunk_end_norm)
  if hybrid.embed is not None:
    emb = hybrid.embed
    dense("/attn_body/preproc", emb.preprocess)
    dense("/attn_body/matmul", emb.project)
    norm("/attn_body/ln0", emb.ln0)
    put("/attn_body/mult_gate/w", torch.relu(emb.mult_gate))
    put("/attn_body/position/w", emb.position[None])
    put("/attn_body/ffn/dense1/w", torch.cat((emb.ffn.gate.weight.double().T, emb.ffn.up.weight.double().T), dim=1))
    put("/attn_body/ffn/dense1/b", torch.cat((torch.zeros_like(emb.ffn.up.bias), emb.ffn.up.bias)))
    dense("/attn_body/ffn/dense2", emb.ffn.down)
    put("/attn_body/ffn/alpha/w", emb.alpha)
    norm("/attn_body/ln1", emb.ln1)
  else:
    dense("/embed/linear", net.embedding_layer)
    norm("/embed/norm", net.embedding_norm)
  dense("/head/premap", net.headPremap)
  dense("/head/shared", net.headSharedLinear)
  for name in ("value", "value2", "unc", "mlh"):
    head = getattr(net, f"{name}_head", None)
    if head is not None:
      dense(f"/head/{name}/fc", head.fc)
      dense(f"/head/{name}/out", head.fcFinal)
  return out





def served_meta(net) -> dict:
  """What the serving forward needs besides the tensors (the builder reads the same facts from the net's config)."""
  hybrid = net.bt6h
  blocks = list(net.transformer_layer)
  K = len(blocks[0].inner)
  cfg = BT6Config(**{**hybrid.layer_config.__dict__, "layers": len(blocks) * K,
                     "edge_sites": tuple(b * K + K - 1 for b in hybrid.site_blocks)})
  config = net.config
  return {
    "layer_config": cfg, "blocks": len(blocks), "inner_layers": K, "site_blocks": hybrid.site_blocks,
    "outer_norm": config.NetDef_NormType, "outer_norm_eps": 1e-6,
    "proj_activation": config.NetDef_NBTProjActivation, "proj_shift": bool(config.NetDef_NBTProjActivationShift),
    "embedding": "bt6" if hybrid.embed is not None else "linear", "embedding_eps": cfg.norm_eps,
    "heads_activation": config.NetDef_HeadsActivationType, "head_front_act": bool(net.head_front_act),
    "premap_per_square": net.HEAD_PREMAP_PER_SQUARE, "value2": net.value2_loss_weight > 0,
    "mlh": net.moves_left_loss_weight > 0,
    "terms_inner_layers": tuple(hybrid.terms_inner_layers),
    "smolgen": getattr(blocks[0], "shared_smolgen", None) is not None,
    "smolgen_activation": getattr(config, "NetDef_BT6SmolgenActivation", "Swish"),
  }


@torch.no_grad()
def forward_served_hybrid(t: dict, squares: torch.Tensor, meta: dict) -> dict:
  """The hybrid's forward computed from the SERVED tensors (export_served_hybrid), in kernel form: the executable
  version of HYBRID_SERVING_SPEC.txt. The BT6 parts are bt6_net's served_* functions, i.e. the code that was gated
  against the real BT6 program. `t`: {name: float32 tensor} (bt6_net.served_to_torch); `squares`: the record
  [B, 64, F] as the net receives it (already byte / 100). Returns policy [B, 1858], value [B, 3] (and value2, unc,
  mlh when built)."""
  from activation_functions import to_activation
  from bt6_net import _dense, _served_glu, served_layer, served_policy, served_prologue, served_site

  cfg = meta["layer_config"]
  batch = squares.shape[0]
  squares = squares.float()

  def norm(x, prefix, kind, eps):
    if kind == "LayerNorm":
      return F.layer_norm(x, x.shape[-1:], t[f"{prefix}/scale"], t[f"{prefix}/bias"], eps)
    assert kind == "RMSNorm", kind
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * t[f"{prefix}/scale"]

  def project(x):
    y = to_activation(meta["proj_activation"])(x)
    return y - y.mean(dim=-1, keepdim=True) if meta["proj_shift"] else y

  # ---- relation tensor, pair normaliser, edge-state seed
  pieces, previous = pieces_from_squares(squares)
  E_bool = BT6Relations().to(squares.device)(pieces, previous)
  E = E_bool.float()
  S, e = served_prologue(t, E_bool, cfg)

  # ---- embedding; the two blunder bytes (119, 120) are zeroed in the embedding input and kept for value2
  blunders = squares[:, 0, 119:121]
  record = torch.cat((squares[:, :, :119], torch.zeros_like(squares[:, :, 119:121]), squares[:, :, 121:]), dim=2)
  if meta["embedding"] == "bt6":
    position = _dense(record[:, :, 1:13].reshape(batch, 768), t, "/attn_body/preproc").view(batch, 64, -1)
    x = _dense(torch.cat((record, position), dim=-1), t, "/attn_body/matmul")
    x = norm(F.mish(x), "/attn_body/ln0", "LayerNorm", meta["embedding_eps"])
    x = x * t["/attn_body/mult_gate/w"] + t["/attn_body/position/w"].view(64, -1)
    x = norm(x + t["/attn_body/ffn/alpha/w"] * _served_glu(x, t, "/attn_body/ffn"), "/attn_body/ln1", "LayerNorm",
             meta["embedding_eps"])
  else:
    x = norm(_dense(record, t, "/embed/linear"), "/embed/norm", meta["outer_norm"], meta["outer_norm_eps"])

  # ---- trunk: NBT blocks of BT6 layers, edge sites at block ends
  def nbt_proj(y, prefix):
    # the NBT projections are bias-free today; the export writes "/b" when one has a bias, and then it is applied
    y = y @ t[f"{prefix}/w"]
    return y + t[f"{prefix}/b"] if f"{prefix}/b" in t else y

  K = meta["inner_layers"]
  terms = meta.get("terms_inner_layers", tuple(range(K)))

  def smolgen_bias(n, b):
    # the block's shared smolgen on the normalised block input n [B, 64, D] -> [B, heads, 64, 64]
    act = to_activation(meta["smolgen_activation"])
    p = f"/nbt{b}/smolgen"
    s = _dense(n, t, f"{p}/compress").reshape(batch, -1)
    s = norm(act(_dense(s, t, f"{p}/dense1")), f"{p}/ln1", meta["outer_norm"], meta["outer_norm_eps"])
    s = norm(act(_dense(s, t, f"{p}/dense2")), f"{p}/ln2", meta["outer_norm"], meta["outer_norm_eps"])
    return _dense(s.reshape(batch, cfg.heads, -1), t, "/smolgen/prep").reshape(batch, cfg.heads, 64, 64)

  for b in range(meta["blocks"]):
    n_in = norm(x, f"/nbt{b}/norm_in", meta["outer_norm"], meta["outer_norm_eps"])
    bias = smolgen_bias(n_in, b) if meta.get("smolgen") else None
    h = nbt_proj(project(n_in), f"/nbt{b}/down")
    H = None
    for j in range(K):
      h, H = served_layer(t, b * K + j, h, E, S, e, cfg, bias=None if j in terms else bias)
    x = x + nbt_proj(project(norm(h, f"/nbt{b}/norm_out", meta["outer_norm"], meta["outer_norm_eps"])), f"/nbt{b}/up")
    if b in meta["site_blocks"]:
      e = served_site(t, b * K + K - 1, e, H, cfg)
  x = norm(x, "/trunk/final_norm", meta["outer_norm"], meta["outer_norm_eps"])

  # ---- heads
  out = {"policy": served_policy(t, x, E, S)}
  act = to_activation(meta["heads_activation"])
  front = _dense(x, t, "/head/premap")
  front = act(front) if meta["head_front_act"] else front
  front = _dense(front.reshape(batch, -1), t, "/head/shared")
  front = act(front) if meta["head_front_act"] else front

  def head(name, value):
    return _dense(act(_dense(value, t, f"/head/{name}/fc")), t, f"/head/{name}/out")

  out["value"] = head("value", front)
  if meta["value2"]:
    out["value2"] = head("value2", torch.cat((front, blunders), dim=-1))
  if "/head/unc/fc/w" in t:                       # exported only when the net has the head (as value2 / mlh)
    out["unc"] = head("unc", front)
  if meta["mlh"]:
    out["mlh"] = head("mlh", front)
  return out
