"""Serving export (kernel-builder carrier) of the two NBT-BT6 "terms" variants WITH the move-token decoder.

Sibling of `bt6_hybrid.export_served_hybrid` (which serves the original hybrid: BT6 layers, BT6 policy head, no
decoder, and refuses these two). Export code only: nothing here is on the training path.

  variant "bt"  (BT6Hybrid + UseMoveTokens):  the inner layers are bt6_net.BT6Layer (LayerNorm eps 1e-3, q / k / v / g,
                2 * sigmoid output gate, sigmoid GLU of hidden BT6FFNMultiplier x inner width); the block's shared
                smolgen (BT6SharedSmolgen) biases the TERM-FREE layers only.
  variant "nt"  (BT6Hybrid + BT6NativeLayers [+ UseMoveTokens]): the inner layers are our own EncoderLayer (RMSNorm,
                joint qkv, SwiGLU); the block's shared smolgen (NBTSharedSmolgen) biases ALL inner layers; the term
                positions carry BT6's term reads on their own Q / K (bt6_native_terms.BT6NativeTerms):
                    L = t_node (Q.K / sqrt(hd) + relation terms + pair read) + t_edge (edge read) + bias
                    H = L (1 + door);   A = softmax(H) * 2 sigmoid(gate);   no output gate.

    export_served_terms(net)                  -> {carrier name: numpy array}
    served_terms_meta(net)                    -> what the kernel-form forward needs besides the tensors
    tokens_from_squares(net, squares)         -> uint8 [B, M, 4] move tokens, the program's /input/move_tokens
    forward_served_terms(t, squares, meta, tokens) -> the forward computed from the SERVED tensors, in kernel form

NAME PLAN (dense weights [in, out] FP16; relation / edge tables FP32, as the hybrid's carrier):
  /prologue/*                                 as export_served_hybrid (BT6Net.export_served, prologue part)
  /embed/linear/{w,b}, /embed/norm/scale      linear embedding + outer norm
  /nbt{b}/norm_in/scale, /nbt{b}/down/w, /nbt{b}/norm_out/scale, /nbt{b}/up/w
  /nbt{b}/smolgen/{compress,dense1,dense2}/{w,b}, /nbt{b}/smolgen/{ln1,ln2}/scale, /smolgen/prep/{w,b}
  /encoder{i}/mha/egt/{qk_scale, attack, key, query, scaled_coefficients, constant_bias, edge_read/w, door/w, gate/w,
                       gate/b}                FP32, for EVERY inner layer i = b * K + j, with BT6's folds; a term-free
                                              layer has qk_scale = 1 / sqrt(hd) and zero tables
  /encoder{i}/edge_site/*                     FP32, i = b * K + K - 1 for the site blocks
  variant bt, per layer: BT6's names -- /encoder{i}/ln{1,2}/{scale,bias}, mha/qkvg/{w,b}, mha/out/{w,b}, mha/alpha/w,
                         ffn/dense1/{w [d, 2 hid] = [gate | up], b = [0 | up bias]}, ffn/dense2/{w [hid, d], b},
                         ffn/alpha/w
  variant nt, per layer: /encoder{i}/ln{1,2}/scale, mha/qkv/w [d, 3 d] = Q | K | V with the heads side by side inside
                         each (the trainer's rows are interleaved per head), NO bias and NOT scaled (the 1 / sqrt(hd)
                         is in qk_scale), mha/out/{w,b}, ffn/dense1/{w [d, 2 ffn] = [SiLU gate | up],
                         b = [gate bias | 0]}, ffn/dense2/{w [ffn, d], b}
  /trunk/final_norm/scale
  /head/premap/{w,b}, /head/shared/{w,b}, /head/{value,value2,unc}/{fc,out}/{w,b}    (value2/fc/w is [F + 2, hidden]:
                         the last two rows are the two blunder inputs)
  /const/one [1]                              1.0: the residual scale of the skip GEMMs
  no-triplet sites (BT6Triplet false)         /encoder{i}/edge_site/ has readback and ffn tensors only (no triplet/*)
  /mt/w_edge/w [d_e, dm] FP32                 only with MoveTokenEdgeInput: token input += rms(e_final[from, to]) @ w
  /mt/...                                     the move-token decoder, exactly the names, shapes and folds of the
                         kernel builder's Ceres front end (lczero_triton/ceres/_names.py plan_decoder_buffers,
                         carrier.py decoder_tensors_from_state_dict): w_in split into its from | to square blocks,
                         every block's square-norm gain folded into xkv / pm_kv and all of them concatenated into
                         /mt/kv/w, 1 / sqrt(head depth) folded into the query rows, mt_pol_bias re-indexed by
                         (from * 64 + to) * 4 + slot (FP32).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bt6_hybrid import pieces_from_squares, served_meta
from bt6_net import BT6Config, BT6Net, _dense, rms, served_layer, served_prologue, served_site
from bt6_relations import BT6Relations

MT = "/mt"
MT_FLOOR = -30.0
ONE = "/const/one"


def variant_of(net) -> str:
  """'nt' (our own layers with BT6 term reads) or 'bt' (BT6 layers)."""
  assert getattr(net, "bt6h", None) is not None, "not a BT6Hybrid net"
  return "nt" if getattr(net, "bt6_native", False) else "bt"


def qkv_permutation(heads: int, head_depth: int) -> np.ndarray:
  """Row order turning the trainer's per-head [Q | K | V] qkv rows into Q | K | V with the heads inside each."""
  width = heads * head_depth
  order = np.empty(3 * width, dtype=np.int64)
  for part in range(3):
    for head in range(heads):
      source = head * 3 * head_depth + part * head_depth
      target = part * width + head * head_depth
      order[target:target + head_depth] = np.arange(source, source + head_depth)
  return order


# ====================================================================================================================
# Export
# ====================================================================================================================
@torch.no_grad()
def export_served_terms(net) -> dict:
  """Served tensors of a BT6-hybrid CeresNet of variant "bt" or "nt" with the move-token decoder."""
  hybrid = net.bt6h
  variant = variant_of(net)
  if hybrid.policy is not None or getattr(net, "move_tokens", None) is None:
    raise NotImplementedError("export_served_terms: the net must have the move-token decoder and no BT6 policy head "
                              "(the original hybrid is served by bt6_hybrid.export_served_hybrid)")
  if hybrid.embed is not None:
    raise NotImplementedError("export_served_terms: only the linear embedding (BT6Embedding 'linear') is exported")
  if hybrid.share_readers:
    raise NotImplementedError("export_served_terms: BT6ShareReadersPerBlock is not exported")
  if getattr(net, "attn_templates", None) is not None or getattr(net, "input_skip_a", None) is not None:
    raise NotImplementedError("export_served_terms: AttnTemplates / NBTInputSkip have no lc0ex front end yet "
                              "(read these nets through the ONNX export)")
  cfg = hybrid.layer_config
  blocks = list(net.transformer_layer)
  K = len(blocks[0].inner)
  heads, hd = cfg.heads, cfg.head_dim

  shell = BT6Net.__new__(BT6Net)
  nn.Module.__init__(shell)
  shell.config = BT6Config(**{**cfg.__dict__, "layers": len(blocks) * K,
                              "edge_sites": tuple(b * K + K - 1 for b in hybrid.site_blocks)})
  shell.relations = hybrid.relations
  shell.prologue = hybrid.prologue
  if variant == "bt":
    # exactly export_served_hybrid's route: BT6Net.export_served folds the prologue, every layer and every site
    shell.layers = nn.ModuleList([layer for block in blocks for layer in block.inner])
    shell.sites = nn.ModuleDict({str(b * K + K - 1): hybrid.sites[str(b)] for b in hybrid.site_blocks})
    out = shell.export_served(parts=("prologue", "layers"))
  else:
    out = shell.export_served(parts=())      # the prologue tensors (always written)
  out.pop("/policy/qk_scale", None)          # BT6's policy head is not part of these nets

  def f64(tensor):
    return tensor.detach().double().cpu()

  def put(name, tensor, dtype=np.float16):
    array = (tensor if isinstance(tensor, np.ndarray) else f64(tensor).numpy()).astype(np.float64)
    with np.errstate(over="ignore"):
      converted = array.astype(dtype)
    if not np.isfinite(converted).all():
      raise ValueError(f"{name}: a value does not fit {np.dtype(dtype).name}")
    out[name] = np.ascontiguousarray(converted)

  def dense(prefix, linear, bias=True):
    put(f"{prefix}/w", linear.weight.T)
    if linear.bias is not None:
      assert bias, f"{prefix}: unexpected bias"
      put(f"{prefix}/b", linear.bias)
    else:
      assert not bias, f"{prefix}: the plan has a bias, the module has none"

  def scale(prefix, module):
    assert getattr(module, "bias", None) is None and hasattr(module, "scale"), f"{prefix}: an RMSNorm is expected"
    assert not getattr(module, "_scale_folded", False)
    put(f"{prefix}/scale", module.scale)

  put(ONE, torch.ones(1))
  dense("/embed/linear", net.embedding_layer)
  scale("/embed/norm", net.embedding_norm)
  for b, block in enumerate(blocks):
    assert not block.proj_activation_shift and block.down.bias is None and block.up.bias is None
    scale(f"/nbt{b}/norm_in", block.norm_in)
    dense(f"/nbt{b}/down", block.down, bias=False)
    scale(f"/nbt{b}/norm_out", block.norm_out)
    dense(f"/nbt{b}/up", block.up, bias=False)
    smolgen = getattr(block, "shared_smolgen", None)
    assert smolgen is not None, "the block has no shared smolgen (BT6SharedSmolgen / NBTSharedSmolgen)"
    dense(f"/nbt{b}/smolgen/compress", smolgen.sm1)
    dense(f"/nbt{b}/smolgen/dense1", smolgen.sm2)
    scale(f"/nbt{b}/smolgen/ln1", smolgen.ln1)
    dense(f"/nbt{b}/smolgen/dense2", smolgen.sm3)
    scale(f"/nbt{b}/smolgen/ln2", smolgen.ln2)
  dense("/smolgen/prep", net.smolgenPrepLayer)
  scale("/trunk/final_norm", net.trunk_end_norm)

  if variant == "nt":
    pro = hybrid.prologue
    D, T = f64(pro.D), f64(pro.T)
    offsets = T.T[:, pro.relidx.cpu()]                                        # [P, 64, 64]
    order = qkv_permutation(heads, hd)
    for b, block in enumerate(blocks):
      for j, layer in enumerate(block.inner):
        i = b * K + j
        p = f"/encoder{i}"
        att = layer.attention
        assert layer.pre_norm and layer.alpha == 1 and att.use_qkv and att.qkv.bias is None
        assert not att.use_smolgen and not att.use_rpe and not att.use_rope and att.softcap_cutoff == 0
        assert not getattr(att, "use_gated_attn_out", False) and not att.use_qk_norm
        scale(f"{p}/ln1", layer.ln1)
        put(f"{p}/mha/qkv/w", f64(att.qkv.weight)[torch.from_numpy(order)].T)
        dense(f"{p}/mha/out", att.W_h)
        scale(f"{p}/ln2", layer.ln2)
        mlp = layer.mlp
        assert mlp.linear3.bias is None
        put(f"{p}/ffn/dense1/w", torch.cat((f64(mlp.linear1.weight).T, f64(mlp.linear3.weight).T), dim=1))
        put(f"{p}/ffn/dense1/b", torch.cat((f64(mlp.linear1.bias), torch.zeros_like(f64(mlp.linear1.bias)))))
        dense(f"{p}/ffn/dense2", mlp.linear2)
        g = f"{p}/mha/egt"
        terms = getattr(att, "bt6t", None)
        if terms is None:
          z = lambda *shape: torch.zeros(*shape, dtype=torch.float64)
          put(f"{g}/qk_scale", torch.full((heads,), 1.0 / math.sqrt(hd), dtype=torch.float64), np.float32)
          put(f"{g}/attack", z(heads, cfg.relation_channels), np.float32)
          put(f"{g}/key", z(heads, cfg.relation_channels, hd), np.float32)
          put(f"{g}/query", z(heads, cfg.relation_channels, hd), np.float32)
          put(f"{g}/scaled_coefficients", z(heads, cfg.relation_channels), np.float32)
          put(f"{g}/constant_bias", z(heads, 64, 64), np.float32)
          put(f"{g}/edge_read/w", z(heads, cfg.edge_channels), np.float32)
          put(f"{g}/door/w", z(heads, cfg.edge_channels), np.float32)
          put(f"{g}/gate/w", z(heads, cfg.edge_channels), np.float32)
          put(f"{g}/gate/b", z(heads), np.float32)
        else:
          # BT6's folds (bt6_net.BT6Net.export_served), on the term module's tensors
          t_node, t_edge, C2 = f64(terms.t_node), f64(terms.t_edge), f64(terms.C2)
          put(f"{g}/qk_scale", t_node * (1.0 / math.sqrt(hd)), np.float32)
          put(f"{g}/attack", t_node[:, None] * f64(terms.C1), np.float32)
          put(f"{g}/key", t_node[:, None, None] * f64(terms.Pk), np.float32)
          put(f"{g}/query", t_node[:, None, None] * f64(terms.Pq), np.float32)
          put(f"{g}/scaled_coefficients", t_node[:, None] * (C2 @ D.T), np.float32)
          put(f"{g}/constant_bias", t_node[:, None, None] * torch.einsum("ht,trc->hrc", C2, offsets), np.float32)
          put(f"{g}/edge_read/w", t_edge[:, None] * f64(terms.We), np.float32)
          put(f"{g}/door/w", terms.Wm, np.float32)
          put(f"{g}/gate/w", terms.Wg, np.float32)
          put(f"{g}/gate/b", terms.bg, np.float32)
    assert cfg.rev_edge, "the edge-site export expects the reverse edge"
    for b in hybrid.site_blocks:
      site, s = hybrid.sites[str(b)], f"/encoder{b * K + K - 1}/edge_site"
      put(f"{s}/readback/w", site.readback, np.float32)
      if cfg.triplet:                          # BT6Triplet false: the site has no triplet branch and no such tensors
        put(f"{s}/triplet/value/w", site.triplet_value, np.float32)
        put(f"{s}/triplet/gate/w", site.triplet_gate, np.float32)
        put(f"{s}/triplet/gate/b", site.triplet_gate_bias, np.float32)
        put(f"{s}/triplet/out/w", site.triplet_out, np.float32)
      put(f"{s}/ffn/dense1/w", site.ffn_in, np.float32)
      put(f"{s}/ffn/dense1/b", site.ffn_in_bias, np.float32)
      put(f"{s}/ffn/dense1_rev/w", site.ffn_rev, np.float32)
      put(f"{s}/ffn/dense2/w", site.ffn_out, np.float32)

  # ---- heads
  dense("/head/premap", net.headPremap)
  dense("/head/shared", net.headSharedLinear)
  for name in ("value", "value2", "unc"):
    head = getattr(net, f"{name}_head", None)
    if head is not None and (name != "value2" or net.value2_loss_weight > 0):
      dense(f"/head/{name}/fc", head.fc)
      dense(f"/head/{name}/out", head.fcFinal)

  # ---- move-token decoder (the folds of the builder's ceres/carrier.py decoder_tensors_from_state_dict)
  mt = net.move_tokens
  for flag in ("rich_features", "value_query", "write_back", "square_update", "rel_bias", "minimax", "action_head",
               "opp_pool", "expected_value", "reply_sup"):
    assert not getattr(mt, flag, False), f"export_served_terms: MoveToken option {flag} is not exported"
  assert mt.M_opp == 0 and mt.trunk_mix == 0 and mt.post_move and mt.value_pool == "meanmax"
  d = net.embedding_layer.out_features
  dm = mt.w_in.out_features
  w_in = f64(mt.w_in.weight)
  assert w_in.shape == (dm, 2 * d + 4), f"move_tokens.w_in is {tuple(w_in.shape)}; expected {(dm, 2 * d + 4)}"
  put(f"{MT}/w_in/squares/w", torch.cat((w_in[:, :d].T, w_in[:, d:2 * d].T), dim=1))
  put(f"{MT}/w_in/rest/w", w_in[:, 2 * d:].T)
  put(f"{MT}/w_in/b", mt.w_in.bias)
  if mt.w_edge is not None:
    # MoveTokenEdgeInput: each token adds rms(e_final[:, from, to]) @ w_edge to its input embedding (FP32 table)
    assert tuple(mt.w_edge.shape) == (cfg.edge_channels, dm), f"move_tokens.w_edge is {tuple(mt.w_edge.shape)}"
    put(f"{MT}/w_edge/w", mt.w_edge, np.float32)
  put(f"{MT}/kv/norm/scale", torch.ones(d))
  cross, post = [], []
  for index, blk in enumerate(mt.blocks):
    p = f"{MT}/block{index}"
    qscale = blk.dk ** -0.5
    assert not getattr(blk, "_attn_scale_folded", False) and blk.post_move
    gain = f64(blk.ln_s.scale)[None, :]
    cross.append((f64(blk.xkv.weight) * gain).T)
    post.append((f64(blk.pm_kv.weight) * gain).T)
    qkv = f64(blk.qkv.weight).clone()
    qkv[:dm] *= qscale
    ffn = blk.ffn_in.out_features // 2
    scale(f"{p}/ln1", blk.ln1)
    put(f"{p}/qkv/w", qkv.T)
    put(f"{p}/proj/w", blk.proj.weight.T)
    scale(f"{p}/ln2", blk.ln2)
    put(f"{p}/xq/w", (f64(blk.xq.weight) * qscale).T)
    put(f"{p}/xproj/w", blk.xproj.weight.T)
    scale(f"{p}/ln_pm", blk.ln_pm)
    put(f"{p}/pm_q/w", (f64(blk.pm_q.weight) * qscale).T)
    put(f"{p}/pm_proj/w", blk.pm_proj.weight.T)
    put(f"{p}/pm_dk", blk.pm_dk)
    put(f"{p}/pm_dv", blk.pm_dv)
    scale(f"{p}/ln3", blk.ln3)
    put(f"{p}/ffn/dense1/w", blk.ffn_in.weight.T)
    put(f"{p}/ffn/dense1/b", torch.zeros(2 * ffn))
    put(f"{p}/ffn/dense2/w", blk.ffn_out.weight.T)
  put(f"{MT}/kv/w", torch.cat(cross + post, dim=1))
  scale(f"{MT}/out_ln", mt.out_ln)
  put(f"{MT}/pol/w", mt.pol.weight.T)
  table = torch.zeros(64 * 64 * 4, dtype=torch.float64)
  table[mt.mv_pair_slot.cpu().long()] = f64(mt.mt_pol_bias)
  put(f"{MT}/pol/bias_table", table, np.float32)
  put(f"{MT}/v_inject/w", mt.v_inject.weight.T)
  if net.value2_loss_weight > 0:
    put(f"{MT}/v2_inject/w", mt.v2_inject.weight.T)
  return out


def served_terms_meta(net) -> dict:
  """bt6_hybrid.served_meta plus the variant and the decoder facts."""
  meta = served_meta(net)
  mt = net.move_tokens
  meta.update({
    "variant": variant_of(net), "mt_layers": len(mt.blocks), "mt_heads": mt.blocks[0].h,
    "mv_pair_slot": mt.mv_pair_slot.detach().cpu().long(),
    "smolgen_activation": (net.config.NetDef_SmolgenActivationType if variant_of(net) == "nt"
                           else getattr(net.config, "NetDef_BT6SmolgenActivation", "Swish")),
  })
  return meta


@torch.no_grad()
def tokens_from_squares(net, squares: torch.Tensor) -> torch.Tensor:
  """The decoder's own candidate selection as the program's /input/move_tokens: uint8 [B, M, 4] = from, to, flags
  (bit 0 valid, bits 1..4 the pair's four visibility bits), mover piece type 1..6; padding tokens are 0, 0, 0, 1."""
  mt = net.move_tokens
  squares13 = squares[:, :, 0:13].float()
  B = squares13.shape[0]
  cand, E = mt.candidates(squares13)
  score = cand.float() + (mt.tie_rank.float() * (0.5 / 4096.0)).unsqueeze(0)
  sel = torch.topk(score, mt.M, dim=1).indices
  valid = torch.gather(cand, 1, sel) > 0.5
  fr, to = sel // 64, sel % 64
  bits = (torch.gather(E.reshape(B, 4096, 4), 1, sel.unsqueeze(-1).expand(-1, -1, 4)) > 0.5).long()
  mover = torch.gather(squares13[:, :, 1:7], 1, fr.unsqueeze(-1).expand(-1, -1, 6)).argmax(dim=-1) + 1
  flags = valid.long() | (bits[..., 0] << 1) | (bits[..., 1] << 2) | (bits[..., 2] << 3) | (bits[..., 3] << 4)
  tokens = torch.stack((fr, to, flags, mover), dim=-1)
  padding = torch.tensor([0, 0, 0, 1], device=tokens.device)
  return torch.where(valid.unsqueeze(-1), tokens, padding).to(torch.uint8)


# ====================================================================================================================
# The forward from the served tensors, in kernel form
# ====================================================================================================================
def _rms_norm(x, scale, eps=1e-6):
  return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * scale


def served_native_layer(t: dict, index: int, x, E, S, e, cfg: BT6Config, bias, terms: bool):
  """One inner layer of variant "nt" from /encoder{index}/... (our pre-norm layer; kernel `attention_terms`):
  x [B, 64, d] -> (x, H). `bias` [B, heads, 64, 64] is the block's smolgen bias (added on every layer, unscaled)."""
  batch, heads, hd, width = x.shape[0], cfg.heads, cfg.head_dim, cfg.d_model
  p = f"/encoder{index}"
  g = f"{p}/mha/egt"
  y = _rms_norm(x, t[f"{p}/ln1/scale"])
  lanes = (y @ t[f"{p}/mha/qkv/w"]).view(batch, 64, 3, heads, hd)
  q, k, v = lanes[:, :, 0], lanes[:, :, 1], lanes[:, :, 2]
  scores = torch.einsum("bihd,bjhd->bhij", q, k) * t[f"{g}/qk_scale"].view(1, heads, 1, 1)
  if terms:
    row = torch.einsum("bihd,hcd->bhic", q, t[f"{g}/query"]) + t[f"{g}/attack"][None, :, None, :]
    column = torch.einsum("bjhd,hcd->bhjc", k, t[f"{g}/key"])
    scores = scores + torch.einsum("bhic,bcij->bhij", row, E) + torch.einsum("bhjc,bcij->bhij", column, E)
    scores = scores + S[:, None] * (torch.einsum("hc,bcij->bhij", t[f"{g}/scaled_coefficients"], E)
                                    + t[f"{g}/constant_bias"])
  if bias is not None:
    scores = scores + bias
  if terms:
    read = torch.einsum("hc,bcij->bhij", t[f"{g}/edge_read/w"], e)
    door = torch.einsum("hc,bcij->bhij", t[f"{g}/door/w"], e)
    gate = torch.einsum("hc,bcij->bhij", t[f"{g}/gate/w"], e) + t[f"{g}/gate/b"].view(1, heads, 1, 1)
    H = (scores + read) * (1.0 + door)
    gate = 2.0 * torch.sigmoid(gate)
    if cfg.gate_cap:
      gate = gate.clamp(max=1.0)
    A = torch.softmax(H, dim=-1) * gate
  else:
    H = scores
    A = torch.softmax(H, dim=-1)
  attended = torch.einsum("bhij,bjhd->bihd", A, v).reshape(batch, 64, width)
  x = x + _dense(attended, t, f"{p}/mha/out")
  f = _rms_norm(x, t[f"{p}/ln2/scale"])
  gate, up = _dense(f, t, f"{p}/ffn/dense1").chunk(2, dim=-1)
  x = x + _dense(F.silu(gate) * up, t, f"{p}/ffn/dense2")
  return x, H


def served_decoder(t: dict, x, tokens: torch.Tensor, meta: dict, edge=None):
  """The move-token decoder from /mt/... (the builder's ceres/network.py `_decoder`): trunk state after the trunk norm
  [B, 64, D] and the tokens uint8 [B, M, 4] -> (token logits [B, M, 4] incl. the per-move bias, -30 on invalid tokens;
  pooled [B, 2 dm]). `edge` [B, d_e, 64, 64] (row = from, column = to): the BT6 edge state after the last site, read
  when the carrier has /mt/w_edge/w (MoveTokenEdgeInput; kernel `mt_edge_input`). A padding token reads the pair
  (0, 0); it is masked everywhere downstream, as its whole embedding is."""
  B, M = tokens.shape[0], tokens.shape[1]
  layers, heads = meta["mt_layers"], meta["mt_heads"]
  tok = tokens.long()
  fr, to, flags, mover = tok[..., 0], tok[..., 1], tok[..., 2], tok[..., 3]
  valid = (flags & 1) > 0
  bits = torch.stack([(flags >> s) & 1 for s in (1, 2, 3, 4)], dim=-1).to(x.dtype)
  dm = t[f"{MT}/w_in/b"].shape[0]
  dk = dm // heads
  P = x @ t[f"{MT}/w_in/squares/w"]                                           # [B, 64, 2 dm]
  take = lambda src, idx: torch.gather(src, 1, idx.unsqueeze(-1).expand(-1, -1, src.shape[-1]))
  h = take(P[..., :dm], fr) + take(P[..., dm:], to) + bits @ t[f"{MT}/w_in/rest/w"] + t[f"{MT}/w_in/b"]
  if f"{MT}/w_edge/w" in t:
    assert edge is not None, "the carrier has /mt/w_edge/w: the decoder needs the final edge state"
    e_tok = take(edge.float().flatten(2).transpose(1, 2), fr * 64 + to)        # [B, M, d_e]
    e_tok = e_tok * torch.rsqrt(e_tok.pow(2).mean(-1, keepdim=True) + 1e-6)
    h = h + e_tok @ t[f"{MT}/w_edge/w"]
  n = _rms_norm(x, t[f"{MT}/kv/norm/scale"])
  KV = n @ t[f"{MT}/kv/w"]                                                    # [B, 64, 4 L dm]
  key_bias = (~valid).to(x.dtype).view(B, 1, 1, M) * -1e4
  ar64 = torch.arange(64, device=x.device).view(1, 1, 64)
  oh_to = (ar64 == to.unsqueeze(-1)).to(x.dtype).unsqueeze(1)                 # [B, 1, M, 64]
  oh_from = (ar64 == fr.unsqueeze(-1)).to(x.dtype).unsqueeze(1)

  def split(a):
    return a.reshape(a.shape[0], a.shape[1], heads, dk).transpose(1, 2)

  def merge(a):
    return a.transpose(1, 2).reshape(B, M, dm)

  for b in range(layers):
    p = f"{MT}/block{b}"
    q, k, v = (_rms_norm(h, t[f"{p}/ln1/scale"]) @ t[f"{p}/qkv/w"]).chunk(3, dim=-1)
    a = torch.softmax(split(q) @ split(k).transpose(2, 3) + key_bias, dim=-1)
    h = h + merge(a @ split(v)) @ t[f"{p}/proj/w"]
    q = _rms_norm(h, t[f"{p}/ln2/scale"]) @ t[f"{p}/xq/w"]
    k, v = KV[..., b * 2 * dm:b * 2 * dm + dm], KV[..., b * 2 * dm + dm:(b + 1) * 2 * dm]
    a = torch.softmax(split(q) @ split(k).transpose(2, 3), dim=-1)
    h = h + merge(a @ split(v)) @ t[f"{p}/xproj/w"]
    q = split(_rms_norm(h, t[f"{p}/ln_pm/scale"]) @ t[f"{p}/pm_q/w"])           # the scale is folded into pm_q
    o = (layers + b) * 2 * dm
    k, v = split(KV[..., o:o + dm]), split(KV[..., o + dm:o + 2 * dm])
    dk_to, dv_to = split(t[f"{p}/pm_dk"][mover]), split(t[f"{p}/pm_dv"][mover])
    dk_fr, dv_fr = t[f"{p}/pm_dk"][0].view(1, heads, 1, dk), t[f"{p}/pm_dv"][0].view(1, heads, 1, dk)
    s = q @ k.transpose(2, 3) + (q * dk_to).sum(-1, keepdim=True) * oh_to + (q * dk_fr).sum(-1, keepdim=True) * oh_from
    a = torch.softmax(s, dim=-1)
    out = a @ v + (a * oh_to).sum(-1, keepdim=True) * dv_to + (a * oh_from).sum(-1, keepdim=True) * dv_fr
    h = h + merge(out) @ t[f"{p}/pm_proj/w"]
    gate, up = (_rms_norm(h, t[f"{p}/ln3/scale"]) @ t[f"{p}/ffn/dense1/w"] + t[f"{p}/ffn/dense1/b"]).chunk(2, dim=-1)
    h = h + (F.silu(gate) * up) @ t[f"{p}/ffn/dense2/w"]
  xo = _rms_norm(h, t[f"{MT}/out_ln/scale"])
  slots = (fr * 64 + to).unsqueeze(-1) * 4 + torch.arange(4, device=x.device)
  logits = xo @ t[f"{MT}/pol/w"] + t[f"{MT}/pol/bias_table"][slots]
  logits = torch.where(valid.unsqueeze(-1), logits, torch.full_like(logits, MT_FLOOR))
  w = valid.to(xo.dtype).unsqueeze(-1)
  count = w.sum(dim=1)
  mean_pool = (xo * w).sum(dim=1) / count.clamp_min(1.0)
  max_pool = torch.where(count > 0, (xo + (w - 1.0) * 1e4).amax(dim=1), torch.zeros_like(mean_pool))
  return logits, torch.cat((mean_pool, max_pool), dim=-1)


def policy_from_tokens(t: dict, token_logits: torch.Tensor, tokens: torch.Tensor, mv_pair_slot: torch.Tensor):
  """Token logits [B, M, 4] (incl. bias) -> the net's [B, 1858] policy output (floor + bias on absent moves)."""
  B = tokens.shape[0]
  tok = tokens.long()
  pair = tok[..., 0] * 64 + tok[..., 1]
  valid = (tok[..., 2] & 1) > 0
  table = t[f"{MT}/pol/bias_table"].view(4096, 4)
  raw = torch.where(valid.unsqueeze(-1), token_logits - table[pair], torch.full_like(token_logits, MT_FLOOR))
  buf = torch.full((B, 4097, 4), MT_FLOOR, dtype=token_logits.dtype, device=token_logits.device)
  index = torch.where(valid, pair, torch.full_like(pair, 4096))               # padding tokens go to a spare row
  buf = buf.scatter(1, index.unsqueeze(-1).expand(-1, -1, 4), raw)[:, :4096]
  slot = mv_pair_slot.to(token_logits.device)
  return buf.reshape(B, 16384).index_select(1, slot) + table.reshape(-1).index_select(0, slot)


@torch.no_grad()
def forward_served_terms(t: dict, squares: torch.Tensor, meta: dict, tokens: torch.Tensor, taps: dict | None = None) -> dict:
  """The forward of a "bt" / "nt" terms net computed from the SERVED tensors (export_served_terms), in kernel form.
  `t`: {name: float32 tensor} (bt6_net.served_to_torch); `squares` [B, 64, F] as the net receives it (byte / 100);
  `tokens` uint8 [B, M, 4] (tokens_from_squares, or what the engine builds). Returns policy [B, 1858] (the net's
  output layout), policy_tokens [B, M, 4], value, value2 [B, 3] (WDL logits, decoder injects included)."""
  from activation_functions import to_activation

  cfg = meta["layer_config"]
  variant = meta["variant"]
  batch = squares.shape[0]
  squares = squares.float()
  assert meta["outer_norm"] == "RMSNorm" and meta["embedding"] == "linear" and not meta["proj_shift"]
  eps = meta["outer_norm_eps"]
  project = to_activation(meta["proj_activation"])
  smol_act = to_activation(meta["smolgen_activation"])

  def keep(name, value):
    if taps is not None:
      taps[name] = value
    return value

  pieces, previous = pieces_from_squares(squares)
  E_bool = BT6Relations().to(squares.device)(pieces, previous)
  E = E_bool.float()
  S, e = served_prologue(t, E_bool, cfg)
  keep("edge_norm", S)
  keep("state_seed", e)

  blunders = squares[:, 0, 119:121]
  record = torch.cat((squares[:, :, :119], torch.zeros_like(squares[:, :, 119:121]), squares[:, :, 121:]), dim=2)
  x = keep("embedding", _rms_norm(_dense(record, t, "/embed/linear"), t["/embed/norm/scale"], eps))

  K = meta["inner_layers"]
  term_positions = meta["terms_inner_layers"]
  for b in range(meta["blocks"]):
    n_in = _rms_norm(x, t[f"/nbt{b}/norm_in/scale"], eps)
    p = f"/nbt{b}/smolgen"
    s = _dense(n_in, t, f"{p}/compress").reshape(batch, -1)
    s = _rms_norm(smol_act(_dense(s, t, f"{p}/dense1")), t[f"{p}/ln1/scale"], eps)
    s = _rms_norm(smol_act(_dense(s, t, f"{p}/dense2")), t[f"{p}/ln2/scale"], eps)
    bias = keep(f"smolgen{b}", _dense(s.reshape(batch, cfg.heads, -1), t, "/smolgen/prep").reshape(batch, cfg.heads, 64, 64))
    h = keep(f"block_in{b}", project(n_in) @ t[f"/nbt{b}/down/w"])
    H = None
    for j in range(K):
      i = b * K + j
      if variant == "bt":
        h, H = served_layer(t, i, h, E, S, e, cfg, bias=None if j in term_positions else bias)
      else:
        h, H = served_native_layer(t, i, h, E, S, e, cfg, bias, j in term_positions)
      keep(f"layer{i}", h)
    x = keep(f"block{b}", x + project(_rms_norm(h, t[f"/nbt{b}/norm_out/scale"], eps)) @ t[f"/nbt{b}/up/w"])
    if b in meta["site_blocks"]:
      keep(f"logits{b * K + K - 1}", H)
      e = keep(f"state{b}", served_site(t, b * K + K - 1, e, H, cfg))
  x = keep("trunk_norm", _rms_norm(x, t["/trunk/final_norm/scale"], eps))

  token_logits, pooled = served_decoder(t, x, tokens, meta, edge=e)
  keep("mt_pooled", pooled)
  out = {"policy_tokens": token_logits,
         "policy": policy_from_tokens(t, token_logits, tokens, meta["mv_pair_slot"])}
  act = to_activation(meta["heads_activation"])
  front = _dense(x, t, "/head/premap")
  front = act(front) if meta["head_front_act"] else front
  front = _dense(front.reshape(batch, -1), t, "/head/shared")
  front = keep("head_shared", act(front) if meta["head_front_act"] else front)
  out["value"] = _dense(act(_dense(front, t, "/head/value/fc") + pooled @ t[f"{MT}/v_inject/w"]), t, "/head/value/out")
  if meta["value2"]:
    hidden = _dense(torch.cat((front, blunders), dim=-1), t, "/head/value2/fc") + pooled @ t[f"{MT}/v2_inject/w"]
    out["value2"] = _dense(act(hidden), t, "/head/value2/out")
  return out
