# License Notice
"""
This file is part of the CeresTrain project at https://github.com/dje-dev/CeresTrain.
Copyright (C) 2023- by David Elliott and the CeresTrain Authors.
GNU GPL v3.0 — see <http://www.gnu.org/licenses/>.
"""
# End of License Notice

"""Dual-Plane Attention — the P-plane (piece tokens) of the dual-plane
concept (F:/cout/Findings/dual_plane_concept.md). Stage A1 scope: a small
piece-token transformer with relation-typed attention biases, read into the
VALUE family only via zero-init injects. The S-plane (square trunk) is
untouched, so policy isolation is provable (vpc smoke pattern).

Mechanism summary:
  - 32 piece slots selected by occupancy TopK over the 64 squares
    (deterministic tie-break by square index; chess has <=32 pieces, so all
    real pieces are always selected; empty slots are attention-masked).
    TopK + Gather were TRT-validated by the PTV day-zero build.
  - Piece token init: Linear(13 piece one-hot + 8 file one-hot + 8 rank
    one-hot -> dp) — identity + location, the classic piece-square pairing.
  - Relation-typed P<->P bias: the parent computes VisibilityChannels E
    [B, 64, 64, C] ONCE; we double-gather to piece pairs [B, 32, 32, C] and
    project per-head with a ZERO-INIT linear (bias starts silent; the
    magnitude trajectory is the does-the-net-want-this diagnostic, FGH
    lesson).
  - num_layers pre-norm P-blocks (MHA + FFN). Optional soft-min heads
    (quantifiers over PIECES — "is ANY piece hanging") via the same signed
    LSE formulation as dot_product_attention.
  - One cross-read P->S: piece tokens attend the FINAL square flow
    (queries = pieces, keys/values = squares), zero-init out-projection.
  - Output: [B, 2*dp] (masked mean-pool + masked soft-min pool over piece
    tokens), consumed by ceres_net through zero-init Linears into the
    value1/value2 hidden pre-activation (the proven inject pattern).

Everything is matmul/softmax/topk/gather — export/TRT-safe by construction.
Zero-init injects => exact step-0 bit-identity for ALL heads.
"""

import torch
from torch import nn
from rms_norm import make_norm


class DualPlanePBlock(nn.Module):
  """Pre-norm piece-plane encoder block with additive relation bias."""

  def __init__(self, dp: int, heads: int, rel_channels: int, ffn_mult: int,
               norm_type: str, softmin_heads: int = 0, rel_gains: bool = False,
               edge_update: bool = False):
    super().__init__()
    assert dp % heads == 0
    self.dp, self.heads, self.dk = dp, heads, dp // heads
    self.rel_channels = rel_channels
    self.softmin_heads = softmin_heads
    self.ln1 = make_norm(norm_type, dp)
    self.qkv = nn.Linear(dp, 3 * dp, bias=False)
    self.proj = nn.Linear(dp, dp, bias=False)
    # Relation-typed bias: per-head zero-init projection of the gathered
    # edge channels. Zero-init => block starts as PLAIN attention.
    self.rel_proj = nn.Linear(rel_channels, heads, bias=False)
    nn.init.zeros_(self.rel_proj.weight)
    # Relational-type modulation ("P-plane smolgen", catalogue idea, 2026-08-20):
    # a masked mean of the block's normed piece tokens conditions per-head,
    # per-channel GAINS on the relation-bias weights — the global position
    # picks WHICH relation types matter now ("weight check edges up, xray
    # down"), rather than generating raw slot-pair maps (slot semantics vary
    # per position, so classic smolgen maps would be uninterpretable here).
    # W_eff = rel_proj.weight * (1 + delta): zero-init delta => exact no-op,
    # and the modulator can only scale relations the net has already grown
    # (gradient reaches gain_mlp only once rel_proj is nonzero).
    #
    # ⛔⛔ DO NOT ENABLE FOR ANY NET THAT WILL BE SERVED — TRT-UNSERVABLE.
    # Measured 2026-08-22, matched pair (rgsmk vs t91pkg, same chassis/corpus/
    # steps, only this flag differs), EB cmp both orders on a CACHED engine,
    # reproduced independently by the user:
    #     rel-gains  547 EPS   vs   control  25,050 EPS   =  46x SLOWER.
    # Not a measurement artifact and not an export bug:
    #   * ORT runs both at the SAME speed (0.597 s vs 0.603 s / 64 pos), so the
    #     arithmetic is free — this is purely a TRT compilation outcome.
    #   * The graph diff is exactly 4 extra MatMul nodes (one per P-block) whose
    #     SECOND operand is dynamic: 50 -> 54 dynamic-weight MatMuls, while the
    #     178 constant-weight ones are unchanged.
    #   * Cause: the control's rel_proj weight is CONSTANT, so TRT bakes it in
    #     and collapses the batch into the GEMM's M dimension — one big
    #     [B*1024, C] x [C, H]. Here W_eff is per-sample, so the batch cannot
    #     collapse: it becomes B separate [1024, C] x [C, H] GEMMs per block
    #     (~900 tiny launches per forward at batch 224), with K=C=20 and N=H,
    #     far too skinny for tensor cores. TRT then picks generic tactics —
    #     visible as a SMALLER engine (69.9 MB vs 75.8 MB) despite more layers.
    # NOT fixable by re-exporting: delta depends on the channel c, so it cannot
    # be factored out of the sum over c. Channel selectivity IS the mechanism,
    # and it is also exactly what makes it unservable.
    # Servable redesigns if the hypothesis is worth revisiting:
    #   (a) per-HEAD gain only (delta over (b,h), not c) — then it factors out
    #       to delta * sharedGEMM, essentially free, but loses "which relation
    #       type matters now", which was the point;
    #   (b) let delta pick among K fixed channel-weightings — K constant GEMMs
    #       plus a per-sample weighting of K scalars, still fusable.
    # Also note the ORIGINAL 5M verdict ("policy-neutral", parked) is suspect
    # for an unrelated reason: gain_mlp's gradient is proportional to rel_proj's
    # magnitude, which is zero-init, so the modulator barely activates inside a
    # short run. The hypothesis is untested; only this IMPLEMENTATION is dead.
    self.rel_gains = rel_gains
    if rel_gains:
      self.gain_mlp = nn.Linear(dp, heads * rel_channels, bias=False)
      nn.init.zeros_(self.gain_mlp.weight)
    if softmin_heads > 0:
      assert softmin_heads <= heads,           f'DualPlaneSoftMinHeads ({softmin_heads}) > heads ({heads}) — tau-broadcast kolliderer (runde-3-vern)'
      self.softmin_log_tau = nn.Parameter(torch.zeros(softmin_heads))
    # LAERT KANT-OPPDATERING (boelge 6, 2026-08-29 — EGT-ens manglende halvdel):
    # kanten leser sine to endepunkt-noder + seg selv og oppdateres residualt;
    # attention-biasen leser deretter LEVENDE kanter, og node-tokenene faar en
    # degree-refresh fra dem (aggregeringsruten — vinnerveien per softmax-doer-
    # doktrinen, konvergent maalt i begge programmene). Zero-init paa BEGGE
    # output-projeksjonene => eksakt step-0 no-op; inputs er rike (rel_gains-
    # laerdommen: gradientvei fra steg en). Kun dense ops (dpdiff-TRT-lekse).
    self.edge_update = edge_update
    if edge_update:
      _h = 32
      self.eu_e = nn.Linear(rel_channels, _h)
      self.eu_i = nn.Linear(dp, _h, bias=False)
      self.eu_j = nn.Linear(dp, _h, bias=False)
      self.eu_out = nn.Linear(_h, rel_channels, bias=False)
      nn.init.zeros_(self.eu_out.weight)
      self.eu_deg = nn.Linear(2 * rel_channels, dp, bias=False)
      nn.init.zeros_(self.eu_deg.weight)
    self.ln2 = make_norm(norm_type, dp)
    self.ffn1 = nn.Linear(dp, ffn_mult * dp, bias=False)
    self.ffn2 = nn.Linear(ffn_mult * dp, dp, bias=False)
    self.act = nn.Mish()

  def forward(self, x, rel_pair, pad_bias):
    # x [B, 32, dp]; rel_pair [B, 32, 32, C]; pad_bias [B, 1, 1, 32]
    # Returnerer (x, rel_pair) — kant-tilstanden traades gjennom blokkene.
    B = x.shape[0]
    if self.edge_update:
      _ei = self.eu_i(x).unsqueeze(2)                      # [B,32,1,h]
      _ej = self.eu_j(x).unsqueeze(1)                      # [B,1,32,h]
      _ee = self.eu_e(rel_pair.to(x.dtype))                # [B,32,32,h]
      rel_pair = rel_pair.to(x.dtype) + self.eu_out(torch.nn.functional.mish(_ee + _ei + _ej))
      # degree-refresh fra levende kanter (samme aksesemantikk som rel_degrees)
      x = x + self.eu_deg(torch.cat([rel_pair.sum(dim=2), rel_pair.sum(dim=1)], dim=-1))
    h = self.ln1(x)
    q, k, v = self.qkv(h).chunk(3, dim=-1)
    q = q.reshape(B, 32, self.heads, self.dk).transpose(1, 2)
    k = k.reshape(B, 32, self.heads, self.dk).transpose(1, 2)
    v = v.reshape(B, 32, self.heads, self.dk).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(2, 3)) * (self.dk ** -0.5)
    if self.rel_gains:
      # Masked mean of normed tokens -> per-head/channel gain deltas.
      _w = (pad_bias.reshape(B, 32, 1) > -5e3).to(h.dtype)   # midpoint: occ>0.5 = real slot
      _g = (h * _w).sum(dim=1) / _w.sum(dim=1).clamp_min(1.0)          # [B, dp]
      delta = self.gain_mlp(_g).reshape(B, self.heads, self.rel_channels)
      W_eff = self.rel_proj.weight.unsqueeze(0) * (1.0 + delta)        # [B, H, C]
      bias = torch.matmul(rel_pair.reshape(B, 32 * 32, -1).to(scores.dtype),
                          W_eff.transpose(1, 2).to(scores.dtype))      # [B, 1024, H]
      scores = scores + bias.permute(0, 2, 1).reshape(B, self.heads, 32, 32)
    else:
      scores = scores + self.rel_proj(rel_pair.to(scores.dtype)).permute(0, 3, 1, 2)
    scores = scores + pad_bias.to(scores.dtype)          # mask empty slots as keys
    A = torch.softmax(scores, dim=-1)
    if self.softmin_heads > 0:
      ks = self.softmin_heads
      A_s, V_s = A[:, :ks].float(), v[:, :ks].float()
      tau = torch.exp(self.softmin_log_tau.clamp(-4.0, 4.0)).reshape(1, ks, 1, 1)
      negtv = -tau * V_s
      # Max-shift over VALID keys only: empty slots carry never-trained
      # garbage V; letting them set the shift would underflow every real
      # term (their A-weight is 0 so they contribute nothing to s, but the
      # SHIFT is global). pad_bias is [B,1,1,32] keyed on keys — move it to
      # the key axis of negtv ([B,ks,32keys,dk]).
      _key_pad = pad_bias.reshape(pad_bias.shape[0], 1, 32, 1).float()
      m = (negtv + _key_pad).amax(dim=2, keepdim=True)
      s = torch.matmul(A_s, torch.exp(negtv + _key_pad - m))
      H_s = -(torch.log(s.clamp_min(1e-20)) + m) / tau
      out = torch.cat([H_s.to(v.dtype), torch.matmul(A[:, ks:], v[:, ks:])], dim=1)
    else:
      out = torch.matmul(A, v)
    out = out.transpose(1, 2).reshape(B, 32, self.dp)
    x = x + self.proj(out)
    x = x + self.ffn2(self.act(self.ffn1(self.ln2(x))))
    return x, rel_pair


class DualPlane(nn.Module):
  """P-plane stack + S-plane cross-read + pooled outputs (Stage A1 scope)."""

  def __init__(self, s_dim: int, rel_channels: int, norm_type: str,
               dp: int = 128, heads: int = 4, layers: int = 2,
               ffn_mult: int = 2, softmin_heads: int = 2,
               interleave_cross: bool = False,
               edge_update: bool = False,
               rel_degrees: bool = False,
               rel_degrees2: bool = False,
               rel_gains: bool = False,
               king_flight: bool = False,
               king_zone: bool = False):
    super().__init__()
    self.dp = dp
    # interleave_cross: apply the (weight-shared, zero-init) S-plane
    # cross-read after EVERY P-block instead of once at the end — each
    # P-block then reasons over piece states refreshed with square context.
    # Weight sharing keeps the param count flat; zero-init x_out keeps the
    # exact step-0 no-op regardless of how many times it is applied.
    self.interleave_cross = interleave_cross
    # Piece identity+location embedding: 13 one-hot + 8 file + 8 rank.
    self.embed = nn.Linear(13 + 16, dp, bias=False)
    fr = torch.zeros(64, 16)
    for sq in range(64):
      fr[sq, sq % 8] = 1.0
      fr[sq, 8 + sq // 8] = 1.0
    self.register_buffer('filerank', fr, persistent=False)
    # Relation DEGREES as token features (catalogue idea A, 2026-08-20):
    # each piece token receives its per-channel row/column sums of the
    # relation tensor — "I attack 2, I am attacked by 3, defended by 1" —
    # overload/hanging detection is DEGREE COUNTING, which attention
    # otherwise spends capacity re-deriving. One reduce over an already-
    # computed tensor; zero-init projection => exact step-0 no-op.
    # Descriptive facts only (sac-rule compliant).
    self.rel_degrees = rel_degrees
    if rel_degrees:
      self.deg_proj = nn.Linear(2 * rel_channels, dp, bias=False)
      nn.init.zeros_(self.deg_proj.weight)
    # SECOND-ORDER degrees (2026-08-20 night; the computation-over-relations
    # class that produced both wins, one hop deeper): per channel, how
    # COVERED are the pieces I target (sum_j rel[i,j]*indeg[j]) and how
    # covered are the pieces targeting me (sum_j rel[j,i]*indeg[j]) — the
    # exchange-calculus primitives ("my target is defended twice", "my
    # attacker is protected") as DESCRIPTIVE counting, not a material prior
    # (sac-rule compliant: no piece values anywhere). Two tiny per-channel
    # matmuls over the already-gathered rel_pair, kept in source dtype (the
    # kf-v1 lesson: never .float() a big tensor before reducing it).
    self.rel_degrees2 = rel_degrees2
    if rel_degrees2:
      self.deg2_proj = nn.Linear(2 * rel_channels, dp, bias=False)
      nn.init.zeros_(self.deg2_proj.weight)
    # kf2 (2026-08-20 night, after kf-v1 parked): the SAME flight-zone
    # information delivered to the tokens that USE it — every piece gets its
    # own per-channel coverage of the two king zones ("how much of the enemy
    # king's 3x3 do I cover / how much of my own king's do I defend"),
    # instead of a summary parked in the king's token that attention must
    # learn to route (v1's measured failure: mate themes flat, forcing
    # themes taxed). Reads the ALREADY-GATHERED e_rows tensor — no full-E
    # reduction (v1's 9% TRT cost class avoided by construction).
    # Descriptive facts only; zero-init projection => exact step-0 no-op.
    self.king_zone = king_zone
    if king_zone:
      self.kz_proj = nn.Linear(2 * rel_channels, dp, bias=False)
      nn.init.zeros_(self.kz_proj.weight)
    # King flight-zone features v1 (catalogue idea #6; PARKED on evidence —
    # neutral aggregate, ~9% TRT cost — but kept flag-gated for the record).
    # Per king: each of the 8 neighbor squares contributes its per-channel
    # in-degree from E ("who covers this flight square"), its 13-dim
    # occupancy and an on-board bit; the concat is zero-init projected into
    # the KING'S OWN token before the P-blocks.
    self.king_flight = king_flight
    if king_flight or king_zone:
      _nbr = torch.zeros(64, 8, dtype=torch.long)
      _onb = torch.zeros(64, 8)
      _dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
      for sq in range(64):
        f, r = sq % 8, sq // 8
        for j, (df, dr) in enumerate(_dirs):
          nf, nr = f + df, r + dr
          if 0 <= nf < 8 and 0 <= nr < 8:
            _nbr[sq, j] = nr * 8 + nf
            _onb[sq, j] = 1.0
          else:
            _nbr[sq, j] = sq          # off-board: self-pad, on-board bit 0
      self.register_buffer('kf_nbr', _nbr, persistent=False)
      self.register_buffer('kf_onb', _onb, persistent=False)
      if king_flight:
        self.kf_proj = nn.Linear(8 * (rel_channels + 13 + 1), dp, bias=False)
        nn.init.zeros_(self.kf_proj.weight)
    self.blocks = nn.ModuleList([
        DualPlanePBlock(dp, heads, rel_channels, ffn_mult, norm_type,
                        softmin_heads=softmin_heads, rel_gains=rel_gains,
                        edge_update=edge_update)
        for _ in range(layers)])
    # Cross-read P->S on the FINAL square flow.
    self.x_ln_p = make_norm(norm_type, dp)
    self.x_ln_s = make_norm(norm_type, s_dim)
    self.x_q = nn.Linear(dp, dp, bias=False)
    self.x_k = nn.Linear(s_dim, dp, bias=False)
    self.x_v = nn.Linear(s_dim, dp, bias=False)
    self.x_out = nn.Linear(dp, dp, bias=False)
    nn.init.zeros_(self.x_out.weight)                    # cross-read starts silent
    self.out_ln = make_norm(norm_type, dp)

  def forward(self, squares13, vis_E, s_flow):
    """squares13 [B,64,13] one-hot; vis_E [B,64,64,C]; s_flow [B,64,S].
    Returns [B, 2*dp] pooled piece summary (masked mean + masked soft-min)."""
    x, rel_pair, sel, slot_occ = self.run_pblocks(squares13, vis_E,
                                                  s_flow if self.interleave_cross else None)
    p, xt, sl, oc = self.finish(x, s_flow, sel, slot_occ)
    return p, xt, sl, oc

  def run_pblocks(self, squares13, vis_E, interleave_s_flow=None):
    """Fase 1 (boelge 9): alt som er TRUNK-UAVHENGIG — embed, feature-injects
    og P-blokkene (med levende kanter). Muliggjoer kant->trunk-eksport: fasen
    kan kjoeres FOER trunken, og de ferdig-utviklede kantene bias-loeftes inn
    i trunk-attention. Returns (x_tokens, rel_pair_final, sel, slot_occ)."""
    B = squares13.shape[0]
    occ = 1.0 - squares13[:, :, 0]                       # [B, 64]
    # Deterministic slot selection: occupancy + tiny positional tie-break.
    # Runde-3/4: tie-break i fp32 OG med steg > fp16-ulp (9.8e-4 ved 1.0) —
    # fp16-eksporten regner Add-en i fp16 (Add er ikke paa blokklisten), saa
    # 1e-4 kollapset der. 2e-3-steget (maks 64*2e-3=0.128 << 1.0-gapet mot
    # tomme felter) overlever begge presisjoner; settet var alltid riktig,
    # dette gjelder slot-ORDEN/bit-paritet eager-vs-ORT/TRT.
    tie = torch.arange(64, device=squares13.device, dtype=torch.float32) * 2e-3
    sel = torch.topk(occ.float() + tie, k=32, dim=1).indices     # [B, 32]
    slot_occ = torch.gather(occ, 1, sel)                 # [B, 32] 1=piece, 0=empty
    feats = torch.cat([squares13,
                       self.filerank.to(squares13.dtype).unsqueeze(0).expand(B, 64, 16)],
                      dim=-1)                            # [B, 64, 29]
    x = self.embed(torch.gather(feats, 1, sel.unsqueeze(-1).expand(-1, -1, 29)))
    # Pair-gather the relation channels: E[b, sel_i, sel_j, :].
    C = vis_E.shape[-1]
    e_rows = torch.gather(vis_E, 1, sel.reshape(B, 32, 1, 1).expand(-1, -1, 64, C))
    rel_pair = torch.gather(e_rows, 2, sel.reshape(B, 1, 32, 1).expand(-1, 32, -1, C))
    pad_bias = ((slot_occ - 1.0) * 1e4).reshape(B, 1, 1, 32)  # empty slots: -1e4 as keys

    if self.rel_degrees:
      # out-degree (what I do to others) + in-degree (what others do to me),
      # per channel; empty slots' garbage degrees are harmless (their tokens
      # are masked from keys/pools everywhere downstream).
      _deg = torch.cat([rel_pair.sum(dim=2), rel_pair.sum(dim=1)], dim=-1)   # [B, 32, 2C]
      x = x + self.deg_proj(_deg.to(x.dtype))

    if self.rel_degrees2:
      _rp = rel_pair.permute(0, 3, 1, 2)                   # [B, C, 32, 32], source dtype
      _ind = _rp.sum(dim=2)                                # [B, C, 32] coverage of slot j
      _t2 = torch.matmul(_rp, _ind.unsqueeze(-1)).squeeze(-1)                 # my targets' coverage
      _a2 = torch.matmul(_rp.transpose(2, 3), _ind.unsqueeze(-1)).squeeze(-1) # my attackers' coverage
      _d2 = torch.cat([_t2, _a2], dim=1).permute(0, 2, 1)  # [B, 32, 2C]
      x = x + self.deg2_proj(_d2.to(x.dtype))

    if self.king_zone:
      # kf2: per-piece coverage of both king zones, read from e_rows (piece ->
      # all squares) which is already gathered above. Zone = king square + its
      # 8 neighbors (off-board self-pads contribute the king square again —
      # harmless, uniform). Channel layout: kings at 6 (ours), 12 (theirs).
      _kz_feats = []
      for _kch in (12, 6):                                 # enemy zone first, then own
        _ksq = squares13[:, :, _kch].argmax(dim=1)         # [B]
        _zone = torch.cat([_ksq.unsqueeze(1), self.kf_nbr[_ksq]], dim=1)   # [B, 9]
        _zi = _zone.reshape(B, 1, 9, 1).expand(-1, 32, -1, C)
        _kz_feats.append(torch.gather(e_rows, 2, _zi).sum(dim=2))          # [B, 32, C]
      x = x + self.kz_proj(torch.cat(_kz_feats, dim=-1).to(x.dtype))

    if self.king_flight:
      # v1 (parked): king-token flight-zone summary. Sum in the source dtype
      # and cast the SMALL result — a .float() before the reduce materializes
      # the whole [B,64,64,C] tensor in fp32 (~7.6 % eager cost measured
      # 2026-08-20).
      _indeg_all = vis_E.sum(dim=1).float()                # [B, 64, C] who-covers-square
      for _kch in (6, 12):
        _ksq = squares13[:, :, _kch].argmax(dim=1)         # [B]
        _knb = self.kf_nbr[_ksq]                           # [B, 8]
        _deg_n = torch.gather(_indeg_all, 1,
                              _knb.unsqueeze(-1).expand(-1, -1, _indeg_all.shape[-1]))
        _occ_n = torch.gather(squares13, 1, _knb.unsqueeze(-1).expand(-1, -1, 13))
        _onb_n = self.kf_onb[_ksq].unsqueeze(-1)           # [B, 8, 1]
        _kf = torch.cat([_deg_n.to(x.dtype), _occ_n.to(x.dtype), _onb_n.to(x.dtype)],
                        dim=-1).reshape(B, -1)             # [B, 8*(C+14)]
        _kmask = (sel == _ksq.unsqueeze(1)).to(x.dtype).unsqueeze(-1)   # [B, 32, 1]
        x = x + _kmask * self.kf_proj(_kf).unsqueeze(1)

    # NB (boelge 9): interleave_cross er inkompatibel med fase-splitten
    # (cross-read trenger s_flow inne i blokk-loekka) — avvises hoeyt i
    # ceres_net naar kant->trunk er paa. I fase-splitt-modus kjoeres
    # cross-read kun i finish().
    for blk in self.blocks:
      x, rel_pair = blk(x, rel_pair, pad_bias)
      if self.interleave_cross:
        # Review-funn (boelge 9): s_flow som ARG, aldri modul-attributt — en
        # graf-tensor paa modulen drepte deepcopy-eksporten for interleave-armer.
        assert interleave_s_flow is not None, 'interleave_cross krever s_flow inn i run_pblocks'
        x = self._cross_read(x, interleave_s_flow)
    return x, rel_pair, sel, slot_occ

  def _cross_read(self, xp, s_flow):
    qx = self.x_q(self.x_ln_p(xp))
    s_n = self.x_ln_s(s_flow)
    kx, vx = self.x_k(s_n), self.x_v(s_n)
    a = torch.softmax(torch.matmul(qx, kx.transpose(1, 2)) * (self.dp ** -0.5), dim=-1)
    return xp + self.x_out(torch.matmul(a, vx))

  def finish(self, x, s_flow, sel, slot_occ):
    """Fase 2: cross-read mot trunk-flow + pools."""
    if not self.interleave_cross:
      x = self._cross_read(x, s_flow)
    x = self.out_ln(x)
    # Masked pools over REAL pieces only.
    w = slot_occ.unsqueeze(-1)                           # [B, 32, 1]
    mean_pool = (x * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
    # Masked soft-min over pieces (tau=1): empty slots pushed to +inf side.
    xm = x.float() + (1.0 - w.float()) * 1e4
    smin = -torch.logsumexp(-xm, dim=1) + torch.log(w.float().sum(dim=1).clamp_min(1.0))
    pooled = torch.cat([mean_pool, smin.to(x.dtype)], dim=-1)  # [B, 2*dp]
    # Also expose the raw tokens + slot bookkeeping for decode-side consumers
    # (Stage A3 mover-bilinear reads the mover's token by from-square).
    return pooled, x, sel, slot_occ
