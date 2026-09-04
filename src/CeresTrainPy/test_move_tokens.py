"""Contract tests for the move-token decoder (design B, move_tokens.py).

    python test_move_tokens.py            (from src/CeresTrainPy, CPU, seconds)

1. Candidate builder is a SUPERSET of legal moves: start position (20 legal incl. 8
   double pushes), castling gating, promotions, en passant; if python-chess is
   installed, 150 random positions are checked against board.legal_moves (from-to pairs).
2. Scatter contract: every 1858 index whose (from,to,slot) token exists gets that token's
   logit; absent moves sit at the -30 floor (+ per-move bias); promotion slots differ.
3. Full net: forward/backward finite, gradient reaches the decoder, MLP policy head gets
   no gradient, eval forward finite, stash consumed by compute_loss and diagnostics logged.
4. Bit-pairing: with UseMoveTokens on, every pre-existing parameter is identical to the
   control built with the same TorchSeed.
5. Guards: UseMoveTokens with DualPlanePolicyDecode / PolicyHeadForm=fromto refused.
"""
import os, sys
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_dual_plane_edge_aux import build, random_boards, fake_batch, run_loss
from move_tokens import MoveTokenDecoder, MT_FLOOR
from lc0_moves_1858 import MOVES_1858, FROM_1858, TO_1858

SQ = {f + r: (ord(r) - ord('1')) * 8 + (ord(f) - ord('a')) for f in 'abcdefgh' for r in '12345678'}


def board_from_pieces(own, opp):
  """own/opp: dict square->piece letter (P N B R Q K), stm-relative. Returns [1,64,13]."""
  s = torch.zeros(1, 64, 13); s[0, :, 0] = 1.0
  idx = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6}
  for sq, p in own.items():
    s[0, SQ[sq], 0] = 0; s[0, SQ[sq], idx[p]] = 1.0
  for sq, p in opp.items():
    s[0, SQ[sq], 0] = 0; s[0, SQ[sq], idx[p] + 6] = 1.0
  return s


def pairs_of(cand):
  return {(int(i) // 64, int(i) % 64) for i in torch.nonzero(cand[0] > 0.5).reshape(-1)}


def main():
  dec = MoveTokenDecoder(s_dim=32, norm_type='RMSNorm', dm=32, layers=1, heads=2, max_tokens=64)
  # --- 1. candidate superset ---------------------------------------------
  start_own = {**{f + '2': 'P' for f in 'abcdefgh'}, 'a1': 'R', 'h1': 'R', 'b1': 'N', 'g1': 'N',
               'c1': 'B', 'f1': 'B', 'd1': 'Q', 'e1': 'K'}
  start_opp = {**{f + '7': 'P' for f in 'abcdefgh'}, 'a8': 'R', 'h8': 'R', 'b8': 'N', 'g8': 'N',
               'c8': 'B', 'f8': 'B', 'd8': 'Q', 'e8': 'K'}
  cand, _ = dec.candidates(board_from_pieces(start_own, start_opp))
  P = pairs_of(cand)
  legal_start = {(SQ[f + '2'], SQ[f + '3']) for f in 'abcdefgh'} | {(SQ[f + '2'], SQ[f + '4']) for f in 'abcdefgh'} \
      | {(SQ['b1'], SQ['a3']), (SQ['b1'], SQ['c3']), (SQ['g1'], SQ['f3']), (SQ['g1'], SQ['h3'])}
  assert legal_start <= P, f'start position missing {legal_start - P}'
  # castling = king-takes-rook in the TPG encoding (e1h1 / e1a1): both pairs present at start
  assert (SQ['e1'], SQ['h1']) in P and (SQ['e1'], SQ['a1']) in P, 'king-takes-rook castling pairs missing'
  print(f'  start position: {len(P)} candidates (>= 20 legal, double pushes + king-takes-rook present)')
  # castling pairs follow the king's rank (FRC-compatible); rooks off the rank are not added
  cand, _ = dec.candidates(board_from_pieces({'e1': 'K', 'h1': 'R', 'a1': 'R', 'a5': 'R'}, {'e8': 'K'}))
  P = pairs_of(cand); assert (4, 7) in P and (4, 0) in P and (4, SQ['a5']) not in P, 'castling pairs'
  # promotion + en passant geometry (pawn on 7th: push and capture diagonals present)
  cand, _ = dec.candidates(board_from_pieces({'e1': 'K', 'b7': 'P'}, {'e8': 'K', 'a8': 'R', 'c8': 'B'}))
  P = pairs_of(cand); assert {(SQ['b7'], SQ['b8']), (SQ['b7'], SQ['a8']), (SQ['b7'], SQ['c8'])} <= P
  assert (SQ['b7'], SQ['b8']) in P
  # own-target mask
  cand, _ = dec.candidates(board_from_pieces({'e1': 'K', 'a1': 'R', 'a4': 'P'}, {'e8': 'K'}))
  P = pairs_of(cand); assert (SQ['a1'], SQ['a4']) not in P and (SQ['a1'], SQ['a3']) in P
  print('  castling / promotion / en-passant geometry / own-target mask OK')
  try:
    import chess, random
    random.seed(3); miss = 0; tot = 0; n_cnt = []
    for _ in range(150):
      b = chess.Board()
      for _ in range(random.randint(0, 60)):
        mv = list(b.legal_moves)
        if not mv: break
        b.push(random.choice(mv))
      if b.is_game_over(): continue
      # stm-relative encoding: mirror when black to move
      own, opp = {}, {}
      for sq, pc in b.piece_map().items():
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if b.turn == chess.BLACK: r = 7 - r
        name = 'abcdefgh'[f] + str(r + 1)
        (own if pc.color == b.turn else opp)[name] = pc.symbol().upper()
      cand, _ = dec.candidates(board_from_pieces(own, opp)); P = pairs_of(cand); n_cnt.append(len(P))
      for mv in b.legal_moves:
        fr, to = mv.from_square, mv.to_square
        if b.is_castling(mv):                   # TPG encodes castling as king-takes-rook
          to = chess.H1 if chess.square_file(to) > chess.square_file(fr) else chess.A1
          if b.turn == chess.BLACK: to = chess.square(chess.square_file(to), 7)
        if b.turn == chess.BLACK:
          fr = chess.square(chess.square_file(fr), 7 - chess.square_rank(fr))
          to = chess.square(chess.square_file(to), 7 - chess.square_rank(to))
        tot += 1
        if (fr, to) not in P: miss += 1
    print(f'  python-chess: {tot} legal moves over random positions, missing {miss} '
          f'({100.0 * miss / max(1, tot):.2f} %), candidates mean {sum(n_cnt)/len(n_cnt):.1f} max {max(n_cnt)}')
    assert miss == 0, 'candidate set must be a superset of legal moves'
  except ImportError:
    print('  (python-chess not installed: random-position superset check skipped)')

  # --- 2. scatter contract ----------------------------------------------
  B = 2
  sq = random_boards(B)
  flow = torch.randn(B, 64, 32)
  dec.pol.weight.data.normal_()                       # make token logits non-trivial
  pol, pooled, stats, sel, valid, _ = dec(sq[:, :, 0:13], flow)
  assert pol.shape == (B, 1858) and pooled.shape == (B, 64)
  # recompute reference: for each move index, token logit if its pair is selected+valid
  for b in range(B):
    for m in range(0, 1858, 37):
      pf = FROM_1858[m] * 64 + TO_1858[m]
      hits = [j for j in range(sel.shape[1]) if int(sel[b, j]) == pf and bool(valid[b, j])]
      if hits:
        assert not torch.isclose(pol[b, m] - dec.mt_pol_bias[m], torch.tensor(MT_FLOOR)), 'selected move at floor'
      else:
        assert torch.isclose(pol[b, m] - dec.mt_pol_bias[m], torch.tensor(MT_FLOOR)), 'absent move must be at floor'
  # promotion slots: a7a8 (bare), a7a8q, a7a8r, a7a8b must be able to differ
  s2 = board_from_pieces({'e1': 'K', 'a7': 'P'}, {'e8': 'K'})
  pol2, *_ = dec(s2, torch.randn(1, 64, 32))
  idx = [MOVES_1858.index(x) for x in ('a7a8', 'a7a8q', 'a7a8r', 'a7a8b')]
  vals = [float(pol2[0, i] - dec.mt_pol_bias[i]) for i in idx]
  assert all(v > MT_FLOOR + 1 for v in vals) and len(set(round(v, 5) for v in vals)) == 4, vals
  print('  scatter contract OK (selected moves carry token logits, absent at floor, 4 promotion slots distinct)')

  # --- 3./4. full net ----------------------------------------------------
  ctrl, _ = build({'DualPlanePolicyDecode': False}, {}, 'mtc')
  net, _ = build({'DualPlanePolicyDecode': False, 'UseMoveTokens': True, 'MoveTokenDim': 64,
                  'MoveTokenLayers': 2, 'MoveTokenHeads': 2, 'MoveTokenMax': 64}, {}, 'mt')
  sd_c, sd_n = ctrl.state_dict(), net.state_dict()
  for k in sd_c:
    assert k in sd_n and torch.equal(sd_c[k], sd_n[k]), f'bit-pairing broken at {k}'
  print(f'  bit-pairing OK ({len(set(sd_n) - set(sd_c))} new tensors, all others identical)')
  net.train(); net.zero_grad(set_to_none=True)
  sq = random_boards(4); batch = fake_batch(4, sq)
  loss = run_loss(net, batch, sq); assert torch.isfinite(loss); loss.backward()
  assert net._last_mt is None, 'stash must be consumed'
  dead = [n for n, p in net.move_tokens.named_parameters() if p.grad is None]
  assert not dead, f'move_tokens params without gradient (DDP would abort): {dead}'
  assert net.move_tokens.w_in.weight.grad.abs().sum() > 0
  assert net.policy_head.fcFinal.weight.grad is None, 'MLP policy head must be dead'
  from wd_partition import partition_weight_decay
  partition_weight_decay(net)                       # asserts completeness (the dp_eaux_ lesson)
  # no-candidate board (lone kings, stm king boxed by own pawns): pooled must stay finite/zero
  s0 = board_from_pieces({'a1': 'K', 'a2': 'P', 'b2': 'P', 'b1': 'P'}, {'h8': 'K'})
  s0[0, SQ['b2'], 0] = 0
  _, pooled0, st0, _, v0, _ = net.move_tokens(s0, torch.randn(1, 64, net.EMBEDDING_DIM))
  assert torch.isfinite(pooled0).all() and float(pooled0.abs().max()) < 1e3, 'empty-candidate pool guard'
  net.eval()
  with torch.no_grad(): out = net(sq, None)
  assert torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all()
  print(f'  full net OK: loss {float(loss.detach()):.4f}, all decoder params get grad, MLP head dead, '
        f'wd partition complete, empty-candidate pool guarded, eval finite')
  # ONNX export + ORT parity when the dynamo exporter is available (WSL env has it)
  try:
    import onnx_ir, onnxruntime as ort, numpy as np, tempfile
    path = os.path.join(tempfile.gettempdir(), 'mt_export_test.onnx')
    with torch.no_grad(): ref = net(sq, None)
    torch.onnx.export(net, (sq, None), path, dynamo=True, opset_version=18, input_names=['squares', 'prior'])
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    o = s.run(None, {s.get_inputs()[0].name: sq.numpy()})
    pr = ref[0].numpy(); po = [x for x in o if x.shape == pr.shape][0]
    d = float(np.abs(po - pr).max()); assert d < 1e-3, d
    print(f'  ONNX export + ORT parity OK (policy max|d| {d:.2e})')
  except ImportError:
    print('  (onnx_ir not installed here: export parity check skipped — run under the WSL env)')

  # --- 4b. X-program knobs: rich features + policy-weighted pool + aux MLP CE ---
  net2, _ = build({'DualPlanePolicyDecode': False, 'UseMoveTokens': True, 'MoveTokenDim': 64,
                   'MoveTokenLayers': 2, 'MoveTokenHeads': 2, 'MoveTokenMax': 64,
                   'MoveTokenRichFeatures': True, 'MoveTokenValuePool': 'both'},
                  {'LossMoveTokenAuxMLPMultiplier': 0.3}, 'mt2')
  assert net2.move_tokens.w_in.in_features == 2 * net2.EMBEDDING_DIM + 4 + 17
  assert net2.move_tokens.v_inject.in_features == 3 * 64
  # rich features on a known board: e2 pawn -> e4 is a pawn move, no capture, not promo
  s1 = board_from_pieces({'e1': 'K', 'e2': 'P', 'd1': 'Q'}, {'e8': 'K', 'e5': 'R', 'd4': 'N'})
  cand1, E1 = net2.move_tokens.candidates(s1)
  fr1 = torch.tensor([[SQ['e2'], SQ['d1']]]); to1 = torch.tensor([[SQ['e4'], SQ['d4']]])
  r1 = net2.move_tokens.rich(s1, E1, fr1, to1)
  assert r1.shape == (1, 2, 17)
  assert r1[0, 0, 0] == 1 and r1[0, 0, 1:6].sum() == 0, 'mover one-hot: pawn'
  assert r1[0, 0, 6:12].sum() == 0, 'e2e4 is not a capture'
  assert r1[0, 1, 4] == 1, 'mover one-hot: queen'
  assert r1[0, 1, 6 + 1] == 1, 'd1xd4 captures a knight (opp one-hot index 1)'
  assert r1[0, 0, 12] == 0, 'no promotion flag on e2e4'
  assert float(r1[0, 0, 13]) * 4 == 1.0, 'e4 is attacked by the e5 rook (opp attackers of to = 1)'
  net2.train(); net2.zero_grad(set_to_none=True)
  loss2 = run_loss(net2, batch, sq); assert torch.isfinite(loss2); loss2.backward()
  assert net2._last_mt is None and net2._last_mt_aux_out is None, 'stashes must be consumed'
  dead2 = [n for n, p in net2.move_tokens.named_parameters() if p.grad is None]
  assert not dead2, f'move_tokens params without gradient: {dead2}'
  g_aux = net2.policy_head.fcFinal.weight.grad
  assert g_aux is not None and g_aux.abs().sum() > 0, 'aux MLP policy CE must train the MLP head'
  from wd_partition import partition_weight_decay as _pwd
  _pwd(net2)
  _, pooled2, st2, _, _, _ = net2.move_tokens(s0, torch.randn(1, 64, net2.EMBEDDING_DIM))
  assert pooled2.shape[-1] == 3 * 64 and torch.isfinite(pooled2).all() and float(pooled2.abs().max()) < 1e3
  assert 'mt_polpool_entropy' in st2
  net2.eval()
  with torch.no_grad(): out2 = net2(sq, None)
  assert torch.isfinite(out2[0]).all() and torch.isfinite(out2[1]).all()
  print(f'  X-program knobs OK: rich features (17, checked on a known board), value pool both (3dm), '
        f'aux MLP CE trains the MLP head, loss {float(loss2.detach()):.4f}')
  try:
    import onnx_ir, onnxruntime as ort, numpy as np, tempfile
    path2 = os.path.join(tempfile.gettempdir(), 'mt_export_test2.onnx')
    with torch.no_grad(): ref2 = net2(sq, None)
    torch.onnx.export(net2, (sq, None), path2, dynamo=True, opset_version=18, input_names=['squares', 'prior'])
    s2 = ort.InferenceSession(path2, providers=['CPUExecutionProvider'])
    o2 = s2.run(None, {s2.get_inputs()[0].name: sq.numpy()})
    pr2 = ref2[0].numpy(); po2 = [x for x in o2 if x.shape == pr2.shape][0]
    d2 = float(np.abs(po2 - pr2).max()); assert d2 < 1e-3, d2
    print(f'  ONNX export + ORT parity OK for the knob variant (policy max|d| {d2:.2e})')
  except ImportError:
    print('  (onnx_ir not installed here: knob-variant export parity skipped)')

  # --- 4c. value-order head (2026-09-04 ideation T1-4): training-only per-token scalar ---
  net3, _ = build({'DualPlanePolicyDecode': False, 'UseMoveTokens': True, 'MoveTokenDim': 64,
                   'MoveTokenLayers': 2, 'MoveTokenHeads': 2, 'MoveTokenMax': 64},
                  {'LossMoveTokenValueOrderMultiplier': 0.5, 'MoveTokenValueOrderTopK': 3}, 'mt3')
  assert net3.move_tokens.value_order and tuple(net3.move_tokens.vord.weight.shape) == (1, 64)
  sd3 = net3.state_dict()
  extra3 = set(sd3) - set(sd_n)
  assert extra3 == {'move_tokens.vord.weight'}, extra3
  for k in sd_n:
    assert torch.equal(sd_n[k], sd3[k]), f'bit-pairing broken at {k}'
  net3.train(); net3.zero_grad(set_to_none=True)
  loss3 = run_loss(net3, batch, sq); assert torch.isfinite(loss3); loss3.backward()
  assert net3.move_tokens._last_vord is None and net3._last_mt is None, 'stashes must be consumed'
  g3 = net3.move_tokens.vord.weight.grad
  assert g3 is not None and g3.abs().sum() > 0, 'value-order head must receive gradient'
  dead3 = [n for n, p in net3.move_tokens.named_parameters() if p.grad is None]
  assert not dead3, f'move_tokens params without gradient: {dead3}'
  _pwd(net3)
  # loss contract: tokens in the target's order score lower than the reversed order; no-target row = 0
  from move_tokens import move_token_value_order_loss as _vol
  mv = net3.move_tokens.mv_pair_flat
  m_idx = [0, 1, 2]                                   # three 1858-moves with distinct from-to pairs
  pairs = [int(mv[m]) for m in m_idx]
  assert len(set(pairs)) == 3
  other = next(pp for pp in range(4096) if pp not in pairs)
  sel3 = torch.tensor([pairs + [other], pairs + [other]])
  valid3 = torch.ones(2, 4, dtype=torch.bool)
  tgt = torch.zeros(2, 1858); tgt[0, m_idx[0]] = 0.6; tgt[0, m_idx[1]] = 0.3; tgt[0, m_idx[2]] = 0.1
  good = torch.tensor([[3., 2., 1., 0.], [0., 0., 0., 0.]])
  bad = torch.tensor([[0., 1., 2., 3.], [0., 0., 0., 0.]])
  lg, dg = _vol(good, sel3, valid3, tgt, mv, 3)
  lb, db = _vol(bad, sel3, valid3, tgt, mv, 3)
  assert torch.isfinite(lg) and torch.isfinite(lb) and float(lg) < float(lb), (float(lg), float(lb))
  assert float(dg['mt_vord_top1']) == 1.0 and float(db['mt_vord_top1']) == 0.0
  assert abs(float(dg['mt_vord_rows_with_target']) - 0.5) < 1e-6, 'row 2 has no target mass -> excluded'
  net3.eval()
  with torch.no_grad(): out3 = net3(sq, None)
  assert net3.move_tokens._last_vord is None, 'no stash in eval (export path)'
  assert torch.isfinite(out3[0]).all()
  print(f'  value-order head OK: 1 new tensor, grads flow, ListMLE contract (good {float(lg):.3f} < bad {float(lb):.3f}), '
        f'eval graph untouched')

  # --- 4d. opponent-reply keys + square write-back (2026-09-04 ideation T1-2 / T1-3) ---
  base_over = {'DualPlanePolicyDecode': False, 'UseMoveTokens': True, 'MoveTokenDim': 64,
               'MoveTokenLayers': 2, 'MoveTokenHeads': 2, 'MoveTokenMax': 64}
  # (i) write-back alone is an exact step-0 no-op on every head (zero-init wo)
  net_wb, _ = build(dict(base_over, MoveTokenWriteBack=True), {}, 'mtwb')
  sd_wb = net_wb.state_dict()
  extra_wb = set(sd_wb) - set(sd_n)
  assert extra_wb == {'move_tokens.wb.ln_q.scale', 'move_tokens.wb.wq.weight', 'move_tokens.wb.wkv.weight',
                      'move_tokens.wb.wo.weight'} or all(k.startswith('move_tokens.wb.') for k in extra_wb), extra_wb
  for k in sd_n:
    assert torch.equal(sd_n[k], sd_wb[k]), f'bit-pairing broken at {k}'
  net.eval(); net_wb.eval()
  with torch.no_grad():
    o_ref = net(sq, None); o_wb = net_wb(sq, None)
  for i, (a, b) in enumerate(zip(o_ref, o_wb)):
    if a is not None and b is not None:
      assert torch.equal(a, b), f'write-back must be an exact step-0 no-op (output {i})'
  net_wb.train(); net_wb.zero_grad(set_to_none=True)
  loss_wb = run_loss(net_wb, batch, sq); assert torch.isfinite(loss_wb); loss_wb.backward()
  assert net_wb.move_tokens.wb.wo.weight.grad is not None and net_wb.move_tokens.wb.wo.weight.grad.abs().sum() > 0, \
      'zero-init wo must still receive gradient (value heads read the write-back)'
  dead_wb = [n for n, p in net_wb.move_tokens.named_parameters() if p.grad is None]
  assert not dead_wb, f'move_tokens params without gradient: {dead_wb}'
  _pwd(net_wb)
  # empty-candidate board: write-back must be exactly zero
  net_wb.eval()
  with torch.no_grad():
    *_, wb0 = net_wb.move_tokens(s0, torch.randn(1, 64, net_wb.EMBEDDING_DIM))
  assert wb0 is not None and float(wb0.abs().max()) == 0.0, 'no-candidate board -> zero write-back'
  print('  square write-back OK: exact step-0 no-op on all heads, grads flow into wo, zero on empty boards')
  # (ii) opponent keys: candidate superset vs python-chess with the side to move flipped
  net_o, _ = build(dict(base_over, MoveTokenOppMax=48, MoveTokenOppPool=True), {}, 'mto')
  assert net_o.move_tokens.M_opp == 48 and net_o.move_tokens.v_inject.in_features == 4 * 64
  try:
    import chess, random
    random.seed(5); tot = miss = 0
    for _ in range(120):
      b = chess.Board()
      for _ in range(random.randint(0, 60)):
        mv = list(b.legal_moves)
        if not mv: break
        b.push(random.choice(mv))
      if b.is_game_over(): continue
      own, opp = {}, {}
      for sq_, pc in b.piece_map().items():
        f, r = chess.square_file(sq_), chess.square_rank(sq_)
        if b.turn == chess.BLACK: r = 7 - r
        (own if pc.color == b.turn else opp)['abcdefgh'[f] + str(r + 1)] = pc.symbol().upper()
      s13 = board_from_pieces(own, opp)
      _, E = net_o.move_tokens.candidates(s13)
      cand_o = net_o.move_tokens.candidates_opp(s13, E)[0]
      b2 = b.copy(); b2.turn = not b.turn
      for mv in b2.pseudo_legal_moves:
        fr, to = mv.from_square, mv.to_square
        if b2.is_castling(mv):                  # king-takes-rook on the OPPONENT's back rank
          to = chess.square(7 if chess.square_file(to) > chess.square_file(fr) else 0,
                            0 if b2.turn == chess.WHITE else 7)
        if b.turn == chess.BLACK:
          fr = chess.square(chess.square_file(fr), 7 - chess.square_rank(fr))
          to = chess.square(chess.square_file(to), 7 - chess.square_rank(to))
        tot += 1
        if float(cand_o[fr * 64 + to]) < 0.5: miss += 1
    print(f'  opponent candidates: {tot} pseudo-legal replies over random positions, missing {miss}')
    assert miss == 0, 'opponent candidate set must be a superset of the opponent pseudo-legal replies'
  except ImportError:
    print('  (python-chess not installed: opponent superset check skipped)')
  net_o.train(); net_o.zero_grad(set_to_none=True)
  loss_o = run_loss(net_o, batch, sq); assert torch.isfinite(loss_o); loss_o.backward()
  dead_o = [n for n, p in net_o.move_tokens.named_parameters() if p.grad is None]
  assert not dead_o, f'move_tokens params without gradient: {dead_o}'
  assert net_o.move_tokens.opp_side.grad is not None and net_o.move_tokens.opp_side.grad.abs().sum() > 0
  _pwd(net_o)
  net_o.eval()
  with torch.no_grad():
    _, pooled_o, st_o, _, _, _ = net_o.move_tokens(s0, torch.randn(1, 64, net_o.EMBEDDING_DIM))
    out_o = net_o(sq, None)
  assert pooled_o.shape[-1] == 4 * 64 and torch.isfinite(pooled_o).all() and float(pooled_o.abs().max()) < 1e3
  assert 'mt_opp_count_mean' in st_o and torch.isfinite(out_o[0]).all() and torch.isfinite(out_o[1]).all()
  # logit monitor: present in training stats, absent in eval
  net_o.train()
  _, _, st_tr, _, _, _ = net_o.move_tokens(sq[:, :, 0:13].float(), torch.randn(sq.shape[0], 64, net_o.EMBEDDING_DIM))
  assert 'mt_qk_max_self' in st_tr and 'mt_qk_max_cross' in st_tr and torch.isfinite(st_tr['mt_qk_max_self'])
  assert 'mt_qk_max_self' not in st_o
  print(f'  opponent keys OK: {net_o.move_tokens.M_opp} opp tokens as extra K/V, pool 4dm, grads flow, empty-board guard, '
        f'logit monitor self {float(st_tr["mt_qk_max_self"]):.2f} / cross {float(st_tr["mt_qk_max_cross"]):.2f}')
  try:
    import onnx_ir, onnxruntime as ort, numpy as np, tempfile
    for tag, nn_ in (('wb', net_wb), ('opp', net_o)):
      nn_.eval()
      pth = os.path.join(tempfile.gettempdir(), f'mt_export_test_{tag}.onnx')
      with torch.no_grad(): ref_ = nn_(sq, None)
      torch.onnx.export(nn_, (sq, None), pth, dynamo=True, opset_version=18, input_names=['squares', 'prior'])
      ss = ort.InferenceSession(pth, providers=['CPUExecutionProvider'])
      oo = ss.run(None, {ss.get_inputs()[0].name: sq.numpy()})
      pr_ = ref_[0].numpy(); po_ = [x for x in oo if x.shape == pr_.shape][0]
      dd = float(np.abs(po_ - pr_).max()); assert dd < 1e-3, dd
      print(f'  ONNX export + ORT parity OK for {tag} (policy max|d| {dd:.2e})')
  except ImportError:
    print('  (onnx_ir not installed here: opp/write-back export parity skipped)')

  # --- 4e. post-move attention + learned value query: grads, step-0 no-op, fused identity, ONNX parity
  net4, _ = build({'DualPlanePolicyDecode': False, 'UseMoveTokens': True, 'MoveTokenDim': 64,
                   'MoveTokenLayers': 2, 'MoveTokenHeads': 2, 'MoveTokenMax': 64,
                   'MoveTokenPostMove': True, 'MoveTokenValueQuery': True}, {}, 'mt4')
  assert net4.move_tokens.v_inject.in_features == 3 * 64, 'value query widens the pool by dm'
  # step-0: pm_proj is zero => post-move attention is an exact no-op vs the same net without it
  net4b, _ = build({'DualPlanePolicyDecode': False, 'UseMoveTokens': True, 'MoveTokenDim': 64,
                    'MoveTokenLayers': 2, 'MoveTokenHeads': 2, 'MoveTokenMax': 64,
                    'MoveTokenValueQuery': True}, {}, 'mt4b')
  net4.eval(); net4b.eval()
  with torch.no_grad():
    sd4 = net4.state_dict()
    net4b.load_state_dict({k: v for k, v in sd4.items() if k in net4b.state_dict()}, strict=True)
    net4.move_tokens.export_fused = False; net4b.move_tokens.export_fused = False
    o4 = net4(sq, None); o4b = net4b(sq, None)
  d0 = float((o4[0] - o4b[0]).abs().max())
  assert d0 < 1e-5, f'post-move attention must be a step-0 no-op (zero pm_proj): {d0}'
  net4.train(); net4.zero_grad(set_to_none=True)
  loss4 = run_loss(net4, batch, sq); assert torch.isfinite(loss4); loss4.backward()
  dead4 = [n for n, p in net4.move_tokens.named_parameters() if p.grad is None]
  assert not dead4, f'params without gradient: {dead4}'
  # zero-init pm_proj => inner post-move params get zero grad at step 0 (the projection trains first,
  # the v_inject pattern); what must hold is: grads EXIST (DDP) and pm_proj itself gets signal.
  assert net4.move_tokens.blocks[0].pm_dk.grad is not None
  assert net4.move_tokens.blocks[0].pm_proj.weight.grad.abs().sum() > 0, 'pm_proj must get gradient at step 0'
  assert net4.move_tokens.vq_block.vq.grad is not None
  from wd_partition import partition_weight_decay as _pwd4
  _pwd4(net4)
  net4.eval()
  with torch.no_grad():
    for _i, _blk in enumerate(net4.move_tokens.blocks):
      _blk.ln_s.scale.copy_(torch.rand(_blk.ln_s.scale.shape, generator=torch.Generator().manual_seed(21 + _i)) + 0.5)
      _blk.pm_proj.weight.normal_(0, 0.05, generator=torch.Generator().manual_seed(31 + _i))
    net4.move_tokens.vq_block.ln_s.scale.copy_(torch.rand(net4.move_tokens.vq_block.ln_s.scale.shape, generator=torch.Generator().manual_seed(41)) + 0.5)
    net4.move_tokens.export_fused = False; ref4 = net4(sq, None)
    net4.move_tokens.export_fused = True; assert net4.move_tokens._fusable(); fus4 = net4(sq, None)
  d4p = float((fus4[0] - ref4[0]).abs().max()); d4v = float((fus4[1] - ref4[1]).abs().max())
  assert d4p < 1e-4 and d4v < 1e-4, (d4p, d4v)
  _, pooled4, _, _, _, _ = net4.move_tokens(s0, torch.randn(1, 64, net4.EMBEDDING_DIM))
  assert torch.isfinite(pooled4).all() and float(pooled4.abs().max()) < 1e3
  print(f'  post-move + value-query OK: step-0 no-op (|d| {d0:.1e}), all params get grad, fused identity '
        f'(policy {d4p:.1e}, value {d4v:.1e}), empty-board guard')
  try:
    import onnx_ir, onnxruntime as ort, numpy as np, tempfile
    path4 = os.path.join(tempfile.gettempdir(), 'mt_export_test4.onnx')
    with torch.no_grad(): ref4o = net4(sq, None)
    torch.onnx.export(net4, (sq, None), path4, dynamo=True, opset_version=18, input_names=['squares', 'prior'])
    s4 = ort.InferenceSession(path4, providers=['CPUExecutionProvider'])
    o4o = s4.run(None, {s4.get_inputs()[0].name: sq.numpy()})
    pr4 = ref4o[0].numpy(); po4 = [x for x in o4o if x.shape == pr4.shape][0]
    d4 = float(np.abs(po4 - pr4).max()); assert d4 < 1e-3, d4
    print(f'  ONNX export + ORT parity OK for post-move + value-query (policy max|d| {d4:.2e})')
  except ImportError:
    print('  (onnx_ir not installed here: post-move/value-query export parity skipped)')

  # --- 4d. export-fused eval path == unfused (shared square norm + single K/V GEMM, one-gather assembly)
  net.eval()
  with torch.no_grad():
    for _i, _blk in enumerate(net.move_tokens.blocks):     # non-trivial norm scales so the fold is exercised
      _blk.ln_s.scale.copy_(torch.rand(_blk.ln_s.scale.shape, generator=torch.Generator().manual_seed(11 + _i)) + 0.5)
    net.move_tokens.export_fused = False
    ref_u = net(sq, None)
    net.move_tokens.export_fused = True
    assert net.move_tokens._fusable()
    out_f = net(sq, None)
  dpol = float((out_f[0] - ref_u[0]).abs().max()); dval = float((out_f[1] - ref_u[1]).abs().max())
  assert dpol < 1e-4 and dval < 1e-4, (dpol, dval)
  print(f'  export-fused path OK: identical to unfused (policy max|d| {dpol:.1e}, value {dval:.1e})')

  # --- 4c. export-time token cap: equivariance => identical logits when all fit ---
  net.eval()
  with torch.no_grad():
    cnt = net.move_tokens.candidates(sq[:, :, 0:13])[0].sum(dim=1)
    fit = cnt <= 56
    assert int(fit.sum()) >= 2, f'need >= 2 boards within the cap: {cnt.tolist()}'
    sq_fit = sq[fit]
    ref_full = net(sq_fit, None)[0].clone()
    old_M = net.move_tokens.set_export_max(56)
    cap = net(sq_fit, None)[0]
    net.move_tokens.M = old_M
  dmax = float((cap - ref_full).abs().max())
  assert dmax < 1e-4, f'export cap changed logits: {dmax}'
  print(f'  export-time cap OK: M 64 -> 56 gives identical policy (max|d| {dmax:.1e}) on {int(fit.sum())} boards '
        f'with <= 56 candidates (counts {cnt.tolist()})')

  # --- 5. guards ---------------------------------------------------------
  for name, over in (('with plane decode', {'UseMoveTokens': True}),
                     ('rich w/o move tokens', {'DualPlanePolicyDecode': False, 'MoveTokenRichFeatures': True}),
                     ('bad pool name', {'DualPlanePolicyDecode': False, 'UseMoveTokens': True, 'MoveTokenValuePool': 'max'}),
                     ('with fromto', {'DualPlanePolicyDecode': False, 'UseMoveTokens': True})):
    try:
      if name == 'with fromto':
        os.environ['CERES_POLICY_HEAD_FORM'] = 'fromto'
      build(over, {}, 'rej'); raise SystemExit(f'FAIL: {name} not rejected')
    except ValueError as e:
      print(f'  rejection OK ({name}): {str(e)[:60]}')
    finally:
      os.environ.pop('CERES_POLICY_HEAD_FORM', None)
  print('ALL OK')


if __name__ == '__main__':
  main()
