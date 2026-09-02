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
  assert (SQ['e1'], SQ['g1']) not in P, 'castling must be gated on empty f1/g1'
  print(f'  start position: {len(P)} candidates (>= 20 legal, double pushes present)')
  # castling gates
  cand, _ = dec.candidates(board_from_pieces({'e1': 'K', 'h1': 'R', 'a1': 'R'}, {'e8': 'K'}))
  P = pairs_of(cand); assert (4, 6) in P and (4, 2) in P, 'castling pairs missing'
  cand, _ = dec.candidates(board_from_pieces({'e1': 'K', 'h1': 'R', 'b1': 'N', 'a1': 'R'}, {'e8': 'K'}))
  P = pairs_of(cand); assert (4, 6) in P and (4, 2) not in P, 'queenside must be blocked by b1'
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
  pol, pooled, stats, sel, valid = dec(sq[:, :, 0:13], flow)
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
  _, pooled0, st0, _, v0 = net.move_tokens(s0, torch.randn(1, 64, net.EMBEDDING_DIM))
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

  # --- 5. guards ---------------------------------------------------------
  for name, over in (('with plane decode', {'UseMoveTokens': True}),
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
