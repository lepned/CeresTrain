"""Unit tests of the Fry building blocks (config.py "Fry building blocks"; FRY_BLOCKS_NOTES.md). Small nets, CPU.

  (a) SquareFeatures: the vectorised row computation == a per-square reference on game positions of both colours;
      codes from our record; zero table = identity; live table changes the function; gradients; as an add-on to the
      Linear embedding and to the BT6-form embedding of the 384 base arm's layout
  (b) grouped-query attention with its own q/k width: == a direct einsum reference (query head h reads key / value
      head h // group); == our plain attention when the grouping and widths are the plain ones; with templates; per-layer
      widths; gradients; refusals
  (c) FryInput: tokens from our record == the board encoder (colour and non-legal e.p. excepted, by design); 70
      tokens through the trunk, template bias 0 on metadata rows / columns; the heads read 64 squares; with our
      move-token decoder; refusals
  (d) heads: PolicyHeadFry on our 1858 moves == Fry's 1968-move formula for every legal move (promotions with and
      without capture, knight promotion on the plain index, castling both ways, both colours); ValueHeadFry; served
      WDL with and without the WDL head's draw
  (e) input skip with layer 0
  (f) optimizer: the 'fry' weight-decay scheme and Muon scope as train.py forms them; wd = 1 / (0.17 lr steps)
  (g) all blocks together: loss, a gradient on every parameter, bf16 autocast, state_dict round trip, ONNX
  (h) torch.compile forward + backward == eager

  python test_fry_blocks.py
"""
import math
import os
import random
import sys
import tempfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_bt6_hybrid as H
from test_bt6_hybrid import batch_for, run_loss
from fry_testlib import (TEST_FENS, frame, fry_move_table, fry_tokens, legal_move_indices, piece_codes, record_from_board,
                         square_feature_rows)

PLAIN = {'ModelDim': 64, 'NumLayers': 3, 'NumHeads': 4, 'PreNorm': True, 'NormType': 'RMSNorm',
         'FFNMultiplier': 1, 'FFNActivationType': 'SwiGLU', 'HeadsActivationType': 'Mish', 'HeadWidthMultiplier': 4,
         'NonLinearAttention': False, 'SoftCapCutoff': 0, 'UseQKNorm': True,
         'SmolgenDimPerSquare': 0, 'SmolgenDim': 0, 'SmolgenToHeadDivisor': 1, 'SmolgenActivationType': 'Swish',
         'UseRPE': False, 'UseRPE_V': False, 'UseRoPE': False, 'NBTInnerLayers': 0}
OPT = {'NumTrainingPositions': 1000, 'BatchSizeForwardPass': 4, 'BatchSizeBackwardPass': 4, 'Optimizer': 'Muon',
       'LearningRateBase': 1e-3, 'LossValueMultiplier': 0.1, 'LossValue2Multiplier': 0, 'LossPolicyMultiplier': 1,
       'LossMLHMultiplier': 0, 'LossUNCMultiplier': 0.01, 'LossUncertaintyPolicyMultiplier': 0, 'LossValueDMultiplier': 0,
       'TorchSeed': 777, 'TPGV3': 0, 'AuxFeaturesPerSquare': 0}
GQA = {'AttnKVHeads': 2, 'AttnQKDims': [4, 8, 12], 'AttnValueDim': 16, 'AttnQKVBias': True}
FRY_HEADS = {'PolicyHeadFry': True, 'ValueHeadFry': True, 'ValueHLGaussServe': True}
OPT_HLG = {**OPT, 'HLGaussWeight': 0.5, 'HLGaussBuckets': 128}
BT6_FLAT = {**PLAIN, 'AttnTemplates': 4, 'BT6Hybrid': True, 'BT6NativeLayers': True, 'BT6TermsLayers': [0, 1, 2],
            'BT6EdgeSites': [1], 'BT6Embedding': 'linear', 'BT6Triplet': False, 'BT6SiteHidden': 32, 'BT6PolicyWidth': 64,
            'UseMoveTokens': True, 'MoveTokenDim': 32, 'MoveTokenLayers': 1, 'MoveTokenHeads': 2, 'MoveTokenFFNMult': 1,
            'MoveTokenMax': 96, 'MoveTokenPostMove': True, 'MoveTokenEdgeInput': True}
ALL = {**PLAIN, 'AttnTemplates': 4, 'NBTInputSkip': True, 'InputSkipLayer0': True, 'FryInput': True, 'SquareFeatures': True,
       'FFNNoBias': True, 'NormEps': 1e-5, **GQA, **FRY_HEADS}


def build(net, opt=None):
  saved = H.NET_SMALL, H.OPT_BASE
  try:
    H.NET_SMALL, H.OPT_BASE = dict(net), dict(opt or OPT)
    return H.build()
  finally:
    H.NET_SMALL, H.OPT_BASE = saved


def refused(net, what, opt=None):
  try:
    build(net, opt)
  except (ValueError, AssertionError) as e:
    assert what in str(e), (what, str(e))
  else:
    raise AssertionError(f'not refused: {what}')


def game_boards(n, seed=7):
  """Positions from random games (both colours, captures, castling, e.p., promotions as they come) + the fixed list."""
  rng = random.Random(seed)
  out = [chess.Board(f) for f in TEST_FENS]
  while len(out) < n:
    b = chess.Board()
    for _ in range(rng.randint(4, 120)):
      moves = list(b.legal_moves)
      if not moves:
        break
      b.push(rng.choice(moves))
      if rng.random() < 0.15:
        out.append(b.copy())
  return out[:n]


def recs_of(boards):
  return torch.from_numpy(np.stack([record_from_board(b) for b in boards]))


def live(net, seed=11):
  """Non-trivial values for the zero-initialised tensors, so that every path carries signal."""
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for n, p in net.named_parameters():
      if float(p.abs().max()) == 0.0:
        p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    if getattr(net, 'input_skip_a', None) is not None:
      net.input_skip_a.copy_(1.0 + 0.2 * torch.randn(net.input_skip_a.shape, generator=g))
      net.input_skip_b.copy_(0.3 * torch.randn(net.input_skip_b.shape, generator=g))


# ----------------------------------------------------------------------------------------------- (a)
def test_square_features():
  from square_features import SquareFeatures, codes_from_record, NUM_ROWS, NUM_SLOTS
  boards = game_boards(300)
  rec = recs_of(boards)
  codes = codes_from_record(rec)
  want_codes = np.stack([piece_codes(frame(b)) for b in boards])
  assert codes.dtype == torch.int64 and (codes.numpy() == want_codes).all(), 'codes from the record'
  sf = SquareFeatures(24)
  rows = sf.rows(codes).numpy()
  want = np.stack([square_feature_rows(c) for c in want_codes])
  assert rows.shape == (len(boards), 64, NUM_SLOTS) and (rows == want).all(), np.argwhere(rows != want)[:5]
  assert rows.min() >= 0 and rows.max() < NUM_ROWS
  used = len(np.unique(rows))
  # the summed lookup: training / eval path (embedding_bag) == gather + sum, and gradients reach exactly the used rows
  with torch.no_grad():
    sf.table.normal_()
  out = sf(codes)
  ref = sf.table[torch.from_numpy(rows)].sum(dim=2)
  assert out.shape == (len(boards), 64, 24) and float((out - ref).abs().max()) < 1e-4
  out.sum().backward()
  hit = (sf.table.grad.abs().sum(dim=1) > 0).numpy()
  assert hit.sum() == used and set(np.nonzero(hit)[0]) == set(np.unique(rows))
  n_black = sum(1 for b in boards if b.turn == chess.BLACK)
  print(f'(a) square features: {len(boards)} positions ({n_black} Black to move): codes from the record and all 64 x 23 rows == '
        f'the per-square reference; {used} of {NUM_ROWS} rows used; summed lookup == gather + sum; gradient on exactly the used rows')

  # as an add-on to the Linear embedding: zero table = the net without it; a live table changes the output
  sq = recs_of(boards[:6])
  for name, base in (('Linear embedding (plain trunk)', {**PLAIN, 'AttnTemplates': 4}),
                     ('BT6-form "linear" embedding, flat BT6 trunk + decoder (the 384 base arm layout)',
                      {**PLAIN, 'AttnTemplates': 4, 'BT6Hybrid': True, 'BT6NativeLayers': True, 'BT6TermsLayers': [0, 1, 2],
                       'BT6EdgeSites': [1], 'BT6Embedding': 'linear', 'BT6Triplet': False, 'BT6SiteHidden': 32, 'BT6PolicyWidth': 64,
                       'UseMoveTokens': True, 'MoveTokenDim': 32, 'MoveTokenLayers': 1, 'MoveTokenHeads': 2, 'MoveTokenFFNMult': 1,
                       'MoveTokenMax': 96, 'MoveTokenPostMove': True, 'MoveTokenEdgeInput': True})):
    ref_net = build(base).eval()
    net = build({**base, 'SquareFeatures': True}).eval()
    state = net.state_dict()
    ref_net.load_state_dict({k: state[k] for k in ref_net.state_dict()}, strict=True)
    assert set(state) - set(ref_net.state_dict()) == {'square_features.table'}
    with torch.no_grad():
      a, b = net(sq, None), ref_net(sq, None)
    d0 = max(float((a[0] - b[0]).abs().max()), float((a[1] - b[1]).abs().max()))
    assert d0 == 0.0, d0
    with torch.no_grad():
      net.square_features.table.normal_(0, 0.3)
      c = net(sq, None)
    assert float((c[1] - a[1]).abs().max()) > 1e-4
    net.train()
    loss = run_loss(net, batch_for(6), sq)
    loss.backward()
    assert float(net.square_features.table.grad.abs().max()) > 0
    from wd_partition import partition_weight_decay
    decay, no_decay = partition_weight_decay(net)
    assert 'square_features.table' in no_decay
    print(f'    add-on to the {name}: zero table == the net without the key (bit-identical), live table changes the output, '
          f'table gets a gradient, no-decay class in the default partition')


# ----------------------------------------------------------------------------------------------- (b)
def test_gqa():
  from dot_product_attention import DotProductAttention
  torch.manual_seed(5)
  D, Hq, G, dqk, dv, T, B = 48, 6, 3, 5, 8, 70, 3
  att = DotProductAttention(T, T, Hq, D // Hq, 'RMSNorm', 1e-5, use_qk_norm=True, gqa_kv_heads=G, gqa_qk_dim=dqk, gqa_v_dim=dv,
                            qkv_bias=True, qk_norm_eps=1e-5)
  with torch.no_grad():
    att.qLN.scale.uniform_(0.5, 1.5); att.kLN.scale.uniform_(0.5, 1.5)
  x = torch.randn(B, T, D)
  bias = torch.randn(B, Hq, T, T)
  got = att(x, x, x, x, piece_relation_bias=bias)
  # direct reference
  def rms(v, gain):
    return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-5) * gain
  q = rms(att.q_proj(x).reshape(B, T, Hq, dqk), att.qLN.scale)
  k = rms(att.k_proj(x).reshape(B, T, G, dqk), att.kLN.scale)
  v = att.v_proj(x).reshape(B, T, G, dv)
  heads = []
  for h in range(Hq):
    g = h // (Hq // G)
    s = torch.einsum('bid,bjd->bij', q[:, :, h], k[:, :, g]) / math.sqrt(dqk) + bias[:, h]
    heads.append(torch.einsum('bij,bjd->bid', torch.softmax(s, -1), v[:, :, g]))
  want = att.W_h(torch.cat(heads, dim=-1))
  err = float((got - want).abs().max())
  assert got.shape == (B, T, D) and err < 1e-5, err
  assert att.q_proj.weight.shape == (Hq * dqk, D) and att.k_proj.weight.shape == (G * dqk, D) and att.v_proj.weight.shape == (G * dv, D)
  assert att.W_h.weight.shape == (D, Hq * dv) and att.qLN.scale.shape == (dqk,) and att.qkv is None
  got.sum().backward()
  assert all(p.grad is not None and float(p.grad.abs().max()) > 0 for p in att.parameters())
  print(f'(b) grouped-query attention ({Hq} query heads on {G} key/value heads, q/k width {dqk}, value width {dv}, {T} tokens, '
        f'with a logit bias): == the einsum reference with query head h on key/value head h // {Hq // G} (max diff {err:.1e}); '
        f'gradient on every parameter')
  # the plain grouping and widths: == our fused-qkv attention with the same weights
  torch.manual_seed(6)
  Hq, dk = 4, 12
  plain = DotProductAttention(64, 64, Hq, dk, 'RMSNorm', 1e-6, use_qk_norm=True)
  same = DotProductAttention(64, 64, Hq, dk, 'RMSNorm', 1e-6, use_qk_norm=True, gqa_kv_heads=Hq, gqa_qk_dim=dk, gqa_v_dim=dk)
  with torch.no_grad():
    w = plain.qkv.weight.reshape(Hq, 3, dk, Hq * dk)                  # fused rows: per head [q | k | v]
    same.q_proj.weight.copy_(w[:, 0].reshape(Hq * dk, -1)); same.k_proj.weight.copy_(w[:, 1].reshape(Hq * dk, -1))
    same.v_proj.weight.copy_(w[:, 2].reshape(Hq * dk, -1))
    same.W_h.load_state_dict(plain.W_h.state_dict())
    plain.qLN.scale.uniform_(0.5, 1.5); same.qLN.scale.copy_(plain.qLN.scale)
    plain.kLN.scale.uniform_(0.5, 1.5); same.kLN.scale.copy_(plain.kLN.scale)
  x = torch.randn(2, 64, Hq * dk)
  bias = torch.randn(2, Hq, 64, 64)
  err = float((plain(x, x, x, x, piece_relation_bias=bias) - same(x, x, x, x, piece_relation_bias=bias)).abs().max())
  assert same.q_proj.bias is None and err < 1e-5, err
  print(f'    with kv heads = heads and q/k = value width = ModelDim / heads: == our fused-qkv attention on the same weights ({err:.1e})')
  # in a net: per-layer widths, with templates; identity of the template path; parameter shapes
  net = build({**PLAIN, 'AttnTemplates': 4, **GQA})
  widths = [(l.attention.qk_width, l.attention.kv_heads, l.attention.v_width) for l in net.transformer_layer]
  assert widths == [(4, 2, 16), (8, 2, 16), (12, 2, 16)], widths
  assert [tuple(l.attention.q_proj.weight.shape) for l in net.transformer_layer] == [(16, 64), (32, 64), (48, 64)]
  assert all(l.attention.q_proj.bias is not None and l.tmix is not None for l in net.transformer_layer)
  sq = recs_of(game_boards(6))
  live(net)
  seen = {}
  hk = net.transformer_layer[1].attention.register_forward_pre_hook(lambda m, a, kw: seen.update(prb=kw.get('piece_relation_bias')), with_kwargs=True)
  net.train()
  loss = run_loss(net, batch_for(6), sq)
  hk.remove()
  assert seen['prb'] is not None and seen['prb'].shape == (6, 4, 64, 64)
  loss.backward()
  dead = [n for n, p in net.named_parameters() if 'transformer_layer' in n and (p.grad is None or float(p.grad.abs().max()) == 0.0)]
  assert torch.isfinite(loss) and not dead, dead
  print(f'    in a net with attention templates: per-layer (q/k width, kv heads, value width) {widths}; the template bias reaches '
        f'the grouped attention; loss finite; gradient on every trunk parameter')
  for bad, what in (({**PLAIN, **GQA, 'UseQKNorm': False}, 'needs UseQKNorm'),
                    ({**PLAIN, **GQA, 'AttnKVHeads': 3}, 'must divide NumHeads'),
                    ({**PLAIN, **GQA, 'AttnQKDims': [4, 8]}, 'one entry per (distinct) layer'),
                    ({**PLAIN, 'AttnQKVBias': True}, 'silent no-op'),
                    ({**PLAIN, **GQA, 'UseRoPE': True}, 'does not combine with UseRoPE'),
                    ({**PLAIN, **GQA, 'NBTInnerLayers': 2, 'NBTWidthDivisor': 2, 'NBTInnerHeads': 4}, 'plain encoder trunk'),
                    ({**BT6_FLAT, **GQA, 'BT6FusedAttention': True}, 'BT6FusedAttention'),
                    ({**BT6_FLAT, **GQA, 'BT6EdgeValues': True}, 'BT6EdgeValues')):
    refused(bad, what)
  print('    refusals OK (no QK-norm, kv heads not dividing, list length, bias without the block, RoPE, NBT trunk, BT6 fused kernel / edge values)')
  # on the flat BT6 trunk of the 384 base arm (term reads on every layer, edge sites, decoder with edge input):
  # the plain grouping and widths == the fused-qkv net on the same weights, with live reader tables
  ref = build(BT6_FLAT)
  same = build({**BT6_FLAT, 'AttnKVHeads': 4, 'AttnQKDims': 16, 'AttnValueDim': 16})
  H.randomise(ref); live(ref)
  state, target = ref.state_dict(), same.state_dict()
  for k in target:
    if k in state:
      target[k] = state[k]
  for i, layer in enumerate(ref.transformer_layer):
    w = layer.attention.qkv.weight.detach().reshape(4, 3, 16, 64)
    for j, nm in enumerate(('q_proj', 'k_proj', 'v_proj')):
      target[f'transformer_layer.{i}.attention.{nm}.weight'] = w[:, j].reshape(64, 64)
  same.load_state_dict(target, strict=True)
  sq = recs_of(game_boards(6))
  ref.eval(); same.eval()
  with torch.no_grad():
    a, b = ref(sq, None), same(sq, None)
  d = max(float((a[0] - b[0]).abs().max()), float((a[1] - b[1]).abs().max()))
  assert d < 2e-4, d
  net = build({**BT6_FLAT, **GQA})
  assert [tuple(l.attention.bt6t.Pq.shape) for l in net.transformer_layer] == [(4, 34, 4), (4, 34, 8), (4, 34, 12)]
  H.randomise(net); live(net)
  net.train()
  loss = run_loss(net, batch_for(6), sq)
  loss.backward()
  dead = [n for n, p in net.named_parameters() if 'transformer_layer' in n and (p.grad is None or float(p.grad.abs().max()) == 0.0)]
  assert torch.isfinite(loss) and not dead, dead
  print(f'    on the flat BT6 trunk (term reads on every layer, edge sites, decoder with edge input; the 384 base arm layout): with '
        f'the plain grouping and widths == the fused-qkv net on the same weights and live reader tables (max diff {d:.1e}); with '
        f'narrow q/k the q- / k-side relation tables take the layer width {[l.attention.qk_width for l in net.transformer_layer]}; '
        f'loss finite, gradient on every trunk parameter (edge-site readback included)')


# ----------------------------------------------------------------------------------------------- (c)
def test_fry_input():
  from fry_input import tokens_from_record
  boards = game_boards(300)
  rec = recs_of(boards)
  tok = tokens_from_record(rec).numpy()
  want = np.stack([fry_tokens(b) for b in boards])
  white = np.array([b.turn == chess.WHITE for b in boards])
  cols = np.nonzero((tok != want).any(axis=0))[0].tolist()
  rows69 = np.nonzero(tok[:, 69] != want[:, 69])[0]
  assert set(cols) <= {64, 69}, cols
  assert (tok[white, 64] == want[white, 64]).all() and (tok[~white, 64] == 14).all() and (want[~white, 64] == 13).all()
  # e.p.: the record path is pseudo-legal; it may differ from the legal token only where the capture is not legal
  for r in rows69:
    fb = frame(boards[r])
    assert want[r, 69] == 17 and fb.ep_square is not None and tok[r, 69] == 17 + fb.ep_square and not fb.has_legal_en_passant()
  n_ep = int((want[:, 69] != 17).sum())
  i_pin = TEST_FENS.index('8/8/8/8/k1pP3R/8/8/3K4 b - d3 0 1')
  assert i_pin in rows69.tolist(), 'the pinned e.p. test position must be one of the differing rows'
  print(f'(c) FryInput tokens from our record == the board encoder on {len(boards)} positions ({int((~white).sum())} Black to move, '
        f'{n_ep} with a legal e.p. capture) in all 70 columns except: colour (the record has none: always 14) and e.p. on '
        f'{len(rows69)} positions where the capture is pseudo-legal but not legal (e.g. the pinned-pawn test position)')

  net = build({**PLAIN, 'AttnTemplates': 4, 'NBTInputSkip': True, 'InputSkipLayer0': True, 'FryInput': True, 'SquareFeatures': True})
  assert net.embedding_layer is None and net.embedding_norm is None and net.fry_input.pos_embedding.shape == (70, 64)
  live(net)
  sq = rec[:6]
  seen = {'x': [], 'prb': []}
  def grab(m, a, kw):
    seen['x'].append(a[0]); seen['prb'].append(kw.get('piece_relation_bias'))
  hooks = [l.attention.register_forward_pre_hook(grab, with_kwargs=True) for l in net.transformer_layer]
  net.eval()
  with torch.no_grad():
    out = net(sq, None)
    for h in hooks:
      h.remove()
    net._fry_tokens_override = torch.from_numpy(want[:6])
    out_t = net(sq, None)
    net._fry_tokens_override = None
  assert all(x.shape == (6, 70, 64) for x in seen['x']) and all(p.shape == (6, 4, 70, 70) for p in seen['prb'])
  prb = seen['prb'][1]
  assert float(prb[:, :, 64:, :].abs().max()) == 0.0 and float(prb[:, :, :, 64:].abs().max()) == 0.0 and float(prb[:, :, :64, :64].abs().max()) > 0
  assert out[0].shape == (6, 1858) and out[1].shape == (6, 3)
  w6 = white[:6]
  assert float((out[0][w6] - out_t[0][w6]).abs().max()) == 0.0, 'White to move: record path == token path'
  assert float((out[1][~w6] - out_t[1][~w6]).abs().max()) > 0, 'Black to move: the colour token differs'
  # the metadata tokens matter: change a castling right in the record -> the value changes
  sq2 = sq.clone(); sq2[:, :, 112] = 1.0 - sq2[:, :, 112]
  with torch.no_grad():
    out2 = net(sq2, None)
  assert float((out2[1] - out[1]).abs().max()) > 1e-5
  net.train()
  loss = run_loss(net, batch_for(6), sq)
  loss.backward()
  dead = [n for n, p in net.named_parameters() if p.grad is None or float(p.grad.abs().max()) == 0.0]
  assert torch.isfinite(loss) and not [n for n in dead if n.startswith(('fry_input', 'square_features', 'transformer_layer', 'attn_templates', 'input_skip'))], dead
  print('    70 tokens through a plain trunk with templates + input skip (layer 0 included) + square features: every layer sees '
        '[B, 70, D]; template bias is 0 on metadata rows / columns; heads give [B, 1858] / [B, 3]; record path == token path for '
        'White to move; a castling right changes the value; gradient on every embedding / trunk parameter')
  # with our move-token decoder (reads the 64 square states) and depth probes
  dec = {**PLAIN, 'AttnTemplates': 4, 'FryInput': True, 'SquareFeatures': True, 'UseMoveTokens': True, 'MoveTokenDim': 32,
         'MoveTokenLayers': 1, 'MoveTokenHeads': 2, 'MoveTokenFFNMult': 1, 'MoveTokenMax': 96, 'MoveTokenPostMove': True}
  netd = build(dec, {**OPT, 'DepthProbeValueWeight': 0.25})
  live(netd)
  netd.train()
  loss = run_loss(netd, batch_for(6), sq)
  loss.backward()
  assert torch.isfinite(loss) and float(netd.fry_input.tok_embedding.weight.grad.abs().max()) > 0
  print('    with our move-token decoder as the policy and the depth probes: loss finite, gradients reach the token embedding')
  for bad, what in (({**PLAIN, 'FryInput': True, 'PreNorm': False}, 'plain pre-norm encoder trunk'),
                    ({**PLAIN, 'FryInput': True, 'NBTInnerLayers': 2, 'NBTWidthDivisor': 2, 'NBTInnerHeads': 4}, 'plain pre-norm encoder trunk'),
                    ({**PLAIN, 'FryInput': True, 'SmolgenDimPerSquare': 8, 'SmolgenDim': 64}, 'smolgen'),
                    ({**PLAIN, 'FryInput': True, 'BT6Hybrid': True, 'BT6NativeLayers': True, 'BT6EdgeSites': [1], 'BT6Embedding': 'linear'}, 'BT6Hybrid'),
                    ({**PLAIN, 'FryInput': True, 'UseRPE': True}, 'UseRPE'),
                    ({**dec, 'MoveTokenEdgeInput': True}, 'MoveTokenEdgeInput'),
                    ({**PLAIN, 'FryInput': True, 'RegisterTokens': 2}, 'RegisterTokens'),
                    ({**PLAIN, 'InputSkipLayer0': True}, 'InputSkipLayer0 needs NBTInputSkip')):
    refused(bad, what)
  print('    refusals OK (post-norm, NBT trunk, smolgen, BT6 hybrid, RPE, decoder edge input, register tokens, layer-0 skip without the skip)')


# ----------------------------------------------------------------------------------------------- (d)
def test_heads():
  from fry_heads import FryPolicyHead, FryValueHead
  torch.manual_seed(9)
  D = 32
  boards = game_boards(120)
  codes = torch.from_numpy(np.stack([piece_codes(frame(b)) for b in boards]))
  y = torch.randn(len(boards), 64, D)
  table = fry_move_table()
  n_moves = n_promo = n_kn = n_castle = 0
  for king_castling in (True, False):
    head = FryPolicyHead(D, king_castling=king_castling)
    with torch.no_grad():
      got = head(y, codes)
      F, T, P = head.from_proj(y), head.to_proj(y), head.promo_proj(y)
      L = torch.einsum('bfd,btd->bft', F, T) / math.sqrt(D)
    assert got.shape == (len(boards), 1858) and got.dtype == torch.float32
    worst = 0.0
    for b, board in enumerate(boards):
      fb = frame(board)
      for uci, ours, his in legal_move_indices(board):
        f, t, promo = table[his]                                   # Fry's own formula for his id
        want = L[b, f, t] + (P[b, t, 'nbrq'.index(promo)] if promo else 0.0)
        mv = chess.Move.from_uci(uci)
        if fb.is_castling(mv) and not king_castling:
          rook = 7 if t > f else 0                                 # our data convention: the king-takes-rook pair
          want = L[b, f, rook]
        worst = max(worst, abs(float(got[b, ours] - want)))
        if king_castling:
          n_moves += 1; n_promo += bool(promo); n_kn += promo == 'n'; n_castle += fb.is_castling(mv)
    assert worst < 1e-5, worst
  # a non-pawn stepping from rank 7 to rank 8 gets no promotion term
  b = chess.Board('4k3/R7/8/8/8/8/8/4K3 w - - 0 1')
  c = torch.from_numpy(piece_codes(b)).unsqueeze(0)
  head = FryPolicyHead(D)
  with torch.no_grad():
    g = head(y[:1], c)[0]
    L = torch.einsum('fd,td->ft', head.from_proj(y[0]), head.to_proj(y[0])) / math.sqrt(D)
  i = [o for u, o, _ in legal_move_indices(b) if u == 'a7a8'][0]
  assert abs(float(g[i] - L[48, 56])) < 1e-6
  print(f'(d) PolicyHeadFry on our 1858 moves == Fry\'s 1968-move formula for every legal move of {len(boards)} positions: {n_moves} moves, '
        f'{n_promo} promotions ({n_kn} knight promotions on the plain index), {n_castle} castling moves (king-castling read for his '
        f'weights; king-takes-rook pair for ours); a rook stepping a7-a8 gets no promotion term')
  vh = FryValueHead(D, 70, 128)
  out = vh(torch.randn(5, 70, D))
  assert out.shape == (5, 128) and vh.fc.weight.shape == (D, 70 * 16)
  # in a net: served WDL with the WDL head's draw, and with no draw
  sq = recs_of(boards[:6])
  for draw in ('wdl', 'none'):
    opt = {**OPT_HLG, 'LossValueMultiplier': 0.1 if draw == 'wdl' else 0}
    net = build({**PLAIN, **FRY_HEADS, 'ValueHLGaussServeDraw': draw}, opt)
    assert net.policy_head is None and net.fry_policy is not None and isinstance(net.hlg_head, FryValueHead)
    live(net)
    grab = {}
    hk = net.hlg_head.register_forward_hook(lambda m, a, o: grab.update(hlg=o))
    net.eval()
    with torch.no_grad():
      o = net(sq, None)
    hk.remove()
    wdl = torch.exp(o[1].double())
    q = (torch.softmax(grab['hlg'].double(), -1) * net.hlg_centers.double()).sum(-1)
    assert float((wdl.sum(-1) - 1).abs().max()) < 1e-5 and float((wdl[:, 0] - wdl[:, 2] - q).abs().max()) < 1e-5
    assert (float(wdl[:, 1].max()) < 1e-6) == (draw == 'none')
    net.train()
    loss = run_loss(net, batch_for(6), sq)
    loss.backward()
    for n in ('fry_policy.from_proj.weight', 'fry_policy.to_proj.weight', 'hlg_head.squeeze.weight', 'hlg_head.fc.weight', 'hlg_head.out.weight'):
      assert float(dict(net.named_parameters())[n].grad.abs().max()) > 0, n
    assert torch.isfinite(loss)
    print(f'    Fry heads in a net, ValueHLGaussServeDraw {draw!r}: served W - L == the HL-Gauss Q, W + D + L = 1, '
          f'D {"= 0" if draw == "none" else "from the WDL head"}; loss finite, gradients on both heads')
  # ValueHeadFry as the body of the TRAINING-ONLY HL-Gauss head (no ValueHLGaussServe): the WDL head stays the served value
  net = build({**PLAIN, 'ValueHeadFry': True}, OPT_HLG)
  assert isinstance(net.hlg_head, FryValueHead) and net.policy_head is not None and not net.hlg_serve
  net.train()
  loss = run_loss(net, batch_for(6), sq)
  loss.backward()
  assert torch.isfinite(loss) and float(net.hlg_head.fc.weight.grad.abs().max()) > 0
  print('    ValueHeadFry without ValueHLGaussServe (training-only HL-Gauss head with his body, our MLP policy head): loss finite, gradient on the head')
  refused({**PLAIN, **FRY_HEADS, 'UseMoveTokens': True, 'MoveTokenDim': 32, 'MoveTokenLayers': 1, 'MoveTokenHeads': 2}, 'PolicyHeadFry does not combine', OPT_HLG)
  refused({**PLAIN, 'ValueHeadFry': True}, 'needs HLGaussWeight')
  refused({**PLAIN, **FRY_HEADS, 'ValueHLGaussServeDraw': 'none'}, 'LossValueMultiplier must be 0', OPT_HLG)
  print('    refusals OK (PolicyHeadFry with the decoder, ValueHeadFry without the HL-Gauss loss, no-draw serving with a trained WDL head)')


# ----------------------------------------------------------------------------------------------- (e)
def test_input_skip_layer0():
  sq = recs_of(game_boards(6))
  base = {**PLAIN, 'AttnTemplates': 4, 'NBTInputSkip': True}
  a = build(base).eval()
  b = build({**base, 'InputSkipLayer0': True}).eval()
  b.load_state_dict(a.state_dict(), strict=True)
  with torch.no_grad():
    oa, ob = a(sq, None), b(sq, None)
  assert float((oa[1] - ob[1]).abs().max()) == 0.0, 'a = 1, b = 0 at init: identical'
  # layer 0 reads (a0 + b0) * embedding
  with torch.no_grad():
    b.input_skip_a[0] = 0.7; b.input_skip_b[0] = 0.5
  seen = {}
  h1 = b.embedding_norm.register_forward_hook(lambda m, i, o: seen.update(e=o))
  h2 = b.transformer_layer[0].register_forward_pre_hook(lambda m, i: seen.update(x=i[0]))
  with torch.no_grad():
    oc = b(sq, None)
  h1.remove(); h2.remove()
  assert float((seen['x'] - 1.2 * seen['e']).abs().max()) < 1e-5 and float((oc[1] - ob[1]).abs().max()) > 1e-6
  with torch.no_grad():
    a.input_skip_a[0] = 0.7; a.input_skip_b[0] = 0.5
    od = a(sq, None)
  assert float((od[1] - oa[1]).abs().max()) == 0.0, 'without InputSkipLayer0 the layer-0 scalars are unused'
  # the flat BT6 trunk runs its own layer loop: the layer-0 scalars act there too
  f = build({**BT6_FLAT, 'NBTInputSkip': True, 'InputSkipLayer0': True}).eval()
  H.randomise(f)
  with torch.no_grad():
    f0 = f(sq, None)
    f.input_skip_b[0] = 0.5
    f1 = f(sq, None)
  assert float((f1[1] - f0[1]).abs().max()) > 1e-6, 'InputSkipLayer0 on the flat BT6 trunk'
  b.train()
  live(b)
  run_loss(b, batch_for(6), sq).backward()
  assert all(float(g) != 0 for g in b.input_skip_a.grad) and all(float(g) != 0 for g in b.input_skip_b.grad)
  print('(e) input skip: with InputSkipLayer0 layer 0 reads (a0 + b0) * embedding (identical at init); without it the layer-0 '
        'scalars are unused (the existing NBTInputSkip form); also on the flat BT6 trunk; all scalars get a gradient')


# ----------------------------------------------------------------------------------------------- (f)
def fry_partition(net):
  """The two 'fry' rules exactly as train.py states them (train.py builds them inline; same predicates here)."""
  named = dict(net.named_parameters())
  muon = [n for n, p in named.items() if p.ndim == 2 and 'transformer_layer' in n and 'lora' not in n.lower()]
  no_decay = {n for n in named if '.qLN.' in n or '.kLN.' in n or n.endswith('input_skip_a') or n.endswith('input_skip_b')}
  return named, muon, no_decay


def test_optimizer():
  import re
  src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py'), encoding='utf-8').read()
  assert "return p.ndim == 2 and 'transformer_layer' in n and 'lora' not in n.lower()" in src, 'train.py: the fry Muon predicate changed'
  assert re.search(r"no_decay = \{pn for pn in param_dict if '\.qLN\.' in pn or '\.kLN\.' in pn or pn\.endswith\('input_skip_a'\) "
                   r"or pn\.endswith\('input_skip_b'\)\}", src), 'train.py: the fry no-decay set changed'
  net = build(ALL, OPT_HLG)
  named, muon, no_decay = fry_partition(net)
  kinds = sorted({re.sub(r'transformer_layer\.\d+\.', '', n) for n in muon})
  assert kinds == ['attention.W_h.weight', 'attention.k_proj.weight', 'attention.q_proj.weight', 'attention.v_proj.weight',
                   'mlp.linear1.weight', 'mlp.linear2.weight', 'mlp.linear3.weight', 'tmix.mix.weight'], kinds
  nd_kinds = sorted({re.sub(r'transformer_layer\.\d+\.', '', n) for n in no_decay})
  assert nd_kinds == ['attention.kLN.scale', 'attention.qLN.scale', 'input_skip_a', 'input_skip_b'], nd_kinds
  decayed = sorted({re.sub(r'transformer_layer\.\d+\.', '', n) for n in named if n not in no_decay})
  for must in ('ln1.scale', 'ln2.scale', 'trunk_end_norm.scale', 'attn_templates.table', 'square_features.table',
               'fry_input.tok_embedding.weight', 'fry_input.pos_embedding', 'attention.q_proj.bias', 'tmix.mix.bias',
               'fry_policy.from_proj.bias', 'hlg_head.out.weight'):
    assert must in decayed, must
  # our default partition keeps these out of decay -- the difference the scheme makes
  from wd_partition import partition_weight_decay
  d0, nd0 = partition_weight_decay(net)
  moved = sorted({re.sub(r'transformer_layer\.\d+\.', '', n) for n in nd0 if n not in no_decay})
  print(f'(f) optimizer, scheme "fry": Muon on {len(muon)} matrices of kinds {kinds}; no decay only on {nd_kinds}; decayed in his '
        f'scheme but not in our default partition: {len(moved)} kinds, e.g. {moved[:8]}')
  # the Muon optimizer of the trainer accepts the partition; a few steps reduce the loss; no-decay params are not decayed
  from muon import Muon
  mu = [named[n] for n in muon]
  ad = [p for n, p in named.items() if n not in set(muon)]
  opt = Muon(lr=2e-3, wd=0.05, momentum=0.95, adamw_betas=(0.9, 0.999), muon_params=mu, adamw_params=ad,
             wd_scales={named[n]: 0.0 for n in no_decay})
  sq = recs_of(game_boards(8))
  batch = batch_for(8)
  net.train()
  losses = []
  gains0 = net.transformer_layer[0].attention.qLN.scale.detach().clone()
  for _ in range(8):
    opt.zero_grad()
    loss = run_loss(net, batch, sq)
    loss.backward()
    opt.step()
    losses.append(float(loss))
  assert all(math.isfinite(l) for l in losses) and losses[-1] < losses[0], losses
  print(f'    trainer Muon (momentum 0.95) + AdamW with that partition: 8 steps, loss {losses[0]:.3f} -> {losses[-1]:.3f}')
  # weight decay from the schedule: 1 / (0.17 * lr * steps)
  cfg = None
  saved = H.NET_SMALL, H.OPT_BASE
  try:
    H.NET_SMALL = dict(PLAIN)
    H.OPT_BASE = {**OPT, 'WeightDecayFromSchedule': 0.17, 'WeightDecay': None, 'LearningRateBase': 1.5e-3,
                  'NumTrainingPositions': 6_000_000_000, 'BatchSizeBackwardPass': 4104, 'BatchSizeForwardPass': 4104}
    cfg = H.make_config()
  finally:
    H.NET_SMALL, H.OPT_BASE = saved
  assert abs(cfg.Opt_WeightDecay - 1.0 / (0.17 * 1.5e-3 * (6e9 / 4104))) < 1e-12 and abs(cfg.Opt_WeightDecay - 0.00268) < 2e-5
  refused(PLAIN, 'remove the WeightDecay key', {**OPT, 'WeightDecayFromSchedule': 0.17, 'WeightDecay': 0.01})
  refused(PLAIN, 'MuonHonorNoDecay must be off', {**OPT, 'WeightDecayScheme': 'fry', 'MuonHonorNoDecay': True})
  refused(PLAIN, 'full-length runs', {**OPT, 'WeightDecayFromSchedule': 0.17, 'WeightDecay': None})
  print(f'    WeightDecayFromSchedule 0.17 at lr 1.5e-3, 6.0B positions, batch 4104: wd = {cfg.Opt_WeightDecay:.5f} (his table: 0.0026-0.0027); '
        f'refusals OK (explicit WeightDecay alongside, MuonHonorNoDecay alongside, a run too short for the rule)')


# ----------------------------------------------------------------------------------------------- (g)
def test_all_together():
  import onnxruntime as ort
  boards = game_boards(12)
  sq = recs_of(boards)
  net = build(ALL, OPT_HLG)
  n_par = sum(p.numel() for p in net.parameters())
  live(net)
  net.train()
  loss = run_loss(net, batch_for(12), sq)
  loss.backward()
  dead = [n for n, p in net.named_parameters() if p.grad is None]
  zero = [n for n, p in net.named_parameters() if p.grad is not None and float(p.grad.abs().max()) == 0.0]
  assert torch.isfinite(loss) and not dead, dead
  # zero gradients are allowed only on heads the config does not train (loss weight 0)
  assert all(n.startswith(('unc_head', 'unc_policy', 'mlh_head', 'qdev', 'value2_head')) for n in zero), zero
  # bf16 autocast (CPU): the policy logits and the served log-WDL stay float32, the loss is finite
  net.zero_grad()
  with torch.autocast('cpu', dtype=torch.bfloat16):
    outs = net(sq, None)
  assert outs[0].dtype == torch.float32 and outs[1].dtype == torch.float32, (outs[0].dtype, outs[1].dtype)
  with torch.autocast('cpu', dtype=torch.bfloat16), torch.no_grad():     # (no bf16 backward on this CPU's oneDNN)
    loss16 = run_loss(net, batch_for(12), sq)
  assert torch.isfinite(loss16) and abs(float(loss16.detach()) - float(loss.detach())) < 0.05 * abs(float(loss.detach())), (float(loss16), float(loss))
  # state_dict round trip
  net.eval()
  with torch.no_grad():
    a = net(sq, None)
  twin = build(ALL, OPT_HLG).eval()
  twin.load_state_dict(net.state_dict(), strict=True)
  with torch.no_grad():
    b = twin(sq, None)
  assert float((a[0] - b[0]).abs().max()) == 0.0 and float((a[1] - b[1]).abs().max()) == 0.0
  # ONNX (opset 18, fp32) == PyTorch
  class SquaresOnly(torch.nn.Module):
    def __init__(self, inner):
      super().__init__(); self.inner = inner
    def forward(self, squares):
      out = self.inner(squares, None)
      return out[0], out[1], out[4]
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'net.onnx')
    torch.onnx.export(SquaresOnly(net), (sq[:2],), path, do_constant_folding=True, export_params=True, opset_version=18,
                      input_names=['squares'], output_names=['policy', 'value', 'value2'],
                      dynamic_axes={n: {0: 'batch_size'} for n in ('squares', 'policy', 'value', 'value2')})
    import onnx
    ops = sorted({n.op_type for n in onnx.load(path).graph.node})
    so = ort.SessionOptions(); so.log_severity_level = 3
    got = ort.InferenceSession(path, so, providers=['CPUExecutionProvider']).run(None, {'squares': sq.numpy().astype(np.float32)})
  errs = []
  for ya, yb in zip(got, (a[0], a[1], a[4])):
    errs.append(float(np.abs(ya.astype(np.float64) - yb.numpy().astype(np.float64)).max()))
  assert max(errs) < 1e-3, errs
  assert 'ArgMax' not in ops and 'CumSum' not in ops and 'Loop' not in ops and 'ScatterND' not in ops, ops
  print(f'(g) all blocks together ({n_par:,} parameters): loss finite, a gradient on every parameter (zero only on the heads '
        f'this config does not train: {sorted({n.split(".")[0] for n in zero})}); under bf16 autocast (CPU, forward only) policy / served value stay float32 and the loss agrees ({float(loss16.detach()):.4f} vs {float(loss.detach()):.4f}); state_dict round trip identical; ONNX (opset 18, '
        f'12 positions, batch axis dynamic) == PyTorch (max diff {max(errs):.1e}); no ArgMax / CumSum / Loop / ScatterND in the graph')


def test_compile():
  # torch.compile (as train.py runs it, CPU here): forward + backward == eager, with all blocks, and with square
  # features + grouped-query attention on the flat BT6 trunk
  sq = recs_of(game_boards(4))
  for name, net_def, opt in (('all blocks (FryInput, square features, grouped attention, Fry heads)', ALL, OPT_HLG),
                             ('flat BT6 trunk + decoder + square features + grouped attention', {**BT6_FLAT, 'SquareFeatures': True, **GQA}, OPT)):
    net = build(net_def, opt)
    H.randomise(net); live(net)
    net.train()
    eager = run_loss(net, batch_for(4), sq)
    cnet = torch.compile(net)
    loss = run_loss(cnet, batch_for(4), sq)
    loss.backward()
    assert torch.isfinite(loss) and abs(float(loss.detach()) - float(eager.detach())) < 1e-4, (float(loss), float(eager))
    print(f'(h) torch.compile forward + backward, {name}: loss {float(loss.detach()):.5f} == eager {float(eager.detach()):.5f}')


if __name__ == '__main__':
  torch.set_num_threads(8)
  test_square_features()
  test_gqa()
  test_fry_input()
  test_heads()
  test_input_skip_layer0()
  test_optimizer()
  test_all_together()
  test_compile()
  print('test_fry_blocks: OK')
