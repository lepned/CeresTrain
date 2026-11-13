"""Tests for the NBT-BT6 hybrid (NetDef BT6Hybrid; bt6_hybrid.py, bt6_net.py, nbt_layer.forward_bt6, ceres_net.py).

Run from src/CeresTrainPy (CPU):

    python test_bt6_hybrid.py            all tests on small nets (about a minute)
    python test_bt6_hybrid.py table      parameter tables + optimiser-group tables of the two target configs
                                         (builds 2048/1024 and 2048/896, 8 x 3, on CPU: needs ~4 GB RAM)
    python test_bt6_hybrid.py cost       time / allocation of BT6Relations + pair / edge tensors per 1,024 positions

1. Relations from OUR record == BT6Relations on lc0's planes for the same positions (current and previous position);
   copy-filled history and a record without history bytes both give EMPTY gained / lost.
2. Equivalence with bt6_net: (a) every inner layer and every edge site of the hybrid, fed the tensors it received in
   the hybrid forward, equals a stand-alone bt6_net.BT6Layer / BT6EdgeSite with the same weights (bitwise);
   (b) the whole hybrid forward equals `forward_served_hybrid` on its own export -- the kernel-form code
   (bt6_net.served_layer / served_site / served_policy / served_prologue) that was gated against the real BT6 program.
3. Gradient checkpoint per block: bit-identical forward on / off, gradients equal within float noise.
4. Optimiser scopes: weight-decay partition complete, reader tables and scales in no_decay and out of Muon, GLU
   kernels under the 'ffn-only' scope; every parameter receives a gradient from the real loss path.
5. bf16 autocast (CPU): logit assembly, softmax, gates, rms and relation einsums run in float32; the stream stays bf16.
6. Config: decoder-bound keys and incompatible trunk options are refused with a message.
7. Checkpoint-style state_dict save / reload gives identical outputs.
"""

import io
import json
import os
import shutil
import sys
import tempfile
import time

os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bt6_planes import encode_fens, expand_planes, parse_fen, read_fens  # noqa: E402

FEN_FILE = '/mnt/c/lc0_bt6/check/selftest_7040.fen'
FALLBACK_FENS = [
  'r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2QK2R b KQ - 3 8',
  '5r1k/ppp1q2p/8/3BnQ2/4p1n1/2N3K1/PPP2PP1/R4R2 w - - 1 21',
  '8/P6k/8/8/8/8/7p/K7 b - - 12 60',
  'rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3',
  '6k1/p3b2p/1p1pP3/2p3P1/1Pnp3B/P6P/3Q3K/8 w - - 0 38',
  '2r1kb2/5p1p/p3p3/1p2P1Q1/1Pq5/P1p3P1/2P4P/2KR1R2 w - - 0 28',
]

NET_SMALL = {
  'ModelDim': 128, 'NumLayers': 2, 'NumHeads': 4, 'PreNorm': False, 'NormType': 'RMSNorm',
  'FFNMultiplier': 1, 'FFNActivationType': 'SwiGLU', 'HeadsActivationType': 'Mish', 'HeadWidthMultiplier': 4,
  'NonLinearAttention': False, 'SoftCapCutoff': 0,
  'SmolgenDimPerSquare': 0, 'SmolgenDim': 0, 'SmolgenToHeadDivisor': 1, 'SmolgenActivationType': 'Swish',
  'UseRPE': False, 'UseRPE_V': False, 'UseRoPE': False,
  'NBTInnerLayers': 2, 'NBTWidthDivisor': 2, 'NBTInnerHeads': 4, 'NBTInnerPreNorm': True,
  'NBTProjActivation': 'None', 'NBTProjNorm': 'Norm',
  'BT6Hybrid': True, 'BT6EdgeSites': [0], 'BT6PolicyWidth': 64, 'BT6SoftPolicyWidth': 32,
  'BT6SiteHidden': 32, 'BT6EmbeddingDense': 16, 'BT6Embedding': 'bt6',
}
# THE target: 1536 outer / 768 inner (24 heads x 32) / 10 blocks x 3 inner layers, sites after blocks 0, 2, 4, 6, 8
NET_TARGET = {
  **NET_SMALL, 'ModelDim': 1536, 'NumLayers': 10, 'NumHeads': 24, 'NBTInnerLayers': 3, 'NBTWidthDivisor': 2,
  'NBTInnerHeads': 24, 'BT6EdgeSites': [0, 2, 4, 6, 8], 'BT6PolicyWidth': 1024, 'BT6SoftPolicyWidth': 256,
  'BT6SiteHidden': 64, 'BT6EmbeddingDense': 512,
}
# The earlier target shape, kept as a second table: 2048 / 1024 (32 x 32) / 8 x 3
NET_2048 = {**NET_TARGET, 'ModelDim': 2048, 'NumLayers': 8, 'NumHeads': 32, 'NBTInnerHeads': 32,
            'BT6EdgeSites': [0, 2, 3, 4, 6]}
OPT_BASE = {
  'NumTrainingPositions': 1000, 'BatchSizeForwardPass': 4, 'BatchSizeBackwardPass': 4,
  'Optimizer': 'Muon', 'LearningRateBase': 1e-4, 'LossValueMultiplier': 1.0,
  'LossValue2Multiplier': 0.5, 'LossPolicyMultiplier': 1, 'LossMLHMultiplier': 0.1,
  'LossUNCMultiplier': 0.01, 'LossUncertaintyPolicyMultiplier': 0, 'LossValueDMultiplier': 0,
  'TorchSeed': 777, 'TPGV3': 0, 'AuxFeaturesPerSquare': 0,
  'SoftPolicyWeight': 2.0, 'SoftPolicyTemp': 4.0,
}
DATA_BASE = {'SourceType': 'DirectFromPositionGenerator', 'TrainingFilesDirectory': '/none', 'FractionQ': 1,
             'WDLLabelSmoothing': 0}
EXEC_BASE = {'ID': 'bt6h_test', 'DeviceType': 'cpu', 'DeviceIDs': [0], 'DataType': 'Float32',
             'UseHistory': True, 'DropoutRate': 0, 'EngineType': 'CSharpViaTorchscript'}


def make_config(net_over=None, opt_over=None, tag='bt6h_test', data_over=None):
  from config import Configuration
  d = tempfile.mkdtemp(prefix='bt6h_')
  try:
    for suf, obj in (('net', {**NET_SMALL, **(net_over or {})}), ('opt', {**OPT_BASE, **(opt_over or {})}),
                     ('data', {**DATA_BASE, **(data_over or {})}), ('exec', {**EXEC_BASE, 'ID': tag}), ('monitoring', {})):
      with open(os.path.join(d, f'{tag}_ceres_{suf}.json'), 'w') as f:
        json.dump(obj, f)
    return Configuration(d, tag)
  finally:
    shutil.rmtree(d, ignore_errors=True)


def build(net_over=None, opt_over=None, quiet=True):
  from ceres_net import CeresNet
  cfg = make_config(net_over, opt_over)
  torch.manual_seed(int(OPT_BASE['TorchSeed']))
  sink = io.StringIO()
  stdout = sys.stdout
  if quiet:
    sys.stdout = sink
  try:
    net = CeresNet(None, cfg,
                   policy_loss_weight=cfg.Opt_LossPolicyMultiplier, value_loss_weight=cfg.Opt_LossValueMultiplier,
                   moves_left_loss_weight=cfg.Opt_LossMLHMultiplier, unc_loss_weight=cfg.Opt_LossUNCMultiplier,
                   value2_loss_weight=cfg.Opt_LossValue2Multiplier, q_deviation_loss_weight=cfg.Opt_LossQDeviationMultiplier,
                   value_diff_loss_weight=cfg.Opt_LossValueDMultiplier, value2_diff_loss_weight=cfg.Opt_LossValue2DMultiplier,
                   action_loss_weight=cfg.Opt_LossActionMultiplier, uncertainty_policy_weight=cfg.Opt_LossUncertaintyPolicyMultiplier,
                   action_uncertainty_loss_weight=cfg.Opt_LossActionUncertaintyMultiplier, q_ratio=cfg.Data_FractionQ)
  finally:
    sys.stdout = stdout
  return net


# --------------------------------------------------------------------------- records from FENs
def frame_board(fen):
  """64 piece codes of the position in the frame of the side to move (0-5 ours P..K, 6-11 theirs, 12 empty)."""
  squares, side, castle, ep, halfmove = parse_fen(fen)
  out = [12] * 64
  for s in range(64):
    piece = squares[s ^ 56 if side == 1 else s]
    if piece != 12:
      colour, kind = divmod(piece, 6)
      out[s] = (0 if colour == side else 6) + kind
  return out, side, castle, halfmove


def record_from_fen(fen, previous_board=None, zero_history=False):
  """Our 137-float square record of a FEN position (TPGSquareRecord layout). Missing history is filled with copies of
  the position, as the TPG writer and v6_dataset do; `previous_board` (frame codes) replaces history position 1;
  `zero_history` leaves the history bytes empty (a record written without history)."""
  board, side, castle, halfmove = frame_board(fen)
  rec = np.zeros((64, 137), dtype=np.float32)
  for h in range(8):
    position = previous_board if (h == 1 and previous_board is not None) else board
    if h > 0 and zero_history:
      continue
    for s in range(64):
      rec[s, h * 13 + (0 if position[s] == 12 else 1 + position[s])] = 1.0
  us, them = side, side ^ 1
  rec[:, 112] = (castle >> (us * 2)) & 1
  rec[:, 113] = (castle >> (us * 2 + 1)) & 1
  rec[:, 114] = (castle >> (them * 2)) & 1
  rec[:, 115] = (castle >> (them * 2 + 1)) & 1
  rec[:, 116] = min(halfmove, 100) / 50.0
  for s in range(64):
    rec[s, 121 + s // 8] = 1.0
    rec[s, 129 + s % 8] = 1.0
  return rec


def fens(count):
  return read_fens(FEN_FILE, count) if os.path.exists(FEN_FILE) else (FALLBACK_FENS * (count // len(FALLBACK_FENS) + 1))[:count]


def records(count, with_previous=True):
  """Records of real positions; history position 1 is another real position (dense gained / lost)."""
  f = fens(count)
  boards = [frame_board(x)[0] for x in f]
  recs = [record_from_fen(x, boards[(i + 1) % count] if with_previous else None) for i, x in enumerate(f)]
  return torch.from_numpy(np.stack(recs))


def batch_for(B, seed=5):
  g = torch.Generator().manual_seed(seed)
  pol = torch.zeros(B, 1858)
  for b in range(B):
    idx = torch.randperm(1858, generator=g)[:10]
    w = torch.rand(10, generator=g) + 0.05
    pol[b, idx] = w / w.sum()
  wdl = torch.tensor([[0.4, 0.3, 0.3]]).expand(B, 3).clone()
  return {'policies': pol, 'wdl_deblundered': wdl, 'wdl_q': wdl, 'wdl_nondeblundered': wdl,
          'mlh': torch.rand(B, 1, generator=g), 'unc': torch.rand(B, 1, generator=g) * 0.1,
          'uncertainty_policy': torch.zeros(B, 1),
          'q_deviation_lower': torch.zeros(B, 1), 'q_deviation_upper': torch.zeros(B, 1)}


def run_loss(m, batch, sq):
  from losses import LossCalculator
  outs = m(sq, None)
  (policy_out, value_out, mlh_out, unc_out, value2_out, qdl, qdu, unc_pol, action_out, _) = outs[:10]
  return m.compute_loss(LossCalculator(nn.Linear(4, 4)), batch, policy_out, value_out, mlh_out, unc_out, value2_out,
                        qdl, qdu, unc_pol, None, None, None, action_out, None, 0, 0, 0, False)


def randomise(net, seed=3, scale=0.3):
  """Non-trivial values for the zero-initialised reader tables, gates and site maps (so that every path is live)."""
  from bt6_hybrid import is_bt6_layer_reader
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for name, p in net.named_parameters():
      live = (is_bt6_layer_reader(name) or name.startswith('bt6h.sites') or name.endswith('.mix') or '.g.' in name
              or name.endswith(('.up.weight', 'mult_gate', 'position')))
      if live:
        noise = torch.randn(p.shape, generator=g) * scale
        if name.endswith('.t_node'):
          p.copy_(0.5 + noise.abs())
        elif name.endswith('mult_gate'):
          p.copy_(1.0 + noise)
        elif name.endswith(('.g.weight', '.up.weight')):
          p.copy_(noise * 0.2 / max(1.0, p.shape[1] ** 0.5) * 3)
        else:
          p.copy_(noise)


# --------------------------------------------------------------------------- 1. relations from the record
def test_relations_from_record():
  from bt6_hybrid import pieces_from_squares
  from bt6_relations import BT6Relations
  rel = BT6Relations()
  n = 256
  f = fens(n)
  masks, values = encode_fens(f)
  planes = expand_planes(masks, values)
  lc0_pieces = torch.from_numpy(planes[:, 0:12] > 0.5)
  # (a) FEN-only: the record's copy-filled history against lc0's planes for the same positions. lc0's engine fill
  #     puts an e.p. pawn back in the copies; our record's copy is the position itself -> compare on current pieces,
  #     and on the previous position only where the two fills agree (no e.p. square)
  rec = torch.from_numpy(np.stack([record_from_fen(x) for x in f]))
  pieces, previous = pieces_from_squares(rec)
  assert torch.equal(pieces, lc0_pieces), 'current pieces: record bytes 1..12 != lc0 planes 0..11'
  E = rel(pieces, previous)
  assert int(E[:, 13:17].sum()) == 0, 'copy-filled history must give empty gained / lost'
  assert torch.equal(E[:, :13], rel(lc0_pieces, lc0_pieces)[:, :13])
  # (b) a real previous position (another board): record bytes 14..25 against planes built from the same boards
  recp = records(n)
  pieces_p, previous_p = pieces_from_squares(recp)
  expected_previous = torch.roll(lc0_pieces, -1, dims=0)
  assert torch.equal(pieces_p, lc0_pieces) and torch.equal(previous_p, expected_previous)
  Ep = rel(pieces_p, previous_p)
  assert torch.equal(Ep, rel(lc0_pieces, expected_previous)) and int(Ep[:, 13:17].sum()) > 1000
  # (c) a record without history bytes: treated as "nothing changed", not as "everything gained"
  rec0 = torch.from_numpy(np.stack([record_from_fen(x, zero_history=True) for x in f[:32]]))
  p0, q0 = pieces_from_squares(rec0)
  assert torch.equal(q0, p0) and int(rel(p0, q0)[:, 13:17].sum()) == 0
  assert int(rel(p0, None)[:, 13:15].sum()) > 0   # what BT6 itself sees with zero planes: gained = all attacks
  print(f'1. relations from the record: {n} positions, bytes 1..12 / 14..25 == lc0 planes 0..11 / 13..24 as boards; '
        f'copy-filled history and missing history bytes -> gained / lost empty ({int(Ep[:, 13:17].sum())} set bits with real previous boards)')


# --------------------------------------------------------------------------- 2. equivalence with bt6_net
def test_equivalence():
  from bt6_hybrid import export_served_hybrid, forward_served_hybrid, served_meta
  from bt6_net import BT6EdgeSite, BT6Layer, served_to_torch
  net = build({'NumLayers': 3, 'BT6EdgeSites': [0, 1]}).eval()
  randomise(net)
  sq = records(6)
  # (a) stand-alone bt6_net layers / sites with the same weights, fed what the hybrid fed its own
  captured = []
  hooks = []
  for b, block in enumerate(net.transformer_layer):
    for j, layer in enumerate(block.inner):
      hooks.append(layer.register_forward_hook(lambda m, i, o, key=('layer', b, j): captured.append((key, m, i, o))))
  for key, site in net.bt6h.sites.items():
    hooks.append(site.register_forward_hook(lambda m, i, o, key=('site', key): captured.append((key, m, i, o))))
  with torch.no_grad():
    out = net(sq, None)
  for h in hooks:
    h.remove()
  cfg = net.bt6h.layer_config
  worst = 0.0
  n_layers = n_sites = 0
  last_H = {}
  for key, module, inputs, output in captured:
    if key[0] == 'layer':
      twin = BT6Layer(cfg)
      twin.load_state_dict(module.state_dict())
      with torch.no_grad():
        x2, H2 = twin(*inputs)
      assert torch.equal(x2, output[0]) and torch.equal(H2, output[1]), key
      last_H[key[1]] = output[1]
      n_layers += 1
    else:
      twin = BT6EdgeSite(cfg)
      twin.load_state_dict(module.state_dict())
      with torch.no_grad():
        e2 = twin(*inputs)
      assert torch.equal(e2, output), key
      # the site read the logits of the LAST inner layer of its block
      assert torch.equal(inputs[1], last_H[int(key[1])]), 'site must read H of the last inner layer of its block'
      n_sites += 1
  assert n_layers == 6 and n_sites == 2
  # the edge state really changes at a site and is what later layers read
  layer_inputs = {key[1:]: inputs for key, _, inputs, _ in captured if key[0] == 'layer'}
  assert torch.equal(layer_inputs[(0, 0)][3], layer_inputs[(0, 1)][3])
  assert not torch.equal(layer_inputs[(0, 1)][3], layer_inputs[(1, 0)][3]), 'state must be updated after block 0'
  assert not torch.equal(layer_inputs[(1, 1)][3], layer_inputs[(2, 0)][3]), 'state must be updated after block 1'
  # (b) the whole forward against the kernel-form forward on the export. The export stores dense weights in FP16,
  #     so the net is first given FP16-representable weights (the FP32 folds are what is under test).
  with torch.no_grad():
    for name, p in net.named_parameters():
      from bt6_hybrid import is_bt6_layer_reader
      if not (is_bt6_layer_reader(name) or name.startswith(('bt6h.prologue', 'bt6h.sites'))):
        p.copy_(p.half().float())
    out = net(sq, None)
  served = served_to_torch(export_served_hybrid(net))
  ref = forward_served_hybrid(served, sq, served_meta(net))
  pairs = {'policy': out[0], 'value': out[1], 'mlh': out[2], 'unc': out[3], 'value2': out[4]}
  for name, tensor in pairs.items():
    diff = float((tensor.double() - ref[name].double()).abs().max())
    scale = max(1.0, float(tensor.abs().max()))
    worst = max(worst, diff / scale)
    assert diff < 2e-4 * scale, (name, diff, scale)
  print(f'2. equivalence: {n_layers} inner layers and {n_sites} edge sites == stand-alone bt6_net modules with the same '
        f'weights (bitwise); whole forward vs the kernel-form forward on the export: max relative diff {worst:.2e} '
        f'over policy / value / value2 / unc / mlh ({len(served)} served tensors)')


# --------------------------------------------------------------------------- 3. gradient checkpoint
def test_checkpoint():
  sq = records(4)
  batch = batch_for(4)
  results = {}
  for flag in (False, True):
    net = build({'NumLayers': 3, 'BT6EdgeSites': [0, 1], 'BT6CheckpointBlocks': flag}).train()
    randomise(net)
    outs = net(sq, None)
    net._last_sp_out = None
    net.zero_grad()
    loss = run_loss(net, batch, sq)
    loss.backward()
    results[flag] = ([o.detach().clone() for o in outs[:5]], float(loss),
                     {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None})
  for a, b in zip(results[False][0], results[True][0]):
    assert torch.equal(a, b), 'forward must be bit-identical with the checkpoint on and off'
  assert results[False][1] == results[True][1]
  ga, gb = results[False][2], results[True][2]
  assert ga.keys() == gb.keys()
  worst = max(float((ga[n] - gb[n]).abs().max() / (ga[n].abs().max() + 1e-12)) for n in ga)
  assert worst < 1e-4, worst
  print(f'3. checkpoint per block: forward bit-identical on / off, loss equal, {len(ga)} gradients equal '
        f'(max relative difference {worst:.1e})')


# --------------------------------------------------------------------------- 4. optimiser scopes and participation
def test_scopes_and_participation():
  from bt6_hybrid import bt6_is_ffn_kernel, bt6_muon_excluded, is_bt6_layer_reader, parameter_group_table
  from wd_partition import partition_weight_decay
  net = build().train()
  decay, no_decay = partition_weight_decay(net)      # asserts completeness itself
  readers = [n for n, _ in net.named_parameters() if is_bt6_layer_reader(n)]
  assert readers and all(n in no_decay for n in readers)
  assert len(readers) == 10 * sum(len(b.inner) for b in net.transformer_layer)
  for n, p in net.named_parameters():
    if n.startswith('bt6h.'):
      is_kernel = n.endswith('.weight') and p.ndim == 2 and '.ln' not in n
      assert (n in decay) == is_kernel, (n, n in decay)
      assert bt6_muon_excluded(n)
  assert all(n in decay for n, _ in net.named_parameters() if bt6_is_ffn_kernel(n))
  for scope in ('all-non-trunk', 'ffn-only'):
    rows = parameter_group_table(net, scope)
    muon = [r for r in rows if r[0] == 'Muon']
    assert muon and all('reader' not in r[2] and 'gate projection' not in r[2] and 'bt6' not in r[2].lower() for r in muon)
    if scope == 'ffn-only':
      assert all('GLU' in r[2] for r in muon)
  # every parameter must receive a gradient from the real loss path (DDP participation)
  randomise(net)
  sq, batch = records(4), batch_for(4)
  loss = run_loss(net, batch, sq)
  loss.backward()
  missing = [n for n, p in net.named_parameters() if p.grad is None]
  zero = [n for n, p in net.named_parameters() if p.grad is not None and float(p.grad.abs().max()) == 0.0]
  assert not missing, f'parameters without gradient: {missing}'
  assert not zero, f'parameters with an all-zero gradient: {zero}'
  print(f'4. scopes: partition complete ({len(decay)} decay / {len(no_decay)} no_decay tensors), {len(readers)} reader '
        f'tables and scales in no_decay and out of Muon, bt6h.* out of Muon, GLU kernels under ffn-only; '
        f'all {len(list(net.parameters()))} parameter tensors receive a gradient from the loss')


# --------------------------------------------------------------------------- 5. autocast
def test_autocast():
  net = build({'NumLayers': 3, 'BT6EdgeSites': [0, 1]}).eval()
  randomise(net)
  sq = records(4)
  seen = {}
  def layer_hook(m, i, o):
    seen.setdefault('H', set()).add(o[1].dtype)
    seen.setdefault('x', set()).add(o[0].dtype)
    seen.setdefault('e_in', set()).add(i[3].dtype)
    seen.setdefault('E', set()).add(i[1].E.dtype)
  hooks = [layer.register_forward_hook(layer_hook) for block in net.transformer_layer for layer in block.inner]
  hooks += [site.register_forward_hook(lambda m, i, o: seen.setdefault('e_out', set()).add(o.dtype))
            for site in net.bt6h.sites.values()]
  with torch.no_grad():
    with torch.autocast('cpu', dtype=torch.bfloat16):
      out = net(sq, None)
  for h in hooks:
    h.remove()
  with torch.no_grad():
    ref = net(sq, None)
  f32, bf16 = {torch.float32}, {torch.bfloat16}
  # H leaves the layer in the ASSEMBLY dtype (bf16 under autocast with BT6LogitAssembly 'autocast'); the softmax and
  # the gates inside attention_core are float32 (checked by test 9's comparison with the float32 reference)
  assert seen['H'] == bf16 and seen['e_in'] == f32 and seen['e_out'] == f32 and seen['E'] == f32, seen
  assert seen['x'] == bf16, f'the inner stream should be bf16 under autocast, saw {seen["x"]}'
  assert out[0].dtype == torch.float32, 'policy logits are assembled in float32'
  rel = max(float((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-6)) for a, b in zip(out[:5], ref[:5]))
  assert all(torch.isfinite(o.float()).all() for o in out[:5]) and rel < 0.15, rel
  print(f'5. bf16 autocast (CPU): E, edge state in and out of sites float32, H (logit assembly) bf16; inner stream {sorted(str(d) for d in seen["x"])}; '
        f'outputs finite, max relative deviation from the float32 forward {rel:.3f}')


# --------------------------------------------------------------------------- 6. config refusals
def test_config_refusals():
  cases = [
    ({'UseMoveTokens': True}, {}, 'MoveTokens'),
    ({'MoveTokenDim': 96}, {}, 'MoveToken'),
    ({}, {'LossMoveTokenSoftPolicyMultiplier': 0.5}, 'MoveToken'),
    ({}, {'LossMoveTokenValueOrderMultiplier': 0.5}, 'MoveToken'),
    ({}, {'LearningRateDecoderRatio': 0.5}, 'ecoder'),
    ({'NBTSharedSmolgen': True}, {}, ''),
    ({'SmolgenDimPerSquare': 8, 'SmolgenDim': 32}, {}, 'smolgen'),
    ({'BT6EdgeSites': [1]}, {}, 'BT6EdgeSites'),
    ({'EGTEdgeStream': True, 'EGTEdgeSites': [0]}, {}, ''),
    ({'NBTInnerLayers': 0}, {}, ''),
    ({'BT6Hybrid': False}, {}, 'BT6Hybrid is off'),
  ]
  refused = 0
  for net_over, opt_over, needle in cases:
    try:
      make_config(net_over, opt_over)
    except (ValueError, AssertionError) as error:
      assert needle in str(error), (net_over, opt_over, str(error))
      refused += 1
    else:
      raise AssertionError(f'config accepted: {net_over} {opt_over}')
  print(f'6. config: {refused} decoder-bound / incompatible settings refused with a message')
  net8 = {k: v for k, v in NET_SMALL.items() if k != 'BT6EdgeSites'}
  from config import Configuration  # default site schedule for 8 blocks
  d = tempfile.mkdtemp(prefix='bt6h_')
  try:
    for suf, obj in (('net', {**net8, 'NumLayers': 8}), ('opt', OPT_BASE), ('data', DATA_BASE), ('exec', EXEC_BASE), ('monitoring', {})):
      with open(os.path.join(d, f'x_ceres_{suf}.json'), 'w') as f:
        json.dump(obj, f)
    assert Configuration(d, 'x').NetDef_BT6EdgeSites == [0, 2, 3, 4, 6]
    with open(os.path.join(d, 'x_ceres_net.json'), 'w') as f:
      json.dump({**net8, 'NumLayers': 10}, f)
    assert Configuration(d, 'x').NetDef_BT6EdgeSites == [0, 2, 4, 6, 8]
  finally:
    shutil.rmtree(d, ignore_errors=True)
  assert make_config({'NBTInnerWidth': 56, 'NBTInnerHeads': 7}).NetDef_NBTMidDim == 56   # e.g. 2048 / 896 at scale


# --------------------------------------------------------------------------- 7. save / reload
def test_save_reload():
  net = build().eval()
  randomise(net)
  sq = records(4)
  buffer = io.BytesIO()
  torch.save(net.state_dict(), buffer)
  buffer.seek(0)
  twin = build().eval()
  twin.load_state_dict(torch.load(buffer))
  with torch.no_grad():
    a, b = net(sq, None), twin(sq, None)
  assert all(torch.equal(x, y) for x, y in zip(a[:5], b[:5]))
  linear = build({'BT6Embedding': 'linear'}).eval()
  with torch.no_grad():
    assert linear(sq, None)[0].shape == (4, 1858)
  assert linear.embedding_layer is not None and linear.bt6h.embed is None and net.embedding_layer is None
  no_soft = build(opt_over={'SoftPolicyWeight': 0})
  assert no_soft.bt6h.soft_policy is None and net.bt6h.soft_policy is not None
  print('7. state_dict save / reload: identical outputs; BT6Embedding linear builds and runs; the soft-policy head exists '
        'only when SoftPolicyWeight > 0')


# --------------------------------------------------------------------------- 9. fast layer form vs the literal form
def test_fast_layer_vs_reference():
  """BT6Layer.forward (matmul form on the shared relation context) against forward_reference (the literal einsum
  form of BT6_SPEC A3): outputs and ALL parameter / input gradients (float32); then the size of the bf16
  logit-assembly rounding under autocast."""
  net = build({'NumLayers': 3, 'BT6EdgeSites': [0, 1]}).train()
  randomise(net)
  sq = records(4)
  hybrid = net.bt6h
  layer = net.transformer_layer[1].inner[0]
  torch.manual_seed(11)
  x = torch.randn(4, 64, hybrid.mid_dim)
  results = {}
  for name, fn in (('fast', layer.forward), ('reference', layer.forward_reference)):
    ctx, S, e = hybrid.start(sq)          # a fresh graph per form (both backward through the prologue)
    xin = x.clone().requires_grad_(True)
    ein = e.detach().clone().requires_grad_(True)
    out, H = fn(xin, ctx, S, ein, hybrid.prologue)
    net.zero_grad()
    (out.pow(2).mean() + H.pow(2).mean()).backward()
    grads = {n: p.grad.detach().clone() for n, p in layer.named_parameters() if p.grad is not None}
    grads.update({'x': xin.grad.clone(), 'e': ein.grad.clone(),
                  **{f'prologue.{n}': p.grad.detach().clone() for n, p in hybrid.prologue.named_parameters() if p.grad is not None}})
    results[name] = (out.detach(), H.detach(), grads)
  rel = lambda a, b: float((a - b).abs().max() / (b.abs().max() + 1e-12))
  d_out, d_H = rel(results['fast'][0], results['reference'][0]), rel(results['fast'][1], results['reference'][1])
  ga, gb = results['fast'][2], results['reference'][2]
  assert ga.keys() == gb.keys() and len(ga) > 20
  d_grad = max(rel(ga[n], gb[n]) for n in ga)
  assert d_out < 2e-5 and d_H < 2e-5 and d_grad < 2e-3, (d_out, d_H, d_grad)
  # the custom autograd function of the attention core: numerical gradient check in float64 on a small case, with
  # and without the gate cap, with H used downstream (as at an edge site) and unused
  from bt6_net import attention_core
  torch.manual_seed(5)
  for cap in (False, True):
    L = torch.randn(2, 3, 6, 6, dtype=torch.float64, requires_grad=True)
    door = (0.3 * torch.randn(2, 3, 6, 6, dtype=torch.float64)).requires_grad_(True)
    gl = torch.randn(2, 3, 6, 6, dtype=torch.float64, requires_grad=True)
    vals = torch.randn(2, 3, 6, 4, dtype=torch.float64, requires_grad=True)
    og = torch.randn(2, 3, 6, 4, dtype=torch.float64, requires_grad=True)
    for with_gate in (True, False):
      extra = (og,) if with_gate else ()
      fn = lambda L, door, gl, vals, *og: attention_core(L, door, gl, vals, cap, og[0] if og else None, 2.0, custom=True)
      assert torch.autograd.gradcheck(fn, (L, door, gl, vals) + extra, eps=1e-6, atol=1e-6)
      assert torch.autograd.gradcheck(lambda *a: fn(*a)[0], (L, door, gl, vals) + extra, eps=1e-6, atol=1e-6)
      assert torch.autograd.gradcheck(lambda *a: fn(*a)[1], (L, door, gl, vals) + extra, eps=1e-6, atol=1e-6)
      o1, h1 = attention_core(L, door, gl, vals, cap, og if with_gate else None, 2.0, custom=True)
      o2, h2 = attention_core(L, door, gl, vals, cap, og if with_gate else None, 2.0, custom=False)
      assert torch.allclose(o1, o2, atol=1e-12) and torch.allclose(h1, h2, atol=1e-12)
      g1 = torch.autograd.grad((o1.pow(2).sum() + h1.sin().sum()), (L, door, gl, vals) + extra)
      g2 = torch.autograd.grad((o2.pow(2).sum() + h2.sin().sum()), (L, door, gl, vals) + extra)
      assert all(torch.allclose(a, b, atol=1e-10) for a, b in zip(g1, g2))
  # bf16 assembly vs float32 assembly under autocast, whole net
  net.eval()
  with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
    a = net(sq, None)
    for block in net.transformer_layer:
      for l in block.inner:
        l.cfg.assembly_autocast = False
    b = net(sq, None)
    for block in net.transformer_layer:
      for l in block.inner:
        l.cfg.assembly_autocast = True
  d_assembly = max(float((u.float() - w.float()).abs().max() / (w.float().abs().max() + 1e-6)) for u, w in zip(a[:5], b[:5]))
  print(f'9. attention_core: float64 gradcheck passes (with / without cap and output gate; out only, H only, both), custom backward == autograd; '
        f'fast layer form vs the literal einsum form: output {d_out:.1e}, H {d_H:.1e}, {len(ga)} gradients {d_grad:.1e} '
        f'(max relative, float32); under bf16 autocast the '
        f'bf16 logit assembly changes the outputs by {d_assembly:.3f} relative to float32 assembly (same autocast run)')


# --------------------------------------------------------------------------- 8. a few optimiser steps on a fixed batch
def test_fixed_batch_steps():
  net = build({'BT6CheckpointBlocks': True}).train()
  sq, batch = records(16), batch_for(16)
  optimizer = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.0)
  losses = []
  for _ in range(12):
    optimizer.zero_grad()
    loss = run_loss(net, batch, sq)
    assert torch.isfinite(loss), loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    losses.append(float(loss.detach()))
  assert losses[-1] < 0.8 * losses[0] and all(np.isfinite(losses)), losses
  moved = [n for n, p in net.named_parameters() if n.endswith(('.C1', '.Pq', '.C2', '.t_edge', '.Wm', '.Wg', 'readback'))
           and float(p.detach().abs().max()) > 0]
  print(f'8. fixed batch, 12 AdamW steps through compute_loss (checkpointing on): loss {losses[0]:.3f} -> {losses[-1]:.3f}, all '
        f'finite; {len(moved)} zero-initialised relation / edge tensors have left zero')


# --------------------------------------------------------------------------- tables / cost (on request)
def tables():
  from bt6_hybrid import parameter_group_table
  count = lambda module: sum(p.numel() for p in module.parameters()) if module is not None else 0
  shapes = [('1536 outer / 768 inner (24 x 32) / 10 x 3, policy width 1024', NET_TARGET, ('bt6', 'linear')),
            ('1536 outer / 768 inner (24 x 32) / 10 x 3, policy width 512', {**NET_TARGET, 'BT6PolicyWidth': 512}, ('bt6', 'linear'))]
  if 'all' in sys.argv:
    shapes.append(('2048 outer / 1024 inner (32 x 32) / 8 x 3, policy width 1024', NET_2048, ('bt6', 'linear')))
  for label, net_def, embeddings in shapes:
    for embedding in embeddings:
      net = build({**net_def, 'BT6Embedding': embedding})
      blocks = list(net.transformer_layer)
      inner = [l for b in blocks for l in b.inner]
      readers = sum(p.numel() for l in inner for n, p in l.named_parameters() if n in
                    ('t_node', 't_edge', 'C1', 'Pk', 'Pq', 'C2', 'We', 'Wm', 'Wg', 'bg'))
      rows = [
        ('BT6 inner layers', f'{len(inner)} x {count(inner[0]):,}', sum(count(l) for l in inner)),
        ('  of which relation / pair / edge reader tables', f'{len(inner)} x {readers // len(inner):,}', readers),
        ('NBT down + up projections', f'{len(blocks)} x {count(blocks[0].down) + count(blocks[0].up):,}',
         sum(count(b.down) + count(b.up) for b in blocks)),
        ('outer norms (norm_in, norm_out, trunk end)', '', sum(count(b.norm_in) + count(b.norm_out) for b in blocks) + count(net.trunk_end_norm)),
        ('prologue (D, T, Pin, Toff)', '', count(net.bt6h.prologue)),
        ('edge sites', f'{len(net.bt6h.sites)} x {count(next(iter(net.bt6h.sites.values()))):,}', count(net.bt6h.sites)),
        (f'embedding ({embedding})', '', count(net.bt6h.embed) + count(net.embedding_layer) + count(net.embedding_norm)),
        ('BT6 policy head', f'width {net.config.NetDef_BT6PolicyWidth}', count(net.bt6h.policy)),
        ('soft-policy head (training-only)', f'width {net.config.NetDef_BT6SoftPolicyWidth}', count(net.bt6h.soft_policy)),
        ('shared head front end (premap + shared linear)', '', count(net.headPremap) + count(net.headSharedLinear)),
      ]
      listed = sum(r[2] for r in rows if not r[0].startswith('  '))
      total = count(net)
      rows.append(('value / value2 / unc / mlh and other heads', '', total - listed))
      print(f'\n=== {label}, embedding {embedding}: {total:,} parameters')
      for name, detail, n in rows:
        print(f'  {name:52s} {detail:22s} {n:>13,}')
      if embedding == 'bt6' and net.config.NetDef_BT6PolicyWidth == 1024:
        for scope in ('all-non-trunk', 'ffn-only'):
          table = parameter_group_table(net, scope)
          print(f'  -- optimiser groups, MuonAdamWScope {scope!r}:')
          print(f'     {"group":6s} {"wd":9s} {"family":64s} {"tensors":>8s} {"parameters":>14s}')
          for group, wd, fam, tensors, n in table:
            print(f'     {group:6s} {wd:9s} {fam:64s} {tensors:8d} {n:14,}')
          for group in ('Muon', 'AdamW'):
            print(f'     {group:6s} total: {sum(r[3] for r in table if r[0] == group)} tensors, '
                  f'{sum(r[4] for r in table if r[0] == group):,} parameters')
      del net


def cost():
  from bt6_hybrid import BT6Hybrid
  hybrid = BT6Hybrid(1536, 768, 24, 10, 3, [0, 2, 4, 6, 8], policy_width=1024)
  sq = records(1024)
  with torch.no_grad():
    hybrid.start(sq[:64])
    for label, fn in (('BT6Relations (bool E)', lambda: hybrid.relations(*__import__('bt6_hybrid').pieces_from_squares(sq))),
                      ('relations + float32 E + prologue (S, e0)', lambda: hybrid.start(sq))):
      best = 1e9
      for _ in range(3):
        started = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - started)
      print(f'cost: {label}: {best * 1e3:.0f} ms per 1,024 positions on CPU ({torch.get_num_threads()} threads)')
    E, S, e = hybrid.start(sq)
    E = E.E
  mb = lambda t: t.numel() * t.element_size() / 2**20
  print(f'cost: per 1,024 positions: E float32 {tuple(E.shape)} {mb(E):.0f} MiB (bool {mb(E) / 4:.0f} MiB), S {tuple(S.shape)} '
        f'{mb(S):.0f} MiB, e0 {tuple(e.shape)} {mb(e):.0f} MiB; one pairwise logit tensor [B, 24, 64, 64] float32 '
        f'{1024 * 24 * 4096 * 4 / 2**20:.0f} MiB')
  print(f'cost: at forward batch 1536: E {mb(E) * 1.5 / 1024:.2f} GiB, e {mb(e) * 1.5 / 1024:.2f} GiB per state version, '
        f'pairwise tensor {1536 * 24 * 4096 * 4 / 2**30:.2f} GiB')


if __name__ == '__main__':
  mode = sys.argv[1] if len(sys.argv) > 1 else 'tests'
  if mode == 'table':
    tables()
  elif mode == 'cost':
    cost()
  else:
    test_relations_from_record()
    test_equivalence()
    test_checkpoint()
    test_scopes_and_participation()
    test_autocast()
    test_config_refusals()
    test_save_reload()
    test_fixed_batch_steps()
    test_fast_layer_vs_reference()
    print('test_bt6_hybrid: OK')
