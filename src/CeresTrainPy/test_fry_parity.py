"""Fry (fry-final-117) in our trainer against the author's ONNX file. CPU, float32 on our side.

Our CeresNet is built from fry_parity.FRY_NET (768 x 28, 70 tokens, square features, 24-on-12 grouped-query attention
with q/k width 8 / 16 / 24, 8 templates, input skip in every layer, his policy and value heads), the released
checkpoint's tensors are copied onto it (fry_parity.load_fry_checkpoint) and the outputs are compared with the ONNX
(FP16 trunk, residual stream scaled by 1/16) in onnxruntime on the verification positions, stage by stage so that
a disagreement is localised:

  0. tables: the 1968-move table == the checkpoint's integer buffers; the checkpoint is consumed completely
  1. embedding output (token + position + square features)
  2. the input of every layer (after the a * x + b * x0 mix) and the trunk-end norm output
  3. heads: the 128 value logits, the value (expected score), the policy logits of every legal move after the index
     mapping (our 1858 layout <- his 1968)
  4. the same with the tokens built from OUR RECORD inside forward (no token override): equal for White to move;
     Black to move differs by the colour token only (the record does not carry it) -- measured, not hidden

  python test_fry_parity.py [n_positions]         (default: all 450 verification positions)

Needs: the ONNX, the checkpoint tensors (npz) and the verification set; paths below.
"""
import os
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fry_parity as FP
from fry_testlib import fry_move_table, fry_tokens, legal_move_indices, record_from_board

ONNX_PATH = os.environ.get('FRY_ONNX', '/mnt/c/Dev/Chess/Networks/Fry/fry-final-117.fp16.onnx')
NPZ_PATH = os.environ.get('FRY_NPZ', '/mnt/c/Dev/Chess/fry_review/release/fry_sd.npz')
VERIFY_PATH = os.environ.get('FRY_VERIFY', '/mnt/c/Dev/Chess/fry_review/verify_set.npz')
STREAM_SCALE = 16.0          # the ONNX carries the residual stream at 1/16


def to_float32(model):
  """The same graph with every FLOAT16 tensor and cast made FLOAT: the FP16-rounded WEIGHTS stay as they are, the
  activations are no longer rounded to FP16. A second oracle that separates 'our forward differs' from 'the FP16
  file rounds': stream channels reach |x| ~ 6000 in training units, where one FP16 step of the 1/16-scaled stream is 4."""
  import onnx
  from onnx import numpy_helper, TensorProto
  def conv(t):
    if t.data_type == TensorProto.FLOAT16:
      t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
  for init in model.graph.initializer:
    conv(init)
  for node in model.graph.node:
    for a in node.attribute:
      if a.type == onnx.AttributeProto.TENSOR:
        conv(a.t)
      if node.op_type in ('Cast', 'ConstantOfShape') and a.name == 'to' and a.i == TensorProto.FLOAT16:
        a.i = TensorProto.FLOAT
  del model.graph.value_info[:]
  return model


def round_like_the_file(net):
  """Give our net the weights the ONNX file actually holds: trunk weights, biases, skip scalars, embeddings and
  tables rounded to FP16 (the stream-writing ones -- embeddings, square-feature table, attention output projection,
  FFN down projection -- after the file's 1/16 scaling), the norm gains too (FP32 tensors in the file, but holding
  FP16-rounded values, the trunk-end norm included), heads untouched (FP32 in the file, equal to the checkpoint); the
  768-wide norms' epsilon as the file computes it (1e-5 on the 1/16 stream = 2.56e-3 in training units); and the
  attention logit divisor sqrt(q/k width) rounded to FP16 as the file's constant is."""
  def r16(t, scaled):
    k = 16.0 if scaled else 1.0
    return (t / k).half().float() * k
  with torch.no_grad():
    for name, p in net.named_parameters():
      if name.startswith(('fry_policy.', 'hlg_head.')):
        continue          # heads: FP32 in the file, equal to the checkpoint
      if name.startswith(('fry_input.', 'square_features.')) or '.attention.W_h.' in name or name.endswith('mlp.linear2.weight'):
        p.copy_(r16(p, True))
      elif name.startswith(('transformer_layer.', 'attn_templates.', 'input_skip_', 'trunk_end_norm.')):
        p.copy_(r16(p, False))   # includes every norm gain: stored as FP32 in the file, but with FP16-rounded values
  for layer in net.transformer_layer:
    layer.ln1.eps = layer.ln2.eps = 1e-5 * 256
    # the file's logit divisor is an FP16 constant: 2.828125 for sqrt(8), 4.8984375 for sqrt(24) (1.1e-4 off)
    layer.attention.qk_div = float(np.float16(layer.attention.qk_div))
  net.trunk_end_norm.eps = 1e-5 * 256


def onnx_session(extra_outputs, float32=False):
  import onnx
  import onnxruntime as ort
  model = onnx.load(ONNX_PATH)
  if float32:
    model = to_float32(model)
  have = {o for n in model.graph.node for o in n.output}
  for name in extra_outputs:
    assert name in have, f'ONNX has no tensor {name}'
    model.graph.output.append(onnx.ValueInfoProto(name=name))
  so = ort.SessionOptions()
  so.log_severity_level = 3
  return ort.InferenceSession(model.SerializeToString(), so, providers=['CPUExecutionProvider'])


def stat(name, ours, theirs, scale_note=''):
  ours, theirs = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
  d = np.abs(ours - theirs)
  rms = float(np.sqrt((theirs ** 2).mean()))
  k = np.unravel_index(d.argmax(), d.shape)
  rel = float((d / np.maximum(np.abs(theirs), rms)).max())      # error relative to the element (or the rms if larger)
  print(f'   {name:30s} max |diff| {d.max():.5f} (reference there {theirs[k]:+.3f})   mean |diff| {d.mean():.6f}   '
        f'rms of reference {rms:.4f}   max relative {rel:.5f} {scale_note}')
  return float(d.max()), rms, rel


def main():
  n_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 0
  torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))
  sd = np.load(NPZ_PATH)
  ver = np.load(VERIFY_PATH, allow_pickle=True)
  fens = [str(f) for f in ver['fens']]
  if n_pos:
    fens = fens[:n_pos]
  boards = [chess.Board(f) for f in fens]
  n_black = sum(1 for b in boards if b.turn == chess.BLACK)

  # ---- 0. tables and the checkpoint map
  table = fry_move_table()
  assert [m[0] for m in table] == sd['from_squares'].tolist() and [m[1] for m in table] == sd['to_squares'].tolist(), \
    'the 1968-move table differs from the checkpoint from / to buffers'
  ids, to_sq, pieces = sd['promotion_move_ids'].tolist(), sd['promotion_to_squares'].tolist(), sd['promotion_pieces'].tolist()
  assert ids == list(range(1792, 1968)) and all(table[i][1] == t and 'nbrq'.index(table[i][2]) == p for i, t, p in zip(ids, to_sq, pieces)), \
    'promotion ids / pieces differ from the checkpoint buffers (projection order N, B, R, Q)'
  t0 = time.time()
  net = FP.build_fry_net().eval()
  info = FP.load_fry_checkpoint(net, sd)
  n_ckpt = sum(int(np.asarray(sd[k]).size) for k in sd.files if k not in FP.CHECKPOINT_BUFFERS)
  assert not info['theirs_unused'], info['theirs_unused']
  assert info['loaded_elements'] == n_ckpt == 113_227_484, (info['loaded_elements'], n_ckpt)
  fry_side = [n for n in info['ours_untouched']
              if n.startswith(('transformer_layer', 'fry_', 'square_features', 'attn_templates', 'input_skip', 'trunk_end_norm', 'hlg_head'))]
  assert not fry_side, f'parameters of the Fry part of our net that the checkpoint did not write: {fry_side}'
  n_ours = sum(p.numel() for n, p in net.named_parameters() if n not in info['ours_untouched'])
  assert n_ours == n_ckpt, (n_ours, n_ckpt)
  n_extra = sum(p.numel() for n, p in net.named_parameters() if n in info['ours_untouched'])
  print(f'0. move table == checkpoint buffers (1968 ids, 176 promotion ids, projection order N B R Q); checkpoint '
        f'consumed completely: {info["loaded"]} of our parameters written, {n_ckpt:,} elements = his parameter count; '
        f'our net has {n_extra:,} more parameters in heads Fry does not have (shared head front-end, WDL / uncertainty '
        f'heads: {sorted(set(n.split(".")[0] for n in info["ours_untouched"]))}); build + load {time.time() - t0:.0f} s')

  # ---- ONNX with intermediate outputs
  layer_in = [f'/net/encoder.{i}/Add_output_0' for i in range(FP.LAYERS)]
  extra = ['/net/Add_1_output_0'] + layer_in + ['/net/ln/Mul_2_output_0']
  mode = os.environ.get('FRY_ORACLE', 'fp16')
  assert mode in ('fp16', 'fp32', 'exact'), mode
  as_f32 = mode != 'fp16'
  sess = onnx_session(extra, float32=as_f32)
  if mode == 'exact':
    round_like_the_file(net)
  print('   ORACLE: ' + {'fp16': 'the ONNX file as released (FP16 trunk) against the checkpoint weights in our net',
                         'fp32': 'the ONNX graph with every FP16 tensor / cast turned FP32 (its FP16-rounded weights, unrounded '
                                 'activations) against the checkpoint weights in our net',
                         'exact': 'the ONNX graph with every FP16 tensor / cast turned FP32, against our net holding the SAME '
                                  'FP16-rounded weights and norm epsilon as the file: only float32 summation order differs'}[mode])
  tokens = np.stack([fry_tokens(b) for b in boards])
  t0 = time.time()
  ref = {}
  names = ['policy', 'value', 'value_logits'] + extra
  outs = [[] for _ in names]
  for s in range(0, len(boards), 64):
    got = sess.run(names, {'board': tokens[s:s + 64]})
    for k, g in enumerate(got):
      outs[k].append(np.asarray(g, dtype=np.float32))
  ref = {n: np.concatenate(o) for n, o in zip(names, outs)}
  print(f'   onnxruntime: {len(boards)} positions ({n_black} Black to move) in {time.time() - t0:.0f} s')

  # ---- our net with the same tokens (override) and hooks on the stages
  rec = torch.from_numpy(np.stack([record_from_board(b) for b in boards]))
  tok = torch.from_numpy(tokens)
  seen = {'layer_in': [None] * FP.LAYERS}
  hooks = []
  for i, layer in enumerate(net.transformer_layer):
    hooks.append(layer.register_forward_pre_hook(lambda m, a, i=i: seen['layer_in'].__setitem__(i, a[0].detach())))
  hooks.append(net.trunk_end_norm.register_forward_hook(lambda m, a, o: seen.__setitem__('end', o.detach())))
  hooks.append(net.hlg_head.register_forward_hook(lambda m, a, o: seen.__setitem__('hlg', o.detach())))

  def run(override):
    pol, val, hlg, lay, end, emb = [], [], [], [[] for _ in range(FP.LAYERS)], [], []
    with torch.no_grad():
      for s in range(0, len(boards), 64):
        net._fry_tokens_override = tok[s:s + 64] if override else None
        out = net(rec[s:s + 64], None)
        pol.append(out[0].float().numpy()); val.append(out[1].float().numpy()); hlg.append(seen['hlg'].numpy())
        end.append(seen['end'].numpy())
        for i in range(FP.LAYERS):
          lay[i].append(seen['layer_in'][i].numpy())
    net._fry_tokens_override = None
    return (np.concatenate(pol), np.concatenate(val), np.concatenate(hlg), [np.concatenate(x) for x in lay], np.concatenate(end))

  t0 = time.time()
  pol, val, hlg, lay, end = run(True)
  print(f'   our net (float32, CPU): {time.time() - t0:.0f} s')

  # ---- 1. embedding. Layer 0 reads (a0 + b0) * x0, so x0 = its input / (a0 + b0)
  a0 = float((net.input_skip_a[0] + net.input_skip_b[0]).detach())
  print('1. embedding output (token + position + square features; ONNX tensor x 16):')
  e_max, e_rms, e_rel = stat('x0', lay[0] / a0, ref['/net/Add_1_output_0'] * STREAM_SCALE)

  # ---- 2. trunk, layer by layer
  print('2. input of each layer after the skip mix (ONNX tensor x 16), then the trunk-end norm output:')
  worst = worst_mean = 0.0
  for i in range(FP.LAYERS):
    r = ref[layer_in[i]].astype(np.float64) * STREAM_SCALE
    d = np.abs(lay[i].astype(np.float64) - r)
    rms = float(np.sqrt((r ** 2).mean()))
    rel = float((d / np.maximum(np.abs(r), rms)).max())
    worst, worst_mean = max(worst, rel), max(worst_mean, d.mean() / rms)
    if i in (0, 1, 2, 7, 14, 21, 27):
      k = np.unravel_index(d.argmax(), d.shape)
      print(f'   layer {i:2d} input                 max |diff| {d.max():.5f} (reference there {r[k]:+.3f}, largest |reference| {np.abs(r).max():.1f})   '
            f'mean |diff| {d.mean():.6f}   rms of reference {rms:.4f}   max relative {rel:.5f}')
  print(f'   over the 28 layer inputs: worst max relative difference {worst:.5f}, worst mean |diff| / rms {worst_mean:.6f}')
  n_max, n_rms, n_rel = stat('trunk-end norm output', end, ref['/net/ln/Mul_2_output_0'])

  # ---- 3. heads
  print('3. heads:')
  h_max = stat('value logits (128 buckets)', hlg, ref['value_logits'])[0]
  centres = (np.arange(128) + 0.5) / 128
  p_bins = torch.softmax(torch.from_numpy(hlg).double(), dim=-1).numpy()
  score = (p_bins * centres).sum(-1)
  w_served = np.exp(val[:, 0].astype(np.float64))            # served log-WDL with D = 0: W = (1 + Q) / 2 = expected score
  v_max = stat('value (expected score)', score, ref['value'])[0]
  s_max = stat('served W (D = 0) vs value', w_served, ref['value'])[0]
  d_all, n_moves, top_same, kl = [], 0, 0, []
  worst_pos = (0.0, -1)
  for b, board in enumerate(boards):
    lm = legal_move_indices(board)
    if not lm:
      continue
    ours = np.array([pol[b, o] for _, o, _ in lm], dtype=np.float64)
    his = np.array([ref['policy'][b, f] for _, _, f in lm], dtype=np.float64)
    d = np.abs(ours - his)
    d_all.append(d)
    n_moves += len(lm)
    if d.max() > worst_pos[0]:
      worst_pos = (float(d.max()), b)
    top_same += int(ours.argmax() == his.argmax())
    po, ph = np.exp(ours - ours.max()), np.exp(his - his.max())
    po, ph = po / po.sum(), ph / ph.sum()
    kl.append(float((ph * (np.log(ph + 1e-30) - np.log(po + 1e-30))).sum()))
  d_all = np.concatenate(d_all)
  n_with = sum(1 for b in boards if any(True for _ in b.legal_moves))
  print(f'   policy logits on legal moves       max |diff| {d_all.max():.5f}   mean |diff| {d_all.mean():.6f}   over {n_moves} legal moves of '
        f'{n_with} positions; same top move {top_same}/{n_with}; KL(his || ours) over legal moves: mean {np.mean(kl):.2e}, max {np.max(kl):.2e}')
  print(f'   worst position: #{worst_pos[1]} {fens[worst_pos[1]]} (max logit diff {worst_pos[0]:.5f})')
  if mode == 'exact':
    # same weights, same epsilon, float32 on both sides
    tol = dict(embedding_rel=e_rel < 1e-5, trunk_rel=worst < 5e-3, trunk_mean=worst_mean < 2e-5, end_norm=n_max < 2e-3,
               value_logits=h_max < 2e-3, value=max(v_max, s_max) < 2e-5, policy=d_all.max() < 2e-3, top_move=top_same == n_with)
  elif mode == 'fp32':
    # unrounded activations: what remains is the FP16 rounding of the file's weights
    tol = dict(embedding_rel=e_rel < 2e-3, trunk_mean=worst_mean < 2e-3, value_logits=h_max < 0.5,
               value=max(v_max, s_max) < 2e-3, policy=d_all.max() < 0.5, policy_mean=d_all.mean() < 0.01, top_move=top_same >= n_with - 2)
  else:
    # the released file: FP16 activations on a stream with channels up to |x| ~ 6000 (single elements can be off by
    # several units while the mean difference stays ~1e-3 of the rms)
    tol = dict(embedding_rel=e_rel < 0.02, trunk_mean=worst_mean < 2e-3, value_logits=h_max < 0.5,
               value=max(v_max, s_max) < 2e-3, policy=d_all.max() < 0.5, policy_mean=d_all.mean() < 0.01, top_move=top_same >= n_with - 2)
  assert all(tol.values()), f'outside the tolerances: {[k for k, v in tol.items() if not v]}'

  # ---- 4. tokens from our record
  pol_r, val_r, hlg_r, _, _ = run(False)
  from fry_input import tokens_from_record
  tok_r = tokens_from_record(rec).numpy()
  white = np.array([b.turn == chess.WHITE for b in boards])
  diff_cols = sorted(set(np.nonzero(tok_r != tokens)[1].tolist()))
  same_white = bool((tok_r[white] == tokens[white]).all())
  only_colour = bool((np.delete(tok_r, 64, axis=1) == np.delete(tokens, 64, axis=1)).all())
  print(f'4. tokens built from OUR RECORD inside forward: identical to the board encoder for all {int(white.sum())} White-to-move '
        f'positions: {same_white}; token columns that differ anywhere: {diff_cols} (64 = colour; the record has none, so Black '
        f'to move gets the White id); everything but the colour token identical on all positions: {only_colour}')
  dv_w = np.abs(np.exp(val_r[white, 0]) - np.exp(val[white, 0])).max() if white.any() else 0.0
  dp_w = np.abs(pol_r[white] - pol[white]).max() if white.any() else 0.0
  print(f'   White to move, record path vs token path: max |policy logit diff| {dp_w:.2e}, max |value diff| {dv_w:.2e}')
  if (~white).any():
    dv_b = np.abs(np.exp(val_r[~white, 0].astype(np.float64)) - np.exp(val[~white, 0].astype(np.float64)))
    same_top = 0
    dl = []
    for b in np.nonzero(~white)[0]:
      lm = legal_move_indices(boards[b])
      if not lm:
        same_top += 1
        continue
      idx = [o for _, o, _ in lm]
      same_top += int(pol_r[b, idx].argmax() == pol[b, idx].argmax())
      dl.append(np.abs(pol_r[b, idx] - pol[b, idx]).max())
    print(f'   Black to move with the White colour token (what a record-fed Fry would see): value differs by mean {dv_b.mean():.4f} / '
          f'max {dv_b.max():.4f} expected score; policy logits by up to {max(dl):.3f} (mean of per-position max {np.mean(dl):.3f}); '
          f'same top move {same_top}/{int((~white).sum())}')
  assert same_white and only_colour and dp_w < 1e-4 and dv_w < 1e-5
  for h in hooks:
    h.remove()
  print('PARITY OK')


if __name__ == '__main__':
  main()
