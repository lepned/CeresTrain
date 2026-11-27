"""Piece-dependent attention templates (config TemplatesPieceQuery / TemplatesPieceKey, attn_templates.py). CPU.

  (a) D = 0: the mixer's bias with the position's piece maps equals the plain-template bias EXACTLY (torch.equal),
      query / key / both, FP32 and under CPU bf16 autocast
  (b) random non-zero D, random codes: bias == a straightforward reference loop over (b, i, j)
        bias[b, h, i, j] = sum_t c[b, i, h, t] * (T[t, i, j] + Dq[t, code(b, i), i, j] + Dk[t, code(b, j), i, j])
  (c) codes from real records (square_features.codes_from_record): the gathered maps pick the row of the piece
      actually standing on the square (one python-chess board, a few squares checked by hand)
  (d) gradient reaches D through the mixer while the templates are zero (as at step 0)
  (e) config refusals

  python test_piece_templates.py
"""
import io
import os
import sys

import numpy as np
import torch

from attn_templates import AttnTemplateBank, AttnTemplateMixer, NUM_PIECE_CODES


def bank_pair(T=8, pq=False, pk=False, seed=3):
  torch.manual_seed(seed)
  bank = AttnTemplateBank(T, piece_query=pq, piece_key=pk)
  with torch.no_grad():
    bank.table.normal_()
  return bank


def test_zero_equal():
  B, H, D_IN = 5, 4, 48
  g = torch.Generator().manual_seed(1)
  codes = torch.randint(0, NUM_PIECE_CODES, (B, 64), generator=g)
  x = torch.randn(B, 64, D_IN, generator=g)
  for pq, pk in ((True, False), (False, True), (True, True)):
    bank = bank_pair(pq=pq, pk=pk)
    torch.manual_seed(9)
    mixer = AttnTemplateMixer(D_IN, H, bank)
    ref = mixer(x)
    got = mixer(x, bank.position_tables(codes, torch.float32))
    assert torch.equal(ref, got), (pq, pk, float((ref - got).abs().max()))
    with torch.autocast('cpu', dtype=torch.bfloat16):
      ref16 = mixer(x)
      got16 = mixer(x, bank.position_tables(codes, torch.bfloat16))
    assert ref16.dtype == torch.bfloat16 and torch.equal(ref16, got16), (pq, pk)
  print('(a) D = 0: bias == control bias exactly (query / key / both; fp32 and bf16 autocast)')


def reference_bias(c, T, Dq, Dk, codes):
  """c [B, i, h, t], T [t, i, j], D* [t, 13, i, j] or None, codes [B, 64] -> [B, h, i, j] by explicit loops."""
  B = c.shape[0]
  M = torch.zeros(B, 64, T.shape[0], 64, dtype=torch.float64)     # [b, i, t, j]
  for b in range(B):
    for i in range(64):
      for j in range(64):
        v = T[:, i, j].double().clone()
        if Dq is not None:
          v += Dq[:, int(codes[b, i]), i, j].double()
        if Dk is not None:
          v += Dk[:, int(codes[b, j]), i, j].double()
        M[b, i, :, j] = v
  out = torch.zeros(B, c.shape[2], 64, 64, dtype=torch.float64)
  for b in range(B):
    for i in range(64):
      out[b, :, i, :] = c[b, i].double() @ M[b, i]                  # [h, t] @ [t, j]
  return out


def test_reference_loop():
  B, H, D_IN = 3, 4, 32
  g = torch.Generator().manual_seed(2)
  codes = torch.randint(0, NUM_PIECE_CODES, (B, 64), generator=g)
  x = torch.randn(B, 64, D_IN, generator=g)
  for pq, pk in ((True, False), (False, True), (True, True)):
    bank = bank_pair(pq=pq, pk=pk)
    with torch.no_grad():
      for p in (bank.piece_query, bank.piece_key):
        if p is not None:
          p.normal_(0, 0.7, generator=g)
    mixer = AttnTemplateMixer(D_IN, H, bank)
    with torch.no_grad():
      got = mixer(x, bank.position_tables(codes, torch.float32)).double()
      c = mixer.coefficients(x)
      want = reference_bias(c, bank.table, bank.piece_query, bank.piece_key, codes)
      plain = mixer(x).double()
    err = float((got - want).abs().max())
    assert err < 1e-4, (pq, pk, err)
    assert float((plain - want).abs().max()) > 0.1, 'piece term had no effect'
    print(f'(b) random D (query {pq}, key {pk}): max |bias - reference loop| {err:.2e}; '
          f'|plain - reference| {float((plain - want).abs().max()):.2f}')


def test_real_codes():
  import chess
  from fry_testlib import record_from_board
  from square_features import codes_from_record
  board = chess.Board('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3')
  rec = torch.from_numpy(np.stack([record_from_board(board)]).astype(np.float32))
  codes = codes_from_record(rec)
  # White to move: frame = board; a1 = 0. f3 (21) our knight = 2, c6 (42) their knight = 8, e1 (4) our king = 6, e5 (36) their pawn 7
  assert [int(codes[0, s]) for s in (21, 42, 4, 36, 28, 35)] == [2, 8, 6, 7, 1, 0], codes[0]
  bank = AttnTemplateBank(2, piece_query=True, piece_key=True)
  with torch.no_grad():
    bank.piece_query.copy_(torch.arange(bank.piece_query.numel(), dtype=torch.float32).reshape(bank.piece_query.shape))
    bank.piece_key.zero_()
    pos = bank.position_tables(codes, torch.float32)               # [1, i, t, j]
    assert torch.equal(pos[0, 21, 1, 5], bank.piece_query[1, 2, 21, 5])
    assert torch.equal(pos[0, 42, 0, 63], bank.piece_query[0, 8, 42, 63])
    bank.piece_query.zero_()
    bank.piece_key.copy_(torch.arange(bank.piece_key.numel(), dtype=torch.float32).reshape(bank.piece_key.shape))
    pos = bank.position_tables(codes, torch.float32)
    assert torch.equal(pos[0, 0, 1, 21], bank.piece_key[1, 2, 0, 21])      # key f3 = our knight
    assert torch.equal(pos[0, 63, 0, 4], bank.piece_key[0, 6, 63, 4])      # key e1 = our king
  print('(c) real record: codes and gathered rows pick the piece on the query / key square')


def test_gradient():
  bank = AttnTemplateBank(8, piece_query=True, piece_key=True)          # all tables zero, as at step 0
  mixer = AttnTemplateMixer(32, 4, bank)
  g = torch.Generator().manual_seed(4)
  codes = torch.randint(0, NUM_PIECE_CODES, (2, 64), generator=g)
  x = torch.randn(2, 64, 32, generator=g)
  out = mixer(x, bank.position_tables(codes, torch.float32))
  (out * torch.randn(out.shape, generator=g)).sum().backward()
  gq, gk = float(bank.piece_query.grad.abs().sum()), float(bank.piece_key.grad.abs().sum())
  assert gq > 0 and gk > 0 and float(mixer.mix.weight.grad.abs().sum()) == 0.0
  # only rows of codes that occur get a gradient
  used = set(int(v) for v in codes.reshape(-1))
  for k in range(NUM_PIECE_CODES):
    gk_k = float(bank.piece_query.grad[:, k].abs().sum())
    assert (gk_k > 0) == (k in used), (k, gk_k)
  print(f'(d) gradient at zero tables: |dD_query| {gq:.3g}, |dD_key| {gk:.3g} (mixer 0, as the plain templates)')


def test_refusals():
  from config import Configuration
  import json, tempfile
  here = os.path.dirname(os.path.abspath(__file__))
  src = os.environ.get('CERES_PZ_D2_CONFIGS', '')
  if not os.path.isdir(src):
    print('(e) skipped (no d2 configs at', src, ')')
    return
  base = {p: json.load(open(os.path.join(src, f'pzc256x8d2_ctl_ceres_{p}.json'))) for p in ('net', 'opt', 'data', 'exec', 'monitoring')}
  cases = [({'AttnTemplates': 0, 'TemplatesPieceQuery': True}, 'need AttnTemplates'),
           ({'TemplatesPieceKey': True, 'NBTInnerLayers': 2}, 'plain encoder')]
  for change, what in cases:
    with tempfile.TemporaryDirectory() as d:
      for p, v in base.items():
        v = dict(v)
        if p == 'net':
          v.update(change)
          v.pop('AttnTemplateScope', None) if change.get('AttnTemplates') == 0 else None
        json.dump(v, open(os.path.join(d, f'x_ceres_{p}.json'), 'w'))
      sink, so = io.StringIO(), sys.stdout
      sys.stdout = sink
      try:
        Configuration(d, 'x')
        raise AssertionError(f'no refusal: {change}')
      except ValueError as e:
        assert what in str(e), (change, str(e))
      finally:
        sys.stdout = so
  print('(e) config refusals ok')


if __name__ == '__main__':
  test_zero_equal()
  test_reference_loop()
  test_real_codes()
  test_gradient()
  test_refusals()
  print('ALL PIECE-TEMPLATE TESTS OK')
