"""The record path of the Fry blocks on REAL training positions (lc0 v6 chunks of the T80 corpus, read from a tar).

For every position two independent routes are compared:
  (r) the trainer's own loader code (v6_dataset.V6ChunkDataset._records_to_arrays) builds our 137-float square record;
      the in-graph code reads it: square_features.codes_from_record, fry_input.tokens_from_record, SquareFeatures.rows
  (b) this file decodes the lc0 planes into a python-chess board (frame of the side to move, mirrored back to the
      real colour for Black) and uses the board-based encoder / per-square reference of fry_testlib

Checks:
  1. the board decode is right: the legal moves of the board, as OUR 1858 indices, are exactly the moves the lc0
     record marks legal (probs >= 0) -- castling as king-takes-rook, en passant, promotions included
  2. token t of the record is square t in the frame of the side to move, for White AND Black to move
  3. the 70 tokens: record path == board path in every column except colour (the record has none), and en passant
     only where the capture is pseudo-legal but not legal (counted)
  4. the 23 square-feature rows per square: module on the record == per-square reference on the board

  python test_fry_record_real.py [tar] [max positions]
"""
import gzip
import os
import sys
import tarfile
import types

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('CERES_AUX_FEATURES_PER_SQUARE', '0')

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_TAR = '/mnt/d/TAR/training-run1-test80-20240427-1117.tar'


def read_records(tar_path, max_positions):
  import v6_dataset as V
  out = []
  n = 0
  with tarfile.open(tar_path, 'r') as tf:
    for member in tf:
      if not member.name.endswith('.gz'):
        continue
      data = gzip.decompress(tf.extractfile(member).read())
      version = int(np.frombuffer(data[:4], dtype='<u4')[0])
      stride, dtype = V._BY_VERSION[version]
      recs = np.frombuffer(data[:len(data) // stride * stride], dtype=dtype)
      out.append(recs)
      n += len(recs)
      if n >= max_positions:
        break
  return out


def board_from_record(rec):
  """lc0 record -> (python-chess board with the REAL side to move, frame board). Planes: 13 per history position
  (ours P N B R Q K, theirs P N B R Q K, repetition); a plane is a u64 whose byte r is rank r with file a in the
  MSB. Standard chess (rooks on a / h files) is assumed for the castling flags."""
  planes = rec['planes']
  def squares_of(plane):
    b = int(planes[plane])
    return [r * 8 + f for r in range(8) for f in range(8) if (b >> (8 * r + (7 - f))) & 1]
  fb = chess.Board(None)
  for k, piece in enumerate((chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)):
    for s in squares_of(k):
      fb.set_piece_at(s, chess.Piece(piece, chess.WHITE))
    for s in squares_of(6 + k):
      fb.set_piece_at(s, chess.Piece(piece, chess.BLACK))
  fb.turn = chess.WHITE
  rights = 0
  if rec['us_oo']: rights |= chess.BB_H1
  if rec['us_ooo']: rights |= chess.BB_A1
  if rec['them_oo']: rights |= chess.BB_H8
  if rec['them_ooo']: rights |= chess.BB_A8
  fb.castling_rights = rights
  # en passant: their pawn stands on rank 5 now and stood on rank 7 one ply ago (and not the other way round)
  now, before = set(squares_of(6)), set(squares_of(13 + 6))
  has_history = any(int(planes[13 + k]) for k in range(12))
  if has_history:
    for f in range(8):
      if (32 + f) in now and (48 + f) in before and (32 + f) not in before and (48 + f) not in now:
        fb.ep_square = 40 + f
  black = int(rec['input_format']) == 1 and int(rec['stm_or_ep']) == 1
  standard = fb.clean_castling_rights() == rights     # False: a Chess960 game (king / rooks not on e / a / h files)
  return (fb.mirror() if black else fb), fb, black, standard


def main():
  tar_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TAR
  max_positions = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
  if not os.path.exists(tar_path):
    print(f'SKIPPED: {tar_path} not found')
    return
  import v6_dataset as V
  from fry_input import tokens_from_record
  from fry_testlib import fry_tokens, legal_move_indices, piece_codes, square_feature_rows
  from square_features import SquareFeatures, codes_from_record

  chunks = read_records(tar_path, max_positions)
  recs = np.concatenate(chunks)[:max_positions]
  formats = sorted(set(recs['input_format'].tolist()))
  assert formats == [1], f'input formats {formats}: this test reads the side to move from the classical format (1)'
  arrays = V.V6ChunkDataset._records_to_arrays(types.SimpleNamespace(), recs)
  squares = torch.from_numpy(np.asarray(arrays[9], dtype=np.float32))
  assert squares.shape[1:] == (64, 137), squares.shape

  # 2. token t of the record is square t (rank / file one-hots)
  enc = squares[:, :, 121:129].argmax(-1) * 8 + squares[:, :, 129:137].argmax(-1)
  assert bool((enc == torch.arange(64)).all()), 'token t is not square t in the record'

  codes_r = codes_from_record(squares)
  tok_r = tokens_from_record(squares, codes_r).numpy()
  sf = SquareFeatures(8)
  rows_r = sf.rows(codes_r).numpy()

  n = len(recs)
  n_black = n_castle = n_ep_flag = n_ep_pseudo = n_ep_legal = n_promo = 0
  legal_bad = tok_bad = rows_bad = 0
  ep_only_diff = 0
  n_rows_checked = 0
  n_frc = 0
  for i in range(n):
    board, fb, black, standard = board_from_record(recs[i])
    if not standard:
      n_frc += 1       # T80 contains Chess960 games; this test's board decode handles standard castling only
      continue
    n_black += int(black)
    # 1. legal moves == the record's legal mask
    lm = legal_move_indices(board)
    mine = sorted(o for _, o, _ in lm)
    theirs = np.nonzero(np.nan_to_num(recs[i]['probs'], nan=-1.0) >= 0)[0].tolist()
    if mine != theirs:
      legal_bad += 1
      if legal_bad <= 3:
        print('   legal-move mismatch:', board.fen(), sorted(set(mine) ^ set(theirs))[:8])
    n_castle += int(any(fb.is_castling(m) for m in fb.legal_moves))
    n_promo += int(any(m.promotion for m in fb.legal_moves))
    # 3. tokens
    tok_b = fry_tokens(board)
    assert tok_b[64] == (13 if black else 14)
    flag = bool(squares[i, 32:40, 118].sum() > 0)
    n_ep_flag += int(flag)
    n_ep_pseudo += int(tok_r[i, 69] != 17)
    n_ep_legal += int(tok_b[69] != 17)
    d = np.nonzero(tok_r[i] != tok_b)[0].tolist()
    d = [c for c in d if c != 64]
    if d == [69] and tok_b[69] == 17 and tok_r[i, 69] != 17:
      ep_only_diff += 1            # pseudo-legal in the record path, not legal on the board
    elif d:
      tok_bad += 1
      if tok_bad <= 3:
        print('   token mismatch:', board.fen(), d, tok_r[i, d], tok_b[d])
    assert (tok_r[i, 64] == 14)
    # 4. square-feature rows (every position: the vectorised module against the per-square reference; the reference
    #    is slow, so a subsample)
    if i % 10 == 0:
      n_rows_checked += 1
      if not (rows_r[i] == square_feature_rows(piece_codes(fb))).all():
        rows_bad += 1
        if rows_bad <= 3:
          k = np.argwhere(rows_r[i] != square_feature_rows(piece_codes(fb)))[:5]
          print('   row mismatch:', fb.fen(), k.tolist())
  print(f'{n} positions read from {os.path.basename(tar_path)} ({len(chunks)} games); {n_frc} with castling rights on non-standard '
        f'squares (Chess960 games) left out of the board comparison')
  n -= n_frc
  print(f'{n} positions compared ({n_black} Black to move, {n_castle} with a legal '
        f'castling move, {n_promo} with a legal promotion, record e.p. byte set in {n_ep_flag})')
  print(f'1. legal moves of the decoded board == moves the lc0 record marks legal (our 1858 indices): {n - legal_bad}/{n}')
  print(f'2. token t of the record is square t (rank * 8 + file in the frame of the side to move): all {n} positions, both colours')
  print(f'3. 70 tokens, record path vs board path: {n - tok_bad - ep_only_diff}/{n} identical outside the colour token; '
        f'{ep_only_diff} differ ONLY in the e.p. token (record path: one of our pawns stands beside the pushed pawn = {n_ep_pseudo} '
        f'positions; legal capture on the board = {n_ep_legal}); other mismatches: {tok_bad}. Colour token: always 14 from the '
        f'record, 13 on the board for the {n_black} Black-to-move positions')
  print(f'4. square-feature rows, module on the record == per-square reference on the board: {n_rows_checked - rows_bad}/{n_rows_checked} positions '
        f'(every 10th), 64 x 23 rows each')
  assert legal_bad == 0 and tok_bad == 0 and rows_bad == 0
  print('RECORD PATH OK')


if __name__ == '__main__':
  main()
