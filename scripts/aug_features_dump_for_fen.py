"""Dump augmented-feature bytes for a set of FENs. Used as the reference oracle
for the Phase 2 Python ↔ C# equality test (Ceres tests/AugFeatSanity).

For each FEN: computes our_attackers, opp_attackers, shifted_net per square
matching aug_features.py's encoding exactly, then emits the bytes that C#
should produce for the same position.

Output format (binary, per FEN):
  uint32 nFens (little-endian) — written once at the start
  for each FEN:
    [192 bytes]  64 squares * 3 channels * 1 byte each, row-major
                 byte[sq*3 + 0] = our_attackers byte
                 byte[sq*3 + 1] = opp_attackers byte
                 byte[sq*3 + 2] = net_shifted byte

Square indexing matches python-chess (a1=0, h8=63) AND our PerSquareAttacks
C# convention. Caller is responsible for orientation: this script outputs
in REAL board coordinates (white pieces always WHITE, black pieces always
BLACK) — the C# side must apply us-to-move flipping when consuming.

Usage:
  python aug_features_dump_for_fen.py --fens fens.txt --out aug_bytes.bin
"""
import argparse, struct, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'CeresTrainPy'))
import chess


def aug_bytes_for_board(board: chess.Board) -> bytes:
  """Returns 192 bytes: 64 squares × 3 channels (our, opp, net_shifted).
  WHITE in this convention always = our_attackers; the equality test compares
  byte-for-byte against PerSquareAttacks.Compute which uses the same convention.
  """
  out = bytearray(64 * 3)
  for sq in range(64):
    w_mask = board.attackers_mask(chess.WHITE, sq)
    b_mask = board.attackers_mask(chess.BLACK, sq)
    w = bin(w_mask).count('1')
    b = bin(b_mask).count('1')
    out[sq * 3 + 0] = (w * 100) // 8
    out[sq * 3 + 1] = (b * 100) // 8
    out[sq * 3 + 2] = ((w - b + 8) * 100) // 16
  return bytes(out)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--fens', required=True, help='one FEN per line')
  ap.add_argument('--out',  required=True, help='output binary file')
  args = ap.parse_args()

  with open(args.fens) as f:
    fens = [line.strip() for line in f if line.strip() and not line.startswith('#')]

  with open(args.out, 'wb') as f:
    f.write(struct.pack('<I', len(fens)))
    for fen in fens:
      board = chess.Board(fen)
      f.write(aug_bytes_for_board(board))

  print(f'wrote {len(fens)} FENs × 192 bytes = {4 + len(fens) * 192} total bytes to {args.out}')


if __name__ == '__main__':
  main()
