"""
Scan TPG files for Chess960 (Fischer Random) records.

Detection rule: a record is Chess960 if a king is NOT on the e-file (file index 4)
AND that side still has castling rights. In standard chess, the king must be on
e1/e8 for castling to be legal — moving the king kills both castle rights. So a
king with castling rights on (say) c1/d1/f1/g1 means the king started there,
which only happens in 960.

Usage:
    python3 inspect_tpg_for_960.py <path-to-tpg-dir>          # scan whole dir
    python3 inspect_tpg_for_960.py <path-to-tpg.zst> [N]      # scan one file, N records (default 12288 = one block)

Schema (per TPGRecord.cs / TPGSquareRecord.cs, V2 layout, 9378 bytes/record):
  Per record: 610 bytes of metadata + 64 squares * 137 bytes = 9378
  Per square (137 bytes):
    [0 : 104]  pieceTypeHistory[8 history positions * 13 one-hot bytes]
                 indices: 0=empty, 1-6=our P/N/B/R/Q/K, 7-12=opp P/N/B/R/Q/K
    [104:112]  historyRepetitionCounts[8]
    [112:121]  9 ByteScaled fields (CanOO, CanOOO, OppCanOO, OppCanOOO,
                                    Move50Count, PlySinceLastMove, IsEnPassant,
                                    QPosBlunders, QNegBlunders)
    [121:129]  rankEncoding[8] one-hot (rank 0 = back rank "us")
    [129:137]  fileEncoding[8] one-hot (file 0 = a-file)

Output:
    Per-file: kings-with-castle counts, files seen, verdict
    Total: clear "CHESS960 PRESENT" or "all records standard".
"""
import os
import sys
import glob

import numpy as np
import zstandard


BYTES_PER_POS = 9378
SQUARE_OFFSET = 9378 - 64 * 137                # = 610
SQUARE_SIZE   = 137
NUM_SQUARES   = 64

PIECE_HIST_OFFSET     = 0     # bytes [0:104] = 8 history * 13 piece-onehot
CASTLE_OO_OFFSET      = 112
CASTLE_OOO_OFFSET     = 113
OPP_CASTLE_OO_OFFSET  = 114
OPP_CASTLE_OOO_OFFSET = 115
RANK_ENC_OFFSET       = 121   # one-hot 8 bytes
FILE_ENC_OFFSET       = 129   # one-hot 8 bytes

OUR_KING_PIECE_IDX = 6        # bytes [0..12] = current position one-hot, 6=our K
OPP_KING_PIECE_IDX = 12


def _argmax_onehot(byte_slice):
    """Given an 8-byte one-hot, return the index of the set bit (or -1 if all zero)."""
    nz = np.flatnonzero(byte_slice)
    return int(nz[0]) if len(nz) > 0 else -1


def inspect_block(record_bytes):
    """Inspect one record (9378 bytes). Returns dict of findings."""
    findings = {
        'has_960_signal': False,
        'our_king_file': -1,
        'opp_king_file': -1,
        'our_can_oo': False,
        'our_can_ooo': False,
        'opp_can_oo': False,
        'opp_can_ooo': False,
    }

    # Walk all 64 squares to find the kings and read castling rights from
    # any one of them (castling fields are stored per-square but identical
    # across all squares in a record).
    castle_read = False
    for sq in range(NUM_SQUARES):
        sq_off = SQUARE_OFFSET + sq * SQUARE_SIZE
        sq_bytes = record_bytes[sq_off:sq_off + SQUARE_SIZE]

        if not castle_read:
            findings['our_can_oo']  = sq_bytes[CASTLE_OO_OFFSET] > 0
            findings['our_can_ooo'] = sq_bytes[CASTLE_OOO_OFFSET] > 0
            findings['opp_can_oo']  = sq_bytes[OPP_CASTLE_OO_OFFSET] > 0
            findings['opp_can_ooo'] = sq_bytes[OPP_CASTLE_OOO_OFFSET] > 0
            castle_read = True

        # Current-position piece one-hot (history index 0 = bytes [0:13])
        cur_piece = sq_bytes[0:13]
        if cur_piece[OUR_KING_PIECE_IDX] > 0:
            file_idx = _argmax_onehot(sq_bytes[FILE_ENC_OFFSET:FILE_ENC_OFFSET + 8])
            findings['our_king_file'] = file_idx
        elif cur_piece[OPP_KING_PIECE_IDX] > 0:
            file_idx = _argmax_onehot(sq_bytes[FILE_ENC_OFFSET:FILE_ENC_OFFSET + 8])
            findings['opp_king_file'] = file_idx

    # Detection: castling rights alive AND king not on e-file (file 4).
    # In standard chess, castling rights die the moment the king moves off e1/e8.
    if findings['our_king_file'] not in (-1, 4):
        if findings['our_can_oo'] or findings['our_can_ooo']:
            findings['has_960_signal'] = True
    if findings['opp_king_file'] not in (-1, 4):
        if findings['opp_can_oo'] or findings['opp_can_ooo']:
            findings['has_960_signal'] = True

    return findings


def scan_file(path, max_records=12288):
    """Decompress the head of a .zst file and inspect records."""
    bytes_to_read = max_records * BYTES_PER_POS
    with open(path, 'rb') as f:
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        data = reader.read(bytes_to_read)

    n_records = len(data) // BYTES_PER_POS
    if n_records == 0:
        return {'path': path, 'records_scanned': 0, 'note': 'empty/short read'}

    arr = np.frombuffer(data[:n_records * BYTES_PER_POS], dtype=np.uint8).reshape(n_records, BYTES_PER_POS)

    n_960 = 0
    our_king_files = np.zeros(8, dtype=np.int64)
    opp_king_files = np.zeros(8, dtype=np.int64)
    our_castle_alive = 0
    opp_castle_alive = 0
    castle_alive_total = 0
    sample_960_findings = []

    for i in range(n_records):
        f = inspect_block(arr[i])
        if 0 <= f['our_king_file'] <= 7:
            our_king_files[f['our_king_file']] += 1
        if 0 <= f['opp_king_file'] <= 7:
            opp_king_files[f['opp_king_file']] += 1
        our_castle = f['our_can_oo'] or f['our_can_ooo']
        opp_castle = f['opp_can_oo'] or f['opp_can_ooo']
        if our_castle: our_castle_alive += 1
        if opp_castle: opp_castle_alive += 1
        if our_castle or opp_castle: castle_alive_total += 1
        if f['has_960_signal']:
            n_960 += 1
            if len(sample_960_findings) < 5:
                sample_960_findings.append((i, f))

    return {
        'path': path,
        'records_scanned': n_records,
        'records_with_castling_alive': castle_alive_total,
        'records_with_960_signal': n_960,
        'our_king_file_dist': our_king_files.tolist(),  # files a..h (0..7)
        'opp_king_file_dist': opp_king_files.tolist(),
        'sample_960_findings': sample_960_findings[:5],
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)

    target = sys.argv[1]
    max_records = int(sys.argv[2]) if len(sys.argv) >= 3 else 12288

    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, '*.zst')))
    elif os.path.isfile(target):
        files = [target]
    else:
        print(f'Not found: {target}'); sys.exit(1)

    print(f'Scanning {len(files)} file(s), up to {max_records} records each')
    print('Detection rule: king NOT on e-file AND castling rights alive => Chess960\n')

    grand_960 = 0
    grand_castle = 0
    grand_records = 0
    for fpath in files:
        r = scan_file(fpath, max_records=max_records)
        records = r['records_scanned']
        n_960 = r['records_with_960_signal']
        n_cas = r['records_with_castling_alive']
        verdict = 'CHESS960 DETECTED' if n_960 > 0 else 'standard only'
        print(f'  {os.path.basename(fpath)}')
        print(f'    records={records}  castle-alive={n_cas}  960-signal={n_960}  =>  {verdict}')
        # King file distribution helps spot suspicious spreads even without castling
        ufiles = r['our_king_file_dist']
        ofiles = r['opp_king_file_dist']
        print(f'    our king files [a-h]: {ufiles}')
        print(f'    opp king files [a-h]: {ofiles}')
        if n_960 > 0:
            print(f'    sample 960 records:')
            for idx, f in r['sample_960_findings']:
                print(f'      record#{idx}  our_king_file={f["our_king_file"]} (castles {f["our_can_oo"]}/{f["our_can_ooo"]})  opp_king_file={f["opp_king_file"]} (castles {f["opp_can_oo"]}/{f["opp_can_ooo"]})')
        grand_960 += n_960
        grand_castle += n_cas
        grand_records += records

    print(f'\nTotal: {grand_records} records, {grand_castle} with castling alive, {grand_960} with 960 signal')
    if grand_960 == 0:
        print('VERDICT: all sampled records are standard chess.')
    else:
        rate = 100.0 * grand_960 / max(grand_castle, 1)
        print(f'VERDICT: CHESS960 PRESENT — {grand_960} 960-signaling records ({rate:.1f}% of records-with-castling-alive).')


if __name__ == '__main__':
    main()
