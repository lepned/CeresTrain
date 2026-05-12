#!/usr/bin/env python3
"""Dump N tactical TPG records as study-ready FENs with per-record signal values.

Filters by composite (DQVV + KLDPolicy) — these are the genuine "hard for the
network" signals, NOT PlayedMoveQSuboptimality (which is dominated by self-play
sampling noise).

Output: a markdown file listing each record with FEN, side-to-move, search W/D/L,
deblundered/non-deblundered targets, raw V (= search_Q - DQVV), KLDPolicy, and
the top-K policy entries (idx + probability — search's recommended moves).

Usage:
    python dump_tactical_records.py <input_dir> <output_md>
        [--min-dqvv 0.5]
        [--min-kld 0.5]
        [--min-abs-q 0.0]
        [--n 20]
        [--top-k 5]

Notes:
- FEN castling/EP/halfmove are placeholders (- - 0 1). Position structure is correct.
- Square encoding is side-to-move-relative (Square.Reversed for Black). Decoder
  flips ranks for Black-to-move to produce standard-orientation FENs.
- Policy values are un-normalized soft targets. Top-1 by P is the search's
  most-visited move. Convert idx→UCI with a Lc0 1858-move map (not included).
"""
import argparse, glob, os, sys, time
import numpy as np
import zstandard

BPP = 9378
OFFSET_WDLND  = 0       # WDLResultNonDeblundered (3*float32)
OFFSET_WDLD   = 12      # WDLResultDeblundered (3*float32)
OFFSET_WDLQ   = 24      # WDLQ search (3*float32)
OFFSET_PQSUB  = 36      # PlayedMoveQSuboptimality (float32)
OFFSET_STM    = 40      # IsWhiteToMove (uint8)
OFFSET_KLD    = 96      # KLDPolicy (float32)
OFFSET_MLH    = 100     # MLH (float32)
OFFSET_DQVV   = 104     # DeltaQVersusV (float32)
OFFSET_PIP    = 112     # PolicyIndexInParent (int16) — move played from parent into this position
OFFSET_POLI   = 242     # 92*int16
OFFSET_POLV   = 426     # 92*float16
OFFSET_SQ     = 610     # 64*137 bytes


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('input_dir', help='Directory containing TPG .zst files')
    ap.add_argument('output_md', help='Output markdown file path')
    ap.add_argument('--min-dqvv', type=float, default=0.20,
                    help='Keep |DeltaQVersusV| >= this. NN value disagreed with search. (default 0.20)')
    ap.add_argument('--min-kld', type=float, default=2.5,
                    help='Keep KLDPolicy >= this. NN policy strongly disagreed with search. (default 2.5)')
    ap.add_argument('--min-abs-q', type=float, default=0.60,
                    help='Keep |search_Q| >= this. Decisive position. (default 0.60)')
    ap.add_argument('--min-top-p', type=float, default=0.20,
                    help='Keep records where the search top-1 policy P is ABOVE this. '
                         'Ensures search has a clear preference (training target). (default 0.20)')
    ap.add_argument('--max-top-p', type=float, default=0.50,
                    help='Keep records where the search top-1 policy P is BELOW this. '
                         'Excludes trivially-converged positions where the answer is overwhelming. (default 0.50)')
    ap.add_argument('--min-top2-p', type=float, default=0.10,
                    help='Keep records where the SECOND-best move (unique idx) policy P >= this. '
                         'Excludes pure forcing positions where only one move makes sense. (default 0.10)')
    ap.add_argument('--min-pieces', type=int, default=18,
                    help='Minimum total piece count on board. (default 18 — full middlegame)')
    ap.add_argument('--max-material-imbalance', type=float, default=1.0,
                    help='Maximum |white material − black material| in standard piece values '
                         '(P=1, N=B=3, R=5, Q=9). Excludes positions where one side is materially '
                         'ahead — material-driven positions, not tactical. (default 1.0 = ≤1 pawn)')
    ap.add_argument('--n', type=int, default=20,
                    help='Number of records to dump (default 20)')
    ap.add_argument('--top-k', type=int, default=5,
                    help='Top-K policy moves per record (default 5)')
    return ap.parse_args()


def decode_position_to_fen(rec):
    """Reconstruct FEN board from a TPG record's 64*137 squares plus IsWhiteToMove.

    Square encoding is side-to-move-relative (rank flipped for Black via
    Square.Reversed = (File, 7-Rank)). We flip back so the FEN is standard.

    Returns: (fen_str, ascii_board_str)
    """
    is_white = bool(rec[OFFSET_STM])
    sq = rec[OFFSET_SQ:OFFSET_SQ + 64*137].reshape(64, 137)

    PIECE_OUR = ['.', 'P', 'N', 'B', 'R', 'Q', 'K']  # plane0 idx 0..6
    PIECE_OPP = ['p', 'n', 'b', 'r', 'q', 'k']        # plane0 idx 7..12

    # board[absolute_rank][file] = piece char (uppercase=white, lowercase=black)
    board = [['.' for _ in range(8)] for _ in range(8)]
    for s in range(64):
        plane0  = sq[s, 0:13]
        rank_e  = sq[s, 121:129]
        file_e  = sq[s, 129:137]
        if rank_e.max() == 0 or file_e.max() == 0:
            continue
        rel_rank = int(rank_e.argmax())
        file     = int(file_e.argmax())

        # Convert relative→absolute rank
        abs_rank = (7 - rel_rank) if not is_white else rel_rank

        pi = int(plane0.argmax())
        if plane0[pi] == 0:
            ch = '.'
        elif pi < 7:
            ch = PIECE_OUR[pi]    # our piece
        else:
            ch = PIECE_OPP[pi-7]  # opp piece
        # Display: uppercase=white, lowercase=black. "Our" depends on STM:
        #   - white-to-move: our=white (uppercase already correct)
        #   - black-to-move: our=black, swap to lowercase; opp=white, swap to upper
        if not is_white:
            ch = ch.swapcase()
        board[abs_rank][file] = ch

    # FEN board: ranks 8..1 top-to-bottom
    fen_ranks = []
    for r in range(7, -1, -1):
        row, empties = '', 0
        for f in range(8):
            ch = board[r][f]
            if ch == '.':
                empties += 1
            else:
                if empties: row += str(empties); empties = 0
                row += ch
        if empties: row += str(empties)
        fen_ranks.append(row)
    stm = 'w' if is_white else 'b'
    fen = '/'.join(fen_ranks) + f' {stm} - - 0 1'

    # ASCII board (standard orientation, rank 8 at top)
    lines = ['  a b c d e f g h']
    for r in range(7, -1, -1):
        lines.append(f'{r+1} ' + ' '.join(board[r][f] for f in range(8)))
    ascii_board = '\n'.join(lines)
    return fen, ascii_board


def main():
    args = parse_args()
    inputs = sorted(glob.glob(os.path.join(args.input_dir, '*.zst')))
    if not inputs:
        print(f'no .zst files in {args.input_dir}', file=sys.stderr)
        sys.exit(1)

    print(f'Scanning {len(inputs)} files for: |DQVV|>={args.min_dqvv} '
          f'AND KLDPolicy>={args.min_kld}'
          + (f' AND |Q|>={args.min_abs_q}' if args.min_abs_q > 0 else '')
          + f' (target N={args.n})')

    found = []  # list of dicts
    for fi, path in enumerate(inputs):
        if len(found) >= args.n:
            break
        try:
            with open(path, 'rb') as f:
                raw = zstandard.ZstdDecompressor().stream_reader(f).read(BPP * 12288)
            if len(raw) < BPP * 12288:
                continue
        except Exception as e:
            print(f'  skip {os.path.basename(path)}: {e}', file=sys.stderr)
            continue

        arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, BPP)
        wdlq = np.frombuffer(arr[:, OFFSET_WDLQ:OFFSET_WDLQ+12].tobytes(),
                             dtype=np.float32).reshape(-1, 3)
        dqvv = np.frombuffer(arr[:, OFFSET_DQVV:OFFSET_DQVV+4].tobytes(),
                             dtype=np.float32)
        kld  = np.frombuffer(arr[:, OFFSET_KLD:OFFSET_KLD+4].tobytes(),
                             dtype=np.float32)
        search_q = wdlq[:, 0] - wdlq[:, 2]

        mask = (np.abs(dqvv) >= args.min_dqvv) & (kld >= args.min_kld)
        if args.min_abs_q > 0:
            mask &= np.abs(search_q) >= args.min_abs_q

        # Top-K policy P per record — needed for filter and ranking.
        # The policy array has 92 slots; some slots may be DUPLICATES of the same move idx
        # (padding behavior). Compute true top-1 P and true top-2 *unique-idx* P.
        pol_val_all = np.frombuffer(arr[:, OFFSET_POLV:OFFSET_POLV+184].tobytes(),
                                    dtype=np.float16).reshape(-1, 92).astype(np.float32)
        pol_idx_all = np.frombuffer(arr[:, OFFSET_POLI:OFFSET_POLI+184].tobytes(),
                                    dtype=np.int16).reshape(-1, 92)
        top_slot = np.argmax(pol_val_all, axis=1)
        ar = np.arange(pol_val_all.shape[0])
        top_p = pol_val_all[ar, top_slot]
        top_idx_val = pol_idx_all[ar, top_slot]
        # Mask out all slots whose idx equals the top idx (handles duplicate-padding)
        masked = np.where(pol_idx_all == top_idx_val[:, None], -1.0, pol_val_all)
        top2_p = masked.max(axis=1)
        # If everything was masked (single-move policy), top2_p will be -1 — clamp to 0
        top2_p = np.maximum(top2_p, 0.0)

        if args.max_top_p < 1.0:
            mask &= top_p <= args.max_top_p
        if args.min_top_p > 0.0:
            mask &= top_p >= args.min_top_p
        if args.min_top2_p > 0.0:
            mask &= top2_p >= args.min_top2_p

        # Piece count + material balance from plane-0 of square encoding
        if args.min_pieces > 0 or args.max_material_imbalance < 99:
            sq = arr[:, OFFSET_SQ:OFFSET_SQ + 64*137].reshape(-1, 64, 137)
            # Plane 0 layout: byte 0 = empty, bytes 1..6 = our P/N/B/R/Q/K, bytes 7..12 = opp P/N/B/R/Q/K
            # Counts per side use ByteScaled (0 or 100); we just check >0 across squares.
            # Standard piece values: P=1, N=3, B=3, R=5, Q=9, K=0
            VALUES = np.array([1, 3, 3, 5, 9, 0], dtype=np.float32)  # P, N, B, R, Q, K
            our_counts = np.array([(sq[:, :, 1+i] > 0).sum(axis=1) for i in range(6)]).T  # (N, 6)
            opp_counts = np.array([(sq[:, :, 7+i] > 0).sum(axis=1) for i in range(6)]).T  # (N, 6)
            our_material = (our_counts * VALUES).sum(axis=1)
            opp_material = (opp_counts * VALUES).sum(axis=1)
            # Imbalance from side-to-move's perspective; we just want |our - opp|
            imbalance = np.abs(our_material - opp_material)
            piece_counts = our_counts.sum(axis=1) + opp_counts.sum(axis=1) + 2  # +2 for two kings
            if args.min_pieces > 0:
                mask &= piece_counts >= args.min_pieces
            if args.max_material_imbalance < 99:
                mask &= imbalance <= args.max_material_imbalance

        hits = np.where(mask)[0]
        # "Hardness" composite ranking score (training-data quality):
        #   KLD            : NN policy was wrong (high = NN sharply misaligned with search)
        #   |DQVV|         : NN value was wrong
        #   |Q|            : position is decisive (wrong move costs real points)
        #   top_p          : search converged on a clear best move (sharp supervision target)
        # Note: prefer decisive search to get well-defined training targets.
        # Use --min-top-p to set a floor (default 0.0) and exclude flat distributions.
        scores = kld[hits] * np.abs(dqvv[hits]) * np.abs(search_q[hits]) * top_p[hits]
        order = np.argsort(-scores)
        for h in order:
            if len(found) >= args.n:
                break
            idx = hits[h]
            found.append({'rec': arr[idx].copy(),
                          'src_file': os.path.basename(path),
                          'src_rec': int(idx)})
        print(f'  [{fi+1}/{len(inputs)}] {os.path.basename(path):40s} '
              f'matches={len(hits):>5d}  cumulative kept={len(found)}')

    if not found:
        print('No matches.', file=sys.stderr)
        sys.exit(1)

    # Write markdown
    with open(args.output_md, 'w', encoding='utf-8') as out:
        out.write(f'# Tactical TPG records — {len(found)} samples\n\n')
        out.write(f'Filter: `|DQVV| >= {args.min_dqvv}` AND `KLDPolicy >= {args.min_kld}`')
        if args.min_abs_q > 0:
            out.write(f' AND `|search_Q| >= {args.min_abs_q}`')
        out.write(f'\n\nRanked by `KLD * |DQVV| * |Q| * top_p` (descending) — '
                  f'favors positions where search converged decisively (clear training target) '
                  f'AND both NN heads were wrong.\n\n')
        out.write('FEN castling/EP/halfmove are defaults (`- - 0 1`); '
                  'paste the FEN into a chess GUI to view position.\n\n')
        out.write('Top-K moves are the search\'s most-visited (highest policy mass) — '
                  'a Lc0 1858-move map is needed to convert idx→UCI; for now use any '
                  'Lc0-aware tool to map.\n\n---\n\n')

        for ri, rec_info in enumerate(found):
            r = rec_info['rec']
            wdl_nd = np.frombuffer(r[OFFSET_WDLND:OFFSET_WDLND+12].tobytes(), dtype=np.float32)
            wdl_d  = np.frombuffer(r[OFFSET_WDLD:OFFSET_WDLD+12].tobytes(),   dtype=np.float32)
            wdl_q  = np.frombuffer(r[OFFSET_WDLQ:OFFSET_WDLQ+12].tobytes(),   dtype=np.float32)
            pq_sub = float(np.frombuffer(r[OFFSET_PQSUB:OFFSET_PQSUB+4].tobytes(), dtype=np.float32)[0])
            kld    = float(np.frombuffer(r[OFFSET_KLD:OFFSET_KLD+4].tobytes(),  dtype=np.float32)[0])
            mlh    = float(np.frombuffer(r[OFFSET_MLH:OFFSET_MLH+4].tobytes(),  dtype=np.float32)[0])
            dqvv   = float(np.frombuffer(r[OFFSET_DQVV:OFFSET_DQVV+4].tobytes(),dtype=np.float32)[0])
            pip    = int(np.frombuffer(r[OFFSET_PIP:OFFSET_PIP+2].tobytes(), dtype=np.int16)[0])
            pol_idx = np.frombuffer(r[OFFSET_POLI:OFFSET_POLI+184].tobytes(), dtype=np.int16)
            pol_val = np.frombuffer(r[OFFSET_POLV:OFFSET_POLV+184].tobytes(), dtype=np.float16)
            stm = bool(r[OFFSET_STM])
            search_qv = wdl_q[0] - wdl_q[2]
            # DeltaQVersusV is stored as |OriginalQ - BestQ| (sign lost).
            # Pick the in-range candidate; if both candidates are in [-1,+1] it's ambiguous.
            v_below = search_qv - dqvv
            v_above = search_qv + dqvv
            in_range_below = -1.0 <= v_below <= 1.0
            in_range_above = -1.0 <= v_above <= 1.0
            if in_range_below and not in_range_above:
                raw_v_str = f'{v_below:+.4f}'
            elif in_range_above and not in_range_below:
                raw_v_str = f'{v_above:+.4f}'
            elif in_range_below and in_range_above:
                raw_v_str = f'{v_below:+.4f} OR {v_above:+.4f} (sign-ambiguous)'
            else:
                # Neither in range — clamp both, show closest-to-valid
                raw_v_str = f'(out-of-range; |DQVV|={dqvv:.3f} > {1+abs(search_qv):.3f})'

            fen, ascii_board = decode_position_to_fen(r)

            out.write(f'## Record {ri+1} — `{rec_info["src_file"]}` rec {rec_info["src_rec"]}\n\n')
            out.write(f'**FEN**: `{fen}`\n\n')
            out.write('```\n' + ascii_board + '\n```\n\n')
            out.write(f'- **Side to move**: {"White" if stm else "Black"}\n')
            out.write(f'- **DeltaQVersusV** ⭐: `{dqvv:+.4f}`  (search Q − raw V)\n')
            out.write(f'- **KLDPolicy** ⭐: `{kld:.4f}`  (policy head vs search visits)\n')
            out.write(f'- **WDLQ search** : W={wdl_q[0]:.3f} D={wdl_q[1]:.3f} L={wdl_q[2]:.3f} → Q=`{search_qv:+.4f}`\n')
            out.write(f'- **Raw NN V**   : `{raw_v_str}`  (DQVV is unsigned; in-range candidate shown)\n')
            out.write(f'- **WDLDeblund (target)**     : W={wdl_d[0]:.3f} D={wdl_d[1]:.3f} L={wdl_d[2]:.3f}\n')
            out.write(f'- **WDLNonDebl (game outcome)**: W={wdl_nd[0]:.3f} D={wdl_nd[1]:.3f} L={wdl_nd[2]:.3f}\n')
            out.write(f'- **PlayedMoveQSubopt** (note: noisy signal): `{pq_sub:+.4f}`\n')
            out.write(f'- **MLH (raw)**: {mlh:.2f}  → decoded plies-left ≈ {((mlh/0.1)**2)/100:.1f}\n')
            out.write(f'- **PolicyIndexInParent**: {pip}  (move played from parent → this position)\n')

            order = np.argsort(-pol_val.astype(np.float32))
            out.write(f'\n**Top {args.top_k} unique policy moves at this position** (search recommended):\n\n')
            out.write('| Rank | Idx | P |\n|---|---|---|\n')
            seen = set()
            shown = 0
            for slot in order:
                idx = int(pol_idx[slot])
                if idx in seen:
                    continue
                seen.add(idx)
                out.write(f'| {shown+1} | {idx} | {float(pol_val[slot]):.4f} |\n')
                shown += 1
                if shown >= args.top_k:
                    break
            out.write('\n---\n\n')

    print(f'\nWrote {len(found)} records to {args.output_md}')


if __name__ == '__main__':
    main()
