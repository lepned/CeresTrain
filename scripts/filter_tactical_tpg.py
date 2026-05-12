#!/usr/bin/env python3
"""Filter TPG records by tactical signal (|DeltaQVersusV|).

Reads .zst TPG files from <input_dir>, retains records where the search-Q vs
raw-NN-V gap exceeds threshold (and optionally where |search_Q| is decisive),
and emits drop-in-compatible .zst shards to <output_dir>.

Output preserves the 9378-byte TPG record layout exactly. Each output shard is
written in BYTES_PER_BLOCK-aligned chunks (12,288 records = 3*4096), so the
trainer's tpg_dataset.py loader reads them without modification.

Usage:
    python filter_tactical_tpg.py <input_dir> <output_dir>
        [--min-dqvv 0.30]
        [--min-abs-q 0.0]
        [--num-output-sets 16]
        [--max-records-out 0]
        [--zstd-level 11]
        [--only-closed-files]

Examples:
    # Sharp tactical (|DQVV| >= 0.30, ~4% yield) into 16 round-robin shards
    python filter_tactical_tpg.py E:/T80_tpg E:/T80_tactical --min-dqvv 0.30

    # Very sharp + decisive (~2% yield)
    python filter_tactical_tpg.py E:/T80_tpg E:/T80_tactical \\
        --min-dqvv 0.30 --min-abs-q 0.50
"""
import argparse, glob, os, sys, time
import numpy as np
import zstandard

BYTES_PER_POS = 9378
BATCH_SIZE = 4096
POS_PER_BLOCK = 12288  # = 3 * BATCH_SIZE
BYTES_PER_BLOCK = POS_PER_BLOCK * BYTES_PER_POS  # ~115 MB raw

# TPGRecord field offsets (from Ceres TPGRecord.cs)
OFFSET_WDLQ = 24            # 3 * float32 = 12 bytes  (search-Q WDL)
OFFSET_PQSUB = 36           # 1 * float32 = 4 bytes   (PlayedMoveQSuboptimality)
OFFSET_KLD = 96             # 1 * float32 = 4 bytes   (KLDPolicy)
OFFSET_DQVV = 104           # 1 * float32 = 4 bytes   (DeltaQVersusV)
OFFSET_POLI = 242           # 92 * int16 = 184 bytes
OFFSET_POLV = 426           # 92 * float16 = 184 bytes
OFFSET_SQ = 610             # 64 * 137 bytes


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('input_dir', help='Directory containing input .zst files')
    ap.add_argument('output_dir', help='Output directory for filtered .zst files')
    ap.add_argument('--min-dqvv', type=float, default=0.20,
                    help='Keep records where |DeltaQVersusV| >= this. NN value disagreed with search. '
                         '(NN here = the Lc0 net that played the T80 game, not necessarily orig.) (default 0.20)')
    ap.add_argument('--min-kld', type=float, default=1.0,
                    help='Keep records where KLDPolicy >= this. NN policy disagreed with search. '
                         'Lower threshold (1.0 vs 3.0) to capture universally-hard positions, not just Lc0-specific ones. '
                         '(default 1.0)')
    ap.add_argument('--min-abs-q', type=float, default=0.60,
                    help='Keep only when |search_Q| >= this. Decisive position — wrong move costs the game. '
                         '(default 0.60)')
    ap.add_argument('--min-top-p', type=float, default=0.20,
                    help='Keep records where search top-1 policy P >= this. '
                         'Ensures search has a clear preference (training target). (default 0.20)')
    ap.add_argument('--max-top-p', type=float, default=0.50,
                    help='Keep records where search top-1 policy P <= this. '
                         'Excludes trivially-converged positions where the answer is overwhelming. (default 0.50)')
    ap.add_argument('--min-top2-p', type=float, default=0.10,
                    help='Keep records where second-best move policy P >= this. '
                         'Excludes pure forcing-line positions where only one move makes sense. (default 0.10)')
    ap.add_argument('--min-pieces', type=int, default=14,
                    help='Minimum total piece count on board. Excludes endgame trivials, '
                         'forces real middlegame complexity. (default 14)')
    ap.add_argument('--min-played-q-subopt', type=float, default=0.0,
                    help='Additional filter (DEPRECATED for hard-position discovery): '
                         'noisy in self-play data due to temperature/Dirichlet sampling. '
                         '(default 0.0 = no filter)')
    ap.add_argument('--num-output-sets', type=int, default=16,
                    help='Round-robin kept records across this many output shards '
                         '(default 16, matches gen-tpg convention)')
    ap.add_argument('--max-records-out', type=int, default=0,
                    help='Stop after writing N records total (0 = no limit)')
    ap.add_argument('--zstd-level', type=int, default=11,
                    help='Zstd compression level (default 11, matches gen-tpg)')
    ap.add_argument('--only-closed-files', action='store_true',
                    help='Skip files modified within last 60s (avoids races with active gen-tpg)')
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    inputs = sorted(glob.glob(os.path.join(args.input_dir, '*.zst')))
    if not inputs:
        print(f'ERROR: no .zst files in {args.input_dir}', file=sys.stderr)
        sys.exit(1)

    if args.only_closed_files:
        cutoff = time.time() - 60
        inputs = [p for p in inputs if os.path.getmtime(p) < cutoff]
        if not inputs:
            print('ERROR: no closed (>60s old) .zst files', file=sys.stderr)
            sys.exit(1)

    print(f'Inputs: {len(inputs)} .zst files in {args.input_dir}')
    fparts = [f'|DQVV| >= {args.min_dqvv}', f'KLDPolicy >= {args.min_kld}']
    if args.min_abs_q > 0:
        fparts.append(f'|Q| >= {args.min_abs_q}')
    if args.min_top_p > 0:
        fparts.append(f'top-1 P >= {args.min_top_p}')
    if args.min_pieces > 0:
        fparts.append(f'pieces >= {args.min_pieces}')
    if args.min_played_q_subopt > 0:
        fparts.append(f'PlayedQSubopt >= {args.min_played_q_subopt}')
    print(f'Filter: {" AND ".join(fparts)}')
    print(f'Output: {args.num_output_sets} shards in {args.output_dir}')

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    base = os.path.join(args.output_dir, f'tactical_{timestamp}')
    cctx = zstandard.ZstdCompressor(level=args.zstd_level)
    out_paths = [f'{base}.tpg_set{i}.zst' for i in range(args.num_output_sets)]
    out_fhs = [open(p, 'wb') for p in out_paths]
    out_zs = [cctx.stream_writer(fh) for fh in out_fhs]
    out_buffers = [bytearray() for _ in range(args.num_output_sets)]
    out_counts = [0] * args.num_output_sets

    BLOCK_BYTES = POS_PER_BLOCK * BYTES_PER_POS

    def flush_block_aligned(idx, force_partial=False):
        """Write whole BLOCK-aligned chunks from buffer to output stream.

        Trainer reads BYTES_PER_BLOCK at a time; partial blocks at file end are
        skipped silently. So we only emit POS_PER_BLOCK-multiples per file.
        """
        buf = out_buffers[idx]
        n_records = len(buf) // BYTES_PER_POS
        n_blocks = n_records // POS_PER_BLOCK
        n_to_write = n_blocks * POS_PER_BLOCK
        if n_to_write == 0:
            return 0
        nbytes = n_to_write * BYTES_PER_POS
        out_zs[idx].write(bytes(buf[:nbytes]))
        del buf[:nbytes]
        out_counts[idx] += n_to_write
        return n_to_write

    n_in = 0
    n_kept = 0
    next_out = 0
    t_start = time.time()
    stopped_early = False

    for fi, in_path in enumerate(inputs):
        try:
            fh = open(in_path, 'rb')
            rdr = zstandard.ZstdDecompressor().stream_reader(fh)
        except Exception as e:
            print(f'  [{fi+1}/{len(inputs)}] OPEN-ERR {os.path.basename(in_path)}: {e}', file=sys.stderr)
            continue

        file_in = 0
        file_kept = 0
        try:
            while True:
                block = rdr.read(BLOCK_BYTES)
                if not block or len(block) < BLOCK_BYTES:
                    break  # partial trailing block — skip (matches trainer behavior)

                arr = np.frombuffer(block, dtype=np.uint8).reshape(-1, BYTES_PER_POS)
                # Vectorized field extraction
                wdlq_bytes = arr[:, OFFSET_WDLQ:OFFSET_WDLQ + 12].tobytes()
                wdlq = np.frombuffer(wdlq_bytes, dtype=np.float32).reshape(-1, 3)
                pqsub = np.frombuffer(arr[:, OFFSET_PQSUB:OFFSET_PQSUB + 4].tobytes(),
                                      dtype=np.float32)
                kld = np.frombuffer(arr[:, OFFSET_KLD:OFFSET_KLD + 4].tobytes(),
                                    dtype=np.float32)
                dqvv = np.frombuffer(arr[:, OFFSET_DQVV:OFFSET_DQVV + 4].tobytes(),
                                     dtype=np.float32)
                search_q = wdlq[:, 0] - wdlq[:, 2]

                mask = (np.abs(dqvv) >= args.min_dqvv) & (kld >= args.min_kld)
                if args.min_abs_q > 0:
                    mask &= np.abs(search_q) >= args.min_abs_q
                if args.min_played_q_subopt > 0:
                    mask &= pqsub >= args.min_played_q_subopt
                # Top-K policy P: also need top-2 unique-idx P for filter
                pol_val_all = np.frombuffer(arr[:, OFFSET_POLV:OFFSET_POLV + 184].tobytes(),
                                            dtype=np.float16).reshape(-1, 92).astype(np.float32)
                pol_idx_all = np.frombuffer(arr[:, OFFSET_POLI:OFFSET_POLI + 184].tobytes(),
                                            dtype=np.int16).reshape(-1, 92)
                top_slot_b = np.argmax(pol_val_all, axis=1)
                ar = np.arange(pol_val_all.shape[0])
                top_p = pol_val_all[ar, top_slot_b]
                top_idx_val = pol_idx_all[ar, top_slot_b]
                masked = np.where(pol_idx_all == top_idx_val[:, None], -1.0, pol_val_all)
                top2_p = np.maximum(masked.max(axis=1), 0.0)
                if args.min_top_p > 0:
                    mask &= top_p >= args.min_top_p
                if args.max_top_p < 1.0:
                    mask &= top_p <= args.max_top_p
                if args.min_top2_p > 0:
                    mask &= top2_p >= args.min_top2_p
                if args.min_pieces > 0:
                    sq = arr[:, OFFSET_SQ:OFFSET_SQ + 64*137].reshape(-1, 64, 137)
                    occupied = (sq[:, :, 1:13] > 0).any(axis=2)
                    mask &= occupied.sum(axis=1) >= args.min_pieces

                kept = arr[mask]
                file_in += arr.shape[0]
                file_kept += kept.shape[0]

                # Round-robin distribute records across output shards
                # (vectorized: split kept into N_SETS chunks by row index mod N_SETS)
                if kept.shape[0] > 0:
                    n_sets = args.num_output_sets
                    # Compute target shard for each kept row
                    targets = (np.arange(kept.shape[0]) + next_out) % n_sets
                    for s in range(n_sets):
                        sel = kept[targets == s]
                        if sel.shape[0]:
                            out_buffers[s].extend(sel.tobytes())
                    next_out = (next_out + kept.shape[0]) % n_sets

                # Flush full BLOCK-aligned chunks per shard
                for i in range(args.num_output_sets):
                    if len(out_buffers[i]) >= BLOCK_BYTES:
                        flush_block_aligned(i)

                if args.max_records_out and sum(out_counts) >= args.max_records_out:
                    stopped_early = True
                    break
        except Exception as e:
            print(f'  [{fi+1}/{len(inputs)}] READ-ERR {os.path.basename(in_path)}: {e}',
                  file=sys.stderr)
        finally:
            try: rdr.close()
            except Exception: pass
            try: fh.close()
            except Exception: pass

        n_in += file_in
        n_kept += file_kept
        elapsed = time.time() - t_start
        rate = n_in / elapsed if elapsed > 0 else 0
        kp = 100 * file_kept / max(file_in, 1)
        print(f'  [{fi+1}/{len(inputs)}] {os.path.basename(in_path):40s} '
              f'in={file_in:>9,} kept={file_kept:>8,} ({kp:.2f}%) '
              f'rate={rate:>8,.0f} pos/s')

        if stopped_early:
            print(f'  Reached max-records-out={args.max_records_out}, stopping.')
            break

    # Final flush
    for i in range(args.num_output_sets):
        flush_block_aligned(i)

    # Close streams
    for zs in out_zs: zs.close()
    for fh in out_fhs: fh.close()

    # Report per-shard sizes (some shards may have 0 records if filter is sharp)
    print()
    print('=== Final ===')
    print(f'Input records:   {n_in:>14,}')
    print(f'Filter passed:   {n_kept:>14,}  ({100*n_kept/max(n_in,1):.3f}%)')
    print(f'Records written: {sum(out_counts):>14,}  '
          f'(BLOCK-aligned; {n_kept - sum(out_counts):,} partial-block records dropped)')
    print(f'Output dir:      {args.output_dir}')
    print(f'Per-shard counts:')
    for i, (c, p) in enumerate(zip(out_counts, out_paths)):
        size = os.path.getsize(p)
        size_mb = size / (1024 * 1024)
        if c == 0:
            os.remove(p)  # clean up empty output files
            print(f'  set{i:>2}: {c:>10,} records  ({size_mb:>7.1f} MB) [empty, removed]')
        else:
            print(f'  set{i:>2}: {c:>10,} records  ({size_mb:>7.1f} MB)')
    print(f'Elapsed: {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    main()
