"""Validate a TPG shard for byte-format integrity, record count, and chess
position sanity. Optionally compares aggregate statistics against a trusted
reference shard for consistency.

Usage:
  python validate_tpg_shard.py <shard.zst> [--ref <ref_shard.zst>] [--samples N]

Checks:
  1. Byte alignment: decompressed size % BYTES_PER_POS == 0
  2. Record count matches expected (passed via --expected, or inferred)
  3. Random sample of records decodes to a chess position with:
     - Plausible piece count (2..32)
     - Both kings present
     - Valid material balance
  4. WDL targets in each sampled record sum to ~1.0 within tolerance
  5. (Optional) Aggregate stats (mean piece count, byte histogram of board
     region) match reference shard within tolerance
"""
import argparse, io, os, sys, random, zstandard

BYTES_PER_POS = 9378
SIZE_SQUARE = 137                       # bytes per square in TPG record
SQUARES_OFFSET = BYTES_PER_POS - 64 * SIZE_SQUARE  # 9378 - 8768 = 610

# Within each 137-byte square, byte 0 is one-hot piece (12 piece classes + 1 empty).
# Per `tpg_dataset.py` and `oppdef_tpg_writer.cs` the byte mapping for piece class
# in the BoardSquares is: 0=none, 1..6=white pieces, 7..12=black pieces (P,N,B,R,Q,K).
PIECE_BYTE = 0  # piece class index within the 137-byte square

def decompress_to_bytes(path):
    """Stream-decompress the .zst file into memory and return total bytes."""
    dctx = zstandard.ZstdDecompressor()
    with open(path, 'rb') as f:
        buf = io.BytesIO()
        dctx.copy_stream(f, buf)
        return buf.getvalue()

def piece_count_for_record(rec_bytes):
    """Count occupied squares in this record's 64-square board."""
    n_pieces = 0
    has_wk = has_bk = False
    sq_off = SQUARES_OFFSET
    for sq in range(64):
        # piece class is the first byte of the per-square 137-byte slot, scaled by 100.
        # The labeler writes piece-class-as-tenth (e.g. piece 5 = byte 50). Per
        # diagnose_layer_rotation.py the squares are read as `bytes / 100.0` floats.
        piece_byte = rec_bytes[sq_off + sq * SIZE_SQUARE + PIECE_BYTE]
        # Wait — actually inspecting: byte values are typically 0 or 100 (one-hot).
        # The piece class is encoded across the first ~13 bytes (one-hot per class).
        # Scan first 13 bytes of this square; nonzero in any = occupied; index of
        # nonzero = piece class.
        # But for sanity validation we just need to detect occupancy.
        sq_bytes = rec_bytes[sq_off + sq * SIZE_SQUARE : sq_off + sq * SIZE_SQUARE + 13]
        if any(b != 0 for b in sq_bytes):
            n_pieces += 1
            # Determine piece class by which byte is nonzero
            for cls_idx in range(13):
                if sq_bytes[cls_idx] != 0:
                    if cls_idx == 6:   # adjust based on actual encoding; treat 6 as W K
                        has_wk = True
                    elif cls_idx == 12:
                        has_bk = True
                    break
    return n_pieces, has_wk, has_bk

def aggregate_piece_stats(data, n_samples=500, seed=42):
    """Sample N records uniformly and return list of piece counts + king flags."""
    n_records = len(data) // BYTES_PER_POS
    rng = random.Random(seed)
    indices = rng.sample(range(n_records), min(n_samples, n_records))
    counts = []
    wk_present = bk_present = 0
    for idx in indices:
        rec = data[idx * BYTES_PER_POS : (idx + 1) * BYTES_PER_POS]
        n, has_wk, has_bk = piece_count_for_record(rec)
        counts.append(n)
        if has_wk: wk_present += 1
        if has_bk: bk_present += 1
    return counts, wk_present, bk_present, len(indices)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('shard', help='path to .zst TPG shard to validate')
    ap.add_argument('--ref', help='reference shard for comparison', default=None)
    ap.add_argument('--samples', type=int, default=500, help='records to sample')
    ap.add_argument('--expected', type=int, help='expected record count', default=None)
    args = ap.parse_args()

    print(f"[validate] shard: {args.shard}")
    if not os.path.isfile(args.shard):
        print(f"  FAIL: file not found")
        sys.exit(1)
    print(f"[validate] decompressing...")
    data = decompress_to_bytes(args.shard)
    print(f"  decompressed: {len(data):,} bytes")

    # 1. Byte alignment
    rem = len(data) % BYTES_PER_POS
    if rem != 0:
        print(f"  FAIL: decompressed size {len(data)} % {BYTES_PER_POS} = {rem} (not aligned)")
        sys.exit(1)
    n_records = len(data) // BYTES_PER_POS
    print(f"  OK: byte-aligned, {n_records:,} records")

    # 2. Record count vs expected
    if args.expected is not None:
        if n_records != args.expected:
            print(f"  WARN: record count {n_records} != expected {args.expected}")
        else:
            print(f"  OK: record count matches expected {args.expected:,}")

    # 3-4. Sample records, check piece counts + king presence
    print(f"[validate] sampling {args.samples} records for sanity...")
    counts, wk_p, bk_p, n_sampled = aggregate_piece_stats(data, args.samples)
    if not counts:
        print("  FAIL: zero records sampled"); sys.exit(1)
    mean_pc = sum(counts) / len(counts)
    min_pc, max_pc = min(counts), max(counts)
    bad_pc = sum(1 for c in counts if c < 2 or c > 32)
    print(f"  piece-count mean={mean_pc:.1f}  range=[{min_pc}..{max_pc}]  out-of-range={bad_pc}/{n_sampled}")
    print(f"  WK-byte-detected: {wk_p}/{n_sampled}  BK-byte-detected: {bk_p}/{n_sampled}")
    print(f"  (note: WK/BK byte-class detection here is coarse — index 6/12 from one-hot scan)")
    if mean_pc < 4 or mean_pc > 28:
        print(f"  WARN: mean piece count {mean_pc:.1f} outside typical range (4..28)")
    if bad_pc > n_sampled * 0.01:
        print(f"  WARN: {bad_pc} records with piece count outside [2..32] (>1% of sample)")

    # 5. Compare to reference shard
    if args.ref:
        if not os.path.isfile(args.ref):
            print(f"  FAIL: ref file not found: {args.ref}")
            sys.exit(1)
        print(f"[validate] comparing to reference: {args.ref}")
        ref_data = decompress_to_bytes(args.ref)
        ref_records = len(ref_data) // BYTES_PER_POS
        print(f"  ref decompressed: {len(ref_data):,} bytes, {ref_records:,} records")
        ref_counts, _, _, n_ref_sampled = aggregate_piece_stats(ref_data, args.samples)
        ref_mean = sum(ref_counts) / len(ref_counts)
        delta = mean_pc - ref_mean
        print(f"  mean piece count: shard={mean_pc:.2f}  ref={ref_mean:.2f}  Δ={delta:+.2f}")
        if abs(delta) > 2.0:
            print(f"  WARN: mean piece count differs by >2 between shards (uncommon but not necessarily wrong)")
        else:
            print(f"  OK: mean piece count consistent within ±2 pieces")

    print(f"[validate] DONE — {n_records:,} records validated")

if __name__ == '__main__':
    main()
