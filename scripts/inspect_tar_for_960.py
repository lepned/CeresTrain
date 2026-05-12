"""
Sample Lc0 training TAR files to estimate the ratio of Chess960 vs standard-chess games.

Logic:
  - Each TAR contains many .gz files (one per game).
  - Each game's first position (8356 bytes for V6 format) carries the board state
    of the start of that game.
  - We extract the 832-byte BoardsHistory blob (bytes [7440 : 7440+832]) and
    fingerprint it. Standard-chess games all share the same fingerprint at move 0
    (after the constant Lc0 mirror); 960 games each produce a different one.
  - We pick the most common fingerprint as the "standard-chess reference" and
    count games matching vs not matching it.

Usage:
    python3 inspect_tar_for_960.py <tar-dir> [num-tars] [games-per-tar]

Defaults: 10 TARs sampled, 100 games each.
"""
import os
import sys
import glob
import tarfile
import gzip
import hashlib
import random
from collections import Counter

V6_LEN = 8356
BOARDS_OFFSET = 8 + 1858 * 4         # = 7440 (after Version + InputFormat + Policies)
BOARDS_LEN = 832                     # 8 history boards × 13 planes × 8 bytes


def fingerprint_first_position(record_bytes):
    """Return SHA-1 of the BoardsHistory blob from the first V6 record."""
    if len(record_bytes) < V6_LEN:
        return None
    boards = record_bytes[BOARDS_OFFSET : BOARDS_OFFSET + BOARDS_LEN]
    return hashlib.sha1(boards).hexdigest()


def sample_tar(tar_path, num_games=100):
    """Return list of fingerprints from the first num_games games in this TAR."""
    fingerprints = []
    errors = 0
    with tarfile.open(tar_path, 'r') as tf:
        for member in tf:
            if not member.isfile():
                continue
            if len(fingerprints) >= num_games:
                break
            try:
                f = tf.extractfile(member)
                if f is None:
                    continue
                raw = f.read()
                # Game files are typically gzipped inside the TAR
                if raw[:2] == b'\x1f\x8b':
                    try:
                        decompressed = gzip.decompress(raw)
                    except Exception:
                        errors += 1
                        continue
                else:
                    decompressed = raw
                fp = fingerprint_first_position(decompressed)
                if fp:
                    fingerprints.append(fp)
                else:
                    errors += 1
            except Exception:
                errors += 1
    return fingerprints, errors


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    tar_dir = sys.argv[1]
    num_tars = int(sys.argv[2]) if len(sys.argv) >= 3 else 10
    games_per_tar = int(sys.argv[3]) if len(sys.argv) >= 4 else 100

    all_tars = sorted(glob.glob(os.path.join(tar_dir, '*.tar')))
    if not all_tars:
        print(f'No .tar files in {tar_dir}'); sys.exit(1)

    # Reproducible random sample
    random.seed(42)
    sampled = random.sample(all_tars, min(num_tars, len(all_tars)))
    sampled.sort()
    print(f'Sampling {len(sampled)} of {len(all_tars)} TARs, up to {games_per_tar} games each')
    print('-' * 80)

    all_fingerprints = []
    total_errors = 0
    for tar_path in sampled:
        fps, errs = sample_tar(tar_path, num_games=games_per_tar)
        all_fingerprints.extend(fps)
        total_errors += errs
        # Per-TAR breakdown
        c = Counter(fps)
        top = c.most_common(1)
        unique = len(c)
        most = top[0][1] if top else 0
        print(f'  {os.path.basename(tar_path):60s}  games={len(fps):4d}  unique-fingerprints={unique:4d}  most-common={most}')

    print('-' * 80)
    print(f'\nTotal games sampled: {len(all_fingerprints)}')
    print(f'Decode errors:       {total_errors}')

    counts = Counter(all_fingerprints)
    if not counts:
        print('No fingerprints collected.'); sys.exit(1)

    # The most common fingerprint is the standard-chess starting position
    # (every standard game shares it; 960 games each have unique starts).
    ranked = counts.most_common(10)
    std_fp, std_n = ranked[0]
    total = sum(counts.values())
    non_std = total - std_n

    print(f'\nMost common fingerprint: {std_fp[:16]}...   count={std_n}  ({100.0*std_n/total:.2f}% of sampled games)')
    print(f'  → assumed to be the STANDARD chess starting position')
    print(f'\nNon-standard fingerprints: {non_std}  ({100.0*non_std/total:.2f}% of sampled games)')
    print(f'Unique fingerprints total: {len(counts)}')

    print('\nTop fingerprints by frequency:')
    for fp, n in ranked:
        marker = '  (STANDARD)' if fp == std_fp else ''
        print(f'  {fp[:16]}...  count={n:4d}{marker}')

    print()
    if non_std == 0:
        print('VERDICT: All sampled games appear standard chess (single shared start position).')
    else:
        pct_960 = 100.0 * non_std / total
        print(f'VERDICT: {non_std} games ({pct_960:.2f}%) have non-standard starting positions — likely Chess960.')


if __name__ == '__main__':
    main()
