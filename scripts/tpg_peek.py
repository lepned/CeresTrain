"""Corpus-level summary of raw TPG shards (no model, no GPU, CPU only).

Answers "what does the training data actually look like": how sharp the policy
targets are, how drawish the value targets are, how often deblundering rewrites
the game result, and how two corpora in a mix compare to each other.

Usage:
  python scripts/tpg_peek.py <shard-or-dir> [<shard-or-dir> ...] [--mb 120] [--json out.json]
  python scripts/tpg_peek.py D:/T80_v2_sub D:/TPG_TAR_LC0_T91_combined_v3 --mb 120

Square width (137 = V2 shards, 141 = V3 shards with 4 aux bytes) is detected per
shard from the data itself (see `detect_square_bytes`), so mixed-corpus runs like
T80_v2 + T91 need no per-path flags. Override with --square-bytes if detection
is ever ambiguous.

WHY THIS EXISTS / WHY IT IS NOT tpg_eval.py
-------------------------------------------
`scripts/tpg_eval.py` carries a banner recording that its results were measured
untrustworthy on 2026-08-17: it ranked three checkpoints in the exact reverse of
their known EngineBattle gates, because its BYTES_PER_POS / SQUARES_OFFSET /
POLICIES_INDICES_OFFSET constants were hand-derived and never cross-checked.

This reader does not repeat that. The record layout is written once, as a table
of (name, size) fields, and three independent checks must pass:

  1. The fields sum to exactly BYTES_PER_POS - 64*square_bytes, i.e. the header
     and the policy block together account for every byte before the squares.
     Asserted at import for BOTH 137 and 141 widths.
  2. BYTES_PER_POS reproduces tpg_dataset.py's own on-disk constants
     (9378 for V2, 9634 for V3). Covered by test_tpg_peek.py, which reads those
     numbers out of tpg_dataset.py rather than trusting a copy here.
  3. Per shard at runtime, the decoded wdl_q rows must be valid probability
     vectors. A wrong stride desynchronises the view within a few hundred
     records and this check collapses -- which is what drives the width
     detection and what makes a silently wrong decode loud instead of quiet.

If you change the layout, change LAYOUT and let the checks fail; do not patch a
raw offset.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import zstandard

# TPGRecord.MAX_MOVES -- mirrored from src/CeresTrainPy/tpg_dataset.py (MAX_MOVES = 92).
# test_tpg_peek.py asserts this still matches that file.
MAX_MOVES = 92

# Record layout, matching the V2 TPGRecord struct in TPGRecord.cs
# (LayoutKind.Sequential, Pack=1) as decoded by tpg_dataset.py. Sizes in bytes.
# `None` name = present on disk but not decoded here.
LAYOUT = [
    ("wdl_nondeblundered", 3 * 4),
    ("wdl_deblundered", 3 * 4),
    ("wdl_q", 3 * 4),
    ("played_q_suboptimality", 1 * 4),
    (None, 4 + 42),            # IsWhiteToMove, Unused1, PUNIMSelf/Opponent, UnusedArray[42]
    (None, 4 + 2 + 2 + 2),     # NumSearchNodes, RefModel1 NumNodes/Value, RefModel1BestMove
    ("kld_policy", 1 * 4),     # KLDPolicy; trainer reuses it as uncertainty_policy
    ("mlh_raw", 1 * 4),
    ("delta_q_versus_v", 1 * 4),   # trainer reuses it as uncertainty
    ("q_deviation_lower", 1 * 2),
    ("q_deviation_upper", 1 * 2),
    ("policy_index_in_parent", 1 * 2),
    (None, 64 + 64),           # PlyUntilSquareChangePiece, PlyUntilSquarePieceCapture
    ("policies_indices", MAX_MOVES * 2),
    ("policies_values", MAX_MOVES * 2),
]

HEADER_AND_POLICY_BYTES = sum(size for _, size in LAYOUT)

SQUARE_BYTES_CHOICES = (137, 141)


def bytes_per_pos(square_bytes):
    """On-disk record size. Mirrors tpg_dataset.py: 9378 + (square_bytes - 137) * 64."""
    return 9378 + (square_bytes - 137) * 64


# Check 1: every byte before the squares is accounted for, at both widths.
for _sq in SQUARE_BYTES_CHOICES:
    _expected = bytes_per_pos(_sq) - 64 * _sq
    assert HEADER_AND_POLICY_BYTES == _expected, (
        f"LAYOUT sums to {HEADER_AND_POLICY_BYTES} but square_bytes={_sq} leaves "
        f"{_expected} bytes before the squares -- the layout table is wrong"
    )


def _read_prefix(path, want_bytes):
    """Decompress at most want_bytes from the front of a .zst shard."""
    dctx = zstandard.ZstdDecompressor()
    buf = bytearray()
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        while len(buf) < want_bytes:
            chunk = reader.read(1 << 22)
            if not chunk:
                break
            buf.extend(chunk)
    return bytes(buf)


def _decode(raw, square_bytes):
    """Decode a flat record buffer into named field arrays."""
    bpp = bytes_per_pos(square_bytes)
    n = len(raw) // bpp
    if n == 0:
        return 0, {}
    data = np.frombuffer(raw[: n * bpp], dtype=np.uint8).reshape(n, bpp)

    out = {}
    offset = 0
    for name, size in LAYOUT:
        if name is not None:
            out[name] = np.ascontiguousarray(data[:, offset : offset + size])
        offset += size
    assert offset == HEADER_AND_POLICY_BYTES  # redundant with the import check, cheap

    def as_f32(name, count):
        return out[name].view(np.float32).reshape(-1, count)

    fields = {
        "wdl_nondeblundered": as_f32("wdl_nondeblundered", 3),
        "wdl_deblundered": as_f32("wdl_deblundered", 3),
        "wdl_q": as_f32("wdl_q", 3),
        "played_q_suboptimality": as_f32("played_q_suboptimality", 1).ravel(),
        "kld_policy": np.abs(as_f32("kld_policy", 1).ravel()),
        "delta_q_versus_v": np.abs(as_f32("delta_q_versus_v", 1).ravel()),
        "policies_indices": out["policies_indices"].view(np.int16).reshape(-1, MAX_MOVES),
        "policies_values": out["policies_values"].view(np.float16).reshape(-1, MAX_MOVES).astype(np.float32),
    }
    # mlh is stored preprocessed; tpg_dataset.py undoes it as square(x / 0.1) / 100,
    # which is in trainer units. Plies = that * 100.
    mlh_raw = as_f32("mlh_raw", 1).ravel()
    fields["mlh_plies"] = np.square(mlh_raw / 0.1)
    return n, fields


def _wdl_validity(fields):
    """Fraction of rows whose wdl_q is a valid probability vector.

    This is the runtime stride check (check 3). With the wrong square width the
    float view walks off the field boundary and this collapses toward zero.
    """
    wdl = fields["wdl_q"]
    finite = np.isfinite(wdl).all(axis=1)
    in_range = ((wdl >= -1e-4) & (wdl <= 1 + 1e-4)).all(axis=1)
    sums_to_one = np.abs(wdl.sum(axis=1) - 1.0) < 1e-3
    return float((finite & in_range & sums_to_one).mean())


def detect_square_bytes(raw, min_score=0.98, margin=0.20):
    """Infer 137 vs 141 from the data by testing which stride decodes valid WDL rows.

    Returns (square_bytes, scores). Raises if neither width is convincing or the
    two are too close to call -- better to stop than to report a silently
    misaligned decode, which is exactly how tpg_eval.py went wrong.
    """
    scores = {}
    # Probing the wrong stride reads arbitrary bytes as float32, so overflow and
    # invalid-value warnings are the expected outcome of the losing candidate, not
    # a problem to report. Real decodes outside this probe stay un-silenced.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for sq in SQUARE_BYTES_CHOICES:
            n, fields = _decode(raw, sq)
            scores[sq] = _wdl_validity(fields) if n else 0.0
    best = max(scores, key=scores.get)
    other = min(scores, key=scores.get)
    if scores[best] < min_score:
        raise ValueError(
            f"could not identify square width: scores {scores} (best below {min_score}). "
            "Pass --square-bytes explicitly if this shard is a new format."
        )
    if scores[best] - scores[other] < margin:
        raise ValueError(
            f"square width ambiguous: scores {scores} (margin < {margin}). "
            "Pass --square-bytes explicitly."
        )
    return best, scores


def decode_checked(raw, square_bytes, label="<buffer>", min_validity=0.98):
    """Decode and refuse to return anything from a misaligned view.

    Returns (n, fields, validity). Raises ValueError if the wdl_q rows are not
    probability vectors, which is what a wrong stride produces. Reporting plausible
    numbers from a misaligned decode is the tpg_eval.py failure mode; this is the
    single place that prevents it, so all readers go through here.
    """
    n, fields = _decode(raw, square_bytes)
    if n == 0:
        raise ValueError(f"{label}: no complete records at square_bytes={square_bytes}")
    validity = _wdl_validity(fields)
    if validity < min_validity:
        raise ValueError(
            f"{label}: only {validity:.1%} of wdl_q rows are valid probability vectors "
            f"at square_bytes={square_bytes} -- decode is misaligned, refusing to report"
        )
    return n, fields, validity


def summarize(path, want_mb=120, square_bytes=None):
    """Decode the front of one shard and return summary statistics."""
    raw = _read_prefix(path, int(want_mb * 1024 * 1024))
    if square_bytes is None:
        square_bytes, _ = detect_square_bytes(raw)
    n, f, validity = decode_checked(raw, square_bytes, label=path)

    # Policy targets. The C# writer pads unused slots by REPLICATING the last
    # (index, value) pair, so the padding suffix must be masked before any sum or
    # count -- tpg_dataset.py carries the same mask (bug found 2026-08-28).
    idx = f["policies_indices"]
    valid = np.ones_like(idx, dtype=bool)
    valid[:, 1:] = idx[:, 1:] != idx[:, :-1]
    probs = np.where(valid, f["policies_values"], 0.0)
    totals = probs.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    probs = probs / totals

    n_moves = valid.sum(axis=1)
    top1 = probs.max(axis=1)
    entropy = -(np.where(probs > 0, probs * np.log(np.maximum(probs, 1e-12)), 0.0)).sum(axis=1)
    q = f["wdl_q"][:, 0] - f["wdl_q"][:, 2]
    deblundered = np.abs(f["wdl_deblundered"] - f["wdl_nondeblundered"]).max(axis=1) > 1e-6

    def pcts(x):
        return {
            "mean": float(np.mean(x)),
            "p10": float(np.percentile(x, 10)),
            "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)),
        }

    return {
        "shard": os.path.basename(path),
        "path": path,
        "square_bytes": square_bytes,
        "bytes_per_pos": bytes_per_pos(square_bytes),
        "records": int(n),
        "wdl_validity": validity,
        "legal_moves": pcts(n_moves),
        "top1_target_mass": pcts(top1),
        "policy_entropy": pcts(entropy),
        "frac_near_onehot_top1_gt_0.9": float((top1 > 0.9).mean()),
        "frac_flat_top1_lt_0.3": float((top1 < 0.3).mean()),
        "q_mean": float(q.mean()),
        "frac_decisive_absq_gt_0.8": float((np.abs(q) > 0.8).mean()),
        "frac_balanced_absq_lt_0.1": float((np.abs(q) < 0.1).mean()),
        "draw_share": float(f["wdl_q"][:, 1].mean()),
        "kld_policy": pcts(f["kld_policy"]),
        "delta_q_versus_v": pcts(f["delta_q_versus_v"]),
        "played_q_suboptimality": pcts(f["played_q_suboptimality"]),
        "frac_played_subopt_gt_0.05": float((f["played_q_suboptimality"] > 0.05).mean()),
        "mlh_plies": pcts(f["mlh_plies"]),
        "frac_deblundered": float(deblundered.mean()),
    }


def print_summary(s):
    def row(label, d):
        print(f"  {label:<22} mean {d['mean']:8.3f}   p10 {d['p10']:7.3f}  p50 {d['p50']:7.3f}  p90 {d['p90']:7.3f}")

    print(f"\n=== {s['shard']}  ({s['square_bytes']} B/sq, {s['bytes_per_pos']} B/rec, "
          f"n={s['records']}, wdl valid {s['wdl_validity']:.1%}) ===")
    row("legal moves/pos", s["legal_moves"])
    row("top-1 target mass", s["top1_target_mass"])
    row("policy entropy", s["policy_entropy"])
    print(f"  {'near-one-hot >0.9':<22} {s['frac_near_onehot_top1_gt_0.9']:7.1%}"
          f"     flat <0.3 {s['frac_flat_top1_lt_0.3']:7.1%}")
    print(f"  {'Q (w-l)':<22} mean {s['q_mean']:+8.3f}   |Q|>0.8 {s['frac_decisive_absq_gt_0.8']:6.1%}"
          f"   |Q|<0.1 {s['frac_balanced_absq_lt_0.1']:6.1%}")
    print(f"  {'draw share of wdl_q':<22} {s['draw_share']:8.3f}")
    row("KLDPolicy", s["kld_policy"])
    row("DeltaQVersusV", s["delta_q_versus_v"])
    row("played-Q subopt", s["played_q_suboptimality"])
    print(f"  {'  subopt > 0.05':<22} {s['frac_played_subopt_gt_0.05']:7.1%}")
    row("MLH (plies)", s["mlh_plies"])
    print(f"  {'deblundered != raw':<22} {s['frac_deblundered']:7.1%} of rows")


def expand(paths):
    """Expand directories to their .zst shards; keep explicit files as given."""
    out = []
    for p in paths:
        p = p.rstrip("/\\")
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.zst"))))
        else:
            out.append(p)
    # .tgt.zst sidecars (survival targets) are not TPG records.
    return [p for p in out if not p.endswith(".tgt.zst")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="shard .zst files and/or corpus directories")
    ap.add_argument("--mb", type=float, default=120,
                    help="MB to decompress from the front of each shard (default 120, ~13K records)")
    ap.add_argument("--square-bytes", type=int, choices=SQUARE_BYTES_CHOICES, default=None,
                    help="override per-shard width detection (137 = V2, 141 = V3)")
    ap.add_argument("--per-shard", action="store_true",
                    help="summarize every shard found, not just the first of each directory")
    ap.add_argument("--json", metavar="FILE", help="also write the summaries as JSON")
    args = ap.parse_args(argv)

    shards = expand(args.paths)
    if not args.per_shard:
        seen, keep = set(), []
        for p in shards:
            d = os.path.dirname(p)
            if d not in seen:
                seen.add(d)
                keep.append(p)
        shards = keep
    if not shards:
        print("no .zst shards found", file=sys.stderr)
        return 1

    results = []
    for path in shards:
        try:
            s = summarize(path, want_mb=args.mb, square_bytes=args.square_bytes)
        except (ValueError, OSError) as exc:
            print(f"\n=== {os.path.basename(path)} ===\n  SKIPPED: {exc}", file=sys.stderr)
            continue
        results.append(s)
        print_summary(s)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
