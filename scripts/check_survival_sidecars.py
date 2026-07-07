"""Structural + statistical validation of K-ply survival sidecars vs their TPG shards.

Checks per shard pair (<shard>.zst, <shard>.tgt.zst):
  H. header: magic TPGT, version 1, channels 1, K
  C. row count == record count
  1. empty-square agreement: label==0 exactly where the record square's current-board
     one-hot says empty (catches ANY record-order desync and any slot-flip error)
  2. kings always survive: label==K+1 on king squares (kings are never captured)
  3. distribution: label histogram, capture-within-K rate by piece class
"""
import glob, os, sys
import numpy as np
import zstandard as zstd

V2 = 9378
SQ = 137
PREFIX = V2 - 64 * SQ  # 610
K = 8

corpus = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/kovax_lc0_4cells_tpg_v2_surv"

def read_exact(reader, n):
    parts = []
    remaining = n
    while remaining > 0:
        piece = reader.read(remaining)
        if not piece:
            break
        parts.append(piece)
        remaining -= len(piece)
    return b"".join(parts)


def stream_pairs(shard_path, tgt_path, block_records=65536):
    """Yield (recs[N,V2], labels[N,64]) blocks streamed from shard+sidecar in lockstep."""
    ds, dt = zstd.ZstdDecompressor(), zstd.ZstdDecompressor()
    with open(shard_path, "rb") as fs, open(tgt_path, "rb") as ft:
        rs, rt = ds.stream_reader(fs), dt.stream_reader(ft)
        hdr = read_exact(rt, 16)
        assert hdr[:4] == b"TPGT" and hdr[4] == 1 and hdr[5] == 1 and hdr[6] == K, f"bad header {hdr[:8].hex()}"
        while True:
            raw = read_exact(rs, block_records * V2)
            if not raw:
                # main stream done; sidecar must be done too
                assert not read_exact(rt, 1), "sidecar longer than shard"
                break
            n = len(raw) // V2
            assert len(raw) % V2 == 0, "shard not record-aligned"
            lab = read_exact(rt, n * 64)
            assert len(lab) == n * 64, "sidecar shorter than shard"
            yield (np.frombuffer(raw, dtype=np.uint8).reshape(n, V2),
                   np.frombuffer(lab, dtype=np.uint8).reshape(n, 64))

tot = {"recs": 0, "empty_mismatch": 0, "king_bad": 0}
hist = np.zeros(K + 2, dtype=np.int64)
cap_by_class = {"our_pawn": [0, 0], "our_piece": [0, 0], "opp_pawn": [0, 0], "opp_piece": [0, 0]}

# Any .zst with a matching sidecar qualifies: game shards (*.tpg_setN.zst) and
# tablebase endgame streams (*.dat.zst) share the sidecar naming convention.
shards = sorted(glob.glob(os.path.join(corpus, "*.zst")))
shards = [s for s in shards if not s.endswith(".tgt.zst")]
# Optional argv[2]: substring filter on shard basename (e.g. "set3.") so big corpora
# can be validated with one process per shard in parallel.
if len(sys.argv) > 2:
    shards = [s for s in shards if sys.argv[2] in os.path.basename(s)]
assert shards, "no shards found"

for shard in shards:
    tgt = shard[:-4] + ".tgt.zst"
    assert os.path.exists(tgt), f"missing sidecar for {shard}"

    n_shard = 0
    for recs, labels in stream_pairs(shard, tgt):
        n = recs.shape[0]
        n_shard += n
        squares = recs[:, PREFIX:].reshape(n, 64, SQ)
        onehot = squares[:, :, :13]                      # current-board piece one-hot
        empty = onehot[:, :, 0] == 100                   # [n, 64]
        king = (onehot[:, :, 6] == 100) | (onehot[:, :, 12] == 100)

        # ALL-ZERO rows = deliberately unsupervised records (no observed continuation);
        # every square is masked from the loss, so exclude them from structural checks.
        supervised = labels.any(axis=1)                  # [n]
        tot.setdefault("masked", 0)
        tot["masked"] += int((~supervised).sum())

        tot["recs"] += n
        tot["empty_mismatch"] += int((((labels == 0) != empty) & supervised[:, None]).sum())
        tot["king_bad"] += int((labels[king & supervised[:, None]] != K + 1).sum())

        occ = labels[~empty]
        hist += np.bincount(occ, minlength=K + 2)

        is_cap = (labels >= 1) & (labels <= K)
        for name, mask in (("our_pawn", onehot[:, :, 1] == 100),
                           ("our_piece", (onehot[:, :, 1:6] == 100).any(axis=2) & (onehot[:, :, 1] != 100)),
                           ("opp_pawn", onehot[:, :, 7] == 100),
                           ("opp_piece", (onehot[:, :, 7:12] == 100).any(axis=2) & (onehot[:, :, 7] != 100))):
            cap_by_class[name][0] += int(is_cap[mask].sum())
            cap_by_class[name][1] += int(mask.sum())
    print(f"{os.path.basename(shard)}: {n_shard} recs OK")

print(f"\nTOTAL records: {tot['recs']}")
print(f"unsupervised (all-zero) rows: {tot.get('masked', 0)}")
print(f"empty-square mismatches: {tot['empty_mismatch']}  (MUST be 0)")
print(f"king label violations:   {tot['king_bad']}  (MUST be 0)")
print(f"label histogram (piece squares): d=1..{K} then survive:")
print("  " + " ".join(f"{d}:{hist[d]}" for d in range(1, K + 2)))
print("capture-within-%d rate by class:" % K)
for name, (c, t) in cap_by_class.items():
    print(f"  {name:9s}: {100*c/max(t,1):.2f}% of {t}")
ok = tot["empty_mismatch"] == 0 and tot["king_bad"] == 0
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
