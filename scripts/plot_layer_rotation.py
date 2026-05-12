"""Plot per-layer rotation profile from layer_rotation_*.csv.

Reads the CSV produced by diagnose_layer_rotation.py and emits a PNG showing
alignment_in_subspace vs transformer-layer index, one curve per sub-layer type
(attention.qkv, W_h, sm1, sm2, sm3, mlp.linear1, linear2, ...).

Lower alignment = more rotation = where LoRA gets traction. Curves near 1.0
are pass-through layers that LoRA cannot meaningfully act on.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAYER_RE = re.compile(r"^transformer_layer\.(\d+)\.(.+)$")


def load(csv_path: Path):
    by_kind: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    non_tx: list[tuple[str, float, float]] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            align = float(row["alignment_in_subspace"])
            eff_y = float(row["eff_rank_y"])
            m = LAYER_RE.match(name)
            if m:
                idx = int(m.group(1))
                kind = m.group(2)  # e.g. attention.W_h, mlp.linear2
                by_kind[kind].append((idx, align, eff_y))
            else:
                non_tx.append((name, align, eff_y))
    for k in by_kind:
        by_kind[k].sort()
    return by_kind, non_tx


def plot(by_kind, non_tx, out_path: Path, title_suffix: str):
    kinds_sorted = sorted(by_kind.keys())
    fig, (ax_a, ax_e) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    cmap = plt.get_cmap("tab10")
    for i, kind in enumerate(kinds_sorted):
        rows = by_kind[kind]
        xs = [r[0] for r in rows]
        ys_a = [r[1] for r in rows]
        ys_e = [r[2] for r in rows]
        ax_a.plot(xs, ys_a, marker="o", ms=4, lw=1.2, color=cmap(i % 10), label=kind)
        ax_e.plot(xs, ys_e, marker="o", ms=4, lw=1.2, color=cmap(i % 10), label=kind)

    ax_a.set_ylabel("alignment_in_subspace\n(lower = more rotated = LoRA traction)")
    ax_a.set_title(f"Per-layer rotation profile {title_suffix}")
    ax_a.axhline(1.0, color="grey", lw=0.5, ls=":")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(fontsize=8, ncol=2, loc="lower left")

    ax_e.set_ylabel("eff_rank_y (output effective rank)")
    ax_e.set_xlabel("transformer layer index")
    ax_e.set_yscale("log")
    ax_e.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")

    if non_tx:
        print("\nnon-transformer layers (alignment, eff_rank_y):")
        for name, a, e in sorted(non_tx, key=lambda r: r[1]):
            print(f"  {a:.4f}  eff_rank_y={e:8.2f}  {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, nargs="?",
                    default=Path("/mnt/c/Dev/Chess/CeresTrain/layer_rotation_c1_640_34.csv"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    csv_path = args.csv
    out = args.out or csv_path.with_suffix(".png")
    by_kind, non_tx = load(csv_path)
    plot(by_kind, non_tx, out, title_suffix=f"({csv_path.name})")


if __name__ == "__main__":
    main()
