"""
Generate a 2x3 grid of recall-vs-distance-computations plots for a
Steiner percentage sweep (e.g. 1%, 5%, 15%, 25%, 50%).

Rows: R=32, R=64
Columns: Recall@1, Recall@10, Recall@100

Each subplot shows baseline (grey) + one colored line per percentage.

Usage:
    python scripts/plot_steiner_sweep.py \
        --jsons r32_1pct.json r32_5pct.json r32_15pct.json r32_25pct.json r32_50pct.json \
        --jsons64 r64_1pct.json r64_5pct.json r64_15pct.json r64_25pct.json r64_50pct.json \
        --labels "1%" "5%" "15%" "25%" "50%" \
        --output steiner_sweep.png

    Or for a single R value (1x3 grid):
    python scripts/plot_steiner_sweep.py \
        --jsons r32_1pct.json r32_5pct.json r32_25pct.json r32_50pct.json \
        --labels "1%" "5%" "25%" "50%" \
        --output steiner_sweep_r32.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Color palette: up to 8 distinct colors
SWEEP_COLORS = [
    "#e41a1c",  # red
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#17becf",  # cyan
    "#d62728",  # dark red
]

SWEEP_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]

RECALL_COLS = [
    ("recall_at_1",  "Recall@1"),
    ("recall_at_10", "Recall@10"),
    ("recall_at_k",  "Recall@100"),
]


def _extract(payload, metric_key, method_prefix=None):
    """Return [(dist_comps, recall), ...] sorted by dist_comps.

    If method_prefix is given, merge all methods starting with that prefix.
    Otherwise return all non-baseline points merged.
    """
    pts = []
    for method_name, variants in payload["results"].items():
        if method_name == "baseline":
            continue
        if method_prefix and not method_name.startswith(method_prefix):
            continue
        for v in variants.values():
            pts.append((v["avg_distance_computations"], v.get(metric_key, 0.0)))
    pts.sort()
    return pts


def _extract_baseline(payload, metric_key):
    """Return [(dist_comps, recall), ...] for baseline."""
    pts = []
    if "baseline" in payload["results"]:
        for v in payload["results"]["baseline"].values():
            pts.append((v["avg_distance_computations"], v.get(metric_key, 0.0)))
    pts.sort()
    return pts


def main():
    parser = argparse.ArgumentParser(description="Steiner percentage sweep plot")
    parser.add_argument("--jsons", nargs="+", required=True,
                        help="JSON result files for R=32 (or single R), one per percentage")
    parser.add_argument("--jsons64", nargs="*", default=None,
                        help="JSON result files for R=64, one per percentage (optional; adds second row)")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for each percentage (e.g. '1%%' '5%%' '25%%' '50%%')")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Method name prefixes to include (default: all non-baseline)")
    parser.add_argument("--output", default="steiner_sweep.png")
    args = parser.parse_args()

    assert len(args.jsons) == len(args.labels), \
        f"Number of --jsons ({len(args.jsons)}) must match --labels ({len(args.labels)})"
    if args.jsons64:
        assert len(args.jsons64) == len(args.labels), \
            f"Number of --jsons64 ({len(args.jsons64)}) must match --labels ({len(args.labels)})"

    # Load JSONs
    data_r32 = []
    for path in args.jsons:
        with open(path) as f:
            data_r32.append(json.load(f))

    data_r64 = []
    if args.jsons64:
        for path in args.jsons64:
            with open(path) as f:
                data_r64.append(json.load(f))

    two_rows = bool(data_r64)
    nrows = 2 if two_rows else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 5 * nrows), squeeze=False)

    method_prefix = args.methods[0] if args.methods else None

    for row, (data_list, R_label) in enumerate(
        [(data_r32, "R=32")] + ([(data_r64, "R=64")] if two_rows else [])
    ):
        for col, (metric_key, metric_label) in enumerate(RECALL_COLS):
            ax = axes[row][col]

            # Baseline (from first json — should be same across all)
            baseline_pts = _extract_baseline(data_list[0], metric_key)
            if baseline_pts:
                ax.plot(
                    [p[0] for p in baseline_pts], [p[1] for p in baseline_pts],
                    color="#555555", linewidth=2.5, marker="o", markersize=5,
                    label="Baseline", zorder=10,
                )

            # One line per percentage
            for idx, (data, label) in enumerate(zip(data_list, args.labels)):
                pts = _extract(data, metric_key, method_prefix)
                if not pts:
                    continue
                color = SWEEP_COLORS[idx % len(SWEEP_COLORS)]
                marker = SWEEP_MARKERS[idx % len(SWEEP_MARKERS)]
                ax.plot(
                    [p[0] for p in pts], [p[1] for p in pts],
                    color=color,
                    linestyle="-",
                    marker=marker,
                    markersize=4,
                    linewidth=1.6,
                    label=label,
                )

            ax.set_xlabel("Avg Distance Computations", fontsize=10)
            if col == 0:
                ax.set_ylabel(R_label, fontsize=12, fontweight="bold")
            ax.set_title(f"{metric_label}  ({R_label})", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            ax.set_xlim(left=0)

            if row == 0 and col == 2:
                ax.legend(fontsize=8, loc="lower right", ncol=1)

    # Overall title
    ds = data_r32[0].get("dataset", {})
    fig.suptitle(
        f"Steiner Percentage Sweep — {ds.get('name', 'dataset')}, "
        f"train={ds.get('train_size')}, queries={ds.get('query_count')}, "
        f"top_k={data_r32[0].get('search', {}).get('top_k', '?')}, "
        f"α={data_r32[0].get('graph', {}).get('alpha', '?')}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
