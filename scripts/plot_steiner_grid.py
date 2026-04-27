"""
Generate a 2x3 grid of recall-vs-distance-computations plots.

Rows: R=32, R=64
Columns: Recall@1, Recall@10, Recall@100

Each subplot contains 7 lines: baseline + 2 methods x 3 Steiner percentages.

Usage:
    python scripts/plot_steiner_grid.py \
        --r32-5   <json>  --r32-15  <json>  --r32-25  <json> \
        --r64-5   <json>  --r64-15  <json>  --r64-25  <json> \
        --output steiner_grid.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLES = {
    "cluster_centroid": {"label": "Centroid"},
    "random_midpoint":  {"label": "Random Mid"},
}

PCT_COLORS     = {5: "#1f77b4", 15: "#ff7f0e", 25: "#2ca02c"}  # blue / orange / green
PCT_LINESTYLES = {5: "-", 15: "-", 25: "-"}
PCT_MARKERS    = {5: "o", 15: "s", 25: "D"}

RECALL_COLS = [
    ("recall_at_1",  "Recall@1"),
    ("recall_at_10", "Recall@10"),
    ("recall_at_k",  "Recall@100"),
]


def _extract(payload, metric_key):
    """Return {method: [(dist_comps, recall), ...]} sorted by dist_comps."""
    curves = {}
    for method_name, variants in payload["results"].items():
        pts = []
        for v in variants.values():
            pts.append((v["avg_distance_computations"], v.get(metric_key, 0.0)))
        pts.sort()
        curves[method_name] = pts
    return curves


def main():
    parser = argparse.ArgumentParser(description="2x3 Steiner grid plot")
    parser.add_argument("--r32-5",  required=True)
    parser.add_argument("--r32-15", required=True)
    parser.add_argument("--r32-25", required=True)
    parser.add_argument("--r64-5",  required=True)
    parser.add_argument("--r64-15", required=True)
    parser.add_argument("--r64-25", required=True)
    parser.add_argument("--output", default="steiner_grid.png")
    args = parser.parse_args()

    grid = {
        32: {5:  args.r32_5,  15: args.r32_15, 25: args.r32_25},
        64: {5:  args.r64_5,  15: args.r64_15, 25: args.r64_25},
    }

    # Load all JSONs
    data = {}
    for R in [32, 64]:
        data[R] = {}
        for pct in [5, 15, 25]:
            with open(grid[R][pct]) as f:
                data[R][pct] = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey="col")

    for row, R in enumerate([32, 64]):
        for col, (metric_key, metric_label) in enumerate(RECALL_COLS):
            ax = axes[row][col]

            # Plot baseline once (from 5% run — baseline is same across pcts)
            baseline_curves = _extract(data[R][5], metric_key)
            if "baseline" in baseline_curves:
                pts = baseline_curves["baseline"]
                ax.plot(
                    [p[0] for p in pts], [p[1] for p in pts],
                    color="#555555", linewidth=2.5, marker="o", markersize=5,
                    label="Baseline", zorder=10,
                )

            # Plot each method x percentage
            for method, style in METHOD_STYLES.items():
                for pct in [5, 15, 25]:
                    curves = _extract(data[R][pct], metric_key)
                    if method not in curves:
                        continue
                    pts = curves[method]
                    ax.plot(
                        [p[0] for p in pts], [p[1] for p in pts],
                        color=PCT_COLORS[pct],
                        linestyle=PCT_LINESTYLES[pct],
                        marker=PCT_MARKERS[pct],
                        markersize=4,
                        linewidth=1.6,
                        label=f"{style['label']} {pct}%",
                    )

            ax.set_xlabel("Avg Distance Computations", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"R={R}", fontsize=12, fontweight="bold")
            ax.set_title(f"{metric_label}  (R={R})", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            ax.set_xlim(left=0)

            if row == 0 and col == 2:
                ax.legend(fontsize=7, loc="lower right", ncol=1)

    # Overall title
    ds = data[32][5].get("dataset", {})
    fig.suptitle(
        f"Steiner Point Recall Tradeoff \u2014 {ds.get('name', 'dataset')}, "
        f"train={ds.get('train_size')}, queries={ds.get('query_count')}, "
        f"top_k={data[32][5].get('search', {}).get('top_k', '?')}, "
        f"\u03b1={data[32][5].get('graph', {}).get('alpha', '?')}",
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
