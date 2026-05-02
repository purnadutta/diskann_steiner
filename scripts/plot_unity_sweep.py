"""
Generate a 2x3 grid of recall-vs-distance-computations plots for Unity results.

Rows: R=32, R=64
Columns: Recall@1, Recall@10, Recall@100

Features:
- Colorblind-friendly Wong 2011 palette
- Y-axis starts above 0 (auto-scaled to data range with padding)
- Full metadata in title: dataset, size, queries, Steiner %, metric, alpha, top_k

Usage:
    python scripts/plot_unity_sweep.py \
        --results-dir results_unity \
        --dataset fashion-mnist-784-euclidean \
        --train-size 60000 \
        --hidden-counts 3000 9000 15000 \
        --output plots_unity/fashionmnist_sweep.png
"""
from __future__ import annotations

import argparse
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Colorblind-friendly palette (Wong 2011 / IBM Design)
SWEEP_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
]

SWEEP_MARKERS = ["s", "D", "^", "v", "P", "X"]
SWEEP_LINESTYLES = ["--", "-.", ":", (0, (3, 1, 1, 1)), "-", (0, (5, 1))]  # dashed, dash-dot, dotted, dash-dot-dot, solid, long-dash

RECALL_COLS = [
    ("recall_at_1",  "Recall@1"),
    ("recall_at_10", "Recall@10"),
    ("recall_at_k",  "Recall@100"),
]


def load_results(results_dir, dataset, train_size, hidden_count, R):
    """Find and load the JSON for a given config."""
    pattern = f"*{dataset}*t{train_size}*h{hidden_count}_R{R}.json"
    matches = glob.glob(str(Path(results_dir) / pattern))
    if not matches:
        # Try without underscore before R
        pattern2 = f"*{dataset}*t{train_size}*h{hidden_count}*R{R}.json"
        matches = glob.glob(str(Path(results_dir) / pattern2))
    if not matches:
        print(f"  WARNING: No file found for {dataset} t={train_size} h={hidden_count} R={R}")
        return None
    with open(matches[0]) as f:
        return json.load(f)


def extract_points(payload, metric_key, method="cluster_centroid"):
    """Return [(visible_dist_comps, recall), ...] sorted by dist_comps."""
    pts = []
    if method not in payload["results"]:
        return pts
    for v in payload["results"][method].values():
        # Use visible distance computations (excludes Steiner node computations)
        dist = v.get("avg_visible_distance_computations", v["avg_distance_computations"])
        pts.append((dist, v.get(metric_key, 0.0)))
    pts.sort()
    return pts


def extract_baseline(payload, metric_key):
    """Return [(dist_comps, recall), ...] for baseline."""
    pts = []
    if "baseline" in payload["results"]:
        for v in payload["results"]["baseline"].values():
            pts.append((v["avg_distance_computations"], v.get(metric_key, 0.0)))
    pts.sort()
    return pts


def main():
    parser = argparse.ArgumentParser(description="Unity results sweep plot")
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing JSON result files")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name as it appears in filenames")
    parser.add_argument("--train-size", type=int, required=True,
                        help="Training set size")
    parser.add_argument("--hidden-counts", nargs="+", type=int, required=True,
                        help="Hidden (Steiner) counts for each percentage")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Labels for each percentage (auto-computed if not given)")
    parser.add_argument("--output", default="plots_unity/sweep.png")
    parser.add_argument("--title", default=None)
    parser.add_argument("--use-visible-dist", action="store_true", default=True,
                        help="Use visible distance computations (default: True)")
    parser.add_argument("--y-min", type=float, default=0.8,
                        help="Y-axis minimum (default: 0.8)")
    parser.add_argument("--y-max", type=float, default=1.0,
                        help="Y-axis maximum (default: 1.0)")
    args = parser.parse_args()

    # Auto-compute labels as percentages
    if args.labels:
        labels = args.labels
    else:
        labels = [f"{h*100//args.train_size}% ({h:,})" for h in args.hidden_counts]

    R_values = [32, 64]

    # Load all data
    all_data = {}  # (hidden_count, R) -> payload
    for h in args.hidden_counts:
        for R in R_values:
            payload = load_results(args.results_dir, args.dataset, args.train_size, h, R)
            if payload:
                all_data[(h, R)] = payload

    if not all_data:
        print("ERROR: No data loaded!")
        return

    # Get metadata from first available payload
    sample = next(iter(all_data.values()))
    ds_info = sample.get("dataset", {})
    graph_info = sample.get("graph", {})
    search_info = sample.get("search", {})

    metric_raw = ds_info.get("metric", "unknown")
    # Clean up verbose metric strings
    if "cosine" in metric_raw.lower() or "inner product" in metric_raw.lower():
        metric_display = "cosine (L2-normalized)"
    elif "euclidean" in metric_raw.lower():
        metric_display = "euclidean"
    else:
        metric_display = metric_raw
    query_count = ds_info.get("query_count", "?")
    alpha = graph_info.get("alpha", "?")
    top_k = search_info.get("top_k", sample.get("search", {}).get("top_k", 100))
    # Try to get top_k from results
    for method_data in sample["results"].values():
        for v in method_data.values():
            if "top_k" in v:
                top_k = v["top_k"]
                break
        break

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)

    for row, R in enumerate(R_values):
        for col, (metric_key, metric_label) in enumerate(RECALL_COLS):
            ax = axes[row][col]

            # Collect all y-values for auto-scaling
            all_y = []

            # Baseline (from first available hidden count for this R)
            baseline_pts = None
            for h in args.hidden_counts:
                if (h, R) in all_data:
                    baseline_pts = extract_baseline(all_data[(h, R)], metric_key)
                    break

            if baseline_pts:
                all_y.extend([p[1] for p in baseline_pts])
                ax.plot(
                    [p[0] for p in baseline_pts], [p[1] for p in baseline_pts],
                    color="#333333", linewidth=2.0, linestyle="--", dashes=(5, 3),
                    marker="o", markersize=5, markerfacecolor="none", markeredgewidth=1.5,
                    label="Baseline (no Steiner)", zorder=10,
                )

            # One line per Steiner percentage
            for idx, (h, label) in enumerate(zip(args.hidden_counts, labels)):
                if (h, R) not in all_data:
                    continue
                pts = extract_points(all_data[(h, R)], metric_key)
                if not pts:
                    continue
                all_y.extend([p[1] for p in pts])
                color = SWEEP_COLORS[idx % len(SWEEP_COLORS)]
                marker = SWEEP_MARKERS[idx % len(SWEEP_MARKERS)]
                ls = SWEEP_LINESTYLES[idx % len(SWEEP_LINESTYLES)]
                ax.plot(
                    [p[0] for p in pts], [p[1] for p in pts],
                    color=color,
                    linestyle=ls,
                    marker=marker,
                    markersize=7,
                    linewidth=2.5,
                    label=f"Steiner {label}",
                )

            ax.set_ylim(args.y_min, args.y_max)

            ax.set_xlabel("Avg Distance Computations (visible only)", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"R = {R}", fontsize=12, fontweight="bold")
            ax.set_title(f"{metric_label}  (R={R})", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)

            if row == 0 and col == 2:
                ax.legend(fontsize=9, loc="lower right", ncol=1, framealpha=0.9)

    # Build detailed title
    if args.title:
        plot_title = args.title
    else:
        dataset_clean = args.dataset.replace("-", " ").replace("_", " ").title()
        pcts = ", ".join(labels)
        plot_title = (
            f"{dataset_clean}\n"
            f"n={args.train_size:,} | queries={query_count} | top_k={top_k} | "
            f"metric={metric_display} | alpha={alpha} | Steiner: {pcts}"
        )

    fig.suptitle(plot_title, fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
