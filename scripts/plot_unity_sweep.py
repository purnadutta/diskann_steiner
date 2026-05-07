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


def extract_points(payload, metric_key, method="cluster_centroid", use_total=True):
    """Return [(dist_comps, recall), ...] sorted by dist_comps.

    If use_total=True, uses total distance computations (data + Steiner).
    If use_total=False, uses only visible (data) distance computations.
    """
    pts = []
    if method not in payload["results"]:
        return pts
    for v in payload["results"][method].values():
        if use_total:
            dist = v["avg_distance_computations"]
        else:
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
    parser.add_argument("--visible-only", action="store_true", default=False,
                        help="Use only visible (data) distance computations instead of total")
    parser.add_argument("--y-min", type=float, default=0.8,
                        help="Y-axis minimum (default: 0.8)")
    parser.add_argument("--y-max", type=float, default=1.0,
                        help="Y-axis maximum (default: 1.0)")
    parser.add_argument("--x-max", type=float, default=None,
                        help="X-axis maximum (auto if not set)")
    parser.add_argument("--x-maxes", nargs=6, type=float, default=None,
                        help="Per-panel x-max: R@1-R32 R@1-R64 R@10-R32 R@10-R64 R@100-R32 R@100-R64")
    parser.add_argument("--transpose", action="store_true", default=False,
                        help="Use 3x2 layout (rows=recall, cols=R) instead of 2x3")
    parser.add_argument("--paper", action="store_true", default=False,
                        help="Paper-friendly: larger fonts, no metadata title, cleaner look")
    parser.add_argument("--R-values", nargs="+", type=int, default=None,
                        help="Which R values to plot (default: 32 64)")
    args = parser.parse_args()

    # Auto-compute labels as percentages
    if args.labels:
        labels = args.labels
    else:
        labels = [f"{h*100//args.train_size}% ({h:,})" for h in args.hidden_counts]

    R_values = args.R_values if args.R_values else [32, 64]

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

    # Font sizes
    if args.paper:
        fs_title = 14
        fs_axlabel = 12
        fs_tick = 11
        fs_legend = 11
        fs_suptitle = 15
        lw_base = 2.5
        lw_steiner = 3.0
        ms = 8
    else:
        fs_title = 11
        fs_axlabel = 10
        fs_tick = 9
        fs_legend = 9
        fs_suptitle = 12
        lw_base = 2.0
        lw_steiner = 2.5
        ms = 7

    # Create figure layout based on number of R values and transpose flag
    n_R = len(R_values)
    n_metrics = len(RECALL_COLS)
    if args.transpose:
        fig, axes = plt.subplots(n_metrics, n_R, figsize=(6 * n_R, 5 * n_metrics), squeeze=False)
    else:
        fig, axes = plt.subplots(n_R, n_metrics, figsize=(6 * n_metrics, 5 * n_R), squeeze=False)

    for r_idx, R in enumerate(R_values):
        for m_idx, (metric_key, metric_label) in enumerate(RECALL_COLS):
            if args.transpose:
                ax = axes[m_idx][r_idx]
            else:
                ax = axes[r_idx][m_idx]

            # Baseline (from first available hidden count for this R)
            baseline_pts = None
            for h in args.hidden_counts:
                if (h, R) in all_data:
                    baseline_pts = extract_baseline(all_data[(h, R)], metric_key)
                    break

            if baseline_pts:
                ax.plot(
                    [p[0] for p in baseline_pts], [p[1] for p in baseline_pts],
                    color="#333333", linewidth=lw_base, linestyle="--", dashes=(5, 3),
                    marker="o", markersize=ms-2, markerfacecolor="none", markeredgewidth=1.5,
                    label="Baseline (no Steiner)", zorder=10,
                )

            # One line per Steiner percentage
            for idx, (h, label) in enumerate(zip(args.hidden_counts, labels)):
                if (h, R) not in all_data:
                    continue
                pts = extract_points(all_data[(h, R)], metric_key, use_total=not args.visible_only)
                if not pts:
                    continue
                color = SWEEP_COLORS[idx % len(SWEEP_COLORS)]
                marker = SWEEP_MARKERS[idx % len(SWEEP_MARKERS)]
                ls = SWEEP_LINESTYLES[idx % len(SWEEP_LINESTYLES)]
                ax.plot(
                    [p[0] for p in pts], [p[1] for p in pts],
                    color=color,
                    linestyle=ls,
                    marker=marker,
                    markersize=ms,
                    linewidth=lw_steiner,
                    label=f"Steiner {label}",
                )

            ax.set_ylim(args.y_min, args.y_max)
            # Per-panel x-max: order is R@1-R32, R@1-R64, R@10-R32, R@10-R64, R@100-R32, R@100-R64
            panel_xmax = args.x_max
            if args.x_maxes:
                panel_idx = m_idx * 2 + r_idx
                panel_xmax = args.x_maxes[panel_idx]
            ax.set_xlim(left=0, right=panel_xmax)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=fs_tick)

            dist_label = "Avg Distance Computations (visible only)" if args.visible_only else "Avg Distance Computations (total)"

            if args.transpose:
                ax.set_title(f"{metric_label}  (R={R})", fontsize=fs_title, fontweight="bold")
                if m_idx == 2:  # bottom row
                    ax.set_xlabel(dist_label, fontsize=fs_axlabel)
                if r_idx == 0:  # left column
                    ax.set_ylabel(metric_label, fontsize=fs_axlabel)
                # Legend in top-right subplot
                if m_idx == 0 and r_idx == 1:
                    ax.legend(fontsize=fs_legend, loc="lower right", ncol=1, framealpha=0.9)
            else:
                ax.set_title(f"{metric_label}  (R={R})", fontsize=fs_title)
                ax.set_xlabel(dist_label, fontsize=fs_axlabel)
                if m_idx == 0:
                    ax.set_ylabel(f"R = {R}", fontsize=fs_axlabel+2, fontweight="bold")
                if r_idx == 0 and m_idx == 2:
                    ax.legend(fontsize=fs_legend, loc="lower right", ncol=1, framealpha=0.9)

    # Build title
    if args.paper:
        plot_title = args.title if args.title else ""
    elif args.title:
        plot_title = args.title
    else:
        dataset_clean = args.dataset.replace("-", " ").replace("_", " ").title()
        pcts = ", ".join(labels)
        plot_title = (
            f"{dataset_clean}\n"
            f"n={args.train_size:,} | queries={query_count} | top_k={top_k} | "
            f"metric={metric_display} | alpha={alpha} | Steiner: {pcts}"
        )

    if plot_title:
        fig.suptitle(plot_title, fontsize=fs_suptitle, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
