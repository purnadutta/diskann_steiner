"""
Compare Steiner insertion modes (end / start / random) on a single plot.

Usage:
    python scripts/compare_insertion_modes.py \
        --end  data/.../t50000_q10000_h1024.json \
        --start data/.../t50000_q10000_h1024_insstart.json \
        --random data/.../t50000_q10000_h1024_insrandom.json \
        --output insertion_comparison.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODE_LABELS = {"end": "Steiner at end", "start": "Steiner at start", "random": "Steiner random"}
# Per-mode line style: solid, dashed, dotted
MODE_LINESTYLES = {"end": "-", "start": "--", "random": ":"}

# Colors per method (shared across modes)
METHOD_COLORS = {
    "baseline": "#555555",
    "cluster_centroid": "#1f77b4",
    "bridge": "#ff7f0e",
    "failure_driven": "#2ca02c",
    "pairwise_interpolation": "#d62728",
    "local_knn_mean": "#9467bd",
    "noisy_copy": "#8c564b",
    "hierarchical_centroid": "#e377c2",
    "directional_centroid": "#7f7f7f",
    "boundary_shell": "#bcbd22",
    "random_line": "#17becf",
    "random_line_anchor": "#aec7e8",
    "targeted_noisy_replicas": "#ffbb78",
}


def _extract_curves(payload: dict) -> dict[str, list[tuple[float, float, float]]]:
    """Return {method: [(dist_comps, recall@1, recall@k), ...]} sorted by beam size."""
    curves = {}
    for method_name, variants in payload["results"].items():
        points = []
        for variant in variants.values():
            points.append((
                variant["avg_distance_computations"],
                variant["recall_at_1"],
                variant["recall_at_k"],
            ))
        points.sort(key=lambda p: p[0])
        curves[method_name] = points
    return curves


def main():
    parser = argparse.ArgumentParser(description="Compare insertion modes")
    parser.add_argument("--end", required=True, help="JSON from --steiner-insertion end")
    parser.add_argument("--start", required=True, help="JSON from --steiner-insertion start")
    parser.add_argument("--random", required=True, help="JSON from --steiner-insertion random")
    parser.add_argument("--output", default="insertion_mode_comparison.png")
    parser.add_argument("--metric", choices=["recall_at_1", "recall_at_k"], default="recall_at_k")
    args = parser.parse_args()

    recall_idx = 1 if args.metric == "recall_at_1" else 2
    recall_label = "Recall@1" if args.metric == "recall_at_1" else "Recall@k"

    mode_files = {"end": args.end, "start": args.start, "random": args.random}
    mode_data = {}
    for mode, path in mode_files.items():
        with open(path) as f:
            mode_data[mode] = json.load(f)

    # Collect all methods across modes (excluding baseline — plotted once)
    all_methods = set()
    for payload in mode_data.values():
        all_methods.update(payload["results"].keys())
    all_methods.discard("baseline")
    all_methods = sorted(all_methods)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Plot baseline once (same across all modes since baseline has no Steiner points)
    baseline_curves = _extract_curves(mode_data["end"])
    if "baseline" in baseline_curves:
        pts = baseline_curves["baseline"]
        ax.plot(
            [p[0] for p in pts], [p[recall_idx] for p in pts],
            color=METHOD_COLORS.get("baseline", "gray"),
            linestyle="-", linewidth=2.5, marker="o", markersize=6,
            label="baseline", zorder=10,
        )

    # Plot each method × mode combination
    for method in all_methods:
        color = METHOD_COLORS.get(method, "black")
        for mode, payload in mode_data.items():
            curves = _extract_curves(payload)
            if method not in curves:
                continue
            pts = curves[method]
            ls = MODE_LINESTYLES[mode]
            label = f"{method} ({MODE_LABELS[mode]})"
            ax.plot(
                [p[0] for p in pts], [p[recall_idx] for p in pts],
                color=color, linestyle=ls, linewidth=1.8,
                marker="s" if mode == "end" else ("^" if mode == "start" else "D"),
                markersize=5, label=label,
            )

    ax.set_xlabel("Avg Distance Computations", fontsize=12)
    ax.set_ylabel(recall_label, fontsize=12)
    ds = mode_data["end"].get("dataset", {})
    title = (
        f"Steiner Insertion Order Comparison — {ds.get('name', 'dataset')}\n"
        f"train={ds.get('train_size')}, queries={ds.get('query_count')}, "
        f"hidden={mode_data['end'].get('steiner', {}).get('requested_hidden_count')}, "
        f"ParlayANN Vamana (R={mode_data['end']['graph']['requested_max_degree']}, "
        f"α={mode_data['end']['graph']['alpha']})"
    )
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Print summary table
    print(f"\n{'Method':<25} {'Mode':<10} {'AUC(R@k)':<12} {'Best R@k':<10} {'Best R@1':<10}")
    print("-" * 70)
    for method in ["baseline"] + all_methods:
        for mode, payload in mode_data.items():
            summary = payload.get("summary", {}).get("by_method", {}).get(method)
            if summary is None:
                continue
            if method == "baseline" and mode != "end":
                continue  # baseline is same across modes
            print(
                f"{method:<25} {mode:<10} "
                f"{summary.get('recall_at_k_auc', 0):<12.6f} "
                f"{summary.get('best_variant_recall_at_k', 0):<10.4f} "
                f"{summary.get('best_variant_recall_at_1', 0):<10.4f}"
            )


if __name__ == "__main__":
    main()
