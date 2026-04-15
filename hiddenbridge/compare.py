from __future__ import annotations

import argparse
import json
from pathlib import Path


def _dataset_label(dataset_name: str) -> str:
    known = {
        "glove-200-angular": "GloVe-200 (angular)",
        "sift-128-euclidean": "SIFT-128 (euclidean)",
        "dbpedia-openai3-large-1536-n1000000-test10000": "DBpedia OpenAI-3 1536",
        "dbpedia-openai3-large-3072-n101000-test1000": "DBpedia OpenAI-3 3072",
    }
    return known.get(dataset_name, dataset_name)


def _load_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _method_styles() -> dict[str, dict[str, str]]:
    return {
        "baseline": {
            "label": "Baseline graph",
            "color": "#176BEF",
            "marker": "^",
            "linestyle": "-",
        },
        "pairwise_interpolation": {
            "label": "Pairwise interpolation",
            "color": "#F59E0B",
            "marker": "o",
            "linestyle": "--",
        },
        "cluster_centroid": {
            "label": "Cluster centroids",
            "color": "#107C10",
            "marker": "s",
            "linestyle": "-",
        },
        "local_knn_mean": {
            "label": "Local k-NN means",
            "color": "#0EA5E9",
            "marker": "H",
            "linestyle": ":",
        },
        "random_line": {
            "label": "Random line",
            "color": "#8B5CF6",
            "marker": "P",
            "linestyle": "-.",
        },
        "random_line_anchor": {
            "label": "Random line + anchor",
            "color": "#6D28D9",
            "marker": "X",
            "linestyle": "--",
        },
        "noisy_copy": {
            "label": "Noisy copies",
            "color": "#D7263D",
            "marker": "D",
            "linestyle": ":",
        },
        "bridge": {
            "label": "Bridge points",
            "color": "#009FB7",
            "marker": "v",
            "linestyle": "-.",
        },
        "hierarchical_centroid": {
            "label": "Hierarchical centroids",
            "color": "#0F766E",
            "marker": "h",
            "linestyle": "--",
        },
        "directional_centroid": {
            "label": "Directional centroids",
            "color": "#7C3AED",
            "marker": "<",
            "linestyle": ":",
        },
        "boundary_shell": {
            "label": "Boundary shell",
            "color": "#E11D48",
            "marker": ">",
            "linestyle": "-.",
        },
        "failure_driven": {
            "label": "Failure-driven",
            "color": "#374151",
            "marker": "*",
            "linestyle": "-",
        },
        "targeted_noisy_replicas": {
            "label": "Targeted noisy replicas",
            "color": "#B45309",
            "marker": "p",
            "linestyle": "--",
        },
        "noisy_anchors": {
            "label": "Noisy-anchor Steiner",
            "color": "#D7263D",
            "marker": "D",
            "linestyle": ":",
        },
    }


def _payload_actual_hidden_count(payload: dict[str, object], method_name: str) -> int:
    if method_name == "baseline":
        return 0
    steiner_methods = payload.get("steiner", {}).get("methods", {})
    if not isinstance(steiner_methods, dict):
        return 0
    method_metadata = steiner_methods.get(method_name, {})
    if not isinstance(method_metadata, dict):
        return 0
    return int(method_metadata.get("actual_hidden_count", 0))


def _shared_method_hidden_counts(
    payloads: list[dict[str, object]],
) -> dict[str, int | None]:
    method_names = sorted(
        {
            str(method_name)
            for payload in payloads
            for method_name in payload.get("results", {}).keys()
        }
    )
    counts: dict[str, int | None] = {}
    for method_name in method_names:
        values = {
            _payload_actual_hidden_count(payload, method_name) for payload in payloads
        }
        counts[method_name] = next(iter(values)) if len(values) == 1 else None
    return counts


def _legend_label(
    method_name: str,
    base_label: str,
    shared_hidden_counts: dict[str, int | None],
) -> str:
    actual_hidden_count = shared_hidden_counts.get(method_name)
    if method_name == "baseline" or not actual_hidden_count:
        return base_label
    return f"{base_label} (S={actual_hidden_count})"


def _plot_x_bounds(payloads: list[dict[str, object]]) -> tuple[float, float]:
    import math

    xs = []
    for payload in payloads:
        for series in payload["results"].values():
            for item in series.values():
                xs.append(float(item["avg_distance_computations"]))
    if not xs:
        return (1.0, 10.0)
    min_x = min(xs)
    max_x = max(xs)
    if min_x <= 0.0:
        min_x = min(x for x in xs if x > 0.0)
    log_min = math.log10(min_x)
    log_max = math.log10(max_x)
    pad = 0.045 * max(1.0, log_max - log_min)
    return (10 ** (log_min - pad), 10 ** (log_max + pad))


def _plot_x_ticks(payloads: list[dict[str, object]]) -> list[float]:
    import numpy as np

    xs = []
    for payload in payloads:
        for series in payload["results"].values():
            for item in series.values():
                xs.append(float(item["avg_distance_computations"]))
    if not xs:
        return [1.0, 10.0]
    unique_xs = np.asarray(sorted(set(xs)), dtype=np.float64)
    if unique_xs.size <= 8:
        return [float(item) for item in unique_xs]
    target_count = min(8, max(5, int(round(np.sqrt(unique_xs.size)))))
    selected_indices = np.linspace(0, unique_xs.size - 1, num=target_count, dtype=int)
    tick_values = []
    seen = set()
    for idx in selected_indices.tolist():
        value = float(unique_xs[int(idx)])
        rounded = int(round(value))
        if rounded in seen:
            continue
        seen.add(rounded)
        tick_values.append(value)
    if float(unique_xs[0]) not in tick_values:
        tick_values.insert(0, float(unique_xs[0]))
    if float(unique_xs[-1]) not in tick_values:
        tick_values.append(float(unique_xs[-1]))
    return sorted(set(tick_values))


def _format_x_tick(value: float) -> str:
    rounded = round(float(value), 1)
    if rounded >= 100:
        return f"{int(round(rounded)):,}"
    if rounded >= 10:
        return f"{rounded:.0f}"
    return f"{rounded:.1f}"


def _style_tradeoff_axis(axis, x_limits, x_ticks) -> None:
    from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

    axis.set_xscale("log")
    axis.set_xlim(*x_limits)
    axis.xaxis.set_major_locator(FixedLocator(x_ticks))
    axis.set_xticklabels([_format_x_tick(item) for item in x_ticks])
    axis.set_ylim(0.0, 1.02)
    axis.set_axisbelow(True)
    axis.margins(x=0.03, y=0.02)
    axis.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9))
    )
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.grid(which="major", axis="y", color="#CBD5E1", linewidth=1.2, alpha=0.98)
    axis.grid(which="major", axis="x", color="#CBD5E1", linewidth=1.0, alpha=0.84)
    axis.grid(which="minor", axis="x", color="#E5E7EB", linestyle=":", linewidth=0.95)
    axis.tick_params(axis="both", labelsize=11.5, width=1.0, length=5, pad=6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#94A3B8")
    axis.spines["bottom"].set_color("#94A3B8")
    axis.spines["left"].set_linewidth(1.1)
    axis.spines["bottom"].set_linewidth(1.1)


def save_benchmark_comparison_plot(
    payloads: list[dict[str, object]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if not payloads:
        raise ValueError("Expected at least one payload")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.labelsize": 13.5,
            "axes.titlesize": 16.0,
            "legend.fontsize": 10.2,
            "xtick.labelsize": 11.2,
            "ytick.labelsize": 11.2,
            "font.size": 11.8,
        }
    )
    styles = _method_styles()
    shared_hidden_counts = _shared_method_hidden_counts(payloads)
    x_limits = _plot_x_bounds(payloads)
    x_ticks = _plot_x_ticks(payloads)
    row_count = len(payloads)
    figure, axes = plt.subplots(
        row_count,
        2,
        figsize=(18.8, max(6.2 * row_count, 7.2)),
        dpi=240,
        squeeze=False,
        sharex=True,
        sharey="col",
    )
    figure.patch.set_facecolor("white")
    panels = [
        ("recall_at_1", "Recall@1"),
        ("recall_at_k", None),
    ]

    for row_idx, payload in enumerate(payloads):
        dataset_name = str(payload["dataset"]["name"])
        top_k = int(payload["search"]["top_k"])
        for col_idx, (metric_key, default_ylabel) in enumerate(panels):
            axis = axes[row_idx][col_idx]
            ylabel = default_ylabel or f"Recall@{top_k}"
            for method_name, series in payload["results"].items():
                style = styles.get(
                    method_name,
                    {
                        "label": method_name.replace("_", " ").title(),
                        "color": "#6B7280",
                        "marker": "o",
                    },
                )
                points = sorted(
                    series.values(),
                    key=lambda item: item["avg_distance_computations"],
                )
                xs = [float(item["avg_distance_computations"]) for item in points]
                ys = [float(item[metric_key]) for item in points]
                axis.plot(
                    xs,
                    ys,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style.get("linestyle", "-"),
                    linewidth=2.95,
                    markersize=8.8,
                    markeredgecolor="white",
                    markeredgewidth=1.0,
                    alpha=0.98,
                    label=_legend_label(
                        method_name=method_name,
                        base_label=style["label"],
                        shared_hidden_counts=shared_hidden_counts,
                    ),
                )
            axis.set_ylim(0.0, 1.02)
            axis.set_xlabel("Average distance computations per query")
            axis.set_ylabel(ylabel)
            axis.set_title(
                f"{_dataset_label(dataset_name)}",
                fontsize=15.6,
                fontweight="bold",
                pad=11,
            )
            _style_tradeoff_axis(axis, x_limits=x_limits, x_ticks=x_ticks)

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, max(1, len(labels))),
        frameon=True,
        bbox_to_anchor=(0.5, 0.985),
        borderpad=0.7,
        handlelength=2.6,
        edgecolor="#CBD5E1",
        facecolor="white",
    )
    figure.suptitle(
        "DiskANN-Style Steiner Point Comparison Across Benchmarks",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.008,
        (
            "All Steiner nodes are routing-only and are never returned in final search results. "
            "Curves sweep beam size on a fixed-degree Vamana/DiskANN-style graph."
        ),
        ha="center",
        va="bottom",
        fontsize=11,
        color="#374151",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.018, 0.055, 0.982, 0.925))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Metric JSON files produced by hiddenbridge.experiment",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination PNG path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [_load_payload(Path(path)) for path in args.inputs]
    save_benchmark_comparison_plot(payloads=payloads, output_path=Path(args.output))


if __name__ == "__main__":
    main()
