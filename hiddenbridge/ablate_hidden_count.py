from __future__ import annotations

import argparse
import json
from pathlib import Path

from hiddenbridge.compare import _dataset_label
from hiddenbridge.experiment import (
    _format_x_tick,
    _plot_method_styles,
    _plot_x_bounds,
    _plot_x_ticks,
    _style_tradeoff_axis,
)


def _load_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _curve_points(
    payload: dict[str, object],
    method_name: str,
) -> tuple[list[float], list[float], list[float]]:
    series = payload["results"][method_name]
    points = sorted(series.values(), key=lambda item: item["avg_distance_computations"])
    xs = [float(item["avg_distance_computations"]) for item in points]
    r1 = [float(item["recall_at_1"]) for item in points]
    rk = [float(item["recall_at_k"]) for item in points]
    return xs, r1, rk


def _actual_hidden_count(payload: dict[str, object], method_name: str) -> int:
    if method_name == "baseline":
        return 0
    method_meta = payload.get("steiner", {}).get("methods", {}).get(method_name, {})
    return int(method_meta.get("actual_hidden_count", 0))


def _legend_label(method_name: str, payload: dict[str, object]) -> str:
    base_label = _plot_method_styles().get(
        method_name,
        {"label": method_name.replace("_", " ").title()},
    )["label"]
    hidden_count = _actual_hidden_count(payload, method_name)
    if method_name == "baseline":
        return base_label
    return f"{base_label} (S={hidden_count})"


def save_hidden_count_ablation_plot(
    *,
    payloads: list[dict[str, object]],
    methods: list[str],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if not payloads:
        raise ValueError("Expected at least one payload")

    dataset_name = str(payloads[0]["dataset"]["name"])
    top_k = int(payloads[0]["search"]["top_k"])
    x_limits = _plot_x_bounds(payloads)
    x_ticks = _plot_x_ticks(payloads)
    style_map = _plot_method_styles()
    hidden_counts = sorted(
        {
            _actual_hidden_count(payload, method_name)
            for payload in payloads
            for method_name in methods
            if method_name in payload["results"] and method_name != "baseline"
        }
    )
    linestyle_cycle = {
        count: style
        for count, style in zip(hidden_counts, ["--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))])
    }

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
    figure, axes = plt.subplots(1, 2, figsize=(19.0, 8.8), dpi=240, sharex=True)
    figure.patch.set_facecolor("white")

    metric_panels = [
        ("recall_at_1", "Recall@1"),
        ("recall_at_k", f"Recall@{top_k}"),
    ]

    baseline_payload = payloads[0]
    baseline_style = style_map["baseline"]
    baseline_xs, baseline_r1, baseline_rk = _curve_points(baseline_payload, "baseline")

    for axis, (metric_key, metric_label) in zip(axes, metric_panels):
        baseline_ys = baseline_r1 if metric_key == "recall_at_1" else baseline_rk
        axis.plot(
            baseline_xs,
            baseline_ys,
            color=baseline_style["color"],
            marker=baseline_style["marker"],
            linestyle="-",
            linewidth=3.2,
            markersize=9.0,
            markeredgecolor="white",
            markeredgewidth=1.0,
            alpha=0.98,
            label="Baseline graph",
            zorder=5,
        )

        for method_name in methods:
            if method_name == "baseline":
                continue
            base_style = style_map.get(
                method_name,
                {"label": method_name.replace("_", " ").title(), "color": "#6B7280", "marker": "o"},
            )
            for payload in payloads:
                if method_name not in payload["results"]:
                    continue
                hidden_count = _actual_hidden_count(payload, method_name)
                xs, r1, rk = _curve_points(payload, method_name)
                ys = r1 if metric_key == "recall_at_1" else rk
                axis.plot(
                    xs,
                    ys,
                    color=base_style["color"],
                    marker=base_style["marker"],
                    linestyle=linestyle_cycle.get(hidden_count, "--"),
                    linewidth=2.8,
                    markersize=8.2,
                    markeredgecolor="white",
                    markeredgewidth=0.95,
                    alpha=0.97,
                    label=_legend_label(method_name, payload),
                )

        axis.set_xlabel("Average distance computations per query")
        axis.set_ylabel(metric_label)
        axis.set_title(metric_label, fontsize=15.4, fontweight="bold", pad=12)
        _style_tradeoff_axis(axis, x_limits=x_limits, x_ticks=x_ticks)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    dedup_handles = []
    dedup_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        dedup_handles.append(handle)
        dedup_labels.append(label)

    figure.legend(
        dedup_handles,
        dedup_labels,
        loc="upper center",
        frameon=True,
        ncol=min(5, max(1, len(dedup_labels))),
        fontsize=10.2,
        bbox_to_anchor=(0.5, 0.972),
        borderpad=0.7,
        handlelength=2.6,
        edgecolor="#CBD5E1",
        facecolor="white",
    )
    figure.suptitle(
        f"Steiner Budget Ablation on {_dataset_label(dataset_name)}",
        fontsize=19,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        (
            "Same 100k-database GloVe DiskANN-style setup. "
            "Legend shows the actual routing-only Steiner count S for each method."
        ),
        ha="center",
        va="bottom",
        fontsize=11,
        color="#374151",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.018, 0.075, 0.982, 0.905))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--methods",
        default="cluster_centroid,failure_driven",
        help="Comma-separated methods to include alongside baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [_load_payload(Path(path)) for path in args.inputs]
    methods = ["baseline"] + [item.strip() for item in args.methods.split(",") if item.strip()]
    save_hidden_count_ablation_plot(
        payloads=payloads,
        methods=methods,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
