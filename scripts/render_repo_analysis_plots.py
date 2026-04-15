from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hiddenbridge.ablate_hidden_count import save_hidden_count_ablation_plot
from hiddenbridge.compare import save_benchmark_comparison_plot
from hiddenbridge.experiment import _save_tradeoff_plot


def _load_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _repo_root() -> Path:
    return REPO_ROOT


def _source_root() -> Path:
    return _repo_root().parent / "steiner_diskann" / "artifacts"


def _output_root() -> Path:
    return _repo_root() / "docs" / "analysis_figures"


def main() -> None:
    source_root = _source_root()
    output_root = _output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    tradeoff_sources = {
        "glove_100k_tradeoff.png": source_root
        / "glove_100k_200"
        / "diskann_steiner_tradeoff_glove_200_angular_t100000_q500_h512.json",
        "glove_full_tradeoff.png": source_root
        / "glove_full_200"
        / "diskann_steiner_tradeoff_glove_200_angular_t1183514_q1000_h1024.json",
        "openai_1536_100k_tradeoff.png": source_root
        / "openai_100k_1536"
        / "diskann_steiner_tradeoff_dbpedia_openai3_large_1536_n1000000_test10000_t100000_q1000_h512.json",
        "openai_3072_100k_tradeoff.png": source_root
        / "openai_100k_3072"
        / "diskann_steiner_tradeoff_dbpedia_openai3_large_3072_n101000_test1000_t100000_q1000_h512.json",
        "sift_full_tradeoff.png": source_root
        / "sift"
        / "diskann_steiner_tradeoff_sift_128_euclidean_t1000000_q10000_h1024.json",
    }

    comparison_payloads = [
        _load_payload(tradeoff_sources["glove_100k_tradeoff.png"]),
        _load_payload(tradeoff_sources["openai_1536_100k_tradeoff.png"]),
        _load_payload(tradeoff_sources["openai_3072_100k_tradeoff.png"]),
        _load_payload(tradeoff_sources["sift_full_tradeoff.png"]),
    ]

    ablation_paths = [
        source_root
        / "ablation"
        / "glove_100k_200"
        / "diskann_steiner_tradeoff_glove_200_angular_t100000_q500_h512.json",
        source_root
        / "ablation"
        / "glove_100k_200"
        / "diskann_steiner_tradeoff_glove_200_angular_t100000_q500_h4096.json",
        source_root
        / "ablation"
        / "glove_100k_200"
        / "diskann_steiner_tradeoff_glove_200_angular_t100000_q500_h32768.json",
    ]
    ablation_payloads = [_load_payload(path) for path in ablation_paths]

    for output_name, payload_path in tradeoff_sources.items():
        payload = _load_payload(payload_path)
        _save_tradeoff_plot(payload=payload, plot_path=output_root / output_name)

    save_benchmark_comparison_plot(
        payloads=comparison_payloads,
        output_path=output_root / "cross_dataset_comparison.png",
    )
    save_hidden_count_ablation_plot(
        payloads=ablation_payloads,
        methods=["cluster_centroid", "bridge"],
        output_path=output_root / "glove_100k_hidden_count_ablation.png",
    )


if __name__ == "__main__":
    main()
