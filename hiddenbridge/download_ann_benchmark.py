from __future__ import annotations

import argparse
import json

from hiddenbridge.datasets import download_ann_benchmark_dataset


KNOWN_DATASETS = {
    "glove": {
        "dataset_subdir": "glove-200-angular",
        "dataset_url": "http://ann-benchmarks.com/glove-200-angular.hdf5",
        "embedding_dim": 200,
        "metric": "angular",
        "normalize": True,
    },
    "sift": {
        "dataset_subdir": "sift-128-euclidean",
        "dataset_url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "embedding_dim": 128,
        "metric": "euclidean",
        "normalize": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(KNOWN_DATASETS), required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = KNOWN_DATASETS[args.dataset]
    result = download_ann_benchmark_dataset(
        data_root=args.data_root,
        dataset_subdir=spec["dataset_subdir"],
        dataset_url=spec["dataset_url"],
        embedding_dim=spec["embedding_dim"],
        metric=spec["metric"],
        normalize=spec["normalize"],
        n=args.n,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
