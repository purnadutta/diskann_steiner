from __future__ import annotations

import argparse
import json

from hiddenbridge.datasets import (
    download_ann_benchmark_dataset,
    download_big_ann_dataset,
    download_huggingface_dataset,
)


KNOWN_DATASETS = {
    "glove-200": {
        "dataset_subdir": "glove-200-angular",
        "dataset_url": "http://ann-benchmarks.com/glove-200-angular.hdf5",
        "embedding_dim": 200,
        "metric": "angular",
        "normalize": True,
    },
    "glove-100": {
        "dataset_subdir": "glove-100-angular",
        "dataset_url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "embedding_dim": 100,
        "metric": "angular",
        "normalize": True,
    },
    "glove-25": {
        "dataset_subdir": "glove-25-angular",
        "dataset_url": "http://ann-benchmarks.com/glove-25-angular.hdf5",
        "embedding_dim": 25,
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
    "fashion-mnist": {
        "dataset_subdir": "fashion-mnist-784-euclidean",
        "dataset_url": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        "embedding_dim": 784,
        "metric": "euclidean",
        "normalize": False,
    },
    "gist": {
        "dataset_subdir": "gist-960-euclidean",
        "dataset_url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "embedding_dim": 960,
        "metric": "euclidean",
        "normalize": False,
    },
    "nytimes": {
        "dataset_subdir": "nytimes-256-angular",
        "dataset_url": "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
        "embedding_dim": 256,
        "metric": "angular",
        "normalize": True,
    },
    "lastfm": {
        "dataset_subdir": "lastfm-64-dot",
        "dataset_url": "http://ann-benchmarks.com/lastfm-64-dot.hdf5",
        "embedding_dim": 64,
        "metric": "angular",
        "normalize": True,
    },
    "mnist": {
        "dataset_subdir": "mnist-784-euclidean",
        "dataset_url": "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
        "embedding_dim": 784,
        "metric": "euclidean",
        "normalize": False,
    },
}

# big-ann-benchmarks (NeurIPS'23) — binary format, larger downloads
BIG_ANN_DATASETS = {
    "yfcc": {
        "dataset_subdir": "yfcc-10M-u8-192-l2",
        "base_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/base.10M.u8bin",
        "query_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/query.public.100K.u8bin",
        "embedding_dim": 192,
        "metric": "euclidean",
        "dtype": "uint8",
        "normalize": False,
    },
    "text2image": {
        "dataset_subdir": "text2image-10M-f32-200-ip",
        "base_url": "https://storage.yandexcloud.net/yr-secret-share/ann-datasets/T2I/base.10M.fbin",
        "query_url": "https://storage.yandexcloud.net/yr-secret-share/ann-datasets/T2I/query.public.100K.fbin",
        "embedding_dim": 200,
        "metric": "angular",
        "dtype": "float32",
        "normalize": True,
    },
}

# Hugging Face datasets (parquet format)
HF_DATASETS = {
    "openai-ada2": {
        "dataset_subdir": "dbpedia-openai-ada2-1536",
        "hf_repo": "KShivendu/dbpedia-entities-openai-1M",
        "embedding_column": "openai",
        "embedding_dim": 1536,
        "metric": "angular",
        "normalize": True,
        "format": "huggingface",
    },
    "openai-3-large": {
        "dataset_subdir": "dbpedia-openai3-large-3072",
        "hf_repo": "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M",
        "embedding_column": "openai",
        "embedding_dim": 3072,
        "metric": "angular",
        "normalize": True,
        "format": "huggingface",
    },
}


ALL_DATASETS = {**KNOWN_DATASETS, **BIG_ANN_DATASETS, **HF_DATASETS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(ALL_DATASETS), required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset in HF_DATASETS:
        spec = HF_DATASETS[args.dataset]
        print(f"Downloading HuggingFace dataset: {args.dataset} ({spec['hf_repo']})")
        result = download_huggingface_dataset(
            data_root=args.data_root,
            dataset_subdir=spec["dataset_subdir"],
            hf_repo=spec["hf_repo"],
            embedding_column=spec["embedding_column"],
            embedding_dim=spec["embedding_dim"],
            metric=spec["metric"],
            normalize=spec["normalize"],
            n=args.n,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    elif args.dataset in BIG_ANN_DATASETS:
        spec = BIG_ANN_DATASETS[args.dataset]
        print(f"Downloading big-ann-benchmark dataset: {args.dataset}")
        print(f"  WARNING: These are large files (1-8 GB). Download may take a while.")
        result = download_big_ann_dataset(
            data_root=args.data_root,
            dataset_subdir=spec["dataset_subdir"],
            base_url=spec["base_url"],
            query_url=spec["query_url"],
            embedding_dim=spec["embedding_dim"],
            metric=spec["metric"],
            dtype=spec["dtype"],
            normalize=spec["normalize"],
            n=args.n,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    else:
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
