from __future__ import annotations

import argparse
import json

from hiddenbridge.datasets import register_local_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset-subdir", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = register_local_dataset(
        data_root=args.data_root,
        dataset_subdir=args.dataset_subdir,
        train_file=args.train_file,
        test_file=args.test_file,
        metric=args.metric,
        normalize=args.normalize,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
