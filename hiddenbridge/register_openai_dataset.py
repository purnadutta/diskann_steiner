from __future__ import annotations

import argparse
import json

from hiddenbridge.datasets import register_local_dataset


def _default_dataset_subdir(dimension: int) -> str:
    return f"openai-{dimension}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dimension", type=int, choices=[1536, 3072], required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--dataset-subdir")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = register_local_dataset(
        data_root=args.data_root,
        dataset_subdir=args.dataset_subdir or _default_dataset_subdir(args.dimension),
        train_file=args.train_file,
        test_file=args.test_file,
        metric="cosine",
        normalize=True,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
