from __future__ import annotations

import json
import tempfile
from pathlib import Path


def dataset_paths(data_root: str | Path, dataset_subdir: str) -> dict[str, Path]:
    root = Path(data_root).expanduser().resolve()
    dataset_dir = root / dataset_subdir
    indices_dir = dataset_dir / "indices"
    analysis_dir = dataset_dir / "analysis"
    graph_navigation_analysis_dir = analysis_dir / "graph_navigation"
    return {
        "root": root,
        "dataset_dir": dataset_dir,
        "indices_dir": indices_dir,
        "analysis_dir": analysis_dir,
        "graph_navigation_analysis_dir": graph_navigation_analysis_dir,
        "train_file": dataset_dir / "train.npy",
        "test_file": dataset_dir / "test.npy",
        "ground_truth_neighbors_file": dataset_dir / "ground_truth_neighbors.npy",
        "ground_truth_distances_file": dataset_dir / "ground_truth_distances.npy",
        "sample_indices_file": dataset_dir / "sample_indices.npy",
        "dataset_metadata_file": dataset_dir / "dataset_metadata.json",
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def normalize_rows_inplace(matrix) -> None:
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_rows = norms.squeeze(1) <= 0.0
    if np.any(zero_rows):
        matrix[zero_rows] = 0.0
        matrix[zero_rows, 0] = 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= norms


def download_ann_benchmark_dataset(
    *,
    data_root: str | Path,
    dataset_subdir: str,
    dataset_url: str,
    embedding_dim: int,
    metric: str,
    normalize: bool,
    n: int = 0,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, object]:
    import h5py
    import numpy as np
    import requests

    paths = dataset_paths(data_root, dataset_subdir)
    train_file = paths["train_file"]
    test_file = paths["test_file"]
    metadata_file = paths["dataset_metadata_file"]
    sample_indices_file = paths["sample_indices_file"]
    ground_truth_neighbors_file = paths["ground_truth_neighbors_file"]
    ground_truth_distances_file = paths["ground_truth_distances_file"]
    paths["dataset_dir"].mkdir(parents=True, exist_ok=True)

    if train_file.exists() and test_file.exists() and not overwrite:
        return {
            "status": "skipped",
            "dataset_subdir": dataset_subdir,
            "train_file": str(train_file),
            "test_file": str(test_file),
            "metadata_file": str(metadata_file),
        }

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir) / f"{dataset_subdir}.hdf5"
        with requests.get(dataset_url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with local_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        with h5py.File(local_path, "r") as handle:
            train_dataset = handle["train"]
            test_dataset = handle["test"]
            if len(train_dataset.shape) != 2 or int(train_dataset.shape[1]) != int(embedding_dim):
                raise ValueError(
                    f"Expected train embeddings with dim {embedding_dim}, got {train_dataset.shape}"
                )
            if len(test_dataset.shape) != 2 or int(test_dataset.shape[1]) != int(embedding_dim):
                raise ValueError(
                    f"Expected test embeddings with dim {embedding_dim}, got {test_dataset.shape}"
                )

            full_train_count = int(train_dataset.shape[0])
            test_count = int(test_dataset.shape[0])
            sample_indices = None
            if 0 < int(n) < full_train_count:
                rng = np.random.default_rng(seed)
                sample_indices = np.sort(
                    rng.choice(full_train_count, size=int(n), replace=False)
                )
                train_embeddings = np.asarray(train_dataset[sample_indices], dtype=np.float32)
            else:
                train_embeddings = np.asarray(train_dataset, dtype=np.float32)

            test_embeddings = np.asarray(test_dataset, dtype=np.float32)
            neighbors = (
                np.asarray(handle["neighbors"], dtype=np.int32)
                if "neighbors" in handle and sample_indices is None
                else None
            )
            distances = (
                np.asarray(handle["distances"], dtype=np.float32)
                if "distances" in handle and sample_indices is None
                else None
            )

    if normalize:
        normalize_rows_inplace(train_embeddings)
        normalize_rows_inplace(test_embeddings)

    np.save(train_file, train_embeddings)
    np.save(test_file, test_embeddings)
    if sample_indices is not None:
        np.save(sample_indices_file, sample_indices)
    elif sample_indices_file.exists():
        sample_indices_file.unlink()

    if neighbors is not None:
        np.save(ground_truth_neighbors_file, neighbors)
    elif ground_truth_neighbors_file.exists():
        ground_truth_neighbors_file.unlink()

    if distances is not None:
        np.save(ground_truth_distances_file, distances)
    elif ground_truth_distances_file.exists():
        ground_truth_distances_file.unlink()

    metadata = {
        "dataset_name": dataset_subdir,
        "dataset_url": dataset_url,
        "dimension": int(embedding_dim),
        "metric": metric,
        "original_train_count": full_train_count,
        "saved_train_count": int(train_embeddings.shape[0]),
        "test_count": test_count,
        "sampled_train": sample_indices is not None,
        "vectors_normalized": bool(normalize),
        "seed": int(seed),
        "train_file": str(train_file),
        "test_file": str(test_file),
        "sample_indices_file": str(sample_indices_file) if sample_indices is not None else None,
        "ground_truth_neighbors_file": str(ground_truth_neighbors_file)
        if neighbors is not None
        else None,
        "ground_truth_distances_file": str(ground_truth_distances_file)
        if distances is not None
        else None,
    }
    _write_json(metadata_file, metadata)
    return metadata


def register_local_dataset(
    *,
    data_root: str | Path,
    dataset_subdir: str,
    train_file: str | Path,
    test_file: str | Path,
    metric: str,
    normalize: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    import numpy as np

    source_train = Path(train_file).expanduser().resolve()
    source_test = Path(test_file).expanduser().resolve()
    if not source_train.exists() or not source_test.exists():
        raise FileNotFoundError("train_file and test_file must exist")

    paths = dataset_paths(data_root, dataset_subdir)
    dest_train = paths["train_file"]
    dest_test = paths["test_file"]
    metadata_file = paths["dataset_metadata_file"]
    paths["dataset_dir"].mkdir(parents=True, exist_ok=True)

    if dest_train.exists() and dest_test.exists() and not overwrite:
        return {
            "status": "skipped",
            "dataset_subdir": dataset_subdir,
            "train_file": str(dest_train),
            "test_file": str(dest_test),
            "metadata_file": str(metadata_file),
        }

    train = np.array(np.load(source_train, mmap_mode="r"), dtype=np.float32, copy=True)
    test = np.array(np.load(source_test, mmap_mode="r"), dtype=np.float32, copy=True)
    if normalize:
        normalize_rows_inplace(train)
        normalize_rows_inplace(test)

    np.save(dest_train, train)
    np.save(dest_test, test)
    metadata = {
        "dataset_name": dataset_subdir,
        "metric": metric,
        "dimension": int(train.shape[1]),
        "saved_train_count": int(train.shape[0]),
        "test_count": int(test.shape[0]),
        "vectors_normalized": bool(normalize),
        "source_train_file": str(source_train),
        "source_test_file": str(source_test),
        "train_file": str(dest_train),
        "test_file": str(dest_test),
        "ground_truth_neighbors_file": None,
        "ground_truth_distances_file": None,
    }
    _write_json(metadata_file, metadata)
    return metadata
