from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from hiddenbridge.datasets import dataset_paths


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(path)


def _dataset_slug(dataset_subdir: str) -> str:
    cleaned = []
    for ch in str(dataset_subdir):
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    return slug or "dataset"


def _result_paths(
    data_root: str | Path,
    dataset_subdir: str,
    train_size: int,
    query_count: int,
    hidden_count: int,
    max_degree: int = 0,
) -> dict[str, Path]:
    graph_dir = dataset_paths(data_root, dataset_subdir)["graph_navigation_analysis_dir"]
    r_suffix = f"_R{max_degree}" if max_degree > 0 else ""
    stem = (
        "diskann_steiner_tradeoff_"
        f"{_dataset_slug(dataset_subdir)}_t{train_size}_q{query_count}_h{hidden_count}"
        f"{r_suffix}"
    )
    return {
        "metrics_file": graph_dir / f"{stem}.json",
        "plot_file": graph_dir / f"{stem}.png",
        "curve_table_file": graph_dir / f"{stem}_curve_table.csv",
        "method_summary_file": graph_dir / f"{stem}_method_summary.csv",
    }


def _arrange_augmented_vectors(original_vectors, hidden_vectors, mode: str, seed: int):
    """Build augmented vector array with Steiner points at specified positions.

    Args:
        mode: "end" (append hidden after original), "start" (prepend hidden before
              original), or "random" (interleave at random positions).

    Returns:
        augmented_vectors: (N+H, D) array
        orig_to_aug: (N,) int array — orig_to_aug[i] = augmented index of original vector i
        hidden_to_aug: (H,) int array — hidden_to_aug[j] = augmented index of hidden vector j
        is_visible: (N+H,) bool array — True at positions holding original vectors
        aug_to_orig: (N+H,) int array — maps augmented index to original index, -1 for hidden
    """
    import numpy as np

    n = int(original_vectors.shape[0])
    h = int(hidden_vectors.shape[0])
    total = n + h

    if mode == "end":
        augmented = np.vstack([original_vectors, hidden_vectors]).astype(np.float32, copy=False)
        orig_to_aug = np.arange(n, dtype=np.int32)
        hidden_to_aug = np.arange(n, n + h, dtype=np.int32)
    elif mode == "start":
        augmented = np.vstack([hidden_vectors, original_vectors]).astype(np.float32, copy=False)
        orig_to_aug = np.arange(h, h + n, dtype=np.int32)
        hidden_to_aug = np.arange(h, dtype=np.int32)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        positions = rng.permutation(total)
        orig_positions = np.sort(positions[:n])
        hidden_positions = np.sort(positions[n:])
        augmented = np.empty((total, original_vectors.shape[1]), dtype=np.float32)
        augmented[orig_positions] = original_vectors
        augmented[hidden_positions] = hidden_vectors
        orig_to_aug = orig_positions.astype(np.int32)
        hidden_to_aug = hidden_positions.astype(np.int32)
    else:
        raise ValueError(f"Unsupported steiner_insertion mode: {mode}")

    is_visible = np.zeros(total, dtype=np.bool_)
    is_visible[orig_to_aug] = True
    aug_to_orig = np.full(total, -1, dtype=np.int32)
    for i in range(n):
        aug_to_orig[orig_to_aug[i]] = i
    return augmented, orig_to_aug, hidden_to_aug, is_visible, aug_to_orig


def _resolved_metric_name(
    dataset_subdir: str,
    dataset_metadata: dict[str, object] | None,
) -> str:
    metadata = dataset_metadata or {}
    raw_metric = str(metadata.get("metric", "")).strip().lower()
    if "euclidean" in raw_metric or raw_metric == "l2":
        return "euclidean"
    if "angular" in raw_metric or "cosine" in raw_metric or "inner product" in raw_metric:
        return "cosine"
    dataset_slug = str(dataset_subdir).strip().lower()
    if "euclidean" in dataset_slug or "l2" in dataset_slug or "sift" in dataset_slug:
        return "euclidean"
    return "cosine"


def _metric_display_name(metric: str) -> str:
    if metric == "euclidean":
        return "euclidean via squared L2 distance"
    return "cosine via inner product on l2-normalized vectors"


def _prepare_vectors_for_metric(matrix, metric: str, fallback_vector=None, skip_normalize: bool = False):
    import numpy as np

    matrix = np.asarray(matrix, dtype=np.float32)
    if metric == "cosine" and not skip_normalize:
        return _normalize_rows(matrix, fallback_vector=fallback_vector)
    return matrix.astype(np.float32, copy=False)


def _prepare_vector_for_metric(vector, metric: str, fallback_vector=None, skip_normalize: bool = False):
    return _prepare_vectors_for_metric([vector], metric, fallback_vector=fallback_vector, skip_normalize=skip_normalize)[0]


def _score_batch(query_vectors, base_vectors, metric: str):
    import numpy as np

    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    base_vectors = np.asarray(base_vectors, dtype=np.float32)
    if metric == "cosine":
        return np.asarray(query_vectors @ base_vectors.T, dtype=np.float32)

    query_norms = np.sum(query_vectors * query_vectors, axis=1, keepdims=True)
    base_norms = np.sum(base_vectors * base_vectors, axis=1, keepdims=True).T
    sq_dists = query_norms + base_norms - (2.0 * (query_vectors @ base_vectors.T))
    return np.asarray(-np.maximum(sq_dists, 0.0), dtype=np.float32)


def _score_vector(query_vector, base_vectors, metric: str):
    return _score_batch([query_vector], base_vectors, metric)[0]


def _pair_score(left_vector, right_vector, metric: str) -> float:
    import numpy as np

    left_vector = np.asarray(left_vector, dtype=np.float32)
    right_vector = np.asarray(right_vector, dtype=np.float32)
    if metric == "cosine":
        return float(left_vector @ right_vector)
    diff = left_vector - right_vector
    return -float(diff @ diff)


def _score_to_distance(score: float, metric: str) -> float:
    if metric == "cosine":
        return max(0.0, 1.0 - float(score))
    return max(0.0, -float(score))


def _normalize_rows(matrix, fallback_vector=None):
    import numpy as np

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_rows = norms.squeeze(1) <= 0.0
    if np.any(zero_rows):
        repaired = np.array(matrix, dtype=np.float32, copy=True)
        if fallback_vector is None:
            fallback = np.zeros((int(matrix.shape[1]),), dtype=np.float32)
            fallback[0] = 1.0
        else:
            fallback = np.asarray(fallback_vector, dtype=np.float32).reshape(-1)
            fallback_norm = float(np.linalg.norm(fallback))
            if fallback_norm <= 0.0:
                fallback = np.zeros((int(matrix.shape[1]),), dtype=np.float32)
                fallback[0] = 1.0
            else:
                fallback = fallback / fallback_norm
        repaired[zero_rows] = fallback
        matrix = repaired
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / norms).astype(np.float32, copy=False)


def _normalize_vector(vector):
    return _normalize_rows([vector])[0]


def _sample_rows(matrix, count: int, seed: int):
    import numpy as np

    row_count = int(matrix.shape[0])
    if count <= 0 or count >= row_count:
        indices = np.arange(row_count, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(row_count, size=int(count), replace=False)).astype(
            np.int64
        )
    sampled = np.array(matrix[indices], dtype=np.float32, copy=True)
    return sampled, indices


def _batched_exact_search(
    base_vectors,
    query_vectors,
    top_k: int,
    batch_size: int,
    metric: str = "cosine",
):
    import numpy as np

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    if top_k > int(base_vectors.shape[0]):
        raise ValueError("top_k cannot exceed the number of base vectors")

    scores = np.empty((int(query_vectors.shape[0]), top_k), dtype=np.float32)
    ids = np.empty((int(query_vectors.shape[0]), top_k), dtype=np.int64)
    try:
        import faiss

        dim = int(base_vectors.shape[1])
        if metric == "euclidean":
            index = faiss.IndexFlatL2(dim)
        else:
            index = faiss.IndexFlatIP(dim)
        index.add(np.array(base_vectors, dtype=np.float32, copy=True))

        for start in range(0, int(query_vectors.shape[0]), batch_size):
            stop = min(start + batch_size, int(query_vectors.shape[0]))
            batch = np.array(query_vectors[start:stop], dtype=np.float32, copy=True)
            batch_scores, batch_ids = index.search(batch, top_k)
            if metric == "euclidean":
                batch_scores = -batch_scores
            scores[start:stop] = batch_scores
            ids[start:stop] = batch_ids
    except ModuleNotFoundError:
        # Keep the experiment runnable in lightweight local setups even if FAISS is absent.
        base_vectors = np.asarray(base_vectors, dtype=np.float32)
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        for start in range(0, int(query_vectors.shape[0]), batch_size):
            stop = min(start + batch_size, int(query_vectors.shape[0]))
            batch = query_vectors[start:stop]
            batch_scores = _score_batch(batch, base_vectors, metric)
            if top_k >= int(base_vectors.shape[0]):
                batch_ids = np.argsort(-batch_scores, axis=1)[:, :top_k]
            else:
                batch_ids = np.argpartition(-batch_scores, kth=top_k - 1, axis=1)[:, :top_k]
                unsorted_scores = np.take_along_axis(batch_scores, batch_ids, axis=1)
                local_order = np.argsort(-unsorted_scores, axis=1)
                batch_ids = np.take_along_axis(batch_ids, local_order, axis=1)
            scores[start:stop] = np.take_along_axis(batch_scores, batch_ids, axis=1)
            ids[start:stop] = batch_ids.astype(np.int64, copy=False)
    return scores, ids


def _self_excluding_exact_knn(
    vectors,
    neighbor_count: int,
    batch_size: int,
    metric: str = "cosine",
):
    import numpy as np

    if int(vectors.shape[0]) <= 1:
        raise ValueError("Need at least two vectors to build a graph")

    neighbor_count = min(int(neighbor_count), int(vectors.shape[0]) - 1)
    search_scores, search_ids = _batched_exact_search(
        base_vectors=vectors,
        query_vectors=vectors,
        top_k=neighbor_count + 1,
        batch_size=batch_size,
        metric=metric,
    )

    row_count = int(vectors.shape[0])
    filtered_ids = np.full((row_count, neighbor_count), -1, dtype=np.int32)
    filtered_scores = np.full((row_count, neighbor_count), -np.inf, dtype=np.float32)
    for row_idx in range(row_count):
        out_pos = 0
        for candidate_id, candidate_score in zip(search_ids[row_idx], search_scores[row_idx]):
            candidate_id = int(candidate_id)
            if candidate_id < 0 or candidate_id == row_idx:
                continue
            filtered_ids[row_idx, out_pos] = candidate_id
            filtered_scores[row_idx, out_pos] = float(candidate_score)
            out_pos += 1
            if out_pos >= neighbor_count:
                break
        if out_pos < neighbor_count:
            raise ValueError(
                f"Could not find {neighbor_count} self-excluding neighbors for row {row_idx}"
            )
    return filtered_ids, filtered_scores


def _resolved_nlist(train_count: int, requested_nlist: int) -> int:
    if requested_nlist > 0:
        return min(int(requested_nlist), max(1, int(train_count) - 1))
    return max(1, int(math.sqrt(int(train_count))))


def _resolved_nprobe(nlist: int, requested_nprobe: int) -> int:
    if requested_nprobe > 0:
        return min(int(requested_nprobe), int(nlist))
    return max(1, min(int(nlist), int(math.sqrt(int(nlist)))))


def _resolved_training_sample_size(
    train_count: int, nlist: int, requested_sample_size: int
) -> int:
    if requested_sample_size > 0:
        return min(int(train_count), max(int(nlist), int(requested_sample_size)))
    return min(int(train_count), max(int(nlist) * 100, 100_000))


def _load_or_build_ivf_index(
    vectors,
    *,
    requested_nlist: int,
    requested_nprobe: int,
    requested_train_sample_size: int,
    batch_size: int,
    seed: int,
    metric: str,
    existing_index_file: Path | None = None,
):
    import faiss
    import numpy as np

    if existing_index_file is not None and existing_index_file.exists():
        index = faiss.read_index(str(existing_index_file))
        index.nprobe = _resolved_nprobe(int(index.nlist), requested_nprobe)
        return index, {
            "index_source": "existing_file",
            "nlist": int(index.nlist),
            "nprobe": int(index.nprobe),
            "train_sample_size": None,
        }

    train_count = int(vectors.shape[0])
    dimension = int(vectors.shape[1])
    nlist = _resolved_nlist(train_count, requested_nlist)
    train_sample_size = _resolved_training_sample_size(
        train_count, nlist, requested_train_sample_size
    )

    rng = np.random.default_rng(seed)
    if train_sample_size < train_count:
        sample_indices = np.sort(
            rng.choice(train_count, size=train_sample_size, replace=False)
        )
        training_vectors = np.array(vectors[sample_indices], dtype=np.float32, copy=True)
    else:
        training_vectors = np.array(vectors, dtype=np.float32, copy=True)

    if metric == "euclidean":
        quantizer = faiss.IndexFlatL2(dimension)
        faiss_metric = faiss.METRIC_L2
    else:
        quantizer = faiss.IndexFlatIP(dimension)
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        nlist,
        faiss_metric,
    )
    index.train(training_vectors)
    del training_vectors

    for start in range(0, train_count, batch_size):
        stop = min(start + batch_size, train_count)
        batch = np.array(vectors[start:stop], dtype=np.float32, copy=True)
        index.add(batch)

    index.nprobe = _resolved_nprobe(nlist, requested_nprobe)
    return index, {
        "index_source": "built_in_memory",
        "nlist": int(nlist),
        "nprobe": int(index.nprobe),
        "train_sample_size": int(train_sample_size),
    }


def _self_excluding_ivf_knn(
    vectors,
    neighbor_count: int,
    batch_size: int,
    index,
    overfetch: int,
    metric: str,
):
    import numpy as np

    row_count = int(vectors.shape[0])
    neighbor_count = min(int(neighbor_count), int(index.ntotal) - 1)
    initial_search_k = min(int(index.ntotal), max(neighbor_count + 1, neighbor_count + int(overfetch)))
    filtered_ids = np.full((row_count, neighbor_count), -1, dtype=np.int32)
    filtered_scores = np.full((row_count, neighbor_count), -np.inf, dtype=np.float32)

    unresolved_rows = np.arange(row_count, dtype=np.int64)
    search_k = initial_search_k
    while unresolved_rows.size > 0:
        next_unresolved = []
        for start in range(0, unresolved_rows.size, batch_size):
            stop = min(start + batch_size, unresolved_rows.size)
            batch_rows = unresolved_rows[start:stop]
            batch_queries = np.array(vectors[batch_rows], dtype=np.float32, copy=True)
            batch_scores, batch_ids = index.search(batch_queries, search_k)
            if metric == "euclidean":
                batch_scores = -batch_scores
            for local_idx, row_idx in enumerate(batch_rows.tolist()):
                out_pos = 0
                row_ids = batch_ids[local_idx]
                row_scores = batch_scores[local_idx]
                for candidate_id, candidate_score in zip(row_ids, row_scores):
                    candidate_id = int(candidate_id)
                    if candidate_id < 0 or candidate_id == int(row_idx):
                        continue
                    filtered_ids[row_idx, out_pos] = candidate_id
                    filtered_scores[row_idx, out_pos] = float(candidate_score)
                    out_pos += 1
                    if out_pos >= neighbor_count:
                        break
                if out_pos < neighbor_count:
                    next_unresolved.append(int(row_idx))
        if not next_unresolved:
            break
        if search_k >= int(index.ntotal):
            raise ValueError(
                "IVF candidate generation could not find enough self-excluding neighbors"
            )
        unresolved_rows = np.asarray(next_unresolved, dtype=np.int64)
        search_k = min(int(index.ntotal), max(search_k * 2, neighbor_count + 64))

    return filtered_ids, filtered_scores


def _prune_candidates(
    node_id: int,
    candidate_ids,
    vectors,
    max_degree: int,
    alpha: float,
    metric: str,
):
    import numpy as np

    unique_candidates: list[int] = []
    seen: set[int] = set()
    for candidate_id in candidate_ids:
        candidate_id = int(candidate_id)
        if candidate_id < 0 or candidate_id == node_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        unique_candidates.append(candidate_id)

    if not unique_candidates:
        return np.empty((0,), dtype=np.int32)

    source_vector = vectors[node_id]
    candidate_array = np.asarray(unique_candidates, dtype=np.int32)
    candidate_vectors = vectors[candidate_array]
    source_scores = _score_vector(source_vector, candidate_vectors, metric)
    order = np.argsort(-source_scores, kind="stable")
    ordered_candidates = candidate_array[order]
    ordered_source_scores = source_scores[order]

    selected: list[int] = []
    for candidate_id, source_score in zip(ordered_candidates, ordered_source_scores):
        if len(selected) >= max_degree:
            break
        source_distance = _score_to_distance(float(source_score), metric)
        if not selected:
            selected.append(int(candidate_id))
            continue

        candidate_vector = vectors[int(candidate_id)]
        selected_vectors = vectors[np.asarray(selected, dtype=np.int32)]
        pair_scores = _score_vector(candidate_vector, selected_vectors, metric)
        pair_distances = np.asarray(
            [_score_to_distance(float(item), metric) for item in pair_scores],
            dtype=np.float32,
        )
        if np.any(pair_distances < alpha * source_distance):
            continue
        selected.append(int(candidate_id))

    if len(selected) < max_degree:
        for candidate_id in ordered_candidates:
            candidate_id = int(candidate_id)
            if candidate_id in selected:
                continue
            selected.append(candidate_id)
            if len(selected) >= max_degree:
                break

    return np.asarray(selected, dtype=np.int32)


def _build_vamana_style_graph(
    vectors,
    candidate_ids,
    max_degree: int,
    alpha: float,
    metric: str,
):
    import numpy as np

    node_count = int(vectors.shape[0])
    pruned_outgoing = [
        _prune_candidates(
            node_id=node_id,
            candidate_ids=candidate_ids[node_id],
            vectors=vectors,
            max_degree=max_degree,
            alpha=alpha,
            metric=metric,
        )
        for node_id in range(node_count)
    ]

    merged_candidates: list[set[int]] = [set(map(int, neighbors)) for neighbors in pruned_outgoing]
    for node_id, neighbors in enumerate(pruned_outgoing):
        for neighbor_id in neighbors:
            merged_candidates[int(neighbor_id)].add(int(node_id))

    adjacency: list[np.ndarray] = []
    for node_id in range(node_count):
        union_candidates = list(merged_candidates[node_id])
        for candidate_id in candidate_ids[node_id]:
            candidate_id = int(candidate_id)
            if candidate_id >= 0:
                union_candidates.append(candidate_id)
            if len(union_candidates) >= max_degree * 6:
                break
        neighbors = _prune_candidates(
            node_id=node_id,
            candidate_ids=union_candidates,
            vectors=vectors,
            max_degree=max_degree,
            alpha=alpha,
            metric=metric,
        )
        if neighbors.shape[0] < max_degree:
            fill_ids = []
            seen = set(map(int, neighbors.tolist()))
            for candidate_id in candidate_ids[node_id]:
                candidate_id = int(candidate_id)
                if candidate_id < 0 or candidate_id == node_id or candidate_id in seen:
                    continue
                fill_ids.append(candidate_id)
                seen.add(candidate_id)
                if neighbors.shape[0] + len(fill_ids) >= max_degree:
                    break
            if fill_ids:
                neighbors = np.concatenate(
                    [neighbors, np.asarray(fill_ids, dtype=np.int32)], axis=0
                )[:max_degree]
        adjacency.append(neighbors.astype(np.int32, copy=False))
    return adjacency


def _build_fixed_degree_graph(
    candidate_ids,
    max_degree: int,
):
    import numpy as np

    adjacency: list[np.ndarray] = []
    for node_id in range(int(candidate_ids.shape[0])):
        selected: list[int] = []
        seen: set[int] = set()
        for candidate_id in candidate_ids[node_id]:
            candidate_id = int(candidate_id)
            if candidate_id < 0 or candidate_id == node_id or candidate_id in seen:
                continue
            seen.add(candidate_id)
            selected.append(candidate_id)
            if len(selected) >= int(max_degree):
                break
        adjacency.append(np.asarray(selected, dtype=np.int32))
    return adjacency


def _graph_reachable(adjacency: list, start_node: int) -> set[int]:
    visited = {int(start_node)}
    stack = [int(start_node)]
    while stack:
        node_id = stack.pop()
        for neighbor_id in adjacency[node_id]:
            neighbor_id = int(neighbor_id)
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            stack.append(neighbor_id)
    return visited


def _append_unique_neighbors(existing_neighbors, extra_neighbors, node_id: int):
    import numpy as np

    merged: list[int] = []
    seen: set[int] = set()
    for candidate_id in list(existing_neighbors) + list(extra_neighbors):
        candidate_id = int(candidate_id)
        if candidate_id < 0 or candidate_id == int(node_id) or candidate_id in seen:
            continue
        seen.add(candidate_id)
        merged.append(candidate_id)
    return np.asarray(merged, dtype=np.int32)


def _ensure_reachability_from_entry(
    adjacency: list,
    candidate_ids,
    vectors,
    entry_point: int,
    metric: str,
):
    import numpy as np

    node_count = len(adjacency)
    reachable = _graph_reachable(adjacency, int(entry_point))
    if len(reachable) == node_count:
        return adjacency

    while len(reachable) != node_count:
        reachable_set = set(reachable)
        progress_made = False
        for node_id in range(node_count):
            if node_id in reachable_set:
                continue
            bridge_id = None
            for candidate_id in candidate_ids[node_id]:
                candidate_id = int(candidate_id)
                if candidate_id in reachable_set:
                    bridge_id = candidate_id
                    break
            if bridge_id is None:
                reachable_ids = np.asarray(sorted(reachable_set), dtype=np.int32)
                reachable_vectors = vectors[reachable_ids]
                scores = _score_vector(vectors[node_id], reachable_vectors, metric)
                bridge_id = int(reachable_ids[int(np.argmax(scores))])

            adjacency[node_id] = _append_unique_neighbors(
                existing_neighbors=adjacency[node_id].tolist(),
                extra_neighbors=[int(bridge_id)],
                node_id=node_id,
            )
            adjacency[bridge_id] = _append_unique_neighbors(
                existing_neighbors=adjacency[bridge_id].tolist(),
                extra_neighbors=[int(node_id)],
                node_id=int(bridge_id),
            )
            progress_made = True
        if not progress_made:
            break
        reachable = _graph_reachable(adjacency, int(entry_point))

    if len(reachable) != node_count:
        unreachable = sorted(set(range(node_count)) - set(reachable))
        raise RuntimeError(
            "Failed to make the graph reachable from the entry point. "
            f"Still unreachable: {unreachable[:8]}"
        )

    return adjacency


def _pick_medoid_entry_point(vectors, metric: str) -> int:
    import numpy as np

    center = np.asarray(vectors, dtype=np.float32).mean(axis=0)
    if metric == "cosine":
        norm = float(np.linalg.norm(center))
        if norm <= 0.0:
            return 0
        center = (center / norm).astype(np.float32, copy=False)
    scores = _score_vector(center, vectors, metric)
    return int(np.argmax(scores))


def _effective_graph_params(node_count: int, max_degree: int, candidate_pool: int) -> tuple[int, int]:
    degree = max(1, min(int(max_degree), int(node_count) - 1))
    pool = max(degree + 1, int(candidate_pool))
    pool = min(pool, int(node_count) - 1)
    return degree, pool


def _collect_unique_edge_rows(vectors, adjacency):
    edge_rows: list[tuple[float, int, int]] = []
    for source_id, neighbors in enumerate(adjacency):
        for target_id in neighbors:
            target_id = int(target_id)
            if target_id <= source_id:
                continue
            score = _pair_score(vectors[source_id], vectors[target_id], "cosine")
            edge_rows.append((score, int(source_id), int(target_id)))
    return edge_rows


def _select_long_graph_edges(
    original_vectors,
    baseline_adjacency,
    hidden_count: int,
    metric: str,
):
    import numpy as np

    edge_rows: list[tuple[float, int, int]] = []
    for source_id, neighbors in enumerate(baseline_adjacency):
        for target_id in neighbors:
            target_id = int(target_id)
            if target_id <= source_id:
                continue
            score = _pair_score(original_vectors[source_id], original_vectors[target_id], metric)
            edge_rows.append((score, int(source_id), int(target_id)))
    if not edge_rows:
        raise ValueError("Could not derive any original-original edges from the baseline graph")

    edge_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    node_cap = 8
    selected_edges: list[tuple[int, int]] = []
    per_node_counts: dict[int, int] = {}

    for _, source_id, target_id in edge_rows:
        if per_node_counts.get(source_id, 0) >= node_cap:
            continue
        if per_node_counts.get(target_id, 0) >= node_cap:
            continue
        selected_edges.append((source_id, target_id))
        per_node_counts[source_id] = per_node_counts.get(source_id, 0) + 1
        per_node_counts[target_id] = per_node_counts.get(target_id, 0) + 1
        if len(selected_edges) >= hidden_count:
            break

    if len(selected_edges) < hidden_count:
        seen = set(selected_edges)
        for _, source_id, target_id in edge_rows:
            edge = (source_id, target_id)
            if edge in seen:
                continue
            seen.add(edge)
            selected_edges.append(edge)
            if len(selected_edges) >= hidden_count:
                break

    edge_array = np.asarray(selected_edges[:hidden_count], dtype=np.int32)
    midpoint_vectors = original_vectors[edge_array[:, 0]].astype(np.float32, copy=True)
    midpoint_vectors += original_vectors[edge_array[:, 1]].astype(np.float32, copy=False)
    midpoint_vectors *= 0.5
    midpoint_vectors = _prepare_vectors_for_metric(
        midpoint_vectors,
        metric=metric,
        fallback_vector=original_vectors[0],
    )
    return midpoint_vectors, edge_array


def _build_kmeans_centroids(
    vectors,
    hidden_count: int,
    niter: int,
    seed: int,
    metric: str,
):
    import numpy as np

    hidden_count = min(int(hidden_count), int(vectors.shape[0]))
    if hidden_count <= 0:
        return np.empty((0, int(vectors.shape[1])), dtype=np.float32)

    try:
        import faiss

        dimension = int(vectors.shape[1])
        kmeans = faiss.Kmeans(
            d=dimension,
            k=int(hidden_count),
            niter=int(niter),
            nredo=1,
            spherical=(metric == "cosine"),
            seed=int(seed),
            verbose=False,
        )
        training_data = np.array(vectors, dtype=np.float32, copy=True)
        kmeans.train(training_data)
        centroids = np.asarray(kmeans.centroids, dtype=np.float32)
        return _prepare_vectors_for_metric(
            centroids,
            metric=metric,
            fallback_vector=vectors[0],
        )
    except ModuleNotFoundError:
        # Simple spherical Lloyd iterations provide a no-dependency fallback for local runs.
        rng = np.random.default_rng(seed)
        centroids = vectors[
            rng.choice(int(vectors.shape[0]), size=hidden_count, replace=False)
        ].astype(np.float32, copy=True)
        for _ in range(max(1, int(niter))):
            scores = _score_batch(vectors, centroids, metric)
            assignments = np.argmax(scores, axis=1)
            updated = np.array(centroids, dtype=np.float32, copy=True)
            for cluster_id in range(hidden_count):
                member_ids = np.where(assignments == cluster_id)[0]
                if member_ids.size == 0:
                    updated[cluster_id] = vectors[int(rng.integers(0, int(vectors.shape[0])))]
                    continue
                updated[cluster_id] = _prepare_vector_for_metric(
                    vectors[member_ids].mean(axis=0),
                    metric=metric,
                    fallback_vector=vectors[0],
                )
            if np.allclose(updated, centroids, atol=1e-4):
                centroids = updated
                break
            centroids = updated
        return _prepare_vectors_for_metric(
            centroids,
            metric=metric,
            fallback_vector=vectors[0],
        )


def _assign_to_centroids(vectors, centroids, batch_size: int, metric: str):
    _, ids = _batched_exact_search(
        base_vectors=centroids,
        query_vectors=vectors,
        top_k=1,
        batch_size=batch_size,
        metric=metric,
    )
    return ids[:, 0].astype("int32", copy=False)


def _allocate_counts_with_caps(total: int, caps, weights):
    import numpy as np

    total = max(0, int(total))
    caps = np.asarray(caps, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float64)
    if caps.size == 0:
        return np.empty((0,), dtype=np.int32)
    if np.sum(caps) <= 0 or total <= 0:
        return np.zeros_like(caps)
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(caps, dtype=np.float64)

    counts = np.zeros_like(caps)
    for _ in range(total):
        scores = np.full(caps.shape, -1.0, dtype=np.float64)
        available = counts < caps
        if not np.any(available):
            break
        scores[available] = weights[available] / (counts[available] + 1.0)
        best_idx = int(np.argmax(scores))
        counts[best_idx] += 1
    return counts


def _graph_distance_within_limit(adjacency: list, source_id: int, target_id: int, limit: int) -> bool:
    from collections import deque

    if int(source_id) == int(target_id):
        return True
    queue = deque([(int(source_id), 0)])
    visited = {int(source_id)}
    while queue:
        node_id, depth = queue.popleft()
        if depth >= limit:
            continue
        for neighbor_id in adjacency[node_id]:
            neighbor_id = int(neighbor_id)
            if neighbor_id == int(target_id):
                return True
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            queue.append((neighbor_id, depth + 1))
    return False


def _reverse_degree_counts(adjacency: list, node_count: int):
    import numpy as np

    reverse_degrees = np.zeros((node_count,), dtype=np.float32)
    for node_id in range(node_count):
        reverse_degrees[node_id] += float(len(adjacency[node_id]))
        for neighbor_id in adjacency[node_id]:
            reverse_degrees[int(neighbor_id)] += 1.0
    return reverse_degrees


def _local_boundary_scores(vectors, adjacency: list):
    import numpy as np

    scores = np.zeros((int(vectors.shape[0]),), dtype=np.float32)
    for node_id, neighbors in enumerate(adjacency):
        if len(neighbors) == 0:
            scores[node_id] = 1.0
            continue
        sims = vectors[neighbors] @ vectors[node_id]
        scores[node_id] = float(1.0 - np.mean(sims))
    return scores


def _principal_directions(cluster_vectors, max_count: int):
    import numpy as np

    if int(cluster_vectors.shape[0]) < 2 or max_count <= 0:
        return np.empty((0, int(cluster_vectors.shape[1])), dtype=np.float32)
    centered = cluster_vectors - cluster_vectors.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    directions = np.asarray(vh[:max_count], dtype=np.float32)
    return directions


def _random_orthogonal_unit(direction, rng):
    import numpy as np

    orth = rng.standard_normal(direction.shape[0]).astype(np.float32)
    orth -= np.dot(orth, direction) * direction
    norm = float(np.linalg.norm(orth))
    if norm <= 1e-8:
        orth = np.roll(direction, 1)
        orth -= np.dot(orth, direction) * direction
        norm = float(np.linalg.norm(orth))
    if norm <= 1e-8:
        orth = np.zeros_like(direction)
        orth[0] = 1.0
        orth -= np.dot(orth, direction) * direction
        norm = float(np.linalg.norm(orth))
    return orth / norm


def _unit_scale(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    max_value = float(np.max(values))
    if max_value <= 0.0:
        return np.zeros_like(values)
    return values / max_value


def _select_diverse_offsets(vectors, count: int, seed: int, metric: str):
    import numpy as np

    if int(vectors.shape[0]) == 0 or count <= 0:
        return []
    target_count = min(int(count), int(vectors.shape[0]))
    rng = np.random.default_rng(seed)
    center = np.asarray(vectors, dtype=np.float32).mean(axis=0)
    if metric == "cosine":
        center = _prepare_vector_for_metric(center, metric=metric, fallback_vector=vectors[0])
    center_scores = _score_vector(center, vectors, metric)
    first_idx = int(np.argmin(center_scores))
    selected = [first_idx]
    min_distances = np.asarray(
        [
            _score_to_distance(float(score), metric)
            for score in _score_vector(vectors[first_idx], vectors, metric)
        ],
        dtype=np.float32,
    )

    while len(selected) < target_count:
        scores = np.array(min_distances, copy=True)
        scores[selected] = -1.0
        best_idx = int(np.argmax(scores))
        if best_idx in selected or float(scores[best_idx]) <= 1e-6:
            remaining = [idx for idx in range(int(vectors.shape[0])) if idx not in selected]
            if not remaining:
                break
            best_idx = int(rng.choice(remaining))
        selected.append(best_idx)
        min_distances = np.minimum(
            min_distances,
            np.asarray(
                [
                    _score_to_distance(float(score), metric)
                    for score in _score_vector(vectors[best_idx], vectors, metric)
                ],
                dtype=np.float32,
            ),
        )
    return [int(item) for item in selected]


def _prepare_method_payload(
    hidden_vectors,
    description: str,
    metadata: dict[str, object],
    entry_hidden_offsets: list[int] | None = None,
    query_seed_hidden_offsets: list[int] | None = None,
    query_seed_top_m: int | None = None,
):
    return {
        "hidden_vectors": hidden_vectors,
        "description": description,
        "metadata": metadata,
        "entry_hidden_offsets": list(entry_hidden_offsets or []),
        "query_seed_hidden_offsets": list(query_seed_hidden_offsets or []),
        "query_seed_top_m": int(query_seed_top_m or 0),
    }


def _build_pairwise_interpolation_steiner(
    *,
    original_vectors,
    baseline_adjacency,
    hidden_count: int,
    metric: str,
    **_,
):
    midpoint_vectors, edge_array = _select_long_graph_edges(
        original_vectors=original_vectors,
        baseline_adjacency=baseline_adjacency,
        hidden_count=hidden_count,
        metric=metric,
    )
    return _prepare_method_payload(
        hidden_vectors=midpoint_vectors,
        description="Midpoints inserted on long baseline graph edges.",
        metadata={
            "selected_edge_count": int(edge_array.shape[0]),
        },
    )


def _build_random_midpoint_steiner(
    *,
    original_vectors,
    hidden_count: int,
    seed: int,
    metric: str,
    **_,
):
    """Randomly sample pairs of original points and use their midpoint as Steiner points.

    Unlike pairwise_interpolation (which targets long graph edges) or bridge
    (which targets weak connectivity), this method is completely agnostic to
    graph structure and distances.  It serves as a simple randomised baseline
    for Steiner point placement.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n = int(original_vectors.shape[0])
    idx_a = rng.integers(0, n, size=hidden_count)
    idx_b = rng.integers(0, n, size=hidden_count)
    # Avoid self-pairs
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % n

    hidden_vectors = (
        original_vectors[idx_a].astype(np.float32, copy=True)
        + original_vectors[idx_b].astype(np.float32, copy=False)
    ) * 0.5
    hidden_vectors = _prepare_vectors_for_metric(
        hidden_vectors, metric=metric, fallback_vector=original_vectors[0],
    )
    return _prepare_method_payload(
        hidden_vectors=hidden_vectors,
        description=(
            "Midpoints of randomly sampled pairs of original points. "
            "Graph-agnostic and distance-agnostic baseline for Steiner placement."
        ),
        metadata={
            "pair_count": int(hidden_count),
        },
    )


def _build_cluster_centroid_steiner(
    *,
    original_vectors,
    hidden_count: int,
    kmeans_niter: int,
    query_seed_candidate_count: int,
    query_seed_top_m: int,
    seed: int,
    metric: str,
    **_,
):
    centroids = _build_kmeans_centroids(
        vectors=original_vectors,
        hidden_count=hidden_count,
        niter=kmeans_niter,
        seed=seed,
        metric=metric,
    )
    seed_offsets = _select_diverse_offsets(
        centroids,
        count=max(1, min(int(query_seed_candidate_count), int(centroids.shape[0]))),
        seed=seed + 31,
        metric=metric,
    )
    return _prepare_method_payload(
        hidden_vectors=centroids,
        description="Spherical k-means centroids used as routing-only nodes.",
        metadata={
            "cluster_count": int(centroids.shape[0]),
            "kmeans_niter": int(kmeans_niter),
        },
        query_seed_hidden_offsets=seed_offsets,
        query_seed_top_m=max(1, int(query_seed_top_m)),
    )


def _build_random_line_core(
    *,
    original_vectors,
    hidden_count: int,
    seed: int,
    line_gap_fraction: float,
    line_shift_scale: float,
    include_anchor: bool,
):
    import numpy as np

    rng = np.random.default_rng(seed)
    node_count = int(original_vectors.shape[0])
    dim = int(original_vectors.shape[1])
    requested_count = max(1, min(int(hidden_count), max(2, int(round(math.sqrt(node_count))))))

    center = original_vectors.mean(axis=0).astype(np.float32, copy=False)
    if float(np.linalg.norm(center)) <= 1e-8:
        center = original_vectors[_pick_medoid_entry_point(original_vectors)].astype(
            np.float32, copy=False
        )

    direction = rng.standard_normal(dim).astype(np.float32)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-8:
        direction[0] = 1.0
        direction_norm = 1.0
    direction /= direction_norm

    shift_direction = _random_orthogonal_unit(direction, rng)
    centered = original_vectors - center[None, :]
    projections = centered @ direction
    projected_min = float(np.min(projections))
    projected_max = float(np.max(projections))
    span = max(projected_max - projected_min, 1e-4)
    min_gap = max(1e-4, float(line_gap_fraction) * span / max(requested_count, 1))

    line_point_count = requested_count - (1 if include_anchor else 0)
    line_point_count = max(0, line_point_count)
    candidate_quantiles = np.linspace(0.04, 0.96, max(line_point_count * 5, 8))
    candidate_scalars = np.quantile(projections, candidate_quantiles)
    selected_scalars: list[float] = []
    for scalar in candidate_scalars:
        scalar = float(scalar)
        if any(abs(scalar - existing) < min_gap for existing in selected_scalars):
            continue
        selected_scalars.append(scalar)
        if len(selected_scalars) >= line_point_count:
            break
    if len(selected_scalars) < line_point_count:
        fallback_scalars = np.linspace(projected_min, projected_max, line_point_count)
        for scalar in fallback_scalars:
            scalar = float(scalar)
            if any(abs(scalar - existing) < min_gap * 0.6 for existing in selected_scalars):
                continue
            selected_scalars.append(scalar)
            if len(selected_scalars) >= line_point_count:
                break

    line_points = []
    for scalar in selected_scalars[:line_point_count]:
        point = center + float(scalar) * direction + float(line_shift_scale) * shift_direction
        line_points.append(point)
    if line_points:
        hidden_vectors = _normalize_rows(line_points, fallback_vector=center)
    else:
        hidden_vectors = np.empty((0, dim), dtype=np.float32)

    entry_hidden_offsets: list[int] = []
    if include_anchor:
        anchor_vector = _normalize_vector(center)
        if hidden_vectors.shape[0] == 0:
            hidden_vectors = anchor_vector[None, :].astype(np.float32, copy=False)
            entry_hidden_offsets = [0]
        else:
            hidden_vectors = np.vstack([hidden_vectors, anchor_vector[None, :]]).astype(
                np.float32, copy=False
            )
            entry_hidden_offsets = [int(hidden_vectors.shape[0]) - 1]

    metadata = {
        "actual_hidden_count": int(hidden_vectors.shape[0]),
        "line_gap_fraction": float(line_gap_fraction),
        "line_shift_scale": float(line_shift_scale),
        "random_line_point_count": int(line_point_count),
        "includes_global_anchor": bool(include_anchor),
    }
    description = (
        "Steiner points placed along a random projected line through the dataset mean."
    )
    if include_anchor:
        description += " Includes a synthetic global anchor used as the search start."

    return _prepare_method_payload(
        hidden_vectors=hidden_vectors,
        description=description,
        metadata=metadata,
        entry_hidden_offsets=entry_hidden_offsets,
    )


def _build_random_line_steiner(
    *,
    original_vectors,
    hidden_count: int,
    seed: int,
    line_gap_fraction: float,
    line_shift_scale: float,
    **_,
):
    return _build_random_line_core(
        original_vectors=original_vectors,
        hidden_count=hidden_count,
        seed=seed,
        line_gap_fraction=line_gap_fraction,
        line_shift_scale=line_shift_scale,
        include_anchor=False,
    )


def _build_random_line_anchor_steiner(
    *,
    original_vectors,
    hidden_count: int,
    seed: int,
    line_gap_fraction: float,
    line_shift_scale: float,
    **_,
):
    return _build_random_line_core(
        original_vectors=original_vectors,
        hidden_count=hidden_count,
        seed=seed + 97,
        line_gap_fraction=line_gap_fraction,
        line_shift_scale=line_shift_scale,
        include_anchor=True,
    )


def _build_noisy_copy_steiner(
    *,
    original_vectors,
    hidden_count: int,
    noise_std: float,
    seed: int,
    metric: str,
    **_,
):
    import numpy as np

    rng = np.random.default_rng(seed)
    target_count = min(int(hidden_count), int(original_vectors.shape[0]))
    selected_ids = np.sort(
        rng.choice(int(original_vectors.shape[0]), size=target_count, replace=False)
    ).astype(np.int32)
    hidden_vectors = original_vectors[selected_ids].astype(np.float32, copy=True)
    noise = rng.standard_normal(hidden_vectors.shape).astype(np.float32, copy=False)
    hidden_vectors += float(noise_std) * noise
    hidden_vectors = _prepare_vectors_for_metric(
        hidden_vectors,
        metric=metric,
        fallback_vector=original_vectors[0],
    )
    return _prepare_method_payload(
        hidden_vectors=hidden_vectors,
        description="Small-noise replicas around a random subset of original points.",
        metadata={
            "selected_point_count": int(selected_ids.shape[0]),
            "noise_std": float(noise_std),
        },
    )


def _build_bridge_steiner(
    *,
    original_vectors,
    baseline_adjacency,
    baseline_candidate_ids,
    hidden_count: int,
    metric: str,
    **_,
):
    import numpy as np

    node_count = int(original_vectors.shape[0])
    selected_pairs: list[tuple[int, int]] = []
    pair_rows: list[tuple[float, int, int]] = []
    for source_id in range(node_count):
        local_neighbors = baseline_candidate_ids[source_id][: min(24, baseline_candidate_ids.shape[1])]
        adjacency_set = set(int(item) for item in baseline_adjacency[source_id])
        for target_id in local_neighbors:
            target_id = int(target_id)
            if target_id <= source_id or target_id in adjacency_set:
                continue
            if source_id in set(int(item) for item in baseline_adjacency[target_id]):
                continue
            if _graph_distance_within_limit(
                adjacency=baseline_adjacency,
                source_id=source_id,
                target_id=target_id,
                limit=2,
            ):
                continue
            score = _pair_score(original_vectors[source_id], original_vectors[target_id], metric)
            pair_rows.append((score, source_id, target_id))

    pair_rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    per_node_counts: dict[int, int] = {}
    for _, source_id, target_id in pair_rows:
        if per_node_counts.get(source_id, 0) >= 6:
            continue
        if per_node_counts.get(target_id, 0) >= 6:
            continue
        selected_pairs.append((source_id, target_id))
        per_node_counts[source_id] = per_node_counts.get(source_id, 0) + 1
        per_node_counts[target_id] = per_node_counts.get(target_id, 0) + 1
        if len(selected_pairs) >= hidden_count:
            break

    if len(selected_pairs) < hidden_count:
        fallback_vectors, fallback_pairs = _select_long_graph_edges(
            original_vectors=original_vectors,
            baseline_adjacency=baseline_adjacency,
            hidden_count=hidden_count - len(selected_pairs),
            metric=metric,
        )
        del fallback_vectors
        for pair in fallback_pairs.tolist():
            pair_tuple = (int(pair[0]), int(pair[1]))
            if pair_tuple in selected_pairs:
                continue
            selected_pairs.append(pair_tuple)
            if len(selected_pairs) >= hidden_count:
                break

    pair_array = np.asarray(selected_pairs[:hidden_count], dtype=np.int32)
    hidden_vectors = original_vectors[pair_array[:, 0]].astype(np.float32, copy=True)
    hidden_vectors += original_vectors[pair_array[:, 1]].astype(np.float32, copy=False)
    hidden_vectors *= 0.5
    hidden_vectors = _prepare_vectors_for_metric(
        hidden_vectors,
        metric=metric,
        fallback_vector=original_vectors[0],
    )
    return _prepare_method_payload(
        hidden_vectors=hidden_vectors,
        description=(
            "Bridge points between geometrically close but weakly connected regions in the "
            "baseline graph."
        ),
        metadata={
            "bridge_pair_count": int(pair_array.shape[0]),
            "path_limit_for_weak_connection": 2,
        },
    )


def _build_hierarchical_centroid_steiner(
    *,
    original_vectors,
    hidden_count: int,
    kmeans_niter: int,
    exact_batch_size: int,
    query_seed_candidate_count: int,
    query_seed_top_m: int,
    seed: int,
    metric: str,
    **_,
):
    import numpy as np

    hidden_count = min(int(hidden_count), int(original_vectors.shape[0]))
    coarse_count = max(1, min(hidden_count, int(round(math.sqrt(hidden_count)))))
    fine_budget = max(0, hidden_count - coarse_count)

    coarse_centroids = _build_kmeans_centroids(
        vectors=original_vectors,
        hidden_count=coarse_count,
        niter=kmeans_niter,
        seed=seed,
        metric=metric,
    )
    fine_centroids = []
    if fine_budget > 0 and coarse_centroids.shape[0] > 0:
        assignments = _assign_to_centroids(
            vectors=original_vectors,
            centroids=coarse_centroids,
            batch_size=exact_batch_size,
            metric=metric,
        )
        cluster_sizes = np.bincount(assignments, minlength=int(coarse_centroids.shape[0]))
        caps = np.maximum(cluster_sizes, 0)
        fine_counts = _allocate_counts_with_caps(
            total=fine_budget,
            caps=caps,
            weights=np.maximum(cluster_sizes, 1),
        )
        for cluster_id, cluster_fine_count in enumerate(fine_counts.tolist()):
            if cluster_fine_count <= 0:
                continue
            member_ids = np.where(assignments == cluster_id)[0]
            if member_ids.size == 0:
                continue
            cluster_vectors = original_vectors[member_ids]
            if member_ids.size <= cluster_fine_count:
                fine_centroids.append(cluster_vectors.astype(np.float32, copy=True))
                continue
            fine_centroids.append(
                _build_kmeans_centroids(
                    vectors=cluster_vectors,
                    hidden_count=cluster_fine_count,
                    niter=max(8, kmeans_niter // 2),
                    seed=seed + 1000 + cluster_id,
                    metric=metric,
                )
            )

    blocks = [coarse_centroids]
    blocks.extend(fine_centroids)
    hidden_vectors = np.vstack(blocks).astype(np.float32, copy=False)[:hidden_count]
    hidden_vectors = _normalize_rows(hidden_vectors, fallback_vector=original_vectors[0])
    coarse_seed_count = max(1, min(int(query_seed_candidate_count), int(coarse_centroids.shape[0])))
    seed_offsets = list(range(coarse_seed_count))
    return _prepare_method_payload(
        hidden_vectors=hidden_vectors,
        description="Coarse and fine centroid levels that act as a routing hierarchy.",
        metadata={
            "coarse_centroid_count": int(coarse_centroids.shape[0]),
            "fine_centroid_count": int(hidden_vectors.shape[0] - coarse_centroids.shape[0]),
            "kmeans_niter": int(kmeans_niter),
        },
        query_seed_hidden_offsets=seed_offsets,
        query_seed_top_m=max(1, int(query_seed_top_m)),
    )


def _build_directional_centroid_steiner(
    *,
    original_vectors,
    hidden_count: int,
    kmeans_niter: int,
    exact_batch_size: int,
    seed: int,
    directional_directions_per_cluster: int,
    directional_offset_scale: float,
    metric: str,
    **_,
):
    import numpy as np

    directions_per_cluster = max(1, int(directional_directions_per_cluster))
    points_per_cluster = 1 + (2 * directions_per_cluster)
    cluster_count = max(1, min(int(hidden_count), max(1, int(hidden_count) // points_per_cluster)))

    centroids = _build_kmeans_centroids(
        vectors=original_vectors,
        hidden_count=cluster_count,
        niter=kmeans_niter,
        seed=seed,
        metric=metric,
    )
    assignments = _assign_to_centroids(
        vectors=original_vectors,
        centroids=centroids,
        batch_size=exact_batch_size,
        metric=metric,
    )

    hidden_points: list[np.ndarray] = []
    rng = np.random.default_rng(seed + 17)
    for cluster_id in range(int(centroids.shape[0])):
        centroid = centroids[cluster_id]
        hidden_points.append(centroid.astype(np.float32, copy=True))
        if len(hidden_points) >= hidden_count:
            break
        member_ids = np.where(assignments == cluster_id)[0]
        cluster_vectors = original_vectors[member_ids] if member_ids.size > 0 else centroids[cluster_id][None, :]
        directions = _principal_directions(cluster_vectors, directions_per_cluster)
        if directions.shape[0] == 0:
            random_direction = _random_orthogonal_unit(centroid, rng)
            directions = random_direction[None, :]
        for direction in directions:
            for sign in (1.0, -1.0):
                if len(hidden_points) >= hidden_count:
                    break
                point = centroid + float(sign) * float(directional_offset_scale) * direction
                hidden_points.append(_normalize_vector(point))
            if len(hidden_points) >= hidden_count:
                break

    while len(hidden_points) < hidden_count:
        base_id = len(hidden_points) % int(centroids.shape[0])
        base = centroids[base_id]
        jitter = rng.standard_normal(base.shape[0]).astype(np.float32)
        hidden_points.append(_normalize_vector(base + 0.5 * directional_offset_scale * jitter))

    hidden_vectors = np.asarray(hidden_points[:hidden_count], dtype=np.float32)
    return _prepare_method_payload(
        hidden_vectors=hidden_vectors,
        description=(
            "Cluster centroids plus small centroid-offset points along principal directions."
        ),
        metadata={
            "cluster_count": int(cluster_count),
            "directions_per_cluster": int(directions_per_cluster),
            "directional_offset_scale": float(directional_offset_scale),
        },
    )


def _build_boundary_shell_steiner(
    *,
    original_vectors,
    hidden_count: int,
    kmeans_niter: int,
    exact_batch_size: int,
    seed: int,
    shell_scale: float,
    metric: str,
    **_,
):
    import numpy as np

    cluster_count = max(1, min(int(hidden_count), int(round(math.sqrt(hidden_count * 4)))))
    centroids = _build_kmeans_centroids(
        vectors=original_vectors,
        hidden_count=cluster_count,
        niter=kmeans_niter,
        seed=seed,
        metric=metric,
    )
    assignments = _assign_to_centroids(
        vectors=original_vectors,
        centroids=centroids,
        batch_size=exact_batch_size,
        metric=metric,
    )

    hidden_points: list[np.ndarray] = []
    per_cluster_quota = max(1, int(math.ceil(hidden_count / max(1, cluster_count))))
    for cluster_id in range(int(centroids.shape[0])):
        member_ids = np.where(assignments == cluster_id)[0]
        if member_ids.size == 0:
            continue
        centroid = centroids[cluster_id]
        member_vectors = original_vectors[member_ids]
        sims = member_vectors @ centroid
        order = np.argsort(sims, kind="stable")
        for local_idx in order[:per_cluster_quota]:
            direction = member_vectors[local_idx] - centroid
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1e-8:
                continue
            point = centroid + (1.0 + float(shell_scale)) * direction
            hidden_points.append(_normalize_vector(point))
            if len(hidden_points) >= hidden_count:
                break
        if len(hidden_points) >= hidden_count:
            break

    if len(hidden_points) < hidden_count:
        hidden_points.extend(centroids[: hidden_count - len(hidden_points)])

    hidden_vectors = np.asarray(hidden_points[:hidden_count], dtype=np.float32)
    return _prepare_method_payload(
        hidden_vectors=hidden_vectors,
        description="Shell points slightly outside local clusters to act as gateway nodes.",
        metadata={
            "cluster_count": int(cluster_count),
            "shell_scale": float(shell_scale),
        },
    )


def _build_failure_driven_steiner(
    *,
    original_vectors,
    baseline_adjacency,
    hidden_count: int,
    validation_signals,
    metric: str,
    **_,
):
    import numpy as np

    failure_pair_stats = validation_signals["failure_pair_stats"]
    pair_rows = []
    for pair_key, item in failure_pair_stats.items():
        del pair_key
        count = int(item["count"])
        source_id = int(item["source_id"])
        target_id = int(item["target_id"])
        score = _pair_score(original_vectors[source_id], original_vectors[target_id], metric)
        pair_rows.append((count, score, source_id, target_id))
    pair_rows.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))

    selected_pairs: list[tuple[int, int]] = []
    per_node_counts: dict[int, int] = {}
    for count, sim, source_id, target_id in pair_rows:
        del count, sim
        if per_node_counts.get(source_id, 0) >= 6:
            continue
        if per_node_counts.get(target_id, 0) >= 6:
            continue
        selected_pairs.append((source_id, target_id))
        per_node_counts[source_id] = per_node_counts.get(source_id, 0) + 1
        per_node_counts[target_id] = per_node_counts.get(target_id, 0) + 1
        if len(selected_pairs) >= hidden_count:
            break

    if len(selected_pairs) < hidden_count:
        fallback_vectors, fallback_pairs = _select_long_graph_edges(
            original_vectors=original_vectors,
            baseline_adjacency=baseline_adjacency,
            hidden_count=hidden_count - len(selected_pairs),
            metric=metric,
        )
        del fallback_vectors
        for pair in fallback_pairs.tolist():
            pair_tuple = (int(pair[0]), int(pair[1]))
            if pair_tuple in selected_pairs:
                continue
            selected_pairs.append(pair_tuple)
            if len(selected_pairs) >= hidden_count:
                break

    pair_array = np.asarray(selected_pairs[:hidden_count], dtype=np.int32)
    hidden_vectors = (
        0.6 * original_vectors[pair_array[:, 0]].astype(np.float32, copy=True)
        + 0.4 * original_vectors[pair_array[:, 1]].astype(np.float32, copy=False)
    )
    hidden_vectors = _prepare_vectors_for_metric(
        hidden_vectors,
        metric=metric,
        fallback_vector=original_vectors[0],
    )
    return _prepare_method_payload(
        hidden_vectors=hidden_vectors,
        description=(
            "Validation-driven Steiner points between stuck search nodes and missed true "
            "neighbors."
        ),
        metadata={
            "failed_query_count": int(validation_signals["failed_query_count"]),
            "selected_failure_pair_count": int(pair_array.shape[0]),
            "validation_query_count": int(validation_signals["query_count"]),
        },
    )


def _build_targeted_noisy_replica_steiner(
    *,
    original_vectors,
    baseline_adjacency,
    hidden_count: int,
    validation_signals,
    noise_std: float,
    seed: int,
    **_,
):
    import numpy as np

    visit_counts = np.asarray(validation_signals["visit_counts"], dtype=np.float32)
    reverse_degrees = _reverse_degree_counts(baseline_adjacency, int(original_vectors.shape[0]))
    boundary_scores = _local_boundary_scores(original_vectors, baseline_adjacency)

    importance = (
        0.45 * _unit_scale(visit_counts)
        + 0.35 * _unit_scale(reverse_degrees)
        + 0.20 * _unit_scale(boundary_scores)
    )
    target_count = min(int(hidden_count), int(original_vectors.shape[0]))
    selected_ids = np.argsort(-importance, kind="stable")[:target_count].astype(np.int32)

    rng = np.random.default_rng(seed + 203)
    hidden_vectors = original_vectors[selected_ids].astype(np.float32, copy=True)
    noise = rng.standard_normal(hidden_vectors.shape).astype(np.float32, copy=False)
    hidden_vectors += float(noise_std) * 0.75 * noise
    hidden_vectors = _normalize_rows(hidden_vectors, fallback_vector=original_vectors[0])
    return _prepare_method_payload(
        hidden_vectors=hidden_vectors,
        description=(
            "Small-noise replicas targeted at hubs, bottlenecks, and frequently visited nodes."
        ),
        metadata={
            "selected_point_count": int(selected_ids.shape[0]),
            "noise_std": float(noise_std) * 0.75,
            "importance_signal_weights": {
                "visit_counts": 0.45,
                "reverse_degree": 0.35,
                "boundary_score": 0.20,
            },
        },
    )


def _build_local_knn_mean_steiner(
    *,
    original_vectors,
    baseline_adjacency,
    baseline_candidate_ids,
    hidden_count: int,
    validation_signals,
    local_mean_neighbor_count: int,
    query_seed_candidate_count: int,
    query_seed_top_m: int,
    seed: int,
    metric: str,
    **_,
):
    import numpy as np

    visit_counts = np.asarray(validation_signals["visit_counts"], dtype=np.float32)
    reverse_degrees = _reverse_degree_counts(baseline_adjacency, int(original_vectors.shape[0]))
    boundary_scores = _local_boundary_scores(original_vectors, baseline_adjacency)
    importance = (
        0.50 * _unit_scale(visit_counts)
        + 0.30 * _unit_scale(reverse_degrees)
        + 0.20 * _unit_scale(boundary_scores)
    )
    anchor_count = min(int(hidden_count), int(original_vectors.shape[0]))
    anchor_ids = np.argsort(-importance, kind="stable")[:anchor_count].astype(np.int32)

    neighbor_count = max(1, min(int(local_mean_neighbor_count), baseline_candidate_ids.shape[1]))
    hidden_vectors = []
    representative_rows = []
    for anchor_id in anchor_ids.tolist():
        selected = [int(anchor_id)]
        for candidate_id in baseline_candidate_ids[int(anchor_id)]:
            candidate_id = int(candidate_id)
            if candidate_id < 0 or candidate_id == int(anchor_id) or candidate_id in selected:
                continue
            selected.append(candidate_id)
            if len(selected) >= neighbor_count + 1:
                break
        rep_ids = np.asarray(selected, dtype=np.int32)
        representative_rows.append(rep_ids.tolist())
        hidden_vectors.append(
            _normalize_vector(original_vectors[rep_ids].mean(axis=0, dtype=np.float32))
        )

    hidden_vectors = np.asarray(hidden_vectors, dtype=np.float32)
    seed_offsets = _select_diverse_offsets(
        hidden_vectors,
        count=max(1, min(int(query_seed_candidate_count), int(hidden_vectors.shape[0]))),
        seed=seed + 19,
        metric=metric,
    )
    return _prepare_method_payload(
        hidden_vectors=hidden_vectors,
        description=(
            "Local k-NN means around important baseline nodes, plus diversified Steiner seeds "
            "used at query time."
        ),
        metadata={
            "anchor_count": int(anchor_ids.shape[0]),
            "neighbor_count": int(neighbor_count),
            "query_seed_candidate_count": int(len(seed_offsets)),
            "query_seed_top_m": int(max(1, int(query_seed_top_m))),
            "importance_signal_weights": {
                "visit_counts": 0.50,
                "reverse_degree": 0.30,
                "boundary_score": 0.20,
            },
        },
        entry_hidden_offsets=seed_offsets[: max(1, min(int(query_seed_top_m), len(seed_offsets)))],
        query_seed_hidden_offsets=seed_offsets,
        query_seed_top_m=max(1, int(query_seed_top_m)),
    )


def _method_registry():
    return {
        "pairwise_interpolation": _build_pairwise_interpolation_steiner,
        "cluster_centroid": _build_cluster_centroid_steiner,
        "random_midpoint": _build_random_midpoint_steiner,
        "local_knn_mean": _build_local_knn_mean_steiner,
        "random_line": _build_random_line_steiner,
        "random_line_anchor": _build_random_line_anchor_steiner,
        "noisy_copy": _build_noisy_copy_steiner,
        "bridge": _build_bridge_steiner,
        "hierarchical_centroid": _build_hierarchical_centroid_steiner,
        "directional_centroid": _build_directional_centroid_steiner,
        "boundary_shell": _build_boundary_shell_steiner,
        "failure_driven": _build_failure_driven_steiner,
        "targeted_noisy_replicas": _build_targeted_noisy_replica_steiner,
    }


def _default_methods() -> list[str]:
    return list(_method_registry().keys())


def _beam_search_topk(
    query_vector,
    vectors,
    adjacency,
    entry_points,
    beam_size: int,
    top_k: int,
    visible_count: int,
    metric: str,
    query_seed_node_ids=None,
    query_seed_top_m: int = 0,
    return_trace: bool = False,
    is_visible=None,
    aug_to_orig=None,
):
    import heapq
    import numpy as np

    if beam_size <= 0:
        raise ValueError("beam_size must be positive")

    # Support both legacy visible_count threshold and explicit is_visible mask.
    _is_visible = is_visible
    _aug_to_orig = aug_to_orig

    def _node_is_visible(node_id: int) -> bool:
        if _is_visible is not None:
            return bool(_is_visible[node_id])
        return node_id < visible_count

    def _to_orig_id(node_id: int) -> int:
        if _aug_to_orig is not None:
            return int(_aug_to_orig[node_id])
        return node_id

    visited_scores: dict[int, float] = {}
    active: list[list[object]] = []
    top_real_heap: list[tuple[float, int]] = []
    expanded_order: list[int] = []

    # active is sorted descending by score: [(score, node_id, expanded), ...]
    # We use (-score, node_id) tuples for bisect (ascending order = descending score).
    import bisect
    _active_sort_keys = []  # parallel to active: (-score, node_id)

    def insert_active(node_id: int, score: float) -> None:
        key = (-float(score), int(node_id))
        pos = bisect.bisect_left(_active_sort_keys, key)
        if pos >= beam_size:
            return
        _active_sort_keys.insert(pos, key)
        active.insert(pos, [float(score), int(node_id), False])
        if len(active) > beam_size:
            active.pop()
            _active_sort_keys.pop()

    def visit(node_id: int) -> None:
        node_id = int(node_id)
        if node_id in visited_scores:
            return
        score = _pair_score(vectors[node_id], query_vector, metric)
        visited_scores[node_id] = score
        insert_active(node_id, score)
        if _node_is_visible(node_id):
            if len(top_real_heap) < top_k:
                heapq.heappush(top_real_heap, (score, node_id))
            elif score > top_real_heap[0][0]:
                heapq.heapreplace(top_real_heap, (score, node_id))

    if query_seed_node_ids:
        seed_node_ids = [int(node_id) for node_id in query_seed_node_ids]
        seed_vectors = vectors[np.asarray(seed_node_ids, dtype=np.int32)]
        seed_scores = _score_vector(query_vector, seed_vectors, metric)
        for node_id, score in zip(seed_node_ids, seed_scores.tolist()):
            if node_id in visited_scores:
                continue
            visited_scores[node_id] = float(score)
            if _node_is_visible(node_id):
                if len(top_real_heap) < top_k:
                    heapq.heappush(top_real_heap, (float(score), int(node_id)))
                elif float(score) > top_real_heap[0][0]:
                    heapq.heapreplace(top_real_heap, (float(score), int(node_id)))
        top_m = max(0, min(int(query_seed_top_m), len(seed_node_ids)))
        if top_m > 0:
            order = np.argsort(-seed_scores, kind="stable")[:top_m]
            for offset in order.tolist():
                insert_active(seed_node_ids[int(offset)], float(seed_scores[int(offset)]))

    for entry_point in entry_points:
        visit(int(entry_point))

    while True:
        expand_idx = None
        for idx, item in enumerate(active):
            if not bool(item[2]):
                expand_idx = idx
                break
        if expand_idx is None:
            break

        active[expand_idx][2] = True
        node_id = int(active[expand_idx][1])
        expanded_order.append(node_id)

        # Batch distance computation for all unvisited neighbors
        neighbors = adjacency[node_id]
        unvisited = [int(nid) for nid in neighbors if int(nid) not in visited_scores]
        if unvisited:
            nid_array = np.asarray(unvisited, dtype=np.int32)
            neighbor_vecs = vectors[nid_array]
            if metric == "cosine":
                scores = (query_vector @ neighbor_vecs.T).tolist()
            else:
                diff = neighbor_vecs - query_vector
                scores = (-(diff * diff).sum(axis=1)).tolist()
            for nid, score in zip(unvisited, scores):
                visited_scores[nid] = score
                insert_active(nid, score)
                if _node_is_visible(nid):
                    if len(top_real_heap) < top_k:
                        heapq.heappush(top_real_heap, (score, nid))
                    elif score > top_real_heap[0][0]:
                        heapq.heapreplace(top_real_heap, (score, nid))

    # Map predicted IDs back to original-space indices for ground truth comparison.
    predicted_ids = [
        _to_orig_id(node_id) for _, node_id in sorted(top_real_heap, reverse=True)
    ]
    predicted_ids = predicted_ids[:top_k]
    hidden_visits = sum(1 for node_id in visited_scores if not _node_is_visible(node_id))
    payload = {
        "predicted_ids": predicted_ids,
        "distance_computations": int(len(visited_scores)),
        "visible_distance_computations": int(len(visited_scores) - hidden_visits),
        "hidden_distance_computations": int(hidden_visits),
        "hop_count": int(len(expanded_order)),
    }
    if return_trace:
        payload["expanded_order"] = list(expanded_order)
        payload["visited_node_ids"] = [int(node_id) for node_id in visited_scores]
    return payload


def _collect_validation_signals(
    *,
    original_vectors,
    baseline_adjacency,
    baseline_entry_points,
    query_vectors,
    ground_truth_ids,
    validation_query_count: int,
    validation_beam_size: int,
    top_k: int,
    failure_recall_threshold: float,
    metric: str,
):
    import numpy as np

    query_count = min(int(validation_query_count), int(query_vectors.shape[0]))
    visible_count = int(original_vectors.shape[0])
    visit_counts = np.zeros((visible_count,), dtype=np.int64)
    expansion_counts = np.zeros((visible_count,), dtype=np.int64)
    failure_pair_stats: dict[tuple[int, int], dict[str, object]] = {}
    failed_query_count = 0

    for query_id in range(query_count):
        result = _beam_search_topk(
            query_vector=query_vectors[query_id],
            vectors=original_vectors,
            adjacency=baseline_adjacency,
            entry_points=baseline_entry_points,
            beam_size=int(validation_beam_size),
            top_k=int(top_k),
            visible_count=visible_count,
            metric=metric,
            return_trace=True,
        )
        truth_ids = [int(item) for item in ground_truth_ids[query_id]]
        predicted_ids = [int(item) for item in result["predicted_ids"]]
        predicted_set = set(predicted_ids)
        truth_set = set(truth_ids)
        recall_at_k = len(predicted_set & truth_set) / float(top_k)

        for node_id in result["visited_node_ids"]:
            if int(node_id) < visible_count:
                visit_counts[int(node_id)] += 1
        for node_id in result["expanded_order"]:
            if int(node_id) < visible_count:
                expansion_counts[int(node_id)] += 1

        if recall_at_k >= float(failure_recall_threshold) and truth_ids[0] in predicted_set:
            continue

        failed_query_count += 1
        # These traces let later Steiner builders target where the search actually gets stuck.
        visible_expanded = [int(node_id) for node_id in result["expanded_order"] if int(node_id) < visible_count]
        visible_visited = [int(node_id) for node_id in result["visited_node_ids"] if int(node_id) < visible_count]
        stuck_id = visible_expanded[-1] if visible_expanded else visible_visited[-1]
        best_visited_id = max(
            visible_visited,
            key=lambda node_id: _pair_score(
                original_vectors[node_id],
                query_vectors[query_id],
                metric,
            ),
            default=stuck_id,
        )
        missed_ids = [int(node_id) for node_id in truth_ids if int(node_id) not in predicted_set]
        if not missed_ids:
            missed_ids = [int(truth_ids[0])]
        for missed_id in missed_ids[:2]:
            key = (int(best_visited_id), int(missed_id))
            item = failure_pair_stats.setdefault(
                key,
                {
                    "count": 0,
                    "source_id": int(best_visited_id),
                    "target_id": int(missed_id),
                    "recall_sum": 0.0,
                    "stuck_sum": 0,
                },
            )
            item["count"] = int(item["count"]) + 1
            item["recall_sum"] = float(item["recall_sum"]) + float(recall_at_k)
            item["stuck_sum"] = int(item["stuck_sum"]) + int(stuck_id)

    return {
        "query_count": int(query_count),
        "beam_size": int(validation_beam_size),
        "failure_recall_threshold": float(failure_recall_threshold),
        "failed_query_count": int(failed_query_count),
        "visit_counts": visit_counts.tolist(),
        "expansion_counts": expansion_counts.tolist(),
        "failure_pair_stats": failure_pair_stats,
    }


def _run_single_query(args):
    """Worker function for parallel beam search. Runs one query and returns raw counts."""
    (query_id, query_vector, vectors, adjacency, entry_points, beam_size,
     top_k, visible_count, metric, query_seed_node_ids, query_seed_top_m,
     is_visible, aug_to_orig, truth_ids_list) = args

    result = _beam_search_topk(
        query_vector=query_vector,
        vectors=vectors,
        adjacency=adjacency,
        entry_points=entry_points,
        beam_size=beam_size,
        top_k=top_k,
        visible_count=visible_count,
        metric=metric,
        query_seed_node_ids=query_seed_node_ids,
        query_seed_top_m=query_seed_top_m,
        is_visible=is_visible,
        aug_to_orig=aug_to_orig,
    )
    predicted_ids = result["predicted_ids"]
    truth_ids = truth_ids_list
    truth_top1 = int(truth_ids[0])
    predicted_top1 = int(predicted_ids[0]) if predicted_ids else -1
    predicted_set = {int(x) for x in predicted_ids}
    truth_set = {int(x) for x in truth_ids}

    recall_at_1_hit = 1 if predicted_top1 == truth_top1 else 0
    top1_in_topk_hit = 1 if truth_top1 in predicted_set else 0
    predicted_top10 = set(int(x) for x in predicted_ids[:10])
    truth_top10 = set(int(x) for x in truth_ids[:10])
    recall_at_10 = len(predicted_top10 & truth_top10) / min(10.0, float(top_k))
    recall_at_k = len(predicted_set & truth_set) / float(top_k)

    return {
        "recall_at_1_hit": recall_at_1_hit,
        "recall_at_10": recall_at_10,
        "recall_at_k": recall_at_k,
        "top1_in_topk_hit": top1_in_topk_hit,
        "distance_computations": int(result["distance_computations"]),
        "visible_distance_computations": int(result["visible_distance_computations"]),
        "hidden_distance_computations": int(result["hidden_distance_computations"]),
        "hop_count": int(result["hop_count"]),
    }


# Module-level globals for shared data in worker processes (set via _init_worker).
_shared_vectors = None
_shared_adjacency = None
_shared_ground_truth_ids = None
_shared_query_vectors = None


def _init_worker(vectors, adjacency, ground_truth_ids, query_vectors):
    """Initialize shared read-only data in each worker process."""
    global _shared_vectors, _shared_adjacency, _shared_ground_truth_ids, _shared_query_vectors
    _shared_vectors = vectors
    _shared_adjacency = adjacency
    _shared_ground_truth_ids = ground_truth_ids
    _shared_query_vectors = query_vectors


def _worker_run_query(args):
    """Thin wrapper that reads shared data from module globals."""
    (query_id, beam_size, top_k, visible_count, metric,
     entry_points, query_seed_node_ids, query_seed_top_m,
     is_visible, aug_to_orig) = args

    return _run_single_query((
        query_id,
        _shared_query_vectors[query_id],
        _shared_vectors,
        _shared_adjacency,
        entry_points,
        beam_size,
        top_k,
        visible_count,
        metric,
        query_seed_node_ids,
        query_seed_top_m,
        is_visible,
        aug_to_orig,
        [int(x) for x in _shared_ground_truth_ids[query_id]],
    ))


def _get_num_workers():
    """Return number of parallel workers. Leaves 1 core free for OS."""
    import os
    return max(1, (os.cpu_count() or 1) - 1)


def _evaluate_graph(
    method_name: str,
    vectors,
    adjacency,
    query_vectors,
    ground_truth_ids,
    beam_sizes,
    visible_count: int,
    entry_points,
    top_k: int,
    metric: str,
    query_seed_node_ids=None,
    query_seed_top_m: int = 0,
    is_visible=None,
    aug_to_orig=None,
    max_comp_fraction: float = 0.0,
):
    import multiprocessing as mp

    query_count = int(query_vectors.shape[0])
    variants: dict[str, dict[str, object]] = {}
    comp_limit = float(max_comp_fraction) * float(visible_count) if max_comp_fraction > 0 else 0.0
    num_workers = _get_num_workers()

    # Use fork-based pool so workers inherit vectors/adjacency without copying.
    ctx = mp.get_context("fork")
    pool = ctx.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(vectors, adjacency, ground_truth_ids, query_vectors),
    )

    try:
        for beam_size in beam_sizes:
            work_items = [
                (query_id, int(beam_size), int(top_k), visible_count, metric,
                 entry_points, query_seed_node_ids, int(query_seed_top_m),
                 is_visible, aug_to_orig)
                for query_id in range(query_count)
            ]

            results_list = pool.map(_worker_run_query, work_items, chunksize=max(1, query_count // (num_workers * 4)))

            total_distance_comps = 0
            total_visible_distance_comps = 0
            total_hidden_distance_comps = 0
            total_hops = 0
            recall_at_1_hits = 0
            recall_at_10_sum = 0.0
            recall_at_k_sum = 0.0
            top1_in_topk_hits = 0

            for r in results_list:
                recall_at_1_hits += r["recall_at_1_hit"]
                recall_at_10_sum += r["recall_at_10"]
                recall_at_k_sum += r["recall_at_k"]
                top1_in_topk_hits += r["top1_in_topk_hit"]
                total_distance_comps += r["distance_computations"]
                total_visible_distance_comps += r["visible_distance_computations"]
                total_hidden_distance_comps += r["hidden_distance_computations"]
                total_hops += r["hop_count"]

            avg_dist_comps = total_distance_comps / float(query_count)
            variant_id = f"{method_name}_beam{beam_size}"
            variants[variant_id] = {
                "method": method_name,
                "beam_size": int(beam_size),
                "query_count": query_count,
                "top_k": int(top_k),
                "avg_distance_computations": avg_dist_comps,
                "avg_visible_distance_computations": total_visible_distance_comps
                / float(query_count),
                "avg_hidden_distance_computations": total_hidden_distance_comps
                / float(query_count),
                "avg_hop_count": total_hops / float(query_count),
                "recall_at_1": recall_at_1_hits / float(query_count),
                "recall_at_10": recall_at_10_sum / float(query_count),
                "recall_at_k": recall_at_k_sum / float(query_count),
                "true_top1_hit_in_topk": top1_in_topk_hits / float(query_count),
            }
            print(f"  [{method_name}] beam={beam_size}: R@1={variants[variant_id]['recall_at_1']:.4f}, "
                  f"R@10={variants[variant_id]['recall_at_10']:.4f}, "
                  f"avg_comps={avg_dist_comps:.0f} ({num_workers} workers)")
            # Stop early if distance computations exceed the fraction threshold.
            if comp_limit > 0 and avg_dist_comps >= comp_limit:
                break
    finally:
        pool.close()
        pool.join()

    return variants


def _curve_auc(variants: dict[str, dict[str, object]], metric_key: str) -> float:
    import numpy as np

    points = sorted(variants.values(), key=lambda item: item["avg_distance_computations"])
    xs = np.asarray(
        [float(item["avg_distance_computations"]) for item in points],
        dtype=np.float64,
    )
    ys = np.asarray([float(item[metric_key]) for item in points], dtype=np.float64)
    if xs.size == 0:
        return 0.0
    if xs.size == 1:
        return float(ys[0])
    log_xs = np.log(xs)
    x_span = float(log_xs[-1] - log_xs[0])
    if x_span <= 0.0:
        return float(np.mean(ys))
    return float(np.trapezoid(ys, log_xs) / x_span)


def _summarize_method_results(
    results: dict[str, dict[str, dict[str, object]]],
) -> dict[str, object]:
    by_method: dict[str, dict[str, object]] = {}
    baseline_by_beam = {
        int(item["beam_size"]): item for item in results["baseline"].values()
    }
    for method_name, variants in results.items():
        ordered_variants = sorted(variants.items(), key=lambda item: item[1]["recall_at_k"], reverse=True)
        best_variant_id, best_variant = ordered_variants[0]
        summary = {
            "recall_at_1_auc": _curve_auc(variants, "recall_at_1"),
            "recall_at_k_auc": _curve_auc(variants, "recall_at_k"),
            "best_variant_id": best_variant_id,
            "best_variant_beam_size": int(best_variant["beam_size"]),
            "best_variant_recall_at_1": float(best_variant["recall_at_1"]),
            "best_variant_recall_at_k": float(best_variant["recall_at_k"]),
            "best_variant_avg_distance_computations": float(
                best_variant["avg_distance_computations"]
            ),
        }
        if method_name != "baseline":
            deltas_r1 = []
            deltas_rk = []
            deltas_comp = []
            deltas_hops = []
            for item in variants.values():
                beam_sz = int(item["beam_size"])
                if beam_sz not in baseline_by_beam:
                    continue
                baseline_item = baseline_by_beam[beam_sz]
                deltas_r1.append(float(item["recall_at_1"]) - float(baseline_item["recall_at_1"]))
                deltas_rk.append(float(item["recall_at_k"]) - float(baseline_item["recall_at_k"]))
                deltas_comp.append(
                    float(item["avg_distance_computations"])
                    - float(baseline_item["avg_distance_computations"])
                )
                deltas_hops.append(
                    float(item["avg_hop_count"]) - float(baseline_item["avg_hop_count"])
                )
            if deltas_r1:
                summary["matched_beam_mean_delta_recall_at_1"] = sum(deltas_r1) / len(deltas_r1)
                summary["matched_beam_mean_delta_recall_at_k"] = sum(deltas_rk) / len(deltas_rk)
                summary["matched_beam_mean_delta_distance_computations"] = (
                    sum(deltas_comp) / len(deltas_comp)
                )
                summary["matched_beam_mean_delta_hop_count"] = sum(deltas_hops) / len(deltas_hops)
        by_method[method_name] = summary

    rank_r1 = sorted(
        by_method.items(),
        key=lambda item: float(item[1]["recall_at_1_auc"]),
        reverse=True,
    )
    rank_rk = sorted(
        by_method.items(),
        key=lambda item: float(item[1]["recall_at_k_auc"]),
        reverse=True,
    )
    return {
        "by_method": by_method,
        "ranking_by_recall_at_1_auc": [item[0] for item in rank_r1],
        "ranking_by_recall_at_k_auc": [item[0] for item in rank_rk],
        "best_method_by_recall_at_k_auc": rank_rk[0][0] if rank_rk else None,
    }


def _plot_method_styles() -> dict[str, dict[str, str]]:
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


def _method_actual_hidden_count(
    payload: dict[str, object],
    method_name: str,
) -> int:
    if method_name == "baseline":
        return 0
    steiner_methods = payload.get("steiner", {}).get("methods", {})
    if not isinstance(steiner_methods, dict):
        return 0
    method_metadata = steiner_methods.get(method_name, {})
    if not isinstance(method_metadata, dict):
        return 0
    return int(method_metadata.get("actual_hidden_count", 0))


def _plot_label_for_method(
    payload: dict[str, object],
    method_name: str,
    base_label: str,
) -> str:
    actual_hidden_count = _method_actual_hidden_count(payload, method_name)
    if method_name == "baseline" or actual_hidden_count <= 0:
        return base_label
    return f"{base_label} (S={actual_hidden_count})"


def _plot_x_bounds(payloads: list[dict[str, object]]):
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


def _style_tradeoff_axis(axis, x_limits, x_ticks):
    from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

    axis.set_xscale("log")
    axis.set_xlim(*x_limits)
    axis.set_ylim(0.0, 1.02)
    axis.set_axisbelow(True)
    axis.margins(x=0.03, y=0.02)
    axis.xaxis.set_major_locator(FixedLocator(x_ticks))
    axis.set_xticklabels([_format_x_tick(item) for item in x_ticks])
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


def _save_tradeoff_plot(
    payload: dict[str, object],
    plot_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.labelsize": 13.5,
            "axes.titlesize": 16.5,
            "legend.fontsize": 10.4,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "font.size": 12.0,
        }
    )
    method_styles = _plot_method_styles()
    x_limits = _plot_x_bounds([payload])
    x_ticks = _plot_x_ticks([payload])

    figure, axes = plt.subplots(1, 3, figsize=(24, 8.9), dpi=240, sharex=True)
    figure.patch.set_facecolor("white")
    panels = [
        ("recall_at_1", "Recall@1"),
        ("recall_at_10", "Recall@10"),
        ("recall_at_k", f"Recall@{payload['search']['top_k']}"),
    ]

    for axis, (metric_key, metric_label) in zip(axes, panels):
        for method_name, series in payload["results"].items():
            style = method_styles.get(
                method_name,
                {
                    "label": method_name.replace("_", " ").title(),
                    "color": "#6B7280",
                    "marker": "o",
                },
            )
            points = sorted(series.values(), key=lambda item: item["avg_distance_computations"])
            xs = [float(item["avg_distance_computations"]) for item in points]
            ys = [float(item[metric_key]) for item in points]
            axis.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                linestyle=style.get("linestyle", "-"),
                linewidth=3.05,
                markersize=9.2,
                markeredgecolor="white",
                markeredgewidth=1.05,
                alpha=0.98,
                label=_plot_label_for_method(
                    payload=payload,
                    method_name=method_name,
                    base_label=style["label"],
                ),
            )
        axis.set_xlabel("Average distance computations per query")
        axis.set_ylabel(metric_label)
        axis.set_title(metric_label, fontsize=15.5, fontweight="bold", pad=12)
        _style_tradeoff_axis(axis, x_limits=x_limits, x_ticks=x_ticks)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        frameon=True,
        ncol=min(4, max(1, len(labels))),
        fontsize=10.4,
        bbox_to_anchor=(0.5, 0.972),
        borderpad=0.7,
        handlelength=2.6,
        edgecolor="#CBD5E1",
        facecolor="white",
    )
    figure.suptitle(
        (
            "DiskANN-style graph search on "
            f"{payload['dataset']['name']} with routing-only Steiner points"
        ),
        fontsize=18.5,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"Dataset subset: {payload['dataset']['train_size']} train vectors, "
            f"{payload['dataset']['query_count']} queries, dimension {payload['dataset']['dimension']}. "
            f"Requested hidden budget: {payload['steiner']['requested_hidden_count']}. "
            "Returned neighbors are always restricted to original data points."
        ),
        ha="center",
        va="bottom",
        fontsize=11,
        color="#374151",
    )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.018, 0.075, 0.982, 0.90))
    figure.savefig(plot_path, bbox_inches="tight")
    plt.close(figure)


def _build_curve_table_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_name, series in payload["results"].items():
        for variant_id, item in sorted(
            series.items(),
            key=lambda pair: int(pair[1]["beam_size"]),
        ):
            rows.append(
                {
                    "dataset": payload["dataset"]["name"],
                    "variant_id": variant_id,
                    "method": method_name,
                    "actual_hidden_count": _method_actual_hidden_count(
                        payload=payload,
                        method_name=method_name,
                    ),
                    "beam_size": int(item["beam_size"]),
                    "avg_distance_computations": float(item["avg_distance_computations"]),
                    "avg_visible_distance_computations": float(
                        item["avg_visible_distance_computations"]
                    ),
                    "avg_hidden_distance_computations": float(
                        item["avg_hidden_distance_computations"]
                    ),
                    "avg_hop_count": float(item["avg_hop_count"]),
                    "recall_at_1": float(item["recall_at_1"]),
                    "recall_at_10": float(item.get("recall_at_10", 0.0)),
                    "recall_at_k": float(item["recall_at_k"]),
                    "true_top1_hit_in_topk": float(item["true_top1_hit_in_topk"]),
                }
            )
    return rows


def _build_method_summary_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    summary = payload["summary"]["by_method"]
    for method_name, item in summary.items():
        rows.append(
            {
                "dataset": payload["dataset"]["name"],
                "method": method_name,
                "actual_hidden_count": _method_actual_hidden_count(
                    payload=payload,
                    method_name=method_name,
                ),
                "recall_at_1_auc": float(item["recall_at_1_auc"]),
                "recall_at_k_auc": float(item["recall_at_k_auc"]),
                "best_variant_id": item["best_variant_id"],
                "best_variant_beam_size": int(item["best_variant_beam_size"]),
                "best_variant_recall_at_1": float(item["best_variant_recall_at_1"]),
                "best_variant_recall_at_k": float(item["best_variant_recall_at_k"]),
                "best_variant_avg_distance_computations": float(
                    item["best_variant_avg_distance_computations"]
                ),
                "matched_beam_mean_delta_recall_at_1": float(
                    item.get("matched_beam_mean_delta_recall_at_1", 0.0)
                ),
                "matched_beam_mean_delta_recall_at_k": float(
                    item.get("matched_beam_mean_delta_recall_at_k", 0.0)
                ),
                "matched_beam_mean_delta_distance_computations": float(
                    item.get("matched_beam_mean_delta_distance_computations", 0.0)
                ),
                "matched_beam_mean_delta_hop_count": float(
                    item.get("matched_beam_mean_delta_hop_count", 0.0)
                ),
            }
        )
    return rows


def _resolved_candidate_source(candidate_source: str, node_count: int) -> str:
    candidate_source = str(candidate_source).strip().lower()
    if candidate_source == "auto":
        return "ivf" if int(node_count) > 50_000 else "exact"
    if candidate_source not in {"exact", "ivf"}:
        raise ValueError(f"Unsupported candidate_source: {candidate_source}")
    return candidate_source


def _resolved_graph_build_strategy(graph_build_strategy: str, candidate_source: str, node_count: int) -> str:
    graph_build_strategy = str(graph_build_strategy).strip().lower()
    if graph_build_strategy == "auto":
        return "fixed" if candidate_source == "ivf" or int(node_count) > 50_000 else "vamana"
    if graph_build_strategy not in {"vamana", "fixed"}:
        raise ValueError(f"Unsupported graph_build_strategy: {graph_build_strategy}")
    return graph_build_strategy


def _candidate_graph_inputs(
    *,
    vectors,
    neighbor_count: int,
    batch_size: int,
    candidate_source: str,
    ivf_nlist: int,
    ivf_nprobe: int,
    ivf_train_sample_size: int,
    ivf_batch_size: int,
    ivf_overfetch: int,
    seed: int,
    metric: str,
    existing_index_file: Path | None = None,
):
    candidate_source = _resolved_candidate_source(candidate_source, int(vectors.shape[0]))
    if candidate_source == "exact":
        candidate_ids, candidate_scores = _self_excluding_exact_knn(
            vectors=vectors,
            neighbor_count=neighbor_count,
            batch_size=batch_size,
            metric=metric,
        )
        return candidate_ids, candidate_scores, {
            "candidate_source": "exact",
        }

    index, ivf_metadata = _load_or_build_ivf_index(
        vectors=vectors,
        requested_nlist=ivf_nlist,
        requested_nprobe=ivf_nprobe,
        requested_train_sample_size=ivf_train_sample_size,
        batch_size=ivf_batch_size,
        seed=seed,
        metric=metric,
        existing_index_file=existing_index_file,
    )
    candidate_ids, candidate_scores = _self_excluding_ivf_knn(
        vectors=vectors,
        neighbor_count=neighbor_count,
        batch_size=batch_size,
        index=index,
        overfetch=ivf_overfetch,
        metric=metric,
    )
    return candidate_ids, candidate_scores, {
        "candidate_source": "ivf",
        **ivf_metadata,
    }


def _build_graph_from_candidates(
    *,
    vectors,
    candidate_ids,
    max_degree: int,
    alpha: float,
    graph_build_strategy: str,
    metric: str,
):
    if graph_build_strategy == "fixed":
        adjacency = _build_fixed_degree_graph(
            candidate_ids=candidate_ids,
            max_degree=max_degree,
        )
    elif graph_build_strategy == "vamana":
        adjacency = _build_vamana_style_graph(
            vectors=vectors,
            candidate_ids=candidate_ids,
            max_degree=max_degree,
            alpha=alpha,
            metric=metric,
        )
    else:
        raise ValueError(f"Unsupported graph_build_strategy: {graph_build_strategy}")
    return adjacency


def run_diskann_steiner_experiment(
    data_root: str | Path = "data",
    dataset_subdir: str = "glove-200-angular",
    train_size: int = 12_000,
    query_count: int = 1_000,
    hidden_count: int = 1_024,
    top_k: int = 10,
    max_degree: int = 32,
    candidate_pool: int = 96,
    beam_sizes_csv: str = "4,8,16,32,64,128,256",
    methods_csv: str = ",".join(_default_methods()),
    candidate_source: str = "auto",
    graph_build_strategy: str = "auto",
    ivf_nlist: int = 0,
    ivf_nprobe: int = 0,
    ivf_train_sample_size: int = 0,
    ivf_batch_size: int = 100_000,
    ivf_overfetch: int = 64,
    alpha: float = 1.2,
    exact_batch_size: int = 1_024,
    kmeans_niter: int = 20,
    noise_std: float = 0.002,
    validation_query_count: int = 250,
    validation_beam_size: int = 16,
    failure_recall_threshold: float = 0.5,
    local_mean_neighbor_count: int = 8,
    query_seed_candidate_count: int = 24,
    query_seed_top_m: int = 4,
    line_gap_fraction: float = 0.35,
    line_shift_scale: float = 0.0005,
    directional_directions_per_cluster: int = 2,
    directional_offset_scale: float = 0.08,
    shell_scale: float = 0.12,
    max_comp_fraction: float = 0.1,
    seed: int = 42,
    use_parlay: bool = False,
    parlay_binary: str = "",
    parlay_beam_width: int = 128,
    parlay_num_passes: int = 2,
    metric_override: str = "",
    no_normalize: bool = False,
) -> dict[str, object]:
    import faiss
    import numpy as np

    if train_size <= 0 or query_count <= 0 or hidden_count <= 0:
        raise ValueError("train_size, query_count, and hidden_count must be positive")
    if top_k <= 0 or max_degree <= 0 or candidate_pool <= 0:
        raise ValueError("top_k, max_degree, and candidate_pool must be positive")
    if exact_batch_size <= 0 or kmeans_niter <= 0:
        raise ValueError("exact_batch_size and kmeans_niter must be positive")
    if ivf_batch_size <= 0 or ivf_overfetch <= 0:
        raise ValueError("ivf_batch_size and ivf_overfetch must be positive")
    if noise_std <= 0.0:
        raise ValueError("noise_std must be positive")
    if alpha < 1.0:
        raise ValueError("alpha must be at least 1.0")
    if validation_query_count <= 0 or validation_beam_size <= 0:
        raise ValueError("validation_query_count and validation_beam_size must be positive")
    if not (0.0 <= failure_recall_threshold <= 1.0):
        raise ValueError("failure_recall_threshold must be in [0, 1]")
    if local_mean_neighbor_count <= 0:
        raise ValueError("local_mean_neighbor_count must be positive")
    if query_seed_candidate_count < 0 or query_seed_top_m < 0:
        raise ValueError("query_seed_candidate_count and query_seed_top_m must be non-negative")

    if beam_sizes_csv.strip():
        beam_sizes = sorted({int(item.strip()) for item in beam_sizes_csv.split(",") if item.strip()})
    else:
        beam_sizes = []
    # Auto-generate beam sizes: start at 4, double until distance computations
    # would exceed max_comp_fraction of train_size (checked at eval time).
    if not beam_sizes:
        beam_sizes = [4]
    b = max(beam_sizes)
    while b < train_size:
        b = int(b * 2) if b < 64 else int(b * 1.5)
        beam_sizes.append(b)
    beam_sizes = sorted(set(beam_sizes))
    if not beam_sizes or beam_sizes[0] <= 0:
        raise ValueError("Expected at least one positive beam size")

    requested_methods = [item.strip() for item in methods_csv.split(",") if item.strip()]
    registry = _method_registry()
    unknown_methods = sorted(set(requested_methods) - set(registry))
    if unknown_methods:
        raise ValueError(f"Unknown Steiner methods: {unknown_methods}")

    faiss.omp_set_num_threads(8)
    paths = dataset_paths(data_root, dataset_subdir)
    train_file = paths["train_file"]
    test_file = paths["test_file"]
    dataset_metadata_file = paths["dataset_metadata_file"]
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Missing dataset files for {dataset_subdir!r} under {Path(data_root)!s}"
        )

    dataset_metadata = {}
    if dataset_metadata_file.exists():
        dataset_metadata = json.loads(dataset_metadata_file.read_text())
    if metric_override and metric_override.strip():
        override = metric_override.strip().lower()
        if override in ("euclidean", "l2"):
            metric = "euclidean"
        elif override in ("cosine", "angular"):
            metric = "cosine"
        else:
            raise ValueError(f"Unknown metric_override: {metric_override!r}. Use 'euclidean' or 'cosine'.")
        print(f"  Metric override: using {metric!r} instead of dataset default")
    else:
        metric = _resolved_metric_name(dataset_subdir, dataset_metadata)

    train_embeddings = np.load(train_file, mmap_mode="r")
    test_embeddings = np.load(test_file, mmap_mode="r")
    full_train_count = int(train_embeddings.shape[0])
    if train_size > int(train_embeddings.shape[0]):
        raise ValueError(
            f"Requested train_size={train_size}, but only {train_embeddings.shape[0]} exist"
        )
    if query_count > int(test_embeddings.shape[0]):
        raise ValueError(
            f"Requested query_count={query_count}, but only {test_embeddings.shape[0]} exist"
        )
    if top_k > train_size:
        raise ValueError("top_k cannot exceed train_size")

    original_vectors, train_indices = _sample_rows(train_embeddings, train_size, seed)
    query_vectors, query_indices = _sample_rows(test_embeddings, query_count, seed + 1)
    original_vectors = _prepare_vectors_for_metric(original_vectors, metric=metric, skip_normalize=no_normalize)
    query_vectors = _prepare_vectors_for_metric(query_vectors, metric=metric, skip_normalize=no_normalize)

    full_dataset_mode = int(original_vectors.shape[0]) == full_train_count
    ground_truth_neighbors_file = paths["ground_truth_neighbors_file"]
    # Skip saved ground truth when metric_override is active, because saved
    # ground truth was computed under the original dataset metric, not the
    # overridden one.
    use_saved_gt = (full_dataset_mode
                    and ground_truth_neighbors_file.exists()
                    and not metric_override)
    if use_saved_gt:
        saved_ground_truth = np.load(ground_truth_neighbors_file, mmap_mode="r")
        ground_truth_ids = np.asarray(
            saved_ground_truth[query_indices, :top_k],
            dtype=np.int32,
        )
        ground_truth_source = "saved_full_dataset_ground_truth"
    else:
        _, ground_truth_ids = _batched_exact_search(
            base_vectors=original_vectors,
            query_vectors=query_vectors,
            top_k=top_k,
            batch_size=exact_batch_size,
            metric=metric,
        )
        ground_truth_ids = ground_truth_ids.astype(np.int32, copy=False)
        ground_truth_source = "brute_force_exact_search"

    baseline_entry_point = _pick_medoid_entry_point(original_vectors, metric=metric)
    baseline_max_degree, baseline_candidate_pool = _effective_graph_params(
        node_count=int(original_vectors.shape[0]),
        max_degree=max_degree,
        candidate_pool=candidate_pool,
    )

    if use_parlay:
        # ---- ParlayANN graph build (parallel C++) ----
        from hiddenbridge.parlay_backend import parlay_build_for_python_search
        import time as _time
        print(f"[ParlayANN] Building baseline graph ({original_vectors.shape[0]} points)...")
        _t0 = _time.monotonic()
        baseline_adjacency, _, parlay_build_time = parlay_build_for_python_search(
            original_vectors=original_vectors,
            hidden_vectors=None,
            max_degree=baseline_max_degree,
            beam_width=parlay_beam_width,
            alpha=alpha,
            num_passes=parlay_num_passes,
            parlay_binary=parlay_binary or None,
            metric=metric,
        )
        print(f"[ParlayANN] Baseline graph built in {_time.monotonic() - _t0:.2f}s "
              f"(ParlayANN reports {parlay_build_time:.2f}s)")
        # ParlayANN doesn't expose candidate IDs, but some Steiner builders
        # (bridge, local_knn_mean) need them. Derive from adjacency + exact kNN.
        baseline_candidate_ids, _ = _self_excluding_exact_knn(
            vectors=original_vectors,
            neighbor_count=baseline_candidate_pool,
            batch_size=exact_batch_size,
            metric=metric,
        )
        baseline_candidate_metadata = {"candidate_source": "parlay"}
        baseline_graph_strategy = "parlay_vamana"
    else:
        # ---- Original Python graph build ----
        baseline_candidate_ids, _, baseline_candidate_metadata = _candidate_graph_inputs(
            vectors=original_vectors,
            neighbor_count=baseline_candidate_pool,
            batch_size=exact_batch_size,
            candidate_source=candidate_source,
            ivf_nlist=ivf_nlist,
            ivf_nprobe=ivf_nprobe,
            ivf_train_sample_size=ivf_train_sample_size,
            ivf_batch_size=ivf_batch_size,
            ivf_overfetch=ivf_overfetch,
            seed=seed,
            metric=metric,
            existing_index_file=(
                paths["indices_dir"]
                / ("ivf_flat_l2.faiss" if metric == "euclidean" else "ivf_flat_ip.faiss")
            )
            if full_dataset_mode
            else None,
        )
        baseline_graph_strategy = _resolved_graph_build_strategy(
            graph_build_strategy=graph_build_strategy,
            candidate_source=baseline_candidate_metadata["candidate_source"],
            node_count=int(original_vectors.shape[0]),
        )
        baseline_adjacency = _build_graph_from_candidates(
            vectors=original_vectors,
            candidate_ids=baseline_candidate_ids,
            max_degree=baseline_max_degree,
            alpha=alpha,
            graph_build_strategy=baseline_graph_strategy,
            metric=metric,
        )
        if baseline_graph_strategy == "vamana":
            baseline_adjacency = _ensure_reachability_from_entry(
                adjacency=baseline_adjacency,
                candidate_ids=baseline_candidate_ids,
                vectors=original_vectors,
                entry_point=baseline_entry_point,
                metric=metric,
            )

    validation_signals = _collect_validation_signals(
        original_vectors=original_vectors,
        baseline_adjacency=baseline_adjacency,
        baseline_entry_points=[int(baseline_entry_point)],
        query_vectors=query_vectors,
        ground_truth_ids=ground_truth_ids,
        validation_query_count=validation_query_count,
        validation_beam_size=validation_beam_size,
        top_k=top_k,
        failure_recall_threshold=failure_recall_threshold,
        metric=metric,
    )

    build_kwargs = {
        "original_vectors": original_vectors,
        "baseline_adjacency": baseline_adjacency,
        "baseline_candidate_ids": baseline_candidate_ids,
        "hidden_count": hidden_count,
        "top_k": top_k,
        "kmeans_niter": kmeans_niter,
        "exact_batch_size": exact_batch_size,
        "noise_std": noise_std,
        "line_gap_fraction": line_gap_fraction,
        "line_shift_scale": line_shift_scale,
        "directional_directions_per_cluster": directional_directions_per_cluster,
        "directional_offset_scale": directional_offset_scale,
        "shell_scale": shell_scale,
        "validation_signals": validation_signals,
        "local_mean_neighbor_count": local_mean_neighbor_count,
        "query_seed_candidate_count": query_seed_candidate_count,
        "query_seed_top_m": query_seed_top_m,
        "metric": metric,
    }

    # When --no-normalize is set, temporarily override _prepare_vectors_for_metric
    # so Steiner builders also skip normalization
    if no_normalize:
        _original_prepare = _prepare_vectors_for_metric
        def _patched_prepare(matrix, metric: str, fallback_vector=None, skip_normalize: bool = False):
            return _original_prepare(matrix, metric=metric, fallback_vector=fallback_vector, skip_normalize=True)
        import hiddenbridge.experiment as _self_module
        _self_module._prepare_vectors_for_metric = _patched_prepare

    method_specs: dict[str, dict[str, object]] = {}
    for method_offset, method_name in enumerate(requested_methods):
        builder = registry[method_name]
        method_specs[method_name] = builder(
            **build_kwargs,
            seed=seed + (method_offset * 101),
        )

    # Restore original _prepare_vectors_for_metric if patched
    if no_normalize:
        _self_module._prepare_vectors_for_metric = _original_prepare

    results = {
        "baseline": _evaluate_graph(
            method_name="baseline",
            vectors=original_vectors,
            adjacency=baseline_adjacency,
            query_vectors=query_vectors,
            ground_truth_ids=ground_truth_ids,
            beam_sizes=beam_sizes,
            visible_count=int(original_vectors.shape[0]),
            entry_points=[int(baseline_entry_point)],
            top_k=top_k,
            metric=metric,
            max_comp_fraction=max_comp_fraction,
        )
    }

    steiner_metadata: dict[str, object] = {}
    graph_entry_points: dict[str, list[int]] = {"baseline": [int(baseline_entry_point)]}
    effective_params_by_method: dict[str, dict[str, int]] = {
        "baseline": {
            "max_degree": int(baseline_max_degree),
            "candidate_pool": int(baseline_candidate_pool),
            "node_count": int(original_vectors.shape[0]),
        }
    }
    candidate_generation_by_method: dict[str, dict[str, object]] = {
        "baseline": {
            **baseline_candidate_metadata,
            "graph_build_strategy": baseline_graph_strategy,
        }
    }

    for method_name in requested_methods:
        hidden_vectors = method_specs[method_name]["hidden_vectors"]
        # Always append Steiner points after originals; ParlayANN shuffles internally.
        augmented_vectors = np.vstack([original_vectors, hidden_vectors]).astype(np.float32, copy=False)
        visible_count = int(original_vectors.shape[0])

        # Every Steiner variant reuses the same graph builder and search loop for a clean comparison.
        method_max_degree, method_candidate_pool = _effective_graph_params(
            node_count=int(augmented_vectors.shape[0]),
            max_degree=max_degree,
            candidate_pool=candidate_pool,
        )

        if use_parlay:
            # ---- ParlayANN graph build (parallel C++) ----
            from hiddenbridge.parlay_backend import build_vamana_graph as _parlay_build
            import time as _time
            print(f"[ParlayANN] Building {method_name} graph ({augmented_vectors.shape[0]} points)...")
            _t0 = _time.monotonic()
            adjacency, parlay_build_time = _parlay_build(
                augmented_vectors,
                max_degree=method_max_degree,
                beam_width=parlay_beam_width,
                alpha=alpha,
                num_passes=parlay_num_passes,
                parlay_binary=parlay_binary or None,
                metric=metric,
            )
            print(f"[ParlayANN] {method_name} graph built in {_time.monotonic() - _t0:.2f}s "
                  f"(ParlayANN reports {parlay_build_time:.2f}s)")
            candidate_ids = None
            method_candidate_metadata = {"candidate_source": "parlay"}
            method_graph_strategy = "parlay_vamana"
        else:
            # ---- Original Python graph build ----
            candidate_ids, _, method_candidate_metadata = _candidate_graph_inputs(
                vectors=augmented_vectors,
                neighbor_count=method_candidate_pool,
                batch_size=exact_batch_size,
                candidate_source=candidate_source,
                ivf_nlist=ivf_nlist,
                ivf_nprobe=ivf_nprobe,
                ivf_train_sample_size=ivf_train_sample_size,
                ivf_batch_size=ivf_batch_size,
                ivf_overfetch=ivf_overfetch,
                seed=seed + 10_000 + len(method_name),
                metric=metric,
            )
            method_graph_strategy = _resolved_graph_build_strategy(
                graph_build_strategy=graph_build_strategy,
                candidate_source=method_candidate_metadata["candidate_source"],
                node_count=int(augmented_vectors.shape[0]),
            )
            adjacency = _build_graph_from_candidates(
                vectors=augmented_vectors,
                candidate_ids=candidate_ids,
                max_degree=method_max_degree,
                alpha=alpha,
                graph_build_strategy=method_graph_strategy,
                metric=metric,
            )

        # Map entry points and seed offsets through the hidden_to_aug mapping.
        # Entry points and query seeds: Steiner offsets are relative to hidden_vectors,
        # so augmented index = visible_count + offset (since we always append).
        entry_hidden_offsets = method_specs[method_name].get("entry_hidden_offsets", [])
        if entry_hidden_offsets:
            entry_points = [
                int(visible_count + int(offset)) for offset in entry_hidden_offsets
            ]
        else:
            entry_points = [int(_pick_medoid_entry_point(augmented_vectors, metric=metric))]
        query_seed_offsets = method_specs[method_name].get("query_seed_hidden_offsets", [])
        query_seed_node_ids = [
            int(visible_count + int(offset)) for offset in query_seed_offsets
        ]
        method_query_seed_top_m = int(method_specs[method_name].get("query_seed_top_m", 0))
        if method_graph_strategy == "vamana" and candidate_ids is not None:
            adjacency = _ensure_reachability_from_entry(
                adjacency=adjacency,
                candidate_ids=candidate_ids,
                vectors=augmented_vectors,
                entry_point=int(entry_points[0]),
                metric=metric,
            )

        results[method_name] = _evaluate_graph(
            method_name=method_name,
            vectors=augmented_vectors,
            adjacency=adjacency,
            query_vectors=query_vectors,
            ground_truth_ids=ground_truth_ids,
            beam_sizes=beam_sizes,
            visible_count=visible_count,
            entry_points=entry_points,
            top_k=top_k,
            metric=metric,
            query_seed_node_ids=query_seed_node_ids,
            query_seed_top_m=method_query_seed_top_m,
            max_comp_fraction=max_comp_fraction,
        )
        graph_entry_points[method_name] = [int(item) for item in entry_points]
        effective_params_by_method[method_name] = {
            "max_degree": int(method_max_degree),
            "candidate_pool": int(method_candidate_pool),
            "node_count": int(augmented_vectors.shape[0]),
        }
        candidate_generation_by_method[method_name] = {
            **method_candidate_metadata,
            "graph_build_strategy": method_graph_strategy,
        }
        steiner_metadata[method_name] = {
            "description": method_specs[method_name]["description"],
            "actual_hidden_count": int(hidden_vectors.shape[0]),
            "query_seed_candidate_count": int(len(query_seed_node_ids)),
            "query_seed_top_m": int(method_query_seed_top_m),
            **method_specs[method_name]["metadata"],
        }

    output_paths = _result_paths(
        data_root=data_root,
        dataset_subdir=dataset_subdir,
        train_size=int(original_vectors.shape[0]),
        query_count=int(query_vectors.shape[0]),
        hidden_count=hidden_count,
        max_degree=max_degree,
    )

    payload = {
        "status": "completed",
        "dataset": {
            "name": dataset_subdir,
            "metric": _metric_display_name(metric),
            "train_size": int(original_vectors.shape[0]),
            "query_count": int(query_vectors.shape[0]),
            "dimension": int(original_vectors.shape[1]),
            "sampled_train_indices_min": int(train_indices.min()),
            "sampled_train_indices_max": int(train_indices.max()),
            "sampled_query_indices_min": int(query_indices.min()),
            "sampled_query_indices_max": int(query_indices.max()),
            "source_metadata": dataset_metadata,
        },
        "graph": {
            "style": ("ParlayANN parallel Vamana graph" if use_parlay
                      else "Simplified Vamana/DiskANN-style fixed-degree proximity graph"),
            "requested_max_degree": int(max_degree),
            "requested_candidate_pool": int(candidate_pool),
            "alpha": float(alpha),
            "effective_params_by_method": effective_params_by_method,
            "candidate_generation_by_method": candidate_generation_by_method,
            "entry_points": graph_entry_points,
        },
        "search": {
            "algorithm": "beam search over graph neighbors",
            "beam_sizes": [int(item) for item in beam_sizes],
            "top_k": int(top_k),
            "ground_truth_source": ground_truth_source,
            "distance_computation_definition": (
                "One exact query-to-node score evaluation for each unique visited node, "
                "including hidden Steiner nodes"
            ),
            "validation_query_count": int(validation_signals["query_count"]),
            "validation_beam_size": int(validation_signals["beam_size"]),
            "failure_recall_threshold": float(validation_signals["failure_recall_threshold"]),
            "query_seed_candidate_count": int(query_seed_candidate_count),
            "query_seed_top_m": int(query_seed_top_m),
        },
        "steiner": {
            "requested_hidden_count": int(hidden_count),
            "methods": steiner_metadata,
        },
        "results": results,
        "validation": {
            "failed_query_count": int(validation_signals["failed_query_count"]),
        },
        "artifacts": {
            "metrics_file": str(output_paths["metrics_file"]),
            "plot_file": str(output_paths["plot_file"]),
            "curve_table_file": str(output_paths["curve_table_file"]),
            "method_summary_file": str(output_paths["method_summary_file"]),
        },
    }
    payload["summary"] = _summarize_method_results(payload["results"])

    curve_rows = _build_curve_table_rows(payload)
    method_rows = _build_method_summary_rows(payload)
    _write_json(output_paths["metrics_file"], payload)
    _write_csv(
        output_paths["curve_table_file"],
        fieldnames=[
            "dataset",
            "variant_id",
            "method",
            "actual_hidden_count",
            "beam_size",
            "avg_distance_computations",
            "avg_visible_distance_computations",
            "avg_hidden_distance_computations",
            "avg_hop_count",
            "recall_at_1",
            "recall_at_10",
            "recall_at_k",
            "true_top1_hit_in_topk",
        ],
        rows=curve_rows,
    )
    _write_csv(
        output_paths["method_summary_file"],
        fieldnames=[
            "dataset",
            "method",
            "actual_hidden_count",
            "recall_at_1_auc",
            "recall_at_k_auc",
            "best_variant_id",
            "best_variant_beam_size",
            "best_variant_recall_at_1",
            "best_variant_recall_at_k",
            "best_variant_avg_distance_computations",
            "matched_beam_mean_delta_recall_at_1",
            "matched_beam_mean_delta_recall_at_k",
            "matched_beam_mean_delta_distance_computations",
            "matched_beam_mean_delta_hop_count",
        ],
        rows=method_rows,
    )
    _save_tradeoff_plot(payload=payload, plot_path=output_paths["plot_file"])
    return payload


def main(
    data_root: str | Path = "data",
    dataset: str = "glove-200",
    dataset_subdir: str | None = None,
    train_size: int = 12_000,
    query_count: int = 1_000,
    hidden_count: int = 1_024,
    top_k: int = 10,
    max_degree: int = 32,
    candidate_pool: int = 96,
    beam_sizes_csv: str = "4,8,16,32,64,128,256",
    methods_csv: str = ",".join(_default_methods()),
    candidate_source: str = "auto",
    graph_build_strategy: str = "auto",
    ivf_nlist: int = 0,
    ivf_nprobe: int = 0,
    ivf_train_sample_size: int = 0,
    ivf_batch_size: int = 100_000,
    ivf_overfetch: int = 64,
    alpha: float = 1.2,
    exact_batch_size: int = 1_024,
    kmeans_niter: int = 20,
    noise_std: float = 0.002,
    validation_query_count: int = 250,
    validation_beam_size: int = 16,
    failure_recall_threshold: float = 0.5,
    local_mean_neighbor_count: int = 8,
    query_seed_candidate_count: int = 24,
    query_seed_top_m: int = 4,
    line_gap_fraction: float = 0.35,
    line_shift_scale: float = 0.0005,
    directional_directions_per_cluster: int = 2,
    directional_offset_scale: float = 0.08,
    shell_scale: float = 0.12,
    max_comp_fraction: float = 0.1,
    seed: int = 42,
    use_parlay: bool = False,
    parlay_binary: str = "",
    parlay_beam_width: int = 128,
    parlay_num_passes: int = 2,
    metric_override: str = "",
    no_normalize: bool = False,
) -> None:
    # Resolve dataset name to subdir and auto-download if needed
    registry = _dataset_registry()
    if dataset_subdir is None:
        if dataset not in registry:
            raise ValueError(
                f"Unknown dataset {dataset!r}. Known: {sorted(registry.keys())}"
            )
        spec = registry[dataset]
        dataset_subdir = spec["dataset_subdir"]
    else:
        # Manual override — try to find spec for auto-download
        spec = None
        for s in registry.values():
            if s["dataset_subdir"] == dataset_subdir:
                spec = s
                break

    # Auto-download if dataset files are missing
    paths = dataset_paths(data_root, dataset_subdir)
    if not paths["train_file"].exists() or not paths["test_file"].exists():
        if spec is None:
            raise FileNotFoundError(
                f"Dataset files missing for {dataset_subdir!r} and no download spec found"
            )
        fmt = spec.get("format", "hdf5")
        if fmt == "bigann":
            print(f"Dataset {dataset_subdir!r} not found locally. Downloading from big-ann-benchmarks...")
            print(f"  WARNING: These are large files (1-8 GB). Download may take a while.")
            from hiddenbridge.datasets import download_big_ann_dataset
            download_big_ann_dataset(
                data_root=data_root,
                dataset_subdir=spec["dataset_subdir"],
                base_url=spec["base_url"],
                query_url=spec["query_url"],
                embedding_dim=spec["embedding_dim"],
                metric=spec["metric"],
                dtype=spec.get("dtype", "float32"),
                normalize=spec["normalize"],
            )
        elif fmt == "huggingface":
            print(f"Dataset {dataset_subdir!r} not found locally. Downloading from HuggingFace...")
            from hiddenbridge.datasets import download_huggingface_dataset
            download_huggingface_dataset(
                data_root=data_root,
                dataset_subdir=spec["dataset_subdir"],
                hf_repo=spec["hf_repo"],
                embedding_column=spec["embedding_column"],
                embedding_dim=spec["embedding_dim"],
                metric=spec["metric"],
                normalize=spec["normalize"],
            )
        else:
            print(f"Dataset {dataset_subdir!r} not found locally. Downloading from ann-benchmarks.com...")
            from hiddenbridge.datasets import download_ann_benchmark_dataset
            download_ann_benchmark_dataset(
                data_root=data_root,
                dataset_subdir=spec["dataset_subdir"],
                dataset_url=spec["dataset_url"],
                embedding_dim=spec["embedding_dim"],
                metric=spec["metric"],
                normalize=spec["normalize"],
            )
        print(f"Download complete: {dataset_subdir}")

    result = run_diskann_steiner_experiment(
        data_root=data_root,
        dataset_subdir=dataset_subdir,
        train_size=train_size,
        query_count=query_count,
        hidden_count=hidden_count,
        top_k=top_k,
        max_degree=max_degree,
        candidate_pool=candidate_pool,
        beam_sizes_csv=beam_sizes_csv,
        methods_csv=methods_csv,
        candidate_source=candidate_source,
        graph_build_strategy=graph_build_strategy,
        ivf_nlist=ivf_nlist,
        ivf_nprobe=ivf_nprobe,
        ivf_train_sample_size=ivf_train_sample_size,
        ivf_batch_size=ivf_batch_size,
        ivf_overfetch=ivf_overfetch,
        alpha=alpha,
        exact_batch_size=exact_batch_size,
        kmeans_niter=kmeans_niter,
        noise_std=noise_std,
        validation_query_count=validation_query_count,
        validation_beam_size=validation_beam_size,
        failure_recall_threshold=failure_recall_threshold,
        local_mean_neighbor_count=local_mean_neighbor_count,
        query_seed_candidate_count=query_seed_candidate_count,
        query_seed_top_m=query_seed_top_m,
        line_gap_fraction=line_gap_fraction,
        line_shift_scale=line_shift_scale,
        directional_directions_per_cluster=directional_directions_per_cluster,
        directional_offset_scale=directional_offset_scale,
        shell_scale=shell_scale,
        max_comp_fraction=max_comp_fraction,
        seed=seed,
        use_parlay=use_parlay,
        parlay_binary=parlay_binary,
        parlay_beam_width=parlay_beam_width,
        parlay_num_passes=parlay_num_passes,
        metric_override=metric_override,
        no_normalize=no_normalize,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _dataset_registry():
    """Known datasets from ann-benchmarks.com. Keys are short names for --dataset."""
    return {
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
            "embedding_dim": 65,
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
        # big-ann-benchmarks (NeurIPS'23) — binary format, large downloads
        "yfcc": {
            "dataset_subdir": "yfcc-10M-u8-192-l2",
            "base_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/base.10M.u8bin",
            "query_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/query.public.100K.u8bin",
            "embedding_dim": 192,
            "metric": "euclidean",
            "dtype": "uint8",
            "normalize": False,
            "format": "bigann",
        },
        "yfcc-angular": {
            "dataset_subdir": "yfcc-192-angular",
            "embedding_dim": 192,
            "metric": "angular",
            "normalize": True,
        },
        "text2image": {
            "dataset_subdir": "text2image-10M-f32-200-ip",
            "base_url": "https://storage.yandexcloud.net/yr-secret-share/ann-datasets/T2I/base.10M.fbin",
            "query_url": "https://storage.yandexcloud.net/yr-secret-share/ann-datasets/T2I/query.public.100K.fbin",
            "embedding_dim": 200,
            "metric": "angular",
            "dtype": "float32",
            "normalize": True,
            "format": "bigann",
        },
        "msturing": {
            "dataset_subdir": "msturing-10M-f32-100-l2",
            "base_url": "https://comp21storage.blob.core.windows.net/publiccontainer/comp21/MSFT-TURING-ANNS/base1b.fbin.crop_nb_10000000",
            "query_url": "https://comp21storage.blob.core.windows.net/publiccontainer/comp21/MSFT-TURING-ANNS/query100K.fbin",
            "embedding_dim": 100,
            "metric": "euclidean",
            "dtype": "float32",
            "normalize": False,
            "format": "bigann",
        },
        # HuggingFace datasets
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


def parse_args() -> argparse.Namespace:
    registry = _dataset_registry()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="glove-200",
                        choices=sorted(registry.keys()),
                        help="Dataset short name (auto-downloads from ann-benchmarks.com if not cached)")
    parser.add_argument("--dataset-subdir", default=None,
                        help="Override dataset subdirectory (advanced; usually set automatically by --dataset)")
    parser.add_argument("--train-size", type=int, default=12_000)
    parser.add_argument("--query-count", type=int, default=1_000)
    parser.add_argument("--hidden-count", type=int, default=1_024)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-degree", type=int, default=32)
    parser.add_argument("--candidate-pool", type=int, default=96)
    parser.add_argument("--beam-sizes-csv", default="4",
                        help="Initial beam sizes (auto-extended until 10%% of dataset is reached)")
    parser.add_argument("--methods-csv", default=",".join(_default_methods()))
    parser.add_argument("--candidate-source", default="auto")
    parser.add_argument("--graph-build-strategy", default="auto")
    parser.add_argument("--ivf-nlist", type=int, default=0)
    parser.add_argument("--ivf-nprobe", type=int, default=0)
    parser.add_argument("--ivf-train-sample-size", type=int, default=0)
    parser.add_argument("--ivf-batch-size", type=int, default=100_000)
    parser.add_argument("--ivf-overfetch", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--exact-batch-size", type=int, default=1_024)
    parser.add_argument("--kmeans-niter", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.002)
    parser.add_argument("--validation-query-count", type=int, default=250)
    parser.add_argument("--validation-beam-size", type=int, default=16)
    parser.add_argument("--failure-recall-threshold", type=float, default=0.5)
    parser.add_argument("--local-mean-neighbor-count", type=int, default=8)
    parser.add_argument("--query-seed-candidate-count", type=int, default=24)
    parser.add_argument("--query-seed-top-m", type=int, default=4)
    parser.add_argument("--line-gap-fraction", type=float, default=0.35)
    parser.add_argument("--line-shift-scale", type=float, default=0.0005)
    parser.add_argument("--directional-directions-per-cluster", type=int, default=2)
    parser.add_argument("--directional-offset-scale", type=float, default=0.08)
    parser.add_argument("--shell-scale", type=float, default=0.12)
    parser.add_argument("--max-comp-fraction", type=float, default=0.1,
                        help="Stop increasing beam width when avg distance computations exceed "
                             "this fraction of train_size (e.g., 0.1 for 10%%). Default 0.1.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-parlay", action="store_true", default=False,
                        help="Use ParlayANN C++ for graph build (much faster, parallel)")
    parser.add_argument("--parlay-binary", default="",
                        help="Path to ParlayANN neighbors binary")
    parser.add_argument("--parlay-beam-width", type=int, default=128,
                        help="Beam width for ParlayANN graph build")
    parser.add_argument("--parlay-num-passes", type=int, default=2,
                        help="Number of build passes for ParlayANN")
    parser.add_argument("--metric-override", default="",
                        choices=["", "euclidean", "cosine"],
                        help="Override the dataset's default distance metric (e.g., run GloVe with euclidean)")
    parser.add_argument("--no-normalize", action="store_true", default=False,
                        help="Skip L2 normalization even for cosine/angular datasets")
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
