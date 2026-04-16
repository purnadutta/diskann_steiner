"""
ParlayANN backend for HiddenBridge.

Replaces the pure-Python Vamana graph build and beam search with
ParlayANN's parallel C++ implementation via subprocess calls to the
pre-built `neighbors` binary.

Data flow:
  1. Python generates augmented vectors (base + Steiner) as numpy
  2. This module writes them to .fbin files
  3. Calls ParlayANN neighbors binary to build the Vamana graph
  4. Reads the resulting graph file back as Python adjacency lists
  5. For search: calls neighbors binary which does parallel beam search
     and outputs recall/QPS CSVs, OR loads the graph and does search
     in Python (for per-query distance-computation counting)

The module also supports a "graph-only" mode where ParlayANN builds
the graph but search is still done in Python (to preserve the exact
distance-computation counting that HiddenBridge needs for its plots).
"""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# File format helpers (fbin = DiskANN/ParlayANN binary vector format)
# ---------------------------------------------------------------------------

def write_fbin(path: str | Path, vectors: np.ndarray) -> None:
    """Write vectors to .fbin format: [uint32 n][uint32 d][float32 data...]."""
    vectors = np.asarray(vectors, dtype=np.float32)
    assert vectors.ndim == 2
    n, d = vectors.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, d))
        f.write(vectors.tobytes())


def read_fbin(path: str | Path) -> np.ndarray:
    """Read vectors from .fbin format."""
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(n * d * 4), dtype=np.float32)
    return data.reshape(n, d)


def write_groundtruth_bin(path: str | Path, ids: np.ndarray) -> None:
    """Write ground truth in ParlayANN format: [uint32 n][uint32 k][uint32 ids...]."""
    ids = np.asarray(ids, dtype=np.uint32)
    assert ids.ndim == 2
    n, k = ids.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, k))
        f.write(ids.tobytes())


def read_parlay_graph(path: str | Path) -> list[np.ndarray]:
    """
    Read a ParlayANN graph binary file.

    Format:
      [uint32 n][uint32 max_degree]
      [uint32 degree_0][uint32 degree_1]...[uint32 degree_{n-1}]
      [uint32 edges_0_0]...[uint32 edges_0_{deg0-1}]
      [uint32 edges_1_0]...[uint32 edges_1_{deg1-1}]
      ...
    """
    with open(path, "rb") as f:
        n, max_deg = struct.unpack("<II", f.read(8))
        degrees = np.frombuffer(f.read(n * 4), dtype=np.uint32).copy()
        adjacency = []
        for i in range(n):
            deg = int(degrees[i])
            if deg > 0:
                neighbors = np.frombuffer(f.read(deg * 4), dtype=np.uint32).copy()
                adjacency.append(neighbors.astype(np.int32))
            else:
                adjacency.append(np.empty(0, dtype=np.int32))
    return adjacency


# ---------------------------------------------------------------------------
# ParlayANN binary discovery
# ---------------------------------------------------------------------------

def find_parlay_binary(search_paths: list[str | Path] | None = None) -> Path | None:
    """Find the ParlayANN neighbors binary."""
    candidates = []
    if search_paths:
        candidates.extend([Path(p) for p in search_paths])

    # Common locations relative to steiner_diskANN repo
    repo_root = Path(__file__).resolve().parent.parent
    candidates.extend([
        repo_root.parent / "ParlayANN" / "algorithms" / "vamana" / "neighbors",
        repo_root / "ParlayANN" / "algorithms" / "vamana" / "neighbors",
        Path.home() / "Desktop" / "research" / "synthesis" / "ParlayANN" / "algorithms" / "vamana" / "neighbors",
    ])

    for candidate in candidates:
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Graph build via ParlayANN
# ---------------------------------------------------------------------------

def build_vamana_graph(
    vectors: np.ndarray,
    *,
    max_degree: int = 32,
    beam_width: int = 128,
    alpha: float = 1.2,
    num_passes: int = 2,
    parlay_binary: str | Path | None = None,
    work_dir: str | Path | None = None,
    metric: str = "cosine",
) -> tuple[list[np.ndarray], float]:
    """
    Build a Vamana graph using ParlayANN's C++ implementation.

    Parameters
    ----------
    vectors : np.ndarray
        (n, d) float32 array of all vectors (base + Steiner).
    max_degree : int
        Maximum out-degree (R parameter).
    beam_width : int
        Build beam width (L parameter).
    alpha : float
        Pruning parameter.
    num_passes : int
        Number of build passes.
    parlay_binary : path, optional
        Path to the ParlayANN neighbors binary.
    work_dir : path, optional
        Directory for temporary files. If None, uses a temp dir.
    metric : str
        "cosine" or "euclidean".

    Returns
    -------
    adjacency : list[np.ndarray]
        adjacency[i] is an int32 array of neighbor indices for node i.
    build_time : float
        Wall-clock build time in seconds.
    """
    binary = Path(parlay_binary) if parlay_binary else find_parlay_binary()
    if binary is None or not binary.exists():
        raise FileNotFoundError(
            "ParlayANN neighbors binary not found. "
            "Set parlay_binary= or ensure ParlayANN is built at ../ParlayANN/"
        )

    vectors = np.asarray(vectors, dtype=np.float32)
    n, d = vectors.shape

    cleanup = False
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="parlay_build_")
        cleanup = True
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    data_file = work_dir / "vectors.fbin"
    graph_file = work_dir / "graph.bin"
    # We need a dummy query file and groundtruth for the binary to run
    dummy_query_file = work_dir / "dummy_queries.fbin"
    dummy_gt_file = work_dir / "dummy_gt"

    write_fbin(data_file, vectors)
    # Write a single dummy query (the binary requires it)
    write_fbin(dummy_query_file, vectors[:1])
    # Write dummy groundtruth
    write_groundtruth_bin(dummy_gt_file, np.zeros((1, 1), dtype=np.uint32))

    # For cosine distance on normalized vectors, use Euclidean in ParlayANN.
    # On l2-normalized vectors: ||a-b||^2 = 2(1 - a·b), so Euclidean
    # distance preserves the same nearest-neighbor ordering as cosine.
    # This avoids MIPS-specific pruning quirks in ParlayANN's Vamana build
    # that can produce lower-quality graphs for our use case.
    dist_func = "Euclidian"

    cmd = [
        str(binary),
        "-R", str(max_degree),
        "-L", str(beam_width),
        "-alpha", str(alpha),
        "-num_passes", str(num_passes),
        "-k", "1",
        "-data_type", "float",
        "-dist_func", dist_func,
        "-base_path", str(data_file),
        "-query_path", str(dummy_query_file),
        "-gt_path", str(dummy_gt_file),
        "-res_path", str(work_dir / "dummy_res.csv"),
        "-graph_outfile", str(graph_file),
    ]

    t0 = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    build_time = time.monotonic() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"ParlayANN build failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )

    if not graph_file.exists():
        raise FileNotFoundError(
            f"ParlayANN did not produce graph file at {graph_file}\n"
            f"STDOUT:\n{result.stdout[-2000:]}"
        )

    adjacency = read_parlay_graph(graph_file)

    # Parse actual build time from ParlayANN output if available
    for line in result.stdout.split("\n"):
        if "Graph built in" in line:
            try:
                build_time = float(line.split("built in")[1].split("seconds")[0].strip())
            except (ValueError, IndexError):
                pass

    return adjacency, build_time


# ---------------------------------------------------------------------------
# Batch beam search via ParlayANN (for QPS benchmarking)
# ---------------------------------------------------------------------------

def parlay_batch_search(
    vectors: np.ndarray,
    queries: np.ndarray,
    graph_file: str | Path,
    ground_truth_ids: np.ndarray,
    *,
    k: int = 10,
    n_base: int = 0,
    parlay_binary: str | Path | None = None,
    work_dir: str | Path | None = None,
    metric: str = "cosine",
) -> dict:
    """
    Run ParlayANN's parallel beam search for QPS benchmarking.

    This uses the full C++ search pipeline with multiple beam widths.
    Returns raw QPS numbers (not per-query distance computation counts).

    For the recall-vs-distance-computations tradeoff plots that
    HiddenBridge needs, use build_vamana_graph() + Python beam search.
    """
    binary = Path(parlay_binary) if parlay_binary else find_parlay_binary()
    if binary is None:
        raise FileNotFoundError("ParlayANN neighbors binary not found")

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="parlay_search_")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    data_file = work_dir / "vectors.fbin"
    query_file = work_dir / "queries.fbin"
    gt_file = work_dir / "groundtruth"
    res_file = work_dir / "results.csv"

    write_fbin(data_file, np.asarray(vectors, dtype=np.float32))
    write_fbin(query_file, np.asarray(queries, dtype=np.float32))
    write_groundtruth_bin(gt_file, np.asarray(ground_truth_ids, dtype=np.uint32))

    # Use Euclidean for cosine on normalized vectors (see build_vamana_graph comment)
    dist_func = "Euclidian"

    cmd = [
        str(binary),
        "-k", str(k),
        "-data_type", "float",
        "-dist_func", dist_func,
        "-base_path", str(data_file),
        "-query_path", str(query_file),
        "-gt_path", str(gt_file),
        "-res_path", str(res_file),
        "-graph_outfile", str(work_dir / "graph.bin"),
        "-n_base", str(n_base),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ParlayANN search failed:\n{result.stderr[-2000:]}")

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "results_file": str(res_file),
    }


# ---------------------------------------------------------------------------
# High-level: build graph with ParlayANN, return adjacency for Python search
# ---------------------------------------------------------------------------

def parlay_build_for_python_search(
    original_vectors: np.ndarray,
    hidden_vectors: np.ndarray | None = None,
    *,
    max_degree: int = 32,
    beam_width: int = 128,
    alpha: float = 1.2,
    num_passes: int = 2,
    parlay_binary: str | Path | None = None,
    work_dir: str | Path | None = None,
    metric: str = "cosine",
) -> tuple[list[np.ndarray], int, float]:
    """
    Build graph with ParlayANN, return adjacency for use with Python beam search.

    This is the primary integration point: ParlayANN builds the graph fast
    (parallel Vamana), then HiddenBridge's Python beam search evaluates it
    with exact distance-computation counting.

    Parameters
    ----------
    original_vectors : np.ndarray
        (n_base, d) base dataset vectors.
    hidden_vectors : np.ndarray or None
        (n_hidden, d) Steiner point vectors. If None, builds baseline.
    max_degree, beam_width, alpha, num_passes : graph build params
    parlay_binary : path to neighbors binary
    work_dir : scratch directory
    metric : "cosine" or "euclidean"

    Returns
    -------
    adjacency : list[np.ndarray]
        Full graph adjacency (n_base + n_hidden nodes).
    visible_count : int
        Number of real (non-Steiner) nodes = n_base.
    build_time : float
        Graph build time in seconds.
    """
    original_vectors = np.asarray(original_vectors, dtype=np.float32)
    n_base = original_vectors.shape[0]

    if hidden_vectors is not None and len(hidden_vectors) > 0:
        hidden_vectors = np.asarray(hidden_vectors, dtype=np.float32)
        all_vectors = np.concatenate([original_vectors, hidden_vectors], axis=0)
    else:
        all_vectors = original_vectors

    adjacency, build_time = build_vamana_graph(
        all_vectors,
        max_degree=max_degree,
        beam_width=beam_width,
        alpha=alpha,
        num_passes=num_passes,
        parlay_binary=parlay_binary,
        work_dir=work_dir,
        metric=metric,
    )

    return adjacency, n_base, build_time
