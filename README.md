# HiddenBridge

HiddenBridge is a standalone research codebase for studying whether routing-only Steiner points can improve DiskANN-style graph nearest-neighbor search.

This repo focuses on:

- building a simplified Vamana/DiskANN-style fixed-degree proximity graph
- running beam-search graph traversal on top of that graph
- evaluating `recall@k`, average distance computations, and hop count
- comparing multiple Steiner-point augmentation strategies
- plotting recall-vs-compute tradeoffs and Steiner-budget ablations

Steiner points are used only for graph construction and graph navigation. They are never returned as final nearest-neighbor results.

## What Is Included

- core DiskANN-style graph build and beam-search evaluation
- 11 Steiner construction methods
- plotting for per-run tradeoff curves
- multi-run comparison plotting
- hidden-count ablation plotting
- local dataset download helpers for ANN-Benchmarks GloVe and SIFT
- local dataset registration helpers for precomputed array datasets, including OpenAI `1536` and `3072` embeddings

This repo intentionally does **not** include Modal wrappers, Modal configs, or downloaded dataset/artifact files.

## Steiner Methods

Implemented methods:

- `pairwise_interpolation`
  Midpoints or convex-combination bridge points between selected original pairs.
- `cluster_centroid`
  K-means centroids used as routing-only nodes.
- `random_line`
  Steiner points laid out along a random projected line through the dataset.
- `random_line_anchor`
  Random-line Steiner points plus a synthetic global anchor start node.
- `noisy_copy`
  Small-noise replicas around selected original points.
- `bridge`
  Bridge points between geometrically close but weakly connected regions.
- `hierarchical_centroid`
  Coarse and fine centroid levels that act as a routing hierarchy.
- `directional_centroid`
  Cluster centroids plus centroid-offset points along principal directions.
- `boundary_shell`
  Slightly outside-cluster shell points used as gateways.
- `failure_driven`
  Steiner points placed using baseline search failure traces.
- `targeted_noisy_replicas`
  Noisy replicas around hubs, bottlenecks, and frequently visited nodes.

## Installation

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`faiss-cpu` is strongly recommended for anything beyond toy runs.

## Dataset Layout

HiddenBridge expects each dataset under:

```text
data/<dataset_subdir>/
  train.npy
  test.npy
  dataset_metadata.json
  ground_truth_neighbors.npy          # optional but used when available
  ground_truth_distances.npy          # optional
  indices/                            # optional cached IVF indices
  analysis/graph_navigation/          # output plots/json/csv
```

## Download ANN-Benchmarks Datasets

Download full GloVe-200:

```bash
python -m hiddenbridge.download_ann_benchmark \
  --dataset glove \
  --data-root data
```

Download full SIFT-128:

```bash
python -m hiddenbridge.download_ann_benchmark \
  --dataset sift \
  --data-root data
```

## Register Existing Arrays

For datasets you already have as `train.npy` and `test.npy`, such as OpenAI embedding exports:

```bash
python -m hiddenbridge.register_dataset \
  --data-root data \
  --dataset-subdir dbpedia-openai3-large-3072-n101000-test1000 \
  --train-file /path/to/train.npy \
  --test-file /path/to/test.npy \
  --metric cosine \
  --normalize
```

Use `--metric euclidean` for Euclidean datasets and omit `--normalize` unless the metric should be cosine/IP.

## Register OpenAI Embedding Datasets

For OpenAI embedding exports, HiddenBridge includes a convenience wrapper that fixes the metric to cosine and enables normalization automatically.

Register a `1536`-dimensional OpenAI dataset:

```bash
python -m hiddenbridge.register_openai_dataset \
  --dimension 1536 \
  --train-file /path/to/openai_1536_train.npy \
  --test-file /path/to/openai_1536_test.npy \
  --data-root data
```

Register a `3072`-dimensional OpenAI dataset:

```bash
python -m hiddenbridge.register_openai_dataset \
  --dimension 3072 \
  --train-file /path/to/openai_3072_train.npy \
  --test-file /path/to/openai_3072_test.npy \
  --data-root data
```

If you want a specific dataset folder name instead of the defaults, pass `--dataset-subdir`.

## Run An Experiment

Example: GloVe `100k` database with two strong Steiner methods:

```bash
python -m hiddenbridge.experiment \
  --data-root data \
  --dataset-subdir glove-200-angular \
  --train-size 100000 \
  --query-count 500 \
  --hidden-count 512 \
  --top-k 10 \
  --max-degree 32 \
  --candidate-pool 96 \
  --beam-sizes-csv 8,16,32,64 \
  --methods-csv cluster_centroid,failure_driven \
  --candidate-source ivf \
  --graph-build-strategy fixed \
  --ivf-nprobe 32 \
  --validation-query-count 250 \
  --validation-beam-size 16
```

Example: full SIFT with all `10,000` queries:

```bash
python -m hiddenbridge.experiment \
  --data-root data \
  --dataset-subdir sift-128-euclidean \
  --train-size 1000000 \
  --query-count 10000 \
  --hidden-count 1024 \
  --top-k 10 \
  --max-degree 32 \
  --candidate-pool 96 \
  --beam-sizes-csv 8,16,32,64 \
  --methods-csv cluster_centroid,failure_driven,bridge,pairwise_interpolation \
  --candidate-source ivf \
  --graph-build-strategy fixed \
  --ivf-nprobe 32 \
  --validation-query-count 500 \
  --validation-beam-size 16
```

Example: OpenAI `1536` embeddings with a `100k` graph and `1,000` queries:

```bash
python -m hiddenbridge.experiment \
  --data-root data \
  --dataset-subdir openai-1536 \
  --train-size 100000 \
  --query-count 1000 \
  --hidden-count 512 \
  --top-k 10 \
  --max-degree 32 \
  --candidate-pool 96 \
  --beam-sizes-csv 8,16,32,64 \
  --methods-csv cluster_centroid,failure_driven,bridge,pairwise_interpolation \
  --candidate-source ivf \
  --graph-build-strategy fixed \
  --ivf-nprobe 32 \
  --validation-query-count 250 \
  --validation-beam-size 16
```

Example: OpenAI `3072` embeddings with a `100k` graph and `1,000` queries:

```bash
python -m hiddenbridge.experiment \
  --data-root data \
  --dataset-subdir openai-3072 \
  --train-size 100000 \
  --query-count 1000 \
  --hidden-count 512 \
  --top-k 10 \
  --max-degree 32 \
  --candidate-pool 96 \
  --beam-sizes-csv 8,16,32,64 \
  --methods-csv cluster_centroid,failure_driven,bridge,pairwise_interpolation \
  --candidate-source ivf \
  --graph-build-strategy fixed \
  --ivf-nprobe 32 \
  --validation-query-count 250 \
  --validation-beam-size 16
```

## Important Experiment Knobs

- `--train-size`
  Number of original database vectors used in the graph.
- `--query-count`
  Number of held-out queries used for evaluation.
- `--hidden-count`
  Requested Steiner-node budget.
- `--max-degree`
  Graph out-degree cap.
- `--candidate-pool`
  Candidate pool size before pruning or fixed-degree selection.
- `--beam-sizes-csv`
  Search beams to evaluate.
- `--methods-csv`
  Comma-separated Steiner methods to run.
- `--candidate-source`
  `exact`, `ivf`, or `auto`.
- `--graph-build-strategy`
  `vamana`, `fixed`, or `auto`.
- `--ivf-nprobe`
  IVF probe count when candidate generation uses IVF.

## Outputs

Each run writes:

- metrics JSON
- tradeoff plot PNG
- per-curve CSV table
- per-method summary CSV table

under:

```text
data/<dataset_subdir>/analysis/graph_navigation/
```

## Compare Multiple Runs

Compare multiple datasets or multiple experiment outputs:

```bash
python -m hiddenbridge.compare \
  --inputs \
    data/glove-200-angular/analysis/graph_navigation/run_a.json \
    data/sift-128-euclidean/analysis/graph_navigation/run_b.json \
  --output artifacts/benchmark_comparison.png
```

## Hidden-Count Ablation Plot

Combine several GloVe `100k` hidden-count runs into one plot:

```bash
python -m hiddenbridge.ablate_hidden_count \
  --inputs \
    data/glove-200-angular/analysis/graph_navigation/run_h512.json \
    data/glove-200-angular/analysis/graph_navigation/run_h4096.json \
    data/glove-200-angular/analysis/graph_navigation/run_h32768.json \
  --methods cluster_centroid,failure_driven \
  --output artifacts/glove_hidden_count_ablation.png
```

## Notes

- This is a simplified DiskANN-style research implementation, not the official Microsoft DiskANN codebase.
- Returned neighbors are always restricted to original dataset points.
- For large runs, IVF is only used to generate graph candidates. Search itself remains graph traversal.
- Exact brute-force ground truth is used when not already available from the dataset.
