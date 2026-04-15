# Setup And Data

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`faiss-cpu` is strongly recommended for non-toy runs.

## Dataset Layout

```text
data/<dataset_subdir>/
  train.npy
  test.npy
  dataset_metadata.json
  ground_truth_neighbors.npy
  ground_truth_distances.npy
  indices/
  analysis/graph_navigation/
```

Ground-truth files and `indices/` are optional, but used when available.

## Download ANN-Benchmarks Datasets

Download GloVe:

```bash
python -m hiddenbridge.download_ann_benchmark --dataset glove --data-root data
```

Download SIFT:

```bash
python -m hiddenbridge.download_ann_benchmark --dataset sift --data-root data
```

## Register Existing Arrays

```bash
python -m hiddenbridge.register_dataset \
  --data-root data \
  --dataset-subdir my-dataset \
  --train-file /path/to/train.npy \
  --test-file /path/to/test.npy \
  --metric cosine \
  --normalize
```

Use `--metric euclidean` for Euclidean datasets. Omit `--normalize` unless you want cosine/IP-style behavior.

## Register OpenAI Embedding Datasets

OpenAI `1536`:

```bash
python -m hiddenbridge.register_openai_dataset \
  --dimension 1536 \
  --train-file /path/to/openai_1536_train.npy \
  --test-file /path/to/openai_1536_test.npy \
  --data-root data
```

OpenAI `3072`:

```bash
python -m hiddenbridge.register_openai_dataset \
  --dimension 3072 \
  --train-file /path/to/openai_3072_train.npy \
  --test-file /path/to/openai_3072_test.npy \
  --data-root data
```

Next:
- [Run Experiments](run_experiments.md)
- [Analyze Outputs](analysis.md)
