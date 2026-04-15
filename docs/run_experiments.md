# Run Experiments

## Main Runner

All experiments use:

```bash
python -m hiddenbridge.experiment ...
```

## GloVe 100k Example

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
  --methods-csv cluster_centroid,bridge,pairwise_interpolation \
  --candidate-source ivf \
  --graph-build-strategy fixed \
  --ivf-nprobe 32 \
  --validation-query-count 250 \
  --validation-beam-size 16
```

## Full SIFT Example

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
  --methods-csv cluster_centroid,bridge,pairwise_interpolation \
  --candidate-source ivf \
  --graph-build-strategy fixed \
  --ivf-nprobe 32 \
  --validation-query-count 500 \
  --validation-beam-size 16
```

## OpenAI 100k Examples

OpenAI `1536`:

```bash
python -m hiddenbridge.experiment \
  --data-root data \
  --dataset-subdir openai-1536 \
  --train-size 100000 \
  --query-count 1000 \
  --hidden-count 512 \
  --beam-sizes-csv 8,16,32,64 \
  --methods-csv cluster_centroid,bridge,pairwise_interpolation \
  --candidate-source ivf \
  --graph-build-strategy fixed \
  --ivf-nprobe 32
```

OpenAI `3072`:

```bash
python -m hiddenbridge.experiment \
  --data-root data \
  --dataset-subdir openai-3072 \
  --train-size 100000 \
  --query-count 1000 \
  --hidden-count 512 \
  --beam-sizes-csv 8,16,32,64 \
  --methods-csv cluster_centroid,bridge,pairwise_interpolation \
  --candidate-source ivf \
  --graph-build-strategy fixed \
  --ivf-nprobe 32
```

## High-Value Knobs

- `--train-size`: number of original database vectors
- `--query-count`: number of held-out evaluation queries
- `--hidden-count`: requested Steiner budget
- `--beam-sizes-csv`: beam widths to evaluate
- `--max-degree`: graph out-degree cap
- `--candidate-pool`: candidate list size before pruning or trimming
- `--candidate-source`: `exact`, `ivf`, or `auto`
- `--graph-build-strategy`: `fixed`, `vamana`, or `auto`
- `--ivf-nprobe`: IVF probe count when IVF is used

More detail:
- [Method Parameters](method_parameters.md)
- [Analyze Outputs](analysis.md)
