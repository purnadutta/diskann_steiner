# Method Parameters

## Shared Parameters

- `--dataset-subdir`: dataset folder under `data/`
- `--train-size`: number of original database vectors
- `--query-count`: number of held-out evaluation queries
- `--hidden-count`: requested Steiner budget
- `--top-k`: cutoff used for `recall@k`
- `--beam-sizes-csv`: beam widths to evaluate
- `--max-degree`: graph out-degree cap
- `--candidate-pool`: candidate list size before pruning or trimming
- `--candidate-source`: `exact`, `ivf`, or `auto`
- `--graph-build-strategy`: `fixed`, `vamana`, or `auto`
- `--ivf-nlist`: IVF coarse cluster count
- `--ivf-nprobe`: IVF probe count
- `--ivf-overfetch`: extra IVF candidates before trimming
- `--alpha`: Vamana pruning aggressiveness
- `--seed`: global random seed

## Extra Parameters Used By Some Methods

- `--kmeans-niter`
- `--noise-std`
- `--validation-query-count`
- `--validation-beam-size`
- `--failure-recall-threshold`
- `--query-seed-candidate-count`
- `--query-seed-top-m`
- `--line-gap-fraction`
- `--line-shift-scale`
- `--directional-directions-per-cluster`
- `--directional-offset-scale`
- `--shell-scale`
- `--local-mean-neighbor-count`
- `--exact-batch-size`

## Method Notes

### `pairwise_interpolation`

- Main knob: `--hidden-count`
- Uses midpoint-like Steiner nodes on selected original pairs

### `cluster_centroid`

- Main knobs: `--hidden-count`, `--kmeans-niter`
- Query-seeding knobs: `--query-seed-candidate-count`, `--query-seed-top-m`
- Uses k-means centroids as routing-only nodes

### `local_knn_mean`

- Main knobs: `--hidden-count`, `--local-mean-neighbor-count`
- Validation knobs: `--validation-query-count`, `--validation-beam-size`
- Uses local neighborhood means around important nodes

### `random_line`

- Main knobs: `--hidden-count`, `--line-gap-fraction`, `--line-shift-scale`
- Places Steiner points along a random projected line

### `random_line_anchor`

- Same knobs as `random_line`
- Adds a synthetic global anchor start node

### `noisy_copy`

- Main knobs: `--hidden-count`, `--noise-std`
- Adds small-noise replicas of selected original points

### `bridge`

- Main knob: `--hidden-count`
- Adds midpoint-like bridge nodes between close but weakly connected regions

### `hierarchical_centroid`

- Main knobs: `--hidden-count`, `--kmeans-niter`, `--exact-batch-size`
- Query-seeding knobs: `--query-seed-candidate-count`, `--query-seed-top-m`
- Uses coarse and fine centroid layers

### `directional_centroid`

- Main knobs: `--hidden-count`, `--kmeans-niter`, `--exact-batch-size`
- Direction knobs: `--directional-directions-per-cluster`, `--directional-offset-scale`
- Adds centroid-offset points along principal directions

### `boundary_shell`

- Main knobs: `--hidden-count`, `--kmeans-niter`, `--exact-batch-size`, `--shell-scale`
- Pushes Steiner points slightly outside local clusters

### `failure_driven`

- Main knobs: `--hidden-count`
- Validation knobs: `--validation-query-count`, `--validation-beam-size`, `--failure-recall-threshold`
- Adds Steiner points based on failed validation traces

### `targeted_noisy_replicas`

- Main knobs: `--hidden-count`, `--noise-std`
- Validation knobs: `--validation-query-count`, `--validation-beam-size`
- Adds noisy replicas around hubs and bottlenecks

## Suggested Starting Points

- `cluster_centroid`: `--hidden-count 512 --kmeans-niter 20 --query-seed-candidate-count 24 --query-seed-top-m 4`
- `bridge`: `--hidden-count 512`
- `pairwise_interpolation`: `--hidden-count 512`
- `random_line`: `--line-gap-fraction 0.35 --line-shift-scale 0.0005`
- `noisy_copy`: `--noise-std 0.002`
- `directional_centroid`: `--directional-directions-per-cluster 2 --directional-offset-scale 0.08`
- `boundary_shell`: `--shell-scale 0.12`

Related:
- [Run Experiments](run_experiments.md)
- [Results Snapshot](results.md)
