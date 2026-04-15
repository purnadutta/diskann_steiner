# Steiner Methods

HiddenBridge currently includes 12 Steiner constructions:

- `pairwise_interpolation`
- `cluster_centroid`
- `local_knn_mean`
- `random_line`
- `random_line_anchor`
- `noisy_copy`
- `bridge`
- `hierarchical_centroid`
- `directional_centroid`
- `boundary_shell`
- `failure_driven`
- `targeted_noisy_replicas`

Use one method with `--methods-csv cluster_centroid` or compare several at once with something like `--methods-csv cluster_centroid,bridge,pairwise_interpolation`.

## Shared Parameter Guide

The experiment runner accepts one shared superset of parameters for every method. Each Steiner construction uses only a subset of them.

### Shared Parameters For Almost Every Run

- `--dataset-subdir`
  Which dataset folder under `data/` to load.
- `--train-size`
  Number of original database vectors included in the graph.
- `--query-count`
  Number of held-out queries used for evaluation.
- `--hidden-count`
  Requested Steiner budget.
- `--top-k`
  Final result cutoff used for `recall@k`.
- `--beam-sizes-csv`
  Search beams evaluated for the recall-vs-compute curve.
- `--max-degree`
  Out-degree cap for the DiskANN-style graph.
- `--candidate-pool`
  Candidate pool size before graph pruning or fixed-degree selection.
- `--candidate-source`
  Candidate generation mode: `exact`, `ivf`, or `auto`.
- `--graph-build-strategy`
  Graph builder mode: `fixed`, `vamana`, or `auto`.
- `--ivf-nlist`
  IVF coarse cluster count when candidate generation uses FAISS IVF.
- `--ivf-nprobe`
  Number of IVF coarse clusters scanned per query or per node.
- `--ivf-overfetch`
  Extra IVF candidates fetched before trimming to the candidate pool.
- `--alpha`
  Vamana-style pruning aggressiveness. Only matters when `--graph-build-strategy vamana` is used.
- `--seed`
  Global random seed used by stochastic builders.

### Shared Parameters That Only Matter For Some Methods

- `--kmeans-niter`
  Number of k-means iterations.
- `--noise-std`
  Noise scale for noisy-replica methods.
- `--validation-query-count`
  Number of baseline validation queries used to mine search traces.
- `--validation-beam-size`
  Beam size used when collecting those validation traces.
- `--failure-recall-threshold`
  Threshold for deciding which validation searches count as failures.
- `--query-seed-candidate-count`
  Number of Steiner nodes considered as possible query-time seed nodes.
- `--query-seed-top-m`
  How many of those Steiner seed nodes are actually inserted at query time.
- `--line-gap-fraction`
  Minimum gap between neighboring random-line Steiner points.
- `--line-shift-scale`
  Tiny orthogonal offset for random-line Steiner points.
- `--directional-directions-per-cluster`
  Number of principal directions kept per cluster.
- `--directional-offset-scale`
  Offset size along each principal direction.
- `--shell-scale`
  How far boundary-shell points are pushed outside their local cluster.
- `--local-mean-neighbor-count`
  Neighborhood size used when averaging local `k`-NN means.
- `--exact-batch-size`
  Batch size for exact assignment/scoring steps used inside some methods.

## Method Details

### `pairwise_interpolation`

- Main knobs: `--hidden-count`
- Indirectly important: `--max-degree`, `--candidate-pool`, `--candidate-source`, `--graph-build-strategy`
- What it does:
  Chooses long baseline graph edges and inserts midpoint Steiner nodes.

### `cluster_centroid`

- Main knobs: `--hidden-count`, `--kmeans-niter`
- Query-seeding knobs: `--query-seed-candidate-count`, `--query-seed-top-m`
- Randomness knob: `--seed`
- What it does:
  Runs k-means and uses centroids as routing-only Steiner nodes.

### `local_knn_mean`

- Main knobs: `--hidden-count`, `--local-mean-neighbor-count`
- Validation-trace knobs: `--validation-query-count`, `--validation-beam-size`
- Query-seeding knobs: `--query-seed-candidate-count`, `--query-seed-top-m`
- Randomness knob: `--seed`
- What it does:
  Finds important baseline nodes, averages each node with a small local neighborhood, and uses those means as Steiner nodes plus diversified query-time seeds.

### `random_line`

- Main knobs: `--hidden-count`, `--line-gap-fraction`, `--line-shift-scale`
- Randomness knob: `--seed`
- What it does:
  Places Steiner points along a random projected line through the dataset.
- Important note:
  The actual number of line points is capped to roughly `sqrt(n)` and may be smaller than `--hidden-count`.

### `random_line_anchor`

- Main knobs: `--hidden-count`, `--line-gap-fraction`, `--line-shift-scale`
- Randomness knob: `--seed`
- What it does:
  Same construction as `random_line`, but also adds a synthetic global anchor that can be used as the search start.

### `noisy_copy`

- Main knobs: `--hidden-count`, `--noise-std`
- Randomness knob: `--seed`
- What it does:
  Selects original points uniformly at random and adds slightly perturbed copies.

### `bridge`

- Main knobs: `--hidden-count`
- Indirectly important: `--max-degree`, `--candidate-pool`, `--candidate-source`, `--graph-build-strategy`
- What it does:
  Inserts midpoint-like bridge points between geometrically close pairs that look weakly connected in the baseline graph.

### `hierarchical_centroid`

- Main knobs: `--hidden-count`, `--kmeans-niter`, `--exact-batch-size`
- Query-seeding knobs: `--query-seed-candidate-count`, `--query-seed-top-m`
- Randomness knob: `--seed`
- What it does:
  Splits the Steiner budget into coarse centroids and finer within-cluster centroids.

### `directional_centroid`

- Main knobs: `--hidden-count`, `--kmeans-niter`, `--exact-batch-size`
- Directional knobs: `--directional-directions-per-cluster`, `--directional-offset-scale`
- Randomness knob: `--seed`
- What it does:
  Adds a centroid plus small offsets along principal directions for each cluster.

### `boundary_shell`

- Main knobs: `--hidden-count`, `--kmeans-niter`, `--exact-batch-size`, `--shell-scale`
- Randomness knob: `--seed`
- What it does:
  Places Steiner points slightly outside cluster boundaries so they behave like gateway nodes.

### `failure_driven`

- Main knobs: `--hidden-count`
- Validation-trace knobs: `--validation-query-count`, `--validation-beam-size`, `--failure-recall-threshold`
- Indirectly important: `--max-degree`, `--candidate-pool`, `--candidate-source`, `--graph-build-strategy`
- What it does:
  Runs baseline search on validation queries and inserts Steiner points between stuck nodes and missed true neighbors.

### `targeted_noisy_replicas`

- Main knobs: `--hidden-count`, `--noise-std`
- Validation-trace knobs: `--validation-query-count`, `--validation-beam-size`
- Randomness knob: `--seed`
- What it does:
  Adds noisy replicas only around important nodes such as hubs, bottlenecks, and frequently visited states.

## Suggested Starting Settings

- `pairwise_interpolation`
  Start with `--hidden-count 512` or `1024`.
- `bridge`
  Start with `--hidden-count 512` or `1024` and a reasonably strong baseline graph such as `--candidate-pool 96`.
- `cluster_centroid`
  Start with `--hidden-count 512`, `--kmeans-niter 20`, `--query-seed-candidate-count 24`, `--query-seed-top-m 4`.
- `hierarchical_centroid`
  Start with `--hidden-count 512`, `--kmeans-niter 20`, `--query-seed-candidate-count 16`, `--query-seed-top-m 4`.
- `local_knn_mean`
  Start with `--hidden-count 512`, `--local-mean-neighbor-count 8`, `--query-seed-candidate-count 24`, `--query-seed-top-m 4`.
- `random_line` or `random_line_anchor`
  Start with `--line-gap-fraction 0.35` and `--line-shift-scale 0.0005`.
- `noisy_copy` or `targeted_noisy_replicas`
  Start with `--noise-std 0.002`.
- `directional_centroid`
  Start with `--directional-directions-per-cluster 2` and `--directional-offset-scale 0.08`.
- `boundary_shell`
  Start with `--shell-scale 0.12`.
- `failure_driven`
  Start with `--validation-query-count 250`, `--validation-beam-size 16`, `--failure-recall-threshold 0.5`.
