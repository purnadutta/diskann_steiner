# HiddenBridge

HiddenBridge studies whether routing-only Steiner points can improve DiskANN-style graph nearest-neighbor search. Steiner nodes may participate in graph build and traversal, but never appear in the final returned neighbors.

This repo is intentionally the core research code only: no Modal wrappers, no downloaded datasets, and no local benchmark artifact dump.

## Start Here

- [Quickstart](docs/quickstart.md)
- [Setup And Data](docs/setup.md)
- [Run Experiments](docs/run_experiments.md)
- [Analyze Outputs](docs/analysis.md)
- [Steiner Methods](docs/methods.md)
- [Method Parameters](docs/method_parameters.md)
- [Results Snapshot](docs/results.md)
- [Bridge Demo](docs/bridge_navigation_demo.html)

## Snapshot

Current non-adaptive takeaway: `cluster_centroid` is the strongest overall method in the current artifact set, while `bridge` is the cleanest geometry-driven win on full `sift-128-euclidean`.

![Conceptual bridge demo](docs/figures/bridge_navigation_demo.gif)

Representative benchmark figure: on `glove-200-angular` with `100k` database points and `500` queries, `cluster_centroid` reaches `Recall@10 = 0.6522` at `825.9` average distance computations versus baseline `0.4190` at `922.6`.

![GloVe 100k cluster-centroid result](docs/figures/glove_100k_cluster_centroid.png)
