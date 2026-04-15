# Results Snapshot

This page summarizes the current local benchmark takeaways represented in the repo figures and artifact summaries.

## Main Takeaway

If `failure_driven` is excluded as too adaptive/post-hoc, the current non-adaptive story is:

- `cluster_centroid` is the strongest overall Steiner method
- `bridge` is the cleanest geometry-driven win on full `sift-128-euclidean`
- `pairwise_interpolation` is usually safe but weak

## Dataset Winners

- `glove-200-angular`, `100k`, `500` queries
  Winner: `cluster_centroid`
- `glove-200-angular`, full `1.18M`, `1000` queries
  Winner among the methods run there: `cluster_centroid`
- `dbpedia-openai3-large-1536`, `100k`, `1000` queries
  Best overall: `failure_driven`
  Best non-adaptive alternative: `bridge`, but only by a small margin
- `dbpedia-openai3-large-3072`, `100k`, `1000` queries
  Best overall: `failure_driven`
  Best non-adaptive alternative: `bridge`, again by a small margin
- `sift-128-euclidean`, full `1M`, all `10000` queries
  Winner: `bridge`

## Interpreting The Repo Figures

The README intentionally shows the strongest clean `cluster_centroid` examples:

- `glove_100k_cluster_centroid.png`
- `glove_full_cluster_centroid.png`

Those are included because they show a larger and easier-to-explain margin over the baseline graph than the other non-adaptive methods.

The bridge concept animation is not a benchmark result. It is a conceptual visualization of why a routing-only Steiner point can improve graph navigability even when the graph is already connected.
