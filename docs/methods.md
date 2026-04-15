# Steiner Methods

HiddenBridge includes 12 Steiner constructions.

## Simple Geometry

- `pairwise_interpolation`
- `bridge`
- `random_line`
- `random_line_anchor`
- `noisy_copy`

## Centroid-Based

- `cluster_centroid`
- `hierarchical_centroid`
- `directional_centroid`
- `boundary_shell`

## Local Or Adaptive

- `local_knn_mean`
- `failure_driven`
- `targeted_noisy_replicas`

Use one method with `--methods-csv cluster_centroid` or compare several at once with `--methods-csv cluster_centroid,bridge,pairwise_interpolation`.

For the actual knobs and per-method notes, see:
- [Method Parameters](method_parameters.md)
- [Results Snapshot](results.md)
