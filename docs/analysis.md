# Analyze Outputs

## Run Outputs

Each experiment writes:

- metrics JSON
- tradeoff plot PNG
- per-curve CSV table
- per-method summary CSV table

under:

```text
data/<dataset_subdir>/analysis/graph_navigation/
```

## Compare Multiple Runs

```bash
python -m hiddenbridge.compare \
  --inputs \
    data/glove-200-angular/analysis/graph_navigation/run_a.json \
    data/sift-128-euclidean/analysis/graph_navigation/run_b.json \
  --output artifacts/benchmark_comparison.png
```

## Hidden-Count Ablation

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
- For large runs, IVF is used only for candidate generation; search itself remains graph traversal.
- Exact brute-force ground truth is used when not already available from the dataset.

Related:
- [Results Snapshot](results.md)
- [Steiner Methods](methods.md)
