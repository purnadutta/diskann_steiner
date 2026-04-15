# Quickstart

Use this page as the short map:

- [Setup And Data](setup.md)
  Installation, dataset layout, ANN-Benchmarks downloads, and dataset registration.
- [Run Experiments](run_experiments.md)
  Core `hiddenbridge.experiment` examples for GloVe, SIFT, and OpenAI datasets.
- [Analyze Outputs](analysis.md)
  Tradeoff plots, comparison plots, ablations, and output locations.
- [Results Snapshot](results.md)
  Current benchmark takeaways.

## Hidden-Count Ablation Plot

```bash
python -m hiddenbridge.ablate_hidden_count \
  --inputs \
    data/glove-200-angular/analysis/graph_navigation/run_h512.json \
    data/glove-200-angular/analysis/graph_navigation/run_h4096.json \
    data/glove-200-angular/analysis/graph_navigation/run_h32768.json \
  --methods cluster_centroid,bridge \
  --output artifacts/glove_hidden_count_ablation.png
```

## Notes

- This is a simplified DiskANN-style research implementation, not the official Microsoft DiskANN codebase.
- Returned neighbors are always restricted to original dataset points.
- For large runs, IVF is only used to generate graph candidates. Search itself remains graph traversal.
- Exact brute-force ground truth is used when not already available from the dataset.
