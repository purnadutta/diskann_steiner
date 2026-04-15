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

## Repo Analysis Figures

Curated repo copies of the main benchmark plots live in:

```text
docs/analysis_figures/
```

These are regenerated from the saved benchmark JSONs with:

```bash
python scripts/render_repo_analysis_plots.py
```

Available repo figures:

- [GloVe 100k tradeoff](analysis_figures/glove_100k_tradeoff.png)
- [GloVe full tradeoff](analysis_figures/glove_full_tradeoff.png)
- [OpenAI 1536 tradeoff](analysis_figures/openai_1536_100k_tradeoff.png)
- [OpenAI 3072 tradeoff](analysis_figures/openai_3072_100k_tradeoff.png)
- [SIFT full tradeoff](analysis_figures/sift_full_tradeoff.png)
- [Cross-dataset comparison](analysis_figures/cross_dataset_comparison.png)
- [GloVe 100k hidden-count ablation](analysis_figures/glove_100k_hidden_count_ablation.png)

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
