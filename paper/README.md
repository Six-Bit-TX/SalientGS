# Paper source

This directory contains the complete ACM MM 2026 camera-ready LaTeX source,
all figure assets, the pre-generated bibliography files, and compiled PDFs.

Build the main paper and supplementary material with:

```bash
latexmk -pdf main.tex
latexmk -pdf supplementary.tex
```

The released-code verification uses equal-weight benchmark macro-averaging.
All 13 scenes, including failed baseline scenes, remain in the reported
dataset aggregates. Per-scene measurements are archived in
[`../results/paper_reproduction_seed42.csv`](../results/paper_reproduction_seed42.csv).
