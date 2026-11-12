# strable-mini-benchmark

Small-scale, open-source STRABLE-style benchmark for tabular learning with strings.

## What this is

A local replication of the **spirit** of STRABLE (Blayer et al., 2026) and TILE (Lefebvre et al., 2026):

- **Pipelines** = string encoder → tabular learner (modular, like STRABLE/TILE)
- **Score** = mean R² (regression) or ROC AUC (classification)
- **Timing** = fit + predict time per 1K samples
- **Data** = a subset of STRABLE's 108 real-world tables with strings
- **Components** = open and cheap: scikit-learn, skrub, sentence-transformers

We do **not** run 70B models, Kumo Tabular, or TabPFN-3.5. This is a hypothesis-generating mini-benchmark, not a replication of the TILE leaderboard.

## What's inside

| File | Purpose |
|------|---------|
| `benchmark.py` | Runs two pipelines on selected STRABLE tables, scores them, times them, saves results to `results/` |
| `results/` | CSV outputs (one row per table × pipeline × fold) |

## Pipelines

1. **TableVectorizer + HistGradientBoosting** (baseline) — skrub's native string handling
2. **TableVectorizer + TabPFN-2.5** (advanced tabular learner) — if available; falls back to HistGradientBoosting

## How to run

```bash
cd strable-mini-benchmark
py benchmark.py
```

Requires: `scikit-learn`, `skrub`, `strable`, `pandas`, `numpy`, `matplotlib`.  
Optionally: `tabpfn` (gated — needs Hugging Face license acceptance).

## What we're NOT claiming

With ~10 tables we cannot establish a generalizable Pareto frontier. STRABLE itself shows rankings stabilize only around N=100 (τ≈0.9 vs oracle). Treat any "winner" here as a hypothesis about *these* tables, not a general rule.

## Reference

- STRABLE: https://arxiv.org/abs/2605.12292 — Blayer, Kim, Lefebvre, et al. (NeurIPS 2026)
- TILE: https://arxiv.org/abs/2610.17065 — Lefebvre, Holzmüller, Blayer, et al.
- `strable` package: https://github.com/soda-inria/strable