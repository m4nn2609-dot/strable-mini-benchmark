"""
Mini STRABLE-style benchmark: string encoder -> tabular learner.

Replicates the spirit of STRABLE (Blayer et al., 2026) and TILE (Lefebvre et al.,
2026) at small scale using open, cheap components only.

Pipelines:
  1. TableVectorizer + HistGradientBoosting   (baseline)
  2. TableVectorizer + TabPFN-2.5             (advanced learner, if available)

Score: mean R^2 (regression) / ROC AUC (classification) across STRABLE's 3 folds.
Timing: fit + predict time per 1K samples per fold.

Outputs: results/scores.csv  (one row per table x pipeline x fold)
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

# Limit internal threading before importing sklearn to avoid joblib
# "concurrent send_bytes() calls" crashes on Windows/Python 3.14.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

import skrub
from skrub import TableVectorizer

import strable

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration: pick a small mix of regression + classification tables
# ---------------------------------------------------------------------------
TABLES = [
    # Regression
    "beer-ratings",
    "aca-federal-upper-limits-wide",
    # Classification
    "chocolate-bar-ratings",
]

# Tables larger than this get subsampled (keeps the mini-benchmark fast).
MAX_ROWS = 5000

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------
def build_pipelines(task: str):
    """Return list of (name, sklearn Pipeline) for the given task."""
    if task in ("regression",):
        learner_base = HistGradientBoostingRegressor(random_state=0)
    else:
        learner_base = HistGradientBoostingClassifier(random_state=0)

    pipelines = [
        ("tablevec_hgb", make_pipeline(TableVectorizer(), learner_base)),
    ]

    # Try TabPFN; fall back gracefully if not installed / not gated.
    try:
        from tabpfn import TabPFNRegressor, TabPFNClassifier

        if task in ("regression",):
            tabpfn = TabPFNRegressor(n_estimators=8, random_state=0)
        else:
            tabpfn = TabPFNClassifier(n_estimators=8, random_state=0)
        pipelines.append(("tablevec_tabpfn", make_pipeline(TableVectorizer(), tabpfn)))
    except Exception as e:  # pragma: no cover - optional dependency
        print(f"  [skip] TabPFN not available: {e}")

    return pipelines


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_pipeline(pipe, X_train, y_train, X_test, y_test, task: str):
    """Fit, predict, return (score, fit_time_s, predict_time_s)."""
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    predict_time = time.perf_counter() - t0

    if task in ("regression",):
        score = r2_score(y_test, y_pred)
    else:
        # ROC AUC: one-vs-rest macro for multiclass, binary for binary.
        try:
            if hasattr(pipe, "predict_proba"):
                y_prob = pipe.predict_proba(X_test)
            else:
                y_prob = y_pred
            if len(np.unique(y_test)) == 2:
                # binary: probability of positive class
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    score = roc_auc_score(y_test, y_prob[:, 1])
                else:
                    score = roc_auc_score(y_test, y_prob)
            else:
                score = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception:
            score = float("nan")

    return score, fit_time, predict_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rows = []

    for table_name in TABLES:
        print(f"\n=== {table_name} ===")
        X, y, task = strable.load_table(table_name)
        folds = strable.get_folds(table_name)

        # Subsample large tables so the mini-benchmark stays fast.
        n_full = len(X)
        if n_full > MAX_ROWS:
            rng = np.random.default_rng(0)
            keep = rng.choice(n_full, size=MAX_ROWS, replace=False)
            keep = np.sort(keep)
            X = X.iloc[keep]
            y = y[keep]
            # Rebuild folds on the subsampled indices: map old positions -> new positions.
            old_to_new = {old: new for new, old in enumerate(keep)}
            folds = [
                (
                    np.array([old_to_new[i] for i in tr if i in old_to_new]),
                    np.array([old_to_new[i] for i in te if i in old_to_new]),
                )
                for tr, te in folds
            ]
            folds = [(tr, te) for tr, te in folds if len(tr) > 0 and len(te) > 0]
            print(f"  [subsampled {n_full} -> {len(X)} rows]")

        n_rows = len(X)
        y = np.asarray(y)  # strable returns y as ndarray; normalize

        print(f"  task={task}, shape={X.shape}, folds={len(folds)}")

        pipelines = build_pipelines(task)

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Encode y for classification learners that need numeric labels.
            if task != "regression":
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)

            for pipe_name, pipe in pipelines:
                score, fit_t, pred_t = score_pipeline(
                    pipe, X_train, y_train, X_test, y_test, task
                )
                total_t = fit_t + pred_t
                per_1k = total_t / n_rows * 1000.0

                rows.append(
                    {
                        "table": table_name,
                        "task": task,
                        "fold": fold_idx,
                        "pipeline": pipe_name,
                        "score": score,
                        "fit_time_s": fit_t,
                        "predict_time_s": pred_t,
                        "total_time_s": total_t,
                        "n_rows": n_rows,
                        "time_per_1k_s": per_1k,
                    }
                )
                print(
                    f"  fold {fold_idx} | {pipe_name:18s} | score={score:.4f} | "
                    f"time={total_t:.2f}s | per1K={per_1k:.3f}s"
                )

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "scores.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # Summary: mean score and mean time per 1K per pipeline (across tables/folds).
    summary = (
        df.groupby(["task", "pipeline"])
        .agg(mean_score=("score", "mean"), mean_time_per_1k=("time_per_1k_s", "mean"), n=("score", "size"))
        .reset_index()
    )
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    summary_path = RESULTS_DIR / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
