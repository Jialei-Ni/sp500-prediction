"""
Evaluation metrics.

Two layers:
  * ML metrics native to each model type (balanced accuracy / log-loss for
    classifiers; MAE / RMSE / R^2 for regressors) -- useful but not directly
    comparable across types.
  * Decision-relevant metrics on the common expected-return signal:
      - Rank IC: Spearman correlation between signal and realized forward return.
        Reported on NON-OVERLAPPING samples (every H-th observation) so the
        overlapping-horizon autocorrelation doesn't inflate it.
      - ICIR: mean(IC)/std(IC) across folds -- IC's consistency.
      - hit rate: directional accuracy of the signal.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    r2_score,
)


def rank_ic(signal: np.ndarray, realized: np.ndarray) -> float:
    """Spearman rank correlation between signal and realized return."""
    s = np.asarray(signal, float)
    r = np.asarray(realized, float)
    mask = np.isfinite(s) & np.isfinite(r)
    if mask.sum() < 3 or np.unique(s[mask]).size < 2:
        return np.nan
    return float(spearmanr(s[mask], r[mask]).correlation)


def nonoverlap_rank_ic(
    signal: np.ndarray, realized: np.ndarray, horizon: int
) -> float:
    """Rank IC on every H-th observation (approx. non-overlapping)."""
    return rank_ic(signal[::horizon], realized[::horizon])


def hit_rate(signal: np.ndarray, realized: np.ndarray) -> float:
    """Fraction of observations where signal and realized share sign."""
    s = np.sign(np.asarray(signal, float))
    r = np.sign(np.asarray(realized, float))
    mask = (s != 0) & (r != 0) & np.isfinite(s) & np.isfinite(r)
    if mask.sum() == 0:
        return np.nan
    return float((s[mask] == r[mask]).mean())


def icir(fold_ics: Sequence[float]) -> float:
    """IC information ratio: mean / std of per-fold ICs."""
    ics = np.array([x for x in fold_ics if np.isfinite(x)], dtype=float)
    if ics.size < 2 or ics.std(ddof=1) == 0:
        return np.nan
    return float(ics.mean() / ics.std(ddof=1))


def classifier_ml_metrics(
    y_true_class: np.ndarray,
    y_pred_class: np.ndarray,
    proba: Optional[np.ndarray] = None,
    labels: Optional[Sequence[int]] = None,
) -> dict:
    out = {
        "accuracy": accuracy_score(y_true_class, y_pred_class),
        "balanced_accuracy": balanced_accuracy_score(y_true_class, y_pred_class),
    }
    if proba is not None and labels is not None:
        try:
            out["log_loss"] = log_loss(y_true_class, proba, labels=labels)
        except ValueError:
            out["log_loss"] = np.nan
    return out


def regressor_ml_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))),
        "r2": r2_score(y_true, y_pred),
    }
