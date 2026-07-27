"""
Turn a model's output into a single expected-return *signal*.

This is the common surface that lets a classifier and a regressor compete on
equal footing:
  * regressor  -> its predicted forward return IS the signal;
  * classifier -> map class probabilities to an expected return via the
                  per-fold bucket mean returns:  signal = sum_k P(k) * rbar_k.

The same expected-return signal feeds both the strategy (stage 03) and Rank IC.
"""

from __future__ import annotations

import numpy as np


def classifier_signal(
    proba: np.ndarray,
    classes: np.ndarray,
    bucket_means: np.ndarray,
    n_buckets: int,
) -> np.ndarray:
    """Probability-weighted expected return.

    `proba` has columns in the order of `classes` (a model's ``classes_``), which
    may be a subset of 0..n_buckets-1 if a fold's training set lacked a class.
    We scatter the columns back to the full bucket layout before dotting with the
    bucket mean returns.
    """
    full = np.zeros((proba.shape[0], n_buckets), dtype=float)
    for col, cls in enumerate(classes):
        full[:, int(cls)] = proba[:, col]
    return full @ bucket_means


def regressor_signal(y_pred: np.ndarray) -> np.ndarray:
    """The regression prediction is already an expected return."""
    return np.asarray(y_pred, dtype=float)
