"""
Time-series cross-validation: expanding walk-forward with an embargo.

Why an embargo? The target is an H-day forward return, so the label at row t
spans [t, t+H]. A training row t therefore "sees into" the test block whenever
t + H reaches the test start. We purge those rows: training uses only rows with
t < test_start - embargo, with embargo = H. This removes the overlap-driven
leakage that a naive chronological split leaves in.

Folds are expanding: every fold trains on [0, train_end) and tests on the next
contiguous forward block, so test blocks are consecutive and non-overlapping in
time (useful later for non-overlapping IC sampling and for stitching a single
out-of-sample equity curve).
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def walk_forward_splits(
    n_samples: int,
    *,
    n_folds: int,
    embargo: int,
    min_train_frac: float = 0.4,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) positional arrays for an expanding walk-forward.

    Parameters
    ----------
    n_samples
        Number of rows in the (already NaN-dropped) dataset.
    n_folds
        Number of contiguous forward test blocks.
    embargo
        Rows to purge from the end of each training window (use the horizon H).
    min_train_frac
        Fraction of the series reserved as the initial training window before the
        first test block begins.
    """
    if not 0 < min_train_frac < 1:
        raise ValueError("min_train_frac must be in (0, 1)")

    initial = int(n_samples * min_train_frac)
    test_total = n_samples - initial
    if test_total < n_folds:
        raise ValueError(
            f"Not enough samples ({n_samples}) for {n_folds} folds "
            f"after reserving {initial} for initial training."
        )

    fold_size = test_total // n_folds

    for i in range(n_folds):
        test_start = initial + i * fold_size
        test_end = n_samples if i == n_folds - 1 else initial + (i + 1) * fold_size

        train_end = test_start - embargo  # purge `embargo` rows before the test block
        if train_end <= 0:
            continue  # first fold too close to the start after embargo; skip

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx
