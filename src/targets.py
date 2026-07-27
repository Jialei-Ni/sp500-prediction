"""
Targets.

The canonical target is the **continuous forward return** over a horizon H:
    fwd_ret_H[t] = Close[t+H] / Close[t] - 1

This continuous form is what regression predicts and what Rank IC is measured
against. Classification labels are derived from it by bucketing -- but the bucket
edges must be fit on TRAINING data only, per fold, never on the whole sample.
That is why bucketing is a separate step (`fit_bucket_edges` / `apply_buckets`)
called inside the walk-forward in the model stage, rather than baked into the
cached dataset.

Note on overlap: consecutive H-day forward returns share H-1 days, so labels are
autocorrelated. The model stage handles this with an embargo around each fold
boundary; targets here are simply the raw forward returns.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Continuous forward return over `horizon` trading days (uses future data;
    the last `horizon` rows are NaN)."""
    return close.shift(-horizon) / close - 1.0


def build_targets(close: pd.Series, horizons: Mapping[str, int]) -> pd.DataFrame:
    """One continuous forward-return column per horizon, named ``fwd_ret_{H}``."""
    out = pd.DataFrame(index=close.index)
    for _, h in horizons.items():
        out[f"fwd_ret_{h}"] = forward_return(close, h)
    return out


def fit_bucket_edges(
    returns_train: pd.Series,
    *,
    n_buckets: int = 5,
    method: str = "percentile",
    fixed_bins: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Compute bucket edges from TRAINING returns only.

    Edges are made open-ended (-inf .. +inf) so out-of-sample values beyond the
    training range still fall into the extreme buckets rather than becoming NaN.

    method="percentile": quantile edges from the training distribution.
    method="fixed":      `fixed_bins` interior cut points (e.g. [-0.03,-0.01,0.01,0.03]).
    """
    if method == "percentile":
        qs = np.linspace(0.0, 1.0, n_buckets + 1)
        edges = returns_train.quantile(qs).to_numpy(dtype=float)
        edges = np.unique(edges)  # guard against duplicate quantiles (ties)
        edges[0], edges[-1] = -np.inf, np.inf
        return edges
    if method == "fixed":
        if fixed_bins is None:
            raise ValueError("method='fixed' requires fixed_bins")
        return np.array([-np.inf, *fixed_bins, np.inf], dtype=float)
    raise ValueError("method must be 'percentile' or 'fixed'")


def apply_buckets(
    returns: pd.Series,
    edges: np.ndarray,
    labels: Optional[Sequence[int]] = None,
) -> pd.Series:
    """Bucket `returns` into ordinal classes 0..k-1 using pre-fit `edges`."""
    if labels is None:
        labels = list(range(len(edges) - 1))
    cut = pd.cut(returns, bins=edges, labels=labels, include_lowest=True)
    return cut.astype("Int64")


def bucket_mean_returns(
    returns_train: pd.Series,
    labels_train: pd.Series,
    n_buckets: int,
) -> np.ndarray:
    """Mean training return within each bucket -- the score used to turn class
    probabilities into an expected-return signal (`sum_k P(k) * r_k`) for the
    classifier, so it competes with regression on a common footing and feeds IC.
    Empty buckets fall back to the class index (monotone tie-break)."""
    means = np.full(n_buckets, np.nan)
    grouped = returns_train.groupby(labels_train.astype("Int64"))
    for k, mean in grouped.mean().items():
        if pd.notna(k):
            means[int(k)] = mean
    for k in range(n_buckets):
        if np.isnan(means[k]):
            means[k] = float(k)
    return means
