"""
Model registry.

Each entry maps a name to ``(kind, builder)`` where kind is "clf" or "reg" and
builder() returns a fresh unfitted estimator. Keeping builders as zero-arg
factories means every walk-forward fold gets a clean model.

Two families, so classification and regression can be compared:
  * classifiers predict the per-fold return bucket;
  * regressors predict the continuous forward return directly.
Both are collapsed to a common expected-return *signal* in `signal.py`, which is
what the strategy consumes and what Rank IC is measured against.

Linear models get median imputation + scaling (they can't take NaNs and are
scale-sensitive); the HistGradientBoosting models handle NaNs natively and need
no scaling.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def _linear_preproc(final_step) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", final_step),
        ]
    )


def logistic_clf() -> Pipeline:
    return _linear_preproc(
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    )


def histgbm_clf() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_depth=3, random_state=RANDOM_STATE
    )


def ridge_reg() -> Pipeline:
    return _linear_preproc(Ridge(alpha=1.0))


def histgbm_reg() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=300, max_depth=3, random_state=RANDOM_STATE
    )


# name -> (kind, builder)
MODEL_REGISTRY: Dict[str, Tuple[str, Callable[[], object]]] = {
    "logistic_clf": ("clf", logistic_clf),
    "histgbm_clf": ("clf", histgbm_clf),
    "ridge_reg": ("reg", ridge_reg),
    "histgbm_reg": ("reg", histgbm_reg),
}


def build_model(name: str) -> Tuple[str, object]:
    """Return (kind, fresh_estimator) for a registered model name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Known: {list(MODEL_REGISTRY)}")
    kind, builder = MODEL_REGISTRY[name]
    return kind, builder()
