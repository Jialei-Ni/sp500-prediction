"""
Walk-forward experiment orchestrator.

`run_experiment` ties the pieces together for every (horizon, model):
  1. build the per-horizon dataset (features + continuous forward target), drop
     rows with no target and an initial warm-up so long-window features exist;
  2. expanding walk-forward with an H-day embargo;
  3. per fold: fit bucket edges + bucket-mean returns on TRAIN ONLY, fit the
     model, predict on TEST, and reduce to the common expected-return signal;
  4. collect the stitched out-of-sample signal series and compute metrics.

Returns
-------
metrics : pd.DataFrame   one row per (horizon, model)
signals : pd.DataFrame   wide, indexed by date: one signal column per
                         (model, horizon) plus realized_h{H} columns -- the
                         input the backtest (stage 03) consumes.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.metrics import (
    classifier_ml_metrics,
    hit_rate,
    icir,
    nonoverlap_rank_ic,
    rank_ic,
    regressor_ml_metrics,
)
from src.models import build_model
from src.signal import classifier_signal, regressor_signal
from src.targets import apply_buckets, bucket_mean_returns, fit_bucket_edges
from src.validation import walk_forward_splits


def _prepare(X: pd.DataFrame, y_cont: pd.Series, warmup: int):
    """Align X and the continuous target; drop no-target rows + warm-up."""
    df = X.join(y_cont.rename("_y"))
    df = df.iloc[warmup:]                 # long-window features now populated
    df = df.dropna(subset=["_y"])         # drop trailing rows with no future
    return df.drop(columns="_y"), df["_y"]


def run_experiment(
    X: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    horizons: Mapping[str, int],
    model_names: Sequence[str],
    n_folds: int,
    min_train_frac: float,
    n_buckets: int,
    bucket_method: str,
    fixed_bins: Sequence[float],
    warmup: int,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    signal_frames: list[pd.Series] = []
    realized_by_h: dict[int, pd.Series] = {}
    labels_full = list(range(n_buckets))

    for hname, H in horizons.items():
        Xh, yh = _prepare(X, targets[f"fwd_ret_{H}"], warmup)
        splits = list(walk_forward_splits(
            len(Xh), n_folds=n_folds, embargo=H, min_train_frac=min_train_frac
        ))
        if verbose:
            print(f"[{hname} H={H}] {len(Xh)} rows, {len(splits)} folds")

        # Realized OOS forward returns (same for every model) over the test span.
        test_pos = np.concatenate([te for _, te in splits])
        realized_by_h[H] = yh.iloc[test_pos]

        for model_name in model_names:
            kind, _ = build_model(model_name)

            oos_signal = pd.Series(index=Xh.index[test_pos], dtype=float)
            fold_ics: list[float] = []
            # accumulators for pooled ML metrics
            clf_true, clf_pred = [], []
            reg_true, reg_pred = [], []
            logloss_folds: list[float] = []

            for tr, te in splits:
                Xtr, Xte = Xh.iloc[tr], Xh.iloc[te]
                ytr, yte = yh.iloc[tr], yh.iloc[te]

                edges = fit_bucket_edges(
                    ytr, n_buckets=n_buckets, method=bucket_method, fixed_bins=fixed_bins
                )

                _, model = build_model(model_name)  # fresh per fold

                if kind == "clf":
                    ytr_c = apply_buckets(ytr, edges).astype(int)
                    yte_c = apply_buckets(yte, edges).astype(int)
                    bmeans = bucket_mean_returns(ytr, ytr_c, n_buckets)

                    model.fit(Xtr, ytr_c)
                    proba = model.predict_proba(Xte)
                    sig = classifier_signal(proba, model.classes_, bmeans, n_buckets)
                    pred_c = model.predict(Xte)

                    clf_true.append(yte_c.to_numpy())
                    clf_pred.append(np.asarray(pred_c))
                    m = classifier_ml_metrics(yte_c, pred_c, proba, labels=model.classes_)
                    if "log_loss" in m and np.isfinite(m["log_loss"]):
                        logloss_folds.append(m["log_loss"])
                else:
                    model.fit(Xtr, ytr)
                    sig = regressor_signal(model.predict(Xte))
                    reg_true.append(yte.to_numpy())
                    reg_pred.append(sig.copy())

                oos_signal.iloc[te - test_pos[0]] = sig  # positions within test span
                fold_ics.append(rank_ic(sig, yte.to_numpy()))

            realized = realized_by_h[H].to_numpy()
            sig_all = oos_signal.to_numpy()

            row = {
                "horizon": hname, "H": H, "model": model_name, "kind": kind,
                "n_oos": len(oos_signal),
                "rank_ic": rank_ic(sig_all, realized),
                "rank_ic_nonoverlap": nonoverlap_rank_ic(sig_all, realized, H),
                "icir": icir(fold_ics),
                "hit_rate": hit_rate(sig_all, realized),
            }
            if kind == "clf":
                yt = np.concatenate(clf_true); yp = np.concatenate(clf_pred)
                row["balanced_accuracy"] = classifier_ml_metrics(yt, yp)["balanced_accuracy"]
                row["accuracy"] = classifier_ml_metrics(yt, yp)["accuracy"]
                row["log_loss"] = float(np.mean(logloss_folds)) if logloss_folds else np.nan
            else:
                yt = np.concatenate(reg_true); yp = np.concatenate(reg_pred)
                row.update(regressor_ml_metrics(yt, yp))
            metric_rows.append(row)

            oos_signal.name = f"{model_name}__h{H}"
            signal_frames.append(oos_signal)

    metrics = pd.DataFrame(metric_rows)

    # Wide signal frame + realized columns for the backtest.
    signals = pd.concat(signal_frames, axis=1)
    for H, r in realized_by_h.items():
        signals[f"realized__h{H}"] = r
    signals = signals.sort_index()

    return metrics, signals
