"""End-to-end offline test of src/features.py + src/targets.py on synthetic data.

Key checks: NO look-ahead (lagged feature at t == raw value at t-1), correct
forward target, per-fold bucket edges, and exclusion of raw price levels.

Run: python smoke_features.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.features import (
    build_feature_matrix, calendar_features, days_since_fomc, macro_features,
)
from src.targets import (
    build_targets, forward_return, fit_bucket_edges, apply_buckets, bucket_mean_returns,
)

rng = np.random.default_rng(0)

TICKERS = ["^GSPC", "^DJI", "^VIX", "^VVIX", "DX-Y.NYB"]
IDX = pd.bdate_range("2015-01-01", periods=400, name="date")   # weekdays only


def make_panel():
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols = pd.MultiIndex.from_product([fields, TICKERS], names=["Field", "Ticker"])
    data = {}
    for tkr in TICKERS:
        close = 100 + np.cumsum(rng.normal(0, 1, len(IDX)))
        close = np.abs(close) + 10
        data[("Close", tkr)] = close
        data[("Open", tkr)] = close * (1 + rng.normal(0, 0.001, len(IDX)))
        data[("High", tkr)] = close * (1 + np.abs(rng.normal(0, 0.003, len(IDX))))
        data[("Low", tkr)] = close * (1 - np.abs(rng.normal(0, 0.003, len(IDX))))
        data[("Volume", tkr)] = rng.integers(1e6, 5e6, len(IDX)).astype(float)
    panel = pd.DataFrame(data, index=IDX)
    panel.columns = pd.MultiIndex.from_tuples(panel.columns, names=["Field", "Ticker"])
    return panel.sort_index(axis=1)


def make_macro():
    days = pd.date_range("2014-12-01", IDX[-1], freq="D", name="date")
    series = ["DFF", "DGS2", "DGS10", "DCOILWTICO", "UNRATE", "GDP", "CORESTICKM159SFRBATL"]
    m = pd.DataFrame(index=days)
    for s in series:
        base = 2 + np.cumsum(rng.normal(0, 0.01, len(days)))
        m[s] = base
        m[f"{s}_pct_change"] = pd.Series(base, index=days).pct_change(fill_method=None)
    return m


def make_fomc():
    dates = pd.to_datetime(["2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
                            "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
                            "2016-01-27", "2016-03-16"])
    return pd.DataFrame({"date": dates, "is_fomc_day": 1, "is_fomc_press_conference": 1})


def test_no_lookahead_returns(panel, X):
    close = panel[("Close", "^GSPC")]
    raw_ret5 = close.pct_change(5, fill_method=None)      # raw Ret_5 at t
    col = X["Ret_5_lag1_^GSPC"]
    # lagged feature at t must equal raw Ret_5 at t-1
    expected = raw_ret5.shift(1).loc[col.index]
    pd.testing.assert_series_equal(col, expected, check_names=False)
    print("  ok: Ret_5_lag1[t] == raw Ret_5[t-1]  (no look-ahead)")


def test_no_lookahead_technical(panel, X):
    from src.technicals import compute_technicals
    tech, _ = compute_technicals(panel, "^GSPC", ["macd"], verbose=False)
    raw = tech["macd"]
    col = X["Technical_lag1_macd"]
    expected = raw.shift(1).loc[col.index]
    pd.testing.assert_series_equal(col.astype(float), expected.astype(float), check_names=False)
    print("  ok: Technical_lag1_macd[t] == raw macd[t-1]  (no look-ahead)")


def test_macro_daily_lagged(panel, macro, X):
    m_td = macro.reindex(panel.index)
    expected = m_td["DFF"].shift(1).loc[X.index]
    pd.testing.assert_series_equal(X["Macro_DFF_lag1"].astype(float), expected.astype(float),
                                   check_names=False)
    # low-frequency series NOT lagged
    assert "Macro_UNRATE" in X.columns and "Macro_UNRATE_lag1" not in X.columns
    print("  ok: daily FRED lagged 1 trading day; low-freq FRED unlagged")


def test_derived_and_calendar(X):
    assert "Macro_term_spread_lag1" in X.columns
    assert "Macro_inflation_gap" in X.columns
    # exactly one weekday flag set per row
    wd = X[["mon", "tues", "wed", "thurs", "fri"]].sum(axis=1)
    assert (wd == 1).all(), "each row should have exactly one weekday flag"
    print("  ok: derived macro features present; weekday one-hot valid")


def test_no_price_levels(X):
    banned = ["Open", "High", "Low", "Close", "Volume"]
    leaked = [c for c in X.columns
              if any(c.startswith(f"{b}_") or c == b for b in banned)]
    assert not leaked, f"raw price levels leaked into features: {leaked}"
    print("  ok: raw OHLCV price levels excluded from feature matrix")


def test_days_since_fomc(panel):
    fomc = make_fomc()
    ds = days_since_fomc(panel.index, fomc)
    assert (ds.dropna() >= 0).all()
    # day exactly on a decision date -> 0
    on_day = pd.Timestamp("2015-03-18")
    if on_day in ds.index:
        assert ds.loc[on_day] == 0
    print("  ok: days_since_fomc non-negative and resets on decision dates")


def test_targets(panel):
    close = panel[("Close", "^GSPC")]
    tgt = build_targets(close, {"week": 5, "month": 21})
    # forward alignment: fwd_ret_5[t] == close[t+5]/close[t]-1
    expected = (close.shift(-5) / close - 1)
    pd.testing.assert_series_equal(tgt["fwd_ret_5"], expected, check_names=False)
    assert tgt["fwd_ret_5"].iloc[-5:].isna().all(), "last H rows must be NaN (no future)"
    print("  ok: forward targets aligned; trailing H rows NaN")


def test_per_fold_buckets(panel):
    close = panel[("Close", "^GSPC")]
    r = forward_return(close, 5).dropna()
    split = int(len(r) * 0.7)
    train, test = r.iloc[:split], r.iloc[split:]

    edges = fit_bucket_edges(train, n_buckets=5, method="percentile")
    assert edges[0] == -np.inf and edges[-1] == np.inf, "edges must be open-ended"

    # an out-of-sample extreme beyond the training range still buckets (not NaN)
    extreme = pd.Series([train.max() * 10, train.min() * 10])
    lab = apply_buckets(extreme, edges)
    assert lab.iloc[0] == 4 and lab.iloc[1] == 0, "extremes must land in end buckets"

    labels_train = apply_buckets(train, edges)
    means = bucket_mean_returns(train, labels_train, 5)
    assert np.all(np.diff(means) >= -1e-9), "bucket mean returns should be monotone"
    print("  ok: bucket edges fit on train only, open-ended; monotone bucket means")


if __name__ == "__main__":
    panel = make_panel()
    macro = make_macro()
    fomc = make_fomc()

    X = build_feature_matrix(
        panel, macro, fomc,
        sp_ticker="^GSPC", start="2015-01-01", end="2016-12-31",
        return_windows=[1, 5, 10, 20, 60, 120], vol_windows=[5, 10, 20, 60],
        technicals=["macd", "rsi_14", "atr_14", "boll_ub", "close_20_z"],
        use_technicals=True, use_fomc=True, verbose=True,
    )

    print("no-look-ahead:")
    test_no_lookahead_returns(panel, X)
    test_no_lookahead_technical(panel, X)
    test_macro_daily_lagged(panel, macro, X)
    print("feature content:")
    test_derived_and_calendar(X)
    test_no_price_levels(X)
    test_days_since_fomc(panel)
    print("targets:")
    test_targets(panel)
    test_per_fold_buckets(panel)
    print("\nALL FEATURE SMOKE TESTS PASSED")
