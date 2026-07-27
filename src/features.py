"""
Feature engineering: turn the raw caches into a prediction-time-safe matrix.

Lagging discipline (all applied here, in one place):
  * MARKET-DERIVED (lag 1 trading day): rolling returns, rolling volatility,
    technical indicators, and the daily FRED series. These include information
    through day T's close, which is not available before the market opens on
    day T.
  * CALENDAR / SCHEDULED (no lag): day-of-week, holiday flags, days_since_fomc.
    Known in advance.
  * LOWER-FREQUENCY FRED (no lag): weekly/monthly/quarterly releases. An
    intentional, documented approximation (proper handling needs ALFRED
    vintages).

The target is NOT built here; see `targets.py`. Buckets are fit per-fold in the
model stage, never on the whole sample.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src.feature_utils import create_lagged_features
from src.technicals import compute_technicals

WEEKDAYS = ["mon", "tues", "wed", "thurs", "fri"]

# Daily FRED series that must be lagged by one trading day. Everything else in
# the macro frame is treated as lower-frequency and left unlagged.
DAILY_FRED = ["DFF", "DGS2", "DGS10", "DCOILWTICO"]


# --------------------------------------------------------------------------- #
# Calendar / holiday (not lagged)
# --------------------------------------------------------------------------- #
def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Day-of-week dummies (Mon=0 .. Fri=4)."""
    wd = index.dayofweek
    out = pd.DataFrame(index=index)
    for i, name in enumerate(WEEKDAYS):
        out[name] = (wd == i).astype("int8")
    return out


def holiday_features(index: pd.DatetimeIndex, start: str, end: str) -> pd.DataFrame:
    """`on_holiday` (a US federal holiday that is still a trading day) plus the
    last trading day before (`pre_holiday`) and first after (`post_holiday`).
    """
    holidays = USFederalHolidayCalendar().holidays(start=start, end=end)
    out = pd.DataFrame(
        0, index=index,
        columns=["on_holiday", "pre_holiday", "post_holiday"], dtype="int8",
    )
    out.loc[index.isin(holidays), "on_holiday"] = 1
    for h in holidays:
        prev = index[index < h]
        if len(prev):
            out.loc[prev[-1], "pre_holiday"] = 1
        nxt = index[index > h]
        if len(nxt):
            out.loc[nxt[0], "post_holiday"] = 1
    return out


def days_since_fomc(index: pd.DatetimeIndex, fomc: pd.DataFrame) -> pd.Series:
    """Calendar days since the most recent FOMC decision (`is_fomc_day == 1`).

    Uses a backward `merge_asof`, so every trading day maps to the last decision
    on or before it. Not lagged: meeting dates are published in advance.
    """
    ann = (
        fomc.loc[fomc["is_fomc_day"] == 1, ["date"]]
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .drop_duplicates()
        .rename(columns={"date": "last_fomc_date"})
        .sort_values("last_fomc_date")
    )
    tmp = pd.DataFrame({"date": pd.to_datetime(index)}).sort_values("date")
    tmp = pd.merge_asof(
        tmp, ann, left_on="date", right_on="last_fomc_date", direction="backward"
    )
    tmp["days_since_fomc"] = (tmp["date"] - tmp["last_fomc_date"]).dt.days
    return tmp.set_index("date")["days_since_fomc"]


# --------------------------------------------------------------------------- #
# Rolling market features (lagged)
# --------------------------------------------------------------------------- #
def rolling_return_features(close: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    """`Ret_{w}` percentage returns per ticker, MultiIndex ``(Ret_w, Ticker)``."""
    frames = []
    for w in windows:
        r = close.pct_change(w, fill_method=None)
        r.columns = pd.MultiIndex.from_product([[f"Ret_{w}"], r.columns])
        frames.append(r)
    return pd.concat(frames, axis=1)


def rolling_vol_features(daily_ret: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    """`Vol_{w}` rolling std of daily returns per ticker, ``(Vol_w, Ticker)``."""
    frames = []
    for w in windows:
        v = daily_ret.rolling(w).std()
        v.columns = pd.MultiIndex.from_product([[f"Vol_{w}"], v.columns])
        frames.append(v)
    return pd.concat(frames, axis=1)


# --------------------------------------------------------------------------- #
# Macro (daily lagged, low-frequency unlagged)
# --------------------------------------------------------------------------- #
def macro_features(
    macro: pd.DataFrame,
    index: pd.DatetimeIndex,
    daily_fred: Sequence[str] = DAILY_FRED,
) -> pd.DataFrame:
    """Assemble macro predictors on the trading-day index.

    Daily series (and their pct_change) are lagged by one trading day. All other
    (lower-frequency) columns are carried as-is. Two derived, prediction-safe
    features are added: the 10y-2y term spread and the core-inflation gap to 2%.
    """
    m = macro.reindex(index)  # daily, forward-filled -> sample onto trading days
    out = pd.DataFrame(index=index)

    daily_set = set(daily_fred)
    for s in daily_fred:
        if s in m.columns:
            out[f"{s}_lag1"] = m[s].shift(1)
        pc = f"{s}_pct_change"
        if pc in m.columns:
            out[f"{pc}_lag1"] = m[pc].shift(1)

    for c in m.columns:
        base = str(c).split("_pct_change")[0]
        if base in daily_set:
            continue  # handled (lagged) above; skip the unlagged daily level
        out[c] = m[c]

    # Derived, prediction-safe macro features.
    if {"DGS10_lag1", "DGS2_lag1"}.issubset(out.columns):
        out["term_spread_lag1"] = out["DGS10_lag1"] - out["DGS2_lag1"]
    if "CORESTICKM159SFRBATL" in out.columns:
        out["inflation_gap"] = out["CORESTICKM159SFRBATL"] - 2.0

    return out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _lag_multiindex(frame: pd.DataFrame) -> pd.DataFrame:
    """Lag every column of a MultiIndex frame by 1 and return only the lagged
    columns (``(name, tkr)`` -> ``(name_lag1, tkr)``)."""
    cols = list(frame.columns)
    lagged = create_lagged_features(frame, cols, lag_period=1, multiindex=True)
    lagged_cols = [(f"{a}_lag1", b) for (a, b) in cols]
    return lagged[lagged_cols]


def _wrap_level0(frame: pd.DataFrame, level0: str) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = pd.MultiIndex.from_product([[level0], frame.columns])
    return frame


def _wrap_empty_second_level(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = pd.MultiIndex.from_product([frame.columns, [""]])
    return frame


def build_feature_matrix(
    panel: pd.DataFrame,
    macro: pd.DataFrame,
    fomc: Optional[pd.DataFrame],
    *,
    sp_ticker: str,
    start: str,
    end: str,
    return_windows: Sequence[int],
    vol_windows: Sequence[int],
    technicals: Sequence[str],
    use_technicals: bool = True,
    use_fomc: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the flat, prediction-time-safe feature matrix.

    Returns a DataFrame indexed by trading day (from `start`) with flattened
    single-level column names, e.g. ``Ret_5_lag1_^GSPC``, ``Technical_lag1_macd``,
    ``mon``, ``days_since_fomc``, ``Macro_DFF_lag1``, ``Macro_term_spread_lag1``.
    Raw OHLCV price levels are intentionally excluded (non-stationary; they enter
    only through returns/volatility/technicals and define the target).
    """
    index = pd.DatetimeIndex(panel.index)
    close = panel["Close"]                       # DataFrame: columns = tickers
    daily_ret = close.pct_change(fill_method=None)

    groups: list[pd.DataFrame] = []

    # Market-derived, lagged.
    ret = rolling_return_features(close, return_windows)
    vol = rolling_vol_features(daily_ret, vol_windows)
    groups.append(_lag_multiindex(pd.concat([ret, vol], axis=1)))

    if use_technicals:
        tech, _ = compute_technicals(panel, sp_ticker, technicals, verbose=verbose)
        groups.append(_lag_multiindex(_wrap_level0(tech, "Technical")))

    # Calendar / scheduled, not lagged.
    groups.append(_wrap_empty_second_level(calendar_features(index)))
    groups.append(_wrap_empty_second_level(holiday_features(index, start, end)))
    if use_fomc and fomc is not None:
        ds = days_since_fomc(index, fomc).to_frame("days_since_fomc")
        groups.append(_wrap_empty_second_level(ds))

    # Macro.
    groups.append(_wrap_level0(macro_features(macro, index), "Macro"))

    X = pd.concat(groups, axis=1)
    X = X.loc[pd.Timestamp(start):]

    # Flatten to single-level column names.
    X.columns = [f"{a}_{b}".rstrip("_") if b != "" else str(a) for a, b in X.columns]
    X = X.dropna(axis=1, how="all")

    if verbose:
        print(f"[features] matrix: {X.shape[0]} rows x {X.shape[1]} cols "
              f"({X.index.min().date()} -> {X.index.max().date()})")
    return X
