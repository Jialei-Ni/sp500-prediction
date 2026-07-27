"""
Backtest engine (index-timing on the S&P via a futures-style position).

The expected-return `signal` (from stage 02, prediction-safe) sets a position
that is refreshed every H trading days and held constant in between, so the
holding period matches the signal horizon without overlapping bets. The position
is applied to the *daily* index return (lagged one day so nothing is earned on
information not yet actionable), which yields a daily equity curve.

Sizing / direction
------------------
- direction: long/short/flat (or long/flat) from the sign of the signal, with an
  optional deadband;
- size: volatility-targeted -- exposure = target_vol / trailing_asset_vol, capped
  at max_leverage -- so risk is roughly constant through calm and turbulent
  regimes; optionally scaled by signal conviction.

Costs
-----
- `cost_bps` charged on turnover each time the position changes (per side);
- optional `roll_cost_bps` charged on |position| on quarterly roll dates.

Everything is in % of equity, so the specific contract (ES vs MES) is cosmetic.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def quarterly_roll_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of Mar/Jun/Sep/Dec within `index` (roll proxy)."""
    s = pd.Series(index, index=index)
    is_roll_month = index.month.isin([3, 6, 9, 12])
    monthly_last = s[is_roll_month].groupby(
        [index[is_roll_month].year, index[is_roll_month].month]
    ).max()
    return pd.DatetimeIndex(sorted(monthly_last.values))


def _direction(signal: pd.Series, positioning: str, deadband: float) -> pd.Series:
    if positioning == "long_flat":
        d = np.where(signal > deadband, 1.0, 0.0)
    elif positioning == "long_short_flat":
        d = np.where(signal > deadband, 1.0, np.where(signal < -deadband, -1.0, 0.0))
    else:
        raise ValueError("positioning must be 'long_short_flat' or 'long_flat'")
    return pd.Series(d, index=signal.index)


def make_positions(
    signal: pd.Series,
    daily_ret: pd.Series,
    *,
    H: int,
    positioning: str = "long_short_flat",
    sizing: str = "vol_target",
    target_vol: float = 0.10,
    max_leverage: float = 3.0,
    deadband: float = 0.0,
    vol_lookback: int = 20,
    conviction: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """Piecewise-constant position series, refreshed every H trading days."""
    idx = daily_ret.index
    sig = signal.reindex(idx)

    direction = _direction(sig.fillna(0.0), positioning, deadband)

    if sizing == "vol_target":
        asset_vol = daily_ret.rolling(vol_lookback).std().shift(1) * np.sqrt(periods_per_year)
        size = (target_vol / asset_vol).clip(upper=max_leverage)
    elif sizing == "fixed":
        size = pd.Series(1.0, index=idx)
    else:
        raise ValueError("sizing must be 'vol_target' or 'fixed'")
    size = size.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    target = direction * size
    if conviction:
        scale = sig.abs().expanding().std().shift(1)
        w = (sig.abs() / scale).clip(upper=1.0).fillna(0.0)
        target = target * w

    # Refresh only on rebalance dates (every H-th trading day); hold in between.
    rebal = idx[::H]
    pos = pd.Series(np.nan, index=idx)
    pos.loc[rebal] = target.loc[rebal]
    return pos.ffill().fillna(0.0)


def run_backtest(
    signal: pd.Series,
    daily_ret: pd.Series,
    *,
    H: int,
    cost_bps: float = 20.0,
    roll: Optional[str] = "quarterly",
    roll_cost_bps: float = 0.0,
    **pos_kwargs,
) -> pd.DataFrame:
    """Run the strategy and return a per-day frame (positions, costs, returns, equity)."""
    daily_ret = daily_ret.dropna()
    pos = make_positions(signal, daily_ret, H=H, **pos_kwargs)

    # Position set at day t is earned on day t+1's return (no same-day look-ahead).
    strat_gross = pos.shift(1).fillna(0.0) * daily_ret

    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])
    cost = turnover * (cost_bps / 1e4)

    if roll == "quarterly" and roll_cost_bps:
        roll_dates = quarterly_roll_dates(daily_ret.index)
        rc = pd.Series(0.0, index=daily_ret.index)
        hit = daily_ret.index.intersection(roll_dates)
        rc.loc[hit] = pos.abs().loc[hit] * (roll_cost_bps / 1e4)
        cost = cost + rc

    strat_ret = strat_gross - cost
    out = pd.DataFrame({
        "asset_ret": daily_ret,
        "position": pos,
        "turnover": turnover,
        "cost": cost,
        "strat_gross": strat_gross,
        "strat_ret": strat_ret,
    })
    out["strat_equity"] = (1.0 + out["strat_ret"]).cumprod()
    return out


def buy_and_hold(daily_ret: pd.Series, *, cost_bps: float = 20.0) -> pd.DataFrame:
    """Long-only benchmark with a single entry cost."""
    daily_ret = daily_ret.dropna()
    cost = pd.Series(0.0, index=daily_ret.index)
    cost.iloc[0] = cost_bps / 1e4
    ret = daily_ret - cost
    return pd.DataFrame({
        "asset_ret": daily_ret,
        "strat_ret": ret,
        "strat_equity": (1.0 + ret).cumprod(),
    })
