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

import re
from typing import Optional, Union

import numpy as np
import pandas as pd


def roll_dates(index: pd.DatetimeIndex,
               roll: Union[str, int, None]) -> pd.DatetimeIndex:
    """Roll (cost) dates within `index`.

    roll may be:
      - None / "none" / False -> no rolls
      - "quarterly" / "q"     -> last trading day of Mar/Jun/Sep/Dec (ES/MES cycle)
      - "monthly"   / "m"     -> last trading day of each month
      - "weekly"    / "w"     -> last trading day of each ISO week
      - int N  or  "<N>d"     -> every N trading days
    The opening day of `index` is never a roll (that is the entry).
    """
    if roll is None or roll is False:
        return pd.DatetimeIndex([])
    idx = pd.DatetimeIndex(index)

    def _last_per(sub, keys):
        if len(sub) == 0:
            return pd.DatetimeIndex([])
        s = pd.Series(sub, index=sub)
        return pd.DatetimeIndex(sorted(s.groupby(list(keys)).max().values))

    if isinstance(roll, (int, np.integer)) and not isinstance(roll, bool):
        if int(roll) < 1:
            raise ValueError("integer roll must be >= 1")
        dates = idx[::int(roll)]
    else:
        r = str(roll).strip().lower()
        if r in ("none", ""):
            return pd.DatetimeIndex([])
        elif r in ("quarterly", "q"):
            sub = idx[idx.month.isin([3, 6, 9, 12])]
            dates = _last_per(sub, [sub.year, sub.month])
        elif r in ("monthly", "m"):
            dates = _last_per(idx, [idx.year, idx.month])
        elif r in ("weekly", "w"):
            iso = idx.isocalendar()
            dates = _last_per(idx, [iso.year.to_numpy(), iso.week.to_numpy()])
        else:
            m = re.fullmatch(r"(\d+)\s*d", r)
            if not m:
                raise ValueError(f"unrecognised roll spec: {roll!r}")
            dates = idx[::int(m.group(1))]
    return dates[dates != idx[0]]

def _direction(signal: pd.Series, positioning: str, deadband: float) -> pd.Series:
    if positioning == "long_flat":
        d = np.where(signal > deadband, 1.0, 0.0)
    elif positioning == "long_short_flat":
        d = np.where(signal > deadband, 1.0, np.where(signal < -deadband, -1.0, 0.0))
    elif positioning == "short_flat":
        d = np.where(signal < deadband, -1.0, 0.0)
        pass
    else:
        raise ValueError("positioning must be 'long_short_flat', 'long_flat', or 'short_flat'")
    return pd.Series(d, index=signal.index)


def make_positions(
    signal: pd.Series,
    daily_ret: pd.Series,
    *,
    H: int,
    hold: Optional[int] = None,
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
    
    hold = H if hold is None else int(hold)
    if hold < 1:
        raise ValueError("hold must be >= 1")

    # Refresh only on rebalance dates (every `hold`-th trading day); hold in between.
    rebal = idx[::hold]
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

    if roll_cost_bps and roll not in (None, False, "none"):
        rdates = roll_dates(daily_ret.index, roll)
        rc = pd.Series(0.0, index=daily_ret.index)
        hit = daily_ret.index.intersection(rdates)
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
