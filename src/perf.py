"""
Performance statistics for a (daily) strategy return series.

All annualised figures assume `periods_per_year` trading periods (252 for daily).
Risk-free rate defaults to 0. These describe realised P&L; prediction-quality
metrics (IC/ICIR) live in `metrics.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak (<= 0)."""
    peak = equity.cummax()
    return equity / peak - 1.0


def performance_summary(
    ret: pd.Series,
    *,
    periods_per_year: int = 252,
    rf: float = 0.0,
    turnover: pd.Series | None = None,
    position: pd.Series | None = None,
) -> dict:
    """Summary stats for a return series.

    Optional `turnover` / `position` add trading diagnostics (avg turnover,
    average gross exposure, number of position changes).
    """
    ret = ret.dropna()
    if ret.empty:
        return {}

    equity = (1.0 + ret).cumprod()
    years = len(ret) / periods_per_year
    ann_ret = ret.mean() * periods_per_year
    ann_vol = ret.std(ddof=1) * np.sqrt(periods_per_year)
    downside = ret[ret < 0].std(ddof=1) * np.sqrt(periods_per_year)
    dd = drawdown(equity)
    max_dd = float(dd.min())
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan

    out = {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": cagr,
        "ann_vol": float(ann_vol),
        "sharpe": float((ann_ret - rf) / ann_vol) if ann_vol > 0 else np.nan,
        "sortino": float((ann_ret - rf) / downside) if downside > 0 else np.nan,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else np.nan,
        "hit_rate": float((ret > 0).mean()),
    }
    if turnover is not None:
        t = turnover.reindex(ret.index).fillna(0.0)
        out["avg_turnover"] = float(t.mean())
        out["n_trades"] = int((t.abs() > 1e-9).sum())
    if position is not None:
        out["avg_gross_exposure"] = float(position.reindex(ret.index).abs().mean())
    return out
