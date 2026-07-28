"""
Backtest visuals: equity curve, drawdown, monthly-return heatmap (fixed +/-10%
color scale), and yearly-return bars. Each function draws onto a Matplotlib Axes
(creating one if not supplied) and returns it, so the notebook controls layout.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from src.perf import drawdown

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _compound(ret: pd.Series, freq: str) -> pd.Series:
    return (1.0 + ret.dropna()).resample(freq).prod() - 1.0


def plot_equity(
    strat_equity: pd.Series,
    bench_equity: Optional[pd.Series] = None,
    *,
    ax: Optional[plt.Axes] = None,
    logy: bool = True,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))
    ax.plot(strat_equity.index, strat_equity.values, label="Strategy", lw=1.6, color="#1b4965")
    if bench_equity is not None:
        ax.plot(bench_equity.index, bench_equity.values, label="Buy & hold",
                lw=1.2, color="#9a9a9a", alpha=0.9)
    if logy:
        ax.set_yscale("log")
    ax.set_title("Equity curve (growth of 1)")
    ax.set_ylabel("equity (log)" if logy else "equity")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    return ax


def plot_drawdown(
    strat_equity: pd.Series, *, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3))
    dd = drawdown(strat_equity)
    ax.fill_between(dd.index, dd.values, 0.0, color="#d1495b", alpha=0.5)
    ax.set_title(f"Drawdown (max {dd.min():.1%})")
    ax.set_ylabel("drawdown")
    ax.grid(alpha=0.25)
    return ax


def monthly_return_table(ret: pd.Series) -> pd.DataFrame:
    """Year x month table of compounded monthly returns."""
    m = _compound(ret, "ME")
    tbl = pd.DataFrame({
        "year": m.index.year, "month": m.index.month, "ret": m.values,
    }).pivot(index="year", columns="month", values="ret")
    tbl = tbl.reindex(columns=range(1, 13))
    tbl.columns = _MONTHS
    return tbl


def plot_monthly_heatmap(
    ret: pd.Series,
    *,
    ax: Optional[plt.Axes] = None,
    vmin: float = -0.10,
    vmax: float = 0.10,
) -> plt.Axes:
    """Monthly-return heatmap with a FIXED diverging color scale (default +/-10%)."""
    tbl = monthly_return_table(ret)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 0.5 * len(tbl) + 1.5))

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    data = tbl.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(12), _MONTHS)
    ax.set_yticks(range(len(tbl)), tbl.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v*100:.1f}", ha="center", va="center",
                        fontsize=7, color="black")
    ax.set_title(f"Monthly returns (%) — color fixed to [{vmin:.0%}, {vmax:.0%}]")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("monthly return")
    return ax


def plot_yearly_bars(ret: pd.Series, *, ax: Optional[plt.Axes] = None) -> plt.Axes:
    y = _compound(ret, "YE")
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5))
    colors = ["#3b7dd8" if v >= 0 else "#d1495b" for v in y.values]
    ax.bar(y.index.year, y.values * 100, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    for x, v in zip(y.index.year, y.values):
        ax.text(x, v * 100, f"{v*100:.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_title("Yearly returns (%)")
    ax.set_ylabel("return (%)")
    ax.grid(alpha=0.25, axis="y")
    return ax


def plot_position(
    position: pd.Series,
    *,
    ax: Optional[plt.Axes] = None,
    direction_only: bool = False,
    max_leverage: Optional[float] = None,
) -> plt.Axes:
    """Signed exposure over time.

    Long stretches are shaded above the zero line, short stretches below it, and
    flat stretches collapse onto the zero line. Because the backtest position is
    piecewise-constant (refreshed every H days and held in between), the series
    is drawn as steps.

    By default the *actual* vol-targeted exposure is plotted, so leverage is
    visible. Pass ``direction_only=True`` to collapse it to long/flat/short at
    +1/0/-1 (a pure regime strip).
    """
    pos = position.dropna()
    if direction_only:
        pos = np.sign(pos)

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 2.5))

    idx = pos.index
    vals = pos.to_numpy(dtype=float)

    long_c, short_c, flat_c = "#2a9d8f", "#d1495b", "#c9c9c9"

    ax.fill_between(idx, vals, 0.0, where=vals > 0, step="post",
                    color=long_c, alpha=0.65, linewidth=0)
    ax.fill_between(idx, vals, 0.0, where=vals < 0, step="post",
                    color=short_c, alpha=0.65, linewidth=0)
    ax.step(idx, vals, where="post", color="#333333", lw=0.6)
    ax.axhline(0.0, color="k", lw=0.8)

    if max_leverage is not None and not direction_only:
        ax.axhline(max_leverage, color="#555555", lw=0.7, ls="--", alpha=0.6)
        ax.axhline(-max_leverage, color="#555555", lw=0.7, ls="--", alpha=0.6)

    if direction_only:
        ax.set_yticks([-1, 0, 1], ["short", "flat", "long"])
        ax.set_ylim(-1.5, 1.5)
        ax.set_title("Position (direction)")
    else:
        ax.set_ylabel("exposure (× equity)")
        ax.set_title("Position (signed exposure)")

    long_frac = float((vals > 0).mean())
    short_frac = float((vals < 0).mean())
    flat_frac = float((vals == 0).mean())
    ax.legend(handles=[
        Patch(facecolor=long_c, alpha=0.65, label=f"long  ({long_frac:.0%})"),
        Patch(facecolor=short_c, alpha=0.65, label=f"short ({short_frac:.0%})"),
        Patch(facecolor=flat_c, label=f"flat  ({flat_frac:.0%})"),
    ], loc="upper left", ncol=3, fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25, axis="y")
    return ax
