"""Offline test of stage 03: backtest engine, perf stats, and figures render.

Run: python smoke_backtest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backtest import run_backtest, buy_and_hold, make_positions, quarterly_roll_dates
from src.perf import performance_summary, drawdown
from src.viz import (
    plot_equity, plot_drawdown, plot_monthly_heatmap, plot_yearly_bars,
    monthly_return_table,
)

rng = np.random.default_rng(11)
N = 252 * 6
IDX = pd.bdate_range("2016-01-01", periods=N, name="date")
# daily asset returns with mild autocorrelation-driven predictability
asset = pd.Series(rng.normal(0.0003, 0.011, N), index=IDX)
# a signal that is (noisily) correlated with the NEXT H-day forward return
H = 21
fwd = asset.rolling(H).sum().shift(-H)
signal = (0.5 * fwd + rng.normal(0, 0.02, N)).reindex(IDX)


def test_positions_and_no_lookahead():
    pos = make_positions(signal, asset, H=H, positioning="long_short_flat",
                         sizing="vol_target", target_vol=0.10, vol_lookback=20)
    # position refreshes only every H days
    changes = pos.diff().abs() > 1e-12
    change_dates = pos.index[changes]
    # all changes should fall on rebalance dates (every H-th index)
    rebal = set(pos.index[::H])
    assert all(d in rebal for d in change_dates), "position changed off-rebalance"
    print(f"  ok: position piecewise-constant, refreshes only on {len(rebal)} rebalance dates")


def test_costs_reduce_return():
    bt0 = run_backtest(signal, asset, H=H, cost_bps=0)
    bt1 = run_backtest(signal, asset, H=H, cost_bps=20)
    assert bt1["strat_ret"].sum() < bt0["strat_ret"].sum(), "costs should reduce return"
    assert (bt1["cost"] >= 0).all()
    # cost only on rebalance/roll days -> mostly zero
    assert (bt1["cost"] == 0).mean() > 0.9
    print("  ok: transaction costs reduce net return and are sparse (rebalance-only)")


def test_vol_target_in_range():
    bt = run_backtest(signal, asset, H=H, cost_bps=0,
                      sizing="vol_target", target_vol=0.10, vol_lookback=20,
                      max_leverage=3.0)
    realized_vol = bt["strat_ret"].std() * np.sqrt(252)
    # long/short/flat + deadband means realized <= target-ish, but same ballpark
    assert 0.0 < realized_vol < 0.20, f"realized vol {realized_vol:.3f} out of range"
    print(f"  ok: vol-target realized annual vol = {realized_vol:.1%} (target 10%)")


def test_perf_summary():
    bt = run_backtest(signal, asset, H=H, cost_bps=20)
    s = performance_summary(bt["strat_ret"], turnover=bt["turnover"], position=bt["position"])
    for k in ["cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar",
              "hit_rate", "avg_turnover", "n_trades"]:
        assert k in s, f"missing perf key {k}"
    assert s["max_drawdown"] <= 0
    assert -1 < s["max_drawdown"] < 0
    print(f"  ok: perf summary complete (Sharpe {s['sharpe']:.2f}, maxDD {s['max_drawdown']:.1%})")


def test_benchmark_and_roll():
    bh = buy_and_hold(asset, cost_bps=20)
    assert (bh["strat_equity"] > 0).all()
    rd = quarterly_roll_dates(IDX)
    assert len(rd) >= 20  # ~4/yr over 6y
    assert all(d.month in (3, 6, 9, 12) for d in rd)
    print(f"  ok: buy&hold benchmark built; {len(rd)} quarterly roll dates")


def test_figures_render(tmp: Path):
    bt = run_backtest(signal, asset, H=H, cost_bps=20)
    bh = buy_and_hold(asset, cost_bps=20)

    tbl = monthly_return_table(bt["strat_ret"])
    assert tbl.shape[1] == 12 and set(tbl.columns[:3]) == {"Jan", "Feb", "Mar"}

    figs = []
    for name, fn in [
        ("equity", lambda ax: plot_equity(bt["strat_equity"], bh["strat_equity"], ax=ax)),
        ("drawdown", lambda ax: plot_drawdown(bt["strat_equity"], ax=ax)),
        ("heatmap", lambda ax: plot_monthly_heatmap(bt["strat_ret"], ax=ax)),
        ("yearly", lambda ax: plot_yearly_bars(bt["strat_ret"], ax=ax)),
    ]:
        fig, ax = plt.subplots(figsize=(10, 4))
        fn(ax)
        p = tmp / f"{name}.png"
        fig.savefig(p, dpi=70, bbox_inches="tight")
        plt.close(fig)
        assert p.exists() and p.stat().st_size > 1000, f"{name} figure did not render"
        figs.append(name)
    print(f"  ok: rendered figures -> {figs}")


if __name__ == "__main__":
    import tempfile
    print("positions:")
    test_positions_and_no_lookahead()
    print("costs / sizing:")
    test_costs_reduce_return()
    test_vol_target_in_range()
    print("performance:")
    test_perf_summary()
    test_benchmark_and_roll()
    print("figures:")
    with tempfile.TemporaryDirectory() as d:
        test_figures_render(Path(d))
    print("\nALL BACKTEST SMOKE TESTS PASSED")
