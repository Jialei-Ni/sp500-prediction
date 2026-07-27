"""
StockStats technical indicators.

`compute_technicals` pulls one ticker's OHLCV out of the raw price panel, runs
it through stockstats, and returns a DataFrame of the requested indicators
aligned to the panel index. Indicators that fail (e.g. an unknown name in a
given stockstats version) are skipped with a warning, matching the original
notebook's tolerant behaviour.

These indicators use data through day T's close, so they are lagged by one
trading day in the feature layer before being used as predictors.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def compute_technicals(
    panel: pd.DataFrame,
    ticker: str,
    indicators: Sequence[str],
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (indicator_frame, computed_names) for `ticker` from `panel`.

    `panel` must have MultiIndex columns ``(Field, Ticker)`` with Open/High/Low/
    Close/Volume present for `ticker`.
    """
    import stockstats  # deferred optional dependency

    ohlcv = pd.DataFrame(
        {
            "open": panel[("Open", ticker)],
            "high": panel[("High", ticker)],
            "low": panel[("Low", ticker)],
            "close": panel[("Close", ticker)],
            "volume": panel[("Volume", ticker)],
        }
    )

    sdf = stockstats.StockDataFrame.retype(ohlcv)

    columns: dict[str, pd.Series] = {}
    computed: list[str] = []
    for feat in indicators:
        try:
            columns[feat] = pd.Series(sdf[feat].to_numpy(), index=panel.index)
            computed.append(feat)
        except Exception as exc:  # noqa: BLE001 - stockstats raises various errors
            if verbose:
                print(f"[technicals] skipping {feat!r}: {exc}")

    result = pd.DataFrame(columns, index=panel.index)
    if verbose:
        print(f"[technicals] computed {len(computed)}/{len(indicators)} indicators for {ticker}")
    return result, computed
