"""
Raw FRED macro acquisition.

Downloads each requested FRED series, reindexes to a continuous daily calendar
and forward-fills (so weekly/monthly/quarterly releases carry forward to every
trading day), and appends a ``{series}_pct_change`` column.

RAW only: lagging is intentionally NOT done here. Look-ahead handling (lagging
the daily series DFF/DGS2/DGS10/DCOILWTICO by one trading day, leaving the
low-frequency series unlagged) belongs to the feature layer, so it lives in one
place next to every other lag. See the module docstring in
``src/feature_utils.py`` for the rationale.

`pyfredapi` is imported lazily so this module can be imported without the
package or an API key present.
"""

from __future__ import annotations

from functools import reduce
from time import sleep
from typing import Sequence

import pandas as pd


def _load_series(series_id: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    import pyfredapi as pf  # deferred: network + optional dependency

    raw = pf.get_series(
        series_id=series_id,
        observation_start=start,
        observation_end=end,
        api_key=api_key,
    )
    s = (
        raw[["date", "value"]]
        .rename(columns={"value": series_id})
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .set_index("date")
        .sort_index()
    )
    s[f"{series_id}_pct_change"] = s[series_id].pct_change(fill_method=None)
    return s


def download_fred(
    series_ids: Sequence[str],
    *,
    start: str,
    end: str,
    api_key: str,
    pause: float = 0.5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download FRED series into one daily, forward-filled wide DataFrame.

    Parameters
    ----------
    series_ids
        FRED series identifiers (e.g. ``["DFF", "DGS10", "UNRATE"]``).
    start, end
        Observation window. Use a `start` a little before the model start so
        forward-fill has values to carry into the first modelled dates.
    api_key
        FRED API key.
    pause
        Seconds to sleep between requests (gentle rate limiting).
    """
    frames = []
    for sid in series_ids:
        if verbose:
            print(f"[fred] downloading {sid} ...")
        frames.append(_load_series(sid, start, end, api_key))
        sleep(pause)

    macro = reduce(lambda left, right: left.join(right, how="outer"), frames)

    daily_index = pd.date_range(start, end, freq="D")
    macro = macro.reindex(daily_index).ffill()
    macro.index.name = "date"
    return macro
