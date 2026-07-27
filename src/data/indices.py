"""
Raw price-panel acquisition (yfinance).

Responsibilities (RAW only -- no returns / volatility / technicals here):
  1. Download OHLCV for the "primary" US tickers (target + US indices), which
     share the NYSE trading calendar.
  2. Download "reference" series (VIX/VVIX, DXY, international indices) that
     trade on other calendars, then forward-fill them onto the primary calendar
     so every row is aligned.
  3. Normalise yfinance's column shape to a consistent 2-level MultiIndex
     ``(Field, Ticker)`` regardless of single-vs-multi-ticker quirks.
  4. Apply the VVIX availability gate (VVIX history only starts ~2007).

The heavy `yfinance` import is deferred into `download_price_panel` so this
module can be imported (and its pure helpers unit-tested) without the package
installed or any network access.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

# OHLCV field names yfinance may emit (auto_adjust=True drops "Adj Close").
_OHLCV_FIELDS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

COL_NAMES = ["Field", "Ticker"]


def _field_level(columns: pd.MultiIndex) -> Optional[int]:
    """Return which MultiIndex level (0 or 1) holds the OHLCV field names."""
    for level in (0, 1):
        values = set(map(str, columns.get_level_values(level)))
        if values & _OHLCV_FIELDS:
            return level
    return None


def normalize_yf_columns(
    df: pd.DataFrame,
    tickers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Coerce a yfinance frame to MultiIndex columns ``(Field, Ticker)``.

    Handles the three shapes yfinance produces across versions / call styles:
      * MultiIndex (Field, Ticker)  -> used as-is
      * MultiIndex (Ticker, Field)  -> levels swapped  (group_by="ticker")
      * flat columns of fields      -> wrapped with the single ticker
    """
    df = df.copy()
    cols = df.columns

    if isinstance(cols, pd.MultiIndex):
        level = _field_level(cols)
        if level is None:
            raise ValueError(
                "Could not locate an OHLCV field level in yfinance columns: "
                f"{list(cols)[:4]}..."
            )
        if level == 1:
            df.columns = df.columns.swaplevel(0, 1)
        # level 0 is now the Field level
    else:
        # Flat columns => a single-ticker download. Wrap into (Field, Ticker).
        if not tickers or len(tickers) != 1:
            raise ValueError(
                "Flat (non-MultiIndex) columns but the number of tickers is not "
                f"exactly 1 (got {tickers!r}); cannot disambiguate."
            )
        df.columns = pd.MultiIndex.from_product([df.columns, [tickers[0]]])

    df.columns = df.columns.set_names(COL_NAMES)
    return df.sort_index(axis=1)


def assemble_panel(
    df_primary: pd.DataFrame,
    df_ref: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Align reference series onto the primary trading calendar and join.

    `df_ref` is expanded onto the union of both indices, forward-filled (so a
    foreign market's last close carries into US trading days it is missing),
    then restricted back to the primary index and left-joined.
    """
    panel = df_primary.sort_index()

    if df_ref is not None and not df_ref.empty:
        master = df_primary.index.union(df_ref.index)
        df_ref_aligned = (
            df_ref.sort_index()
            .reindex(master)
            .ffill()
            .loc[df_primary.index]
        )
        panel = panel.join(df_ref_aligned, how="left")

    panel = panel.sort_index(axis=1).sort_index()
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"
    return panel


def download_price_panel(
    *,
    primary: Sequence[str],
    reference: Sequence[str] = (),
    start: str,
    end: str,
    ref_start: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download and assemble the full raw price panel.

    `primary` share the target's (NYSE) calendar; `reference` series may trade
    on other calendars and are forward-filled onto the primary index. Download
    reference series from `ref_start` (a little before `start`) so early rows
    have a value to forward-fill from.
    """
    import yfinance as yf  # deferred: network + optional dependency

    primary = list(primary)
    reference = list(reference)

    raw_primary = yf.download(
        primary, start=start, end=end, auto_adjust=auto_adjust, progress=False
    )
    df_primary = normalize_yf_columns(raw_primary, primary)

    df_ref = None
    if reference:
        raw_ref = yf.download(
            reference,
            start=ref_start or start,
            end=end,
            auto_adjust=auto_adjust,
            progress=False,
        )
        df_ref = normalize_yf_columns(raw_ref, reference)

    return assemble_panel(df_primary, df_ref)


def apply_vvix_gate(
    panel: pd.DataFrame,
    start_date: str,
    *,
    vvix_min_start: str = "2007-01-01",
    vvix_ticker: str = "^VVIX",
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop VVIX columns when the requested start predates VVIX history.

    VVIX (^VVIX) only has history from ~2007. If a model's start date is earlier
    than `vvix_min_start`, keeping VVIX would inject a long all-NaN prefix, so we
    drop it entirely; otherwise it is retained.
    """
    if pd.Timestamp(start_date) >= pd.Timestamp(vvix_min_start):
        return panel

    if isinstance(panel.columns, pd.MultiIndex):
        mask = panel.columns.get_level_values("Ticker") == vvix_ticker
        to_drop = panel.columns[mask]
    else:
        to_drop = [c for c in panel.columns if vvix_ticker in str(c)]

    if len(to_drop):
        panel = panel.drop(columns=to_drop)
        if verbose:
            print(
                f"[vvix] start {start_date} < {vvix_min_start}: "
                f"dropped {len(to_drop)} VVIX column(s)"
            )
    return panel
