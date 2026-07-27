"""
Generic caching layer for the data-acquisition pipeline.

Design goal
-----------
`load_or_fetch(path, fetch_fn)` returns a cached DataFrame if one exists on
disk, otherwise it calls `fetch_fn()` (the expensive network download), writes
the result to `path`, and returns it. Pass `force_refresh=True` to bypass an
existing cache and re-download.

Format is inferred from the file extension:
  * ``.parquet``  -> preferred for MultiIndex columns / mixed dtypes
                     (this is what makes the yfinance "double heading" survive
                     a round-trip without the messy ``Unnamed: x_level_1`` CSV
                     artefacts).
  * ``.csv``      -> convenient for flat, human-inspectable tables (e.g. the
                     FOMC calendar).

Nothing here touches the network; the network work lives entirely inside the
`fetch_fn` closures defined in the acquisition notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd


def _save(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(path)
        except ImportError as exc:  # pragma: no cover - env dependent
            raise ImportError(
                "Writing parquet requires 'pyarrow' (or 'fastparquet'). "
                "Install with: pip install pyarrow"
            ) from exc
    elif suffix == ".csv":
        df.to_csv(path)
    else:
        raise ValueError(f"Unsupported cache extension: {path.suffix!r} ({path})")


def _load(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        # Flat tables only. MultiIndex-column frames should use parquet so we
        # never have to guess how many header rows a CSV has.
        return pd.read_csv(path, index_col=0)
    raise ValueError(f"Unsupported cache extension: {path.suffix!r} ({path})")


def load_or_fetch(
    path: str | Path,
    fetch_fn: Callable[[], pd.DataFrame],
    *,
    force_refresh: bool = False,
    loader: Optional[Callable[[Path], pd.DataFrame]] = None,
    saver: Optional[Callable[[pd.DataFrame, Path], None]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return a cached DataFrame, fetching and caching it on a miss.

    Parameters
    ----------
    path
        Destination cache file. Extension selects the format (.parquet/.csv).
    fetch_fn
        Zero-argument callable that performs the expensive fetch and returns a
        DataFrame. Only called on a cache miss (or when ``force_refresh``).
    force_refresh
        If True, ignore any existing cache and re-fetch.
    loader, saver
        Optional overrides for custom (de)serialisation.
    verbose
        Print a one-line cache hit/miss/save message.
    """
    path = Path(path)
    load = loader or _load
    save = saver or _save

    if path.exists() and not force_refresh:
        if verbose:
            print(f"[cache] HIT   {path.name}  (skipping fetch)")
        return load(path)

    if verbose:
        reason = "force_refresh" if path.exists() else "not cached"
        print(f"[cache] MISS  {path.name}  ({reason}) -> fetching...")

    df = fetch_fn()

    path.parent.mkdir(parents=True, exist_ok=True)
    save(df, path)

    if verbose:
        n_rows, n_cols = df.shape
        print(f"[cache] SAVE  {path.name}  ({n_rows:,} rows x {n_cols} cols)")

    return df
