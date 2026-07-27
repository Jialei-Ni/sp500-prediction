"""Data-acquisition subpackage (raw external fetches + caching)."""

from .cache import load_or_fetch
from .indices import (
    download_price_panel,
    normalize_yf_columns,
    assemble_panel,
    apply_vvix_gate,
)
from .macro import download_fred
from .fomc import scrape_fomc_calendar

__all__ = [
    "load_or_fetch",
    "download_price_panel",
    "normalize_yf_columns",
    "assemble_panel",
    "apply_vvix_gate",
    "download_fred",
    "scrape_fomc_calendar",
]