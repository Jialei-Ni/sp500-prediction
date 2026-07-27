"""Offline smoke test: verifies the network-free logic of src/data/*.

Run: python smoke_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import (
    normalize_yf_columns, assemble_panel, apply_vvix_gate, load_or_fetch,
)
from src.config import load_config


def _mk_ohlcv(tickers, index, field_level_first=True):
    """Build a fake yfinance-style frame with MultiIndex columns."""
    fields = ["Open", "High", "Low", "Close", "Volume"]
    if field_level_first:
        cols = pd.MultiIndex.from_product([fields, tickers])
    else:  # group_by="ticker" style: (Ticker, Field)
        cols = pd.MultiIndex.from_product([tickers, fields])
    data = np.arange(len(index) * len(cols)).reshape(len(index), len(cols)) * 1.0
    return pd.DataFrame(data, index=index, columns=cols)


def test_normalize_multi_field_first():
    idx = pd.date_range("2020-01-01", periods=3)
    df = _mk_ohlcv(["^GSPC", "^DJI"], idx, field_level_first=True)
    out = normalize_yf_columns(df, ["^GSPC", "^DJI"])
    assert list(out.columns.names) == ["Field", "Ticker"]
    assert set(out.columns.get_level_values("Field")) == {"Open", "High", "Low", "Close", "Volume"}
    assert set(out.columns.get_level_values("Ticker")) == {"^GSPC", "^DJI"}
    print("  ok: MultiIndex (Field, Ticker) passthrough")


def test_normalize_ticker_first_swapped():
    idx = pd.date_range("2020-01-01", periods=3)
    df = _mk_ohlcv(["^GSPC", "^DJI"], idx, field_level_first=False)  # (Ticker, Field)
    out = normalize_yf_columns(df, ["^GSPC", "^DJI"])
    assert list(out.columns.names) == ["Field", "Ticker"]
    assert set(out.columns.get_level_values("Field")) == {"Open", "High", "Low", "Close", "Volume"}
    print("  ok: MultiIndex (Ticker, Field) auto-swapped to (Field, Ticker)")


def test_normalize_flat_single_ticker():
    idx = pd.date_range("2020-01-01", periods=3)
    df = pd.DataFrame(
        {"Open": [1., 2, 3], "High": [1., 2, 3], "Low": [1., 2, 3],
         "Close": [1., 2, 3], "Volume": [1., 2, 3]}, index=idx)
    out = normalize_yf_columns(df, ["^GSPC"])
    assert list(out.columns.names) == ["Field", "Ticker"]
    assert set(out.columns.get_level_values("Ticker")) == {"^GSPC"}
    assert ("Close", "^GSPC") in out.columns
    print("  ok: flat single-ticker frame wrapped to (Field, Ticker)")


def test_assemble_ffill_alignment():
    prim_idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])  # US days
    df_p = _mk_ohlcv(["^GSPC"], prim_idx)
    df_p = normalize_yf_columns(df_p, ["^GSPC"])

    # Reference series present on a DIFFERENT calendar (incl. a gap on 01-03).
    ref_idx = pd.to_datetime(["2020-01-02", "2020-01-06"])
    df_r = pd.DataFrame({"Close": [10.0, 20.0]}, index=ref_idx)
    df_r.columns = pd.MultiIndex.from_product([["Close"], ["^VIX"]], names=["Field", "Ticker"])

    panel = assemble_panel(df_p, df_r)
    assert list(panel.index) == list(prim_idx)                      # restricted to primary
    vix = panel[("Close", "^VIX")]
    assert vix.loc["2020-01-02"] == 10.0
    assert vix.loc["2020-01-03"] == 10.0                            # forward-filled into the gap
    assert vix.loc["2020-01-06"] == 20.0
    assert panel.index.name == "date"
    print("  ok: reference aligned onto primary calendar with forward-fill")


def test_vvix_gate():
    idx = pd.date_range("2005-01-01", periods=2)
    df = _mk_ohlcv(["^GSPC", "^VVIX"], idx)
    df = normalize_yf_columns(df, ["^GSPC", "^VVIX"])

    kept = apply_vvix_gate(df, "2010-01-01", vvix_min_start="2007-01-01", verbose=False)
    assert ("Close", "^VVIX") in kept.columns, "VVIX should be kept when start >= 2007"

    dropped = apply_vvix_gate(df, "2000-01-01", vvix_min_start="2007-01-01", verbose=False)
    assert "^VVIX" not in set(dropped.columns.get_level_values("Ticker")), \
        "VVIX should be dropped when start < 2007"
    assert "^GSPC" in set(dropped.columns.get_level_values("Ticker"))
    print("  ok: VVIX gate drops pre-2007, keeps otherwise")


def test_cache_hit_miss_refresh(tmp: Path):
    calls = {"n": 0}
    idx = pd.date_range("2020-01-01", periods=3)
    payload = _mk_ohlcv(["^GSPC"], idx)  # MultiIndex columns -> parquet round-trip
    payload = normalize_yf_columns(payload, ["^GSPC"])

    def fetch():
        calls["n"] += 1
        return payload

    p = tmp / "indices_raw.parquet"

    a = load_or_fetch(p, fetch, verbose=False)          # miss -> fetch (1)
    b = load_or_fetch(p, fetch, verbose=False)          # hit  -> no fetch
    c = load_or_fetch(p, fetch, verbose=False, force_refresh=True)  # forced (2)

    assert calls["n"] == 2, f"expected 2 fetches, got {calls['n']}"
    # MultiIndex columns + names survive the parquet round-trip:
    assert list(b.columns.names) == ["Field", "Ticker"]
    assert ("Close", "^GSPC") in b.columns
    # parquet drops the index `freq` attribute (real trading-day indices have
    # none anyway); compare values/dtypes/labels but not freq metadata.
    pd.testing.assert_frame_equal(a, b, check_freq=False)
    pd.testing.assert_frame_equal(a, c, check_freq=False)
    print("  ok: cache miss fetches, hit skips, force_refresh re-fetches, MI cols survive parquet")


def test_config_load(root: Path):
    cfg = load_config(root / "config.yaml")
    assert cfg.start == "2000-01-01"
    assert cfg.indices_path.name == "indices_raw.parquet"
    assert cfg.fomc_path.suffix == ".csv"
    assert cfg.vvix_min_start == "2007-01-01"
    print("  ok: config.yaml parsed into typed Config")


def test_fred_key_resolution(root: Path):
    import os

    cfg = load_config(root / "config.yaml")
    var = cfg.fred_api_key_env

    # 1) environment variable takes precedence
    os.environ[var] = "env-key-abc"
    try:
        assert load_config(root / "config.yaml").fred_api_key == "env-key-abc"
    finally:
        os.environ.pop(var, None)

    # 2) legacy fred_api_key.py module fallback (no env var set)
    legacy = root / "fred_api_key.py"
    assert not legacy.exists(), "refusing to overwrite a real fred_api_key.py"
    legacy.write_text('FRED_API_KEY = "module-key-xyz"\n')
    try:
        # fresh interpreter avoids import caching from a prior run
        import subprocess, sys as _sys
        code = (
            "from src.config import load_config;"
            f"print(load_config(r'{root/'config.yaml'}').fred_api_key)"
        )
        out = subprocess.run([_sys.executable, "-c", code], cwd=root,
                             capture_output=True, text=True)
        assert out.stdout.strip() == "module-key-xyz", out.stdout + out.stderr
        print("  ok: FRED key resolves env var first, then legacy fred_api_key.py")
    finally:
        legacy.unlink()


if __name__ == "__main__":
    import tempfile
    root = Path(__file__).resolve().parent
    print("normalize_yf_columns:")
    test_normalize_multi_field_first()
    test_normalize_ticker_first_swapped()
    test_normalize_flat_single_ticker()
    print("assemble_panel:")
    test_assemble_ffill_alignment()
    print("apply_vvix_gate:")
    test_vvix_gate()
    print("cache.load_or_fetch:")
    with tempfile.TemporaryDirectory() as d:
        test_cache_hit_miss_refresh(Path(d))
    print("config.load_config:")
    test_config_load(root)
    test_fred_key_resolution(root)
    print("\nALL SMOKE TESTS PASSED")
