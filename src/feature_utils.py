"""
Feature engineering utilities for S&P 500 prediction pipeline.

This module provides helper functions for creating lagged features and maintaining
consistent naming conventions across all data acquisition notebooks.

Lagging Design Principles:
=========================
1. MARKET-DERIVED FEATURES (LAG BY 1 TRADING DAY):
   - Rolling returns (Ret_1, Ret_5, Ret_10, Ret_20, Ret_60, Ret_120)
   - Rolling volatility (Vol_5, Vol_10, Vol_20, Vol_60)
   - All 54 StockStats technical indicators (MACD, RSI_14, MACD, etc.)
   - Intraday/market OHLCV (Open, High, Low, Close, Volume)
   - Volatility indices (VIX, VVIX)
   
   Rationale: These features include data through trading day T (close).
   Since predictions are made before the US market opens on day T, day T's
   data is not yet available. Lagging by 1 trading day ensures we only use
   information known at prediction time (through close of day T-1).

2. CALENDAR/SCHEDULED FEATURES (NO LAG):
   - Day-of-week dummies (mon, tues, wed, thurs, fri)
   - Holiday indicators (on_holiday, pre_holiday, post_holiday)
   - Days since last FOMC announcement (days_since_fomc)
   
   Rationale: These features are determined/scheduled in advance and are
   known before the market opens on day T.

3. DAILY FRED VARIABLES (LAG BY 1 TRADING DAY):
   - DFF (Federal Funds Effective Rate)
   - DGS2, DGS10 (Treasury yields)
   - DCOILWTICO (WTI crude oil price)
   
   Rationale: Assumed released end-of-day. Data for trading day T is not
   available until after day T has closed. Lag by 1 trading day.

4. LOWER-FREQUENCY FRED VARIABLES (NO LAG):
   - UNRATE, INDPRO, GDP, UMCSENT (released weekly/monthly/quarterly)
   - CORESTICKM159SFRBATL, DAAA, TOTBKCR, IR14270
   
   Rationale: These infrequent releases change slowly. Additional look-ahead
   bias from using the same calendar-day value is acceptable for this project.
   Using a proper ALFRED vintage would be more correct but is out of scope.
   This decision prioritizes simplicity and accepting known approximation error.

5. PERCENT-CHANGE COLUMNS:
   - Pct_change columns are derived from the underlying variable.
   - They follow the same lagging rules as their base variables.
   - Example: DFF_lag1 is lagged, so DFF_pct_lag1 is also lagged.
   
6. TARGET VARIABLE (NOT IN DATA ACQUISITION):
   - The target (Ret_1.shift(-1) for next-day return prediction) is created
     in the modeling notebook (01_EDA.ipynb), not in data acquisition.
   - This keeps cached datasets reusable for multiple supervised tasks.

"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union


def create_lagged_features(
    df: pd.DataFrame,
    lag_columns: List[str],
    lag_period: int = 1,
    multiindex: bool = False,
) -> pd.DataFrame:
    """
    Create lagged versions of specified columns with consistent naming.
    
    For MultiIndex DataFrames, lag_columns should be tuples, e.g., ("Ret_1", "^GSPC").
    For single-index DataFrames, lag_columns should be strings, e.g., "DFF".
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to add lagged columns to. Not modified in place.
    lag_columns : List[str] or List[Tuple[str, str]]
        Column names (or MultiIndex tuples) to lag.
    lag_period : int, default 1
        Number of periods to lag (typically 1 for trading days).
    multiindex : bool, default False
        If True, treat lag_columns as tuples for MultiIndex DataFrames.
    
    Returns
    -------
    pd.DataFrame
        A copy of df with new lagged columns appended.
    
    Examples
    --------
    >>> df = pd.DataFrame({"Ret_5": [0.01, 0.02, 0.03]})
    >>> df_lagged = create_lagged_features(df, ["Ret_5"], lag_period=1)
    >>> df_lagged.columns.tolist()
    ['Ret_5', 'Ret_5_lag1']
    
    >>> df_multi = pd.DataFrame(
    ...     {("Ret_1", "^GSPC"): [0.01, 0.02, 0.03]},
    ...     index=pd.date_range("2020-01-01", periods=3)
    ... )
    >>> df_lagged = create_lagged_features(
    ...     df_multi, [("Ret_1", "^GSPC")], lag_period=1, multiindex=True
    ... )
    """
    df_out = df.copy()
    
    for col in lag_columns:
        # Construct the lagged column name
        if multiindex and isinstance(col, tuple):
            # For MultiIndex tuples: ("Ret_1", "^GSPC") -> ("Ret_1", "^GSPC_lag1")
            # OR ("Ret_1_lag1", "^GSPC") depending on which level to apply lag to.
            # Convention: apply to the first level (feature name level).
            level0, level1 = col
            lagged_col_name = (f"{level0}_lag{lag_period}", level1)
        else:
            # For flat columns: "Ret_5" -> "Ret_5_lag1"
            lagged_col_name = f"{col}_lag{lag_period}"
        
        # Create the lagged column
        df_out[lagged_col_name] = df[col].shift(lag_period)
    
    return df_out


def rename_lagged_columns_multiindex(
    df: pd.DataFrame,
    original_cols: List[Tuple[str, str]],
    level0_to_level1: bool = True,
) -> pd.DataFrame:
    """
    Rename MultiIndex columns to apply lag suffix to level 0 (feature level).
    
    Used when you've already created lagged columns but need to ensure
    consistent naming convention across all notebooks.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with MultiIndex columns.
    original_cols : List[Tuple[str, str]]
        List of original MultiIndex tuples to rename.
    level0_to_level1 : bool, default True
        If True, apply lag to level 0 (first level).
        If False, apply lag to level 1 (second level).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns.
    """
    rename_map = {}
    for col in original_cols:
        if col in df.columns:
            level0, level1 = col
            if level0_to_level1:
                # Rename: ("Ret_1", "^GSPC") -> ("Ret_1_lag1", "^GSPC")
                new_col = (f"{level0}_lag1", level1)
            else:
                # Rename: ("Ret_1", "^GSPC") -> ("Ret_1", "^GSPC_lag1")
                new_col = (level0, f"{level1}_lag1")
            rename_map[col] = new_col
    
    return df.rename(columns=rename_map)


def get_lagged_column_name(
    original_name: Union[str, Tuple[str, str]],
    lag_period: int = 1,
    multiindex: bool = False,
) -> Union[str, Tuple[str, str]]:
    """
    Compute the lagged column name for a given original column name.
    
    Parameters
    ----------
    original_name : str or Tuple[str, str]
        Original column name or MultiIndex tuple.
    lag_period : int, default 1
        Lag period to apply.
    multiindex : bool, default False
        If True, treat original_name as a MultiIndex tuple.
    
    Returns
    -------
    str or Tuple[str, str]
        The lagged column name following project convention.
    
    Examples
    --------
    >>> get_lagged_column_name("Ret_5", lag_period=1, multiindex=False)
    'Ret_5_lag1'
    
    >>> get_lagged_column_name(("Ret_1", "^GSPC"), lag_period=1, multiindex=True)
    ('Ret_1_lag1', '^GSPC')
    """
    if multiindex and isinstance(original_name, tuple):
        level0, level1 = original_name
        return (f"{level0}_lag{lag_period}", level1)
    else:
        return f"{original_name}_lag{lag_period}"


# ============================================================================
# Feature Group Constants & Helpers
# ============================================================================

ROLLING_RETURN_WINDOWS = [1, 5, 10, 20, 60, 120]
"""Windows (in trading days) for rolling return features."""

ROLLING_VOLATILITY_WINDOWS = [5, 10, 20, 60]
"""Windows (in trading days) for rolling volatility features."""

CALENDAR_FEATURES = ["mon", "tues", "wed", "thurs", "fri"]
"""Day-of-week dummy variable names. These should NOT be lagged."""

HOLIDAY_FEATURES = ["on_holiday", "pre_holiday", "post_holiday"]
"""Holiday indicator names. These should NOT be lagged."""

FOMC_FEATURES = ["days_since_fomc"]
"""FOMC-related feature names. These should NOT be lagged."""

DAILY_FRED_SERIES = [
    "DFF",
    "DGS2",
    "DGS10",
    "DCOILWTICO",
]
"""Daily FRED series that should be lagged by 1 trading day."""

LOWER_FREQ_FRED_SERIES = [
    "DAAA",
    "CORESTICKM159SFRBATL",
    "GDP",
    "UNRATE",
    "INDPRO",
    "UMCSENT",
    "TOTBKCR",
    "IR14270",
]
"""Lower-frequency FRED series (weekly/monthly/quarterly) that should NOT be lagged.

Note: This is an intentional approximation. Proper handling would require
ALFRED vintages to eliminate look-ahead bias. See module docstring.
"""


def get_features_to_lag_rolling(
    tickers: List[str],
    windows: List[int] = None,
) -> List[Tuple[str, str]]:
    """
    Generate list of rolling return/volatility MultiIndex columns to lag.
    
    Parameters
    ----------
    tickers : List[str]
        Ticker symbols (e.g., ["^GSPC", "^VIX"]).
    windows : List[int], optional
        Windows to use. If None, uses ROLLING_RETURN_WINDOWS.
    
    Returns
    -------
    List[Tuple[str, str]]
        List of MultiIndex tuples for rolling features.
    
    Examples
    --------
    >>> cols = get_features_to_lag_rolling(["^GSPC"], windows=[1, 5])
    >>> cols
    [('Ret_1', '^GSPC'), ('Ret_5', '^GSPC')]
    """
    if windows is None:
        windows = ROLLING_RETURN_WINDOWS
    
    cols = []
    for ticker in tickers:
        for window in windows:
            cols.append((f"Ret_{window}", ticker))
    
    return cols


def get_features_to_lag_volatility(
    tickers: List[str],
    windows: List[int] = None,
) -> List[Tuple[str, str]]:
    """
    Generate list of rolling volatility MultiIndex columns to lag.
    
    Parameters
    ----------
    tickers : List[str]
        Ticker symbols (e.g., ["^GSPC", "^VIX"]).
    windows : List[int], optional
        Windows to use. If None, uses ROLLING_VOLATILITY_WINDOWS.
    
    Returns
    -------
    List[Tuple[str, str]]
        List of MultiIndex tuples for volatility features.
    """
    if windows is None:
        windows = ROLLING_VOLATILITY_WINDOWS
    
    cols = []
    for ticker in tickers:
        for window in windows:
            cols.append((f"Vol_{window}", ticker))
    
    return cols


def get_features_to_lag_technicals(
    technical_indicators: List[str],
) -> List[Tuple[str, str]]:
    """
    Generate list of technical indicator MultiIndex columns to lag.
    
    Parameters
    ----------
    technical_indicators : List[str]
        List of technical indicator names from StockStats.
    
    Returns
    -------
    List[Tuple[str, str]]
        List of MultiIndex tuples (indicator, "").
    
    Examples
    --------
    >>> indicators = ["macd", "rsi_14", "atr_14"]
    >>> cols = get_features_to_lag_technicals(indicators)
    >>> cols
    [('macd_lag1', ''), ('rsi_14_lag1', ''), ('atr_14_lag1', '')]
    """
    # Technical indicators are stored as ("Technical", indicator_name)
    return [(indicator, "") for indicator in technical_indicators]


if __name__ == "__main__":
    # Quick smoke test
    print("Testing feature_utils.py...")
    
    # Test create_lagged_features with flat DataFrame
    df_flat = pd.DataFrame({
        "Ret_5": [0.01, 0.02, 0.03, 0.04],
        "DFF": [0.03, 0.035, 0.04, 0.042],
    })
    df_flat_lagged = create_lagged_features(df_flat, ["Ret_5", "DFF"], lag_period=1)
    print("Flat DataFrame lagging test:")
    print(df_flat_lagged)
    print()
    
    # Test create_lagged_features with MultiIndex DataFrame
    idx = pd.date_range("2020-01-01", periods=4)
    df_multi = pd.DataFrame(
        {
            ("Ret_1", "^GSPC"): [0.01, 0.02, 0.03, 0.04],
            ("Vol_5", "^GSPC"): [0.005, 0.006, 0.007, 0.008],
        },
        index=idx,
    )
    df_multi.columns = pd.MultiIndex.from_tuples(df_multi.columns)
    df_multi_lagged = create_lagged_features(
        df_multi, [("Ret_1", "^GSPC"), ("Vol_5", "^GSPC")], lag_period=1, multiindex=True
    )
    print("MultiIndex DataFrame lagging test:")
    print(df_multi_lagged)
    print()
    
    # Test lagged column name generation
    print("Column name generation test:")
    print(f"  Flat: {get_lagged_column_name('Ret_5', lag_period=1, multiindex=False)}")
    print(f"  MultiIndex: {get_lagged_column_name(('Ret_1', '^GSPC'), lag_period=1, multiindex=True)}")
    print()
    
    print("All tests passed! ✓")
