# S&P 500 Prediction: Data Acquisition Refactoring

## Overview

This document describes the refactoring of the data acquisition pipeline to move feature engineering into the acquisition notebooks and ensure all features are properly lagged to prevent look-ahead bias.

**Goal**: The cached datasets should be safe to use directly in machine learning without additional shifting, as all features are aligned to the prediction timestamp.

---

## Architecture Changes

### Before Refactoring
```
Holidays_and_Indices.ipynb
  ↓ (indices_from_2000.csv)
     - Rolling returns (forward-looking, not lagged)
     - Rolling volatility (forward-looking, not lagged)
     - Day-of-week dummies
     - Holiday indicators
     
Macro_FRED.ipynb
  ↓ (macro_data.csv)
     - FRED time series (not lagged)
     - Percent-change columns (not lagged)
     
FOMC.ipynb
  ↓ (fomc_calendar_2000_present.csv)
     - Meeting dates
     
01_EDA.ipynb (Modeling)
  ↓
  - Calculate days_since_fomc (merge with FOMC calendar)
  - Calculate 54 technical indicators (StockStats)
  - Derive features (YieldSpread, RealFedFunds, VVIX_VIX)
  - Select features
  - Train model
```

### After Refactoring
```
Holidays_and_Indices.ipynb
  ↓ (indices_from_2000.csv)
     - Rolling returns + lagged versions (Ret_1_lag1, etc.)
     - Rolling volatility + lagged versions (Vol_5_lag1, etc.)
     - 54 Technical indicators + lagged versions (macd_lag1, etc.)
     - days_since_fomc (pre-calculated using FOMC calendar)
     - Day-of-week dummies (NOT lagged - known in advance)
     - Holiday indicators (NOT lagged - known in advance)
     
Macro_FRED.ipynb
  ↓ (macro_data.csv)
     - FRED time series (original values for reference)
     - Lagged daily FRED (DFF_lag1, DGS2_lag1, etc.)
     - Percent-change columns (original + lagged for daily series)
     - Lower-frequency FRED unchanged (no lag)
     
FOMC.ipynb
  ↓ (fomc_calendar_2000_present.csv)
     - Meeting dates (used by Holidays_and_Indices.ipynb)
     
01_EDA.ipynb (Modeling)
  ↓
  - Loads pre-engineered features (all lagged versions available)
  - Derive task-specific features (YieldSpread using lagged macro, etc.)
  - Select features for modeling (prefers lagged versions)
  - Train model
```

---

## Feature Lagging Rules

### Lagged by 1 Trading Day
These features include data through trading day T's close, but predictions are made **before** the market opens on day T. Therefore, they are lagged by 1 trading day:

#### Market-Derived Features (From Holidays_and_Indices.ipynb)
- **Rolling returns** (Ret_1, Ret_5, Ret_10, Ret_20, Ret_60, Ret_120)
  - New columns: `Ret_1_lag1`, `Ret_5_lag1`, ..., `Ret_120_lag1`
  - Lagging rationale: These prices include day T's close; predictions made before market opens
  
- **Rolling volatility** (Vol_5, Vol_10, Vol_20, Vol_60)
  - New columns: `Vol_5_lag1`, `Vol_10_lag1`, ..., `Vol_60_lag1`
  - Lagging rationale: Calculated from returns through day T

- **StockStats technical indicators** (All 54 indicators)
  - Original columns: `Technical_macd`, `Technical_rsi_14`, `Technical_atr_14`, etc.
  - Lagged columns: `Technical_macd_lag1`, `Technical_rsi_14_lag1`, `Technical_atr_14_lag1`, etc.
  - Lagging rationale: Calculated using OHLCV through day T

#### Daily FRED Variables (From Macro_FRED.ipynb)
- **Daily series**: DFF, DGS2, DGS10, DCOILWTICO
  - New columns: `DFF_lag1`, `DGS2_lag1`, `DGS10_lag1`, `DCOILWTICO_lag1`
  - Also: `DFF_pct_change_lag1`, `DGS2_pct_change_lag1`, etc.
  - Lagging rationale: Assumed released end-of-day; not available at market open

#### Market Volatility Indices
- **VIX and VVIX** (if used as features)
  - Lagging rationale: Calculated at market close

### NOT Lagged
These features are determined/scheduled in advance and known before the market opens on day T:

#### Calendar Features (From Holidays_and_Indices.ipynb)
- **Day-of-week dummies**: `mon`, `tues`, `wed`, `thurs`, `fri`
  - Rationale: Known well in advance
  
- **Holiday indicators**: `on_holiday`, `pre_holiday`, `post_holiday`
  - Rationale: Federal holiday calendar is known in advance

#### FOMC Features (From Holidays_and_Indices.ipynb)
- **Days since FOMC**: `days_since_fomc`
  - Rationale: FOMC meeting dates are published; no look-ahead bias

#### Lower-Frequency FRED Variables (From Macro_FRED.ipynb)
- **Weekly/monthly/quarterly series**: DAAA, CORESTICKM159SFRBATL, GDP, UNRATE, INDPRO, UMCSENT, TOTBKCR, IR14270
  - Rationale: These variables change infrequently. Although proper handling would require ALFRED vintages to eliminate look-ahead bias completely, this approximation is acceptable for this project. The additional look-ahead bias from using the same calendar-day value is known and tolerated.

---

## CSV Changes

### indices_from_2000.csv

**New Columns Added:**

| Column Name | Type | Description |
|------------|------|-------------|
| `Ret_1_lag1`, `Ret_5_lag1`, etc. | MultiIndex | Lagged rolling returns (1 day lag) |
| `Vol_5_lag1`, `Vol_10_lag1`, etc. | MultiIndex | Lagged rolling volatility (1 day lag) |
| `Technical_macd`, `Technical_rsi_14`, etc. (all 54) | MultiIndex | StockStats technical indicators |
| `Technical_macd_lag1`, `Technical_rsi_14_lag1`, etc. | MultiIndex | Lagged technical indicators (1 day lag) |
| `days_since_fomc` | MultiIndex | Days since last FOMC announcement (not lagged) |

**Column Structure:**
```
Original structure (preserved): (Feature_Level, Ticker) for rolling returns/volatility
Example: ("Ret_1", "^GSPC"), ("Vol_5", "^DJI"), ("Ret_1_lag1", "^GSPC")

New technical indicators: ("Technical", Indicator_Name)
Example: ("Technical", "macd"), ("Technical", "rsi_14"), ("Technical", "macd_lag1")

FOMC feature: ("days_since_fomc", "")
```

**Backward Compatibility:**
- All original columns preserved
- New lagged columns appended
- Existing code using original columns continues to work
- Modeling notebooks should prefer lagged versions

**File Size Impact:** ~60-70% increase due to ~140+ new lagged columns

---

### macro_data.csv

**New Columns Added:**

| Series ID | Original Column | New Lagged Columns | Frequency |
|-----------|-----------------|-------------------|-----------|
| DFF | `DFF` | `DFF_lag1` | Daily (lagged) |
| DGS2 | `DGS2` | `DGS2_lag1` | Daily (lagged) |
| DGS10 | `DGS10` | `DGS10_lag1` | Daily (lagged) |
| DCOILWTICO | `DCOILWTICO` | `DCOILWTICO_lag1` | Daily (lagged) |
| All daily | `*_pct_change` | `*_pct_change_lag1` | Pct change (lagged) |
| Lower-freq | *Unchanged* | *None* | Monthly/quarterly |

**Backward Compatibility:**
- All original columns preserved (both raw values and pct_change)
- Lower-frequency series unchanged
- New lagged columns for daily series only
- Existing code using original columns continues to work

**Example New Columns:**
- `DFF_lag1`, `DFF_pct_change_lag1`
- `DGS2_lag1`, `DGS2_pct_change_lag1`
- `DGS10_lag1`, `DGS10_pct_change_lag1`
- `DCOILWTICO_lag1`, `DCOILWTICO_pct_change_lag1`

---

## Updated Notebook Structure

### Holidays_and_Indices.ipynb

**Cell 1: Setup**
- Added import of `feature_utils` module for lagging functions
- Added imports for technical indicator definitions

**Cell 2: Holiday & Trading Day Creation** (Unchanged)
- Creates holiday and NYSE trading calendars

**Cell 3: Data Download & Feature Engineering** (Enhanced)
- Downloads OHLC data (same as before)
- Creates day-of-week dummies (same as before, NOT lagged)
- Creates holiday indicators (same as before, NOT lagged)
- Creates rolling returns (same as before)
- Creates rolling volatility (same as before)

**Cell 4 (New): Lag Rolling Features**
- Creates lagged versions of rolling returns (Ret_*_lag1)
- Creates lagged versions of rolling volatility (Vol_*_lag1)
- Uses `feature_utils.create_lagged_features()` for consistency

**Cell 5 (New): Load FOMC Calendar & Calculate days_since_fomc**
- Loads FOMC calendar from cache
- Calculates days_since_fomc using merge_asof (moved from 01_EDA.ipynb)
- Does NOT lag this feature (known in advance)

**Cell 6 (New): Calculate Technical Indicators & Create Lagged Versions**
- Extracts S&P 500 OHLCV
- Calculates 54 StockStats indicators
- Creates lagged versions of all indicators (feature_lag1)
- Uses `feature_utils.create_lagged_features()` for consistency

**Cell 7 (Original Save)**
- Saves enhanced `indices_from_2000.csv` with all new columns

---

### Macro_FRED.ipynb

**Cell 1: Setup**
- Added import of `feature_utils` module

**Cell 2: FRED Series List** (Unchanged)
- Defines 12 FRED series to fetch

**Cell 3: Data Fetching & Processing** (Enhanced)
- Fetches from FRED API (same as before)
- Creates pct_change columns (same as before)
- Forward-fills to daily frequency (same as before)

**Cell 4 (New): Lag Daily FRED Variables**
- Creates lagged versions of daily series only (DFF_lag1, etc.)
- Creates lagged pct_change columns for daily series
- Keeps lower-frequency series unchanged (no lag)
- Uses `.shift(1)` on trading-day basis

**Cell 5 (Original Save)**
- Saves enhanced `macro_data.csv` with new lagged columns

---

### FOMC.ipynb

**No Changes**
- Continues to generate `fomc_calendar_2000_present.csv` as before
- Now used by `Holidays_and_Indices.ipynb` to calculate days_since_fomc

---

### 01_EDA.ipynb (Modeling Notebook)

**Cell 1: Setup** (Updated)
- Removed `stockstats` import (no longer needed)
- Removed `STOCKSTATS_TECHNICALS` import (not used)
- Added comments explaining refactoring

**Cell 2: Load indices_from_2000.csv** (Unchanged)
- Automatically includes technical indicators and days_since_fomc

**Cell 3: Load FOMC Calendar** (Removed/Commented)
- No longer needed; days_since_fomc pre-calculated

**Cell 4: Load macro_data.csv** (Unchanged)
- Now includes lagged daily macro variables

**Cell 5: Calculate days_since_fomc** (Removed)
- Logic moved to Holidays_and_Indices.ipynb
- Column already exists in indices_from_2000.csv

**Cell 6: Join macro data** (Updated)
- Clarified comments about lagged variables
- Merger logic unchanged (works with both original and lagged columns)

**Cell 7: Calculate Technical Indicators** (Removed)
- Logic moved to Holidays_and_Indices.ipynb
- Indicators already in indices_from_2000.csv

**Cell 8: Markdown** (Unchanged)

**Cell 9: Derive Features & Feature Selection** (Updated)
- Updated to use lagged macro variables (DGS10_lag1, DGS2_lag1, DFF_lag1)
- With fallback to original versions if lagged not available
- Added comments about lagged features
- Feature selection logic improved to handle new columns

**Cells 10+: Model Training & Evaluation** (Unchanged)
- Model training code works with the pre-engineered features
- No changes needed

---

## Implementation Details

### Helper Functions (src/feature_utils.py)

**`create_lagged_features(df, lag_columns, lag_period=1, multiindex=False)`**
- Creates lagged versions of specified columns
- Handles both flat and MultiIndex DataFrames
- Naming convention: `{feature_name}_lag{period}`
- Returns DataFrame with original + lagged columns

**`get_lagged_column_name(original_name, lag_period=1, multiindex=False)`**
- Computes lagged column name from original name
- Used for consistency in naming across notebooks

**Feature Group Constants:**
- `ROLLING_RETURN_WINDOWS` = [1, 5, 10, 20, 60, 120]
- `ROLLING_VOLATILITY_WINDOWS` = [5, 10, 20, 60]
- `DAILY_FRED_SERIES` = ["DFF", "DGS2", "DGS10", "DCOILWTICO"]
- `LOWER_FREQ_FRED_SERIES` = [8 lower-frequency series]

---

## Migration Guide for Existing Code

### If Your Code Previously Used Original Columns

**Option 1: Use Lagged Versions (Recommended)**
```python
# OLD
df_model = df[["Ret_5", "Vol_10", "DFF", "macd"]]

# NEW - Prefer lagged versions
df_model = df[["Ret_5_lag1", "Vol_10_lag1", "DFF_lag1", "macd_lag1"]]
```

**Option 2: Keep Using Original Columns**
```python
# Still works, but be aware of look-ahead bias
df_model = df[["Ret_5", "Vol_10", "DFF", "macd"]]
```

### If Your Code Manually Lagged Features

**OLD CODE (no longer needed):**
```python
df["Ret_5_shift"] = df["Ret_5"].shift(1)
```

**NEW CODE:**
```python
# Use pre-calculated lagged version
df_model = df["Ret_5_lag1"]
```

---

## Verification Checklist

- [ ] `src/feature_utils.py` created and tested
- [ ] `Holidays_and_Indices.ipynb` runs without errors
- [ ] `indices_from_2000.csv` contains lagged rolling features
- [ ] `indices_from_2000.csv` contains technical indicators (lagged)
- [ ] `indices_from_2000.csv` contains `days_since_fomc` column
- [ ] `Macro_FRED.ipynb` runs without errors
- [ ] `macro_data.csv` contains lagged daily FRED variables
- [ ] `01_EDA.ipynb` runs without errors
- [ ] Baseline model trains successfully
- [ ] Metrics (RMSE, R², Corr) are in expected range
- [ ] Lagged features preferred in feature selection
- [ ] No unexpected NaNs in final dataset

---

## Known Limitations & Design Decisions

### 1. Lower-Frequency FRED Variables (No Lag)
Lower-frequency FRED series (UNRATE, INDPRO, GDP, UMCSENT, etc.) are not lagged. This means they potentially contain look-ahead bias if their release dates differ significantly from the calendar date. For example, employment data for September is typically released in early October.

**Rationale:** Proper handling would require ALFRED vintages (Federal Reserve's vintage-adjusted data) to eliminate bias completely. This is out of scope for this project.

**Mitigation:** These variables change infrequently and contribute relatively less to daily predictions compared to daily market data.

### 2. International Index Lagging
International indices (Nikkei, DAX, FTSE, etc.) close at different times. Some close before the US market opens (same-day use is valid). The current implementation lags all international indices uniformly for simplicity.

**Recommendation:** If specific international markets are found to be predictive, implement per-ticker lagging logic in future iterations.

### 3. Target Variable Not in Cached Data
The target variable (next-day S&P 500 return) is intentionally NOT created in the acquisition notebooks. It remains task-specific and is created in 01_EDA.ipynb.

**Rationale:** Cached datasets should be reusable for multiple prediction horizons and target definitions (next-day return, next-5-day return, classification, etc.).

---

## Performance Impact

- **Memory**: CSV files increased by ~60-70% due to new lagged columns (roughly doubling features)
- **Load time**: Slightly increased due to larger CSV size; compression on disk is reasonable
- **Model training time**: Varies by model; feature space doubled but many are highly correlated (originals vs. lagged)
- **Recommendation**: Consider feature selection or dimensionality reduction if memory/speed becomes an issue

---

## Future Improvements

1. **Per-ticker lagging logic** for international indices based on market-open times
2. **ALFRED vintage integration** for proper lower-frequency macro variable handling
3. **Feature selection**  to prefer lagged versions automatically in modeling notebooks
4. **Caching**: Cache technical indicators separately to reduce indices_from_2000.csv size
5. **Versioning**: Add CSV version metadata to track schema changes

---

## Contact & Questions

For questions about the refactoring or lagging logic, refer to:
- `src/feature_utils.py` docstrings
- Cell comments in `Holidays_and_Indices.ipynb` and `Macro_FRED.ipynb`
- `REFACTORING_NOTES.md` (this file)
