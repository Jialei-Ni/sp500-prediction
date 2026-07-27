# S&P 500 Index-Timing Pipeline

A refactor of the original notebooks into four staged, cached, leakage-checked
steps: acquire raw data → build features & targets → walk-forward modeling →
backtest & report.

## Layout

```
config.yaml                 run configuration (dates, features, target, model, backtest)
index_ticker.py             ticker lists (existing)
stockstats_technicals.py    technical-indicator list (existing)
fred_macro.py               FRED series list (existing)
fred_api_key.py             (optional) FRED_API_KEY = "..."   -- gitignored
notebooks/
  00_acquire_data.ipynb     fetch + cache raw price panel, FRED macro, FOMC calendar
  01_features_target.ipynb  prediction-safe feature matrix + continuous forward targets
  02_model_walkforward.ipynb expanding walk-forward, models, IC/ICIR, saves OOS signals
  03_backtest_report.ipynb  signal -> strategy, costs/roll, equity/drawdown/heatmap/bars
src/
  config.py                 typed loader for config.yaml
  data/  cache.py indices.py macro.py fomc.py   (stage 00; raw fetch + caching)
  feature_utils.py          (existing) create_lagged_features
  features.py technicals.py targets.py          (stage 01)
  validation.py models.py signal.py metrics.py experiment.py   (stage 02)
  backtest.py perf.py viz.py                    (stage 03)
cache/                      generated data (gitignored)
smoke_*.py                  offline tests (no network required)
```

## Run order

`00 -> 01 -> 02 -> 03`. Each notebook reads the previous stage's cache and skips
work that is already cached. To force a rebuild, set `acquire.force_refresh: true`
in `config.yaml` (this flag also gates the feature/model caches — changing any
`features:`, `target:`, or `model:` setting requires it).

## FRED API key

Resolved in this order: the `FRED_API_KEY` environment variable, else a
project-root `fred_api_key.py` module exposing `FRED_API_KEY = "..."`. The latter
matches the original notebooks and is gitignored.

## Design notes (correctness)

- **No look-ahead.** Market-derived features (returns, volatility, technicals) and
  daily FRED series are lagged one trading day; calendar/holiday/FOMC features are
  not (known in advance). Raw price levels are excluded from features.
- **Leakage-free targets.** The target is the continuous forward return; quantile
  buckets are fit per walk-forward fold on training data only (never the whole
  sample), with an H-day embargo purging overlapping labels around each fold.
- **Fair model comparison.** Classifiers and regressors are both reduced to a
  common expected-return signal, so any difference reflects the model, not the
  formulation. Reported via Rank IC / non-overlapping Rank IC / ICIR plus native
  ML metrics.

## Tests

```
python smoke_test.py        # stage 00: caching, yfinance column shapes, VVIX gate
python smoke_features.py    # stage 01: no-look-ahead, targets, bucketing
python smoke_model.py       # stage 02: embargo, walk-forward experiment
python smoke_backtest.py    # stage 03: positions, costs, vol-target, figures
```

All run offline (no network / API key needed).

## Caveats

Results are a single historical path on one index. The out-of-sample machinery is
sound, but a weak or flat IC / Sharpe is a legitimate finding, not a bug. Costs
default to a conservative 10 bps/turn (configurable). This is research code, not
investment advice.
