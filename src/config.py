"""
Minimal configuration loader.

`load_config()` reads ``config.yaml`` from the project root into a small typed
object. Attribute access for the values the pipeline reads often, plus `.raw`
for the full parsed dict so later stages can add sections without touching this
file.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    project_root: Path
    start: str
    end: str
    ref_start: str
    vvix_min_start: str
    return_windows: list
    vol_windows: list
    use_technicals: bool
    use_fomc: bool
    horizons: dict
    n_buckets: int
    bucket_method: str
    fixed_bins: list
    cache_dir: Path
    indices_path: Path
    macro_path: Path
    fomc_path: Path
    features_path: Path
    targets_path: Path
    signals_path: Path
    metrics_path: Path
    model_names: list
    n_folds: int
    min_train_frac: float
    force_refresh: bool
    fred_api_key_env: str
    request_pause_sec: float
    raw: dict

    @property
    def fred_api_key(self) -> str | None:
        """Resolve the FRED API key.

        Resolution order:
          1. the environment variable named by ``fred_api_key_env`` (default
             ``FRED_API_KEY``), if set;
          2. a project-root ``fred_api_key.py`` module exposing
             ``FRED_API_KEY = "..."`` -- the pattern the original notebooks used.
        Returns None if neither is available.
        """
        key = os.environ.get(self.fred_api_key_env)
        if key:
            return key

        # Legacy fallback: fred_api_key.py at the project root.
        try:
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            import fred_api_key as _mod  # type: ignore
            return getattr(_mod, "FRED_API_KEY", None)
        except ImportError:
            return None


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    project_root = path.resolve().parent
    cache = raw["cache"]
    cache_dir = project_root / cache["dir"]
    feats = raw.get("features", {})
    tgt = raw.get("target", {})
    mdl = raw.get("model", {})

    return Config(
        project_root=project_root,
        start=raw["dates"]["start"],
        end=raw["dates"]["end"],
        ref_start=raw["dates"]["ref_start"],
        vvix_min_start=feats.get("vvix_min_start", "2007-01-01"),
        return_windows=feats.get("return_windows", [1, 5, 10, 20, 60, 120]),
        vol_windows=feats.get("vol_windows", [5, 10, 20, 60]),
        use_technicals=feats.get("use_technicals", True),
        use_fomc=feats.get("use_fomc", True),
        horizons=tgt.get("horizons", {"week": 5, "month": 21, "quarter": 63}),
        n_buckets=tgt.get("n_buckets", 5),
        bucket_method=tgt.get("bucket_method", "percentile"),
        fixed_bins=tgt.get("fixed_bins", [-0.03, -0.01, 0.01, 0.03]),
        cache_dir=cache_dir,
        indices_path=cache_dir / cache["indices_file"],
        macro_path=cache_dir / cache["macro_file"],
        fomc_path=cache_dir / cache["fomc_file"],
        features_path=cache_dir / cache.get("features_file", "features.parquet"),
        targets_path=cache_dir / cache.get("targets_file", "targets.parquet"),
        signals_path=cache_dir / cache.get("signals_file", "oos_signals.parquet"),
        metrics_path=cache_dir / cache.get("metrics_file", "walkforward_metrics.csv"),
        model_names=mdl.get("names", ["logistic_clf", "histgbm_clf", "ridge_reg", "histgbm_reg"]),
        n_folds=mdl.get("n_folds", 6),
        min_train_frac=mdl.get("min_train_frac", 0.4),
        force_refresh=raw["acquire"]["force_refresh"],
        fred_api_key_env=raw["acquire"]["fred_api_key_env"],
        request_pause_sec=raw["acquire"]["request_pause_sec"],
        raw=raw,
    )