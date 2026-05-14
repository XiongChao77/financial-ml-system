"""
factor_layer1_single_asset.py

Layer-1 (single-asset time-series) factor screening tool:
- No model training, no full backtest
- Primary question: does the factor contain stable information about future returns along the time axis (time-series IC)?

For a single asset, IC is best measured as rolling-window time correlation:
  roll_IC(t) = corr( factor_{t-window+1:t}, fwd_ret_{t-window+1:t} )
Then summarize the roll_IC series: mean, |mean|, t-stat, positive ratio, quantiles, etc.

Optional: monthly/weekly sliced IC and conditional IC (bucketed by trend/volatility).

Dependencies: numpy, pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import os, sys
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir, '..'))
import model.train_2head as train
import data_process.common as common
from model import data_loader
EPS = 1e-12


# -------------------------
# Utils
# -------------------------

def winsorize(s: pd.Series, p: Optional[float] = 0.005) -> pd.Series:
    """Two-sided winsorization to suppress outliers. p=None disables it. Uses np.nanpercentile for speed."""
    if p is None or s.empty:
        return s
    arr = np.asarray(s.values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 3:
        return s
    lo, hi = np.nanpercentile(arr, [p * 100, (1 - p) * 100])
    return s.clip(lo, hi)


def _align_xy(x: pd.Series, y: pd.Series, winsor_p: Optional[float]) -> Tuple[pd.Series, pd.Series]:
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    if winsor_p is not None:
        df["x"] = winsorize(df["x"], winsor_p)
        df["y"] = winsorize(df["y"], winsor_p)
    return df["x"], df["y"]


def _is_constant(s: pd.Series, tol: float = 1e-10) -> bool:
    """Return True if series is constant/near-constant (var≈0) to avoid ConstantInputWarning in corr."""
    if s.empty or len(s) < 3:
        return True
    v = s.dropna()
    if len(v) < 3:
        return True
    return float(v.std(ddof=1)) <= tol


def corr_xy(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    if len(x) < 3:
        return np.nan
    if _is_constant(x) or _is_constant(y):
        return np.nan
    return float(x.corr(y, method=method))


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    """Constant/NaN-safe correlation to avoid ConstantInputWarning / RuntimeWarning."""
    if len(x) < 3 or len(y) < 3:
        return np.nan
    if _is_constant(x) or _is_constant(y):
        return np.nan
    try:
        v = float(x.corr(y, method=method))
        return v if not np.isnan(v) else np.nan
    except Exception:
        return np.nan


def t_stat_from_series(s: pd.Series) -> float:
    """t = mean / (std / sqrt(n))"""
    s = s.dropna()
    n = len(s)
    if n < 3:
        return np.nan
    mu = float(s.mean())
    sd = float(s.std(ddof=1))
    if sd <= 0 or np.isnan(sd):
        return np.nan
    return float(mu / (sd / np.sqrt(n)))


@dataclass
class Layer1SingleAssetConfig:
    # Main metric: Rank IC (spearman) is usually more robust; you can switch to pearson if desired.
    method: str = "spearman"
    winsor_p: Optional[float] = 0.005

    # Rolling IC settings (core for single-asset screening)
    rolling_window: int = 1000
    rolling_min_periods: int = 200

    # Time slicing (optional): check stability by month/week (ME=month end; pandas 2.0+)
    slice_freq: str = "ME"
    slice_min_samples: int = 100

    # Keep thresholds (tune for your data size)
    min_abs_ic: float = 0.01              # full-sample |IC|
    min_roll_ic_abs_mean: float = 0.005   # mean rolling |IC|
    min_roll_pos_ratio: float = 0.55      # ratio of rolling IC > 0
    min_roll_same_sign_ratio: float = 0.55
    min_roll_t: float = 1.0               # t-stat of rolling IC

    # Parallelism: set >1 when many factors exist
    n_jobs: int = 1


# -------------------------
# Core computations
# -------------------------

def full_sample_ic(
    factor: pd.Series,
    fwd_ret: pd.Series,
    method: str = "spearman",
    winsor_p: Optional[float] = 0.005,
) -> float:
    x, y = _align_xy(factor, fwd_ret, winsor_p)
    return corr_xy(x, y, method=method)


def rolling_time_ic(
    factor: pd.Series,
    fwd_ret: pd.Series,
    window: int,
    min_periods: int,
    method: str = "spearman",
    winsor_p: Optional[float] = 0.005,
) -> pd.Series:
    """
    Single-asset rolling time-correlation series.
    Vectorized: Spearman = Pearson(rank(x), rank(y)); Pearson uses rolling.corr directly.
    """
    x, y = _align_xy(factor, fwd_ret, winsor_p)
    if len(x) == 0:
        return pd.Series(dtype=float)
    if _is_constant(x) or _is_constant(y):
        return pd.Series(np.nan, index=x.index, name=f"roll_ic_{method}_{window}")

    # Ensure min_periods >= 3 (needed for correlation)
    mp = max(int(min_periods), 3)

    if method == "spearman":
        # Spearman = Pearson(rank(x), rank(y)); rank once, then use pandas vectorized rolling corr
        x_work = x.rank(method="average")
        y_work = y.rank(method="average")
    else:
        x_work = x
        y_work = y

    # Suppress ConstantInputWarning / RuntimeWarning from constant windows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            from scipy.stats import ConstantInputWarning
            warnings.simplefilter("ignore", category=ConstantInputWarning)
        except ImportError:
            warnings.filterwarnings("ignore", message=".*constant.*")
        roll_ic = x_work.rolling(window=window, min_periods=mp).corr(y_work)
    return roll_ic.rename(f"roll_ic_{method}_{window}")


def rolling_ic_stats(roll_ic: pd.Series, ic_full: Optional[float] = None) -> Dict[str, float]:
    s = roll_ic.replace([np.inf, -np.inf], np.nan).dropna()

    # same_sign_ratio: proportion of rolling IC sharing the same sign as ic_full
    same_sign_ratio = np.nan
    if ic_full is not None and (not np.isnan(ic_full)):
        ic_sign = np.sign(ic_full)
        if ic_sign != 0 and len(s) >= 3:
            same_sign_ratio = float((np.sign(s) == ic_sign).mean())

    if len(s) < 3:
        return {
            "roll_ic_mean": np.nan,
            "roll_ic_abs_mean": np.nan,
            "roll_ic_std": np.nan,
            "roll_ic_t": np.nan,
            "roll_pos_ratio": np.nan,
            "roll_same_sign_ratio": same_sign_ratio,
            "roll_q05": np.nan,
            "roll_q50": np.nan,
            "roll_q95": np.nan,
            "roll_n": int(len(s)),
        }

    return {
        "roll_ic_mean": float(s.mean()),
        "roll_ic_abs_mean": float(s.abs().mean()),
        "roll_ic_std": float(s.std(ddof=1)),
        "roll_ic_t": float(t_stat_from_series(s)),
        "roll_pos_ratio": float((s > 0).mean()),
        "roll_same_sign_ratio": same_sign_ratio,
        "roll_q05": float(s.quantile(0.05)),
        "roll_q50": float(s.quantile(0.50)),
        "roll_q95": float(s.quantile(0.95)),
        "roll_n": int(len(s)),
    }


def sliced_ic(
    factor: pd.Series,
    fwd_ret: pd.Series,
    time_index: pd.DatetimeIndex,
    freq: str = "ME",
    method: str = "spearman",
    winsor_p: Optional[float] = 0.005,
    min_samples: int = 100,
) -> pd.Series:
    """Compute sliced correlations by month/week to assess phase-wise stability."""
    x, y = _align_xy(factor, fwd_ret, winsor_p)
    if len(x) == 0:
        return pd.Series(dtype=float)

    # time_index must align with factor; reindex to x (since _align_xy may dropna)
    ti = pd.Series(np.asarray(time_index), index=range(len(time_index)))
    ti_aligned = ti.reindex(x.index)
    if ti_aligned.isna().any() or len(ti_aligned) != len(x):
        return pd.Series(dtype=float)

    df = pd.DataFrame({"x": x.values, "y": y.values}, index=pd.DatetimeIndex(ti_aligned)).dropna()
    if df.empty:
        return pd.Series(dtype=float)

    out = []
    for k, g in df.groupby(pd.Grouper(freq=freq)):
        if len(g) < min_samples:
            continue
        out.append((k, _safe_corr(g["x"], g["y"], method)))
    return pd.Series(dict(out)).sort_index()


def conditional_ic_by_sign(
    factor: pd.Series,
    fwd_ret: pd.Series,
    condition: pd.Series,
    method: str = "spearman",
    winsor_p: Optional[float] = 0.005,
    min_samples: int = 200,
) -> pd.DataFrame:
    """Bucket conditional variable by sign (pos/neg), suitable for 0-centered state variables."""
    x, y = _align_xy(factor, fwd_ret, winsor_p)
    c = condition.reindex(x.index)
    df = pd.concat([x.rename("x"), y.rename("y"), c.rename("c")], axis=1).dropna()
    if len(df) < min_samples:
        return pd.DataFrame()

    rows = []
    for name, mask in {"cond_pos": df["c"] > 0, "cond_neg": df["c"] <= 0}.items():
        g = df[mask]
        if len(g) < min_samples:
            continue
        rows.append({"bucket": name, "n": int(len(g)), "ic": _safe_corr(g["x"], g["y"], method)})
    return pd.DataFrame(rows)


def conditional_ic_by_quantile(
    factor: pd.Series,
    fwd_ret: pd.Series,
    condition: pd.Series,
    q: int = 5,
    method: str = "spearman",
    winsor_p: Optional[float] = 0.005,
    min_samples: int = 200,
) -> pd.DataFrame:
    """Bucket conditional variable by quantiles, suitable for RV_20 / volume_z etc."""
    x, y = _align_xy(factor, fwd_ret, winsor_p)
    c = condition.reindex(x.index)
    df = pd.concat([x.rename("x"), y.rename("y"), c.rename("c")], axis=1).dropna()
    if len(df) < min_samples:
        return pd.DataFrame()

    try:
        df["bucket"] = pd.qcut(df["c"], q=q, duplicates="drop")
    except Exception:
        return pd.DataFrame()

    rows = []
    for b, g in df.groupby("bucket", observed=True):
        if len(g) < min_samples:
            continue
        rows.append({"bucket": str(b), "n": int(len(g)), "ic": _safe_corr(g["x"], g["y"], method)})
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def _screen_one_factor(
    feat: str,
    fac: pd.Series,
    fwd: pd.Series,
    cfg: Layer1SingleAssetConfig,
    time_index: Optional[pd.DatetimeIndex],
    conditions: Optional[Dict[str, Dict]],
    factors_index: pd.Index,
) -> Tuple[Dict, Dict[str, pd.DataFrame]]:
    """Single-factor screening (used by parallel execution)."""
    ic = full_sample_ic(fac, fwd, method=cfg.method, winsor_p=cfg.winsor_p)
    roll = rolling_time_ic(
        fac, fwd,
        window=cfg.rolling_window,
        min_periods=cfg.rolling_min_periods,
        method=cfg.method,
        winsor_p=cfg.winsor_p,
    )
    rstats = rolling_ic_stats(roll, ic_full=ic)

    slice_mean = slice_t = slice_n = np.nan
    if time_index is not None and len(time_index) == len(fac):
        s = sliced_ic(
            fac, fwd, time_index,
            freq=cfg.slice_freq,
            method=cfg.method,
            winsor_p=cfg.winsor_p,
            min_samples=cfg.slice_min_samples,
        )
        slice_n = int(len(s))
        if slice_n >= 3:
            slice_mean = float(s.mean())
            slice_t = float(t_stat_from_series(s))
    
    keep = (
        (not np.isnan(ic)) and (abs(ic) >= cfg.min_abs_ic) and
        (not np.isnan(rstats["roll_ic_abs_mean"])) and (rstats["roll_ic_abs_mean"] >= cfg.min_roll_ic_abs_mean) and
        (not np.isnan(rstats["roll_same_sign_ratio"])) and (rstats["roll_same_sign_ratio"] >= cfg.min_roll_same_sign_ratio) and
        (np.isnan(rstats["roll_ic_t"]) or abs(rstats["roll_ic_t"] )>= cfg.min_roll_t)
    )

    # same_sign_ratio>0.6 and ic_full<0 => stable negative factor (ic_direction=-1); otherwise ic_direction=1
    ssr = rstats.get("roll_same_sign_ratio", np.nan)
    ic_dir = -1 if (
        (not np.isnan(ssr)) and (ssr > 0.6) and
        (not np.isnan(ic)) and (float(ic) < 0)
    ) else 1

    row = {
        "feature": feat,
        "ic_full": float(ic),
        "ic_full_abs": float(abs(ic)) if not np.isnan(ic) else np.nan,
        **rstats,
        "ic_direction": ic_dir,
        "slice_ic_mean": slice_mean,
        "slice_ic_t": slice_t,
        "slice_n": slice_n,
        "keep": bool(keep),
    }

    cond_part: Dict[str, pd.DataFrame] = {}
    if conditions:
        for cname, spec in conditions.items():
            ser = spec.get("series")
            if ser is None:
                continue
            c = pd.Series(np.asarray(ser), index=factors_index, name=cname)
            mode = spec.get("mode", "sign")
            if mode == "sign":
                tab = conditional_ic_by_sign(
                    fac, fwd, c,
                    method=cfg.method,
                    winsor_p=cfg.winsor_p,
                    min_samples=int(spec.get("min_samples", 200)),
                )
            else:
                tab = conditional_ic_by_quantile(
                    fac, fwd, c,
                    q=int(spec.get("q", 5)),
                    method=cfg.method,
                    winsor_p=cfg.winsor_p,
                    min_samples=int(spec.get("min_samples", 200)),
                )
            cond_part[cname] = tab

    return row, cond_part


def layer1_screen_single_asset(
    factors_df: pd.DataFrame,
    fwd_ret: Union[pd.Series, np.ndarray],
    time_index: Optional[Union[pd.DatetimeIndex, Sequence]] = None,
    cfg: Layer1SingleAssetConfig = Layer1SingleAssetConfig(),
    conditions: Optional[Dict[str, Dict]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Returns:
      - summary: full-sample IC + rolling IC stats + (optional) sliced IC + keep flag per factor
      - cond_tables: {cond_name: {feature: DataFrame}}

    time_index: DatetimeIndex aligned with factors_df rows; used by sliced_ic for month/week grouping.
      If using TimeSeriesWindowDataset, you can get it via get_time_index_from_dataset(ds, df) or
      the third return value of build_from_timeseries_window_dataset(..., df=df). If None, sliced_ic is skipped.

    Example conditions:
      {
        "trend_week": {"series": df["MA_WEEK_M_L"], "mode": "sign", "min_samples": 500},
        "vol": {"series": df["RV_20"], "mode": "quantile", "q": 5, "min_samples": 500},
      }
    """
    fwd = pd.Series(np.asarray(fwd_ret), index=factors_df.index, name="fwd_ret")
    ti = pd.DatetimeIndex(time_index) if time_index is not None else None
    n_jobs = getattr(cfg, "n_jobs", 1) or 1

    if n_jobs > 1:
        # Parallel: ThreadPool (pandas/numpy often release GIL)
        rows, cond_tables = [], {}
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = {
                ex.submit(
                    _screen_one_factor,
                    feat, factors_df[feat], fwd, cfg, ti, conditions, factors_df.index,
                ): feat
                for feat in factors_df.columns
            }
            for fut in as_completed(futures):
                row, cond_part = fut.result()
                rows.append(row)
                for cname, tab in cond_part.items():
                    cond_tables.setdefault(cname, {})[futures[fut]] = tab
    else:
        rows, cond_tables = [], {}
        for feat in factors_df.columns:
            row, cond_part = _screen_one_factor(
                feat, factors_df[feat], fwd, cfg, ti, conditions, factors_df.index
            )
            rows.append(row)
            for cname, tab in cond_part.items():
                cond_tables.setdefault(cname, {})[feat] = tab

    summary = pd.DataFrame(rows).sort_values(["keep", "ic_full_abs"], ascending=[False, False]).reset_index(drop=True)
    return summary, cond_tables


def get_selected_factors(summary: pd.DataFrame) -> List[str]:
    """Extract factor names that passed screening (for downstream pipelines)."""
    if "keep" not in summary.columns:
        return []
    return summary.loc[summary["keep"], "feature"].tolist()


def load_selected_factors(path: str) -> List[str]:
    """Load selected factor list from selected_factors.txt for training/GA pipelines."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _build_conditions_safe(
    factors_df: pd.DataFrame,
    trend_col: str = "MA_WEEK_M_L",
    vol_col: str = None,
) -> Optional[Dict[str, Dict]]:
    """
    Safely build conditions: only add entries when the referenced columns exist.
    vol_col prefers RV_20; otherwise tries atr_14 / vol_regime_* etc.
    """
    conditions = {}
    if trend_col in factors_df.columns:
        conditions["trend_week"] = {
            "series": factors_df[trend_col],
            "mode": "sign",
            "min_samples": 500,
        }
    # Volatility proxy: RV_20 may not exist; use atr_14 or the first atr_*/vol_regime_* found.
    vol_candidates = [vol_col] if vol_col else ["RV_20", "atr_14", "atr_16"]
    vol_candidates += [c for c in factors_df.columns if c.startswith("atr_") or c.startswith("vol_regime_")]
    for col in vol_candidates:
        if col and col in factors_df.columns:
            conditions["vol"] = {
                "series": factors_df[col],
                "mode": "quantile",
                "q": 5,
                "min_samples": 500,
            }
            break
    return conditions if conditions else None


# -------------------------
# Helper for your Dataset-style pipeline
# -------------------------

def get_time_index_from_dataset(ds, df: pd.DataFrame) -> Optional[pd.DatetimeIndex]:
    """
    Build time_index from TimeSeriesWindowDataset and the original df.
    Used by layer1_screen_single_asset for sliced_ic (monthly/weekly stability checks).
    df must be the DataFrame passed to the Dataset, containing open_time_ms_utc or open_time_date_utc.
    """
    if not hasattr(ds, "indices") or ds.indices is None:
        return None
    time_col = "open_time_ms_utc" if "open_time_ms_utc" in df.columns else "open_time_date_utc"
    if time_col not in df.columns:
        return None
    try:
        idx = np.asarray(ds.indices)
        # indices refer to rows in the original df (each sample maps to the last bar of the window)
        ts = df.loc[idx, time_col]
        if time_col == "open_time_ms_utc":
            return pd.to_datetime(ts, unit="ms", utc=True)
        return pd.to_datetime(ts, utc=True)
    except Exception:
        return None


def build_from_timeseries_window_dataset(
    ds,
    feature_names: Optional[List[str]] = None,
    drop_cols: Optional[Iterable[str]] = None,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DatetimeIndex]]:
    """
    Adapter for TimeSeriesWindowDataset:
      factors = ds.X[:, -1, :]
      fwd_ret = ds.returns
    If df (the Dataset's original DataFrame) is provided, also return time_index for sliced_ic.
    """
    X = ds.X
    R = ds.returns

    if hasattr(X, "detach"):
        X_last = X[:, -1, :].detach().cpu().numpy()
    else:
        X_last = np.asarray(X)[:, -1, :]

    cols = list(feature_names) if feature_names is not None else list(ds.feature_names)
    factors = pd.DataFrame(X_last, columns=cols)

    drops = set(drop_cols or [])
    if "z_ret" in factors.columns:
        drops.add("z_ret")
    if drops:
        factors = factors.drop(columns=[c for c in drops if c in factors.columns])

    if hasattr(R, "detach"):
        ret = R.detach().cpu().numpy()
    else:
        ret = np.asarray(R)
    fwd_ret = pd.Series(ret, index=factors.index, name="fwd_ret")

    time_index = get_time_index_from_dataset(ds, df) if df is not None else None
    return factors, fwd_ret, time_index

def main():
    # 1. Basic setup
    logger, _ = common.setup_session_logger(sub_folder='correlation_result')
    pre_task = common.BaseDefine()
    pre_task.interval = '15m'
    train_cfg = train.TrainConfig()
    
    csv_path = os.path.join(common.PROJECT_DATA_DIR, f"{pre_task.symbol}_{pre_task.interval}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing data: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df = common.clean_data_quality_auto(df, logger)
    df = common.attach_attr(df, common.FEATURE_GROUP_LIST, para=pre_task)
    common.attach_label(df, pre_task)
    # Ensure z_ret exists
    if 'z_ret' not in df.columns:
        logger.warning("z_ret not found in df; attempting to compute it.")
        df['z_ret'] = df['close'].pct_change(pre_task.predict_num).shift(-pre_task.predict_num)

    # 2. Build feature list
    # Trick: include z_ret temporarily for dataset alignment so we can extract synchronized cross-sections.
    all_cols = [c for c in df.columns if c not in data_loader.DROP_FEATURES]
    if 'z_ret' not in all_cols: all_cols.append('z_ret')
    
    ds = data_loader.TimeSeriesWindowDataset(
        df=df, 
        kline_interval_ms=common.get_interval_ms(pre_task.interval),
        feature_cols=all_cols,
        label_col='label',
        window=train_cfg.data_cfg.window,
        stride=train_cfg.stride,
        use_cache=False
    )
    # 1) Extract cross-sectional factors + forward returns and time_index from dataset
    factors_df, fwd_ret, time_index = build_from_timeseries_window_dataset(
        ds,
        drop_cols=["z_ret"],
        df=df,
    )

    # 2) Config (defaults are ok to start)
    cfg = Layer1SingleAssetConfig(
        rolling_window=1000,
        rolling_min_periods=300,
    )

    # 3) Conditional IC (safely built only when columns exist)
    conditions = _build_conditions_safe(factors_df)

    # 4) Run layer-1 factor screening
    summary, cond_tables = layer1_screen_single_asset(
        factors_df=factors_df,
        fwd_ret=fwd_ret,
        time_index=time_index,
        cfg=cfg,
        conditions=conditions,
    )

    # 5) Output: save CSV and print results
    out_dir = os.path.join(common.PERSISTENCE_DIR, "factor_layer1_result")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "factor_screen_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    logger.info(f"Factor screening results saved: {summary_path}")

    selected = get_selected_factors(summary)
    logger.info(f"Factors kept: {len(selected)} / {len(summary)}")
    if selected:
        logger.info(f"Kept factors: {selected[:20]}{' ...' if len(selected) > 20 else ''}")
    else:
        logger.warning("No factors passed screening. Consider relaxing min_* thresholds in cfg.")

    # Save factor list for GA / training pipelines
    selected_path = os.path.join(out_dir, "selected_factors.txt")
    with open(selected_path, "w", encoding="utf-8") as f:
        f.write("\n".join(selected))
    logger.info(f"Selected factor list: {selected_path}")

    print("\n📊 Factor screening top 15 (sorted by keep + ic_full_abs):")
    print(summary.head(15).to_string())
if __name__ == "__main__":
    main()