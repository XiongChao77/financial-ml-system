"""
factor_layer1_single_asset.py

Layer-1（单资产时间序列）因子粗筛工具：
- 不训练模型、不做完整回测
- 主要回答：因子是否在“时间维度”上对未来收益有稳定信息（Time-series IC）

单资产下的“IC”建议使用 rolling window 的时间相关：
  roll_IC(t) = corr( factor_{t-window+1:t}, fwd_ret_{t-window+1:t} )
并对 roll_IC 序列统计：均值、|均值|、t-stat、正值占比、分位数等。

可选：按月/周切片相关（slice IC）与条件 IC（按趋势/波动分桶）。

依赖：numpy, pandas
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
    """两侧截断抑制极端值。p=None 表示不处理。用 np.nanpercentile 加速。"""
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
    """常数或近常数（方差≈0）则 True，避免 corr 触发 ConstantInputWarning。"""
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
    """常数/NaN 安全的相关计算，避免 ConstantInputWarning / RuntimeWarning。"""
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
    # 主度量：Rank IC（spearman）更稳；你也可以改成 pearson
    method: str = "spearman"
    winsor_p: Optional[float] = 0.005

    # rolling IC 参数（单资产的核心）
    rolling_window: int = 1000
    rolling_min_periods: int = 200

    # 时间切片（可选）：用于观察“按月/周”是否稳定（ME=month end，pandas 2.0+）
    slice_freq: str = "ME"
    slice_min_samples: int = 100

    # keep 的门槛（按你数据规模调）
    min_abs_ic: float = 0.01              # 全样本 |IC|
    min_roll_ic_abs_mean: float = 0.005   # rolling |IC| 的均值
    min_roll_pos_ratio: float = 0.55      # rolling IC > 0 的比例
    min_roll_same_sign_ratio: float = 0.55
    min_roll_t: float = 1.0               # rolling IC 的 t-stat

    # 并行：因子数多时可设为 >1
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
    单资产：rolling window 的时间相关序列。
    向量化实现：Spearman = rank 后 rolling Pearson，Pearson 直接用 rolling.corr。
    """
    x, y = _align_xy(factor, fwd_ret, winsor_p)
    if len(x) == 0:
        return pd.Series(dtype=float)
    if _is_constant(x) or _is_constant(y):
        return pd.Series(np.nan, index=x.index, name=f"roll_ic_{method}_{window}")

    # 确保 min_periods 至少 3（相关计算需要）
    mp = max(int(min_periods), 3)

    if method == "spearman":
        # Spearman = Pearson(rank(x), rank(y))，一次性 rank 后利用 pandas 向量化 rolling
        x_work = x.rank(method="average")
        y_work = y.rank(method="average")
    else:
        x_work = x
        y_work = y

    # 抑制 rolling 窗口内常数列导致的 ConstantInputWarning / RuntimeWarning
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

    # same_sign_ratio：rolling IC 与 ic_full 同号比例
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
    """按月/周切片计算相关，用于观察“分阶段”是否稳定。"""
    x, y = _align_xy(factor, fwd_ret, winsor_p)
    if len(x) == 0:
        return pd.Series(dtype=float)

    # time_index 需与 factor 同长、同序；对齐到 x 的 index（_align_xy 可能 dropna）
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
    """条件变量按正/负两桶（适合以0为中性的状态变量）。"""
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
    """条件变量按分位数分桶（适合 RV_20 / volume_z 等）。"""
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
    """单因子筛选，供并行调用。"""
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

    # same_sign_ratio>0.6 且 ic_full<0：负向稳定因子，标注 -1；其余为 1
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
    返回：
      - summary: 每个因子的全样本 IC + rolling IC 统计 + (可选)切片 IC + keep
      - cond_tables: {cond_name: {feature: DataFrame}}

    time_index: 与 factors_df 等长、逐行对应的 DatetimeIndex，用于 sliced_ic 按月/周分组。
      若使用 TimeSeriesWindowDataset，可通过 get_time_index_from_dataset(ds, df) 或
      build_from_timeseries_window_dataset(..., df=df) 的第三返回值获取。传 None 则跳过 sliced_ic。

    conditions 示例：
      {
        "trend_week": {"series": df["MA_WEEK_M_L"], "mode": "sign", "min_samples": 500},
        "vol": {"series": df["RV_20"], "mode": "quantile", "q": 5, "min_samples": 500},
      }
    """
    fwd = pd.Series(np.asarray(fwd_ret), index=factors_df.index, name="fwd_ret")
    ti = pd.DatetimeIndex(time_index) if time_index is not None else None
    n_jobs = getattr(cfg, "n_jobs", 1) or 1

    if n_jobs > 1:
        # 并行：ThreadPool（pandas/numpy 多释放 GIL，可加速）
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
    """从 screening 结果中提取筛选通过的因子名列表，便于下游 pipeline 使用。"""
    if "keep" not in summary.columns:
        return []
    return summary.loc[summary["keep"], "feature"].tolist()


def load_selected_factors(path: str) -> List[str]:
    """从 selected_factors.txt 加载筛选后的因子列表，供训练/GA 等下游使用。"""
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
    安全构建 conditions：仅在对应列存在时加入。
    vol_col 优先 RV_20，否则尝试 atr_14 / vol_regime_* 等。
    """
    conditions = {}
    if trend_col in factors_df.columns:
        conditions["trend_week"] = {
            "series": factors_df[trend_col],
            "mode": "sign",
            "min_samples": 500,
        }
    # 波动率代理：RV_20 可能不存在，用 atr_14 或第一个 atr_* / vol_regime_* 替代
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
    从 TimeSeriesWindowDataset 和原始 df 构造 time_index。
    用于 layer1_screen_single_asset 的 sliced_ic（按月/周观察因子稳定性）。
    要求：df 为传入 Dataset 的原始 DataFrame，需含 open_time_ms_utc 或 open_time_date_utc。
    """
    if not hasattr(ds, "indices") or ds.indices is None:
        return None
    time_col = "open_time_ms_utc" if "open_time_ms_utc" in df.columns else "open_time_date_utc"
    if time_col not in df.columns:
        return None
    try:
        idx = np.asarray(ds.indices)
        # indices 为原始 df 的 index 标签（每样本对应窗口最后一根 K 线的行）
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
    适配 TimeSeriesWindowDataset：
      factors = ds.X[:, -1, :]
      fwd_ret = ds.returns
    若传入 df（传入 Dataset 的原始 DataFrame），则同时返回 time_index 供 sliced_ic 使用。
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
    if "return_rate" in factors.columns:
        drops.add("return_rate")
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
    # 1. 基础准备
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
    # 确保 return_rate 存在
    if 'return_rate' not in df.columns:
        logger.warning("df 中未找到 return_rate，将尝试计算。")
        df['return_rate'] = df['close'].pct_change(pre_task.predict_num).shift(-pre_task.predict_num)

    # 2. 构造特征列表
    # 核心技巧：将 return_rate 暂时放入 feat_cols 参与 Dataset 窗口对齐，这样能拿到与特征同步的截面数据
    all_cols = [c for c in df.columns if c not in data_loader.DROP_FEATURES]
    if 'return_rate' not in all_cols: all_cols.append('return_rate')
    
    ds = data_loader.TimeSeriesWindowDataset(
        df=df, 
        kline_interval_ms=common.get_interval_ms(pre_task.interval),
        feature_cols=all_cols,
        label_col='label',
        window=train_cfg.data_cfg.window,
        stride=train_cfg.stride,
        use_cache=False
    )
    # 1) 从 dataset 里取“单资产截面特征 + 未来收益”及 time_index
    factors_df, fwd_ret, time_index = build_from_timeseries_window_dataset(
        ds,
        drop_cols=["return_rate"],
        df=df,
    )

    # 2) 配置（可先用默认）
    cfg = Layer1SingleAssetConfig(
        rolling_window=1000,
        rolling_min_periods=300,
    )

    # 3) 条件 IC（安全构建：列存在才加入，避免 KeyError）
    conditions = _build_conditions_safe(factors_df)

    # 4) 跑 Layer-1 因子筛选
    summary, cond_tables = layer1_screen_single_asset(
        factors_df=factors_df,
        fwd_ret=fwd_ret,
        time_index=time_index,
        cfg=cfg,
        conditions=conditions,
    )

    # 5) 输出：保存 CSV + 打印筛选结果
    out_dir = os.path.join(common.PERSISTENCE_DIR, "factor_layer1_result")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "factor_screen_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    logger.info(f"因子筛选结果已保存: {summary_path}")

    selected = get_selected_factors(summary)
    logger.info(f"通过筛选的因子数: {len(selected)} / {len(summary)}")
    if selected:
        logger.info(f"保留因子: {selected[:20]}{' ...' if len(selected) > 20 else ''}")
    else:
        logger.warning("无因子通过筛选，可适当放宽 cfg 中的 min_* 阈值")

    # 保存筛选后因子列表，供 GA / 训练 pipeline 使用
    selected_path = os.path.join(out_dir, "selected_factors.txt")
    with open(selected_path, "w", encoding="utf-8") as f:
        f.write("\n".join(selected))
    logger.info(f"筛选因子列表: {selected_path}")

    print("\n📊 因子筛选 Top 15 (按 keep + ic_full_abs):")
    print(summary.head(15).to_string())
if __name__ == "__main__":
    main()