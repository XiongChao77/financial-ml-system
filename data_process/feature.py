from abc import ABC, abstractmethod
import logging,math,os
import pandas as pd
import numpy as np
from numba import njit, float64, int64

EPS = 1e-9 # 防止除以 0
_logger = logging.getLogger()
#All features should be based on this
class FeatureBase(ABC):
    def __init__(self, factory, **kwargs): 
        self.params = kwargs
        self.features :list[str]= []
        self.factory : FeatureFactory= factory
        self.kline_interval_ms:int = kwargs.get('kline_interval_ms', None)
    #方差和均值做除数区别:方差会放大小波动，缩小大波动

    def _apply_squashing(self, vals, scale, method):
            """
            统一的长尾压制逻辑应用器
            vals: 已经经过 (X-mu)/sigma 处理的标准分
            scale: 线性区半径。数值越大，压制越晚触发。
            """
            if method is None:
                # 如果不压制，且 scale 为 1.0，直接返回以节省 CPU
                return vals #if (scale == 1.0 or scale == None) else (vals / scale)
            # 1.  动态确定信任半径 S
            if scale is None:
                # 取当前窗口绝对值的分位数作为 S，代表“90% 的数据分布范围”
                scale = np.nanpercentile(np.abs(vals), 95, keepdims=True)
                # print(f"***********************{self.__class__.__name__} scale is {scale}*******************")
                # scale = np.maximum(scale, 1.0) # 保底为 1，防止波动过小时过度放大噪声
            else:
                scale = scale
            if method == 'tanh':
                # tanh(1.0) = 0.76. 所以 scale 直接用 raw_s 即可
                adj_scale = scale / 1.0 
                # print(f"***********************{self.__class__.__name__} tanh adj_scale is {adj_scale}*******************")
                result = np.tanh(vals / adj_scale)
            elif method == 'log':
                # ln(1 + 1.22) = 0.8. 如果想让 95% 分位达到 0.8，需要除以 raw_s / 1.22
                adj_scale = scale #* 1.22 
                # print(f"***********************{self.__class__.__name__} log adj_scale is {adj_scale}*******************")
                result = np.sign(vals) * np.log1p(np.abs(vals / adj_scale))

            # for pct in range(50, 100,5):
            #     result_m = np.nanpercentile(np.abs(result), pct, keepdims=True)
            #     print(f"***********************{self.__class__.__name__} {pct} scale result is {result_m}*******************")
            # for pct in range(96, 100,1):
            #     result_m = np.nanpercentile(np.abs(result), pct, keepdims=True)
            #     print(f"***********************{self.__class__.__name__} {pct} scale result is {result_m}*******************")
            return result # method=None，保持线性

    def _get_target_indices(self, feature_cols: list[str], target_feature_cols: list[str]):
        """
        [叠加过滤核心] 
        1. 检查特征名是否存在于 factory._feature_index 中
        2. 检查特征名是否存在于当前传入的 feature_cols 列表中
        返回: (有效索引列表, 有效特征名列表)
        """
        # 转换为 set 提高查找效率
        cols_set = set(feature_cols)
        valid_indices = []
        valid_names = []
        
        for f in target_feature_cols:
            # 叠加判断：既要在全局索引库中，也要在本次处理的列中
            if f in self.factory._feature_index and f in cols_set:
                valid_indices.append(self.factory._feature_index[f])
                valid_names.append(f)
                
        return valid_indices, valid_names

    def _normalize_z_score(self, X, feature_cols, target_feature_cols, feature_base, factory, scale=None, method=None):
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return
        mu_base, sigma_base = factory.get_base_stats(feature_base)
        mu_base_3d = mu_base[:, :, np.newaxis] if mu_base.ndim == 2 else mu_base.reshape(-1, 1, 1)
        denom = sigma_base[:, :, np.newaxis] 
        standardized = np.where(
            denom > EPS,
            (X[:, :, target_indices] - mu_base_3d) / denom,
            0.0
        )
        # 2. 应用倍率缩放与长尾压制
        X[:, :, target_indices] = self._apply_squashing(standardized, scale, method)

    def _normalize_z_score_group(self, X, feature_cols, target_feature_cols, factory, scale=None, method=None):
        """
        Group Z-Score Normalization (联合标准化)
        逻辑：将一组特征视为一个整体，计算它们在整个窗口时间轴+特征轴上的统一均值和标准差。
        优点：保护特征组内部的相对距离（如 Upper 与 Lower 的间距）。
        """
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return
        
        # 2. 提取目标组数据，形状为 (Samples, Time, Group_Features)
        vals = X[:, :, target_indices]
        
        # 3.  核心计算：跨时间轴 (axis=1) 和 特征轴 (axis=2) 进行池化统计
        # 这样每一个 Batch 样本都会得到唯一的 [1, 1, 1] 形状的统计量
        group_mu = np.nanmean(vals, axis=(1, 2), keepdims=True)
        group_sigma = np.nanstd(vals, axis=(1, 2), keepdims=True)
        
        # 4. 执行标准化计算
        # 使用 EPS 保护，若 group 内没有任何波动 (std=0)，则结果归零
        standardized = np.where(
            group_sigma > EPS,
            (vals - group_mu) / group_sigma,
            0.0
        )
        X[:, :, target_indices] = self._apply_squashing(standardized, scale, method)

    #method 'tanh'/'log'
    def _normalize_signal_group(self, X, feature_cols, target_feature_cols, factory, scale=None, method=None):
        """
        全功能零轴锚定组缩放：
        k: 线性区缩放因子。k 越小，线性区间越宽，压制越晚触发。
        method: 'tanh' (映射到 -1~1) 或 'log' (Symmetric Log1p)。
        """
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return
        
        # 1. 提取组数据并计算 RMS (确保零轴不偏移，组内比例一致)
        vals = X[:, :, target_indices]
        rms_group = np.sqrt(np.nanmean(vals**2, axis=(1, 2), keepdims=True))
        
        # 2. 无量纲化基础缩放
        standardized = np.where(rms_group > EPS, vals / (rms_group), 0.0)

        X[:, :, target_indices] = self._apply_squashing(standardized, scale, method)

    def _normalize_z_score_rel(self, X, feature_cols, target_feature_cols, feature_base, factory, scale=None, method=None):
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return
        mu_base, sigma_base = factory.get_base_stats(feature_base)
        denom = sigma_base[:, :, np.newaxis] 
        mu_self = np.nanmean(X[:, :, target_indices], axis=1, keepdims=True)
        standardized = np.where(
            denom > EPS,
            (X[:, :, target_indices] - mu_self) / denom,
            0.0
        )
        X[:, :, target_indices] = self._apply_squashing(standardized, scale, method)

    def _normalize_winsorized_z_score_group(self, X, feature_cols, target_feature_cols, feature_base, factory):
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return
        vals = X[:, :, target_indices]
        
        # 1.  预处理：在算统计量前，先把原始值限制在 1% ~ 99% 分位数之间
        # 这样可以防止极端插针拉大标准差
        p_low = np.nanpercentile(vals, 1, axis=1, keepdims=True)
        p_high = np.nanpercentile(vals, 99, axis=1, keepdims=True)
        np.clip(vals, p_low, p_high, out=vals)
        
        mu_base, sigma_base = factory.get_base_stats(feature_base)
        denom = sigma_base[:, :, np.newaxis] 
        mu_self = np.nanmean(vals, axis=1, keepdims=True)
        X[:, :, target_indices] = np.where(
            denom > EPS,
            (vals - mu_self) / denom,
            0.0
        )

    def _normalize_winsorized_z_score(self, X, feature_cols, target_feature_cols, factory):
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return
        vals = X[:, :, target_indices]
        
        # 1.  预处理：在算统计量前，先把原始值限制在 1% ~ 99% 分位数之间
        # 这样可以防止极端插针拉大标准差
        p_low = np.nanpercentile(vals, 1, axis=1, keepdims=True)
        p_high = np.nanpercentile(vals, 99, axis=1, keepdims=True)
        vals_clipped = np.clip(vals, p_low, p_high)
        
        # 2. 基于“温顺”后的数据计算统计量
        mu_win = np.nanmean(vals_clipped, axis=1, keepdims=True)
        sigma_win = np.nanstd(vals_clipped, axis=1, keepdims=True)
        
        # 3. 用“温顺”的统计量去归一化“原始”数据（或者也归一化温顺数据）
        X[:, :, target_indices] = np.where(
            sigma_win > EPS,
            (vals_clipped - mu_win) / sigma_win,
            0.0
        )

    def _normalize_signal(
        self, X, feature_cols, target_feature_cols, feature_base, factory
    ):
        """
        用于变化率 / 方向 / 相对偏离特征
        0 为中性，对称
        σ=0 → 0
        """
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return

        mu, sigma = self.factory.get_base_stats(feature_base)

        X[:, :, target_indices] = np.where(
            sigma[:, :, None] > 0,
            (X[:, :, target_indices] - mu[:, :, None]) / sigma[:, :, None],
            0.0
        )

    #抑制长尾
    def _normalize_volume_rlc(self, X, feature_cols, target_feature_cols, feature_base):
        """
        Relative Log-Compression (RLC) - 相对对数压缩缩放
        针对长尾分布（如成交量、成交额）设计，结合了无量纲化与极值抑制。
        
        :param feature_base: 作为无量纲化基准的特征名（如 'volume' 或 'quote_asset_volume'）
        """
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        if not target_indices:
            return

        # 2. 获取基准统计量 (使用 self.factory 访问)
        # 获取基准列的均值 mu，用于将目标特征转化为“均值的倍数”
        mu_base, _ = self.factory.get_base_stats(feature_base)
        mu_base = mu_base[:, :, np.newaxis] # 增加维度以支持广播计算 (M, 1, 1)

        # 严格检查：如果基准列均值为 0，说明数据源该列全为 0，属于严重异常
        if np.any(mu_base == 0):
            raise ValueError(f"RLC Error: Base feature '{feature_base}' has zero mean. Data invalid.")

        # 3. 提取数据
        X_target = X[:, :, target_indices]

        # 4. 执行无量纲化 + 对数压缩
        # 公式：log1p( X / mu_base )
        # 物理意义：
        # - 0 依然映射为 0 (log1p(0))
        # - 当 X 等于均值时，映射为 log(2) ≈ 0.69
        # - 当 X 是均值的 100 倍时，映射为 log(101) ≈ 4.6
        # 这种方式在保留了“放量”相对强度的同时，极大地抑制了长尾极值
        X[:, :, target_indices] = np.log1p(np.maximum(X_target / mu_base, 0.0))
    def _normalize_scs(self, X, feature_cols, target_feature_cols, feature_base):
        """
        Structural-Consistency Scaling (SCS) 结构一致性缩放
        
        逻辑：
        1. 识别索引：获取目标特征组和基准特征（如 Close）在 X 中的位置。
        2. 无量纲化：将所有目标特征除以基准特征，转化为相对比例（Ratios）。
        3. 整体池化：在单个样本(M)的时间轴(T)和特征轴(F_sub)上计算全局 Min/Max。
        4. 等比映射：将整组比例值线性缩放到 [0, 1]，保留它们之间的相对几何距离。
        """
        # 1. 获取索引
        target_indices, _ = self._get_target_indices(feature_cols, target_feature_cols)
        base_idx = self.factory._feature_index.get(feature_base)
        
        if not target_indices or base_idx is None:
            return

        # 2. 提取数据与基准 (M, T, F_sub) 和 (M, T, 1)
        X_target = X[:, :, target_indices]
        X_base = X[:, :, [base_idx]]
        # --- 严格检查 A: 基准特征是否存在 0 ---
        # 比如 Close 价格为 0，这在金融数据中属于严重异常
        if np.any(X_base == 0):
            # 找到异常发生的样本索引，方便排查数据源
            sample_idx = np.where(X_base == 0)[0][0]
            raise ValueError(f"SCS Error: Base feature '{feature_base}' contains zero at sample index {sample_idx}. "
                             f"This indicates data corruption or price zeroing.")
        # 3. 计算相对比例 (Ratio = P / Base)
        ratios = X_target / X_base

        # 4. 计算组内全局统计量
        group_min = np.nanmin(ratios, axis=(1, 2), keepdims=True)
        group_max = np.nanmax(ratios, axis=(1, 2), keepdims=True)
        group_diff = group_max - group_min

        # --- 严格检查 B: 组内是否存在波动 ---
        # 如果 max == min，说明整组特征在整个窗口内完全没有变化（例如全是一字板或常数）
        # 这种情况会导致 Min-Max 缩放公式的分母为 0
        if np.any(group_diff == 0):
            raise ZeroDivisionError(f"SCS Error: Feature group {target_feature_cols} has zero variance "
                                    f"within the window. Check if these features are constant.")

        # 5. 执行缩放
        X[:, :, target_indices] = (ratios - group_min) / group_diff
    @abstractmethod
    def generate(self,df:pd.DataFrame,kline_interval_ms) -> None: ...
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory) : pass
    @abstractmethod
    def _min_history_request(self, kline_interval_ms:int = None) -> int: pass
    def min_history_request(self, kline_interval_ms:int = None) -> int:
        min_request= self._min_history_request(kline_interval_ms)
        _logger.debug(f"{self.__class__.__name__} min_history_request {min_request}")
        return min_request
"""
直接在原数据上添加 MACD 与多条均线（同时包含 SMA 与 EMA）。
生成列：
    - MACD_DIF, MACD_DEA, MACD
    - MA_{w}  （简单移动均线）
    - EMA_{w} （指数移动均线，严格：窗口未满置 NaN）
"""
class FeatureMACD(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fast = kwargs.get('fast', 12)
        self.slow = kwargs.get('slow', 16)
        self.signal = kwargs.get('signal', 9)
        
        prefix = f"MACD_{self.fast}_{self.slow}"
        
        # 1. 新增：原始绝对值特征组 (Price Unit)
        self.macd_abs_group = [
            f'{prefix}_DIF', 
            f'{prefix}_DEA', 
            f'{prefix}_HIST'
        ]
        
        # 2. 现有的百分比特征组 (Percentage/Momentum)
        self.macd_pct_group = [
            f'{prefix}_DIF_PCT', 
            f'{prefix}_HIST_PCT', 
            f'{prefix}_HIST_ACCEL'
        ]
        
        # 3. 独立特征 (Ratio)
        self.sig_dist = [f'{prefix}_SIG_DIST']
        
        # 汇总所有特征
        self.features = self.macd_abs_group + self.macd_pct_group + self.sig_dist

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close = df['close'].astype(float)
        prefix = f"MACD_{self.fast}_{self.slow}"
        res = {}
        
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal, adjust=False).mean()
        hist = dif - dea
        
        res[f'{prefix}_DIF'] = dif
        res[f'{prefix}_DEA'] = dea
        res[f'{prefix}_HIST'] = hist
        
        dif_pct = np.where(ema_slow != 0, (ema_fast - ema_slow) / ema_slow, np.nan)
        dif_pct_s = pd.Series(dif_pct, index=df.index)
        dea_pct = dif_pct_s.ewm(span=self.signal, adjust=False).mean()
        hist_pct = (dif_pct_s - dea_pct)
        
        res[f'{prefix}_DIF_PCT'] = dif_pct_s
        res[f'{prefix}_HIST_PCT'] = hist_pct
        res[f'{prefix}_HIST_ACCEL'] = hist_pct.diff().fillna(0)
        res[f'{prefix}_SIG_DIST'] = np.where(dea_pct != 0, (dif_pct_s - dea_pct) / np.abs(dea_pct), np.nan)

        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        分层归一化策略
        """
        # 1. 原始绝对值组：属于价格量纲。
        # 使用 _normalize_z_score_rel 挂钩价格基准（如 'close' 或 'high'），消除币种价格差异。
        self._normalize_z_score_rel(
            X, feature_cols, self.macd_abs_group, 
            feature_base="close", factory=factory, method='log'
        )
        
        # 2. MACD 动力百分比组 (DIF_PCT, HIST_PCT, ACCEL)
        # 使用零轴锚定组缩放
        self._normalize_signal_group(
            X, feature_cols, self.macd_pct_group, 
            factory=factory, scale=None, method='log'
        )
        
        # 3. 处理 SIG_DIST (单独缩放，比率的比率)
        self._normalize_signal_group(
            X, feature_cols, self.sig_dist, 
            factory=factory, scale=None, method='log'
        )

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        return int((self.slow + self.signal) * 4)

class FeatureMAStructure(FeatureBase):
    """
    多时间尺度 MA 结构特征（仅保留相对关系）

    输出特征（默认）：
    - BAR_S_L   = log(MA_bar_short / MA_bar_long)
    - BAR_M_L   = log(MA_bar_mid   / MA_bar_long)
    - DAY_S_L   = log(MA_day_short / MA_day_long)
    - WEEK_M_L  = log(MA_week_mid  / MA_week_long)

    可选：
    - Δ_BAR_S_L
    - Δ_DAY_S_L
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ===== 配置 =====
        self.bar_windows  = kwargs.get("bar_windows",  (7, 21, 63))   # short, mid, long
        self.day_windows  = kwargs.get("day_windows",  (5, 20))       # short, long
        self.week_windows = kwargs.get("week_windows", (7, 25))       # mid, long
        self.slope_window = kwargs.get("slope_window", 5)

        self.add_delta = kwargs.get("add_delta", False)
        self.method = kwargs.get("method", "sma").lower()  # 'sma' or 'ema'
        self.strict = kwargs.get("strict", True)

        # ===== 注册特征 =====
        self.features = [
            "MA_BAR_S_L",
            "MA_BAR_M_L",
            "MA_DAY_S_L",
            "MA_WEEK_M_L",
            "MA_WEEK_L_SLOPE",
        ]

        if self.add_delta:
            self.features += [
                "D_MA_BAR_S_L",
                "D_MA_DAY_S_L",
            ]

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _ma(self, series: pd.Series, window: int) -> pd.Series:
        if self.method == "ema":
            ma = series.ewm(span=window, adjust=False).mean()
            if self.strict:
                ma = ma.where(series.expanding().count() >= window, np.nan)
            return ma
        else:
            return series.rolling(
                window=window,
                min_periods=window if self.strict else 1
            ).mean()

    def _calc_klines_per_day_week(self, kline_interval_ms: int) -> tuple[int, int]:
        one_day_ms = 24 * 60 * 60 * 1000
        one_week_ms = 7 * one_day_ms
        kpd = max(int(round(one_day_ms / kline_interval_ms)), 1)
        kpw = max(int(round(one_week_ms / kline_interval_ms)), 1)
        return kpd, kpw

    # ------------------------------------------------------------------
    # 生成特征
    # ------------------------------------------------------------------
    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close = df["close"].astype(float)
        kpd, kpw = self._calc_klines_per_day_week(kline_interval_ms)
        res = {}

        # === BAR 级 ===
        b_s, b_m, b_l = self.bar_windows
        ma_bs = self._ma(close, b_s); ma_bm = self._ma(close, b_m); ma_bl = self._ma(close, b_l)
        res["MA_BAR_S_L"] = np.log(ma_bs / ma_bl)
        res["MA_BAR_M_L"] = np.log(ma_bm / ma_bl)

        # === DAY / WEEK 级 ===
        d_s, d_l = self.day_windows
        ma_ds = self._ma(close, d_s * kpd); ma_dl = self._ma(close, d_l * kpd)
        res["MA_DAY_S_L"] = np.log(ma_ds / ma_dl)

        w_m, w_l = self.week_windows
        ma_wm = self._ma(close, w_m * kpw); ma_wl = self._ma(close, w_l * kpw)
        res["MA_WEEK_M_L"] = np.log(ma_wm / ma_wl)
        res["MA_WEEK_L_SLOPE"] = np.log(ma_wl / ma_wl.shift(self.slope_window))

        # === Δ ===
        if self.add_delta:
            res["D_MA_BAR_S_L"] = pd.Series(res["MA_BAR_S_L"]).diff().fillna(0.0)
            res["D_MA_DAY_S_L"] = pd.Series(res["MA_DAY_S_L"]).diff().fillna(0.0)

        return pd.DataFrame(res, index=df.index)
    # ------------------------------------------------------------------
    # 归一化
    # ------------------------------------------------------------------
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        所有特征都是：
        - log-ratio
        - 以 0 为中性
        → 使用自缩放 Z-Score + log 压制即可
        """
        for f in self.features:
            if f in feature_cols:
                self._normalize_z_score_rel(
                    X,
                    feature_cols,
                    [f],
                    feature_base=f,
                    factory=factory,
                    method="log",
                )

    # ------------------------------------------------------------------
    # 最小历史需求
    # ------------------------------------------------------------------
    def _min_history_request(self, kline_interval_ms: int) -> int:
            # 因为计算斜率需要多回看 slope_window 周期
            kpd, kpw = self._calc_klines_per_day_week(self.kline_interval_ms)
            max_week = max(self.week_windows) * kpw
            
            base = max(max(self.bar_windows), max(self.day_windows) * kpd, max_week)
            
            # 斜率需要额外的 buffer
            base += self.slope_window

            if self.method == "ema":
                base = int(base * 3.5)
            return int(base + 2)


#Dimensionless
class FeatureRsi(FeatureBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.period:int = kwargs.get('period', 14)
        self.price_col = kwargs.get('price_col', 'close')
        self.strict = kwargs.get('strict', True)# 严格型：窗口未满为 NaN；宽松型：尽早给值
        self.prefix = kwargs.get('prefix', "RSI")
        self.features :list[str]= [f"{self.prefix}_{self.period}"]
    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close = df[self.price_col].astype(float)
        res = {}
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.ewm(alpha=1/float(self.period), adjust=False,
                            min_periods=self.period if self.strict else 1).mean()
        avg_loss = loss.ewm(alpha=1/float(self.period), adjust=False,
                            min_periods=self.period if self.strict else 1).mean()

        rs = avg_gain / (avg_loss + EPS)
        rsi_values = np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))
        rsi_values = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, rsi_values)
        
        valid = close.expanding().count() >= self.period
        res[f"{self.prefix}_{self.period}"] = np.where(valid, rsi_values, np.nan)
        return pd.DataFrame(res, index=df.index)
            
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        RSI 范围 [0, 100]。
        处理：(RSI / 100) - 0.5
        结果范围：[-0.5, 0.5]
        0.0 对应 RSI 50 (中性)
        """
        target_indices, _ = self._get_target_indices(feature_cols, self.features)
        if not target_indices:
            return
        
        # 简单缩放，保留绝对位置信息
        X[:, :, target_indices] = (X[:, :, target_indices] / 100.0) - 0.5
    def _min_history_request(self, kline_interval_ms:int = None) -> int:
        """
        计算 RSI 所需的最小历史 K 线数量。
        RSI 使用 Wilder 平滑（alpha=1/period），本质上也是一种 EMA。
        为了使数值收敛并与标准软件一致，通常建议提供至少 5 到 10 倍周期的历史数据。
        """
        # 基础周期，例如 14
        base_period = self.period
        # 为了保证实盘计算精度，建议使用 6 倍周期作为预热
        # 14 * 6 = 84 根 K 线
        # 这样可以确保 EMA 的初始权重衰减到足够小，数值达到稳定状态
        warmup_factor = 6
        return int(base_period * warmup_factor)
"""
经典 KDJ：
    RSV = (C - LLV(n)) / (HHV(n) - LLV(n)) * 100
    K = EMA(RSV, alpha=1/m1)
    D = EMA(K,   alpha=1/m2)
    J = 3*K - 2*D
输出列：{prefix}_K, {prefix}_D, {prefix}_J
"""
#Dimensionless
class FeatureKdj(FeatureBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.features :list[str]= []
        self.n:int = kwargs.get('n', 9)
        self.m1 = kwargs.get('m1', 3)
        self.m2 = kwargs.get('m2', 3)
        self.high_col = kwargs.get('high_col', "high")
        self.low_col = kwargs.get('low_col', "low")
        self.close_col = kwargs.get('close_col', "close")
        self.strict = kwargs.get('strict', True)# 严格型：窗口未满为 NaN；宽松型：尽早给值
        self.prefix = kwargs.get('prefix', "KDJ")
        k_col, d_col, j_col = f"{self.prefix}_K", f"{self.prefix}_D", f"{self.prefix}_J"
        self.features = [k_col, d_col, j_col]
    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        high, low, close = df[self.high_col].astype(float), df[self.low_col].astype(float), df[self.close_col].astype(float)
        res = {}
        llv = low.rolling(window=self.n, min_periods=self.n if self.strict else 1).min()
        hhv = high.rolling(window=self.n, min_periods=self.n if self.strict else 1).max()
        diff = hhv - llv
        rsv = np.where(diff == 0, 50.0, (close - llv) / (diff + EPS) * 100.0)
        
        rsv_s = pd.Series(rsv, index=df.index)
        K = rsv_s.ewm(alpha=1/float(self.m1), adjust=False, min_periods=self.m1 if self.strict else 1).mean()
        D = K.ewm(alpha=1/float(self.m2), adjust=False, min_periods=self.m2 if self.strict else 1).mean()
        J = 3 * K - 2 * D

        valid = close.expanding().count() >= self.n
        res[f"{self.prefix}_K"] = K.where(valid, np.nan)
        res[f"{self.prefix}_D"] = D.where(valid, np.nan)
        res[f"{self.prefix}_J"] = J.where(valid, np.nan)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        #KDJ 是 0-100 指标，使用简单缩放
        target_indices, _ = self._get_target_indices(feature_cols, self.features)
        if not target_indices:
            return

        # 映射到 [-0.5, 0.5]
        X[:, :, target_indices] = (X[:, :, target_indices] / 100.0) - 0.5
    def _min_history_request(self, kline_interval_ms:int = None) -> int:
        """
        计算 KDJ 所需的最小历史 K 线数量。
        1. 基础窗口 n: 计算 RSV 需要 n 根 K 线。
        2. 平滑窗口 m1, m2: K 和 D 使用 alpha=1/m 的递推平滑。
        为了确保 EWM 收敛精度，平滑部分建议提供 4 倍以上的缓冲区。
        """
        # 1. 基础 RSV 窗口
        base_n = self.n
        
        # 2. 平滑收敛需求 (K 和 D 是嵌套平滑)
        # 参照 EMA 的收敛逻辑，通常取 (m1 + m2) * 4 作为安全冗余
        warmup_buffer = int((self.m1 + self.m2) * 4)
        
        # 总需求 = 基础窗口 + 预热缓冲区
        # 例如默认参数 (9, 3, 3) -> 9 + (3+3)*4 = 33 根
        return base_n + warmup_buffer

class FeatureContainer:
    def __init__(self,feature:type[FeatureBase],  **kwargs):
        self.feature = feature
        self.parameters = kwargs

class FeatureATRRegime(FeatureBase):
    """
    多周期波动率环境分析器
    用途：
    1. atr_{w}: 输入模型或下单参考的百分比波动率 (NATR)
    2. vol_regime: 当前短期波动相对于长期的分布位置 (用于过滤非趋势市场)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #  现在支持通过 windows 参数配置多个周期，例如 [14, 100, 1000]
        self.windows = kwargs.get('windows', [14, 100, 1000])
        self.short_w = self.windows[0]  # 以第一个窗口作为“短期”基准
        self.long_w = self.windows[-1]  # 以最后一个窗口作为“长期”背景
        
        self.features = []
        # 动态注册所有 ATR 特征
        for w in self.windows:
            self.features.append(f'atr_{w}')
            
        # 保留环境参考特征
        self.features.append(f'vol_regime_{self.long_w}')
        self.features.append('Vol_Trend')

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close, h, l = df['close'].astype(float), df['high'], df['low']
        pc = close.shift(1)
        res = {}
        tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
        
        atr_series_map = {}
        for w in self.windows:
            atr_w = tr.rolling(w).mean()
            atr_series_map[w] = atr_w
            res[f'atr_{w}'] = np.where(close > 0, atr_w / close, 0.0)
        
        short_atr = atr_series_map[self.short_w]
        long_atr_ref = short_atr.rolling(self.long_w).mean()
        res[f'vol_regime_{self.long_w}'] = np.where(long_atr_ref > 0, short_atr / long_atr_ref, 1.0)
        res['Vol_Trend'] = short_atr.diff(5) / (short_atr.shift(5) + EPS)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # 1. 批量处理所有 ATR 特征
        # 采用独立 Z-Score + Tanh 压制，确保不同尺度的波动率在模型输入层量纲一致
        for w in self.windows:
            col = f'atr_{w}'
            if col in feature_cols:
                self._normalize_z_score(X, feature_cols, [col], feature_base= col, factory = factory, method='tanh')

        # 2. 处理环境参考特征 (Regime)
        regime_col = f'vol_regime_{self.long_w}'
        if regime_col in feature_cols:
            self._normalize_z_score(X, feature_cols, [regime_col], feature_base = regime_col, factory =factory, method='log')

        # 3. 处理 Vol_Trend (变化率)
        if 'Vol_Trend' in feature_cols:
            idx = feature_cols.index('Vol_Trend')
            X[:, :, idx] = self._apply_squashing(X[:, :, idx], scale=1.0, method='tanh')

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # 必须满足最大窗口的需求
        return int(max(self.windows) * 1.2)
    
class FeatureVolMa(FeatureBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.vol_ma_windows:list = kwargs.get('vol_ma_windows', (5, 10, 20))
        self.ma_features = []
        self.ma_features_ratio = []
        for w in self.vol_ma_windows:
            self.ma_features.append(f'VOL_MA_{w}')
            self.ma_features_ratio.append(f'VOL_ratio_{w}')
            self.features.extend([f'VOL_MA_{w}', f'VOL_ratio_{w}'])
    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        res = {}
        vol = df['volume']
        for w in self.vol_ma_windows:
            vol_ma = vol.rolling(w).mean()
            res[f'VOL_MA_{w}'] = vol_ma
            res[f'VOL_ratio_{w}'] = np.where(vol_ma > EPS, vol / vol_ma, 0.0)
        return pd.DataFrame(res, index=df.index)
            
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize_volume_rlc(X, feature_cols, self.ma_features, "volume")
        self._normalize_z_score_group(X, feature_cols , self.ma_features , factory= factory, method = 'log', scale= 3)
        self._normalize_z_score_group(X, feature_cols , self.ma_features_ratio , factory= factory, method = 'log')
    def _min_history_request(self, kline_interval_ms:int = None) -> int:
        """
        计算成交量均线所需的最小历史 K 线数量。
        逻辑：取所有配置窗口中的最大值，并增加少量缓冲区确保标准化计算稳定。
        """
        if not self.vol_ma_windows:
            return 0
        # 1. 获取最大窗口期 (例如 20)
        max_window = max(self.vol_ma_windows)
        # 2. 增加缓冲区
        # 对于 SMA 来说，max_window 根就能出数，
        # 但为了 normalize 逻辑在时间轴上有统计意义，建议提供 1.5 倍到 2 倍的数据
        return int(max_window * 1.5)

class FeatureQavMa(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 1. 获取配置项
        self.windows = kwargs.get('windows', [7, 25, 99])    # 支持多个窗口长度
        self.add_surge = kwargs.get('add_surge', True)      # 是否生成爆发比率 (Intensity)
        self.add_slope = kwargs.get('add_slope', True)      # 是否生成趋势斜率 (Consistency)
        self.add_bias = kwargs.get('add_bias', True)        # 是否生成 VWAP 偏离 (Cost)
        self.slope_step = kwargs.get('slope_step', 3)       # 斜率计算跨度
        
        self.surge_features = []
        self.slope_features = []
        self.bias_features = []

        # 2. 预注册特征名
        for w in self.windows:
            if self.add_surge:
                col = f'QAV_SURGE_{w}'
                self.features.append(col)
                self.surge_features.append(col)
            
            if self.add_slope:
                col = f'QAV_SLOPE_{w}'
                self.features.append(col)
                self.slope_features.append(col)
        
        if self.add_bias:
            # VWAP Bias 通常只需要一个（代表当前与成交均价的偏离）
            col = 'VWAP_BIAS'
            self.features.append(col)
            self.bias_features.append(col)

    def _slope_reg_vectorized(self, series: pd.Series, steps: int) -> pd.Series:
        """
        [无 Log 稳健版] 最小二乘法线性回归斜率
        使用全窗口数据进行拟合，彻底消除端点噪声。
        """
        if steps <= 1: return pd.Series(np.nan, index=series.index)
        
        n = float(steps)
        # x 轴坐标是 [0, 1, 2, ..., n-1]
        x_mean = (n - 1) / 2.0
        var_x = (n * (n**2 - 1)) / 12.0 # x 的方差
        
        # 向量化计算 y 的统计量
        y_filled = series.fillna(0)
        s1 = y_filled.cumsum()
        s2 = s1.cumsum()
        
        sum_y = s1 - s1.shift(steps)
        shift_s1 = s1.shift(steps)
        shift_s2 = s2.shift(steps)
        
        # 利用累加和性质计算 sum(x*y)
        weighted_sum_rev = (s2 - shift_s2) - steps * shift_s1
        sum_xy = (steps * sum_y) - weighted_sum_rev
        
        # 斜率公式: Beta = Cov(x,y) / Var(x)
        y_mean = sum_y / n
        slope = (sum_xy - n * x_mean * y_mean) / var_x
        
        #  关键：因为没有用 log，为了保持不同价格/成交量下的可比性
        # 我们将斜率“标准化”：即 这里的 slope 代表的是 相对于均值的变动比例
        # 这样 100 变成 110 和 10 变成 11 的斜率在量纲上才一致
        return slope / (y_mean + EPS)

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        qav = df['quote_asset_volume'].astype(float)
        vol = df['volume'].astype(float)
        close = df['close'].astype(float)
        res = {}

        for w in self.windows:
            ma_qav = qav.rolling(w).mean()
            if self.add_surge:
                res[f'QAV_SURGE_{w}'] = np.where(ma_qav > EPS, qav / ma_qav, 1.0)
            if self.add_slope:
                res[f'QAV_SLOPE_{w}'] = self._slope_reg_vectorized(ma_qav, self.slope_step)

        if self.add_bias:
            vwap = np.where(vol > EPS, qav / vol, 0.0)
            res['VWAP_BIAS'] = np.where(vwap > EPS, (close / vwap) - 1.0, 0.0)

        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # 1. 爆发比率组：波动极大，长尾重，必须开启 suppress=True
        if self.surge_features:
            for f in self.surge_features:
                self._normalize_z_score_rel(X, feature_cols, [f], factory=factory,feature_base=f, method= 'log')

        # 2. 趋势斜率组：log1p 后的斜率波动稳定，常规归一化即可
        if self.slope_features:
            # 斜率可以放在一起组归一化，因为它们量纲相同（百分比变化）
            self._normalize_z_score_rel(X, feature_cols, self.slope_features, 
                                          feature_base=self.slope_features[0], factory=factory, method= 'log')

        # 3. VWAP 偏离组：百分比变化量，单独归一化.范围 [-1,1]
        if self.bias_features:  
            self._normalize_z_score_rel(X, feature_cols, self.bias_features, 
                                          feature_base='VWAP_BIAS', factory=factory, method= 'log')
            
        # 4. 别忘了最后的物理保护
        # X[:, :, indices] = np.clip(X[:, :, indices], -5.0, 5.0) # 在 FeatureFactory 层做或者在这里做都行

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # 需要最大窗口 + 斜率步长作为预热
        max_w = max(self.windows) if self.windows else 0
        return int((max_w + self.slope_step) * 1.5)
# ==== 4. OBV ====
class FeatureOBV(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = ['OBV']

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        close = df['close']
        vol = df['volume'].astype(float)
        
        # 1. 基础方向逻辑
        sign = np.where(close > close.shift(1), 1,
                        np.where(close < close.shift(1), -1, 0))
        obv_raw = (sign * vol).cumsum()
        
        # 2. 这里的无量纲化应该在 normalize 阶段配合 Factory 完成
        # 或者在 generate 阶段先存下原始值，我们这里统一在 normalize 处理更严谨
        df['OBV'] = obv_raw

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize_z_score_rel(X, feature_cols , self.features , feature_base = self.features[0], factory= factory, method = 'log')  #Self-Normalization
    def _min_history_request(self, kline_interval_ms:int = None) -> int:
        """
        计算 OBV 所需的最小历史 K 线数量。
        OBV 是一个累积指标，为了让标准化 (Normalization) 逻辑获得稳定的
        均值和标准差，建议提供至少 2 倍于模型窗口 (window) 的历史数据。
        """
        model_window = getattr(self, 'window', 100) 
        
        # 为了让累积值的波动率在归一化时趋于平稳
        return int(model_window * 2)
# ==== PVT ====
class FeaturePVT(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = ['PVT']

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        # 1. 计算价格变动率 (pct_change)
        # 注意：df['close'] 必须先转为 float，避免类型问题
        close = df['close'].astype(float)
        volume = df['quote_asset_volume'].astype(float) # 建议用成交额 QAV 代替 VOL
        
        pct = close.pct_change().fillna(0)
        
        # 2. 计算当前增量：变动率 * 成交量
        incremental_pvt = pct * volume
        
        # 3. 累加得到标准 PVT
        # pvt_raw 是一个带有“记忆”的长序列
        pvt_raw = incremental_pvt.cumsum()
        
        # 4. 存储到 DataFrame
        df['PVT'] = pvt_raw

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # 4. 最后执行对称组缩放，将分布锁定在 [-0.5, 0.5]
        # 此时由于经过了 log1p，数据的有效信息会在这个区间内分布得非常均匀
        self._normalize_z_score_rel(X, feature_cols , self.features , feature_base = 'quote_asset_volume', factory= factory)  #Self-Normalization
    def _min_history_request(self, kline_interval_ms:int = None) -> int:
        """
        计算 PVT 所需的最小历史 K 线数量。
        PVT 是累积指标，需要足够的样本量来让归一化 (Normalization) 过程中的
        均值和标准差趋于稳定。
        """
        # 建议参考模型训练时的 window 大小 (通常在 100 左右)
        # 如果类中没有保存 window，则给一个安全的保守估计值
        model_window = getattr(self, 'window', 100)
        # 2倍窗口可以确保在第一个推理窗口 [T-window, T] 之前，
        # 已经有一段数据用于形成指标的初始统计分布
        return int(model_window * 2)
    
# ==== VWAP（滚动） ====
class FeatureWAP(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 获取窗口配置，默认使用你喜欢的 7, 25, 99 风格（或者是你指定的 20, 48, 96）
        self.vwap_windows: list = kwargs.get('vwap_windows', (20, 48, 96))
        self.add_bias: bool = kwargs.get('add_bias', True)  # 是否生成偏离度特征
        
        self.absolute_features = []  # 原始价格量纲特征
        self.ratio_features = []     # 偏离度百分比特征

        for w in self.vwap_windows:
            # 1. 注册原始 VWAP
            vwap_col = f'VWAP_{w}'
            self.features.append(vwap_col)
            self.absolute_features.append(vwap_col)
            
            # 2. 注册 VWAP Bias
            if self.add_bias:
                bias_col = f'VWAP_Bias_{w}'
                self.features.append(bias_col)
                self.ratio_features.append(bias_col)

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        qav, vol, close = df['quote_asset_volume'].astype(float), df['volume'].astype(float), df['close'].astype(float)
        res = {}
        for w in self.vwap_windows:
            rolling_qav = qav.rolling(w).sum()
            rolling_vol = vol.rolling(w).sum()
            vwap_series = np.where(rolling_vol > EPS, rolling_qav / rolling_vol, close)
            res[f'VWAP_{w}'] = vwap_series
            if self.add_bias:
                res[f'VWAP_Bias_{w}'] = np.where(vwap_series > EPS, (close / vwap_series) - 1.0, 0.0)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        按照物理意义对齐进行归一化
        """
        # 1. 原始 VWAP：属于价格量纲，必须和 close 挂钩进行组归一化
        if self.absolute_features:
            self._normalize_z_score_rel(X, feature_cols, self.absolute_features, feature_base = "close", factory=factory, method='log')
        
        # 2. VWAP Bias：属于百分比量纲，波动较小且平稳，单独归一化
        if self.ratio_features:
            self._normalize_z_score_rel(X, feature_cols, self.ratio_features, feature_base = self.ratio_features[-1], factory=factory, method='log')
            # for f in self.ratio_features:
            #     # 偏离度通常在 [-0.1, 0.1] 之间，使用自缩放即可
            #     self._normalize_z_score_rel(X, feature_cols, [f], f, factory=factory)
    def _min_history_request(self, kline_interval_ms:int = None) -> int:
            """
            计算滚动 VWAP 所需的最小历史 K 线数量。
            逻辑：取配置窗口中的最大值，并增加缓冲区以确保标准化逻辑在实盘初期具有统计稳定性。
            """
            if not self.vwap_windows:
                return 0
                
            # 1. 获取最大窗口期 (例如 96)
            max_window = max(self.vwap_windows)
            
            # 2. 增加缓冲区 (建议 1.5 倍)
            # 确保 rolling sum 能够完整计算，且 normalize 过程有足够的历史样本作为背景
            return int(max_window * 1.5)
# ==== CMF ====
#如果价格在收盘时处于当根 K 线的高位，并且成交量放大，那就说明机构在“吸筹”；反之则是“派发”
class FeatureCFM(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 支持传入多个窗口，例如 (10, 20, 60)
        self.cmf_windows: list = kwargs.get('cmf_windows', [10, 20, 60])
        
        self.ratio_features = []
        for w in self.cmf_windows:
            col = f'CMF_{w}'
            self.features.append(col)
            self.ratio_features.append(col)

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float) # 如果你想更进阶，这里可以换成 quote_asset_volume
        
        range_hl = high - low
        
        # 1. 计算乘数 (Money Flow Multiplier)
        # 逻辑：((收盘-最低) - (最高-收盘)) / (最高-最低)
        # 本质是看收盘价在 K 线振幅中的相对站位
        mfm = np.where(range_hl > EPS, 
                       ((close - low) - (high - close)) / range_hl, 
                       0.0)
        
        # 2. 计算资金流成交量 (MFV)
        mfv = mfm * volume
        
        # 3. 循环计算不同窗口的 CMF
        for w in self.cmf_windows:
            mfv_sum = mfv.rolling(w).sum()
            vol_sum = volume.rolling(w).sum()
            
            # CMF = 窗口内 MFV 总和 / 窗口内总成交量. 范围 [-1, 1]
            df[f'CMF_{w}'] = np.where(vol_sum > EPS, mfv_sum / vol_sum, 0.0)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        🚀 CMF 的归一化建议：
        CMF 已经是无量纲的比率（-1.0 到 1.0），且均值通常接近 0。
        建议使用自缩放归一化，捕捉其超买超卖的震荡信号。
        """
        # self._normalize_z_score_rel(X, feature_cols, self.ratio_features, self.ratio_features[0], factory=factory)  #根据测试单独归一化更好
        if self.ratio_features:
            for f in self.ratio_features:
                # 这种震荡指标不需要和 close 挂钩，单独进行 Z-Score 即可
                self._normalize_z_score_rel(X, feature_cols, [f], feature_base = f, factory=factory, method='log')

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # 取最大的窗口并增加缓冲区
        max_w = max(self.cmf_windows) if self.cmf_windows else 20
        return int(max_w * 1.5)

# ==== MFI 衡量资金在流入还是流出，以及流出的力度有多大====
class FeatureMFI(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 延续黄金周期思路，支持多窗口配置
        self.mfi_windows: list = kwargs.get('mfi_windows', [14, 25, 99])
        
        self.ratio_features = []
        for w in self.mfi_windows:
            col = f'MFI_{w}'
            self.features.append(col)
            self.ratio_features.append(col)

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        high, low, close = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
        volume = df['quote_asset_volume'].astype(float)
        res = {}

        tp = (high + low + close) / 3.0
        mf = tp * volume
        tp_diff = tp.diff()
        
        pos_mf = pd.Series(np.where(tp_diff > 0, mf, 0.0), index=df.index)
        neg_mf = pd.Series(np.where(tp_diff < 0, mf, 0.0), index=df.index)

        for w in self.mfi_windows:
            p_sum = pos_mf.rolling(w).sum()
            n_sum = neg_mf.rolling(w).sum()
            total_mf = p_sum + n_sum
            res[f'MFI_{w}'] = np.divide(100.0 * p_sum, total_mf, out=np.full_like(p_sum, 50.0), where=total_mf > 0)

        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        🚀 零均值对称归一化：将 [0, 100] 映射到 [-0.5, 0.5]
        这能让模型更敏锐地识别“流入占优”还是“流出占优”
        """
        for f in self.ratio_features:
            if f in feature_cols:
                idx = feature_cols.index(f)
                # 线性映射：(val / 100) - 0.5
                X[:, :, idx] = (X[:, :, idx] / 100.0) - 0.5

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        max_w = max(self.mfi_windows) if self.mfi_windows else 14
        return int(max_w * 2.1)
    
class FeatureATS(FeatureBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.features = ['ATS']
    def generate(self,df:pd.DataFrame, kline_interval_ms:int):
        # ==== 2. 平均每笔成交量 ====
        df['ATS'] = np.where(df['number_of_trades'] > EPS, df['volume'] / df['number_of_trades'], 0.0 )
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        ATS 代表单笔成交力度。
        """
        self._normalize_z_score_rel(X=X, feature_cols=feature_cols, target_feature_cols=self.features, feature_base='ATS', factory= factory)  # <--- 必须自缩放
    def _min_history_request(self, kline_interval_ms:int = None) -> int:
        """
        计算平均单笔成交量 (ATS) 所需的最小历史 K 线数量。
        虽然指标本身是逐 K 线计算的，但为了使 Z-Score 标准化逻辑稳定，
        建议提供至少与模型观察窗口等长的历史数据。
        """
        # 建议参考模型训练时的 window 大小 (通常在 100 左右)
        model_window = getattr(self, 'window', 100)
        
        # 提供 1 倍窗口作为标准化计算的基础样本
        return int(model_window)

class FeatureAdvancedVol(FeatureBase):
    """
    高级波动率建模：提供比 ATR 更高效的波动率估计量
    包含：Parkinson, Garman-Klass
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.windows = kwargs.get('windows', [14, 100])
        self.features = []
        for w in self.windows:
            self.features.append(f'vol_parkinson_{w}')
            self.features.append(f'vol_gk_{w}')

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        o, h, l, c = df['open'].astype(float), df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
        res = {}
        log_hl = np.log(h / (l + EPS))**2
        log_co = np.log(c / (o + EPS))**2
        for w in self.windows:
            const_p = 1.0 / (4.0 * np.log(2.0))
            res[f'vol_parkinson_{w}'] = np.sqrt(const_p * log_hl.rolling(w).mean())
            const_gk = 2.0 * np.log(2.0) - 1.0
            gk_term = 0.5 * log_hl - const_gk * log_co
            res[f'vol_gk_{w}'] = np.sqrt(gk_term.rolling(w).mean())
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # 波动率特征通常具有明显的长尾分布
        # 建议使用 z-score + log 压制，这样模型可以更好地识别“波动率突增”信号
        for f in self.features:
            if f in feature_cols:
                self._normalize_z_score(
                    X, feature_cols, [f], 
                    feature_base=f, 
                    factory=factory, 
                    method='log'
                )

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        return int(max(self.windows) * 1.5)

class FeatureFractalPersistence(FeatureBase):
    """
    分形与持久性指标：识别趋势的纯度与记忆性
    包含：Hurst Exponent (方差尺度法), Efficiency Ratio (ER/Kaufman)

    - ER: 净位移/路径总长 ∈ [0,1]，1 为纯趋势，0 为震荡
    - Hurst: 方差尺度法滚动估计，H>0.5 趋势延续，H<0.5 均值回归，H=0.5 随机游走
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 建议窗口：14 (短线灵敏), 126 (波段基调)
        self.windows = kwargs.get('windows', [14, 126])
        self.features = []
        for w in self.windows:
            self.features.append(f'hurst_{w}')
            self.features.append(f'er_{w}')

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close = df['close'].astype(float)
        diff = close.diff().abs()
        res = {}
        for w in self.windows:
            # ER
            net_change = (close - close.shift(w)).abs()
            total_path = diff.rolling(w).sum()
            res[f'er_{w}'] = np.where(total_path > EPS, net_change / total_path, 0.0)
            # Hurst
            log_close = np.log(close + EPS)
            std_1 = log_close.diff().rolling(w).std()
            std_w = log_close.diff(w).rolling(w).std()
            ratio = np.where(std_1 > EPS, std_w / (std_1 + EPS), 1.0)
            res[f'hurst_{w}'] = np.clip(np.log(np.maximum(ratio, EPS)) / np.log(max(w, 2)), 0.0, 1.0)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # ER 已经在 [0, 1] 空间，将其平移至 [-0.5, 0.5] 即可
        er_indices, _ = self._get_target_indices(feature_cols, [f for f in self.features if 'er_' in f])
        if er_indices:
            X[:, :, er_indices] = X[:, :, er_indices] - 0.5

        # Hurst 指数通常分布在 [0, 1] 之间，中心点在 0.5
        hurst_indices, _ = self._get_target_indices(feature_cols, [f for f in self.features if 'hurst_' in f])
        if hurst_indices:
            # 同样平移至 [-0.5, 0.5]，让 0 代表随机游走
            X[:, :, hurst_indices] = X[:, :, hurst_indices] - 0.5

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        return int(max(self.windows) * 2)

class FeatureOrderFlow(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.windows = kwargs.get('windows', [14, 49])
        self.poc_bias_step = kwargs.get('poc_bias_step', [7, 25, 99, 200, 400, 900])
        
        self.features = []
        for w in self.windows:
            self.features.extend([f'imbalance_{w}', f'vpin_{w}', f'trade_density_{w}'])
        for w in self.poc_bias_step:
            self.features.append(f'poc_bias_{w}')

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        # 1. 基础数据准备
        vol = df['volume'].astype(float)
        taker_vol = df['taker_buy_base_volume'].astype(float)
        trades = df['number_of_trades'].astype(float)
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        # 预计算 Typical Price 和 PV (Price * Volume) 用于 VWAP
        tp = (high + low + close) / 3.0
        pv = tp * vol
        
        # 结果容器
        res = {}

        # 2. 瞬时买卖比例 (向量化)
        raw_imbalance_series = pd.Series(
            np.divide(taker_vol, vol, out=np.full_like(vol, 0.5), where=vol > EPS),
            index=df.index
        )

        # 3. 计算常规窗口特征
        for w in self.windows:
            rolling_imb = raw_imbalance_series.rolling(w)
            res[f'imbalance_{w}'] = rolling_imb.mean() - 0.5
            res[f'vpin_{w}'] = rolling_imb.std()
            
            trades_ma = trades.rolling(w).mean()
            res[f'trade_density_{w}'] = np.divide(trades, trades_ma, out=np.full_like(trades, 1.0), where=trades_ma > EPS)

        # 4. 向量化计算 POC Bias (核心优化点)
        for w in self.poc_bias_step:
            # 使用向量化的滚动求和代替原来的 apply
            rolling_pv_sum = pv.rolling(w).sum()
            rolling_vol_sum = vol.rolling(w).sum()
            
            # 计算滚动 VWAP
            vwap = np.divide(rolling_pv_sum, rolling_vol_sum, out=np.full_like(close, np.nan), where=rolling_vol_sum > EPS)
            
            # 计算对数偏离度
            res[f'poc_bias_{w}'] = np.log(close / (vwap + EPS))

        # 5. 一次性转换为 DataFrame，彻底解决碎片化警告
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # 1. Imbalance: 范围 [-0.5, 0.5], 已经是中心化的，直接 Z-Score
        imbalance_feats = [f for f in self.features if 'imbalance_' in f]
        self._normalize_z_score_group(X, feature_cols, imbalance_feats, factory, method='tanh')

        # 2. Trade Density: 长尾特征，均值为 1.0，使用 Log 压缩
        density_features = [f for f in self.features if 'trade_density_' in f]
        for f in density_features:
            if f in feature_cols:
                idx = feature_cols.index(f)
                X[:, :, idx] = np.log1p(X[:, :, idx])

        # 3. POC Bias: 价格偏离，使用相对价格归一化
        poc_feats = [f for f in self.features if 'poc_bias_' in f]
        self._normalize_signal_group(X, feature_cols, poc_feats, factory, method='log')

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        return int(int(max(max(self.windows), max(self.poc_bias_step)))* 1.5)

class FeatureClassicFactors(FeatureBase):
    """
    经典因子类：统计矩、信息离散度与极值位置
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.windows = kwargs.get('windows', [20, 100])
        self.features = []
        for w in self.windows:
            self.features.extend([
                f'skew_{w}', f'kurt_{w}', 
                f'id_factor_{w}', f'dist_to_high_{w}'
            ])

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        close = df['close'].astype(float)
        returns = close.pct_change()

        for w in self.windows:
            # 1. 统计矩 (Skew & Kurt)
            # 捕获收益率分布的非对称性与肥尾特征
            df[f'skew_{w}'] = returns.rolling(w).skew()
            df[f'kurt_{w}'] = returns.rolling(w).kurt()

            # 2. 信息离散度 (ID Factor)
            # 计算上涨 K 线与下跌 K 线的比例差
            pos_bars = (returns > 0).rolling(w).sum()
            neg_bars = (returns < 0).rolling(w).sum()
            # 归一化占比差值
            df[f'id_factor_{w}'] = (pos_bars - neg_bars) / w

            # 3. 极值位置 (Distance to High)
            # 使用对数距离，保持量纲一致性
            rolling_high = df['high'].rolling(w).max()
            df[f'dist_to_high_{w}'] = np.log(rolling_high / (close + EPS))

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # 1. Skew/Kurt: 通常在 [-3, 3] 左右，使用 Z-Score 即可
        skew_kurt = [f for f in self.features if 'skew' in f or 'kurt' in f]
        self._normalize_z_score_group(X, feature_cols, skew_kurt, factory, method='tanh')

        # 2. ID Factor: 天然在 [-1, 1] 之间，0 为平衡
        id_feats = [f for f in self.features if 'id_factor' in f]
        # 无需平移，直接进行 Z-Score 映射以增强信号强度
        self._normalize_signal_group(X, feature_cols, id_feats, factory)

        # 3. Dist to High: 正数且长尾，使用 Log 压缩
        dist_feats = [f for f in self.features if 'dist_to_high' in f]
        self._normalize_signal_group(X, feature_cols, dist_feats, factory, method='log')

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # 统计矩计算至少需要一定样本量（建议 20 以上）才能稳定
        return int(max(self.windows) * 1.5)

class FeatureMomentum(FeatureBase):
    """
    经典因子型动量（Momentum Factors）

    Features:
      1) MOM_h              = log(close / close.shift(h))
      2) MOM_h_SKIPk        = log(close.shift(k) / close.shift(h+k))   (default k=1)
      3) MOM_h_RV{v}        = MOM_h / (RV_v + EPS), RV_v = rolling std of log returns

    Notes:
      - All are causal (<= t).
      - Designed as factor-type features: directly tied to realized returns.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # --- config ---
        self.horizons: list[int] = kwargs.get("horizons", [5, 10, 20, 60])
        self.include_skip: bool = kwargs.get("include_skip", True)
        self.skip_k: int = kwargs.get("skip_k", 1)
        self.skip_horizon: int | None = kwargs.get("skip_horizon", 20)  # only build skip for this horizon; None => for all

        self.include_vol_adj: bool = kwargs.get("include_vol_adj", True)
        self.vol_window: int = kwargs.get("vol_window", 20)
        self.vol_adj_horizon: int | None = kwargs.get("vol_adj_horizon", 20)  # only build vol-adj for this horizon; None => for all

        # --- feature names ---
        self.mom_cols = [f"MOM_{h}" for h in self.horizons]

        self.skip_cols = []
        if self.include_skip:
            if self.skip_horizon is None:
                self.skip_cols = [f"MOM_{h}_SKIP{self.skip_k}" for h in self.horizons]
            else:
                self.skip_cols = [f"MOM_{self.skip_horizon}_SKIP{self.skip_k}"]

        self.vol_adj_cols = []
        if self.include_vol_adj:
            if self.vol_adj_horizon is None:
                self.vol_adj_cols = [f"MOM_{h}_RV{self.vol_window}" for h in self.horizons]
            else:
                self.vol_adj_cols = [f"MOM_{self.vol_adj_horizon}_RV{self.vol_window}"]

        self.features = self.mom_cols + self.skip_cols + self.vol_adj_cols

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close = df["close"].astype(float)
        log_close = np.log(close + EPS)
        res = {}
        for h in self.horizons:
            res[f"MOM_{h}"] = (log_close - log_close.shift(h))
        if self.include_skip:
            k = int(self.skip_k)
            for h in (self.horizons if self.skip_horizon is None else [self.skip_horizon]):
                res[f"MOM_{h}_SKIP{k}"] = (log_close.shift(k) - log_close.shift(h + k))
        if self.include_vol_adj:
            rv = log_close.diff().rolling(window=self.vol_window).std()
            for h in (self.horizons if self.vol_adj_horizon is None else [self.vol_adj_horizon]):
                res[f"MOM_{h}_RV{self.vol_window}"] = np.where(rv > EPS, res[f"MOM_{h}"] / (rv + EPS), np.nan)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        动量类因子都是 0 为中性、对称分布（log-return / ratio）。
        推荐用零轴锚定组缩放（RMS）+ 对称 log 压制，保留“各周期动量强弱对比”。
        """
        # 1) plain momentum as one group
        self._normalize_signal_group(
            X, feature_cols, self.mom_cols,
            factory=factory, scale=None, method="log"
        )

        # 2) skip momentum (if any) - keep separate group (分布略不同)
        if self.skip_cols:
            self._normalize_signal_group(
                X, feature_cols, self.skip_cols,
                factory=factory, scale=None, method="log"
            )

        # 3) vol-adjusted momentum (if any) - separate group
        if self.vol_adj_cols:
            self._normalize_signal_group(
                X, feature_cols, self.vol_adj_cols,
                factory=factory, scale=None, method="log"
            )

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        """
        最小历史长度：
          - plain MOM: max(horizons)
          - skip MOM: max(h+skip_k)
          - vol adj: needs vol_window + 1 (for diff) and horizon itself
        """
        max_h = max(self.horizons) if self.horizons else 1

        need = max_h

        if self.include_skip:
            k = int(self.skip_k)
            if self.skip_horizon is None:
                need = max(need, max_h + k)
            else:
                need = max(need, int(self.skip_horizon) + k)

        if self.include_vol_adj:
            # rv needs vol_window bars of log_ret, log_ret needs 1 extra
            need = max(need, int(self.vol_window) + 1)
            if self.vol_adj_horizon is None:
                need = max(need, max_h)
            else:
                need = max(need, int(self.vol_adj_horizon))

        # buffer
        return int(need + 2)


@njit(float64[:](float64[:], int64, int64), cache=True)
def calc_rolling_top_k_reference(data, window, k):
    n = len(data)
    out = np.full(n, np.nan)
    
    # 手动指定 k 的类型为整数，防止 Numba 推导错误
    k_int = int(k)
    
    for i in range(window, n):
        # 1. 明确切片，获取窗口数据
        window_data = data[i-window : i]
        
        # 2. 使用 partition 获取极值
        # 这里我们手动把结果赋值给一个变量，帮助 Numba 锁定类型
        top_k_elements = np.partition(window_data, -k_int)[-k_int:]
        
        # 3. 手动累加求平均，避开 top_k_elements.mean() 的潜在错误
        # 这是 Numba 最稳健的写法
        accumulator = 0.0
        for val in top_k_elements:
            accumulator += val
        
        out[i] = accumulator / k_int
        
    return out

class FeatureVolumeEvent(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.windows = kwargs.get('windows', [5000, 1500, 500])
        self.top_k = kwargs.get('top_k', 3)
        # 动态注册所有特征
        self.features = []
        for w in self.windows:
            self.features.append(f'vol_event_flag_{w}')

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        # 确保传入的是 float64 类型的 Numpy 数组
        vol_values = df['volume'].values.astype(np.float64)
        
        for w in self.windows:
            flag_col = f'vol_event_flag_{w}'
            
            # 调用手动循环优化的 JIT 函数
            v_ref_raw_values = calc_rolling_top_k_reference(vol_values, w, self.top_k)
            
            # 转换为 Series 进行后续的 EWM 处理
            v_ref_raw = pd.Series(v_ref_raw_values, index=df.index)
            # 适当调大 alpha 让基准回落稍快，保持灵敏
            v_ref = v_ref_raw.ewm(alpha=1 / (w * 2), adjust=False).mean()
            
            # 这里的计算是在 Pandas/Numpy 层面，非常稳健
            ratio_col = np.where(v_ref > EPS, df['volume'] / v_ref, np.nan)
            df[flag_col] = (ratio_col >= 1.0).astype(int)
            
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # Event-level feature.
        # Do NOT normalize / z-score / center.
        pass

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # 必须返回所有窗口中最大的那个，确保初始化数据足够
        return int(max(self.windows) * 1.2)
    
class FeatureCandle(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 1. 绝对量级类：受价格影响，长尾极其严重
        self.feat_magnitude = ['body', 'upper_wick', 'lower_wick', 'max_range', 'body_mom']
        # 2. 比例类：已天然缩放在 [0, 1]
        self.feat_ratio = ['body_pct', 'upper_wick_pct', 'lower_wick_pct', 'close_pos', 'doji_score']
        # 3. 评分类：描述方向，通常在 [-1, 1]
        self.feat_score = ['wick_bias'] 
        
        self.features = self.feat_magnitude + self.feat_ratio + self.feat_score

    def generate(self, df: pd.DataFrame, kline_interval_ms: int = None) -> pd.DataFrame:
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        res = {}

        # 绝对量级
        res['body'] = np.abs(c - o)
        res['upper_wick'] = h - np.maximum(o, c)
        res['lower_wick'] = np.minimum(o, c) - l
        res['max_range'] = h - l
        res['body_mom'] = pd.Series(res['body']).diff().fillna(0)

        # 比例类
        rng = res['max_range']
        res['body_pct'] = np.where(rng > 0, res['body'] / rng, 0.0)
        res['upper_wick_pct'] = np.where(rng > 0, res['upper_wick'] / rng, 0.0)
        res['lower_wick_pct'] = np.where(rng > 0, res['lower_wick'] / rng, 0.0)
        res['close_pos'] = np.where(rng > 0, (c - l) / rng, 0.5)
        res['doji_score'] = 1.0 - res['body_pct']
        res['wick_bias'] = res['upper_wick_pct'] - res['lower_wick_pct']

        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        分层压缩策略：
        - Magnitude: 价格归一化 -> 组 Z-Score -> Log1p 压缩
        - Ratio: 中心化到 [-0.5, 0.5] -> 可选 Log 增强
        """
        self._normalize_z_score_group(X,feature_cols,self.feat_magnitude, factory, method = 'log')

        # 2. 处理 Ratio 类别
        # self._normalize_signal_group(X,feature_cols,self.feat_ratio, factory, method = 'log')
        # 2. 处理 Ratio 类别: 移动到 [-0.5, 0.5]
        # 获取 Ratio 特征在 X 中的列索引
        ratio_indices = [feature_cols.index(f) for f in self.feat_ratio if f in feature_cols]
        
        if ratio_indices:
            # 执行平移： [0, 1] -> [-0.5, 0.5]
            X[:, :, ratio_indices] = X[:, :, ratio_indices] - 0.5
        # 3. 处理 Score 类别 (wick_bias 等)

        self._normalize_z_score(X,feature_cols,self.feat_score, self.feat_score[0] , factory)
    def _min_history_request(self, kline_interval_ms:int = None) -> int:
        """
        计算 K线形态特征所需的最小历史 K 线数量。
        虽然大部分形态是单根计算，但 gap 和 body_mom 依赖前一根数据，
        且 Z-Score 标准化需要足够的样本背景。
        """
        # 1. 基础依赖：diff/shift 至少需要 2 根
        base_dependency = 2
        
        # 2. 标准化稳定性需求：建议参考模型观察窗口 (Window)
        # 如果类中没有保存 window，则给一个安全的保守值
        model_window = getattr(self, 'window', 100)
        
        # 返回模型窗口大小，确保 Z-Score 计算有足够的样本数
        return max(base_dependency, int(model_window))

class FeatureDonchian(FeatureBase):
    """
    多周期唐奇安通道 (Multi-Period Donchian Channels):
    支持传入一个周期列表，同时生成价格骨架与形态特征。
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 支持列表输入，默认为 [20]
        self.periods = kwargs.get('periods', [20])
        
        self.price_features = []
        self.ratio_features = []
        self.pos_features = []  # 专门记录 POS 特征以便中心化处理

        # 动态构建特征列表
        for p in self.periods:
            # 1. 价格类
            self.price_features.extend([
                f"DONCHIAN_UPPER_{p}",
                f"DONCHIAN_LOWER_{p}",
                f"DONCHIAN_MIDDLE_{p}"
            ])
            # 2. 比例/形态类
            p_pos = f"DONCHIAN_POS_{p}"
            self.pos_features.append(p_pos)
            self.ratio_features.extend([
                p_pos,
                f"DONCHIAN_BW_{p}",
                f"DONCHIAN_DIST_U_{p}",
                f"DONCHIAN_DIST_L_{p}"
            ])

        self.features = self.price_features + self.ratio_features

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        high, low, close = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
        res = {}
        for p in self.periods:
            upper, lower = high.rolling(p).max(), low.rolling(p).min()
            middle = (upper + lower) / 2
            range_hl = upper - lower
            res[f"DONCHIAN_UPPER_{p}"], res[f"DONCHIAN_LOWER_{p}"], res[f"DONCHIAN_MIDDLE_{p}"] = upper, lower, middle
            res[f"DONCHIAN_POS_{p}"] = np.where(range_hl > EPS, (close - lower) / range_hl, 0.5)
            res[f"DONCHIAN_BW_{p}"] = np.where(middle > EPS, range_hl / middle, 0.0)
            res[f"DONCHIAN_DIST_U_{p}"] = (upper - close) / (close + EPS)
            res[f"DONCHIAN_DIST_L_{p}"] = (close - lower) / (close + EPS)
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        利用 Factory 的组归一化能力，一次性处理所有周期的特征
        """
        # 1. 价格类：所有周期的 Upper/Lower/Middle 统一挂钩 Close 进行相对标准化
        if self.price_features:
            self._normalize_z_score_rel(
                X, feature_cols, self.price_features, 
                feature_base="close", factory=factory, method='log'
            )

        # 2. POS 类：中心化到 [-0.5, 0.5]
        # 这样 0 代表中轨，正数代表上半区，负数代表下半区
        pos_indices = [factory._feature_index[f] for f in self.pos_features if f in factory._feature_index]
        if pos_indices:
            X[:, :, pos_indices] = X[:, :, pos_indices] - 0.5

        # 3. 其他比例类 (BW, DIST)：使用信号组归一化 + Log 压缩
        # 这样不同周期的波动率挤压程度具有可比性
        other_ratios = [f for f in self.ratio_features if f not in self.pos_features]
        if other_ratios:
            self._normalize_signal_group(
                X, feature_cols, other_ratios, 
                factory=factory, method='log'
            )

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # 请求最大周期的一定倍数，确保归一化统计量稳定
        return int(max(self.periods) * 1.5)

class FeatureKeltner(FeatureBase):
    """
    肯特纳通道 (Keltner Channels):
    Middle Band = EMA(Close, N)
    Upper Band = Middle + Multiplier * ATR(N)
    Lower Band = Middle - Multiplier * ATR(N)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.period = kwargs.get('period', 20)
        self.multiplier = kwargs.get('multiplier', 2.0)
        self.features = [
            f"KELTNER_UPPER_{self.period}",
            f"KELTNER_LOWER_{self.period}",
            f"KELTNER_MIDDLE_{self.period}"
        ]

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close, high, low = df['close'].astype(float), df['high'].astype(float), df['low'].astype(float)
        res = {}
        middle = close.ewm(span=self.period, adjust=False).mean()
        tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(window=self.period).mean()
        res[f"KELTNER_UPPER_{self.period}"] = middle + (self.multiplier * atr)
        res[f"KELTNER_LOWER_{self.period}"] = middle - (self.multiplier * atr)
        res[f"KELTNER_MIDDLE_{self.period}"] = middle
        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # 同样使用 SCS 归一化，保护通道几何形态
        # self._normalize_scs(X, feature_cols, self.features, "close")
        self._normalize_z_score_rel(X, feature_cols , self.features , feature_base = "close", factory= factory, method='log')

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # EMA 需要约 4 倍周期进行预热收敛
        return int(self.period * 4)

class FeatureBoll(FeatureBase):
    """
    Bollinger Bands (布林带):
    Middle Band = SMA(Close, N)
    Upper Band = Middle + Multiplier * StdDev(Close, N)
    Lower Band = Middle - Multiplier * StdDev(Close, N)
    Bandwidth = (Upper - Lower) / Middle  (波动率量化)
    %B = (Close - Lower) / (Upper - Lower) (价格在通道内的位置)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.period = kwargs.get('period', 20)
        self.multiplier = kwargs.get('multiplier', 2.0)
        
        # 价格通道特征
        self.price_features = [
            f"BOLL_UPPER_{self.period}",
            f"BOLL_LOWER_{self.period}",
            f"BOLL_MIDDLE_{self.period}"
        ]
        # 无量纲衍生特征
        self.ratio_features = [
            f"BOLL_BW_{self.period}", # Bandwidth: 描述波动率挤压
            f"BOLL_PB_{self.period}"  # %B: 描述价格相对位置
        ]
        self.features = self.price_features + self.ratio_features

    def generate(self, df: pd.DataFrame, kline_interval_ms: int) -> pd.DataFrame:
        close = df['close'].astype(float)
        res = {}
        
        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper = middle + (self.multiplier * std)
        lower = middle - (self.multiplier * std)
        
        res[f"BOLL_UPPER_{self.period}"] = upper
        res[f"BOLL_LOWER_{self.period}"] = lower
        res[f"BOLL_MIDDLE_{self.period}"] = middle
        res[f"BOLL_BW_{self.period}"] = (upper - lower) / (middle + EPS)
        res[f"BOLL_PB_{self.period}"] = (close - lower) / (upper - lower + EPS)

        return pd.DataFrame(res, index=df.index)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # A. 通道价格类：使用 Z-Score 与收盘价对齐
        self._normalize_z_score_rel(X, feature_cols, self.price_features, feature_base="close", factory=factory, method= 'log')
        
        # B. 百分比位置 %B: 范围约在 [0, 1]，进行中心化处理
        pb_idx = [factory._feature_index[f] for f in [self.ratio_features[1]] if f in factory._feature_index]
        if pb_idx:
            X[:, :, pb_idx] = X[:, :, pb_idx] - 0.5
            
        # C. 带宽 BW: 是正数且具有长尾特征，使用 log1p 压制
        bw_idx = [factory._feature_index[f] for f in [self.ratio_features[0]] if f in factory._feature_index]
        if bw_idx:
            X[:, :, bw_idx] = np.log1p(X[:, :, bw_idx])

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # SMA 基础窗口
        return self.period
    
class FeatureOrigin(FeatureBase):   #增加一个 taker_buy_base_volume/volume 占比参数，taker_buy_quote_volume/quote_asset_volume 占比参数
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.price_base_features = ['open', 'high', 'low', 'close', ]#[]
        self.volume_base_features = ['taker_buy_base_volume', 'volume']#[] # feature used as basic must be the last!!!
        self.quote_base_features  = ['taker_buy_quote_volume', 'quote_asset_volume']#[]   #the basic is quote_asset
        self.self_based_features = ['number_of_trades']#[]
        self.features = self.price_base_features + self.volume_base_features + self.quote_base_features + self.self_based_features
    def generate(self,df:pd.DataFrame, kline_interval_ms: int = None):
        # for f in self.factory.base_features:
        #     df[f'base_{f}'] = df[f]
        pass
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize_z_score_rel(X, feature_cols , self.price_base_features , feature_base = self.price_base_features[-1], factory= factory, method= 'log')
        self._normalize_z_score_rel(X, feature_cols , self.volume_base_features , feature_base = self.volume_base_features[-1], factory= factory, method= 'log')
        self._normalize_z_score_rel(X, feature_cols , self.quote_base_features , feature_base = self.quote_base_features[-1], factory= factory, method= 'log')
        # self._normalize_z_score_group(X, feature_cols , self.price_base_features , factory= factory, method= 'log')
        # self._normalize_z_score_group(X, feature_cols , self.volume_base_features , factory= factory, method= 'log')
        # self._normalize_z_score_group(X, feature_cols , self.quote_base_features , factory= factory, method= 'log')#_normalize_z_score_group
        for f in self.self_based_features:
            self._normalize_z_score(X, feature_cols , [f] , feature_base = f, factory= factory, method= 'log')
            
    def _min_history_request(self, kline_interval_ms:int = None) -> int:
        return 1

#均线 动量 结构
# --- 1. 价格趋势与指标类 ---
FCVolumeEvent = FeatureContainer(FeatureVolumeEvent, **{"windows": [5000, 1500, 1000, 500,200,100], "top_k": 3})
FCMACD        = FeatureContainer(FeatureMACD, **{"fast": 12, "slow": 26, "signal": 9})
FCMA          = FeatureContainer(FeatureMAStructure, bar_windows=(7, 21, 63),day_windows=(5, 20),week_windows=(7, 25),add_delta=True,method="sma",strict=True,)
FCRSI         = FeatureContainer(FeatureRsi, **{"period": 14, "price_col": 'close', "strict": True, "prefix": 'RSI'})
FCKDJ         = FeatureContainer(FeatureKdj, **{"n": 9, "m1": 3, "m2": 3, "strict": True, "prefix": 'KDJ'})

# --- 2. 价格通道类 ---
FCDonchian    = FeatureContainer(FeatureDonchian, **{"periods": [7,25,99]})
FCKeltner     = FeatureContainer(FeatureKeltner, **{"period": 14, "multiplier": 2})
FCBoll        = FeatureContainer(FeatureBoll, **{"period": 25})

# --- 3. 量能与成交活跃度类 ---
FCVolMa       = FeatureContainer(FeatureVolMa, **{"vol_ma_windows": [7, 14, 25, 99]})
FCQavMa       = FeatureContainer(FeatureQavMa, **{"windows": [7, 25, 99]})
FCOBV         = FeatureContainer(FeatureOBV, **{})
FCPVT         = FeatureContainer(FeaturePVT, **{})
FCWAP         = FeatureContainer(FeatureWAP, **{"vwap_windows": [7,25,99], "add_bias": True})
FCCFM         = FeatureContainer(FeatureCFM, **{"cmf_windows": [7,25,99]})
FCMFI         = FeatureContainer(FeatureMFI, **{"mfi_windows": [7,25,99,200,500,999]})
FCATS         = FeatureContainer(FeatureATS, **{})

FCAdvancedVol = FeatureContainer(FeatureAdvancedVol, windows = [14, 100])
FCFractalPersistence = FeatureContainer(FeatureFractalPersistence, windows = [14, 126])
FCOrderFlow = FeatureContainer(FeatureOrderFlow, windows = [14, 49],poc_bias_step= [7,25,99,200,300,400,500,600,700,800,900], include_poc_bias=True)
FCOrderClassicFactors = FeatureContainer(FeatureClassicFactors, windows = [20, 100])
FCOrderMomentum = FeatureContainer(FeatureMomentum, horizons=[10, 20, 60],include_skip=True, skip_horizon=20,
                                        include_vol_adj=True, vol_adj_horizon=20, vol_window=20)


FCATR         = FeatureContainer(FeatureATRRegime, windows = [14])   #14, 16, 1000 , 2000, 5000

# --- 4. K线形态与原始数据 ---
FCCandle      = FeatureContainer(FeatureCandle, **{})
FCOrigin      = FeatureContainer(FeatureOrigin, **{})

# ==============================================================================
# 2. 最终特征配置列表 (FEATURE_GROUP_LIST)
# ==============================================================================

FEATURE_GROUP_LIST = [
    # 1. 自定义的成交量爆发特征 (窗口 512，对比前 2 强)
    FCVolumeEvent, 

    # 2. 价格趋势与指标类    FeatureMA > FeatureRsi/FeatureKdj/FeatureMACD   what happen to FeatureMACD??
    FCMACD,   # （12，26，9），（6，13，5）或（10，20，7）
    FCMA,     # slope 值搭配使用
    FCRSI,
    FCKDJ,

    # 价格通道类，2选1   FeatureKeltner >> FeatureBoll/FeatureDonchian
    FCDonchian, 
    FCKeltner,
    FCBoll,

    # 3. 量能与成交活跃度类 FeatureQavMa > FeatureMFI/FeatureWAP > FeatureCFM  > FeaturePVT >FeatureVolMa
    FCVolMa,
    FCQavMa,
    # FCOBV,    # 等于 FeaturePVT 丢掉幅度信息。不如 FeaturePVT，直接丢弃
    FCPVT,    # 累积性变量，对短期预测作用小，不如动量
    FCWAP,
    FCCFM,
    FCMFI,
    # FCATS,  # 负作用
    FCAdvancedVol,
    FCFractalPersistence,
    FCOrderFlow,
    FCOrderClassicFactors,
    FCOrderMomentum,
    FCATR,

    # 4. K线形态类
    FCCandle,
    FCOrigin,
]

class FeatureFactory:
    def __init__(self,kline_interval_ms:int,feature_group_list:list[FeatureContainer]= FEATURE_GROUP_LIST,feature_conf_list= []):
        self.all_feature_list = []  #feature names
        self.selected_feature_list = []  #feature names
        self.price_features = {}
        self._kline_interval_ms = kline_interval_ms
        self._X = None
        self._feature_index = None
        self._base_stats_pool = None
        self.feature_group_list :list[FeatureBase] = []
        self.base_features= ['open', 'high', 'low', 'close', 'taker_buy_base_volume', 'volume','taker_buy_quote_volume', 'quote_asset_volume','number_of_trades']
        for container in feature_group_list:
            instance =container.feature(factory = self,kline_interval_ms=kline_interval_ms, **container.parameters)
            self.feature_group_list.append(instance)
            self.all_feature_list.extend(instance.features)
        if feature_conf_list:
            filtered_groups = []
            for group in self.feature_group_list:
                for f in feature_conf_list:
                    if f in group.features:
                        filtered_groups.append(group)
                        break
            self.feature_group_list = filtered_groups

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        # 存储所有特征类生成的 DataFrame 块
        feature_blocks = [df] 

        for f in self.feature_group_list:
            # 获取该组生成的特征块
            # 注意：需确保每个特征类的 generate 此时都返回 DataFrame 或 dict
            res_block = f.generate(df, self._kline_interval_ms)
            if res_block is not None and not res_block.empty:
                feature_blocks.append(res_block)
        # 核心：一次性水平拼接。这是处理高维特征最快的方式。
        combined_df = pd.concat(feature_blocks, axis=1)
        # 如果担心内存碎片影响后续 normalize，可以 copy 一下
        return combined_df.copy()

    def _prepare_normalize_context(self, X, feature_cols):
        self._X = X
        self._feature_cols = tuple(feature_cols)
        self._feature_index = {f: i for i, f in enumerate(feature_cols)}
        self._base_stats_pool = {}

    def get_base_stats(self, base_feature):
        if self._X is None or self._feature_index is None:
            raise RuntimeError("prepare_normalize_context() must be called first")

        base_idx = (
            self._feature_index[base_feature]
            if isinstance(base_feature, str)
            else base_feature
        )

        if base_idx not in self._base_stats_pool:
            base = self._X[:, :, base_idx]
            mu = np.nanmean(base, axis=1, keepdims=True)
            sigma = np.nanstd(base, axis=1, keepdims=True)
            denom = sigma #+ 0.1 * np.abs(mu) + EPS
            self._base_stats_pool[base_idx] = (mu, denom)

        return self._base_stats_pool[base_idx]

    def fit_group_magnitude(self, df_list, target_cols, base_col):
        """
        在训练集上预计算组统计量
        df_list: 训练集的 DataFrame 列表
        target_cols: ['body', 'upper_wick', 'lower_wick', ...]
        base_col: 'close'
        """
        all_ratios = []
        for df in df_list:
            # 1. 计算比例：X / close
            # 这里简单用当前价格，也可以用滚动均值，视你 generate 逻辑而定
            base_price = df[base_col].replace(0, np.nan)
            for col in target_cols:
                ratio = (df[col] / base_price).dropna()
                all_ratios.append(ratio.values)
        
        # 2. 池化所有比例值，计算全局统计量
        combined_ratios = np.concatenate(all_ratios)
        group_mu = np.mean(combined_ratios)
        group_std = np.std(combined_ratios)
        
        # 3. 存储结果
        group_key = "_".join(sorted(target_cols))
        self.group_stats[group_key] = (group_mu, group_std)
        return group_mu, group_std

    def normalize(self, X: np.ndarray, feature_cols: list[str]):
        self._prepare_normalize_context(X, feature_cols)
        for group in self.feature_group_list:
            if any(f in feature_cols for f in group.features):
                group.normalize(X, feature_cols, self)
                
    def get_global_min_history(self) -> int:
        """遍历所有已注册特征，返回其中最大的历史需求"""
        return max([f.min_history_request(self._kline_interval_ms) for f in self.feature_group_list])