from abc import ABC, abstractmethod
import logging,math,os
import pandas as pd
import numpy as np
from numba import njit, float64, int64
import torch

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

    def _normalize_z_score(self, X, feature_cols, target_feature_cols, feature_base, factory, scale=None, method=None):
        target_indices = [factory._feature_index[f] for f in target_feature_cols]
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
        # 1. 安全获取特征索引
        target_indices = [factory._feature_index[f] for f in target_feature_cols if f in factory._feature_index]
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
        target_indices = [factory._feature_index[f] for f in target_feature_cols if f in factory._feature_index]
        if not target_indices: 
            return
        
        # 1. 提取组数据并计算 RMS (确保零轴不偏移，组内比例一致)
        vals = X[:, :, target_indices]
        rms_group = np.sqrt(np.nanmean(vals**2, axis=(1, 2), keepdims=True))
        
        # 2. 无量纲化基础缩放
        standardized = np.where(rms_group > EPS, vals / (rms_group), 0.0)

        X[:, :, target_indices] = self._apply_squashing(standardized, scale, method)

    def _normalize_z_score_rel(self, X, feature_cols, target_feature_cols, feature_base, factory, scale=None, method=None):
        target_indices = [factory._feature_index[f] for f in target_feature_cols]
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
        target_indices = [factory._feature_index[f] for f in target_feature_cols]
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
        target_indices = [factory._feature_index[f] for f in target_feature_cols]
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
        target_indices = [factory._feature_index[f] for f in target_feature_cols]
        if not target_indices:
            return

        mu, sigma = factory.get_base_stats(feature_base)

        X[:, :, target_indices] = np.where(
            sigma[:, :, None] > 0,
            (X[:, :, target_indices] - mu[:, :, None]) / sigma[:, :, None],
            0.0
        )

    def _normalize_magnitude(
        self, X, feature_cols, target_feature_cols, feature_base, factory
    ):
        """
        用于幅度 / 强度 / level 特征
        不做 (x-mu)/σ
        """
        target_indices = [factory._feature_index[f] for f in target_feature_cols]
        if not target_indices:
            return

        # 推荐默认：log 压缩
        X[:, :, target_indices] = np.log1p(
            np.maximum(X[:, :, target_indices], 0.0)
        )
    def register_features(self, features:list[str]):
        self.factory.feature_map.update({self:features})
    def prepare_depend_feature(self,df, feature_name:str):
        for feature_class , feature_list in self.factory.feature_map.items():
            if feature_name in feature_list:
                feature_class.generate(df,self.kline_interval_ms)
                return
        raise RuntimeError(f"request feature {feature_name} not exist")

    #抑制长尾
    def _normalize_volume_rlc(self, X, feature_cols, target_feature_cols, feature_base):
        """
        Relative Log-Compression (RLC) - 相对对数压缩缩放
        针对长尾分布（如成交量、成交额）设计，结合了无量纲化与极值抑制。
        
        :param feature_base: 作为无量纲化基准的特征名（如 'volume' 或 'quote_asset_volume'）
        """
        # 1. 获取索引
        target_indices = [self.factory._feature_index[f] for f in target_feature_cols if f in self.factory._feature_index]
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
        target_indices = [self.factory._feature_index[f] for f in target_feature_cols if f in self.factory._feature_index]
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
        min_request= self._min_history_request()
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

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        close = df['close'].astype(float)
        prefix = f"MACD_{self.fast}_{self.slow}"
        
        # --- A. 基础 EMA 计算 ---
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        
        # --- B. 计算原始绝对值指标 (DIF, DEA, HIST) ---
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal, adjust=False).mean()
        hist = dif - dea
        
        df[f'{prefix}_DIF'] = dif
        df[f'{prefix}_DEA'] = dea
        df[f'{prefix}_HIST'] = hist
        
        # --- C. 计算百分比化指标 (保持原有逻辑) ---
        # 去除分母为 0 的风险
        dif_pct = np.where(ema_slow != 0, (ema_fast - ema_slow) / ema_slow, np.nan)
        dif_pct_s = pd.Series(dif_pct, index=df.index)
        dea_pct = dif_pct_s.ewm(span=self.signal, adjust=False).mean()
        hist_pct = (dif_pct_s - dea_pct)
        
        df[f'{prefix}_DIF_PCT'] = dif_pct_s
        df[f'{prefix}_HIST_PCT'] = hist_pct
        df[f'{prefix}_HIST_ACCEL'] = hist_pct.diff().fillna(0)
        
        # SIG_DIST: DIF 与 DEA 的乖离度
        df[f'{prefix}_SIG_DIST'] = np.where(
            dea_pct != 0, 
            (dif_pct_s - dea_pct) / np.abs(dea_pct), 
            np.nan
        )

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
        
"""
基于时间列自动推断 1 根K线的时间长度，计算：
    * 周均线（SMA/EMA）：如 7W / 25W，并可选计算周均线斜率
    * 日均线（SMA/EMA）：如 5D / 10D / 20D（新增，天数可参数化）

- 时间列支持：字符串/DatetimeIndex/带/不带时区；若为数字则按毫秒时间戳处理
- 周均线列名：SMA_{w}W / EMA_{w}W
- 日均线列名：SMA_{d}D / EMA_{d}D
- 仅对“周均线”计算斜率，保持原函数语义不变
"""
class FeatureMA(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weeks = kwargs.get('weeks', [7, 25])
        self.days = kwargs.get('days', [5, 10, 20])
        self.bars = kwargs.get('bars', [7, 25, 99])
        self.method = kwargs.get('method', 'sma').upper()
        self.strict = kwargs.get('strict', True)
        self.add_slope = kwargs.get('add_slope', False)
        self.slope_method = kwargs.get('slope_method', 'reg')
        self.slope_weeks = kwargs.get('slope_weeks', 2)  # 斜率回看基准周期
        
        self.absolute_features = []
        self.ratio_features = []

        # 1. 统一注册逻辑：遍历三个维度 (Bars, Days, Weeks)
        # 维度配置：(列表, 后缀名)
        dimensions = [
            (self.bars, "B"),
            (self.days, "D"),
            (self.weeks, "W")
        ]

        for val_list, suffix in dimensions:
            for v in val_list:
                # 注册均线值
                ma_col = f"{self.method}_{v}{suffix}"
                self.features.append(ma_col)
                self.absolute_features.append(ma_col)
                
                # 注册斜率
                if self.add_slope:
                    prefix = "SLOPE_DIFF" if self.slope_method == 'diff' else "SLOPE_REG"
                    slope_col = f"{prefix}_{ma_col}_{self.slope_weeks}W"
                    self.features.append(slope_col)
                    self.ratio_features.append(slope_col)

    def _calculate_klines_count(self, kline_interval_ms: float) -> tuple[int, int]:
        """
        [纯数学计算] 根据给定的 K 线周期毫秒数，计算每天和每周包含多少根 K 线。
        kline_interval_ms: e.g., 900000 (15m), 60000 (1m)
        返回: (klines_per_day, klines_per_week)
        """
        if kline_interval_ms <= 0:
            # 默认保底值 (假设为 15m 周期: 24*4=96)
            return 96, 96 * 7
            
        one_day_ms = 24 * 60 * 60 * 1000
        one_week_ms = 7 * one_day_ms

        # 使用 round 确保由于浮点数微小误差（如 14.99999）导致的取整错误
        klines_per_day = max(int(round(one_day_ms / kline_interval_ms)), 1)
        klines_per_week = max(int(round(one_week_ms / kline_interval_ms)), 1)
        
        return klines_per_day, klines_per_week

    def _slope_reg_vectorized(self, series: pd.Series, steps: int, strict: bool = True) -> pd.Series:
        """
        [优化] 向量化线性回归斜率计算。
        通过预计算 x 的统计量，将 O(N*W) 的滑动窗口回归简化为 O(N) 的向量运算。
        series: 均线序列 (pd.Series)
        steps:  回看窗口根数 (int)
        strict: 如果窗口内数据不足 steps 根，是否返回 NaN
        """
        if steps <= 1: 
            return pd.Series(np.nan, index=series.index)
        
        # 1. 预计算 x 轴（时间轴）的常量
        # x = [0, 1, 2, ..., steps-1]
        n = float(steps)
        x_mean = (n - 1) / 2.0
        # x 的方差和 sum(x^2) 的简化公式
        var_x = (n * (n**2 - 1)) / 12.0
        
        # 2. 向量化计算 y 的和 以及 x*y 的和
        # 我们使用 cumsum 来模拟 rolling_sum
        y_filled = series.fillna(0)
        s1 = y_filled.cumsum()
        s2 = s1.cumsum()
        
        # sum_y = 当前点的累计和 - steps 之前的累计和
        sum_y = s1 - s1.shift(steps)
        
        # 利用双重累加和计算 sum(i * y_i)
        # 物理意义：s2 包含了 y 的加权历史信息
        shift_s1 = s1.shift(steps)
        shift_s2 = s2.shift(steps)
        weighted_sum_rev = (s2 - shift_s2) - steps * shift_s1
        sum_xy = (steps * sum_y) - weighted_sum_rev
        
        # 3. 最小二乘法公式: slope = Cov(x, y) / Var(x)
        y_mean = sum_y / n
        slope = (sum_xy - n * x_mean * y_mean) / var_x
        
        # 4. 严格模式处理 (NaN 填充)
        if strict:
            # 只有当窗口内实际非空数据达到 steps 时才有效
            valid = series.rolling(window=steps).count() >= steps
            slope = slope.where(valid, np.nan)
            
        return slope

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        klines_per_day, klines_per_week = self._calculate_klines_count(kline_interval_ms)
        close = df['close'].astype(float)
        m = self.method.lower()

        # ---- 内部工具函数 ----
        def _get_ma(series, window):
            if m == 'sma':
                return series.rolling(window=window, min_periods=window if self.strict else 1).mean()
            else:
                ema = series.ewm(span=window, adjust=False).mean()
                if self.strict:
                    ema = ema.where(series.expanding().count() >= window, np.nan)
                return ema

        def _get_slope(series, steps):
            if steps <= 1: return pd.Series(np.nan, index=series.index)
            if self.slope_method == 'diff':
                # 使用你要求的百分比变化率逻辑
                return (series / series.shift(steps).replace(0, np.nan) - 1.0) / steps
            else:
                # 线性回归斜率（建议也在此基础上进行百分比处理，或保持原样配合自缩放归一化）
                return self._slope_reg_vectorized(series, steps, self.strict)

        # 2. 统一生成逻辑
        dimensions = [
            (self.bars, "B", 1),
            (self.days, "D", klines_per_day),
            (self.weeks, "W", klines_per_week)
        ]

        for val_list, suffix, multiplier in dimensions:
            for v in val_list:
                window = max(int(round(v * multiplier)), 1)
                ma_col = f"{self.method}_{v}{suffix}"
                
                # 计算并存储均线
                df[ma_col] = _get_ma(close, window)
                
                # 计算并存储斜率
                if self.add_slope:
                    # 斜率的回看步长统一以 slope_weeks 为准，或者你也可以改为以 v 为准
                    steps = max(int(round(klines_per_week * float(self.slope_weeks))), 1)
                    slope_col = f"{'SLOPE_DIFF' if self.slope_method == 'diff' else 'SLOPE_REG'}_{ma_col}_{self.slope_weeks}W"
                    
                    df[slope_col] = _get_slope(df[ma_col], steps)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        统一归一化逻辑
        """
        # 均线值：跟随 close 波动，进行组归一化，解决“海拔”问题
        if self.absolute_features:
            self._normalize_z_score_rel(X, feature_cols, self.absolute_features, feature_base = "close", factory=factory, method='log')
        
        # 斜率：每个斜率的波动率差异极大，强制使用自缩放 + 长尾压制
        if self.ratio_features:
            for f in self.ratio_features:
                #  这里使用带 suppress=True 的自缩放版本，防止分母塌陷和极端值
                self._normalize_z_score_rel(X, feature_cols, [f], feature_base= f, factory=factory, method='log')
    def _min_history_request(self, kline_interval_ms: int) -> int:
        """
        [实现] 根据当前 K 线周期，计算模型所需的最少历史 K 线数量
        """
        # 调用封装好的周期计算函数
        k_day, k_week = self._calculate_klines_count(kline_interval_ms)
        
        # 1. 均线最大窗口 (取周线和日线中较大的需求)
        max_d_window = max(self.days) * k_day if self.days else 0
        max_w_window = max(self.weeks) * k_week if self.weeks else 0
        base_required = max(max_d_window, max_w_window)
        
        # 2. 考虑斜率窗口 (回看步数)
        slope_buffer = 0
        if self.add_slope:
            slope_buffer = self.slope_weeks * k_week
            
        # 3. EMA 精度补偿因子 (EMA 需要约 3.5 倍窗口才能使初始权重衰减至 <1%)
        multiplier = 3.5 if self.method.lower() == 'ema' else 1.0
        
        # 总请求量 = (最大周期 * 预热系数) + 斜率回看
        total_needed = int(base_required * multiplier + slope_buffer)
        
        # 保底返回 200 根，确保数据量足够计算基础指标
        return max(total_needed, 200)

#Dimensionless
class FeatureRsi(FeatureBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.period:int = kwargs.get('period', 14)
        self.price_col = kwargs.get('price_col', 'close')
        self.strict = kwargs.get('strict', True)# 严格型：窗口未满为 NaN；宽松型：尽早给值
        self.prefix = kwargs.get('prefix', "RSI")
        self.features :list[str]= [f"{self.prefix}_{self.period}"]
    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        out = df
        close = out[self.price_col].astype(float)

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        # 1. 计算平滑值
        avg_gain = gain.ewm(alpha=1/float(self.period), adjust=False,
                            min_periods=self.period if self.strict else 1).mean()
        avg_loss = loss.ewm(alpha=1/float(self.period), adjust=False,
                            min_periods=self.period if self.strict else 1).mean()

        # 2.  健壮性计算：直接处理 rs，不要 replace(0, np.nan)
        # 加上 EPS 仅仅是为了防止分母为 0 时的 RuntimeWarning 刷屏，逻辑判断会覆盖它
        rs = avg_gain / (avg_loss + EPS)
        rsi_values = np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))

        # 3. 处理死水行情 (涨跌均为 0)
        rsi_values = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, rsi_values)

        # 4.  赋值与 Strict 模式裁剪
        col = f"{self.prefix}_{self.period}"
        
        # 只有当数据根数足够时才保留结果，否则置为 NaN
        valid = close.expanding().count() >= self.period
        out[col] = np.where(valid, rsi_values, np.nan)
            
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        RSI 范围 [0, 100]。
        处理：(RSI / 100) - 0.5
        结果范围：[-0.5, 0.5]
        0.0 对应 RSI 50 (中性)
        """
        target_indices = [feature_cols.index(f) for f in self.features if f in feature_cols]
        
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
    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        for c in (self.high_col, self.low_col, self.close_col):
            if c not in df.columns:
                raise ValueError(f"缺少列 {c}")

        out = df
        high = out[self.high_col].astype(float)
        low = out[self.low_col].astype(float)
        close = out[self.close_col].astype(float)

        llv = low.rolling(window=self.n, min_periods=self.n if self.strict else 1).min()
        hhv = high.rolling(window=self.n, min_periods=self.n if self.strict else 1).max()

        # ---  核心改进：处理 RSV 的除零风险 ---
        diff = hhv - llv
        # 如果周期内最高等于最低，说明没有波动，RSV 设为 50
        rsv = np.where(diff == 0, 50.0, (close - llv) / (diff + EPS) * 100.0)

        # 用 EWM 实现等价的递推平滑
        # 注意：这里的 rsv 是 numpy 数组，可以直接传给 pd.Series 进行平滑
        rsv_s = pd.Series(rsv, index=df.index)
        
        K = rsv_s.ewm(alpha=1/float(self.m1), adjust=False,
                    min_periods=self.m1 if self.strict else 1).mean()
        D = K.ewm(alpha=1/float(self.m2), adjust=False,
                min_periods=self.m2 if self.strict else 1).mean()
        J = 3 * K - 2 * D

        k_col, d_col, j_col = f"{self.prefix}_K", f"{self.prefix}_D", f"{self.prefix}_J"
        
        valid = close.expanding().count() >= self.n
        out[k_col] = K.where(valid, np.nan)
        out[d_col] = D.where(valid, np.nan)
        out[j_col] = J.where(valid, np.nan)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        #KDJ 是 0-100 指标，使用简单缩放
        target_indices = [feature_cols.index(f) for f in self.features if f in feature_cols]
        if not target_indices: return

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

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        close = df['close'].astype(float)
        h, l = df['high'], df['low']
        pc = close.shift(1)
        
        # 1. 计算 True Range (基础波幅)
        tr = np.maximum(h - l, np.maximum((h - pc).abs(), (l - pc).abs()))
        
        # 2. 循环生成多个周期的 NATR (百分比化 ATR)
        atr_series_map = {}
        for w in self.windows:
            atr_w = tr.rolling(w).mean()
            atr_series_map[w] = atr_w
            # 存储为百分比特征，方便模型理解和下单参考
            df[f'atr_{w}'] = np.where(close > 0, atr_w / close, 0.0)
        
        # 3. 计算“波动率一致性环境” (Regime)
        # 逻辑：短期(14) ATR / 长期(1000) ATR 均值
        short_atr = atr_series_map[self.short_w]
        long_atr_ref = short_atr.rolling(self.long_w).mean()
        
        # 结果 > 1 代表当前比长期更活跃，值越稳定代表波动越均匀
        df[f'vol_regime_{self.long_w}'] = np.where(long_atr_ref > 0, short_atr / long_atr_ref, 1.0)
        
        # 4. 波动率趋势 (短期动量)
        df['Vol_Trend'] = short_atr.diff(5) / (short_atr.shift(5) + EPS)

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
    def generate(self,df:pd.DataFrame, kline_interval_ms:int):
        # ==== 1. 成交量均线 + 比值 ====
        for w in self.vol_ma_windows:
            vol_ma = df['volume'].rolling(w).mean()
            df[f'VOL_MA_{w}'] = vol_ma
            # 使用 np.where 进行条件判断：
            # 如果 vol_ma > 0，计算比值；否则直接给 0.0
            df[f'VOL_ratio_{w}'] = np.where(
                vol_ma > EPS,              # 考虑浮点数精度，用一个极小值代替 0
                df['volume'] / vol_ma, 
                0.0
            )
            # 将比值限制在 [0, 50] 之间，防止极端离群值破坏特征分布
            # df[f'VOL_ratio_{w}'] = df[f'VOL_ratio_{w}'].clip(upper=50.0)
            
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

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        qav = df['quote_asset_volume'].astype(float)
        vol = df['volume'].astype(float)
        close = df['close'].astype(float)

        # 1. 计算均线基础 (预计算，减少循环内的重复计算)
        for w in self.windows:
            ma_qav = qav.rolling(w).mean()
            
            # --- 爆发比率 (Surge): 当前成交额 / 均值 ---
            if self.add_surge:
                df[f'QAV_SURGE_{w}'] = np.where( ma_qav > EPS,  qav / ma_qav, 1.0 )
            
            # --- 趋势斜率 (Slope): 取对数后再算斜率，解决 QAV 长尾问题 ---
            if self.add_slope:
                #  这里不再是简单的 (ma - ma.shift)
                # 而是使用回归函数，拟合过去 slope_step 长度内 ma_qav 的走向
                # 这样即使最后一根 K 线是插针，由于前面几根线的支撑，斜率也不会乱跳
                df[f'QAV_SLOPE_{w}'] = self._slope_reg_vectorized(ma_qav, self.slope_step)

        # 2. 计算 VWAP 偏离度 (Bias)
        if self.add_bias:
            # VWAP = 成交额 / 成交量 (这里是单根 K 线的成交均价)
            # 如果没有成交量，vwap 设为 0
            vwap = np.where(vol > EPS, qav / vol, 0.0)
            # 只有当 vwap > 0 时才计算偏离度，否则直接设为 0.0
            # 这样即使没有成交，特征值也是平稳的，不会产生 inf 或 nan
            df['VWAP_BIAS'] = np.where( vwap > EPS,  (close / vwap) - 1.0,   0.0)

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
        max_w = max(self.vol_ma_windows) if self.vol_ma_windows else 0
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
        # 这里的 window 建议与 TimeSeriesWindowDataset 中的 window 保持一致
        # 如果类中没有保存 window，可以给一个稳健的默认值（如 200）
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

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        # 使用 QAV (quote_asset_volume) 代替 price * volume，含金量更高
        qav = df['quote_asset_volume'].astype(float)
        vol = df['volume'].astype(float)
        close = df['close'].astype(float)

        for w in self.vwap_windows:
            # 计算滚动窗口内的总成交额和总成交量
            rolling_qav = qav.rolling(w).sum()
            rolling_vol = vol.rolling(w).sum()
            
            #  计算原始 VWAP：成交额 / 成交量
            # 安全防护：如果窗口内无成交量，VWAP 暂设为当前 close (或者 NaN)
            vwap_series = np.where(rolling_vol > EPS, rolling_qav / rolling_vol, close)
            vwap_col = f'VWAP_{w}'
            df[vwap_col] = vwap_series
            
            #  计算 VWAP Bias：(当前价 / 平均成本) - 1
            if self.add_bias:
                bias_col = f'VWAP_Bias_{w}'
                # 只有在 vwap 有效且大于 0 时计算偏离
                df[bias_col] = np.where(
                    vwap_series > EPS, 
                    (close / vwap_series) - 1.0, 
                    0.0
                )

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

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        # 1. 基础数据准备 (建议使用 QAV 以获取更真实的资金流信息)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        volume = df['quote_asset_volume'].astype(float) 

        # 2. 计算典型价格 Typical Price (TP) 和 资金流 (MF)
        tp = (high + low + close) / 3.0
        mf = tp * volume
        
        # 3. 确定流入与流出方向
        tp_diff = tp.diff()
        # tp_diff > 0 为流入，tp_diff < 0 为流出
        pos_mf = np.where(tp_diff > 0, mf, 0.0)
        neg_mf = np.where(tp_diff < 0, mf, 0.0)
        
        # 转换为 Series 方便滑动窗口运算
        pos_mf_series = pd.Series(pos_mf, index=df.index)
        neg_mf_series = pd.Series(neg_mf, index=df.index)

        # 4. 多窗口占比计算
        for w in self.mfi_windows:
            p_sum = pos_mf_series.rolling(w).sum()
            n_sum = neg_mf_series.rolling(w).sum()
            
            # 总资金流：流入 + 流出
            total_mf = p_sum + n_sum
            
            # --- 核心改造：占比版公式 ---
            # 如果总资金流大于 0，计算流入占比；否则给中性值 50.0
            # 这里使用了 np.divide 的 where 参数，能完美规避除零警告，且无需 EPS
            mfi = np.divide(
                100.0 * p_sum, 
                total_mf, 
                out=np.full_like(p_sum, 50.0), # 默认填充中性值
                where=total_mf > 0
            )
            
            df[f'MFI_{w}'] = mfi

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
        self.ratio_features = []
        # 动态注册所有特征
        self.features = []
        for w in self.windows:
            self.features.append(f'vol_event_ratio_{w}')
            self.features.append(f'vol_event_flag_{w}')
            self.ratio_features.append(f'vol_event_ratio_{w}')

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        # 确保传入的是 float64 类型的 Numpy 数组
        vol_values = df['volume'].values.astype(np.float64)
        
        for w in self.windows:
            ratio_col = f'vol_event_ratio_{w}'
            flag_col = f'vol_event_flag_{w}'
            
            # 调用手动循环优化的 JIT 函数
            v_ref_raw_values = calc_rolling_top_k_reference(vol_values, w, self.top_k)
            
            # 转换为 Series 进行后续的 EWM 处理
            v_ref_raw = pd.Series(v_ref_raw_values, index=df.index)
            # 适当调大 alpha 让基准回落稍快，保持灵敏
            v_ref = v_ref_raw.ewm(alpha=1 / (w * 2), adjust=False).mean()
            
            # 这里的计算是在 Pandas/Numpy 层面，非常稳健
            df[ratio_col] = np.where(v_ref > EPS, df['volume'] / v_ref, np.nan)
            df[flag_col] = (df[ratio_col] >= 1.0).astype(int)
            
            _logger.debug(f"[{self.__class__.__name__}] Success: {ratio_col}")
        # 在 generate 之后加一行
        for w in self.windows:
            count = df[f'vol_event_flag_{w}'].sum()
            _logger.debug(f"Window {w} triggered {count} events.")

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # Event-level feature.
        # Do NOT normalize / z-score / center.
        self._normalize_z_score_rel(X, feature_cols , self.ratio_features , feature_base = self.ratio_features[0], factory= factory)

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

    def generate(self, df: pd.DataFrame, kline_interval_ms: int = None):
        o, h, l, c = df['open'], df['high'], df['low'], df['close']

        # === A. 绝对量级类 ===
        df['body'] = np.abs(c - o)
        df['upper_wick'] = h - np.maximum(o, c)
        df['lower_wick'] = np.minimum(o, c) - l
        df['max_range'] = h - l
        df['body_mom'] = df['body'].diff().fillna(0)

        # === B. 比例类 [0, 1] ===
        # 移除 EPS，改用安全触发
        rng = df['max_range']
        
        # 使用 np.where 避免除零，若振幅为 0（一字板）则比例设为 0
        df['body_pct'] = np.where(rng > 0, df['body'] / rng, 0.0)
        df['upper_wick_pct'] = np.where(rng > 0, df['upper_wick'] / rng, 0.0)
        df['lower_wick_pct'] = np.where(rng > 0, df['lower_wick'] / rng, 0.0)
        df['close_pos'] = np.where(rng > 0, (c - l) / rng, 0.5) # 一字板视为中性
        df['doji_score'] = 1.0 - df['body_pct']

        # === C. 评分类 ===
        df['wick_bias'] = df['upper_wick_pct'] - df['lower_wick_pct']

        # df[self.features] = df[self.features].replace([np.inf, -np.inf], np.nan).fillna(0)
        return df

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
        self.prefix = kwargs.get('prefix', 'DONCHIAN')
        
        self.price_features = []
        self.ratio_features = []
        self.pos_features = []  # 专门记录 POS 特征以便中心化处理

        # 动态构建特征列表
        for p in self.periods:
            # 1. 价格类
            self.price_features.extend([
                f"{self.prefix}_UPPER_{p}",
                f"{self.prefix}_LOWER_{p}",
                f"{self.prefix}_MIDDLE_{p}"
            ])
            # 2. 比例/形态类
            p_pos = f"{self.prefix}_POS_{p}"
            self.pos_features.append(p_pos)
            self.ratio_features.extend([
                p_pos,
                f"{self.prefix}_BW_{p}",
                f"{self.prefix}_DIST_U_{p}",
                f"{self.prefix}_DIST_L_{p}"
            ])

        self.features = self.price_features + self.ratio_features

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)

        for p in self.periods:
            # --- 基础计算 ---
            upper = high.rolling(window=p).max()
            lower = low.rolling(window=p).min()
            middle = (upper + lower) / 2
            range_hl = upper - lower

            # --- 写入价格类特征 ---
            df[f"{self.prefix}_UPPER_{p}"] = upper
            df[f"{self.prefix}_LOWER_{p}"] = lower
            df[f"{self.prefix}_MIDDLE_{p}"] = middle

            # --- 写入形态特征 (模型友好) ---
            # 1. POS: 价格在通道内的相对位置 [0, 1]
            df[f"{self.prefix}_POS_{p}"] = np.where(range_hl > EPS, (close - lower) / range_hl, 0.5)
            
            # 2. BW: 带宽 (波动率挤压)
            df[f"{self.prefix}_BW_{p}"] = np.where(middle > EPS, range_hl / middle, 0.0)
            
            # 3. DIST: 距离上下轨的百分比距离
            df[f"{self.prefix}_DIST_U_{p}"] = (upper - close) / (close + EPS)
            df[f"{self.prefix}_DIST_L_{p}"] = (close - lower) / (close + EPS)

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
        self.prefix = kwargs.get('prefix', 'KELTNER')
        self.features = [
            f"{self.prefix}_UPPER_{self.period}",
            f"{self.prefix}_LOWER_{self.period}",
            f"{self.prefix}_MIDDLE_{self.period}"
        ]

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        # 1. 计算中轨 (EMA)
        middle = close.ewm(span=self.period, adjust=False).mean()
        
        # 2. 计算 ATR (Average True Range)
        tr = np.maximum((high - low), 
                        np.maximum(abs(high - close.shift(1)), 
                                   abs(low - close.shift(1))))
        atr = tr.rolling(window=self.period).mean()
        
        # 3. 计算上下轨
        upper = middle + (self.multiplier * atr)
        lower = middle - (self.multiplier * atr)
        
        df[self.features[0]] = upper
        df[self.features[1]] = lower
        df[self.features[2]] = middle

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
        self.prefix = kwargs.get('prefix', 'BOLL')
        
        # 价格通道特征
        self.price_features = [
            f"{self.prefix}_UPPER_{self.period}",
            f"{self.prefix}_LOWER_{self.period}",
            f"{self.prefix}_MIDDLE_{self.period}"
        ]
        # 无量纲衍生特征
        self.ratio_features = [
            f"{self.prefix}_BW_{self.period}", # Bandwidth: 描述波动率挤压
            f"{self.prefix}_PB_{self.period}"  # %B: 描述价格相对位置
        ]
        self.features = self.price_features + self.ratio_features

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        close = df['close'].astype(float)
        
        # 1. 计算中轨 (SMA) 和滚动标准差
        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        
        # 2. 计算上下轨
        upper = middle + (self.multiplier * std)
        lower = middle - (self.multiplier * std)
        
        # 3. 计算 Bandwidth (带宽) - 衡量波动率大小
        bandwidth = (upper - lower) / (middle + EPS)
        
        # 4. 计算 %B - 价格在通道中的相对位置 (0=下轨, 1=上轨)
        percent_b = (close - lower) / (upper - lower + EPS)
        
        df[self.price_features[0]] = upper
        df[self.price_features[1]] = lower
        df[self.price_features[2]] = middle
        df[self.ratio_features[0]] = bandwidth
        df[self.ratio_features[1]] = percent_b

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
        self.price_base_features = ['open', 'high', 'close', 'low',]#[]
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
FCVolumeEvent = FeatureContainer(FeatureVolumeEvent, **{"windows": [5000, 1500], "top_k": 3})
FCMACD        = FeatureContainer(FeatureMACD, **{"fast": 12, "slow": 26, "signal": 9})
FCMA          = FeatureContainer(FeatureMA, **{"weeks": [7], "days": [], "bars": [7, 14], "method": 'sma', "strict": True, "add_slope": True})
FCRSI         = FeatureContainer(FeatureRsi, **{"period": 14, "price_col": 'close', "strict": True, "prefix": 'RSI'})
FCKDJ         = FeatureContainer(FeatureKdj, **{"n": 9, "m1": 3, "m2": 3, "strict": True, "prefix": 'KDJ'})

# --- 2. 价格通道类 ---
FCDonchian    = FeatureContainer(FeatureDonchian, **{"period": [7]})
FCKeltner     = FeatureContainer(FeatureKeltner, **{"period": 14, "multiplier": 2})
FCBoll        = FeatureContainer(FeatureBoll, **{"period": 25})

# --- 3. 量能与成交活跃度类 ---
FCVolMa       = FeatureContainer(FeatureVolMa, **{"vol_ma_windows": [14]})
FCQavMa       = FeatureContainer(FeatureQavMa, **{"windows": [49]})
FCOBV         = FeatureContainer(FeatureOBV, **{})
FCPVT         = FeatureContainer(FeaturePVT, **{})
FCWAP         = FeatureContainer(FeatureWAP, **{"vwap_windows": [7], "add_bias": True})
FCCFM         = FeatureContainer(FeatureCFM, **{"cmf_windows": [25]})
FCMFI         = FeatureContainer(FeatureMFI, **{"mfi_windows": [25]})
FCATS         = FeatureContainer(FeatureATS, **{})
FCATR         = FeatureContainer(FeatureATRRegime, windows = [14])   #14, 16, 1000 , 2000, 5000

# --- 4. K线形态与原始数据 ---
FCCandle      = FeatureContainer(FeatureCandle, **{})
FCOrigin      = FeatureContainer(FeatureOrigin, **{})

# ==============================================================================
# 2. 最终特征配置列表 (FEATURE_CONFIG_LIST)
# ==============================================================================

FEATURE_CONFIG_LIST = [
    # 1. 自定义的成交量爆发特征 (窗口 512，对比前 2 强)
    # FCVolumeEvent, 

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
    FCOBV,    # 等于 FeaturePVT 丢掉幅度信息。不如 FeaturePVT，直接丢弃
    FCPVT,    # 累积性变量，对短期预测作用小，不如动量
    FCWAP,
    FCCFM,
    FCMFI,
    FCATS,  # 负作用
    FCATR,

    # 4. K线形态类
    FCCandle,
    FCOrigin,
]

class FeatureFactory:
    def __init__(self, feature_conf_list:list[FeatureContainer], kline_interval_ms:int):
        self.all_feature_list = []  #feature names
        self.feature_map :dict[FeatureBase:[str]]= {}
        self.price_features = {}
        self._kline_interval_ms = kline_interval_ms
        self._X = None
        self._feature_index = None
        self._base_stats_pool = None
        self.feature_list :list[FeatureBase] = []
        self.base_features= ['open', 'high', 'low', 'close', 'taker_buy_base_volume', 'volume','taker_buy_quote_volume', 'quote_asset_volume','number_of_trades']
        for container in feature_conf_list:
            # 使用 **params 将字典解包为关键字参数传递给构造函数
            instance =container.feature(factory = self,kline_interval_ms=kline_interval_ms, **container.parameters)
            self.feature_list.append(instance)
            self.all_feature_list.extend(instance.features)

    def generate(self,df):
        for f in self.feature_list:     f.generate(df, self._kline_interval_ms)
        # cols_to_drop = [c for c in self.base_features if c in df.columns]
        # if cols_to_drop:
        #     df.drop(columns=cols_to_drop, inplace=True)
        #     _logger.debug(f"Dropped base features: {cols_to_drop}")
        # return df

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
        for f in self.feature_list:
            f.normalize(X, feature_cols, self)
    
    def get_global_min_history(self) -> int:
        """遍历所有已注册特征，返回其中最大的历史需求"""
        return max([f.min_history_request(self._kline_interval_ms) for f in self.feature_list])