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

    def _normalize_old(self, X, feature_cols, target_feature_cols, feature_base, factory):
        target_indices = [factory._feature_index[f] for f in target_feature_cols]
        if not target_indices:
            return
        mu, denom = factory.get_base_stats(feature_base)
        # 广播计算: (M, T, F_sub) - (M, 1, 1) / (M, 1, 1)
        # 注意：mu 和 denom 需要增加一个维度以匹配特征维度 F
        X[:, :, target_indices] = (X[:, :, target_indices] - mu[:, :, np.newaxis]) / denom[:, :, np.newaxis]
    def _normalize_z_score(self, X, feature_cols, target_feature_cols, feature_base, factory):
        target_indices = [factory._feature_index[f] for f in target_feature_cols]
        if not target_indices:
            return

        # 1. 获取基准列的波动率 (Sigma)
        # sigma_base 形状通常为 (M, 1) -> 我们需要它变为 (M, 1, 1) 以匹配 X (M, T, F)
        mu_base, sigma_base = factory.get_base_stats(feature_base)
        denom = sigma_base[:, :, np.newaxis] 

        # 2. 核心改进：一次性计算 target 区域内所有特征各自的 mu
        # X[:, :, target_indices] 的形状是 (M, T, len(target_indices))
        # 我们沿时间轴 (axis=1) 计算均值
        mu_self = np.nanmean(X[:, :, target_indices], axis=1, keepdims=True)

        # 3. 向量化赋值
        # 这里的计算会同时跨样本、跨时间和跨特征列进行广播
        X[:, :, target_indices] = np.where(
            denom > EPS,
            (X[:, :, target_indices] - mu_self) / denom,
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
    def _normalize_group_min_max(self, X, feature_cols, target_feature_cols, symmetric=False):
        """
        Skeleton-Kinetic Structural Scaling (SKSS) 核心实现
        坚决保护特征分布不被扭曲。
        """
        target_indices = [self.factory._feature_index[f] for f in target_feature_cols if f in self.factory._feature_index]
        if not target_indices: return
        
        X_group = X[:, :, target_indices]

        if symmetric:
            # 零轴对称模式：保护 MACD/Returns 等动能指标的符号语义
            abs_max = np.nanmax(np.abs(X_group), axis=(1, 2), keepdims=True)
            if np.any(abs_max == 0):
                raise ZeroDivisionError(f"SKSS Failure: Symmetric group {target_feature_cols} has no amplitude.")
            X[:, :, target_indices] = (X_group / abs_max) * 0.5
        else:
            # 整体结构模式：保护价格骨架 (Level) 的相对几何间距
            g_min = np.nanmin(X_group, axis=(1, 2), keepdims=True)
            g_max = np.nanmax(X_group, axis=(1, 2), keepdims=True)
            g_diff = g_max - g_min
            if np.any(g_diff == 0):
                raise ZeroDivisionError(f"SKSS Failure: Skeleton group {target_feature_cols} has collapsed.")
            X[:, :, target_indices] = (X_group - g_min) / g_diff

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
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.fast:int = kwargs.get('fast', 12)
        self.slow:int = kwargs.get('slow', 26)
        self.signal:int = kwargs.get('signal', 9)
        self.features = ['MACD_DIF', 'MACD_DEA', 'MACD']
    def generate(self,df:pd.DataFrame, kline_interval_ms:int):
        out = df
        close = out['close'].astype(float)
        # ---- MACD ----
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal, adjust=False).mean()
        macd = 2 * (dif - dea)

        out['MACD_DIF'] = dif
        out['MACD_DEA'] = dea
        out['MACD'] = macd
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize_group_min_max(X, feature_cols , self.features , symmetric = True)
    def _min_history_request(self, kline_interval_ms:int = None) -> int:
        """
        计算 MACD 所需的最小历史 K 线数量。
        由于 EMA 需要数据预热 (Warmup) 才能收敛，
        通常建议请求 (Slow + Signal) * 4 的长度以确保精度。
        """        
        # 基础周期：最慢的均线 + 信号线周期
        base_cycle = self.slow + self.signal
        # 乘以 4 倍用于 EMA 收敛 (Warmup Buffer)
        # 例如 (26 + 9) * 4 = 140 根 K 线
        return int(base_cycle * 4)
        
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
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.weeks = kwargs.get('weeks', [7,25])
        self.days = kwargs.get('days', [5, 10, 20])
        self.bars = kwargs.get('bars', [7, 25, 99])
        self.method = kwargs.get('method', 'sma') # 'sma' 或 'ema'
        self.strict = kwargs.get('strict', True)# 严格型：窗口未满为 NaN；宽松型：尽早给值
        self.add_slope = kwargs.get('add_slope', False)
        self.slope_method = kwargs.get('slope_method', 'reg')# 'diff' 或 'reg'
        self.slope_weeks = kwargs.get('slope_weeks', 2)     # 斜率回看窗口（单位：周）
        # 注册特征名
        # 1. 注册原生 K 线均线 (如 SMA_7B)
        for b in self.bars:
            ma_col = f"{self.method.upper()}_{b}B"
            self.features.append(ma_col)
        for d in self.days:
            ma_col = f"{'SMA' if self.method=='sma' else 'EMA'}_{d}D"
            self.features.append(ma_col)
        for w in self.weeks:
            ma_w_col = f"{'SMA' if self.method=='sma' else 'EMA'}_{w}W"
            self.features.append(ma_w_col)
            # 周线斜率（保持原行为）
            if self.add_slope:
                slope_col = f"{'SLOPE_DIFF_' if self.method=='diff' else 'SLOPE_REG_'}{ma_w_col}_{self.slope_weeks}W"
                self.features.append(slope_col)   #no need for normalize

    def _calculate_klines_count(self, kline_interval_ms: float):
        """
        [纯数学计算] 根据给定的周期毫秒数，计算每天和每周包含多少根 K 线。
        不再依赖 DataFrame。
        """
        if kline_interval_ms <= 0:
            # 默认保底值 (假设 15m)
            return 96, 96 * 7
            
        one_day_ms = 24 * 60 * 60 * 1000
        one_week_ms = 7 * one_day_ms

        # 使用 round 确保浮点精度误差不会导致向下取整错误
        klines_per_day = max(int(round(one_day_ms / kline_interval_ms)), 1)
        klines_per_week = max(int(round(one_week_ms / kline_interval_ms)), 1)
        
        return klines_per_day, klines_per_week
    
    def generate(self,df:pd.DataFrame, kline_interval_ms:int):
        price_col='close'

        interval_ms = kline_interval_ms
        klines_per_day, klines_per_week = self._calculate_klines_count(interval_ms)

        out = df
        close = out[price_col].astype(float)

        m = self.method.lower()
        if m not in ('sma', 'ema'):
            raise ValueError("method 只能为 'sma' 或 'ema'")

        # ---- 通用均线计算子函数 ----
        def _ma(series: pd.Series, window: int) -> pd.Series:
            if m == 'sma':
                min_p = window if self.strict else 1
                return series.rolling(window=window, min_periods=min_p).mean()
            else:
                ema = series.ewm(span=window, adjust=False).mean()
                if self.strict:
                    counts = series.expanding(min_periods=1).count()
                    ema = ema.where(counts >= window, np.nan)
                return ema

        # ---- 斜率函数（仅用于周均线）----
        def _slope_diff(series: pd.Series, steps: int) -> pd.Series:
            """用 steps 根（≈ slope_weeks 周）差分近似斜率；返回每根K线的平均变化量"""
            if steps <= 0:
                return pd.Series(np.nan, index=series.index)
            return (series - series.shift(steps)) / steps

        # [优化] 向量化线性回归斜率计算 (Double Cumsum)
        def _slope_reg_vectorized(series: pd.Series, steps: int) -> pd.Series:
            if steps <= 1: return pd.Series(np.nan, index=series.index)
            
            n = float(steps)
            x_mean = (n - 1) / 2.0
            var_x = (n * (n**2 - 1)) / 12.0
            
            y_filled = series.fillna(0)
            s1 = y_filled.cumsum()
            s2 = s1.cumsum()
            
            sum_y = s1 - s1.shift(steps)
            shift_s1 = s1.shift(steps)
            shift_s2 = s2.shift(steps)
            
            # 计算 sum(x*y)
            weighted_sum_rev = (s2 - shift_s2) - steps * shift_s1
            sum_xy = (steps * sum_y) - weighted_sum_rev
            
            y_mean = sum_y / n
            slope = (sum_xy - n * x_mean * y_mean) / var_x
            
            if self.strict:
                valid = series.rolling(steps).count() >= steps
                slope = slope.where(valid, np.nan)
            return slope

        # 计算 —— 原生 K 线均线 (新增)
        # 这里直接按根数计算，不随时间周期缩放
        for b in self.bars:
            b = int(b)
            if b <= 0: continue
            ma_b = _ma(close, b)
            ma_col = f"{self.method.upper()}_{b}B"
            out[ma_col] = ma_b

        # 3) 计算 —— 日线均线（新增）
        for d in self.days:
            d = int(d)
            if d <= 0:
                continue
            window_d = max(klines_per_day * d, 1)
            ma_d = _ma(close, window_d)
            if m == 'sma':
                ma_col = f"SMA_{d}D"
            else:
                ma_col = f"EMA_{d}D"
            out[ma_col] = ma_d

        # 4) 计算 —— 周线均线（原有）
        for w in self.weeks:
            w = int(w)
            if w <= 0:
                continue
            window_w = max(klines_per_week * w, 1)
            ma_w = _ma(close, window_w)
            ma_w_col = f"{'SMA' if m=='sma' else 'EMA'}_{w}W"
            out[ma_w_col] = ma_w
            
            # 周线斜率（保持原行为）
            if self.add_slope:
                steps = max(int(round(klines_per_week * float(self.slope_weeks))), 1)

                if self.slope_method == 'diff':
                    slope = _slope_diff(ma_w, steps)
                    slope_col = f"SLOPE_DIFF_{ma_w_col}_{self.slope_weeks}W"
                elif self.slope_method == 'reg':
                    slope = _slope_reg_vectorized(ma_w, steps)
                    slope_col = f"SLOPE_REG_{ma_w_col}_{self.slope_weeks}W"
                else:
                    raise ValueError("slope_method 只能为 'diff' 或 'reg'")
                if self.strict:
                    valid_ma = ma_w.notna()
                    valid_slope = ma_w.rolling(steps).count() >= steps
                    slope = slope.where(valid_ma & valid_slope, np.nan)
                out[slope_col] = slope

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize_z_score(X, feature_cols , self.features , "close", factory= factory)
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

        # 2. 🌟 健壮性计算：直接处理 rs，不要 replace(0, np.nan)
        # 加上 EPS 仅仅是为了防止分母为 0 时的 RuntimeWarning 刷屏，逻辑判断会覆盖它
        rs = avg_gain / (avg_loss + EPS)
        rsi_values = np.where(avg_loss == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))

        # 3. 处理死水行情 (涨跌均为 0)
        rsi_values = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, rsi_values)

        # 4. 🌟 赋值与 Strict 模式裁剪
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

        # --- 🌟 核心改进：处理 RSV 的除零风险 ---
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
            df[f'VOL_ratio_{w}'] = df[f'VOL_ratio_{w}'].clip(upper=50.0)
            
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize_z_score(X, feature_cols , self.ma_features , feature_base = 'volume', factory= factory)
        # self._normalize_z_score(X, feature_cols , self.ma_features_ratio , feature_base = self.ma_features_ratio[0], factory= factory)
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
        # 默认窗口建议包含短、中、长，例如 (5, 20, 60)
        self.vol_ma_windows: list = kwargs.get('vol_ma_windows', (5, 20, 60))
        self.slope_step = kwargs.get('slope_step', 3)  # 斜率步长，越大越迟钝/稳健
        
        self.level_stack_features = [] # 方案 A
        self.gradient_features = []    # 方案 B
        self.ratio_features = []       # 原始比率
        self.vwap_features = ['VWAP_Inside_Ratio', 'VWAP_Close_Dev']

        # 预注册特征名
        if len(self.vol_ma_windows) >= 2:
            # 方案 A: 构造相邻窗口的层级，如 MA5/MA20
            for i in range(len(self.vol_ma_windows) - 1):
                self.level_stack_features.append(f'QAV_Stack_{self.vol_ma_windows[i]}_{self.vol_ma_windows[i+1]}')
        
        for w in self.vol_ma_windows:
            self.gradient_features.append(f'QAV_Grad_{w}')
            self.ratio_features.append(f'QAV_ratio_{w}')
            
        self.features = (self.level_stack_features + self.gradient_features + 
                         self.ratio_features + self.vwap_features)

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        # 1. 计算 VWAP 及其衍生特征
        vwap = np.where(df['volume'] > 0, df['quote_asset_volume'] / df['volume'], df['close'])
        range_hl = df['high'] - df['low']
        df['VWAP_Inside_Ratio'] = np.where(range_hl > EPS, ((vwap - df['low']) / range_hl) - 0.5, 0.0)
        df['VWAP_Close_Dev'] = (vwap / df['close']) - 1

        # 2. 计算各窗口均线 (作为中间变量，不直接存入 self.features)
        ma_series = {}
        for w in self.vol_ma_windows:
            ma_val = df['quote_asset_volume'].rolling(w).mean()
            ma_series[w] = ma_val
            
            # 计算原始比率
            df[f'QAV_ratio_{w}'] = np.where(ma_val > EPS, df['quote_asset_volume'] / ma_val, 0.0)

            # --- 方案 B: 量能偏离度 (不灵敏斜率) ---
            # 使用 shift(slope_step) 跨步计算，捕捉中线趋势而非随机波动
            prev_ma = ma_val.shift(self.slope_step)
            df[f'QAV_Grad_{w}'] = np.where(prev_ma > EPS, (ma_val / prev_ma) - 1, 0.0)

        # --- 方案 A: 量能层级 (Level Stack) ---
        # 描述量能是在收缩还是扩张
        if len(self.vol_ma_windows) >= 2:
            for i in range(len(self.vol_ma_windows) - 1):
                w_short, w_long = self.vol_ma_windows[i], self.vol_ma_windows[i+1]
                col_name = f'QAV_Stack_{w_short}_{w_long}'
                df[col_name] = np.where(ma_series[w_long] > EPS, ma_series[w_short] / ma_series[w_long], 1.0)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        按照 SKSS 原则进行分布保护
        """
        # 1. 方案 A (Level Stack): 它是均线的比值，通常在 1.0 附近波动
        # 使用 SCS 结构化缩放，基准设为 None，即在组内进行 Min-Max
        if self.level_stack_features:
            self._normalize_group_min_max(X, feature_cols, self.level_stack_features, symmetric=False)

        # 2. 方案 B (Gradient) & VWAP_Dev: 它们是变化率，基于 0 对称
        # 使用对称缩放，固定零轴在 0.5
        kinetic_group = self.gradient_features + ['VWAP_Close_Dev']
        self._normalize_group_min_max(X, feature_cols, kinetic_group, symmetric=True)

        # 3. 原始比率 (Ratios): 长尾分布，使用 log1p 压制
        ratio_indices = [self.factory._feature_index[f] for f in self.ratio_features if f in self.factory._feature_index]
        if ratio_indices:
            X[:, :, ratio_indices] = np.log1p(np.maximum(X[:, :, ratio_indices], 0.0))

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
        """
        OBV 的无量纲化：
        使用成交量的均值作为标尺，然后进行对称组缩放。
        """
        target_indices = [self.factory._feature_index['OBV']]
        
        # 1. 获取成交量的均值 mu (M, 1, 1)
        mu_vol, _ = self.factory.get_base_stats('volume')
        mu_vol = mu_vol[:, :, np.newaxis]

        # 2. 无量纲化：OBV / mu_volume
        if np.any(mu_vol == 0):
            raise ValueError("OBV Normalization Error: Volume mean is zero.")
            
        X[:, :, target_indices] = X[:, :, target_indices] / mu_vol

        # 3. 最后进行对称组缩放 [-0.5, 0.5]
        # 这一步会处理 OBV 的长尾并锚定零轴
        self._normalize_group_min_max(X, feature_cols, self.features, symmetric=True)
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
        # 1. 计算原始 PVT
        pct = df['close'].pct_change()
        pvt_raw = (pct * df['volume']).cumsum()
        
        # 2. 无量纲化：除以 Volume
        # 得到：单位成交量下的累积收益贡献
        pvt_unit = np.where(df['volume'] > 0, pvt_raw / df['volume'], 0.0)

        # 3. 长尾压制：对称对数压缩 (Symmetric Log)
        # 这样既压制了 20% 涨幅带来的脉冲，又保留了正负号
        df['PVT'] = np.sign(pvt_unit) * np.log1p(np.abs(pvt_unit))

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # 4. 最后执行对称组缩放，将分布锁定在 [-0.5, 0.5]
        # 此时由于经过了 log1p，数据的有效信息会在这个区间内分布得非常均匀
        self._normalize_group_min_max(X, feature_cols, self.features, symmetric=True)
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
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.vwap_windows:list = kwargs.get('vwap_windows', (20, 48, 96))
        for w in self.vwap_windows:
            self.features.append(f'VWAP_{w}')
    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        # 1. 预计算 PV，避免在循环中重复计算
        pv = df['close'] * df['volume']
        
        for w in self.vwap_windows:
            rolling_vol = df['volume'].rolling(w).sum()
            rolling_pv = pv.rolling(w).sum()
            
            # 2. 识别有效区域：既要有足够的窗口，成交量又要大于 0
            # rolling_vol.notna() 过滤掉冷启动期
            # rolling_vol > 0 过滤掉无成交期
            valid_mask = (rolling_vol > 0) & (rolling_vol.notna())
            
            # 3. 初始化为 NaN (比初始化为 0 更利于后续 ffill 或审计)
            vwap = pd.Series(np.nan, index=df.index)
            
            # 4. 赋值
            vwap[valid_mask] = rolling_pv[valid_mask] / rolling_vol[valid_mask]
            
            # 5. 如果你确实希望无成交的地方是 0 而不是 NaN：
            vwap[rolling_vol == 0] = 0.0
            
            df[f'VWAP_{w}'] = vwap
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize_scs(X, feature_cols, self.features, feature_base='close')
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
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.cmf_window:int = kwargs.get('cmf_window', 20)
        self.features = ['CMF']
    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        # 1. 计算价格区间
        range_hl = df['high'] - df['low']
        
        # 2. 🌟 按照你的理解：若无波动 (一字线)，MFM 赋值为 0
        # 这里的 EPS 只是为了消灭 RuntimeWarning，逻辑会被 np.where 完美覆盖
        mfm = np.where(range_hl == 0, 0.0, 
                       ((df['close'] - df['low']) - (df['high'] - df['close'])) / (range_hl + EPS))
        
        # 3. 计算资金流成交量 (MFV)
        mfv = mfm * df['volume']
        
        # 4. 计算窗口内的累积值
        mfv_sum = mfv.rolling(self.cmf_window).sum()
        vol_sum = df['volume'].rolling(self.cmf_window).sum()
        
        # 5. 🌟 再次防护：如果 20 根线内完全没成交量，结果也赋值为 0
        df['CMF'] = np.where(vol_sum == 0, 0.0, mfv_sum / (vol_sum + EPS))
        
    def _min_history_request(self, kline_interval_ms:int = None) -> int:
        """
        计算 Chaikin Money Flow (CMF) 所需的最小历史 K 线数量。
        逻辑：基于 cmf_window，并增加缓冲区以确保滚动求和逻辑的统计稳定性。
        """
        # 1. 基础窗口需求 (例如默认的 20)
        base_window = self.cmf_window
        
        # 2. 增加 50% 的缓冲区 (Warmup Buffer)
        # 确保在第一个推理窗口之前，指标已有足够的背景样本
        # 例如 20 * 1.5 = 30 根
        return int(base_window * 1.5)

# ==== MFI ====
class FeatureMFI(FeatureBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.mfi_window:int = kwargs.get('mfi_window', 14)
        self.features = ['MFI']
    def generate(self,df:pd.DataFrame, kline_interval_ms:int):
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        pos = np.where(tp > tp.shift(1), mf, 0)
        neg = np.where(tp < tp.shift(1), mf, 0)
        pos_sum = pd.Series(pos).rolling(self.mfi_window).sum()
        neg_sum = pd.Series(neg).rolling(self.mfi_window).sum()
        mfi = np.where(neg_sum == 0, 100.0, 100.0 - (100.0 / (1.0 + pos_sum / (neg_sum))))
        # 3. 补充一个细节：如果上涨和下跌都是 0 (比如停盘或没成交量)，给 50 中性值
        mfi = np.where((pos_sum == 0) & (neg_sum == 0), 50.0, mfi)
        df['MFI'] = mfi
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        MFI 范围固定为 [0, 100]。
        处理方式：直接除以 100，将其映射到 [0, 1] 区间。
        （可选：再减去 0.5 映射到 [-0.5, 0.5] 以实现零均值化）
        """
        # 1. 找到 MFI 列的索引
        target_indices = [feature_cols.index(f) for f in self.features if f in feature_cols]
        
        if not target_indices:
            return

        # 2. 简单缩放
        # MFI / 100.0  -> 范围 [0, 1]
        # (MFI / 100.0) - 0.5 -> 范围 [-0.5, 0.5] (推荐，配合 tanh 激活函数更好)
        X[:, :, target_indices] = (X[:, :, target_indices] / 100.0) - 0.5
    def _min_history_request(self, kline_interval_ms:int = None) -> int:
        """
        计算资金流量指数 (MFI) 所需的最小历史 K 线数量。
        逻辑：基于 mfi_window，增加缓冲区以确保滚动求和逻辑的稳定性。
        """
        # 1. 基础窗口需求 (例如默认的 14)
        base_window = self.mfi_window
        
        # 2. 增加缓冲区 (建议 1.5 倍至 2 倍)
        # 确保在第一个推理窗口之前，数据已足以生成稳定的指标分布
        return int(base_window * 2)
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
        self._normalize_volume_rlc(X, feature_cols, self.features, feature_base='ATS')
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
        
        # 动态注册所有特征
        self.features = []
        for w in self.windows:
            self.features.append(f'vol_event_ratio_{w}')
            self.features.append(f'vol_event_flag_{w}')

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
        pass

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        # 必须返回所有窗口中最大的那个，确保初始化数据足够
        return int(max(self.windows) * 1.2)
    
class FeatureCandle(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 1. 绝对量级类 (Magnitude)
        # 特征：数值随价格高低变化，分布无界
        # 处理：需要 Z-Score (标准化) 或 Robust Scaling
        self.feat_magnitude = ['body', 'upper_wick', 'lower_wick', 'max_range', 'body_mom']
        
        # 2. 比例类 (Ratios) [0, 1]
        # 特征：描述形状，数值在 0 到 1 之间
        # 处理：中心化 (减去 0.5)，使其分布在 [-0.5, 0.5]
        self.feat_ratio = ['body_pct', 'upper_wick_pct', 'lower_wick_pct', 'close_pos', 'doji_score']
        
        # 3. 评分类 (Scores) [-1, 1]
        # 特征：描述倾向性，数值本身就在 0 附近
        # 处理：通常保持原样，或者做简单的 Clip 截断
        self.feat_score = ['wick_bias'] #'gap'
        
        # 汇总所有特征名，供父类使用
        self.features = self.feat_magnitude + self.feat_ratio + self.feat_score

    def generate(self, df: pd.DataFrame, kline_interval_ms: int = None):
        """
        计算基础特征，此时生成的数据包含：
        1. 巨大的绝对值 (如 body=1000)
        2. 0~1 的比率
        3. 小数点后的 Score
        """
        # 避免修改原始 DataFrame
        o, h, l, c = df['open'], df['high'], df['low'], df['close']

        # === A. 绝对量级类 (Magnitude) ===
        # 计算基础物理量
        df['body'] = np.abs(c - o)
        df['upper_wick'] = h - np.maximum(o, c)
        df['lower_wick'] = np.minimum(o, c) - l
        
        # max_range: 全天振幅
        df['max_range'] = (h - l)
        
        # body_mom: 实体的变化动量 (当前实体大小 - 上一根实体大小)
        # 注意：这里 fillna(0) 处理第一根 K 线
        df['body_mom'] = df['body'].diff().fillna(0)

        # === B. 比例类 (Ratios) [0, 1] ===
        # 计算分母，加上 EPS 防止一字板导致除零
        rng = df['max_range'] + EPS
        
        df['body_pct'] = df['body'] / rng
        df['upper_wick_pct'] = df['upper_wick'] / rng
        df['lower_wick_pct'] = df['lower_wick'] / rng
        
        # Close 在 K 线中的相对位置 (0=Low, 1=High)
        df['close_pos'] = (c - l) / rng
        
        # 十字星评分: 实体占比越小，分数越高 (1.0 = 完美十字星)
        df['doji_score'] = 1.0 - df['body_pct']

        # === C. 评分类 (Scores) [-1, 1] ===
        # 影线偏向: >0 上影线长(空头), <0 下影线长(多头)
        df['wick_bias'] = df['upper_wick_pct'] - df['lower_wick_pct']
        
        # Gap: 跳空比例 (Open - Prev_Close) / Prev_Close
        # 注意：使用 shift(1) 获取昨收，fillna 处理第一行
        prev_close = c.shift(1).fillna(o)
        # df['gap'] = (o - prev_close) / (prev_close + EPS)

        # === 清洗 ===
        # 将计算过程中可能产生的 inf 替换为 nan，最后统一填充
        cols_to_clean = self.features
        df[cols_to_clean] = df[cols_to_clean].replace([np.inf, -np.inf], np.nan)
        df[cols_to_clean] = df[cols_to_clean].fillna(0)
        
        return df

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        分层归一化策略执行
        - Magnitude 类: 使用 factory 统一获取价格均值进行无量纲化
        - Ratio 类: 中心化到 [-0.5, 0.5]
        - Score 类: 保持原样
        """
        # 1. Magnitude 类 -> 无量纲化 + 组归一化
        mag_indices = [factory._feature_index[f] for f in self.feat_magnitude if f in feature_cols]
        
        if mag_indices:
            # 使用工厂方法获取 'close' 的均值 (mu)
            # mu 的形状自动被处理为 (Sample, 1, 1)，完美匹配广播要求
            mu_price, _ = factory.get_base_stats('close')
            mu_price = mu_price[:, :, np.newaxis]  # 统一升维，后续计算更简洁
            mag_data = np.where(mu_price > EPS, X[:, :, mag_indices] / mu_price, 0.0)
            
            # Step B: 计算组整体统计量 (跨 Time 和 Feature 维度池化)
            group_mu = np.nanmean(mag_data, axis=(1, 2), keepdims=True)
            group_sigma = np.nanstd(mag_data, axis=(1, 2), keepdims=True)
            
            # Step C: 标准化并写回
            X[:, :, mag_indices] = np.where(
                group_sigma > 0,
                (mag_data - group_mu) / group_sigma,
                0.0
            )

        # 2. Ratio 类 -> Center at 0 (平移)
        # 原始范围 [0, 1] -> 映射到 [-0.5, 0.5]
        ratio_indices = [feature_cols.index(f) for f in self.feat_ratio if f in feature_cols]
        if ratio_indices:
            X[..., ratio_indices] = np.clip(X[..., ratio_indices], 0, 1) - 0.5

        # 3. Score 类 -> Pass (保持原样)
        pass
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
    唐奇安通道 (Donchian Channels):
    Upper Band = Highest High of N periods
    Lower Band = Lowest Low of N periods
    Middle Band = (Upper + Lower) / 2
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.period = kwargs.get('period', 20)
        self.prefix = kwargs.get('prefix', 'DONCHIAN')
        self.features = [
            f"{self.prefix}_UPPER_{self.period}",
            f"{self.prefix}_LOWER_{self.period}",
            f"{self.prefix}_MIDDLE_{self.period}"
        ]

    def generate(self, df: pd.DataFrame, kline_interval_ms: int):
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        # 计算滚动最高与最低
        upper = high.rolling(window=self.period).max()
        lower = low.rolling(window=self.period).min()
        middle = (upper + lower) / 2
        
        df[self.features[0]] = upper
        df[self.features[1]] = lower
        df[self.features[2]] = middle

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        # 使用 SCS 保持价格骨架的一致性，将绝对价格转化为比例关系
        # self._normalize_scs(X, feature_cols, self.features, "close")
        self._normalize_z_score(X, feature_cols , self.features , feature_base = "close", factory= factory)

    def _min_history_request(self, kline_interval_ms: int = None) -> int:
        return self.period

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
        self._normalize_z_score(X, feature_cols , self.features , feature_base = "close", factory= factory)

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
        self._normalize_z_score(X, feature_cols, self.price_features, feature_base="close", factory=factory)
        
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
    
class FeatureOrigin(FeatureBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.price_base_features = ['open', 'high', 'low', 'close']#[]
        self.volume_base_features = ['taker_buy_base_volume', 'volume']#[] # feature used as basic must be the last!!!
        self.quote_base_features  = ['taker_buy_quote_volume', 'quote_asset_volume']#[]   #the basic is quote_asset
        self.self_based_features = ['number_of_trades']#[]
        self.features = self.price_base_features + self.volume_base_features + self.quote_base_features + self.self_based_features
        # for f in ['open', 'high', 'low', 'close']:
        #     self.price_base_features.append(f'base_{f}')
        # for f in ['taker_buy_base_volume', 'volume']:
        #     self.volume_base_features.append(f'base_{f}')
        # for f in ['taker_buy_quote_volume', 'quote_asset_volume']:
        #     self.quote_base_features.append(f'base_{f}')
        # for f in ['number_of_trades']:
        #     self.self_based_features.append(f'base_{f}')
    def generate(self,df:pd.DataFrame, kline_interval_ms: int = None):
        # for f in self.factory.base_features:
        #     df[f'base_{f}'] = df[f]
        pass
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize_z_score(X, feature_cols , self.price_base_features , feature_base = "close", factory= factory)
        self._normalize_z_score(X, feature_cols , self.volume_base_features , feature_base = "volume", factory= factory)
        self._normalize_z_score(X, feature_cols , self.quote_base_features , feature_base = "quote_asset_volume", factory= factory)
        for f in self.self_based_features:
            self._normalize_z_score(X, feature_cols , [f] , feature_base = f, factory= factory)
    # def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
    #     self._normalize_group_min_max(X, feature_cols , self.price_base_features)
    #     self._normalize_volume_rlc(X, feature_cols , self.volume_base_features , feature_base = "volume")
    #     self._normalize_volume_rlc(X, feature_cols , self.quote_base_features , feature_base = "quote_asset_volume")
    #     for f in self.self_based_features:
    #         self._normalize_volume_rlc(X, feature_cols , [f] , feature_base = f)
    def _min_history_request(self, kline_interval_ms:int = None) -> int:
        return 1

#均线 动量 结构
#01000001110001111  过拟合风险
#01000010000010011
FEATURE_CONFIG_LIST = [
    # 1. 自定义的成交量爆发特征 (窗口 512，对比前 2 强)
    (FeatureVolumeEvent, {"windows": [5000, 1500], "top_k": 3}),
    # 2. 价格趋势与指标类
    (FeatureMACD, {"fast": 12, "slow": 26, "signal": 9}),
    (FeatureMA, {  "weeks": [7, 25],  "days": [5, 10, 20],  "bars": [],  "method": 'sma',  "strict": True,  "add_slope": False }),
    (FeatureRsi, {"period": 14, "price_col": 'close', "strict": True, "prefix": 'RSI'}),
    (FeatureKdj, {"n": 9, "m1": 3, "m2": 3, "strict": True, "prefix": 'KDJ'}),
    # #价格通道类，2选1
    (FeatureDonchian, {"period": 20}),
    (FeatureKeltner, {"period": 20, "multiplier":2 }),
    (FeatureBoll, {"period": 20}),

    # # 3. 量能与成交活跃度类
    (FeatureVolMa, {"vol_ma_windows": (5, 10, 20)}),
    (FeatureQavMa, {"vol_ma_windows": (5, 10, 20)}),  #tired,rview this later
    (FeatureOBV, {}),
    (FeaturePVT, {}),
    (FeatureWAP, {"vwap_windows": (20, 48, 96)}),
    (FeatureCFM, {"cmf_window": 20}),
    (FeatureMFI, {"mfi_window": 14}),
    (FeatureATS, {}),
    # 4. K线形态类
    (FeatureCandle, {}),
    (FeatureOrigin, {}),
]

class FeatureFactory:
    def __init__(self, feature_conf_list:list[dict[FeatureBase:[]]], kline_interval_ms:int):
        self.all_feature_list = []  #feature names
        self.feature_map :dict[FeatureBase:[str]]= {}
        self.price_features = {}
        self._kline_interval_ms = kline_interval_ms
        self._X = None
        self._feature_index = None
        self._base_stats_pool = None
        self.feature_list :list[FeatureBase] = []
        self.base_features= ['open', 'high', 'low', 'close', 'taker_buy_base_volume', 'volume','taker_buy_quote_volume', 'quote_asset_volume','number_of_trades']
        for cls, params in feature_conf_list:
            # 使用 **params 将字典解包为关键字参数传递给构造函数
            instance = cls(factory = self,kline_interval_ms=kline_interval_ms, **params) 
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

    def _normalize_price_features(self):
        pass

    def normalize(self, X: np.ndarray, feature_cols: list[str]):
        self._prepare_normalize_context(X, feature_cols)
        for f in self.feature_list:
            f.normalize(X, feature_cols, self)
        self._normalize_price_features()

    def get_global_min_history(self) -> int:
        """遍历所有已注册特征，返回其中最大的历史需求"""
        return max([f.min_history_request(self._kline_interval_ms) for f in self.feature_list])