from abc import ABC, abstractmethod
import logging,math
import pandas as pd
import numpy as np
import os

EPS = 1e-6 # 防止除以 0

#All features should be based on this
class FeatureBase(ABC):
    def __init__(self, **kwargs): 
        self.params = kwargs
        self.features :list[str]= []
    def _normalize(self, X, feature_cols, target_feature_cols, feature_base, factory):
        target_indices = [factory._feature_index[f] for f in target_feature_cols]
        if not target_indices:
            return
        mu, denom = factory.get_base_stats(feature_base)
        # 广播计算: (M, T, F_sub) - (M, 1, 1) / (M, 1, 1)
        # 注意：mu 和 denom 需要增加一个维度以匹配特征维度 F
        X[:, :, target_indices] = (X[:, :, target_indices] - mu[:, :, np.newaxis]) / denom[:, :, np.newaxis]
    @abstractmethod
    def generate(self,df:pd.DataFrame) -> None: ...
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory) : pass
    @abstractmethod
    def min_history_request(self, kline_interval_ms:int = None) -> int: ...
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
    def generate(self,df:pd.DataFrame):
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
        self._normalize(X, feature_cols , self.features , feature_base = "MACD_DEA", factory = factory)
    def min_history_request(self, kline_interval_ms:int = None) -> int:
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
                # self.features.append(slope_col)

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
    
    def generate(self,df:pd.DataFrame):
        kline_col='open_time_date_utc'
        price_col='close'
        # 1) 标准化时间为 datetime64[ns, UTC]
        tmp = df[[kline_col]].dropna().copy()
        if len(tmp) < 3:
            raise ValueError("数据过少，无法稳定判断K线周期")

        s = tmp[kline_col]
        if np.issubdtype(s.dtype, np.number):
            dt = pd.to_datetime(s.astype('int64'), unit='ms', utc=True)
        else:
            dt = pd.to_datetime(s, utc=True, errors='coerce')
            if dt.isna().any():
                bad_n = int(dt.isna().sum())
                raise ValueError(f"时间列存在无法解析的值（{bad_n} 条），请清洗后再试")

        # 2) 用相邻时间差的中位数估算单K线周期（纳秒）
        dt_sorted = dt.sort_values()
        diffs_ns = np.diff(dt_sorted.astype('int64').to_numpy()).astype(float)  # ns
        kline_ns = np.median(diffs_ns)
        if not np.isfinite(kline_ns) or kline_ns <= 0:
            raise ValueError("检测到非正或无效的K线周期，请检查时间数据")
        
        klines_per_day,klines_per_week = self._calculate_klines_count(kline_ns/1_000_000)

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
            x_arr = np.arange(n)
            var_x = ((x_arr - x_mean) ** 2).sum()
            
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
        self._normalize(X, feature_cols , self.features , "close", factory= factory)
    def min_history_request(self, kline_interval_ms: int) -> int:
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
        self.features :list[str]= []
        self.period:int = kwargs.get('period', 14)
        self.price_col = kwargs.get('price_col', 'close')
        self.strict = kwargs.get('strict', True)# 严格型：窗口未满为 NaN；宽松型：尽早给值
        self.prefix = kwargs.get('prefix', "RSI")
    def generate(self,df:pd.DataFrame):
        """
        Wilder 风格 RSI（使用 EWM, alpha=1/period）
        输出列：{prefix}_{period}
        """

        out = df.copy()
        close = out[self.price_col].astype(float)

        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        # Wilder 平滑：alpha=1/period
        avg_gain = gain.ewm(alpha=1/float(self.period), adjust=False,
                            min_periods=self.period if self.strict else 1).mean()
        avg_loss = loss.ewm(alpha=1/float(self.period), adjust=False,
                            min_periods=self.period if self.strict else 1).mean()

        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100.0 - (100.0 / (1.0 + rs))

        col = f"{self.prefix}_{self.period}"
        out[col] = rsi

        # 严格模式：窗口未满置 NaN
        if self.strict:
            valid = close.expanding().count() >= self.period
            out[col] = out[col].where(valid, np.nan)
        self.features = [col]
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
    def min_history_request(self, kline_interval_ms:int = None) -> int:
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
    def generate(self,df:pd.DataFrame):
        for c in (self.high_col, self.low_col, self.close_col):
            if c not in df.columns:
                raise ValueError(f"缺少列 {c}")

        out = df
        high = out[self.high_col].astype(float)
        low = out[self.low_col].astype(float)
        close = out[self.close_col].astype(float)

        llv = low.rolling(window=self.n, min_periods=self.n if self.strict else 1).min()
        hhv = high.rolling(window=self.n, min_periods=self.n if self.strict else 1).max()

        rsv = (close - llv) / (hhv - llv + 1e-12) * 100.0

        # 用 EWM 实现等价的递推平滑（alpha=1/m）
        K = rsv.ewm(alpha=1/float(self.m1), adjust=False,
                    min_periods=self.m1 if self.strict else 1).mean()
        D = K.ewm(alpha=1/float(self.m2), adjust=False,
                min_periods=self.m2 if self.strict else 1).mean()
        J = 3 * K - 2 * D

        k_col, d_col, j_col = f"{self.prefix}_K", f"{self.prefix}_D", f"{self.prefix}_J"
        out[k_col], out[d_col], out[j_col] = K, D, J

        # 严格模式：必须至少有 n 根形成 RSV，且各自平滑窗口就绪
        if self.strict:
            valid_rsv = close.expanding().count() >= self.n
            out[k_col] = out[k_col].where(valid_rsv, np.nan)
            out[d_col] = out[d_col].where(valid_rsv, np.nan)
            out[j_col] = out[j_col].where(valid_rsv, np.nan)
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        #KDJ 是 0-100 指标，使用简单缩放
        target_indices = [feature_cols.index(f) for f in self.features if f in feature_cols]
        if not target_indices: return

        # 映射到 [-0.5, 0.5]
        X[:, :, target_indices] = (X[:, :, target_indices] / 100.0) - 0.5
    def min_history_request(self, kline_interval_ms:int = None) -> int:
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
        for w in self.vol_ma_windows:
            self.ma_features.append(f'VOL_MA_{w}')
    def generate(self,df):
        # ==== 1. 成交量均线 + 比值 ====
        for w in self.vol_ma_windows:
            vol_ma = df['volume'].rolling(w).mean()
            df[f'VOL_MA_{w}'] = vol_ma
            df[f'VOL_ratio_{w}'] = df['volume'] / (vol_ma.replace(0, np.nan))
            self.features.extend([f'VOL_MA_{w}', f'VOL_ratio_{w}'])
            
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize(X, feature_cols , self.ma_features , feature_base = 'volume', factory= factory)
    def min_history_request(self, kline_interval_ms:int = None) -> int:
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

# ==== 3. 成交额均线 + 比值 ====
class FeatureQavMa(FeatureBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.vol_ma_windows:list = kwargs.get('vol_ma_windows', [])
        self.qa_features = []
        for w in self.vol_ma_windows:
            self.qa_features.append(f'QAV_MA_{w}')
    def generate(self,df):
        # ==== 1. 成交量均线 + 比值 ====
        for w in self.vol_ma_windows:
            qav_ma = df['quote_asset_volume'].rolling(w).mean()
            df[f'QAV_MA_{w}'] = qav_ma
            df[f'QAV_ratio_{w}'] = df['quote_asset_volume'] / (qav_ma.replace(0, np.nan))
            self.features.extend([f'QAV_MA_{w}', f'QAV_ratio_{w}'])
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize(X, feature_cols , self.qa_features , feature_base = 'quote_asset_volume', factory = factory)
    def min_history_request(self, kline_interval_ms:int = None) -> int:
        """
        计算成交额均线 (QAV MA) 所需的最小历史 K 线数量。
        """
        if not self.vol_ma_windows:
            return 0
            
        # 1. 获取所有配置窗口中的最大值 (例如 20)
        max_window = max(self.vol_ma_windows)
        
        # 2. 增加缓冲区 (建议 1.5 倍) 
        # 确保在计算第一个有效特征点时，标准化逻辑已有足够的样本背景
        return int(max_window * 1.5)
# ==== 4. OBV ====
class FeatureOBV(FeatureBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.features = ['OBV']
    def generate(self,df):
        close = df['close']
        sign = np.where(close > close.shift(1), 1,
            np.where(close < close.shift(1), -1, 0))
        df['OBV'] = (sign * df['volume']).cumsum()
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize(X, feature_cols , self.features , feature_base = self.features[0], factory= factory)  #Self-Normalization
    def min_history_request(self, kline_interval_ms:int = None) -> int:
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
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.features = ['PVT']
    def generate(self,df):
        pct = df['close'].pct_change()
        df['PVT'] = (pct * df['volume']).cumsum()
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize(X, feature_cols , self.features , feature_base = self.features[0], factory= factory)  #Self-Normalization
    def min_history_request(self, kline_interval_ms:int = None) -> int:
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
    def generate(self,df):
        for w in self.vwap_windows:
            pv = df['close'] * df['volume']
            vwap = pv.rolling(w).sum() / df['volume'].rolling(w).sum()
            df[f'VWAP_{w}'] = vwap
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize(X, feature_cols , self.features , feature_base = 'close', factory= factory)
    def min_history_request(self, kline_interval_ms:int = None) -> int:
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
class FeatureCFM(FeatureBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.cmf_window:int = kwargs.get('cmf_window', 20)
    def generate(self,df):
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
            (df['high'] - df['low']).replace(0, np.nan)
        mfv = mfm * df['volume']
        df['CMF'] = mfv.rolling(self.cmf_window).sum() / df['volume'].rolling(self.cmf_window).sum()
        self.features = ['CMF']
    def min_history_request(self, kline_interval_ms:int = None) -> int:
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
    def generate(self,df):
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        pos = np.where(tp > tp.shift(1), mf, 0)
        neg = np.where(tp < tp.shift(1), mf, 0)
        pos_sum = pd.Series(pos).rolling(self.mfi_window).sum()
        neg_sum = pd.Series(neg).rolling(self.mfi_window).sum()
        mfi = 100 - (100 / (1 + pos_sum / (neg_sum.replace(0, np.nan))))
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
    def min_history_request(self, kline_interval_ms:int = None) -> int:
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
    def generate(self,df):
        # ==== 2. 平均每笔成交量 ====
        df['ATS'] = df['volume'] / (df['number_of_trades'].replace(0, np.nan))
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        ATS 代表单笔成交力度。
        我们需要捕捉的是它相对于自身历史水平的波动（Z-Score）。
        """
        self._normalize(X=X, feature_cols=feature_cols, target_feature_cols=self.features, feature_base='ATS', factory= factory)  # <--- 必须自缩放
    def min_history_request(self, kline_interval_ms:int = None) -> int:
        """
        计算平均单笔成交量 (ATS) 所需的最小历史 K 线数量。
        虽然指标本身是逐 K 线计算的，但为了使 Z-Score 标准化逻辑稳定，
        建议提供至少与模型观察窗口等长的历史数据。
        """
        # 建议参考模型训练时的 window 大小 (通常在 100 左右)
        model_window = getattr(self, 'window', 100)
        
        # 提供 1 倍窗口作为标准化计算的基础样本
        return int(model_window)
class FeatureCandle(FeatureBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化不同类别的特征列表
        self.feat_magnitude = ['body', 'upper_wick', 'lower_wick', 'range', 'body_mom']  # 绝对值类
        self.feat_ratio = ['body_pct', 'upper_wick_pct', 'lower_wick_pct', 'close_pos', 'doji_score']      # 0~1 比例类
        self.feat_score = ['hammer_score', 'shooting_score', 'wick_bias', 'gap']      # -1~1 评分类
        # 汇总所有特征
        self.features = self.feat_magnitude + self.feat_ratio + self.feat_score

    def generate(self, df: pd.DataFrame):
        o, h, l, c = df['open'], df['high'], df['low'], df['close']

        # --- A. 绝对量级类 (Magnitude) ---
        # 这些特征单位是“价格”，必须去量纲
        df['body'] = np.abs(c - o)
        df['upper_wick'] = h - np.maximum(o, c)
        df['lower_wick'] = np.minimum(o, c) - l
        df['range'] = (h - l).replace(0, np.nan)
        df['body_mom'] = df['body'].diff()  # 实体大小的变化量

        # --- B. 比例类 (Ratios) [0, 1] ---
        # 这些特征描述“形状”，不随价格涨跌改变
        # 分母加上 EPS 防止除零
        rng = df['range'] + EPS
        
        df['body_pct'] = df['body'] / rng
        df['upper_wick_pct'] = df['upper_wick'] / rng
        df['lower_wick_pct'] = df['lower_wick'] / rng
        
        # Close 在 K 线中的相对位置 (0=Low, 1=High) —— 这是一个极强的特征
        df['close_pos'] = (c - l) / rng
        
        # 十字星评分 (实体越小分越高)
        df['doji_score'] = 1.0 - df['body_pct']

        # --- C. 评分类 (Scores) [-1, 1] ---
        # 描述多空倾向
        df['hammer_score'] = df['lower_wick_pct'] - df['upper_wick_pct']  # 正值=锤子(多)
        df['shooting_score'] = df['upper_wick_pct'] - df['lower_wick_pct'] # 正值=流星(空)
        df['wick_bias'] = df['upper_wick_pct'] - df['lower_wick_pct']     # 影线偏向
        
        # Gap (跳空比例)
        # 注意：加密货币 24h 交易 gap 很多时候是 0，但在维护或其他情况会有
        df['gap'] = (o - c.shift(1)) / c.shift(1)

    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        """
        分层归一化策略：
        1. Magnitude: 自缩放 (Z-Score)，消除价格绝对值影响。
        2. Ratio: 减去 0.5，平移到 [-0.5, 0.5]。
        3. Score: 保持原样 (本身就在 -1~1 之间)。
        """
        
        # 1. 处理绝对量级类 (Magnitude) -> Self Z-Score
        # 必须对每个特征单独计算 Z-Score，因为 range 的均值肯定比 upper_wick 大
        # 我们这里循环调用 _normalize 针对每一个单独特征，或者批量自缩放
        # 为了效率，我们稍微修改一下逻辑：分别对每个特征做“自缩放”
        
        for feat in self.feat_magnitude:
            # 这里的 base 就是特征自己
            self._normalize(X, feature_cols, [feat], feature_base=feat, factory= factory)

        # 2. 处理比例类 (Ratios) -> Center at 0
        # 将 [0, 1] 映射到 [-0.5, 0.5]
        ratio_indices = [feature_cols.index(f) for f in self.feat_ratio if f in feature_cols]
        if ratio_indices:
            X[:, :, ratio_indices] = X[:, :, ratio_indices] - 0.5

        # 3. 处理评分类 (Scores) -> Pass
        # 它们已经是 [-1, 1] 且 0 有特殊含义，无需处理
        # 唯一例外是 'gap'，如果它的数值过小 (如 0.0001)，可能需要放大
        # 但通常 Transformer/LSTM 能处理这种小数值，不做处理也行。
    def min_history_request(self, kline_interval_ms:int = None) -> int:
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

class FeatureOrigin(FeatureBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.price_base_features = ['open', 'high', 'low', 'close']
        self.volume_base_features = ['taker_buy_base_volume', 'volume'] # feature used as basic must be the last!!!
        self.quote_base_features  = ['taker_buy_quote_volume', 'quote_asset_volume' ]   #the basic is quote_asset
        self.self_based_features = ['number_of_trades']
    def generate(self,df:pd.DataFrame):     pass
    def normalize(self, X: np.ndarray, feature_cols: list[str], factory):
        self._normalize(X, feature_cols , self.price_base_features , feature_base = "close", factory= factory)
        self._normalize(X, feature_cols , self.volume_base_features , feature_base = "volume", factory= factory)
        self._normalize(X, feature_cols , self.quote_base_features , feature_base = "quote_asset_volume", factory= factory)
        for f in self.self_based_features:
            self._normalize(X, feature_cols , [f] , feature_base = f, factory= factory)
    def min_history_request(self, kline_interval_ms:int = None) -> int:
        return 1