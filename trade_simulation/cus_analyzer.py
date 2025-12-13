import numpy as np
import backtrader as bt
import math

class CusAnalyzer(bt.Analyzer):
    """
    综合风险分析器 (High Cohesion Version)
    职责：
    1. 监控持仓暴露 (Position Exposure)
    2. 监控日内最大回撤 (Daily Max Drawdown - FTMO Standard)
    """
    def start(self):
        # --- 1. 持仓暴露相关状态 ---
        self._ratios = []
        self._max_exposure = 0.0

        # --- 2. 日内回撤相关状态 ---
        self._daily_stats = []        # 存储每天的记录
        self._curr_date = None        # 当前处理的日期
        self._day_start_equity = self.strategy.broker.getvalue() # 当日开盘净值
        self._day_min_equity = self.day_start_equity             # 当日最低净值
        # --- 3. 全局最低净值相关状态  ---
        self._global_min_equity = self.strategy.broker.getvalue()

    def next(self):
        """每个 Bar 结束时调用，分发逻辑"""
        # 1. 追踪持仓暴露
        self._track_exposure()
        
        # 2. 追踪日内回撤
        self._track_daily_drawdown()

        # 3. 追踪全局最低净值 【新增】
        self._track_global_min()

    def stop(self):
        """回测结束，执行最终统计汇总"""
        # 记录最后一天的回撤数据 (因为 next 不会触发日期变更)
        self._record_day(self._curr_date)
        
        # 计算最终指标
        exposure_metrics = self._finalize_exposure()
        drawdown_metrics = self._finalize_drawdown()

        # 直接获取最低净值
        global_metrics = {
            'global_min_equity': self._global_min_equity
        }

        # 合并结果
        self.rets = {**exposure_metrics, **drawdown_metrics, **global_metrics}

    def get_analysis(self):
        return self.rets

    # =========================================================
    # 私有方法区：全局最低净值逻辑 (Global Min Logic) 【新增】
    # =========================================================
    def _track_global_min(self):
        """简单的追踪逻辑：只记录历史出现过的最低值"""
        current_equity = self.strategy.broker.getvalue()
        if current_equity < self._global_min_equity:
            self._global_min_equity = current_equity

    # =========================================================
    # 私有方法区：持仓暴露逻辑 (Position Exposure Logic)
    # =========================================================
    def _track_exposure(self):
        """计算当前 Bar 的持仓市值占比"""
        equity = float(self.strategy.broker.getvalue())
        if equity <= 0: return

        # 计算总持仓市值
        gross_value = 0.0
        for d in self.strategy.datas:
            pos = self.strategy.getposition(d)
            if pos.size:
                gross_value += abs(pos.size) * float(d.close[0])
        
        ratio = gross_value / equity
        self._ratios.append(ratio)
        self._max_exposure = max(self._max_exposure, ratio)

    def _finalize_exposure(self):
        """计算暴露统计指标"""
        if not self._ratios:
            return {
                'avg_pos_ratio': 0.0, 'std_pos_ratio': 0.0, 
                'p95_pos_ratio': 0.0, 'max_pos_ratio': 0.0
            }
        
        arr = np.asarray(self._ratios, dtype=float)
        return {
            'avg_pos_ratio': float(arr.mean()),
            'std_pos_ratio': float(arr.std(ddof=0)),
            'p95_pos_ratio': float(np.quantile(arr, 0.95)),
            'max_pos_ratio': float(self._max_exposure),
        }

    # =========================================================
    # 私有方法区：日内回撤逻辑 (Daily Drawdown Logic)
    # =========================================================
    def _track_daily_drawdown(self):
        """核心逻辑：检测日期变更，维护当日最低净值"""
        dt = self.strategy.data.datetime.date(0) # 获取当前 K 线日期
        current_equity = self.strategy.broker.getvalue()

        # 初始化第一天
        if self._curr_date is None:
            self._curr_date = dt

        # 检测日期变更 (新的一天开始了)
        if self._curr_date != dt:
            # 1. 结算前一天的回撤
            self._record_day(self._curr_date)
            
            # 2. 重置当天状态
            self._curr_date = dt
            self._day_start_equity = current_equity
            self._day_min_equity = current_equity
        else:
            # 同一天内：更新最低净值
            if current_equity < self._day_min_equity:
                self._day_min_equity = current_equity

    def _record_day(self, date_obj):
        """将单日统计存入列表"""
        if date_obj is None: return

        # 逻辑：(最低 - 开盘) / 开盘
        # 负值代表亏损 (e.g. -0.04)
        if self._day_start_equity > 0:
            dd_pct = (self._day_min_equity - self._day_start_equity) / self._day_start_equity
        else:
            dd_pct = 0.0
            
        self._daily_stats.append({
            'date': str(date_obj),
            'dd_pct': dd_pct
        })

    def _finalize_drawdown(self):
        """找出历史上最严重的单日回撤"""
        if not self._daily_stats:
            return {
                'max_daily_dd': 0.0, 
                'max_daily_dd_date': None,
                'daily_dd_violation_days': 0
            }

        # 找到回撤最大(即数值最小, 因为是负数)的那一天
        # key 设为 dd_pct，以防其他字段干扰
        worst_day = min(self._daily_stats, key=lambda x: x['dd_pct'])
        
        # 统计超过 -4% (FTMO 警戒线) 的天数
        violation_count = sum(1 for x in self._daily_stats if x['dd_pct'] < -0.04)

        return {
            'max_daily_dd': worst_day['dd_pct'], # 例如 -0.05
            'max_daily_dd_date': worst_day['date'],
            'daily_dd_violation_days': violation_count
        }

    @property
    def day_start_equity(self):
        """提供属性访问，方便外部(如策略)在盘中获取今日初始权益，用于熔断判断"""
        return self._day_start_equity