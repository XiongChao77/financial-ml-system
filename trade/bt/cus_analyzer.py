import numpy as np
import backtrader as bt
import math

class CusAnalyzer(bt.Analyzer):
    """
    综合风险分析器 (High Cohesion Version)
    职责：
    1. 监控持仓暴露 (Position Exposure)
    2. 监控日内最大回撤 (Daily Max Drawdown - FTMO Standard)
    3. 空仓分布 (Flat/No-position Distribution)  <-- 新增
    """
    def start(self):
        # --- 1. 持仓暴露相关状态 ---
        self._ratios = []
        self._max_exposure = 0.0

        # --- 2. 日内回撤相关状态 ---
        self._daily_stats = []
        self._curr_date = None
        self._day_start_equity = self.strategy.broker.getvalue()
        self._day_min_equity = self._day_start_equity

        # --- 3. 全局最低净值相关状态 ---
        self._global_min_equity = self.strategy.broker.getvalue()

        # --- FLAT DIST: 空仓分布状态（新增）---
        self._flat_start_date = None        # 连续空仓段起始日期
        self._flat_periods_days = []        # 每段空仓长度（单位：天）
        self._flat_bucket_edges = [1, 3, 7, 14, 30]  # 桶边界：0-1,2-3,4-7,8-14,15-30,>30

    def next(self):
        """每个 Bar 结束时调用，分发逻辑"""
        self._track_exposure()
        self._track_daily_drawdown()
        self._track_global_min()

        # --- FLAT DIST: 空仓分布跟踪（新增）---
        self._track_flat_distribution()

    def stop(self):
        """回测结束，执行最终统计汇总"""
        self._record_day(self._curr_date)

        exposure_metrics = self._finalize_exposure()
        drawdown_metrics = self._finalize_drawdown()

        global_metrics = {'global_min_equity': self._global_min_equity}

        # --- FLAT DIST: 回测结束时收尾（新增）---
        flat_metrics = self._finalize_flat_distribution()

        self.rets = {**exposure_metrics, **drawdown_metrics, **global_metrics, **flat_metrics}

    def get_analysis(self):
        return self.rets

    # =========================================================
    # 全局最低净值
    # =========================================================
    def _track_global_min(self):
        current_equity = self.strategy.broker.getvalue()
        if current_equity < self._global_min_equity:
            self._global_min_equity = current_equity

    # =========================================================
    # 持仓暴露
    # =========================================================
    def _track_exposure(self):
        equity = float(self.strategy.broker.getvalue())
        if equity <= 0:
            return

        gross_value = 0.0
        for d in self.strategy.datas:
            pos = self.strategy.getposition(d)
            if pos.size:
                gross_value += abs(pos.size) * float(d.close[0])

        ratio = gross_value / equity
        self._ratios.append(ratio)
        self._max_exposure = max(self._max_exposure, ratio)

    def _finalize_exposure(self):
        if not self._ratios:
            return {'avg_pos_ratio': 0.0, 'std_pos_ratio': 0.0, 'p95_pos_ratio': 0.0, 'max_pos_ratio': 0.0}

        arr = np.asarray(self._ratios, dtype=float)
        return {
            'avg_pos_ratio': float(arr.mean()),
            'std_pos_ratio': float(arr.std(ddof=0)),
            'p95_pos_ratio': float(np.quantile(arr, 0.95)),
            'max_pos_ratio': float(self._max_exposure),
        }

    # =========================================================
    # 日内回撤
    # =========================================================
    def _track_daily_drawdown(self):
        dt = self.strategy.data.datetime.date(0)
        current_equity = self.strategy.broker.getvalue()

        if self._curr_date is None:
            self._curr_date = dt

        if self._curr_date != dt:
            self._record_day(self._curr_date)
            self._curr_date = dt
            self._day_start_equity = current_equity
            self._day_min_equity = current_equity
        else:
            if current_equity < self._day_min_equity:
                self._day_min_equity = current_equity

    def _record_day(self, date_obj):
        if date_obj is None:
            return

        if self._day_start_equity > 0:
            dd_pct = (self._day_min_equity - self._day_start_equity) / self._day_start_equity
        else:
            dd_pct = 0.0

        self._daily_stats.append({'date': str(date_obj), 'dd_pct': dd_pct})

    def _finalize_drawdown(self):
        if not self._daily_stats:
            return {
                'max_daily_dd': 0.0,
                'max_daily_dd_date': None,
                'daily_dd_violation_days': 0
            }

        worst_day = min(self._daily_stats, key=lambda x: x['dd_pct'])
        violation_count_4 = sum(1 for x in self._daily_stats if x['dd_pct'] < -0.04)
        violation_count_5 = sum(1 for x in self._daily_stats if x['dd_pct'] < -0.049)
        violation_count_3 = sum(1 for x in self._daily_stats if x['dd_pct'] < -0.029)
        all_drawdowns = [x['dd_pct'] for x in self._daily_stats]

        return {
            'max_daily_dd': worst_day['dd_pct'],
            'max_daily_dd_date': worst_day['date'],
            'daily_dd_violation_days': violation_count_4,
            'daily_dd_max_violation_days': violation_count_5,
            'daily_dd_max_3_violation_days': violation_count_3,
            'daily_returns_list': all_drawdowns,
        }

    # =========================================================
    # FLAT DIST: 空仓分布（新增）
    # =========================================================
    def _has_any_position(self) -> bool:
        """多数据源兼容：任意 data 有持仓就算非空仓"""
        for d in self.strategy.datas:
            pos = self.strategy.getposition(d)
            if pos.size:
                return True
        return False

    def _track_flat_distribution(self):
        dt = self.strategy.data.datetime.date(0)
        has_pos = self._has_any_position()

        if not has_pos:
            if self._flat_start_date is None:
                self._flat_start_date = dt
        else:
            if self._flat_start_date is not None:
                # 这里按“空仓起始日到开仓日之前”的天数来算
                days = (dt - self._flat_start_date).days
                self._flat_periods_days.append(int(days))
                self._flat_start_date = None

    def _finalize_flat_distribution(self):
        # stop 时如果还在空仓，把最后一段也记上（包含最后一天，所以 +1）
        if self._flat_start_date is not None:
            dt_end = self.strategy.data.datetime.date(0)
            days = (dt_end - self._flat_start_date).days + 1
            self._flat_periods_days.append(int(days))
            self._flat_start_date = None

        if not self._flat_periods_days:
            return {
                'flat_count': 0,
                'flat_max_days': 0,
                'flat_mean_days': 0.0,
                'flat_p50': 0, 'flat_p75': 0, 'flat_p90': 0, 'flat_p95': 0, 'flat_p99': 0,
                'flat_tail_mean_10pct': 0.0,
                'flat_tail_mean_5pct': 0.0,
                'flat_ge_7d': 0, 'flat_ge_14d': 0, 'flat_ge_30d': 0,
                'flat_bucket_pct': {},
                'flat_periods_raw': [],
            }

        arr = np.asarray(self._flat_periods_days, dtype=float)

        # 分位数
        p50 = int(np.percentile(arr, 50))
        p75 = int(np.percentile(arr, 75))
        p90 = int(np.percentile(arr, 90))
        p95 = int(np.percentile(arr, 95))
        p99 = int(np.percentile(arr, 99))

        # 尾部均值（抗异常值，比 max 更稳）
        q90 = np.percentile(arr, 90)
        q95 = np.percentile(arr, 95)
        tail10 = float(arr[arr >= q90].mean()) if (arr >= q90).any() else 0.0
        tail5  = float(arr[arr >= q95].mean()) if (arr >= q95).any() else 0.0

        # 桶分布：0-1, 2-3, 4-7, 8-14, 15-30, >30
        # 注意：这里用“段长度 days”来分桶
        edges = self._flat_bucket_edges
        labels = ["0-1", "2-3", "4-7", "8-14", "15-30", ">30"]
        counts = [0] * 6
        for v in arr:
            v = int(v)
            if v <= edges[0]:
                counts[0] += 1
            elif v <= edges[1]:
                counts[1] += 1
            elif v <= edges[2]:
                counts[2] += 1
            elif v <= edges[3]:
                counts[3] += 1
            elif v <= edges[4]:
                counts[4] += 1
            else:
                counts[5] += 1
        total = float(len(arr))
        bucket_pct = {labels[i]: float(counts[i] / total) for i in range(6)}

        return {
            'flat_count': int(len(arr)),
            'flat_max_days': int(arr.max()),
            'flat_mean_days': float(arr.mean()),
            'flat_p50': p50, 'flat_p75': p75, 'flat_p90': p90, 'flat_p95': p95, 'flat_p99': p99,
            'flat_tail_mean_10pct': tail10,
            'flat_tail_mean_5pct': tail5,
            'flat_ge_7d': int((arr >= 7).sum()),
            'flat_ge_14d': int((arr >= 14).sum()),
            'flat_ge_30d': int((arr >= 30).sum()),
            'flat_bucket_pct': bucket_pct,
            'flat_periods_raw': [int(x) for x in self._flat_periods_days],
        }

    @property
    def day_start_equity(self):
        return self._day_start_equity
