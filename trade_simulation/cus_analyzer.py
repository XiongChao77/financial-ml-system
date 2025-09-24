import numpy as np
import backtrader as bt
from backtrader.utils.date import num2date

class CusAnalyzer(bt.Analyzer):
    """
    兼容 exactbars=1 的绩效+暴露分析器
    输出:
      - start_value, end_value, gross_return
      - days, years, cagr
      - avg_pos_ratio, std_pos_ratio, p95_pos_ratio, max_pos_ratio
    说明:
      - 总收益率 = end_value / start_value - 1
      - CAGR = (end_value / start_value) ** (1/years) - 1
      - 持仓比例 = 所有数据|头寸市值|之和 / 当时账户净值
    """
    def start(self):
        # 期初净值（此时通常全现金，无持仓）
        self.start_value = float(self.strategy.broker.getvalue())
        self.end_value = None
        self.start_dt = None
        self.end_dt = None
        self._ratios = []
        self._max_ratio = 0.0

    def _gross_exposure_value(self):
        expv = 0.0
        for d in self.strategy.datas:
            pos = self.strategy.getposition(d)
            if pos.size:
                expv += abs(pos.size) * float(d.close[0])
        return expv

    def next(self):
        # 记录起止时间（只需看第一根与最新一根）
        dt = num2date(self.data.datetime[0])
        if self.start_dt is None:
            self.start_dt = dt
        self.end_dt = dt

        # 记录当期暴露比例
        equity = float(self.strategy.broker.getvalue())
        if equity > 0:
            ratio = self._gross_exposure_value() / equity
            self._ratios.append(ratio)
            if ratio > self._max_ratio:
                self._max_ratio = ratio

        # 不在 next 中取 end_value，避免多次覆盖；stop 时统一取

    def stop(self):
        # 期末净值
        self.end_value = float(self.strategy.broker.getvalue())

        # 持有时长
        if (self.start_dt is not None) and (self.end_dt is not None):
            days = (self.end_dt - self.start_dt).total_seconds() / 86400.0
        else:
            days = 0.0
        years = days / 365.25 if days > 0 else float('nan')

        # 收益
        gross_return = (self.end_value / self.start_value - 1.0) if self.start_value > 0 else float('nan')
        cagr = ((self.end_value / self.start_value) ** (1.0 / years) - 1.0) if (self.start_value > 0 and years and years > 0) else float('nan')

        # 暴露分布
        if self._ratios:
            arr = np.asarray(self._ratios, dtype=float)
            avg_pos_ratio = float(arr.mean())
            std_pos_ratio = float(arr.std(ddof=0))
            p95_pos_ratio = float(np.quantile(arr, 0.95))
            max_pos_ratio = float(self._max_ratio)
        else:
            avg_pos_ratio = std_pos_ratio = p95_pos_ratio = max_pos_ratio = 0.0

        self.rets = {
            'start_value': self.start_value,
            'end_value': self.end_value,
            'gross_return': gross_return,
            'days': days,
            'years': years,
            'cagr': cagr,
            'avg_pos_ratio': avg_pos_ratio,
            'std_pos_ratio': std_pos_ratio,
            'p95_pos_ratio': p95_pos_ratio,
            'max_pos_ratio': max_pos_ratio,
        }
