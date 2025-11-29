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
        # 资金曲线
        self.equity_curve = []

        # 回撤曲线
        self.drawdown_curve = []

        # 收益曲线（逐 bar 收益）
        self.return_curve = []

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

        # ---- 资金 ----
        equity = float(self.strategy.broker.getvalue())

        # 资金曲线
        self.equity_curve.append({
            "dt": int(dt.timestamp()),
            "value": equity
        })

        # ---- 收益 ----
        if len(self.equity_curve) > 1:
            prev = self.equity_curve[-2]["value"]
            ret = (equity - prev) / prev if prev != 0 else 0
        else:
            ret = 0.0

        self.return_curve.append({
            "dt": int(dt.timestamp()),
            "ret": ret
        })

        # ---- 回撤 ----
        peak = max([p["value"] for p in self.equity_curve])
        dd = (equity - peak) / peak if peak != 0 else 0

        self.drawdown_curve.append({
            "dt": int(dt.timestamp()),
            "dd": dd
        })

        # ---- 暴露统计 ----
        if equity > 0:
            ratio = self._gross_exposure_value() / equity
            self._ratios.append(ratio)
            self._max_ratio = max(self._max_ratio, ratio)

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

        # ---- 计算 rolling return（30-bar）----
        rr30 = []
        eq = [v["value"] for v in self.equity_curve]
        dt = [v["dt"] for v in self.equity_curve]

        win = 30  # 滚动窗口大小，可根据你的时间周期调整

        for i in range(len(eq)):
            if i < win:
                rr30.append({"dt": dt[i], "rolling": 0})
            else:
                r = (eq[i] - eq[i - win]) / eq[i - win]
                rr30.append({"dt": dt[i], "rolling": r})

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
            # ⭐ 资金曲线
            "equity_curve": self.equity_curve,
            # ⭐ 回撤曲线（每根 bar）
            "drawdown_curve": self.drawdown_curve,
            # ⭐ 收益曲线（逐 bar）
            "return_curve": self.return_curve,
            # ⭐ 滚动收益曲线（30-bar）
            "rolling_return_30": rr30
        }
