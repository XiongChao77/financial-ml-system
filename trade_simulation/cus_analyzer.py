import numpy as np
import backtrader as bt

class CusAnalyzer(bt.Analyzer):
    """
    极简版分析器：仅专注于计算持仓暴露（Position Exposure）统计。
    持仓比例 = 所有数据|头寸市值|之和 / 当时账户净值
    """
    def start(self):
        # 存储每一步的持仓比例
        self._ratios = []
        # 记录最大持仓比例
        self._max_ratio = 0.0

    def _gross_exposure_value(self):
        """计算所有持仓的市值总和（总资产暴露）"""
        expv = 0.0
        # 遍历所有数据源（即所有交易的品种）
        for d in self.strategy.datas:
            pos = self.strategy.getposition(d)
            # 如果有头寸（多头或空头）
            if pos.size:
                # 累加：头寸数量的绝对值 * 当前价格
                expv += abs(pos.size) * float(d.close[0])
        return expv

    def next(self):
        """在每个 Bar 结束时调用"""
        equity = float(self.strategy.broker.getvalue())
        
        # 只有账户净值大于零时才计算比例
        if equity > 0:
            ratio = self._gross_exposure_value() / equity
            self._ratios.append(ratio)
            self._max_ratio = max(self._max_ratio, ratio)

    def stop(self):
        """回测结束时调用，执行最终计算"""
        
        # 暴露分布统计
        if self._ratios:
            arr = np.asarray(self._ratios, dtype=float)
            
            # 确保统计值是标准的 Python float 类型
            avg_pos_ratio = float(arr.mean())
            std_pos_ratio = float(arr.std(ddof=0)) # 使用 ddof=0 确保计算的是总体标准差
            p95_pos_ratio = float(np.quantile(arr, 0.95))
            max_pos_ratio = float(self._max_ratio)
        else:
            # 无交易或无持仓，所有比例设为 0
            avg_pos_ratio = std_pos_ratio = p95_pos_ratio = max_pos_ratio = 0.0

        # 设置最终输出
        self.rets = {
            'avg_pos_ratio': avg_pos_ratio,
            'std_pos_ratio': std_pos_ratio,
            'p95_pos_ratio': p95_pos_ratio,
            'max_pos_ratio': max_pos_ratio,
        }
        
    def get_analysis(self):
        """暴露给外部的最终分析结果"""
        return self.rets