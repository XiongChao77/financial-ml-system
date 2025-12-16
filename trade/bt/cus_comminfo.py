import backtrader as bt

class CommInfo_Cryptocurrency(bt.CommInfoBase):
    params = (
        ('leverage', 1.0),    # 【关键参数】默认杠杆倍数
    )

    # def __init__(self, **kwargs):
    #     super(CommInfo_Cryptocurrency, self).__init__()

    def getsize(self, price, cash):
        '''Returns the needed size to meet a cash operation at a given price'''
        if not self._stocklike:
            return (self.p.leverage * (cash / self.get_margin(price)))

        return (self.p.leverage * (cash / price))
    
    def get_margin(self, price):
        """
        【核心逻辑】
        Backtrader 每次开仓都会调用这个函数查询“需要多少本金”。
        保证金 = 价格 / 杠杆
        """
        return price / self.p.leverage