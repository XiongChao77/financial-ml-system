import backtrader as bt

class CommInfo_Cryptocurrency(bt.CommInfoBase):
    params = (
    )

    # def __init__(self, **kwargs):
    #     super(CommInfo_Cryptocurrency, self).__init__()

    def getsize(self, price, cash):
        '''Returns the needed size to meet a cash operation at a given price'''
        if not self._stocklike:
            return (self.p.leverage * (cash / self.get_margin(price)))

        return (self.p.leverage * (cash / price))