import backtrader as bt

class CommInfo_Cryptocurrency(bt.CommInfoBase):
    params = (
        ('leverage', 1.0),    # [key parameter] default leverage
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
        [core logic]
        Backtrader calls this on every open to ask "how much capital is required".
        margin = price / leverage
        """
        return price / self.p.leverage