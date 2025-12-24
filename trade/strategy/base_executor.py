from abc import ABC, abstractmethod

class BaseExecutor:
    @abstractmethod
    def user_order_target_percent(self, target_pct: float, stop_loss: float = None):
        pass
    def user_close(self, size=None, **kwargs):
        pass