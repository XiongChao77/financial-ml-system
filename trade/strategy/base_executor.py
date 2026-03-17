from abc import ABC, abstractmethod

class BaseExecutor:
    @abstractmethod
    def get_account_equity(self):
        pass
    @abstractmethod
    def get_current_state(self):
        pass
    @abstractmethod
    def get_server_time(self):
        pass
    @abstractmethod
    def user_close(self, size=None, **kwargs):
        pass
    @abstractmethod
    def user_order(self, size, is_buy, stop_loss=None, take_profit=None):
        pass
    @abstractmethod
    def get_last_position_open_time(self):  #return UTC time
        pass