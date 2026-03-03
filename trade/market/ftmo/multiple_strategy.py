import os,sys,torch,logging,time
import math
from functools import reduce
import pandas as pd
import numpy as np
from multiprocessing import Process, Queue, Manager
from datetime import datetime
from typing import Optional
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..' , '..'))

from data_process import common
from model.train_2head import TrainConfig
from trade.bt.simulation import StrategyPara
from data_process.common import FeatureFactory
from model.data_loader import TimeSeriesWindowDataset
from model import model_loader
from trade.strategy.strategy_ml import FtmoBrain, MarketState, Signal, PositionDir
from trade.market.ftmo import mt5_executor
from trade.market.bybit.bybit_executor import BybitExecutor 
from trade.market.binance_data_feed import BinanceDataFeed

ADDITIONAL_FEATURES = ["atr_14"]

def ms_to_seconds(s):
    return s//1000

def get_base_interval_seconds(intervals: list[int]) -> int:
    """
    输入: ['5m', '15m', '1h']
    输出: 300 (秒)
    """
    seconds_list = [ms_to_seconds(i) for i in intervals]
    # 计算所有秒数的最大公约数 (Greatest Common Divisor)
    gcd_seconds = reduce(math.gcd, seconds_list)
    return gcd_seconds

def sleep_until_next_tick(base_seconds: int):
    """
    精准休眠到下一个 base_seconds 的整倍数时间点
    """
    now = time.time()
    # 计算距离下一个整点还差多少秒
    wait_time = base_seconds - (now % base_seconds)
    
    # 增加 0.5s 缓冲，确保交易所数据已更新
    self.info(f"sleep {wait_time}s from now")
    time.sleep(wait_time + 0.5)
    self.info(f"wake up")

class StrategyType:
    MT5 = "MT5"
    BYBIT = "BYBIT"
  
class StrategyHolder:
    def __init__(self,strategy_hash,strategy_type:str,path:str,tarin_out_path:str,pre_para:common.BaseDefine, train_para:TrainConfig, st_para:StrategyPara):
        self.strategy_hash = strategy_hash
        self.pre_para = pre_para
        self.train_para = train_para
        self.st_para = st_para
        self.type = strategy_type
        self.path = path
        self.tarin_out_path = tarin_out_path
        self.model:Optional[model_loader.ModelHandler] = None
        self.brain:FtmoBrain = None

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class WindowConfig:
    # key 为 window size (int), value 为特征列表
    items: Dict[int, List[str]] = field(default_factory=dict)
    data_feed: Optional[BinanceDataFeed] = None
    factory: Optional[FeatureFactory] = None
    min_bars_needed: Optional[int] = 0
    last_excute_time_s = 0
    last_candle_time = None

    @property
    def max_window(self) -> int:
        """获取当前 interval 下最大的 window size"""
        return max(self.items.keys()) if self.items else 0

@dataclass
class IntervalConfig:
    # key 为 interval (如 '5m'), value 为 WindowConfig 对象
    intervals: Dict[str, WindowConfig] = field(default_factory=dict)

@dataclass
class SymbolRegistry:
    # key 为 symbol (如 'BTCUSDT'), value 为 IntervalConfig 对象
    symbols: Dict[str, IntervalConfig] = field(default_factory=dict)

@dataclass
class TradingConfig:
    # key 为 interval (如 '5m'), value 为 WindowConfig 对象
    trading_type: Dict[str, SymbolRegistry] = field(default_factory=dict)

# ============================================================
# 数据中心与推理机
# ============================================================
class MasterController:
    def __init__(self, strategy_path):
        self.strategy_path = strategy_path
        self.strategies:dict[int,StrategyHolder] = {}
        self.label_col = None
        self.queues = []        # 每个子进程对应的 Queue
        self.strategy_input = TradingConfig()   #symbol:{interval:window:feature_conf_list}
        self.feature_conf_list:list = []  #feature_conf_list
        self.logger, _ = common.setup_session_logger(sub_folder="master", symbol="GLOBAL")
        self.debug = False
        self.init()

    def init(self):
        self._prepare_registry()

    def _prepare_registry(self):
        if not os.path.exists(self.strategy_path):
            raise FileNotFoundError(f"Strategy configuration file not found: {self.strategy_path}")
        self.df_configs = pd.read_csv(os.path.join(self.strategy_path,'strategy.csv'), skipinitialspace=True, comment="#")
        records_file = os.path.join(self.strategy_path, 'selected_configs.jsonl')
        records = common.load_selected_configs(records_file)
        for _, row in self.df_configs.iterrows():
            strategy_hash = str(row.iloc[0])
            strategy_type = str(row.iloc[1])
            config_path   = str(row.iloc[2])
            train_out_path = os.path.join(self.strategy_path,'valid_train_out',strategy_hash)
            if not train_out_path or not os.path.exists(train_out_path):
                raise FileNotFoundError(f"Model output path not found: {train_out_path}")
            record = None
            for r in records:
                if strategy_hash == r['short']['params']['hash']:
                    record = r
                    break
            if record == None:
                raise RuntimeError(f"{strategy_hash} not found in {records_file}")
            params =record["short"]
            pre_para= common.BaseDefine(**params["params"]["common"])
            train_para = TrainConfig(**params["params"]["train"])
            st_para = StrategyPara(**params["params"]["strategy"])
            if strategy_hash in self.strategies:
                raise RuntimeError(f"Duplicate strategy hash detected: {strategy_hash}. Please check CSV for duplicate configurations.")
            self.strategies[strategy_hash]= StrategyHolder(strategy_hash,strategy_type,config_path,train_out_path,pre_para, train_para, st_para)
            self.strategies[strategy_hash].model = model_loader.ModelHandler(tarin_out_path=train_out_path, device='cpu')
            self.label_col = self.strategies[strategy_hash].model.label_col
            self.logger.info(f"load strategy {strategy_hash} {strategy_type}")
        self.logger.info(f"load total {len(self.strategies)} strategies ")
        for hash_value,strategy in self.strategies.items():
            feature_list = strategy.train_para.feature_conf_list
            if feature_list != self.feature_conf_list and self.feature_conf_list:
                raise RuntimeError("Multiple feature combinations are not currently supported.")
            self.feature_conf_list = feature_list
            t_cfg = self.strategy_input.trading_type.setdefault(strategy.pre_para.trading_type, SymbolRegistry())
            s_cfg = t_cfg.symbols.setdefault(strategy.pre_para.symbol, IntervalConfig())
            w_cfg = s_cfg.intervals.setdefault(strategy.pre_para.interval, WindowConfig())
            w_cfg.items[strategy.pre_para.candlestick_num] = self.feature_conf_list
            if strategy.type == StrategyType.MT5:
                executor = mt5_executor.MT5Executor(strategy.path, strategy.pre_para.symbol,int(hash_value, 16), logger=self.logger)
            elif strategy.type == StrategyType.BYBIT:
                executor = BybitExecutor(strategy.path, strategy.pre_para.symbol)
            else:
                raise RuntimeError(f"invalid strategy type :{strategy.type}")
            strategy.brain = FtmoBrain(
                            executor,
                            trade_risk=strategy.st_para.trade_risk,
                            max_layers=1,
                            holdbar=strategy.st_para.holdbar,
                            allow_long=strategy.st_para.allow_long,
                            allow_short=strategy.st_para.allow_short,
                            thresh=strategy.st_para.thresh,
                            stop_loss_long = strategy.st_para.stop_loss_long,
                            stop_loss_short = strategy.st_para.stop_loss_short,
                            atr_sl_mult_long = strategy.st_para.atr_sl_mult_long,
                            atr_sl_mult_short = strategy.st_para.atr_sl_mult_short,
                            max_daily_loss_pct = strategy.st_para.max_daily_loss_pct,
                        )
        
        for trading_type, symbols in self.strategy_input.trading_type.items():
            for symbol, interval_items in symbols.symbols.items():
                for interval, window_items in interval_items.intervals.items():
                    window_items.factory = FeatureFactory(common.get_interval_ms(interval), feature_conf_list=self.feature_conf_list+ADDITIONAL_FEATURES)
                    window_items.min_bars_needed = window_items.factory.get_global_min_history() + max(window_items.items.keys())*2
                    window_items.data_feed = BinanceDataFeed(symbol, interval, trading_type, max_len=window_items.min_bars_needed + 500) #buffer
                    window_items.data_feed.initialize_cache(window_items.min_bars_needed, common.get_interval_ms(interval))
                    initial_df = window_items.data_feed.get_latest_data()
                    window_items.last_candle_time = initial_df.iloc[-1]["open_time_date_utc"] if not initial_df.empty else None
                    self.logger.info(f"History Required: {window_items.min_bars_needed} bars")

    def execute_strategy(self,strategy:StrategyHolder,current_price,pred,pred_prob,atr):
        try:
            curr_dir, curr_layers, curr_vol = strategy.brain.executor.get_current_state()
            state = MarketState(
                price=current_price,
                signal=Signal(int(pred)),
                pred_prob=float(pred_prob),
                position_dir=curr_dir,
                layers=curr_layers,
                current_time=strategy.brain.executor.get_server_time(),
                account_balance=strategy.brain.executor.get_account_equity(),
                atr=atr,
                slow_atr=None, vol_regime=None
            )
            strategy.brain.decide(state)
            self.logger.info(f"🧠 {strategy.strategy_hash} {strategy.pre_para.symbol} Decided: Signal={pred} | Price={current_price}")
        except Exception as e:
            self.logger.error(f"Error in execute strategy {strategy.strategy_hash} {strategy.pre_para.symbol} {strategy.pre_para.interval}: {e}")

    def run_invalid_signal(self,symbol:str,interval_str:str, i_config:WindowConfig):
        for window in i_config.items.keys():
            for hash_value,strategy in self.strategies.items():
                if strategy.pre_para.symbol == symbol and strategy.pre_para.interval == interval_str and strategy.pre_para.candlestick_num == window:
                    self.execute_strategy(strategy,current_price=None,pred=Signal.INVALID,pred_prob=1,atr=None )


    def run_forever(self):
        self.logger.info(f"🚀 Master Controller started. Handling {len(self.strategies)} strategies.")
        # 1. 启动时计算所有 interval 的 GCD
        all_intervals = set()
        for trading_type, symbols in self.strategy_input.trading_type.items():
            for symbol, interval_items in symbols.symbols.items():
                all_intervals.update(interval_items.intervals.keys())
        base_step_s = get_base_interval_seconds([common.get_interval_ms(i) for i in all_intervals])
        self.logger.info(f"⏰ Scheduler started with base step: {base_step_s}s")
        while True:
            # 2. 精准休眠
            if self.debug == False:
                sleep_until_next_tick(base_step_s)
            
            # 3. 醒来后检查哪些 interval 到时了
            now_ts = int(time.time())
            
            # 1. 初始化本次需要执行的策略快照
            activate_strategy = TradingConfig()
            
            for trading_type, symbols in self.strategy_input.trading_type.items():
                for symbol, s_config in symbols.symbols.items():
                    for interval_str, i_config in s_config.intervals.items():
                        interval_seconds = ms_to_seconds(common.get_interval_ms(interval_str))
                        if self.debug == False:
                            interval_seconds = ms_to_seconds(common.get_interval_ms(interval_str))
                        else:
                            interval_seconds = 10
                        if now_ts - i_config.last_excute_time_s > (interval_seconds -1):# (interval_seconds -1):
                            i_config.last_excute_time_s = now_ts
                            if trading_type not in activate_strategy.trading_type:
                                activate_strategy.trading_type[trading_type] = SymbolRegistry()
                            if symbol not in activate_strategy.trading_type[trading_type].symbols:
                                activate_strategy.trading_type[trading_type].symbols[symbol] = IntervalConfig()
                            activate_strategy.trading_type[trading_type].symbols[symbol].intervals[interval_str] = i_config


            for trading_type, symbols in activate_strategy.trading_type.items():
                for symbol, s_config in symbols.symbols.items():
                    for interval_str, i_config in s_config.intervals.items():
                        df = i_config.data_feed.get_latest_data()
                        if df is  None or df.empty:
                            self.logger.warning(f"empty candle data for {symbol} {interval_str}, generate invalid signal ")
                            self.run_invalid_signal(symbol,interval_str,i_config)
                            continue
                        self.logger.info(f"new candle data for {trading_type} {symbol} {interval_str}")
                        try:
                            df_with_feature = i_config.factory.generate(df)
                        except Exception as e:
                            df_with_feature = None
                            self.logger.error(f"Error in factory generate {trading_type} {symbol} {interval_str}: {e}")
                        if df_with_feature is  None or df_with_feature.empty:
                            self.run_invalid_signal(symbol,interval_str,i_config)
                            continue
                        current_candle_time = df_with_feature.iloc[-1]["open_time_date_utc"]
                        if i_config.last_candle_time == current_candle_time:
                            self.logger.info(f"✨ {symbol} {interval_str} no new Candle, lastest is {current_candle_time} | Buffer Size: {len(df)}")
                            if self.debug == False:
                                continue
                        i_config.last_candle_time = current_candle_time
                        self.logger.info(f"✨ {symbol} {interval_str} New Candle Closed: {current_candle_time} | Buffer Size: {len(df)}")
                        for window in i_config.items.keys():                            
                            ds = TimeSeriesWindowDataset(
                                df=df_with_feature, 
                                kline_interval_ms = common.get_interval_ms(interval_str),
                                feature_cols=self.feature_conf_list, 
                                label_col=self.label_col, 
                                window=window,
                                is_live=True,
                            )
                            for hash_value,strategy in self.strategies.items():
                                self.logger.info(f"check {hash_value} {strategy.pre_para.symbol} {strategy.pre_para.interval} {strategy.pre_para.candlestick_num} and {symbol}  {interval_str} {window}")
                                if strategy.pre_para.symbol == symbol and strategy.pre_para.interval == interval_str and strategy.pre_para.candlestick_num == window:
                                    try:
                                        df_pred, model_stats = strategy.model.predict_with_ds(ds,df_with_feature,is_live=True,diff_thresh = None)
                                        last_row = df_pred.iloc[-1]
                                        self.execute_strategy(strategy, last_row["close"], last_row["pred"], last_row["pred_prob"], last_row['atr_14'])
                                        
                                    except Exception as e:
                                        self.logger.error(f"Error in strategy work {strategy.pre_para.symbol}: {e}")

if __name__ == "__main__":
    strategy_path = os.path.join(common.PERSISTENCE_DIR,"market_prepare","strategy_0")
    master = MasterController(strategy_path)
    master.run_forever()