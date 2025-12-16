import time
import logging
import json
import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import MetaTrader5 as mt5

# 添加项目路径以导入自定义模块
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

# 引入自定义模块
from data_process import common
from data_process.feature import FeatureFactory, FEATURE_CONFIG
from trade_simulation import model_loader
from trade_simulation.strategy.ftmo_strategy import FtmoBrain, MarketState, PositionDir, ActionType, Signal

# ============================================================
# 配置区域
# ============================================================
class LiveConfig:
    # 交易品种映射
    SYMBOL_BINANCE = "BTCUSDT"  # 数据源品种
    SYMBOL_FTMO = "BTCUSD"      # 交易执行品种 (FTMO通常是 BTCUSD)
    
    # 时间周期 (分钟)
    TIMEFRAME = 15
    
    # 每次请求的历史K线数量 (必须满足特征计算的窗口需求，例如 200+)
    LOOKBACK_BARS = 300 
    
    # 策略参数 (需与 bt_ftmo.py 保持一致)
    HOLD_BAR = 1
    MAX_LAYERS = 1
    TRADE_RISK = 0.98   # 每次仓位占比 (0.98 表示 98% 仓位, FTMO杠杆低需注意)
    THRESH = 0.40       # 置信度阈值
    ALLOW_LONG = True
    ALLOW_SHORT = True
    
    # MT5 魔法数字
    MAGIC_NUMBER = 888888
    
    # 轮询间隔 (秒)
    POLL_INTERVAL = 5 

# ============================================================
# 1. 数据源：Binance Data Feed
# ============================================================
class BinanceDataFeed:
    """
    负责从 Binance 获取 K 线数据并清洗为模型所需的格式
    """
    BASE_URL = "https://api.binance.com/api/v3/klines"
    
    def __init__(self, symbol, interval_minutes):
        self.symbol = symbol
        self.interval = f"{interval_minutes}m"
        self.logger = logging.getLogger("BinanceFeed")
    
    def fetch_ohlcv(self, limit=200):
        try:
            params = {
                "symbol": self.symbol,
                "interval": self.interval,
                "limit": limit
            }
            # 获取数据
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            data = response.json()
            
            if not isinstance(data, list):
                self.logger.error(f"Binance API Error: {data}")
                return None

            # 转换为 DataFrame
            # Binance API 返回结构: 
            # [Open time, Open, High, Low, Close, Volume, Close time, Quote asset vol, Number of trades, Taker buy base vol, Taker buy quote vol, Ignore]
            cols = [
                "open_time_ms_utc", "open", "high", "low", "close", "volume", 
                "close_time_ms", "quote_asset_volume", "number_of_trades", 
                "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
            ]
            
            df = pd.DataFrame(data, columns=cols)
            
            # 类型转换
            numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume", 
                           "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # 时间处理 (转换为 UTC)
            df["open_time_date_utc"] = pd.to_datetime(df["open_time_ms_utc"], unit='ms', utc=True)
            df["close_time_ms_utc"] = pd.to_datetime(df["close_time_ms"], unit='ms', utc=True)
            
            # 移除未闭合的最新一根K线 (实盘决策通常基于已完成的K线)
            # Binance 返回的最后一根通常是正在进行的
            current_time_ms = int(time.time() * 1000)
            last_close_time = df.iloc[-1]["close_time_ms"]
            
            if last_close_time > current_time_ms:
                # 如果最后一根K线还没走完，去掉它
                df = df.iloc[:-1].reset_index(drop=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data: {e}")
            return None

# ============================================================
# 2. 执行器：MT5 Executor
# ============================================================
class MT5Executor:
    """
    负责与 MT5 交互：查询持仓、执行订单
    """
    def __init__(self, symbol, magic_number):
        self.symbol = symbol
        self.magic = magic_number
        self.logger = logging.getLogger("MT5Executor")
        
        if not mt5.initialize():
            self.logger.critical("MT5 Initialize Failed!")
            raise RuntimeError("MT5 Init Failed")
            
        self.logger.info(f"Connected to MT5. Account: {mt5.account_info().login}")

    def get_current_state(self):
        """
        获取当前策略在 MT5 上的持仓状态
        返回: (PositionDir, Layers, Lots)
        """
        positions = mt5.positions_get(symbol=self.symbol)
        
        # 筛选属于本策略的持仓 (Magic Number)
        my_pos = [p for p in positions if p.magic == self.magic]
        
        if not my_pos:
            return PositionDir.FLAT, 0, 0.0
        
        # 假设 FTMO 是 Netting 账户或我们会自行处理合并，这里简单取第一个持仓
        # 如果有多个持仓，需要聚合计算
        total_vol = sum(p.volume for p in my_pos)
        direction = PositionDir.LONG if my_pos[0].type == mt5.ORDER_TYPE_BUY else PositionDir.SHORT
        
        # 估算当前层数 (这里简化处理，假设每次固定手数或基于比例反推)
        # 实盘中层数维护比较复杂，这里简单将 有持仓=1层 (如果你的策略是固定每层比例)
        # 或者你需要把层数持久化到本地文件
        layers = 1 if total_vol > 0 else 0
        
        return direction, layers, total_vol

    def execute_order(self, action_type: ActionType, target_dir: PositionDir, target_pct: float):
        """
        执行交易指令
        """
        # 1. 获取账户信息计算手数
        account = mt5.account_info()
        if not account:
            self.logger.error("Could not get account info")
            return
            
        balance = account.balance
        
        # 2. 获取当前价格
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            self.logger.error(f"Symbol {self.symbol} not found")
            return
            
        # 3. 计算目标手数 (FTMO BTCUSD Contract Size = 1)
        # Target Value = Balance * target_pct
        # Lots = Target Value / Price
        if target_pct is None: target_pct = 0.0
        
        target_value = balance * abs(target_pct)
        # 这里的 price 用 mid price 估算
        price_est = (tick.bid + tick.ask) / 2
        
        target_lots = round(target_value / price_est, 2)
        
        # 最小手数检查 (通常是 0.01)
        if target_lots < 0.01: target_lots = 0.01
        
        self.logger.info(f"Action: {action_type} | Dir: {target_dir} | Pct: {target_pct:.2f} | Calc Lots: {target_lots}")

        # 4. 执行逻辑分支
        if action_type == ActionType.CLOSE:
            self.close_all()
            
        elif action_type in [ActionType.OPEN, ActionType.REVERSE, ActionType.PYRAMID]:
            # 对于反手，先平仓再开仓 (简单稳健的做法)
            if action_type == ActionType.REVERSE:
                self.close_all()
                time.sleep(1) # 等待成交
            
            # 开新仓 / 加仓
            is_buy = (target_dir == PositionDir.LONG)
            order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
            price = tick.ask if is_buy else tick.bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": target_lots,
                "type": order_type,
                "price": price,
                "deviation": 200, # 滑点允许
                "magic": self.magic,
                "comment": f"AutoTrade {action_type.value}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"Order Failed: {result.comment} ({result.retcode})")
            else:
                self.logger.info(f"Order Executed: {result.volume} lots @ {result.price}")

    def close_all(self):
        """平掉所有属于本策略的持仓"""
        positions = mt5.positions_get(symbol=self.symbol)
        my_pos = [p for p in positions if p.magic == self.magic]
        
        for pos in my_pos:
            tick = mt5.symbol_info_tick(self.symbol)
            type_close = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price_close = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": pos.volume,
                "type": type_close,
                "position": pos.ticket,
                "price": price_close,
                "magic": self.magic,
                "comment": "Close All"
            }
            mt5.order_send(request)
            self.logger.info(f"Closed position {pos.ticket}")

# ============================================================
# 3. 主控程序：LiveBot
# ============================================================
class LiveBot:
    def __init__(self):
        self.logger = common.setup_logger(log_name='ftmo_live', log_path=os.path.join(common.TEMPORARY_DIR, 'live.log'))
        
        self.logger.info("Initializing Live Bot...")
        
        # 1. 组件初始化
        self.data_feed = BinanceDataFeed(LiveConfig.SYMBOL_BINANCE, LiveConfig.TIMEFRAME)
        self.executor = MT5Executor(LiveConfig.SYMBOL_FTMO, LiveConfig.MAGIC_NUMBER)
        self.model_handler = model_loader.ModelHandler() # 自动加载训练好的模型
        
        # 2. 策略大脑初始化
        # 注意：这里传入 trade_risk (比如 0.98), Brain 会用它来计算 target_pct
        self.brain = FtmoBrain(
            trade_risk=LiveConfig.TRADE_RISK, 
            max_layers=LiveConfig.MAX_LAYERS,
            holdbar=LiveConfig.HOLD_BAR,
            allow_long=LiveConfig.ALLOW_LONG,
            allow_short=LiveConfig.ALLOW_SHORT,
            thresh=LiveConfig.THRESH
        )
        
        self.last_candle_time = None
        self.logger.info("Bot Initialized. Waiting for candles...")

    def run_step(self):
        """
        单次执行逻辑
        """
        # 1. 获取 Binance 数据
        df = self.data_feed.fetch_ohlcv(limit=LiveConfig.LOOKBACK_BARS)
        if df is None or df.empty:
            self.logger.warning("Empty dataframe from Binance")
            return

        # 检查是否是新 K 线
        current_candle_time = df.iloc[-1]["open_time_date_utc"]
        if self.last_candle_time == current_candle_time:
            return # 还没收盘，或者是同一根K线
        
        self.logger.info(f"New Candle Detected: {current_candle_time}")
        
        # 2. 特征工程 & 模型预测
        try:
            # A. 特征计算 (FeatureFactory)
            # 必须先重命名 ignore 列以匹配 common.attach_attr 的逻辑
            if 'ignore' in df.columns: df.rename({'ignore':'label'}, axis=1, inplace=True)
            
            # 使用 FeatureFactory 生成特征
            FeatureFactory(FEATURE_CONFIG).generate(df)
            
            # B. 计算动态阈值 (复用 common.py 逻辑)
            # 这会生成 threshold 和 stop_threshold 列
            df = common.calculate_thresholds(df)
            
            # C. 模型推理
            # ModelHandler 内部会进行 TimeSeriesWindowDataset 处理和归一化
            # 注意：predict 返回的是包含 pred 和 pred_prob 的 DataFrame
            df_pred, _ = self.model_handler.predict(df)
            
            # 获取最新一根 K 线的预测结果
            last_row = df_pred.iloc[-1]
            pred_signal = last_row["pred"]
            pred_prob = last_row["pred_prob"]
            current_price = last_row["close"]
            
            self.logger.info(f"Predict: Signal={pred_signal}, Prob={pred_prob:.4f}, Price={current_price}")
            
        except Exception as e:
            self.logger.error(f"Prediction Pipeline Error: {e}")
            import traceback
            traceback.print_exc()
            return

        # 3. 获取 MT5 当前状态
        curr_dir, curr_layers, curr_vol = self.executor.get_current_state()
        self.logger.info(f"MT5 State: Dir={curr_dir}, Layers={curr_layers}, Vol={curr_vol}")

        # 4. 构建 MarketState
        state = MarketState(
            price=current_price,
            signal=Signal(int(pred_signal)), # 转换为 Enum
            pred_prob=float(pred_prob),
            position_dir=curr_dir,
            layers=curr_layers
        )

        # 5. Brain 决策
        action = self.brain.decide(state)
        
        if action.action == ActionType.HOLD:
            self.logger.info("Decision: HOLD")
        else:
            self.logger.info(f"Decision: {action.action} -> TargetDir: {action.target_dir}")
            
            # 6. 执行交易
            self.executor.execute_order(
                action_type=action.action,
                target_dir=action.target_dir,
                target_pct=action.target_pct
            )
        
        # 更新时间戳，防止重复执行
        self.last_candle_time = current_candle_time

    def start(self):
        while True:
            try:
                self.run_step()
                time.sleep(LiveConfig.POLL_INTERVAL)
            except KeyboardInterrupt:
                self.logger.info("Bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Main Loop Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = LiveBot()
    bot.start()