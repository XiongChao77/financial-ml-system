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
sys.path.append(os.path.join(current_work_dir, "..", '..'))

# 引入自定义模块
from data_process import common
from data_process.common import FeatureFactory, FEATURE_CONFIG
from trade_simulation import model_loader
from trade_simulation.strategy.ftmo_strategy import FtmoBrain, MarketState, PositionDir, ActionType, Signal



pd.set_option("display.max_columns", None)   # 不限制列数
pd.set_option("display.width", None)         # 自动宽度（别强行换行）
pd.set_option("display.max_colwidth", None)  # 单元格内容不截断
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
    TRADE_RISK = 0.01   # 每次仓位占比 (0.98 表示 98% 仓位, FTMO杠杆低需注意)
    THRESH = 0.40       # 置信度阈值
    ALLOW_LONG = True
    ALLOW_SHORT = True
    
    # MT5 魔法数字
    MAGIC_NUMBER = 888888
    
    # 轮询间隔 (秒)
    POLL_INTERVAL = 1

# ============================================================
# 1. 数据源：Binance Data Feed
# ============================================================
class BinanceDataFeed:
    """
    智能数据馈送器：
    1. 内部维护一个 self.local_cache (DataFrame)。
    2. 启动时自动计算时间并拉取全量历史。
    3. 运行时只拉取增量数据并拼接。
    4. 自动修剪过长的数据，保持内存轻量。
    """
    BASE_URL = "https://api.binance.com/api/v3/klines"
    MAX_LIMIT_PER_REQ = 1000
    
    def __init__(self, symbol, interval_minutes, max_len=5000):
        self.symbol = symbol
        self.interval = f"{interval_minutes}m"
        self.logger = logging.getLogger("BinanceFeed")
        
        # 核心：内存中的数据缓存
        self.local_cache = None 
        
        # 内存中最多保留多少根 K 线 (防止无限增长)
        # 只要大于 feature 这里的 required_history 即可
        self.max_cache_len = max_len 

    def _process_data(self, data):
        """[内部工具] 原始 List 转 DataFrame"""
        if not data: return None
        
        cols = [
            "open_time_ms", "open", "high", "low", "close", "volume", 
            "close_time_ms", "quote_asset_volume", "number_of_trades", 
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ]
        df = pd.DataFrame(data, columns=cols)
        
        # 类型转换
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume", 
                       "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # 生成时间字符串 (仅用于日志或调试，计算尽量用 ms)
        df["open_time_date_utc"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)\
                                    .dt.strftime("%Y-%m-%d %H:%M:%S")
        return df

    def _fetch_range_api(self, start_ts, end_ts=None):
        """[内部工具] 纯粹的 API 分页抓取逻辑"""
        if end_ts is None:
            end_ts = int(time.time() * 1000)
            
        all_dfs = []
        curr = start_ts
        
        while True:
            if curr >= end_ts: break
            
            params = {
                "symbol": self.symbol, 
                "interval": self.interval,
                "startTime": curr,
                "endTime": end_ts,
                "limit": self.MAX_LIMIT_PER_REQ
            }
            
            try:
                resp = requests.get(self.BASE_URL, params=params, timeout=5)
                data = resp.json()
            except Exception as e:
                self.logger.error(f"Network error: {e}")
                break

            if not isinstance(data, list) or not data:
                break
                
            df_batch = self._process_data(data)
            all_dfs.append(df_batch)
            
            # 更新游标：最后一根的收盘时间 + 1ms
            last_close = int(data[-1][6])
            curr = last_close + 1
            
            if len(data) < self.MAX_LIMIT_PER_REQ:
                break # 抓完了
            
            time.sleep(0.1)

        if not all_dfs: return None
        return pd.concat(all_dfs, ignore_index=True)

    def initialize_cache(self, required_bars, interval_ms):
        """
        [启动预热] 计算需要的时间，拉取数据并初始化 local_cache
        """
        self.logger.info("Initializing local data cache...")
        
        # 1. 计算回溯时间 (加 1.5 倍 Buff)
        duration_ms = required_bars * interval_ms * 1.2 + 100
        start_time = int(time.time() * 1000) - int(duration_ms)
        
        # 2. 拉取全量
        df = self._fetch_range_api(start_time)
        
        if df is not None and not df.empty:
            # 3. 去重并存入缓存
            df.drop_duplicates("open_time_ms", inplace=True)
            df.sort_values("open_time_ms", inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            self.local_cache = df
            self.logger.info(f"Cache initialized with {len(df)} bars. Last: {df.iloc[-1]['open_time_date_utc']}")
        else:
            raise RuntimeError("Failed to initialize data cache!")

    def get_latest_data(self):
        """
        [增量更新] 获取最新数据，更新缓存，并返回完整的 DataFrame 供特征工程使用
        """
        if self.local_cache is None:
            self.logger.warning("Cache is empty, running initialization...")
            # 这是一个保底，实际上应该在 start 显式调用 initialize
            return None

        # 1. 确定增量抓取的起点
        # 起点 = 缓存中最后一根的收盘时间 + 1ms
        last_k = self.local_cache.iloc[-1]
        start_time = int(last_k["close_time_ms"]) + 1
        
        # 2. 抓取增量 (通常这里只会抓到 0 条或 1-2 条)
        new_df = self._fetch_range_api(start_time)
        
        if new_df is not None and not new_df.empty:
            self.logger.info(f"Updates found: {len(new_df)} new bars.")
            
            # 3. 拼接更新
            # concat 是 pandas 比较昂贵的操作，但对于 (5000行 + 1行) 来说非常快
            self.local_cache = pd.concat([self.local_cache, new_df], ignore_index=True)
            
            # 4. 安全清洗 (去重 + 排序)
            self.local_cache.drop_duplicates("open_time_ms", inplace=True)
            
            # 5. 内存管理 (剪枝)
            # 如果缓存超过最大长度，切掉头部的旧数据
            if len(self.local_cache) > self.max_cache_len:
                self.local_cache = self.local_cache.iloc[-self.max_cache_len:].reset_index(drop=True)
        
        # 6. 返回**副本**给策略使用 (防止外部修改污染缓存)
        # 此时需要剔除未走完的 K 线（Binance API 总是返回最新的一根未闭合 K 线）
        
        # 获取当前系统时间
        current_time = int(time.time() * 1000)
        
        # 这里的 copy 很重要，特征工程会修改 df
        export_df = self.local_cache.copy() 
        
        if not export_df.empty:
            last_close_time = export_df.iloc[-1]["close_time_ms"]
            # 如果最后一根 K 线的收盘时间在未来，说明它没走完 -> 剔除
            if last_close_time > current_time:
                export_df = export_df.iloc[:-1]
        
        return export_df

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
        self.logger = common.setup_logger(log_name='ftmo_live', log_path=os.path.join(common.TEMPORARY_DIR, 'market_ftmo'))
        
        self.logger.info("Initializing Live Bot...")

        self.executor = MT5Executor(LiveConfig.SYMBOL_FTMO, LiveConfig.MAGIC_NUMBER)
        self.model_handler = model_loader.ModelHandler() # 自动加载训练好的模型

        # 1. 设置参数
        self.interval_ms = LiveConfig.TIMEFRAME * 60 * 1000 
        self.factory = FeatureFactory(FEATURE_CONFIG)
        
        # 2. 计算历史需求 (数量)
        self.min_bars_needed = self.factory.get_global_min_history(self.interval_ms)
        self.logger.info(f"History Required: {self.min_bars_needed} bars")
        
        # 3. 初始化数据源 (带缓存)
        # max_len 设置得比 min_bars_needed 大一些，比如 +500，留有余地
        self.data_feed = BinanceDataFeed(
            LiveConfig.SYMBOL_BINANCE, 
            LiveConfig.TIMEFRAME, 
            max_len = self.min_bars_needed + 500
        )
        
        # 4. 执行数据预热 (Warmup) -> 填充内存
        self.data_feed.initialize_cache(self.min_bars_needed, self.interval_ms)
        
        # 记录一下初始化后的最后一根时间
        initial_df = self.data_feed.get_latest_data()
        self.last_candle_time = initial_df.iloc[-1]["open_time_date_utc"] if not initial_df.empty else None

    def run_step(self):
        """
        每隔几秒运行一次
        """
        # 1. 获取最新数据 (DataFeed 内部自动处理增量更新)
        # 这里的 df 已经是清洗好、长度足够、且剔除了未闭合 K 线的完美数据
        df = self.data_feed.get_latest_data()
        
        if df is None or df.empty:
            return

        # 2. 检查是否有新 K 线产生
        # 比较这一轮拿到的最新时间 vs 上一轮处理的时间
        current_candle_time = df.iloc[-1]["open_time_date_utc"]
        
        if self.last_candle_time == current_candle_time:
            # 时间没变，说明没有新 K 线收盘 -> 跳过
            pass# return 
            
        self.logger.info(f"✨ New Candle Closed: {current_candle_time} | Buffer Size: {len(df)}")
        
        # 2. 特征工程 & 模型预测
        try:
            # A. 特征计算 (FeatureFactory)
            print("after rename:", df.head(5))
            print("columns:", df.columns.tolist())
            # 使用 FeatureFactory 生成特征
            self.factory.generate(df)
            
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