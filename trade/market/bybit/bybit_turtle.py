import time
import logging
import pandas as pd
import argparse
import sys
import os
import signal
from datetime import datetime

# 路径适配
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..', '..'))

from bybit_engine import BybitEngine
from data_process.common import setup_session_logger
from trade.strategy.strategy_ftmo import PositionDir, ActionType
# 假设 TurtleBrain 在此路径，请确保该文件存在
from trade.strategy.strategy_turtle import TurtleBrain 

# ================= 配置区域 =================
class TurtleConfig:
    # 策略参数
    SYMBOL         = "DOGEUSDT"
    TIMEFRAME      = 240       # 分钟 (Bybit: 1, 3, 5, 15, 60, 240, D)
    ENTRY_PERIOD   = 15        # 唐奇安入场周期
    EXIT_PERIOD    = 10        # 唐奇安离场周期
    ATR_PERIOD     = 20
    MAX_LAYERS     = 1         # 最大加仓层数
    RISK_PER_UNIT  = 0.01      # 单笔风险 1%
    MAX_DAILY_LOSS = 0.5      # 最大回撤限制
    UPPER_LIMIT    = 0.7
    UNIT_PCT_SCALE = 2
    
    # 轮询间隔 (秒)
    POLL_INTERVAL  = 10

# ================= 数据源适配器 =================
class BybitDataFeed:
    def __init__(self, engine: BybitEngine, symbol: str, interval: int):
        self.engine = engine
        self.symbol = symbol
        self.interval = str(interval) # Bybit API 需要字符串
        self.logger = logging.getLogger("BybitDataFeed")

    def get_latest_data(self, limit=200) -> pd.DataFrame:
        """从 Bybit 获取 K 线并转换为 DataFrame"""
        try:
            # 调用 engine 的 http 接口
            res = self.engine.http.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.interval,
                limit=limit
            )
            
            if res.get('retCode') != 0:
                self.logger.error(f"获取 K 线失败: {res.get('retMsg')}")
                return None

            # Bybit 返回数据格式: [startTime, open, high, low, close, volume, turnover]
            # 且顺序是倒序（最新在最前）
            raw_list = res['result']['list']
            data = []
            for row in raw_list:
                data.append({
                    "open_time": int(row[0]), # 毫秒时间戳
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5])
                })
            
            # 转为 DataFrame 并按时间正序排列
            df = pd.DataFrame(data)
            df = df.sort_values("open_time").reset_index(drop=True)
            
            # 转换时间索引
            df['open_time_date_utc'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time_date_utc', inplace=True)
            
            return df
        except Exception as e:
            self.logger.error(f"数据处理异常: {e}")
            return None

# ================= 执行器适配器 =================
class BybitTurtleExecutor:
    def __init__(self, engine: BybitEngine, symbol: str):
        self.engine = engine
        self.symbol = symbol
        self.logger = logging.getLogger("BybitExecutor")
        
        # 初始化精度信息
        self.qty_step = 0.0
        self.tick_size = 0.0
        self.min_qty = 0.0
        self._init_symbol_info()

    def _init_symbol_info(self):
        """同步交易所精度配置，防止 Invalid Volume"""
        try:
            res = self.engine.http.get_instruments_info(category="linear", symbol=self.symbol)
            if res['retCode'] == 0:
                info = res['result']['list'][0]
                self.qty_step = float(info['lotSizeFilter']['qtyStep'])
                self.min_qty = float(info['lotSizeFilter']['minOrderQty'])
                self.tick_size = float(info['priceFilter']['tickSize'])
                self.logger.info(f"✅ 精度同步: QtyStep={self.qty_step}, Tick={self.tick_size}")
        except Exception as e:
            self.logger.error(f"精度同步失败: {e}")

    def get_account_equity(self):
        """获取 USDT 账户净值"""
        res = self.engine.http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if res.get('retCode') == 0:
            return float(res['result']['list'][0]['coin'][0]['equity'])
        return 0.0

    def get_current_state(self):
        """
        返回: (PositionDir, layers, avg_price)
        适配 TurtleBrain 的接口需求
        """
        try:
            res = self.engine.http.get_positions(category="linear", symbol=self.symbol)
            if res['retCode'] != 0: 
                return PositionDir.FLAT, 0, 0.0
            
            pos_list = res['result']['list']
            if not pos_list: 
                return PositionDir.FLAT, 0, 0.0

            pos = pos_list[0]
            size = float(pos['size'])
            avg_price = float(pos['avgPrice']) if size > 0 else 0.0
            side = pos['side'] # 'Buy' or 'Sell'

            if size == 0:
                return PositionDir.FLAT, 0, 0.0
            
            # 简单的层数估算（海龟逻辑通常需要自己记录，这里简化为 1 层代表有持仓）
            # 如果需要严格的层数逻辑，需要在外部记录或通过 size/unit_size 推算
            direction = PositionDir.LONG if side == 'Buy' else PositionDir.SHORT
            return direction, 1, avg_price

        except Exception as e:
            self.logger.error(f"获取持仓状态失败: {e}")
            return PositionDir.FLAT, 0, 0.0

    def user_order(self, size, is_buy, stop_loss=None):
        """
        执行下单逻辑
        size: 币的数量 (Base Coin)
        is_buy: 方向
        stop_loss: 止损比例 (如 0.05 代表 5%)
        """
        # 1. 精度对齐
        qty = round(float(size) / self.qty_step) * self.qty_step
        qty = max(self.min_qty, qty)
        qty_str = str(qty)

        # 2. 获取当前价格用于计算 SL 价格
        # 注意：这里使用市价单，所以 entry_price 近似为当前 ticker 价格
        tickers = self.engine.http.get_tickers(category="linear", symbol=self.symbol)
        curr_price = float(tickers['result']['list'][0]['lastPrice'])
        
        # 3. 计算止损价格 (Bybit 需要具体价格，Brain 给的是比例)
        sl_price = 0.0
        if stop_loss:
            if is_buy:
                raw_sl = curr_price * (1 - stop_loss)
            else:
                raw_sl = curr_price * (1 + stop_loss)
            sl_price = round(raw_sl / self.tick_size) * self.tick_size

        side = "Buy" if is_buy else "Sell"
        self.logger.info(f"🐢 执行下单: {side} {qty_str} @ 市价 | SL: {sl_price}")

        try:
            # 4. 调用 engine 下单，带上 stopLoss 参数
            # Market 单不需要传 price
            order_params = {
                "category": "linear",
                "symbol": self.symbol,
                "side": side,
                "orderType": "Market",
                "qty": qty_str,
                "positionIdx": 0, # 单向持仓模式
                "reduceOnly": False
            }
            
            if sl_price > 0:
                order_params["stopLoss"] = str(sl_price)

            # 使用 HTTP 接口下单更稳妥，或者用 engine.ws_trade.place_order
            # 这里为了简单直接用 HTTP，因为海龟不是高频策略
            res = self.engine.http.place_order(**order_params)
            
            if res['retCode'] == 0:
                self.logger.info(f"✅ 下单成功: ID {res['result']['orderId']}")
            else:
                self.logger.error(f"❌ 下单失败: {res['retMsg']}")
                
        except Exception as e:
            self.logger.error(f"下单异常: {e}")

    def user_close(self):
        """全平当前持仓"""
        try:
            # 获取持仓
            res = self.engine.http.get_positions(category="linear", symbol=self.symbol)
            for pos in res['result']['list']:
                size = float(pos['size'])
                if size > 0:
                    side = "Sell" if pos['side'] == "Buy" else "Buy"
                    self.logger.info(f"正在平仓: {pos['side']} {size}")
                    
                    self.engine.http.place_order(
                        category="linear",
                        symbol=self.symbol,
                        side=side,
                        orderType="Market",
                        qty=str(size),
                        positionIdx=0,
                        reduceOnly=True
                    )
        except Exception as e:
            self.logger.error(f"平仓异常: {e}")

# ================= 主机器人逻辑 =================
class BybitTurtleBot:
    def __init__(self, engine, is_long_account):
        self.logger, _ = setup_session_logger(
            sub_folder="BybitTurtle", 
            console_level=logging.DEBUG, 
            file_level=logging.DEBUG
        )
        self.engine = engine
        self.symbol = TurtleConfig.SYMBOL
        
        # 初始化组件
        self.data_feed = BybitDataFeed(engine, self.symbol, TurtleConfig.TIMEFRAME)
        self.executor = BybitTurtleExecutor(engine, self.symbol)
        
        # 初始化大脑
        self.brain = TurtleBrain(
            executor=self.executor,
            entry_period=TurtleConfig.ENTRY_PERIOD,
            exit_period=TurtleConfig.EXIT_PERIOD,
            atr_period=TurtleConfig.ATR_PERIOD,
            max_layers=TurtleConfig.MAX_LAYERS,
            risk_per_unit=TurtleConfig.RISK_PER_UNIT,
            max_daily_loss_pct=TurtleConfig.MAX_DAILY_LOSS,
            upper_limit = TurtleConfig.UPPER_LIMIT,
            unit_pct_scale = TurtleConfig.UNIT_PCT_SCALE,
        )
        
        self.last_candle_time = None
        self.stop_signal = False
        
        # 设置杠杆
        self.engine.set_leverage(self.symbol, "10")
        try:
            # 尝试切换为单向持仓模式
            res = self.engine.http.switch_position_mode(
                category="linear", 
                symbol=self.symbol, 
                mode=0 # 0: 单向持仓
            )
            if res.get('retCode') == 0:
                self.logger.info(f"✅ [{self.symbol}] 成功切换为单向持仓模式")
        except Exception as e:
            # 如果报错代码是 110025，说明已经是目标模式，直接忽略即可
            if "110025" in str(e):
                self.logger.info(f"ℹ️ [{self.symbol}] 已经是单向持仓模式，无需修改")
            else:
                self.logger.error(f"⚠️ 切换持仓模式失败: {e}")

    def run_step(self):
        # 1. 获取数据
        df = self.data_feed.get_latest_data()
        if df is None or df.empty: return

        # 2. 检查 K 线闭合 (基于 open_time)
        current_candle_time = df.iloc[-1].name
        if self.last_candle_time == current_candle_time:
            # 只有在时间变动时才运行逻辑
            return 
        
        self.logger.info(f"📊 新 K 线闭合: {current_candle_time} | Close: {df.iloc[-1]['close']}")
        
        # 3. 获取实盘状态
        curr_dir, _, last_price = self.executor.get_current_state()
        equity = self.executor.get_account_equity()
        
        # 计算持仓占比 (简单模拟，如需精确需计算名义价值)
        current_price = df.iloc[-1]['close']
        curr_pos_size_pct = 0.0
        if curr_dir != PositionDir.FLAT:
            curr_pos_size_pct = 0.1 # 只要有持仓，给个非0值让Brain知道

        # 4. 大脑决策
        # 注意：Brain 内部会调用 executor.user_order 进行下单
        self.brain.decide(
            df=df,
            current_time=pd.to_datetime(datetime.now()), # 或使用 server time
            account_balance=equity,
            curr_dir=curr_dir,
            curr_pos_size=curr_pos_size_pct,
            last_entry_price=last_price
        )
        
        self.last_candle_time = current_candle_time

    def run(self):
        self.logger.info("🚀 Bybit Turtle Strategy Started...")
        last_heartbeat = 0
        
        while not self.stop_signal:
            try:
                # 增加心跳：每 5 分钟打印一次，证明循环没卡死
                if time.time() - last_heartbeat > 300:
                    self.logger.info("💓 Heartbeat: Bot is still alive and cycling...")
                    last_heartbeat = time.time()
                
                self.run_step()
                time.sleep(TurtleConfig.POLL_INTERVAL)
            except Exception as e:
                self.logger.error(f"主循环异常: {e}", exc_info=True)
                time.sleep(10)

if __name__ == "__main__":
    # 解析参数，复用 Martingale 的逻辑
    parser = argparse.ArgumentParser(description="Bybit Turtle Bot")
    parser.add_argument("-t", "--testnet", action="store_true", help="Run on Testnet")
    args = parser.parse_args()

    # 路径配置
    keypath = 'Maringale'
    side_path = 'Long' # 默认用 Long 文件夹下的 key
    BASE = os.path.dirname(os.path.abspath(__file__))
    
    # 假设你的 Key 文件结构如下
    # keys/Maringale/Long/hmac_api_key
    API_K = os.path.join(BASE, "keys", keypath, side_path, "hmac_api_key")
    API_S = os.path.join(BASE, "keys", keypath, side_path, "hmac_secret")
    RSA_K = os.path.join(BASE, "keys", keypath, side_path, "api_key")
    RSA_P = os.path.join(BASE, "keys", "bybit_rsa.pem")

    if not os.path.exists(API_K):
        print(f"❌ Key file not found: {API_K}")
        sys.exit(1)

    # 初始化引擎
    engine = BybitEngine(API_K, API_S, RSA_K, RSA_P, testnet=args.testnet)
    
    # 启动机器人
    bot = BybitTurtleBot(engine, is_long_account=True)
    
    # 信号处理
    def signal_handler(sig, frame):
        print("\n👋 Stop signal received...")
        bot.stop_signal = True
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    bot.run()