import os, sys, logging, time, threading, math, signal
from typing import Dict, List
from enum import Enum

# 引用你的基座
from bybit_engine import BybitEngine 
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..' , '..'))
from data_process import common
# -----------------------------------------------------------------------------
# 1. 配置与状态类
# -----------------------------------------------------------------------------
class MartingaleConfig:
    def __init__(self, symbol, base_order_value, max_safety_orders, 
                 price_deviation, safety_order_step_scale, 
                 volume_scale, tp_pct, stop_loss_pct):
        self.symbol = symbol
        self.base_order_value = base_order_value   # 首单金额 (USDT)
        self.max_safety_orders = max_safety_orders # 最大补仓次数
        self.price_deviation = price_deviation     # 初次补仓跌幅 (如 0.01 = 1%)
        self.safety_order_step_scale = safety_order_step_scale # 跌幅扩展系数 (如 1.05 表示网格越来越稀)
        self.volume_scale = volume_scale           # 加仓倍率 (如 1.5 表示每次买更多)
        self.tp_pct = tp_pct                       # 目标止盈率 (如 0.01 = 1%)
        self.stop_loss_pct = stop_loss_pct         # 止损率 (如 0.10 = 10% 亏损即止损)

# -----------------------------------------------------------------------------
# 基础枚举与数据类
# -----------------------------------------------------------------------------
class MarketState(Enum):
    OSCILLATION = "OSCILLATION"  # 震荡
    TREND_UP = "TREND_UP"        # 单边上涨
    TREND_DOWN = "TREND_DOWN"    # 单边下跌

class OrderSide(Enum):
    BUY = "Buy"
    SELL = "Sell"
    NONE = "None"

class BotState(Enum):
    RUNNING = "RUNNING"
    STOPPED = "STOPPED" # 止损触发后进入此状态

class SymbolState:
    def __init__(self, config: MartingaleConfig):
        self.config = config
        self.lock = threading.RLock()
        
        # 动态数据
        self.status = BotState.RUNNING
        self.current_pos_qty = 0.0      # 当前持仓数量
        self.avg_price = 0.0            # 持仓均价
        self.current_step = 0           # 当前处于第几层 (0=刚开单)
        self.total_profit = 0.0         # 累计收益
        self.win_count = 0              # 止盈次数
        self.stop_loss_count = 0        # 止损次数
        
        # 交易所规则缓存
        self.tick_size = 0.0001
        self.qty_step = 0.1
        self.min_qty = 1.0
        # --- 核心方向锁定 ---
        self.current_side = OrderSide.NONE        # None, "Buy" (做多马丁), "Sell" (做空马丁)   
# -----------------------------------------------------------------------------
# 2. 马丁机器人核心
# -----------------------------------------------------------------------------
class MartingaleBot:
    def __init__(self, engine: BybitEngine, configs: list, account_stop_loss_pct):
        self.logger, _ = common.setup_session_logger(
            sub_folder=self.__class__.__name__, 
            console_level=logging.DEBUG, 
            file_level=logging.DEBUG
        )
        self.engine = engine
        self.markets: Dict[str, SymbolState] = {}
        self.stop_signal = False
        self.account_stop_loss_pct = account_stop_loss_pct
        self.starting_balance = 0.0
        self.start_time = time.time()
        
        # 初始化配置
        for cfg in configs:
            self.markets[cfg.symbol] = SymbolState(cfg)
            # 设置高频杠杆
            self.engine.set_leverage(cfg.symbol, 10) 
            self.logger.info(f"Load Config: {cfg.symbol} | Base:{cfg.base_order_value}U | MaxSafety:{cfg.max_safety_orders}")

    # -------------------------------------------------------------------------
    # 辅助计算
    # -------------------------------------------------------------------------
    def adjust_qty(self, raw_qty, symbol_state):
        """数量精度对齐"""
        s = symbol_state
        if raw_qty < s.min_qty: return str(s.min_qty)
        steps = raw_qty / s.qty_step
        qty = math.floor(steps) * s.qty_step
        
        # 处理小数位格式化
        if s.qty_step >= 1:
            return str(int(qty))
        else:
            precision = int(math.ceil(-math.log10(s.qty_step)))
            return f"{qty:.{precision}f}"

    def adjust_price(self, raw_price, symbol_state):
        """价格精度对齐"""
        s = symbol_state
        steps = raw_price / s.tick_size
        price = round(steps) * s.tick_size
        precision = int(math.ceil(-math.log10(s.tick_size)))
        return f"{price:.{precision}f}"

    def update_instrument_info(self):
        """同步交易所规则"""
        res = self.engine.http.get_instruments_info(category=self.engine.category)
        if res.get('retCode') == 0:
            info_map = {item['symbol']: item for item in res['result']['list']}
            for s in self.markets:
                if s in info_map:
                    m = self.markets[s]
                    m.tick_size = float(info_map[s]['priceFilter']['tickSize'])
                    m.min_qty = float(info_map[s]['lotSizeFilter']['minOrderQty'])
                    m.qty_step = float(info_map[s]['lotSizeFilter']['qtyStep'])

    def decide_base_side(self, symbol):
        """
        根据 Z-Score 或趋势判断首单方向
        """
        m = self.markets[symbol]
        # 使用你已有的 Z-Score 逻辑
        # z_val > 0 通常代表强势，z_val < 0 代表弱势
        if m.trend_direction == 1: 
            return "Buy"  # 顺势做多
        elif m.trend_direction == -1:
            return "Sell" # 顺势做空
        else:
            # 震荡市：可以根据 RSI 超买超卖，或者随机选一个
            return "Buy"

    def update_micro_market_status_volume(self, symbol):
        """
        趋势熔断逻辑 (V9 适配版)：Z-Score 概率模型 + 筹码分布 (VPVR)
        旨在精准识别 20% 的极端波动时间
        """
        import math
        m = self.markets[symbol]
        
        # 1. 获取 K 线数据 (100根 1m 线)
        res = self.engine.http.get_kline(category=self.engine.category, symbol=symbol, interval=30, limit=100)   #1, 3, 5, 15, 30
        if res.get('retCode') != 0 or not res['result']['list']: 
            return

        # 提取收盘价与成交量
        k_list = res['result']['list']
        prices = [float(k[4]) for k in k_list]
        volumes = [float(k[5]) for k in k_list]
        prices.reverse() # 转为正序
        volumes.reverse()

        # 2. 计算 Z-Score (统计学偏离度)
        n = len(prices)
        ma = sum(prices) / n
        variance = sum((p - ma) ** 2 for p in prices) / n
        std_dev = math.sqrt(variance)
        curr = prices[-1]
        
        # 计算 Z-Score (不取绝对值，保留方向)
        z_val = (curr - ma) / std_dev if std_dev > 0 else 0
        z_score_abs = abs(z_val)
        
        # 3. 计算筹码集中度 (VPVR 简化版)
        bin_count = 10
        min_p, max_p = min(prices), max(prices)
        # 防止除零
        price_range = (max_p - min_p)
        interval = price_range / bin_count if price_range > 0 else 0.0001
        
        profile = [0.0] * bin_count
        for p, v in zip(prices, volumes):
            # 将价格归入对应的成交量桶
            idx = min(int((p - min_p) / interval), bin_count - 1)
            profile[idx] += v
        
        # 计算最大成交量桶占比
        total_vol = sum(profile)
        concentration = max(profile) / total_vol if total_vol > 0 else 0
        
        # 4. 判定状态与方向
        with m.lock:
            # 逻辑：价格偏离过大 (Z > 1.28) 或者 筹码极度分散 (Concentration < 0.10)
            # 1.28 对应正态分布双侧约 20% 的尾部区域
            # if z_score_abs > 1.5 or concentration < 0.10:
            if concentration < 0.10:
                if z_val > 0:
                    m.market_state = MarketState.TREND_UP
                    m.trend_direction = 1
                else:
                    m.market_state = MarketState.TREND_DOWN
                    m.trend_direction = -1
                
                self.logger.warning(f"🚨 [{symbol}] 趋势触发: Z={z_val:.2f}, POC={concentration:.2f}")
            else:
                m.market_state = MarketState.OSCILLATION
                m.trend_direction = 0

    # -------------------------------------------------------------------------
    # 核心流程
    # -------------------------------------------------------------------------
    def startup(self):
        self.logger.info("🚀 高频马丁机器人启动...")
        self.update_instrument_info()
        
        # 1. 启动 WS
        self.engine.start_stream(self.on_order_update)
        
        # 2. 初始扫描（恢复状态）
        for symbol in self.markets:
            self.sync_position(symbol)
            self.reconcile(symbol)
        
        # 3. 守护线程
        threading.Thread(target=self.run_loop, daemon=True).start()
        
        while not self.stop_signal:
            time.sleep(1)

    def sync_position(self, symbol):
        m = self.markets[symbol]
        res = self.engine.http.get_positions(category=self.engine.category, symbol=symbol)
        
        with m.lock:
            if res.get('retCode') == 0:
                pos_list = res['result']['list']
                size = 0.0
                avg_price = 0.0
                side = None
                
                for p in pos_list:
                    s = float(p.get('size', 0))
                    if s > 0:
                        size = s
                        avg_price = float(p.get('avgPrice', 0))
                        side = p.get('side') # "Buy" 或 "Sell"
                        break 
                
                m.current_pos_qty = size
                m.avg_price = avg_price
                m.current_side = OrderSide.BUY if side == "Buy" else OrderSide.SELL # 锁定当前轮次方向
                
                if size == 0:
                    m.current_step = 0
                    m.current_side = OrderSide.NONE
                else:
                    # 这里的 Step 推算逻辑保持不变，但要基于 base_qty
                    base_qty_est = m.config.base_order_value / avg_price
                    m.current_step = 1 
                    current_calc_qty = base_qty_est
                    for i in range(1, m.config.max_safety_orders + 1):
                        safety_qty = base_qty_est * (m.config.volume_scale ** (i-1))
                        current_calc_qty += safety_qty
                        if size >= current_calc_qty * 0.9:
                            m.current_step = i + 1
                        else: break

    def on_order_update(self, message):
        """WS 回调：只关心 Filled"""
        data = message.get('data', [])
        for order in data:
            if order['orderStatus'] == "Filled":
                symbol = order['symbol']
                if symbol in self.markets:
                    side = order['side']
                    qty = float(order['qty'])
                    price = float(order.get('avgPrice', order['price']))
                    self.handle_execution(symbol, side, qty, price)

    def handle_execution(self, symbol, side, qty, price):
        """⚡ 成交处理"""
        m = self.markets[symbol]
        
        # 重新同步一次持仓，保证数据绝对准确（避免本地计算误差）
        # 高频模式下，也可以选择本地累加，这里为了稳健选择 API 同步
        # 为了速度，我们先本地计算，再异步校验
        with m.lock:
            if side == "Buy":
                # 📥 买入成交 (Base 或 Safety)
                self.logger.info(f"🟢 [{symbol}] 买入成交! 价格:{price}")
                m.current_step += 1
            elif side == "Sell":
                # 💰 卖出成交 (止盈 或 止损)
                self.logger.info(f"🎉 [{symbol}] 卖出成交! 价格:{price}")
                # 判断是全部卖出还是部分
                # 这里简单假设是全平止盈
                m.current_step = 0
                m.total_profit += (price - m.avg_price) * qty
                m.win_count += 1
        
        # 立即触发策略重算
        # 稍微延时一点点确保交易所结算完成
        threading.Thread(target=self.delayed_reconcile, args=(symbol, 1.0)).start()

    def delayed_reconcile(self, symbol, delay):
        time.sleep(delay)
        self.sync_position(symbol)
        self.reconcile(symbol)

    def check_global_risk(self):
        """
        🌍 全局风控：保护总本金安全
        """
        if self.starting_balance <= 0: return False
        
        # 计算所有运行币种的总浮亏
        total_unrealized_pnl = 0.0
        for s, m in self.markets.items():
            # 实时计算每个币种的 PnL
            ticker = self.engine.http.get_tickers(category=self.engine.category, symbol=s)
            curr_price = float(ticker['result']['list'][0]['lastPrice'])
            
            if m.current_pos_qty > 0:
                pnl = (curr_price - m.avg_price) * m.current_pos_qty if m.current_side == "Buy" else \
                      (m.avg_price - curr_price) * m.current_pos_qty
                total_unrealized_pnl += pnl

        # 亏损比例判断
        drawdown = abs(total_unrealized_pnl) / self.starting_balance
        if total_unrealized_pnl < 0 and drawdown >= self.account_stop_loss_pct:
            self.logger.critical(f"🆘 [全局熔断] 总回撤 {drawdown:.2%} 触发阈值！立即执行全账户清仓！")
            self.emergency_wipe_all_account_positions(use_market=True)
            self.stop_signal = True
            return True
        return False

    # -------------------------------------------------------------------------
    # 🧠 策略大脑 (Reconcile)
    # -------------------------------------------------------------------------
    def check_stop_loss(self, symbol):
        """🛡️ 止损检查"""
        m = self.markets[symbol]
        if m.current_pos_qty <= 0 or m.status == BotState.STOPPED:
            return False

        # 获取当前市价
        res = self.engine.http.get_tickers(category=self.engine.category, symbol=symbol)
        if res.get('retCode') == 0:
            curr_price = float(res['result']['list'][0]['lastPrice'])
            # 计算浮亏
            pnl_pct = (curr_price - m.avg_price) / m.avg_price
            
            if pnl_pct < -m.config.stop_loss_pct:
                self.logger.warning(f"🚨 [{symbol}] 触发止损! 浮亏: {pnl_pct:.2%} (阈值: -{m.config.stop_loss_pct:.2%})")
                
                # 1. 撤销所有挂单
                self.engine.cancel_all_http(symbol)
                # 2. 市价全平
                self.engine.place_order(symbol, "Sell", m.current_pos_qty, curr_price, f"STOP_LOSS_{int(time.time())}", order_type="Market", is_reduce=True)
                # 3. 锁定状态
                m.status = BotState.STOPPED
                m.stop_loss_count += 1
                m.current_step = 0
                return True
        return False

    # -------------------------------------------------------------------------
    # 核心：自动对齐与异常修复 (Reconciliation)
    # -------------------------------------------------------------------------
    def reconcile(self, symbol):
        """
        核心修复逻辑：对比 [应有状态] 与 [现有状态]，不一致则执行修复
        """
        m = self.markets[symbol]
        if m.status == BotState.STOPPED: return

        with m.lock:       
            # 1. 先检查止损 (最高优先级)
            if self.check_stop_loss(symbol): return

            # 2. 获取当前挂单列表 (Reality)
            # 为了防止网络抖动，这里直接从 HTTP 获取最新的挂单
            open_orders = self.fetch_symbol_open_orders(symbol)
            
            # 3. 策略逻辑分叉
            if m.current_pos_qty == 0:
                # --- 情况 A: 空仓 ---
                # 应有状态：必须有一个 BASE 买单
                has_base = any("BASE" in o['orderLinkId'] for o in open_orders)
                if not has_base:
                    self.logger.info(f"🔄 [修复] {symbol} 发现空仓且无首单，正在重新挂入...")
                    m.current_side = self.decide_base_side(symbol) # 预测方向
                    self.place_base_order(symbol, m.current_side)
            
            else:   #if m.current_pos_qty > 0:
                # 1. 始终保持止盈单 (TP)
                has_tp = any("TP" in o['orderLinkId'] for o in open_orders)
                if not has_tp: self.place_tp_order(symbol)
                
                # 2. 补仓控制：只要没到最大层数，就继续挂补仓单
                has_safety = any("SAFETY" in o['orderLinkId'] for o in open_orders)
                if m.current_step <= m.config.max_safety_orders:
                    if not has_safety: self.place_safety_order(symbol)
                else:
                    # 🌸 达到最大层数，不再操作补仓，依靠全局止损保护
                    self.logger.info(f"🛡️ [{symbol}] 已达最大层数 {m.config.max_safety_orders}，仓位锁定中...")

    def fetch_symbol_open_orders(self, symbol):
        """获取当前活跃挂单"""
        orders = []
        res = self.engine.http.get_open_orders(category=self.engine.category, symbol=symbol)
        if res.get('retCode') == 0:
            orders = res['result']['list']
        return orders

    def report_status(self):
        """打印统计报表"""
        print("\n" + "="*90)
        print(f"📊 马丁简报 | 运行: {int(time.time()-self.start_time)}s")
        print(f"{'SYMBOL':<10} | {'SIDE':<5} | {'STEP':<4} | {'POS_VAL':<10} | {'AVG_P':<10} | {'PROFIT':<10}")
        print("-" * 90)
        for s, m in self.markets.items():
            pos_val = m.current_pos_qty * m.avg_price
            side_str = m.current_side if m.current_side else "NONE"
            print(f"{s:<10} | {side_str:<5} | {m.current_step:<4} | {pos_val:<10.2f} | {m.avg_price:<10.4f} | {m.total_profit:<10.4f}")
        print("="*90 + "\n")

    # -------------------------------------------------------------------------
    # 抽取出的原子下单方法
    # -------------------------------------------------------------------------
    def place_base_order(self, symbol, side:OrderSide):
        m = self.markets[symbol]
        res = self.engine.http.get_tickers(category=self.engine.category, symbol=symbol)
        curr_price = float(res['result']['list'][0]['lastPrice'])
        qty_str = self.adjust_qty(m.config.base_order_value / curr_price, m)
        price_str = self.adjust_price(curr_price, m)
        self.engine.place_order(symbol, side.value, qty_str, price_str, f"BASE_{int(time.time())}")

    def place_tp_order(self, symbol):
        m = self.markets[symbol]
        # 多头：均价之上卖出；空头：均价之下买入
        if m.current_side == "Buy":
            tp_price = m.avg_price * (1 + m.config.tp_pct)
            side = "Sell"
        else:
            tp_price = m.avg_price * (1 - m.config.tp_pct)
            side = "Buy"

        tp_price_str = self.adjust_price(tp_price, m)
        tp_qty_str = self.adjust_qty(m.current_pos_qty, m)
        self.engine.place_order(symbol, side, tp_qty_str, tp_price_str, f"TP_{int(time.time())}", is_reduce=True)

    def place_sl_order(self, symbol):
        m = self.markets[symbol]
        # 多头：触发价在均价之下；空头：触发价在均价之上
        if m.current_side == "Buy":
            sl_trigger = m.avg_price * (1 - m.config.stop_loss_pct)
            side = "Sell"
        else:
            sl_trigger = m.avg_price * (1 + m.config.stop_loss_pct)
            side = "Buy"

        params = {
            "category": self.engine.category,
            "symbol": symbol,
            "side": side,
            "orderType": "Market", 
            "qty": self.adjust_qty(m.current_pos_qty, m),
            "triggerPrice": self.adjust_price(sl_trigger, m),
            "triggerBy": "LastPrice",
            "orderLinkId": f"SL_{int(time.time())}",
            "reduceOnly": True,
            "positionIdx": 0
        }
        self.engine.http.place_order(**params)

    def place_safety_order(self, symbol):
        m = self.markets[symbol]
        # 补仓间距计算
        drop = m.config.price_deviation * (m.config.safety_order_step_scale ** (m.current_step - 1))
        
        # 多头：更低价买入；空头：更高价卖出
        if m.current_side == "Buy":
            price = m.avg_price * (1 - drop)
            side = "Buy"
        else:
            price = m.avg_price * (1 + drop)
            side = "Sell"
            
        val = m.config.base_order_value * (m.config.volume_scale ** m.current_step)
        qty_str = self.adjust_qty(val / price, m)
        price_str = self.adjust_price(price, m)
        self.engine.place_order(symbol, side, qty_str, price_str, f"SAFETY_{m.current_step}_{int(time.time())}")

    # -------------------------------------------------------------------------
    # 监控循环
    # -------------------------------------------------------------------------
    def run_loop(self):
        market_state_update_time = time.time()
        while not self.stop_signal:
            try:
                if time.time() - market_state_update_time > 30:
                    for symbol in self.markets:
                        # 1. 更新趋势与 Z-Score
                        self.update_micro_market_status_volume(symbol)
                for symbol in self.markets:
                    # 2. 同步状态
                    self.sync_position(symbol)
                    # 3. 审计与修复
                    self.reconcile(symbol)
                
                self.report_status()
            except Exception as e:
                self.logger.error(f"❌ 守护循环异常: {e}")
            time.sleep(5)

# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. 密钥配置
    BASE = os.path.dirname(os.path.abspath(__file__))
    API_K = os.path.join(BASE, "keys", "hmac_api_key")
    API_S = os.path.join(BASE, "keys", "hmac_secret")
    RSA_K = os.path.join(BASE, "keys", "api_key")     
    RSA_P = os.path.join(BASE, "keys", "bybit_rsa.pem")

    # 2. 策略配置
    # 高频马丁思路：小跌幅(0.8%)就补，小利润(0.6%)就跑，快速周转
    configs = [
        MartingaleConfig(
            symbol="DOGEUSDT",
            base_order_value=10.0,   # 首单 10 U
            max_safety_orders=6,     # 最多补 6 次
            price_deviation=0.008,   # 跌 0.8% 补第一次
            safety_order_step_scale=1.2, # 之后每次跌幅间距扩大 1.2倍
            volume_scale=1.3,        # 每次加仓金额是上次的 1.3 倍
            tp_pct=0.006,            # 赚 0.6% 就跑 (高频)
            stop_loss_pct=0.15       # 浮亏 15% 认赔离场
        ),
        MartingaleConfig(
            symbol="SOLUSDT",
            base_order_value=15.0,
            max_safety_orders=5,
            price_deviation=0.01,
            safety_order_step_scale=1.1,
            volume_scale=1.5,
            tp_pct=0.008,
            stop_loss_pct=0.10
        ),
    ]

    # 3. 启动
    engine = BybitEngine(API_K, API_S, RSA_K, RSA_P)
    bot = MartingaleBot(engine, configs, account_stop_loss_pct =0.2)
    
    def signal_handler(sig, frame):
        print("\n👋 停止机器人...")
        bot.stop_signal = True
        engine.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    bot.startup()