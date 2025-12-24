# ======================================================
# Gate.io Long Grid Strategy - 完整优化版 (修复版v3)
# ======================================================

import ccxt
import asyncio
import websockets
import json
import time
import statistics
import math
import hmac
import hashlib
import requests
from datetime import datetime
from typing import Dict, Optional, List

# ======================================================
# 全局配置参数
# ======================================================
class Config:
    # === Gate.io API 配置 ===
    # 实盘配置
    GATE_API_KEY = ""
    GATE_API_SECRET = ""
    
    # 模拟盘配置
    TESTNET_API_KEY = "your_testnet_api_key"
    TESTNET_API_SECRET = "your_testnet_api_secret"
    
    # === 交易配置 ===
    SYMBOL = "ETH/USDT:USDT"
    SETTLE = "usdt"
    CONTRACT = "ETH_USDT"
    
    # === 网格参数 ===
    PRICE_LOW = 2500
    PRICE_HIGH = 3500
    GRID_NUM = 100
    GRID_QTY = 1
    LEVERAGE = 100
    
    # === 手续费配置 ===
    MAKER_FEE = 0.0002  # 开仓手续费 0.02%
    TAKER_FEE = 0.0005  # 平仓手续费 0.05%
    
    # === 策略指标参数 ===
    VOL_WINDOW = 60          # 波动率窗口
    ATR_PERIOD = 14          # ATR周期
    MA_SHORT = 20            # 短期均线
    MA_LONG = 60             # 长期均线
    TREND_THRESHOLD = 0.001  # 趋势阈值
    CCI_PERIOD = 20          # CCI周期
    FREEZE_IN_DOWNTREND = True  # 下跌趋势中冻结开仓
    
    # === 企业微信通知 ===
    WECOM_WEBHOOK = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=ee15450b-daf8-40db-aa00-1c1f2a9ff332"
    ENABLE_WECOM = True
    
    # === 运行模式 ===
    TESTNET = False  # True=模拟盘, False=实盘
    DRY_RUN = False  # True=本地模拟(不发送真实订单)

cfg = Config()

# ======================================================
# API配置管理
# ======================================================
class APIConfig:
    def __init__(self, testnet: bool = False):
        self.testnet = testnet
        
        if testnet:
            # 模拟盘配置
            self.api_key = cfg.TESTNET_API_KEY
            self.api_secret = cfg.TESTNET_API_SECRET
            self.base_url = "https://fx-api-testnet.gateio.ws"  # 模拟盘地址
            self.ws_url = "wss://fx-ws-testnet.gateio.ws/v4/ws/usdt"
        else:
            # 实盘配置
            self.api_key = cfg.GATE_API_KEY
            self.api_secret = cfg.GATE_API_SECRET
            self.base_url = "https://api.gateio.ws"
            self.ws_url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
    
    def get_api_key(self):
        return self.api_key
    
    def get_api_secret(self):
        return self.api_secret
    
    def get_base_url(self):
        return self.base_url
    
    def get_ws_url(self):
        return self.ws_url

# ======================================================
# 企业微信通知
# ======================================================
class WeComNotifier:
    def __init__(self, webhook_url: str, enabled: bool = True):
        self.webhook_url = webhook_url
        self.enabled = enabled
    
    def send(self, message: str):
        if not self.enabled or not self.webhook_url or "your_key_here" in self.webhook_url:
            return
        
        try:
            data = {
                "msgtype": "text",
                "text": {
                    "content": message
                }
            }
            requests.post(self.webhook_url, json=data, timeout=5)
        except Exception as e:
            print(f"企业微信通知发送失败: {e}")

# ======================================================
# 日志管理器
# ======================================================
class Logger:
    def __init__(self, notifier: WeComNotifier):
        self.notifier = notifier
    
    def log(self, message: str, notify: bool = False):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        if notify:
            self.notifier.send(message)

# ======================================================
# 价格数据源 (WebSocket)
# ======================================================
class PriceFeed:
    def __init__(self, contract: str, logger: Logger, api_config: APIConfig):
        self.contract = contract
        self.logger = logger
        self.api_config = api_config
        self.price = None
        self.ws_url = api_config.get_ws_url()
        self.running = False
        self.task = None
    
    async def connect(self):
        self.running = True
        
        while self.running:
            try:
                async with websockets.connect(
                    self.ws_url, 
                    ping_interval=20,
                    ping_timeout=60,
                    close_timeout=10
                ) as ws:
                    subscribe_msg = {
                        "time": int(time.time()),
                        "channel": "futures.tickers",
                        "event": "subscribe",
                        "payload": [self.contract]
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    self.logger.log(f"WebSocket已连接,订阅 {self.contract}")
                    
                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                            data = json.loads(msg)
                            
                            if data.get("event") == "update" and data.get("channel") == "futures.tickers":
                                result = data.get("result", [])
                                if result and len(result) > 0:
                                    ticker = result[0]
                                    self.price = float(ticker.get("last", 0))
                        
                        except asyncio.TimeoutError:
                            await ws.send(json.dumps({"time": int(time.time()), "channel": "futures.ping"}))
                            
            except Exception as e:
                self.logger.log(f"WebSocket错误: {e}")
                await asyncio.sleep(5)
    
    def start(self):
        self.task = asyncio.create_task(self.connect())
    
    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
    
    def get_price(self, symbol: str) -> Optional[float]:
        return self.price

# ======================================================
# 仓位管理器
# ======================================================
class PositionManager:
    def __init__(self, settle: str, contract: str, logger: Logger, api_config: APIConfig):
        self.settle = settle
        self.contract = contract
        self.logger = logger
        self.api_config = api_config
    
    def get_position(self, symbol: str) -> Dict:
        """获取当前仓位"""
        try:
            url = f"{self.api_config.get_base_url()}/api/v4/futures/{self.settle}/positions/{self.contract}"
            timestamp = str(int(time.time()))
            
            query_string = ""
            body_hash = hashlib.sha512("".encode('utf-8')).hexdigest()
            sign_string = f"GET\n/api/v4/futures/{self.settle}/positions/{self.contract}\n{query_string}\n{body_hash}\n{timestamp}"
            
            signature = hmac.new(
                self.api_config.get_api_secret().encode('utf-8'),
                sign_string.encode('utf-8'),
                hashlib.sha512
            ).hexdigest()
            
            headers = {
                'KEY': self.api_config.get_api_key(),
                'SIGN': signature,
                'Timestamp': timestamp
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                
                pos = None
                if isinstance(result, list):
                    pos = result[0] if result else None
                elif isinstance(result, dict):
                    pos = result
                
                if pos:
                    size = float(pos.get('size', 0))
                    
                    if size != 0:
                        return {
                            'size': abs(size),
                            'side': 'long' if size > 0 else 'short',
                            'entry_price': float(pos.get('entry_price', 0)),
                            'unrealized_pnl': float(pos.get('unrealised_pnl', 0))
                        }
            
            return {'size': 0.0, 'side': None, 'entry_price': 0, 'unrealized_pnl': 0}
            
        except Exception as e:
            if 'SSL' not in str(e):
                self.logger.log(f"获取仓位失败: {e}")
            return {'size': 0.0, 'side': None, 'entry_price': 0, 'unrealized_pnl': 0}

    def get_balance(self) -> Dict:
        """获取账户余额信息"""
        try:
            url = f"{self.api_config.get_base_url()}/api/v4/futures/{self.settle}/accounts"
            timestamp = str(int(time.time()))
            
            query_string = ""
            body_hash = hashlib.sha512("".encode('utf-8')).hexdigest()
            sign_string = f"GET\n/api/v4/futures/{self.settle}/accounts\n{query_string}\n{body_hash}\n{timestamp}"
            
            signature = hmac.new(
                self.api_config.get_api_secret().encode('utf-8'),
                sign_string.encode('utf-8'),
                hashlib.sha512
            ).hexdigest()
            
            headers = {
                'KEY': self.api_config.get_api_key(),
                'SIGN': signature,
                'Timestamp': timestamp
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                
                if result:
                    return {
                        'total': float(result.get('total', 0)),
                        'available': float(result.get('available', 0)),
                        'equity': float(result.get('total', 0)),  # equity = total
                        'unrealized_pnl': float(result.get('unrealised_pnl', 0)),
                        'position_margin': float(result.get('position_margin', 0)),
                        'order_margin': float(result.get('order_margin', 0))
                    }
            
            return {
                'total': 0,
                'available': 0,
                'equity': 0,
                'unrealized_pnl': 0,
                'position_margin': 0,
                'order_margin': 0
            }
            
        except Exception as e:
            if 'SSL' not in str(e):
                self.logger.log(f"获取余额失败: {e}")
            return {
                'total': 0,
                'available': 0,
                'equity': 0,
                'unrealized_pnl': 0,
                'position_margin': 0,
                'order_margin': 0
            }
# ======================================================
# 订单管理器
# ======================================================
class OrderManager:
    def __init__(self, settle: str, contract: str, dry_run: bool, logger: Logger, api_config: APIConfig):
        self.settle = settle
        self.contract = contract
        self.dry_run = dry_run
        self.logger = logger
        self.api_config = api_config
        self.last_order_time = {}
        self.order_cooldown = 2
        
        # 🔑 问题2修复: 改用订单ID作为key,而不是价格
        self.pending_orders = set()  # 记录活跃的订单ID
        self.active_close_order = None  # 当前活跃的平仓订单价格
        self.active_open_order = None   # 当前活跃的开仓订单价格
    
    def get_active_orders(self) -> List[Dict]:
        """获取所有活跃订单"""
        try:
            url = f"{self.api_config.get_base_url()}/api/v4/futures/{self.settle}/orders"
            timestamp = str(int(time.time()))
            
            query_string = f"contract={self.contract}&status=open"
            body_hash = hashlib.sha512("".encode('utf-8')).hexdigest()
            sign_string = f"GET\n/api/v4/futures/{self.settle}/orders\n{query_string}\n{body_hash}\n{timestamp}"
            
            signature = hmac.new(
                self.api_config.get_api_secret().encode('utf-8'),
                sign_string.encode('utf-8'),
                hashlib.sha512
            ).hexdigest()
            
            headers = {
                'KEY': self.api_config.get_api_key(),
                'SIGN': signature,
                'Timestamp': timestamp
            }
            
            params = {'contract': self.contract, 'status': 'open'}
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                return response.json()
            return []
            
        except Exception as e:
            if 'SSL' not in str(e):
                self.logger.log(f"获取订单失败: {e}")
            return []
    
    def sync_orders(self):
        """🔑 同步交易所的真实订单状态"""
        try:
            active_orders = self.get_active_orders()
            
            # 重置本地状态
            self.active_close_order = None
            self.active_open_order = None
            
            for order in active_orders:
                price = float(order.get('price', 0))
                size = int(order.get('size', 0))
                
                if size < 0:  # 平仓订单
                    self.active_close_order = price
                elif size > 0:  # 开仓订单
                    self.active_open_order = price
            
            if self.active_close_order:
                self.logger.log(f"🔄 同步到活跃平仓订单: {self.active_close_order}")
            if self.active_open_order:
                self.logger.log(f"🔄 同步到活跃开仓订单: {self.active_open_order}")
                
        except Exception as e:
            self.logger.log(f"同步订单失败: {e}")
    
    def place_order(self, contract: str, side: str, price: float, size: float, 
                   is_close: bool = False, entry_price: float = 0) -> Dict:
        """下单 - Gate.io单向持仓模式"""
        
        # 🔑 问题2修复: 检查是否已有同类型订单
        if is_close and self.active_close_order is not None:
            self.logger.log(f"⏸️ 已有平仓订单@{self.active_close_order},跳过")
            return {'status': 'already_pending'}
        
        if not is_close and self.active_open_order is not None:
            self.logger.log(f"⏸️ 已有开仓订单@{self.active_open_order},跳过")
            return {'status': 'already_pending'}
        
        # 冷却检查
        key = f"{price}"
        now = time.time()
        if key in self.last_order_time:
            if now - self.last_order_time[key] < self.order_cooldown:
                return {'status': 'cooldown'}
        
        if self.dry_run:
            action = "平多" if is_close else "开多"
            self.logger.log(f"[模拟] {action} {size} @ {price}")
            return {'status': 'simulated', 'id': 'dry_run'}
        
        try:
            url = f"{self.api_config.get_base_url()}/api/v4/futures/{self.settle}/orders"
            timestamp = str(int(time.time()))
            
            # Gate.io API规则
            if is_close:
                order_size = -int(size)
                action = "平多"
                reduce_only = True
            else:
                order_size = int(size)
                action = "开多"
                reduce_only = False
            
            order_data = {
                'contract': self.contract,
                'size': order_size,
                'price': str(price),
                'tif': 'gtc',
                'text': 't-grid',
                'reduce_only': reduce_only
            }
            
            body = json.dumps(order_data)
            
            query_string = ""
            body_hash = hashlib.sha512(body.encode('utf-8')).hexdigest()
            sign_string = f"POST\n/api/v4/futures/{self.settle}/orders\n{query_string}\n{body_hash}\n{timestamp}"
            
            signature = hmac.new(
                self.api_config.get_api_secret().encode('utf-8'),
                sign_string.encode('utf-8'),
                hashlib.sha512
            ).hexdigest()
            
            headers = {
                'KEY': self.api_config.get_api_key(),
                'SIGN': signature,
                'Timestamp': timestamp,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, headers=headers, data=body)
            
            if response.status_code == 201:
                result = response.json()
                order_id = result.get('id')
                self.last_order_time[key] = now
                
                # 🔑 更新活跃订单状态
                if is_close:
                    self.active_close_order = price
                else:
                    self.active_open_order = price
                
                # 🔑 问题3修复: 计算含手续费的盈亏
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                coin = self.contract.split('_')[0]
                
                if is_close:
                    # 计算盈亏 = (平仓价 - 成本价) * 数量 - 手续费
                    gross_pnl = (price - entry_price) * abs(order_size) if entry_price > 0 else 0
                    open_fee = entry_price * abs(order_size) * cfg.MAKER_FEE  # 开仓手续费
                    close_fee = price * abs(order_size) * cfg.TAKER_FEE       # 平仓手续费
                    net_pnl = gross_pnl - open_fee - close_fee
                    
                    msg = (
                        f"【{coin}币种网格策略通知】\n"
                        f"✅ 平多成功\n"
                        f"币种: {coin}\n"
                        f"价格: {price}\n"
                        f"数量: {abs(order_size)} 张\n"
                        f"方向: 平多\n"
                        f"成交时间: {current_time}\n"
                        f"盈亏: {net_pnl:.4f} USDT (已扣手续费)"
                    )
                else:
                    msg = (
                        f"【{coin}币种网格策略通知】\n"
                        f"✅ 开多委托成功\n"
                        f"币种: {coin}\n"
                        f"价格: {price}\n"
                        f"数量: {abs(order_size)} 张\n"
                        f"方向: 开多\n"
                        f"委托时间: {current_time}"
                    )
                
                self.logger.log(msg, notify=True)
                
                return {'status': 'success', 'id': order_id, 'info': result}
            else:
                error_data = response.json()
                error_msg = error_data.get('message', response.text)
                
                # 只记录非余额不足的错误
                if error_data.get('label') != 'INSUFFICIENT_AVAILABLE':
                    self.logger.log(f"❌ {action}失败: {error_msg}")
                
                return {'status': 'error', 'error': response.text}
                
        except Exception as e:
            if 'SSL' not in str(e):
                self.logger.log(f"❌ 下单异常: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def cancel_order(self, order_id: str):
        """取消指定订单"""
        try:
            url = f"{self.api_config.get_base_url()}/api/v4/futures/{self.settle}/orders/{order_id}"
            timestamp = str(int(time.time()))
            
            query_string = ""
            body_hash = hashlib.sha512("".encode('utf-8')).hexdigest()
            sign_string = f"DELETE\n/api/v4/futures/{self.settle}/orders/{order_id}\n{query_string}\n{body_hash}\n{timestamp}"
            
            signature = hmac.new(
                self.api_config.get_api_secret().encode('utf-8'),
                sign_string.encode('utf-8'),
                hashlib.sha512
            ).hexdigest()
            
            headers = {
                'KEY': self.api_config.get_api_key(),
                'SIGN': signature,
                'Timestamp': timestamp
            }
            
            response = requests.delete(url, headers=headers)
            
            if response.status_code == 200:
                self.logger.log(f"✅ 已取消订单 {order_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.log(f"取消订单失败: {e}")
            return False
    
    def cancel_all_orders(self, contract: str):
        """取消所有挂单"""
        if self.dry_run:
            return
        
        try:
            url = f"{self.api_config.get_base_url()}/api/v4/futures/{self.settle}/orders"
            timestamp = str(int(time.time()))
            
            query_string = f"contract={self.contract}"
            body_hash = hashlib.sha512("".encode('utf-8')).hexdigest()
            sign_string = f"DELETE\n/api/v4/futures/{self.settle}/orders\n{query_string}\n{body_hash}\n{timestamp}"
            
            signature = hmac.new(
                self.api_config.get_api_secret().encode('utf-8'),
                sign_string.encode('utf-8'),
                hashlib.sha512
            ).hexdigest()
            
            headers = {
                'KEY': self.api_config.get_api_key(),
                'SIGN': signature,
                'Timestamp': timestamp
            }
            
            params = {'contract': self.contract}
            response = requests.delete(url, headers=headers, params=params)
            
            if response.status_code == 200:
                result = response.json()
                cancelled_count = len(result) if isinstance(result, list) else 0
                self.logger.log(f"✅ 已取消 {cancelled_count} 个挂单")
                self.active_close_order = None
                self.active_open_order = None
            else:
                self.logger.log(f"⚠️ 取消挂单响应: {response.text}")
                
        except Exception as e:
            self.logger.log(f"❌ 取消挂单失败: {e}")

# ======================================================
# 网格策略
# ======================================================
class LongGridStrategy:
    def __init__(self, symbol: str, cfg: Config, order_mgr: OrderManager,
                 pos_mgr: PositionManager, price_feed: PriceFeed, logger: Logger):
        self.symbol = symbol
        self.low = cfg.PRICE_LOW
        self.high = cfg.PRICE_HIGH
        self.num = cfg.GRID_NUM
        self.qty = cfg.GRID_QTY
        self.order_mgr = order_mgr
        self.pos_mgr = pos_mgr
        self.price_feed = price_feed
        self.logger = logger
        
        self.vol_window = cfg.VOL_WINDOW
        self.atr_period = cfg.ATR_PERIOD
        self.ma_short = cfg.MA_SHORT
        self.ma_long = cfg.MA_LONG
        self.trend_threshold = cfg.TREND_THRESHOLD
        self.cci_period = cfg.CCI_PERIOD
        self.freeze_in_downtrend = cfg.FREEZE_IN_DOWNTREND
        
        self.base_step = (self.high - self.low) / self.num
        self.grid_prices = self.build_grid(self.base_step)
        
        self.price_history = []
        self.vol_history = []
        self.atr_history = []
        self.tr_history = []
        self.cci_history = []
        
        self.last_trend = "RANGE"
        self.last_position_check = 0
        self.last_pos_log = 0
        self.last_sync = 0  # 上次同步订单时间
        
        self.grid_opened = {}
        self.last_position_size = 0  # 🔑 问题1修复: 记录上次仓位大小
    
    def log(self, msg: str, notify: bool = False):
        self.logger.log(msg, notify)
    
    def build_grid(self, step: float) -> List[float]:
        prices = []
        p = self.low
        while p <= self.high:
            prices.append(round(p, 2))
            p += step
        return prices
    
    def get_grid_index(self, price: float) -> int:
        """获取价格所在的网格索引"""
        if price < self.low:
            return 0
        if price > self.high:
            return self.num - 1
        
        index = int((price - self.low) / self.base_step)
        return min(index, self.num - 1)
    
    def calc_std_vol(self) -> Optional[float]:
        if len(self.price_history) < self.vol_window:
            return None
        data = self.price_history[-self.vol_window:]
        if len(data) < 2:
            return None
        try:
            return statistics.stdev(data)
        except:
            return None
    
    def calc_atr(self) -> Optional[float]:
        if len(self.tr_history) < self.atr_period:
            return None
        data = self.tr_history[-self.atr_period:]
        return statistics.mean(data) if data else None
    
    def calc_cci(self) -> Optional[float]:
        if len(self.price_history) < self.cci_period:
            return None
        data = self.price_history[-self.cci_period:]
        tp = statistics.mean(data)
        sma = tp
        mad = statistics.mean([abs(x - sma) for x in data])
        if mad == 0:
            return 0
        return (tp - sma) / (0.015 * mad)
    
    def detect_trend(self, cci_value: Optional[float]) -> str:
        if len(self.price_history) < self.ma_long:
            return "RANGE"
        fast = statistics.mean(self.price_history[-self.ma_short:])
        slow = statistics.mean(self.price_history[-self.ma_long:])
        if slow == 0:
            return "RANGE"
        diff = (fast - slow) / slow
        if diff > self.trend_threshold and (cci_value is None or cci_value >= 0):
            return "UP"
        elif diff < -self.trend_threshold and (cci_value is None or cci_value <= 0):
            return "DOWN"
        else:
            return "RANGE"

    def on_tick(self):
        price = self.price_feed.get_price(self.symbol)
        if price is None:
            return

        try:
            price = float(price)
        except:
            return

        # 更新价格历史
        if self.price_history:
            prev_price = self.price_history[-1]
            tr = abs(price - prev_price)
            self.tr_history.append(tr)
            if len(self.tr_history) > self.atr_period * 5:
                self.tr_history.pop(0)
        
        self.price_history.append(price)
        max_len = max(self.vol_window, self.ma_long, self.cci_period) * 5
        if len(self.price_history) > max_len:
            self.price_history.pop(0)

        # 计算指标
        vol_now = self.calc_std_vol()
        atr_now = self.calc_atr()

        if vol_now:
            self.vol_history.append(vol_now)
            if len(self.vol_history) > 500:
                self.vol_history.pop(0)

        if atr_now:
            self.atr_history.append(atr_now)
            if len(self.atr_history) > 500:
                self.atr_history.pop(0)

        # 动态网格调整
        ratio_eff = 1.0
        if vol_now and len(self.vol_history) > 30:
            vol_avg = statistics.mean(self.vol_history)
            if vol_avg > 1e-10:
                ratio_vol = vol_now / vol_avg
                ratio_eff = ratio_vol

        if atr_now and len(self.atr_history) > 30:
            atr_avg = statistics.mean(self.atr_history)
            if atr_avg > 1e-10:
                ratio_atr = atr_now / atr_avg
                ratio_eff = math.sqrt(ratio_eff * ratio_atr) if ratio_eff != 1.0 else ratio_atr

        ratio_eff = max(0.5, min(2.5, ratio_eff))

        if self.base_step > 0:
            new_step = self.base_step * ratio_eff
            self.grid_prices = self.build_grid(new_step)

        # CCI和趋势
        cci_val = self.calc_cci()
        if cci_val is not None:
            self.cci_history.append(cci_val)
            if len(self.cci_history) > 500:
                self.cci_history.pop(0)

        trend = self.detect_trend(cci_val)
        if trend != self.last_trend:
            cci_str = f"{cci_val:.2f}" if cci_val is not None else "N/A"
            self.log(f"📊 趋势变化: {self.last_trend} → {trend} (CCI={cci_str})")
            self.last_trend = trend

        # 冻结条件(只保留趋势冻结)
        freeze_long = False
        if self.freeze_in_downtrend and trend == "DOWN":
            freeze_long = True

        # 获取仓位
        now = time.time()
        if now - self.last_position_check < 5:
            return
        self.last_position_check = now
        
        # 🔑 定期同步订单状态(每30秒)
        if now - self.last_sync > 30:
            self.order_mgr.sync_orders()
            self.last_sync = now
        
        pos = self.pos_mgr.get_position(self.symbol)
        size = pos.get("size", 0.0) if pos else 0.0
        entry_price = pos.get("entry_price", 0) if pos else 0
        
        # 🔑 问题1修复: 检测仓位变化,清除对应的挂单状态
        if size != self.last_position_size:
            if size == 0 and self.last_position_size > 0:
                # 仓位被平掉,清除平仓订单状态
                self.order_mgr.active_close_order = None
                self.log(f"✅ 仓位已平,清除平仓订单状态")
            elif size > 0 and self.last_position_size == 0:
                # 新开仓位,清除开仓订单状态
                self.order_mgr.active_open_order = None
                self.log(f"✅ 已开新仓,清除开仓订单状态")
            
            self.last_position_size = size
        
        # 定期输出仓位
        if now - self.last_pos_log > 60:
            if size > 0:
                coin = self.order_mgr.contract.split('_')[0]
                msg = (
                    f"【{coin}币种网格策略通知】\n"
                    f"📍 当前持仓信息\n"
                    f"币种: {coin}\n"
                    f"持仓数量: {size} 张\n"
                    f"成本价格: {entry_price:.2f}\n"
                    f"浮动盈亏: {pos.get('unrealized_pnl', 0):.4f} USDT"
                )
                self.log(msg, notify=True)
            self.last_pos_log = now

        # 根据持仓状态重置网格
        if size == 0:
            freeze_long = False
            self.grid_opened.clear()
        else:
            current_grid = self.get_grid_index(price)
            grids_to_clear = [g for g in self.grid_opened.keys() if g < current_grid]
            for g in grids_to_clear:
                del self.grid_opened[g]

        valid_grids = [p for p in self.grid_prices if self.low <= p <= self.high]
        if not valid_grids:
            return

        buy_levels = [p for p in valid_grids if p < price]
        buy_price = buy_levels[-1] if buy_levels else valid_grids[0]
        
        sell_levels = [p for p in valid_grids if p > price]

        # 核心逻辑
        if size > 0:
            # 有多仓 → 平仓
            if sell_levels:
                take_price = sell_levels[0]
                
                # 🔑 问题1修复: 确保平仓价格高于成本价
                if take_price <= entry_price:
                    self.log(f"⚠️ 平仓价{take_price}低于成本价{entry_price},等待更好价格")
                    return
                
                self.log(f"📉 [平仓逻辑] 持仓{size}张 成本{entry_price:.2f} → 挂sell单@{take_price} 平仓")
                self.order_mgr.place_order(
                    contract=self.symbol,
                    side="sell",
                    price=take_price,
                    size=size,
                    is_close=True,
                    entry_price=entry_price
                )
        else:
            # 无仓位 → 开仓
            if not freeze_long and price <= buy_price * 1.0005:
                buy_grid_index = self.get_grid_index(buy_price)
                
                if buy_grid_index not in self.grid_opened:
                    self.log(f"📈 [开仓逻辑] 网格{buy_grid_index} → 挂buy单@{buy_price} 开多")
                    result = self.order_mgr.place_order(
                        contract=self.symbol,
                        side="buy",
                        price=buy_price,
                        size=self.qty,
                        is_close=False
                    )
                    
                    if result.get('status') == 'success':
                        self.grid_opened[buy_grid_index] = True
                        self.log(f"✅ 网格{buy_grid_index}已标记为已开仓")

    def run(self):
        self.on_tick()

# ======================================================
# 主程序入口
# ======================================================
async def main():
    # 初始化API配置
    api_config = APIConfig(testnet=cfg.TESTNET)
    
    notifier = WeComNotifier(cfg.WECOM_WEBHOOK, cfg.ENABLE_WECOM)
    logger = Logger(notifier)
    
    mode_str = "🧪 模拟盘" if cfg.TESTNET else "💰 实盘"
    
    logger.log("=" * 60)
    logger.log(f"🚀 Gate.io 网格策略启动 ({mode_str})", notify=True)
    logger.log(f"交易对: {cfg.SYMBOL}")
    logger.log(f"网格范围: [{cfg.PRICE_LOW}, {cfg.PRICE_HIGH}]")
    logger.log(f"网格数量: {cfg.GRID_NUM}")
    logger.log(f"每格数量: {cfg.GRID_QTY}")
    logger.log(f"杠杆: {cfg.LEVERAGE}x")
    logger.log(f"开仓手续费: {cfg.MAKER_FEE*100}%")
    logger.log(f"平仓手续费: {cfg.TAKER_FEE*100}%")
    logger.log(f"本地模拟: {cfg.DRY_RUN}")
    logger.log("=" * 60)
    
    # 设置杠杆
    try:
        url = f"{api_config.get_base_url()}/api/v4/futures/{cfg.SETTLE}/positions/{cfg.CONTRACT}/leverage"
        timestamp = str(int(time.time()))
        
        query_params = {
            'leverage': '0',
            'cross_leverage_limit': str(cfg.LEVERAGE)
        }
        query_string = f"leverage=0&cross_leverage_limit={cfg.LEVERAGE}"
        
        body_hash = hashlib.sha512("".encode('utf-8')).hexdigest()
        sign_string = f"POST\n/api/v4/futures/{cfg.SETTLE}/positions/{cfg.CONTRACT}/leverage\n{query_string}\n{body_hash}\n{timestamp}"
        
        signature = hmac.new(
            api_config.get_api_secret().encode('utf-8'),
            sign_string.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()
        
        headers = {
            'KEY': api_config.get_api_key(),
            'SIGN': signature,
            'Timestamp': timestamp
        }
        
        response = requests.post(url, headers=headers, params=query_params)
        if response.status_code == 200:
            logger.log(f"✅ 全仓模式设置成功: {cfg.LEVERAGE}x")
        else:
            logger.log(f"⚠️ 杠杆设置响应: {response.text}")
                
    except Exception as e:
        logger.log(f"❌ 设置杠杆失败: {e}")
    
    # 初始化组件
    price_feed = PriceFeed(cfg.CONTRACT, logger, api_config)
    pos_mgr = PositionManager(cfg.SETTLE, cfg.CONTRACT, logger, api_config)
    order_mgr = OrderManager(cfg.SETTLE, cfg.CONTRACT, cfg.DRY_RUN, logger, api_config)
    
    price_feed.start()
    await asyncio.sleep(2)
    
    # 启动时同步订单
    order_mgr.sync_orders()
    
    strategy = LongGridStrategy(
        symbol=cfg.SYMBOL,
        cfg=cfg,
        order_mgr=order_mgr,
        pos_mgr=pos_mgr,
        price_feed=price_feed,
        logger=logger
    )
    
    logger.log("✅ 策略初始化完成,开始运行...", notify=True)
    
    try:
        while True:
            strategy.run()
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.log("⏹️ 收到停止信号...", notify=True)
    except Exception as e:
        logger.log(f"❌ 策略异常: {e}", notify=True)
    finally:
        await price_feed.stop()
        order_mgr.cancel_all_orders(cfg.SYMBOL)
        logger.log("✅ 策略已停止", notify=True)

if __name__ == "__main__":
    asyncio.run(main())