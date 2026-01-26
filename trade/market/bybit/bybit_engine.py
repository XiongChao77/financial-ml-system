import time, json, logging, threading, contextlib, os, sys, websocket,uuid
from pybit.unified_trading import HTTP, WebSocket, WebSocketTrading
from websocket import WebSocketConnectionClosedException 

class BybitEngine:
    """
    Bybit V5 通用量化基座 (V8.1)
    集成 HTTP(配置/查询) + WS_Trade(交易) + 环境自愈逻辑
    """
    def __init__(self, api_key_path, hmac_secret_path, rsa_key_path, rsa_pem_path, testnet=False):
        self.api_key = self._load_key(api_key_path)
        self.hmac_secret = self._load_key(hmac_secret_path)
        self.rsa_key = self._load_key(rsa_key_path)
        self.rsa_pem_path = self._load_key(rsa_pem_path)
        self.testnet = testnet
        self.category = "linear"
        
        # 1. HTTP 客户端 (配置中心)
        self.http = HTTP(testnet=testnet, api_key=self.api_key, api_secret=self.hmac_secret,timeout=10,)
        
        # 2. WS 交易客户端
        self.ws_trade = WebSocketTrading(testnet=testnet, api_key=self.api_key, api_secret=self.hmac_secret)
        
        # 3. WS 数据流客户端
        self.ws_stream = WebSocket(
            testnet=testnet, channel_type="private", 
            rsa_authentication=True, api_key=self.rsa_key, api_secret=self.rsa_pem_path
        )

    def _load_key(self, path):
        try: return open(path, 'r').read().strip()
        except: return ""

    # --- 新增：环境配置逻辑 ---

    def setup_hedge_env(self, symbol, leverage="10"):
        """
        一键初始化交易环境：设置双向持仓 + 调整杠杆
        """
        print(f"🛠️  正在为 {symbol} 配置双向对冲环境...")
        
        # 1. 尝试切换为双向持仓模式 (mode=3)
        # 注意：如果有持仓或挂单，此操作会报错
        try:
            res = self.http.switch_position_mode(
                category=self.category,
                symbol=symbol,
                mode=3 
            )
            if res.get('retCode') == 0:
                print(f"   ✅ 模式切换成功：双向持仓已开启")
            elif res.get('retCode') == 110025:
                print(f"   ℹ️ 模式确认：已经是双向持仓模式")
        except Exception as e:
            print(f"   ⚠️ 模式切换异常: {e}")

        # 2. 设置杠杆
        try:
            self.http.set_leverage(
                category=self.category,
                symbol=symbol,
                buyLeverage=leverage,
                sellLeverage=leverage
            )
            print(f"   ✅ 杠杆设置成功：{leverage}x")
        except Exception as e:
            # 如果杠杆没变动，Bybit会抛出异常，通常可以直接忽略
            pass

    def set_leverage(self, symbol, leverage):
        """
        智能杠杆设置：先检查，不一致才修改
        leverage: 目标杠杆，如 "5"
        """
        try:
            # 1. 获取当前持仓/风险限额信息
            # 注意：即使没有持仓，这个接口也会返回该币种默认的杠杆设置
            pos_res = self.http.get_positions(category=self.category, symbol=symbol)
            
            if pos_res.get('retCode') == 0 and pos_res['result']['list']:
                # 获取当前第一条记录（通常是 POS 0 或 POS 1）的杠杆
                current_lev = pos_res['result']['list'][0].get('leverage', "0")
                
                # 2. 比较：如果已经一致，直接跳过
                if str(current_lev) == str(leverage):
                    # self.logger.debug(f"ℹ️ [{symbol}] 杠杆已是 {leverage}x，跳过设置")
                    return True

            # 3. 不一致时才执行设置
            res = self.http.set_leverage(
                category=self.category,
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            
            if res.get('retCode') == 0:
                print(f"✅ [{symbol}] 杠杆成功更新为 {leverage}x")
            elif res.get('retCode') == 110043:
                # 兜底处理：万一查询时有延迟，这里依然捕获不修改的错误
                pass 
            else:
                print(f"⚠️ [{symbol}] 杠杆设置失败: {res.get('retMsg')}")
                
        except Exception as e:
            print(f"❌ [{symbol}] 检查/设置杠杆发生异常: {e}")

    # --- 核心交易逻辑 ---

    def place_order(self, symbol, side, qty, price, link_id, 
                    order_type="Limit", pos_idx=0, is_reduce=False, callback=None):
        """
        通过 WebSocket 发送订单：修复 pybit 位置参数缺失问题
        """
        # 如果调用方没有传 callback，我们需要给一个默认的空函数，防止报错
        if callback is None:
            callback = lambda response: None 

        order_params = {
            "category": self.category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "orderLinkId": link_id,
            "positionIdx": pos_idx,
            "reduceOnly": is_reduce,
        }

        if order_type == "Limit":
            order_params["price"] = str(price)
            order_params["timeInForce"] = "PostOnly"
        else:
            order_params["timeInForce"] = "GTC"

        try:
            #  关键修复：将 callback 作为第一个位置参数传入
            self.ws_trade.place_order(callback, **order_params)
            
        except Exception as e:
            self.logger.error(f"❌ WebSocket 下单异常: {e}")

    def start_stream(self, order_callback):
        self.ws_stream.order_stream(callback=order_callback)

    def cancel_all_http(self, symbol):
        """
        通过 HTTP 撤销所有挂单：增强健壮性版
        """
        try:
            #  修复点：明确 category 并增加 settleCoin 辅助判定
            # 对于 USDT 永续合约，settleCoin 必须是 "USDT"
            return self.http.cancel_all_orders(
                category=self.category, # 确保是 "linear"
                symbol=symbol,
                settleCoin="USDT"       # 增加此参数通常能解决 110074 错误
            )
        except Exception as e:
            print(f"❌ [{symbol}] HTTP 全撤指令发生异常: {e}")
            return {"retCode": -1, "retMsg": str(e)}
    
    def stop(self):
        try:
            if hasattr(self, 'ws_stream'): self.ws_stream.exit()
            if hasattr(self, 'ws_trade'): self.ws_trade.exit()
        except WebSocketConnectionClosedException: pass
        print("engine safety close")

# --- 线程异常静默处理 (保持不变) ---
def silent_thread_exception_handler(args):
    if args.exc_type == websocket._exceptions.WebSocketConnectionClosedException:
        return
    sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)

threading.excepthook = silent_thread_exception_handler