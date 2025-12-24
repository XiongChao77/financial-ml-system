import os, sys, logging, time, uuid, json, base64, requests, threading
from Crypto.Hash import SHA256
from Crypto.Signature import PKCS1_v1_5
from Crypto.PublicKey import RSA

# -----------------------------------------------------------------------------
# 配置区域
# -----------------------------------------------------------------------------
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..' , '..'))

# 尝试导入 common，如果没有则使用模拟 logger
try:
    from data_process import common
except ImportError:
    class MockLogger:
        def info(self, msg): print(f"[INFO] {msg}")
        def warning(self, msg): print(f"[WARN] {msg}")
        def error(self, msg): print(f"[ERROR] {msg}")
    common = type('obj', (object,), {'setup_session_logger': lambda **kwargs: (MockLogger(), None)})

class SequentialGridBot_V5_3:
    def __init__(self, api_key_path, private_key_path, symbol_configs):
        self.logger, _ = common.setup_session_logger(sub_folder='grid_v5_3', console_level=logging.INFO, file_level=logging.INFO)
        self.url = "https://api.bybit.com"
        self.api_key = self._load_key(api_key_path)
        self.private_key_path = private_key_path
        self.category = "linear"
        self.session = requests.Session()
        
        # 基础状态
        self.fee_rate = 0.0002 # Taker/Maker 混合预估 (保守)
        self.start_balance = 0
        self.current_balance = 0
        self.total_profit = 0.0
        self.start_time = time.time()
        self.stop_new_pairs = False 
        
        self.markets = {}
        # 交易统计 (修复之前报错的 bug)
        self.trade_stats = {s: 0 for s in symbol_configs.keys()}

        # 初始化配置
        for symbol, cfg in symbol_configs.items():
            self.markets[symbol] = {
                "budget_pct": cfg.get("budget_pct", 0.1),
                "max_layers": cfg.get("max_layers", 5),
                "qty_step": cfg.get("qty_step", 0),
                "base_offset": cfg.get("base_offset", 0.003),
                "start_price_gap": 0.0005, # 0.05% 启动间距
                
                # 动态数据
                "grid_nodes": [], 
                "node_counter": 0,
                "tick_size": 0.0001, # 默认值，稍后更新
                "min_qty": 1,
                "market_state": "OSCILLATION", # 默认假设震荡
                "last_check_time": 0
            }
            self.set_hedge_mode(symbol)
            time.sleep(0.2)
        
        self.logger.info("🚀 V5.3 终极网格启动 | ID标记恢复 | 趋势熔断 | 利润保底")
        
        # 🌟 初始化三部曲
        self.get_wallet_balance() 
        self.start_balance = self.current_balance
        self.update_instrument_info()     # 1. 获取精度
        self.sync_state_from_exchange()   # 2. 恢复状态
        self.update_micro_market_status() # 3. 初始趋势分析

    def _load_key(self, path): return open(path, 'r').read().strip()
    
    def _gen_signature(self, ts, payload):
        pk = RSA.importKey(open(self.private_key_path, 'r').read())
        h = SHA256.new((str(ts) + self.api_key + "5000" + payload).encode("utf-8"))
        return base64.b64encode(PKCS1_v1_5.new(pk).sign(h)).decode()

    def request(self, method, endpoint, params=""):
        ts = str(int(time.time() * 1000))
        sig = self._gen_signature(ts, params)
        headers = {'X-BAPI-API-KEY':self.api_key,'X-BAPI-SIGN':sig,'X-BAPI-SIGN-TYPE':'2','X-BAPI-TIMESTAMP':ts,'X-BAPI-RECV-WINDOW':"5000",'Content-Type':'application/json'}
        url = self.url + endpoint + ("?" + params if method == "GET" else "")
        try:
            res = self.session.request(method, url, headers=headers, data=params if method == "POST" else None, timeout=5)
            return res.json()
        except Exception as e:
            self.logger.warning(f"⚠️ Req Error: {e}")
            return {"retCode": -1}

    def set_hedge_mode(self, symbol):
        """强制切换为双向持仓模式"""
        # mode: 3 代表双向 (Hedge), 0 代表单向 (One-way)
        res = self.request("POST", "/v5/position/switch-mode", json.dumps({
            "category": self.category, "symbol": symbol, "mode": 3 
        }))
        if res.get('retCode') == 0:
            self.logger.info(f"✅ [{symbol}] 已切换至双向持仓模式")

    # -----------------------------------------------------------------------------
    # ID 编解码与状态恢复 (核心功能)
    # -----------------------------------------------------------------------------
    def generate_order_link_id(self, symbol, node_id, side):
        """生成: V5_SymbolShort_NodeID_Side_Random"""
        short_sym = symbol.replace("USDT", "")
        uid = uuid.uuid4().hex[:4]
        side_code = "B" if side == "Buy" else "S"
        return f"V5_{short_sym}_{node_id}_{side_code}_{uid}"

    def parse_order_link_id(self, order_link_id):
        """解析: 返回 (valid, symbol_short, node_id, side_code)"""
        try:
            parts = order_link_id.split('_')
            if len(parts) != 5 or parts[0] != "V5": return False, None, None, None
            return True, parts[1], int(parts[2]), parts[3]
        except: return False, None, None, None

    def sync_state_from_exchange(self):
        """从链上挂单反向重构内存状态"""
        self.logger.info("🔗 正在扫描链上订单以恢复状态...")
        for symbol in self.markets:
            m = self.markets[symbol]
            m["grid_nodes"] = [] 
        
            res = self.request("GET", "/v5/order/realtime", f"category={self.category}&symbol={symbol}&limit=50")
            if res.get('retCode') != 0: continue
            
            recovered_nodes = {}
            active_orders = res['result']['list']
            
            for order in active_orders:
                oid = order['orderLinkId']
                valid, sym_short, node_id, side_code = self.parse_order_link_id(oid)
                
                # 过滤非本策略订单
                if not valid or sym_short != symbol.replace("USDT", ""): continue
                
                m["node_counter"] = max(m["node_counter"], node_id)
                
                if node_id not in recovered_nodes:
                    recovered_nodes[node_id] = {
                        "id": node_id, "qty": float(order['qty']),
                        "entry_price": 0.0, "entry_id": None, "exit_id": None,
                        "status": "UNKNOWN", "create_time": time.time()
                    }
                
                node = recovered_nodes[node_id]
                price = float(order['price'])

                if side_code == 'B':
                    node["status"] = "WAITING_ENTRY"
                    node["entry_id"] = oid
                    node["entry_price"] = price
                elif side_code == 'S':
                    node["status"] = "HOLDING"
                    node["exit_id"] = oid
                    # 反推买入价 (近似)
                    node["entry_price"] = price / (1 + m["base_offset"])

            for nid, node in recovered_nodes.items():
                m["grid_nodes"].append(node)
                icon = "🟢 等买" if node['status']=="WAITING_ENTRY" else "🔴 持仓"
                self.logger.info(f"   ♻️ [{symbol}] 恢复节点 #{nid}: {icon}")
            
            self.logger.info(f"   ✅ {symbol} 状态同步完成，当前层数: {len(m['grid_nodes'])}")

    # -----------------------------------------------------------------------------
    # 市场分析与风控
    # -----------------------------------------------------------------------------
    def update_instrument_info(self):
        self.logger.info("🔍 同步精度与利润体检...")
        res = self.request("GET", "/v5/market/instruments-info", f"category={self.category}")
        if res.get('retCode') == 0:
            all_info = {item['symbol']: item for item in res['result']['list']}
            for symbol in self.markets:
                if symbol in all_info:
                    info = all_info[symbol]
                    self.markets[symbol]["tick_size"] = float(info['priceFilter']['tickSize'])
                    self.check_profit_viability(symbol)

    def check_profit_viability(self, symbol):
        m = self.markets[symbol]
        res = self.request("GET", "/v5/market/tickers", f"category={self.category}&symbol={symbol}")
        if res.get('retCode') != 0: return
        last_price = float(res['result']['list'][0]['lastPrice'])
        
        gap = last_price * m["base_offset"]
        fee = last_price * (self.fee_rate * 2) # 双边费率
        net_pct = ((gap - fee) / last_price) * 100
        ticks = gap / m["tick_size"]
        
        self.logger.info(f"📊 [{symbol}] 体检: Offset={m['base_offset']*100:.2f}% | 净利预测={net_pct:.3f}% | Ticks={ticks:.1f}")
        if ticks < 4 or net_pct < 0.05:
            self.logger.warning(f"   ❌ {symbol} 利润过薄！请增加 base_offset！ ticks:{ticks} net_pct:{net_pct}")

    def update_micro_market_status(self):
        """趋势熔断逻辑: 动态分位数版 (旨在仅熔断约 20% 的极端波动时间)"""
        import math
        now = time.time()
        for symbol in self.markets:
            m = self.markets[symbol]
            #
            res = self.request("GET", "/v5/market/kline", f"category={self.category}&symbol={symbol}&interval=1&limit=100")
            if res.get('retCode') != 0 or not res['result']['list']: continue

            prices = [float(k[4]) for k in res['result']['list']]
            volumes = [float(k[5]) for k in res['result']['list']]
            prices.reverse() 

            # 1. 计算动态波动率 (标准差)
            n = len(prices)
            ma = sum(prices) / n
            # 计算方差与标准差
            variance = sum((p - ma) ** 2 for p in prices) / n
            std_dev = math.sqrt(variance)
            
            curr = prices[-1]
            # 计算当前价格偏离了多少个标准差 (Z-Score)
            # 💡 逻辑：只有当偏离度超过 1.28 sigma 时，才认为属于那 20% 的极端行情
            z_score = abs(curr - ma) / std_dev if std_dev > 0 else 0
            
            # 2. 筹码集中度 (简化版 VPVR)
            bin_count = 10
            min_p, max_p = min(prices), max(prices)
            interval = (max_p - min_p) / bin_count if max_p != min_p else 0.0001
            profile = [0] * bin_count
            for p, v in zip(prices, volumes):
                idx = min(int((p - min_p) / interval), bin_count - 1)
                profile[idx] += v
            concentration = max(profile) / sum(profile) if sum(profile) > 0 else 0
            
            # 3. 动态判定
            # z_score > 1.28 替代了死板的 0.003
            # concentration 阈值从 0.15 调低到 0.10，增加对轻微趋势的容忍度
            if z_score > 1.28 or concentration < 0.10:
                m["market_state"] = "TREND"
            else:
                m["market_state"] = "OSCILLATION"
            
            m["last_check_time"] = now
            self.logger.info(f"🔍 [{symbol}] 状态: {m['market_state']} (check_time:{now}, POC:{concentration:.2f})")

    # -----------------------------------------------------------------------------
    # 核心交易逻辑
    # -----------------------------------------------------------------------------
    def get_order_status(self, symbol, order_id):
        """主动查询订单最终状态，防止裸空"""
        # 优先查历史，因为如果 realtime 查不到可能是成交了也可能是撤单了
        res = self.request("GET", "/v5/order/history", f"category={self.category}&symbol={symbol}&orderLinkId={order_id}&limit=1")
        if res.get('retCode') == 0 and res['result']['list']:
            return res['result']['list'][0]['orderStatus'] # Filled, Cancelled, Rejected
        
        # 如果历史没查到，查一下 realtime (极少情况)
        res = self.request("GET", "/v5/order/realtime", f"category={self.category}&symbol={symbol}&orderLinkId={order_id}")
        if res.get('retCode') == 0 and res['result']['list']:
            return res['result']['list'][0]['orderStatus'] # New, PartiallyFilled
            
        return "Unknown"

    def create_grid_node(self, symbol, entry_price, is_initial=False):
        if self.stop_new_pairs: return
        m = self.markets[symbol]
        
        # 1. 熔断检查
        if m["market_state"] == "TREND": 
            return # 趋势中不接刀

        # 2. 严格层数限制
        if len(m["grid_nodes"]) >= m["max_layers"]: return

        # 3. 价格计算
        if is_initial:
            buy_price = entry_price * (1 - m["start_price_gap"])
        else:
            buy_price = entry_price * (1 - m["base_offset"])
            
        # 精度对齐
        ts = m["tick_size"]
        buy_price = round(buy_price / ts) * ts

        # 4. 数量计算
        self.get_wallet_balance()
        if self.current_balance <= 0: return
        budget = self.current_balance * m["budget_pct"] / m["max_layers"]
        raw_qty = budget / buy_price
        
        if m["qty_step"] == 0: qty = max(1, round(raw_qty))
        elif m["qty_step"] == -2: qty = max(100, round(raw_qty, -2))
        else: qty = round(raw_qty, m["qty_step"])

        # 5. 生成 ID 并下单
        m["node_counter"] += 1
        entry_id = self.generate_order_link_id(symbol, m["node_counter"], "Buy")
        
        res = self.request("POST", "/v5/order/create", json.dumps({
            "category": self.category, "symbol": symbol, "side": "Buy",
            "orderType": "Limit", "qty": str(qty), 
            "price": f"{buy_price:.5f}", "timeInForce": "PostOnly", 
            "orderLinkId": entry_id
        }))

        if res.get('retCode') == 0:
            m["grid_nodes"].append({
                "id": m["node_counter"], "status": "WAITING_ENTRY",
                "entry_id": entry_id, "exit_id": None,
                "entry_price": buy_price, "qty": qty, "create_time": time.time()
            })
            tag = "🚀 启动" if is_initial else "📉 补单"
            self.logger.info(f"➕ [{symbol}] {tag} #{m['node_counter']} | Price: {buy_price:.5f}")
        else:
            self.logger.warning(f"下单失败: {res}")

    def place_exit_order(self, symbol, node):
        m = self.markets[symbol]
        tick_size = m["tick_size"]
        
        # 1. 利润保底计算
        raw_sell = node["entry_price"] * (1 + m["base_offset"])
        min_profit_price = node["entry_price"] + (tick_size * 5) # 5个Tick
        fee_cover_price = node["entry_price"] * (1 + self.fee_rate * 2.5) # 费率覆盖
        
        safe_price = max(raw_sell, min_profit_price, fee_cover_price)
        final_price = round(safe_price / tick_size) * tick_size
        price_str = format(final_price, 'f').rstrip('0').rstrip('.')

        # 2. 生成 ID
        exit_id = self.generate_order_link_id(symbol, node["id"], "Sell")
        
        res = self.request("POST", "/v5/order/create", json.dumps({
            "category": self.category, "symbol": symbol, "side": "Sell",
            "orderType": "Limit", "qty": str(node["qty"]), 
            "price": price_str, "timeInForce": "PostOnly", 
            "orderLinkId": exit_id
        }))
        
        if res.get('retCode') == 0:
            node["exit_id"] = exit_id
            node["status"] = "HOLDING"
            self.logger.info(f"🔒 [{symbol}] 止盈挂单: {price_str}")
            return True
        return False

    def get_wallet_balance(self):
        res = self.request("GET", "/v5/account/wallet-balance", "accountType=UNIFIED&coin=USDT")
        if res.get('retCode') == 0:
            try:
                coin = res['result']['list'][0]['coin'][0]
                self.current_balance = float(coin['walletBalance'])
            except: pass
        return self.current_balance

    def get_all_tickers(self):
        res = self.request("GET", "/v5/market/tickers", "category=linear")
        if res.get('retCode') == 0:
            return {item['symbol']: float(item['lastPrice']) for item in res['result']['list']}
        return {}

    def clear_all_and_liquidate(self):
        self.stop_new_pairs = True
        self.logger.info("🚨 紧急全平启动...")
        for s in self.markets:
            self.request("POST", "/v5/order/cancel-all", json.dumps({"category": self.category, "symbol": s}))
            res = self.request("GET", "/v5/position/list", f"category={self.category}&symbol={s}")
            for p in res.get('result',{}).get('list',[]):
                if float(p['size']) > 0:
                    side = "Sell" if p['side'] == "Buy" else "Buy"
                    self.request("POST", "/v5/order/create", json.dumps({
                        "category": self.category, "symbol": s, "side": side,
                        "orderType": "Market", "qty": p['size'], "reduceOnly": True
                    }))
        self.logger.info("✅ 全平完成")

    def listen_keyboard(self):
        while True:
            cmd = input()
            if cmd == 'x': 
                if input("Confirm Exit (y/n)? ") == 'y': self.clear_all_and_liquidate()

    def run_loop(self):
        self.logger.info("🎬 开始主循环...")
        threading.Thread(target=self.listen_keyboard, daemon=True).start()
        
        last_check_tick = time.time()
        last_report_tick = time.time()
        
        while True:
            now = time.time()
            
            # A. 定时趋势分析 (120s)
            if now - last_check_tick > 120:
                self.update_micro_market_status()
                last_check_tick = now
            
            all_tickers = self.get_all_tickers()
            
            for symbol in self.markets:
                m = self.markets[symbol]
                curr_price = all_tickers.get(symbol)
                if not curr_price: continue

                # B. 获取实时订单 (Active Orders)
                res = self.request("GET", "/v5/order/realtime", f"category={self.category}&symbol={symbol}")
                on_chain_ids = set()
                if res.get('retCode') == 0:
                    on_chain_ids = {o['orderLinkId'] for o in res['result']['list']}

                # C. 状态机轮询
                for node in m["grid_nodes"][:]:
                    
                    # 状态 1: 等待买入
                    if node["status"] == "WAITING_ENTRY":
                        if node["entry_id"] not in on_chain_ids:
                            # ⚠️ 防裸空: 确认是真的成交了
                            status = self.get_order_status(symbol, node["entry_id"])
                            if status == "Filled":
                                if self.place_exit_order(symbol, node):
                                    # 尝试补下一层 (严格检查层数)
                                    if len(m["grid_nodes"]) < m["max_layers"]:
                                        self.create_grid_node(symbol, node["entry_price"], is_initial=False)
                            elif status in ["Cancelled", "Rejected"]:
                                self.logger.warning(f"⚠️ 买单被撤，移除节点 #{node['id']}")
                                m["grid_nodes"].remove(node)

                    # 状态 2: 持仓等卖
                    elif node["status"] == "HOLDING":
                        if node["exit_id"] not in on_chain_ids:
                            # 确认成交
                            status = self.get_order_status(symbol, node["exit_id"])
                            if status == "Filled":
                                profit = (node["entry_price"] * m["base_offset"]) * node["qty"]
                                self.total_profit += profit
                                self.trade_stats[symbol] += 1
                                self.logger.info(f"💰 [{symbol}] 止盈完成! +{profit:.4f} U")
                                m["grid_nodes"].remove(node)
                            elif status in ["Cancelled", "Rejected"]:
                                self.logger.warning(f"⚠️ 卖单被撤，请手动检查 #{node['id']}")
                                # 此时很危险，通常需要重新挂卖单，这里简单处理为报警
                
                # D. 启动逻辑 (仅当空仓且处于震荡期)
                if len(m["grid_nodes"]) == 0 and not self.stop_new_pairs:
                    self.create_grid_node(symbol, curr_price, is_initial=True)
                
                time.sleep(0.05) # 币种间微暂停

            # E. 战报
            if now - last_report_tick > 60:
                self.report_status(now)
                last_report_tick = now

            time.sleep(0.5) # 主循环暂停

    def report_status(self, now):
        # 🌟 改进后的简报
        # os.system('clear' if os.name == 'posix' else 'cls') # 可选：清屏让输出更整洁
        # 🌟 关键：打印前强制更新一次余额
        self.get_wallet_balance() 
        
        total_seconds = int(now - self.start_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60

        # 1. 账户实际余额增长 (包含已结利润)
        actual_growth = self.current_balance - self.start_balance
        
        self.logger.info(f"🎬 已运行: {hours}小时 {minutes}分钟")
        self.logger.info(f"💰 初始资金: {self.start_balance:.2f} | 当前可用余额: {self.current_balance:.2f}")
        # 2. 已实现净利润 (累加每对成交的 Net Profit)
        self.logger.info(f"📈 累计网格净利: {self.total_profit:+.4f} USDT (扣除手续费)")
        # 3. 账户增长情况 (注意：受挂单保证金占用影响)
        self.logger.info(f"📊 账户余额变动: {actual_growth:+.4f} USDT")
        self.logger.info("-" * 50)

        for s, m in self.markets.items():
            wins = self.trade_stats[s]
            self.logger.info(f"💰 {s:12} | 获利次数: {wins:3} | budget_pct: {m['budget_pct']:2f} | base_offset: {m['base_offset']:4f}")
        self.logger.info("-" * 50)

if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    API_K, PRI_K = os.path.join(BASE, "keys", "api_key"), os.path.join(BASE, "keys", "bybit_rsa.pem")
    
    # 🌟 终极配置：Offset 必须给足，防止精度亏损
    CONFIGS = {
        "DOGEUSDT": {"budget_pct": 0.2, "max_layers": 5, "base_offset": 0.00035, "qty_step": 0},
        "ARCUSDT":  {"budget_pct": 0.1, "max_layers": 5, "base_offset": 0.001, "qty_step": 0}, 
        "ASTRUSDT": {"budget_pct": 0.1, "max_layers": 5, "base_offset": 0.001, "qty_step": -2}, 
        "APEXUSDT": {"budget_pct": 0.1, "max_layers": 5, "base_offset": 0.001, "qty_step": 0}, 
        "1000RATSUSDT": {"budget_pct": 0.1, "max_layers": 5, "base_offset": 0.003, "qty_step": -2},
    }

    bot = SequentialGridBot_V5_3(API_K, PRI_K, CONFIGS)
    bot.run_loop()