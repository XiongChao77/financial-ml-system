import time, threading, math, logging, sys, os, signal
from dataclasses import dataclass, field
from enum import Enum
import argparse
from typing import Dict
from bybit_engine import BybitEngine
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..' , '..'))
from data_process.common import setup_session_logger
from collections import OrderedDict

class OrderLabel:
    ENTRY = "Entry"      # 首单/初始仓位
    SO = "SO"            # 补仓单 (Safety Order)
    TP = "TP"            # 止盈单 (Take Profit)
    SL = "SL"            # 止损单 (Stop Loss)
    MARKET = "MARKET"    # 市场单/震荡模式处理
    CANCEL_ALL = "CANCEL_ALL"

class MarketState(Enum):
    OSCILLATION = "OSCILLATION"  # 震荡
    TREND_UP = "TREND_UP"        # 单边上涨
    TREND_DOWN = "TREND_DOWN"    # 单边下跌
    
@dataclass
class SymbolConfig:
    """单币种马丁策略参数类"""
    symbol: str
    budget_pct: float            # 总预算 (百分比)
    base_gap: float          # 基础网格间距 (如 0.005 代表 0.5%)
    trend_bias: float        # 趋势偏置量 (如 0.002 代表 0.2%). 开首单时偏向某一边的幅度，必须小于 base_gap
    max_layers: int          # 最大补仓层数
    volume_scale: float      # 加仓倍率 (Martingale)
    step_scale: float        # 间距倍率
    profit_target: float     # 固定的止盈百分比 (如 0.005 代表 0.5%)
                             # 无论第几层，都要保持这个利润空间
    # 以下参数将由交易所动态覆盖
    qty_step: float          # 数量精度 (交易所限制)
    tick_size: float         # 价格精度 (交易所限制)
    min_qty: float           # 最小下单量
    # 🌟 费率参数 (由 API 填充)
    maker_fee: float = 0.0002   # 默认 0.02%
    taker_fee: float = 0.00055  # 默认 0.055%
    # 核心矩阵：存储每一层的系数
    # 格式: {layer_int: {"p_factor": f, "q_factor": f, "avg_p_factor": f, ...}}
    matrix: list = field(default_factory=list)
    # 可以在类中定义一些便捷方法
    def get_max_so_gap(self, layer: int) -> float:
        """计算第 N 层相对于均价的百分比间距"""
        return self.base_gap * (self.step_scale ** (max(0, layer - 1)))
    @property
    def estimated_round_trip_fee(self):
        #计算主要操作的手续费 (纯 Maker 模式)
        return self.maker_fee * 2
    def build_matrix(self, is_long: bool):
        self.matrix = []
        cum_q_factor = 0.0
        cum_notional_factor = 0.0
        cum_gap = 0.0
        
        # 🌟 计算双边手续费成本因子
        # 成本 = 开仓费率 + 平仓费率
        fee_cost_pct = self.maker_fee + self.maker_fee
        
        for i in range(1, self.max_layers + 1):
            if i > 1:
                cum_gap += self.base_gap * (self.step_scale ** (i - 2))
            
            p_factor = (1 - cum_gap) if is_long else (1 + cum_gap)
            q_factor = self.volume_scale ** (i - 1)
            
            cum_q_factor += q_factor
            cum_notional_factor += (q_factor * p_factor)
            avg_p_factor = cum_notional_factor / cum_q_factor
            
            # 🌟 计算该层“理论净利润率因子” (Net Profit Factor)
            # 逻辑：(均价 * (1 + 利润目标)) - (均价 * (1 + 手续费成本))
            net_profit_factor = self.profit_target - fee_cost_pct
            
            self.matrix.append({
                "layer": i,
                "p_factor": p_factor,
                "q_factor": q_factor,
                "avg_p_factor": avg_p_factor,
                "cum_q_factor": cum_q_factor,
                "net_profit_factor": net_profit_factor # 存入矩阵，后续直接查表
            })
# ================= 核心状态管理 (State) =================
class SymbolState:
    def __init__(self, symbol:str, conf):
        self.symbol = symbol
        self.conf:SymbolConfig = conf
        self.lock = threading.RLock()
        self.last_order_ts = int(time.time() * 1000)
        # print(f"{sys._getframe().f_lineno} {time.time()} last_order_ts {self.last_order_ts}")

        # 🌟 核心：有符号仓位 (正数=多头, 负数=空头, 0=无仓位)
        self.signed_pos_qty = 0.0  
        self.avg_entry_price = 0.0
        self.base_price = 0
        self.base_qty = 0
        
        # 状态标记
        self.layer_count = 0        # 当前补仓到了第几层
        self.trend_score = 0.0      # Z-Score 趋势分 (正=涨, 负=跌)
        self.order_submit_sl = False
        #running status
        self.last_result = 0    # -1:SL, 1:TP
        self.loss_count = 0
        self.last_result_updte = time.time()
        self.market_state = MarketState.OSCILLATION
        # 0: 无趋势, 1: 上涨, -1: 下跌
        self.trend_direction = 0  
        # 任务防抖锁
        self.is_processing = False 
        self.last_processing_time = 0
        
        # 统计
        self.total_profit = 0.0
        self.total_fees = 0.0
        self.tp_count = 0           # 🌟 累计止盈次数
        self.sl_count = 0           # 🌟 累计止损次数
        self.tp_layer_dist = {}     # 例如: {1: 15, 2: 5, 3: 1}

class MartingaleBot:
    def __init__(self, engine: BybitEngine, configs: list, is_long_account):
        self.version = 'V1' # 马丁网格
        self.logger, _ = setup_session_logger(
            sub_folder=self.__class__.__name__+self.version, 
            console_level=logging.INFO, 
            file_level=logging.INFO
        )
        self.engine    = engine
        self.configs: list[SymbolConfig]= configs
        self.markets:Dict[str,SymbolState] = {}
        self.total_equity = 0.0
        self.init_total_equity = -1
        self.max_loss_ratio = 0.2
        self.stop_signal = False
        self.is_long_account = is_long_account
        self.leverage = 10
        self.setup()

    def setup(self):
        self.logger.info("🚀 V10 策略启动中...")
        self.update_wallet_balance()
        
        for cfg in self.configs:
            # 1. 同步交易所精度
            info = self.engine.get_symbol_info(cfg.symbol) # 假设你在engine增加了此函数
            if info:
                cfg.tick_size, cfg.qty_step, cfg.min_qty = info['tick_size'], info['qty_step'], info['min_qty']
            # 2. 🌟 同步真实费率
            maker, taker = self.engine.get_real_fee_rate(cfg.symbol)
            cfg.maker_fee = maker
            cfg.taker_fee = taker
            cfg.build_matrix(is_long=self.is_long_account)
            self.logger.info(f"💎 [{cfg.symbol}] info update: Maker {maker:.4%} | Taker {taker:.4%} | tick_size {cfg.tick_size} | qty_step {cfg.qty_step} | min_qty {cfg.min_qty} ")
            self.check_profit_viability(cfg)
            # 2. 初始化环境
            # self.engine.http.switch_position_mode(category="linear", symbol=cfg.symbol, mode=0)
            self.engine.set_leverage(cfg.symbol, self.leverage)
            self.markets[cfg.symbol] = SymbolState(cfg.symbol, cfg)

        self.initial_risk_report()

        self.engine.start_stream(self.on_ws_order_notify)
        for symbol in self.markets.keys():
            self.engine.market_close_all(symbol)

    def check_profit_viability(self, cfg: SymbolConfig):
        """
        启动自检：确保止盈目标能覆盖真实的 Maker 手续费
        """
        # 马丁网格核心流程是：Maker 买入 -> Maker 卖出
        # 成本 = 开仓费 + 平仓费
        cost = cfg.maker_fee * 2 
        
        # 净利润空间
        net_margin = cfg.profit_target - cost
        
        # 安全阈值：至少要有 0.2% 的净利，或者是费率的 2 倍
        if net_margin < 0.001:
            self.logger.warning("=" * 60)
            self.logger.warning(f"⚠️  [{cfg.symbol}] 利润空间过窄警告！预计净利: {net_margin:.2%} (除去滑点可能微乎其微)")
            # self.logger.warning(f"   - 设定止盈: {cfg.profit_target:.2%}")
            # self.logger.warning(f"   - 真实 Maker 费率: {cfg.maker_fee:.4%} (双边: {cost:.4%})")
            # self.logger.warning(f"   - 预计净利: {net_margin:.2%} (除去滑点可能微乎其微)")
            # self.logger.warning("   -> 建议: 提高 profit_target 或 升级 VIP 等级")
            self.logger.warning("=" * 60)
        else:
            self.logger.info(f"✅ [{cfg.symbol}] 利润模型健康 (净利空间: {net_margin:.2%})")

    def initial_risk_report(self):
        """
        🛡️ 增强版风险报告 - 增加手续费损耗与净收益分析
        """
        for m in self.markets.values():
            with m.lock:
                p_ref = self.get_last_price(m.symbol)
                if p_ref <= 0: continue
                
                side = "Long" if self.is_long_account else "Short"
                self.logger.info("=" * 115)
                self.logger.info(f"📊 [{m.symbol} {side}] 深度回撤与净收益分析 | Maker费率: {m.conf.maker_fee:.4%}")
                self.logger.info("-" * 115)
                
                # 增加“预估净利(U)”这一列
                header = f"{'层级':<4} | {'成交价格':<10} | {'实时均价':<10} | {'需反弹%':<8} | {'累计持仓(U)':<12} | {'预估净利(U)'}"
                print(header)
                print("-" * 115)
                
                m.base_qty = 5/ p_ref
                for row in m.conf.matrix:
                    layer = row['layer']
                    fill_p = p_ref * row['p_factor']
                    avg_p = p_ref * row['avg_p_factor']
                    
                    # 止盈位
                    tp_p = avg_p * (1 + m.conf.profit_target if self.is_long_account else 1 - m.conf.profit_target)
                    rebound_needed = abs((tp_p - fill_p) / fill_p) * 100
                    
                    # 计算统计
                    q_total = m.base_qty * row['cum_q_factor']
                    total_notional = q_total * avg_p
                    
                    # 🌟 计算该层离场时的“净利润”
                    # 公式：总名义价值 * 净利润率因子
                    net_profit_u = total_notional * row['net_profit_factor']
                    
                    # 收益颜色提醒 (如果手续费太高导致净利极低)
                    profit_status = "⚠️ LOW" if net_profit_u < 0.1 else "OK"

                    print(f"L{layer:<3} | {fill_p:>10.4f} | {avg_p:>10.4f} | {rebound_needed:>7.2f}% | {total_notional:>12.2f} | {net_profit_u:>10.4f} {profit_status}")

                # --- 最终风险汇总 ---
                final_layer = m.conf.matrix[-1]
                avg_price = p_ref * final_layer['avg_p_factor']
                tp_price = avg_price * (1 + m.conf.profit_target if self.is_long_account else 1 - m.conf.profit_target)
                sl_price = avg_price * (1 - m.conf.profit_target if self.is_long_account else 1 + m.conf.profit_target)
                
                # 止损损耗计算
                loss_at_sl = abs((m.base_qty * final_layer['cum_q_factor']) * (sl_price - avg_price))
                loss_pct = (loss_at_sl / self.total_equity) * 100

                self.logger.info(f"🔍 [{m.symbol} {side}] 关键结论:")
                self.logger.info(f"   - 均价锚点: {avg_price:.5f} ({((avg_price-p_ref)/p_ref*100):.3f}% from Ref)")
                self.logger.info(f"   - 🎯 止盈目标: {tp_price:.5f} ({((tp_price-p_ref)/p_ref*100):.3f}% from Ref)")
                self.logger.info(f"   - 💀 镜像止损: {sl_price:.5f} ({((sl_price-p_ref)/p_ref*100):.3f}% from Ref)")
                self.logger.info(f"   - 💥 满仓风险: 预计损耗 {loss_at_sl:.2f} USDT ({loss_pct:.2f}% 净值)")
                
                # 提醒反弹力度
                max_rebound = abs((tp_price - (p_ref * final_layer['p_factor'])) / (p_ref * final_layer['p_factor'])) * 100
                self.logger.info(f"   - ⚡ 生死线反弹需求: 价格触及 L{final_layer['layer']} 后需反弹 {max_rebound:.2f}% 即可获利出场")
        
        self.logger.info("=" * 105)

    def update_wallet_balance(self):
        res = self.engine.http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if res.get('retCode') == 0:
            self.total_equity = float(res['result']['list'][0]['coin'][0]['equity'])
            if self.init_total_equity == -1:
                self.init_total_equity = self.total_equity

    def on_ws_order_notify(self, message):
        orders = message.get('data', [])
        if not orders:
            return
        # 1. 预处理：按时间戳从小到大排序（防止 Bybit 乱序推送）
        # 确保如果 L1 和 L2 同时在 data 里，先处理 L1
        orders.sort(key=lambda x: int(x.get('execTime', 0)))
        for order in message.get('data', []):
            symbol = order['symbol']
            if symbol in self.markets:
                if order['orderStatus'] == 'Filled':
                    if order['orderLinkId'] == '':
                        continue    #cancle/market order, ignore
                    self.handle_fill(symbol,order)# call directly when test
                    # threading.Thread(target=self.handle_fill, args=(symbol,order), daemon=True).start()
                elif order['orderStatus'] in ['Cancelled', 'Rejected']:
                    self.logger.debug(f"ℹ️ 订单 {order['orderLinkId']} 状态变更: {order['orderStatus']}")
            else:
                self.logger.warning(f"unecpected symbol {symbol}, close all")
                self.engine.market_close_all(symbol)

    #止盈，bybit会自动取消价格更高的TP订单
    def place_tp_order(self,m:SymbolState):
        tp_raw_price = m.avg_entry_price * (1 + (m.conf.profit_target * (1 if (m.signed_pos_qty > 0) else -1)))
        tp_price = round(tp_raw_price / m.conf.tick_size) * m.conf.tick_size
        side = "Sell" if (m.signed_pos_qty > 0) else "Buy"
        tp_qty = abs(m.signed_pos_qty)
        order_id = self.generate_order_link_id(m.last_order_ts, m.layer_count ,label = OrderLabel.TP)
        self.engine.place_order(m.symbol, side, tp_qty, tp_price, order_id, is_reduce=True)
        self.logger.info(f"place tp order {side} order , qty {tp_qty}, price {tp_price} order_id {order_id}")

    def _prepare_so_order_params(self, m: SymbolState, layer: int):
        """
        🧪 通用 SO 订单参数生成器 (不执行下单)
        计算公式: 
        Price_{layer} = BasePrice /times P_Factor_{layer}$
        Qty_{layer} = BaseQty /times Q_Factor_{layer}$
        """
        # 1. 查表获取系数
        row = m.conf.matrix[layer - 1]
        
        # 2. 计算物理价格与数量
        raw_p = m.base_price * row['p_factor']
        price = round(raw_p / m.conf.tick_size) * m.conf.tick_size
        self.logger.debug(f"_prepare_so_order_params layer {layer} price {price}")
        raw_q = m.base_qty * row['q_factor']
        qty = round(raw_q / m.conf.qty_step) * m.conf.qty_step
        
        # 3. 构造参数字典 (符合 Bybit V5 批量下单格式)
        side = "Buy" if self.is_long_account else "Sell"
        label = OrderLabel.ENTRY if layer == 1 else OrderLabel.SO
        # print(f"{sys._getframe().f_lineno} {time.time()} last_order_ts {m.last_order_ts}, layer {layer}")
        order_id = self.generate_order_link_id(m.last_order_ts, layer ,label = label)

        return {
            "symbol": m.symbol,
            "side": side,
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(price),
            "orderLinkId": order_id,
            "reduceOnly": False
        }

    def _prepare_sl_order_params(self, m: SymbolState):
        """
        🛡️ 镜像止损参数生成器 (Maker 模式)
        计算逻辑：基于满仓均价因子，计算出与止盈对称的止损位
        """
        # 1. 获取矩阵最后一层 (满仓状态)
        final_layer = m.conf.matrix[-1]
        
        # 2. 推算理论满仓均价
        # $AvgP_{final} = BasePrice \times AvgPFactor_{final}$
        theo_avg_p = m.base_price * final_layer['avg_p_factor']
        
        # 3. 计算镜像对称距离
        # 距离 = 满仓均价 * 止盈目标
        dist = theo_avg_p * m.conf.profit_target
        
        # 4. 计算止损价格 (多头在均价下，空头在均价上)
        if self.is_long_account:
            sl_raw_p = theo_avg_p - dist
            side = "Sell"
        else:
            sl_raw_p = theo_avg_p + dist
            side = "Buy"
            
        sl_price = round(sl_raw_p / m.conf.tick_size) * m.conf.tick_size
        
        # 5. 计算冗余平仓数量 (1.001倍 + reduceOnly 确保全平)
        full_qty = m.base_qty * final_layer['cum_q_factor'] * 1.001
        sl_qty = round(full_qty / m.conf.qty_step) * m.conf.qty_step
        # 🌟 增加打印信息：监控 SL 核心参数
        # print(f"[{sys._getframe().f_lineno}] 🛡️ {m.symbol} SL Calc | Side: {side} | TheoAvgP: {theo_avg_p:.6f} | Dist: {dist:.6f} | SL_Price: {sl_price} | SL_Qty: {sl_qty} | TS: {m.last_order_ts}")
        return {
            "symbol": m.symbol,
            "side": side,
            "orderType": "Limit",
            "qty": str(sl_qty),
            "price": str(sl_price),
            "orderLinkId": self.generate_order_link_id(m.last_order_ts, m.conf.max_layers, OrderLabel.SL),
            "reduceOnly": True  # 🌟 必须开启，确保只减仓
        }

    #对称止损
    def place_sl_order(self, m: SymbolState):
        """🛠️ 执行止损单挂设"""
        params = self._prepare_sl_order_params(m)
        
        #限价
        # self.engine.place_order(
        #     symbol=params['symbol'],
        #     side=params['side'],
        #     qty=float(params['qty']),
        #     price=float(params['price']),
        #     link_id=params['orderLinkId'],
        #     is_reduce=True # 内部对应 reduceOnly=True
        # )
        # 🌟 注意：市价条件单在触发前不需要传 price，只需 triggerPrice
        self.engine.place_order(
            symbol=params['symbol'],
            side=params['side'],
            qty=float(params['qty']),
            price=None,                     # 🌟 市价单不设限价
            triggerPrice=params['price'], # 🌟 传入触发价
            link_id=params['orderLinkId'],
            is_reduce=True,
            order_type="Market",             # 🌟 显式指定 Market
            triggerDirection = 2 if self.is_long_account else 1
        )
        self.logger.info(f"🛡️ [{m.symbol}] 镜像止损单已就位 | 价格: {params['price']} | 数量: {params['qty']}")
        
    def handle_fill(self, symbol, order):
        m = self.markets[symbol]
        price = float(order.get('avgPrice', order['price']))
        qty = float(order['cumExecQty'])
        signed_delta = qty if order['side'] == 'Buy' else -qty
        
        with m.lock:
            # 判定加仓还是减仓
            valid, ts, layer_count ,label = self.parse_order_link_id(order['orderLinkId'])
            if valid == False or ts != m.last_order_ts:
                # self.engine.cancel_single_order(order['symbol'], order['orderLinkId'])
                if valid == False:
                    self.logger.error(f"cancle invalid order {order['orderLinkId']}, current version {self.version} ts {m.last_order_ts}")
                elif ts < m.last_order_ts:
                    self.logger.error(f"cancle expired order {order['orderLinkId']}, current version {self.version} ts {m.last_order_ts}")
                elif ts > m.last_order_ts:
                    self.logger.warning(f"cancle newer order {order['orderLinkId']}, current version {self.version} ts {m.last_order_ts}") 
                return
            if layer_count != m.layer_count:    #允许这种情况，增加容错
                self.logger.warning(f" 检测到跨层成交 ! layer {m.layer_count} -> {layer_count}")
                m.layer_count = layer_count
            is_inc = (m.signed_pos_qty * signed_delta) > 0 or m.signed_pos_qty == 0
            if is_inc:
                if m.order_submit_sl == False:
                    # --- 3. 挂出“终极限价止损单” (目标：Maker 平仓) ---
                    # 基于矩阵最后一层的满仓均价因子进行对称计算
                    m.order_submit_sl = True
                    self.place_sl_order(m)
                old_abs = abs(m.signed_pos_qty)
                m.avg_entry_price = ((m.avg_entry_price * old_abs) + (price * qty)) / (old_abs + qty)
                m.signed_pos_qty += signed_delta
                if m.layer_count < m.conf.max_layers:
                    m.layer_count += 1
                    self.place_tp_order(m)  #no need to cancle the previous TP
                        # self.logger.info(f"handle_fill {side} Buy order , qty {tp_qty}, price {tp_price} order_id {order_id}")
            else: 
                m.signed_pos_qty += signed_delta
                if label == OrderLabel.TP:
                    m.tp_count += 1  # 🌟 记录止盈
                    m.tp_layer_dist[layer_count] = m.tp_layer_dist.get(layer_count, 0) + 1
                    m.last_result = OrderLabel.TP
                    m.last_result_updte = time.time()
                    m.loss_count = 0
                    self.logger.info(" 止盈触发 ")
                elif label == OrderLabel.SL:
                    m.sl_count += 1  # 🌟 记录止损
                    if m.last_result == OrderLabel.SL:
                        m.loss_count += 1
                    else:
                        m.loss_count = 1
                    m.last_result = OrderLabel.SL
                    m.last_result_updte = time.time()
                    self.logger.info(" 止损触发 ")
                if abs(m.signed_pos_qty) >= m.conf.min_qty:
                    self.logger.error("unexpected uncompleted decrease in position, please check !")
                    # self.engine.cancel_all_http(symbol)
                    # self.engine.market_close_all(symbol)
                else:
                    if label != OrderLabel.TP and label != OrderLabel.SL: #止盈/止损
                        self.logger.error("unexpected order id {order['orderLinkId']} layer_count {layer_count} label {label}")
                    else:
                        self.logger.info(" position is 0, new order ")
                self.engine.market_close_all(symbol)
                self.deploy_full_martingale_grid(symbol)  #new order
            self.logger.info(f"📊 {symbol} 仓位更新: {m.signed_pos_qty:.2f} @ {m.avg_entry_price:.4f}")

    def generate_order_link_id(self, last_order_ts =0, layer_count =0 , label = ''): 
        """生成唯一 ID: V9:SYMBOL:INDEX:SIDE:TIMESTAMP"""
        # 毫秒级防碰撞
        link_id = f"{self.version}:{last_order_ts}:{layer_count}:{label}"
        self.logger.debug(f"generate new order {link_id}")
        return link_id

    def parse_order_link_id(self, link_id):
        """
        解析 ID
        返回: (valid, timestamp, layer_count, label)
        """
        try:
            parts = link_id.split(':')
            if len(parts) < 4 or parts[0] != self.version:
                return False, None, None, None
            
            ts = int(parts[1])
            layer_count = int(parts[2])
            label = str(parts[3])
            return True, ts, layer_count ,label
        except Exception:
            return False, None, None, None

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

    def deploy_full_martingale_grid(self, symbol, p_ref =0, q1 =0):
        """
        ⚡ 全场布阵：批量挂出开仓/补仓单 + 确定性限价止损(Maker)
        """
        left_equity_ratio = self.total_equity/self.init_total_equity
        if left_equity_ratio < (1 - self.max_loss_ratio):
            self.stop_signal = True
            self.logger.warning(f" Quity ratio {left_equity_ratio} is less than {self.max_loss_ratio}, Emergency Stop ")
        m = self.markets[symbol]
        with m.lock:
            if m.loss_count == 1 and time.time() - m.last_result_updte > 60:    #1 min内禁止交易
                self.logger.info(f"trade forbiden loss_count {m.loss_count}")
                return
            elif m.loss_count > 1 and time.time() - m.last_result_updte > 60*2*m.loss_count:    #行情异常，禁止交易
                self.logger.info(f"trade forbiden loss_count {m.loss_count}")
                return
            elif (m.market_state == MarketState.TREND_DOWN and self.is_long_account) or \
                (m.market_state == MarketState.TREND_UP and not self.is_long_account):
                self.logger.info(f"trade forbiden symbol {symbol} market_state {m.market_state.value}")
                return
            m.order_submit_sl = False
            # 🌟 核心：在此锁定基准锚点
            res = self.engine.http.get_tickers(category=self.engine.category, symbol=symbol)
            
            if res.get('retCode') == 0:
                m.base_price = float(res['result']['list'][0]['lastPrice'])
                m.base_qty = self.total_equity * m.conf.budget_pct / m.base_price
            else:
                self.logger.warning("deploy update price fail! ")
                return
            
            # 1. 循环生成所有层级的订单请求 (L1-Lmax)
            order_requests = []
            m.last_order_ts = int(time.time() * 1000)
            # print(f"{sys._getframe().f_lineno} {time.time()} last_order_ts {m.last_order_ts}")
            for i in range(1, m.conf.max_layers + 1):
                req = self._prepare_so_order_params(m, i)
                order_requests.append(req)

            # --- 2. 批量下单 (L1-Lmax) ---
            if order_requests:
                res = self.engine.http.place_batch_order(category="linear", request=order_requests)
                if res.get('retCode') == 0:
                    self.logger.info(f"✅ [{symbol}] L1-L{m.conf.max_layers} 批量布阵成功")
                else:
                    self.logger.error(f"❌ 批量下单失败: {res.get('retMsg')}")
                    return
            m.layer_count = 1

    def get_all_open_orders(self, category="linear"):
        """
        🚀 一次性获取当前账户下所有币种的活动挂单
        """
        # 🌟 关键点：不传 symbol 参数，Bybit 会返回该 category 下的所有挂单
        res = self.engine.http.get_open_orders(category="linear", settleCoin="USDT")
        
        if res.get('retCode') == 0:
            return res.get('result', {}).get('list', [])
        else:
            self.logger.error(f"❌ 批量获取订单失败: {res.get('retMsg')}")
            return []
        
    def reconcile_all_markets(self):
        # 1. 批量抓取所有挂单
        all_orders = self.get_all_open_orders()
        
        # 2. 将订单按 symbol 归类
        # 使用 dict[str, list] 结构：{'MNTUSDT': [...], 'BTCUSDT': [...]}
        orders_by_symbol = {}
        for order in all_orders:
            s = order['symbol']
            if s not in orders_by_symbol:
                orders_by_symbol[s] = []
            orders_by_symbol[s].append(order)
        return orders_by_symbol

    def get_open_orders(self, symbol):
        res = self.engine.http.get_open_orders(category="linear", symbol=symbol)
        return res.get('result', {}).get('list', [])

    def get_last_price(self, symbol):
        res = self.engine.http.get_tickers(category="linear", symbol=symbol)
        return float(res['result']['list'][0]['lastPrice'])

    def recover_layer_from_history(self, symbol):
        """
        📜 通过历史成交记录精准恢复层数
        """
        result = False, 0, 0, 0
        try:
            # 1. 获取最近 20 笔成交记录 (Bybit V5 接口),考虑到订单频率不同，只能逐个symbol读取
            res = self.engine.http.get_executions(
                category="linear",
                symbol=symbol,
                limit=100
            )
            
            if res.get('retCode') != 0:
                return result

            exec_list = res.get('result', {}).get('list', [])
            if not exec_list:
                return result

            # 2. 找到最后一笔有效的成交
            order_list :dict[int,list]= {}
            for order in exec_list:
                link_id = order.get('orderLinkId', '')
                if link_id == '':
                    continue
                valid, ts, layer_count ,label = self.parse_order_link_id(link_id)
                if valid:
                    if ts not in order_list:
                        order_list[ts] = [(order,layer_count ,label)]
                    else:
                        order_list[ts].append((order,layer_count ,label))
            newest_orders_index = max(order_list.keys())
            newest_orders =  order_list[newest_orders_index]
            if not newest_orders :
                return result
            so_orders = {}
            for (order,layer_count ,label) in newest_orders:
                self.logger.info(f" label {label}| {OrderLabel.ENTRY}| {order.get('orderLinkId', '')} |layer_count{layer_count} ")
                if label in [OrderLabel.ENTRY, OrderLabel.SO]:   #开仓/加仓 订单
                    so_orders[layer_count] = order
                    self.logger.info(f"add to so_orders layer_count {layer_count} | {order.get('orderLinkId', '')}")
            sorted_so_order_list = OrderedDict(sorted(so_orders.items()))
            self.logger.info(f"num of sorted_so_order_list {len(sorted_so_order_list)}")
            last_layer_count = 0
            total_position = 0
            find_all_order = True
            for layer_count,order in sorted_so_order_list.items():
                if layer_count != last_layer_count+1:
                    self.logger.warning(f"layer count {last_layer_count} miss!!!")
                    find_all_order = False
                    total_position += float(order['cumExecQty'])
                    last_layer_count = layer_count
                    break
            if find_all_order != True:
                return result
            result = True, newest_orders_index, last_layer_count, total_position
            return result     
        except Exception as e:
            self.logger.error(f"❌ 恢复层数失败: {e}")
            
        return result # 默认安全返回

    def sync_local_pos(self):
        """同步本地仓位并恢复层数"""
        res = self.engine.http.get_positions(category="linear", settleCoin="USDT")
        if res.get('retCode') != 0:
            self.logger.warning("sync get_positions fail ")
            return
        pos_list = res.get('result', {}).get('list', [])
        active_symbols = []
        all_market_orders:Dict[str:list] = self.reconcile_all_markets()

        if not pos_list:
            for symbol in self.markets.keys():
                market_orders = all_market_orders.get(symbol , [])
                if len(market_orders)==0:
                    self.logger.info(" no position1,no order, start new order")
                    self.deploy_full_martingale_grid(symbol)
            return
        
        for p in pos_list:
            qty = float(p['size'])
            symbol = p['symbol']
            #test stage
            market_orders = all_market_orders.get(symbol , [])
            if qty != 0 or  len(market_orders)!=0:
                continue
            else:
                self.logger.info(" no position,no order, start new order")
                self.deploy_full_martingale_grid(symbol)
                continue
            if qty > 0:
                active_symbols.append(symbol)
                m = self.markets[symbol]
                with m.lock:
                    self.logger.debug(f" symbol {symbol} signed_pos_qty {m.signed_pos_qty} qty {qty} side {p['side']}")
                    self.logger.debug(f" symbol {symbol} avg_entry_price {m.avg_entry_price} avgPrice {p['avgPrice']}")
                    m.signed_pos_qty = qty * (1 if p['side'] == 'Buy' else -1)
                    m.avg_entry_price = float(p['avgPrice'])
                    # 🌟 传入同步回来的均价进行推算
                    valid, ts, last_layer_count, total_position = self.recover_layer_from_history(symbol)
                    sync_result = False
                    if valid == False:
                        self.logger.error(f"{symbol} position sync fail ! position {qty}")
                    else:
                        if qty != total_position:
                            self.logger.error(f"{symbol} position sync fail ! qty {qty}, total_position {total_position}")
                        elif ts != m.last_order_ts:
                            self.logger.error(f"{symbol} position sync fail ! ts {ts}, last_order_ts {m.last_order_ts}")
                        elif last_layer_count != m.layer_count:
                            self.logger.warning(f"{symbol} sync layer, update layer_count {m.layer_count} -> {last_layer_count}")
                            sync_result =True
                        else:
                            sync_result =True
                    if sync_result == False:
                        # self.engine.cancel_all_http(symbol)
                        self.engine.market_close_all(symbol)
                        self.update_wallet_balance()
                        self.deploy_full_martingale_grid(symbol)
                    else: #check exist oder, TP/SO/SL
                        market_orders = all_market_orders.get(symbol , [])
                        #应该有两个订单，止盈单 + 止损/补仓单
                        find_tp = False
                        find_so = False
                        find_sl = False
                        target_tp_order_id = self.generate_order_link_id(m.last_order_ts, m.layer_count, label = OrderLabel.TP)  #止盈单
                        target_sl_order_id = self.generate_order_link_id(m.last_order_ts, m.layer_count,  label = OrderLabel.SL)  #止损单
                        miss_so_layers = {}
                        for i in range(last_layer_count, m.conf.max_layers+1):
                            m.layer_count = i
                            target_so_order_id = self.generate_order_link_id(m.last_order_ts, m.layer_count, label = OrderLabel.SO)  #补仓单
                            miss_so_layers[target_so_order_id] = i
                        m.layer_count = last_layer_count #recover

                        for order in market_orders:
                            link_id = order.get('orderLinkId', '')
                            if link_id == target_tp_order_id:
                                find_tp = True
                            elif link_id in miss_so_layers:
                                miss_so_layers.pop(link_id)
                            elif link_id == target_sl_order_id:
                                find_sl = True 
                        if find_tp == False:
                            self.logger.warning(f"{symbol}  止盈单缺失，补充止盈单 ")
                            self.place_tp_order(m)
                        so_order_requests = []
                        for layer in miss_so_layers.values():
                            so_order_requests.append(self._prepare_so_order_params(m, layer))
                        if so_order_requests:
                            res = self.engine.http.place_batch_order(category="linear", request=so_order_requests)
                            if res.get('retCode') == 0:
                                self.logger.info(f"✅ [{symbol}] L1-L{m.conf.max_layers} 批量补单成功")
                            else:
                                self.logger.error(f"❌ 批量补单失败: {res.get('retMsg')}")
                                return
                        if find_sl == False:
                            self.logger.warning(f"{symbol}  止损单缺失，补充止损单 ")
                            self.place_sl_order(m)     
            else:
                self.logger.info(" no position,check market_orders first")
                #没有仓位，先检查是否有订单存在
                market_orders = all_market_orders.get(symbol , [])
                if len(market_orders) == 0:
                    self.logger.info(f"sync market_orders len {len(market_orders)}")
                    # self.engine.cancel_all_http(symbol)
                    self.update_wallet_balance()
                    self.deploy_full_martingale_grid(symbol)
        # for symbol in self.markets.keys():
        #     if symbol not in active_symbols:
        #         self.logger.warning(f" {symbol} not found , cancel all ")
        #         m = self.markets[symbol]
        #         with m.lock:
        #             m.signed_pos_qty, m.avg_entry_price, m.layer_count = 0.0, 0.0, 0
        #             self.engine.cancel_all_http(symbol)
        #             self.update_wallet_balance()
        #             self.deploy_full_martingale_grid(symbol)

    def print_runtime_report(self):
        """📊 实时运行状态报告"""
        self.update_wallet_balance()  # 更新最新净值
        
        self.logger.info(f"{'币种':<10} | {'层级':<4} | {'止盈次数 (分布)':<35} | {'止损':<5} | {'胜率':<8}")
        self.logger.info("-" * 110)
        
        for symbol, m in self.markets.items():
            with m.lock:
                # 将字典 {1: 10, 2: 5} 转为字符串 "L1:10, L2:5"
                dist_items = sorted(m.tp_layer_dist.items())
                dist_str = ", ".join([f"L{k}:{v}" for k, v in dist_items]) if dist_items else "None"
                
                total_trades = m.tp_count + m.sl_count
                win_rate = (m.tp_count / total_trades * 100) if total_trades > 0 else 0
                
                self.logger.info(
                    f"{symbol:<10} | L{m.layer_count:<3} | {dist_str:<35} | {m.sl_count:>5} | {win_rate:>7.1f}%"
                ) 

    def run(self):
        last_report_time = 0
        while not self.stop_signal:
            current_time = time.time()
            
            # 1. 更新市场状态（趋势检测）
            for symbol in self.markets.keys():
                self.update_micro_market_status_volume(symbol)
            
            # 2. 同步仓位与自动补单
            self.sync_local_pos()
            
            # 3. 每 60 秒打印一次运行报告
            if current_time - last_report_time >= 60:
                self.print_runtime_report()
                last_report_time = current_time
                
            time.sleep(30) # 保持 30 秒的扫描频率

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Martingale Bot V1")
    group = parser.add_mutually_exclusive_group(required=True)
    # 🌟 简化为 -l 和 -s
    group.add_argument("-l", "--long", action="store_true", help="Run Long Account")
    group.add_argument("-s", "--short", action="store_true", help="Run Short Account")
    
    args = parser.parse_args()
    is_long_account = args.long
    # 配置你的 Key 路径
    demoTrading = False
    keypath = 'Testnet' if demoTrading == True else 'Maringale'
    side_path = 'Long' if is_long_account == True else 'Short'
    BASE = os.path.dirname(os.path.abspath(__file__))
    API_K = os.path.join(BASE, "keys", keypath, side_path, "hmac_api_key")
    API_S = os.path.join(BASE, "keys", keypath, side_path, "hmac_secret")
    RSA_K = os.path.join(BASE, "keys", keypath, side_path, "api_key")
    RSA_P = os.path.join(BASE, "keys", "bybit_rsa.pem")

    # ================= 配置区域 (CONFIG) =================
    # 建议根据不同币种波动率调整 gap 和 trend_bias
    CONFIGS = [
        # SymbolConfig(币种,    预算,   基础间距, 趋势偏置, 最大层数, 数量倍率, 间距倍率, 止盈目标)
        SymbolConfig("MNTUSDT", 0.05,  0.008,   0.002,      6,      1.4,    1.1,        0.0008,   0,0,0),
        SymbolConfig("DOGEUSDT", 0.05, 0.0008,  0.0003,      6,      1.4,    1.2,        0.0008,   0,0,0),
        SymbolConfig("RAVEUSDT", 0.05, 0.002,  0.0003,      6,      1.7,    1.2,        0.002,   0,0,0),     #fee rate 0.0004
        # SymbolConfig("ADAUSDT", 0.05,  0.01,   0.001,      5,      1.3,    1.1,        0.008,   0,0,0),
    ]

    #分成多头/空头两个账户可以显著降低逻辑耦合。统计数据更加清晰
    # 初始化引擎
    engine = BybitEngine(API_K, API_S, RSA_K, RSA_P, testnet = demoTrading)
    # 启动机器人
    bot    = MartingaleBot(engine, CONFIGS, is_long_account = is_long_account)
    
    # 信号处理
    def signal_handler(sig, frame):
        print("\n👋 Stop signal received...")
        bot.stop_signal = True
        engine.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    bot.run()