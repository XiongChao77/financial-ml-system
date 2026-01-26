import sys
import os
import time
import logging
import MetaTrader5 as mt5

# 路径适配
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..', '..'))

from trade.market.ftmo.mt5_executor import MT5Executor
from Quant.trade.market.ftmo.market_ml import LiveConfig
from trade.strategy.strategy_ml import PositionDir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestExecution")

def run_test():
    print("="*60)
    print("🧪 FTMO MT5 接口完整性压力测试 (含 user_order)")
    print(f"📡 目标品种: {LiveConfig.SYMBOL_FTMO}")
    print("="*60)
    
    try:
        executor = MT5Executor(
            symbol=LiveConfig.SYMBOL_FTMO, 
            magic=LiveConfig.MAGIC_NUMBER, 
        )
    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        return

    symbol_info = mt5.symbol_info(executor.symbol)
    if symbol_info is None:
        logger.error(f"❌ 无法获取品种信息")
        return

    print(f"\n[检查 1] 品种规格校验 ({executor.symbol})")
    print(f"  - 合约大小: {symbol_info.trade_contract_size}")
    print(f"  - 报价精度: {symbol_info.digits}")

    # ---------------------------------------------------------
    # 测试场景 0: 原始下单接口测试 (Raw user_order)
    # ---------------------------------------------------------
    # 这是你执行器中最核心的逻辑：将币数转换为 Lots
    test_size = 1000.0 
    expected_lots = test_size / symbol_info.trade_contract_size
    
    print(f"\n[测试 0] 正在执行: 原始 user_order 接口 (买入 {test_size} 单位)...")
    print(f"  - 预期成交手数应为: {expected_lots} Lots")
    
    # 直接调用底层接口
    executor.user_order(size=test_size, is_buy=True, stop_loss=0.01)
    
    time.sleep(3)
    _print_status(executor)
    
    input(f">> [操作确认] 请检查 MT5 中 Volume 是否为 {expected_lots}。按回车清理并继续...")
    executor.close_all()
    time.sleep(2)

    # ---------------------------------------------------------
    # 测试场景 A: 开多单 (基于百分比)
    # ---------------------------------------------------------
    print("\n[测试 A] 正在执行: 目标仓位 1% (多单)...")
    executor.user_order_target_percent(target_pct=0.01)
    
    time.sleep(3)
    _print_status(executor)
    
    input(">> [操作确认] 检查百分比开仓是否成功。按回车继续测试加仓...")

    # ---------------------------------------------------------
    # 测试场景 B: 加仓
    # ---------------------------------------------------------
    print("\n[测试 B] 正在执行: 目标仓位 2% (加仓)...")
    executor.user_order_target_percent(target_pct=0.02)
    
    time.sleep(3)
    _print_status(executor)

    input(">> [操作确认] 检查手数是否翻倍。按回车测试反手...")

    # ---------------------------------------------------------
    # 测试场景 C: 反手 (Reverse)
    # ---------------------------------------------------------
    print("\n[测试 C] 正在执行: 目标仓位 -1% (全平并开空)...")
    executor.user_order_target_percent(target_pct=-0.01)
    
    time.sleep(3)
    _print_status(executor)
    
    input(">> [操作确认] 确认多单消失且现在持有空单。按回车执行全平...")

    # ---------------------------------------------------------
    # 测试场景 D: 全平
    # ---------------------------------------------------------
    print("\n[测试 D] 正在执行: 目标仓位 0% (全平)...")
    executor.user_order_target_percent(target_pct=0.0)
    
    time.sleep(3)
    _print_status(executor)
    
    print("\n[测试 E] 显式调用 close_all()...")
    executor.close_all()
    
    print("\n✅ 所有接口测试完成。")

def _print_status(executor):
    direction, layers, vol = executor.get_current_state() #
    logger.info(f"📊 当前实盘状态: 方向={direction.name}, 层数={layers}, 总手数={vol}")

if __name__ == "__main__":
    run_test()