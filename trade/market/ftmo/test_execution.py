import sys
import os
import time
import logging

# 路径适配
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..', '..'))

from trade.market.ftmo.ftmo_executor import MT5Executor
from trade.market.ftmo.market_ftmo import LiveConfig
from data_process import common

# 配置日志到控制台，方便实时看
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test():
    print("="*60)
    print("🧪 MT5 Executor Logic Test Suite")
    print("⚠️  WARNING: This will execute REAL TRADES on your connected MT5 account.")
    print("⚠️  Ensure you are logged into a DEMO account.")
    print("="*60)
    
    # 1. 初始化 Executor
    # 使用极小的止损比例，方便观察 SL 设置
    try:
        executor = MT5Executor(
            symbol=LiveConfig.SYMBOL_FTMO, 
            magic_number=LiveConfig.MAGIC_NUMBER, 
            sl_scale=1.0, 
            tp_ratio=1.0
        )
    except Exception as e:
        print(f"❌ Init failed: {e}")
        return

    # 打印基础信息，检查 Crypto 适配是否正确
    print(f"\n[Check 1] Symbol Info for {executor.symbol}")
    print(f"  - Contract Size: {executor.contract_size}")
    print(f"  - Digits: {executor.digits}")
    print(f"  - Min Vol: {executor.min_vol}")
    print(f"  - Vol Step: {executor.vol_step}")

    # 注入一个模拟的止损阈值 (比如 1% 的波动率)
    # 这样 calculated SL = Price * (1 - 0.01 * sl_scale)
    executor.update_context(stop_threshold_pct=0.01)

    # ---------------------------------------------------------
    # 测试场景 A: 开多单 (Open Long)
    # ---------------------------------------------------------
    print("\n[Test A] Opening Long Position (Target: 1% Equity)...")
    # 假设我们要用 1% 的资金买入
    executor.user_order_target_percent(target_pct=0.01)
    
    time.sleep(3) # 等待 MT5 反应
    _print_status(executor)
    
    input(">> Press Enter to continue to ADD POSITION (Pyramiding)...")

    # ---------------------------------------------------------
    # 测试场景 B: 加仓 (Increase Long)
    # ---------------------------------------------------------
    print("\n[Test B] Increasing Long Position (Target: 2% Equity)...")
    # 目标仓位变成 2%，应该会再买入一份
    executor.user_order_target_percent(target_pct=0.02)
    
    time.sleep(3)
    _print_status(executor)

    input(">> Press Enter to continue to REDUCE POSITION (Partial Close)...")

    # ---------------------------------------------------------
    # 测试场景 C: 减仓 (Reduce Long)
    # ---------------------------------------------------------
    print("\n[Test C] Reducing Long Position (Target: 0.5% Equity)...")
    # 目标变成 0.5%，应该会卖出大部分持仓
    executor.user_order_target_percent(target_pct=0.005)
    
    time.sleep(3)
    _print_status(executor)
    
    input(">> Press Enter to continue to REVERSE (Flip to Short)...")

    # ---------------------------------------------------------
    # 测试场景 D: 反手 (Reverse to Short)
    # ---------------------------------------------------------
    print("\n[Test D] Reversing to Short (Target: -1% Equity)...")
    # 目标变成 -1%，应该平掉剩余多单，并开空单
    executor.user_order_target_percent(target_pct=-0.01)
    
    time.sleep(3)
    _print_status(executor)
    
    input(">> Press Enter to continue to CLOSE ALL...")

    # ---------------------------------------------------------
    # 测试场景 E: 清仓 (Close All)
    # ---------------------------------------------------------
    print("\n[Test E] Closing All Positions...")
    executor.close_all()
    
    time.sleep(3)
    _print_status(executor)
    
    print("\n✅ Test Complete.")

def _print_status(executor):
    direction, layers, vol = executor.get_current_state()
    print(f"   -> Current MT5 State: Dir={direction.name}, Vol={vol}")

if __name__ == "__main__":
    run_test()