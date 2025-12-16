import MetaTrader5 as mt5
import time

mt5.initialize()
symbol = "BTCUSD"

# 预热一次
mt5.symbol_info_tick(symbol)

start_time = time.perf_counter()
loop_count = 1000

for _ in range(loop_count):
    tick = mt5.symbol_info_tick(symbol)

end_time = time.perf_counter()

avg_time = (end_time - start_time) / loop_count
print(f"平均每次调用耗时: {avg_time * 1000:.4f} ms")
# 在我的机器上结果通常是 0.08 ms 左右