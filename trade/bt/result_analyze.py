import numpy as np
import pandas as pd
import logging

def analyze_pnl_distribution(trade_list):
    """
    输入: trade_list (list of dict), 每个元素至少包含 {'pnl': float, 'date': ...}
          或者直接是 pnl 的 list
    """
    logger = logging.getLogger("trade")
    
    # 1. 提取 PnL 数据
    if not trade_list:
        logger.warning("没有交易记录，无法分析分布。")
        return

    # 兼容性处理：如果是字典列表，提取 pnl 字段；如果是纯数字列表，直接用
    if isinstance(trade_list[0], dict):
        # 假设你的 trade_log 里净利润字段叫 'net_pnl' 或 'pnl'
        pnls = [t.get('net_pnl', t.get('pnl', 0)) for t in trade_list]
    else:
        pnls = trade_list

    df = pd.DataFrame({'pnl': pnls})
    df['is_win'] = df['pnl'] > 0

    # ====================================================
    # A. 盈利分布统计 (Profit Distribution)
    # ====================================================
    logger.info("\n" + "="*40)
    logger.info("📊 盈亏分布统计 (PnL Distribution)")
    logger.info("="*40)
    
    stats = df['pnl'].describe()
    skew = df['pnl'].skew()
    kurt = df['pnl'].kurt()
    
    logger.info(f"交易总数: {int(stats['count'])}")
    logger.info(f"平均盈亏: ${stats['mean']:.2f}")
    logger.info(f"中位数盈亏: ${stats['50%']:.2f}")
    logger.info(f"最大盈利: ${stats['max']:.2f}")
    logger.info(f"最大亏损: ${stats['min']:.2f}")
    logger.info(f"标准差 (波动): {stats['std']:.2f}")
    logger.info(f"偏度 (Skew): {skew:.2f} ({( '正偏/右肥尾' if skew > 0 else '负偏/左肥尾' )})")
    
    # 胜率分布
    win_count = df['is_win'].sum()
    loss_count = len(df) - win_count
    win_rate = win_count / len(df) * 100
    
    avg_win = df[df['pnl'] > 0]['pnl'].mean() if win_count > 0 else 0
    avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if loss_count > 0 else 0
    p_l_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    logger.info(f"胜率: {win_rate:.2f}% (胜 {win_count} / 负 {loss_count})")
    logger.info(f"盈亏比 (Avg Win / Avg Loss): {p_l_ratio:.2f}")

    # ====================================================
    # B. 连续亏损分析 (Consecutive Loss Analysis)
    # ====================================================
    logger.info("\n" + "-"*40)
    logger.info("📉 连续亏损/盈利分析 (Streaks)")
    logger.info("-"*40)

    # 计算连续序列
    # 逻辑：利用 cumsum 和 shift 找出状态变化的分组
    # 1: 盈利, -1: 亏损
    df['sign'] = np.where(df['pnl'] > 0, 1, -1)
    # 只有当符号发生变化时，group id 才会增加
    df['streak_id'] = (df['sign'] != df['sign'].shift()).cumsum()
    
    # 统计每个 streak_id 的长度和类型
    streak_stats = df.groupby(['streak_id', 'sign']).size().reset_index(name='length')
    
    # 分离出连赢和连亏
    winning_streaks = streak_stats[streak_stats['sign'] == 1]['length']
    losing_streaks = streak_stats[streak_stats['sign'] == -1]['length']

    # 1. 连亏统计
    if not losing_streaks.empty:
        max_losing_streak = losing_streaks.max()
        avg_losing_streak = losing_streaks.mean()
        
        logger.info(f"🛑 最大连续亏损次数: {max_losing_streak} 次")
        logger.info(f"🛑 平均连续亏损次数: {avg_losing_streak:.2f} 次")
        
        # 连亏分布表 (Histogram)
        logger.info("\n连亏次数分布表:")
        logger.info(f"{'连亏长度':<10} | {'发生次数':<10} | {'占比':<10}")
        logger.info("-" * 36)
        
        loss_dist = losing_streaks.value_counts().sort_index()
        total_streaks = len(losing_streaks)
        for length, count in loss_dist.items():
            pct = count / total_streaks * 100
            # 简单的 ASCII 条形图
            bar = "█" * int(pct // 5) 
            logger.info(f"{length:<10} | {count:<10} | {pct:5.1f}% {bar}")
    else:
        logger.info("恭喜！没有发生连续亏损 (这不太可能...)")

    # 2. 连赢统计 (可选)
    if not winning_streaks.empty:
        logger.info(f"\n✅ 最大连续盈利次数: {winning_streaks.max()} 次")
    
    return {
        "max_losing_streak": max_losing_streak if not losing_streaks.empty else 0,
        "loss_streak_dist": losing_streaks.value_counts().to_dict() if not losing_streaks.empty else {}
    }

def analyze_trade_dependency(trade_list):
    """
    分析交易结果的依赖性 (Dependency Analysis)
    判断是否存在：亏损后更容易亏损 (Positive Correlation)
    """
    logger = logging.getLogger("trade")
    
    if not trade_list:
        return

    # 1. 数据准备：转为 1 (Win) 和 -1 (Loss)
    # 假设 trade_list 里是 pnl 数字
    # 过滤掉 0 (平盘通常不算)
    outcomes = [1 if t > 0 else -1 for t in trade_list if t != 0]
    
    n = len(outcomes)
    if n < 2:
        return

    # 2. 统计状态转移
    # ww: Win followed by Win
    # wl: Win followed by Loss
    # lw: Loss followed by Win
    # ll: Loss followed by Loss
    counts = {
        (1, 1): 0,  # W -> W
        (1, -1): 0, # W -> L
        (-1, 1): 0, # L -> W
        (-1, -1): 0 # L -> L
    }

    for i in range(n - 1):
        current = outcomes[i]
        next_trade = outcomes[i+1]
        counts[(current, next_trade)] += 1

    # 3. 计算基础概率
    total_wins = outcomes.count(1)
    total_losses = outcomes.count(-1)
    p_win = total_wins / n
    p_loss = total_losses / n

    # 4. 计算条件概率
    # P(L|L): 已知当前亏损，下一笔继续亏损的概率
    l_to_l_count = counts[(-1, -1)]
    l_to_w_count = counts[(-1, 1)]
    # 分母是“所有以 Loss 开头的转换”
    total_l_start = l_to_l_count + l_to_w_count
    
    p_l_given_l = l_to_l_count / total_l_start if total_l_start > 0 else 0
    
    # P(W|W): 已知当前盈利，下一笔继续盈利的概率
    w_to_w_count = counts[(1, 1)]
    w_to_l_count = counts[(1, -1)]
    total_w_start = w_to_w_count + w_to_l_count
    
    p_w_given_w = w_to_w_count / total_w_start if total_w_start > 0 else 0

    # 5. 输出结果
    logger.info("\n" + "="*40)
    logger.info("🔗 交易依赖性分析 (Sequential Dependency)")
    logger.info("="*40)
    
    logger.info(f"基础败率 P(L): {p_loss:.2%}")
    logger.info(f"条件败率 P(L|L): {p_l_given_l:.2%} (亏损后继续亏的概率)")
    
    delta = p_l_given_l - p_loss
    
    if delta > 0.05:
        logger.warning(f"⚠️ 发现正相关! 亏损后继续亏的概率增加了 {delta*100:.1f}% -> 建议: 连亏时减仓")
    elif delta < -0.05:
        logger.info(f"✅ 发现负相关 (均值回归). 亏损后更容易赢 -> 建议: 保持仓位")
    else:
        logger.info(f"⚖️ 结果接近随机独立 (差异 < 5%). 亏损不影响下一次结果.")

    logger.info("-" * 40)
    logger.info(f"基础胜率 P(W): {p_win:.2%}")
    logger.info(f"条件胜率 P(W|W): {p_w_given_w:.2%} (盈利后继续赢的概率)")
    
    # Z-Score 显著性检验 (Runs Test)
    # R: 实际游程数 (符号变化的次数 + 1)
    runs = 1
    for i in range(n - 1):
        if outcomes[i] != outcomes[i+1]:
            runs += 1
            
    # E_R: 期望游程数
    n1 = total_wins
    n2 = total_losses
    exp_runs = 1 + (2 * n1 * n2) / n
    std_runs = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1)))
    
    z_score = (runs - exp_runs) / std_runs if std_runs > 0 else 0
    
    logger.info("-" * 40)
    logger.info(f"Z-Score (Runs Test): {z_score:.4f}")
    if abs(z_score) > 1.96:
        if z_score < 0:
            logger.info("结论: 显著的正相关 (聚类). 连赢或连亏现象明显 -> 适合趋势策略/反马丁")
        else:
            logger.info("结论: 显著的负相关 (震荡). 赢亏交替频繁 -> 适合回归策略/马丁")
    else:
        logger.info("结论: 随机性强 (Random). 无法通过上一笔结果预测下一笔.")