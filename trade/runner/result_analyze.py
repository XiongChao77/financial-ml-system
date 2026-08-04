import numpy as np
import pandas as pd
import logging

def analyze_pnl_distribution(trade_list):
    """
    Input: trade_list (list of dict), every element carries at least {'pnl': float, 'date': ...}
           or simply a list of pnl values
    """
    logger = logging.getLogger("trade")
    
    # 1. Extract the PnL data
    if not trade_list:
        logger.warning("没有交易记录，无法分析分布。")
        return

    # Compatibility: pull the pnl field out of a list of dicts, or use a plain list of numbers as is
    if isinstance(trade_list[0], dict):
        # Assumes the net profit field of your trade_log is called 'net_pnl' or 'pnl'
        pnls = [t.get('net_pnl', t.get('pnl', 0)) for t in trade_list]
    else:
        pnls = trade_list

    df = pd.DataFrame({'pnl': pnls})
    df['is_win'] = df['pnl'] > 0

    # ====================================================
    # A. Profit distribution
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
    
    # Win rate distribution
    win_count = df['is_win'].sum()
    loss_count = len(df) - win_count
    win_rate = win_count / len(df) * 100
    
    avg_win = df[df['pnl'] > 0]['pnl'].mean() if win_count > 0 else 0
    avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if loss_count > 0 else 0
    p_l_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    logger.info(f"胜率: {win_rate:.2f}% (胜 {win_count} / 负 {loss_count})")
    logger.info(f"盈亏比 (Avg Win / Avg Loss): {p_l_ratio:.2f}")

    # ====================================================
    # B. Consecutive loss analysis
    # ====================================================
    logger.info("\n" + "-"*40)
    logger.info("📉 连续亏损/盈利分析 (Streaks)")
    logger.info("-"*40)

    # Build the streaks
    # Logic: cumsum + shift to group the state changes
    # 1: win, -1: loss
    df['sign'] = np.where(df['pnl'] > 0, 1, -1)
    # The group id only increases when the sign flips
    df['streak_id'] = (df['sign'] != df['sign'].shift()).cumsum()
    
    # Length and type of every streak_id
    streak_stats = df.groupby(['streak_id', 'sign']).size().reset_index(name='length')
    
    # Split winning and losing streaks
    winning_streaks = streak_stats[streak_stats['sign'] == 1]['length']
    losing_streaks = streak_stats[streak_stats['sign'] == -1]['length']

    # 1. Losing streak statistics
    if not losing_streaks.empty:
        max_losing_streak = losing_streaks.max()
        avg_losing_streak = losing_streaks.mean()
        
        logger.info(f"🛑 最大连续亏损次数: {max_losing_streak} 次")
        logger.info(f"🛑 平均连续亏损次数: {avg_losing_streak:.2f} 次")
        
        # Losing streak histogram
        logger.info("\n连亏次数分布表:")
        logger.info(f"{'连亏长度':<10} | {'发生次数':<10} | {'占比':<10}")
        logger.info("-" * 36)
        
        loss_dist = losing_streaks.value_counts().sort_index()
        total_streaks = len(losing_streaks)
        for length, count in loss_dist.items():
            pct = count / total_streaks * 100
            # Simple ASCII bar chart
            bar = "█" * int(pct // 5) 
            logger.info(f"{length:<10} | {count:<10} | {pct:5.1f}% {bar}")
    else:
        logger.info("恭喜！没有发生连续亏损 (这不太可能...)")

    # 2. Winning streak statistics (optional)
    if not winning_streaks.empty:
        logger.info(f"\n✅ 最大连续盈利次数: {winning_streaks.max()} 次")
    
    return {
        "max_losing_streak": max_losing_streak if not losing_streaks.empty else 0,
        "loss_streak_dist": losing_streaks.value_counts().to_dict() if not losing_streaks.empty else {}
    }

def analyze_trade_dependency(trade_list):
    """
    Dependency analysis of the trade results
    Tests whether a loss makes the next loss more likely (positive correlation)
    """
    logger = logging.getLogger("trade")
    
    if not trade_list:
        return

    # 1. Prepare the data: map to 1 (win) and -1 (loss)
    # Assumes trade_list holds pnl numbers
    # Drop 0 (break-even usually does not count)
    outcomes = [1 if t > 0 else -1 for t in trade_list if t != 0]
    
    n = len(outcomes)
    if n < 2:
        return

    # 2. Count the state transitions
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

    # 3. Base probabilities
    total_wins = outcomes.count(1)
    total_losses = outcomes.count(-1)
    p_win = total_wins / n
    p_loss = total_losses / n

    # 4. Conditional probabilities
    # P(L|L): probability that the next trade loses given the current one lost
    l_to_l_count = counts[(-1, -1)]
    l_to_w_count = counts[(-1, 1)]
    # The denominator is "every transition starting from a loss"
    total_l_start = l_to_l_count + l_to_w_count
    
    p_l_given_l = l_to_l_count / total_l_start if total_l_start > 0 else 0
    
    # P(W|W): probability that the next trade wins given the current one won
    w_to_w_count = counts[(1, 1)]
    w_to_l_count = counts[(1, -1)]
    total_w_start = w_to_w_count + w_to_l_count
    
    p_w_given_w = w_to_w_count / total_w_start if total_w_start > 0 else 0

    # 5. Output
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
    
    # Z-score significance test (runs test)
    # R: actual number of runs (sign changes + 1)
    runs = 1
    for i in range(n - 1):
        if outcomes[i] != outcomes[i+1]:
            runs += 1
            
    # E_R: expected number of runs
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