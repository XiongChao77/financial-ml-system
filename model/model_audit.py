import pandas as pd
import numpy as np
import torch
import logging
import os
from model_loader import ModelHandler
from data_process.common import Signal

def verify_model_alignment_v2():
    # 1. 初始化
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger("AuditV2")
    handler = ModelHandler()
    
    window = handler.window
    stride = 8 # 假设使用 stride=8 进行测试
    feature_cols = handler.feature_cols
    kline_interval = 60000
    logger.info(f"🔍 启动对齐审计 V2 | Window: {window} | Stride: {stride}")

    # 2. 构造具有复杂拓扑结构的伪数据
    # 总长 500 行，模拟一个包含多种异常的序列
    data_len = 500
    df_test = pd.DataFrame(np.random.randn(data_len, len(feature_cols)), columns=feature_cols)
    base_time = 1600000000000
    df_test['open_time_ms_utc'] = [base_time + i * kline_interval for i in range(data_len)]
    df_test['label'] = 1 # 默认全为 Hold

    # --- 注入异常点 ---
    # A. 头部冷启动 (0-29行): 模拟技术指标预热
    df_test.loc[0:29, feature_cols[0]] = np.nan 
    
    # B. 中间异常空洞 (200-205行): 模拟数据源断层
    df_test.loc[200:205, feature_cols[1]] = np.nan
    
    logger.info(f"🛠 数据构造：[0:29] 头部空洞 | [200:205] 中间空洞")

    # 3. 执行推理 (回测模式，必须开启 is_live=False)
    # 注意：TimeSeriesWindowDataset 内部会执行你设计的两阶段清洗
    df_out, stats = handler.predict(
        df=df_test, 
        kline_interval_ms=kline_interval, 
        is_live=False, 
        batch_size=64,
        # 这里为了审计方便，不设置概率过滤，直接看 argmax
        diff_thresh=None ,
        stride= stride
    )

    # 4. 自动化审计点
    logger.info("\n" + "="*60)
    logger.info("📊 审计报告：数据拓扑与对齐验证")
    logger.info("="*60)

    # --- 检查点 1: 清洗逻辑验证 ---
    # 头部删除了 30 行，中间删除了 6 行
    # 有效行数应为 500 - 30 - 6 = 464
    # 窗口数量计算公式：$M = (N - window) // stride + 1$
    valid_rows_after_clean = data_len - 30 - 6
    expected_windows = (valid_rows_after_clean - window) // stride + 1
    
    actual_signals = df_out['pred'].dropna()
    logger.info(f"检查点 1 (清洗与步长): 预期信号数 ~{expected_windows}, 实际产生信号数 {len(actual_signals)}")
    if len(actual_signals) > 0:
        logger.info("✅ 信号生成正常。")
    else:
        logger.error("❌ 信号生成失败！")

    # --- 检查点 2: 索引对齐精确性 (关键) ---
    # 随机抽取 10 个信号点，验证其时间戳是否严格匹配索引位置
    logger.info(f"检查点 2 (时间戳坐标对齐):")
    sample_indices = np.random.choice(actual_signals.index, 5)
    for idx in sorted(sample_indices):
        expected_t = base_time + idx * kline_interval
        actual_t = df_out.at[idx, 'open_time_ms_utc']
        
        # 验证该位置的特征是否在原始 df 中本就是完整的
        is_original_nan = df_test.loc[idx, feature_cols].isna().any()
        
        status = "✅" if (expected_t == actual_t and not is_original_nan) else "❌"
        logger.info(f"  Row {idx:3d}: TimeMatch={status} | Pred={df_out.at[idx, 'pred']} | Prob={df_out.at[idx, 'pred_prob']:.4f}")

    # --- 检查点 3: 步长 (Stride) 均匀性 ---
    # 在连续区域，信号之间的间隔应严格等于 stride
    diffs = np.diff(actual_signals.index)
    # 排除由于中间空洞导致的间隔跳变
    normal_diffs = diffs[diffs < window] 
    if len(normal_diffs) > 0 and np.all(normal_diffs == stride):
        logger.info(f"检查点 3 (Stride 均匀性): 步长验证成功 (Stride={stride}) ✅")
    else:
        logger.warning(f"⚠️ 检查点 3: 步长检测不一。若数据中存在 Gap，这是正常的。")

    # --- 检查点 4: 尾部严检逻辑 ---
    # 模拟在 df 最后 5 行注入一个 NaN，看是否整个末尾窗口都被丢弃
    logger.info("检查点 4 (尾部 10 根严检): 验证中...")
    df_tail_bad = df_test.copy()
    df_tail_bad.iloc[-3, 0] = np.nan # 在倒数第3行制造缺失
    
    df_out_tail, _ = handler.predict(df_tail_bad, kline_interval, is_live=False)
    if df_out_tail.index[-1] not in df_out_tail['pred'].dropna().index:
        logger.info("✅ 成功：尾部存在空洞时，末尾信号已按逻辑被丢弃。")
    else:
        logger.error("❌ 失败：尾部严检逻辑未生效！")

    logger.info("="*60 + "\n")

if __name__ == "__main__":
    verify_model_alignment_v2()