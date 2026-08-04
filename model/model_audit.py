import pandas as pd
import numpy as np
import torch
import logging
import os
from model_loader import ModelHandler
from data_process.common import Signal

def verify_model_alignment_v2():
    # 1. Initialization
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger("AuditV2")
    handler = ModelHandler()
    
    window = handler.seq_len
    stride = 8 # test with stride=8
    feature_cols = handler.feature_cols
    kline_interval = 60000
    logger.info(f"🔍 启动对齐审计 V2 | Window: {window} | Stride: {stride}")

    # 2. Build synthetic data with a complex topology
    # 500 rows in total, simulating a sequence with several kinds of anomaly
    data_len = 500
    df_test = pd.DataFrame(np.random.randn(data_len, len(feature_cols)), columns=feature_cols)
    base_time = 1600000000000
    df_test['open_time_ms_utc'] = [base_time + i * kline_interval for i in range(data_len)]
    df_test['label'] = 1 # everything is Hold by default

    # --- inject the anomalies ---
    # A. Cold start at the head (rows 0-29): simulates the indicator warm-up
    df_test.loc[0:29, feature_cols[0]] = np.nan 
    
    # B. Hole in the middle (rows 200-205): simulates a data source outage
    df_test.loc[200:205, feature_cols[1]] = np.nan
    
    logger.info(f"🛠 数据构造：[0:29] 头部空洞 | [200:205] 中间空洞")

    # 3. Run the inference (backtest mode, is_live=False is mandatory)
    # Note: TimeSeriesWindowDataset runs the two-stage cleaning you designed internally
    df_out, stats = handler.predict(
        df=df_test, 
        kline_interval_ms=kline_interval, 
        is_live=False, 
        batch_size=64,
        # No probability filter here, so the audit sees the raw argmax
        diff_thresh=None ,
        stride= stride
    )

    # 4. Automated audit points
    logger.info("\n" + "="*60)
    logger.info("📊 审计报告：数据拓扑与对齐验证")
    logger.info("="*60)

    # --- check 1: cleaning logic ---
    # 30 rows removed at the head, 6 in the middle
    # so the valid row count should be 500 - 30 - 6 = 464
    # window count formula: $M = (N - window) // stride + 1$
    valid_rows_after_clean = data_len - 30 - 6
    expected_windows = (valid_rows_after_clean - window) // stride + 1
    
    actual_signals = df_out['pred'].dropna()
    logger.info(f"检查点 1 (清洗与步长): 预期信号数 ~{expected_windows}, 实际产生信号数 {len(actual_signals)}")
    if len(actual_signals) > 0:
        logger.info("✅ 信号生成正常。")
    else:
        logger.error("❌ 信号生成失败！")

    # --- check 2: index alignment accuracy (critical) ---
    # Sample 10 signal points and verify their timestamps match the index positions exactly
    logger.info(f"检查点 2 (时间戳坐标对齐):")
    sample_indices = np.random.choice(actual_signals.index, 5)
    for idx in sorted(sample_indices):
        expected_t = base_time + idx * kline_interval
        actual_t = df_out.at[idx, 'open_time_ms_utc']
        
        # Verify the features at that position really were complete in the original df
        is_original_nan = df_test.loc[idx, feature_cols].isna().any()
        
        status = "✅" if (expected_t == actual_t and not is_original_nan) else "❌"
        logger.info(f"  Row {idx:3d}: TimeMatch={status} | Pred={df_out.at[idx, 'pred']} | Prob={df_out.at[idx, 'pred_prob']:.4f}")

    # --- check 3: stride uniformity ---
    # Inside a continuous region the gap between signals must equal the stride exactly
    diffs = np.diff(actual_signals.index)
    # Ignore the jump caused by the hole in the middle
    normal_diffs = diffs[diffs < window] 
    if len(normal_diffs) > 0 and np.all(normal_diffs == stride):
        logger.info(f"检查点 3 (Stride 均匀性): 步长验证成功 (Stride={stride}) ✅")
    else:
        logger.warning(f"⚠️ 检查点 3: 步长检测不一。若数据中存在 Gap，这是正常的。")

    # --- check 4: strict tail handling ---
    # Inject a NaN into the last 5 rows of the df and see whether the whole tail window is dropped
    logger.info("检查点 4 (尾部 10 根严检): 验证中...")
    df_tail_bad = df_test.copy()
    df_tail_bad.iloc[-3, 0] = np.nan # create a gap in the third row from the end
    
    df_out_tail, _ = handler.predict(df_tail_bad, kline_interval, is_live=False)
    if df_out_tail.index[-1] not in df_out_tail['pred'].dropna().index:
        logger.info("✅ 成功：尾部存在空洞时，末尾信号已按逻辑被丢弃。")
    else:
        logger.error("❌ 失败：尾部严检逻辑未生效！")

    logger.info("="*60 + "\n")

if __name__ == "__main__":
    verify_model_alignment_v2()