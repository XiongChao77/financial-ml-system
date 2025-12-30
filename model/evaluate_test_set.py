import os,sys
import pandas as pd
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from model.model_loader import ModelHandler
from data_process import common

def run_test_evaluation():
    # 1. 环境初始化
    logger, _ = common.setup_session_logger(sub_folder='evaluation')
    
    # 2. 严格复刻训练集切分逻辑
    df = common.load_test_df()
    # window = common.CANDLESTICK_NUM  # 120
    # stride = 4                       # 必须与 train.py 的全量数据集 stride 一致
    # train_ratio = 0.7
    # val_ratio = 0.15

    # # 计算窗口总数 M 并推导测试集起始行索引
    # M = (len(df) - window) // stride + 1
    # test_start_window = int(M * (train_ratio + val_ratio))
    # df_test_start_row = test_start_window * stride
    
    # # 切出测试集原始数据
    # df_test = df.iloc[df_test_start_row:].copy()
    # logger.info(f"📋 Sliced Test Set: {len(df_test)} rows starting from index {df_test_start_row}")

    # 3. 初始化模型处理器
    # 可选 'Best_F1' 或 'Best_Loss'，对应你训练保存的两个版本
    handler = ModelHandler(loss_fun='Best_F1')

    # 4. 执行推理与评估
    # is_live=False 会触发 evaluate_performance 打印详细报告
    # diff_thresh=None 表示使用模型直接预测的 argmax 结果
    df_out, stats = handler.predict_v2(df, kline_interval_ms = common.load_interval_ms(), is_live = False, diff_thresh = None,
                                                       cache_path=os.path.join(common.TEMPORARY_DIR,"trade_cache.pt"), use_cache = False )

    # 5. 输出精简结果
    print("\n" + "="*40)
    print(f"🎯 Final Test Metrics: {stats}")
    print("="*40)

if __name__ == "__main__":
    run_test_evaluation()