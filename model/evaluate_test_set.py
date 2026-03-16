import os,sys
import pandas as pd
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from model.model_loader import ModelHandler
from data_process import common

def run_test_evaluation():
    # 1. Environment setup
    logger, _ = common.setup_session_logger(sub_folder='evaluation')
    
    # 2. Reproduce training split logic strictly
    df = common.load_test_df()
    # window = common.BaseDefine.predict_num  # 120
    # stride = 4                       # Must match train.py full-dataset stride
    # train_ratio = 0.7
    # val_ratio = 0.15

    # # Compute total windows M and derive test start row index
    # M = (len(df) - window) // stride + 1
    # test_start_window = int(M * (train_ratio + val_ratio))
    # df_test_start_row = test_start_window * stride
    
    # # Slice raw test set data
    # df_test = df.iloc[df_test_start_row:].copy()
    # logger.info(f"📋 Sliced Test Set: {len(df_test)} rows starting from index {df_test_start_row}")

    # 3. Initialize model handler
    # Options: 'Best_F1' or 'Best_Loss', matching the two saved variants during training
    handler = ModelHandler(loss_fun='Best_F1')

    # 4. Run inference and evaluation
    # is_live=False triggers evaluate_performance to print detailed report
    # diff_thresh=None uses argmax of model output directly
    df_out, stats = handler.predict(df, kline_interval_ms = common.load_interval_ms(), is_live = False, diff_thresh = None,
                                                       cache_path=os.path.join(common.TEMPORARY_DIR,"trade_cache.pt"), use_cache = False )

    # 5. Print concise results
    print("\n" + "="*40)
    print(f"🎯 Final Test Metrics: {stats}")
    print("="*40)

if __name__ == "__main__":
    run_test_evaluation()