# data_loader.py (新文件)
from typing import List
import numpy as np
import pandas as pd
import torch,os
from torch.utils.data import Dataset
from data_process.common import *

#number_of_trades 和vloume高度重合，统计相关性低,quote_asset_volume和vloume高度重合.
#taker_buy_quote_volume--taker_buy_base_volume,
DROP_FEATURES =['threshold', 'stop_threshold', 'label', 'return_rate', 'open_time_ms_utc', 'open_time_date_utc',
                 'close_time_ms_utc', 'ignore' ]
LOW_CORRELATION_FEATURES = ['number_of_trades','quote_asset_volume', 'taker_buy_quote_volume']

# ====================================================================
# --- 2. TimeSeriesWindowDataset CLASS ---
# ====================================================================
class TimeSeriesWindowDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str], 
        label_col: str, 
        window: int
    ):
        # === 过滤逻辑：必须在提取 values 之前执行 ===
        clean_feature_cols = [col for col in feature_cols if col not in DROP_FEATURES]
        self.feature_names = clean_feature_cols
        self.feature_count = len(clean_feature_cols)
        
        # === 【修复 1】: 彻底清洗 NaN ===
        # 这一步至关重要！任何技术指标产生的 NaN 都会导致训练崩溃
        df_clean = df[clean_feature_cols + [label_col]].copy()
        
        # 检查是否有 NaN
        nan_rows = df_clean[df_clean.isnull().any(axis=1)]

        if not nan_rows.empty:
            print(f"Warning: Found {nan_rows.shape[0]} rows containing NaNs (Total NaNs: {df_clean.isnull().sum().sum()}).")
            
            # 【新增调试输出】打印前 5 行和后 5 行包含 NaN 的数据，或只打印后 10 行。
            # 这里选择打印后 10 行，因为技术指标的 NaN 通常出现在时间序列的前端。
            # print("\n--- Last 10 rows containing NaNs before dropping ---")
            # print(nan_rows.tail(10).to_string())
            # print("----------------------------------------------------\n")
            
            df_clean.dropna(inplace=True)
            df_clean.reset_index(drop=True, inplace=True)
            print(f"--- Data Length after Drop: {len(df_clean)}---")
            if df_clean.empty:
                 raise RuntimeError("Dataset became empty after dropping NaNs. Check TA windows.")

        values = df_clean[clean_feature_cols].to_numpy(dtype=np.float32, copy=True)   # [N, F]
        labels = df_clean[label_col].astype(int).to_numpy()                       # [N]
        # === 过滤逻辑结束 ===
        
        # 1. 划分窗口 (Partitioning)
        X3d = _as_strided_windows(values, window) # [M, T, F]
        y_all = labels[window-1:]
        
        FeatureFactory(FEATURE_CONFIG).normalize(X3d , clean_feature_cols)
        FeatureOrigin().normalize(X3d , clean_feature_cols)

        # 3. 转换为 PyTorch Tensor
        self.X = torch.from_numpy(X3d)
        self.y = torch.from_numpy(y_all)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

    # ========== 【新增】 保存处理后数据的方法 ==========
    def save_debug_data(self, output_dir: str):
        """
        将处理后的 Tensor 数据保存到文件，用于检查归一化结果。
        1. debug_full_tensor.pt: 完整的 (M, T, F) 数据
        2. debug_snapshot_last_step.csv: 每个窗口最后一个时间步的数据 (M, F) -> 最直观，用来查错
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\n[Debug] Saving processed data to {output_dir} ...")

        # 1. 保存完整的 Tensor 数据
        torch.save({
            "X": self.X,
            "y": self.y,
            "features": self.feature_names
        }, os.path.join(output_dir, "debug_full_tensor.pt"))

        # 2. 生成人类可读的 CSV (取每个窗口的最后一帧 t = window-1)
        # 这代表了模型在做预测那个时刻看到的“归一化后”特征值
        # 形状变换: [M, T, F] -> [M, F] (取 T的最后一个索引)
        last_step_data = self.X[:, -1, :].numpy()
        
        df_debug = pd.DataFrame(last_step_data, columns=self.feature_names)
        df_debug["label"] = self.y.numpy()
        
        csv_path = os.path.join(output_dir, "debug_snapshot_last_step.csv")
        # **【关键修复】** 使用 float_format 确保输出足够的精度，避免 CSV 写入问题
        df_debug.to_csv(csv_path, index_label="window_idx", float_format='%.6f')
        
        print(f"[Debug] Snapshot CSV saved: {csv_path}")
        print(f"[Debug] Check this CSV to verify if values are in range [-3, 3] (approx).")
        
        # 3. 简单统计检查 (直接打印到控制台)
        print("\n=== Data Statistics Check (Last Step) ===")
        # 检查是否有 NaN
        nan_count = np.isnan(last_step_data).sum()
        print(f"Total NaNs in processed data: {nan_count}")
        if nan_count > 0:
            print("!!! WARNING: NaNs found in processed data. Check scaling logic! !!!")
        
        # 打印前几列的均值方差，确认是否接近 0 和 1
        print(f"{'Feature':<25} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}")
        print("-" * 75)
        for i, col in enumerate(self.feature_names):
            # # 为了不刷屏，只打印前10个特征
            # if i >= 10: 
            #     print(f"... and {len(self.feature_names) - 10} more features")
            #     break
            col_data = last_step_data[:, i]
            print(f"{col:<25} | {col_data.mean():.4f}     | {col_data.std():.4f}     | {col_data.min():.4f}     | {col_data.max():.4f}")
        print("=========================================\n")

    # ========== 【新增】 最终数据异常值检测方法 ==========
    def inspect_final_data(self, clip_limit: float = 5.0):
            """
            检查最终的 self.X 数据中是否存在超出给定 Z-Score 限制的异常值。
            打印 Nan/Inf 出现的位置，并按绝对值打印最大的 5 个异常值。
            """
            print(f"\n--- Starting Final Data Outlier Inspection (Limit: +/- {clip_limit:.1f}) ---")
            
            # 转换为 NumPy 数组进行检查
            X_np = self.X.numpy()
            
            # 1. 检查 NaN 或 Inf
            nan_or_inf_mask = np.isnan(X_np) | np.isinf(X_np)
            if nan_or_inf_mask.any():
                print("🚨 CRITICAL ERROR: Found NaN or Inf values in the final processed data (self.X).")
                nan_loc = np.where(nan_or_inf_mask)
                if nan_loc[0].size > 0:
                    f_idx = nan_loc[2][0]
                    print(f"  -> First NaN/Inf detected at Window: {nan_loc[0][0]}, Feature: {self.feature_names[f_idx]}")
                
            # 2. 检查 Z-Score Outliers (超过 ±clip_limit)
            is_outlier = (X_np > clip_limit) | (X_np < -clip_limit)
            outlier_locs = np.where(is_outlier)
            
            if outlier_locs[0].size > 0:
                print(f"⚠️ WARNING: Found {outlier_locs[0].size} total outliers exceeding +/- {clip_limit:.1f}.")
                
                # --- 新增逻辑: 排序并打印最大的 5 个异常值 ---
                
                # 1. 提取所有异常值
                # 使用 outlier_locs 数组作为索引，提取对应的值
                outlier_values = X_np[outlier_locs]
                
                # 2. 获取按绝对值排序的索引 (降序)
                sorted_indices = np.argsort(np.abs(outlier_values))[::-1]
                
                print(f"--- Top {min(5, sorted_indices.size)} Largest Outliers (by Magnitude) ---")
                
                # 3. 打印前 5 个最大异常值的位置
                for i in range(min(5, sorted_indices.size)):
                    
                    # 获取在 outlier_locs 数组中的原始索引
                    original_outlier_idx = sorted_indices[i] 
                    
                    # 使用原始索引获取三维坐标
                    w_idx, t_idx, f_idx = (outlier_locs[0][original_outlier_idx],
                                        outlier_locs[1][original_outlier_idx],
                                        outlier_locs[2][original_outlier_idx])
                    
                    value = outlier_values[original_outlier_idx]
                    feature_name = self.feature_names[f_idx]
                    
                    print(f"  [TOP {i+1}] Window: {w_idx:<5} | Bar: {t_idx:<3} | Feature: {feature_name:<20} | Value: {value:.4f}")

            else:
                print("✅ All data points are within the +/- limit. Data quality is good.")
                
            print("-" * 40)

def _as_strided_windows(a2d: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """
    Turn [N, F] into overlapping [M, T, F] with custom stride S.
    M = (N - T) // S + 1
    """
    S = max(1, int(stride)) # 确保步长至少为 1
    N, F = a2d.shape
    M = (N - window) // S + 1 # 计算新的样本数量 M
    
    if M <= 0:
        raise ValueError(f"Data length {N} must be >= window {window}.")
    
    s0, s1 = a2d.strides
    
    # 关键修改：样本轴（第一个维度 M）的步长现在是 s0 * S
    view = np.lib.stride_tricks.as_strided(
        a2d, 
        shape=(M, window, F), 
        strides=(s0 * S, s0, s1), # <--- 引入步长 S
        writeable=False
    )
    return view.copy()