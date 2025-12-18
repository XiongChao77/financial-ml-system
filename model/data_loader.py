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
            label_col: str = None,
            window: int = CANDLESTICK_NUM,
            is_live: bool = False,
    ):
        self.is_live = is_live
        time_col = 'open_time_ms_utc'
        
        # 1. 基础检查
        if time_col not in df.columns:
            raise ValueError(f"Dataset 必须包含 '{time_col}' 列以进行时间连续性检查！")

        # 2. 列准备
        clean_feature_cols = [col for col in feature_cols if col not in DROP_FEATURES]
        self.feature_names = clean_feature_cols
        self.feature_count = len(clean_feature_cols)

        cols_to_extract = list(clean_feature_cols)
        has_label = (label_col is not None) and (label_col in df.columns)
        if has_label: cols_to_extract.append(label_col)
        if time_col not in cols_to_extract: cols_to_extract.append(time_col)

        # 3. 统一清洗 (仅保留一份 df_work 节省内存)
        df_work = df[cols_to_extract].copy()
        
        # 回测模式下记录原始索引映射
        if not self.is_live:
            df_work['orig_index'] = df.index
        
        df_work.dropna(inplace=True)
        df_work.reset_index(drop=True, inplace=True)
        print(f"--- Data Length after Drop: {len(df_work)}---")
        if df_work.empty:
            raise RuntimeError("Dataset became empty after dropping NaNs.")

        # 4. 提取基础 array
        values = df_work[clean_feature_cols].to_numpy(dtype=np.float32)
        timestamps = df_work[time_col].to_numpy(dtype=np.int64)

        # 5. 窗口化
        X3d = _as_strided_windows(values, window) # [M, T, F]
        time_windows = _as_strided_windows(timestamps.reshape(-1, 1), window).squeeze(-1)

        assert X3d.shape[0] == time_windows.shape[0], "特征与时间窗口不一致！"

        if False:
            # 6. 连续性检查逻辑
            global_diffs = np.diff(timestamps)
            expected_interval = np.median(global_diffs) if len(global_diffs) > 0 else 0
            
            # A. 全局检查 (Window内缺失 <= 5)
            global_actual_span = time_windows[:, -1] - time_windows[:, 0]
            global_ideal_span = (window - 1) * expected_interval
            mask_global = ((global_actual_span - global_ideal_span) / (expected_interval + 1e-9)) <= 5.1

            # B. 尾部检查 (末端10个缺失 <= 2)
            check_tail_count = 10
            if window > check_tail_count:
                tail_actual_span = time_windows[:, -1] - time_windows[:, -(check_tail_count + 1)]
                tail_ideal_span = check_tail_count * expected_interval
                mask_tail = ((tail_actual_span - tail_ideal_span) / (expected_interval + 1e-9)) <= 2.1
            else:
                mask_tail = True

            valid_mask = mask_global & mask_tail
        else:
                    # === 回退逻辑 ===
                    # 创建一个全为 True 的掩码，即保留所有窗口
                    valid_mask = np.ones(len(X3d), dtype=bool)
                    print("⚠️ [Continuity] Check SKIPPED (check_continuity=False). All windows kept.")
        # 7. 【关键步骤】：执行统一过滤
        original_count = len(X3d)
        X3d_filtered = X3d[valid_mask]
        
        # 处理标签
        labels = df_work[label_col].values[window-1:] if has_label else np.zeros(original_count)
        y_filtered = labels[valid_mask]

        # 处理索引映射
        if not self.is_live:
            all_window_indices = df_work['orig_index'].values[window-1:]
            self.indices = all_window_indices[valid_mask]
        else:
            self.indices = None

        # 打印过滤信息
        dropped = original_count - len(X3d_filtered)
        if dropped > 0:
            print(f"⚠️ [Continuity] Dropped {dropped} windows. Remaining: {len(X3d_filtered)} ({len(X3d_filtered)/original_count:.2%})")
        # 【新增：安全性检查】
        if len(X3d_filtered) == 0:
            raise RuntimeError("经过时间连续性检查后，没有任何窗口活下来！请检查数据质量。")
        # 8. 归一化 (在过滤后的数据上执行)
        feature_factory = FeatureFactory(FEATURE_CONFIG)
        feature_factory.normalize(X3d_filtered, clean_feature_cols)
        FeatureOrigin().normalize(X3d_filtered, clean_feature_cols, feature_factory)

        # 9. 最终赋值给 Tensor
        self.X = torch.from_numpy(X3d_filtered)
        self.y = torch.from_numpy(y_filtered).long() # 确保标签是 long 类型

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# ========== 【修改】 增加 save_file 参数控制是否落盘 ==========
    def save_debug_data(self, output_dir: str, save_file: bool = True):
        """
        检查并保存处理后的数据。
        
        Args:
            output_dir (str): 输出目录路径
            save_file (bool): 如果为 True，则保存 .pt 和 .csv 文件；
                              如果为 False，仅在控制台打印统计信息 (用于快速 Debug)。
        """
        # 0. 预先提取最后一帧数据 (用于后续的保存或统计)
        # 形状变换: [M, T, F] -> [M, F] (取 T的最后一个索引)
        last_step_data = self.X[:, -1, :].numpy()

        # --- A. 文件保存逻辑 (仅当 save_file=True 时执行) ---
        if save_file:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print(f"\n[Debug] Saving processed data to {output_dir} ...")

            # 1. 保存完整的 Tensor 数据
            torch.save({
                "X": self.X,
                "y": self.y,
                "features": self.feature_names
            }, os.path.join(output_dir, "debug_full_tensor.pt"))

            # 2. 生成人类可读的 CSV
            df_debug = pd.DataFrame(last_step_data, columns=self.feature_names)
            df_debug["label"] = self.y.numpy()
            
            csv_path = os.path.join(output_dir, "debug_snapshot_last_step.csv")
            # 使用 float_format 确保精度
            df_debug.to_csv(csv_path, index_label="window_idx", float_format='%.6f')
            
            print(f"[Debug] Snapshot CSV saved: {csv_path}")
            print(f"[Debug] Check this CSV to verify if values are in range [-3, 3] (approx).")
        else:
            print(f"\n[Debug] Skip saving files (save_file=False). Showing statistics only...")

        # --- B. 统计检查逻辑 (始终执行) ---
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
            # 防止特征太多刷屏
            # if i >= 20: 
            #     print(f"... and {len(self.feature_names) - 20} more features")
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