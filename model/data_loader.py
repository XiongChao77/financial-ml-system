# data_loader.py (新文件)
from typing import List
import numpy as np
import pandas as pd
import torch,os
from torch.utils.data import Dataset

PRICE_BASE_FEATURES = [ 'SMA' , 'EMA' ,'open', 'high', 'low', 'close'] #will match features by this, and the basic is price
VOLUME_BASE_FEATURES = ['volume','taker_buy_base_volume', ]  #the basic is volume
QUOTE_ASSET_VOLUME_FEATURES = ['quote_asset_volume','taker_buy_quote_volume' ]   #the basic is quote_asset
MACD_FEATURES =['MACD'] #base:MACD_DEA
KDJ_FEATURES =['KDJ'] #base:KDJ_D
SELF_BASE_FEATURES = ['number_of_trades','SLOPE_REG_','N_SLOPE_REG_','RSI']   #the basic is theirself
DROP_FEATURES =['threshold','label','open_time_dt_utc', 'close_time_dt_utc']

# --- 基准映射表 ---
BASIS_MAP = {
    'PRICE_BASE': 'close',
    'VOLUME_BASE': 'volume',
    'QUOTE_ASSET_VOLUME_BASE': 'quote_asset_volume',
    'MACD_BASE': 'MACD_DEA',
    'KDJ_BASE': 'KDJ_D',
}

# 辅助函数：根据特征名返回所需的基准列名 (Basis Column Name)
def get_scaling_basis_col(feature_name: str) -> str | None:
    """
    通过子串匹配，确定特征应该使用哪个 t=0 原始特征作为缩放基准。
    """
    
    # --- 1. PRICE_BASE: 匹配所有价格相关指标 (MACD/SMA/EMA) ---
    # 这类指标通常有后缀 (e.g., MACD_N, SMA_5D_N)，使用子串匹配确保识别。
    for match_str in PRICE_BASE_FEATURES:
        if feature_name.startswith(match_str):
            return BASIS_MAP['PRICE_BASE'] # 返回 'close'

    # --- 2. QUOTE_ASSET_VOLUME_FEATURES: 匹配成交额相关特征 ---
    # 优先匹配成交额组，以避免 'volume' 产生歧义。
    for match_str in QUOTE_ASSET_VOLUME_FEATURES:
        if feature_name.startswith(match_str):
            return BASIS_MAP['QUOTE_ASSET_VOLUME_BASE'] # 返回 'quote_asset_volume'

    # --- 3. VOLUME_BASE_FEATURES: 匹配基础成交量特征 ---
    for match_str in VOLUME_BASE_FEATURES:
        if feature_name.startswith(match_str):
            return BASIS_MAP['VOLUME_BASE'] # 返回 'volume'

    for match_str in MACD_FEATURES:
        if feature_name.startswith(match_str):
            return BASIS_MAP['MACD_BASE'] # 返回 'MACD_BASE'
        
    for match_str in KDJ_FEATURES:
        if feature_name.startswith(match_str):
            return BASIS_MAP['KDJ_BASE'] # 返回 'KDJ_D'
        
    # ---  . SELF_BASE_FEATURES: 匹配自身基准特征 ---
    for match_str in SELF_BASE_FEATURES:
        if feature_name.startswith(match_str):
            return feature_name
        
    # --- 5. 默认处理 ---
    # 未匹配到的特征（例如：RSI, KDJ, open_change_pos, 原始 OHLCV）不进行处理
    # 这些特征将保持原值，或需要在其他地方定义它们的基准。
    return None

# === 核心逻辑：可替换的缩放函数 ===
# 这里可以定义多种缩放策略：
def scale_policy_relative_start(X3d: np.ndarray, price_idx: np.ndarray):
    """ 实现原有逻辑：除以窗口的第一个时间步的均值。 """
    # (实现 _minmax_group_inplace 的逻辑)
    if price_idx.size == 0: return None
    sub = X3d[:, :, price_idx]
    first_mean = sub[:, :1, :].mean(axis=2, keepdims=True)
    denom = np.where(first_mean != 0, first_mean, 1.0)
    X3d[:, :, price_idx] = sub / denom
    
def scale_policy_zscore(X3d: np.ndarray, price_idx: np.ndarray):
    """ 实现 Z-Score 归一化。 """
    if price_idx.size == 0: return None
    sub = X3d[:, :, price_idx]
    mean_window = sub.mean(axis=(1, 2), keepdims=True)
    std_window = sub.std(axis=(1, 2), keepdims=True)
    eps = 1e-6
    X3d[:, :, price_idx] = (sub - mean_window) / (std_window + eps)

def scale_policy_none(X3d: np.ndarray, price_idx: np.ndarray):
    """ 不做任何缩放处理 (用于测试)。 """
    return X3d

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
            print(f"\n--- Data Length after Drop: {len(df_clean)}---")
            if df_clean.empty:
                 raise RuntimeError("Dataset became empty after dropping NaNs. Check TA windows.")

        values = df_clean[clean_feature_cols].to_numpy(dtype=np.float32, copy=True)   # [N, F]
        labels = df_clean[label_col].astype(int).to_numpy()                       # [N]
        # === 过滤逻辑结束 ===
        
        # 1. 划分窗口 (Partitioning)
        X3d = _as_strided_windows(values, window) # [M, T, F]
        y_all = labels[window-1:]
        
        # 2. 统一动态缩放 (保留原有的基准列逻辑，但改为 Z-Score)
        f_map = {col: i for i, col in enumerate(clean_feature_cols)} # 注意使用 clean_feature_cols
        eps = 1e-6 # 防止除以 0
        
        # === 【核心修复 1】提前复制所有基准列的原始数据 ===
        # 存储所有基准列的 3D 窗口数据副本，防止在缩放循环中被修改
        basis_data_cache = {}
        required_bases = set(BASIS_MAP.values())
        
        for basis_col in required_bases:
            if basis_col in f_map:
                basis_idx = f_map[basis_col]
                # 复制原始窗口数据，确保后续循环中使用的统计量是基于原始值计算的
                basis_data_cache[basis_col] = X3d[:, :, basis_idx:basis_idx + 1].copy()
            else:
                # 这不应该发生，除非配置错误
                raise RuntimeError(f"Critical Error: Required basis column '{basis_col}' is missing.")
        
        # === 【核心修复 2】遍历特征并使用缓存的原始数据进行缩放 ===
        for f_idx, feature_name in enumerate(clean_feature_cols):
            
            if feature_name == 'SMA_7W' or feature_name == 'SMA_25W':
                pass
            # 1. 获取缩放基准列名
            basis_col = get_scaling_basis_col(feature_name)

            if basis_col is None:
                continue # 未定义的特征，跳过缩放
            # 2. 提取基准列数据并计算统计量
            if basis_col == feature_name:
                # 自缩放：使用当前特征的原始数据
                basis_window = X3d[:, :, f_idx:f_idx + 1] # 自缩放不涉及依赖其他列，直接使用自身未缩放的版本
            else:
                # 依赖其他列：使用缓存的原始基准数据
                if basis_col not in basis_data_cache:
                     raise RuntimeError(f"Error: Basis column '{basis_col}' not found in cache.")
                basis_window = basis_data_cache[basis_col]
            
            # 计算基准列在每个窗口内的均值和标准差
            # axis=1 表示沿时间轴计算，keepdims=True 保持维度以便广播 [M, 1, 1]
            mu = np.mean(basis_window, axis=1, keepdims=True)
            sigma = np.std(basis_window, axis=1, keepdims=True)

            # --- 【新增调试陷阱】抓取 SMA_25W 的计算细节 ---
            if feature_name == 'SMA_25W' or feature_name == 'SMA_7W': 
                # 1. 临时计算一下缩放结果
                temp_denom = sigma + eps
                temp_scaled = (X3d[:, :, f_idx:f_idx + 1] - mu) / temp_denom
                
                # 2. 查找绝对值最大的位置
                max_val = np.max(np.abs(temp_scaled))
                
                # 3. 如果最大值异常大（比如 > 20），打印计算过程
                if max_val > 20.0:
                    # 找到最大值的索引
                    flat_idx = np.argmax(np.abs(temp_scaled))
                    # 转换回 (Window, Time, Feature) 坐标
                    m, t, _ = np.unravel_index(flat_idx, temp_scaled.shape)
                    
                    print(f"\n🔍 [DEBUG TRAP] Investigating SMA_25W Explosion")
                    print(f"---------------------------------------------")
                    print(f"Location -> Window Index: {m}, Time Step: {t}")
                    print(f"Equation -> (Raw_Value - Mu) / Sigma = Result")
                    
                    raw_val = X3d[m, t, f_idx]
                    curr_mu = mu[m, 0, 0]
                    curr_sigma = sigma[m, 0, 0]
                    
                    print(f"  Raw Value (X) : {raw_val:.4f}")
                    print(f"  Mean (Mu)     : {curr_mu:.4f}")
                    print(f"  Diff (X - Mu) : {raw_val - curr_mu:.6f}")
                    print(f"  Sigma (Std)   : {curr_sigma:.8f}  <-- 重点看这个！")
                    print(f"  Calc Result   : {(raw_val - curr_mu) / (curr_sigma + eps):.4f}")
                    print(f"---------------------------------------------")
                    
                    # 只要抓到一次典型的就够了，不用一直刷屏，可以取消下面的注释让它只报一次
                    # break

            # --- C. 应用 Z-Score 缩放 (修改点) ---
            # 使用基准列的均值和标准差来缩放当前特征
            # 这样如果是 Open/High/Low，它们都会减去 Close 的均值，除以 Close 的标准差
            # 从而保留了 K 线结构，同时将数值范围限制在合理区间
            X3d[:, :, f_idx:f_idx + 1] = (X3d[:, :, f_idx:f_idx + 1] - mu) / (sigma + eps)

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

def _as_strided_windows(a2d: np.ndarray, window: int) -> np.ndarray:
    """
    Turn [N, F] into overlapping [M, T, F] with stride tricks, then copy for safety.
    M = N - T + 1
    """
    N, F = a2d.shape
    M = N - window + 1
    if M <= 0:
        raise ValueError(f"Data length {N} must be >= window {window}.")
    s0, s1 = a2d.strides
    view = np.lib.stride_tricks.as_strided(a2d, shape=(M, window, F), strides=(s0, s0, s1), writeable=False)
    return view.copy()