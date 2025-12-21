# data_loader.py (新文件)
from typing import List
import numpy as np
import pandas as pd
import torch,os,logging
from torch.utils.data import Dataset
from data_process import common

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
            kline_interval_ms: int,
            feature_cols: List[str],
            label_col: str = None,
            window: int = common.CANDLESTICK_NUM,
            is_live: bool = False,
    ):
        self.is_live = is_live
        time_col = 'open_time_ms_utc'
        self.logger = logging.getLogger()
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
        
        if True:
            # --- 🔍 智能审计逻辑：区分冷启动与中间异常 ---

            # 1. 找出所有包含 NaN 的行
            nan_mask = df_work.isna().any(axis=1)
            df_nan = df_work[nan_mask].copy()

            if not df_nan.empty:
                cold_start_count = 0
                middle_gap_count = 0
                
                # 记录每种类型的异常详情
                gap_details = [] 
                
                self.logger.debug(f"📊 [Data Audit] 总计检测到 {len(df_nan)} 行数据包含 NaN")

                # 预先计算每列的第一个有效索引（First Valid Index）
                # 这是区分冷启动的关键线索
                first_valid_indices = {col: df_work[col].first_valid_index() for col in clean_feature_cols}

                for idx, row in df_nan.iterrows():
                    actual_nan_cols = row.index[row.isna()].tolist()
                    
                    is_middle_gap = False
                    cols_middle_gap = []
                    cols_cold_start = []

                    for col in actual_nan_cols:
                        first_idx = first_valid_indices.get(col)
                        
                        # 如果当前索引比该列第一个有效值还要早，就是冷启动
                        if first_idx is None or idx < first_idx:
                            cols_cold_start.append(col)
                        else:
                            # 否则，这就是“中间空洞”，属于异常！
                            is_middle_gap = True
                            cols_middle_gap.append(col)

                    if is_middle_gap:
                        middle_gap_count += 1
                        gap_details.append(f"   - 行号 {idx:6d} | 时间: {row[time_col]} | ❌ 异常空洞列: {cols_middle_gap}")
                    else:
                        cold_start_count += 1

                # --- 汇报结果 ---
                self.logger.debug(f"✅ 冷启动缺失: {cold_start_count} 行 (属于正常计算窗口等待)")
                self.logger.debug(f"🚨 中间异常缺失: {middle_gap_count} 行 (请检查数据源或特征逻辑!)")

                if gap_details:
                    self.logger.debug("📍 中间异常详情 (前 20 条):")
                    for detail in gap_details[:20]:
                        self.logger.debug(detail)

                # 汇总受灾特征
                self.logger.debug("🧐 缺失特征分布:")
                missing_stats = df_nan.isna().sum()
                for col_name, count in missing_stats[missing_stats > 0].items():
                    type_str = "冷启动" if idx < (first_valid_indices.get(col_name) or 0) else "可能含异常"
                    self.logger.debug(f"   - [{col_name:15s}]: 缺失 {count:6d} 行")

        
        # 回测模式下记录原始索引映射
        if not self.is_live:
            df_work['orig_index'] = df.index
        
        df_work.dropna(inplace=True)
        df_work.reset_index(drop=True, inplace=True)
        self.logger.debug(f"--- Data Length after Drop: {len(df_work)}---")
        if df_work.empty:
            raise RuntimeError("Dataset became empty after dropping NaNs.")

        # 4. 提取基础 array
        values = df_work[clean_feature_cols].to_numpy(dtype=np.float32)
        timestamps = df_work[time_col].to_numpy(dtype=np.int64)

        # 5. 窗口化
        X3d = _as_strided_windows(values, window) # [M, T, F]
        time_windows = _as_strided_windows(timestamps.reshape(-1, 1), window).squeeze(-1)

        assert X3d.shape[0] == time_windows.shape[0], "特征与时间窗口不一致！"

        if True:
            # 6. 连续性检查逻辑 (优化版)
            # 直接使用从 JSON 读取的精确值
            interval_ms = kline_interval_ms

            # A. 全局检查 (Window内缺失 <= 5)
            global_actual_span = time_windows[:, -1] - time_windows[:, 0]
            # 使用物理标准计算理想跨度
            global_ideal_span = (window - 1) * interval_ms
            mask_global = (global_actual_span - global_ideal_span) <= (5.1 * interval_ms)

            # B. 尾部检查 (末端 10 个缺失 <= 2)
            check_tail_count = 10
            if window > check_tail_count:
                tail_actual_span = time_windows[:, -1] - time_windows[:, -(check_tail_count + 1)]
                tail_ideal_span = check_tail_count * interval_ms
                mask_tail = (tail_actual_span - tail_ideal_span) <= (2 * interval_ms)
            else:
                mask_tail = True

            valid_mask = mask_global & mask_tail
        else:
            # === 回退逻辑 ===
            # 创建一个全为 True 的掩码，即保留所有窗口
            valid_mask = np.ones(len(X3d), dtype=bool)
            self.logger.debug("⚠️ [Continuity] Check SKIPPED (check_continuity=False). All windows kept.")
        # 7. 【核心修改】：执行统一过滤 (输入连续性 + 标签有效性)
        original_count = len(X3d)
        
        # 先获取所有窗口对应的原始标签
        # (df_work 已经 reset_index 了，所以 window-1: 正好对应窗口最后一根)
        labels_all = df_work[label_col].values[window-1:] if has_label else np.zeros(original_count)
        
        if has_label:
            # 创建标签有效性掩码：只有不等于 -1 的才是有效的
            label_valid_mask = (labels_all != common.Signal.INVALID)
            # 最终掩码 = 输入连续 & 标签有效
            final_mask = valid_mask & label_valid_mask
        else:
            final_mask = valid_mask
        if True:
            # ---------------------------------------------------------
            # 🔍 【时空全显】打印异常窗口内的每一根 K 线时间戳
            # ---------------------------------------------------------
            # 找到被 mask_global 或 mask_tail 拦截的索引
            failed_indices = np.where(~valid_mask)[0]

            if len(failed_indices) > 0:
                self.logger.debug(f"\n🚨 [CRITICAL AUDIT] 发现 {len(failed_indices)} 个异常窗口。")
                
                # 为了防止刷屏，我们只展示前 2 个被拦截的窗口明细
                for window_idx in failed_indices[:2]:
                    # 提取该窗口内的所有 136 个时间戳
                    internal_timestamps = time_windows[window_idx] 
                    
                    self.logger.debug(f"\n📂 窗口索引: {window_idx} (明细清单):")
                    self.logger.debug(f"{'Step':>5} | {'Row_in_df':>10} | {'Timestamp_ms':>15} | {'Diff_with_prev'}")
                    self.logger.debug("-" * 65)

                    for step in range(len(internal_timestamps)):
                        curr_t = internal_timestamps[step]
                        # 计算与上一根的差值
                        diff_str = ""
                        if step > 0:
                            diff = curr_t - internal_timestamps[step-1]
                            diff_str = f"<- {diff} ms"
                            # 如果差值不是 interval_ms，高亮标注
                            if diff != interval_ms:
                                diff_str += " ⚠️ [GAP DETECTED]"
                        
                        # 计算该数据在 df_work 中的实际行号
                        actual_row = window_idx + step
                        self.logger.debug(f"{step:5d} | {actual_row:10d} | {curr_t:15d} | {diff_str}")
                    
                    self.logger.debug(f"\n[Summary] 窗口 {window_idx}: 理想跨度 {global_ideal_span} | 实际跨度 {global_actual_span[window_idx]}")
                
                if len(failed_indices) > 2:
                    self.logger.debug(f"\n... 还有 {len(failed_indices)-2} 个异常窗口未展开明细。")
            
            self.logger.debug("\n" + "="*80 + "\n")

        if True:
            # ---------------------------------------------------------
            # 🔍 增强 Debug：追踪被过滤的行号与原因
            # ---------------------------------------------------------
            # 建立原始索引参考 (窗口最后一根 K 线在 df_work 中的位置)
            all_indices = np.arange(window - 1, window - 1 + original_count)
            
            # 1. 找出被连续性检查 (valid_mask) 过滤掉的行
            mask_continuity_dropped = ~valid_mask
            if np.any(mask_continuity_dropped):
                dropped_idx = all_indices[mask_continuity_dropped]
                df_dropped_cont = df_work.iloc[dropped_idx]
                self.logger.debug(f"❌ [Debug] 因时间不连续被过滤: {len(df_dropped_cont)} 行")
                # 打印前 5 条详细信息
                for i, (idx, row) in enumerate(df_dropped_cont.head(40).iterrows()):
                    self.logger.debug(f"   行号: {idx} | 时间: {row[time_col]} | 原因: 时间跨度异常")

            # 2. 找出被标签有效性 (label_valid_mask) 过滤掉的行
            if has_label:
                mask_label_dropped = ~label_valid_mask
                if np.any(mask_label_dropped):
                    # 注意：这里我们只看通过了连续性检查但标签无效的，或者看全部标签无效的
                    dropped_idx = all_indices[mask_label_dropped]
                    df_dropped_label = df_work.iloc[dropped_idx]
                    self.logger.debug(f"⚠️ [Debug] 因标签为 INVALID(-1) 被过滤: {len(df_dropped_label)} 行")
                    # 打印前 5 条详细信息
                    for i, (idx, row) in enumerate(df_dropped_label.head(40).iterrows()):
                        self.logger.debug(f"   行号: {idx} | 时间: {row[time_col]} | 标签值: {row[label_col]}")

        # 3. 统计最终留存情况
        self.logger.info(f"✅ 过滤汇总: 原始 {original_count} -> 连续性过滤后 {np.sum(valid_mask)} -> 最终剩余 {np.sum(final_mask)}")
        # 执行物理过滤
        X3d_filtered = X3d[final_mask]
        y_filtered = labels_all[final_mask]

        # 处理索引映射 (用于 predict_v2 的精准回填)
        if not self.is_live:
            all_window_indices = df_work['orig_index'].values[window-1:]
            self.indices = all_window_indices[final_mask]
        else:
            self.indices = None

        # 打印过滤信息，让我们知道丢了多少“断层”数据
        dropped = original_count - len(X3d_filtered)
        if dropped > 0:
            self.logger.warning(f"⚠️ [Dataset] Dropped {dropped} samples (Incomplete windows or Invalid labels).")
            self.logger.warning(f"📊 Remaining: {len(X3d_filtered)} ({len(X3d_filtered)/original_count:.2%})")
            
        # 8. 归一化 (在过滤后的数据上执行)
        feature_factory = common.FeatureFactory(common.FEATURE_CONFIG_LIST, kline_interval_ms)
        feature_factory.normalize(X3d_filtered, clean_feature_cols)

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

            self.logger.info(f"\n[Debug] Saving processed data to {output_dir} ...")

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
            
            self.logger.debug(f"[Debug] Snapshot CSV saved: {csv_path}")
            self.logger.debug(f"[Debug] Check this CSV to verify if values are in range [-3, 3] (approx).")
        else:
            self.logger.debug(f"\n[Debug] Skip saving files (save_file=False). Showing statistics only...")

        # --- B. 统计检查逻辑 (始终执行) ---
        # 3. 简单统计检查 (直接打印到控制台)
        self.logger.debug("\n=== Data Statistics Check (Last Step) ===")
        
        # 检查是否有 NaN
        nan_count = np.isnan(last_step_data).sum()
        self.logger.debug(f"Total NaNs in processed data: {nan_count}")
        if nan_count > 0:
            self.logger.debug("!!! WARNING: NaNs found in processed data. Check scaling logic! !!!")
        
        # 打印前几列的均值方差，确认是否接近 0 和 1
        self.logger.debug(f"{'Feature':<25} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}")
        self.logger.debug("-" * 75)
        for i, col in enumerate(self.feature_names):
            # 防止特征太多刷屏
            # if i >= 20: 
            #     self.logger.debug(f"... and {len(self.feature_names) - 20} more features")
            #     break
            
            col_data = last_step_data[:, i]
            self.logger.debug(f"{col:<25} | {col_data.mean():.4f}     | {col_data.std():.4f}     | {col_data.min():.4f}     | {col_data.max():.4f}")
        self.logger.debug("=========================================\n")

    # ========== 【新增】 最终数据异常值检测方法 ==========
    def inspect_final_data(self, clip_limit: float = 5.0):
            """
            检查最终的 self.X 数据中是否存在超出给定 Z-Score 限制的异常值。
            打印 Nan/Inf 出现的位置，并按绝对值打印最大的 5 个异常值。
            """
            self.logger.debug(f"\n--- Starting Final Data Outlier Inspection (Limit: +/- {clip_limit:.1f}) ---")
            
            # 转换为 NumPy 数组进行检查
            X_np = self.X.numpy()
            
            # 1. 检查 NaN 或 Inf
            nan_or_inf_mask = np.isnan(X_np) | np.isinf(X_np)
            if nan_or_inf_mask.any():
                self.logger.debug("🚨 CRITICAL ERROR: Found NaN or Inf values in the final processed data (self.X).")
                nan_loc = np.where(nan_or_inf_mask)
                if nan_loc[0].size > 0:
                    f_idx = nan_loc[2][0]
                    self.logger.debug(f"  -> First NaN/Inf detected at Window: {nan_loc[0][0]}, Feature: {self.feature_names[f_idx]}")
                
            # 2. 检查 Z-Score Outliers (超过 ±clip_limit)
            is_outlier = (X_np > clip_limit) | (X_np < -clip_limit)
            outlier_locs = np.where(is_outlier)
            
            if outlier_locs[0].size > 0:
                self.logger.debug(f"⚠️ WARNING: Found {outlier_locs[0].size} total outliers exceeding +/- {clip_limit:.1f}.")
                
                # --- 新增逻辑: 排序并打印最大的 5 个异常值 ---
                
                # 1. 提取所有异常值
                # 使用 outlier_locs 数组作为索引，提取对应的值
                outlier_values = X_np[outlier_locs]
                
                # 2. 获取按绝对值排序的索引 (降序)
                sorted_indices = np.argsort(np.abs(outlier_values))[::-1]
                
                self.logger.debug(f"--- Top {min(5, sorted_indices.size)} Largest Outliers (by Magnitude) ---")
                
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
                    
                    self.logger.debug(f"  [TOP {i+1}] Window: {w_idx:<5} | Bar: {t_idx:<3} | Feature: {feature_name:<20} | Value: {value:.4f}")

            else:
                self.logger.debug("✅ All data points are within the +/- limit. Data quality is good.")
                
            self.logger.debug("-" * 40)

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