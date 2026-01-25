from typing import List,Optional,Tuple
import numpy as np
import pandas as pd
import torch,os,logging,hashlib
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
class TimeSeriesWindowDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            df: pd.DataFrame = None,
            feature_config_list = common.FEATURE_CONFIG_LIST,
            kline_interval_ms: int = None,
            feature_cols: List[str] = None,
            label_col: str = None,
            window: int = common.CANDLESTICK_NUM,
            stride: int = 1,
            is_live: bool = False,
            cache_path: Optional[str] = None,
            use_cache: bool = False,
            show_feature_distribution = False
    ):
        self.logger = logging.getLogger()
        self.is_live = is_live
        self.show_feature_distribution = show_feature_distribution
        self.stride = stride
        self.window = window
        self.kline_interval_ms = kline_interval_ms
        self.feature_cols_requested = feature_cols  # 记录请求的特征列表
        self.label_col = label_col
        self.time_col = 'open_time_ms_utc'
        self.factory = common.FeatureFactory(feature_config_list, self.kline_interval_ms)


        # 🌟 核心修改：增加特征过滤逻辑
        if feature_cols is not None:
            # 由于 self.factory.all_feature_list 已是一维列表，直接转 set 进行高效匹配
            available_features = set(self.factory.all_feature_list)
            # 过滤请求列表：只保留 Factory 确实能够生成的特征
            filtered_cols = [f for f in feature_cols if f in available_features]
            # 如果有特征被剔除，打印警告以便排查配置问题
            if len(filtered_cols) < len(feature_cols):
                missing = set(feature_cols) - available_features
                self.logger.warning(f"🚫 过滤掉 Factory 未定义的特征: {missing}")
            # 更新请求参数，确保后续 _prepare_data 和缓存校验使用的是过滤后的列表
            feature_cols = filtered_cols
            self.feature_cols_requested = feature_cols 

        # --- 1. Load from Cache ---
        if use_cache and cache_path and os.path.exists(cache_path):
            if self._load_from_cache(cache_path):
                self.logger.warning(f"🚀 参数匹配，成功从缓存加载: {cache_path}")
                return
            else:
                self.logger.info("🔄 参数已变更或缓存失效，将重新处理原始数据...")

        # --- 2. Process Data ---
        if df is None or feature_cols is None:
            raise ValueError("Data and feature_cols must be provided if cache is not found.")

        self.logger.info("⚙️ Cache not found or invalid. Processing raw data...")
        
        # A. Data Preparation & Audit
        df_work, clean_features = self._prepare_data(df, feature_cols, label_col)
        self.feature_names = clean_features
        self.feature_count = len(clean_features)

        # B. Window Generation
        X3d, time_windows = self._generate_windows(df_work)

        # C. Filter & Align (Continuity + Labels)
        X_filtered, y_filtered, final_indices = self._filter_and_align(
            df_work, X3d, time_windows, label_col
        )

        # D. Finalize (Normalization & Tensor Conversion)
        self._finalize_dataset(X_filtered, y_filtered, final_indices)

        # --- 3. Save to Cache ---
        if cache_path:
            self._save_to_cache(cache_path)

    def _save_to_cache(self, path: str):
        """保存处理后的数据及所有关键初始化参数"""
        data_to_save = {
            "X": self.X,
            "y": self.y,
            "indices": self.indices,
            "feature_names": self.feature_names,
            "feature_count": self.feature_count,
            # 关键参数快照
            "stride": self.stride,
            "window": self.window,
            "kline_interval_ms": self.kline_interval_ms,
            "label_col": self.label_col,
            "feature_cols_requested": self.feature_cols_requested,
            "symbol": common.symbol,
            "interval": common.interval,
        }
        torch.save(data_to_save, path)
        self.logger.info(f"💾 数据处理完成，缓存已更新至: {path}")

    def _load_from_cache(self, path: str) -> bool:
        """从磁盘加载并校验参数一致性"""
        try:
            checkpoint = torch.load(path, weights_only=False) 
            
            # --- 严格参数比对逻辑 ---
            # 1. 基础标量参数校验
            mismatch_reasons = []
            if self.stride != checkpoint.get("stride"):
                mismatch_reasons.append(f"stride ({checkpoint.get('stride')} -> {self.stride})")
            
            if self.window != checkpoint.get("window"):
                mismatch_reasons.append(f"window ({checkpoint.get('window')} -> {self.window})")
            
            if self.kline_interval_ms != checkpoint.get("kline_interval_ms"):
                mismatch_reasons.append(f"interval ({checkpoint.get('kline_interval_ms')} -> {self.kline_interval_ms})")
            
            if self.label_col != checkpoint.get("label_col"):
                mismatch_reasons.append(f"label_col ({checkpoint.get('label_col')} -> {self.label_col})")

            if common.symbol != checkpoint.get("symbol"):
                mismatch_reasons.append(f"symbol ({checkpoint.get('symbol')} -> {common.symbol})")

            if common.interval != checkpoint.get("interval"):
                mismatch_reasons.append(f"interval ({checkpoint.get('interval')} -> {common.interval})")

            # 2. 特征列表校验 (内容与顺序)
            cached_features = checkpoint.get("feature_cols_requested", [])
            if self.feature_cols_requested != cached_features:
                mismatch_reasons.append("feature_cols (列表内容或顺序已变更)")

            # 如果有任何不匹配，返回 False 触发重算
            if mismatch_reasons:
                self.logger.warning(f"⚠️ 缓存参数不匹配: {', '.join(mismatch_reasons)}")
                return False
                
            # 参数完全一致，执行赋值
            self.X = checkpoint["X"]
            self.y = checkpoint["y"]
            self.indices = checkpoint["indices"]
            self.feature_names = checkpoint["feature_names"]
            self.feature_count = checkpoint["feature_count"]
            return True

        except Exception as e:
            self.logger.error(f"❌ 缓存加载失败: {e}")
            return False

    # ----------------------------------------------------------------
    # --- Internal Pipeline Methods ---
    # ----------------------------------------------------------------

    def _prepare_data(self, df: pd.DataFrame, feature_cols: List[str], label_col: str):
        """
        分两阶段清洗数据：
        1. 剔除开头的冷启动 NaN (预期内)
        2. 剔除中间和末尾的异常 NaN (风险项)
        """
        # --- 基础列筛选 ---
        clean_features = [c for c in feature_cols if c not in DROP_FEATURES]
        cols = clean_features + ([label_col] if label_col and label_col in df.columns else []) + [self.time_col]
        
        df_work = df[cols].copy()
        if not self.is_live:
            df_work['orig_index'] = df.index

        total_rows = len(df_work)
        self.logger.debug(f"📊 [Data Clean] 开始清洗数据，原始总行数: {total_rows}")

        # --- 第一部分：删除起始的冷启动 NaN ---
        # 逻辑：找到第一行没有任何 NaN 的位置，删除它之前的所有行
        is_valid_row = df_work.notna().all(axis=1)
        if is_valid_row.any():
            first_valid_idx_label = is_valid_row.idxmax()  # 找到第一个全 True 的索引标签
            first_valid_loc = df_work.index.get_loc(first_valid_idx_label) # 获取该标签的物理位置
            
            if first_valid_loc > 0:
                df_work = df_work.iloc[first_valid_loc:].copy()
                self.logger.debug(f"✂️ [Step 1] 剔除头部冷启动: {first_valid_loc} 行 (由于指标预热等原因)")
            else:
                self.logger.debug(f"✅ [Step 1] 无头部冷启动数据。")
        else:
            raise RuntimeError("❌ [Data Clean] 错误：数据集中没有任何一行是完整的！请检查特征计算逻辑。")

        # --- 第二部分：删除中间或末尾的 NaN (异常空洞) ---
        before_gap_clean = len(df_work)
        df_work.dropna(inplace=True) # 此时剩下的都是中间或末尾的 NaN
        after_gap_clean = len(df_work)
        
        gap_count = before_gap_clean - after_gap_clean
        if gap_count > 0:
            # 中间空洞通常意味着数据源质量问题，建议用 WARNING 或 ERROR 级别
            self.logger.error(f"🚨 [Step 2] 发现并剔除中间/末尾异常空洞: {gap_count} 行！请检查数据源完整性。")
        else:
            self.logger.debug(f"✅ [Step 2] 未发现中间空洞。")

        # --- 最终重置索引 ---
        df_work.reset_index(drop=True, inplace=True)
        self.logger.info(f"✨ [Data Clean] 清洗完成: {total_rows} -> {len(df_work)} 行 (总计剔除 {total_rows - len(df_work)} 行)")

        return df_work, clean_features

    def _generate_windows(self, df_work: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Creates 3D windows using custom stride."""
        values = df_work[self.feature_names].to_numpy(dtype=np.float32)
        timestamps = df_work[self.time_col].to_numpy(dtype=np.int64)

        X3d = _as_strided_windows(values, self.window, self.stride)
        time_windows = _as_strided_windows(timestamps.reshape(-1, 1), self.window, self.stride).squeeze(-1)
        return X3d, time_windows

    def _filter_and_align(self, df_work, X3d, time_windows, label_col):
        """
        过滤逻辑增强：
        1. 全局检查：窗口内总缺失不能超过 5 个 K 线。
        2. 尾部检查：窗口内最后 10 个数据必须完全连续（零缺失）。
        3. 标签对齐：支持 stride，并区分过滤原因。
        """
        original_count = len(X3d)
        has_label = (label_col is not None) and (label_col in df_work.columns)
        interval = self.kline_interval_ms

        # --- A. 连续性掩码计算 ---
        
        # 1. 全局跨度检查 (例如允许 5 个 K 线以内的空洞)
        global_actual_span = time_windows[:, -1] - time_windows[:, 0]
        global_ideal_span = (self.window - 1) * interval
        # 这里你可以根据注释修改允许的阈值，目前设为 0 容忍
        mask_global = (global_actual_span <= global_ideal_span) 

        # 2. 严格的尾部检查 (最后 10 根 K 线)
        check_tail_count = 10
        tail_actual_span = time_windows[:, -1] - time_windows[:, -(check_tail_count + 1)]
        tail_ideal_span = check_tail_count * interval
        mask_tail = (tail_actual_span <= tail_ideal_span)

        # --- B. 标签有效性检查 ---
        if has_label:
            raw_labels = df_work[label_col].values[self.window - 1 :: self.stride]
            labels_all = raw_labels[:original_count]
            mask_label = (labels_all != common.Signal.INVALID)
        else:
            labels_all = np.zeros(original_count)
            mask_label = np.ones(original_count, dtype=bool)

        # --- C. 统计过滤原因 (分层统计) ---
        # 1. 因为全局跨度不合格被踢除的
        fail_global = np.sum(~mask_global)
        # 2. 全局合格但尾部不连续的
        fail_tail = np.sum(mask_global & ~mask_tail)
        # 3. 时间校验合格但标签无效的
        fail_label = np.sum(mask_global & mask_tail & ~mask_label)
        
        # 最终合并掩码
        final_mask = mask_global & mask_tail & mask_label
        final_count = np.sum(final_mask)

        # --- D. 索引对齐 ---
        final_indices = None
        if not self.is_live:
            raw_orig = df_work['orig_index'].values[self.window - 1 :: self.stride]
            final_indices = raw_orig[:original_count][final_mask]

        # --- E. 详细审计打印 ---
        self.logger.info(f"📊 数据过滤审计 [窗口长度: {self.window}]:")
        self.logger.info(f"   - 原始窗口总数: {original_count}")
        if fail_global > 0:
            self.logger.warning(f"   - ❌ 丢弃 (全局跨度超限): {fail_global}")
        if fail_tail > 0:
            self.logger.warning(f"   - ❌ 丢弃 (尾部10根不连续): {fail_tail}")
        if fail_label > 0:
            self.logger.warning(f"   - ❌ 丢弃 (标签无效/INVALID): {fail_label}")
        self.logger.info(f"   - ✅ 最终保留数量: {final_count} ({final_count/original_count:.2%})")
        
        return X3d[final_mask], labels_all[final_mask], final_indices

    def _finalize_dataset(self, X_filtered, y_filtered, final_indices):
        """Normalization and conversion to Tensors."""
        self.factory.normalize(X_filtered, self.feature_names)

        self.X = torch.from_numpy(X_filtered)
        self.y = torch.from_numpy(y_filtered).long()
        self.indices = final_indices
        # --- 自动打印统计信息供 Review ---
        if self.show_feature_distribution:
            self.print_feature_stats()
    # ----------------------------------------------------------------
    # --- 调试与数据复核方法 ---
    # ----------------------------------------------------------------

    def print_feature_stats(self):
        """
        打印所有特征的统计信息 (Mean, Std, Min, Max)，用于 review 归一化效果。
        参考之前的逻辑，主要查看窗口最后一帧 (Last Step) 的分布。
        """
        if self.X is None or self.X.shape[0] == 0:
            self.logger.warning("⚠️ 没有数据可以进行统计复核。")
            return

        # 提取最后一帧数据 [Batch, Feature]
        # X 的形状是 [N, Window, Feature]
        last_step_data = self.X[:, -1, :].numpy()
        
        self.logger.info("\n" + "="*90)
        self.logger.info(f"📊 数据处理复核 (特征统计 - 最后一帧) | 样本数: {len(last_step_data)}")
        self.logger.info("-" * 90)
        self.logger.info(f"{'Feature Name':<35} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10}")
        self.logger.info("-" * 90)

        for i, name in enumerate(self.feature_names):
            feat_slice = last_step_data[:, i]
            mean_v = np.mean(feat_slice)
            std_v  = np.std(feat_slice)
            min_v  = np.min(feat_slice)
            max_v  = np.max(feat_slice)
            
            # 使用简单的颜色或符号提醒异常值 (例如 std 远离 1 或 mean 远离 0)
            alert = " ⚠️" if abs(mean_v) > 0.1 or abs(std_v - 1.0) > 0.2 else ""
            
            self.logger.info(
                f"{name:<35} | {mean_v:10.4f} | {std_v:10.4f} | {min_v:10.4f} | {max_v}{alert}"
            )
        
        self.logger.info("="*90 + "\n")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]
def should_regenerate_cache(cache_path, data_path, feature_file, data_cfg):
    """
    通过校验文件修改时间和配置哈希，决定是否需要重新生成缓存。
    """
    if not os.path.exists(cache_path):
        return True # 缓存不存在，必须生成

    # 1. 检查文件修改时间 (mtime)
    # 如果原始数据 CSV 或特征定义文件 feature.py 在缓存之后被修改过，则失效
    cache_mtime = os.path.getmtime(cache_path)
    if os.path.getmtime(data_path) > cache_mtime:
        return True
    if os.path.getmtime(feature_file) > cache_mtime:
        return True

    # 2. 检查关键配置是否改变 (Hash 校验)
    # 将影响数据生成的参数转为字符串并计算哈希
    config_str = f"{data_cfg.window}_{data_cfg.feature_cols}_{data_cfg.label_col}"
    current_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    # 这里建议在生成缓存时，顺便存一个 .hash 文件
    hash_path = cache_path + ".hash"
    if not os.path.exists(hash_path):
        return True
    
    with open(hash_path, "r") as f:
        old_hash = f.read().strip()
    
    return current_hash != old_hash
    
# --- Global Utility ---
def _as_strided_windows(a2d: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    S = max(1, int(stride))
    N, F = a2d.shape
    M = (N - window) // S + 1
    if M <= 0: return np.empty((0, window, F))
    
    s0, s1 = a2d.strides
    view = np.lib.stride_tricks.as_strided(
        a2d, shape=(M, window, F), strides=(s0 * S, s0, s1), writeable=False
    )
    return view# copy will happen in _filter_and_align X3d[final_mask]