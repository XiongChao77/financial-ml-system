from typing import List,Optional,Tuple
import numpy as np
import pandas as pd
import torch,os,logging,hashlib
from torch.utils.data import Dataset
from data_process import common

# number_of_trades and volume are highly correlated; quote_asset_volume and volume are also highly correlated.
#taker_buy_quote_volume--taker_buy_base_volume,
DROP_FEATURES =['threshold_long', 'stop_threshold_long','threshold_short', 'stop_threshold_short', 'label', 'trend_strength', 'open_time_ms_utc', 'open_time_date_utc',
                 'close_time_ms_utc', 'ignore' ]
LOW_CORRELATION_FEATURES = ['number_of_trades','quote_asset_volume', 'taker_buy_quote_volume']

# ====================================================================
# --- 2. TimeSeriesWindowDataset CLASS ---
# ====================================================================
class TimeSeriesWindowDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            kline_interval_ms:int,
            feature_cols,
            label_col: str,
            window: int,
            stride: int = 1,
            is_live: bool = False,
            cache_path: Optional[str] = None,
            use_cache: bool = False,
            show_feature_distribution = True
    ):
        self.logger = logging.getLogger("dataset")
        self.is_live = is_live
        self.show_feature_distribution = show_feature_distribution
        self.stride = stride
        self.window = window
        self.kline_interval_ms = kline_interval_ms
        self.feature_cols = feature_cols  # Keep a copy of requested feature list
        self.label_col = label_col
        self.time_col = 'open_time_ms_utc'
        self.factory = common.FeatureFactory(self.kline_interval_ms)

        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {list(missing)}")
        if 'atr_14' in feature_cols:
            raise RuntimeError("atr_14 is used for strategy now , can't be normalize")
        if self.is_live == True:
            self.stride = 1
        self.logger.debug(f"Features num:{len(feature_cols)},: {feature_cols}")

        # --- 1. Load from Cache ---
        if use_cache and cache_path and os.path.exists(cache_path):
            if self._load_from_cache(cache_path):
                self.logger.warning(f"🚀 Parameters match, loaded from cache: {cache_path}")
                return
            else:
                self.logger.info("🔄 Parameters changed or cache invalid, re-processing raw data...")

        # --- 2. Process Data ---
        if df is None or feature_cols is None:
            raise ValueError("Data and feature_cols must be provided if cache is not found.")

        self.logger.info("⚙️ Cache not found or invalid. Processing raw data...")
        
        # A. Data Preparation & Audit
        df_work, clean_features = self._prepare_data(df, feature_cols, label_col)
        self.feature_names = clean_features
        self.feature_count = len(clean_features)
        cols_set = set(self.feature_cols)
        unused = [f for f in self.factory.all_feature_list if f not in cols_set]    #keep order
        self.logger.debug(f"feature unused: {unused}")
        # B. Window Generation
        X3d, time_windows = self._generate_windows(df_work)

        # C. Filter & Align (Continuity + Labels)
        X_filtered, y_filtered, final_indices = self._filter_and_align(
            df_work, X3d, time_windows, label_col
        )

        # D. Finalize (Normalization & Tensor Conversion)
        self._finalize_dataset(X_filtered, y_filtered, final_indices)

        # --- 3. Save to Cache ---
        if use_cache and cache_path:
            self._save_to_cache(cache_path)

    def _save_to_cache(self, path: str):
        """Save processed data and all key initialization parameters."""
        data_to_save = {
            "X": self.X,
            "y": self.y,
            "returns": self.returns,
            "indices": self.indices,
            "feature_names": self.feature_names,
            "feature_count": self.feature_count,
            # Key parameter snapshot
            "stride": self.stride,
            "window": self.window,
            "kline_interval_ms": self.kline_interval_ms,
            "label_col": self.label_col,
            "feature_cols": self.feature_cols,
            "symbol": common.BaseDefine.symbol,
            "interval": common.BaseDefine.interval,
        }
        torch.save(data_to_save, path)
        self.logger.info(f"💾 Data processing done, cache updated: {path}")

    def _load_from_cache(self, path: str) -> bool:
        """Load from disk and validate parameter consistency."""
        try:
            checkpoint = torch.load(path, weights_only=False) 
            
            # --- Strict parameter comparison ---
            # 1. Basic scalar parameter validation
            mismatch_reasons = []
            self.returns = checkpoint.get("returns") 
            
            # Backward compatibility: older caches may not contain returns
            if self.returns is None:
                self.logger.error("🚨 Cache missing 'returns'! Please delete cache file and regenerate.")
                return False

            if self.stride != checkpoint.get("stride"):
                mismatch_reasons.append(f"stride ({checkpoint.get('stride')} -> {self.stride})")
            
            if self.window != checkpoint.get("window"):
                mismatch_reasons.append(f"window ({checkpoint.get('window')} -> {self.window})")
            
            if self.kline_interval_ms != checkpoint.get("kline_interval_ms"):
                mismatch_reasons.append(f"interval ({checkpoint.get('kline_interval_ms')} -> {self.kline_interval_ms})")
            
            if self.label_col != checkpoint.get("label_col"):
                mismatch_reasons.append(f"label_col ({checkpoint.get('label_col')} -> {self.label_col})")

            if common.BaseDefine.symbol != checkpoint.get("symbol"):
                mismatch_reasons.append(f"symbol ({checkpoint.get('symbol')} -> {common.BaseDefine.symbol})")

            if common.BaseDefine.interval != checkpoint.get("interval"):
                mismatch_reasons.append(f"interval ({checkpoint.get('interval')} -> {common.BaseDefine.interval})")

            # 2. Feature list validation (content and order)
            cached_features = checkpoint.get("feature_cols", [])
            if self.feature_cols != cached_features:
                mismatch_reasons.append("feature_cols (list content or order changed)")

            # If anything mismatches, return False to trigger recomputation
            if mismatch_reasons:
                self.logger.warning(f"⚠️ Cache parameter mismatch: {', '.join(mismatch_reasons)}")
                return False
                
            # All parameters match; assign cached tensors
            self.X = checkpoint["X"]
            self.y = checkpoint["y"]
            self.indices = checkpoint["indices"]
            self.feature_names = checkpoint["feature_names"]
            self.feature_count = checkpoint["feature_count"]
            return True

        except Exception as e:
            self.logger.error(f"❌ Cache load failed: {e}")
            return False

    # ----------------------------------------------------------------
    # --- Internal Pipeline Methods ---
    # ----------------------------------------------------------------

    def _prepare_data(self, df: pd.DataFrame, feature_cols: List[str], label_col: str):
        """
        Clean data in two stages:
        1. Remove initial cold-start NaNs (expected)
        2. Remove abnormal NaNs in the middle or tail (risky)
        """
        # --- Basic column selection ---
        clean_features = [c for c in feature_cols if c not in DROP_FEATURES]
        cols = clean_features + ([label_col] if label_col and label_col in df.columns else []) + [self.time_col]
        
        # Core change: always include trend_strength in extracted columns (but it's not part of clean_features)
        if 'trend_strength' in df.columns:
            if 'trend_strength' not in cols:
                cols.append('trend_strength')

        df_work = df[cols].copy()
        if not self.is_live:
            df_work['orig_index'] = df.index

        total_rows = len(df_work)
        self.logger.debug(f"📊 [Data Clean] Start cleaning, total rows: {total_rows}")

        # --- Part 1: remove initial cold-start NaNs ---
        # Logic: find the first row with no NaNs, drop all rows before it
        is_valid_row = df_work.notna().all(axis=1)
        if is_valid_row.any():
            first_valid_idx_label = is_valid_row.idxmax()  # First index label where all columns are valid
            first_valid_loc = df_work.index.get_loc(first_valid_idx_label)  # Physical position for that label
            
            if first_valid_loc > 0:
                df_work = df_work.iloc[first_valid_loc:].copy()
                self.logger.debug(f"✂️ [Step 1] Dropped head cold-start: {first_valid_loc} rows (indicator warmup, etc.)")
            else:
                self.logger.debug("✅ [Step 1] No head cold-start rows.")
        else:
            raise RuntimeError("❌ [Data Clean] Error: no complete rows found! Please check feature computation logic.")

        # --- Part 2: remove NaNs in the middle or tail (abnormal gaps) ---
        before_gap_clean = len(df_work)
        df_work.dropna(inplace=True)  # Remaining NaNs are in the middle or tail
        after_gap_clean = len(df_work)
        
        gap_count = before_gap_clean - after_gap_clean
        if gap_count > 0:
            # Middle gaps usually indicate data quality issues; keep as WARNING/ERROR
            self.logger.error(f"🚨 [Step 2] Found and removed abnormal middle/tail gaps: {gap_count} rows! Please check data completeness.")
        else:
            self.logger.debug("✅ [Step 2] No middle gaps found.")

        # --- Final index reset ---
        df_work.reset_index(drop=True, inplace=True)
        self.logger.info(f"✨ [Data Clean] Done: {total_rows} -> {len(df_work)} rows (dropped {total_rows - len(df_work)} rows)")

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
        Enhanced filtering logic:
        1. Global check: total missing span within the window must be within tolerance.
        2. Tail check: last N points in the window must be fully continuous (no gaps).
        3. Label alignment: support stride and track filtering reasons.
        """
        original_count = len(X3d)
        has_label = (label_col is not None) and (label_col in df_work.columns)
        interval = self.kline_interval_ms

        # --- A. Continuity mask computation ---
        
        # 1. Global span check (e.g., allow limited gaps)
        global_actual_span = time_windows[:, -1] - time_windows[:, 0]
        global_ideal_span = (self.window - 1) * interval
        # You may tune tolerance here; currently 0 tolerance
        mask_global = (global_actual_span <= global_ideal_span) 

        # 2. Strict tail check (last N bars)
        check_tail_count = 2
        tail_actual_span = time_windows[:, -1] - time_windows[:, -(check_tail_count + 1)]
        tail_ideal_span = check_tail_count * interval
        mask_tail = (tail_actual_span <= tail_ideal_span)

        # --- B. Label validity check ---
        if has_label:
            raw_labels = df_work[label_col].values[self.window - 1 :: self.stride]
            labels_all = raw_labels[:original_count]
            mask_label = (labels_all != common.Signal.INVALID)
        else:
            labels_all = np.zeros(original_count)
            mask_label = np.ones(original_count, dtype=bool)

        # --- C. Filter reason stats (hierarchical) ---
        # 1. Failed due to global span
        fail_global = np.sum(~mask_global)
        # 2. Global ok but tail not continuous
        fail_tail = np.sum(mask_global & ~mask_tail)
        # 3. Time ok but label invalid
        fail_label = np.sum(mask_global & mask_tail & ~mask_label)
        
        # Final combined mask
        final_mask = mask_global & mask_tail & mask_label
        final_count = np.sum(final_mask)

        # --- D. Index alignment ---
        final_indices = None
        if not self.is_live:
            raw_orig = df_work['orig_index'].values[self.window - 1 :: self.stride]
            final_indices = raw_orig[:original_count][final_mask]

        # --- E. Detailed audit logs ---
        self.logger.info(f"📊 Data filter audit [window: {self.window}]:")
        self.logger.info(f"   - Original window count: {original_count}")
        if fail_global > 0:
            self.logger.warning(f"   - ❌ Dropped (global span exceeded): {fail_global}")
        if fail_tail > 0:
            self.logger.warning(f"   - ❌ Dropped (tail not continuous): {fail_tail}")
        if fail_label > 0:
            self.logger.warning(f"   - ❌ Dropped (label invalid/INVALID): {fail_label}")
        self.logger.info(f"   - ✅ Final kept: {final_count} ({final_count/original_count:.2%})")
    
        if 'trend_strength' in df_work.columns:
            # Use the same slicing as labels: start at window-1 and sample by stride
            aligned_returns = df_work['trend_strength'].values[self.window - 1 :: self.stride]
            df_work.drop(columns=['trend_strength'], inplace=True)
            # Truncate to match window count
            self.returns = aligned_returns[:original_count][final_mask]
        else:
            self.logger.warning("⚠️ Missing trend_strength in df_work; returns set to 0")
            self.returns = np.zeros(final_count)

        return X3d[final_mask], labels_all[final_mask], final_indices

    def _finalize_dataset(self, X_filtered, y_filtered, final_indices):
        """Normalization and conversion to Tensors."""
        self.factory.normalize(X_filtered, self.feature_names)

        self.X = torch.from_numpy(X_filtered)
        self.y = torch.from_numpy(y_filtered).long()
        self.returns = torch.from_numpy(self.returns).float()
        self.indices = final_indices
        # --- Auto-print stats for review ---
        if self.show_feature_distribution:
            self.print_feature_stats()
    # ----------------------------------------------------------------
    # --- Debugging and data review helpers ---
    # ----------------------------------------------------------------

    def print_feature_stats(self):
        """
        Print statistics (Mean, Std, Min, Max) for all features to review normalization quality.
        Following the previous logic, we mainly inspect the distribution of the last step in each window.
        """
        if self.X is None or self.X.shape[0] == 0:
            self.logger.warning("⚠️ No data available for stats review.")
            return

        num_features_in_data = self.X.shape[2]
        num_feature_names = len(self.feature_names)

        if num_features_in_data != num_feature_names:
            msg = (f"❌ Dimension mismatch! Feature columns in data ({num_features_in_data}) "
                   f"do not match number of feature names ({num_feature_names}).")
            self.logger.critical(msg)
            # If mismatched, some unexpected feature slipped in; stop immediately
            raise RuntimeError(msg)

        # Extract last-step data [Batch, Feature]
        # X shape: [N, Window, Feature]
        last_step_data = self.X[:, -1, :].numpy()
        
        self.logger.info("\n" + "="*90)
        self.logger.info(f"📊 Data processing review (feature stats - last step) | samples: {len(last_step_data)}")
        self.logger.info("-" * 90)
        self.logger.info(f"{'Feature Name':<35} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10}")
        self.logger.info("-" * 90)

        for i, name in enumerate(self.feature_names):
            feat_slice = last_step_data[:, i]
            mean_v = np.mean(feat_slice)
            std_v  = np.std(feat_slice)
            min_v  = np.min(feat_slice)
            max_v  = np.max(feat_slice)
            
            # Simple marker for suspicious values (e.g., std far from 1 or mean far from 0)
            alert = " ⚠️" if abs(mean_v) > 0.1 or abs(std_v - 1.0) > 0.2 else ""
            
            self.logger.info(
                f"{name:<35} | {mean_v:10.4f} | {std_v:10.4f} | {min_v:10.4f} | {max_v}{alert}"
            )
        
        self.logger.info("="*90 + "\n")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.returns[i]
def should_regenerate_cache(cache_path, data_path, feature_file, data_cfg):
    """
    Decide whether to regenerate cache by checking file modification times and a config hash.
    """
    if not os.path.exists(cache_path):
        return True  # Cache does not exist; must generate

    # 1. Check modification time (mtime)
    # If raw CSV or feature definition file was modified after cache, invalidate
    cache_mtime = os.path.getmtime(cache_path)
    if os.path.getmtime(data_path) > cache_mtime:
        return True
    if os.path.getmtime(feature_file) > cache_mtime:
        return True

    # 2. Check if critical config changed (hash check)
    # Convert config affecting cache generation into a string and hash it
    config_str = f"{data_cfg.window}_{data_cfg.feature_cols}_{data_cfg.label_col}"
    current_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    # Suggestion: write a .hash file together when creating cache
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