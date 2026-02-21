import os, sys, time, json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score,confusion_matrix
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
# 引入自定义模块
from data_process.common import *
from model.model_factory import ModelFactory
from model.data_loader import TimeSeriesWindowDataset
from model.models.fusion_wrapper import FusionWrapper
# -----------------------------------------------------------------------------
# Encapsulated Model Handler
# -----------------------------------------------------------------------------
# model_loader.py

class ModelHandler:
    def __init__(self,tarin_out_path , device, task_desc_path = None):
        self.device = device
        self.logger = logging.getLogger("trade")
        
        # 1. 读取 Task Index
        if task_desc_path is None:
            task_desc_path = os.path.join(tarin_out_path, "task_description.json")
            
        if not os.path.exists(task_desc_path):
            raise FileNotFoundError(f"Task Description not found: {task_desc_path}")
            
        with open(task_desc_path, "r", encoding="utf-8") as f:
            self.task_desc = json.load(f)
            
        self.task_type = self.task_desc.get("task_type", "single")
        self.base_dir = os.path.dirname(task_desc_path)
        
        # 2. 根据类型初始化
        self.logger.info(f"🚀 Loading Task: {self.task_type.upper()}")
        
        if self.task_type == "single":
            self._load_single_mode()
        elif self.task_type in ["trigger_direction", "long_short_ovr"]:
            self._load_pipeline_mode()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def _init_config_from_meta(self, meta):
        """
        从具体的 Meta 字典中提取 Dataset 配置。
        """
        self.feature_cols = meta["feature_cols"]
        self.window = int(meta["window"])
        # Pipeline 模式下，最终输出通常映射回 3 分类，这里暂时取 meta 中的定义
        # 如果子模型是二分类，wrapper 会处理成 3 分类
        self.classes = meta.get("classes", [0, 1]) 
        self.label_col = meta.get("label_col", "label")
        
        self.raw_config = meta.get("feature_group_list", [])
        self.feature_group_list = []
        for class_name, params in self.raw_config:
            if class_name in globals():
                cls = globals()[class_name] 
                self.feature_group_list.append(FeatureContainer(cls, **params))

    def _load_single_mode(self):
        files = self.task_desc["models"]["main"]
        meta_path = os.path.join(self.base_dir, files["meta"])
        model_path = os.path.join(self.base_dir, files["model"])
        
        # 1. 读取 Meta 并初始化配置
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._init_config_from_meta(meta)
        self.classes = meta["classes"] # Single 模式直接用 Meta 里的 classes

        # 2. 加载模型
        self.model, _ = ModelFactory.load_from_checkpoint(
            model_path=model_path,
            meta_path=meta_path,
            device=self.device
        )
        self.model.eval()

    def _load_pipeline_mode(self):
        sub_models_map = self.task_desc["models"]
        loaded_sub_models = {}
        
        #  关键：确定谁是“主配置”提供者
        # 通常 Trigger/Direction 模式下，Trigger 是第一步，我们用它的配置初始化 Dataset
        if "trigger" in sub_models_map:
            primary_key = "trigger"
        elif "long_ovr" in sub_models_map:
            primary_key = "long_ovr"
        else:
            primary_key = list(sub_models_map.keys())[0]

        # 1. 先加载主配置
        primary_files = sub_models_map[primary_key]
        primary_meta_path = os.path.join(self.base_dir, primary_files["meta"])
        with open(primary_meta_path, "r", encoding="utf-8") as f:
            primary_meta = json.load(f)
            
        self.logger.info(f"📋 Using configuration from primary sub-model: '{primary_key}'")
        self._init_config_from_meta(primary_meta)
        
        # 修正：Pipeline 模式对外永远是 3 分类 [Short, Neutral, Long]
        # 即使子模型 Meta 里写的是 [0, 1]，Loader 对外表现必须统一
        self.classes = [0, 1, 2]

        # 2. 循环加载所有子模型
        for name, files in sub_models_map.items():
            model_path = os.path.join(self.base_dir, files["model"])
            meta_path = os.path.join(self.base_dir, files["meta"])
            
            # 可选：检查子模型配置是否与主配置冲突
            # if name != primary_key:
            #     check_consistency(primary_meta, meta_path)

            self.logger.info(f"   🔄 Loading sub-model '{name}'...")
            model, _ = ModelFactory.load_from_checkpoint(
                model_path=model_path,
                meta_path=meta_path,
                device=self.device
            )
            model.eval()
            loaded_sub_models[name] = model
            
        # 3. 组装 Wrapper
        self.model = FusionWrapper(loaded_sub_models, mode=self.task_type)
        self.model.to(self.device)
        self.model.eval()
        

    def predict(self, df, kline_interval_ms, is_live=True, batch_size=2048, diff_thresh=None, min_thresh=0.3, stride =1,
                   cache_path = '', use_cache= False):
        """
        执行推理，并支持基于概率差值的策略增强。
        
        :param df: 输入的 DataFrame (包含原始特征)
        :param is_live: 是否为实盘模式。
                        - True (实盘): 优化内存，仅输出最后一根 K 线的信号。
                        - False (回测): 记录索引映射，确保信号与时间轴严格对齐。
        :param batch_size: 批处理大小
        :param diff_thresh: 概率差阈值 (P_long - P_short)。
        :param min_thresh: 最小概率门槛，确保胜率。
        :return: (df_out, stats) 
                 df_out 包含完整 K 线及 'pred', 'pred_prob', 'net_score' 等列
        """
        self.logger.info(f"Starting inference pipeline (Mode={'Live' if is_live else 'Backtest'}, diff_thresh={diff_thresh})...")
        
        # 1. 准备数据：传入 is_live 标志以控制索引记录逻辑
        ds = TimeSeriesWindowDataset(
            df=df, 
            kline_interval_ms = kline_interval_ms,
            feature_cols=self.feature_cols, 
            label_col=self.label_col, 
            window=self.window,
            is_live=is_live,
            stride= stride,
            cache_path = cache_path,
            use_cache = use_cache,
        )
        
        # 检查是否产生了有效窗口（可能因为数据太短或全部不连续而被丢弃）
        if len(ds) == 0:
            self.logger.warning("No valid windows generated after continuity check!")
            df_empty = df.copy()
            for c in ['pred', 'pred_prob', 'prob_short', 'prob_neutral', 'prob_long', 'net_score']:
                df_empty[c] = np.nan
            return df_empty, {}

        self.logger.info(f"Dataset created. Valid windows: {len(ds)}")
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # 2. 推理循环 (获取原始 Logits -> Probabilities)
        probs_list = []
        with torch.no_grad():
            for xb, _, _ in dl:
                xb = xb.to(self.device)
                _, fused_probs = self.model(xb, return_fused=True) 
                
                # 转换回 numpy 以便后续处理
                probs_list.append(fused_probs.cpu().numpy())

        # 拼接所有批次结果
        probs_all = np.concatenate(probs_list)
        p_short = probs_all[:, 0]   # 下跌概率
        p_neutral = probs_all[:, 1] # 震荡概率
        p_long = probs_all[:, 2]    # 上涨概率
        net_score = p_long - p_short # 净得分

        # 3. 生成最终信号逻辑
        if diff_thresh is not None:
            final_pred = np.full(len(probs_all), int(Signal.NEUTRAL))
            final_conf = np.zeros(len(probs_all))
            
            # 做多逻辑
            mask_long = (net_score > diff_thresh) & (p_long > min_thresh)
            final_pred[mask_long] = int(Signal.POSITIVE )
            final_conf[mask_long] = net_score[mask_long]
            
            # 做空逻辑
            mask_short = (net_score < -diff_thresh) & (p_short > min_thresh)
            final_pred[mask_short] = int(Signal.NEGATIVE)
            final_conf[mask_short] = -net_score[mask_short]
        else:
            final_pred = probs_all.argmax(axis=1)
            final_conf = probs_all.max(axis=1)

        # 4. 【核心修复】：精准对齐与回填
        # 创建副本并初始化新列为 NaN，确保不连续的“空洞”被保留以进行持仓管理
        df_out = df.copy()
        cols_to_init = ['pred', 'pred_prob', 'prob_short', 'prob_neutral', 'prob_long', 'net_score']
        for c in cols_to_init:
            df_out[c] = np.nan
        
        if not is_live:
            # === 回测模式：通过 ds.indices 将信号“钉”在正确的原始时间戳上 ===
            if ds.indices is not None:
                # 确保索引和预测值长度对齐
                valid_len = min(len(ds.indices), len(final_pred))
                active_indices = ds.indices[:valid_len]
                
                df_out.loc[active_indices, 'pred'] = final_pred[:valid_len]
                df_out.loc[active_indices, 'pred_prob'] = final_conf[:valid_len]
                df_out.loc[active_indices, 'prob_short'] = p_short[:valid_len]
                df_out.loc[active_indices, 'prob_neutral'] = p_neutral[:valid_len]
                df_out.loc[active_indices, 'prob_long'] = p_long[:valid_len]
                df_out.loc[active_indices, 'net_score'] = net_score[:valid_len]
        else:
            # === 实盘模式：仅回填最新一根 K 线的结果 ===
            if len(final_pred) > 0:
                last_idx = df.index[-1]
                df_out.at[last_idx, 'pred'] = final_pred[-1]
                df_out.at[last_idx, 'pred_prob'] = final_conf[-1]
                df_out.at[last_idx, 'net_score'] = net_score[-1]

        # 5. 计算评估指标 (仅在非实盘且包含标签列时执行)
        stats = {}
        if not is_live and self.label_col in df_out.columns:
            # 仅评估有预测值且有标签的部分
            df_valid = df_out.dropna(subset=['pred', self.label_col])
            if not df_valid.empty:
                y_true = df_valid[self.label_col].values.astype(int)
                y_pred = df_valid['pred'].values.astype(int)
                stats = self.evaluate_performance(y_true, y_pred)
        stats['feature_config'] = self.raw_config
        stats['feature_cols']   = self.feature_cols
        self.logger.info(f"Inference complete. Valid signals: {len(final_pred)}")
        return df_out, stats
    
    def predict_with_ds(self, ds, df, is_live=True, batch_size=2048, diff_thresh=None, min_thresh=0.3):
        self.logger.info(f"Starting inference pipeline (Mode={'Live' if is_live else 'Backtest'}, diff_thresh={diff_thresh})...")
        
        # 检查是否产生了有效窗口（可能因为数据太短或全部不连续而被丢弃）
        if len(ds) == 0:
            self.logger.warning("No valid windows generated after continuity check!")
            df_empty = df.copy()
            for c in ['pred', 'pred_prob', 'prob_short', 'prob_neutral', 'prob_long', 'net_score']:
                df_empty[c] = np.nan
            return df_empty, {}

        self.logger.info(f"Dataset created. Valid windows: {len(ds)}")
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # 2. 推理循环 (获取原始 Logits -> Probabilities)
        probs_list = []
        with torch.no_grad():
            for xb, _, _ in dl:
                xb = xb.to(self.device)
                _, fused_probs = self.model(xb, return_fused=True) 
                
                # 转换回 numpy 以便后续处理
                probs_list.append(fused_probs.cpu().numpy())

        # 拼接所有批次结果
        probs_all = np.concatenate(probs_list)
        p_short = probs_all[:, 0]   # 下跌概率
        p_neutral = probs_all[:, 1] # 震荡概率
        p_long = probs_all[:, 2]    # 上涨概率
        net_score = p_long - p_short # 净得分

        # 3. 生成最终信号逻辑
        if diff_thresh is not None:
            final_pred = np.full(len(probs_all), int(Signal.NEUTRAL))
            final_conf = np.zeros(len(probs_all))
            
            # 做多逻辑
            mask_long = (net_score > diff_thresh) & (p_long > min_thresh)
            final_pred[mask_long] = int(Signal.POSITIVE )
            final_conf[mask_long] = net_score[mask_long]
            
            # 做空逻辑
            mask_short = (net_score < -diff_thresh) & (p_short > min_thresh)
            final_pred[mask_short] = int(Signal.NEGATIVE)
            final_conf[mask_short] = -net_score[mask_short]
        else:
            final_pred = probs_all.argmax(axis=1)
            final_conf = probs_all.max(axis=1)

        # 4. 【核心修复】：精准对齐与回填
        # 创建副本并初始化新列为 NaN，确保不连续的“空洞”被保留以进行持仓管理
        df_out = df.copy()
        cols_to_init = ['pred', 'pred_prob', 'prob_short', 'prob_neutral', 'prob_long', 'net_score']
        for c in cols_to_init:
            df_out[c] = np.nan
        
        if not is_live:
            # === 回测模式：通过 ds.indices 将信号“钉”在正确的原始时间戳上 ===
            if ds.indices is not None:
                # 确保索引和预测值长度对齐
                valid_len = min(len(ds.indices), len(final_pred))
                active_indices = ds.indices[:valid_len]
                
                df_out.loc[active_indices, 'pred'] = final_pred[:valid_len]
                df_out.loc[active_indices, 'pred_prob'] = final_conf[:valid_len]
                df_out.loc[active_indices, 'prob_short'] = p_short[:valid_len]
                df_out.loc[active_indices, 'prob_neutral'] = p_neutral[:valid_len]
                df_out.loc[active_indices, 'prob_long'] = p_long[:valid_len]
                df_out.loc[active_indices, 'net_score'] = net_score[:valid_len]
        else:
            # === 实盘模式：仅回填最新一根 K 线的结果 ===
            if len(final_pred) > 0:
                last_idx = df.index[-1]
                df_out.at[last_idx, 'pred'] = final_pred[-1]
                df_out.at[last_idx, 'pred_prob'] = final_conf[-1]
                df_out.at[last_idx, 'net_score'] = net_score[-1]

        # 5. 计算评估指标 (仅在非实盘且包含标签列时执行)
        stats = {}
        if not is_live and self.label_col in df_out.columns:
            # 仅评估有预测值且有标签的部分
            df_valid = df_out.dropna(subset=['pred', self.label_col])
            if not df_valid.empty:
                y_true = df_valid[self.label_col].values.astype(int)
                y_pred = df_valid['pred'].values.astype(int)
                stats = self.evaluate_performance(y_true, y_pred)
        stats['feature_config'] = self.raw_config
        stats['feature_cols']   = self.feature_cols
        self.logger.info(f"Inference complete. Valid signals: {len(final_pred)}")
        return df_out, stats
    
    def scan_thresholds(self, df, kline_interval_ms, thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3], batch_size=1024):
        """
        一次性扫描多个阈值，对比模型性能。
        
        原理：
        1. 只做一次推理 (Inference Once)，获取原始概率。
        2. 在内存中循环应用不同的 diff_thresh 逻辑。
        3. 计算每个阈值下的开单密度、精准率、召回率。
        """
        self.logger.info(f"🔍 Scanning thresholds: {thresholds}...")
        
        # 1. 准备数据 & 推理 (复用 Dataset 逻辑)
        ds = TimeSeriesWindowDataset(
            df=df, 
            kline_interval_ms = kline_interval_ms,
            feature_config_list = self.feature_group_list,
            feature_cols=self.feature_cols, 
            label_col=self.label_col, 
            window=self.window,
            is_live = False,
        )
        # shuffle=False 保证顺序一致
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # 2. 获取原始概率 (Probabilities)
        probs_list = []
        self.model.eval()
        with torch.no_grad():
            for xb, _ in dl:
                xb = xb.to(self.device)
                #  统一调用融合接口，获取 [B, 3] 的概率分布
                _, fused_probs = self.model(xb, return_fused=True) 
                probs_list.append(fused_probs.cpu().numpy())
        
        if not probs_list:
            self.logger.warning("No predictions generated!")
            return pd.DataFrame()

        # 拼接所有批次: [N, 3]
        probs_all = np.concatenate(probs_list)
        
        # 3. 对齐真实标签 (y_true)
        # TimeSeriesWindowDataset 从 window-1 开始产生输出
        valid_idx = df.index[self.window-1:]
        
        # 截断以防止长度不匹配
        min_len = min(len(valid_idx), len(probs_all))
        valid_idx = valid_idx[:min_len]
        probs_all = probs_all[:min_len]
        
        # 检查是否有标签用于评估
        if self.label_col not in df.columns:
            self.logger.warning("No label column found in df, cannot evaluate performance.")
            return pd.DataFrame()
            
        y_true = df.loc[valid_idx, self.label_col].values.astype(int)

        # 4. 预计算净得分 (Net Score)
        # 假设 0:Short, 1:Neutral, 2:Long
        p_short = probs_all[:, 0]
        p_long = probs_all[:, 2]
        net_score = p_long - p_short # 范围 [-1, 1]

        # 5. 循环评估每个阈值
        results = []
        
        for th in thresholds:
            # 初始化预测为 1 (震荡)
            preds = np.full(len(y_true), int(Signal.NEUTRAL))
            
            # 应用阈值逻辑
            preds[net_score > th] = int(Signal.POSITIVE )
            preds[net_score < -th] = int(Signal.NEGATIVE)
            
            # 计算指标
            # output_dict=True 返回字典方便提取
            report = classification_report(y_true, preds, output_dict=True, zero_division=0)
            
            # 统计开单数量 (非震荡的单子)
            n_short = np.sum(preds == int(Signal.NEGATIVE))
            n_long = np.sum(preds == int(Signal.POSITIVE ))
            total_signals = n_short + n_long
            coverage = total_signals / len(y_true)
            
            # 提取关键指标
            res_row = {
                "Threshold": th,
                "Signals": total_signals,   # 开单总数
                "Coverage": coverage,       # 覆盖率 (开单频率)
                
                # 精确率 (Precision): 做的单子里有多少是对的？
                "Prec_Short": report[str(int(Signal.NEGATIVE))]['precision'],
                "Prec_Long": report[str(int(Signal.POSITIVE ))]['precision'],
                
                # 召回率 (Recall): 所有的机会抓住了多少？
                "Rec_Short": report[str(int(Signal.NEGATIVE))]['recall'],
                "Rec_Long": report[str(int(Signal.POSITIVE ))]['recall'],
                
                # 综合 F1 (宏平均)
                "Macro_F1": report['macro avg']['f1-score']
            }
            results.append(res_row)

        # 6. 生成汇总 DataFrame
        df_res = pd.DataFrame(results)
        
        # 打印 ASCII 表格
        print("\n" + "="*80)
        print("📊 Threshold Scan Report (寻找最佳 Diff Threshold)")
        print("="*80)
        # 格式化打印
        print(df_res.to_string(formatters={
            'Threshold': '{:.2f}'.format,
            'Coverage': '{:.2%}'.format,
            'Prec_Short': '{:.2%}'.format,
            'Prec_Long': '{:.2%}'.format,
            'Rec_Short': '{:.2%}'.format,
            'Rec_Long': '{:.2%}'.format,
            'Macro_F1': '{:.4f}'.format
        }))
        print("="*80 + "\n")
        
        return df_res

    def evaluate_performance(self, y_true, y_pred):
        """
        返回的 stats 保证 json.dumps 可直接序列化
        """
        # 确保是 numpy array（方便统一处理）
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        stats = {}

        # ===== 基础指标 =====
        stats["accuracy"] = float(accuracy_score(y_true, y_pred))
        stats["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        stats["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        stats["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        stats["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

        # ===== 分类报告（dict）=====
        stats["classification_report"] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        # sklearn 这里通常是 str key（'0','1','2' / 'accuracy' / 'macro avg'），但我们最后统一 json_safe

        # ===== 混淆矩阵 =====
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        stats["confusion_matrix"] = {
            "labels": [int(x) for x in labels],     # 强制 Python int
            "matrix": cm.tolist(),                  # list[list[int]]
        }

        # ===== 分布信息（注意：key 转 str/int）=====
        unique_t, cnt_t = np.unique(y_true, return_counts=True)
        unique_p, cnt_p = np.unique(y_pred, return_counts=True)

        stats["label_distribution_true"] = {int(k): int(v) for k, v in zip(unique_t, cnt_t)}
        stats["label_distribution_pred"] = {int(k): int(v) for k, v in zip(unique_p, cnt_p)}

        # ===== 信号指标（按你项目的 Signal 定义改 NEUTRAL/NEG/POS）=====
        NEUTRAL = int(Signal.NEUTRAL)
        NEG = int(Signal.NEGATIVE)
        POS = int(Signal.POSITIVE)

        mask_signal = (y_pred != NEUTRAL)
        n_total = int(len(y_pred))
        n_signal = int(mask_signal.sum())

        signal = {
            "total_samples": n_total,
            "signal_count": n_signal,
            "coverage": float(n_signal / n_total) if n_total > 0 else 0.0,
        }

        if n_signal > 0:
            y_true_sig = y_true[mask_signal]
            y_pred_sig = y_pred[mask_signal]
            signal["directional_accuracy"] = float(np.mean(y_true_sig == y_pred_sig))

            # long / short 单独统计（基于预测触发）
            for name, cls in [("short", NEG), ("long", POS)]:
                m = (y_pred == cls)
                cnt = int(m.sum())
                signal[f"{name}_count"] = cnt
                signal[f"{name}_win_rate"] = float(np.mean(y_true[m] == cls)) if cnt > 0 else None

        stats["signal"] = signal

        # ===== 最后一刀：强制 JSON-safe（保证不会再炸）=====
        stats = json_safe(stats)

        return stats
