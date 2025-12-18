import os, sys, time, json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
# 引入自定义模块
from data_process.common import *
from model.model_factory import ModelFactory
from model.data_loader import TimeSeriesWindowDataset 
# -----------------------------------------------------------------------------
# Encapsulated Model Handler
# -----------------------------------------------------------------------------
class ModelHandler:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("trade")
        
        self.meta_path = os.path.join(TEMPORARY_DIR, "torch_model_train_meta.json")
        self.model_path = os.path.join(TEMPORARY_DIR, "torch_model_train_info.pt")
        
        self._load_metadata()
        self._load_model()

    def _load_metadata(self):
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Meta file not found: {self.meta_path}")
            
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
            
        self.feature_cols = self.meta["feature_cols"]
        self.window = int(self.meta["window"])
        self.classes = self.meta["classes"]
        self.label_col = self.meta.get("label_col", "label")
        # 仅用于日志
        self.model_type = self.meta["model_type"]
        self.model_version = self.meta.get("model_version", "unknown")

    def _load_model(self):
        self.logger.record(
            f"Loading model: {self.model_type.upper()} "
            f"(version={self.model_version}) on {self.device}..."
        )

        self.model, _ = ModelFactory.load_from_checkpoint(
            model_path=self.model_path,
            meta_path=self.meta_path,
            device=self.device,
        )
        self.model.eval()

    def predict(self, df, is_live=True, batch_size=1024, diff_thresh=None, min_thresh=0.3):
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
        self.logger.record(f"Starting inference pipeline (Mode={'Live' if is_live else 'Backtest'}, diff_thresh={diff_thresh})...")
        
        # 1. 准备数据：传入 is_live 标志以控制索引记录逻辑
        ds = TimeSeriesWindowDataset(
            df=df, 
            feature_cols=self.feature_cols, 
            label_col=self.label_col, 
            window=self.window,
            is_live=is_live
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
            for xb, _ in dl:
                xb = xb.to(self.device)
                if getattr(self.model, "supports_lengths", False):
                    logits = self.model(xb, lengths=None)
                else:
                    logits = self.model(xb)
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)

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
            final_pred[mask_long] = int(Signal.LONG)
            final_conf[mask_long] = net_score[mask_long]
            
            # 做空逻辑
            mask_short = (net_score < -diff_thresh) & (p_short > min_thresh)
            final_pred[mask_short] = int(Signal.SHORT)
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
        
        self.logger.record(f"Inference complete. Valid signals: {len(final_pred)}")
        return df_out, stats
    
    def scan_thresholds(self, df, thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3], batch_size=1024):
        """
        一次性扫描多个阈值，对比模型性能。
        
        原理：
        1. 只做一次推理 (Inference Once)，获取原始概率。
        2. 在内存中循环应用不同的 diff_thresh 逻辑。
        3. 计算每个阈值下的开单密度、精准率、召回率。
        """
        self.logger.record(f"🔍 Scanning thresholds: {thresholds}...")
        
        # 1. 准备数据 & 推理 (复用 Dataset 逻辑)
        ds = TimeSeriesWindowDataset(
            df=df, 
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
                # 兼容不同模型接口
                if getattr(self.model, "supports_lengths", False):
                    logits = self.model(xb, lengths=None)
                else:
                    logits = self.model(xb)
                
                # Softmax 转概率
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
        
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
            preds[net_score > th] = int(Signal.LONG)
            preds[net_score < -th] = int(Signal.SHORT)
            
            # 计算指标
            # output_dict=True 返回字典方便提取
            report = classification_report(y_true, preds, output_dict=True, zero_division=0)
            
            # 统计开单数量 (非震荡的单子)
            n_short = np.sum(preds == int(Signal.SHORT))
            n_long = np.sum(preds == int(Signal.LONG))
            total_signals = n_short + n_long
            coverage = total_signals / len(y_true)
            
            # 提取关键指标
            res_row = {
                "Threshold": th,
                "Signals": total_signals,   # 开单总数
                "Coverage": coverage,       # 覆盖率 (开单频率)
                
                # 精确率 (Precision): 做的单子里有多少是对的？
                "Prec_Short": report[str(int(Signal.SHORT))]['precision'],
                "Prec_Long": report[str(int(Signal.LONG))]['precision'],
                
                # 召回率 (Recall): 所有的机会抓住了多少？
                "Rec_Short": report[str(int(Signal.SHORT))]['recall'],
                "Rec_Long": report[str(int(Signal.LONG))]['recall'],
                
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

    def evaluate_performance(self, y_true, y_pred, labels=None):
        """
        生成符合要求的 Test Report 格式日志
        """
        self.logger.record("=== Test Report ===")
        
        # 1. 生成主要分类报告 (Precision, Recall, F1)
        # digits=4 确保保留4位小数 (例如 0.0956)
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        # logger 默认会处理换行，直接打印即可
        self.logger.record("\n" + report)

        # 2. 宏平均 F1 (单独打印)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        self.logger.record(f"Test macro-F1:{macro_f1}")
        
        # 3. 真实标签分布
        self.logger.record("\n=== True label proportion (Test set) ===")
        unique_labels, counts = np.unique(y_true, return_counts=True)
        total_samples = len(y_true)
        
        for label, count in zip(unique_labels, counts):
            proportion = count / total_samples
            self.logger.record(f"label {label}: {count} samples, {proportion:.4f} of total")

        # 返回 UI 需要的简单指标字典
        return {
            "accuracy": f"{accuracy_score(y_true, y_pred):.2%}",
            "precision": f"{precision_score(y_true, y_pred, average='weighted', zero_division=0):.2%}",
            "recall": f"{recall_score(y_true, y_pred, average='weighted', zero_division=0):.2%}",
            "f1_score": f"{f1_score(y_true, y_pred, average='weighted', zero_division=0):.2%}"
        }