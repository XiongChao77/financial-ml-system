import os, sys, time, json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
MODEL_PATH = os.path.join(current_work_dir, '..', 'model', "torch_model_train_info.pt")
META_PATH  = os.path.join(current_work_dir, '..', 'model', "torch_model_train_meta.json")
# 引入自定义模块
from data_process.common import *
from model.cnn import CNN1D
from model.lstm import LSTM1D
from model.data_loader import TimeSeriesWindowDataset 
# -----------------------------------------------------------------------------
# Encapsulated Model Handler
# -----------------------------------------------------------------------------
class ModelHandler:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("backtest")
        
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
        self.model_type = self.meta.get("model_type", "cnn") # 默认为 cnn

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.logger.info(f"Loading {self.model_type.upper()} model on {self.device}...")
        
        # 加载权重状态字典
        state = torch.load(self.model_path, map_location=self.device)
        channel = state.get("channel", len(self.feature_cols))
        n_classes = len(state.get("classes", self.classes))

        # 根据类型初始化模型架构
        if self.model_type == "lstm":
            hidden_size = self.meta.get("lstm_hidden", 64)
            num_layers = self.meta.get("lstm_layers", 2)
            bidirectional = self.meta.get("bidirectional", 2)
            self.model = LSTM1D(
                input_size=channel,
                hidden_size=hidden_size,
                num_layers=num_layers,
                n_classes=n_classes,
                p_drop=0.0,
                bidirectional = bidirectional
            )
        else:
            self.model = CNN1D(
                channel=channel, 
                n_classes=n_classes, 
                p_drop=0.0
            )
            
        self.model.load_state_dict(state["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, df, batch_size=1024):
        """
        输入原始 DataFrame，输出包含预测结果的 DataFrame 和 评估指标
        """
        self.logger.info("Starting inference pipeline...")
        
        # 1. 准备数据
        ds = TimeSeriesWindowDataset(
            df=df, 
            feature_cols=self.feature_cols, 
            label_col=self.label_col, 
            window=self.window
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # 2. 推理循环
        preds, confs = [], []
        with torch.no_grad():
            for xb, _ in dl:
                xb = xb.to(self.device)
                logits = self.model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds.append(probs.argmax(axis=1))
                confs.append(probs.max(axis=1))

        if not preds:
            self.logger.warning("No predictions generated!")
            return df, {}

        preds = np.concatenate(preds)
        confs = np.concatenate(confs)

        # 3. 结果对齐 (TimeSeriesWindowDataset 从 window-1 开始产生输出)
        valid_idx = df.index[self.window-1:]
        
        # 截断以匹配长度（防止数据尾部对其问题）
        min_len = min(len(valid_idx), len(preds))
        valid_idx = valid_idx[:min_len]
        preds = preds[:min_len]
        confs = confs[:min_len]

        # 4. 写入 DataFrame
        df_out = df.copy()
        df_out['pred'] = np.nan
        df_out['conf'] = np.nan
        df_out.loc[valid_idx, 'pred'] = preds
        df_out.loc[valid_idx, 'conf'] = confs

        # 5. 计算评估指标 (如果有标签)
        stats = {}
        if self.label_col in df_out.columns:
            # 只评估有效预测部分
            df_valid = df_out.loc[valid_idx]
            y_true = df_valid[self.label_col].values.astype(int)
            y_pred = df_valid['pred'].values.astype(int)
            stats = self.evaluate_performance(y_true, y_pred) # 使用你脚本中已有的函数
        
        # 删除 NaN 行（可选，取决于是否需要保留前面的数据用于绘图，回测通常需要保留）
        # df_out.dropna(subset=['pred'], inplace=True) 
        
        self.logger.info(f"Inference complete. Valid samples: {len(preds)}")
        return df_out, stats
    
    def evaluate_performance(self, y_true, y_pred, labels=None):
        """
        生成符合要求的 Test Report 格式日志
        """
        self.logger.info("\n=== Test Report ===")
        
        # 1. 生成主要分类报告 (Precision, Recall, F1)
        # digits=4 确保保留4位小数 (例如 0.0956)
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        # logger 默认会处理换行，直接打印即可
        self.logger.info("\n" + report)

        # 2. 宏平均 F1 (单独打印)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        self.logger.info(f"Test macro-F1:{macro_f1}")
        
        # 3. 真实标签分布
        self.logger.info("\n=== True label proportion (Test set) ===")
        unique_labels, counts = np.unique(y_true, return_counts=True)
        total_samples = len(y_true)
        
        for label, count in zip(unique_labels, counts):
            proportion = count / total_samples
            self.logger.info(f"label {label}: {count} samples, {proportion:.4f} of total")

        # 返回 UI 需要的简单指标字典
        return {
            "accuracy": f"{accuracy_score(y_true, y_pred):.2%}",
            "precision": f"{precision_score(y_true, y_pred, average='weighted', zero_division=0):.2%}",
            "recall": f"{recall_score(y_true, y_pred, average='weighted', zero_division=0):.2%}",
            "f1_score": f"{f1_score(y_true, y_pred, average='weighted', zero_division=0):.2%}"
        }