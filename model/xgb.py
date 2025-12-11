import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np
import os
import json

class XGBoostAdapter:
    """
    XGBoost 适配器，使其行为像一个 PyTorch Module。
    支持输入 3D Tensor [Batch, Window, Feature]，内部自动展平为 2D。
    """
    def __init__(self, input_dim=None, n_classes=3, params=None):
        self.input_dim = input_dim  # 展平后的维度 (Window * Features)
        self.n_classes = n_classes
        
        # 默认 XGBoost 参数
        self.params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'objective': 'multi:softprob',
            'num_class': n_classes,
            'tree_method': 'hist',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'eval_metric': ['merror', 'mlogloss']
        }
        if params:
            self.params.update(params)
            
        self.model = xgb.XGBClassifier(**self.params)
        self._is_fitted = False

    def forward(self, x):
        """
        推理入口，兼容 PyTorch 调用方式: logits = model(x)
        输入 x: Tensor [Batch, Window, Feature]
        输出: Tensor [Batch, n_classes] (Raw Logits / Margins)
        """
        # 1. 数据转换 Tensor -> Numpy
        if isinstance(x, torch.Tensor):
            device = x.device
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
            device = 'cpu'

        # 2. 形状展平 [B, T, F] -> [B, T*F]
        B = x_np.shape[0]
        x_flat = x_np.reshape(B, -1)

        # 3. 推理
        if not self._is_fitted:
            # 如果还没训练，返回随机 Logits (防止初始化时报错)
            return torch.randn(B, self.n_classes).to(device)
        
        # 关键：output_margin=True 让 XGB 返回 Logits 而不是概率
        # 这样 model_loader 里的 torch.softmax(logits) 才能正常工作
        # 注意：XGBClassifier 的 predict_proba 不支持 output_margin，
        # 我们使用 booster 的底层 predict
        dmatrix = xgb.DMatrix(x_flat)
        logits_np = self.model.get_booster().predict(dmatrix, output_margin=True)
        
        return torch.from_numpy(logits_np).to(device)

    def fit_loader(self, train_loader, val_loader=None):
        """
        专门为 DataLoader 设计的训练函数，一次性提取所有数据进行训练
        """
        print(f"Converting DataLoader to Numpy for XGBoost (Input Dim: {self.input_dim})...")
        
        # 提取训练集
        X_train, y_train = self._loader_to_numpy(train_loader)
        eval_set = [(X_train, y_train)]
        
        # 提取验证集
        if val_loader:
            X_val, y_val = self._loader_to_numpy(val_loader)
            eval_set.append((X_val, y_val))
            
        print(f"Starting XGBoost Training (Train samples: {len(y_train)})...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=10  # 打印频率
        )
        self._is_fitted = True
        return self.model.evals_result()

    def _loader_to_numpy(self, loader):
        """辅助函数：将 DataLoader 中的所有 batch 拼接成大 Numpy 数组"""
        X_list, y_list = [], []
        for x, y in loader:
            # x: [B, T, F] -> flatten -> [B, T*F]
            x_flat = x.numpy().reshape(x.shape[0], -1)
            X_list.append(x_flat)
            y_list.append(y.numpy())
            
        return np.concatenate(X_list), np.concatenate(y_list)

    # --- 以下方法用于伪装成 PyTorch Module，防止 train.py 报错 ---
    
    def to(self, device):
        # XGBoost 自己管理 device，这里只需返回 self
        # 如果需要切换 GPU，可以在这里修改 self.model.device
        return self

    def eval(self):
        return self

    def train(self):
        return self
    
    def parameters(self):
        # 返回空迭代器，骗过 optimizer
        return iter([])

    def state_dict(self):
        # 导出模型为 JSON 字符串，放入字典
        if not self._is_fitted:
            return {}
        
        # 保存为临时文件再读回来，或者使用 save_model/load_model 的 buffer
        # 这里为了简化，我们只保存核心 booster
        # 注意：这里返回的格式必须能被 torch.save 序列化
        return {"xgb_json": self.model.get_booster().save_config()}

    def load_state_dict(self, state_dict):
        if "xgb_json" in state_dict:
             # 实际上加载需要 load_model(fname)，这里仅作占位
             # 真正的加载逻辑建议在 ModelHandler 里特殊处理，或者在这里通过临时文件加载
             pass

    def save_model(self, path):
        """原生保存方法"""
        self.model.save_model(path)

    def load_model(self, path):
        """原生加载方法"""
        self.model.load_model(path)
        self._is_fitted = True