import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
import tempfile
import os
import joblib

from model.models.model_base import BaseTimeSeriesModel

class XGBoostAdapter(BaseTimeSeriesModel):
    """
    XGBoost Adapter implementing the 'Dual-Head' (2+2) architecture.
    
    Structure:
      - Head A (Trigger): Binary Classifier (Neutral vs. Action)
      - Head B (Direction): Binary Classifier (Short vs. Long), trained only on Action samples.
    
    Compatibility:
      - Input: PyTorch Tensors [B, T, F] -> Flattened internally to [B, T*F]
      - Output: PyTorch Tensors (logits or probs)
    """
    MODEL_TYPE = "xgboost"
    MODEL_VERSION = 1

    def __init__(
        self,
        input_size: int,   # Feature count per step
        window_size: int,  # Time steps
        n_classes: int = 3,
        
        # XGBoost Hyperparameters
        xgb_depth: int = 6,
        xgb_estimators: int = 100,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        tree_method: str = "hist", # 'hist' (cpu) or 'gpu_hist' (gpu)
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        
        **kwargs
    ):
        super().__init__()
        
        self.input_size = input_size
        self.window_size = window_size
        self.flatten_dim = input_size * window_size
        self.n_classes = n_classes
        print(f"XGBoostAdapter device : {device} ")
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        print(f"XGBoostAdapter device_type : {self.device_type} ")

        # Shared Params
        self.params = {
            'max_depth': int(xgb_depth),
            'n_estimators': int(xgb_estimators),
            'learning_rate': float(learning_rate),
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_jobs': -1
        }
        # 🌟 兼容性处理：适配新旧版本
        if self.device_type == "cuda":
            # 尝试新版 2.0+ 的 device 参数写法
            self.params['tree_method'] = 'hist'
            self.params['device'] = 'cuda'
        else:
            self.params['tree_method'] = 'hist'
            self.params['device'] = 'cpu'
        
        # We hold two separate boosters
        # 1. Trigger Model: Predicts P(Action)
        self.clf_trigger = xgb.XGBClassifier(**self.params)
        
        # 2. Direction Model: Predicts P(Long | Action)
        self.clf_direction = xgb.XGBClassifier(**self.params)
        
        # State flag
        self.is_fitted = False

    def _preprocess(self, x: torch.Tensor) -> np.ndarray:
        """Flatten [B, T, F] -> [B, T*F]"""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        # Shape Check
        if x.ndim == 3:
            B, T, F = x.shape
            assert T == self.window_size and F == self.input_size, \
                f"Shape mismatch: expected ({self.window_size}, {self.input_size}), got ({T}, {F})"
            x = x.reshape(B, -1)
        return x

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor, x_val=None, y_val=None):
        """
        Special Training Logic for 2+2 Architecture.
        """
        X = self._preprocess(x_train)
        y = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
        
        eval_set_trig = []
        eval_set_dir = []
        
        if x_val is not None and y_val is not None:
            X_v = self._preprocess(x_val)
            y_v = y_val.cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val
            
            # Prep Validation Targets
            y_v_trig = (y_v != 1).astype(int)
            mask_v_act = (y_v != 1)
            
            eval_set_trig = [(X_v, y_v_trig)]
            
            if mask_v_act.any():
                # Neutral(1) -> Skip, Short(0)->0, Long(2)->1
                y_v_dir = np.where(y_v[mask_v_act] == 2, 1, 0)
                eval_set_dir = [(X_v[mask_v_act], y_v_dir)]

        # --- Task A: Trigger (All Data) ---
        # Label: Neutral(1) -> 0, Short(0)/Long(2) -> 1
        y_trig = (y != 1).astype(int)
        
        print(f"[XGBoost] Training Trigger Head (Samples: {len(X)})...")
        self.clf_trigger.fit(
            X, y_trig, 
            eval_set=eval_set_trig if eval_set_trig else None,
            verbose=False
        )

        # --- Task B: Direction (Action Data Only) ---
        # Mask: Keep only non-neutral
        mask_action = (y != 1)
        X_dir = X[mask_action]
        y_dir_raw = y[mask_action]
        
        if len(X_dir) > 0:
            # Label: Short(0) -> 0, Long(2) -> 1
            y_dir = np.where(y_dir_raw == 2, 1, 0)
            
            print(f"[XGBoost] Training Direction Head (Samples: {len(X_dir)})...")
            self.clf_direction.fit(
                X_dir, y_dir,
                eval_set=eval_set_dir if eval_set_dir else None,
                verbose=False
            )
        else:
            print("[XGBoost] Warning: No action samples found. Direction head not trained.")

        self.is_fitted = True

    def forward(self, x: torch.Tensor, return_fused=False):
        """
        修正后的前向传播：
        1. return_fused=True  -> 返回 [B] 预测标签, [B, 3] 融合概率
        2. return_fused=False -> 返回 [B, 2] Trigger Logits, [B, 2] Direction Logits (用于 Loss 计算)
        """
        if not self.is_fitted:
            device = x.device
            B = x.size(0)
            return torch.zeros(B, 2).to(device), torch.zeros(B, 2).to(device)

        X = self._preprocess(x)
        
        # 1. 获取两个 Booster 的原始概率
        # Trigger: [p_neutral, p_action] -> [B, 2]
        probs_trig_np = self.clf_trigger.predict_proba(X) 
        # Direction: [p_short, p_long] -> [B, 2]
        probs_dir_np = self.clf_direction.predict_proba(X) 
        
        p_trig = torch.from_numpy(probs_trig_np).float().to(x.device)
        p_dir = torch.from_numpy(probs_dir_np).float().to(x.device)
        
        # 2. 只有在需要 return_fused 时才进行 3 分类融合
        if return_fused:
            p_neutral = p_trig[:, 0]
            p_action  = p_trig[:, 1]
            p_short   = p_action * p_dir[:, 0]
            p_long    = p_action * p_dir[:, 1]
            
            fused_probs = torch.stack([p_short, p_neutral, p_long], dim=1) # [B, 3]
            fused_preds = torch.argmax(fused_probs, dim=1)
            return fused_preds, fused_probs
            
        # 3. 默认返回原始头的 Logits (转换为 log-prob 以兼容 CrossEntropy)
        # 这样 mtl_manager 就能拿到正确的 [B, 2] 维度进行计算
        logits_trig = torch.log(p_trig + 1e-9)
        logits_dir = torch.log(p_dir + 1e-9)
        
        return logits_trig, logits_dir

    # ============================================================
    # Persistence (Since XGBoost isn't a torch module)
    # ============================================================
    
    def state_dict(self):
        """Custom serialization"""
        # Save boosters to temporary files and read bytes
        return {
            "trigger_booster": self.clf_trigger,
            "direction_booster": self.clf_direction,
            "params": self.params,
            "is_fitted": self.is_fitted
        }

    def load_state_dict(self, state_dict):
        self.clf_trigger = state_dict["trigger_booster"]
        self.clf_direction = state_dict["direction_booster"]
        self.params = state_dict["params"]
        self.is_fitted = state_dict["is_fitted"]

    def export_meta(self, **extra) -> dict:
        return {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "input_size": self.input_size,
            "window_size": self.window_size,
            "xgb_params": self.params,
            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        # Re-instantiate
        model = cls(
            input_size=meta["input_size"],
            window_size=meta["window_size"],
            xgb_depth=meta["xgb_params"]["max_depth"],
            xgb_estimators=meta["xgb_params"]["n_estimators"],
            learning_rate=meta["xgb_params"]["learning_rate"],
            device=device
        )
        model.load_state_dict(state["state_dict"])
        return model