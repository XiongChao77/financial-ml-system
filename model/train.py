#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List, Dict
from tqdm import tqdm
from collections import Counter

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

# 路径设置
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from data_process import common
from model.data_loader import TimeSeriesWindowDataset
from model.model_factory import ModelFactory

# ==============================================================================
# 1. 配置定义 (Configuration)
# ==============================================================================

@dataclass
class DataConfig:
    csv_path: str = common.train_data_path
    feature_cols: list = field(default_factory=list)
    label_col: str = "label"
    window: int = common.CANDLESTICK_NUM
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 128

@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 42
    save_dir: str = common.TEMPORARY_DIR

@dataclass
class LSTMConfig:
    model_type: str = "lstm"
    model_version: int = 4
    hidden_size: int = 64
    num_layers: int = 2
    bidirectional: bool = True
    lstm_dropout: float = 0.4
    head_dropout: float = 0.2
    p_drop: float = 0.3
    readout: str = ['last' , 'meanmax' , 'attn', 'mix'][1]
    head: str = ['linear' , 'mlp'][1]
    in_locked_p: float = 0.05               # V4 locked dropout on inputs
    out_locked_p: float = 0              # V4 locked dropout on LSTM outputs (before pooling)
    input_norm: bool = True                # V4 LayerNorm on input features
    input_proj_dim: int | None = None      # V4 optional projection before LSTM.一个可选的线性层，将原始特征维度（如 48）映射到一个新的维度 $D$ 后再送入.降维
    logit_clip: float | None = None        # V4 

@dataclass
class TransformerConfig:
    model_type: str = "transformer"
    model_version: int = 2
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.3
    attn_dropout: float = 0.1
    drop_path: float = 0.05
    max_len: Optional[int] = None
    use_alibi: bool = False
    pos_encoding: str = "none"
    cls_token: bool = False
    readout: str = "mix"
    head: str = "linear"
    ffn_type: str = "swiglu"

@dataclass
class ConvLSTMConfig:
    model_type: str = "conv_lstm"
    model_version: int = 1
    d_model: int = 96
    conv_layers: int = 3
    conv_kernel: int = 5
    conv_dropout: float = 0.10
    conv_dilations: str = ""
    lstm_hidden: int = 64
    lstm_layers: int = 2
    bidirectional: bool = True
    lstm_dropout: float = 0.2
    input_norm: bool = True
    in_locked_p: float = 0.05
    out_locked_p: float = 0.05
    head_dropout: float = 0.2
    readout: str = "mix"
    head: str = "linear"
    logit_clip: Optional[float] = None
    p_drop: Optional[float] = None

@dataclass
class XGBoostConfig:
    model_type: str = "xgboost"
    model_version: int = 1
    xgb_depth: int = 6
    xgb_estimators: int = 100
    learning_rate: float = 3e-4

@dataclass
class CNNConfig:
    model_type: str = "cnn"
    model_version: int = 1
    p_drop: float = 0.3
    tau: float = 16.0
    use_tpool: bool = False
# ==============================================================================
# 2. 工厂 (Factory)
# ==============================================================================

class ModelConfigFactory:
    @staticmethod
    def get_default_config(model_type: str):
        if model_type == "lstm":        return LSTMConfig()
        if model_type == "transformer": return TransformerConfig()
        if model_type == "conv_lstm":   return ConvLSTMConfig()
        if model_type == "xgboost":     return XGBoostConfig()
        if model_type == "cnn":         return CNNConfig()
        raise ValueError(f"Unknown model_type: {model_type}")

# ==============================================================================
# 3. 核心逻辑 (Core Logic)
# ==============================================================================

def run_training(logger:logging, data_cfg: DataConfig, train_cfg: TrainConfig, model_cfg):
    """
    接收配置对象，执行训练。
    """
    # 0. 初始化环境
    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Model: {model_cfg.model_type} version: {model_cfg.model_version}")

    # 1. 准备数据
    df = common.load_train_df()
    logger.info(f"Using TimeSeriesWindowDataset with window={data_cfg.window} Origin data len {len(df)}...")
    
    feature_cols = data_cfg.feature_cols if data_cfg.feature_cols else list(df.columns)
    logger.info(f"Features num:{len(feature_cols)},: {feature_cols}")

    full_ds = TimeSeriesWindowDataset(
        df=df, kline_interval_ms= common.load_interval_ms() , feature_cols=feature_cols, label_col=data_cfg.label_col, window=data_cfg.window
    )
    logger.info(f"📊 [Dataset Check] Final features used in training ({full_ds.feature_count}):")
    logger.info(f"{full_ds.feature_names}")
    # ========== 【新增】调用保存 Debug 数据 ==========
    # 保存目录设置在 exported_project_files/model/debug_data 下
    if True:
        debug_dir = os.path.join(common.PROJECT_DATA_DIR, "debug_data")
        full_ds.save_debug_data(debug_dir, save_file= False)
        # exit()

    if False:
        full_ds.inspect_final_data()
        exit()

    # 显存预加载优化
    logger.info(f"Pre-loading entire dataset to {device}...")
    # full_ds.X = full_ds.X.to(device) # 根据显存情况开启
    # full_ds.y = full_ds.y.to(device)
    logger.info("Data loaded to VRAM.")

    M = len(full_ds)
    logger.info(f"Total windows (M) = {M}, window = {data_cfg.window}")

    # 2. 切分数据
    tr_rng, va_rng, te_rng = chrono_split_by_window_ends(M, data_cfg.train_ratio, data_cfg.val_ratio)
    
    ds_tr = SeqDataset(full_ds.X[tr_rng[0]:tr_rng[1]].numpy(), full_ds.y[tr_rng[0]:tr_rng[1]].numpy())
    ds_va = SeqDataset(full_ds.X[va_rng[0]:va_rng[1]].numpy(), full_ds.y[va_rng[0]:va_rng[1]].numpy())
    ds_te = SeqDataset(full_ds.X[te_rng[0]:te_rng[1]].numpy(), full_ds.y[te_rng[0]:te_rng[1]].numpy())

    # 3. 计算权重
    y_tr_np = full_ds.y[tr_rng[0]:tr_rng[1]].numpy()
    classes = np.unique(y_tr_np)
    cw_balanced = compute_class_weight("balanced", classes=classes, y=y_tr_np)
    class_weights = torch.tensor(cw_balanced, dtype=torch.float32, device=device)
    logger.info(f"Class weights: {dict(zip(classes, cw_balanced))}")

    # 4. DataLoader
    dl_tr = DataLoader(ds_tr, batch_size=data_cfg.batch_size, shuffle=True, pin_memory=(device.type=="cuda"))
    dl_va = DataLoader(ds_va, batch_size=data_cfg.batch_size, shuffle=False, pin_memory=(device.type=="cuda"))
    dl_te = DataLoader(ds_te, batch_size=data_cfg.batch_size, shuffle=False, pin_memory=(device.type=="cuda"))

    # 5. 构建模型 (参数解包)
    logger.info(f"Initializing model: type={model_cfg.model_type}, version={model_cfg.model_version}")
    
    params = asdict(model_cfg)
    m_type = params.pop('model_type')
    m_ver = params.pop('model_version')
    
    # 默认值修正 logic
    if hasattr(model_cfg, 'max_len') and params['max_len'] is None:
        params['max_len'] = data_cfg.window

    # 特殊处理 XGBoost
    if m_type == 'xgboost':
        model = ModelFactory.build_for_training(
            model_type=m_type, model_version=m_ver, device=device,
            input_size=full_ds.feature_count, n_classes=len(classes),
            input_dim=data_cfg.window * full_ds.feature_count, 
            xgb_params=params
        )
    else:
        model = ModelFactory.build_for_training(
            model_type=m_type, model_version=m_ver, device=device,
            input_size=full_ds.feature_count, n_classes=len(classes),
            **params
        )

    # 6. 训练准备
    criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(device)
    # criterion = FocalLoss(alpha=None, gamma=2.0).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights) #中立裁判，不偏袒任何一类
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # 7. 训练循环
    best_val, best_state, wait = float("inf"), None, 0
    
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        tr_loss, tr_total = 0.0, 0
        
        for xb, yb in tqdm(dl_tr, desc=f"Epoch {epoch}/{train_cfg.epochs}", ncols=100):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            # 【修复1】恢复 set_to_none=True
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * xb.size(0)
            tr_total += xb.size(0)
            
        tr_loss /= max(1, tr_total)
        va_loss, yv_true, yv_pred = eval_epoch(model, dl_va, device, criterion)
        
        scheduler.step(va_loss)
        va_f1 = f1_score(yv_true, yv_pred, average="macro") if len(yv_true) else float("nan")
        
        logger.info(f"Epoch {epoch:03d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} | va_macroF1 {va_f1:.4f}")

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= train_cfg.patience:
                logger.info(f"Early stopping. epoch {epoch}")
                break

    if best_state: model.load_state_dict(best_state)

    # 8. 测试与保存
    _, yt_true, yt_pred = eval_epoch(model, dl_te, device, criterion)
    logger.info("\n=== Test Report ===")
    logger.info(classification_report(yt_true, yt_pred, digits=4))
    logger.info("Test macro-F1:{}".format(f1_score(yt_true, yt_pred, average="macro")))

    # 【修复2】恢复 Test Set Label Proportion 统计日志
    counts = Counter(yt_true)
    total = sum(counts.values())
    classes_sorted = sorted(counts.keys())
    true_pct = {c: counts[c] / total for c in classes_sorted}
    logger.info("\n=== True label proportion (Test set) ===")
    for c in classes_sorted:
        logger.info(f"label {c}: {counts[c]} samples, {true_pct[c]:.4f} of total")

    # 【修复3】恢复混淆矩阵保存
    cm = confusion_matrix(yt_true, yt_pred, labels=classes)
    pd.DataFrame(
        cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]
    ).to_csv(os.path.join(train_cfg.save_dir, "confmat_cnn.csv"), index=True)
    logger.info("Saved confusion matrix -> confmat_cnn.csv")

    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes.tolist(),
        "channel": full_ds.feature_count,
        "window": data_cfg.window,
        "feature_cols": full_ds.feature_names,
        "label_col": data_cfg.label_col,
    }, os.path.join(train_cfg.save_dir, "torch_model_train_info.pt"))

    # ===== 保存 meta（模型自描述）=====
    meta = model.export_meta(
        feature_cols=full_ds.feature_names,
        label_col=data_cfg.label_col,
        classes=classes.tolist(),
        window=data_cfg.window,
    )

    with open(os.path.join(common.TEMPORARY_DIR, "torch_model_train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Saved model -> torch_model_train_info.pt")
    logger.info("Saved meta  -> torch_model_train_meta.json")

# ==============================================================================
# 4. 辅助函数
# ==============================================================================

def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def chrono_split_by_window_ends(M, tr_r, va_r):
    n_tr = int(M * tr_r); n_va = int(M * va_r)
    return (0, n_tr), (n_tr, n_tr + n_va), (n_tr + n_va, M)

class SeqDataset(Dataset):
    def __init__(self, X, y): self.X=torch.from_numpy(X); self.y=torch.from_numpy(y)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer('alpha', alpha) 
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None: alpha_t = self.alpha[targets]; focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

import torch.nn.functional as F

@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    tl, yt, yp = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        tl += criterion(logits, yb).item() * xb.size(0)
        yp.append(logits.argmax(1).cpu().numpy())
        yt.append(yb.cpu().numpy())
    if not yt: return float("nan"), np.array([]), np.array([])
    return tl/len(loader.dataset), np.concatenate(yt), np.concatenate(yp)

def main(logger:logging.Logger):
    # 示例：直接在这里配置参数，取代了命令行参数
    
    # 1. 数据配置
    d_cfg = DataConfig()
    
    # 2. 训练配置
    t_cfg = TrainConfig()
    
    # 3. 模型配置 (在此处切换模型)
    # m_cfg = ModelConfigFactory.get_default_config("transformer")
    # m_cfg.d_model = 128
    
    # m_cfg = ModelConfigFactory.get_default_config("transformer")
    m_cfg = [LSTMConfig(), TransformerConfig(), ConvLSTMConfig(), CNNConfig(), XGBoostConfig()][0]
    m_cfg.model_version = 4

    logger.info(f"Training {m_cfg.model_type}...")
    run_training(logger,d_cfg, t_cfg, m_cfg)
# ==============================================================================
# 5. 调用入口 (Main Entry)
# ==============================================================================

if __name__ == "__main__":
    logger, _ = common.setup_session_logger(sub_folder='train', file_level = logging.DEBUG)
    main(logger)