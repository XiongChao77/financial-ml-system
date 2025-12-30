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
import torch.nn.functional as F
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
    epochs: int = 30
    lr: float = 8e-5    #3e-4
    weight_decay: float = 2e-3
    patience: int = 5
    seed: int = 42
    save_dir: str = common.TEMPORARY_DIR
    stride = 4
    use_cache = True
    lambda_trig: float = 1.0  # Trigger 任务权重
    lambda_dir: float = 0.1   # Direction 任务权重 (设为 0 即可实现第一阶段只练 Trigger)

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
    model_version: int = 3
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.3
    attn_dropout: float = 0.1
    drop_path: float = 0
    in_locked_p: float = 0
    max_len: Optional[int] = None
    use_alibi: bool = True
    pos_encoding: str = "none"
    cls_token: bool = False
    readout: str = "cls" #"cls" | "meanmax" | "attn" | "mix"
    head: str = "linear"
    ffn_type: str = "swiglu"

@dataclass
class ConvLSTMConfig:
    model_type: str = "conv_lstm"
    model_version: int = 1
    d_model: int = 128
    hidden_size = 64
    conv_layers: int = 5
    conv_kernel: int = 5
    conv_dropout: float = 0.10
    conv_dilations: str = ""
    bidirectional: bool = True
    lstm_dropout: float = 0.2
    input_norm: bool = True
    in_locked_p: float = 0.05
    out_locked_p: float = 0.05
    head_dropout: float = 0.1
    readout: str = "mix"    # 'last'|'meanmax'|'attn'|'mix'
    head: str = "linear"    # 'linear'|'mlp'
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

def run_training(feature_config_list, logger:logging, data_cfg: DataConfig, train_cfg: TrainConfig, model_cfg):
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

    full_ds = TimeSeriesWindowDataset(feature_config_list = feature_config_list,
        df=df, kline_interval_ms= common.load_interval_ms() , feature_cols=feature_cols, label_col=data_cfg.label_col, window=data_cfg.window,
        cache_path=os.path.join(common.TEMPORARY_DIR,"train_cache.pt"), stride =train_cfg.stride, use_cache = train_cfg.use_cache,
    )
    logger.info(f"📊 [Dataset Check] Final features used in training ({full_ds.feature_count}):")
    logger.warning(f"{full_ds.feature_names}")

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # 7. 调用封装好的训练引擎
    logger.info("🚀 Starting training engine...")
    results = train_engine(
        model=model,
        dl_tr=dl_tr,
        dl_va=dl_va,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        train_cfg=train_cfg,
    )
    # 8. 评估与保存 (调用新封装的评估函数)
    final_metrics = evaluate_and_save_results(
        results=results,
        model=model,
        dl_te=dl_te,
        device=device,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        full_ds=full_ds,
        feature_config_list = feature_config_list,
        classes=classes,
        logger=logger
    )
    # diagnose_confidence(results, model=model,dl_te=dl_te, device=device,logger=logger,save_dir = common.TEMPORARY_DIR)
    # find_best_threshold(results, model=model,dl_te=dl_te, device=device,logger=logger)
    return final_metrics
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

def train_engine(
    model: nn.Module,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    logger: logging.Logger,
    train_cfg: TrainConfig # 传入 TrainConfig 以获取 lambda_trig 和 lambda_dir
):
    """
    核心训练引擎：内部自动计算 2+2 任务的类别权重，支持分阶段调优。
    """
    # --- 1. 动态计算类别权重 ---
    # 从 DataLoader 的 Dataset 中提取原始标签
    # y_raw 的定义为: 0: SHORT, 1: NEUTRAL, 2: LONG
    y_raw = dl_tr.dataset.y.numpy() if torch.is_tensor(dl_tr.dataset.y) else dl_tr.dataset.y

    # A. 为 Trigger 任务计算权重 (0: Neutral, 1: Action)
    y_trig = (y_raw != 1).astype(int) 
    cw_trig = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_trig)
    weights_trig = torch.tensor(cw_trig, dtype=torch.float32, device=device)
    weights_trig[1] = weights_trig[1] * 1.1 # 调整对 Action 类的重视，目标越小，多数类的Recall越高

    # B. 为 Direction 任务计算权重 (0: Short, 1: Long)
    mask_dir = (y_raw != 1)
    y_dir = np.where(y_raw[mask_dir] == 2, 1, 0) # 将 2(Long) 映射为 1, 0(Short) 映射为 0
    cw_dir = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_dir)
    weights_dir = torch.tensor(cw_dir, dtype=torch.float32, device=device)

    logger.info(f"⚖️ [Internal Weight] Trigger (Neut vs Act): {cw_trig}")
    logger.info(f"⚖️ [Internal Weight] Direction (Short vs Long): {cw_dir}")

    # --- 2. 初始化带权重的损失函数 ---
    criterion_trig = nn.CrossEntropyLoss(weight=weights_trig) 
    criterion_dir = nn.CrossEntropyLoss(weight=weights_dir)
    # 初始化监控指标
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_state_loss = None 
    best_state_f1 = None
    wait = 0

    for epoch in range(1, train_cfg.epochs + 1):
        # --- 训练阶段 ---
        model.train()
        tr_loss, tr_total = 0.0, 0
        
        for xb, yb in tqdm(dl_tr, desc=f"Epoch {epoch}/{train_cfg.epochs}", ncols=100, leave=False):
            xb, yb = xb.to(device), yb.to(device)
            
            # 1. 动态标签映射
            # Task A (Trigger): yb=1(震荡) -> 0, yb=0/2(有信号) -> 1
            target_trig = (yb != 1).long()
            # Task B (Direction): yb=0(看空) -> 0, yb=2(看多) -> 1
            target_dir = torch.where(yb == 2, 1, 0).long()
            action_mask = (yb != 1) # 只有非震荡样本才计算方向损失

            # 2. 前向传播 (双头输出)
            logits_trig, logits_dir = model(xb)

            # 3. 计算多任务 Loss
            loss_trig = criterion_trig(logits_trig, target_trig)
            
            # 方向 Loss 只在 action_mask 上计算
            if action_mask.any():
                loss_dir = criterion_dir(logits_dir[action_mask], target_dir[action_mask])
            else:
                loss_dir = 0.0

            # 4. 分阶段权重加权
            # 通过调节 lambda，你可以实现“先练 Trigger，再练 Direction”
            loss = (train_cfg.lambda_trig * loss_trig) + (train_cfg.lambda_dir * loss_dir)

            # 5. 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * xb.size(0)
            tr_total += xb.size(0)

        tr_loss /= max(1, tr_total)

        # --- 验证阶段 ---
        # 注意：eval_epoch 也需要修改，见下文
        va_loss, yv_true, yv_pred = eval_epoch(model, dl_va, device, train_cfg)
        
        # 保持你原本的 F1 计算逻辑（Macro-F1 对三分类平衡性最敏感）
        va_f1 = f1_score(yv_true, yv_pred, average="macro") if len(yv_true) else 0.0
        scheduler.step(va_loss)

        logger.info(f"Epoch {epoch:03d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} | va_macroF1 {va_f1:.4f}")

        # --- 保持你原本的双指标并行监控逻辑 ---
        if va_f1 > best_val_f1 + 1e-6:
            best_val_f1 = va_f1
            best_state_f1 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"🌟 [Metric] New Best Macro-F1: {va_f1:.4f}")

        if va_loss < best_val_loss - 1e-6:
            best_val_loss = va_loss
            best_state_loss = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"📉 [Metric] New Best va_loss: {va_loss:.4f}")
            wait = 0 
        else:
            wait += 1
            if wait >= train_cfg.patience:
                logger.warning(f"🛑 [Early Stop] Triggered after {train_cfg.patience} epochs.")
                break

    return {
        "best_f1_state": best_state_f1,
        "best_loss_state": best_state_loss,
        "f1_score": best_val_f1,
        "loss_score": best_val_loss
    }

def evaluate_and_save_results(
    results: dict,
    model: nn.Module,
    dl_te: DataLoader,
    # 这里不需要额外传入 criterion，因为 eval_epoch 内部会根据 2+2 逻辑处理
    device: torch.device,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    full_ds: TimeSeriesWindowDataset,
    feature_config_list,
    classes: np.ndarray,
    logger: logging.Logger
):
    """
    评估函数：通过传入 train_cfg 来协调 2+2 分类的 Loss 权重。
    """
    tasks = [
        ("Best_F1", results["best_f1_state"], results["f1_score"]),
        ("Best_Loss", results["best_loss_state"], results["loss_score"])
    ]

    final_metrics = {}

    for suffix, state, val_score in tasks:
        if state is None: continue
            
        model.load_state_dict(state)
        # 🌟 修改点：传入 train_cfg 替代原本的 criterion
        test_loss, yt_true, yt_pred = eval_epoch(model, dl_te, device, train_cfg)

        # --- 以下逻辑 100% 保留，因为它们基于已经“还原”的三分类标签 ---
        report_dict = classification_report(yt_true, yt_pred, output_dict=True, zero_division=0)
        test_f1 = report_dict['macro avg']['f1-score']

        logger.info(f"\n{'='*20} Evaluating Model Version: {suffix} {'='*20}")
        logger.info("\n=== Optimized Test Report ===")
        # 使用你原本的格式化打印函数
        logger.info(format_custom_report(report_dict))
        logger.info(f"Test macro-F1: {test_f1:.4f}")

        # 统计标签比例 (保留)
        counts = Counter(yt_true)
        total = sum(counts.values())
        logger.info(f"[{suffix}] True label proportion (Test set): " + 
                    ", ".join([f"{c}: {counts[c]/total:.2%}" for c in sorted(counts.keys())]))

        # 保存混淆矩阵 (保留)
        cm = confusion_matrix(yt_true, yt_pred, labels=classes)
        cm_path = os.path.join(train_cfg.save_dir, f"confmat_{suffix.lower()}.csv")
        pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]).to_csv(cm_path, index=True)

        feature_config_info = [
        (cls.__name__, params) for cls, params in feature_config_list
        ]

        # 保存 .pt 和 Meta (保留)
        pt_path = os.path.join(train_cfg.save_dir, f"model_{suffix.lower()}_info.pt")
        torch.save({
            "state_dict": state,
            "feature_config_list": feature_config_info, # 🌟 保存配置
            "classes": classes.tolist(),
            "channel": full_ds.feature_count,
            "window": data_cfg.window,
            "feature_cols": full_ds.feature_names,
            "label_col": data_cfg.label_col,
            "val_score": val_score,
            "test_f1": test_f1,
            "version_type": suffix
        }, pt_path)
        
        meta = model.export_meta(
            feature_cols=full_ds.feature_names,
            label_col=data_cfg.label_col,
            classes=classes.tolist(),
            window=data_cfg.window,
            model_version_tag=suffix,
            feature_config_list = feature_config_info
        )
        with open(os.path.join(train_cfg.save_dir, f"model_{suffix.lower()}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if suffix == "Best_F1":
            final_metrics = {
                "test_f1": test_f1,
                "val_f1": val_score,
                "test_loss": test_loss,
                "precision_short": report_dict.get('0', {}).get('precision', 0),
                "recall_short": report_dict.get('0', {}).get('recall', 0),
                "precision_long": report_dict.get('2', {}).get('precision', 0),
                "recall_long": report_dict.get('2', {}).get('recall', 0),
                "overfit_gap": abs(val_score - test_f1)
            }
            
    logger.error(f"Final Metrics Collected: {final_metrics}")
    return final_metrics

def format_custom_report(report_dict):
    """
    将 classification_report 的字典输出格式化为美观的表格字符串
    """
    header = f"{'Class':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}"
    sep = "-" * 65
    lines = [header, sep]
    
    # 1. 遍历具体的类别 (0, 1, 2)
    for label in sorted([k for k in report_dict.keys() if k.isdigit()]):
        v = report_dict[label]
        lines.append(
            f"{label:<10} | {v['precision']:10.4f} | {v['recall']:10.4f} | {v['f1-score']:10.4f} | {v['support']:8.0f}"
        )
    
    lines.append(sep)
    
    # 2. 统计摘要
    # Accuracy
    acc = report_dict.get('accuracy', 0)
    lines.append(f"{'Accuracy':<35} | {acc:10.4f} | {report_dict['macro avg']['support']:8.0f}")
    
    # Macro & Weighted Avg
    for avg in ['macro avg', 'weighted avg']:
        v = report_dict[avg]
        lines.append(
            f"{avg.capitalize():<10} | {v['precision']:10.4f} | {v['recall']:10.4f} | {v['f1-score']:10.4f} | {v['support']:8.0f}"
        )
        
    return "\n" + "\n".join(lines)

import matplotlib.pyplot as plt
import seaborn as sns

def diagnose_confidence(results, model, dl_te, device, logger, save_dir):
    """
    诊断模型预测的置信度分布。
    分析模型是在“坚定地犯错”还是“犹豫地对”。
    """
    model.eval()
    versions = [
        ("Best_F1", results["best_f1_state"]),
        ("Best_Loss", results["best_loss_state"])
    ]

    plt.figure(figsize=(15, 6))

    for i, (suffix, state) in enumerate(versions):
        if state is None: continue
        model.load_state_dict(state)
        
        all_probs = []
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for xb, yb in dl_te:
                xb = xb.to(device)
                logits = model(xb)
                # 使用 Softmax 将输出转化为概率
                probs = torch.softmax(logits, dim=1) 
                all_probs.append(probs.cpu().numpy())
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_trues.append(yb.numpy())

        probs_np = np.concatenate(all_probs) # [N, 3]
        preds_np = np.concatenate(all_preds)
        trues_np = np.concatenate(all_trues)

        # 提取模型对预测类别的“信心”（即最大概率值）
        confidences = np.max(probs_np, axis=1)
        
        # 区分：预测正确的信心 vs 预测错误的信心
        correct_mask = (preds_np == trues_np)
        conf_correct = confidences[correct_mask]
        conf_wrong = confidences[~correct_mask]

        # 绘图
        plt.subplot(1, 2, i+1)
        sns.histplot(conf_correct, color="green", label="Correct Preds", kde=True, stat="density", alpha=0.5)
        sns.histplot(conf_wrong, color="red", label="Wrong Preds", kde=True, stat="density", alpha=0.5)
        plt.title(f"Confidence Distribution: {suffix}")
        plt.xlabel("Max Probability (Confidence)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plot_path = os.path.join(save_dir, "confidence_diagnosis.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"📊 Confidence diagnosis plot saved to: {plot_path}")
    plt.close()

def find_best_threshold(results, model, dl_te, device, logger):
    """
    寻找最佳入场阈值。
    目标：在保持一定交易频率的前提下，尽可能提升 Precision。
    """
    model.eval()
    state = results["best_f1_state"]
    if state is None: return
    model.load_state_dict(state)

    all_probs = []
    all_trues = []
    with torch.no_grad():
        for xb, yb in dl_te:
            logits = model(xb.to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_trues.append(yb.numpy())

    probs = np.concatenate(all_probs) # [N, 3]
    trues = np.concatenate(all_trues)
    
    # 尝试不同的阈值门槛
    thresholds = np.linspace(0.34, 0.70, 20)
    search_results = []

    for th in thresholds:
        # 逻辑：只有当 Buy(0) 或 Sell(2) 的概率 > th 时才触发信号，否则视为 Hold(1)
        preds = []
        for p in probs:
            if p[0] > th: preds.append(0)
            elif p[2] > th: preds.append(2)
            else: preds.append(1)
        
        preds = np.array(preds)
        
        # 计算 Buy 和 Sell 的综合 Precision
        # 排除掉全是 Hold 的情况
        signal_mask = (preds != 1)
        if signal_mask.sum() == 0: continue
        
        # 计算信号准确率
        signal_acc = (preds[signal_mask] == trues[signal_mask]).mean()
        # 计算信号覆盖率（占总样本比例）
        coverage = signal_mask.mean()
        
        search_results.append({
            "threshold": th,
            "precision": signal_acc,
            "coverage": coverage,
            "count": signal_mask.sum()
        })

    # 打印搜索报告
    logger.info("\n📊 --- Threshold Search Report (Signal Precision vs Coverage) ---")
    logger.info(f"{'Threshold':<10} | {'Precision':<10} | {'Coverage':<10} | {'Signal Count'}")
    for r in search_results:
        logger.info(f"{r['threshold']:.3f}      | {r['precision']:.4f}     | {r['coverage']:.2%}      | {r['count']}")
    
    return search_results

@torch.no_grad()
def eval_epoch(model, loader, device, train_cfg):
    """
    适配 2+2 分类的验证函数：合并双头输出为三分类标签。
    """
    model.eval()
    tl, yt, yp = 0.0, [], []
    
    # 2+2 模式下的 Loss 计算组件
    criterion_trig = nn.CrossEntropyLoss()
    criterion_dir = nn.CrossEntropyLoss()

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        
        # 1. 前向传播：获取双头 Logits
        l_trig, l_dir = model(xb)
        
        # 2. 计算组合验证 Loss (用于监控监控 best_loss_state)
        # 映射 yb 到 2+2 的目标
        t_trig = (yb != 1).long()                   # 0/2 -> 1, 1 -> 0
        t_dir = torch.where(yb == 2, 1, 0).long()   # 2 -> 1, 0 -> 0
        act_mask = (yb != 1)
        
        loss_trig = criterion_trig(l_trig, t_trig)
        loss_dir = criterion_dir(l_dir[act_mask], t_dir[act_mask]) if act_mask.any() else 0.0
        
        # 按训练时的 lambda 权重叠加
        v_loss = (train_cfg.lambda_trig * loss_trig) + (train_cfg.lambda_dir * loss_dir)
        tl += v_loss.item() * xb.size(0)

        # 3. 核心：将双头预测“缝合”回三分类 (0, 1, 2)
        # 获取各头的最大概率索引
        p_trig = l_trig.argmax(1) # [B], 0: Neutral, 1: Action
        p_dir = l_dir.argmax(1)   # [B], 0: Short, 1: Long
        
        # 初始默认全为 1 (Neutral)
        batch_preds = torch.ones_like(yb) 
        # 如果 Trigger 判定有信号 (==1)，则根据 Direction 填入 2(Long) 或 0(Short)
        batch_preds = torch.where(p_trig == 1, torch.where(p_dir == 1, 2, 0), batch_preds)
        
        yp.append(batch_preds.cpu().numpy())
        yt.append(yb.cpu().numpy())

    if not yt: return float("nan"), np.array([]), np.array([])
    return tl/len(loader.dataset), np.concatenate(yt), np.concatenate(yp)

def main(feature_config_list, logger:logging.Logger):
    # 示例：直接在这里配置参数，取代了命令行参数
    
    # 1. 数据配置
    d_cfg = DataConfig()
    
    # 2. 训练配置
    t_cfg = TrainConfig()
    
    # 3. 模型配置 (在此处切换模型)
    # m_cfg = ModelConfigFactory.get_default_config("transformer")
    # m_cfg.d_model = 128
    
    # m_cfg = ModelConfigFactory.get_default_config("transformer")
    m_cfg = [LSTMConfig(), TransformerConfig(), ConvLSTMConfig(), CNNConfig(), XGBoostConfig()][2]
    # m_cfg.model_version = 1

    logger.info(f"Training {m_cfg.model_type}...")
    return run_training(feature_config_list, logger,d_cfg, t_cfg, m_cfg)
# ==============================================================================
# 5. 调用入口 (Main Entry)
# ==============================================================================

if __name__ == "__main__":
    logger, _ = common.setup_session_logger(sub_folder='train', file_level = logging.DEBUG)
    # 1. 准备冠军 Mask (第 8 代的最佳适应度组合)
    best_mask_str = "01000001110001111"
    # best_mask_str = "01000010000010011"   #过拟合风险最低
    mask = [int(bit) for bit in best_mask_str]

    # 2. 按照 GA 脚本的逻辑拆分
    # 排除 FeatureOrigin 作为可选基因，将其设为必选
    evolvable_config = [item for item in common.FEATURE_CONFIG_LIST if item[0].__name__ != "FeatureOrigin"]
    mandatory_config = [item for item in common.FEATURE_CONFIG_LIST if item[0].__name__ == "FeatureOrigin"]

    # 3. 根据 Mask 提取特征
    selected_evolvable = [evolvable_config[i] for i, bit in enumerate(mask) if bit == 1]
    FINAL_FEATURE_CONFIG = selected_evolvable + mandatory_config
    
    # 4. 打印结果
    logger.info("🏆 === 最终选中的特征组合 (Final Selection) ===")
    logger.info("-" * 50)
    for i, (cls, params) in enumerate(FINAL_FEATURE_CONFIG):
        logger.info(f"{i+1}. {cls.__name__:<20} | 参数: {params}")
    logger.info("-" * 50)
    logger.info(f"📊 总特征组数量: {len(FINAL_FEATURE_CONFIG)}")
    main(FINAL_FEATURE_CONFIG, logger)