#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,copy
import sys
import json
import logging,shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import WeightedRandomSampler, Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Path Settings
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from data_process import common
from model.data_loader import TimeSeriesWindowDataset
from model.model_factory import ModelFactory
from model.models.fusion_wrapper import FusionWrapper
# ==============================================================================
# 1. Configuration
# ==============================================================================
@dataclass
class ConvLSTMConfig:
    model_type: str = "conv_lstm"
    model_version: int = 1
    d_model: int = 96
    hidden_size: int = 64
    conv_layers: int = 3
    conv_kernel: int = 5
    conv_dropout: float = 0.2
    # conv_dilations: Tuple[int] = (1, 2, 4)
    bidirectional: bool = True
    lstm_dropout: float = 0.2
    input_norm: bool = True
    in_locked_p: float = 0.05
    out_locked_p: float = 0.05
    head_dropout: float = 0.3
    readout: str = "mix"    # 'last'|'meanmax'|'attn'|'mix'
    head: str = "linear"    # 'linear'|'mlp'
    logit_clip: Optional[float] = None
    use_feature_selector: bool = False

@dataclass
class DataConfig:
    csv_path: str = common.train_data_path
    feature_cols: list = field(default_factory=list)
    label_col: str = "label"
    window: int = common.CommonDefine.predict_num
    train_ratio: float = 0.7
    val_ratio: float = 0.15

@dataclass
class TrainConfig:
    #  PIPELINE MODE: "trigger_direction" OR "long_short_ovr"
    pipeline_mode: str = "trigger_direction" 
    
    model_cfg = ConvLSTMConfig()
    data_cfg = DataConfig()
    
    epochs: int = 20
    batch_size: int = 1024
    lr: float = 5e-3
    gate_lr: float = 5e-3     # feature selector
    weight_decay: float = 1e-3 #1e-3
    patience: int = 8
    seed: int = 42
    save_dir: str = common.TRAIN_OUT_DIR
    stride: int = 2
    use_cache: bool = True
    # Weights for binary tasks [Weight for Class 0, Weight for Class 1]
    # Class 1 is usually the "Signal" or "Action", so we weight it higher.
    mag_alpha: float = 0
    mag_limit: float = 4.0
    miss_penalty: float = 5  # 踏空惩罚
    flip_penalty: float = 4.0  # 做反惩罚 (针对 OvR 任务中的 Opposite Trend)

# ==============================================================================
# 2. Dataset & Helper Classes
# ==============================================================================

class SeqDataset(Dataset):
    def __init__(self, X, y, returns): 
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.r = torch.from_numpy(returns).float() # 存储收益率
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i], self.r[i]

def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def chrono_split(M, tr_r, va_r):
    n_tr = int(M * tr_r); n_va = int(M * va_r)
    return (0, n_tr), (n_tr, n_tr + n_va), (n_tr + n_va, M)

def get_balanced_sampler(labels):
    """Simple binary balanced sampler"""
    class_counts = np.bincount(labels)
    total_n = len(labels)
    # Inverse frequency
    weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def format_report(report_dict):
    """Format classification report into a clean table with global F1 metrics"""
    lines = [f"{'Class':<10} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Supp':<6}"]
    lines.append("-" * 60)
    
    # 打印每个类别的明细
    for label in sorted(report_dict.keys()):
        if label in ['accuracy', 'macro avg', 'weighted avg']: continue
        v = report_dict[label]
        lines.append(f"{label:<10} | {v['precision']:<8.4f} | {v['recall']:<8.4f} | {v['f1-score']:<8.4f} | {v['support']:<6}")
    
    lines.append("-" * 60)
    
    # 核心全局指标
    acc = report_dict.get('accuracy', 0)
    macro_f1 = report_dict.get('macro avg', {}).get('f1-score', 0)
    weighted_f1 = report_dict.get('weighted avg', {}).get('f1-score', 0)
    
    lines.append(f"{'Accuracy':<10} | {acc:<8.4f}")
    lines.append(f"{'Macro F1':<10} | {macro_f1:<8.4f}  (不考虑样本量，两类同等重要)")
    lines.append(f"{'Weight F1':<10} | {weighted_f1:<8.4f} (考虑样本量，Class 0 权重极大)")
    
    return "\n".join(lines)

def analyze_confidence(subtask_name, probs, trues, save_dir, bins=20):
    """
    Generates a Confidence Distribution plot similar to the user's reference.
    Compares 'Correct Preds' vs 'Wrong Preds' across max probability scores.
    """
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    
    # Separate confidences based on correctness
    correct_mask = (preds == trues)
    conf_correct = confidences[correct_mask]
    conf_wrong = confidences[~correct_mask]

    # Create the figure
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--'})

    # Plot Correct Predictions (Green)
    if len(conf_correct) > 0:
        sns.histplot(conf_correct, bins=bins, kde=True, color='green', 
                     label='Correct Preds', stat="density", alpha=0.5, element="bars")
    
    # Plot Wrong Predictions (Red)
    if len(conf_wrong) > 0:
        sns.histplot(conf_wrong, bins=bins, kde=True, color='red', 
                     label='Wrong Preds', stat="density", alpha=0.5, element="bars")

    # Formatting to match the provided style
    plt.title(f"Confidence Distribution: {subtask_name.upper()}", fontsize=14)
    plt.xlabel("Max Probability (Confidence)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    
    # Save the plot
    file_path = os.path.join(save_dir, f"diag_{subtask_name}_confidence.png")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    
    print(f"📊 Confidence diagnosis saved to: {file_path}")
# ==============================================================================
# 3. Label Processing Logic (The Core Difference)
# ==============================================================================

def prepare_data_for_subtask(X_raw, y_raw, rb_raw, subtask_type: str):
    """
    同步过滤特征、标签和收益率，确保索引对齐。
    """
    if subtask_type == "trigger":
        y_new = (y_raw != 1).astype(int)
        return X_raw, y_new, rb_raw # Trigger 不删样本，直接返回
        
    elif subtask_type == "direction":
        mask = (y_raw != 1) # 仅保留 Long(2) 和 Short(0)
        X_filt = X_raw[mask]
        y_filt = y_raw[mask]
        rb_filt = rb_raw[mask] # <--- 核心修复：同步过滤收益率
        # Map: 0(Short) -> 0, 2(Long) -> 1
        y_new = np.where(y_filt == 2, 1, 0)
        return X_filt, y_new, rb_filt

    elif subtask_type == "long_ovr":
        y_new = (y_raw == 2).astype(int)
        return X_raw, y_new, rb_raw
        
    elif subtask_type == "short_ovr":
        y_new = (y_raw == 0).astype(int)
        return X_raw, y_new, rb_raw
    
    else:
        raise ValueError(f"Unknown subtask: {subtask_type}")

# ==============================================================================
# 4. Generic Binary Training Engine
# ==============================================================================
def print_feature_importance(model, feature_names, logger, subtask_name, tag=""):
    """
    Extracts and prints weights from the FeatureSelector module.
    """
    if hasattr(model, 'feature_selector') and hasattr(model.feature_selector, 'importance_logits'):
        logger.info(f"🔍 [Feature Importance] Analysis for {subtask_name.upper()}{(' ' + tag) if tag else ''}")

        with torch.no_grad():
            # keep consistent with FeatureSelector forward: sigmoid(logits / 0.1)
            weights = torch.sigmoid(model.feature_selector.importance_logits / 0.1).cpu().numpy()

        importance_map = sorted(zip(feature_names, weights), key=lambda x: x[1], reverse=True)

        logger.info(f"{'Feature Name':<25} | {'Weight (Sigmoid)':<15}")
        logger.info("-" * 45)
        for name, weight in importance_map[:10]:
            logger.info(f"{name:<25} | {weight:<15.4f}")

        if len(importance_map) > 15:
            logger.info("...")
            for name, weight in importance_map[-5:]:
                logger.info(f"{name:<25} | {weight:<15.4f}")
        logger.info("-" * 45 + "\n")
    else:
        logger.info(f"ℹ️ Feature Selector is disabled for {subtask_name.upper()}.")

def get_trigger_sampler(y, pos_ratio=0.3):
    """
    y: 0/1 trigger label
    pos_ratio: 采样后 POS 在 batch 中的大致比例
    """
    y = np.asarray(y)
    cnt_pos = (y == 1).sum()
    cnt_neg = (y == 0).sum()

    # 目标：pos_ratio : (1 - pos_ratio)
    w_pos = pos_ratio / max(cnt_pos, 1)
    w_neg = (1 - pos_ratio) / max(cnt_neg, 1)

    weights = np.where(y == 1, w_pos, w_neg)
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def compute_soft_binary_loss(logits, y_sub, rb, subtask_type, train_cfg:TrainConfig, device):
    probs = torch.softmax(logits, dim=1)
    p_neg, p_pos = probs[:, 0], probs[:, 1]
    
    # 1. 基础幅度权重 (行情越大，权重越高)
    mag_weights = 1.0 + torch.log1p(train_cfg.mag_alpha * torch.abs(rb))
    mag_weights = torch.clamp(mag_weights, max=train_cfg.mag_limit)

    # 2. 任务特定的软化惩罚
    penalty = torch.zeros_like(p_neg)
    is_pos = (y_sub == 1)
    is_neg = (y_sub == 0)

    if subtask_type == "trigger":
        # 错过行情惩罚：真实是有信号(1)，但预测为无信号(0)的概率高
        penalty[is_pos] += p_neg[is_pos] * train_cfg.miss_penalty
    elif subtask_type in ["long_ovr", "short_ovr"]:
        # 错过行情惩罚：真实是有信号(1)，但预测为无信号(0)的概率高
        penalty[is_pos] += p_neg[is_pos] * train_cfg.flip_penalty

    # 3. 最终权重融合
    final_weights = mag_weights * (1.0 + penalty)
    final_weights = final_weights / (final_weights.mean() + 1e-8)

    # 4. 计算带权重的 NLL Loss
    log_probs = torch.log_softmax(logits, dim=1)
    # 直接提取 NLL Loss
    loss_samples = -log_probs.gather(1, y_sub.unsqueeze(1)).squeeze()

    return (loss_samples * final_weights).mean()

def train_binary_model(
    model, dl_tr, dl_va, dl_te,
    device, train_cfg: TrainConfig, logger,
    subtask_name: str,
):
    logger.info(f"🚀 [Start Joint Training] Subtask: {subtask_name.upper()}")

    # 1. Setup StrategyPara
    use_gate = bool(train_cfg.model_cfg.use_feature_selector and hasattr(model, "feature_selector"))

    if use_gate:
        gate_params = [model.feature_selector.importance_logits]
        base_params = [p for n, p in model.named_parameters() if "feature_selector" not in n]
        
        optimizer = torch.optim.AdamW([
            {"params": base_params, "lr": train_cfg.lr, "weight_decay": train_cfg.weight_decay},
            {"params": gate_params, "lr": train_cfg.gate_lr, "weight_decay": 0.0}
        ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    # 2. Shared Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=train_cfg.patience // 2
    )

    best_f1 = 0.0
    best_va_loss = float('inf')
    best_state_f1 = None
    best_state_loss = None
    wait = 0
    
    best_val_probs, best_val_trues, best_val_preds = None, None, None

    # --- Training Loop ---
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        tr_loss_list = []

        # 注意：这里解包 3 个值 (xb, y_sub, rb)，对应 SeqDataset 的返回
        for xb, y_sub, rb in tqdm(dl_tr, desc=f"Ep {epoch}", leave=False):
            xb, y_sub, rb = xb.to(device), y_sub.to(device), rb.to(device)
            
            optimizer.zero_grad()
            logits = model(xb)
            
            # 训练损失
            loss = compute_soft_binary_loss(
                logits, y_sub, rb, 
                subtask_name, train_cfg, device
            )
            
            loss.backward()
            optimizer.step()
            tr_loss_list.append(loss.item())

        # --- Validation ---
        model.eval()
        val_preds, val_trues, val_probs = [], [], []
        val_loss_sum = 0.0
        
        with torch.no_grad():
            # 统一解包逻辑：xb (特征), yb (子任务标签), rb (收益率)
            for xb, yb, rb in dl_va:
                xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
                logits = model(xb)
                
                # ✅ 关键修改：验证集同样使用自定义损失函数
                loss = compute_soft_binary_loss(
                    logits, yb, rb, 
                    subtask_name, train_cfg, device
                )
                
                val_loss_sum += loss.item()
                probs = torch.softmax(logits, dim=1)
                val_probs.append(probs.cpu().numpy())
                val_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        # 指标计算
        avg_tr_loss = float(np.mean(tr_loss_list)) if tr_loss_list else 0.0
        avg_va_loss = val_loss_sum / max(1, len(dl_va))
        
        v_preds = np.concatenate(val_preds) if val_preds else np.array([])
        v_trues = np.concatenate(val_trues) if val_trues else np.array([])
        v_probs = np.concatenate(val_probs) if val_probs else np.array([])
        val_f1 = f1_score(v_trues, v_preds, average="macro") if len(v_trues) else 0.0

        logger.info(
            f"Ep {epoch:02d} | tr_loss {avg_tr_loss:.4f} | va_loss {avg_va_loss:.4f} | va_macroF1 {val_f1:.4f}"
        )

        scheduler.step(avg_va_loss)

        # 保存逻辑与 Early Stopping
        progress_made = False
        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_state_f1 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_probs, best_val_trues, best_val_preds = v_probs, v_trues, v_preds
            progress_made = True

        if avg_va_loss < best_va_loss - 1e-6:
            best_va_loss = avg_va_loss
            best_state_loss = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            progress_made = True

        if progress_made:
            wait = 0
        else:
            wait += 1
            if wait >= train_cfg.patience:
                logger.warning(f"🛑 Early Stop at Epoch {epoch}")
                break

    if best_state_f1 is not None:
        model.load_state_dict(best_state_f1)

    # 报告与诊断
    logger.info(f"\n{'#'*10} [{subtask_name.upper()}] FINAL REPORT {'#'*10}")
    if best_val_trues is not None:
        logger.info("\n" + format_report(classification_report(best_val_trues, best_val_preds, output_dict=True, zero_division=0)))
        analyze_confidence(subtask_name, best_val_probs, best_val_trues, train_cfg.save_dir)

    return best_state_f1

# ==============================================================================
# 5. Pipeline Logic & Fusion
# ==============================================================================

def run_pipeline(feature_config_list, logger, train_cfg: TrainConfig):
    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Raw Data ONCE
    df = common.load_train_df()
    logger.info(f"Using TimeSeriesWindowDataset with window={train_cfg.data_cfg.window} Origin data len {len(df)}...")
    
    feature_cols = train_cfg.data_cfg.feature_cols if train_cfg.data_cfg.feature_cols else list(df.columns)
    logger.info(f"Features num:{len(feature_cols)},: {feature_cols}")
    full_ds = TimeSeriesWindowDataset(
        feature_config_list=feature_config_list,
        df=df, kline_interval_ms=common.load_interval_ms(),
        feature_cols=feature_cols, 
        label_col=train_cfg.data_cfg.label_col,
        window=train_cfg.data_cfg.window, 
        stride=train_cfg.stride, 
        use_cache=train_cfg.use_cache,
        cache_path=os.path.join(common.TEMPORARY_DIR,"train_cache.pt"), show_feature_distribution=True
    )
    X_raw, y_raw = full_ds.X.numpy(), full_ds.y.numpy()
    returns_raw = full_ds.returns.numpy()

    # Define tasks based on mode
    if train_cfg.pipeline_mode == "trigger_direction":
        tasks = ["trigger","direction"]    #"trigger", "direction"
    elif train_cfg.pipeline_mode == "long_short_ovr":
        tasks = ["long_ovr", "short_ovr"] #["long_ovr", "short_ovr"]
    else:
        raise ValueError("Invalid pipeline mode")

    models = {}
    
    # 2. Train Loop for each Subtask
    for task_name in tasks:
        # A. Preprocess Data for this specific task
        X_t, y_t, rb = prepare_data_for_subtask(X_raw, y_raw,full_ds.returns, task_name)
        
        # Split (Ensure Chronological consistency)
        tr_rng, va_rng, te_rng = chrono_split(len(y_t), train_cfg.data_cfg.train_ratio, train_cfg.data_cfg.val_ratio)
        
        ds_tr = SeqDataset(X_t[tr_rng[0]:tr_rng[1]], y_t[tr_rng[0]:tr_rng[1]],rb[tr_rng[0]:tr_rng[1]].numpy())
        ds_va = SeqDataset(X_t[va_rng[0]:va_rng[1]], y_t[va_rng[0]:va_rng[1]],rb[va_rng[0]:va_rng[1]].numpy())
        ds_te = SeqDataset(X_t[te_rng[0]:te_rng[1]], y_t[te_rng[0]:te_rng[1]],rb[te_rng[0]:te_rng[1]].numpy())
        
        # B. Build Model
        model = ModelFactory.build_for_training(
            device=device,
            input_size=full_ds.feature_count,
            n_classes=2, # Always binary
            **asdict(train_cfg.model_cfg)
        )
        
        # C. Train
        # dl_tr = DataLoader(ds_tr, batch_size=train_cfg.batch_size, num_workers=4, shuffle=True) #sampler=get_balanced_sampler(ds_tr.y.numpy()))
        cfg_task = copy.deepcopy(train_cfg)
        if task_name == 'direction':
            dl_tr = DataLoader(ds_tr, batch_size=train_cfg.batch_size, num_workers=4, shuffle=True)
            train_cfg.lr = 5e-3
            train_cfg.gate_lr = 1e-2
            train_cfg.weight_decay = 1e-3
        else :
            dl_tr = DataLoader(ds_tr, batch_size=train_cfg.batch_size, num_workers=4, shuffle=True)
            if task_name == 'trigger':
                sampler = get_trigger_sampler(ds_tr.y.numpy(), pos_ratio=0.25)
                dl_tr = DataLoader(ds_tr, batch_size=train_cfg.batch_size, num_workers=4, sampler=sampler)
                train_cfg.lr = 2e-3
                train_cfg.gate_lr = 2e-3
                train_cfg.weight_decay = 1e-3
            elif task_name == 'long_ovr':
                sampler = get_trigger_sampler(ds_tr.y.numpy(), pos_ratio=0.1)
                dl_tr = DataLoader(ds_tr, batch_size=train_cfg.batch_size, num_workers=4, sampler=sampler)
                train_cfg.lr = 2e-3
                train_cfg.gate_lr = 2e-3
                train_cfg.weight_decay = 1e-3
            elif task_name == 'short_ovr':
                sampler = get_trigger_sampler(ds_tr.y.numpy(), pos_ratio=0.1)
                dl_tr = DataLoader(ds_tr, batch_size=train_cfg.batch_size, num_workers=4, sampler=sampler)
                train_cfg.lr = 2e-3
                train_cfg.gate_lr = 5e-4
                train_cfg.weight_decay = 1e-3
        dl_va = DataLoader(ds_va, batch_size=train_cfg.batch_size, shuffle=False)
        dl_te = DataLoader(ds_te, batch_size=train_cfg.batch_size, shuffle=False)
        
        best_state = train_binary_model(model, dl_tr, dl_va, dl_te, device, train_cfg, logger, task_name)
        
        # Reload best state for fusion
        model.load_state_dict(best_state)
        if train_cfg.model_cfg.use_feature_selector:
            print_feature_importance(model, full_ds.feature_names, logger, task_name)

        model.eval()
        models[task_name] = model

    if len(tasks)!=2:
        exit()

    # 3. Fusion Evaluation (The Important Part)
    # We must evaluate on the TEST portion of the RAW data (y_raw) to compare against real 3-class ground truth
    _, _, te_rng_raw = chrono_split(len(y_raw), train_cfg.data_cfg.train_ratio, train_cfg.data_cfg.val_ratio)
    
    X_test_raw = torch.from_numpy(X_raw[te_rng_raw[0]:te_rng_raw[1]]).float().to(device)
    y_test_true = y_raw[te_rng_raw[0]:te_rng_raw[1]] # 0, 1, 2

    logger.info(f"\n{'='*20} 🧩 FUSION EVALUATION: {train_cfg.pipeline_mode.upper()} {'='*20}")
    
    evaluate_and_save_pipeline(
            models_dict=models,
            X_test_raw=X_test_raw, 
            y_test_true=y_test_true, 
            device=device,
            train_cfg=train_cfg,
            full_ds=full_ds,
            feature_config_list=feature_config_list,
            logger=logger
        )

@torch.no_grad()
def evaluate_fusion_trigger_direction(models, X, y_true, logger):
    """
    Logic: 
    1. Trigger Model decides: Action vs Neutral.
    2. IF Action: Direction Model decides Long vs Short.
    3. ELSE: Neutral.
    """
    model_trig = models["trigger"]
    model_dir = models["direction"]
    
    # 1. Trigger Inference
    logits_trig = model_trig(X)
    preds_trig = torch.argmax(logits_trig, dim=1).cpu().numpy() # 0:Neutral, 1:Action
    
    # 2. Direction Inference
    logits_dir = model_dir(X)
    preds_dir = torch.argmax(logits_dir, dim=1).cpu().numpy() # 0:Short, 1:Long
    
    # 3. Fuse
    # Start with all Neutral (1)
    final_preds = np.full_like(y_true, 1) 
    
    # Where Trigger says Action (1)
    action_mask = (preds_trig == 1)
    
    # Map Direction outputs: 0->0(Short), 1->2(Long)
    mapped_dir = np.where(preds_dir == 1, 2, 0)
    
    # Apply
    final_preds[action_mask] = mapped_dir[action_mask]
    
    # Report
    print_metrics(y_true, final_preds, logger, "Trigger + Direction Fusion")

@torch.no_grad()
def evaluate_fusion_ovr(models, X, y_true, logger):
    """
    Logic:
    Long Model: 1=Long, 0=Others
    Short Model: 1=Short, 0=Others
    
    Fusion:
    - Long=1, Short=0 -> Long (2)
    - Long=0, Short=1 -> Short (0)
    - Long=0, Short=0 -> Neutral (1)
    - Long=1, Short=1 -> Conflict (1, Neutral)
    """
    model_long = models["long_ovr"]
    model_short = models["short_ovr"]
    
    # 1. Inference
    preds_long = torch.argmax(model_long(X), dim=1).cpu().numpy()
    preds_short = torch.argmax(model_short(X), dim=1).cpu().numpy()
    
    # 2. Fuse
    final_preds = np.full_like(y_true, 1) # Default Neutral
    
    # Long Condition
    long_mask = (preds_long == 1) & (preds_short == 0)
    final_preds[long_mask] = 2
    
    # Short Condition
    short_mask = (preds_short == 1) & (preds_long == 0)
    final_preds[short_mask] = 0
    
    # Report
    print_metrics(y_true, final_preds, logger, "Long-OVR + Short-OVR Fusion")

def print_metrics(y_true, y_pred, logger, title):
    logger.info(f"\n📊 --- {title} ---")
    
    # Sample and Prediction Distribution
    total = len(y_true)
    c_true, c_pred = Counter(y_true), Counter(y_pred)
    logger.info(f"📈 Sample Distribution (Data): NEGATIVE(0): {c_true.get(0,0)/total:.2%}, NEUTRAL(1): {c_true.get(1,0)/total:.2%}, POSITIVE (2): {c_true.get(2,0)/total:.2%}")
    logger.info(f"🔮 Prediction Distribution (Model): NEGATIVE(0): {c_pred.get(0,0)/total:.2%}, NEUTRAL(1): {c_pred.get(1,0)/total:.2%}, POSITIVE (2): {c_pred.get(2,0)/total:.2%}")
    
    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    logger.info("\n" + format_report(report))
    
    #  CORE F1 METRICS
    f1_short = report.get('0', {}).get('f1-score', 0)
    f1_long = report.get('2', {}).get('f1-score', 0)
    macro_f1 = report.get('macro avg', {}).get('f1-score', 0)
    trend_f1 = (f1_short + f1_long) / 2
    
    logger.info(f"🏆 Final Fusion Macro F1: {macro_f1:.4f}")
    logger.info(f"🎯 Trend F1 (Short/Long Avg): {trend_f1:.4f}")
    
    # Risk Metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    flip_rate = (cm[0, 2] + cm[2, 0]) / ((y_true == 0).sum() + (y_true == 2).sum() + 1e-6)
    logger.info(f"☠️ Fatal Flip Rate: {flip_rate:.2%}")

def evaluate_and_save_pipeline(
    models_dict, 
    X_test_raw,    # 传入原始测试集特征
    y_test_true,   # 传入原始测试集标签 (0,1,2)
    device, 
    train_cfg: TrainConfig, 
    full_ds,
    feature_config_list,
    logger
):
    """
    集评估、保存、索引生成于一体的管线终点函数。
    使用 FusionWrapper 确保推理逻辑的一致性。
    """
    # --- 1. 实例化推理包装器 (与 model_loader 加载逻辑一致) ---
    fusion_model = FusionWrapper(models_dict, mode=train_cfg.pipeline_mode)
    fusion_model.to(device)
    fusion_model.eval()

    # --- 2. 统一评估逻辑 ---
    logger.info(f"\n{'='*20} 🧩 FUSION EVALUATION: {train_cfg.pipeline_mode.upper()} {'='*20}")
    placeholder_returns = np.zeros(len(y_test_true))
    test_ds = SeqDataset(X_test_raw.cpu().numpy(), y_test_true,placeholder_returns) # 转为 CPU Numpy 构造 Dataset
    test_dl = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False)
    
    all_fused_probs = []
    
    with torch.no_grad():
        for xb, _, _ in tqdm(test_dl, desc="Evaluating Fusion", leave=False):
            xb = xb.to(device)
            # 调用包装器获取 3 分类概率
            _, fused_probs = fusion_model(xb, return_fused=True) 
            all_fused_probs.append(fused_probs.cpu()) # 移回 CPU 释放显存
            
    # 拼接结果并计算预测值
    fused_probs_all = torch.cat(all_fused_probs, dim=0).numpy()
    final_preds = np.argmax(fused_probs_all, axis=1)

    # 调用原有的指标打印函数
    print_metrics(y_test_true, final_preds, logger, f"{train_cfg.pipeline_mode.upper()} Pipeline")

    # --- 3. 独立保存子模型 ---
    save_dir = train_cfg.save_dir
    feature_config_info = [
        (c.feature.__name__, c.parameters) for c in feature_config_list
    ]
    
    sub_model_map = {}
    for name, model in models_dict.items():
        suffix = "best"
        file_prefix = f"model_{name}_{suffix}"
        
        model_path = os.path.join(save_dir, f"{file_prefix}_info.pt")
        meta_path = os.path.join(save_dir, f"{file_prefix}_meta.json")
        
        # 使用子模型自身的保存逻辑，包含各自的架构参数
        model.save_checkpoint(
            model_path=model_path,
            meta_path=meta_path,
            window=train_cfg.data_cfg.window,
            feature_cols=full_ds.feature_names,
            label_col=train_cfg.data_cfg.label_col,
            classes=[0, 1], # 子模型固定为二分类
            feature_config_list=feature_config_info
        )
        
        sub_model_map[name] = {
            "model": f"{file_prefix}_info.pt",
            "meta": f"{file_prefix}_meta.json"
        }
    
    # --- 4. 生成 Task Description 索引文件 ---
    task_desc = {
        "task_type": train_cfg.pipeline_mode,
        "timestamp": pd.Timestamp.now().isoformat(),
        "models": sub_model_map
    }
    
    task_path = os.path.join(save_dir, "task_description.json")
    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(task_desc, f, ensure_ascii=False, indent=2)
        
    logger.info(f"🎉 Task description and models saved to: {save_dir}")

def main(logger:logging.Logger):
    if os.path.exists(common.TRAIN_OUT_DIR):
        shutil.rmtree(common.TRAIN_OUT_DIR)
    os.makedirs(common.TRAIN_OUT_DIR, exist_ok=True)

    # Configure Features
    feature_config_list = [
        common.FCMA,
        common.FCQavMa,
        common.FCCandle,
        common.FCOrigin,
    ]
    feature_config_list = [
        # 1. 自定义的成交量爆发特征 (窗口 512，对比前 2 强)
        # FCVolumeEvent, 

        # 2. 价格趋势与指标类    FeatureMA > FeatureRsi/FeatureKdj/FeatureMACD   what happen to FeatureMACD??
        # common.FCMACD,   # （12，26，9），（6，13，5）或（10，20，7）
        common.FCMA,     # slope 值搭配使用
        # common.FCRSI,
        common.FCKDJ,

        # # 价格通道类，2选1   FeatureKeltner >> FeatureBoll/FeatureDonchian
        # common.FCDonchian, 
        common.FCKeltner,
        # common.FCBoll,

        # # 3. 量能与成交活跃度类 FeatureQavMa > FeatureMFI/FeatureWAP > FeatureCFM  > FeaturePVT >FeatureVolMa
        # # FCVolMa,
        common.FCQavMa,
        # common.FCOBV,    # 等于 FeaturePVT 丢掉幅度信息。不如 FeaturePVT，直接丢弃
        common.FCPVT,    # 累积性变量，对短期预测作用小，不如动量
        # common.FCWAP,
        common.FCCFM,
        # common.FCMFI,
        # # FCATS,  # 负作用

        # # 4. K线形态类
        common.FCCandle,
        common.FCOrigin,
    ]

    # Configure Training
    cfg = TrainConfig()
    
    #  SET YOUR MODE HERE
    # Option : "long_short_ovr"/long_short_ovr
    cfg.pipeline_mode = "long_short_ovr"  # Change this to switch modes
    
    # Run
    run_pipeline(feature_config_list, logger, cfg)
# ==============================================================================
# 6. Main Entry
# ==============================================================================

if __name__ == "__main__":
    # Setup Logger
    logger, _ = common.setup_session_logger(sub_folder='train', file_level=logging.DEBUG)
    main(logger)