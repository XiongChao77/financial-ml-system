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
    conv_dilations: Tuple[int] = (1, 2, 4)
    bidirectional: bool = True
    lstm_dropout: float = 0.2
    input_norm: bool = True
    in_locked_p: float = 0.05
    out_locked_p: float = 0.05
    head_dropout: float = 0.3
    readout: str = "mix"    # 'last'|'meanmax'|'attn'|'mix'
    head: str = "linear"    # 'linear'|'mlp'
    logit_clip: Optional[float] = None
    use_feature_selector: bool = True

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
    lr: float = 1e-3
    weight_decay: float = 1e-3 #1e-3
    patience: int = 8
    seed: int = 42
    save_dir: str = common.TEMPORARY_DIR
    stride: int = 1
    use_cache: bool = True
    
    # Weights for binary tasks [Weight for Class 0, Weight for Class 1]
    # Class 1 is usually the "Signal" or "Action", so we weight it higher.
    binary_pos_weight: float = 3.0 

# ==============================================================================
# 2. Dataset & Helper Classes
# ==============================================================================

class SeqDataset(Dataset):
    def __init__(self, X, y): 
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

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

def prepare_data_for_subtask(X_raw, y_raw, subtask_type: str):
    """
    Transforms raw multi-class labels (0:Short, 1:Neutral, 2:Long)
    into binary labels and filters data for specific sub-tasks.
    
    Returns: X_filtered, y_transformed
    """
    if subtask_type == "trigger":
        # Task: Detect Action (0,2) vs Neutral (1)
        # 0 (Neutral) -> 0
        # 1 (Short/Long) -> 1
        y_new = (y_raw != 1).astype(int)
        return X_raw, y_new
        
    elif subtask_type == "direction":
        # Task: Long (2) vs Short (0)
        # Filter: Remove Neutral (1)
        mask = (y_raw != 1)
        X_filt = X_raw[mask]
        y_filt = y_raw[mask]
        # Map: 0(Short) -> 0, 2(Long) -> 1
        y_new = np.where(y_filt == 2, 1, 0)
        return X_filt, y_new

    elif subtask_type == "long_ovr":
        # Task: Long (2) vs Others (0, 1)
        # 0 (Others) -> 0
        # 1 (Long) -> 1
        y_new = (y_raw == 2).astype(int)
        return X_raw, y_new
        
    elif subtask_type == "short_ovr":
        # Task: Short (0) vs Others (1, 2)
        # 0 (Others) -> 0
        # 1 (Short) -> 1
        y_new = (y_raw == 0).astype(int)
        return X_raw, y_new
    
    else:
        raise ValueError(f"Unknown subtask: {subtask_type}")

# ==============================================================================
# 4. Generic Binary Training Engine
# ==============================================================================
def print_feature_importance(model, feature_names, logger, subtask_name):
    """
    Extracts and prints weights from the FeatureSelector module.
    """
    # Check if the model has the FeatureSelector parameter
    if hasattr(model, 'feature_selector') and hasattr(model.feature_selector, 'importance_logits'):
        logger.info(f"🔍 [Feature Importance] Analysis for {subtask_name.upper()}")
        
        with torch.no_grad():
            # Apply sigmoid just like the model does during forward pass
            weights = torch.sigmoid(model.feature_selector.importance_logits).cpu().numpy()
        
        # Pair with names and sort
        importance_map = sorted(zip(feature_names, weights), key=lambda x: x[1], reverse=True)
        
        # Print Top 10 and Bottom 5 for brevity
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

def train_binary_model(
    model, dl_tr, dl_va, dl_te, 
    device, train_cfg: TrainConfig, logger, 
    subtask_name: str, 
    pos_weight_multiplier: float = 3.0
):
    """
    Generic training loop with immediate sub-task evaluation.
    """
    logger.info(f"🚀 [Start Training] Subtask: {subtask_name.upper()}")
    
    # Loss with Manual Class Weighting
    weights = torch.tensor([1.0, pos_weight_multiplier], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    
    best_f1, best_state = 0.0, None
    wait = 0
    best_val_probs, best_val_trues, best_val_preds = None, None, None

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        tr_loss_list = []
        for xb, yb in tqdm(dl_tr, desc=f"Ep {epoch} {subtask_name}", leave=False, ncols=100):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(); logits = model(xb) 
            loss = criterion(logits, yb);
            #  如果开启了选择器，增加 L1 惩罚项项，强迫权重向 0 靠拢
            if train_cfg.model_cfg.use_feature_selector and hasattr(model, 'feature_selector'):
                # 将参数分为两组
                gate_params = [model.feature_selector.importance_logits]
                base_params = [p for n, p in model.named_parameters() if 'feature_selector' not in n]
                
                optimizer = torch.optim.AdamW([
                    {'params': base_params, 'lr': train_cfg.lr},
                    {'params': gate_params, 'lr': train_cfg.lr * 10}  # 👈 给门控层 10 倍学习率
                ], weight_decay=train_cfg.weight_decay)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            tr_loss_list.append(loss.item())
            
        model.eval()
        val_preds, val_trues, val_probs = [], [], []
        val_loss_sum = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb); loss = criterion(logits, yb)
                val_loss_sum += loss.item()
                probs = torch.softmax(logits, dim=1)
                val_probs.append(probs.cpu().numpy())
                val_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                val_trues.append(yb.cpu().numpy())
                
        # Epoch Metrics
        avg_tr_loss = np.mean(tr_loss_list)
        avg_va_loss = val_loss_sum / len(dl_va)
        v_preds, v_trues, v_probs = np.concatenate(val_preds), np.concatenate(val_trues), np.concatenate(val_probs)
        val_f1 = f1_score(v_trues, v_preds, average="macro")
        
        logger.info(f"Ep {epoch:02d} | tr_loss {avg_tr_loss:.4f} | va_loss {avg_va_loss:.4f} | va_macroF1 {val_f1:.4f}")
        scheduler.step(avg_va_loss)

        if val_f1 > best_f1:
            best_f1, best_state = val_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_probs, best_val_trues, best_val_preds = v_probs, v_trues, v_preds
            wait = 0
        else:
            wait += 1
            if wait >= train_cfg.patience: break
                
    #  SUB-TASK REPORT
    logger.info(f"\n{'#'*10} [{subtask_name.upper()}] SUB-TASK REPORT {'#'*10}")
    counts = Counter(best_val_trues)
    total = len(best_val_trues)
    dist_str = ", ".join([f"Class {k}: {counts[k]/total:.2%}" for k in sorted(counts.keys())])
    logger.info(f"📈 Sample Distribution (Data): {dist_str}")
    logger.info("\n" + format_report(classification_report(best_val_trues, best_val_preds, output_dict=True, zero_division=0)))
    
    # Visual Diagnostic
    analyze_confidence(subtask_name, best_val_probs, best_val_trues, train_cfg.save_dir)
    return best_state

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
        cache_path=os.path.join(common.TEMPORARY_DIR,"train_cache.pt"),
    )
    X_raw, y_raw = full_ds.X.numpy(), full_ds.y.numpy()
    
    # Define tasks based on mode
    if train_cfg.pipeline_mode == "trigger_direction":
        tasks = ["trigger", "direction"]
    elif train_cfg.pipeline_mode == "long_short_ovr":
        tasks = ["long_ovr","short_ovr"] #["long_ovr", "short_ovr"]
    else:
        raise ValueError("Invalid pipeline mode")

    models = {}
    
    # 2. Train Loop for each Subtask
    for task_name in tasks:
        # A. Preprocess Data for this specific task
        X_t, y_t = prepare_data_for_subtask(X_raw, y_raw, task_name)
        
        # Split (Ensure Chronological consistency)
        tr_rng, va_rng, te_rng = chrono_split(len(y_t), train_cfg.data_cfg.train_ratio, train_cfg.data_cfg.val_ratio)
        
        ds_tr = SeqDataset(X_t[tr_rng[0]:tr_rng[1]], y_t[tr_rng[0]:tr_rng[1]])
        ds_va = SeqDataset(X_t[va_rng[0]:va_rng[1]], y_t[va_rng[0]:va_rng[1]])
        ds_te = SeqDataset(X_t[te_rng[0]:te_rng[1]], y_t[te_rng[0]:te_rng[1]])
        
        # B. Build Model
        model = ModelFactory.build_for_training(
            device=device,
            input_size=full_ds.feature_count,
            n_classes=2, # Always binary
            **asdict(train_cfg.model_cfg)
        )
        
        # C. Train
        dl_tr = DataLoader(ds_tr, batch_size=train_cfg.batch_size, sampler=get_balanced_sampler(ds_tr.y.numpy()), num_workers=4)
        dl_va = DataLoader(ds_va, batch_size=train_cfg.batch_size, shuffle=False)
        dl_te = DataLoader(ds_te, batch_size=train_cfg.batch_size, shuffle=False)
        
        best_state = train_binary_model(model, dl_tr, dl_va, dl_te, device, train_cfg, logger, task_name, train_cfg.binary_pos_weight)
        
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
    
    if train_cfg.pipeline_mode == "trigger_direction":
        evaluate_fusion_trigger_direction(models, X_test_raw, y_test_true, logger)
    else:
        evaluate_fusion_ovr(models, X_test_raw, y_test_true, logger)

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

# ==============================================================================
# 6. Main Entry
# ==============================================================================

if __name__ == "__main__":
    # Setup Logger
    logger, _ = common.setup_session_logger(sub_folder='train', file_level=logging.DEBUG)
    
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
        # common.FCMA,     # slope 值搭配使用
        # common.FCRSI,
        # common.FCKDJ,

        # # 价格通道类，2选1   FeatureKeltner >> FeatureBoll/FeatureDonchian
        # common.FCDonchian, 
        common.FCKeltner,
        # common.FCBoll,

        # # 3. 量能与成交活跃度类 FeatureQavMa > FeatureMFI/FeatureWAP > FeatureCFM  > FeaturePVT >FeatureVolMa
        # # FCVolMa,
        # common.FCQavMa,
        # common.FCOBV,    # 等于 FeaturePVT 丢掉幅度信息。不如 FeaturePVT，直接丢弃
        # common.FCPVT,    # 累积性变量，对短期预测作用小，不如动量
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
    # Option : "trigger_direction"/long_short_ovr
    cfg.pipeline_mode = "long_short_ovr"  # Change this to switch modes
    
    # Run
    run_pipeline(feature_config_list, logger, cfg)