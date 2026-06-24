#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,shutil,time
import sys
import json
import logging
import torch
# Path setup
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from data_process import common
# 1. Force-enable persistent graph cache
torch._inductor.config.fx_graph_cache = True

# 2. Use a unified cache directory (recommended under project output)
# This avoids repeated compilation across restarts and parallel runs
cache_dir = os.path.join(common.TRAIN_OUT_DIR, ".inductor_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

# 3. 5090 optimization: if input shapes don't change often, consider disabling dynamic shapes for maximum throughput
# torch._inductor.config.dynamic_shapes = False

import torch.nn as nn
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List, Dict
from tqdm import tqdm
from collections import Counter
from pathlib import Path

from torch.utils.data import WeightedRandomSampler, Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score,accuracy_score,balanced_accuracy_score,matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
from model.data_loader import TimeSeriesWindowDataset
from model.model_factory import ModelFactory
from model.train_config import *
# ==============================================================================
# 3. Core Logic
# ==============================================================================
def get_balanced_sampler(dataset):
    # 1. Extract labels for all samples
    all_labels = dataset.labels 
    
    # 2. Count each class: [Short(0), Neutral(1), Long(2)]
    class_counts = torch.bincount(torch.tensor(all_labels))
    total_n = class_counts.sum().float()
    
    # 3. Compute "natural" proportions
    # Keep Neutral at its original share
    p_neutral = class_counts[1] / total_n
    # Compute total Action share (Long + Short)
    p_action = (class_counts[0] + class_counts[2]) / total_n
    
    # 4. Set target proportions: split p_action equally between Long and Short
    # Index mapping: [0: Short, 1: Neutral, 2: Long]
    target_props = torch.tensor([p_action / 2, p_neutral, p_action / 2]) 
    
    # 5. Sampling weights: Weight = Target_Prop / Actual_Count
    # This makes Long/Short have equal total selection probability while keeping Neutral at natural level
    class_weights = target_props / class_counts.float()
    
    # 6. Assign weight to each sample and create sampler
    sample_weights = [class_weights[label] for label in all_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def apply_feature_direction(X: torch.Tensor, feature_names: List[str], direction_map: Dict[str, int], logger) -> torch.Tensor:
    """
    Multiply feature columns with direction=-1 by -1 to make them positively correlated with returns.
    X: shape [N, T, F], normalized feature tensor
    feature_names: list of feature names aligned with X's 3rd dimension
    direction_map: {feature_name: 1 or -1}
    """
    if direction_map is None or len(direction_map) == 0:
        return X
    
    flip_indices = []
    flip_names = []
    for i, fname in enumerate(feature_names):
        if direction_map.get(fname, 1) == -1:
            flip_indices.append(i)
            flip_names.append(fname)
    
    if flip_indices:
        logger.info(f"🔄 Flipping {len(flip_indices)} features with ic_direction=-1: {flip_names[:10]}{'...' if len(flip_names) > 10 else ''}")
        X[:, :, flip_indices] = -X[:, :, flip_indices]
    
    return X

def prepare_binary_data_for_task(X_raw, y_raw, rb_raw, train_task: TrainTask):
    """
    独立转换逻辑：不影响原始全量数据
    """
    if train_task == TrainTask.SINGLE_MODEL_TRIGGER:
        y_new = (y_raw != 1).long()
        return X_raw, y_new, rb_raw
    if train_task == TrainTask.SINGLE_MODEL_DIR:
        mask = (y_raw != 1) # 仅保留 Long(2) 和 Short(0)
        X_filt = X_raw[mask]
        y_filt = y_raw[mask]
        rb_filt = rb_raw[mask] # <--- 核心修复：同步过滤收益率
        # Map: 0(Short) -> 0, 2(Long) -> 1
        y_new = (y_filt == 2).long()
        return X_filt, y_new, rb_filt
    if train_task == TrainTask.SINGLE_MODEL_TRIGGER:
        # Trigger: 0(Short) & 2(Long) -> 1 (Action), 1(Neutral) -> 0 (Stay)
        y_new = (y_raw != 1).long()
        return X_raw, y_new, rb_raw
        
    elif train_task == TrainTask.SINGLE_MODEL_DIR:
        # Direction: 仅保留 0(Short) 和 2(Long)
        mask = (y_raw != 1)
        # 0 -> 0 (Short), 2 -> 1 (Long)
        y_new = torch.where(y_raw[mask] == 2, 1, 0).long()
        return X_raw[mask], y_new, rb_raw[mask]

    elif train_task == TrainTask.SINGLE_MODEL_LONG_OVR:
        # Long OvR: 2 -> 1, others -> 0
        y_new = (y_raw == 2).long()
        return X_raw, y_new, rb_raw

    elif train_task == TrainTask.SINGLE_MODEL_SHORT_OVR:
        # Short OvR: 0 -> 1, others -> 0
        y_new = (y_raw == 0).long()
        return X_raw, y_new, rb_raw
    
    return X_raw, y_raw, rb_raw

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

def compute_soft_binary_loss(logits, y_sub, rb, train_task: TrainTask, train_cfg: TrainConfig, device):
    """
    针对二分类子任务优化的软损失函数。
    支持：TRIGGER(4), DIR(5), LONG_OVR(6), SHORT_OVR(7)
    """
    probs = torch.softmax(logits, dim=1)
    p_neg, p_pos = probs[:, 0], probs[:, 1]
    
    # 1. 基础幅度权重 (行情越大，样本权重越高)
    # 使用 log1p 压缩极值，避免单次大波动主导梯度
    mag_weights = 1.0 + torch.log1p(train_cfg.mag_alpha * torch.abs(rb))
    mag_weights = torch.clamp(mag_weights, max=train_cfg.mag_limit)

    # 2. 任务特定的软化惩罚 (Penalty Logic)
    penalty = torch.zeros_like(p_neg)
    is_pos = (y_sub == 1) # 真实有信号/看多
    is_neg = (y_sub == 0) # 真实无信号/看空

    # 根据枚举类型分发惩罚逻辑
    if train_task == TrainTask.SINGLE_MODEL_TRIGGER:
        # 【触发任务】：核心是防止“踏空”。若真实有信号(1)但预测为无信号(0)的概率高，加重惩罚
        penalty[is_pos] += p_neg[is_pos] * train_cfg.miss_penalty

    elif train_task == TrainTask.SINGLE_MODEL_DIR:
        # 【方向任务】：核心是防止“做反”。因为没有 Neutral，错即是反
        # 真实看多(1)预测看空(0) 或 真实看空(0)预测看多(1) 都要惩罚
        penalty[is_pos] += p_neg[is_pos] * train_cfg.flip_penalty
        penalty[is_neg] += p_pos[is_neg] * train_cfg.flip_penalty

    elif train_task in [TrainTask.SINGLE_MODEL_LONG_OVR, TrainTask.SINGLE_MODEL_SHORT_OVR]:
        # 【OvR 任务】：针对特定趋势的捕捉，惩罚“错过”该特定趋势的情况
        penalty[is_pos] += p_neg[is_pos] * train_cfg.miss_penalty

    # 3. 最终权重融合与归一化
    # 结合行情幅度和错误代价，并确保 Batch 内平均权重为 1
    final_weights = mag_weights * (1.0 + penalty)
    final_weights = final_weights / (final_weights.mean() + 1e-8)

    # 4. 计算带权重的 NLL Loss
    log_probs = torch.log_softmax(logits, dim=1)
    # 从 log_probs 中提取对应真实标签 y_sub 的对数概率
    loss_samples = -log_probs.gather(1, y_sub.unsqueeze(1)).squeeze()

    return (loss_samples * final_weights).mean()

def train_engine_binary(model, dl_tr, dl_va, optimizer, scheduler, device, logger, train_cfg, train_task:TrainTask):
    """
    专门适配二分类的轻量级引擎
    """
    best_val_loss, best_val_f1 = float("inf"), 0.0
    best_state_loss, best_state_f1 = None, None
    wait = 0

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        tr_loss, tr_total = 0.0, 0
        
        with tqdm(dl_tr, desc=f"Binary Ep {epoch}", ncols=100, leave=False) as pbar:
            for xb, yb, rb in pbar:
                xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
                
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                
                # 调用你上传的软损失函数
                loss = compute_soft_binary_loss(logits, yb, rb, train_task, train_cfg, device)
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                tr_loss += loss.item() * xb.size(0)
                tr_total += xb.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- 验证 (简单化处理) ---
        model.eval()
        va_loss_sum, all_preds, all_trues = 0.0, [], []
        with torch.no_grad():
            for xb, yb, rb in dl_va:
                xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
                logits = model(xb)
                v_loss = compute_soft_binary_loss(logits, yb, rb, train_task, train_cfg, device)
                va_loss_sum += v_loss.item() * xb.size(0)
                all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                all_trues.append(yb.cpu().numpy())

        avg_va_loss = va_loss_sum / max(1, len(dl_va.dataset))
        y_true, y_pred = np.concatenate(all_trues), np.concatenate(all_preds)
        va_f1 = f1_score(y_true, y_pred, average="macro")
        
        scheduler.step(avg_va_loss)
        avg_tr_loss = tr_loss / max(1, tr_total)
        logger.info(f"Epoch {epoch:03d} | tr_loss {avg_tr_loss:.4f} | va_loss {avg_va_loss:.4f} | va_macroF1 {va_f1:.4f}")

        # 保存逻辑
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state_f1 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        elif avg_va_loss < best_val_loss:
            best_val_loss = avg_va_loss
            wait = 0
        else:
            wait += 1

        if wait >= train_cfg.patience: break

    return {"best_f1_state": best_state_f1, "f1_score": best_val_f1}

def build_binary_metrics(
    trues_np,
    preds_np,
    probs_np,
    test_loss: float,
    report_dict: dict,
):
    labels = sorted(np.unique(np.concatenate([trues_np, preds_np])))

    cm = confusion_matrix(trues_np, preds_np, labels=labels)

    true_counts = Counter(trues_np)
    pred_counts = Counter(preds_np)
    total = len(trues_np)

    per_class = {}

    for cls in labels:
        cls_str = str(int(cls))

        base_rate = true_counts[cls] / total if total > 0 else 0.0
        pred_ratio = pred_counts[cls] / total if total > 0 else 0.0

        precision = report_dict.get(cls_str, {}).get("precision", 0.0)
        recall = report_dict.get(cls_str, {}).get("recall", 0.0)
        f1 = report_dict.get(cls_str, {}).get("f1-score", 0.0)
        support = report_dict.get(cls_str, {}).get("support", 0)

        precision_lift = precision / base_rate if base_rate > 0 else None
        abs_uplift = precision - base_rate if base_rate > 0 else None

        cls_pred_mask = preds_np == cls
        if cls_pred_mask.any():
            avg_confidence_when_pred = float(np.max(probs_np[cls_pred_mask], axis=1).mean())
            avg_prob_for_class_when_pred = float(probs_np[cls_pred_mask, int(cls)].mean())
        else:
            avg_confidence_when_pred = None
            avg_prob_for_class_when_pred = None

        per_class[cls_str] = {
            "f1": f1,
            "recall": recall,
            "precision": precision,
            "precision_lift": precision_lift,
            "abs_precision_uplift": abs_uplift,
            "avg_confidence_when_pred": avg_confidence_when_pred,
            "avg_prob_for_class_when_pred": avg_prob_for_class_when_pred,
            "base_rate": base_rate,
            "pred_ratio": pred_ratio,
            "true_count": int(true_counts[cls]),
            "pred_count": int(pred_counts[cls]),
            "support": int(support),
        }

    metrics = {
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "test_loss": float(test_loss),
        "accuracy": float(accuracy_score(trues_np, preds_np)),
        "balanced_accuracy": float(balanced_accuracy_score(trues_np, preds_np)),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "mcc": float(matthews_corrcoef(trues_np, preds_np)),
        "sample_count": int(total),
        "labels": [int(x) for x in labels],
        "true_distribution": {
            str(int(k)): {
                "count": int(v),
                "ratio": float(v / total),
            }
            for k, v in sorted(true_counts.items())
        },
        "pred_distribution": {
            str(int(k)): {
                "count": int(v),
                "ratio": float(v / total),
            }
            for k, v in sorted(pred_counts.items())
        },
        "per_class": per_class,
        "confusion_matrix": {
            "labels": [int(x) for x in labels],
            "matrix": cm.tolist(),
        },
    }

    return metrics

def evaluate_and_save_binary_results(
    results: dict,
    model: nn.Module,
    dl_te: DataLoader,
    device: torch.device,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    full_ds: TimeSeriesWindowDataset,
    feature_list: list[str],
    train_task: TrainTask,
    logger: logging.Logger,
    save_dir: str
):
    """
    专用的二分类评估函数：提供深度指标打印、模型存档及解释性分析。
    适配任务：TRIGGER(4), DIR(5), LONG_OVR(6), SHORT_OVR(7)
    """
    # 1. 定义评估任务（Best F1 和 Best Loss 两个版本）
    tasks = [
        ("Best_F1", results.get("best_f1_state"), results.get("f1_score", 0)),
        ("Best_Loss", results.get("best_loss_state"), results.get("loss_score", 0))
    ]

    final_metrics = {}

    for suffix, state, val_score in tasks:
        if state is None: continue
            
        # A. 加载模型与推理
        model.load_state_dict(state)
        model.eval()
        
        all_probs, all_preds, all_trues = [], [], []
        test_loss_sum = 0.0
        
        with torch.no_grad():
            for xb, yb, rb in dl_te:
                xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
                logits = model(xb)
                
                # 计算测试集 Loss (复用之前的二分类软损失)
                loss = compute_soft_binary_loss(logits, yb, rb, train_task, train_cfg, device)
                test_loss_sum += loss.item() * xb.size(0)
                
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                all_trues.append(yb.cpu().numpy())

        probs_np = np.concatenate(all_probs)
        preds_np = np.concatenate(all_preds)
        trues_np = np.concatenate(all_trues)
        test_loss = test_loss_sum / len(dl_te.dataset)

        # B. 计算核心指标
        # 注意：二分类通常关注 Signal(1) 的表现，但打印 macro avg 以对齐风格
        report_dict = classification_report(trues_np, preds_np, output_dict=True, zero_division=0)
        test_f1 = report_dict['macro avg']['f1-score']

        logger.info(f"\n{'='*20} Evaluating Binary Model: {suffix} | Task: {train_task} {'='*20}")
        
        # 使用自定义格式打印报告 (假设你已有 format_custom_report，否则直接打印 report_dict)
        logger.info(format_custom_report(report_dict))
            
        logger.info(f"Test macro-F1: {test_f1:.4f} | Test Loss: {test_loss:.4f}")

        # C. 统计标签比例
        counts = Counter(trues_np)
        total = sum(counts.values())
        prop_str = ", ".join([f"{int(c)}: {counts[c]/total:.2%}" for c in sorted(counts.keys())])
        logger.info(f"[{suffix}] True label proportion (Test set): {prop_str}")

        # D. 保存混淆矩阵 (CSV 格式)
        cm = confusion_matrix(trues_np, preds_np)
        unique_labels = sorted(np.unique(trues_np))
        cm_df = pd.DataFrame(
            cm, 
            index=[f"true_{int(i)}" for i in unique_labels], 
            columns=[f"pred_{int(i)}" for i in unique_labels]
        )
        cm_df.to_csv(os.path.join(save_dir, f"confmat_binary_{suffix.lower()}.csv"))

        # E. 模型保存 (.pt)
        pt_path = os.path.join(save_dir, f"model_{train_task.lower()}_{suffix.lower()}.pt")
        save_state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
        torch.save({
            "state_dict": save_state,
            "feature_list": feature_list,
            "classes": [0, 1],
            "channel": full_ds.feature_count,
            "window": train_cfg.model_cfg.seq_len,
            "feature_cols": full_ds.feature_names,
            "label_col": data_cfg.label_col,
            "val_score": val_score,
            "test_f1": test_f1,
            "train_task": train_task,
            "version_type": suffix
        }, pt_path)
        
        # F. 导出 Meta (.json)
        meta = model.export_meta(
            feature_cols=full_ds.feature_names,
            label_col=data_cfg.label_col,
            classes=[0, 1],
            window=train_cfg.model_cfg.seq_len,
            model_version_tag=suffix,
        )
        meta["binary_task_type"] = train_task # 注入任务标识
        with open(os.path.join(save_dir, f"model_{train_task.lower()}_{suffix.lower()}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        final_metrics[suffix] = build_binary_metrics(
                                    trues_np=trues_np,
                                    preds_np=preds_np,
                                    probs_np=probs_np,
                                    test_loss=test_loss,
                                    report_dict=report_dict,
                                )
        # G. 特征解释性分析 (仅在配置开启时)
        if getattr(train_cfg.model_cfg, 'use_feature_weighting', False) or getattr(train_cfg.model_cfg, 'use_feature_selector', False):
            try:
                logger.info(f"🎨 Generating Interpretability Analysis for {suffix}...")
                test_xb, test_yb, test_rb = next(iter(dl_te))
                
                # 1. 全局重要性
                importance_df = analyze_feature_importance(
                    model=model, batch_x=test_xb, 
                    feature_names=full_ds.feature_names, device=device
                )
                importance_df.to_csv(os.path.join(save_dir, f"feat_imp_{train_task.lower()}_{suffix.lower()}.csv"), index=False)

                # 2. 样本级贡献 (可视化 0/1 类别的样本)
                analyze_sample_contribution_by_class(
                    model=model, batch_x=test_xb, batch_y=test_yb, batch_r=test_rb,
                    feature_names=full_ds.feature_names, device=device,
                    top_k_per_class=3, save_dir=save_dir, suffix=f"{train_task}_{suffix}"
                )
            except Exception as e:
                logger.error(f"⚠️ Interpretability analysis failed: {e}")

    # H. 生成 Task Description 索引
    primary_suffix = "Best_F1" if getattr(train_cfg, 'best_f1', True) else "Best_Loss"
    task_desc = {
        "task_type": train_task,
        "timestamp": pd.Timestamp.now().isoformat(),
        "models": {
            "main": {
                "model": f"model_{train_task.lower()}_{primary_suffix.lower()}.pt",
                "meta": f"model_{train_task.lower()}_{primary_suffix.lower()}_meta.json"
            }
        }
    }
    
    with open(os.path.join(save_dir, "task_description.json"), "w", encoding="utf-8") as f:
        json.dump(task_desc, f, ensure_ascii=False, indent=2)
        
    logger.info(f"🎉 Binary task evaluation complete. Artifacts saved to: {save_dir}")
    return final_metrics

def diagnose_confidence_binary(results, model, dl_te, device, train_task, logger, save_dir):
    """
    专门适配二分类任务的置信度诊断函数。
    支持：TRIGGER, DIR, LONG_OVR, SHORT_OVR
    """
    model.eval()
    versions = [
        ("Best_F1", results.get("best_f1_state")),
        ("Best_Loss", results.get("best_loss_state"))
    ]

    # 根据任务类型定义标签
    if train_task == TrainTask.SINGLE_MODEL_DIR:
        class_names = {0: "Short (0)", 1: "Long (1)"}
        colors = {0: "#26a69a", 1: "#ef5350"} # 绿/红
    else:
        class_names = {0: "Stay/Neg (0)", 1: "Signal/Pos (1)"}
        colors = {0: "#78909c", 1: "#2196F3"} # 灰/蓝

    plt.figure(figsize=(18, 8))

    for i, (suffix, state) in enumerate(versions):
        if state is None: continue
        model.load_state_dict(state)
        
        all_probs, all_preds, all_trues = [], [], []

        with torch.no_grad():
            for xb, yb, _ in dl_te:
                xb = xb.to(device)
                logits = model(xb) 
                probs = torch.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                all_trues.append(yb.cpu().numpy())

        probs_np = np.concatenate(all_probs)
        preds_np = np.concatenate(all_preds)
        trues_np = np.concatenate(all_trues)

        # 二分类置信度通常指预测类别的概率 (0.5 ~ 1.0)
        confidences = np.max(probs_np, axis=1)
        is_correct = (preds_np == trues_np)

        # 分箱设置：从 0.5 到 1.0
        bins = np.linspace(0.5, 1.0, 11)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax1 = plt.subplot(1, 2, i+1)
        
        # 1. 绘制背景：样本分布 (左轴)
        bin_counts = [np.sum(bin_indices == b) for b in range(len(bins)-1)]
        ax1.bar(bin_centers, bin_counts, width=(bins[1]-bins[0])*0.8, 
                color='#f5f5f5', label='Sample Density', edgecolor='#e0e0e0')
        ax1.set_xlabel('Confidence (Max Probability)', fontsize=12)
        ax1.set_ylabel('Sample Count', color='#9e9e9e')
        ax1.tick_params(axis='y', labelcolor='#9e9e9e')

        # 2. 绘制前景：准确率曲线 (右轴)
        ax2 = ax1.twinx()
        ax2.plot([0.5, 1.0], [0.5, 1.0], '--', color='#bdbdbd', alpha=0.8, label='Perfectly Calibrated')

        for cls_val in [0, 1]:
            cls_bin_accs = []
            for b in range(len(bins) - 1):
                # 筛选条件：置信度区间 + 真实标签
                mask = (bin_indices == b) & (trues_np == cls_val)
                count = np.sum(mask)
                if count > 3: # 二分类样本集中度可能不同，阈值设为3
                    acc = np.mean(is_correct[mask])
                    cls_bin_accs.append(acc)
                else:
                    cls_bin_accs.append(None)

            valid_x = [bin_centers[idx] for idx, val in enumerate(cls_bin_accs) if val is not None]
            valid_y = [val for val in cls_bin_accs if val is not None]
            
            ax2.plot(valid_x, valid_y, marker='o', markersize=5, 
                     linewidth=2, color=colors[cls_val], label=f"Acc: {class_names[cls_val]}")

        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_ylim(0, 1.05)
        
        plt.title(f"{train_task} Reliability: {suffix}", fontsize=14)
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        ax1.grid(axis='y', linestyle=':', alpha=0.3)

    plot_path = os.path.join(save_dir, f"diagnose_{train_task.lower()}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"📊 Binary reliability plot saved to: {plot_path}")
    plt.close()

def run_single_model_binary_task(train_task:TrainTask, full_ds, feature_list,train_cfg: TrainConfig, data_cfg, device, logger, save_dir, experiment):
    """
    全流程二分类 Runner：数据映射 -> 重切分 -> 独立训练 -> 评估
    """
    # 1. 标签映射与样本过滤
    X_t, y_t, rb_t = prepare_binary_data_for_task(full_ds.X, full_ds.y, full_ds.returns, train_task)
    
    # 2. 重新进行时间序列切分
    M = len(y_t)
    tr_rng, va_rng, te_rng = chrono_split_by_window_ends(M, data_cfg.train_ratio, data_cfg.val_ratio)
    
    ds_tr = SeqDataset(X_t[tr_rng[0]:tr_rng[1]], y_t[tr_rng[0]:tr_rng[1]], rb_t[tr_rng[0]:tr_rng[1]])
    ds_va = SeqDataset(X_t[va_rng[0]:va_rng[1]], y_t[va_rng[0]:va_rng[1]], rb_t[va_rng[0]:va_rng[1]])
    ds_te = SeqDataset(X_t[te_rng[0]:te_rng[1]], y_t[te_rng[0]:te_rng[1]], rb_t[te_rng[0]:te_rng[1]])

    # 3. 专用采样器
    sampler_tr = get_trigger_sampler(ds_tr.labels, pos_ratio=0.3)
    dl_tr = DataLoader(ds_tr, batch_size=train_cfg.batch_size, sampler=sampler_tr, shuffle=False)
    dl_va = DataLoader(ds_va, batch_size=train_cfg.batch_size, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=train_cfg.batch_size, shuffle=False)

    # 4. 构建模型 (强制 n_classes=2)
    model = ModelFactory.build_for_training(
        device=device, input_size=full_ds.feature_count, n_classes=2, max_len = train_cfg.model_cfg.seq_len,
        window = train_cfg.model_cfg.seq_len,
        **asdict(train_cfg.model_cfg)
    )

    # 5. 调用独立引擎
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5)

    results = train_engine_binary(model, dl_tr, dl_va, optimizer, scheduler, device, logger, train_cfg, train_task)

    # 6. 评估与保存
    os.makedirs(save_dir, exist_ok=True)
    final_metrics = evaluate_and_save_binary_results(results=results,model=model,dl_te=dl_te,device=device,data_cfg=data_cfg,train_cfg=train_cfg,
        full_ds=full_ds,feature_list=feature_list, train_task=train_task, logger=logger,save_dir=save_dir)
    
    if experiment == False:
        diagnose_confidence_binary(results, model, dl_te, device, train_task, logger, save_dir)
    return final_metrics

def run_training(train_task:TrainTask,feature_direction_map, logger: logging, data_cfg: DataConfig, train_cfg_1: TrainConfig, train_cfg_2: TrainConfig , prep_output_dir:str, save_dir:str,experiment:bool):
    # 0. Initialize environment
    set_seed(train_cfg_1.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"train_task:{train_task} | Device: {device} | Model: {train_cfg_1.model_cfg.model_type} version: {train_cfg_1.model_cfg.model_version}")
    if train_cfg_2:
        logger.info(f"train_task:{train_task} | Device: {device} | Model: {train_cfg_2.model_cfg.model_type} version: {train_cfg_2.model_cfg.model_version}")
    if device.type == 'cuda':
        # Enable TensorFloat32 (TF32): 5090 throughput improves significantly
        torch.set_float32_matmul_precision('high')

    df = common.load_train_df_from_dir(prep_output_dir)
    para = common.load_interval_ms_from_dir(prep_output_dir)
    kline_interval_ms = common.get_interval_ms(para.interval)
    logger.info(f"Using TimeSeriesWindowDataset with window={train_cfg_1.model_cfg.seq_len} Origin data len {len(df)}...")

    feature_list = list(feature_direction_map.keys())
    full_ds = TimeSeriesWindowDataset(
        df=df, kline_interval_ms=kline_interval_ms, feature_cols=feature_list, label_col=data_cfg.label_col, window=train_cfg_1.model_cfg.seq_len,
        cache_path=os.path.join(save_dir,"train_cache.pt"), stride =train_cfg_1.stride, use_cache = train_cfg_1.use_cache, show_feature_distribution=True
    )
    logger.warning(f"📊 [Dataset Check] Final features used in training ({full_ds.feature_count}):"
                   f"{full_ds.feature_names}")
    x_mem = full_ds.X.element_size() * full_ds.X.nelement() / (1024**2)
    y_mem = full_ds.y.element_size() * full_ds.y.nelement() / (1024**2)
    r_mem = full_ds.returns.element_size() * full_ds.returns.nelement() / (1024**2)

    total_gpu_mem_per_process = x_mem + y_mem + r_mem
    logger.info(f"🚀 Estimated GPU VRAM per process: {total_gpu_mem_per_process:.2f} MB")
    # Flip ic_direction=-1 features (multiply by -1) to make them positively correlated with returns
    if feature_direction_map:
        full_ds.X = apply_feature_direction(full_ds.X, full_ds.feature_names, feature_direction_map, logger)

    # VRAM preload optimization
    logger.info(f"Pre-loading entire dataset to {device}...")
    full_ds.X, full_ds.y, full_ds.returns = full_ds.X.to(device), full_ds.y.to(device), full_ds.returns.to(device)

    M = len(full_ds)
    logger.info(f"Total windows (M) = {M}, window = {train_cfg_1.model_cfg.seq_len}")

    # 2. Split data
    tr_rng, va_rng, te_rng = chrono_split_by_window_ends(M, data_cfg.train_ratio, data_cfg.val_ratio)
    
    ds_tr = SeqDataset(full_ds.X[tr_rng[0]:tr_rng[1]], full_ds.y[tr_rng[0]:tr_rng[1]], full_ds.returns[tr_rng[0]:tr_rng[1]])
    # More efficient alternative: just pass the sliced tensors
    ds_va = SeqDataset(full_ds.X[va_rng[0]:va_rng[1]], full_ds.y[va_rng[0]:va_rng[1]], full_ds.returns[va_rng[0]:va_rng[1]])
    ds_te = SeqDataset(full_ds.X[te_rng[0]:te_rng[1]], full_ds.y[te_rng[0]:te_rng[1]], full_ds.returns[te_rng[0]:te_rng[1]])

    X_raw, y_raw, rb_raw = full_ds.X, full_ds.y, full_ds.returns
    if train_task == TrainTask.SINGLE_MODEL_3CLASS:
        return run_single_model_3class_task(
            train_task, ds_tr, ds_va, ds_te, full_ds, feature_list,
            train_cfg_1, data_cfg, device, logger, save_dir, experiment
        )
    elif train_task in [TrainTask.SINGLE_MODEL_TRIGGER, TrainTask.SINGLE_MODEL_DIR, 
                        TrainTask.SINGLE_MODEL_LONG_OVR, TrainTask.SINGLE_MODEL_SHORT_OVR]:
        # 使用新封装的二分类逻辑
        return run_single_model_binary_task(train_task, full_ds, feature_list,train_cfg_1, data_cfg, device, logger, save_dir, experiment)
    elif train_task == TrainTask.TRIGGER_DIR:
        tri_save_dir = os.path.join(save_dir, TrainTask.SINGLE_MODEL_TRIGGER)
        run_single_model_binary_task(TrainTask.SINGLE_MODEL_TRIGGER, full_ds, feature_list,train_cfg_1, data_cfg, device, logger, tri_save_dir, experiment)
        dir_save_dir = os.path.join(save_dir, TrainTask.SINGLE_MODEL_DIR)
        run_single_model_binary_task(TrainTask.SINGLE_MODEL_DIR, full_ds, feature_list,train_cfg_2, data_cfg, device, logger, dir_save_dir, experiment)

    else:
        raise ValueError(f"Unsupported train_task: {train_task}")

def run_single_model_3class_task(
    train_task:TrainTask, ds_tr, ds_va, ds_te, 
    full_ds, feature_list,
    train_cfg, data_cfg, 
    device, logger, save_dir, experiment
):
    """
    专门负责 3分类单模型 任务的训练全流程
    """
    # 1. 计算权重与采样器 (任务相关)
    y_tr_np = ds_tr.y.cpu().numpy()
    sampler_tr = get_balanced_sampler(ds_tr) 
    classes = np.unique(y_tr_np)
    cw_balanced = compute_class_weight("balanced", classes=classes, y=y_tr_np)
    class_weights = torch.tensor(cw_balanced, dtype=torch.float32, device=device)
    logger.info(f"Class weights: {dict(zip(classes, cw_balanced))}")

    # 2. 构建 DataLoader
    dl_tr = DataLoader(ds_tr, batch_size=train_cfg.batch_size, sampler=sampler_tr, shuffle=False, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=train_cfg.batch_size, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=train_cfg.batch_size, shuffle=False)

    # 3. 构建模型
    params = asdict(train_cfg.model_cfg)

    model = ModelFactory.build_for_training(
        max_len =train_cfg.model_cfg.seq_len ,device=device,input_size=full_ds.feature_count, n_classes=len(classes),
        window = train_cfg.model_cfg.seq_len,
        **params
    )

    # 4. 差异化学习率配置
    gate_params = [p for n, p in model.named_parameters() if "feature_weighter" in n]
    backbone_params = [p for n, p in model.named_parameters() if "feature_weighter" not in n]
    
    param_groups = [{"params": backbone_params, "lr": train_cfg.lr}]
    if gate_params:
        logger.info(f"⚡ [Differential LR] Setting gate_lr: {train_cfg.gate_lr} for feature_weighter")
        param_groups.append({"params": gate_params, "lr": train_cfg.gate_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    # 5. 执行训练
    logger.info("🚀 Starting training engine...")
    mtl = MTLManager(device, train_cfg)
    results = train_engine(
        model=model, dl_tr=dl_tr, dl_va=dl_va,
        optimizer=optimizer, scheduler=scheduler,
        device=device, logger=logger, train_cfg=train_cfg, mtl_manager=mtl
    )

    # 6. 评估与诊断
    final_metrics = evaluate_and_save_results(
        train_task, results=results, model=model, dl_te=dl_te,
        device=device, data_cfg=data_cfg, train_cfg=train_cfg,
        full_ds=full_ds, feature_list=feature_list,
        classes=classes, logger=logger, mtl_manager=mtl, save_dir=save_dir
    )

    if not experiment:
        diagnose_confidence(results, model=model, dl_te=dl_te, device=device, logger=logger, save_dir=save_dir)

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
    def __init__(self, X, y, returns):
        self.X = X if torch.is_tensor(X) else torch.from_numpy(X).float()
        self.y = y if torch.is_tensor(y) else torch.from_numpy(y).long()
        self.r = returns if torch.is_tensor(returns) else torch.from_numpy(returns).float()
        self.labels = self.y.cpu().numpy() # 采样器统计仍需在 CPU

    def __len__(self): 
        return self.X.shape[0]

    def __getitem__(self, i): 
        return self.X[i], self.y[i], self.r[i] #  返回三元组

class MTLManager:
    def __init__(self, device, train_cfg:TrainConfig):
        self.device = device
        self.cfg = train_cfg
        self.weights = {}
        self.criteria = {}
        #  新增：对称性惩罚权重，初始建议设为 0.5 ~ 1.0

    def prepare_all(self, y_raw):
        """一次性初始化所有权重和 Loss 函数"""
        # 1. 计算权重
        y_trig = (y_raw != 1).astype(int)
        mask_dir = (y_raw != 1)
        y_dir = np.where(y_raw[mask_dir] == 2, 1, 0)

        cw_main = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_raw)
        cw_trig = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_trig)
        cw_dir = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_dir)
        
        self.weights = {
            'main': torch.tensor(cw_main, dtype=torch.float32, device=self.device),
            'trig': torch.tensor(cw_trig, dtype=torch.float32, device=self.device),
            'dir': torch.tensor(cw_dir, dtype=torch.float32, device=self.device)
        }

        # 2. 初始化 Criteria
        self.criteria = {
            'main': nn.NLLLoss(weight=self.weights['main']),
            'trig': nn.CrossEntropyLoss(weight=self.weights['trig']),
            'dir': nn.CrossEntropyLoss(weight=self.weights['dir'])
        }
        return cw_main, cw_trig, cw_dir

    @staticmethod
    def get_targets(yb):
        """统一标签转换逻辑"""
        target_trig = (yb != 1).long()
        target_dir = torch.where(yb == 2, 1, 0).long()
        action_mask = (yb != 1)
        return target_trig, target_dir, action_mask

    def compute_combined_loss(self, logits_trig, logits_dir, yb, rb, epoch):
        if self.cfg.loss_fun_version == 1:
            return self.compute_combined_loss_v1(logits_trig, logits_dir, yb, rb, epoch)
        elif self.cfg.loss_fun_version == 2:
            return self.compute_combined_loss_v2(logits_trig, logits_dir, yb, rb, epoch)
        elif self.cfg.loss_fun_version == 3:
            return self.compute_combined_loss_v3(logits_trig, logits_dir, yb, rb, epoch)
        elif self.cfg.loss_fun_version == 4:
            return self.compute_combined_loss_v4(logits_trig, logits_dir, yb, rb, epoch)

    def compute_combined_loss_v1(self, logits_trig, logits_dir, yb, rb, epoch: int = 0):
        """
        全功能量化版 Loss：整合了非对称距离惩罚、能量归一化与对称性约束。
        
        flip_penalty: 致命错误惩罚 (0 <-> 2)
        miss_penalty: 保守错误惩罚 (0/2 -> 1)
        """
        t_trig, t_dir, act_mask = self.get_targets(yb)
        
        # 1. 基础子任务 Loss
        loss_trig = self.criteria['trig'](logits_trig, t_trig)
        loss_dir = torch.tensor(0.0, device=self.device)
        if act_mask.any():
            loss_dir = self.criteria['dir'](logits_dir[act_mask], t_dir[act_mask])
        
        # 2. 融合 3-Class 概率分布 [Short(0), Neutral(1), Long(2)]
        p_trig = torch.softmax(logits_trig, dim=1)
        p_dir = torch.softmax(logits_dir, dim=1)
        fused_probs = torch.stack([
            p_trig[:, 1] * p_dir[:, 0], # Short
            p_trig[:, 0],               # Neutral
            p_trig[:, 1] * p_dir[:, 1]  # Long
        ], dim=1)
        
        # 3. 计算逐样本基础 Main Loss (NLL)
        # 使用 1e-10 防止 log(0) 崩溃
        sample_main_loss = F.nll_loss(torch.log(fused_probs + 1e-10), yb, weight=self.weights['main'], reduction='none')

        # --- 🌟 逻辑 A: 三级非对称距离惩罚 ---
        penalty = torch.ones_like(sample_main_loss)
        preds = torch.argmax(fused_probs, dim=1)
        trend_mask = (yb != 1) # 真实标签为趋势的样本

        if trend_mask.any():
            # 1. 致命错误：多空做反 (例如 y=2, pred=0)
            fatal_mask = trend_mask & (preds != yb) & (preds != 1)
            penalty[fatal_mask] = self.cfg.flip_penalty
            
            # 2. 保守错误：趋势预测成了震荡 (例如 y=2, pred=1)
            # 增加此惩罚，防止模型逃向震荡类，迫使其在趋势中寻找微弱信号
            miss_mask = trend_mask & (preds == 1)
            penalty[miss_mask] = self.cfg.miss_penalty

            penalty = penalty / (penalty.mean() + 1e-8)

        loss_main = (sample_main_loss * penalty).mean()

        avg_p_short = fused_probs[:, 0].mean()
        avg_p_long = fused_probs[:, 2].mean()
        bias_loss = torch.zeros((), device=logits_dir.device)

        act = (p_trig[:, 1] > 0.5)   # 或 act = (yb != 1)
        if act.any():
            bias_loss = torch.abs(
                p_dir[act, 0].mean() - p_dir[act, 1].mean()
            )

        bias_lambda = getattr(self.cfg, 'bias_lambda', 0.5)
        total_loss = loss_main + \
                    (self.cfg.lambda_trig * loss_trig) + \
                    (self.cfg.lambda_dir * loss_dir) + \
                    (bias_lambda * bias_loss)
        
        return total_loss, loss_main, loss_trig, loss_dir, self.cfg.lambda_dir
        
    def compute_combined_loss_v2(self, logits_trig, logits_dir, yb, rb, epoch: int = 0):
        t_trig, t_dir, act_mask = self.get_targets(yb)

        # --- 2. 软化子任务 Loss (Sub-task Soft Loss) ---
        # 计算基础子任务
        loss_trig_raw = F.cross_entropy(logits_trig, t_trig, weight=self.weights['trig'], reduction='none')
        loss_trig = loss_trig_raw.mean()

        loss_dir = torch.tensor(0.0, device=self.device)
        if act_mask.any():
            loss_dir_raw = F.cross_entropy(logits_dir[act_mask], t_dir[act_mask], weight=self.weights['dir'], reduction='none')
            loss_dir = loss_dir_raw.mean()

        # --- 3. 融合 3-Class 概率 ---
        p_trig = torch.softmax(logits_trig, dim=1)
        p_dir = torch.softmax(logits_dir, dim=1)
        # [B, 3] 顺序: 0:Short, 1:Neutral, 2:Long
        fused_probs = torch.stack([
            p_trig[:, 1] * p_dir[:, 0], 
            p_trig[:, 0],               
            p_trig[:, 1] * p_dir[:, 1]  
        ], dim=1)

        # --- 4. 核心：软化非对称惩罚 (Soft Penalty) ---
        # 不再使用 argmax，而是根据“分错方向的概率”来计算惩罚
        p_short, p_neutral, p_long = fused_probs[:, 0], fused_probs[:, 1], fused_probs[:, 2]
        
        # 定义惩罚项 penalty = 1.0 + 额外开销
        soft_penalty = torch.ones_like(p_neutral)
        
        # 真实标签为趋势 (Long/Short) 时的惩罚逻辑
        is_long = (yb == 2)
        is_short = (yb == 0)
        
        if is_long.any():
            # Long 样本的风险：预测为 Short 的概率 (Fatal) + 预测为 Neutral 的概率 (Miss)
            soft_penalty[is_long] += (p_short[is_long] * self.cfg.flip_penalty + 
                                      p_neutral[is_long] * self.cfg.miss_penalty)
            
        if is_short.any():
            # Short 样本的风险：预测为 Long 的概率 (Fatal) + 预测为 Neutral 的概率 (Miss)
            soft_penalty[is_short] += (p_long[is_short] * self.cfg.flip_penalty + 
                                       p_neutral[is_short] * self.cfg.miss_penalty)

        # --- 5. 🚀 改进的耦合惩罚 (Additive Coupling) ---
        # 只有在“真正的方向错误概率”较大时才施加额外回报加权，且改乘法为加法
        soft_flip_prob = torch.where(is_long, p_short, torch.where(is_short, p_long, 0.0))
        coupling_weight = soft_flip_prob * torch.abs(rb) * self.cfg.mag_alpha

        # --- 6. 最终复合权重归一化 ---
        # 复合权重 = 错误倾向惩罚 + 回报耦合
        combined_weights = soft_penalty + coupling_weight # 加法结构
        combined_weights = combined_weights / (combined_weights.mean() + 1e-8) # 维持梯度预算

        sample_main_loss = F.nll_loss(torch.log(fused_probs + 1e-10), yb, weight=self.weights['main'], reduction='none')
        loss_main = (sample_main_loss * combined_weights).mean()

        # --- 7. 对称性正则 (保持不变) ---
        bias_loss = torch.zeros((), device=logits_trig.device)
        act_p = (p_trig[:, 1] > 0.5)
        if act_p.any():
            bias_loss = torch.abs(p_dir[act_p, 0].mean() - p_dir[act_p, 1].mean())

        total_loss = loss_main + \
                    (self.cfg.lambda_trig * loss_trig) + \
                    (self.cfg.lambda_dir * loss_dir) + \
                    (getattr(self.cfg, 'bias_lambda', 0.5) * bias_loss)
        
        return total_loss, loss_main, loss_trig, loss_dir, self.cfg.lambda_dir

    def compute_combined_loss_v3(self, logits_trig, logits_dir, yb, rb, epoch: int = 0):
            """
            v4: Financial PnL-Aware Robust Loss
            - Core: Cross Entropy (Classification) + Expected PnL (Regression-like)
            - Robustness: Log-compressed returns for sample weighting
            - Semantics: Explicit penalty for "Wrong Direction" vs "Missing Out"
            """
            eps = 1e-8
            device = self.device

            # ---- 0) Data Prep ----
            # t_trig: 0=Neutral, 1=Action
            # t_dir:  0=Short,   1=Long
            t_trig, t_dir, act_mask = self.get_targets(yb)
            
            # Isolate Long/Short/Neutral masks for easier logic later
            is_short   = (yb == 0)
            is_neutral = (yb == 1) # Assuming 1 is Neutral in your idx map? Check your map!
                                # Based on your v3 code: fused[0]=Short, [1]=Neutral, [2]=Long
                                # usually implies yb=0->Short, yb=1->Neutral, yb=2->Long
            is_long    = (yb == 2)

            # ---- 1) Probabilities (The View) ----
            p_trig = torch.softmax(logits_trig, dim=1) # [B, 2]
            p_dir  = torch.softmax(logits_dir,  dim=1) # [B, 2]

            # Fused Probabilities: [P_Short, P_Neutral, P_Long]
            # P(Short)   = P(Action) * P(Short|Action)
            # P(Neutral) = P(No_Action)
            # P(Long)    = P(Action) * P(Long|Action)
            p_short   = p_trig[:, 1] * p_dir[:, 0]
            p_neutral = p_trig[:, 0]
            p_long    = p_trig[:, 1] * p_dir[:, 1]
            
            fused_probs = torch.stack([p_short, p_neutral, p_long], dim=1)

            # ---- 2) Sample Weighting (Magnitude Only) ----
            # 只保留幅度作为基础权重，不再混入复杂的 penalty，保证梯度稳定
            # Log-compression ensures robustness against extreme volatility ticks
            mag_weights = torch.log1p(torch.abs(rb) * 10.0) + 1.0 
            # Normalize weights in batch to keep learning rate scale stable
            batch_weights = mag_weights / (mag_weights.mean() + eps)

            # ---- 3) Base Classification Loss (NLL) ----
            # Standard CrossEntropy, weighted by market volatility (magnitude)
            nll_loss = F.nll_loss(
                torch.log(fused_probs + eps), 
                yb, 
                reduction='none'
            )
            loss_cls = (nll_loss * batch_weights).mean()

            # ---- 4) Financial Semantic Loss (The "Soul") ----
            
            # A. Expected PnL Proxy (Diff. Sharpe-like)
            # 我们希望：如果是 Long，P_long 越大越好；如果是 Short，P_short 越大越好
            # 这是一个直接的收益最大化项。
            # define semantic sign: Short=-1, Neutral=0, Long=1
            pred_signal = p_long - p_short  # range [-1, 1]
            true_signal = torch.zeros_like(rb)
            true_signal[is_long]  = 1.0
            true_signal[is_short] = -1.0
            
            # PnL Capture: maximize (pred_signal * raw_return)
            # Minimize: - (pred_signal * sign(rb) * |rb|_compressed)
            # We use sign(rb) because yb labels might differ from raw rb sign (due to thresholds)
            # but generally yb aligns with rb. Let's align with LABEL (yb).
            
            pnl_proxy = torch.zeros_like(rb)
            pnl_proxy[is_long]  = -1.0 * p_long[is_long]  * batch_weights[is_long]
            pnl_proxy[is_short] = -1.0 * p_short[is_short] * batch_weights[is_short]
            # Neutral: we want to minimize exposure (abs(pred_signal))
            pnl_proxy[is_neutral] = torch.abs(pred_signal[is_neutral]) * 0.5 * batch_weights[is_neutral]
            
            loss_pnl = pnl_proxy.mean()

            # B. Risk Penalty (Flip & Miss) - Additive!
            # Instead of weighting NLL, we add a specific cost for specific errors.
            risk_cost = torch.zeros_like(nll_loss)
            
            flip_penalty = self.cfg.flip_penalty # e.g., 2.0
            miss_penalty = self.cfg.miss_penalty # e.g., 1.0

            # Case 1: True is Long (2)
            if is_long.any():
                # Error: Predicting Short (Flip) -> Fatal
                risk_cost[is_long] += flip_penalty * (p_short[is_long] ** 2)
                # Error: Predicting Neutral (Miss) -> Regret
                risk_cost[is_long] += miss_penalty * (p_neutral[is_long] ** 2)

            # Case 2: True is Short (0)
            if is_short.any():
                # Error: Predicting Long (Flip) -> Fatal
                risk_cost[is_short] += flip_penalty * (p_long[is_short] ** 2)
                # Error: Predicting Neutral (Miss) -> Regret
                risk_cost[is_short] += miss_penalty * (p_neutral[is_short] ** 2)
                
            # Case 3: True is Neutral (1)
            if is_neutral.any():
                # Error: Predicting Action -> False Alarm (Commission loss)
                # Usually less severe than Flip, similar to Miss
                risk_cost[is_neutral] += miss_penalty * (p_long[is_neutral]**2 + p_short[is_neutral]**2)

            loss_risk = (risk_cost * batch_weights).mean()

            # ---- 5) Bias Regularization (Optional) ----
            # Keep your original logic, it was good.
            bias_loss = torch.tensor(0.0, device=device)
            if self.cfg.bias_lambda > 0:
                conf = p_trig[:, 1]
                w_conf = conf / (conf.mean() + eps)
                bias_loss = torch.abs((w_conf * p_dir[:, 0]).mean() - (w_conf * p_dir[:, 1]).mean())

            # ---- 6) Total Loss Combination ----
            # alpha, beta, gamma to control the mix
            w_cls  = 1.0
            w_pnl  = getattr(self.cfg, "lambda_pnl", 0.5)  # Try 0.1 ~ 1.0
            w_risk = getattr(self.cfg, "lambda_risk", 1.0) # Try 0.5 ~ 2.0

            total_loss = (w_cls * loss_cls) + (w_pnl * loss_pnl) + (w_risk * loss_risk) + (self.cfg.bias_lambda * bias_loss)

            return total_loss, loss_cls, loss_pnl, loss_risk, bias_loss
                                
    def compute_combined_loss_v4(self, logits_trig, logits_dir, yb, trend_strength, epoch: int = 0):
        """
        Simplified v4 (keep core trading-aware parts):
        - trigger CE
        - direction CE (only on action samples, strength-weighted)
        - fused 3-class NLL (log-space, stable)
        - expected cost term (optional strength-weighted on action samples)

        yb: 0=Short, 1=Neutral, 2=Long
        trend_strength: |return| / threshold, >=0; usually >=1 for action samples; may be NaN on invalid rows
        """
        eps = 1e-8
        device = logits_trig.device
        dtype = logits_trig.dtype

        # ---- (0) filter invalid labels and invalid strength (NaN) ----
        valid_mask = (yb >= 0) & (yb <= 2)
        if trend_strength is not None:
            valid_mask = valid_mask & torch.isfinite(trend_strength)

        if not valid_mask.all():
            logits_trig = logits_trig[valid_mask]
            logits_dir  = logits_dir[valid_mask]
            yb          = yb[valid_mask]
            trend_strength = trend_strength[valid_mask]

        # ---- (1) targets for subheads ----
        t_trig, t_dir, act_mask = self.get_targets(yb)  # act_mask: yb!=1

        label_smoothing = float(getattr(self.cfg, "label_smoothing", 0.02))

        # ---- (2) trigger CE ----
        loss_trig = F.cross_entropy(
            logits_trig, t_trig,
            weight=self.weights.get("trig", None),
            label_smoothing=label_smoothing
        )

        # ---- (3) direction CE (action-only) + strength weight ----
        loss_dir = torch.zeros((), device=device, dtype=dtype)
        if act_mask.any():
            ce_dir = F.cross_entropy(
                logits_dir[act_mask], t_dir[act_mask],
                weight=self.weights.get("dir", None),
                reduction="none",
                label_smoothing=label_smoothing
            )

            # trend_strength is >=0, and for action samples usually >=1
            # use max(trend_strength,1) to ensure action weights start at 1
            s = trend_strength[act_mask].clamp(min=1.0)
            w = 1.0 + self.cfg.mag_alpha*torch.log1p(s - 1.0)          # >=1, safe
            w = w / (w.mean().detach() + eps)
            w = w.clamp(max=3.0)

            loss_dir = (ce_dir * w).mean()

        # ---- (4) fused 3-class NLL (log-space stable) ----
        lt = F.log_softmax(logits_trig, dim=1)  # [B,2]
        ld = F.log_softmax(logits_dir,  dim=1)  # [B,2]

        logp_short   = lt[:, 1] + ld[:, 0]
        logp_neutral = lt[:, 0]
        logp_long    = lt[:, 1] + ld[:, 1]
        logp_fused   = torch.stack([logp_short, logp_neutral, logp_long], dim=1)  # [B,3]

        nll_main = F.nll_loss(logp_fused, yb, reduction="mean")
        fused_probs = logp_fused.exp()  # for cost

        # ---- (5) expected cost (trading semantics) ----\
        
        flip = float(self.cfg.flip_penalty) 
        miss = float(self.cfg.miss_penalty)
        false_trade = float(self.cfg.false_trade)

        C = torch.tensor([
            [0.0,   miss,  flip],        # true short
            [false_trade, 0.0, false_trade],  # true neutral
            [flip,  miss,  0.0],         # true long
        ], device=device, dtype=dtype)

        exp_cost = (C[yb] * fused_probs).sum(dim=1)  # [B]

        # optional: mild strength weight ONLY on action samples
        if float(getattr(self.cfg, "cost_use_strength", 1.0)) > 0:
            s_all = torch.where(act_mask, trend_strength.clamp(min=1.0) - 1.0, torch.zeros_like(trend_strength))
            w_cost = 1.0 + self.cfg.mag_alpha * torch.log1p(s_all)   # safe (>=1)
            w_cost = w_cost / (w_cost.mean() + eps)
            loss_cost = (exp_cost * w_cost).mean()
        else:
            loss_cost = exp_cost.mean()

        # ---- (6) total ----
        lam_dir  = self.cfg.lambda_dir
        lam_main = self.cfg.lambda_main
        lam_cost = self.cfg.lambda_cost

        total_loss = loss_trig + lam_dir * loss_dir + lam_main * nll_main + lam_cost * loss_cost
        return total_loss, loss_trig, loss_dir, nll_main, loss_cost
        
class MTLLossTracker:
    def __init__(self, alpha=0.9):
        self.alpha = alpha  # 用于平滑损耗的动量系数
        self.reset()

    def reset(self):
        self.history = {"main": [], "trig": [], "dir": [], "total": []}
        self.smoothed = {"main": 0, "trig": 0, "dir": 0}
        self.steps = 0

    def update(self, l_main, l_trig, l_dir, lam_trig, lam_dir):
        """
        更新 Loss 状态并计算各个部分的贡献比。
        """
        # 转换为标量
        m, t, d = l_main.item(), l_trig.item(), l_dir.item()
        
        # 计算加权后的实际 Loss
        w_t = t * lam_trig
        w_d = d * lam_dir
        total = m + w_t + w_d

        if self.steps == 0:
            self.smoothed["main"], self.smoothed["trig"], self.smoothed["dir"] = m, w_t, w_d
        else:
            self.smoothed["main"] = self.alpha * self.smoothed["main"] + (1 - self.alpha) * m
            self.smoothed["trig"] = self.alpha * self.smoothed["trig"] + (1 - self.alpha) * w_t
            self.smoothed["dir"] = self.alpha * self.smoothed["dir"] + (1 - self.alpha) * w_d

        self.history["main"].append(m)
        self.history["trig"].append(w_t)
        self.history["dir"].append(w_d)
        self.history["total"].append(total)
        self.steps += 1

    def get_ratios(self):
        """计算平滑后的贡献占比"""
        sum_val = sum(self.smoothed.values()) + 1e-10
        return {k: v / sum_val for k, v in self.smoothed.items()}

    def log_report(self, logger, epoch, step, total_steps):
        """打印精确的监控报告"""
        r = self.get_ratios()
        msg = (f"Epoch[{epoch}] Step[{step}/{total_steps}] | "
               f"Ratios: Main({r['main']:.1%}) Trig({r['trig']:.1%}) Dir({r['dir']:.1%}) | "
               f"Loss: Tot({self.history['total'][-1]:.4f}) M({self.history['main'][-1]:.4f})")
        logger.info(msg)

def train_engine(
    model: nn.Module,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    logger: logging.Logger,
    train_cfg: TrainConfig,
    mtl_manager: MTLManager
):
    """
    核心训练引擎 V2 (MTL 联合优化版)
    Loss = Loss_Main(3-class) + λ_trig * Loss_Trig + λ_dir * Loss_Dir
    """
    # --- 1. 准备类别权重 ---
    y_raw = dl_tr.dataset.y.cpu().numpy() if torch.is_tensor(dl_tr.dataset.y) else dl_tr.dataset.y
    cw = mtl_manager.prepare_all(y_raw)
    logger.info(f"⚖️ [MTL Weights Prepared] Main: {cw[0]} | Trig: {cw[1]} | Dir: {cw[2]}")

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_state_loss = None 
    best_state_f1 = None
    wait = 0
    tracker = MTLLossTracker()

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        tracker.reset()
        tr_loss, tr_total = 0.0, 0
        
        
        #  1. 使用 with 语句接管 pbar，这样可以更精准地控制后缀
        with tqdm(dl_tr, desc=f"Epoch {epoch}/{train_cfg.epochs}", ncols=120, leave=False) as pbar:
            for i, (xb, yb, rb) in enumerate(pbar):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                rb = rb.to(device, non_blocking=True)
                # yb: [B]
                cnt_total = yb.numel()
                cnt_action = (yb != 1).sum().item()
                cnt_short = (yb == 0).sum().item()
                cnt_long  = (yb == 2).sum().item()

                if epoch == 1 and i < 5:
                    logger.info(
                        f"[Warmup Batch] "
                        f"action={cnt_action}, short={cnt_short}, long={cnt_long}"
                    )


                # 1. 准备子任务 Target
                target_trig = (yb != 1).long()
                target_dir = torch.where(yb == 2, 1, 0).long()
                action_mask = (yb != 1)

                # 2. 前向传播
                # 获取两个头的原始 Logits
                logits_trig, logits_dir = model(xb, return_fused=False)

                #  调用统一的 Loss 计算
                loss, l_m, l_t, l_d, lam_d = mtl_manager.compute_combined_loss(logits_trig, logits_dir, yb, rb, epoch)

                # 6. 反向传播
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                tr_loss += loss.item() * xb.size(0)
                tr_total += xb.size(0)

                #  2. 更新监测器
                tracker.update(l_m, l_t, l_d, train_cfg.lambda_trig, lam_d)

                #  3. 每 N 步更新进度条后缀，而不是打印新行
                if i % 10 == 0:
                    r = tracker.get_ratios()
                    # 这里的字段会显示在进度条右侧
                    pbar.set_postfix({
                        "Tot": f"{loss.item():.3f}",
                        "M": f"{r['main']*100:4.1f}%", # 强制占用宽度
                        "T": f"{r['trig']*100:4.1f}%",
                        "D": f"{r['dir']*100:4.1f}%"
                    }, refresh=True)
        # 在 train_engine 的 Epoch 循环结尾
        r_final = tracker.get_ratios()
        logger.info(f"🏁 Epoch {epoch} Final Ratios: Main({r_final['main']:.1%}) Trig({r_final['trig']:.1%}) Dir({r_final['dir']:.1%})")

        tr_loss /= max(1, tr_total)

        # --- 验证 ---
        va_loss, yv_true, yv_pred, yt_ret = eval_epoch(model, dl_va, device, mtl_manager)
        va_f1 = f1_score(yv_true, yv_pred, average="macro") if len(yv_true) else 0.0
        scheduler.step(va_loss)

        logger.info(f"Epoch {epoch:03d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} | va_macroF1 {va_f1:.4f}")
        progress_made = False

        # 双指标保存逻辑
        if va_f1 > best_val_f1 + 1e-6:
            best_val_f1 = va_f1
            best_state_f1 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f" [New Best F1] {va_f1:.4f}")
            progress_made = True # 标记有进展

        if va_loss < best_val_loss - 1e-6:
            best_val_loss = va_loss
            best_state_loss = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"📉 [New Best Loss] {va_loss:.4f}")
            progress_made = True # 标记有进展
        if progress_made:
            wait = 0 
        else:
            wait += 1
            if wait >= train_cfg.patience:
                logger.warning(f"🛑 Early Stop at Epoch {epoch}")
                break

    return {
        "best_f1_state": best_state_f1,
        "best_loss_state": best_state_loss,
        "f1_score": best_val_f1,
        "loss_score": best_val_loss
    }

def evaluate_and_save_results(
    train_task:TrainTask,
    results: dict,
    model: nn.Module,
    dl_te: DataLoader,
    # 这里不需要额外传入 criterion，因为 eval_epoch 内部会根据 2+2 逻辑处理
    device: torch.device,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    full_ds: TimeSeriesWindowDataset,
    feature_list:list[str],
    classes: np.ndarray,
    logger: logging.Logger,
    mtl_manager: MTLManager,
    save_dir:str
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
        test_loss, yt_true, yt_pred, yt_ret = eval_epoch(model, dl_te, device, mtl_manager)

        # --- 以下逻辑 100% 保留，因为它们基于已经“还原”的三分类标签 ---
        report_dict = classification_report(yt_true, yt_pred, output_dict=True, zero_division=0)
        test_f1 = report_dict['macro avg']['f1-score']

        logger.info(f"\n{'='*20} Evaluating Model Version: {suffix} {'='*20}")
        logger.info("\n=== Optimized Test Report ===")
        # 使用你原本的格式化打印函数
        logger.info(format_custom_report(report_dict))
        # --- 新增指标计算 ---
        logger.info(f"Test macro-F1: {test_f1:.4f}")

        # 统计标签比例 (保留)
        counts = Counter(yt_true)
        
        total = sum(counts.values())
        logger.info(f"[{suffix}] True label proportion (Test set): " + 
                    ", ".join([f"{c}: {counts[c]/total:.2%}" for c in sorted(counts.keys())]))

        # 保存混淆矩阵 (保留)
        cm = confusion_matrix(yt_true, yt_pred, labels=classes)
        cm_path = os.path.join(save_dir, f"confmat_{suffix.lower()}.csv")
        pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]).to_csv(cm_path, index=True)

        # 保存 .pt 和 Meta (保留)
        pt_path = os.path.join(save_dir, f"model_{suffix.lower()}_info.pt")
        save_state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
        torch.save({
            "state_dict": save_state,
            "feature_list" : feature_list,
            "classes": classes.tolist(),
            "channel": full_ds.feature_count,
            "window": train_cfg.model_cfg.seq_len,
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
            window=train_cfg.model_cfg.seq_len,
            model_version_tag=suffix,
        )
        with open(os.path.join(save_dir, f"model_{suffix.lower()}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        final_metrics.update({
            "test_f1": test_f1,
            "val_f1": val_score,
            "test_loss": test_loss,
            "precision_short": report_dict.get('0', {}).get('precision', 0),
            "recall_short": report_dict.get('0', {}).get('recall', 0),
            "precision_long": report_dict.get('2', {}).get('precision', 0),
            "recall_long": report_dict.get('2', {}).get('recall', 0),
            "overfit_gap": abs(val_score - test_f1)
        })
            
        if getattr(train_cfg.model_cfg,'use_feature_weighting', False):
            logger.info(f"🎨 Generating Interpretability Analysis for {suffix}...")
            
            # 从测试迭代器中取出一个样本 Batch 进行深入分析
            test_xb, test_yb, test_rb = next(iter(dl_te))
            
            # A. 全局特征重要性映射 ( latent weights -> original features )
            importance_df = analyze_feature_importance(
                model=model, 
                batch_x=test_xb, 
                feature_names=full_ds.feature_names, 
                device=device
            )
            # 带有后缀保存，区分 F1 和 Loss 模型
            importance_df.to_csv(os.path.join(save_dir, f"feat_importance_{suffix.lower()}.csv"), index=False)

            # B. 样本级贡献排序分析 ( 识别明星样本 )
            # 建议在评估时传入一个足够大的 Batch，确保 0/2 类别都有样本
            test_xb, test_yb, test_rb = next(iter(dl_te))

            analyze_sample_contribution_by_class(
                model=model,
                batch_x=test_xb,
                batch_y=test_yb,
                batch_r=test_rb,
                feature_names=full_ds.feature_names,
                device=device,
                top_k_per_class=3, # 每个类别看 3 张
                save_dir=save_dir,
                suffix=suffix
            )
            
            # # C. 不同行情下的特征偏好分析
            # plot_regime_importance(
            #     model=model,
            #     batch_x=test_xb,
            #     labels=test_yb,
            #     feature_names=full_ds.feature_names,
            #     device=device
            # )
    #  生成 Task Description (Single Mode)
    primary_suffix = "Best_F1" if train_cfg.best_f1 == True else "Best_Loss"
    
    task_desc = {
        "task_type": train_task,
        "timestamp": pd.Timestamp.now().isoformat(),
        # 移除 common_config
        "models": {
            "main": {
                "model": f"model_{primary_suffix.lower()}_info.pt",
                "meta": f"model_{primary_suffix.lower()}_meta.json"
            }
        }
    }
    
    task_path = os.path.join(save_dir, "task_description.json")
    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(task_desc, f, ensure_ascii=False, indent=2)
        
    logger.info(f"🎉 Task description saved to: {task_path}")
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
    分类别诊断模型置信度与准确率的校准关系。
    X轴: 置信度 (Max Probability)
    Y轴: 分别绘制 Positive(2), Negative(0), Neutral(1) 的准确率曲线
    """
    model.eval()
    versions = [
        ("Best_F1", results.get("best_f1_state")),
        ("Best_Loss", results.get("best_loss_state"))
    ]

    # 定义类别映射与颜色
    class_info = {
        common.Signal.POSITIVE: {"label": "Positive (Buy)", "color": "#ef5350"}, # red
        common.Signal.NEGATIVE: {"label": "Negative (Sell)", "color": "#26a69a"}, # green
        common.Signal.NEUTRAL: {"label": "Neutral (Hold)", "color": "#78909c"}  # 灰色
    }

    plt.figure(figsize=(18, 8))

    for i, (suffix, state) in enumerate(versions):
        if state is None: continue
        model.load_state_dict(state)
        
        all_probs, all_preds, all_trues = [], [], []

        with torch.no_grad():
            for xb, yb, _ in dl_te:
                xb = xb.to(device)
                fused_preds, fused_probs = model(xb, return_fused=True) 
                all_probs.append(fused_probs.cpu().numpy())
                all_preds.append(fused_preds.cpu().numpy())
                all_trues.append(yb.cpu().numpy())

        probs_np = np.concatenate(all_probs)
        preds_np = np.concatenate(all_preds)
        trues_np = np.concatenate(all_trues)

        confidences = np.max(probs_np, axis=1)
        is_correct = (preds_np == trues_np)

        # 分箱设置
        bins = np.linspace(0.33, 1.0, 11)
        bin_indices = np.digitize(confidences, bins) - 1
        bin_centers = (bins[:-1] + bins[1:]) / 2

        ax1 = plt.subplot(1, 2, i+1)
        
        # 1. 绘制背景：总样本分布 (左轴)
        bin_counts = [np.sum(bin_indices == b) for b in range(len(bins)-1)]
        ax1.bar(bin_centers, bin_counts, width=(bins[1]-bins[0])*0.8, 
                color='#f5f5f5', label='Total Samples', edgecolor='#e0e0e0')
        ax1.set_xlabel('Confidence (Max Probability)', fontsize=12)
        ax1.set_ylabel('Sample Count', color='#9e9e9e')
        ax1.tick_params(axis='y', labelcolor='#9e9e9e')

        # 2. 绘制前景：分类别准确率 (右轴)
        ax2 = ax1.twinx()
        ax2.plot([0.33, 1.0], [0.33, 1.0], '--', color='#bdbdbd', alpha=0.8, label='Ideal')

        for cls_val, info in class_info.items():
            cls_bin_accs = []
            for b in range(len(bins) - 1):
                # 筛选条件：属于该置信度区间 且 真实标签为 cls_val
                mask = (bin_indices == b) & (trues_np == cls_val)
                count = np.sum(mask)
                if count > 5: # 样本太少（少于5个）不画点，避免噪点干扰
                    acc = np.mean(is_correct[mask])
                    cls_bin_accs.append(acc)
                else:
                    cls_bin_accs.append(None)

            # 过滤掉 None 值进行绘图
            valid_x = [bin_centers[idx] for idx, val in enumerate(cls_bin_accs) if val is not None]
            valid_y = [val for val in cls_bin_accs if val is not None]
            
            ax2.plot(valid_x, valid_y, marker='o', markersize=4, 
                     linewidth=2, color=info['color'], label=info['label'])

        ax2.set_ylabel('Accuracy per Class', fontsize=12)
        ax2.set_ylim(0, 1.05)
        
        plt.title(f"Class-wise Reliability: {suffix}", fontsize=14)
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        ax1.grid(axis='y', linestyle=':', alpha=0.3)

    plot_path = os.path.join(save_dir, "class_wise_confidence.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"📈 Class-wise reliability plot saved to: {plot_path}")
    plt.close()

def analyze_feature_importance(model, batch_x, feature_names, device='cuda', top_n=20):
    """
    提取样本级权重并映射回原始特征空间进行可视化
    
    参数:
    - model: 训练好的 ConvLSTM1D_V2 模型
    - batch_x: 输入数据 [B, T, F]
    - feature_names: 原始特征名称列表 (长度应等于 F)
    - device: 运行设备
    - top_n: 绘制前 N 个最重要的特征
    """
    model.eval()
    batch_x = batch_x.to(device)
    
    with torch.no_grad():
        # 1. 运行前向传播获取隐藏空间权重 [B, D]
        # 注意：此处假设你已按照之前的建议修改了 forward 返回 weights
        _, _, latent_weights = model(batch_x, return_weights=True)
        
        # 2. 获取 Projection 层的权重矩阵 [D, F]
        # model.proj.weight 形状是 [out_features, in_features]
        proj_weights = model.proj.weight.abs() 
        
    # 3. 映射回原始特征空间
    # 将 Batch 内的权重取平均，得到该批次的平均注意力分布
    avg_latent_weights = latent_weights.mean(dim=0) # [D]
    
    # 矩阵乘法映射重要性: [D] * [D, F] -> [F]
    feature_importance = torch.matmul(avg_latent_weights, proj_weights)
    feature_importance = feature_importance.cpu().numpy()
    
    # 4. 组装数据
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # 5. 可视化
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=importance_df.head(top_n),
        hue='Feature',      # 指定 hue 消除警告
        palette='viridis',
        legend=False        # 配合使用 legend=False
    )
    plt.title(f'Top {top_n} Feature Importance (Mapped from Latent Weights)')
    plt.xlabel('Aggregated Attribution Score')
    plt.ylabel('Original Features')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return importance_df

def plot_regime_importance(model, batch_x, labels, feature_names, device='cuda'):
    """
    根据模型预测的标签分类，分析不同行情下的特征偏好
    """
    model.eval()
    batch_x = batch_x.to(device)
    
    with torch.no_grad():
        # 获取预测和权重
        preds, probs, weights = model(batch_x, return_fused=True, return_weights=True)
        # 核心修正 1：确保所有用于索引和计算的张量都在 CPU 上
        proj_weights = model.proj.weight.abs().cpu()
        weights = weights.cpu()
        preds = preds.cpu() # <--- 添加这一行，修复 RuntimeError

    regimes = {0: 'Short', 1: 'Neutral', 2: 'Long'}
    plt.figure(figsize=(15, 6))

    for label, name in regimes.items():
        # 筛选出属于该类别的样本权重
        mask = (preds == label)
        if mask.any():
            regime_weights = weights[mask].mean(dim=0)
            importance = torch.matmul(regime_weights, proj_weights).numpy()
            
            # 这里的打印信息能帮你快速检查因子有效性
            print(f"[{name}] Top feature: {feature_names[np.argmax(importance)]}")

def analyze_sample_contribution_by_class(model, batch_x, batch_y, batch_r, feature_names, device='cuda', top_k_per_class=3, save_dir=None, suffix=""):
    """
    分别针对 0/1/2 三类标签，找出模型最关注的 Top 样本并保存
    """
    model.eval()
    batch_x = batch_x.to(device)
    
    with torch.no_grad():
        # 获取预测标签和特征权重
        preds, _, latent_weights = model(batch_x, return_fused=True, return_weights=True)
        proj_weights = model.proj.weight.abs() 
        sample_feature_attribution = torch.matmul(latent_weights, proj_weights).cpu().numpy()

    preds_np = preds.cpu().numpy()
    sample_total_energy = sample_feature_attribution.sum(axis=1)
    
    results_df = pd.DataFrame({
        'sample_idx': np.arange(len(batch_r)),
        'pred_label': preds_np,
        'true_label': batch_y.cpu().numpy(),
        'return': batch_r.cpu().numpy(),
        'total_attention': sample_total_energy
    })

    classes = {0: 'Short', 1: 'Neutral', 2: 'Long'}
    
    # 为每个类别创建一张大图
    for cls_val, cls_name in classes.items():
        cls_samples = results_df[results_df['pred_label'] == cls_val]
        
        if cls_samples.empty:
            print(f"⚠️ No samples predicted as {cls_name} in this batch.")
            continue
            
        # 挑选该类别下注意力最高的样本
        top_cls_samples = cls_samples.sort_values(by='total_attention', ascending=False).head(top_k_per_class)
        actual_k = len(top_cls_samples)
        file_name = f"top_samples_{cls_name.lower()}_{suffix.lower()}.png"
        fig, axes = plt.subplots(1, actual_k, figsize=(5 * actual_k, 6), sharey=True)
        if actual_k == 1: axes = [axes]
        
        plt.suptitle(f"File: {file_name}\nTarget Class: {cls_name} | Model: {suffix}", 
                        fontsize=16, fontweight='bold', y=0.98)
        for i, (idx, row) in enumerate(top_cls_samples.iterrows()):
            attr_scores = sample_feature_attribution[int(idx)]
            top_feat_indices = np.argsort(attr_scores)#[-10:] 
            y_data = [feature_names[j] for j in top_feat_indices]

            sns.barplot(
                x=attr_scores[top_feat_indices],
                y=y_data,
                ax=axes[i],
                hue=y_data,
                palette='magma' if cls_val != 1 else 'viridis',
                legend=False
            )
            axes[i].set_title(f"[{cls_name}] Sample #{int(idx)}\nRet: {row['return']:.2%}, True: {int(row['true_label'])}")
            axes[i].set_xlabel("Attribution Score")

        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, file_name)
            plt.savefig(save_path)
            print(f"💾 Saved {cls_name} analysis to: {save_path}")
        plt.close()

def find_best_threshold(results, model, dl_te, device, logger):
    """
    寻找最佳入场阈值 (V2 联合优化适配版)
    逻辑：基于 Net Score (P_long - P_short) 分别扫描多空表现。
    """
    model.eval()
    # 优先加载 Best_Loss 模型，因为加了 Main Loss 后，它的概率校准（Calibration）通常更优
    state = results["best_loss_state"] if results["best_loss_state"] is not None else results["best_f1_state"]
    if state is None: return
    model.load_state_dict(state)

    all_probs = []
    all_trues = []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            #  核心：使用融合后的 3 类概率 [Short(0), Neutral(1), Long(2)]
            _, fused_probs = model(xb, return_fused=True) 
            all_probs.append(fused_probs.cpu().numpy())
            all_trues.append(yb.numpy())

    probs = np.concatenate(all_probs) # [N, 3]
    trues = np.concatenate(all_trues)
    
    # 计算净得分：范围 [-1, 1]，越趋近 1 越倾向多，越趋近 -1 越倾向空
    net_scores = probs[:, 2] - probs[:, 0]
    
    # 扫描区间：从 0.1 (激进) 到 0.7 (极度保守)
    thresholds = np.linspace(0.1, 0.7, 25)
    
    logger.info("\n📊 --- V2 Net-Score Asymmetric Scan Report ---")
    header = (f"{'Thresh':<8} | "
              f"{'L-Prec':<8} {'L-Rec':<8} {'L-Cnt':<6} | "
              f"{'S-Prec':<8} {'S-Rec':<8} {'S-Cnt':<6} | "
              f"{'Neu-Rec':<8}")
    logger.info(header)
    logger.info("-" * len(header))

    total_long = (trues == 2).sum()
    total_short = (trues == 0).sum()
    total_neu = (trues == 1).sum()

    for th in thresholds:
        # 1. 多头统计 (Long)
        l_mask = (net_scores > th)
        l_prec = (trues[l_mask] == 2).mean() if l_mask.any() else 0.0
        l_rec = (trues[l_mask] == 2).sum() / max(1, total_long)
        l_cnt = l_mask.sum()
        
        # 2. 空头统计 (Short)
        s_mask = (net_scores < -th)
        s_prec = (trues[s_mask] == 0).mean() if s_mask.any() else 0.0
        s_rec = (trues[s_mask] == 0).sum() / max(1, total_short)
        s_cnt = s_mask.sum()

        # 3. 震荡召回 (Neutral Recall / 避险率)
        # 即：真正的 Neutral 样本中，有多少被正确地过滤掉了（既没做多也没做空）
        action_mask = l_mask | s_mask
        neu_correct_mask = (trues == 1) & (~action_mask)
        neu_rec = neu_correct_mask.sum() / max(1, total_neu)

        logger.info(f"{th:.3f}    | "
                    f"{l_prec:8.4f} {l_rec:8.4f} {l_cnt:<6} | "
                    f"{s_prec:8.4f} {s_rec:8.4f} {s_cnt:<6} | "
                    f"{neu_rec:8.2%}")
    
    logger.info("-" * len(header))
    return None

@torch.no_grad()
def eval_epoch(model, loader, device, mtl_manager:MTLManager):
    model.eval()
    tl, yt, yp, yr = 0.0, [], [], [] # 🌟 增加 yr 用于存储 trend_strength

    for xb, yb, rb in loader:
        xb, yb, rb = xb.to(device), yb.to(device), rb.to(device)
        logits_trig, logits_dir = model(xb, return_fused=False)
        
        # 使用 MTLManager 计算损失
        loss, _, _, _, _ = mtl_manager.compute_combined_loss(logits_trig, logits_dir, yb, rb, epoch=0)
        tl += loss.item() * xb.size(0)

        # 获取预测标签
        yp_batch, _ = model(xb, return_fused=True)
        yp.append(yp_batch.cpu().numpy())
        yt.append(yb.cpu().numpy())
        yr.append(rb.cpu().numpy()) # 🌟 收集收益率

    # 返回增加了一个返回值：拼接后的收益率数组
    return tl/len(loader.dataset), np.concatenate(yt), np.concatenate(yp), np.concatenate(yr)

def fusion_long_short_ovr(logger: logging.Logger, long_train_output_dir:str, short_train_output_dir:str, save_dir:str):
    task_type = TrainTask.LONG_SHORT_OVR
    rel_long = os.path.relpath(long_train_output_dir, save_dir)
    rel_short = os.path.relpath(short_train_output_dir, save_dir)
    task_desc = {
        "task_type": TrainTask.LONG_SHORT_OVR,
        "timestamp": pd.Timestamp.now().isoformat(),
        "models": {
            "long_ovr": rel_long,
            "short_ovr": rel_short,
        }
    }
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "task_description.json"), "w", encoding="utf-8") as f:
        json.dump(task_desc, f, ensure_ascii=False, indent=2)

    logger.info(f"🎉 {task_type} task complete. Artifacts saved to: {save_dir}")

def copy_dir_if_different(src: str, dst: str, logger: logging.Logger):
    src_path = Path(src).resolve()
    dst_path = Path(dst).resolve()

    if src_path == dst_path:
        logger.info(f"Skip copy: source and target are the same path: {src_path}")
        return

    if dst_path.exists():
        shutil.rmtree(dst_path)

    shutil.copytree(src_path, dst_path)
    logger.info(f"Copied directory: {src_path} -> {dst_path}")

def fusion_trigger_dir(logger: logging.Logger, trigger_train_output_dir:str, dir_train_output_dir:str, save_dir:str):
    task_type = TrainTask.TRIGGER_DIR
    trigger_dir_save_dir = save_dir#os.path.join(save_dir,task_type)
    os.makedirs(trigger_dir_save_dir, exist_ok=True)
    
    tri_new_dir = os.path.join(trigger_dir_save_dir, TrainTask.SINGLE_MODEL_TRIGGER)
    dir_new_dir = os.path.join(trigger_dir_save_dir, TrainTask.SINGLE_MODEL_DIR)
    copy_dir_if_different(src=trigger_train_output_dir, dst=tri_new_dir, logger=logger)
    copy_dir_if_different(src=dir_train_output_dir, dst=dir_new_dir, logger=logger)
    task_desc = {
        "task_type": TrainTask.TRIGGER_DIR,
        "timestamp": pd.Timestamp.now().isoformat(),
        "models": {
            "trigger": TrainTask.SINGLE_MODEL_TRIGGER,
            "direction": TrainTask.SINGLE_MODEL_DIR,
        }
    }
    
    with open(os.path.join(trigger_dir_save_dir, "task_description.json"), "w", encoding="utf-8") as f:
        json.dump(task_desc, f, ensure_ascii=False, indent=2)

    logger.info(f"🎉 {task_type} task complete. Artifacts saved to: {trigger_dir_save_dir}")

#it could be same model with different parameters
def run_task_trade_direction(logger: logging.Logger, train_task: TrainTask, train_cfg_tri: TrainConfig,train_cfg_dir: TrainConfig, 
                             prep_output_dir = common.DATA_OUT_DIR, save_dir: str = common.TRAIN_OUT_DIR,experiment:bool = False):
    os.makedirs(save_dir, exist_ok=True)

    if train_task != TrainTask.TRIGGER_DIR:
        raise RuntimeError(f"train_task:{train_task} not compatible")
    if train_cfg_tri.feature_conf_list != train_cfg_dir.feature_conf_list:
        raise RuntimeError(f"different feature_conf_list not supported")
    if train_cfg_tri.model_cfg.seq_len != train_cfg_dir.model_cfg.seq_len:
        raise RuntimeError(f"different seq_len not supported")
    feature_direction_map_filtered = {}
    for feature_name in train_cfg_tri.feature_conf_list:
        # 从全局 feature_direction_map 中查找方向，如果找不到则默认为 1（正向）
        direction = feature_direction_map.get(feature_name, 1)
        feature_direction_map_filtered[feature_name] = direction
    
    logger.info(f"📋 Using {len(feature_direction_map_filtered)} features from feature_conf_list")

    d_cfg = DataConfig()
    
    return run_training(train_task,feature_direction_map_filtered, logger, d_cfg, train_cfg_tri, train_cfg_dir, prep_output_dir,save_dir,experiment)

def main(logger: logging.Logger,train_task:TrainTask, train_cfg=TrainConfig(), prep_output_dir = common.DATA_OUT_DIR, save_dir: str = common.TRAIN_OUT_DIR,experiment:bool = False):
    os.makedirs(save_dir, exist_ok=True)

    # 根据 feature_conf_list 从全局 feature_direction_map 补充完整方向信息
    feature_direction_map_filtered = {}
    for feature_name in train_cfg.feature_conf_list:
        # 从全局 feature_direction_map 中查找方向，如果找不到则默认为 1（正向）
        direction = feature_direction_map.get(feature_name, 1)
        feature_direction_map_filtered[feature_name] = direction
    
    logger.info(f"📋 Using model {train_cfg.model_cfg.model_type} {len(feature_direction_map_filtered)} features from feature_conf_list")

    d_cfg = DataConfig()
    
    return run_training(train_task,feature_direction_map_filtered, logger, d_cfg, train_cfg, None, prep_output_dir,save_dir,experiment)
# ==============================================================================
# 5. 调用入口 (Main Entry)
# ==============================================================================

if __name__ == "__main__":
    logger, _ = common.setup_session_logger(sub_folder='train', file_level = logging.DEBUG)
    begin_time = time.time()
    save_dir =  os.path.join(common.TRAIN_OUT_DIR,train_task_config)
    main(logger,train_task_config,SingleModelTrigger,save_dir = save_dir)
    save_dir =  os.path.join(common.TRAIN_OUT_DIR)
    # fusion_long_short_ovr(logger,r"/home/chao/work/Quant/output/train/SINGLE_MODEL_LONG_OVR", r"/home/chao/work/Quant/output/train/SINGLE_MODEL_SHORT_OVR",save_dir)
    # run_task_trade_direction(logger, TrainTask.TRIGGER_DIR, SingleModelTrigger, SingleModelDirection, save_dir = save_dir)
    fusion_trigger_dir(logger,
                       os.path.join(save_dir,TrainTask.SINGLE_MODEL_TRIGGER),
                       os.path.join(save_dir,TrainTask.SINGLE_MODEL_DIR),
                       save_dir)
    end_time = time.time()
    logger.info(f"Total training time: {(end_time - begin_time)} seconds")