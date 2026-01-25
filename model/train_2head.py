#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,shutil
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

from torch.utils.data import WeightedRandomSampler, Dataset, DataLoader
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
    readout: str = ['last' , 'meanmax' , 'attn', 'mix'][3]
    head: str = ['linear' , 'mlp'][0]
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
    model_version: int = 2
    d_model: int = 64
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
    head_dropout: float = 0.2
    readout: str = "mix"    # 'last'|'meanmax'|'attn'|'mix'
    head: str = "linear"    # 'linear'|'mlp'
    logit_clip: Optional[float] = None
    p_drop: Optional[float] = None
    task_proj_dim: int = 64
    use_feature_selector: bool = True

@dataclass
class TCNConfig:
    model_type: str = "tcn"
    model_version: int = 1
    num_channels: list = field(default_factory=lambda: [64, 128, 256])
    kernel_size: int = 3
    dropout: float = 0.2
    readout: str = "mix"
    logit_clip: Optional[float] = None

@dataclass
class MambaConfig:
    model_type: str = "mamba"
    model_version: int = 1
    d_model: int = 128
    n_layers: int = 4
    d_state: int = 16
    expand: int = 2
    dropout: float = 0.1
    readout: str = "mix"  # 'last' | 'meanmax' | 'mix'
    logit_clip: Optional[float] = None

@dataclass
class XGBoostConfig:
    model_type: str = "xgboost"
    model_version: int = 1
    xgb_depth: int = 6
    xgb_estimators: int = 100
    learning_rate: float = 3e-4
    # 新增：用于 Flatten 维度计算
    window_size: int = common.CANDLESTICK_NUM

@dataclass
class CNNConfig:
    model_type: str = "cnn"
    model_version: int = 1
    p_drop: float = 0.3
    tau: float = 16.0
    use_tpool: bool = False

@dataclass
class TrainConfig:
    model_cfg = ConvLSTMConfig()
    data_cfg = DataConfig()
    epochs: int = 20
    batch_size: int = 512    #越小模型越敏感，小batch_size自带正则化
    lr: float = 3e-4    #3e-4
    gate_lr: float = 1e-2    #3e-4
    weight_decay: float = 5e-4  # $$L_{total} = L_{original} + \frac{\lambda}{2} \sum \|w\|^2$$  防止过拟合
    patience: int = 8
    seed: int = 42  #设计和验证阶段固定 seed，模型确定之后用多个 seed。
    save_dir: str = common.TRAIN_OUT_DIR
    stride = 2
    use_cache = True
    lambda_trig: float = 0.5  # Trigger 任务权重
    lambda_dir: float = 0.4   # Direction 任务权重 (设为 0 即可实现第一阶段只练 Trigger).lambda_dir需要补偿比例不平衡
    lambda_gate: float = 1e-3
# ==============================================================================
# 3. 核心逻辑 (Core Logic)
# ==============================================================================
def get_balanced_sampler(dataset):
    # 1. 提取所有样本的标签
    all_labels = dataset.labels 
    
    # 2. 统计各类别原始数量: [Short(0), Neutral(1), Long(2)]
    class_counts = torch.bincount(torch.tensor(all_labels))
    total_n = class_counts.sum().float()
    
    # 3. 计算“自然”比例
    # 保持 Neutral 的原始占比不变
    p_neutral = class_counts[1] / total_n
    # 计算 Action (Long + Short) 的总占比
    p_action = (class_counts[0] + class_counts[2]) / total_n
    
    # 4. 设置目标比例: 让 Long 和 Short 平分 p_action
    # 索引对应: [0: Short, 1: Neutral, 2: Long]
    target_props = torch.tensor([p_action / 2, p_neutral, p_action / 2]) 
    
    # 5. 计算采样权重: Weight = Target_Prop / Actual_Count
    # 这样在采样时，Long/Short 被选中的总概率相等，且 Neutral 的总概率维持自然水平
    class_weights = target_props / class_counts.float()
    
    # 6. 为每个样本分配权重并创建采样器
    sample_weights = [class_weights[label] for label in all_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def run_training(feature_config_list:list[common.FeatureContainer], logger:logging, data_cfg: DataConfig, train_cfg: TrainConfig, model_cfg):
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
        cache_path=os.path.join(common.TRAIN_OUT_DIR,"train_cache.pt"), stride =train_cfg.stride, use_cache = train_cfg.use_cache, show_feature_distribution=True
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
    # 🌟 注入平衡采样逻辑
    # 使用你代码中定义的 get_balanced_sampler (Neutral 50%, Short 25%, Long 25%)
    sampler_tr = get_balanced_sampler(ds_tr) 
    
    cw_balanced = compute_class_weight("balanced", classes=classes, y=y_tr_np)
    class_weights = torch.tensor(cw_balanced, dtype=torch.float32, device=device)
    logger.info(f"Class weights: {dict(zip(classes, cw_balanced))}")

    # 4. DataLoader
    # 🌟 针对 5090 优化：增加 num_workers，开启 pin_memory
    dl_tr = DataLoader(
        ds_tr, 
        batch_size=train_cfg.batch_size, 
        sampler=sampler_tr,      # 使用采样器替代 shuffle
        shuffle=False,           # 使用 sampler 时必须设为 False
        num_workers=4,           # 5090 算力强，建议开启多线程读取
        pin_memory=(device.type=="cuda")
    )
    dl_va = DataLoader(ds_va, batch_size=train_cfg.batch_size, shuffle=False, pin_memory=(device.type=="cuda"))
    dl_te = DataLoader(ds_te, batch_size=train_cfg.batch_size, shuffle=False, pin_memory=(device.type=="cuda"))

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
        logger.info(f'xgboost device:{device}')
        model = ModelFactory.build_for_training(
            model_type=m_type, model_version=m_ver, device=device,
            input_size=full_ds.feature_count, n_classes=len(classes),
            input_dim=data_cfg.window * full_ds.feature_count, window_size = model_cfg.window_size,
            xgb_params=params
        )
    else:
        model = ModelFactory.build_for_training(
            model_type=m_type, model_version=m_ver, device=device,
            input_size=full_ds.feature_count, n_classes=len(classes),
            **params
        )
        # 🌟 针对 Mamba 的优化器特殊设置
        if m_type == "mamba":
            # 将参数分类：dt_proj 需要更高的学习率，A_log 需要较低的学习率
            dt_params = []
            other_params = []
            for name, param in model.named_parameters():
                if "dt_proj" in name:
                    dt_params.append(param)
                else:
                    other_params.append(param)
            
            optimizer = torch.optim.AdamW([
                {"params": dt_params, "lr": train_cfg.lr * 10}, # 步长参数学习率放大
                {"params": other_params, "lr": train_cfg.lr}
            ], weight_decay=train_cfg.weight_decay)
        else:
            use_gate = bool(train_cfg.model_cfg.use_feature_selector and hasattr(model, "feature_selector"))

            if use_gate:
                # Differential Learning Rates: Gate usually needs to be more aggressive
                gate_params = [model.feature_selector.importance_logits]
                base_params = [p for n, p in model.named_parameters() if "feature_selector" not in n]
                
                optimizer = torch.optim.AdamW([
                    {"params": base_params, "lr": train_cfg.lr, "weight_decay": train_cfg.weight_decay},
                    {"params": gate_params, "lr": train_cfg.gate_lr, "weight_decay": 0.0} # No decay for logits
                ])
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    # 7. 调用封装好的训练引擎
    logger.info("🚀 Starting training engine...")
    mtl = MTLManager(device, train_cfg)
    if m_type == 'xgboost':
        # 调用新增加的 XGBoost 引擎
        results = train_xgboost_engine(
            model=model,
            dl_tr=dl_tr,
            dl_va=dl_va,
            logger=logger,
            mtl_manager=mtl
        )
        optimizer = None
        scheduler = None
    else:
        # 6. 训练准备
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
        # 现有的 PyTorch 训练引擎
        results = train_engine(
            model=model,
            dl_tr=dl_tr,
            dl_va=dl_va,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            logger=logger,
            train_cfg=train_cfg,
            mtl_manager=mtl
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
        logger=logger,
        mtl_manager = mtl,
    )
    diagnose_confidence(results, model=model,dl_te=dl_te, device=device,logger=logger,save_dir = common.TRAIN_OUT_DIR)
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
    def __init__(self, X, y): 
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        # 🌟 关键：增加 labels 属性供 Sampler 使用，确保为整数
        self.labels = self.y.long().numpy() 

    def __len__(self): 
        return self.X.shape[0]

    def __getitem__(self, i): 
        return self.X[i], self.y[i]

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

class MTLManager:
    def __init__(self, device, train_cfg):
        self.device = device
        self.cfg = train_cfg
        self.weights = {}
        self.criteria = {}
        # 🌟 新增：对称性惩罚权重，初始建议设为 0.5 ~ 1.0
        self.bias_lambda = getattr(train_cfg, 'bias_lambda', 0.5)

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

    def compute_combined_loss(self, logits_trig, logits_dir, yb, flip_penalty=2.0, miss_penalty=1.4):
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
            penalty[fatal_mask] = flip_penalty
            
            # 2. 保守错误：趋势预测成了震荡 (例如 y=2, pred=1)
            # 增加此惩罚，防止模型逃向震荡类，迫使其在趋势中寻找微弱信号
            miss_mask = trend_mask & (preds == 1)
            penalty[miss_mask] = miss_penalty
            
            # --- 🌟 逻辑 B: 惩罚能量归一化 ---
            # 核心：确保 penalty 均值为 1.0。
            # 这样放大错误权重的同事，会相对缩小正确预测的权重，
            # 且保证 Main Loss 总量稳定，不会淹没 lambda_trig/dir 的梯度。
            penalty = penalty / (penalty.mean() + 1e-8)

        # 应用归一化惩罚
        loss_main = (sample_main_loss * penalty).mean()

        # --- 🌟 逻辑 C: 方向对称性约束 (Bias Constraint) ---
        # 计算当前 Batch 预测多头和空头的平均概率差
        # 用于强制拉平模型在不同训练轮次中产生的随机“多/空偏心”
        avg_p_short = fused_probs[:, 0].mean()
        avg_p_long = fused_probs[:, 2].mean()
        bias_loss = torch.zeros((), device=logits_dir.device)

        act = (p_trig[:, 1] > 0.5)   # 或 act = (yb != 1)
        if act.any():
            bias_loss = torch.abs(
                p_dir[act, 0].mean() - p_dir[act, 1].mean()
            )

        
        # 获取 bias_lambda 配置，默认设为 0.5
        bias_lambda = getattr(self.cfg, 'bias_lambda', 0.5)

        # 4. 最终多任务组合
        total_loss = loss_main + \
                    (self.cfg.lambda_trig * loss_trig) + \
                    (self.cfg.lambda_dir * loss_dir) + \
                    (bias_lambda * bias_loss)
        
        return total_loss, loss_main, loss_trig, loss_dir, self.cfg.lambda_dir
    
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

def train_xgboost_engine(model, dl_tr, dl_va, logger, mtl_manager):
    logger.info("🌲 Training XGBoost Dual-Head Model...")
    
    # 1. 提取全量数据
    x_train, y_train = dl_tr.dataset.X, dl_tr.dataset.y
    x_val, y_val = dl_va.dataset.X, dl_va.dataset.y
    
    # 🌟 关键修正：确保 mtl_manager 已经针对当前数据初始化了 criteria
    # prepare_all 内部会设置 self.criteria['trig'] 等
    y_raw_train = y_train.cpu().numpy() if torch.is_tensor(y_train) else y_train
    mtl_manager.prepare_all(y_raw_train) 
    
    # 2. 执行 2+2 分层训练
    model.fit(x_train, y_train, x_val, y_val)
    
    # 3. 验证阶段
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 此时 eval_epoch 内部调用 compute_combined_loss 就不会报 KeyError 了
    va_loss, yv_true, yv_pred = eval_epoch(model, dl_va, device, mtl_manager)
    va_f1 = f1_score(yv_true, yv_pred, average="macro")
    
    logger.info(f"✅ XGBoost Train Done. Val F1: {va_f1:.4f}")
    
    return {
        "best_f1_state": model.state_dict(),
        "best_loss_state": model.state_dict(),
        "f1_score": va_f1,
        "loss_score": va_loss
    }

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
    y_raw = dl_tr.dataset.y.numpy() if torch.is_tensor(dl_tr.dataset.y) else dl_tr.dataset.y
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
        
        
        # 🌟 1. 使用 with 语句接管 pbar，这样可以更精准地控制后缀
        with tqdm(dl_tr, desc=f"Epoch {epoch}/{train_cfg.epochs}", ncols=120, leave=False) as pbar:
            for i, (xb, yb) in enumerate(pbar):
                xb, yb = xb.to(device), yb.to(device)
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

                # 🌟 调用统一的 Loss 计算
                loss, l_m, l_t, l_d, lam_d = mtl_manager.compute_combined_loss(logits_trig, logits_dir, yb)

                # 6. 反向传播
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                tr_loss += loss.item() * xb.size(0)
                tr_total += xb.size(0)

                # 🌟 2. 更新监测器
                tracker.update(l_m, l_t, l_d, train_cfg.lambda_trig, lam_d)

                # 🌟 3. 每 N 步更新进度条后缀，而不是打印新行
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
        lr_base = optimizer.param_groups[0]["lr"]
        logger.info(f"🏁 Epoch {epoch} | LR B/G: {lr_base:.5f} | Final Ratios: Main({r_final['main']:.1%}) Trig({r_final['trig']:.1%}) Dir({r_final['dir']:.1%})")

        tr_loss /= max(1, tr_total)

        # --- 验证 ---
        va_loss, yv_true, yv_pred = eval_epoch(model, dl_va, device, mtl_manager)
        va_f1 = f1_score(yv_true, yv_pred, average="macro") if len(yv_true) else 0.0
        scheduler.step(va_loss)

        logger.info(f"Epoch {epoch:03d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} | va_macroF1 {va_f1:.4f}")
        progress_made = False

        # 双指标保存逻辑
        if va_f1 > best_val_f1 + 1e-6:
            best_val_f1 = va_f1
            best_state_f1 = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"🌟 [New Best F1] {va_f1:.4f}")
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
    results: dict,
    model: nn.Module,
    dl_te: DataLoader,
    # 这里不需要额外传入 criterion，因为 eval_epoch 内部会根据 2+2 逻辑处理
    device: torch.device,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    full_ds: TimeSeriesWindowDataset,
    feature_config_list:list[common.FeatureContainer],
    classes: np.ndarray,
    logger: logging.Logger,
    mtl_manager: MTLManager
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
        if train_cfg.model_cfg.use_feature_selector:
            print_feature_importance(model, full_ds.feature_names, logger, suffix)
        test_loss, yt_true, yt_pred = eval_epoch(model, dl_te, device, mtl_manager)

        # --- 以下逻辑 100% 保留，因为它们基于已经“还原”的三分类标签 ---
        report_dict = classification_report(yt_true, yt_pred, output_dict=True, zero_division=0)
        test_f1 = report_dict['macro avg']['f1-score']

        logger.info(f"\n{'='*20} Evaluating Model Version: {suffix} {'='*20}")
        logger.info("\n=== Optimized Test Report ===")
        # 使用你原本的格式化打印函数
        logger.info(format_custom_report(report_dict))
        # --- 新增指标计算 ---
        t_f1, d_prec, f_rate = calculate_quant_metrics(yt_true, yt_pred, report_dict)
        logger.info(f"🎯 [Quant Metrics] Trend-F1: {t_f1:.4f} | Dir-Precision: {d_prec:.4f} | Flip-Rate: {f_rate:.2%}")
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
        (container.feature.__name__, container.parameters) for container in feature_config_list
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
            
    # 🌟 生成 Task Description (Single Mode)
    primary_suffix = "Best_F1" if results.get("best_f1_state") is not None else "Best_Loss"
    
    task_desc = {
        "task_type": "single",
        "timestamp": pd.Timestamp.now().isoformat(),
        # 移除 common_config
        "models": {
            "main": {
                "model": f"model_{primary_suffix.lower()}_info.pt",
                "meta": f"model_{primary_suffix.lower()}_meta.json"
            }
        }
    }
    
    task_path = os.path.join(train_cfg.save_dir, "task_description.json")
    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(task_desc, f, ensure_ascii=False, indent=2)
        
    logger.info(f"🎉 Task description saved to: {task_path}")
    return final_metrics

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

def calculate_quant_metrics(y_true, y_pred, report_dict=None):
    """
    专门为量化设计的评估指标
    0: Short, 1: Sideways, 2: Long
    """
    if report_dict is None:
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # 1. Trend-F1 (只看涨跌两类的平均)
    f1_0 = report_dict.get('0', {}).get('f1-score', 0)
    f1_2 = report_dict.get('2', {}).get('f1-score', 0)
    trend_f1 = (f1_0 + f1_2) / 2
    
    # 2. Directional Precision (预测为趋势时的准确率)
    # 预测为 0 或 2 的样本中，真正是 0 或 2 的比例
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    pred_trend_total = (y_pred == 0).sum() + (y_pred == 2).sum()
    true_trend_correct = cm[0, 0] + cm[2, 2]
    dir_precision = true_trend_correct / pred_trend_total if pred_trend_total > 0 else 0
    
    # 3. Fatal Flip Rate (做反的概率)
    # 在真实为趋势的样本中，有多少被预测成了相反的方向 (0->2 或 2->0)
    actual_trend_total = (y_true == 0).sum() + (y_true == 2).sum()
    fatal_flips = cm[0, 2] + cm[2, 0]
    flip_rate = fatal_flips / actual_trend_total if actual_trend_total > 0 else 0
    
    return trend_f1, dir_precision, flip_rate

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
                # 🌟 适配 V2：直接获取融合后的概率 [B, 3]
                fused_preds, fused_probs = model(xb, return_fused=True) 
                
                all_probs.append(fused_probs.cpu().numpy())
                all_preds.append(fused_preds.cpu().numpy())
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
            # 🌟 核心：使用融合后的 3 类概率 [Short(0), Neutral(1), Long(2)]
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
def eval_epoch(model, loader, device, mtl_manager):
    model.eval()
    tl, yt, yp = 0.0, [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits_trig, logits_dir = model(xb, return_fused=False)
        
        # 🌟 使用同一套计算逻辑，彻底避免“度量衡”不一致
        loss, _, _, _, _ = mtl_manager.compute_combined_loss(logits_trig, logits_dir, yb)
        tl += loss.item() * xb.size(0)

        yp_batch, _ = model(xb, return_fused=True)
        yp.append(yp_batch.cpu().numpy())
        yt.append(yb.cpu().numpy())

    return tl/len(loader.dataset), np.concatenate(yt), np.concatenate(yp)

def main(logger:logging.Logger):
    if os.path.exists(common.TRAIN_OUT_DIR):
        shutil.rmtree(common.TRAIN_OUT_DIR)
    os.makedirs(common.TRAIN_OUT_DIR, exist_ok=True)
    feature_config_list = [
        # 1. 自定义的成交量爆发特征 (窗口 512，对比前 2 强)
        # FCVolumeEvent, 

        # 2. 价格趋势与指标类    FeatureMA > FeatureRsi/FeatureKdj/FeatureMACD   what happen to FeatureMACD??
        common.FCMACD,   # （12，26，9），（6，13，5）或（10，20，7）
        # common.FCMA,     # slope 值搭配使用
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
        common.FCMFI,
        # # FCATS,  # 负作用

        # # 4. K线形态类
        common.FCCandle,
        common.FCOrigin,
    ]
    
    # 4. 打印结果
    logger.info("🏆 === 最终选中的特征组合 (Final Selection) ===")
    logger.info("-" * 50)
    for i, container in enumerate(feature_config_list):
        logger.info(f"{i+1}. {container.feature.__name__:<20} | 参数: {container.parameters}")
    logger.info("-" * 50)
    logger.info(f"📊 总特征组数量: {len(feature_config_list)}")
    
    # 1. 数据配置
    d_cfg = DataConfig()
    
    # 2. 训练配置
    t_cfg = TrainConfig()
            # 0             1                   2                   3           4               5               6
    m_cfg = [LSTMConfig(), TransformerConfig(), ConvLSTMConfig(), CNNConfig(), XGBoostConfig(), TCNConfig(), MambaConfig()][2]
    # m_cfg.model_version = 1

    logger.info(f"Training {m_cfg.model_type}...")
    return run_training(feature_config_list, logger,d_cfg, t_cfg, m_cfg)
# ==============================================================================
# 5. 调用入口 (Main Entry)
# ==============================================================================

if __name__ == "__main__":
    logger, _ = common.setup_session_logger(sub_folder='train', file_level = logging.DEBUG)
    main(logger)