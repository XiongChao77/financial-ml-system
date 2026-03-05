#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os,shutil,time
import sys
import json
import logging
import torch
# 路径设置
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from data_process import common
# 1. 强制开启持久化图缓存
torch._inductor.config.fx_graph_cache = True

# 2. 指定统一的缓存路径 (建议放在项目目录下)
# 这样即便进程重启，或者并行运行，都能避开重复编译
cache_dir = os.path.join(common.TRAIN_OUT_DIR, ".inductor_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

# 3. 针对 5090 的优化建议：如果输入维度变化频率不高，可以关闭动态形状以换取极限性能
# torch._inductor.config.dynamic_shapes = False

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from model.data_loader import TimeSeriesWindowDataset
from model.model_factory import ModelFactory

# ==============================================================================
# 1. 配置定义 (Configuration)
# ==============================================================================

feature_conf_list = [

    # =========================
    # 原始市场基础信息（Raw Market State）
    # =========================
    "open",
    "high",
    "low",
    "close",
    "volume",
    "number_of_trades",
    "quote_asset_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


@dataclass
class DataConfig:
    label_col: str = "label"
    window: int = common.BaseDefine.predict_num
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
    model_version: int = 3
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
    use_feature_weighting: bool = False

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
    window_size: int = common.BaseDefine.predict_num

@dataclass
class CNNConfig:
    model_type: str = "cnn"
    model_version: int = 1
    p_drop: float = 0.3
    tau: float = 16.0
    use_tpool: bool = False

@dataclass
class TrainConfig:
    model_cfg: ConvLSTMConfig = field(default_factory=ConvLSTMConfig)
    data_cfg: DataConfig = field(default_factory=DataConfig)
    feature_conf_list: List[str] = field(default_factory=lambda: feature_conf_list)
    epochs: int = 100
    batch_size: int = 256#256
    lr: float = 3e-4
    gate_lr: float = 3e-4
    weight_decay: float = 5e-4
    patience: int = 15
    seed: int = 42
    stride: int = 2
    use_cache: bool = False
    lambda_trig: float = 0.5
    lambda_dir: float = 0.7
    lambda_gate: float = 1e-3
    mag_alpha: float = 0
    mag_limit: float = 4.0
    flip_penalty: float = 1.6
    miss_penalty: float = 1.2
    mag_warmup_epochs:int = 8
    temperature:float = 2.0
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

def apply_feature_direction(X: torch.Tensor, feature_names: List[str], direction_map: Dict[str, int], logger) -> torch.Tensor:
    """
    对 direction=-1 的特征列乘以 -1，使其与收益正相关。
    X: shape [N, T, F]，归一化后的特征张量
    feature_names: 特征名列表，与 X 的第 3 维对应
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

def run_strict_subsampling_probe(full_ds, tr_rng, te_rng, logger, n_iterations=5):
    """
    Dissertation Rigorous Probe: 
    通过物理下采样确保 1:1:1 的类别分布，并进行多次自助抽样（Bootstrapping）以消除随机偏置。
    """
    logger.info("🧪 [Rigorous Probe] Executing Strict Subsampling (1:1:1 Ratio)...")

    # 1. 准备原始数据 (只在训练集范围内进行采样)
    X_raw = full_ds.X[tr_rng[0]:tr_rng[1]].cpu().numpy().reshape(tr_rng[1]-tr_rng[0], -1)
    y_raw = full_ds.y[tr_rng[0]:tr_rng[1]].cpu().numpy()
    
    # 2. 识别有效类别的索引 (排除 Signal.INVALID = -1)
    idx_short = np.where(y_raw == 0)[0]
    idx_neutral = np.where(y_raw == 1)[0]
    idx_long = np.where(y_raw == 2)[0]

    # 3. 确定最小样本量 N_min
    n_min = min(len(idx_short), len(idx_neutral), len(idx_long))
    
    if n_min < 50:
        logger.warning(f"⚠️ 样本量极度匮乏 (N_min={n_min})，下采样结果可能不具有统计显著性。")
    else:
        logger.info(f"⚖️ Balancing each class to N={n_min}. Total Training Samples: {n_min * 3}")

    iteration_f1s = []

    # 4. 执行多次采样以确保严谨性
    for i in range(n_iterations):
        # 严格 1:1:1 抽样
        s_choice = np.random.choice(idx_short, n_min, replace=False)
        n_choice = np.random.choice(idx_neutral, n_min, replace=False)
        l_choice = np.random.choice(idx_long, n_min, replace=False)
        
        balanced_idx = np.concatenate([s_choice, n_choice, l_choice])
        np.random.shuffle(balanced_idx) # 打乱顺序防止 Batch 偏差

        X_tr_bal = X_raw[balanced_idx]
        y_tr_bal = y_raw[balanced_idx]

        # 准备测试集 (保持自然分布以观察泛化能力)
        X_te = full_ds.X[te_rng[0]:te_rng[1]].cpu().numpy().reshape(te_rng[1]-te_rng[0], -1)
        y_te = full_ds.y[te_rng[0]:te_rng[1]].cpu().numpy()

        # 5. 训练模型 (此时不需 class_weight='balanced'，因为数据已物理平衡)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_bal)
        X_te_scaled = scaler.transform(X_te)

        lr = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
        lr.fit(X_tr_scaled, y_tr_bal)
        
        # 6. 记录 Macro-F1
        y_pred = lr.predict(X_te_scaled)
        iteration_f1s.append(f1_score(y_te, y_pred, average='macro'))

    # 7. 统计输出
    avg_f1 = np.mean(iteration_f1s)
    std_f1 = np.std(iteration_f1s)
    
    logger.info(f"✅ [Strict Subsampling Result] Mean Macro-F1: {avg_f1:.4f} (±{std_f1:.4f}) over {n_iterations} iterations")
    
    return avg_f1, std_f1
    
def run_logistic_regression_probe(full_ds, tr_rng, te_rng, logger):
    """
    Dissertation Step 4: Simple probe to explore information variation.
    Strictly controls class balance via random subsampling.
    """
    logger.info("🧪 Starting Logistic Regression Probe (Step 4)...")

    # 1. Prepare Data (Flatten Time Series: [N, T, F] -> [N, T*F])
    X_tr = full_ds.X[tr_rng[0]:tr_rng[1]].cpu().numpy().reshape(tr_rng[1]-tr_rng[0], -1)
    y_tr = full_ds.y[tr_rng[0]:tr_rng[1]].cpu().numpy()
    
    X_te = full_ds.X[te_rng[0]:te_rng[1]].cpu().numpy().reshape(te_rng[1]-te_rng[0], -1)
    y_te = full_ds.y[te_rng[0]:te_rng[1]].cpu().numpy()

    # # 2. Strict Class Balancing (Subsampling to the lowest proportion)
    # counts = Counter(y_tr)
    # min_samples = min(counts.values())
    # logger.info(f"⚖️ Strict Balancing: Subsampling all classes to N={min_samples}")

    # balanced_idx = []
    # for label in counts.keys():
    #     idx = np.where(y_tr == label)[0]
    #     selected = np.random.choice(idx, min_samples, replace=False)
    #     balanced_idx.extend(selected)
    
    # X_tr_bal = X_tr[balanced_idx]
    # y_tr_bal = y_tr[balanced_idx]

    logger.info("⚖️ No class balancing: using natural class distribution")
    X_tr_bal = X_tr
    y_tr_bal = y_tr
    # 3. Scaling
    scaler = StandardScaler()
    X_tr_bal = scaler.fit_transform(X_tr_bal)
    X_te = scaler.transform(X_te)

    # 4. Training
    # Using 'multinomial' for 3-class, L2 penalty for stability
    lr_model = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0,class_weight='balanced')
    lr_model.fit(X_tr_bal, y_tr_bal)

    # 5. Evaluation
    y_pred = lr_model.predict(X_te)
    report = classification_report(y_te, y_pred, zero_division=0)
    macro_f1 = f1_score(y_te, y_pred, average='macro')

    logger.info(f"\n--- LR Probe Report (Macro-F1: {macro_f1:.4f}) ---\n{report}")
    
    # 6. Interpretability (Feature Importance)
    # Get average coefficient magnitude across classes for the 'original' features
    # shape: [n_classes, window * features] -> [features]
    coef_abs = np.abs(lr_model.coef_).mean(axis=0).reshape(full_ds.window, -1).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': full_ds.feature_names,
        'Importance': coef_abs
    }).sort_values(by='Importance', ascending=False)
    
    logger.info(f"🔝 Top 5 LR Probe Features: \n{importance_df.head(5).to_string(index=False)}")
    
    return macro_f1, report,importance_df

import matplotlib.pyplot as plt
import seaborn as sns

def plot_f1_sweep(results_df, save_dir):
    """
    Dissertation Tool: 绘制 F1 随标签严格度变化的趋势图。
    帮助论证 'Information Density' (结论 C) 还是 'Simple Rescaling' (结论 A)。
    """
    # 设置绘图风格
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6))
    
    # 转换为数值型确保排序正确
    results_df['threshold_multiplier'] = pd.to_numeric(results_df['threshold_multiplier'])
    results_df['macro_f1'] = pd.to_numeric(results_df['macro_f1'])
    results_df = results_df.sort_values('threshold_multiplier')

    # 核心曲线
    sns.lineplot(
        data=results_df, x='threshold_multiplier', y='macro_f1', 
        marker='o', markersize=8, linewidth=2.5, color='#2c3e50', label='Macro-F1 (LR Probe)'
    )

    # 寻找最优信息点 (Peak)
    best_row = results_df.loc[results_df['macro_f1'].idxmax()]
    plt.axvline(x=best_row['threshold_multiplier'], color='#e74c3c', linestyle='--', alpha=0.6)
    plt.annotate(
        f"Optimal Signal Density\n(F1: {best_row['macro_f1']:.4f})",
        xy=(best_row['threshold_multiplier'], best_row['macro_f1']),
        xytext=(best_row['threshold_multiplier']+0.3, best_row['macro_f1']-0.02),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
        fontsize=10, fontweight='bold', color='#c0392b'
    )

    # 设置标签 (使用 LaTeX 提升学术感)
    plt.title("Impact of Label Strictness on Supervision Information Density", fontsize=14, pad=20)
    plt.xlabel(r"Label Threshold Multiplier ($\lambda$)", fontsize=12)
    plt.ylabel(r"Classification Quality ($Macro-F_1$)", fontsize=12)
    
    # 限制 y 轴范围，突出变化趋势
    margin = 0.05
    plt.ylim(results_df['macro_f1'].min() - margin, results_df['macro_f1'].max() + margin)

    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(save_dir, "dissertation_f1_sweep_plot.png")
    plt.savefig(plot_path, dpi=300) # 300DPI 满足打印出版要求
    plt.close()
    
    print(f"📈 Dissertation plot saved to: {plot_path}")

def run_training(feature_direction_map, logger: logging, data_cfg: DataConfig, train_cfg: TrainConfig, model_cfg, pre_para: common.BaseDefine,prep_output_dir:str, save_dir,experiment:bool):
    # 0. 初始化环境
    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Model: {model_cfg.model_type} version: {model_cfg.model_version}")
    if device.type == 'cuda':
        # 启用 TensorFloat32 (TF32)，5090 的算力吞吐量会大幅提升
        torch.set_float32_matmul_precision('high')

    df = common.load_train_df_from_dir(prep_output_dir)
    kline_interval_ms = common.load_interval_ms_from_dir(prep_output_dir)
    logger.info(f"Using TimeSeriesWindowDataset with window={data_cfg.window} Origin data len {len(df)}...")

    feature_list = list(feature_direction_map.keys())
    label_v_cols = sorted(
        [col for col in df.columns if col.startswith("label_v")],
        key=lambda x: int(x.replace("label_v", ""))
    )
    sweep_results = []
    for label_col in label_v_cols:
        logger.info(f"start training label :{label_col}")
        full_ds = TimeSeriesWindowDataset(
            df=df.copy(), kline_interval_ms=kline_interval_ms, feature_cols=feature_list, label_col=label_col, window=data_cfg.window,
            cache_path=os.path.join(save_dir,"train_cache.pt"), stride =train_cfg.stride, use_cache = False, show_feature_distribution=True
        )
        logger.warning(f"📊 [Dataset Check] Final features used in training ({full_ds.feature_count}):"
                    f"{full_ds.feature_names}")
        x_mem = full_ds.X.element_size() * full_ds.X.nelement() / (1024**2)
        y_mem = full_ds.y.element_size() * full_ds.y.nelement() / (1024**2)
        r_mem = full_ds.returns.element_size() * full_ds.returns.nelement() / (1024**2)

        total_gpu_mem_per_process = x_mem + y_mem + r_mem
        logger.info(f"🚀 Estimated GPU VRAM per process: {total_gpu_mem_per_process:.2f} MB")
        # 对 ic_direction=-1 的特征进行反向（乘以 -1），使其与收益正相关
        if feature_direction_map:
            full_ds.X = apply_feature_direction(full_ds.X, full_ds.feature_names, feature_direction_map, logger)

        # 显存预加载优化
        logger.info(f"Pre-loading entire dataset to {device}...")
        full_ds.X = full_ds.X.to(device)
        full_ds.y = full_ds.y.to(device)
        full_ds.returns = full_ds.returns.to(device) # 之前建议的是 .r，请统一为 .returns
        logger.info("Data loaded to VRAM.")

        M = len(full_ds)
        logger.info(f"Total windows (M) = {M}, window = {data_cfg.window}")

        # 2. 切分数据
        tr_rng, va_rng, te_rng = chrono_split_by_window_ends(M, data_cfg.train_ratio, data_cfg.val_ratio)
        
        lr_f1,report, lr_importance = run_logistic_regression_probe(full_ds, tr_rng, te_rng, logger)
        sweep_results.append({
            "label_column": label_col,
            "threshold_multiplier": int(label_col.replace("label_v", "")) / 10.0, # 将 "v12" 还原为 1.2
            "macro_f1": lr_f1,
            "report":report
        })
    results_df = pd.DataFrame(sweep_results)
    output_path = os.path.join(save_dir, "lr_probe_sweep_results.csv")
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ All sweep tasks completed!")
    logger.info(f"📊 Summary results saved to: {output_path}")
    
    # 打印简要总结
    print("\n" + "="*30)
    print("📈 SWEEP SUMMARY (Macro-F1)")
    print(results_df.to_string(index=False))
    print("="*30)

    plot_f1_sweep(results_df, save_dir)
    return sweep_results

def run_subsampling_probe(feature_direction_map, logger: logging, data_cfg: DataConfig, train_cfg: TrainConfig, model_cfg, pre_para: common.BaseDefine,prep_output_dir:str, save_dir,experiment:bool):
    # 0. 初始化环境
    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Model: {model_cfg.model_type} version: {model_cfg.model_version}")
    if device.type == 'cuda':
        # 启用 TensorFloat32 (TF32)，5090 的算力吞吐量会大幅提升
        torch.set_float32_matmul_precision('high')

    df = common.load_train_df_from_dir(prep_output_dir)
    kline_interval_ms = common.load_interval_ms_from_dir(prep_output_dir)
    logger.info(f"Using TimeSeriesWindowDataset with window={data_cfg.window} Origin data len {len(df)}...")

    feature_list = list(feature_direction_map.keys())
    label_v_cols = sorted(
        [col for col in df.columns if col.startswith("label_v")],
        key=lambda x: int(x.replace("label_v", ""))
    )
    sweep_results = []
    for label_col in label_v_cols:
        logger.info(f"start training label :{label_col}")
        full_ds = TimeSeriesWindowDataset(
            df=df.copy(), kline_interval_ms=kline_interval_ms, feature_cols=feature_list, label_col=label_col, window=data_cfg.window,
            cache_path=os.path.join(save_dir,"train_cache.pt"), stride =train_cfg.stride, use_cache = train_cfg.use_cache, show_feature_distribution=True
        )
        logger.warning(f"📊 [Dataset Check] Final features used in training ({full_ds.feature_count}):"
                    f"{full_ds.feature_names}")
        x_mem = full_ds.X.element_size() * full_ds.X.nelement() / (1024**2)
        y_mem = full_ds.y.element_size() * full_ds.y.nelement() / (1024**2)
        r_mem = full_ds.returns.element_size() * full_ds.returns.nelement() / (1024**2)

        total_gpu_mem_per_process = x_mem + y_mem + r_mem
        logger.info(f"🚀 Estimated GPU VRAM per process: {total_gpu_mem_per_process:.2f} MB")
        # 对 ic_direction=-1 的特征进行反向（乘以 -1），使其与收益正相关
        if feature_direction_map:
            full_ds.X = apply_feature_direction(full_ds.X, full_ds.feature_names, feature_direction_map, logger)

        # 显存预加载优化
        logger.info(f"Pre-loading entire dataset to {device}...")
        full_ds.X = full_ds.X.to(device)
        full_ds.y = full_ds.y.to(device)
        full_ds.returns = full_ds.returns.to(device) # 之前建议的是 .r，请统一为 .returns
        logger.info("Data loaded to VRAM.")

        M = len(full_ds)
        logger.info(f"Total windows (M) = {M}, window = {data_cfg.window}")

        # 2. 切分数据
        tr_rng, va_rng, te_rng = chrono_split_by_window_ends(M, data_cfg.train_ratio, data_cfg.val_ratio)
        
        lr_f1,report, lr_importance = run_logistic_regression_probe(full_ds, tr_rng, te_rng, logger)
        sweep_results.append({
            "label_column": label_col,
            "threshold_multiplier": int(label_col.replace("label_v", "")) / 10.0, # 将 "v12" 还原为 1.2
            "macro_f1": lr_f1,
            "report":report
        })
    results_df = pd.DataFrame(sweep_results)
    output_path = os.path.join(save_dir, "lr_probe_sweep_results.csv")
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ All sweep tasks completed!")
    logger.info(f"📊 Summary results saved to: {output_path}")
    
    # 打印简要总结
    print("\n" + "="*30)
    print("📈 SWEEP SUMMARY (Macro-F1)")
    print(results_df.to_string(index=False))
    print("="*30)

    plot_f1_sweep(results_df, save_dir)
    return sweep_results

def run_fixed_neutral_subsampling_experiment(feature_direction_map, logger, data_cfg, train_cfg, pre_para,prep_output_dir, save_dir):
    # --- 1. 环境准备 ---
    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = common.load_train_df_from_dir(prep_output_dir)
    kline_interval_ms = common.load_interval_ms_from_dir(prep_output_dir)
    feature_list = list(feature_direction_map.keys())
    
    # 获取标签列：从 v01 到 v30
    label_cols = sorted([c for c in df.columns if c.startswith("label_v")], 
                         key=lambda x: int(x.replace("label_v", "")))
    most_strict_col = label_cols[-1]

    # --- 2. 确定全局 Neutral 交集池 ---
    # 只有所有参数都认为是 Neutral 的样本才进入固定池，确保背景纯净
    is_always_neutral = (df[label_cols] == common.Signal.NEUTRAL).all(axis=1)
    neutral_pool_indices = df[is_always_neutral].index.values
    
    # 确定基准数量 N (以最严格标签的 Pos/Neg 最小值为准)
    pos_indices_strict = df[df[most_strict_col] == common.Signal.POSITIVE].index.values
    neg_indices_strict = df[df[most_strict_col] == common.Signal.NEGATIVE].index.values
    N = min(len(pos_indices_strict), len(neg_indices_strict))
    
    logger.info(f"🎯 实验基准 N={N} (源自 {most_strict_col})")
    logger.info(f"🛡️ 全局共识 Neutral 池大小: {len(neutral_pool_indices)}")

    all_results = []

    # --- 3. 开启 10 组重复实验 ---
    for iter_idx in range(1, 11):
        logger.info(f"🌀 [Iteration {iter_idx}/10] 正在生成实验数据集...")
        # 每一组实验使用不同的种子，但该组内的所有参数共享相同的 Neutral 样本
        iter_seed = train_cfg.seed + iter_idx
        np.random.seed(iter_seed)
        
        # 【关键：固定该轮次的中性样本】
        fixed_neutral_idx = np.random.choice(neutral_pool_indices, N, replace=False)

        for label_col in label_cols:
            # 获取当前参数下的趋势样本池
            current_pos_pool = df[df[label_col] == common.Signal.POSITIVE].index.values
            current_neg_pool = df[df[label_col] == common.Signal.NEGATIVE].index.values
            
            # 从当前参数池中随机抽取 N 个
            sampled_pos_idx = np.random.choice(current_pos_pool, N, replace=False)
            sampled_neg_idx = np.random.choice(current_neg_pool, N, replace=False)
            
            # 合并：Fixed Neutral + Variable Trend
            final_indices = np.concatenate([fixed_neutral_idx, sampled_pos_idx, sampled_neg_idx])
            experiment_df = df.loc[final_indices].copy()
            
            # --- 4. 训练与评估 ---
            train_ds = TimeSeriesWindowDataset(
                df=experiment_df, 
                kline_interval_ms=kline_interval_ms, 
                feature_cols=feature_list, 
                label_col=label_col, 
                window=pre_para.candlestick_num,
                use_cache=False, # 抽样数据不建议缓存
                show_feature_distribution=False
            )
            
            # 处理特征方向
            if feature_direction_map:
                train_ds.X = apply_feature_direction(train_ds.X, train_ds.feature_names, feature_direction_map, logger)
            
            # 搬运到 GPU
            train_ds.X, train_ds.y = train_ds.X.to(device), train_ds.y.to(device)
            
            # 简单 Chrono Split (由于已经 shuffle 抽样，此处 split 相当于随机分层)
            M = len(train_ds)
            tr_rng, _, te_rng = chrono_split_by_window_ends(M, data_cfg.train_ratio, data_cfg.val_ratio)
            
            f1, report, _ = run_logistic_regression_probe(train_ds, tr_rng, te_rng, logger)
            
            all_results.append({
                "iteration": iter_idx,
                "label_col": label_col,
                "threshold": int(label_col.replace("label_v", "")) / 10.0,
                "macro_f1": f1,
                "pos_pool_size": len(current_pos_pool)
            })

    # --- 5. 结果汇总 ---
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(save_dir, "fixed_neutral_experiment.csv"), index=False)
    
    summary = results_df.groupby("threshold")["macro_f1"].agg(['mean', 'std']).reset_index()
    print("\n" + "="*50)
    print("📈 FIXED NEUTRAL EXPERIMENT SUMMARY")
    print(summary.to_string(index=False))
    print("="*50)
    
    return results_df
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

# feature_direction_map: 特征名 -> ic_direction (1 正向 / -1 负向)
# 训练前会对 direction=-1 的特征乘以 -1，使其与收益正相关
feature_direction_map = {
    "PVT": -1,
    "BOLL_PB_25": -1,
    "RSI_14": -1,
    "close": -1,
    "KELTNER_MIDDLE_14": -1,
    "low": -1,
    "MOM_20_RV20": -1,
    "DONCHIAN_POS_20": -1,
    "high": -1,
    "OBV": -1,
    "MOM_20_SKIP1": -1,
    "KELTNER_UPPER_14": -1,
    "open": -1,
    "DONCHIAN_DIST_L_20": -1,
    "DONCHIAN_DIST_U_20": 1,
    "MACD_12_26_DIF_PCT": -1,
    "MOM_20": -1,
    "KELTNER_LOWER_14": -1,
    "DONCHIAN_MIDDLE_20": -1,
    "DONCHIAN_LOWER_20": -1,
    "DONCHIAN_UPPER_20": -1,
    "VWAP_7": -1,
    "MFI_25": -1,
    "MOM_10": -1,
    "BOLL_MIDDLE_25": -1,
    "dist_to_high_100": 1,
    "KDJ_K": -1,
    "MA_BAR_S_L": -1,
    "KDJ_D": -1,
    "BOLL_LOWER_25": -1,
    "KDJ_J": -1,
    "vpin_49": 1,
    "MA_BAR_M_L": -1,
    "MACD_12_26_HIST_PCT": -1,
    "BOLL_BW_25": 1,
    "poc_bias_49": -1,
    "BOLL_UPPER_25": -1,
    "MACD_12_26_DIF": -1,
    "vpin_14": 1,
    "MOM_60": -1,
    "D_MA_DAY_S_L": -1,
    "id_factor_20": -1,
    "MACD_12_26_DEA": -1,
    "MACD_12_26_SIG_DIST": -1,
    "VWAP_Bias_7": -1,
    "close_pos": -1,
    "vol_parkinson_100": 1,
    "vol_gk_100": 1,
    "id_factor_100": -1,
    "dist_to_high_20": 1,
    "skew_20": 1,
    "D_MA_BAR_S_L": -1,
    "er_126": 1,
    "imbalance_14": -1,
    "CMF_25": 1,
    "VWAP_BIAS": -1,
    "MACD_12_26_HIST_ACCEL": -1,
    "vol_gk_14": 1,
    "hurst_126": 1,
    "atr_14": 1,
    "vol_parkinson_14": 1,
    "body": 1,
    "body_pct": 1,
    "doji_score": 1,
    "body_mom": 1,
    "imbalance_49": 1,
    "max_range": 1,
    "kurt_100": 1,
    "MACD_12_26_HIST": 1,
    "skew_100": 1,
    "Vol_Trend": 1,
    "poc_bias_14": 1,
    "upper_wick_pct": 1,
    "VOL_ratio_14": 1,
    "kurt_20": 1,
    "ATS": 1,
    "DONCHIAN_BW_20": 1,
    "QAV_SLOPE_49": 1,
    "lower_wick_pct": 1,
    "hurst_14": 1,
    "QAV_SURGE_49": 1,
    "lower_wick": 1,
    "vol_regime_14": 1,
    "wick_bias": 1,
    "trade_density_14": 1,
    "er_14": 1,
    "quote_asset_volume": 1,
    "MA_DAY_S_L": 1,
    "number_of_trades": 1,
    "taker_buy_quote_volume": 1,
    "MA_WEEK_M_L": -1,
    "VOL_MA_14": 1,
    "trade_density_49": 1,
    "upper_wick": 1,
    "volume": 1,
    "taker_buy_base_volume": 1,
}

def main(logger: logging.Logger, train_cfg=TrainConfig(), pre_para=common.BaseDefine(), prep_output_dir = common.DATA_OUT_DIR, save_dir: str = common.TRAIN_OUT_DIR,experiment:bool = False):
    os.makedirs(save_dir, exist_ok=True)

    # 根据 feature_conf_list 从全局 feature_direction_map 补充完整方向信息
    feature_direction_map_filtered = {}
    for feature_name in train_cfg.feature_conf_list:
        # 从全局 feature_direction_map 中查找方向，如果找不到则默认为 1（正向）
        direction = feature_direction_map.get(feature_name, 1)
        feature_direction_map_filtered[feature_name] = direction
    
    logger.info(f"📋 Using {len(feature_direction_map_filtered)} features from feature_conf_list")

    # 1. 数据配置
    d_cfg = DataConfig()

            # 0             1                   2                   3           4               5               6
    m_cfg = [LSTMConfig(), TransformerConfig(), ConvLSTMConfig(), CNNConfig(), XGBoostConfig(), TCNConfig(), MambaConfig()][2]
    # m_cfg.model_version = 1
    
    logger.info(f"Training {m_cfg.model_type}...")
    # return run_training(feature_direction_map_filtered, logger, d_cfg, train_cfg, m_cfg, pre_para,prep_output_dir,save_dir,experiment)
    return run_fixed_neutral_subsampling_experiment(feature_direction_map_filtered, logger, d_cfg, train_cfg,pre_para,prep_output_dir,save_dir)
# ==============================================================================
# 5. 调用入口 (Main Entry)
# ==============================================================================

if __name__ == "__main__":
    logger, _ = common.setup_session_logger(sub_folder='train', file_level = logging.DEBUG)
    begin_time = time.time()
    prep_output_dir = os.path.join(common.PERSISTENCE_DIR, 'dissertation', 'data_process')
    para = common.BaseDefine
    para.candlestick_num =96
    para.predict_num = 6
    para.symbol = 'BTCUSDT'
    para.trading_type = 'spot'
    para.interval = "1h"
    main(logger, pre_para = para, prep_output_dir = prep_output_dir)
    end_time = time.time()
    logger.info(f"Total training time: {(end_time - begin_time)} seconds")