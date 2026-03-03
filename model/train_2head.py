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
    # =========================
    # 一、趋势 / 方向持续（Trend） 描述：价格是否存在延续性
    # =========================
    "MA_WEEK_M_L",        # 长期结构方向（Regime核心）
    "PVT",                    # 量价增强型动量
    "dist_to_high_100",       # 突破型趋势结构
    "id_factor_100",
    "id_factor_20",
    "MFI_999",              # 资金流向极端
    "MFI_99",
    # =========================
    # 二、波动结构（Volatility Regime） 描述：振幅与风险环境
    # =========================
    "vol_gk_100",             # 长期波动
    "vol_gk_14",              # 短期波动
    "skew_100",
    "kurt_100",               # 尾部结构（极端风险）
    # "BOLL_BW_25",           # 需要增益测试后决定
    "RSI_14",                    # 相对强弱指数（动量与过热信号，间接反映波动环境）
    # =========================
    # 三、路径效率 / 市场结构（Efficiency / Regime） 描述：趋势 vs 震荡
    # =========================
    "er_126",                 # 趋势效率比（高质量结构因子）
    # =========================
    # 四、参与强度（Participation / Liquidity） 描述：市场活跃度
    # =========================
    "trade_density_14",       # 连续参与强度
    "vol_event_flag_500",     # 极端成交事件（Regime触发器）
    # =========================
    # 五、订单流 / 失衡（Order Flow） 描述：买卖主导结构
    # =========================
    "vpin_49",                # 中期订单流失衡
    "vpin_14",              # 需要增益测试
    # =========================
    # 六、空间位置结构（Spatial / Price Position） 描述：价格在区间或成本中的位置
    # =========================
    "poc_bias_600",           # 成交密集区偏离（强结构锚点）
    "poc_bias_99",
    "close_pos",              # 区间相对位置
    # =========================
    # 七、K线形态 / 微观博弈（Path Microstructure）
    # =========================
    "upper_wick_pct",
    "lower_wick_pct",
]


@dataclass
class DataConfig:
    label_col: str = "label"
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
    patience: int = 8
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
    logger.info(f"Using TimeSeriesWindowDataset with window={pre_para.candlestick_num} Origin data len {len(df)}...")

    feature_list = list(feature_direction_map.keys())
    full_ds = TimeSeriesWindowDataset(
        df=df, kline_interval_ms=kline_interval_ms, feature_cols=feature_list, label_col=data_cfg.label_col, window=pre_para.candlestick_num,
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
    logger.info(f"Total windows (M) = {M}, window = {pre_para.candlestick_num}")

    # 2. 切分数据
    tr_rng, va_rng, te_rng = chrono_split_by_window_ends(M, data_cfg.train_ratio, data_cfg.val_ratio)
    
    ds_tr = SeqDataset(full_ds.X[tr_rng[0]:tr_rng[1]], full_ds.y[tr_rng[0]:tr_rng[1]], full_ds.returns[tr_rng[0]:tr_rng[1]])
    # More efficient alternative: just pass the sliced tensors
    ds_va = SeqDataset(full_ds.X[va_rng[0]:va_rng[1]], full_ds.y[va_rng[0]:va_rng[1]], full_ds.returns[va_rng[0]:va_rng[1]])
    ds_te = SeqDataset(full_ds.X[te_rng[0]:te_rng[1]], full_ds.y[te_rng[0]:te_rng[1]], full_ds.returns[te_rng[0]:te_rng[1]])
    # 3. 计算权重
    y_tr_np = full_ds.y[tr_rng[0]:tr_rng[1]].cpu().numpy()
    classes = np.unique(y_tr_np)
    #  注入平衡采样逻辑
    # 使用你代码中定义的 get_balanced_sampler (Neutral 50%, Short 25%, Long 25%)
    sampler_tr = get_balanced_sampler(ds_tr) 
    
    cw_balanced = compute_class_weight("balanced", classes=classes, y=y_tr_np)
    class_weights = torch.tensor(cw_balanced, dtype=torch.float32, device=device)
    logger.info(f"Class weights: {dict(zip(classes, cw_balanced))}")

    # 4. DataLoader
    dl_tr = DataLoader(
        ds_tr, 
        batch_size=train_cfg.batch_size, 
        sampler=sampler_tr,      # 使用采样器替代 shuffle
        shuffle=False,           # 使用 sampler 时必须设为 False
        num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=train_cfg.batch_size, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=train_cfg.batch_size, shuffle=False)

    # 5. 构建模型 (参数解包)
    logger.info(f"Initializing model: type={model_cfg.model_type}, version={model_cfg.model_version}")
    
    params = asdict(model_cfg)
    m_type = params.pop('model_type')
    m_ver = params.pop('model_version')
    
    # 默认值修正 logic
    if hasattr(model_cfg, 'max_len') and params['max_len'] is None:
        params['max_len'] = pre_para.candlestick_num

    # 特殊处理 XGBoost
    if m_type == 'xgboost':
        logger.info(f'xgboost device:{device}')
        model = ModelFactory.build_for_training(
            model_type=m_type, model_version=m_ver, device=device,
            input_size=full_ds.feature_count, n_classes=len(classes),
            input_dim=pre_para.candlestick_num * full_ds.feature_count, window_size = pre_para.predict_num,
            xgb_params=params
        )
    else:
        model = ModelFactory.build_for_training(
            model_type=m_type, model_version=m_ver, device=device,
            input_size=full_ds.feature_count, n_classes=len(classes),
            **params
        )
        # model = torch.compile(model)
        # --- 核心修改：实现 gate_lr 差异化学习率 ---
        # 1. 提取参数
        gate_params = []
        backbone_params = []

        for name, param in model.named_parameters():
            if "feature_weighter" in name:
                gate_params.append(param)
            else:
                backbone_params.append(param)

        # 2. 构造参数组
        # 主网络使用 train_cfg.lr (如 3e-4)
        # 加权层使用 train_cfg.gate_lr (如 1e-2)
        param_groups = [
            {"params": backbone_params, "lr": train_cfg.lr},
        ]
        
        if gate_params:
            logger.info(f"⚡ [Differential LR] Setting gate_lr: {train_cfg.gate_lr} for feature_weighter")
            param_groups.append({"params": gate_params, "lr": train_cfg.gate_lr})

        # 3. 重新定义优化器
        optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    # 7. 调用封装好的训练引擎
    logger.info("🚀 Starting training engine...")
    mtl = MTLManager(device, train_cfg)
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
        feature_list = feature_list,
        classes=classes,
        logger=logger,
        mtl_manager = mtl,
        pre_para=pre_para,
        save_dir = save_dir,
    )
    if experiment==False:
        diagnose_confidence(results, model=model,dl_te=dl_te, device=device,logger=logger,save_dir = save_dir)
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
        return self.compute_combined_loss_v2(logits_trig, logits_dir, yb, rb, epoch)

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
    pre_para:common.BaseDefine,
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
            "window": pre_para.candlestick_num,
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
            window=pre_para.candlestick_num,
            model_version_tag=suffix,
        )
        with open(os.path.join(save_dir, f"model_{suffix.lower()}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if suffix == "Best_F1":
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
            
        if train_cfg.model_cfg.use_feature_weighting:
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
            for xb, yb, _ in dl_te:
                xb = xb.to(device)
                fused_preds, fused_probs = model(xb, return_fused=True) 
                
                all_probs.append(fused_probs.cpu().numpy())
                all_preds.append(fused_preds.cpu().numpy())
                all_trues.append(yb.cpu().numpy())

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
    tl, yt, yp, yr = 0.0, [], [], [] # 🌟 增加 yr 用于存储 return_rate

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
    return run_training(feature_direction_map_filtered, logger, d_cfg, train_cfg, m_cfg, pre_para,prep_output_dir,save_dir,experiment)
# ==============================================================================
# 5. 调用入口 (Main Entry)
# ==============================================================================

if __name__ == "__main__":
    logger, _ = common.setup_session_logger(sub_folder='train', file_level = logging.DEBUG)
    begin_time = time.time()
    main(logger)
    end_time = time.time()
    logger.info(f"Total training time: {(end_time - begin_time)} seconds")