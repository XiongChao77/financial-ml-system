#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D-CNN (causal) for 256x9 time-series -> 3-class trend classification
--------------------------------------------------------------------
- 输入: 连续时间序列 CSV（按时间升序），包含 9 个数值特征列 + label(0/1/2)
- 窗口: 每个样本为 256×9（末端时刻的 label 作为该样本标签）
- 切分: 按时间顺序 70%/15%/15% -> Train/Val/Test
- 相对化: 每个窗口内，价格组与成交量组分别按最大值缩放到100（按列名分组）
- 模型: 因果卷积 1D-CNN + GAP + FC
- 评估: classification_report、macro-F1、混淆矩阵
"""

import argparse, json, os, sys, math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import logging
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from data_process import common
from model.data_loader import TimeSeriesWindowDataset
#   model
from model.cnn import CNN1D
from model.lstm import LSTM1D
from model.transformer_v1 import Transformer1D
from model.transformer_v2 import Transformer1D_V2
from model.xgb import XGBoostAdapter

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    创建一个带有 warmup 的学习率调度器。
    前 num_warmup_steps 步线性增加，之后线性减少。
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def chrono_split_idx(n: int, train_ratio=0.7, val_ratio=0.15):
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    tr = np.arange(0, n_tr)
    va = np.arange(n_tr, n_tr + n_va)
    te = np.arange(n_tr + n_va, n)
    return tr, va, te


def chrono_split_by_window_ends(M: int, train_ratio=0.7, val_ratio=0.15):
    """返回 (tr_start,tr_stop), (va_start,va_stop), (te_start,te_stop) on window-ends."""
    n_tr = int(M * train_ratio)
    n_va = int(M * val_ratio)
    tr = (0, n_tr)
    va = (n_tr, n_tr + n_va)
    te = (n_tr + n_va, M)
    return tr, va, te


class SeqDataset(Dataset):
    def __init__(self, X, y):
        # 兼容逻辑：如果是 Tensor 直接用，否则从 Numpy 转换
        self.X = X if torch.is_tensor(X) else torch.from_numpy(X)
        self.y = y if torch.is_tensor(y) else torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class CostSensitiveLoss(nn.Module):
    def __init__(self, penalty_matrix, base_weights=None, lambda_cost=1.0):
        super().__init__()
        self.register_buffer("C", torch.tensor(penalty_matrix, dtype=torch.float32))
        self.lambda_cost = float(lambda_cost)
        self.base_ce = nn.CrossEntropyLoss(
            weight=(
                torch.tensor(base_weights, dtype=torch.float32)
                if base_weights is not None
                else None
            )
        )

    def forward(self, logits, targets):
        # Cross-Entropy 部分（可带类别权重）
        ce = self.base_ce(logits, targets)

        # 期望代价部分：sum_j C[true, j] * p_j
        probs = torch.softmax(logits, dim=1)  # [B, C]
        C_true = self.C.index_select(0, targets)  # [B, C]
        exp_cost = (C_true * probs).sum(dim=1).mean()  # 标量

        return ce + self.lambda_cost * exp_cost


penalty_matrix = [
    [1.0, 1, 1],  # true=0, pred=0/1/2
    [1.0, 1.0, 1.0],  # true=1, pred=0/1/2
    [1, 1, 1.0],  # true=2, pred=0/1/2
]


# ========== 训练/评估 ==========
@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_true.append(yb.cpu().numpy())
    if not y_true:
        return float("nan"), np.array([]), np.array([])
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return total_loss / len(loader.dataset), y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=False, help="按时间升序的 CSV 文件路径")
    ap.add_argument(
        "--feature_cols", default="", help="逗号分隔;留空则用默认9列 + 其它数值列"
    )
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--window", type=int, default=common.CANDLESTICK_NUM)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128*2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    # === 模型选择 ===
    ap.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["cnn", "lstm", "transformer, xgboost"], # 【修改】添加 transformer
        help="Model architecture: cnn, lstm, or transformer",
    )
    ap.add_argument("--lstm_hidden", type=int, default=64, help="LSTM 隐藏层维度")
    ap.add_argument("--lstm_layers", type=int, default=2, help="LSTM 层数")
    ap.add_argument("--bidirectional", action='store_true', help="使用双向 LSTM")  #store_true/store_false
    # === 【新增】Transformer 参数 ===
    ap.add_argument("--trans_d_model", type=int, default=64, help="Transformer 内部维度 (d_model)")
    ap.add_argument("--trans_nhead", type=int, default=8, help="Attention 头数 (d_model 必须能被此数整除)")
    ap.add_argument("--trans_layers", type=int, default=5, help="Transformer Encoder 层数")
    ap.add_argument("--trans_dim_feedforward", type=int, default=256, help="FFN 中间层维度通常是 4倍 d_model")
    # ================
    #  xgboost 特有参数
    ap.add_argument("--xgb_depth", type=int, default=6)
    ap.add_argument("--xgb_estimators", type=int, default=100)
    args = ap.parse_args()

    logger = common.setup_logger(log_name='train', log_path=os.path.join(common.TEMPORARY_DIR, 'training.log'))
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device:{device}")
    is_cude_available = device == "cuda"

    # 1) 读数据（需按时间升序）
    data_path = common.train_data_path
    df = pd.read_csv(data_path)

    # 4) 窗口化 -> [M, T, F], [M]
    T = args.window
    logger.info(f"Using TimeSeriesWindowDataset for windowing and scaling...")
    
    # 实例化 TimeSeriesWindowDataset，它在内部完成了窗口划分和 t=0 缩放
    feat_cols = [col for col in df.columns]
    logger.info(f"Features num:{len(feat_cols)},: {feat_cols}")  # 可选：打印查看
    full_ds = TimeSeriesWindowDataset(
        df=df, feature_cols=feat_cols, label_col=args.label_col, window=T
    )

    # ========== 【新增】调用保存 Debug 数据 ==========
    # 保存目录设置在 exported_project_files/model/debug_data 下
    if False:
        debug_dir = os.path.join(common.TEMPORARY_DIR, "debug_data")
        full_ds.save_debug_data(debug_dir)
        exit()

    if False:
        full_ds.inspect_final_data()
        exit()
    # ===============================================

    # 可用窗口数量 M
    M = len(full_ds)
    logger.info(f"Total windows (M) = {M}, window = {T}, F = {len(feat_cols)}")

    # 3) 按“窗口末端”做时间切分，并构建一次性 Dataset/DataLoader
    tr_rng, va_rng, te_rng = chrono_split_by_window_ends(
        M, args.train_ratio, args.val_ratio
    )
    s_tr, e_tr = tr_rng
    s_va, e_va = va_rng
    s_te, e_te = te_rng

    # 8) 类别权重（直接用 y_tr 计算，无需遍历 DataLoader）
    y_tr = full_ds.y[s_tr:e_tr].numpy()
    classes = np.unique(y_tr)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weights = torch.tensor(cw, dtype=torch.float32, device=device) # <--- 立即移到 GPU
    logger.record(
        "Class weights: {}".format({int(c): float(w) for c, w in zip(classes, cw)})
    )

    # # === 【新增】核心优化：全量数据预加载到 GPU ===
    logger.info(f"Pre-loading entire dataset to {device}...")
    # 直接修改 Dataset 内部的 Tensor，将其移动到 GPU
    # full_ds.X = full_ds.X.to(device)  # too many data
    # full_ds.y = full_ds.y.to(device)
    logger.info("Data loaded to VRAM.")
    
    ds_tr = SeqDataset(full_ds.X[s_tr:e_tr], full_ds.y[s_tr:e_tr]) 
    ds_va = SeqDataset(full_ds.X[s_va:e_va], full_ds.y[s_va:e_va])
    ds_te = SeqDataset(full_ds.X[s_te:e_te], full_ds.y[s_te:e_te])

    train_workers = 8
    use_pin_memory = not ds_tr.X.is_cuda
    dl_tr = DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True, pin_memory=use_pin_memory, num_workers=train_workers, persistent_workers=True
    )
    dl_va = DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False, pin_memory=use_pin_memory, num_workers=train_workers, persistent_workers=True
    )
    dl_te = DataLoader(
        ds_te, batch_size=args.batch_size, shuffle=False, pin_memory=use_pin_memory, num_workers=train_workers, persistent_workers=True
    )

    # 9) 模型/优化器/调度器（原逻辑保持不变）
    feat_cols = full_ds.feature_names
    channel = full_ds.feature_count
    logger.warning(f"Final Features num:{len(feat_cols)},: {feat_cols}")  # 可选：打印查看
    if args.model_type == "lstm":
        logger.record(f"Initializing LSTM (hidden={args.lstm_hidden}, layers={args.lstm_layers}, bidirectional={args.bidirectional})...")
        model = LSTM1D(
            input_size=channel,
            hidden_size=args.lstm_hidden,
            num_layers=args.lstm_layers,
            n_classes=len(classes),
            p_drop=args.dropout,
            bidirectional=args.bidirectional
        ).to(device)
    elif args.model_type == "transformer": # 【新增】
        logger.record(f"Initializing Transformer (d_model={args.trans_d_model}, nhead={args.trans_nhead}, layers={args.trans_layers})...")
        # 检查 d_model 是否能被 nhead 整除
        if args.trans_d_model % args.trans_nhead != 0:
            raise ValueError(f"trans_d_model ({args.trans_d_model}) must be divisible by trans_nhead ({args.trans_nhead})")
            
        model = Transformer1D_V2(
            input_size=channel,
            n_classes=len(classes),
            d_model=args.trans_d_model,
            nhead=args.trans_nhead,
            num_layers=args.trans_layers,
            dim_feedforward=args.trans_dim_feedforward,
            dropout=args.dropout,
            max_len=args.window # 位置编码的最大长度需 >= 窗口长度
        ).to(device)
    elif args.model_type == "xgboost":
        logger.info(f"Initializing XGBoost (depth={args.xgb_depth}, est={args.xgb_estimators})...")
        # 计算展平后的维度 Input Dim = Window * Features
        input_dim = T * len(feat_cols)
        model = XGBoostAdapter(
            input_dim=input_dim,
            n_classes=len(classes),
            params={
                'max_depth': args.xgb_depth,
                'n_estimators': args.xgb_estimators,
                'learning_rate': args.lr
            }
        )
    else:
        logger.record("Initializing CNN...")
        model = CNN1D(
            channel=channel, 
            n_classes=len(classes), 
            p_drop=args.dropout
        ).to(device)
    criterion = CostSensitiveLoss(
        penalty_matrix=penalty_matrix,
        base_weights=class_weights.cpu().numpy(),
        lambda_cost=0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.model_type == "transformer":
        # 1. 计算总步数
        # total_steps = epochs * steps_per_epoch
        steps_per_epoch = len(dl_tr)
        total_steps = args.epochs * steps_per_epoch

        # 2. 设定 Warmup 步数 (通常占总步数的 5% ~ 10%)
        warmup_steps = int(total_steps * 0.1) 

        # 3. 替换掉原来的 ReduceLROnPlateau
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...) # 删掉这行
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4
        )

    # 10) 训练循环（早停基于 val_loss）
    best_f1, best_state, wait = float("-inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_total = 0.0, 0
        for xb, yb in tqdm(dl_tr, desc=f"Epoch {epoch}/{args.epochs}", ncols=100):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if args.model_type == "transformer":
                scheduler.step() # 注意：Warmup 调度器是在每个 step 更新，而不是每个 epoch

            tr_loss += loss.item() * xb.size(0)
            tr_total += xb.size(0)
        tr_loss /= max(1, tr_total)

        va_loss, yv_true, yv_pred = eval_epoch(model, dl_va, device, criterion)
        va_f1 = (
            f1_score(yv_true, yv_pred, average="macro")
            if len(yv_true)
            else 0.0 # 避免 nan 导致比较出错
        )
        if args.model_type != "transformer":
            scheduler.step(va_f1)
        logger.info(
            f"Epoch {epoch:03d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} | va_macroF1 {va_f1:.4f}"
        )

        if va_f1 > best_f1 + 1e-4:  
            best_f1 = va_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                logger.record("Early stopping. epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 11) 测试集评估 + 保存
    te_loss, yt_true, yt_pred = eval_epoch(model, dl_te, device, criterion)
    logger.record("\n=== Test Report ===")
    logger.record(classification_report(yt_true, yt_pred, digits=4))
    logger.record("Test macro-F1:{}".format(f1_score(yt_true, yt_pred, average="macro")))

    # 计算测试集中每个标签的真实占比
    # 假设你的真实标签在 yt_true 里
    counts = Counter(yt_true)
    total = sum(counts.values())
    classes_sorted = sorted(counts.keys())
    true_pct = {c: counts[c] / total for c in classes_sorted}

    logger.record("\n=== True label proportion (Test set) ===")
    for c in classes_sorted:
        logger.record(f"label {c}: {counts[c]} samples, {true_pct[c]:.4f} of total")

    cm = confusion_matrix(yt_true, yt_pred, labels=classes)
    pd.DataFrame(
        cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]
    ).to_csv(os.path.join(common.TEMPORARY_DIR, "confmat_cnn.csv"), index=True)
    logger.info("Saved confusion matrix -> confmat_cnn.csv")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": classes.tolist(),
            "channel": channel,
            "window": T,
            "feature_cols": feat_cols,
            "label_col": args.label_col,
        },
        os.path.join(common.TEMPORARY_DIR, "torch_model_train_info.pt"),
    )
    meta = {
        "feature_cols": feat_cols,
        "label_col": args.label_col,
        "classes": classes.tolist(),
        "window": T,
    }
    with open(
        os.path.join(common.TEMPORARY_DIR, "torch_model_train_meta.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# 1. 保存模型权重 (保持不变)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes.tolist(),
        "channel": channel,
        "window": T,
        "feature_cols": feat_cols,
        "label_col": args.label_col
    }, os.path.join(common.TEMPORARY_DIR, "torch_model_train_info.pt"))

    # 2. 【修改此处】保存元数据 (Meta JSON)
    # 我们将 args.model_type 和 LSTM 参数写入这里
    meta = {
        "model_type": args.model_type,  # 核心：记录是 'cnn' 还是 'lstm'
        "feature_cols": feat_cols,
        "label_col": args.label_col,
        "classes": classes.tolist(),
        "window": T,
        # 如果是 LSTM，记录其架构参数，方便推理时重建
        "lstm_hidden": getattr(args, 'lstm_hidden', 64),
        "lstm_layers": getattr(args, 'lstm_layers', 2),
        "bidirectional": getattr(args, 'bidirectional', False), # 保存这个状态
        "trans_d_model": getattr(args, 'trans_d_model', 128),
        "trans_nhead": getattr(args, 'trans_nhead', 4),
        "trans_layers": getattr(args, 'trans_layers', 2),
        "trans_dim_feedforward": getattr(args, 'trans_dim_feedforward', 512)
    }

    with open(os.path.join(common.TEMPORARY_DIR, "torch_model_train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Saved model -> torch_model_train_info.pt")
    logger.info("Saved meta  -> torch_model_train_meta.json")


if __name__ == "__main__":
    main()
