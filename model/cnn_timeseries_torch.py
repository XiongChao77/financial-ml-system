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

import argparse, json ,os, sys , math
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
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process import common
from model.data_loader import TimeSeriesWindowDataset
# ========== 实用函数 ==========
def setup_logger():
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 文件
    log_file = os.path.join(current_work_dir,'train_log.txt')
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)

    # 格式
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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
        self.X = torch.from_numpy(X)  # [M, T, F]
        self.y = torch.from_numpy(y)  # [M]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# ---- 因果卷积：左填充，右裁剪，杜绝“看未来” ----
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=True):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size,
                         padding=pad, dilation=dilation, bias=bias)
    def forward(self, x):                      # x: [B,C,T]
        y = super().forward(x)                 # 先做常规padding卷积
        cut = (self.kernel_size[0]-1) * self.dilation[0]
        return y[:, :, :-cut] if cut > 0 else y   # 右裁剪，保持长度 & 因果

# ---- ECA: 高效通道注意力（1D版，通道维上做1D卷积） ----
class ECA1D(nn.Module):
    def __init__(self, channels, gamma=2.0, b=1.0, k_override=None):
        super().__init__()
        if k_override is None:
            k = int(abs((math.log2(channels)/gamma) + b))
            k = k if k % 2 == 1 else k + 1
            k = max(3, k)
        else:
            k = k_override if k_override % 2 == 1 else (k_override + 1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k-1)//2, bias=False)
    def forward(self, x):                  # x: [B,C,T]
        y = x.mean(dim=-1, keepdim=True)   # [B,C,1]
        y = self.conv(y.transpose(1,2))    # [B,1,C]
        a = torch.sigmoid(y.transpose(1,2))# [B,C,1]
        return x * a

# ---- 最近更重要：指数衰减加权池化（替代 mean） ----
class TimeDecayPool1D(nn.Module):
    def __init__(self, tau=16.0, learnable=False):
        super().__init__()
        if learnable:
            self.tau = nn.Parameter(torch.tensor(float(tau)))
        else:
            self.register_buffer("tau", torch.tensor(float(tau)))
        self.learnable = learnable
    def forward(self, x):                  # x: [B,C,T]
        B, C, T = x.shape
        t = torch.arange(T, device=x.device, dtype=x.dtype)         # 0..T-1
        tau = self.tau if self.learnable else self.tau.to(x.device)
        w = torch.exp(-(T - 1 - t) / (tau + 1e-8))                  # 近端更大
        w = w / (w.sum() + 1e-12)
        return (x * w.view(1,1,T)).sum(dim=-1)                      # [B,C]

# ---- 整体模型：因果Inception -> 因果融合卷积 -> ECA -> 加权池化 -> FC ----
class CNN1D(nn.Module):
    """
    架构: Causal Inception(k=5/21) -> Causal conv3 -> ECA -> TimeDecayPool -> FC
    - 因果卷积：不利用未来信息
    - ECA：按通道自适应重标定
    - 最近更重要：时间指数衰减加权池化
    输入: [B, T, F]   输出: logits [B, n_classes]
    """
    def __init__(self, channel=9, n_classes=3, p_drop=0.3, tau=16.0):
        super().__init__()
        # 并行因果卷积分支（短/长核）
        self.conv_small = CausalConv1d(channel, 64, kernel_size=5, dilation=1, bias=False)
        self.bn_small   = nn.BatchNorm1d(64)
        self.conv_large = CausalConv1d(channel, 64, kernel_size=21, dilation=1, bias=False)
        self.bn_large   = nn.BatchNorm1d(64)

        # concat 后的 ECA（先做一次通道筛选）
        self.eca_after_concat = ECA1D(channels=128)

        # 融合卷积也使用因果版本，避免再次引入未来
        self.conv_post = CausalConv1d(128, 128, kernel_size=3, dilation=1, bias=False)
        self.bn_post   = nn.BatchNorm1d(128)

        # 融合后的第二次 ECA（可保留或去掉做对比）
        self.eca_after_post = ECA1D(channels=128)

        # 最近更重要的池化
        self.tpool = TimeDecayPool1D(tau=tau, learnable=False)

        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):                  # x: [B,T,F]
        x = x.transpose(1, 2)              # -> [B,F,T]

        s = F.relu(self.bn_small(self.conv_small(x)))   # [B,64,T]
        l = F.relu(self.bn_large(self.conv_large(x)))   # [B,64,T]
        out = torch.cat([s, l], dim=1)                  # [B,128,T]

        out = self.eca_after_concat(out)                # [B,128,T]
        out = F.relu(self.bn_post(self.conv_post(out))) # [B,128,T]
        out = self.eca_after_post(out)                  # [B,128,T]

        out = out.mean(dim=-1) #和tpool二选一
        # out = self.tpool(out)                           # [B,128] (最近更重要)
        out = self.dropout(out)
        return self.fc(out)                             # [B,n_classes]

class CostSensitiveLoss(nn.Module):
    def __init__(self, penalty_matrix, base_weights=None, lambda_cost=1.0):
        super().__init__()
        self.register_buffer("C", torch.tensor(penalty_matrix, dtype=torch.float32))
        self.lambda_cost = float(lambda_cost)
        self.base_ce = nn.CrossEntropyLoss(
            weight=(torch.tensor(base_weights, dtype=torch.float32) if base_weights is not None else None)
        )

    def forward(self, logits, targets):
        # Cross-Entropy 部分（可带类别权重）
        ce = self.base_ce(logits, targets)

        # 期望代价部分：sum_j C[true, j] * p_j
        probs = torch.softmax(logits, dim=1)               # [B, C]
        C_true = self.C.index_select(0, targets)           # [B, C]
        exp_cost = (C_true * probs).sum(dim=1).mean()      # 标量

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
        return float('nan'), np.array([]), np.array([])
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    return total_loss/len(loader.dataset), y_true, y_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=False, help="按时间升序的 CSV 文件路径")
    ap.add_argument("--feature_cols", default="", help="逗号分隔;留空则用默认9列 + 其它数值列")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--window", type=int, default=common.candlestick_num)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default= 128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger  = setup_logger()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device:{device}")
    is_cude_available = device == 'cuda'

    # 1) 读数据（需按时间升序）
    data_path = common.train_data_path
    df = pd.read_csv(data_path)
    split_idx = int(len(df) * common.model_train_rate)
    # 切分数据
    train_df = df.iloc[:split_idx]

    # 2) 选择特征列
    if args.feature_cols.strip():
        feat_cols = [c.strip() for c in args.feature_cols.split(",")]
    else:
        feat_cols = [c for c in common.DEFAULT_FEATURES if c in df.columns]

    # 4) 窗口化 -> [M, T, F], [M]
    T = args.window
    logger.info(f"Using TimeSeriesWindowDataset for windowing and scaling...")
    
    # 实例化 TimeSeriesWindowDataset，它在内部完成了窗口划分和 t=0 缩放
    feat_cols = [col for col in df.columns]
    logger.info(f"Features num:{len(feat_cols)},: {feat_cols}") # 可选：打印查看
    full_ds = TimeSeriesWindowDataset(
        df=df, 
        feature_cols=feat_cols, 
        label_col=args.label_col, 
        window=T
    )

    # ========== 【新增】调用保存 Debug 数据 ==========
    # 保存目录设置在 exported_project_files/model/debug_data 下
    if False:
        debug_dir = os.path.join(current_work_dir, "debug_data")
        full_ds.save_debug_data(debug_dir)
        exit()

    if False:
        full_ds.inspect_final_data()
        exit()
    # ===============================================

    # # === 【新增】核心优化：全量数据预加载到 GPU ===
    # logger.info(f"Pre-loading entire dataset to {device}...")
    # # 直接修改 Dataset 内部的 Tensor，将其移动到 GPU
    # full_ds.X = full_ds.X.to(device) 
    # full_ds.y = full_ds.y.to(device)
    # logger.info("Data loaded to VRAM.")

    # 可用窗口数量 M
    M = len(full_ds)
    logger.info(f"Total windows (M) = {M}, window = {T}, F = {len(feat_cols)}")

    # 3) 按“窗口末端”做时间切分，并构建一次性 Dataset/DataLoader
    tr_rng, va_rng, te_rng = chrono_split_by_window_ends(M, args.train_ratio, args.val_ratio)
    s_tr, e_tr = tr_rng
    s_va, e_va = va_rng
    s_te, e_te = te_rng

    ds_tr = SeqDataset(full_ds.X[s_tr:e_tr].numpy(), full_ds.y[s_tr:e_tr].numpy())
    ds_va = SeqDataset(full_ds.X[s_va:e_va].numpy(), full_ds.y[s_va:e_va].numpy())
    ds_te = SeqDataset(full_ds.X[s_te:e_te].numpy(), full_ds.y[s_te:e_te].numpy())

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  pin_memory=is_cude_available)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, pin_memory=is_cude_available)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, pin_memory=is_cude_available)

    # 8) 类别权重（直接用 y_tr 计算，无需遍历 DataLoader）
    y_tr = full_ds.y[s_tr:e_tr].numpy()
    classes = np.unique(y_tr)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weights = torch.tensor(cw, dtype=torch.float32, device=device)
    logger.info("Class weights: {}".format({int(c): float(w) for c, w in zip(classes, cw)}))

    # 9) 模型/优化器/调度器（原逻辑保持不变）
    feat_cols = full_ds.feature_names
    channel = full_ds.feature_count
    logger.info(f"Final Features num:{len(feat_cols)},: {feat_cols}") # 可选：打印查看
    model = CNN1D(channel=channel, n_classes=len(classes), p_drop=args.dropout).to(device)
    criterion = CostSensitiveLoss(
        penalty_matrix=penalty_matrix,
        base_weights=class_weights.cpu().numpy(),
        lambda_cost=0
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # 10) 训练循环（早停基于 val_loss）
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, args.epochs+1):
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

            tr_loss += loss.item() * xb.size(0)
            tr_total += xb.size(0)
        tr_loss /= max(1, tr_total)

        va_loss, yv_true, yv_pred = eval_epoch(model, dl_va, device, criterion)
        scheduler.step(va_loss)
        va_f1 = f1_score(yv_true, yv_pred, average="macro") if len(yv_true) else float("nan")
        logger.info(f"Epoch {epoch:03d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} | va_macroF1 {va_f1:.4f}")

        if va_loss < best_val - 1e-6:
            best_val, best_state, wait = va_loss, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= args.patience:
                logger.info("Early stopping."); break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 11) 测试集评估 + 保存
    te_loss, yt_true, yt_pred = eval_epoch(model, dl_te, device, criterion)
    logger.info("\n=== Test Report ===")
    logger.info(classification_report(yt_true, yt_pred, digits=4))
    logger.info("Test macro-F1:{}".format(f1_score(yt_true, yt_pred, average="macro")))

    # 计算测试集中每个标签的真实占比
    # 假设你的真实标签在 yt_true 里
    counts = Counter(yt_true)
    total = sum(counts.values())
    classes_sorted = sorted(counts.keys())
    true_pct = {c: counts[c]/total for c in classes_sorted}

    logger.info("\n=== True label proportion (Test set) ===")
    for c in classes_sorted:
        logger.info(f"label {c}: {counts[c]} samples, {true_pct[c]:.4f} of total")

    cm = confusion_matrix(yt_true, yt_pred, labels=classes)
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]) \
      .to_csv(os.path.join(current_work_dir,"confmat_cnn.csv"), index=True)
    # logger.info("Saved confusion matrix -> confmat_cnn.csv")

    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes.tolist(),
        "channel": channel,
        "window": T,
         "feature_cols": feat_cols,
         "label_col": args.label_col
    }, os.path.join(current_work_dir,"cnn_timeseries_torch_model.pt"))
    meta = {
        "feature_cols": feat_cols,
        "label_col": args.label_col,
        "classes": classes.tolist(),
        "window": T
    }
    with open(os.path.join(current_work_dir,"cnn_timeseries_torch_meta.json"),"w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # logger.info("Saved model -> cnn_timeseries_torch_model.pt")
    # logger.info("Saved meta  -> cnn_timeseries_torch_meta.json")

if __name__ == "__main__":
    main()
