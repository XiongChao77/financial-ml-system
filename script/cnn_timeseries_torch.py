#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D-CNN (causal) for 256x9 time-series -> 3-class trend classification
--------------------------------------------------------------------
- 输入: 连续时间序列 CSV（按时间升序），包含 9 个数值特征列 + label(0/1/2)
- 窗口: 每个样本为 256×9（末端时刻的 label 作为该样本标签）
- 切分: 按时间顺序 70%/15%/15% -> Train/Val/Test
- 标准化: 仅用 Train 拟合 StandardScaler，再用于 Val/Test
- 模型: 因果卷积 1D-CNN + GAP + FC
- 评估: classification_report、macro-F1、混淆矩阵
"""

import argparse, json ,os, sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from data import common
# ====== 你可以按需要修改的默认特征列（9维）======
DEFAULT_FEATURES = [
    "Open_price","High_price","Low_price","Close_price",
    "Volume","Quote_asset_volume","Number_of_trades",
    "buy_base_volume","buy_quote_volume"
]

# ========== 实用函数 ==========
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

def make_windows(arr_2d: np.ndarray, labels_1d: np.ndarray, T: int):
    """
    arr_2d: [N, F] 连续序列（按时间升序）
    labels_1d: [N] 与每个时刻对齐的标签(0/1/2)，末端对齐
    返回:
      X: [M, T, F]
      y: [M]
    """
    X, y = [], []
    for end in range(T-1, len(arr_2d)):
        X.append(arr_2d[end-T+1:end+1])  # 以 end 为窗口末端
        y.append(labels_1d[end])
    return np.asarray(X, np.float32), np.asarray(y, np.int64)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # [M, T, F]
        self.y = torch.from_numpy(y)  # [M]
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# ========== 模型定义：因果卷积 CNN ==========
class CausalConv1d(nn.Conv1d):
    """
    通过 padding=(k-1)*dilation 并在 forward 时剪裁右侧，实现因果卷积：
    输出在时刻 t 只依赖 <= t 的输入。
    """
    def __init__(self, in_ch, out_ch, k, dilation=1):
        pad = (k - 1) * dilation
        super().__init__(in_ch, out_ch, k, padding=pad, dilation=dilation)
    def forward(self, x):                 # x: [B, C, T]
        out = super().forward(x)
        return out[..., :x.size(-1)]      # 保持长度 & 因果性

class CNN1D(nn.Module):
    """
    架构: (因果卷积堆叠) -> GAP -> 全连接 -> logits
    通道: 9 -> 64 -> 64 -> 128 -> GAP -> 128 -> 64 -> 3
    """
    def __init__(self, F=9, n_classes=3, p_drop=0.3):
        super().__init__()
        self.feat = nn.Sequential(
            CausalConv1d(F, 64, k=5, dilation=1), nn.ReLU(),
            nn.BatchNorm1d(64),
            CausalConv1d(64,64, k=9, dilation=1), nn.ReLU(),
            nn.BatchNorm1d(64),
            CausalConv1d(64,128,k=9, dilation=1), nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)      # -> [B, 128, 1]
        )
        self.head = nn.Sequential(
            nn.Flatten(),                 # -> [B, 128]
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, n_classes)      # logits
        )
    def forward(self, x):                  # x: [B, T, F]
        x = x.transpose(1, 2)             # -> [B, F, T]
        h = self.feat(x)                  # -> [B, 128, 1]
        return self.head(h)               # -> [B, 3]

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
    ap.add_argument("--batch_size", type=int, default=common.candlestick_num)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) 读数据（需按时间升序）
    data_path = os.path.join(current_work_dir,'..', 'data','data.csv')
    df = pd.read_csv(data_path)

    # 2) 选择特征列
    if args.feature_cols.strip():
        feat_cols = [c.strip() for c in args.feature_cols.split(",")]
    else:
        feat_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
        # 把其它数值型列也一并加入（不含 label）
        extras = [c for c in df.columns if c not in feat_cols + [args.label_col]]
        extras = [c for c in extras if pd.api.types.is_numeric_dtype(df[c])]
        feat_cols += extras

    # 3) 原始数组与标签
    A = df[feat_cols].astype(np.float32).to_numpy()   # [N, F]
    y_all = df[args.label_col].astype(int).to_numpy() # [N]

    # 4) 窗口化 -> [M, T, F], [M]
    T = args.window
    X_all, y_all = make_windows(A, y_all, T=T)  # 末端对齐标签
    M = len(X_all)
    print(f"Total windows (M) = {M}, window = {T}, F = {X_all.shape[-1]}")

    # 5) 按“窗口末端时刻”时间切分
    m_tr = int(M * args.train_ratio)
    m_va = int(M * args.val_ratio)
    X_tr, X_va, X_te = X_all[:m_tr], X_all[m_tr:m_tr+m_va], X_all[m_tr+m_va:]
    y_tr, y_va, y_te = y_all[:m_tr], y_all[m_tr:m_tr+m_va], y_all[m_tr+m_va:]

    # 6) 标准化（仅在 Train 拟合；逐特征、跨时间共享）
    scaler = StandardScaler()
    X_tr_2d = X_tr.reshape(-1, X_tr.shape[-1])   # [m_tr*T, F]
    X_va_2d = X_va.reshape(-1, X_va.shape[-1])
    X_te_2d = X_te.reshape(-1, X_te.shape[-1])

    X_tr = scaler.fit_transform(X_tr_2d).reshape(-1, T, X_tr.shape[-1])
    X_va = scaler.transform(X_va_2d).reshape(-1, T, X_va.shape[-1])
    X_te = scaler.transform(X_te_2d).reshape(-1, T, X_te.shape[-1])

    # 7) DataLoader
    dl_tr = DataLoader(SeqDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(SeqDataset(X_va, y_va), batch_size=args.batch_size, shuffle=False)
    dl_te = DataLoader(SeqDataset(X_te, y_te), batch_size=args.batch_size, shuffle=False)

    # 8) 类别权重（平衡不均衡）
    classes = np.unique(y_tr)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weights = torch.tensor(cw, dtype=torch.float32, device=device)
    print("Class weights:", {int(c): float(w) for c, w in zip(classes, cw)})

    # 9) 模型/优化器/调度器
    F = X_tr.shape[-1]
    model = CNN1D(F=F, n_classes=len(classes), p_drop=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)  # 兼容老版本，不加 verbose

    # 10) 训练循环（早停基于 val_loss）
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss, tr_total = 0.0, 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
            tr_total += xb.size(0)
        tr_loss /= max(1, tr_total)

        va_loss, yv_true, yv_pred = eval_epoch(model, dl_va, device, criterion)
        scheduler.step(va_loss)
        va_f1 = f1_score(yv_true, yv_pred, average="macro") if len(yv_true) else float("nan")
        print(f"Epoch {epoch:03d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} | va_macroF1 {va_f1:.4f}")

        if va_loss < best_val - 1e-6:
            best_val, best_state, wait = va_loss, {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping."); break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 11) 测试集评估 + 保存
    te_loss, yt_true, yt_pred = eval_epoch(model, dl_te, device, criterion)
    print("\n=== Test Report ===")
    print(classification_report(yt_true, yt_pred, digits=4))
    print("Test macro-F1:", f1_score(yt_true, yt_pred, average="macro"))

    # 计算测试集中每个标签的真实占比
    # 假设你的真实标签在 yt_true 里
    counts = Counter(yt_true)
    total = sum(counts.values())
    classes_sorted = sorted(counts.keys())
    true_pct = {c: counts[c]/total for c in classes_sorted}

    print("\n=== True label proportion (Test set) ===")
    for c in classes_sorted:
        print(f"label {c}: {counts[c]} samples, {true_pct[c]:.4f} of total")



    cm = confusion_matrix(yt_true, yt_pred, labels=classes)
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]) \
      .to_csv("confmat_cnn.csv", index=True)
    print("Saved confusion matrix -> confmat_cnn.csv")

    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes.tolist(),
        "F": F,
        "window": T
    }, "cnn_timeseries_torch_model.pt")
    meta = {
        "feature_cols": feat_cols,
        "label_col": args.label_col,
        "classes": classes.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "window": T
    }
    with open("cnn_timeseries_torch_meta.json","w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved model -> cnn_timeseries_torch_model.pt")
    print("Saved meta  -> cnn_timeseries_torch_meta.json")

if __name__ == "__main__":
    main()
