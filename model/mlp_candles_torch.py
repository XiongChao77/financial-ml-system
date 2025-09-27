#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch MLP for candle 3-class classification (chronological split)
-------------------------------------------------------------------
- 输入: CSV，包含数值特征 + label(0/1/2)
- 切分: 按时间顺序 70%/15%/15% -> Train/Val/Test
- 预处理: 仅在 Train 上拟合 StandardScaler，Val/Test 复用
- 模型: MLP(BN+ReLU+Dropout)，CrossEntropyLoss + class weights
- 评估: classification_report、macro-F1、混淆矩阵
- 保存: 模型权重 .pt、混淆矩阵 .csv、元信息 meta.json
"""

import argparse, json ,os
from typing import List
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---- 默认特征列（自动补充其它数值列）----
DEFAULT_FEATURES = [
    "open","high","low","close",
    "volume","quote_asset_volume","number_of_trades",
    "taker_buy_base_volume","taker_buy_quote_volume"
]

# ---------------- 实用函数 ----------------
def set_seed(seed: int = 42):
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

class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 3, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 256, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(64, n_classes)   # logits
        )
    def forward(self, x): return self.net(x)

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_true.append(yb.cpu().numpy())
    if len(y_true)==0:
        return float('nan'), np.array([]), np.array([])
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    return total_loss/len(loader.dataset), y_true, y_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=False, help="Path to dataset.csv")
    ap.add_argument("--feature_cols", default="", help="Comma-separated; empty->auto numeric features")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=8)  # early stopping
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) 读数据
    current_work_dir = os.path.dirname(__file__) 
    data_path = os.path.join(current_work_dir,'data','data.csv')
    df = pd.read_csv(data_path)

    # 2) 选择特征列
    if args.feature_cols.strip():
        feature_cols: List[str] = [c.strip() for c in args.feature_cols.split(",")]
    else:
        feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
        extras = [c for c in df.columns if c not in feature_cols + [args.label_col]]
        extras = [c for c in extras if pd.api.types.is_numeric_dtype(df[c])]
        feature_cols += extras  # 自动把其它数值列也加上

    # 3) 提取 X/y
    X = df[feature_cols].astype(float).to_numpy()
    y = df[args.label_col].astype(int).to_numpy()

    # 4) 按时间划分（避免信息泄漏）
    tr_idx, va_idx, te_idx = chrono_split_idx(len(df), args.train_ratio, args.val_ratio)
    X_tr, X_va, X_te = X[tr_idx], X[va_idx], X[te_idx]
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]

    # 5) 仅在训练集拟合标准化器
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    # 6) 构建 DataLoader
    dl_tr = DataLoader(TabDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    dl_va = DataLoader(TabDataset(X_va, y_va), batch_size=args.batch_size, shuffle=False)
    dl_te = DataLoader(TabDataset(X_te, y_te), batch_size=args.batch_size, shuffle=False)

    # 7) 类别不平衡 -> class weights
    classes = np.unique(y_tr)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weights = torch.tensor(cw, dtype=torch.float32, device=device)
    print("Class weights:", {int(c): float(w) for c, w in zip(classes, cw)})

    # 8) 建模与优化器
    model = MLP(in_dim=X_tr.shape[1], n_classes=len(classes), p_drop=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # 9) 训练循环（早停基于 val_loss）
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 稳定训练
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(dl_tr.dataset)

        va_loss, yv, yv_hat = evaluate(model, dl_va, device, criterion)
        macro_f1 = f1_score(yv, yv_hat, average="macro") if len(yv) else float("nan")
        scheduler.step(va_loss)
        print(f"Epoch {epoch:03d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | val_macroF1 {macro_f1:.4f}")

        if va_loss < best_val - 1e-6:
            best_val, best_state, wait = va_loss, {k: v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 10) 测试集评估与保存
    te_loss, yt, yt_hat = evaluate(model, dl_te, device, criterion)
    print("\n=== Test Report ===")
    print(classification_report(yt, yt_hat, digits=4))
    print("Test macro-F1:", f1_score(yt, yt_hat, average="macro"))

    cm = confusion_matrix(yt, yt_hat, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
    cm_df.to_csv("mlp_confusion_matrix_torch.csv", index=True)
    print("Saved confusion matrix -> mlp_confusion_matrix_torch.csv")

    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": X_tr.shape[1],
        "classes": classes.tolist()
    }, "mlp_candles_torch_model.pt")
    meta = {
        "feature_cols": feature_cols,
        "label_col": args.label_col,
        "classes": classes.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }
    with open("mlp_candles_torch_meta.json","w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved model -> mlp_candles_torch_model.pt")
    print("Saved meta  -> mlp_candles_torch_meta.json")

if __name__ == "__main__":
    main()
