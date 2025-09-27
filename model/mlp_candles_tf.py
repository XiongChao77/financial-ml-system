#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP (TensorFlow/Keras) for 15m candle classification
----------------------------------------------------
- Input: a CSV with columns like
  open, high, low, close, volume, quote_asset_volume,
  number_of_trades, taker_buy_base_volume, taker_buy_quote_volume, label
- Task: 3-class classification (0=down, 1=flat/weak, 2=up)
- Split: chronological Train/Val/Test by ratios
- Preprocess: StandardScaler on numeric features (fit on Train only)
- Model: MLP with BatchNorm + Dropout
- Metrics: accuracy, macro F1; confusion matrix
- Class imbalance: class_weight (auto-computed)

Usage
-----
pip install pandas numpy scikit-learn tensorflow
python mlp_candles_tf.py --csv /path/to/dataset.csv

Notes
-----
- If your label column is not integers {0,1,2}, set --label_map accordingly (e.g., "down:0,weak:1,up:2")
- If your dataset already contains engineered features, list them explicitly with --feature_cols
"""

import argparse
import json
from typing import List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DEFAULT_FEATURES = [
    "open","high","low","close",
    "volume","quote_asset_volume","number_of_trades",
    "taker_buy_base_volume","taker_buy_quote_volume"
]

def parse_label_map(s: str):
    mapping = {}
    for kv in s.split(","):
        if not kv: 
            continue
        k, v = kv.split(":")
        mapping[k.strip()] = int(v.strip())
    return mapping

def build_mlp(input_dim: int, n_classes: int = 3) -> tf.keras.Model:
    inp = layers.Input(shape=(input_dim,), name="features")
    x = layers.BatchNormalization()(inp)
    for u in [256, 128, 64]:
        x = layers.Dense(u, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def chrono_split_idx(n: int, train_ratio=0.7, val_ratio=0.15):
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    tr = np.arange(0, n_train)
    va = np.arange(n_train, n_train + n_val)
    te = np.arange(n_train + n_val, n)
    return tr, va, te

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to prepared dataset CSV")
    ap.add_argument("--feature_cols", type=str, default="", help="Comma-separated feature columns. Leave empty to use defaults.")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--label_map", type=str, default="", help='Optional map like "down:0,weak:1,up:2" if label is string')
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    df = pd.read_csv(args.csv)
    # Determine features
    if args.feature_cols.strip():
        feature_cols: List[str] = [c.strip() for c in args.feature_cols.split(",")]
    else:
        feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
        # Also try to include any extra numeric columns except the label
        extras = [c for c in df.columns if c not in feature_cols + [args.label_col]]
        # keep only numeric extras
        extras = [c for c in extras if pd.api.types.is_numeric_dtype(df[c])]
        feature_cols = feature_cols + extras

    # Label handling
    y = df[args.label_col]
    if args.label_map:
        mapping = parse_label_map(args.label_map)
        y = y.map(mapping)
    y = y.astype(int).to_numpy()

    X = df[feature_cols].astype(float).to_numpy()

    # Chronological split
    tr_idx, va_idx, te_idx = chrono_split_idx(len(df), args.train_ratio, args.val_ratio)
    X_tr, X_va, X_te = X[tr_idx], X[va_idx], X[te_idx]
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    # Class weights for imbalance
    classes = np.unique(y_tr)
    class_weights_arr = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}
    print("Class weights:", class_weight)

    # Build & train
    model = build_mlp(X_tr.shape[1], n_classes=len(classes))
    es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    rl = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        class_weight=class_weight,
        callbacks=[es, rl]
    )

    # Evaluation
    y_proba = model.predict(X_te, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    print("\n=== Test Classification Report (macro avg focus) ===")
    print(classification_report(y_te, y_pred, digits=4))

    macro_f1 = f1_score(y_te, y_pred, average="macro")
    print("Macro F1:", macro_f1)

    cm = confusion_matrix(y_te, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
    cm_df.to_csv("/mnt/data/mlp_confusion_matrix.csv", index=True)

    # Save artifacts
    model.save("/mnt/data/mlp_candles_tf_model.keras")
    meta = {
        "feature_cols": feature_cols,
        "label_col": args.label_col,
        "classes": classes.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist()
    }
    with open("/mnt/data/mlp_candles_tf_meta.json","w",encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nSaved model to /mnt/data/mlp_candles_tf_model.keras")
    print("Saved confusion matrix to /mnt/data/mlp_confusion_matrix.csv")
    print("Saved meta to /mnt/data/mlp_candles_tf_meta.json")

if __name__ == "__main__":
    main()
