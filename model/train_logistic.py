#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from tqdm import tqdm
from collections import Counter
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

# 路径设置
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from data_process import common
from model.data_loader import TimeSeriesWindowDataset

# ==============================================================================
# 1. 配置定义 (Configuration)
# ==============================================================================

@dataclass
class DataConfig:
    label_col: str = "label"
    window: int = common.BaseDefine.predict_num
    train_ratio: float = 0.7
    val_ratio: float = 0.15

@dataclass
class LogisticConfig:
    model_type: str = "logistic"
    model_version: int = 1
    C: float = 1.0  # 正则化强度的倒数，越小正则化越强
    penalty: str = "l2"  # 'l1' or 'l2'
    solver: str = "lbfgs"  # 'lbfgs', 'liblinear', 'saga', etc.
    max_iter: int = 1000
    class_weight: Optional[str] = "balanced"  # None, 'balanced', or dict
    multi_class: str = "multinomial"  # 'ovr' or 'multinomial'
    random_state: int = 42

@dataclass
class TrainConfig:
    model_cfg: LogisticConfig = field(default_factory=LogisticConfig)
    data_cfg: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    save_dir: str = common.TRAIN_OUT_DIR
    stride: int = 2
    use_cache: bool = False
    use_scaler: bool = True  # 是否使用 StandardScaler 标准化特征

# ==============================================================================
# 2. 核心逻辑 (Core Logic)
# ==============================================================================

def apply_feature_direction(X: np.ndarray, feature_names: List[str], direction_map: Dict[str, int], logger) -> np.ndarray:
    """
    对 direction=-1 的特征列乘以 -1，使其与收益正相关。
    X: shape [N, T, F]，归一化后的特征数组
    feature_names: 特征名列表，与 X 的第 2 维对应
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

def flatten_time_series(X: np.ndarray) -> np.ndarray:
    """
    将时间序列窗口数据展平为特征向量
    X: shape [N, T, F] -> [N, T*F]
    """
    N, T, F = X.shape
    return X.reshape(N, T * F)

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)

def chrono_split_by_window_ends(M, tr_r, va_r):
    """按时间顺序切分数据"""
    n_tr = int(M * tr_r)
    n_va = int(M * va_r)
    return (0, n_tr), (n_tr, n_tr + n_va), (n_tr + n_va, M)

def run_training(feature_direction_map, logger: logging.Logger, data_cfg: DataConfig, train_cfg: TrainConfig, model_cfg: LogisticConfig, pre_para: common.BaseDefine = None):
    """
    使用 sklearn LogisticRegression 进行训练和评估
    """
    if pre_para is None:
        pre_para = common.BaseDefine()
    
    # 0. 初始化环境
    set_seed(train_cfg.seed)
    logger.info(f"Model: {model_cfg.model_type} version: {model_cfg.model_version}")

    # 1. 准备数据（从 pre_para.prep_output_dir 读，默认 DATA_OUT_DIR）
    df = common.load_train_df_from_dir(pre_para.prep_output_dir)
    kline_interval_ms = common.load_interval_ms_from_dir(pre_para.prep_output_dir)
    logger.info(f"Using TimeSeriesWindowDataset with window={data_cfg.window} Origin data len {len(df)}...")

    feature_list = list(feature_direction_map.keys())
    full_ds = TimeSeriesWindowDataset(
        df=df, 
        kline_interval_ms=kline_interval_ms, 
        feature_cols=feature_list, 
        label_col=data_cfg.label_col, 
        window=data_cfg.window,
        cache_path=os.path.join(common.TEMPORARY_DIR, "train_cache.pt"), 
        stride=train_cfg.stride, 
        use_cache=train_cfg.use_cache, 
        show_feature_distribution=True
    )
    logger.warning(f"📊 [Dataset Check] Final features used in training ({full_ds.feature_count}): {full_ds.feature_names}")
    
    # 对 ic_direction=-1 的特征进行反向（乘以 -1），使其与收益正相关
    if feature_direction_map:
        X_np = full_ds.X.numpy()
        X_np = apply_feature_direction(X_np, full_ds.feature_names, feature_direction_map, logger)
        full_ds.X = torch.from_numpy(X_np)

    M = len(full_ds)
    logger.info(f"Total windows (M) = {M}, window = {data_cfg.window}")

    # 2. 切分数据
    tr_rng, va_rng, te_rng = chrono_split_by_window_ends(M, data_cfg.train_ratio, data_cfg.val_ratio)
    
    # 提取数据并转换为 numpy
    X_tr = full_ds.X[tr_rng[0]:tr_rng[1]].numpy()
    y_tr = full_ds.y[tr_rng[0]:tr_rng[1]].numpy()
    r_tr = full_ds.returns[tr_rng[0]:tr_rng[1]].numpy()
    
    X_va = full_ds.X[va_rng[0]:va_rng[1]].numpy()
    y_va = full_ds.y[va_rng[0]:va_rng[1]].numpy()
    r_va = full_ds.returns[va_rng[0]:va_rng[1]].numpy()
    
    X_te = full_ds.X[te_rng[0]:te_rng[1]].numpy()
    y_te = full_ds.y[te_rng[0]:te_rng[1]].numpy()
    r_te = full_ds.returns[te_rng[0]:te_rng[1]].numpy()

    # 3. 展平时间序列窗口数据
    logger.info("Flattening time series windows...")
    X_tr_flat = flatten_time_series(X_tr)  # [N_tr, T*F]
    X_va_flat = flatten_time_series(X_va)  # [N_va, T*F]
    X_te_flat = flatten_time_series(X_te)  # [N_te, T*F]
    
    logger.info(f"Flattened feature dimensions: {X_tr_flat.shape[1]} (window={data_cfg.window} * features={full_ds.feature_count})")

    # 4. 特征标准化（可选）
    scaler = None
    if train_cfg.use_scaler:
        logger.info("Applying StandardScaler...")
        scaler = StandardScaler()
        X_tr_flat = scaler.fit_transform(X_tr_flat)
        X_va_flat = scaler.transform(X_va_flat)
        X_te_flat = scaler.transform(X_te_flat)

    # 5. 计算类别权重
    classes = np.unique(y_tr)
    if model_cfg.class_weight == "balanced":
        class_weights = compute_class_weight("balanced", classes=classes, y=y_tr)
        class_weight_dict = dict(zip(classes, class_weights))
        logger.info(f"Class weights: {class_weight_dict}")
    else:
        class_weight_dict = model_cfg.class_weight

    # 6. 构建模型
    logger.info(f"Initializing LogisticRegression: C={model_cfg.C}, penalty={model_cfg.penalty}, solver={model_cfg.solver}")
    
    model = LogisticRegression(
        C=model_cfg.C,
        penalty=model_cfg.penalty,
        solver=model_cfg.solver,
        max_iter=model_cfg.max_iter,
        class_weight=class_weight_dict,
        multi_class=model_cfg.multi_class,
        random_state=model_cfg.random_state,
        n_jobs=-1  # 使用所有CPU核心
    )

    # 7. 训练模型
    logger.info("🚀 Starting training...")
    model.fit(X_tr_flat, y_tr)
    logger.info("✅ Training completed")

    # 8. 验证集评估
    y_va_pred = model.predict(X_va_flat)
    va_f1 = f1_score(y_va, y_va_pred, average="macro")
    logger.info(f"Validation macro-F1: {va_f1:.4f}")

    # 9. 测试集评估与保存
    results = {
        "model": model,
        "scaler": scaler,
        "f1_score": va_f1,
        "loss_score": 0.0  # LogisticRegression 不使用 loss_score，保持兼容性
    }
    
    final_metrics = evaluate_and_save_results(
        results=results,
        model=model,
        X_te=X_te_flat,
        y_te=y_te,
        r_te=r_te,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        full_ds=full_ds,
        feature_list=feature_list,
        classes=classes,
        logger=logger,
        pre_para=pre_para,
    )
    
    return final_metrics

def evaluate_and_save_results(
    results: dict,
    model: LogisticRegression,
    X_te: np.ndarray,
    y_te: np.ndarray,
    r_te: np.ndarray,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    full_ds: TimeSeriesWindowDataset,
    feature_list: List[str],
    classes: np.ndarray,
    logger: logging.Logger,
    pre_para=None
):
    """
    评估函数：计算测试集指标并保存模型
    """
    logger.info(f"\n{'='*20} Evaluating LogisticRegression Model {'='*20}")
    
    # 1. 预测
    y_te_pred = model.predict(X_te)
    y_te_proba = model.predict_proba(X_te)  # [N, n_classes]
    
    # 2. 计算分类报告
    report_dict = classification_report(y_te, y_te_pred, output_dict=True, zero_division=0)
    test_f1 = report_dict['macro avg']['f1-score']
    
    logger.info("\n=== Test Report ===")
    logger.info(format_custom_report(report_dict))
    logger.info(f"Test macro-F1: {test_f1:.4f}")

    # 3. 计算交易指标
    avg_ret, trade_count, win_rate, trades_pnl = calculate_pure_trading_metrics(
        y_te, y_te_pred, r_te, logger, pre_para=pre_para
    )
    
    logger.info(f"| avg_ret: {avg_ret:.6f} | trade_count: {trade_count} | win_rate: {win_rate:.4f}")

    # 4. 统计标签比例
    counts = Counter(y_te)
    total = sum(counts.values())
    logger.info(f"True label proportion (Test set): " + 
                ", ".join([f"{c}: {counts[c]/total:.2%}" for c in sorted(counts.keys())]))

    # 5. 保存混淆矩阵
    cm = confusion_matrix(y_te, y_te_pred, labels=classes)
    cm_path = os.path.join(train_cfg.save_dir, "confmat_logistic.csv")
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]).to_csv(cm_path, index=True)
    logger.info(f"Confusion matrix saved to: {cm_path}")

    # 6. 保存模型和元数据
    model_path = os.path.join(train_cfg.save_dir, "model_logistic.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': results['model'],
            'scaler': results['scaler'],
            'feature_list': feature_list,
            'classes': classes.tolist(),
            'feature_count': full_ds.feature_count,
            'window': data_cfg.window,
            'feature_names': full_ds.feature_names,
            'label_col': data_cfg.label_col,
            'val_f1': results['f1_score'],
            'test_f1': test_f1,
        }, f)
    logger.info(f"Model saved to: {model_path}")

    # 保存 JSON 元数据
    meta = {
        "model_type": "logistic",
        "model_version": train_cfg.model_cfg.model_version,
        "feature_cols": full_ds.feature_names,
        "label_col": data_cfg.label_col,
        "classes": classes.tolist(),
        "window": data_cfg.window,
        "feature_count": full_ds.feature_count,
        "val_f1": float(results['f1_score']),
        "test_f1": float(test_f1),
        "test_avg_ret": float(avg_ret),
        "test_trade_count": int(trade_count),
        "test_win_rate": float(win_rate),
        "model_config": {
            "C": train_cfg.model_cfg.C,
            "penalty": train_cfg.model_cfg.penalty,
            "solver": train_cfg.model_cfg.solver,
            "max_iter": train_cfg.model_cfg.max_iter,
            "multi_class": train_cfg.model_cfg.multi_class,
        }
    }
    
    meta_path = os.path.join(train_cfg.save_dir, "model_logistic_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Metadata saved to: {meta_path}")

    # 生成 Task Description
    task_desc = {
        "task_type": "single",
        "timestamp": pd.Timestamp.now().isoformat(),
        "models": {
            "main": {
                "model": "model_logistic.pkl",
                "meta": "model_logistic_meta.json"
            }
        }
    }
    
    task_path = os.path.join(train_cfg.save_dir, "task_description.json")
    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(task_desc, f, ensure_ascii=False, indent=2)
    logger.info(f"🎉 Task description saved to: {task_path}")

    final_metrics = {
        "test_f1": test_f1,
        "val_f1": results['f1_score'],
        "pure_avg_ret": avg_ret,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "precision_short": report_dict.get('0', {}).get('precision', 0),
        "recall_short": report_dict.get('0', {}).get('recall', 0),
        "precision_long": report_dict.get('2', {}).get('precision', 0),
        "recall_long": report_dict.get('2', {}).get('recall', 0),
    }
    
    return final_metrics

def calculate_pure_trading_metrics(y_true, y_pred, returns, logger, pre_para=None, cooldown_period=None):
    """
    更纯粹的评估方法，同时运行两种评估：
    
    评估1（原始逻辑）：
    1. 只要预测为 Long 或 Short 就视为入场。
    2. PnL = 信号方向 * 实际收益率 (即：Long正确得正，Long错误得负)。
    
    评估2（冷却期逻辑）：
    1. 在一次交易发生后，cooldown_period 之内不会再发生交易。
    2. 默认 cooldown_period = pre_para.predict_num（若未指定）。
    """
    if pre_para is None:
        from data_process import common
        pre_para = common.BaseDefine()
    if cooldown_period is None:
        cooldown_period = pre_para.predict_num
    
    # 信号映射: 0(Short)->-1, 1(Neutral)->0, 2(Long)->1
    signals = np.zeros_like(y_pred)
    signals[y_pred == 2] = 1
    signals[y_pred == 0] = -1
    
    # ========== 评估1：原始逻辑（每次信号都交易） ==========
    pnl_per_sample = signals * returns
    trade_mask = (signals != 0)
    trades_pnl_1 = pnl_per_sample[trade_mask]
    num_trades_1 = len(trades_pnl_1)
    
    if num_trades_1 == 0:
        logger.warning("⚠️ 本轮评估未产生任何交易信号 (Long/Short)")
        avg_ret_1, win_rate_1 = 0.0, 0.0
    else:
        avg_ret_1 = np.mean(trades_pnl_1)
        win_rate_1 = (trades_pnl_1 > 0).sum() / num_trades_1
    
    # ========== 评估2：冷却期逻辑（交易后 cooldown_period 内不交易） ==========
    filtered_signals = np.zeros_like(signals)
    last_trade_idx = -cooldown_period - 1  # 初始化为足够早的位置
    
    for i in range(len(signals)):
        if signals[i] != 0:  # 有交易信号
            if i - last_trade_idx > cooldown_period:  # 距离上次交易超过冷却期
                filtered_signals[i] = signals[i]
                last_trade_idx = i
    
    pnl_per_sample_2 = filtered_signals * returns
    trade_mask_2 = (filtered_signals != 0)
    trades_pnl_2 = pnl_per_sample_2[trade_mask_2]
    num_trades_2 = len(trades_pnl_2)
    
    if num_trades_2 == 0:
        avg_ret_2, win_rate_2 = 0.0, 0.0
    else:
        avg_ret_2 = np.mean(trades_pnl_2)
        win_rate_2 = (trades_pnl_2 > 0).sum() / num_trades_2
    
    # 输出两种评估结果
    logger.info(f"\n{'='*20} Trading Metrics Comparison {'='*20}")
    logger.info(f"评估1（原始）: avg_ret={avg_ret_1:.6f}, trades={num_trades_1}, win_rate={win_rate_1:.4f}")
    logger.info(f"评估2（冷却期={cooldown_period}）: avg_ret={avg_ret_2:.6f}, trades={num_trades_2}, win_rate={win_rate_2:.4f}")
    logger.info(f"{'='*60}\n")
    
    # 返回评估1的结果（保持向后兼容）
    return avg_ret_1, num_trades_1, win_rate_1, trades_pnl_1

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

# ==============================================================================
# 3. 特征方向映射（从原文件复制）
# ==============================================================================

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

feature_conf_list = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "number_of_trades",
    "quote_asset_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "MACD_12_26_DIF",
    "MACD_12_26_DEA",
    "MACD_12_26_HIST",
    "MACD_12_26_DIF_PCT",
    "MACD_12_26_HIST_PCT",
    "MACD_12_26_HIST_ACCEL",
    "MACD_12_26_SIG_DIST",
    "KDJ_K",
    "KDJ_D",
    "KDJ_J",
    "KELTNER_UPPER_14",
    "KELTNER_LOWER_14",
    "KELTNER_MIDDLE_14",
    "QAV_SURGE_49",
    "QAV_SLOPE_49",
    "VWAP_BIAS",
    "PVT",
    "CMF_25",
    "body",
    "upper_wick",
    "lower_wick",
    "max_range",
    "body_mom",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_pos",
    "doji_score",
    "wick_bias",
]

feature_conf_list2 = [
    # ===== 价格 / 位置（强信息，避免全家桶） =====
    "close",
    "close_pos",
    "dist_to_high_100",
    "dist_to_high_20",
    "DONCHIAN_POS_20",

    # ===== 动量 / 反转（多尺度） =====
    "RSI_14",
    "MOM_20",
    "MOM_60",
    "MACD_12_26_DIF_PCT",
    "MACD_12_26_HIST_PCT",

    # ===== 通道 / 偏离 =====
    "BOLL_BW_25",
    "BOLL_PB_25",
    "KELTNER_MIDDLE_14",
    "DONCHIAN_DIST_U_20",
    "DONCHIAN_DIST_L_20",

    # ===== 波动率 / regime =====
    "vol_parkinson_100",
    "vol_gk_100",
    "skew_20",
    "er_126",

    # ===== 成交量 / 资金流 =====
    "PVT",
    # "OBV",
    "CMF_25",

    # ===== 订单流 / 微观结构 =====
    "poc_bias_49",
    "id_factor_20",
    "vpin_49",

    # ===== 补充（弱相关但非冗余） =====
    "VWAP_7",
    # ======basement features=======
    "quote_asset_volume",
]

def main(logger: logging.Logger, feature_conf_list=feature_conf_list, train_cfg=TrainConfig(), pre_para=common.BaseDefine()):
    os.makedirs(train_cfg.save_dir, exist_ok=True)

    # 根据 feature_conf_list 从全局 feature_direction_map 补充完整方向信息
    feature_direction_map_filtered = {}
    for feature_name in feature_conf_list:
        # 从全局 feature_direction_map 中查找方向，如果找不到则默认为 1（正向）
        direction = feature_direction_map.get(feature_name, 1)
        feature_direction_map_filtered[feature_name] = direction
    
    logger.info(f"📋 Using {len(feature_direction_map_filtered)} features from feature_conf_list")

    # 1. 数据配置
    d_cfg = DataConfig()
    
    # 2. 模型配置
    m_cfg = LogisticConfig()
    
    logger.info(f"Training {m_cfg.model_type}...")
    return run_training(feature_direction_map_filtered, logger, d_cfg, train_cfg, m_cfg, pre_para=pre_para)

# ==============================================================================
# 4. 调用入口 (Main Entry)
# ==============================================================================

if __name__ == "__main__":
    logger, _ = common.setup_session_logger(sub_folder='train', file_level=logging.DEBUG)
    main(logger)
