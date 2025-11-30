from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import datetime
import os, sys, time, json
import backtrader as bt
import backtrader.analyzers as btanalyzers 
from datetime import timezone
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging
# --- 新增：引入 sklearn 计算模型指标 ---
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))

# 引入自定义模块
from data_process.common import *
from model.cnn_timeseries_torch import CNN1D
from model.data_loader import TimeSeriesWindowDataset 
from trade_simulation import cus_analyzer,cus_comminfo

MODEL_PATH = os.path.join(current_work_dir, '..', 'model', "cnn_timeseries_torch_model.pt")
META_PATH  = os.path.join(current_work_dir, '..', 'model', "cnn_timeseries_torch_meta.json")
log_file   = os.path.join(current_work_dir, "backtest.log")

class TradeResult():
    def __init__(self) -> None:
        self.times = 0

_cash = 10000
commission = 0 # 0.001 # 0.1%

# ------------- Logging Utilities -------------
def setup_logging(log_file: str, verbose: bool = False):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    level = log_level 

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    root = logging.getLogger("backtest")
    if root.hasHandlers():
        root.handlers.clear()

    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    return root

logger = setup_logging(log_file)

# --- 新增：模型评估函数 ---
def evaluate_performance(y_true, y_pred, labels=None):
    """
    生成符合要求的 Test Report 格式日志
    """
    logger.info("\n=== Test Report ===")
    
    # 1. 生成主要分类报告 (Precision, Recall, F1)
    # digits=4 确保保留4位小数 (例如 0.0956)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    # logger 默认会处理换行，直接打印即可
    logger.info("\n" + report)

    # 2. 宏平均 F1 (单独打印)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    logger.info(f"Test macro-F1:{macro_f1}")
    
    # 3. 真实标签分布
    logger.info("\n=== True label proportion (Test set) ===")
    unique_labels, counts = np.unique(y_true, return_counts=True)
    total_samples = len(y_true)
    
    for label, count in zip(unique_labels, counts):
        proportion = count / total_samples
        logger.info(f"label {label}: {count} samples, {proportion:.4f} of total")

    # 返回 UI 需要的简单指标字典
    return {
        "accuracy": f"{accuracy_score(y_true, y_pred):.2%}",
        "precision": f"{precision_score(y_true, y_pred, average='weighted', zero_division=0):.2%}",
        "recall": f"{recall_score(y_true, y_pred, average='weighted', zero_division=0):.2%}",
        "f1_score": f"{f1_score(y_true, y_pred, average='weighted', zero_division=0):.2%}"
    }

# --- 离线批量推理 ---
def offline_predict_cnn(df, feature_cols, label_col, window, model, device, batch_size: int = 1024):
    logger.info("Starting offline inference...")

    # 1. 创建 Dataset 
    # TimeSeriesWindowDataset 会自动处理:
    #   - 滑动窗口 (Rolling Window)
    #   - 动态 t=0 缩放 (Scaling)
    #   - DROP_FEATURES 过滤 (会忽略时间列等非特征列)
    ds = TimeSeriesWindowDataset(df=df, feature_cols=feature_cols, label_col=label_col, window=window)
    # 2. 创建 DataLoader
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 3. 批量推理
    model.eval()
    preds, confs = [], []
    
    with torch.no_grad():
        for xb, _ in dl: 
            xb = xb.to(device)
            logits = model(xb)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            
            preds.append(probs.argmax(axis=1).astype(np.int64))
            confs.append(probs.max(axis=1).astype(np.float32))
            
    if len(preds) > 0:
        preds = np.concatenate(preds)
        confs = np.concatenate(confs)
    else:
        logger.warning("No predictions generated!")
        return df

    # 4. 对齐索引
    # TimeSeriesWindowDataset 生成的样本从 window-1 开始
    valid_idx = df.index[window-1:]
    
    if len(valid_idx) != len(preds):
        min_len = min(len(valid_idx), len(preds))
        valid_idx = valid_idx[:min_len]
        preds = preds[:min_len]
        confs = confs[:min_len]

    # 5. 构造结果列
    # 先初始化为 NaN
    df_out = df.copy()
    df_out['pred'] = np.nan
    df_out['conf'] = np.nan
    
    # 填入计算结
    df_out.loc[valid_idx, 'pred'] = preds
    df_out.loc[valid_idx, 'conf'] = confs

    # === 在此处调用评估函数 ===
    model_stats = {}
    if label_col in df_out.columns:
        # 提取有效部分的真实标签 (对应 pred 不为 NaN 的部分)
        df_valid = df_out.loc[valid_idx]
        y_true = df_valid[label_col].values.astype(int)
        y_pred = df_valid['pred'].values.astype(int)
        
        # 调用封装好的函数打印报告
        model_stats = evaluate_performance(y_true, y_pred)
    else:
        logger.warning(f"Label column '{label_col}' missing, skipping evaluation.")
    
    logger.info(f"Inference done. Generated {len(preds)} predictions.")
    return df_out , model_stats

# --- DataFeed 扩展 ---
class PandasDataWithPred(bt.feeds.PandasData):
    lines = ('pred', 'conf',)
    params = (('pred', -1), ('conf', -1),)

# --- Strategy ---
class MyStrategy(bt.Strategy):
    params = dict(
        holdbar=1,                 
        trade_risk=0.05,           # 每次加仓 5% 总资金
        max_layers=10,              # 最大加仓层数
        allow_short=True,         
        allow_long=True,           
        thresh=None,               # 置信度阈值
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.bar_executed = None
        self.held_bars = 0
        self.dir = 0     # 当前持仓方向: 1(多), -1(空), 0(无)
        self.layers = 0  # 当前加仓层数
        self.trade_logs = []

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.debug(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")
            elif order.issell():
                logger.debug(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")
            self.bar_executed = len(self)
            
            # 记录交易日志 (修复 UTC 时间戳问题)
            dt = self.data.datetime.datetime()
            dt_utc = dt.replace(tzinfo=timezone.utc)
            record = {
                "dt": int(dt_utc.timestamp()),
                "price": order.executed.price,
                "size": order.executed.size,
                "is_buy": order.isbuy()
            }
            self.trade_logs.append(record)
            
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"Order Canceled/Margin/Rejected: {order.getstatusname()}")

    def stop(self):
        value = self.broker.getvalue()
        logger.info(f"End Value: {value:.2f}")
        #UI
        self.cerebro.trade_logs = self.trade_logs
        
    def next(self):
        # 获取预测结果
        pred = self.data.pred[0]
        conf = self.data.conf[0]

        # 1. 数据有效性检查
        if np.isnan(pred) or np.isnan(conf):
            return

        # 2. 置信度过滤
        if self.params.thresh is not None and conf < self.params.thresh:
            # 如果置信度不够，我们可以选择“保持不动”或者“视为震荡”
            # 这里简单处理：视为震荡信号(target_dir=0)，如果不持仓则不开，如果持仓则可能平仓
            pred = 1 # 强制视为震荡/观望
        
        pred = int(pred)
        
        # 3. 映射信号到方向
        # 假设: 0=空(Short), 1=震荡(Neutral), 2=多(Long)
        target_dir = 0
        if pred == 2 and self.params.allow_long:
            target_dir = 1
        elif pred == 0 and self.params.allow_short:
            target_dir = -1
        else:
            target_dir = 0 # 震荡或不允许的方向

        # 更新持仓时间计数
        if self.position:
            self.held_bars += 1
        else:
            self.held_bars = 0
            self.dir = 0
            self.layers = 0

        # === 4. 核心交易逻辑优化 ===
        
        # 情况 A: 当前无持仓
        if not self.position:
            if target_dir != 0:
                # 只有明确的多/空信号才开仓
                self.dir = target_dir
                self.layers = 1
                target_pct = self.params.trade_risk * self.layers * self.dir
                self.order_target_percent(target=target_pct)
            return

        # 情况 B: 当前有持仓 (self.dir != 0)
        
        # B-1: 信号变为震荡 (Label 1) -> 立即平仓
        if target_dir == 0:
            if self.held_bars >= self.params.holdbar:
                self.close()
                self.dir = 0
                self.layers = 0
            return

        # B-2: 信号反转 (多转空 或 空转多) -> 反手 (Reverse)
        if target_dir != self.dir:
             if self.held_bars >= self.params.holdbar:
                # 记录新方向
                self.dir = target_dir
                self.layers = 1 # 重置层数为1
                # 直接计算反向的目标仓位 (例如从 +0.05 变成 -0.05)
                # Backtrader 会自动平掉旧仓位并开新仓位
                target_pct = self.params.trade_risk * self.layers * self.dir
                self.order_target_percent(target=target_pct)
             return

        # B-3: 信号同向 (同向预测) -> 加仓 (Pyramiding)
        if target_dir == self.dir:
            if self.layers < self.params.max_layers:
                self.layers += 1
                target_pct = self.params.trade_risk * self.layers * self.dir
                self.order_target_percent(target=target_pct)
                logger.debug(f"Pyramiding: Layer {self.layers}, Target {target_pct:.2%}")
            return

        # 强制最后平仓
        if len(self.data) - 1 == len(self) - 1 and self.position:
            self.close()

class Parameters:
    def __init__(self):
        self.allow_short = True
        self.allow_long = True
        self.thresh:float = 0
        self.verbose = False
        
def main():
    args = Parameters()
    if args.verbose:    logger.setLevel(logging.DEBUG)
    logger.info(f"Backtest settings: Short={args.allow_short}, Long={args.allow_long}, Thresh={args.thresh}")

    # 1. 数据加载
    data_path = test_data_path 
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
        
	# 直接读取 CSV，假设其中已包含所有特征列和时间列
    df = pd.read_csv(data_path)
    # 【关键】检查时间列是否存在
    if 'open_time_dt_utc' not in df.columns:
        logger.error("CRITICAL: 'open_time_dt_utc' column missing.")
        sys.exit(1)
    # 【关键】解析时间列
    # 不再调用 attach_attr，避免重复计算和潜在的数据修改
    df['open_time_dt_utc'] = pd.to_datetime(df['open_time_dt_utc'], utc=True)

    # 2. 模型加载
    if not os.path.exists(META_PATH) or not os.path.exists(MODEL_PATH):
        logger.error("Model files not found. Please train first.")
        sys.exit(1)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    feature_cols = meta["feature_cols"]
    window = int(meta["window"])
    classes = meta["classes"]
    label_col = meta.get("label_col", "label") # 确保有真实标签列

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    state = torch.load(MODEL_PATH, map_location=device)
    channel = state.get("channel", len(feature_cols))
    n_classes = len(state.get("classes", classes))
    
    model = CNN1D(channel=channel, n_classes=n_classes, p_drop=0.0).to(device)
    model.load_state_dict(state["state_dict"])
    
    # 3. 离线推理
    df_with_pred , model_stats= offline_predict_cnn(df, feature_cols, label_col, window, model, device)
    df_with_pred.dropna(subset=['pred'], inplace=True)
    logger.info(f"Data range: {df_with_pred['open_time_dt_utc'].min()} to {df_with_pred['open_time_dt_utc'].max()}")

    # 4. Backtrader 执行
    cerebro = bt.Cerebro(runonce=False) 
    cerebro.addstrategy(
        MyStrategy, 
        holdbar = 4,
        allow_short=args.allow_short, 
        allow_long=args.allow_long,
        thresh=args.thresh
    )

    data = PandasDataWithPred(
        dataname=df_with_pred,
        datetime='open_time_dt_utc', 
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1,
        nocase=True
    )

    cerebro.adddata(data)
    cerebro.broker.setcash(_cash)
    cerebro.broker.addcommissioninfo(cus_comminfo.CommInfo_Cryptocurrency(commission=commission))
    # cerebro.broker.set_coc(True)  # 

    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=1)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(cus_analyzer.CusAnalyzer, _name='customize')

    logger.info("Starting Backtest...")
    results = cerebro.run()
    strat = results[0]

    # 5. 结果统计
    #UI
    # 封装统计数据 (合并回测数据和模型指标)
    statistics = generate_backtest_report(strat, model_stats)

    trade_logs = cerebro.trade_logs

    # ========== 转 K线 JSON ==========
    candles = df[['open_time_dt_utc','open','high','low','close']].copy()
    candles.rename(columns={"open_time_dt_utc": "time"}, inplace=True)
    candles["time"] = candles["time"].apply(lambda dt: int(dt.timestamp()))
    candles_json = candles.to_dict(orient="records")

    markers = []
    for t in trade_logs:
        markers.append({
            "time": t["dt"],
            "price": t["price"],
            "size": t["size"],
            "position": "aboveBar" if t["is_buy"] else "belowBar",
            "color": "green" if t["is_buy"] else "red",
            "shape": "arrowUp" if t["is_buy"] else "arrowDown",
            "text": ("BUY" if t["is_buy"] else "SELL") + f" @ {t['price']:.2f}"
        })

    return {
        "candles": candles_json,
        "markers": markers,
        "statistics": statistics
    }


def safe_get(d, keys, default=0):
    """从多层 dict 中取值，避免 KeyError"""
    cur = d
    for k in keys:
        cur = cur.get(k, {})
    return cur if cur != {} else default


def generate_backtest_report(strat, model_stats=None, save_path="full_backtest_report.json"):
    """
    功能：
    1. 读取 analyzers（customize / dd / sharpe / trades）
    2. 打印 summary（保持你现在的格式）
    3. 保存完整信息到 JSON 文件
    4. 返回前端 UI 需要的 statistics（保持你现在的格式）
    """

    # ========== 1. 提取 analyzers ==========
    perf   = strat.analyzers.customize.get_analysis()
    dd     = strat.analyzers.dd.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()

    # ----- 基础数值 -----
    gross_return = perf.get("gross_return", 0)
    cagr         = perf.get("cagr", 0)
    start_value  = perf.get("start_value", 0)
    end_value    = perf.get("end_value", 0)

    # ----- Sharpe -----
    sr = sharpe.get("sharperatio", 0.0) or 0.0

    # ----- 最大回撤 -----
    maxdd_pct  = dd.get("max", {}).get("drawdown", 0.0)
    maxdd_amt  = dd.get("max", {}).get("moneydown", 0.0)
    maxdd_len  = dd.get("max", {}).get("len", 0)

    # ----- 交易统计 -----
    total_trades = safe_get(trades, ["total", "closed"], 0)
    total_won    = safe_get(trades, ["won", "total"], 0)
    win_rate = (total_won / total_trades * 100) if total_trades > 0 else 0.0

    # ========== 2. 打印 summary ==========
    summary = (
        f"SUMMARY | GrossRet: {gross_return*100:.2f}% | CAGR: {cagr*100:.2f}% | "
        f"Sharpe: {sr:.3f} | MaxDD: {maxdd_pct:.2f}% ({maxdd_amt:.0f}) | "
        f"Trades: {total_trades} | WinRate: {win_rate:.2f}%"
    )
    print(summary)

    # ========== 3. 构造完整 JSON report ==========
    full_report = {
        # 基础资金曲线
        "start_value": start_value,
        "end_value": end_value,
        "gross_return": gross_return,
        "cagr": cagr,
        "sharpe": sr,

        # 回撤
        "max_drawdown_pct": maxdd_pct,
        "max_drawdown_amount": maxdd_amt,
        "max_drawdown_duration": maxdd_len,

        # 交易统计
        "total_trades": total_trades,
        "total_won": total_won,
        "win_rate": win_rate,

        # 暴露信息来自 CusAnalyzer
        "avg_pos_ratio": perf.get("avg_pos_ratio", 0),
        "std_pos_ratio": perf.get("std_pos_ratio", 0),
        "p95_pos_ratio": perf.get("p95_pos_ratio", 0),
        "max_pos_ratio": perf.get("max_pos_ratio", 0),

        # 各 Analyzer 原始数据
        "trade_analyzer_raw": trades,
        "drawdown_raw": dd,
        "sharpe_raw": sharpe,

        # 模型预测指标（从外部传入）
        "model_metrics": model_stats or {}
    }

    # 写入 JSON 文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4)

    # ========== 4. 返回前端所需 statistics ==========
    ui_stats = {
        "gross_return": f"{gross_return*100:.2f}%",
        "cagr": f"{cagr*100:.2f}%",
        "sharpe": f"{sr:.3f}",
        "max_drawdown": f"{maxdd_pct:.2f}%",
        "total_trades": total_trades,
        "win_rate": f"{win_rate:.2f}%",
        "start_value": f"{start_value:.2f}",
        "end_value": f"{end_value:.2f}",
        "equity_curve": perf.get("equity_curve", []),
        "drawdown_curve": perf.get("drawdown_curve", []),
        "return_curve": perf.get("return_curve", []),
        "rolling_return_30": perf.get("rolling_return_30", []),
        **(model_stats or {})
    }
    return ui_stats

if __name__ == '__main__':
    main()