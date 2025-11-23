from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import datetime
import os, sys, time, json
import backtrader as bt
import backtrader.analyzers as btanalyzers 
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import logging

current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))

# 引入自定义模块
from data_process.common import *
from model.cnn_timeseries_torch import CNN1D
# 复用 model/data_loader.py 中的 Dataset，确保推理时的缩放/处理与训练完全一致
from model.data_loader import TimeSeriesWindowDataset 
import cus_analyzer

MODEL_PATH = os.path.join(current_work_dir, '..', 'model', "cnn_timeseries_torch_model.pt")
META_PATH  = os.path.join(current_work_dir, '..', 'model', "cnn_timeseries_torch_meta.json")
log_file   = os.path.join(current_work_dir, "backtest.log")

class TradeResult():
    def __init__(self) -> None:
        self.times = 0

_cash = 10000
commission = 0#0.001 # 0.1%

# ------------- Logging Utilities -------------
def setup_logging(log_file: str, verbose: bool = False):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    level = log_level # 来自 common.py

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

# --- 离线批量推理 (使用 DataLoader) ---
def offline_predict_cnn(df, feature_cols, label_col, window, model, device, batch_size: int = 1024):
    """
    使用 DataLoader 跑完模型，返回含 'pred','conf' 的 DataFrame。
    """
    logger.info("Starting offline inference...")
    
    # 1. 创建 Dataset 
    # TimeSeriesWindowDataset 会自动处理:
    #   - 滑动窗口 (Rolling Window)
    #   - 动态 t=0 缩放 (Scaling)
    #   - DROP_FEATURES 过滤 (会忽略时间列等非特征列)
    ds = TimeSeriesWindowDataset(
        df=df,
        feature_cols=feature_cols,
        label_col=label_col,
        window=window
    )
    
    # 2. 创建 DataLoader
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
# 3. 批量推理
    model.eval()
    preds, confs = [], []
    
    with torch.no_grad():
        for xb, _ in dl: # 不需要 yb (标签)
            xb = xb.to(device)
            logits = model(xb)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            
            preds.append(probs.argmax(axis=1).astype(np.int64))
            confs.append(probs.max(axis=1).astype(np.float32))
            
    if len(preds) > 0:
        preds = np.concatenate(preds)
        confs = np.concatenate(confs)
        
        # --- 新增代码: 打印预测结果分布 ---
        total_predictions = len(preds)
        unique_labels, counts = np.unique(preds, return_counts=True)
        
        logger.info("--- Overall Prediction Distribution ---")
        for label, count in zip(unique_labels, counts):
            percentage = (count / total_predictions) * 100
            # 确保只打印 0, 1, 2 (如果存在其他标签，这也能处理)
            if label in (0, 1, 2):
                logger.info(f"Label {label}: {count} samples ({percentage:.2f}%)")
        logger.info("-------------------------------------")
        # --- 新增代码结束 ---
        
    else:
        logger.warning("No predictions generated!")
        return df

    # 4. 对齐索引
    # TimeSeriesWindowDataset 生成的样本从 window-1 开始
    valid_idx = df.index[window-1:]
    
    if len(valid_idx) != len(preds):
        logger.error(f"Shape mismatch: Index len {len(valid_idx)} vs Preds len {len(preds)}")
        # 简单防错
        min_len = min(len(valid_idx), len(preds))
        valid_idx = valid_idx[:min_len]
        preds = preds[:min_len]
        confs = confs[:min_len]

    # 5. 构造结果列
    # 先初始化为 NaN
    df_out = df.copy()
    df_out['pred'] = np.nan
    df_out['conf'] = np.nan
    
    # 填入计算结果
    df_out.loc[valid_idx, 'pred'] = preds
    df_out.loc[valid_idx, 'conf'] = confs
    
    logger.info(f"Inference done. Generated {len(preds)} predictions.")
    return df_out

# --- DataFeed 扩展 ---
class PandasDataWithPred(bt.feeds.PandasData):
    lines = ('pred', 'conf',)
    params = (('pred', -1), ('conf', -1),)

# --- Strategy ---
class MyStrategy(bt.Strategy):
    params = dict(
        holdbar=1,                 
        trade_risk=0.05,           # 每次加仓 5% 总资金
        max_layers=5,              # 最大加仓层数
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

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                logger.debug(f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")
            elif order.issell():
                logger.debug(f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}")
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"Order Canceled/Margin/Rejected: {order.getstatusname()}")

    def stop(self):
        value = self.broker.getvalue()
        logger.info(f"End Value: {value:.2f}")
        
    def next(self):
        if len(self) % 1000 == 0:
            logger.debug(f"Processing bar {len(self)}")

        # 获取预测结果
        pred = self.data.pred[0]
        conf = self.data.conf[0]

        # 1. 数据有效性检查 (跳过 NaN)
        if np.isnan(pred) or np.isnan(conf):
            return

        # 2. 置信度过滤
        if self.params.thresh is not None and conf < self.params.thresh:
            return

        pred = int(pred)
        is_long_signal  = (pred == 2)
        is_short_signal = (pred == 0)
        # Label 1 (Ignore/Oscillation) 不作为开仓信号

        # 更新持仓计数
        if self.position:
            self.held_bars += 1
        else:
            self.held_bars = 0

        # 3. 确定目标方向
        target_dir = 0
        if is_long_signal and self.params.allow_long:
            target_dir = 1
        elif is_short_signal and self.params.allow_short:
            target_dir = -1
        
        # 4. 交易逻辑
        
        # A. 转向 (Reversal) 或 信号消失导致平仓
        # 如果信号方向与持仓方向不同 (例如：多转空，多转无)
        if self.dir != target_dir:
            if self.position:
                # 最小持仓时间限制
                if self.held_bars < self.params.holdbar:
                    return
                self.close() # 平仓
                self.layers = 0
                self.dir = 0
            
            # 如果有新方向，建立底仓
            if target_dir != 0:
                self.dir = target_dir
                self.layers = 1
                target_pct = self.params.trade_risk * self.layers * self.dir
                self.user_order_target_percent(target=target_pct)
            return

        # B. 同向加仓 (Pyramiding)
        if target_dir != 0 and target_dir == self.dir:
            # 补救逻辑：如果记录是空仓但实际没持仓
            if not self.position and self.layers == 0:
                self.layers = 1
                target_pct = self.params.trade_risk * self.layers * self.dir
                self.user_order_target_percent(target=target_pct)
                return
            
            # 加仓
            if self.layers < self.params.max_layers:
                self.layers += 1
                # 计算新的累计目标仓位 (例如 5% -> 10%)
                target_pct = self.params.trade_risk * self.layers * self.dir
                self.user_order_target_percent(target=target_pct)
                logger.debug(f"Pyramiding: Layer {self.layers}, Target {target_pct:.2%}")
            return

        # C. 纯平仓信号 (target_dir == 0) -> 上面 A 逻辑已覆盖，但如果需要单独处理震荡市平仓可在此添加

        # 强制最后平仓
        if len(self.data) - 1 == len(self) - 1 and self.position:
            self.close()

    def user_order_target_percent(self, target):
        cash = self.broker.get_cash() * target
        size = cash / self.dataclose[0]
        if cash > 0:
            self.buy(size=size)
        elif cash < 0:
            self.sell(size=abs(size))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run backtest.")
    parser.add_argument("--allow-short", action="store_true", default=True, help="Allow shorting")
    parser.add_argument("--allow-long", action="store_true", default=True, help="Allow longing")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--thresh", type=float, default=0.4, help="Confidence threshold")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Backtest settings: Short={args.allow_short}, Long={args.allow_long}, Thresh={args.thresh}")

    # 1. 数据加载
    data_path = test_data_path # from common.py
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
        
    # 直接读取 CSV，假设其中已包含所有特征列和时间列
    df = pd.read_csv(data_path)
    
    # 【关键】检查时间列是否存在
    if 'open_time_dt_utc' not in df.columns:
        logger.error("CRITICAL: 'open_time_dt_utc' column missing.")
        logger.error("Please ensure preparation.py saves the time column.")
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
    label_col = meta.get("label_col", "label")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    state = torch.load(MODEL_PATH, map_location=device)
    channel = state.get("channel", len(feature_cols))
    n_classes = len(state.get("classes", classes))
    
    model = CNN1D(channel=channel, n_classes=n_classes, p_drop=0.0).to(device)
    model.load_state_dict(state["state_dict"])
    
    # 3. 离线推理
    # 验证特征列是否齐备
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing feature columns in CSV: {missing_cols}")
        sys.exit(1)

    df_with_pred = offline_predict_cnn(df, feature_cols, label_col, window, model, device)
    
    # 删除预测结果为 NaN 的行 (即前 window-1 行)
    df_with_pred.dropna(subset=['pred'], inplace=True)
    
    logger.info(f"Data range: {df_with_pred['open_time_dt_utc'].min()} to {df_with_pred['open_time_dt_utc'].max()}")

    # 4. Backtrader 执行
    cerebro = bt.Cerebro(runonce=False) 
    
    cerebro.addstrategy(
        MyStrategy, 
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
    cerebro.broker.setcommission(commission=commission)

    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=1)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(cus_analyzer.CusAnalyzer, _name='customize')

    logger.info("Starting Backtest...")
    results = cerebro.run()
    strat = results[0]

    # 5. 结果统计
    perf = strat.analyzers.customize.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    sr = sharpe.get('sharperatio', 0.0)
    if sr is None: sr = 0.0

    mdd_pct = dd.get('max', {}).get('drawdown', 0.0)
    mdd_amt = dd.get('max', {}).get('moneydown', 0.0)
    
    n_closed = trades.get('total', {}).get('closed', 0)
    n_won    = trades.get('won', {}).get('total', 0)
    win_rate = (n_won / n_closed * 100.0) if n_closed > 0 else 0.0

    summary = (
        f"SUMMARY | GrossRet: {perf['gross_return']*100:.2f}% | CAGR: {perf['cagr']*100:.2f}% | "
        f"Sharpe: {sr:.3f} | MaxDD: {mdd_pct:.2f}% ({mdd_amt:.0f}) | "
        f"Trades: {n_closed} | WinRate: {win_rate:.2f}%"
    )
    logger.info(summary)