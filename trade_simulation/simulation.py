from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import datetime  # For datetime objects
import os,sys,time,json  # To find out the script name (in argv[0])

# Import the backtrader platform
import backtrader as bt
import backtrader.analyzers as btanalyzers 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import logging

current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from data.common import *
from model.cnn_timeseries_torch import CNN1D
import cus_analyzer

MODEL_PATH = os.path.join(current_work_dir, '..', 'model',"cnn_timeseries_torch_model.pt")
META_PATH  = os.path.join(current_work_dir, '..', 'model',"cnn_timeseries_torch_meta.json")
log_file= os.path.join(current_work_dir, "backtest.log")

class TradeResult():
    def __init__(self) -> None:
        self.times = 0

_cash = 10000
commission = 0.001
# ------------- Logging Utilities -------------
def setup_logging(log_file: str, verbose: bool = False):
    """
    Configure logging to both console and log file (overwrite mode).
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    level = log_level

    # 文件 handler（覆盖写）
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # 终端 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # 清空旧的 handler，避免重复打印
    root = logging.getLogger("backtest")
    if root.hasHandlers():
        root.handlers.clear()

    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return root

logger = setup_logging(log_file)

# --- NEW: 一次性离线批量推理 ---
def offline_predict_cnn(df: pd.DataFrame,
                        feature_cols: list[str],
                        window: int,
                        scaler: StandardScaler,
                        model: nn.Module,
                        device: torch.device,
                        batch_size: int = 8192) -> pd.DataFrame:
    """
    用训练时的特征列+窗口，在回测前一次性跑完模型，返回含 'pred','conf' 的 df。
    索引与 df 对齐，前 window-1 行为 NaN。
    """
    feat = df[feature_cols].astype(np.float32).copy()

    # 标准化（与训练一致）
    X2d = scaler.transform(feat.values)      # 形状: (T, F)
    T, F = X2d.shape
    if T < window:
        raise ValueError(f"数据长度 {T} 小于窗口 {window}")

    # 构造所有滑窗 (N = T-window+1, window, F)
    strides = (X2d.strides[0], X2d.strides[0], X2d.strides[1])
    shape   = (T - window + 1, window, F)
    X3d = np.lib.stride_tricks.as_strided(X2d, shape=shape, strides=strides).copy()

    # 批量推理
    model.eval()
    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, X3d.shape[0], batch_size):
            xb = torch.from_numpy(X3d[i:i+batch_size]).to(device)   # (B, T, F)
            logits = model(xb)                                      # (B, C)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds.append(probs.argmax(axis=1).astype(np.int64))
            confs.append(probs.max(axis=1).astype(np.float32))
    preds = np.concatenate(preds); confs = np.concatenate(confs)

    # 对齐索引：窗口末尾对应当前时刻
    idx = feat.index[window-1:]
    out = pd.DataFrame({'pred': preds, 'conf': confs}, index=idx)

    # 拼回原 df，前 window-1 行填 NaN
    df_out = df.copy()
    df_out[['pred', 'conf']] = np.nan
    df_out.loc[idx, 'pred'] = out['pred'].values
    df_out.loc[idx, 'conf'] = out['conf'].values
    return df_out

# --- NEW: DataFeed 扩展，增加 'pred' 和 'conf' 两条线 ---
class PandasDataWithPred(bt.feeds.PandasData):
    lines = ('pred', 'conf',)
    params = (('pred', -1), ('conf', -1),)

# Create a Strategy
class MyStrategy(bt.Strategy):
    params = dict(
        holdbar=1,                 # 最短持有K线数，避免过度交易
        trade_risk=0.02,           # 固定风险头寸 sizing 示例
        allow_short=False,         # 是否允许做空
        allow_long=True,           # 是否允许做多
        thresh=None,               # 预测置信度阈值（可选）
    )

    def __init__(self):
        # logger from root
        # 数据句柄
        self.dataclose = self.datas[0].close

        # 交易控制变量（保留你的命名）
        self.bar_executed = None
        self.held_bars = 0
        self.dir = 0
        self.layers = 0

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
            logger.debug("Order Canceled/Margin/Rejected", level=logging.WARNING)

    def notify_trade(self, trade):
        if trade.isclosed:
            logger.debug(f"OPERATION PROFIT, Gross {trade.pnl:.2f}, Net {trade.pnlcomm:.2f}")

    def stop(self):
        # 结尾的关键指标信息：不受 verbose 控制，但仍写入日志文件
        cash = self.broker.getcash()
        value = self.broker.getvalue()
        logger.info(f"当前可用现金: {cash:.2f}")
        logger.info(f"当前总资产: {value:.2f}")
        logger.info(f"(Thresh {self.p.thresh}) Ending Value {self.broker.getvalue():.2f}")
        
    def next(self):
        i = len(self) - 1
        if i % 50000 == 0:
            logger.info(f"bar {i}")

        pred = float(self.data.pred[0])
        conf = float(self.data.conf[0])

        # 预测尚不可用（窗口未满/NaN）
        if np.isnan(pred) or np.isnan(conf):
            return

        pred = int(pred)
        if self.p.thresh is not None and conf < self.p.thresh:
            return

        is_long_signal  = (pred == 2)
        is_flat_signal  = (pred == 1)
        is_short_signal = (pred == 0)

        # 持有计数
        if self.position:
            self.held_bars += 1
        else:
            self.held_bars = 0

        # 目标方向（保留你的风格）
        if is_long_signal and self.p.allow_long:
            new_dir = 1
        elif is_short_signal and self.p.allow_short:
            new_dir = -1
        else:
            new_dir = 0

        # 换向：先平再开
        if self.dir != new_dir:
            if self.position:
                if self.held_bars < self.p.holdbar:
                    return
                self.close()
            self.layers = 0
            self.dir = 0

            if new_dir == 0:
                return

            self.layers = 1
            self.dir = new_dir
            self.user_order_target_percent(target=self.params.trade_risk * self.dir)
            return

        # 同向加仓
        if new_dir != 0:
            if not self.position and self.layers == 0:
                self.layers = 1
                self.dir = new_dir
                self.user_order_target_percent(target=self.params.trade_risk * self.dir)
                return
            if self.layers < 30:
                self.layers += 1
                self.user_order_target_percent(target=self.params.trade_risk * self.dir)
            return

        # 观望→平仓
        if new_dir == 0 and self.position and self.held_bars >= self.p.holdbar:
            self.close()
            self.layers = 0
            self.dir = 0

        # —— 在最后一根K线强制平仓（避免留单）——
        is_last_bar = (len(self.data) - 1 == len(self) - 1)
        if is_last_bar and self.position:
            self.close()  # 平当前数据的仓

    def user_order_target_percent(self, target):
        cash = self.broker.get_cash() * target
        size = cash / self.dataclose[0]
        if cash > 0:
            self.buy(size=size)
        elif cash < 0:
            self.sell(size=abs(size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run backtest with optional logging and reduced console noise.")
    parser.add_argument("--allow-short", action="store_true", default=True, help="是否允许做空（默认允许）")
    parser.add_argument("--allow-long", action="store_true", default=True, help="是否允许做多（默认允许）")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志（默认关闭）")
    parser.add_argument("--log-file", type=str, default=os.path.join(current_work_dir, "backtest.log"),
                        help="日志文件路径（默认 ../logs/backtest.log)")
    args = parser.parse_args()

    # Setup logging & redirect all output to file (overwrite)
    logger.info("Backtest started. verbose=%s, log_file=%s", args.verbose, args.log_file)

    data_path = os.path.join(current_work_dir,'..', 'data', test_data)
    df = pd.read_csv(data_path)
    df = add_weekly_mas(df, weeks=(7, 25), method='sma', strict=False) #用到'open_time_dt_utc'
    # 时间处理（K线开始时间通常是毫秒时间戳）
    df['open_time_dt_utc'] = pd.to_datetime(df['open_time_dt_utc'], utc=True).dt.tz_convert(None)
    df['close_time_dt_utc'] = pd.to_datetime(df['close_time_dt_utc'], utc=True).dt.tz_convert(None)
    logger.info("Columns: {},data range:{} --> {}".format(list(df.columns),df['open_time_dt_utc'].min(),df['open_time_dt_utc'].max()))
    # df = df.iloc[:5000]   #chose a few day only

    # ---------- NEW: 离线生成预测列 ----------
    # 1) 先把训练时用到的特征算出来（保持与训练一致）
    df_feat = add_macd(df)  # 你原来在策略里做的，提前到这里

    # 2) 读取 meta & 恢复 scaler
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    window = int(meta["window"])
    classes = meta["classes"]

    scaler = StandardScaler()
    scaler.mean_  = np.array(meta["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(meta["scaler_scale"], dtype=np.float64)
    scaler.var_   = scaler.scale_ ** 2

    # 3) 加载模型（一次性）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channel = len(feature_cols); n_classes = len(classes)
    model = CNN1D(channel=channel, n_classes=n_classes, p_drop=0.0).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()

    # 4) 一次性离线推理，得到 pred/conf 列
    df_with_pred = offline_predict_cnn(df_feat, feature_cols, window, scaler, model, device)

    # （可选）缓存到 parquet，重复回测直接读取
    # df_with_pred.to_parquet(os.path.join(current_work_dir, "..", "data", "data_with_pred.parquet"))
    # ---------- /NEW ----------


    # Create a cerebro entity
    # cerebro = bt.Cerebro(stdstats=False)
    cerebro = bt.Cerebro(preload=True, runonce=True, stdstats=False)#faster

    # Add a strategy
    cerebro.addstrategy(MyStrategy, allow_short=args.allow_short, allow_long=args.allow_long)

    # Pass it to the backtrader datafeed and add it to the cerebro
    data = PandasDataWithPred(
        dataname=df_with_pred,
        # Do not pass values before this date
        # fromdate=datetime.datetime(2023, 7, 1),
        # # Do not pass values before this date
        # todate=datetime.datetime(2024, 3, 1),
        datetime=df_with_pred.columns.get_loc('open_time_dt_utc'),
        open=df_with_pred.columns.get_loc('open'),
        high=df_with_pred.columns.get_loc('high'),
        low=df_with_pred.columns.get_loc('low'),
        close=df_with_pred.columns.get_loc('close'),
        volume=df_with_pred.columns.get_loc('volume'),
        nocase=True
    )

    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(_cash)
    cerebro.broker.setcommission(commission=commission)  # 0.06% 示例

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)

    # 分析器（回测绩效）
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=1)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='dd')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    # cerebro.addanalyzer(btanalyzers.AnnualReturn, _name='annual') #can't work
    # cerebro.addanalyzer(btanalyzers.TimeReturn, _name='timereturn') #too much. 给出每一根 bar 的投资组合收益率（相对上一根 bar 的净值变化）
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')  # CAGR, 平均收益率
    cerebro.addanalyzer(cus_analyzer.CusAnalyzer, _name='customize')

    # results = cerebro.run(maxcpus=4)
    results = cerebro.run(exactbars=False, maxcpus=4) #faster
    strat = results[0]
    # ---- 提取各分析器的关键信息（只选最有用的字段）----
    sharpe = strat.analyzers.sharpe.get_analysis()
    dd     = strat.analyzers.dd.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    perf   = strat.analyzers.customize.get_analysis()  # 你的一体化分析器（总收益率/CAGR/暴露）

    # Sharpe
    sr = sharpe.get('sharperatio', float('nan'))

    # 最大回撤（% 和 金额）
    mdd_pct  = dd.get('max', {}).get('drawdown', float('nan'))
    mdd_amt  = dd.get('max', {}).get('moneydown', float('nan'))

    # 交易统计（成交笔数、胜率、平均净盈亏）
    n_closed = trades.get('total', {}).get('closed', 0)
    n_won    = trades.get('won', {}).get('total', 0)
    win_rate = (n_won / n_closed * 100.0) if n_closed else 0.0
    avg_pnl  = trades.get('pnl', {}).get('net', {}).get('average', float('nan'))

    # ---- 一条“总览”即可：清晰、浓缩 ----
    logger.info(
        ("SUMMARY | commission:{com:2f}% GrossRet: {ret:.2f}% | CAGR: {cagr:.2f}% | "
        "Sharpe: {sr:.3f} | MaxDD: {mdd:.2f}% ({mdd_amt:.0f}) | "
        "Trades: {nt} | WinRate: {wr:.2f}% | AvgPnL: {ap:.2f} | "
        "Expo avg/p95/max: {avg:.1f}%/{p95:.1f}%/{mx:.1f}%").format(
            com=commission*100.0,
            ret=perf['gross_return']*100.0,
            cagr=perf['cagr']*100.0,
            sr=sr,
            mdd=mdd_pct, mdd_amt=mdd_amt,
            nt=n_closed, wr=win_rate, ap=avg_pnl,
            avg=perf['avg_pos_ratio']*100.0,
            p95=perf['p95_pos_ratio']*100.0,
            mx=perf['max_pos_ratio']*100.0,
    ))

    # ---- 可选：只有在 --verbose 时，才打印详细原始字典（便于排障）----
    if args.verbose:
        logger.info('Sharpe(raw): %s', sharpe)
        logger.info('DrawDown(raw): %s', dd)
        logger.info('Trades(raw): %s', trades)

    # 绘图可能在无GUI环境下失败
    if args.verbose:
        try:
            cerebro.plot(style='bar')
        except Exception as e:
            logger.warning("Plotting skipped: %s", e)
