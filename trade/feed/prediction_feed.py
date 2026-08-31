"""
Feed layer: klines + model inference output.

The test for this layer is "can it exist independently of a trading venue" --
pred / pred_prob / expected_vol / label are computed offline and know nothing about account or position,
so they belong to the feed; equity and current position are only known to the venue and stay out of here.

PredictionFeed is its backtrader side carrier (maps df columns onto lines);
its live counterpart is venue/live/binance_data_feed.py.
"""

import backtrader as bt


class PredictionFeed(bt.feeds.PandasData):
    """Kline feed carrying model signals. Missing columns can be passed as -1 or omitted; the venue degrades safely via line_value()."""

    lines = (
        "pred",
        "pred_prob",
        "label",
        "expected_vol",
        "bars_to_close",
    )
    params = (
        ("pred", -1),
        ("pred_prob", -1),
        ("expected_vol", -1),   # EWMA volatility used by BBM sizing
        ("label", -1),
        ("bars_to_close", -1),
    )


class SignalOnlyFeed(bt.feeds.PandasData):
    """Slim version carrying only pred/pred_prob (for typical strategy backtests, no ATR/label needed)."""

    lines = (
        "pred",
        "pred_prob",
    )
    params = (
        ("pred", -1),
        ("pred_prob", -1),
    )
