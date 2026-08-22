from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Optional

import pandas as pd
import requests
import websocket


class BinanceDataFeed:
    """Maintain closed Binance klines from REST history and WebSocket events."""

    REST_URLS: dict[str, str] = {
        "spot": "https://api.binance.com/api/v3/klines",
        "um": "https://fapi.binance.com/fapi/v1/klines",
        "cm": "https://dapi.binance.com/dapi/v1/klines",
    }
    WEBSOCKET_URLS: dict[str, str] = {
        "spot": "wss://stream.binance.com:9443/ws/{stream}",
        "um": "wss://fstream.binance.com/ws/{stream}",
        "cm": "wss://dstream.binance.com/ws/{stream}",
    }
    MAX_LIMIT_PER_REQUEST = 1000
    RECONNECT_DELAY_SECONDS = 5.0

    def __init__(
        self,
        symbol: str,
        interval: str,
        trading_type: str,
        max_len: int = 5000,
    ):
        if trading_type not in self.REST_URLS:
            raise ValueError(f"Unsupported Binance trading type: {trading_type!r}")
        self.symbol = symbol
        self.interval = interval
        self.trading_type = trading_type
        self.rest_url = self.REST_URLS[trading_type]
        stream = f"{symbol.lower()}@kline_{interval}"
        self.websocket_url = self.WEBSOCKET_URLS[trading_type].format(stream=stream)
        self.max_cache_len = max_len
        self.logger = logging.getLogger("BinanceFeed")

        self.local_cache: Optional[pd.DataFrame] = None
        self._cache_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._websocket_thread: Optional[threading.Thread] = None
        self._websocket_app: Optional[websocket.WebSocketApp] = None
        self._on_closed_candle: Optional[Callable[[int], None]] = None
        self._last_notified_candle_id: Optional[int] = None

    @staticmethod
    def _process_rest_data(data: list[list[Any]]) -> Optional[pd.DataFrame]:
        if not data:
            return None
        columns = [
            "open_time_ms_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        frame = pd.DataFrame(data, columns=columns)
        return BinanceDataFeed._normalize_frame(frame)

    @staticmethod
    def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["open_time_ms_utc"] = pd.to_numeric(
            frame["open_time_ms_utc"], errors="raise"
        ).astype("int64")
        frame["close_time_ms"] = pd.to_numeric(
            frame["close_time_ms"], errors="raise"
        ).astype("int64")
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
        ]
        frame[numeric_columns] = frame[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        frame["open_time_date_utc"] = pd.to_datetime(
            frame["open_time_ms_utc"],
            unit="ms",
            utc=True,
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
        return frame

    def _fetch_range_api(
        self,
        start_ts: int,
        end_ts: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch REST history. Runtime candle delivery does not use this path."""

        if end_ts is None:
            end_ts = int(time.time() * 1000)
        frames: list[pd.DataFrame] = []
        cursor = start_ts

        while cursor < end_ts:
            response = requests.get(
                self.rest_url,
                params={
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "startTime": cursor,
                    "endTime": end_ts,
                    "limit": self.MAX_LIMIT_PER_REQUEST,
                },
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list) or not data:
                break
            frame = self._process_rest_data(data)
            if frame is not None:
                frames.append(frame)
            cursor = int(data[-1][6]) + 1
            if len(data) < self.MAX_LIMIT_PER_REQUEST:
                break
            time.sleep(0.1)

        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    def initialize_cache(self, required_bars: int, interval_ms: int) -> None:
        """Warm the cache with closed REST klines before WebSocket startup."""

        self.logger.info("Initializing Binance closed-kline cache")
        duration_ms = int(required_bars * interval_ms * 1.2) + 100
        now_ms = int(time.time() * 1000)
        frame = self._fetch_range_api(now_ms - duration_ms, now_ms)
        if frame is None or frame.empty:
            raise RuntimeError("Failed to initialize Binance data cache")

        frame = frame.loc[frame["close_time_ms"] <= now_ms].copy()
        if frame.empty:
            raise RuntimeError("Binance history contains no closed klines")
        frame.drop_duplicates("open_time_ms_utc", keep="last", inplace=True)
        frame.sort_values("open_time_ms_utc", inplace=True)
        frame = frame.iloc[-self.max_cache_len :].reset_index(drop=True)

        with self._cache_lock:
            self.local_cache = frame
            self._last_notified_candle_id = int(frame.iloc[-1]["open_time_ms_utc"])
        self.logger.info(
            "Binance cache initialized | symbol=%s interval=%s bars=%d last=%s",
            self.symbol,
            self.interval,
            len(frame),
            frame.iloc[-1]["open_time_date_utc"],
        )

    def start(self, on_closed_candle: Callable[[int], None]) -> None:
        """Start the reconnecting WebSocket listener for closed-kline events."""

        if self.local_cache is None:
            raise RuntimeError("Binance data cache must be initialized before start")
        if self._websocket_thread is not None and self._websocket_thread.is_alive():
            return
        self._on_closed_candle = on_closed_candle
        self._stop_event.clear()
        self._connected_event.clear()
        self._websocket_thread = threading.Thread(
            target=self._run_websocket,
            name=f"binance-kline-{self.symbol}-{self.interval}",
            daemon=True,
        )
        self._websocket_thread.start()

    def _run_websocket(self) -> None:
        while not self._stop_event.is_set():
            app = websocket.WebSocketApp(
                self.websocket_url,
                on_open=self._on_websocket_open,
                on_message=self._on_websocket_message,
                on_error=self._on_websocket_error,
                on_close=self._on_websocket_close,
            )
            self._websocket_app = app
            try:
                app.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                self.logger.exception(
                    "Binance WebSocket stopped unexpectedly | symbol=%s interval=%s",
                    self.symbol,
                    self.interval,
                )
            finally:
                self._connected_event.clear()
                self._websocket_app = None
            if not self._stop_event.wait(self.RECONNECT_DELAY_SECONDS):
                self.logger.warning(
                    "Reconnecting Binance WebSocket | symbol=%s interval=%s",
                    self.symbol,
                    self.interval,
                )

    def _on_websocket_open(self, _app: websocket.WebSocketApp) -> None:
        self._connected_event.set()
        self.logger.info(
            "Binance WebSocket connected | symbol=%s interval=%s",
            self.symbol,
            self.interval,
        )

    def _on_websocket_error(
        self,
        _app: websocket.WebSocketApp,
        error: Any,
    ) -> None:
        if not self._stop_event.is_set():
            self.logger.error(
                "Binance WebSocket error | symbol=%s interval=%s error=%s",
                self.symbol,
                self.interval,
                error,
            )

    def _on_websocket_close(
        self,
        _app: websocket.WebSocketApp,
        status_code: Optional[int],
        message: Optional[str],
    ) -> None:
        self._connected_event.clear()
        if not self._stop_event.is_set():
            self.logger.warning(
                "Binance WebSocket closed | symbol=%s interval=%s code=%s message=%s",
                self.symbol,
                self.interval,
                status_code,
                message,
            )

    def _on_websocket_message(
        self,
        _app: websocket.WebSocketApp,
        raw_message: str,
    ) -> None:
        try:
            payload = json.loads(raw_message)
            if "data" in payload:
                payload = payload["data"]
            kline = payload.get("k")
            if not kline or not kline.get("x"):
                return
            candle_id = self._store_closed_kline(kline)
        except Exception:
            self.logger.exception(
                "Invalid Binance WebSocket kline | symbol=%s interval=%s",
                self.symbol,
                self.interval,
            )
            return

        callback = self._on_closed_candle
        if candle_id is not None and callback is not None:
            try:
                callback(candle_id)
            except Exception:
                self.logger.exception(
                    "Closed-kline callback failed | symbol=%s interval=%s candle=%d",
                    self.symbol,
                    self.interval,
                    candle_id,
                )

    def _store_closed_kline(self, kline: dict[str, Any]) -> Optional[int]:
        candle_id = int(kline["t"])
        row = {
            "open_time_ms_utc": candle_id,
            "open": kline["o"],
            "high": kline["h"],
            "low": kline["l"],
            "close": kline["c"],
            "volume": kline["v"],
            "close_time_ms": int(kline["T"]),
            "quote_asset_volume": kline["q"],
            "number_of_trades": kline["n"],
            "taker_buy_base_volume": kline["V"],
            "taker_buy_quote_volume": kline["Q"],
            "ignore": kline.get("B", "0"),
        }
        frame = self._normalize_frame(pd.DataFrame([row]))

        with self._cache_lock:
            if self.local_cache is None:
                raise RuntimeError("Binance data cache is not initialized")
            self.local_cache = pd.concat(
                [self.local_cache, frame],
                ignore_index=True,
            )
            self.local_cache.drop_duplicates(
                "open_time_ms_utc",
                keep="last",
                inplace=True,
            )
            self.local_cache.sort_values("open_time_ms_utc", inplace=True)
            self.local_cache = self.local_cache.iloc[-self.max_cache_len :].reset_index(
                drop=True
            )
            if (
                self._last_notified_candle_id is not None
                and candle_id <= self._last_notified_candle_id
            ):
                return None
            self._last_notified_candle_id = candle_id
        return candle_id

    def get_latest_data(self) -> Optional[pd.DataFrame]:
        """Return a copy of the closed-kline cache without making a REST request."""

        with self._cache_lock:
            if self.local_cache is None:
                return None
            return self.local_cache.copy()

    def close(self) -> None:
        self._stop_event.set()
        app = self._websocket_app
        if app is not None:
            app.close()
        thread = self._websocket_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._websocket_thread = None
        self._connected_event.clear()

    shutdown = close
