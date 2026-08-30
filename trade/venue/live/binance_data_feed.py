from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

import pandas as pd
import requests
import websocket

from trade.feed.feed_base import ClosedCandleCallback, DataFeedBase


class BinanceDataFeed(DataFeedBase):
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
    MAX_HISTORY_ATTEMPTS = 3
    HISTORY_RETRY_DELAY_SECONDS = 0.5

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
        self._backfill_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._websocket_thread: Optional[threading.Thread] = None
        self._websocket_app: Optional[websocket.WebSocketApp] = None
        self._on_closed_candle: Optional[ClosedCandleCallback] = None
        self._interval_ms = 0

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
            "close_time_ms_utc",
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
            frame["open_time_ms_utc"],
            errors="raise",
        ).astype("int64")
        frame["close_time_ms_utc"] = pd.to_numeric(
            frame["close_time_ms_utc"],
            errors="raise",
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

    @staticmethod
    def _get_open_time_set(frame: pd.DataFrame) -> set[int]:
        if frame.empty or "open_time_ms_utc" not in frame.columns:
            return set()

        return set(
            pd.to_numeric(
                frame["open_time_ms_utc"],
                errors="coerce",
            )
            .dropna()
            .astype("int64")
        )

    @staticmethod
    def _filter_expected_opens(frame: pd.DataFrame, expected_open_times: set[int]) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        open_times = pd.to_numeric(
            frame["open_time_ms_utc"],
            errors="coerce",
        )
        result = frame.loc[open_times.isin(expected_open_times)].copy()
        result.drop_duplicates(
            "open_time_ms_utc",
            keep="last",
            inplace=True,
        )
        result.sort_values("open_time_ms_utc", inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    def _fetch_range_api(self, start_ts: int, end_ts: Optional[int] = None) -> Optional[pd.DataFrame]:
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

            cursor = int(data[-1][0]) + 1
            if len(data) < self.MAX_LIMIT_PER_REQUEST:
                break

            time.sleep(0.1)

        if not frames:
            return None

        return pd.concat(frames, ignore_index=True)

    def initialize_cache(self, required_bars: int, interval_ms: int) -> None:
        """Warm the cache with complete closed REST klines before startup."""

        if required_bars <= 0:
            raise ValueError("required_bars must be positive")
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        if required_bars > self.max_cache_len:
            raise ValueError("required_bars must not exceed the Binance cache capacity")

        self.logger.info("Initializing Binance closed-kline cache")

        now_ms = int(time.time() * 1000)
        latest_open_time_ms = now_ms // interval_ms * interval_ms - interval_ms
        first_open_time_ms = latest_open_time_ms - (required_bars - 1) * interval_ms

        expected_open_times = set(
            range(
                first_open_time_ms,
                latest_open_time_ms + interval_ms,
                interval_ms,
            )
        )

        frame = pd.DataFrame(columns=["open_time_ms_utc", "close_time_ms_utc"])
        missing_open_times = sorted(expected_open_times)

        for attempt in range(1, self.MAX_HISTORY_ATTEMPTS + 1):
            fetch_start_open_time_ms = missing_open_times[0]
            fetch_end_time_ms = missing_open_times[-1] + interval_ms

            try:
                restored = self._fetch_range_api(
                    fetch_start_open_time_ms,
                    fetch_end_time_ms,
                )
            except Exception:
                restored = None
                self.logger.exception(
                    "Binance history request failed | " "symbol=%s interval=%s attempt=%d/%d",
                    self.symbol,
                    self.interval,
                    attempt,
                    self.MAX_HISTORY_ATTEMPTS,
                )

            if restored is not None and not restored.empty:
                restored = self._filter_expected_opens(
                    restored,
                    expected_open_times,
                )
                if not restored.empty:
                    frame = pd.concat(
                        [frame, restored],
                        ignore_index=True,
                    )
                    frame = self._filter_expected_opens(
                        frame,
                        expected_open_times,
                    )

            received_open_times = self._get_open_time_set(frame)
            missing_open_times = sorted(expected_open_times - received_open_times)

            if not missing_open_times:
                break

            if attempt < self.MAX_HISTORY_ATTEMPTS:
                self.logger.warning(
                    "Binance history incomplete; retrying | "
                    "symbol=%s interval=%s attempt=%d/%d missing=%d "
                    "first_missing_open_time_ms=%d "
                    "last_missing_open_time_ms=%d",
                    self.symbol,
                    self.interval,
                    attempt,
                    self.MAX_HISTORY_ATTEMPTS,
                    len(missing_open_times),
                    missing_open_times[0],
                    missing_open_times[-1],
                )
                time.sleep(self.HISTORY_RETRY_DELAY_SECONDS)

        if missing_open_times:
            raise RuntimeError(
                "Failed to acquire complete Binance historical market data "
                f"after {self.MAX_HISTORY_ATTEMPTS} attempts: "
                f"missing open times {missing_open_times}"
            )

        frame.reset_index(drop=True, inplace=True)

        candle_durations = (
            pd.to_numeric(
                frame["close_time_ms_utc"],
                errors="raise",
            )
            - pd.to_numeric(
                frame["open_time_ms_utc"],
                errors="raise",
            )
            + 1
        )

        if len(frame) != required_bars or not candle_durations.eq(interval_ms).all():
            raise RuntimeError("Failed to acquire complete Binance historical market data: " "invalid candle count or duration")

        last_open_time_ms = int(frame.iloc[-1]["open_time_ms_utc"])

        with self._cache_lock:
            self.local_cache = frame
            self._interval_ms = int(interval_ms)

        self.logger.info(
            "Binance cache initialized | symbol=%s interval=%s bars=%d " "last_open_time_ms=%d",
            self.symbol,
            self.interval,
            len(frame),
            last_open_time_ms,
        )

    def start(self, on_closed_candle: ClosedCandleCallback) -> None:
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
                app.run_forever(
                    ping_interval=20,
                    ping_timeout=10,
                )
            except Exception:
                self.logger.exception(
                    "Binance WebSocket stopped unexpectedly | " "symbol=%s interval=%s",
                    self.symbol,
                    self.interval,
                )
            finally:
                self._connected_event.clear()
                self._websocket_app = None

            if not self._stop_event.wait(self.RECONNECT_DELAY_SECONDS):
                self.logger.warning(
                    "Reconnecting Binance WebSocket | " "symbol=%s interval=%s",
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

    def _on_websocket_error(self, _app: websocket.WebSocketApp, error: Any) -> None:
        if not self._stop_event.is_set():
            self.logger.error(
                "Binance WebSocket error | " "symbol=%s interval=%s error=%s",
                self.symbol,
                self.interval,
                error,
            )

    def _on_websocket_close(self, _app: websocket.WebSocketApp, status_code: Optional[int], message: Optional[str]) -> None:
        self._connected_event.clear()
        if not self._stop_event.is_set():
            self.logger.warning(
                "Binance WebSocket closed | " "symbol=%s interval=%s code=%s message=%s",
                self.symbol,
                self.interval,
                status_code,
                message,
            )

    def _on_websocket_message(self, _app: websocket.WebSocketApp, raw_message: str) -> None:
        try:
            payload = json.loads(raw_message)
            if "data" in payload:
                payload = payload["data"]

            kline = payload.get("k")
            if not kline:
                return

            if kline.get("x") is not True:
                self.logger.debug(
                    "Received open Binance kline | " "symbol=%s interval=%s open_time_ms=%s close_time_ms=%s",
                    self.symbol,
                    self.interval,
                    kline.get("t"),
                    kline.get("T"),
                )
                return

            candle_open_time_ms = self._store_closed_kline(kline)
        except Exception:
            self.logger.exception(
                "Invalid Binance WebSocket kline | " "symbol=%s interval=%s",
                self.symbol,
                self.interval,
            )
            return

        callback = self._on_closed_candle
        if candle_open_time_ms is not None and callback is not None:
            try:
                callback(candle_open_time_ms)
            except Exception:
                self.logger.exception(
                    "Closed-kline callback failed | " "symbol=%s interval=%s open_time_ms_utc=%d",
                    self.symbol,
                    self.interval,
                    candle_open_time_ms,
                )

    def _store_closed_kline(self, kline: dict[str, Any]) -> Optional[int]:
        """Store a live closed kline and asynchronously repair detected gaps."""

        candle_open_time_ms = int(kline["t"])
        candle_close_time_ms = int(kline["T"])

        row = {
            "open_time_ms_utc": candle_open_time_ms,
            "open": kline["o"],
            "high": kline["h"],
            "low": kline["l"],
            "close": kline["c"],
            "volume": kline["v"],
            "close_time_ms_utc": candle_close_time_ms,
            "quote_asset_volume": kline["q"],
            "number_of_trades": kline["n"],
            "taker_buy_base_volume": kline["V"],
            "taker_buy_quote_volume": kline["Q"],
            "ignore": kline.get("B", "0"),
        }
        frame = self._normalize_frame(pd.DataFrame([row]))

        missing_start_open_time_ms: Optional[int] = None
        missing_end_open_time_ms: Optional[int] = None

        with self._cache_lock:
            if self.local_cache is None:
                raise RuntimeError("Binance data cache is not initialized")
            if self.local_cache.empty:
                raise RuntimeError("Binance data cache is empty")
            if self._interval_ms <= 0:
                raise RuntimeError("Binance candle interval is not initialized")

            latest_cached_open_time_ms = int(self.local_cache.iloc[-1]["open_time_ms_utc"])

            if candle_open_time_ms <= latest_cached_open_time_ms:
                self.logger.info(
                    "Ignoring stale Binance closed kline | " "symbol=%s interval=%s open_time_ms=%d " "latest_cached_open_time_ms=%d",
                    self.symbol,
                    self.interval,
                    candle_open_time_ms,
                    latest_cached_open_time_ms,
                )
                return None

            open_time_delta = candle_open_time_ms - latest_cached_open_time_ms

            if open_time_delta % self._interval_ms != 0:
                raise RuntimeError("Binance closed-kline timestamp is not aligned with " "the configured interval")

            if open_time_delta > self._interval_ms:
                missing_start_open_time_ms = latest_cached_open_time_ms + self._interval_ms
                missing_end_open_time_ms = candle_open_time_ms - self._interval_ms

                self.logger.warning(
                    "Binance closed-kline gap detected | "
                    "symbol=%s interval=%s latest_cached_open_time_ms=%d "
                    "current_open_time_ms=%d "
                    "first_missing_open_time_ms=%d "
                    "last_missing_open_time_ms=%d",
                    self.symbol,
                    self.interval,
                    latest_cached_open_time_ms,
                    candle_open_time_ms,
                    missing_start_open_time_ms,
                    missing_end_open_time_ms,
                )

            self.local_cache = pd.concat(
                [self.local_cache, frame],
                ignore_index=True,
            )
            self.local_cache.drop_duplicates(
                "open_time_ms_utc",
                keep="last",
                inplace=True,
            )
            self.local_cache.sort_values(
                "open_time_ms_utc",
                inplace=True,
            )
            self.local_cache = self.local_cache.iloc[-self.max_cache_len :].reset_index(drop=True)

        if missing_start_open_time_ms is not None and missing_end_open_time_ms is not None:
            self._start_nonblocking_backfill(
                missing_start_open_time_ms,
                missing_end_open_time_ms,
            )

        return candle_open_time_ms

    def _start_nonblocking_backfill(self, start_open_time_ms: int, end_open_time_ms: int) -> None:
        """Start background cache repair without strategy notifications."""

        if start_open_time_ms > end_open_time_ms:
            return

        thread = threading.Thread(
            target=self._repair_missing_range,
            args=(
                int(start_open_time_ms),
                int(end_open_time_ms),
            ),
            name=(f"binance-backfill-{self.symbol}-" f"{start_open_time_ms}-{end_open_time_ms}"),
            daemon=True,
        )
        thread.start()

    def _repair_missing_range(self, start_open_time_ms: int, end_open_time_ms: int) -> None:
        """Repair a historical cache gap with finite retries and no callback."""

        with self._backfill_lock:
            interval_ms = self._interval_ms
            if interval_ms <= 0:
                self.logger.error(
                    "Cannot repair Binance cache before interval initialization | " "symbol=%s interval=%s",
                    self.symbol,
                    self.interval,
                )
                return

            expected_open_times = set(
                range(
                    int(start_open_time_ms),
                    int(end_open_time_ms) + interval_ms,
                    interval_ms,
                )
            )

            missing_open_times = sorted(expected_open_times)

            for attempt in range(1, self.MAX_HISTORY_ATTEMPTS + 1):
                if self._stop_event.is_set():
                    return

                with self._cache_lock:
                    if self.local_cache is None:
                        return
                    cached_open_times = self._get_open_time_set(self.local_cache)

                missing_open_times = sorted(expected_open_times - cached_open_times)
                if not missing_open_times:
                    return

                fetch_start_open_time_ms = missing_open_times[0]
                fetch_end_open_time_ms = missing_open_times[-1]

                try:
                    self._backfill_cache_once(
                        fetch_start_open_time_ms,
                        fetch_end_open_time_ms,
                        interval_ms,
                    )
                except Exception:
                    self.logger.exception(
                        "Binance background backfill request failed | " "symbol=%s interval=%s attempt=%d/%d " "first_open_time_ms=%d last_open_time_ms=%d",
                        self.symbol,
                        self.interval,
                        attempt,
                        self.MAX_HISTORY_ATTEMPTS,
                        fetch_start_open_time_ms,
                        fetch_end_open_time_ms,
                    )

                with self._cache_lock:
                    if self.local_cache is None:
                        return
                    cached_open_times = self._get_open_time_set(self.local_cache)

                missing_open_times = sorted(expected_open_times - cached_open_times)

                if not missing_open_times:
                    self.logger.info(
                        "Binance background backfill completed | " "symbol=%s interval=%s bars=%d " "first_open_time_ms=%d last_open_time_ms=%d",
                        self.symbol,
                        self.interval,
                        len(expected_open_times),
                        start_open_time_ms,
                        end_open_time_ms,
                    )
                    return

                if attempt < self.MAX_HISTORY_ATTEMPTS:
                    self.logger.warning(
                        "Binance background backfill incomplete; retrying | "
                        "symbol=%s interval=%s attempt=%d/%d missing=%d "
                        "first_missing_open_time_ms=%d "
                        "last_missing_open_time_ms=%d",
                        self.symbol,
                        self.interval,
                        attempt,
                        self.MAX_HISTORY_ATTEMPTS,
                        len(missing_open_times),
                        missing_open_times[0],
                        missing_open_times[-1],
                    )

                    if self._stop_event.wait(self.HISTORY_RETRY_DELAY_SECONDS):
                        return

            self.logger.error(
                "Binance background backfill failed | " "symbol=%s interval=%s missing=%s",
                self.symbol,
                self.interval,
                missing_open_times,
            )

    def _backfill_cache_once(
        self,
        start_open_time_ms: int,
        end_open_time_ms: int,
        interval_ms: int,
    ) -> int:
        """Fetch and merge one historical range without strategy notification."""

        if start_open_time_ms > end_open_time_ms:
            return 0
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        if (end_open_time_ms - start_open_time_ms) % interval_ms != 0:
            raise ValueError("Backfill candle range must align with interval_ms")

        with self._cache_lock:
            if self.local_cache is None:
                raise RuntimeError("Binance data cache is not initialized")
            if self._interval_ms > 0 and interval_ms != self._interval_ms:
                raise ValueError("Backfill interval_ms must match the initialized cache interval")

        now_ms = int(time.time() * 1000)
        fetch_end_ms = min(
            now_ms,
            int(end_open_time_ms) + int(interval_ms),
        )

        frame = self._fetch_range_api(
            start_open_time_ms,
            fetch_end_ms,
        )
        if frame is None or frame.empty:
            return 0

        expected_open_times = set(
            range(
                int(start_open_time_ms),
                int(end_open_time_ms) + int(interval_ms),
                int(interval_ms),
            )
        )
        frame = self._filter_expected_opens(
            frame,
            expected_open_times,
        )
        if frame.empty:
            return 0

        with self._cache_lock:
            if self.local_cache is None:
                raise RuntimeError("Binance data cache is not initialized")

            before_open_times = self._get_open_time_set(self.local_cache)

            self.local_cache = pd.concat(
                [self.local_cache, frame],
                ignore_index=True,
            )
            self.local_cache.drop_duplicates(
                "open_time_ms_utc",
                keep="last",
                inplace=True,
            )
            self.local_cache.sort_values(
                "open_time_ms_utc",
                inplace=True,
            )
            self.local_cache = self.local_cache.iloc[-self.max_cache_len :].reset_index(drop=True)

            after_open_times = self._get_open_time_set(self.local_cache)

        return len(after_open_times - before_open_times)

    def get_latest_data(self) -> Optional[pd.DataFrame]:
        """Return a copy of the closed-kline cache without making a REST request."""

        with self._cache_lock:
            if self.local_cache is None:
                return None
            return self.local_cache.copy()

    def backfill_cache(
        self,
        start_close_time_ms: int,
        end_close_time_ms: int,
        interval_ms: int,
    ) -> int:
        """Synchronously repair cache history without strategy notification."""

        start_open_time_ms = start_close_time_ms - interval_ms + 1
        end_open_time_ms = end_close_time_ms - interval_ms + 1
        with self._backfill_lock:
            return self._backfill_cache_once(
                start_open_time_ms,
                end_open_time_ms,
                interval_ms,
            )

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
