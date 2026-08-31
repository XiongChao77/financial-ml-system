from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import requests
import websocket

from trade.feed.feed_base import ClosedCandleCallback, DataFeedBase


def _format_utc_ms(timestamp_ms: Any) -> str:
    """Format an epoch-millisecond timestamp for human-readable logs."""

    try:
        timestamp = int(timestamp_ms)
    except (TypeError, ValueError):
        return str(timestamp_ms)
    return datetime.fromtimestamp(
        timestamp / 1000,
        tz=timezone.utc,
    ).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]


class BinanceDataFeed(DataFeedBase):
    """Maintain closed Binance klines from REST history and WebSocket events."""

    REST_URLS: dict[str, str] = {
        "spot": "https://api.binance.com/api/v3/klines",
        "um": "https://fapi.binance.com/fapi/v1/klines",
        "cm": "https://dapi.binance.com/dapi/v1/klines",
    }
    WEBSOCKET_URLS: dict[str, str] = {
        "spot": "wss://stream.binance.com:9443/ws/{stream}",
        "um": "wss://fstream.binance.com/market/ws/{stream}",
        "cm": "wss://dstream.binance.com/ws/{stream}",
    }

    MAX_LIMIT_PER_REQUEST = 1000
    RECONNECT_DELAY_SECONDS = 5.0
    MAX_HISTORY_ATTEMPTS = 3
    HISTORY_RETRY_DELAY_SECONDS = 0.5
    MAX_STARTUP_SYNC_ROUNDS = 10

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
                    "first_missing_open_time_utc=%s "
                    "last_missing_open_time_utc=%s",
                    self.symbol,
                    self.interval,
                    attempt,
                    self.MAX_HISTORY_ATTEMPTS,
                    len(missing_open_times),
                    _format_utc_ms(missing_open_times[0]),
                    _format_utc_ms(missing_open_times[-1]),
                )
                time.sleep(self.HISTORY_RETRY_DELAY_SECONDS)

        if missing_open_times:
            readable_missing_open_times = [_format_utc_ms(open_time_ms) for open_time_ms in missing_open_times]
            raise RuntimeError(
                "Failed to acquire complete Binance historical market data "
                f"after {self.MAX_HISTORY_ATTEMPTS} attempts: "
                f"missing open times UTC {readable_missing_open_times}"
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

        with self._cache_lock:
            self.local_cache = frame
            self._interval_ms = int(interval_ms)

        self._synchronize_cache_to_latest_closed(
            required_bars,
            interval_ms,
        )

        with self._cache_lock:
            if self.local_cache is None or self.local_cache.empty:
                raise RuntimeError("Binance data cache became unavailable during startup synchronization")
            final_bar_count = len(self.local_cache)
            last_open_time_ms = int(self.local_cache.iloc[-1]["open_time_ms_utc"])

        self.logger.info(
            "Binance cache initialized | symbol=%s interval=%s bars=%d " "last_open_time_utc=%s",
            self.symbol,
            self.interval,
            final_bar_count,
            _format_utc_ms(last_open_time_ms),
        )

    @staticmethod
    def _latest_closed_open_time_ms(interval_ms: int) -> int:
        now_ms = int(time.time() * 1000)
        return now_ms // interval_ms * interval_ms - interval_ms

    def _missing_required_tail_open_times(
        self,
        target_open_time_ms: int,
        required_bars: int,
        interval_ms: int,
    ) -> list[int]:
        first_required_open_time_ms = target_open_time_ms - (required_bars - 1) * interval_ms
        expected_open_times = set(
            range(
                first_required_open_time_ms,
                target_open_time_ms + interval_ms,
                interval_ms,
            )
        )
        with self._cache_lock:
            if self.local_cache is None:
                raise RuntimeError("Binance data cache is not initialized")
            cached_open_times = self._get_open_time_set(self.local_cache)
        return sorted(expected_open_times - cached_open_times)

    def _synchronize_cache_to_latest_closed(
        self,
        required_bars: int,
        interval_ms: int,
    ) -> None:
        """Catch up candles closed while the initial history was downloading."""

        for sync_round in range(1, self.MAX_STARTUP_SYNC_ROUNDS + 1):
            target_open_time_ms = self._latest_closed_open_time_ms(interval_ms)
            missing_open_times = self._missing_required_tail_open_times(
                target_open_time_ms,
                required_bars,
                interval_ms,
            )

            if missing_open_times:
                self.logger.info(
                    "Synchronizing Binance cache after history load | "
                    "symbol=%s interval=%s round=%d/%d missing_bars=%d "
                    "first_missing_open_time_utc=%s last_missing_open_time_utc=%s",
                    self.symbol,
                    self.interval,
                    sync_round,
                    self.MAX_STARTUP_SYNC_ROUNDS,
                    len(missing_open_times),
                    _format_utc_ms(missing_open_times[0]),
                    _format_utc_ms(missing_open_times[-1]),
                )

            for attempt in range(1, self.MAX_HISTORY_ATTEMPTS + 1):
                if not missing_open_times:
                    break

                try:
                    with self._backfill_lock:
                        self._backfill_cache_once(
                            missing_open_times[0],
                            missing_open_times[-1],
                            interval_ms,
                        )
                except Exception:
                    self.logger.exception(
                        "Binance startup synchronization request failed | "
                        "symbol=%s interval=%s round=%d/%d attempt=%d/%d "
                        "first_missing_open_time_utc=%s last_missing_open_time_utc=%s",
                        self.symbol,
                        self.interval,
                        sync_round,
                        self.MAX_STARTUP_SYNC_ROUNDS,
                        attempt,
                        self.MAX_HISTORY_ATTEMPTS,
                        _format_utc_ms(missing_open_times[0]),
                        _format_utc_ms(missing_open_times[-1]),
                    )

                missing_open_times = self._missing_required_tail_open_times(
                    target_open_time_ms,
                    required_bars,
                    interval_ms,
                )
                if missing_open_times and attempt < self.MAX_HISTORY_ATTEMPTS:
                    time.sleep(self.HISTORY_RETRY_DELAY_SECONDS)

            if missing_open_times:
                readable_missing_open_times = [_format_utc_ms(open_time_ms) for open_time_ms in missing_open_times]
                raise RuntimeError(
                    "Failed to synchronize Binance cache to the latest closed candle: "
                    f"missing open times UTC {readable_missing_open_times}"
                )

            latest_target_open_time_ms = self._latest_closed_open_time_ms(interval_ms)
            if latest_target_open_time_ms <= target_open_time_ms:
                return

            self.logger.info(
                "Binance startup synchronization crossed another candle boundary | "
                "symbol=%s interval=%s synchronized_open_time_utc=%s "
                "new_target_open_time_utc=%s",
                self.symbol,
                self.interval,
                _format_utc_ms(target_open_time_ms),
                _format_utc_ms(latest_target_open_time_ms),
            )

        raise RuntimeError(
            "Failed to synchronize Binance cache because candle boundaries "
            f"kept advancing after {self.MAX_STARTUP_SYNC_ROUNDS} rounds"
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
                    "Received open Binance kline | symbol=%s interval=%s " "open_time_utc=%s close_time_utc=%s",
                    self.symbol,
                    self.interval,
                    _format_utc_ms(kline.get("t")),
                    _format_utc_ms(kline.get("T")),
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
                self.logger.debug(
                    "Received closed Binance kline | symbol=%s interval=%s " "open_time_utc=%s close_time_utc=%s",
                    self.symbol,
                    self.interval,
                    _format_utc_ms(kline.get("t")),
                    _format_utc_ms(kline.get("T")),
                )
                callback(candle_open_time_ms)
            except Exception:
                self.logger.exception(
                    "Closed-kline callback failed | symbol=%s interval=%s " "open_time_utc=%s",
                    self.symbol,
                    self.interval,
                    _format_utc_ms(candle_open_time_ms),
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
                    "Ignoring stale Binance closed kline | symbol=%s interval=%s " "open_time_utc=%s latest_cached_open_time_utc=%s",
                    self.symbol,
                    self.interval,
                    _format_utc_ms(candle_open_time_ms),
                    _format_utc_ms(latest_cached_open_time_ms),
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
                    "symbol=%s interval=%s latest_cached_open_time_utc=%s "
                    "current_open_time_utc=%s "
                    "first_missing_open_time_utc=%s "
                    "last_missing_open_time_utc=%s",
                    self.symbol,
                    self.interval,
                    _format_utc_ms(latest_cached_open_time_ms),
                    _format_utc_ms(candle_open_time_ms),
                    _format_utc_ms(missing_start_open_time_ms),
                    _format_utc_ms(missing_end_open_time_ms),
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
                        "Binance background backfill request failed | " "symbol=%s interval=%s attempt=%d/%d " "first_open_time_utc=%s last_open_time_utc=%s",
                        self.symbol,
                        self.interval,
                        attempt,
                        self.MAX_HISTORY_ATTEMPTS,
                        _format_utc_ms(fetch_start_open_time_ms),
                        _format_utc_ms(fetch_end_open_time_ms),
                    )

                with self._cache_lock:
                    if self.local_cache is None:
                        return
                    cached_open_times = self._get_open_time_set(self.local_cache)

                missing_open_times = sorted(expected_open_times - cached_open_times)

                if not missing_open_times:
                    self.logger.info(
                        "Binance background backfill completed | " "symbol=%s interval=%s bars=%d " "first_open_time_utc=%s last_open_time_utc=%s",
                        self.symbol,
                        self.interval,
                        len(expected_open_times),
                        _format_utc_ms(start_open_time_ms),
                        _format_utc_ms(end_open_time_ms),
                    )
                    return

                if attempt < self.MAX_HISTORY_ATTEMPTS:
                    self.logger.warning(
                        "Binance background backfill incomplete; retrying | "
                        "symbol=%s interval=%s attempt=%d/%d missing=%d "
                        "first_missing_open_time_utc=%s "
                        "last_missing_open_time_utc=%s",
                        self.symbol,
                        self.interval,
                        attempt,
                        self.MAX_HISTORY_ATTEMPTS,
                        len(missing_open_times),
                        _format_utc_ms(missing_open_times[0]),
                        _format_utc_ms(missing_open_times[-1]),
                    )

                    if self._stop_event.wait(self.HISTORY_RETRY_DELAY_SECONDS):
                        return

            self.logger.error(
                "Binance background backfill failed | symbol=%s interval=%s " "missing_open_times_utc=%s",
                self.symbol,
                self.interval,
                [_format_utc_ms(open_time_ms) for open_time_ms in missing_open_times],
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
