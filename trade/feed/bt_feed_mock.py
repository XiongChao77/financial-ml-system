"""Deterministic file-backed candle feed for replaying the live pipeline."""

from __future__ import annotations

import os
import threading
from typing import Optional

import pandas as pd

from trade.feed.feed_base import ClosedCandleCallback, DataFeedBase


class BtDataFeedMock(DataFeedBase):
    """Expose a historical file one closed candle at a time.

    ``initialize_cache`` makes the warm-up prefix visible. Each ``advance`` call
    then exposes exactly one additional candle and invokes the same callback
    used by a live WebSocket feed.
    """

    REQUIRED_COLUMNS = {
        "open_time_ms_utc",
        "close_time_ms_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    def __init__(self, path: str, *, max_len: int = 5000):
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        self.path = os.path.abspath(path)
        self.max_cache_len = int(max_len)
        self._source = self._load_frame(self.path)
        self._cache: Optional[pd.DataFrame] = None
        self._cursor = -1
        self._interval_ms = 0
        self._callback: Optional[ClosedCandleCallback] = None
        self._started = False
        self._lock = threading.Lock()

    @staticmethod
    def _load_frame(path: str) -> pd.DataFrame:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Replay market-data file not found: {path}")
        extension = os.path.splitext(path)[1].casefold()
        if extension == ".csv":
            frame = pd.read_csv(path, encoding="utf-8")
        elif extension in {".feather", ".ft"}:
            frame = pd.read_feather(path)
        elif extension in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        elif extension in {".pkl", ".pickle"}:
            frame = pd.read_pickle(path)
        else:
            raise ValueError(
                "Replay data must be CSV, Feather, Parquet, or Pickle; "
                f"got {extension or '<no extension>'!r}"
            )
        if frame.empty:
            raise ValueError(f"Replay market-data file is empty: {path}")

        missing = sorted(BtDataFeedMock.REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"Replay market data is missing columns: {missing}")

        prepared = frame.copy()
        prepared["open_time_ms_utc"] = pd.to_numeric(
            prepared["open_time_ms_utc"],
            errors="raise",
        ).astype("int64")
        prepared["close_time_ms_utc"] = pd.to_numeric(
            prepared["close_time_ms_utc"],
            errors="raise",
        ).astype("int64")
        if "open_time_date_utc" not in prepared.columns:
            prepared["open_time_date_utc"] = pd.to_datetime(
                prepared["open_time_ms_utc"],
                unit="ms",
                utc=True,
            )

        prepared.sort_values("close_time_ms_utc", inplace=True)
        prepared.reset_index(drop=True, inplace=True)
        if prepared["close_time_ms_utc"].duplicated().any():
            raise ValueError("Replay market data contains duplicate candle times")
        return prepared

    def initialize_cache(self, required_bars: int, interval_ms: int) -> None:
        if required_bars <= 0:
            raise ValueError("required_bars must be positive")
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        if len(self._source) <= required_bars:
            raise ValueError(
                "Replay market data must contain warm-up bars plus at least one "
                f"replay bar: need>{required_bars}, actual={len(self._source)}"
            )

        close_times = self._source["close_time_ms_utc"].to_numpy(dtype="int64")
        gaps = close_times[1:] - close_times[:-1]
        invalid_gap_positions = (gaps != int(interval_ms)).nonzero()[0]
        if len(invalid_gap_positions):
            position = int(invalid_gap_positions[0])
            raise ValueError(
                "Replay market data is not continuous at rows "
                f"{position}/{position + 1}: expected_gap_ms={interval_ms}, "
                f"actual_gap_ms={int(gaps[position])}"
            )

        self._interval_ms = int(interval_ms)
        self._cursor = int(required_bars) - 1
        with self._lock:
            self._cache = self._visible_cache()

    def _visible_cache(self) -> pd.DataFrame:
        start = max(0, self._cursor - self.max_cache_len + 1)
        return self._source.iloc[start : self._cursor + 1].copy()

    def get_latest_data(self) -> Optional[pd.DataFrame]:
        with self._lock:
            return None if self._cache is None else self._cache.copy()

    def start(self, on_closed_candle: ClosedCandleCallback) -> None:
        if self._cache is None:
            raise RuntimeError("Replay cache must be initialized before start")
        self._callback = on_closed_candle
        self._started = True

    def peek_next_close_time_ms(self) -> Optional[int]:
        next_position = self._cursor + 1
        if next_position >= len(self._source):
            return None
        return int(self._source.iloc[next_position]["close_time_ms_utc"])

    def advance(self) -> Optional[int]:
        """Expose one new candle and synchronously enqueue its close event."""

        if not self._started or self._callback is None:
            raise RuntimeError("Replay feed must be started before advance")
        if self._cursor + 1 >= len(self._source):
            return None
        self._cursor += 1
        candle_close_time_ms = int(
            self._source.iloc[self._cursor]["close_time_ms_utc"]
        )
        with self._lock:
            self._cache = self._visible_cache()
        self._callback(candle_close_time_ms)
        return candle_close_time_ms

    def backfill_cache(
        self,
        start_close_time_ms: int,
        end_close_time_ms: int,
        interval_ms: int,
    ) -> int:
        """Historical replay already owns the complete source frame."""

        return 0

    def shutdown(self) -> None:
        self._started = False
        self._callback = None
