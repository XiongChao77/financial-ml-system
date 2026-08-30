"""Shared market-data feed interface used by the live runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import pandas as pd

ClosedCandleCallback = Callable[[int], None]  # candle_open_time_ms


class DataFeedBase(ABC):
    """Provide a rolling cache of closed candles to ``LiveRunner``."""

    @abstractmethod
    def initialize_cache(self, required_bars: int, interval_ms: int) -> None:
        """Populate the initial closed-candle cache."""

    @abstractmethod
    def get_latest_data(self) -> Optional[pd.DataFrame]:
        """Return a snapshot of the currently visible closed candles."""

    @abstractmethod
    def start(self, on_closed_candle: ClosedCandleCallback) -> None:
        """Start delivering timestamps for newly closed candles."""

    @abstractmethod
    def backfill_cache(
        self,
        start_close_time_ms: int,
        end_close_time_ms: int,
        interval_ms: int,
    ) -> int:
        """Fetch missing closed candles into cache without emitting events."""

    @abstractmethod
    def shutdown(self) -> None:
        """Stop delivery and release feed resources."""
