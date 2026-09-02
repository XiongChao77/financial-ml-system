"""Append-only per-feed CSV traces for live market data and predictions."""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Mapping

import numpy as np
import pandas as pd


MARKET_COLUMNS = (
    "open_time_ms_utc",
    "close_time_ms_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)
PREDICTION_COLUMNS = ("pred", "pred_prob", "net_score")


@dataclass(frozen=True)
class PredictionTraceConfig:
    """Destination for one CSV prediction trace per live feed group."""

    output_dir: str

    def __post_init__(self) -> None:
        if not str(self.output_dir).strip():
            raise ValueError("prediction_trace.output_dir must not be empty")


def strategy_prediction_column(strategy_id: str, prediction: str) -> str:
    """Return the stable wide-column name for one strategy prediction value."""

    if prediction not in PREDICTION_COLUMNS:
        raise ValueError(f"Unsupported prediction column: {prediction!r}")
    return f"{strategy_id}__{prediction}"


def _safe_filename_part(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return text.strip("-.") or "unknown"


def _csv_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            return ""
        return format(number, ".17g")
    return value


class FeedPredictionTraceWriter:
    """Write warm-up candles and completed live prediction rows for one feed."""

    def __init__(
        self,
        path: str,
        strategy_ids: list[str],
    ) -> None:
        if not strategy_ids:
            raise ValueError("Prediction trace requires at least one strategy")
        if len(set(strategy_ids)) != len(strategy_ids):
            raise ValueError("Prediction trace strategy IDs must be unique")

        self.path = os.path.abspath(path)
        self.strategy_ids = list(strategy_ids)
        self.fieldnames = [
            "is_warmup",
            *MARKET_COLUMNS,
            *[
                strategy_prediction_column(strategy_id, prediction)
                for strategy_id in self.strategy_ids
                for prediction in PREDICTION_COLUMNS
            ],
        ]
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._handle = open(
            self.path,
            "x",
            encoding="utf-8",
            newline="",
        )
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=self.fieldnames,
            extrasaction="ignore",
        )
        self._writer.writeheader()
        self._handle.flush()
        self._lock = Lock()
        self._closed = False

    @staticmethod
    def _market_values(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
        return {
            column: _csv_value(row.get(column))
            for column in MARKET_COLUMNS
        }

    def write_warmup(self, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            raise ValueError("Prediction trace warm-up frame must not be empty")
        records = [
            {
                "is_warmup": "true",
                **self._market_values(row),
            }
            for _, row in frame.iterrows()
        ]
        self._write_records(records)

    def write_live(
        self,
        row: Mapping[str, Any] | pd.Series,
        predictions: Mapping[str, Mapping[str, Any] | pd.Series],
    ) -> None:
        record = {
            "is_warmup": "false",
            **self._market_values(row),
        }
        for strategy_id, prediction_row in predictions.items():
            if strategy_id not in self.strategy_ids:
                raise ValueError(
                    f"Unknown prediction trace strategy ID: {strategy_id!r}"
                )
            for prediction in PREDICTION_COLUMNS:
                record[strategy_prediction_column(strategy_id, prediction)] = (
                    _csv_value(prediction_row.get(prediction))
                )
        self._write_records([record])

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Prediction trace writer is closed")
            self._writer.writerows(records)
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._handle.flush()
            self._handle.close()


class LivePredictionTraceRecorder:
    """Own the per-feed trace writers for one LiveRunner process session."""

    def __init__(
        self,
        config: PredictionTraceConfig,
        feed_groups,
        *,
        started_at: datetime | None = None,
    ) -> None:
        self.config = config
        timestamp = (started_at or datetime.now(UTC)).astimezone(UTC)
        session = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        self._writers: dict[int, FeedPredictionTraceWriter] = {}

        try:
            for group in feed_groups:
                market = group.market_config
                filename = "_".join(
                    [
                        _safe_filename_part(market.data_source),
                        _safe_filename_part(market.trading_type),
                        _safe_filename_part(market.symbol),
                        _safe_filename_part(market.interval),
                        session,
                    ]
                ) + ".csv"
                path = os.path.join(os.path.abspath(config.output_dir), filename)
                self._writers[id(group)] = FeedPredictionTraceWriter(
                    path,
                    [pipeline.spec.strategy_id for pipeline in group.pipelines],
                )
        except Exception:
            self.close()
            raise

    @property
    def paths(self) -> list[str]:
        return [writer.path for writer in self._writers.values()]

    def record_warmup(self, group, frame: pd.DataFrame) -> None:
        self._writers[id(group)].write_warmup(frame)

    def record_live(
        self,
        group,
        row: Mapping[str, Any] | pd.Series,
        predictions: Mapping[str, Mapping[str, Any] | pd.Series],
    ) -> None:
        self._writers[id(group)].write_live(row, predictions)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()
