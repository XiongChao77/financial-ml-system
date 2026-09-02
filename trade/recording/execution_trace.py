"""Asynchronous CSV recording for normalized live execution reports."""

from __future__ import annotations

import csv
import math
import os
import queue
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from trade.core.execution import ExecutionReport


@dataclass(frozen=True)
class ExecutionTraceConfig:
    """Destination for per-session execution and fill CSV files."""

    output_dir: str

    def __post_init__(self) -> None:
        if not str(self.output_dir).strip():
            raise ValueError("execution_trace.output_dir must not be empty")


def _safe_filename_part(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return text.strip("-.") or "unknown"


def _number(value: Any) -> Any:
    if value is None:
        return ""
    number = float(value)
    return format(number, ".17g") if math.isfinite(number) else ""


def _time(value: datetime | None) -> str:
    return value.astimezone(UTC).isoformat() if value is not None else ""


class LiveExecutionTraceRecorder:
    """Write execution reports off the trading thread."""

    ORDER_FIELDS = (
        "execution_id",
        "strategy_id",
        "strategy_hash",
        "venue",
        "symbol",
        "side",
        "status",
        "reason",
        "decision_at_utc",
        "completed_at_utc",
        "requested_quantity",
        "filled_quantity",
        "decision_price",
        "arrival_price",
        "best_fill_price",
        "worst_fill_price",
        "fill_vwap",
        "latency_slippage_bps",
        "execution_slippage_vwap_bps",
        "execution_slippage_worst_bps",
        "bid",
        "ask",
        "spread_pct",
        "fill_count",
    )
    FILL_FIELDS = (
        "execution_id",
        "strategy_id",
        "venue",
        "symbol",
        "side",
        "fill_index",
        "order_id",
        "deal_id",
        "executed_at_utc",
        "price",
        "quantity",
        "is_aggregate",
    )

    def __init__(
        self,
        config: ExecutionTraceConfig,
        *,
        runner_id: str = "live-runner",
        logger=None,
        started_at: datetime | None = None,
    ) -> None:
        timestamp = (started_at or datetime.now(UTC)).astimezone(UTC)
        session = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        prefix = f"{_safe_filename_part(runner_id)}_{session}"
        output_dir = os.path.abspath(config.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        self.orders_path = os.path.join(output_dir, f"{prefix}_orders.csv")
        self.fills_path = os.path.join(output_dir, f"{prefix}_fills.csv")
        self._logger = logger
        self._queue: queue.Queue[Any] = queue.Queue()
        self._sentinel = object()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="live-execution-trace",
            daemon=True,
        )
        self._thread.start()

    @property
    def paths(self) -> tuple[str, str]:
        return self.orders_path, self.fills_path

    def record(
        self,
        report: ExecutionReport,
        *,
        strategy_id: str,
        strategy_hash: str,
        venue: str,
        symbol: str,
    ) -> None:
        if self._closed:
            return
        self._queue.put_nowait(
            (report, strategy_id, strategy_hash, venue, symbol)
        )

    def _run(self) -> None:
        try:
            with open(self.orders_path, "x", encoding="utf-8", newline="") as orders_handle, open(
                self.fills_path,
                "x",
                encoding="utf-8",
                newline="",
            ) as fills_handle:
                orders_writer = csv.DictWriter(orders_handle, fieldnames=self.ORDER_FIELDS)
                fills_writer = csv.DictWriter(fills_handle, fieldnames=self.FILL_FIELDS)
                orders_writer.writeheader()
                fills_writer.writeheader()
                orders_handle.flush()
                fills_handle.flush()
                while True:
                    item = self._queue.get()
                    if item is self._sentinel:
                        return
                    report, strategy_id, strategy_hash, venue, symbol = item
                    orders_writer.writerow(
                        self._order_row(
                            report,
                            strategy_id,
                            strategy_hash,
                            venue,
                            symbol,
                        )
                    )
                    for index, fill in enumerate(report.fills):
                        fills_writer.writerow(
                            {
                                "execution_id": report.execution_id,
                                "strategy_id": strategy_id,
                                "venue": venue,
                                "symbol": symbol,
                                "side": report.side,
                                "fill_index": index,
                                "order_id": fill.order_id,
                                "deal_id": fill.deal_id,
                                "executed_at_utc": _time(fill.executed_at_utc),
                                "price": _number(fill.price),
                                "quantity": _number(fill.quantity),
                                "is_aggregate": str(fill.is_aggregate).lower(),
                            }
                        )
                    orders_handle.flush()
                    fills_handle.flush()
        except Exception:
            if self._logger is not None:
                self._logger.exception("Live execution trace writer failed")

    @staticmethod
    def _order_row(report, strategy_id, strategy_hash, venue, symbol):
        return {
            "execution_id": report.execution_id,
            "strategy_id": strategy_id,
            "strategy_hash": strategy_hash,
            "venue": venue,
            "symbol": symbol,
            "side": report.side,
            "status": report.status,
            "reason": report.reason,
            "decision_at_utc": _time(report.decision_at_utc),
            "completed_at_utc": _time(report.completed_at_utc),
            "requested_quantity": _number(report.requested_quantity),
            "filled_quantity": _number(report.filled_quantity),
            "decision_price": _number(report.decision_price),
            "arrival_price": _number(report.arrival_price),
            "best_fill_price": _number(report.best_fill_price),
            "worst_fill_price": _number(report.worst_fill_price),
            "fill_vwap": _number(report.fill_vwap),
            "latency_slippage_bps": _number(report.latency_slippage_bps),
            "execution_slippage_vwap_bps": _number(
                report.execution_slippage_vwap_bps
            ),
            "execution_slippage_worst_bps": _number(
                report.execution_slippage_worst_bps
            ),
            "bid": _number(report.bid),
            "ask": _number(report.ask),
            "spread_pct": _number(report.spread_pct),
            "fill_count": len(report.fills),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._sentinel)
        self._thread.join(timeout=5.0)
