#!/usr/bin/env python3
"""Multi-strategy live runner with shared market-data feeds.

Pipeline:

    shared data feed -> feature generation -> model inference -> MarketView
    -> strategy intent -> configured live venue

Each strategy is restored from a shared canonical JSONL backtest report and
selected by its live-config hash.  Live-only settings choose the model artifact,
inference device, and execution venue; strategy and market parameters remain
owned by the report.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import ntpath
import os, sys
import queue
import time
from numbers import Real
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional
from enum import Enum, auto
import pandas as pd
import threading

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", ".."))
from data_process import common
from data_process.utils import config_from_dict_train
from model.model_loader import ModelHandler
from trade.feed.feed_base import DataFeedBase
from trade.core.protocol import (
    AccountView,
    Firm,
    MarketView,
    Observation,
    PositionDir,
    Signal,
    TradeIntent,
)
from trade.runner.config import BrokerConfig
from trade.monitoring.live_monitoring import (
    LiveMonitoringConfig,
    LiveMonitoringService,
    LiveStateRegistry,
    monitoring_config_from_mapping,
)
from trade.venue.live.binance_data_feed import BinanceDataFeed
from trade.core.venue_base import VenueBase
from trade.core.strategy_base import StrategyBase
from trade.venue.live.ctrader.ctrader_venue import CTraderOpenApiConnection
from trade.venue.live.ctrader.ctrader_venue import CTraderVenue

SUPPORTED_BINANCE_DATA_SOURCES = {
    "binance",
    "binance_api",
    "binance_public_data",
}


class RunnerEventType(Enum):
    CLOSED_CANDLE = auto()
    DATA_CHECK = auto()


@dataclass(frozen=True)
class RunnerEvent:
    e_type: RunnerEventType
    group: FeedGroup
    timestamp_ms: int


@dataclass(frozen=True, kw_only=True)
class LiveVenueConfigBase:
    """Live-only connection data. Secrets are deliberately omitted from repr."""

    venue: str
    path: str


@dataclass(frozen=True, kw_only=True)
class LiveVenueConfigHedge:
    firm: Firm
    hedge: bool = False
    # the rest only required when hedge is true
    cost: Optional[float] = None
    challenge_type: Optional[str] = None
    stage: Optional[str] = None
    hedge_venue: Optional[str] = None
    hedge_key_path: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "firm", Firm.parse(self.firm))


@dataclass(frozen=True, kw_only=True)
class LiveVenueConfigMt5(LiveVenueConfigBase, LiveVenueConfigHedge):
    """Live-only connection data. Secrets are deliberately omitted from repr."""

    login: str
    password: str
    server: str

    profit_target: float
    max_loss: float


@dataclass(frozen=True, kw_only=True)
class LiveVenueConfigCtrader(LiveVenueConfigBase, LiveVenueConfigHedge):
    """Live-only connection data. Secrets are deliberately omitted from repr."""

    trader_login: str

    profit_target: float
    max_loss: float


SUPPORTED_VENUES = {
    "mock": LiveVenueConfigBase,
    "mt5": LiveVenueConfigMt5,
    "ctrader": LiveVenueConfigCtrader,
    "bybit": LiveVenueConfigBase,
    "binance": LiveVenueConfigBase,
}


@dataclass
class LiveStrategySpec:
    strategy_id: str
    hash_id: str
    run_live: bool
    model_path: str
    device: str = "auto"
    compound: bool = True
    base_define: common.BaseDefine = None
    train_config: Any = None
    strategy_config: Any = None
    broker_config: BrokerConfig = None
    venue_config: LiveVenueConfigBase = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_live, bool):
            raise TypeError("Live strategy run_live must be a boolean")
        if not isinstance(self.compound, bool):
            raise TypeError("Live strategy compound must be a boolean")


@dataclass
class StrategyPipeline:
    spec: LiveStrategySpec
    model: ModelHandler
    venue: VenueBase
    strategy: StrategyBase
    feature_factory: Any
    interval_ms: int

    @property
    def required_bars(self) -> int:
        feature_history = int(self.feature_factory.get_global_min_history())
        model_history = max(1, int(self.model.seq_len)) * 2
        return feature_history + model_history


@dataclass
class FeedGroup:
    market_config: common.MarketDataSourceConfig
    interval_ms: int
    feed: DataFeedBase
    required_bars: int
    pipelines: list[StrategyPipeline]
    last_processed_candle_open_time_ms: Optional[int] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def name(self) -> str:
        return f"{self.market_config.symbol}_{self.market_config.interval}"


DATA_CHECK_TIMER_DELAY_MS = 2000


@dataclass(frozen=True)
class LiveRunnerConfiguration:
    strategies: list[LiveStrategySpec]
    monitoring: LiveMonitoringConfig | None = None


def _resolve_data_check_timer_interval_ms(feed_groups: Iterable[FeedGroup]) -> tuple[int, int]:
    """Return the shortest feed interval after validating aligned boundaries."""

    interval_entries = [
        (
            group.market_config.interval,
            common.get_interval_ms(group.market_config.interval),
        )
        for group in feed_groups
    ]
    if not interval_entries:
        raise ValueError("LiveRunner requires at least one feed group")
    invalid = [interval for interval, interval_ms in interval_entries if interval_ms <= 0]
    if invalid:
        raise ValueError(f"Invalid feed intervals: {', '.join(invalid)}")

    data_check_timer_interval_ms = min(interval_ms for _, interval_ms in interval_entries)
    data_check_timer_max_ms = max(interval_ms for _, interval_ms in interval_entries)
    incompatible = [f"{interval} ({interval_ms} ms)" for interval, interval_ms in interval_entries if interval_ms % data_check_timer_interval_ms != 0]
    if incompatible:
        raise ValueError(
            "Every feed interval must be an integer multiple of the shortest "
            f"interval ({data_check_timer_interval_ms} ms); incompatible: " + ", ".join(incompatible)
        )
    return data_check_timer_interval_ms, data_check_timer_max_ms


def _expected_closed_candle_open_time_ms(check_boundary_ms: int, interval_ms: int) -> int:
    """Return the latest candle open time that should be closed at a boundary."""

    return check_boundary_ms // interval_ms * interval_ms - interval_ms


def _format_utc_ms(timestamp_ms: int) -> str:
    """Format an epoch-millisecond timestamp for human-readable logs."""

    return datetime.fromtimestamp(
        int(timestamp_ms) / 1000,
        tz=timezone.utc,
    ).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]


def _resolve_path(config_path: str, value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"Path must not be empty in {config_path}")
    if os.path.isabs(raw) or ntpath.isabs(raw):
        return os.path.normpath(raw) if os.path.isabs(raw) else raw
    return os.path.normpath(os.path.join(os.path.dirname(config_path), raw))


def _iter_report_records(path: str) -> Iterable[dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Report file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            yield record


def load_params_from_report(
    strategy_entries: list[LiveStrategySpec],
    report_path: str,
) -> None:
    """Find matching JSONL records and fill strategy parameter configs."""

    from trade.runner.backtest_runner import strategy_config_from_dict

    specs_by_hash: dict[str, list[LiveStrategySpec]] = {}
    for spec in strategy_entries:
        specs_by_hash.setdefault(spec.hash_id, []).append(spec)
    params_by_hash: dict[str, dict[str, Any] | None] = dict.fromkeys(specs_by_hash)

    for record in _iter_report_records(report_path):
        params = record["params"]
        hash_id = params["hash"]
        if hash_id in params_by_hash and params_by_hash[hash_id] is None:
            loaded_params = dict(params)
            market_params = loaded_params["common"]
            train_params = loaded_params["train"]
            strategy_params = loaded_params["strategy"]
            strategy_params["max_daily_loss_pct"] = 0.035

            for spec in specs_by_hash[hash_id]:
                # market_params["interval"] = "1m"
                spec.base_define = common.BaseDefine(**market_params)
                spec.train_config = config_from_dict_train(train_params)
                spec.strategy_config = strategy_config_from_dict(strategy_params)
            params_by_hash[hash_id] = loaded_params
            if all(params is not None for params in params_by_hash.values()):
                break

    missing = sorted(hash_id for hash_id, params in params_by_hash.items() if params is None)
    if missing:
        missing_text = ", ".join(repr(hash_id) for hash_id in missing)
        raise KeyError(f"Hashes {missing_text} were not found in report file {report_path}")


def _venue_section(entry: Mapping[str, Any], venue_kind: str) -> dict[str, Any]:
    target = venue_kind.casefold()
    for key, value in entry.items():
        if str(key).casefold() == target and isinstance(value, Mapping):
            return dict(value)
    for key, value in entry.items():
        names = {part.strip().casefold() for part in str(key).split("/")}
        if target in names and isinstance(value, Mapping):
            return dict(value)
    raise KeyError(f"Missing {venue_kind} venue configuration object")


def _parse_venue_config(
    config_path: str,
    entry: Mapping[str, Any],
) -> LiveVenueConfigBase:
    venue_name = str(entry.get("venue", "")).strip().casefold()
    if venue_name not in SUPPORTED_VENUES:
        choices = ", ".join(sorted(SUPPORTED_VENUES))
        raise ValueError(f"venue must be one of {choices}; got {venue_name!r}")
    venue_class = SUPPORTED_VENUES[venue_name]

    section = _venue_section(entry, venue_name)
    path = section.get("path")
    if venue_name != "mt5":
        path = os.path.realpath(_resolve_path(config_path, path))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{venue_name} key directory not found: {path}")
    return venue_class(
        venue=venue_name,
        path=path,
        **{field.name: section.get(field.name) for field in fields(venue_class) if field.name not in {"venue", "path"}},
    )


def _strategy_entries(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    strategies = payload.get("strategy")
    if not isinstance(strategies, Mapping):
        raise TypeError("Live configuration strategy must be an object")
    return strategies


def load_live_runner_configuration(
    path: str,
    *,
    publish_url: str | None = None,
    runner_id: str | None = None,
) -> LiveRunnerConfiguration:
    """Restore strategies and optional read-only monitoring settings."""

    config_path = os.path.abspath(path)
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("Live configuration root must be an object")

    report_path = _resolve_path(config_path, payload.get("report"))
    if not report_path.lower().endswith(".jsonl"):
        raise ValueError(f"Live configuration report must be a JSONL file: {report_path}")

    raw_strategy_entries = _strategy_entries(payload)
    strategy_entries: list[LiveStrategySpec] = []
    for raw_id, raw_entry in raw_strategy_entries.items():
        strategy_id = str(raw_id).strip()
        entry = dict(raw_entry)
        run_live = entry["run_live"]
        compound = entry.get("compound", True)
        if not isinstance(run_live, bool):
            raise TypeError(f"Live strategy {strategy_id!r} run_live must be a JSON boolean")
        if not run_live:
            continue
        if not isinstance(compound, bool):
            raise TypeError(f"Live strategy {strategy_id!r} compound must be a JSON boolean")
        hash_id = entry["hash"]
        model_path = _resolve_path(config_path, entry["model_path"])
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model artifact directory not found: {model_path}")
        strategy_entries.append(
            LiveStrategySpec(
                strategy_id=strategy_id,
                hash_id=hash_id,
                run_live=run_live,
                model_path=model_path,
                device=str(entry.get("device", "auto")),
                compound=compound,
                broker_config=BrokerConfig(**entry["broker_config"]),
                venue_config=_parse_venue_config(config_path, entry),
            )
        )

    if not strategy_entries:
        raise ValueError("Live configuration contains no run_live strategies")

    load_params_from_report(strategy_entries, report_path)

    monitoring = monitoring_config_from_mapping(
        payload.get("monitoring"),
        publish_url=publish_url or os.environ.get("LIVE_MONITORING_PUBLISH_URL"),
        runner_id=runner_id or os.environ.get("LIVE_RUNNER_ID"),
    )
    return LiveRunnerConfiguration(
        strategies=strategy_entries,
        monitoring=monitoring,
    )


def load_live_strategy_specs(path: str) -> list[LiveStrategySpec]:
    """Restore live strategy specifications without starting monitoring."""

    return load_live_runner_configuration(path).strategies


def _optional_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _prepare_market_frame(
    frame: pd.DataFrame,
    base_define: common.BaseDefine,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("Live data feed returned no closed candles")
    prepared = frame.copy()
    prepared["open_time_ms_utc"] = pd.to_numeric(prepared["open_time_ms_utc"], errors="raise").astype("int64")
    prepared["close_time_ms_utc"] = pd.to_numeric(prepared["close_time_ms_utc"], errors="raise").astype("int64")
    prepared = common.attch_open_time_sn(base_define, prepared)
    prepared = common.calculate_thresholds(prepared, base_define)
    return prepared


def _market_view(
    predicted_frame: pd.DataFrame,
) -> MarketView:
    if predicted_frame is None or predicted_frame.empty:
        raise ValueError("Model returned no live predictions")

    row = predicted_frame.iloc[-1]
    raw_signal = _optional_float(row.get("pred"))
    signal = Signal.INVALID if raw_signal is None else Signal(int(raw_signal))

    close = _optional_float(row.get("close"))
    if close is None or close <= 0:
        raise ValueError("Latest closed candle has no valid close price")

    return MarketView(
        price=close,
        open=_optional_float(row.get("open")),
        high=_optional_float(row.get("high")),
        low=_optional_float(row.get("low")),
        close=close,
        signal=signal,
        pred_prob=_optional_float(row.get("pred_prob"), 0.0) or 0.0,
        expected_vol=_optional_float(row.get("expected_vol")),
        bars_to_close=_optional_float(
            row.get("bars_to_close"),
            math.inf,
        ),
    )


class LiveRunner:
    """Coordinate multiple model strategies while fetching each feed only once."""

    def __init__(
        self,
        specs: list[LiveStrategySpec],
        *,
        logger: Optional[logging.Logger] = None,
        feed_factory: Optional[Callable[[common.MarketDataSourceConfig, int], DataFeedBase]] = None,
        venue_factory: Optional[Callable[[LiveStrategySpec, logging.Logger], Any]] = None,
        prediction_callback: Optional[Callable[[StrategyPipeline, int, pd.Series], None]] = None,
        monitoring_config: LiveMonitoringConfig | None = None,
    ):
        if not specs:
            raise ValueError("LiveRunner requires at least one strategy")
        self.logger = logger or logging.getLogger("trade.live")
        self._feed_factory = feed_factory or self._create_feed
        self.ctrader_connection: CTraderOpenApiConnection | None = None
        self._ctrader_connection_path: str | None = None
        self._ctrader_environment: str | None = None
        self._venue_factory = venue_factory or self._create_venue
        self._prediction_callback = prediction_callback
        self._monitoring_config = monitoring_config
        self._live_registry: LiveStateRegistry | None = None
        self._monitoring_service: LiveMonitoringService | None = None
        self._initialized = False
        self._closed = False
        self.strategy_pipelines: list[StrategyPipeline] = []
        self.feed_groups: list[FeedGroup] = []
        self._events: queue.Queue[RunnerEvent] = queue.Queue()
        self.data_check_timer = None
        self._data_check_timer_interval_ms = 0
        self._next_min_expect_candle_open_time = 0
        self._next_data_check_timer_time_ms = 0
        self._data_check_timer_max_cycle_ms = 0
        self._data_check_timer_cycle_count = 0
        try:
            self._build(specs)
            self._live_registry = LiveStateRegistry(self.strategy_pipelines)
            if self._monitoring_config is not None:
                self._monitoring_service = LiveMonitoringService(
                    self._monitoring_config,
                    self._live_registry,
                    logger=self.logger,
                )
        except Exception:
            self.close()
            raise

    @classmethod
    def from_config(
        cls,
        path: str,
        logger: Optional[logging.Logger] = None,
        *,
        publish_url: str | None = None,
        runner_id: str | None = None,
    ) -> "LiveRunner":
        configuration = load_live_runner_configuration(
            path,
            publish_url=publish_url,
            runner_id=runner_id,
        )
        return cls(
            configuration.strategies,
            logger=logger,
            monitoring_config=configuration.monitoring,
        )

    @staticmethod
    def _create_feed(
        market_config: common.MarketDataSourceConfig,
        max_len: int,
    ):
        if market_config.data_source.casefold() not in SUPPORTED_BINANCE_DATA_SOURCES:
            raise ValueError(f"Unsupported live data source {market_config.data_source!r}; " "only Binance public kline feeds are currently implemented")

        return BinanceDataFeed(
            market_config.symbol,
            market_config.interval,
            market_config.trading_type,
            max_len=max_len,
        )

    @staticmethod
    def _create_feature_generator(spec: LiveStrategySpec):
        return common.FeatureFactory(
            common.get_interval_ms(spec.base_define.interval),
            feature_conf_list=spec.train_config.feature_conf_list,
        )

    @staticmethod
    def _load_model(spec: LiveStrategySpec):
        return ModelHandler(tarin_out_path=spec.model_path, device=spec.device)

    def _create_venue(self, spec: LiveStrategySpec, logger: logging.Logger):
        config = spec.venue_config
        if config.venue == "mock":
            from trade.venue.mock_venue import MockVenue

            return MockVenue(initial_equity=spec.broker_config.initial_equity)
        if config.venue == "mt5":
            if not spec.strategy_id.isascii() or not spec.strategy_id.isdigit():
                raise ValueError("MT5 strategy_id must contain ASCII digits only; " f"got {spec.strategy_id!r}")
            from trade.venue.live.mt5.mt5_venue import MT5Venue

            return MT5Venue(
                config.path,
                spec.base_define.symbol,
                int(spec.strategy_id),
                logger=logger,
                login=config.login,
                password=config.password,
                server=config.server,
                firm=config.firm,
            )
        if config.venue == "bybit":
            from trade.venue.live.bybit.bybit_venue import BybitVenue

            return BybitVenue(
                config.path,
                spec.base_define.symbol,
                f"{spec.strategy_id}:{spec.hash_id}",
                logger=logger,
            )
        if config.venue == "ctrader":
            connection_path = os.path.realpath(config.path)
            environment = "live"
            if self.ctrader_connection is not None:
                if connection_path != self._ctrader_connection_path:
                    raise ValueError(
                        "All cTrader venues sharing a connection must use the same "
                        "credential path; "
                        f"expected {self._ctrader_connection_path!r}, "
                        f"got {connection_path!r}"
                    )
                if environment != self._ctrader_environment:
                    raise ValueError(
                        "All cTrader venues sharing a connection must use the same "
                        "environment; "
                        f"expected {self._ctrader_environment!r}, "
                        f"got {environment!r}"
                    )
            ctrader_venue = CTraderVenue(
                connection_path,
                spec.base_define.symbol,
                f"{spec.strategy_id}:{spec.hash_id}",
                logger=logger,
                trader_login=config.trader_login,
                environment=environment,
                api=self.ctrader_connection,
                firm=config.firm,
            )
            if self.ctrader_connection is None:
                self.ctrader_connection = ctrader_venue.api
                self._ctrader_connection_path = connection_path
                self._ctrader_environment = environment
            return ctrader_venue

        if config.venue == "binance":
            from trade.venue.live.binance.binance_venue import BinanceVenue

            return BinanceVenue(
                config.path,
                spec.base_define.symbol,
                f"{spec.strategy_id}:{spec.hash_id}",
                logger=logger,
            )
        raise ValueError(f"Unsupported venue: {config.venue}")

    @staticmethod
    def _create_strategy(spec: LiveStrategySpec, venue: VenueBase):
        from trade.strategy.strategy_bbm import BbmSignalStrategy, BbmStrategyConfig

        equity = float(venue.get_account_equity())
        if equity <= 0:
            raise RuntimeError("Venue returned invalid account equity for " f"{spec.strategy_id} ({spec.hash_id})")
        leverage = float(spec.broker_config.leverage)
        data_interval_ms = common.get_interval_ms(spec.base_define.interval)

        held_bars = 0
        position = venue.get_current_state()
        if position.dir in {PositionDir.POSITIVE, PositionDir.NEGATIVE}:
            open_time = venue.get_last_position_open_time()
            if open_time is None:
                logger = logging.getLogger("trade")
                logger.warning(
                    "Open position has no opening timestamp | strategy=%s hash=%s",
                    spec.strategy_id,
                    spec.hash_id,
                )
            else:
                now = datetime.now(timezone.utc)
                if open_time.tzinfo is None:
                    open_time = open_time.replace(tzinfo=timezone.utc)
                else:
                    open_time = open_time.astimezone(timezone.utc)
                elapsed_ms = max(
                    0,
                    int((now - open_time).total_seconds() * 1000),
                )
                held_bars = elapsed_ms // data_interval_ms
        if isinstance(spec.strategy_config, BbmStrategyConfig):
            effective_config = replace(
                spec.strategy_config,
                compound=spec.compound,
            )
            spec.strategy_config = effective_config
            return BbmSignalStrategy(
                config=effective_config,
                init_equity=equity,
                data_interval_ms=data_interval_ms,
                exist_hold_bars=int(held_bars),
                leverage=leverage,
            )
        raise TypeError("Live runner currently supports BbmStrategyConfig, got " f"{type(spec.strategy_config).__name__}")

    def _build(self, specs: list[LiveStrategySpec]) -> None:
        grouped_pipelines: list[tuple[common.MarketDataSourceConfig, list[StrategyPipeline]]] = []

        for spec in specs:
            if not spec.run_live:
                self.logger.info(
                    "Non-live strategy skipped | id=%s hash=%s",
                    spec.strategy_id,
                    spec.hash_id,
                )
                continue

            model = self._load_model(spec)
            feature_generator = self._create_feature_generator(spec)
            market_config = common.MarketDataSourceConfig(
                **{field.name: getattr(spec.base_define, field.name) for field in fields(common.MarketDataSourceConfig)}
            )
            feed_entry = next(
                filter(
                    lambda item: item[0] == market_config,
                    grouped_pipelines,
                ),
                None,
            )
            if feed_entry is None:
                feed_pipelines: list[StrategyPipeline] = []
                grouped_pipelines.append((market_config, feed_pipelines))
            else:
                feed_pipelines = feed_entry[1]
            venue = self._venue_factory(spec, self.logger)
            try:
                strategy = self._create_strategy(spec, venue)
                self.logger.info(
                    "Strategy created | id=%s hash=%s venue=%s",
                    spec.strategy_id,
                    spec.hash_id,
                    type(venue).__name__,
                )
            except Exception:
                shutdown = getattr(venue, "shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception:
                        self.logger.exception(
                            "Venue cleanup failed during construction: %s",
                            spec.strategy_id,
                        )
                raise

            pipeline = StrategyPipeline(
                spec=spec,
                model=model,
                venue=venue,
                strategy=strategy,
                feature_factory=feature_generator,
                interval_ms=common.get_interval_ms(spec.base_define.interval),
            )
            self.strategy_pipelines.append(pipeline)
            feed_pipelines.append(pipeline)

        if not self.strategy_pipelines:
            raise ValueError("LiveRunner requires at least one run_live strategy")

        for market_config, pipelines in grouped_pipelines:
            required_bars = max(pipeline.required_bars for pipeline in pipelines)
            interval_ms = common.get_interval_ms(market_config.interval)

            feed = self._feed_factory(market_config, required_bars + 500)
            self.feed_groups.append(
                FeedGroup(
                    market_config=market_config,
                    interval_ms=interval_ms,
                    feed=feed,
                    required_bars=required_bars,
                    pipelines=pipelines,
                )
            )

        self.logger.info(
            "Live runner built | strategies=%d shared_feeds=%d",
            len(self.strategy_pipelines),
            len(self.feed_groups),
        )

    def _start_data_check_timer(self):
        if self._closed:
            return
        now_ms = int(time.time() * 1000)
        self._next_min_expect_candle_open_time = (now_ms // self._data_check_timer_interval_ms) * self._data_check_timer_interval_ms
        self._next_data_check_timer_time_ms = self._next_min_expect_candle_open_time + self._data_check_timer_interval_ms + DATA_CHECK_TIMER_DELAY_MS
        delay_seconds = (self._next_data_check_timer_time_ms - now_ms) / 1000.0
        self.logger.debug(
            "DATA_CHECK timer scheduled | now_utc=%s shortest_interval_ms=%d " "expected_shortest_open_time_utc=%s fire_time_utc=%s " "delay_seconds=%.3f",
            _format_utc_ms(now_ms),
            self._data_check_timer_interval_ms,
            _format_utc_ms(self._next_min_expect_candle_open_time),
            _format_utc_ms(self._next_data_check_timer_time_ms),
            delay_seconds,
        )
        self.data_check_timer = threading.Timer(delay_seconds, self._data_check_timer_handler)
        self.data_check_timer.daemon = True
        self.data_check_timer.start()

    def _data_check_timer_handler(self):
        if self._closed:
            return
        now_ms = int(time.time() * 1000)
        drift_ms = now_ms - self._next_data_check_timer_time_ms
        if abs(drift_ms) > 100:
            self.logger.warning(
                "DATA_CHECK timer drift too large | drift_ms=%d " "scheduled_utc=%s actual_utc=%s",
                drift_ms,
                _format_utc_ms(self._next_data_check_timer_time_ms),
                _format_utc_ms(now_ms),
            )
        check_boundary_ms = self._next_min_expect_candle_open_time + self._data_check_timer_interval_ms
        self.logger.debug(
            "DATA_CHECK timer fired | actual_utc=%s scheduled_utc=%s " "drift_ms=%d check_boundary_utc=%s feed_groups=%d",
            _format_utc_ms(now_ms),
            _format_utc_ms(self._next_data_check_timer_time_ms),
            drift_ms,
            _format_utc_ms(check_boundary_ms),
            len(self.feed_groups),
        )
        for group in self.feed_groups:
            with group.lock:
                last_processed_candle_open_time_ms = group.last_processed_candle_open_time_ms
            expected_open_time_ms = _expected_closed_candle_open_time_ms(
                check_boundary_ms,
                group.interval_ms,
            )
            if last_processed_candle_open_time_ms is None:
                self.logger.debug(
                    "DATA_CHECK group skipped before initialization | symbol=%s " "interval=%s expected_open_time_utc=%s",
                    group.market_config.symbol,
                    group.market_config.interval,
                    _format_utc_ms(expected_open_time_ms),
                )
                continue
            lag_ms = expected_open_time_ms - last_processed_candle_open_time_ms
            missing = lag_ms > 0
            self.logger.debug(
                "DATA_CHECK group evaluated | symbol=%s interval=%s interval_ms=%d "
                "check_boundary_utc=%s expected_open_time_utc=%s "
                "last_processed_open_time_utc=%s lag_ms=%d missing=%s",
                group.market_config.symbol,
                group.market_config.interval,
                group.interval_ms,
                _format_utc_ms(check_boundary_ms),
                _format_utc_ms(expected_open_time_ms),
                _format_utc_ms(last_processed_candle_open_time_ms),
                lag_ms,
                missing,
            )
            if missing:
                self.logger.warning(
                    "Candle missing at DATA_CHECK | symbol=%s interval=%s " "expected_open_time_utc=%s last_processed_open_time_utc=%s " "lag_ms=%d",
                    group.market_config.symbol,
                    group.market_config.interval,
                    _format_utc_ms(expected_open_time_ms),
                    _format_utc_ms(last_processed_candle_open_time_ms),
                    lag_ms,
                )
                self._events.put(
                    RunnerEvent(
                        e_type=RunnerEventType.DATA_CHECK,
                        group=group,
                        timestamp_ms=expected_open_time_ms,
                    )
                )
        self._start_data_check_timer()

    def initialize(self) -> None:
        if self._closed:
            raise RuntimeError("LiveRunner is already closed")
        if self._initialized:
            return

        self._data_check_timer_interval_ms, self._data_check_timer_max_cycle_ms = _resolve_data_check_timer_interval_ms(self.feed_groups)

        for group in self.feed_groups:
            group.interval_ms = common.get_interval_ms(group.market_config.interval)
            group.feed.initialize_cache(group.required_bars, group.interval_ms)
            initial = group.feed.get_latest_data()
            if initial is None or initial.empty:
                raise RuntimeError(f"Failed to warm live feed: {group.market_config}")
            group.last_processed_candle_open_time_ms = int(initial.iloc[-1]["open_time_ms_utc"])
            self.logger.info(
                "Live feed ready | source=%s symbol=%s interval=%s bars=%d",
                group.market_config.data_source,
                group.market_config.symbol,
                group.market_config.interval,
                len(initial),
            )

        for group in self.feed_groups:
            group.feed.start(
                lambda open_time_ms, target=group: self._events.put(RunnerEvent(e_type=RunnerEventType.CLOSED_CANDLE, group=target, timestamp_ms=open_time_ms))
            )
        self._start_data_check_timer()
        self._initialized = True
        if self._monitoring_service is not None:
            self._monitoring_service.start()
        self.logger.info(
            "WebSocket feeds started | data_check_timer_interval_ms=%d delay_ms=%d",
            self._data_check_timer_interval_ms,
            DATA_CHECK_TIMER_DELAY_MS,
        )

    def _predict(self, pipeline: StrategyPipeline, features: pd.DataFrame) -> pd.DataFrame:
        sequence_length = int(pipeline.model.seq_len)
        if len(features) < sequence_length:
            raise RuntimeError(f"Insufficient live features for model window: " f"need={sequence_length}, actual={len(features)}")
        latest_window = features.iloc[-sequence_length:].copy()
        predicted, _ = pipeline.model.predict(
            latest_window,
            kline_interval_ms=pipeline.interval_ms,
            is_live=True,
            batch_size=1,
            diff_thresh=None,
        )
        return predicted

    def _dispatch_invalid_to_group(self, group: FeedGroup, candle_open_time_ms: int):
        try:
            frame = group.feed.get_latest_data()
        except Exception:
            frame = None
            self.logger.exception(
                "Failed to read cached data for INVALID dispatch | " "symbol=%s interval=%s",
                group.market_config.symbol,
                group.market_config.interval,
            )
        market = self._invalid_market_view(frame)
        candle_open_time_utc = pd.Timestamp(
            candle_open_time_ms,
            unit="ms",
            tz="UTC",
        ).to_pydatetime()
        for pipeline in group.pipelines:
            try:
                observation = Observation(
                    market=market,
                    position=pipeline.venue.get_current_state(),
                    account=AccountView(equity=float(pipeline.venue.get_account_equity())),
                    candle_open_time_utc=candle_open_time_utc,
                    daily_reset_date=pipeline.venue.get_daily_reset_date(candle_open_time_utc),
                )
                intent = pipeline.strategy.process(observation)
                pipeline.venue.execute_action(intent)
                self._record_live_cycle(
                    pipeline,
                    pd.Series({"pred": Signal.INVALID.value}),
                    market,
                    intent,
                    candle_open_time_utc,
                )
                self.logger.warning(
                    "Invalid candle processed | id=%s hash=%s symbol=%s " "open_time_utc=%s action=%s",
                    pipeline.spec.strategy_id,
                    pipeline.spec.hash_id,
                    pipeline.spec.base_define.symbol,
                    _format_utc_ms(candle_open_time_ms),
                    intent.action.value,
                )
            except Exception:
                self.logger.exception(
                    "Invalid signal dispatch failed | id=%s hash=%s symbol=%s",
                    pipeline.spec.strategy_id,
                    pipeline.spec.hash_id,
                    pipeline.spec.base_define.symbol,
                )

    def _dispatch(
        self,
        pipeline: StrategyPipeline,
        market: MarketView,
        candle_open_time_utc: datetime,
    ) -> TradeIntent:
        observation = Observation(
            market=market,
            position=pipeline.venue.get_current_state(),
            account=AccountView(equity=float(pipeline.venue.get_account_equity())),
            candle_open_time_utc=candle_open_time_utc,
            daily_reset_date=pipeline.venue.get_daily_reset_date(candle_open_time_utc),
        )
        intent = pipeline.strategy.process(observation)
        pipeline.venue.execute_action(intent)
        self.logger.info(
            "Strategy processed | id=%s hash=%s symbol=%s signal=%s action=%s",
            pipeline.spec.strategy_id,
            pipeline.spec.hash_id,
            pipeline.spec.base_define.symbol,
            market.signal.name,
            intent.action.value,
        )
        return intent

    def _record_live_cycle(
        self,
        pipeline: StrategyPipeline,
        predicted_row: pd.Series,
        market: MarketView,
        intent: TradeIntent,
        updated_at: datetime,
    ) -> None:
        """Update UI-only memory without affecting the trading path."""

        registry = getattr(self, "_live_registry", None)
        if registry is None:
            return
        try:
            registry.record_cycle(
                pipeline,
                predicted_row,
                market,
                intent,
                updated_at,
            )
        except Exception:
            self.logger.exception(
                "Live UI cycle recording failed | strategy=%s",
                pipeline.spec.strategy_id,
            )

    @staticmethod
    def _invalid_market_view(frame: Optional[pd.DataFrame]) -> MarketView:
        row: Mapping[str, Any] = {}
        if frame is not None and not frame.empty:
            row = frame.iloc[-1]
        last_close = _optional_float(row.get("close"), 0.0) or 0.0
        return MarketView(
            price=last_close,
            open=_optional_float(row.get("open")),
            high=_optional_float(row.get("high")),
            low=_optional_float(row.get("low")),
            close=last_close,
            signal=Signal.INVALID,
            pred_prob=1.0,
            expected_vol=_optional_float(row.get("expected_vol")),
            bars_to_close=_optional_float(
                row.get("bars_to_close"),
                math.inf,
            ),
        )

    def _process_closed_candle(self, group: FeedGroup, last_processed_candle_open_time_ms, candle_open_time_ms: int) -> None:
        expected_open_time_ms = last_processed_candle_open_time_ms + group.interval_ms
        if candle_open_time_ms > expected_open_time_ms:
            missed_count = (candle_open_time_ms - expected_open_time_ms) // group.interval_ms
            self.logger.warning(
                "Closed-kline gap detected | symbol=%s interval=%s " "miss_count=%d last_processed_open_time_utc=%s " "new_candle_open_time_utc=%s",
                group.market_config.symbol,
                group.market_config.interval,
                missed_count,
                _format_utc_ms(last_processed_candle_open_time_ms),
                _format_utc_ms(candle_open_time_ms),
            )

        frame = group.feed.get_latest_data()
        if frame is None or frame.empty:
            self.logger.error(
                "Closed-kline event has no cached data | symbol=%s interval=%s " "candle_open_time_utc=%s",
                group.market_config.symbol,
                group.market_config.interval,
                _format_utc_ms(candle_open_time_ms),
            )
            self._dispatch_invalid_to_group(group, candle_open_time_ms)
        else:
            if frame.empty or int(frame.iloc[-1]["open_time_ms_utc"]) != candle_open_time_ms:
                self.logger.error(
                    "Closed-kline event is absent from cache | symbol=%s interval=%s " "candle_open_time_utc=%s",
                    group.market_config.symbol,
                    group.market_config.interval,
                    _format_utc_ms(candle_open_time_ms),
                )
                self._dispatch_invalid_to_group(group, candle_open_time_ms)
            else:
                for pipeline in group.pipelines:
                    try:
                        prepared = _prepare_market_frame(frame, pipeline.spec.base_define)
                        features = pipeline.feature_factory.generate(prepared)
                        predicted = self._predict(pipeline, features)
                        latest_prediction = predicted.iloc[-1]
                        if self._prediction_callback is not None:
                            self._prediction_callback(pipeline, candle_open_time_ms, latest_prediction.copy())
                        market = _market_view(predicted)
                        candle_open_time_utc = pd.Timestamp(
                            candle_open_time_ms,
                            unit="ms",
                            tz="UTC",
                        ).to_pydatetime()
                        intent = self._dispatch(
                            pipeline,
                            market,
                            candle_open_time_utc,
                        )
                        self._record_live_cycle(
                            pipeline,
                            latest_prediction,
                            market,
                            intent,
                            candle_open_time_utc,
                        )
                    except Exception:
                        self.logger.exception(
                            "Strategy cycle failed; dispatching INVALID | " "id=%s hash=%s symbol=%s",
                            pipeline.spec.strategy_id,
                            pipeline.spec.hash_id,
                            pipeline.spec.base_define.symbol,
                        )
                        try:
                            market = self._invalid_market_view(frame)
                            candle_open_time_utc = pd.Timestamp(
                                candle_open_time_ms,
                                unit="ms",
                                tz="UTC",
                            ).to_pydatetime()
                            intent = self._dispatch(
                                pipeline,
                                market,
                                candle_open_time_utc,
                            )
                            self._record_live_cycle(
                                pipeline,
                                pd.Series({"pred": Signal.INVALID.value}),
                                market,
                                intent,
                                candle_open_time_utc,
                            )
                        except Exception:
                            self.logger.exception(
                                "Fallback INVALID dispatch failed | id=%s hash=%s symbol=%s",
                                pipeline.spec.strategy_id,
                                pipeline.spec.hash_id,
                                pipeline.spec.base_define.symbol,
                            )

    def _process_event(self, event: RunnerEvent) -> bool:
        with event.group.lock:
            last_processed_candle_open_time_ms = event.group.last_processed_candle_open_time_ms
            if last_processed_candle_open_time_ms is not None and event.timestamp_ms <= last_processed_candle_open_time_ms:
                self.logger.warning(
                    "Stale runner event ignored | type=%s symbol=%s interval=%s " "last_processed_open_time_utc=%s received_open_time_utc=%s",
                    event.e_type.name,
                    event.group.market_config.symbol,
                    event.group.market_config.interval,
                    _format_utc_ms(last_processed_candle_open_time_ms),
                    _format_utc_ms(event.timestamp_ms),
                )
                return False
            event.group.last_processed_candle_open_time_ms = event.timestamp_ms

        if event.e_type == RunnerEventType.CLOSED_CANDLE:
            self.logger.debug("")
            self._process_closed_candle(
                event.group,
                last_processed_candle_open_time_ms,
                event.timestamp_ms,
            )
        elif event.e_type == RunnerEventType.DATA_CHECK:
            self._dispatch_invalid_to_group(
                event.group,
                event.timestamp_ms,
            )
        else:
            raise ValueError(f"Unsupported runner event type: {event.e_type!r}")
        return True

    def process_pending_events(self) -> int:
        """Process every event currently queued without blocking."""
        if self._closed:
            raise RuntimeError("LiveRunner is already closed")
        processed_count = 0
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                return processed_count
            if self._process_event(event):
                processed_count += 1

    def run_forever(self) -> None:
        try:
            self.initialize()
            self.logger.info("Live runner started")
            while True:
                if self._closed:
                    return
                try:
                    event = self._events.get(
                        timeout=(self._data_check_timer_max_cycle_ms // 1000 + 1),
                    )
                    self.logger.info(
                        "Runner event received | group=%s type=%s time_utc=%s",
                        event.group.name,
                        event.e_type.name,
                        _format_utc_ms(event.timestamp_ms),
                    )
                    self._process_event(event)
                except queue.Empty:
                    pass

        except KeyboardInterrupt:
            self.logger.info("Live runner stopped by user")
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.data_check_timer is not None:
            self.data_check_timer.cancel()
            self.data_check_timer = None
        if getattr(self, "_monitoring_service", None) is not None:
            try:
                self._monitoring_service.stop()
            except Exception:
                self.logger.exception("Live monitoring shutdown failed")
            self._monitoring_service = None
        for pipeline in self.strategy_pipelines:
            try:
                pipeline.strategy.finalize()
            except Exception:
                self.logger.exception(
                    "Strategy finalization failed: %s",
                    pipeline.spec.strategy_id,
                )

        closed = set()
        for pipeline in self.strategy_pipelines:
            venue = pipeline.venue
            if id(venue) in closed:
                continue
            closed.add(id(venue))
            shutdown = getattr(venue, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    self.logger.exception(
                        "Venue shutdown failed: %s",
                        pipeline.spec.strategy_id,
                    )

        for feed_group in self.feed_groups:
            shutdown = getattr(feed_group.feed, "shutdown", None)
            if not callable(shutdown):
                shutdown = getattr(feed_group.feed, "close", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    self.logger.exception(
                        "Live feed shutdown failed: %s",
                        feed_group.market_config,
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple live strategies")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "../../LiveTrading/live_config.json"),
        help="Path to the live strategy JSON configuration",
    )
    parser.add_argument(
        "--publish-url",
        default=None,
        help="Override monitoring.publish_url from the live configuration",
    )
    parser.add_argument(
        "--runner-id",
        default=None,
        help="Override monitoring.runner_id from the live configuration",
    )
    args = parser.parse_args()

    logger, _ = common.setup_session_logger(sub_folder="live_runner", symbol="", console_level=logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    runner = LiveRunner.from_config(
        args.config,
        logger=logger,
        publish_url=args.publish_url,
        runner_id=args.runner_id,
    )
    runner.run_forever()


if __name__ == "__main__":
    main()
