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
import os
import queue
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional
from enum import Enum, auto
import pandas as pd
import threading
from data_process import common
from data_process.utils import config_from_dict_train
from model.model_loader import ModelHandler
from trade.feed.feed_base import DataFeedBase
from trade.core.protocol import (
    AccountView,
    MarketView,
    Observation,
    PositionDir,
    PositionView,
    Signal,
    TradeIntent,
)
from trade.runner.config import BrokerConfig
from trade.venue.live.binance_data_feed import BinanceDataFeed
from trade.core.venue_base import VenueBase

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
    hedge: bool = False
    # the rest only required when hedge is true
    firm: Optional[str] = None
    cost: Optional[float] = None
    challenge_type: Optional[str] = None
    stage: Optional[str] = None
    hedge_venue: Optional[str] = None
    hedge_key_path: Optional[str] = None


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

    account_id: str

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
    enable: bool
    model_path: str
    device: str = "auto"
    compound: bool = True
    market_config: common.MarketDataSourceConfig = None
    train_config: Any = None
    strategy_config: Any = None
    broker_config: BrokerConfig = None
    venue_config: LiveVenueConfigBase = None

    def __post_init__(self) -> None:
        if not isinstance(self.enable, bool):
            raise TypeError("Live strategy enable must be a boolean")
        if not isinstance(self.compound, bool):
            raise TypeError("Live strategy compound must be a boolean")


@dataclass
class StrategyPipeline:
    spec: LiveStrategySpec
    model: ModelHandler
    venue: VenueBase
    strategy: Any
    feature_factory: Any
    interval_ms: int
    enable: bool

    @property
    def required_bars(self) -> int:
        feature_history = int(self.feature_factory.get_global_min_history())
        model_history = max(1, int(self.model.seq_len)) * 2
        return feature_history + model_history


@dataclass
class FeedGroup:
    market_config: common.MarketDataSourceConfig
    interval_ms: int
    enable: bool
    feed: DataFeedBase
    required_bars: int
    pipelines: list[StrategyPipeline]
    last_processed_candle_open_time_ms: Optional[int] = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


DATA_CHECK_TIMER_DELAY_MS = 2000


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
            for spec in specs_by_hash[hash_id]:
                spec.market_config = common.MarketDataSourceConfig(**{field.name: market_params[field.name] for field in fields(common.MarketDataSourceConfig)})
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


def load_live_strategy_specs(path: str) -> list[LiveStrategySpec]:
    """Restore every strategy from live-only config plus its canonical report."""

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
        enable = entry["enable"]
        compound = entry.get("compound", True)
        if not isinstance(enable, bool):
            raise TypeError(f"Live strategy {strategy_id!r} enable must be a JSON boolean")
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
                enable=enable,
                model_path=model_path,
                device=str(entry.get("device", "auto")),
                compound=compound,
                broker_config=BrokerConfig(**entry["broker_config"]),
                venue_config=_parse_venue_config(config_path, entry),
            )
        )

    if not strategy_entries:
        raise ValueError("Live configuration contains no strategies")

    load_params_from_report(strategy_entries, report_path)

    return strategy_entries


def _prepare_market_frame(frame: pd.DataFrame, spec: LiveStrategySpec) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("Live data feed returned no closed candles")
    prepared = frame.copy()
    prepared["open_time_ms_utc"] = pd.to_numeric(prepared["open_time_ms_utc"], errors="raise").astype("int64")
    prepared["close_time_ms_utc"] = pd.to_numeric(prepared["close_time_ms_utc"], errors="raise").astype("int64")
    prepared = common.attch_open_time_sn(spec.market_config, prepared)
    return prepared


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
    ):
        if not specs:
            raise ValueError("LiveRunner requires at least one strategy")
        self.logger = logger or logging.getLogger("trade.live")
        self._feed_factory = feed_factory or self._create_feed
        self._venue_factory = venue_factory or self._create_venue
        self._prediction_callback = prediction_callback
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
        except Exception:
            self.close()
            raise

    @classmethod
    def from_config(
        cls,
        path: str,
        logger: Optional[logging.Logger] = None,
    ) -> "LiveRunner":
        live_config = load_live_strategy_specs(path)
        return cls(live_config, logger=logger)

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
            common.get_interval_ms(spec.market_config.interval),
            feature_conf_list=spec.train_config.feature_conf_list,
        )

    @staticmethod
    def _load_model(spec: LiveStrategySpec):
        return ModelHandler(tarin_out_path=spec.model_path, device=spec.device)

    @staticmethod
    def _create_venue(spec: LiveStrategySpec, logger: logging.Logger):
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
                spec.market_config.symbol,
                int(spec.strategy_id),
                logger=logger,
                login=config.login,
                password=config.password,
                server=config.server,
            )
        if config.venue == "bybit":
            from trade.venue.live.bybit.bybit_venue import BybitVenue

            return BybitVenue(
                config.path,
                spec.market_config.symbol,
                f"{spec.strategy_id}:{spec.hash_id}",
                logger=logger,
            )
        if config.venue == "ctrader":
            from trade.venue.live.ctrader.ctrader_venue import CTraderVenue

            return CTraderVenue(
                config.path,
                spec.market_config.symbol,
                f"{spec.strategy_id}:{spec.hash_id}",
                logger=logger,
                account_id=config.account_id,
                environment="live",
            )
        if config.venue == "binance":
            from trade.venue.live.binance.binance_venue import BinanceVenue

            return BinanceVenue(
                config.path,
                spec.market_config.symbol,
                f"{spec.strategy_id}:{spec.hash_id}",
                logger=logger,
            )
        raise ValueError(f"Unsupported venue: {config.venue}")

    @staticmethod
    def _create_strategy(spec: LiveStrategySpec, venue: Any):
        from trade.strategy.strategy_bbm import BbmSignalStrategy, BbmStrategyConfig
        from trade.strategy.strategy_ml import MlSignalStrategy, MlStrategyConfig

        equity = float(venue.get_account_equity())
        if equity <= 0:
            raise RuntimeError("Venue returned invalid account equity for " f"{spec.strategy_id} ({spec.hash_id})")
        leverage = float(spec.broker_config.leverage)
        bar_interval_ms = common.get_interval_ms(spec.market_config.interval)

        open_time = venue.get_last_position_open_time()
        held_bars = 0
        if open_time is not None:
            now = venue.get_server_time()
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            if open_time.tzinfo is None:
                open_time = open_time.replace(tzinfo=timezone.utc)
            elapsed_ms = max(0, int((now - open_time).total_seconds() * 1000))
            held_bars = elapsed_ms // common.get_interval_ms(spec.market_config.interval)

        if isinstance(spec.strategy_config, MlStrategyConfig):
            return MlSignalStrategy(
                venue,
                config=spec.strategy_config,
                init_equity=equity,
                bar_interval_ms=bar_interval_ms,
                exist_hold_bars=int(held_bars),
                leverage=leverage,
            )
        if isinstance(spec.strategy_config, BbmStrategyConfig):
            return BbmSignalStrategy(
                venue,
                config=spec.strategy_config,
                init_equity=equity,
                bar_interval_ms=bar_interval_ms,
                exist_hold_bars=int(held_bars),
                leverage=leverage,
            )
        raise TypeError("Live runner currently supports MlStrategyConfig and " f"BbmStrategyConfig, got {type(spec.strategy_config).__name__}")

    def _build(self, specs: list[LiveStrategySpec]) -> None:
        grouped_pipelines: list[tuple[common.MarketDataSourceConfig, list[StrategyPipeline]]] = []

        for spec in specs:
            if not spec.enable:
                self.logger.info(
                    "Disabled strategy skipped | id=%s hash=%s",
                    spec.strategy_id,
                    spec.hash_id,
                )
                continue

            model = self._load_model(spec)
            feature_generator = self._create_feature_generator(spec)
            feed_entry = next(
                filter(
                    lambda item: item[0] == spec.market_config,
                    grouped_pipelines,
                ),
                None,
            )
            if feed_entry is None:
                feed_pipelines: list[StrategyPipeline] = []
                grouped_pipelines.append((spec.market_config, feed_pipelines))
            else:
                feed_pipelines = feed_entry[1]
            venue = self._venue_factory(spec, self.logger)
            try:
                strategy = self._create_strategy(spec, venue)
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
                interval_ms=common.get_interval_ms(spec.market_config.interval),
                enable=spec.enable,
            )
            self.strategy_pipelines.append(pipeline)
            feed_pipelines.append(pipeline)

        if not self.strategy_pipelines:
            raise ValueError("LiveRunner requires at least one enabled strategy")

        for market_config, pipelines in grouped_pipelines:
            required_bars = max(pipeline.required_bars for pipeline in pipelines)
            interval_ms = common.get_interval_ms(market_config.interval)
            group_enable = any(pipeline.enable for pipeline in pipelines)

            feed = self._feed_factory(market_config, required_bars + 500)
            self.feed_groups.append(
                FeedGroup(
                    market_config=market_config,
                    interval_ms=interval_ms,
                    enable=group_enable,
                    feed=feed,
                    required_bars=required_bars,
                    pipelines=pipelines,
                )
            )

        self.logger.info(
            "Live runner built | strategies=%d enable=%d shared_feeds=%d",
            len(self.strategy_pipelines),
            sum(pipeline.enable for pipeline in self.strategy_pipelines),
            len(self.feed_groups),
        )

    def _start_data_check_timer(self):
        if self._closed:
            return
        now_ms = int(time.time() * 1000)
        self._next_min_expect_candle_open_time = (now_ms // self._data_check_timer_interval_ms) * self._data_check_timer_interval_ms
        self._next_data_check_timer_time_ms = self._next_min_expect_candle_open_time + self._data_check_timer_interval_ms + DATA_CHECK_TIMER_DELAY_MS
        # self.logger.info(f"_next_data_check_timer_time_ms  {self._next_data_check_timer_time_ms }")
        delay_seconds = (self._next_data_check_timer_time_ms - now_ms) / 1000.0
        self.data_check_timer = threading.Timer(delay_seconds, self._data_check_timer_handler)
        self.data_check_timer.daemon = True
        self.data_check_timer.start()

    def _data_check_timer_handler(self):
        if self._closed:
            return
        now_ms = int(time.time() * 1000)
        fluctuate_ms = abs(now_ms - self._next_data_check_timer_time_ms)
        if fluctuate_ms > 100:
            self.logger.warning(f"data_check_timer fluctuate too much {fluctuate_ms}ms")
        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        # self.logger.info("_data_check_timer_handler %s", timestamp)
        for group in self.feed_groups:
            with group.lock:
                last_processed_candle_open_time_ms = group.last_processed_candle_open_time_ms
            passed_time = self._next_min_expect_candle_open_time - last_processed_candle_open_time_ms
            last_process_time = datetime.fromtimestamp(last_processed_candle_open_time_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            self.logger.debug(
                f"symbol {group.market_config.symbol} interval {group.market_config.interval} passed_time {passed_time}, last process time {last_process_time}"
            )
            if passed_time > 0 and passed_time >= group.interval_ms:
                check_time = datetime.fromtimestamp(self._next_min_expect_candle_open_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                last_process_time = datetime.fromtimestamp(last_processed_candle_open_time_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

                self.logger.warning(
                    f"symbol {group.market_config.symbol} interval {group.market_config.interval} candle miss at time {check_time}, last process time {last_process_time}"
                )
                self._events.put(RunnerEvent(e_type=RunnerEventType.DATA_CHECK, group=group, timestamp_ms=self._next_min_expect_candle_open_time))
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
        market = self._construct_market_view(None, Signal.INVALID, 1.0)
        for pipeline in group.pipelines:
            if pipeline.enable == True:
                observation = Observation(
                    market=market,
                    position=pipeline.venue.get_current_state(),
                    account=AccountView(equity=float(pipeline.venue.get_account_equity())),
                    current_time=pipeline.venue.get_server_time(),
                )
                intent = pipeline.strategy.process(observation)
                self.logger.info(
                    "Strategy processed | id=%s hash=%s symbol=%s signal=%s action=%s",
                    pipeline.spec.strategy_id,
                    pipeline.spec.hash_id,
                    pipeline.spec.market_config.symbol,
                    market.signal.name,
                    intent.action.value,
                )

    def _dispatch(
        self,
        pipeline: StrategyPipeline,
        market: MarketView,
    ) -> TradeIntent:
        observation = Observation(
            market=market,
            position=pipeline.venue.get_current_state(),
            account=AccountView(equity=float(pipeline.venue.get_account_equity())),
            current_time=pipeline.venue.get_server_time(),
        )
        intent = pipeline.strategy.process(observation)
        self.logger.info(
            "Strategy processed | id=%s hash=%s symbol=%s signal=%s action=%s",
            pipeline.spec.strategy_id,
            pipeline.spec.hash_id,
            pipeline.spec.market_config.symbol,
            market.signal.name,
            intent.action.value,
        )
        return intent

    def _construct_market_view(self, frame: Optional[pd.DataFrame], signal, pred_prob) -> MarketView:
        row: Mapping[str, Any] = {}
        if frame is not None and not frame.empty:
            row = frame.iloc[-1]
        last_close = row.get("close", 0)
        return MarketView(
            price=last_close,
            open=row.get("open", 0),
            high=row.get("high", 0),
            low=row.get("low", 0),
            close=last_close,
            signal=signal,
            pred_prob=pred_prob,
            atr_pct=0.0,
            bars_to_close=math.inf,  # cryptocurrency
        )

    def _process_closed_candle(self, group: FeedGroup, last_processed_candle_open_time_ms, candle_open_time_ms: int) -> None:
        expected_open_time_ms = last_processed_candle_open_time_ms + group.interval_ms
        if candle_open_time_ms > expected_open_time_ms:
            missed_count = (candle_open_time_ms - expected_open_time_ms) // group.interval_ms
            last_process_time = datetime.fromtimestamp(last_processed_candle_open_time_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            new_candle_open_time = datetime.fromtimestamp(candle_open_time_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            self.logger.warning(
                f"symbol {group.market_config.symbol} interval {group.market_config.interval} miss count {missed_count} "
                f"last_processed_candle_open_time: {last_process_time} new_candle_open_time: {new_candle_open_time}"
            )

        frame = group.feed.get_latest_data()
        if frame is None or frame.empty:
            self.logger.error(
                "Closed-kline event has no cached data | symbol=%s interval=%s " "candle_open_time_ms=%d",
                group.market_config.symbol,
                group.market_config.interval,
                candle_open_time_ms,
            )
            self._dispatch_invalid_to_group(group, candle_open_time_ms)
        else:
            if frame.empty or int(frame.iloc[-1]["open_time_ms_utc"]) != candle_open_time_ms:
                self.logger.error(
                    "Closed-kline event is absent from cache | symbol=%s interval=%s " "candle_open_time_ms=%d",
                    group.market_config.symbol,
                    group.market_config.interval,
                    candle_open_time_ms,
                )
                self._dispatch_invalid_to_group(group, candle_open_time_ms)
            else:
                for pipeline in group.pipelines:
                    if not pipeline.enable:
                        continue
                    try:
                        prepared = _prepare_market_frame(frame, pipeline.spec)
                        features = pipeline.feature_factory.generate(prepared)
                        predicted = self._predict(pipeline, features)
                        latest_prediction = predicted.iloc[-1]
                        if self._prediction_callback is not None:
                            self._prediction_callback(
                                pipeline,
                                candle_open_time_ms,
                                latest_prediction.copy(),
                            )
                        maket_view = self._construct_market_view(frame, latest_prediction["pred"], latest_prediction["pred_prob"])
                        self._dispatch(pipeline, maket_view)
                    except Exception:
                        self.logger.exception(
                            "Strategy cycle failed; dispatching INVALID | " "id=%s hash=%s symbol=%s",
                            pipeline.spec.strategy_id,
                            pipeline.spec.hash_id,
                            pipeline.spec.market_config.symbol,
                        )
                        try:
                            maket_view = self._construct_market_view(frame, Signal.INVALID, 1.0)
                            self._dispatch(pipeline, maket_view)
                        except Exception:
                            self.logger.exception(
                                "Fallback INVALID dispatch failed | id=%s hash=%s symbol=%s",
                                pipeline.spec.strategy_id,
                                pipeline.spec.hash_id,
                                pipeline.spec.market_config.symbol,
                            )

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
                    with event.group.lock:
                        last_processed_candle_open_time_ms = event.group.last_processed_candle_open_time_ms
                        if last_processed_candle_open_time_ms is not None and event.timestamp_ms <= last_processed_candle_open_time_ms:  # expired event
                            self.logger.warning(
                                "Stale closed-kline event ignored | symbol=%s interval=%s " "last_processed_candle_open_time_ms=%d received_open_time_ms=%d",
                                event.group.market_config.symbol,
                                event.group.market_config.interval,
                                last_processed_candle_open_time_ms,
                                event.timestamp_ms,
                            )
                            continue
                        event.group.last_processed_candle_open_time_ms = event.timestamp_ms
                    if event.e_type == RunnerEventType.CLOSED_CANDLE:
                        self._process_closed_candle(event.group, last_processed_candle_open_time_ms, event.timestamp_ms)
                    elif event.e_type == RunnerEventType.DATA_CHECK:
                        if event.timestamp_ms > last_processed_candle_open_time_ms:
                            self._dispatch_invalid_to_group(event.group, self._next_min_expect_candle_open_time)
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
    args = parser.parse_args()

    logger, _ = common.setup_session_logger(
        sub_folder="live_runner",
        symbol="",
    )
    runner = LiveRunner.from_config(
        args.config,
        logger=logger,
    )
    runner.run_forever()


if __name__ == "__main__":
    main()
