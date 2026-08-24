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
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

from data_process import common
from data_process.utils import config_from_dict_train
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

SUPPORTED_BINANCE_DATA_SOURCES = {
    "binance",
    "binance_api",
    "binance_public_data",
}

@dataclass(frozen=True)
class LiveVenueConfigBase:
    """Live-only connection data. Secrets are deliberately omitted from repr."""
    venue :str
    path: str

class LiveVenueConfigHedge:
    hedge: str = False
    # the rest only required when hedge is true
    firm: Optional[str] = None
    cost: Optional[float] = None
    challenge_type: Optional[str] = None
    stage: Optional[str] = None
    hedge_venue: Optional[str] = None
    hedge_key_path: Optional[str] = None

@dataclass(frozen=True)
class LiveVenueConfigMt5(LiveVenueConfigBase,LiveVenueConfigHedge):
    """Live-only connection data. Secrets are deliberately omitted from repr."""
    login: str
    password: str
    server: str

    profit_target: float
    max_loss: float

@dataclass(frozen=True)
class LiveVenueConfigCtrader(LiveVenueConfigBase,LiveVenueConfigHedge):
    """Live-only connection data. Secrets are deliberately omitted from repr."""
    account_id: str

    profit_target: float
    max_loss: float

SUPPORTED_VENUES = {
    "mt5": LiveVenueConfigMt5,
    "ctrader": LiveVenueConfigCtrader,
    "bybit": LiveVenueConfigBase,
    "binance": LiveVenueConfigBase,
}

@dataclass
class LiveStrategySpec:
    strategy_id: str
    hash_id: str
    enable: str
    model_path: str
    device: str = 'auto'
    compound:str = True
    market_config: common.MarketDataSourceConfig = None
    train_config: Any  = None
    strategy_config: Any  = None
    broker_config: BrokerConfig  = None
    venue_config: LiveVenueConfigBase  = None


@dataclass
class StrategyPipeline:
    spec: LiveStrategySpec
    model: Any
    venue: Any
    strategy: Any
    feature_factory: Any
    interval_ms: int
    enable: str

    @property
    def required_bars(self) -> int:
        feature_history = int(self.feature_factory.get_global_min_history())
        model_history = max(1, int(self.model.seq_len)) * 2
        fixed_hold_bars = getattr(self.spec.strategy_config, "fixed_hold_bars", None)
        hold_history = max(1, int(fixed_hold_bars or 1)) * 2
        return feature_history + max(model_history, hold_history)


@dataclass
class FeedGroup:
    market_config: common.MarketDataSourceConfig
    feed: BinanceDataFeed
    required_bars: int
    pipelines: list[StrategyPipeline]
    last_candle_id: Optional[int] = None


WATCHDOG_DELAY_MS = 500


def _latest_expected_candle_id(boundary_ms: int, interval_ms: int) -> int:
    """Return the open time of the latest candle closed by a time boundary."""

    return boundary_ms // interval_ms * interval_ms - interval_ms


def _resolve_path(config_path: str, value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"Path must not be empty in {config_path}")
    if os.path.isabs(raw) or ntpath.isabs(raw):
        return os.path.normpath(raw) if os.path.isabs(raw) else raw
    return os.path.normpath(os.path.join(os.path.dirname(config_path), raw))


def _required_mapping(parent: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise KeyError(f"Missing object {context}.{key}")
    return dict(value)


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
                spec.market_config = common.MarketDataSourceConfig(
                    **{
                        field.name: market_params[field.name]
                        for field in fields(common.MarketDataSourceConfig)
                    }
                )
                spec.train_config = config_from_dict_train(train_params)
                spec.strategy_config = strategy_config_from_dict(strategy_params)
            params_by_hash[hash_id] = loaded_params
            if all(params is not None for params in params_by_hash.values()):
                break

    missing = sorted(
        hash_id
        for hash_id, params in params_by_hash.items()
        if params is None
    )
    if missing:
        missing_text = ", ".join(repr(hash_id) for hash_id in missing)
        raise KeyError(
            f"Hashes {missing_text} were not found in report file {report_path}"
        )


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
        choices = ", ".join(SUPPORTED_VENUES.values())
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
        **{
            field.name: section.get(field.name)
            for field in fields(venue_class)
            if field.name not in {"venue", "path"}
        },
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
        hash_id = entry["hash"]
        model_path = _resolve_path(config_path, entry['model_path'])
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Model artifact directory not found: {model_path}"
            )
        strategy_entries.append(
            LiveStrategySpec(
                strategy_id=strategy_id,
                hash_id=hash_id,
                enable=enable,
                model_path=model_path,
                device=str(entry.get("device", "auto")),
                compound=str(entry["compound"]),
                broker_config=BrokerConfig(**entry["broker_config"]),
                venue_config=_parse_venue_config(config_path, entry),
            )
        )

    if not strategy_entries:
        raise ValueError("Live configuration contains no strategies")

    load_params_from_report(strategy_entries, report_path)

    return strategy_entries


def _optional_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _bar_time(row: Mapping[str, Any]) -> datetime:
    raw = row.get("open_time_date_utc", row.get("open_time_ms_utc"))
    if raw is None:
        raise ValueError("Live market data has no candle timestamp")
    if isinstance(raw, (int, float)):
        timestamp = pd.Timestamp(raw, unit="ms", tz="UTC")
    else:
        timestamp = pd.Timestamp(raw)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _position_view(raw_state: Any) -> PositionView:
    if isinstance(raw_state, PositionView):
        return raw_state
    if not isinstance(raw_state, (tuple, list)) or len(raw_state) < 3:
        raise TypeError("Venue get_current_state() must return PositionView or a 3-item tuple")
    direction, layers, price = raw_state[:3]
    return PositionView(
        dir=PositionDir(direction),
        layers=int(layers),
        price=float(price or 0.0),
    )


def _prepare_market_frame(frame: pd.DataFrame, spec: LiveStrategySpec) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("Live data feed returned no closed candles")
    prepared = frame.copy()
    interval_ms = common.get_interval_ms(spec.market_config.interval)
    prepared["open_time_ms_utc"] = pd.to_numeric(
        prepared["open_time_ms_utc"], errors="raise"
    ).astype("int64")
    if spec.market_config.market_category.casefold() == "forex":
        prepared = common.add_bars_to_gap(prepared, interval_ms)
        prepared["open_time_sn"] = range(len(prepared))
    else:
        prepared["bars_to_close"] = math.inf
        prepared["open_time_sn"] = prepared["open_time_ms_utc"] // interval_ms
    return prepared


def _market_view(
    predicted_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    pipeline: StrategyPipeline,
) -> tuple[MarketView, datetime]:
    row = predicted_frame.iloc[-1]
    raw_signal = _optional_float(row.get("pred"))
    signal = Signal.INVALID if raw_signal is None else Signal(int(raw_signal))
    fixed_hold_bars = getattr(pipeline.spec.strategy_config, "fixed_hold_bars", None)
    if fixed_hold_bars is None or int(fixed_hold_bars) <= 0:
        atr_pct = 0.0
    else:
        atr_series = common.stop_loss_atr_pct(feature_frame, int(fixed_hold_bars))
        atr_pct = _optional_float(atr_series.iloc[-1], 0.0) or 0.0
    current_time = _bar_time(row)
    close = _optional_float(row.get("close"))
    if close is None or close <= 0:
        raise ValueError("Latest closed candle has no valid close price")

    market = MarketView(
        price=close,
        open=_optional_float(row.get("open")),
        high=_optional_float(row.get("high")),
        low=_optional_float(row.get("low")),
        close=close,
        signal=signal,
        pred_prob=_optional_float(row.get("pred_prob"), 0.0) or 0.0,
        atr_pct=atr_pct,
        expected_vol=_optional_float(row.get("expected_vol")),
        slow_atr=_optional_float(row.get("slow_atr")),
        vol_regime=_optional_float(row.get("vol_regime")),
        bar_interval_ms=pipeline.interval_ms,
        bars_to_close=_optional_float(row.get("bars_to_close"), math.inf),
    )
    return market, current_time


class LiveRunner:
    """Coordinate multiple model strategies while fetching each feed only once."""

    def __init__(
        self,
        specs: list[LiveStrategySpec],
        *,
        logger: Optional[logging.Logger] = None,
    ):
        if not specs:
            raise ValueError("LiveRunner requires at least one strategy")
        self.logger = logger or logging.getLogger("trade.live")
        self._initialized = False
        self._closed = False
        self.strategy_pipelines: list[StrategyPipeline] = []
        self.feed_groups: list[FeedGroup] = []
        self._candle_events: queue.Queue[tuple[FeedGroup, int]] = queue.Queue()
        self._watchdog_interval_ms = 0
        self._next_watchdog_time_ms = 0
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
            raise ValueError(
                f"Unsupported live data source {market_config.data_source!r}; "
                "only Binance public kline feeds are currently implemented"
            )

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
        from model.model_loader import ModelHandler

        return ModelHandler(tarin_out_path=spec.model_path, device=spec.device)

    @staticmethod
    def _create_venue(spec: LiveStrategySpec, logger: logging.Logger):
        config = spec.venue_config
        if config.venue == "mt5":
            if not spec.strategy_id.isascii() or not spec.strategy_id.isdigit():
                raise ValueError(
                    "MT5 strategy_id must contain ASCII digits only; "
                    f"got {spec.strategy_id!r}"
                )
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
                environment='live',
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
            raise RuntimeError(
                "Venue returned invalid account equity for "
                f"{spec.strategy_id} ({spec.hash_id})"
            )
        leverage = float(spec.broker_config.leverage)

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
                exist_hold_bars=int(held_bars),
                leverage=leverage,
            )
        if isinstance(spec.strategy_config, BbmStrategyConfig):
            return BbmSignalStrategy(
                venue,
                config=spec.strategy_config,
                init_equity=equity,
                exist_hold_bars=int(held_bars),
                leverage=leverage,
            )
        raise TypeError(
            "Live runner currently supports MlStrategyConfig and "
            f"BbmStrategyConfig, got {type(spec.strategy_config).__name__}"
        )

    def _build(self, specs: list[LiveStrategySpec]) -> None:
        grouped_pipelines: list[
            tuple[common.MarketDataSourceConfig, list[StrategyPipeline]]
        ] = []

        for spec in specs:
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
            venue = self._create_venue(spec, self.logger)
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

        for market_config, pipelines in grouped_pipelines:
            required_bars = max(pipeline.required_bars for pipeline in pipelines)
            feed = self._create_feed(market_config, required_bars + 500)
            self.feed_groups.append(
                FeedGroup(
                    market_config=market_config,
                    feed=feed,
                    required_bars=required_bars,
                    pipelines=pipelines,
                )
            )

        self.logger.info(
            "Live runner built | strategies=%d enable=%d shared_feeds=%d",
            len(self.strategy_pipelines),
            sum(pipeline.enable == "true" for pipeline in self.strategy_pipelines),
            len(self.feed_groups),
        )

    def set_strategy_enabled(self, strategy_id: str, enable: str) -> None:
        """Enable or disable one strategy without rebuilding its pipeline."""

        enable = enable.lower()
        if enable not in ["false", "true"]:
            raise ValueError("enable must be a boolean string")
        for pipeline in self.strategy_pipelines:
            if pipeline.spec.strategy_id == strategy_id:
                pipeline.enable = enable
                self.logger.info(
                    "Strategy runtime state changed | id=%s enable=%s",
                    strategy_id,
                    enable,
                )
                return
        raise KeyError(f"Unknown live strategy id: {strategy_id!r}")

    def initialize(self) -> None:
        if self._closed:
            raise RuntimeError("LiveRunner is already closed")
        if self._initialized:
            return

        for group in self.feed_groups:
            interval_ms = common.get_interval_ms(group.market_config.interval)
            group.feed.initialize_cache(group.required_bars, interval_ms)
            initial = group.feed.get_latest_data()
            if initial is None or initial.empty:
                raise RuntimeError(
                    f"Failed to warm live feed: {group.market_config}"
                )
            latest_candle_id = int(initial.iloc[-1]["open_time_ms_utc"])
            group.last_candle_id = latest_candle_id
            self.logger.info(
                "Live feed ready | source=%s symbol=%s interval=%s bars=%d",
                group.market_config.data_source,
                group.market_config.symbol,
                group.market_config.interval,
                len(initial),
            )

        self._watchdog_interval_ms = min(
            common.get_interval_ms(group.market_config.interval)
            for group in self.feed_groups
        )
        now_ms = int(time.time() * 1000)
        next_boundary_ms = (
            now_ms // self._watchdog_interval_ms + 1
        ) * self._watchdog_interval_ms
        self._next_watchdog_time_ms = next_boundary_ms + WATCHDOG_DELAY_MS

        for group in self.feed_groups:
            group.feed.start(
                lambda candle_id, target=group: self._candle_events.put(
                    (target, candle_id)
                )
            )
        self._initialized = True
        self.logger.info(
            "WebSocket feeds started | watchdog_interval_ms=%d delay_ms=%d",
            self._watchdog_interval_ms,
            WATCHDOG_DELAY_MS,
        )

    def _predict(self, pipeline: StrategyPipeline, features: pd.DataFrame) -> pd.DataFrame:
        sequence_length = int(pipeline.model.seq_len)
        if len(features) < sequence_length:
            raise RuntimeError(
                f"Insufficient live features for model window: "
                f"need={sequence_length}, actual={len(features)}"
            )
        latest_window = features.iloc[-sequence_length:].copy()
        predicted, _ = pipeline.model.predict(
            latest_window,
            kline_interval_ms=pipeline.interval_ms,
            is_live=True,
            batch_size=1,
            diff_thresh=None,
        )
        return predicted

    def _dispatch(
        self,
        pipeline: StrategyPipeline,
        predicted: pd.DataFrame,
        features: pd.DataFrame,
    ) -> TradeIntent:
        market, current_time = _market_view(predicted, features, pipeline)
        observation = Observation(
            market=market,
            position=_position_view(pipeline.venue.get_current_state()),
            account=AccountView(equity=float(pipeline.venue.get_account_equity())),
            current_time=current_time,
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

    def _dispatch_invalid(
        self,
        pipeline: StrategyPipeline,
        candle_id: int,
        frame: Optional[pd.DataFrame],
    ) -> TradeIntent:
        row: Mapping[str, Any] = {}
        if frame is not None and not frame.empty:
            row = frame.iloc[-1]
        last_close = _optional_float(row.get("close"), 0.0) or 0.0
        market = MarketView(
            price=last_close,
            open=_optional_float(row.get("open")),
            high=_optional_float(row.get("high")),
            low=_optional_float(row.get("low")),
            close=_optional_float(row.get("close")),
            signal=Signal.INVALID,
            pred_prob=1.0,
            atr_pct=0.0,
            bar_interval_ms=pipeline.interval_ms,
            bars_to_close=math.inf,
        )
        observation = Observation(
            market=market,
            position=_position_view(pipeline.venue.get_current_state()),
            account=AccountView(equity=float(pipeline.venue.get_account_equity())),
            current_time=pd.Timestamp(candle_id, unit="ms", tz="UTC").to_pydatetime(),
        )
        intent = pipeline.strategy.process(observation)
        self.logger.warning(
            "Invalid candle signal processed | id=%s hash=%s symbol=%s interval=%s "
            "candle_id=%d action=%s",
            pipeline.spec.strategy_id,
            pipeline.spec.hash_id,
            pipeline.spec.market_config.symbol,
            pipeline.spec.market_config.interval,
            candle_id,
            intent.action.value,
        )
        return intent

    def _emit_invalid_for_group(
        self,
        group: FeedGroup,
        candle_id: int,
        intents: dict[str, TradeIntent],
    ) -> None:
        frame = group.feed.get_latest_data()
        if frame is not None and not frame.empty:
            candle_ids = pd.to_numeric(
                frame["open_time_ms_utc"],
                errors="coerce",
            )
            frame = frame.loc[candle_ids < candle_id].copy()
        for pipeline in group.pipelines:
            if pipeline.enable != "true":
                continue
            try:
                intents[pipeline.spec.strategy_id] = self._dispatch_invalid(
                    pipeline,
                    candle_id,
                    frame,
                )
            except Exception:
                self.logger.exception(
                    "Invalid signal dispatch failed | id=%s hash=%s symbol=%s",
                    pipeline.spec.strategy_id,
                    pipeline.spec.hash_id,
                    pipeline.spec.market_config.symbol,
                )

    def _process_closed_candle(
        self,
        group: FeedGroup,
        candle_id: int,
        intents: dict[str, TradeIntent],
    ) -> None:
        if group.last_candle_id is None:
            group.last_candle_id = candle_id
            return
        if candle_id <= group.last_candle_id:
            return

        interval_ms = common.get_interval_ms(group.market_config.interval)
        missing_candle_id = group.last_candle_id + interval_ms
        while missing_candle_id < candle_id:
            cached = group.feed.get_latest_data()
            received_ids = set()
            if cached is not None and not cached.empty:
                received_ids = set(
                    pd.to_numeric(
                        cached["open_time_ms_utc"],
                        errors="coerce",
                    ).dropna().astype("int64")
                )
            if missing_candle_id in received_ids:
                self._process_closed_candle(group, missing_candle_id, intents)
                missing_candle_id = group.last_candle_id + interval_ms
                continue
            self.logger.error(
                "Closed kline was missed before a later WebSocket event | "
                "symbol=%s interval=%s candle_id=%d",
                group.market_config.symbol,
                group.market_config.interval,
                missing_candle_id,
            )
            self._emit_invalid_for_group(group, missing_candle_id, intents)
            group.last_candle_id = missing_candle_id
            missing_candle_id += interval_ms

        frame = group.feed.get_latest_data()
        if frame is None or frame.empty:
            self.logger.error(
                "Closed-kline event has no cached data | symbol=%s interval=%s "
                "candle_id=%d",
                group.market_config.symbol,
                group.market_config.interval,
                candle_id,
            )
            self._emit_invalid_for_group(group, candle_id, intents)
            group.last_candle_id = candle_id
            return

        candle_ids = pd.to_numeric(frame["open_time_ms_utc"], errors="coerce")
        frame = frame.loc[candle_ids <= candle_id].copy()
        if frame.empty or int(frame.iloc[-1]["open_time_ms_utc"]) != candle_id:
            self.logger.error(
                "Closed-kline event is absent from cache | symbol=%s interval=%s "
                "candle_id=%d",
                group.market_config.symbol,
                group.market_config.interval,
                candle_id,
            )
            self._emit_invalid_for_group(group, candle_id, intents)
            group.last_candle_id = candle_id
            return

        for pipeline in group.pipelines:
            if pipeline.enable != "true":
                continue
            try:
                prepared = _prepare_market_frame(frame, pipeline.spec)
                features = pipeline.feature_factory.generate(prepared)
                predicted = self._predict(pipeline, features)
                intents[pipeline.spec.strategy_id] = self._dispatch(
                    pipeline,
                    predicted,
                    features,
                )
            except Exception:
                self.logger.exception(
                    "Strategy cycle failed; dispatching INVALID | "
                    "id=%s hash=%s symbol=%s",
                    pipeline.spec.strategy_id,
                    pipeline.spec.hash_id,
                    pipeline.spec.market_config.symbol,
                )
                try:
                    intents[pipeline.spec.strategy_id] = self._dispatch_invalid(
                        pipeline,
                        candle_id,
                        frame,
                    )
                except Exception:
                    self.logger.exception(
                        "Fallback INVALID dispatch failed | id=%s hash=%s symbol=%s",
                        pipeline.spec.strategy_id,
                        pipeline.spec.hash_id,
                        pipeline.spec.market_config.symbol,
                    )
        group.last_candle_id = candle_id

    def _run_watchdog(
        self,
        now_ms: int,
        intents: dict[str, TradeIntent],
    ) -> None:
        while now_ms >= self._next_watchdog_time_ms:
            boundary_ms = self._next_watchdog_time_ms - WATCHDOG_DELAY_MS
            for group in self.feed_groups:
                interval_ms = common.get_interval_ms(group.market_config.interval)
                expected_candle_id = _latest_expected_candle_id(
                    boundary_ms,
                    interval_ms,
                )
                while (
                    group.last_candle_id is not None
                    and group.last_candle_id < expected_candle_id
                ):
                    next_candle_id = group.last_candle_id + interval_ms
                    frame = group.feed.get_latest_data()
                    received_ids = set()
                    if frame is not None and not frame.empty:
                        received_ids = set(
                            pd.to_numeric(
                                frame["open_time_ms_utc"],
                                errors="coerce",
                            ).dropna().astype("int64")
                        )
                    if next_candle_id in received_ids:
                        self._process_closed_candle(group, next_candle_id, intents)
                        continue
                    self.logger.error(
                        "Expected WebSocket kline was not received; dispatching INVALID | "
                        "symbol=%s interval=%s candle_id=%d",
                        group.market_config.symbol,
                        group.market_config.interval,
                        next_candle_id,
                    )
                    self._emit_invalid_for_group(group, next_candle_id, intents)
                    group.last_candle_id = next_candle_id
            self._next_watchdog_time_ms += self._watchdog_interval_ms

    def run_step(
        self,
        wait_timeout_seconds: float = 0.0,
    ) -> dict[str, TradeIntent]:
        if self._closed:
            raise RuntimeError("LiveRunner is already closed")
        if wait_timeout_seconds < 0:
            raise ValueError("wait_timeout_seconds must not be negative")
        if not self._initialized:
            self.initialize()
        intents: dict[str, TradeIntent] = {}

        try:
            group, candle_id = self._candle_events.get(
                timeout=wait_timeout_seconds,
            )
            self._process_closed_candle(group, candle_id, intents)
        except queue.Empty:
            pass

        while True:
            try:
                group, candle_id = self._candle_events.get_nowait()
            except queue.Empty:
                break
            self._process_closed_candle(group, candle_id, intents)

        self._run_watchdog(int(time.time() * 1000), intents)
        return intents

    def run_forever(self) -> None:
        try:
            self.initialize()
            self.logger.info("Live runner started")
            while True:
                now_ms = int(time.time() * 1000)
                wait_seconds = max(
                    0.0,
                    (self._next_watchdog_time_ms - now_ms) / 1000.0,
                )
                self.run_step(wait_seconds)
        except KeyboardInterrupt:
            self.logger.info("Live runner stopped by user")
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
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
        default=os.path.join(os.path.dirname(__file__), "live_config.json"),
        help="Path to the live strategy JSON configuration",
    )
    args = parser.parse_args()

    logger, _ = common.setup_session_logger(
        sub_folder="live_runner",
        symbol="MULTI",
    )
    runner = LiveRunner.from_config(
        args.config,
        logger=logger,
    )
    runner.run_forever()


if __name__ == "__main__":
    main()
