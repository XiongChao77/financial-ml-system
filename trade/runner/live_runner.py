#!/usr/bin/env python3
"""Multi-strategy live runner with shared market-data feeds.

Pipeline:

    shared data feed -> feature generation -> model inference -> MarketView
    -> strategy intent -> configured live venue

Each strategy is restored from a canonical backtest report selected by the
top-level live-config hash.  Live-only settings choose the model artifact,
inference device, and execution venue; strategy and market parameters remain
owned by the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import ntpath
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Optional

import pandas as pd

from data_process import common
from data_process.utils import config_from_dict_train, json_safe
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


SUPPORTED_VENUES = {"mt5": "MT5", "bybit": "Bybit", "binance": "Binance"}
SUPPORTED_BINANCE_DATA_SOURCES = {
    "binance",
    "binance_api",
    "binance_public_data",
}


@dataclass(frozen=True)
class FeedKey:
    """Identity of one independently requested live market-data stream."""

    market_category: str
    data_source: str
    trading_type: str
    symbol: str
    interval: str

    @classmethod
    def from_market_config(cls, config: common.BaseDefine) -> "FeedKey":
        data_source = str(config.data_source).casefold()
        if data_source in SUPPORTED_BINANCE_DATA_SOURCES:
            data_source = "binance_public_data"
        return cls(
            market_category=str(config.market_category).casefold(),
            data_source=data_source,
            trading_type=str(config.trading_type).casefold(),
            symbol=str(config.symbol).upper(),
            interval=str(config.interval).casefold(),
        )


@dataclass(frozen=True)
class LiveVenueConfig:
    """Live-only connection data. Secrets are deliberately omitted from repr."""

    kind: str
    path: str
    login: Optional[str] = None
    password: Optional[str] = field(default=None, repr=False)
    server: Optional[str] = None


@dataclass(frozen=True)
class LiveStrategySpec:
    id: str
    hash_id: str
    report_path: str
    model_path: str
    device: str
    market_config: common.BaseDefine
    train_config: Any
    strategy_config: Any
    broker_config: BrokerConfig
    venue_config: LiveVenueConfig

    @property
    def feed_key(self) -> FeedKey:
        return FeedKey.from_market_config(self.market_config)


@dataclass
class StrategyPipeline:
    spec: LiveStrategySpec
    model: Any
    venue: Any
    strategy: Any
    feature_factory: Any
    interval_ms: int
    atr_ref_bars: int

    @property
    def required_bars(self) -> int:
        feature_history = int(self.feature_factory.get_global_min_history())
        model_history = max(1, int(self.model.seq_len)) * 2
        atr_history = max(1, int(self.atr_ref_bars)) * 2
        return feature_history + max(model_history, atr_history)


@dataclass
class FeatureGroup:
    factory: Any
    pipelines: list[StrategyPipeline] = field(default_factory=list)


@dataclass
class FeedGroup:
    key: FeedKey
    feed: Any
    required_bars: int
    feature_groups: list[FeatureGroup]
    last_candle_id: Optional[int] = None
    next_poll_time_ms: int = 0


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

    if path.lower().endswith(".jsonl"):
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
        return

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        for index, record in enumerate(payload):
            if not isinstance(record, dict):
                raise TypeError(f"Expected a JSON object at {path}[{index}]")
            yield record
    else:
        raise TypeError(f"Expected a JSON object or array in {path}")


def _period_reports(record: Mapping[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(record.get("params"), Mapping):
        yield "top", dict(record)
    for period in ("forward", "long", "short"):
        report = record.get(period)
        if isinstance(report, Mapping) and isinstance(report.get("params"), Mapping):
            yield period, dict(report)


def _parameter_signature(report: Mapping[str, Any]) -> str:
    params = json.loads(json.dumps(report["params"], default=str))
    data_params = params.get("data")
    if isinstance(data_params, dict):
        data_params.pop("period", None)
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def load_report_by_hash(path: str, hash_id: str) -> dict[str, Any]:
    """Find one report by params.hash, accepting standalone and period wrappers."""

    matches: list[tuple[str, dict[str, Any]]] = []
    for record in _iter_report_records(path):
        for period, report in _period_reports(record):
            if str(report["params"].get("hash", "")) == str(hash_id):
                matches.append((period, report))

    if not matches:
        raise KeyError(f"Hash {hash_id!r} was not found in report file {path}")

    signatures = {_parameter_signature(report) for _, report in matches}
    if len(signatures) != 1:
        raise ValueError(
            f"Hash {hash_id!r} resolves to conflicting parameter snapshots in {path}"
        )

    priority = {"forward": 0, "long": 1, "short": 2, "top": 3}
    return min(matches, key=lambda item: priority[item[0]])[1]


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
) -> LiveVenueConfig:
    raw_kind = str(entry.get("venue", "")).strip()
    normalized = raw_kind.casefold()
    if normalized not in SUPPORTED_VENUES:
        choices = ", ".join(SUPPORTED_VENUES.values())
        raise ValueError(f"venue must be one of {choices}; got {raw_kind!r}")

    kind = SUPPORTED_VENUES[normalized]
    section = _venue_section(entry, kind)
    if kind == "MT5":
        required = ("path", "login", "password", "server")
        missing = [key for key in required if section.get(key) in (None, "")]
        if missing:
            raise ValueError(f"MT5 configuration is missing: {', '.join(missing)}")
        return LiveVenueConfig(
            kind=kind,
            path=_resolve_path(config_path, section["path"]),
            login=str(section["login"]),
            password=str(section["password"]),
            server=str(section["server"]),
        )

    key_path = section.get("key_path", section.get("path"))
    if not key_path:
        raise ValueError(f"{kind} configuration requires key_path or path")
    resolved_key_path = os.path.realpath(_resolve_path(config_path, key_path))
    if not os.path.isdir(resolved_key_path):
        raise FileNotFoundError(f"{kind} key directory not found: {resolved_key_path}")
    return LiveVenueConfig(
        kind=kind,
        path=resolved_key_path,
    )


def _strategy_entries(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    strategies = payload.get("strategies")
    if strategies is not None:
        if not isinstance(strategies, Mapping):
            raise TypeError("live config strategies must be an object")
        return strategies
    return {
        key: value
        for key, value in payload.items()
        if not str(key).startswith("_")
    }


def _parse_enable(value: Any, strategy_id: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(
        f"Live strategy {strategy_id!r} enable must be a boolean or boolean string"
    )


def load_live_strategy_specs(path: str) -> list[LiveStrategySpec]:
    """Restore every strategy from live-only config plus its canonical report."""

    config_path = os.path.abspath(path)
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("Live configuration root must be an object")

    from trade.runner.backtest_runner import (
        strategy_config_for_preparation,
        strategy_config_from_dict,
    )

    specs = []
    for raw_id, raw_entry in _strategy_entries(payload).items():
        strategy_id = str(raw_id).strip()
        if not strategy_id:
            raise ValueError("Live strategy id must not be empty")
        if not isinstance(raw_entry, Mapping):
            raise TypeError(f"Live strategy {strategy_id!r} must be an object")
        entry = dict(raw_entry)
        if "enable" not in entry:
            raise KeyError(f"Live strategy {strategy_id!r} is missing enable")
        if not _parse_enable(entry["enable"], strategy_id):
            continue

        hash_id = str(entry.get("hash", "")).strip()
        if not hash_id:
            raise ValueError(f"Enabled live strategy {strategy_id!r} has no hash")
        report_path = _resolve_path(config_path, entry.get("report"))
        report = load_report_by_hash(report_path, hash_id)
        params = _required_mapping(report, "params", "report")

        market_params = _required_mapping(params, "common", "report.params")
        train_params = _required_mapping(params, "train", "report.params")
        strategy_params = _required_mapping(params, "strategy", "report.params")
        report_broker_params = _required_mapping(params, "broker", "report.params")

        market_config = common.BaseDefine(**market_params)
        train_config = config_from_dict_train(train_params)
        strategy_config = strategy_config_for_preparation(
            strategy_config_from_dict(strategy_params),
            market_config,
        )
        broker_params = {
            **report_broker_params,
            **dict(entry.get("broker_config") or {}),
        }

        model_path = _resolve_path(config_path, entry.get("model_path"))
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model artifact directory not found: {model_path}")

        specs.append(
            LiveStrategySpec(
                id=strategy_id,
                hash_id=hash_id,
                report_path=report_path,
                model_path=model_path,
                device=str(entry.get("device", "cpu")),
                market_config=market_config,
                train_config=train_config,
                strategy_config=strategy_config,
                broker_config=BrokerConfig(**broker_params),
                venue_config=_parse_venue_config(config_path, entry),
            )
        )

    if not specs:
        raise ValueError("Live configuration contains no enabled strategies")
    return specs


def _feature_signature(spec: LiveStrategySpec) -> str:
    payload = json.dumps(
        json_safe(spec.train_config.feature_conf_list),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _magic_number(hash_id: str) -> int:
    candidate = hash_id.strip().lower()
    if not candidate or any(char not in "0123456789abcdef" for char in candidate):
        candidate = hashlib.sha256(hash_id.encode("utf-8")).hexdigest()
    return int(candidate[:15], 16)


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
    atr_series = common.stop_loss_atr_pct(feature_frame, pipeline.atr_ref_bars)
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
        feed_factory: Optional[Callable[[FeedKey, int], Any]] = None,
        feature_factory: Optional[Callable[[LiveStrategySpec], Any]] = None,
        model_factory: Optional[Callable[[LiveStrategySpec], Any]] = None,
        venue_factory: Optional[Callable[[LiveStrategySpec, logging.Logger], Any]] = None,
        strategy_factory: Optional[Callable[[LiveStrategySpec, Any], Any]] = None,
        process_latest_on_start: bool = False,
    ):
        if not specs:
            raise ValueError("LiveRunner requires at least one strategy")
        self.logger = logger or logging.getLogger("trade.live")
        self.process_latest_on_start = bool(process_latest_on_start)
        self._feed_factory = feed_factory or self._default_feed_factory
        self._feature_factory = feature_factory or self._default_feature_factory
        self._model_factory = model_factory or self._default_model_factory
        self._venue_factory = venue_factory or self._default_venue_factory
        self._strategy_factory = strategy_factory or self._default_strategy_factory
        self._initialized = False
        self._closed = False
        self.pipelines: list[StrategyPipeline] = []
        self.feed_groups: dict[FeedKey, FeedGroup] = {}
        try:
            self._build(specs)
        except Exception:
            self.close()
            raise

    @classmethod
    def from_config(cls, path: str, **kwargs) -> "LiveRunner":
        return cls(load_live_strategy_specs(path), **kwargs)

    @staticmethod
    def _default_feed_factory(key: FeedKey, max_len: int):
        if key.data_source.casefold() not in SUPPORTED_BINANCE_DATA_SOURCES:
            raise ValueError(
                f"Unsupported live data source {key.data_source!r}; "
                "only Binance public kline feeds are currently implemented"
            )
        from trade.venue.live.binance_data_feed import BinanceDataFeed

        return BinanceDataFeed(
            key.symbol,
            key.interval,
            key.trading_type,
            max_len=max_len,
        )

    @staticmethod
    def _default_feature_factory(spec: LiveStrategySpec):
        return common.FeatureFactory(
            common.get_interval_ms(spec.market_config.interval),
            feature_conf_list=spec.train_config.feature_conf_list,
        )

    @staticmethod
    def _default_model_factory(spec: LiveStrategySpec):
        from model.model_loader import ModelHandler

        return ModelHandler(tarin_out_path=spec.model_path, device=spec.device)

    @staticmethod
    def _default_venue_factory(spec: LiveStrategySpec, logger: logging.Logger):
        config = spec.venue_config
        if config.kind == "MT5":
            from trade.venue.live.ftmo.mt5_venue import MT5Venue

            return MT5Venue(
                config.path,
                spec.market_config.symbol,
                _magic_number(f"{spec.id}:{spec.hash_id}"),
                logger=logger,
                login=config.login,
                password=config.password,
                server=config.server,
            )
        if config.kind == "Bybit":
            from trade.venue.live.bybit.bybit_venue import BybitVenue

            return BybitVenue(config.path, spec.market_config.symbol, logger=logger)
        if config.kind == "Binance":
            from trade.venue.live.binance.binance_venue import BinanceVenue

            return BinanceVenue(config.path, spec.market_config.symbol, logger=logger)
        raise ValueError(f"Unsupported venue: {config.kind}")

    @staticmethod
    def _default_strategy_factory(spec: LiveStrategySpec, venue: Any):
        from trade.strategy.strategy_bbm import BbmSignalStrategy, BbmStrategyConfig
        from trade.strategy.strategy_ml import MlSignalStrategy, MlStrategyConfig

        equity = float(venue.get_account_equity())
        if equity <= 0:
            raise RuntimeError(
                f"Venue returned invalid account equity for {spec.id} ({spec.hash_id})"
            )
        leverage = float(spec.broker_config.leverage)

        if isinstance(spec.strategy_config, MlStrategyConfig):
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
                leverage=leverage,
            )
        raise TypeError(
            "Live runner currently supports MlStrategyConfig and "
            f"BbmStrategyConfig, got {type(spec.strategy_config).__name__}"
        )

    def _validate_execution_targets(self, specs: list[LiveStrategySpec]) -> None:
        strategy_ids = [spec.id for spec in specs]
        if len(set(strategy_ids)) != len(strategy_ids):
            raise ValueError("Live strategy IDs must be unique")

        mt5_magics = [
            _magic_number(f"{spec.id}:{spec.hash_id}")
            for spec in specs
            if spec.venue_config.kind == "MT5"
        ]
        if len(set(mt5_magics)) != len(mt5_magics):
            raise ValueError("MT5 strategy IDs produced duplicate magic numbers")

        mt5_connections = {
            (
                spec.venue_config.path,
                spec.venue_config.login,
                spec.venue_config.password,
                spec.venue_config.server,
            )
            for spec in specs
            if spec.venue_config.kind == "MT5"
        }
        if len(mt5_connections) > 1:
            raise ValueError(
                "One LiveRunner process cannot safely switch between multiple MT5 "
                "terminal/account connections"
            )

        exchange_targets = set()
        for spec in specs:
            if spec.venue_config.kind not in {"Bybit", "Binance"}:
                continue
            target = (
                spec.venue_config.kind,
                spec.venue_config.path,
                spec.market_config.symbol,
            )
            if target in exchange_targets:
                raise ValueError(
                    "Multiple strategies cannot safely share one exchange account and "
                    f"symbol without isolated positions: {target[0]} {target[2]}"
                )
            exchange_targets.add(target)

    def _build(self, specs: list[LiveStrategySpec]) -> None:
        self._validate_execution_targets(specs)
        model_cache: dict[tuple[str, str], Any] = {}
        feature_cache: dict[tuple[FeedKey, str], Any] = {}
        grouped_pipelines: dict[FeedKey, dict[str, FeatureGroup]] = {}

        for spec in specs:
            model_key = (spec.model_path, spec.device)
            model = model_cache.get(model_key)
            if model is None:
                model = self._model_factory(spec)
                model_cache[model_key] = model
            signature = _feature_signature(spec)
            feature_key = (spec.feed_key, signature)
            feature_generator = feature_cache.get(feature_key)
            if feature_generator is None:
                feature_generator = self._feature_factory(spec)
                feature_cache[feature_key] = feature_generator
            venue = self._venue_factory(spec, self.logger)
            try:
                strategy = self._strategy_factory(spec, venue)
            except Exception:
                shutdown = getattr(venue, "shutdown", None)
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception:
                        self.logger.exception(
                            "Venue cleanup failed during construction: %s",
                            spec.id,
                        )
                raise

            from trade.runner.backtest_runner import atr_ref_bars_for_strategy

            pipeline = StrategyPipeline(
                spec=spec,
                model=model,
                venue=venue,
                strategy=strategy,
                feature_factory=feature_generator,
                interval_ms=common.get_interval_ms(spec.market_config.interval),
                atr_ref_bars=atr_ref_bars_for_strategy(spec.strategy_config),
            )
            self.pipelines.append(pipeline)
            feature_groups = grouped_pipelines.setdefault(spec.feed_key, {})
            feature_groups.setdefault(
                signature,
                FeatureGroup(factory=feature_generator),
            ).pipelines.append(pipeline)

        for feed_key, feature_groups in grouped_pipelines.items():
            groups = list(feature_groups.values())
            required_bars = max(
                pipeline.required_bars
                for group in groups
                for pipeline in group.pipelines
            )
            feed = self._feed_factory(feed_key, required_bars + 500)
            self.feed_groups[feed_key] = FeedGroup(
                key=feed_key,
                feed=feed,
                required_bars=required_bars,
                feature_groups=groups,
            )

        self.logger.info(
            "Live runner built | strategies=%d shared_feeds=%d models=%d",
            len(self.pipelines),
            len(self.feed_groups),
            len(model_cache),
        )

    def initialize(self) -> None:
        if self._closed:
            raise RuntimeError("LiveRunner is already closed")
        if self._initialized:
            return
        for group in self.feed_groups.values():
            interval_ms = common.get_interval_ms(group.key.interval)
            group.feed.initialize_cache(group.required_bars, interval_ms)
            initial = group.feed.get_latest_data()
            if initial is None or initial.empty:
                raise RuntimeError(f"Failed to warm live feed: {group.key}")
            latest_candle_id = int(initial.iloc[-1]["open_time_ms_utc"])
            if self.process_latest_on_start:
                group.last_candle_id = None
                group.next_poll_time_ms = 0
            else:
                group.last_candle_id = latest_candle_id
                group.next_poll_time_ms = latest_candle_id + interval_ms * 2 + 500
            self.logger.info(
                "Live feed ready | source=%s symbol=%s interval=%s bars=%d",
                group.key.data_source,
                group.key.symbol,
                group.key.interval,
                len(initial),
            )
        self._initialized = True

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
            pipeline.spec.id,
            pipeline.spec.hash_id,
            pipeline.spec.market_config.symbol,
            market.signal.name,
            intent.action.value,
        )
        return intent

    def run_step(self) -> dict[str, TradeIntent]:
        if self._closed:
            raise RuntimeError("LiveRunner is already closed")
        if not self._initialized:
            self.initialize()
        intents: dict[str, TradeIntent] = {}

        for feed_group in self.feed_groups.values():
            now_ms = int(time.time() * 1000)
            if now_ms < feed_group.next_poll_time_ms:
                continue
            try:
                frame = feed_group.feed.get_latest_data()
            except Exception:
                self.logger.exception("Live feed update failed: %s", feed_group.key)
                feed_group.next_poll_time_ms = now_ms + 5_000
                continue
            if frame is None or frame.empty:
                self.logger.error("Live feed returned no data: %s", feed_group.key)
                feed_group.next_poll_time_ms = now_ms + 5_000
                continue
            candle_id = int(frame.iloc[-1]["open_time_ms_utc"])
            if candle_id == feed_group.last_candle_id:
                feed_group.next_poll_time_ms = now_ms + 5_000
                continue
            feed_group.last_candle_id = candle_id
            interval_ms = common.get_interval_ms(feed_group.key.interval)
            feed_group.next_poll_time_ms = max(
                now_ms + 1_000,
                candle_id + interval_ms * 2 + 500,
            )

            for feature_group in feed_group.feature_groups:
                representative = feature_group.pipelines[0].spec
                try:
                    prepared = _prepare_market_frame(frame, representative)
                    features = feature_group.factory.generate(prepared)
                except Exception:
                    self.logger.exception(
                        "Feature generation failed | source=%s symbol=%s interval=%s",
                        feed_group.key.data_source,
                        feed_group.key.symbol,
                        feed_group.key.interval,
                    )
                    continue

                prediction_cache: dict[int, pd.DataFrame] = {}
                for pipeline in feature_group.pipelines:
                    try:
                        model_key = id(pipeline.model)
                        if model_key not in prediction_cache:
                            prediction_cache[model_key] = self._predict(pipeline, features)
                        intents[pipeline.spec.id] = self._dispatch(
                            pipeline,
                            prediction_cache[model_key],
                            features,
                        )
                    except Exception:
                        self.logger.exception(
                            "Strategy cycle failed safely | id=%s hash=%s symbol=%s",
                            pipeline.spec.id,
                            pipeline.spec.hash_id,
                            pipeline.spec.market_config.symbol,
                        )
        return intents

    def run_forever(self, poll_interval_seconds: float = 5.0) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        try:
            self.initialize()
            self.logger.info("Live runner started")
            while True:
                self.run_step()
                time.sleep(poll_interval_seconds)
        except KeyboardInterrupt:
            self.logger.info("Live runner stopped by user")
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for pipeline in self.pipelines:
            try:
                pipeline.strategy.finalize()
            except Exception:
                self.logger.exception("Strategy finalization failed: %s", pipeline.spec.id)

        closed = set()
        for pipeline in self.pipelines:
            venue = pipeline.venue
            if id(venue) in closed:
                continue
            closed.add(id(venue))
            shutdown = getattr(venue, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    self.logger.exception("Venue shutdown failed: %s", pipeline.spec.id)

        for feed_group in self.feed_groups.values():
            shutdown = getattr(feed_group.feed, "shutdown", None)
            if not callable(shutdown):
                shutdown = getattr(feed_group.feed, "close", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    self.logger.exception("Live feed shutdown failed: %s", feed_group.key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple live strategies")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "live_config.json"),
        help="Path to the live strategy JSON configuration",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Seconds between shared-feed polls",
    )
    parser.add_argument(
        "--process-latest-on-start",
        action="store_true",
        help="Process the latest closed candle immediately instead of waiting for a new one",
    )
    args = parser.parse_args()

    logger, _ = common.setup_session_logger(
        sub_folder="live_runner",
        symbol="MULTI",
    )
    runner = LiveRunner.from_config(
        args.config,
        logger=logger,
        process_latest_on_start=args.process_latest_on_start,
    )
    runner.run_forever(args.poll_seconds)


if __name__ == "__main__":
    main()
