"""Synchronous venue facade for the asynchronous cTrader Open API SDK."""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Iterable

from trade.core.protocol import PositionDir
from trade.core.venue_base import VenueBase


CTRADER_SYMBOL_MAP = {
    "BTCUSDT": "BTCUSD",
    "DOGEUSDT": "DOGEUSD",
    "ETHUSDT": "ETHUSD",
}


class CTraderOpenApiConnection:
    """Run Spotware's Twisted client behind a blocking request interface."""

    CLIENT_ID_FILES = ("client_id", "ctrader_client_id")
    CLIENT_SECRET_FILES = ("client_secret", "ctrader_client_secret")
    ACCESS_TOKEN_FILES = ("access_token", "ctrader_access_token")
    ACCOUNT_ID_FILES = ("account_id", "ctid_trader_account_id")

    _reactor_lock = threading.Lock()
    _reactor_ready = threading.Event()
    _reactor_thread: threading.Thread | None = None

    _AUTHENTICATION_MESSAGES = {
        "ProtoOAApplicationAuthReq",
        "ProtoOAGetAccountListByAccessTokenReq",
        "ProtoOAAccountAuthReq",
    }

    def __init__(
        self,
        key_path: str,
        *,
        account_id: int | str | None = None,
        environment: str = "demo",
        timeout: float = 10.0,
        logger: logging.Logger | None = None,
    ):
        try:
            from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol
            from ctrader_open_api.messages import OpenApiMessages_pb2
        except ImportError as exc:
            raise RuntimeError(
                "cTrader support requires the ctrader-open-api package"
            ) from exc

        self._protobuf = Protobuf
        self._messages = OpenApiMessages_pb2
        self._timeout = float(timeout)
        self._logger = logger or logging.getLogger("trade.ctrader.connection")
        self._connected = threading.Event()
        self._authorized = threading.Event()
        self._authentication_lock = threading.Lock()
        self._has_authenticated = False
        self._closed = False
        self._quotes: dict[int, dict[str, float | int]] = {}
        self._quote_events: dict[int, threading.Event] = {}
        self._quote_lock = threading.Lock()

        self._client_id = self._load_first(key_path, self.CLIENT_ID_FILES)
        self._client_secret = self._load_first(key_path, self.CLIENT_SECRET_FILES)
        self._access_token = self._load_first(key_path, self.ACCESS_TOKEN_FILES)
        if account_id in (None, ""):
            account_id_text = self._load_first(
                key_path,
                self.ACCOUNT_ID_FILES,
                required=False,
            )
            account_id = int(account_id_text) if account_id_text else None

        normalized_environment = str(environment).strip().casefold()
        if normalized_environment not in {"demo", "live"}:
            raise ValueError("cTrader environment must be 'demo' or 'live'")
        host = (
            EndPoints.PROTOBUF_LIVE_HOST
            if normalized_environment == "live"
            else EndPoints.PROTOBUF_DEMO_HOST
        )

        self._ensure_reactor()
        self._client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)
        self._call_in_reactor(self._client.startService)
        if not self._connected.wait(self._timeout):
            self.shutdown()
            raise TimeoutError(f"Timed out connecting to cTrader {normalized_environment}")

        try:
            self._authenticate(account_id)
        except Exception:
            self.shutdown()
            raise

    @staticmethod
    def _load_first(
        directory: str,
        candidates: Iterable[str],
        *,
        required: bool = True,
    ) -> str | None:
        for filename in candidates:
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    value = handle.read().strip()
                if value:
                    return value
        if required:
            raise FileNotFoundError(
                f"No cTrader credential file found in {directory}: {tuple(candidates)}"
            )
        return None

    @classmethod
    def _ensure_reactor(cls) -> None:
        from twisted.internet import reactor

        with cls._reactor_lock:
            if cls._reactor_thread is not None:
                return
            if reactor.running:
                cls._reactor_thread = threading.current_thread()
                cls._reactor_ready.set()
                return
            if getattr(reactor, "_startedBefore", False):
                raise RuntimeError("The Twisted reactor cannot be restarted")

            def run_reactor() -> None:
                reactor.run(installSignalHandlers=False)

            cls._reactor_thread = threading.Thread(
                target=run_reactor,
                name="ctrader-open-api-reactor",
                daemon=True,
            )
            cls._reactor_thread.start()
        deadline = time.monotonic() + 5.0
        while not reactor.running and time.monotonic() < deadline:
            time.sleep(0.01)
        if not reactor.running:
            raise RuntimeError("Failed to start the cTrader Open API reactor")
        cls._reactor_ready.set()

    @staticmethod
    def _call_in_reactor(callback, *args) -> None:
        from twisted.internet import reactor

        reactor.callFromThread(callback, *args)

    def _on_connected(self, _client) -> None:
        self._connected.set()
        self._logger.info("cTrader Open API connection ready")
        if self._has_authenticated and not self._closed:
            threading.Thread(
                target=self._reauthenticate,
                name=f"ctrader-reauth-{getattr(self, 'account_id', 'unknown')}",
                daemon=True,
            ).start()

    def _on_disconnected(self, _client, reason) -> None:
        self._connected.clear()
        self._authorized.clear()
        if not self._closed:
            self._logger.warning("cTrader Open API disconnected: %s", reason)

    def _on_message(self, _client, envelope) -> None:
        try:
            message = self._protobuf.extract(envelope)
        except Exception:
            self._logger.exception("Failed to decode a cTrader Open API message")
            return
        if message.__class__.__name__ != "ProtoOASpotEvent":
            return
        symbol_id = int(message.symbolId)
        with self._quote_lock:
            quote = self._quotes.setdefault(symbol_id, {})
            if message.HasField("bid"):
                quote["bid"] = int(message.bid) / 100_000.0
            if message.HasField("ask"):
                quote["ask"] = int(message.ask) / 100_000.0
            if message.HasField("timestamp"):
                quote["timestamp"] = int(message.timestamp)
            self._quote_events.setdefault(symbol_id, threading.Event()).set()

    @staticmethod
    def _set_message_fields(message, fields: dict[str, Any]) -> None:
        for name, value in fields.items():
            target = getattr(message, name)
            if isinstance(value, (list, tuple)) and hasattr(target, "extend"):
                target.extend(value)
            else:
                setattr(message, name, value)

    def _authenticate(self, account_id: int | str | None) -> None:
        with self._authentication_lock:
            self.request(
                "ProtoOAApplicationAuthReq",
                clientId=self._client_id,
                clientSecret=self._client_secret,
            )
            accounts = self.request(
                "ProtoOAGetAccountListByAccessTokenReq",
                accessToken=self._access_token,
            )
            granted_ids = [
                int(item.ctidTraderAccountId)
                for item in accounts.ctidTraderAccount
            ]
            if account_id is None:
                if len(granted_ids) != 1:
                    raise ValueError(
                        "account_id is required when the cTrader token grants "
                        f"{len(granted_ids)} accounts"
                    )
                account_id = granted_ids[0]
            self.account_id = int(account_id)
            if self.account_id not in granted_ids:
                raise ValueError(
                    f"cTrader account {self.account_id} is not granted by the access token"
                )
            self.request(
                "ProtoOAAccountAuthReq",
                ctidTraderAccountId=self.account_id,
                accessToken=self._access_token,
            )
            self._has_authenticated = True
            self._authorized.set()

    def _reauthenticate(self) -> None:
        try:
            self._authenticate(self.account_id)
        except Exception:
            self._logger.exception("cTrader Open API reauthentication failed")

    def request(self, message_name: str, **fields):
        if self._closed:
            raise RuntimeError("cTrader connection is closed")
        if not self._connected.wait(self._timeout):
            raise ConnectionError("cTrader Open API is disconnected")
        if (
            message_name not in self._AUTHENTICATION_MESSAGES
            and not self._authorized.wait(self._timeout)
        ):
            raise ConnectionError("cTrader account session is not authorized")
        message_class = getattr(self._messages, message_name)
        message = message_class()
        self._set_message_fields(message, fields)
        completed = threading.Event()
        outcome: dict[str, Any] = {}

        def send() -> None:
            deferred = self._client.send(
                message,
                responseTimeoutInSeconds=self._timeout,
            )

            def succeeded(envelope):
                try:
                    outcome["response"] = self._protobuf.extract(envelope)
                except Exception as exc:
                    outcome["error"] = exc
                completed.set()
                return envelope

            def failed(failure):
                outcome["error"] = failure
                completed.set()
                return failure

            deferred.addCallbacks(succeeded, failed)

        self._call_in_reactor(send)
        if not completed.wait(self._timeout + 1.0):
            raise TimeoutError(f"Timed out waiting for {message_name}")
        if "error" in outcome:
            raise RuntimeError(
                f"cTrader request {message_name} failed: {outcome['error']}"
            )
        response = outcome["response"]
        response_name = response.__class__.__name__
        if response_name.endswith("ErrorRes") or response_name.endswith("ErrorEvent"):
            raise RuntimeError(
                f"cTrader request {message_name} failed: "
                f"{getattr(response, 'errorCode', response_name)}: "
                f"{getattr(response, 'description', '')}"
            )
        return response

    def subscribe_spots(self, account_id: int, symbol_id: int) -> None:
        event = self._quote_events.setdefault(symbol_id, threading.Event())
        self.request(
            "ProtoOASubscribeSpotsReq",
            ctidTraderAccountId=account_id,
            symbolId=[symbol_id],
            subscribeToSpotTimestamp=True,
        )
        if not event.wait(self._timeout):
            raise TimeoutError(f"Timed out waiting for cTrader quote for {symbol_id}")

    def latest_price(self, symbol_id: int, *, is_buy: bool) -> float:
        event = self._quote_events.setdefault(symbol_id, threading.Event())
        if not event.wait(self._timeout):
            raise TimeoutError(f"No cTrader quote available for {symbol_id}")
        field = "ask" if is_buy else "bid"
        with self._quote_lock:
            price = self._quotes.get(symbol_id, {}).get(field)
        if price is None or not math.isfinite(float(price)) or float(price) <= 0:
            raise RuntimeError(f"cTrader quote has no valid {field} for {symbol_id}")
        return float(price)

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        client = getattr(self, "_client", None)
        if client is not None:
            self._call_in_reactor(client.stopService)


class CTraderVenue(VenueBase):
    """cTrader venue scoped to one account, symbol, and strategy label."""

    MARKET_ORDER = 1
    BUY = 1
    SELL = 2

    def __init__(
        self,
        key_path: str,
        symbol: str,
        magic: int | str | None = None,
        *,
        logger: logging.Logger | None = None,
        account_id: int | str | None = None,
        environment: str = "demo",
        broker_symbol: str | None = None,
        timeout: float = 10.0,
        api: Any = None,
    ):
        self.logger = logger or logging.getLogger("trade.ctrader")
        self.label = str(magic if magic is not None else "financial-ml-system")[:100]
        self.symbol = str(
            broker_symbol
            or CTRADER_SYMBOL_MAP.get(str(symbol).upper(), str(symbol))
        )
        self.api = api or CTraderOpenApiConnection(
            key_path,
            account_id=account_id,
            environment=environment,
            timeout=timeout,
            logger=self.logger,
        )
        self.account_id = int(
            account_id if account_id not in (None, "") else self.api.account_id
        )
        try:
            self._load_account_and_symbol()
        except Exception:
            if api is None:
                self.api.shutdown()
            raise

    @staticmethod
    def _normalized_symbol(value: str) -> str:
        return "".join(character for character in value.upper() if character.isalnum())

    def _load_account_and_symbol(self) -> None:
        trader_response = self.api.request(
            "ProtoOATraderReq",
            ctidTraderAccountId=self.account_id,
        )
        self._limited_risk = bool(
            getattr(trader_response.trader, "isLimitedRisk", False)
        )

        symbols = self.api.request(
            "ProtoOASymbolsListReq",
            ctidTraderAccountId=self.account_id,
            includeArchivedSymbols=False,
        )
        target = self._normalized_symbol(self.symbol)
        light_symbol = next(
            (
                item
                for item in symbols.symbol
                if self._normalized_symbol(
                    str(getattr(item, "symbolName", getattr(item, "name", "")))
                )
                == target
            ),
            None,
        )
        if light_symbol is None:
            raise ValueError(f"cTrader symbol is not available: {self.symbol}")
        self.symbol_id = int(light_symbol.symbolId)

        details = self.api.request(
            "ProtoOASymbolByIdReq",
            ctidTraderAccountId=self.account_id,
            symbolId=[self.symbol_id],
        )
        if not details.symbol:
            raise RuntimeError(f"cTrader returned no details for symbol {self.symbol}")
        symbol_info = details.symbol[0]
        self.volume_min = max(1, int(getattr(symbol_info, "minVolume", 0) or 1))
        self.volume_step = max(1, int(getattr(symbol_info, "stepVolume", 0) or 1))
        self.volume_max = int(getattr(symbol_info, "maxVolume", 0) or 0)
        self.digits = int(symbol_info.digits)
        self._guaranteed_stop_loss = bool(
            getattr(symbol_info, "guaranteedStopLoss", False)
        )
        self.api.subscribe_spots(self.account_id, self.symbol_id)
        self.logger.info(
            "cTrader venue ready | account=%s symbol=%s symbol_id=%s label=%s",
            self.account_id,
            self.symbol,
            self.symbol_id,
            self.label,
        )

    @staticmethod
    def _money(value: int, digits: int) -> float:
        return float(value) / (10 ** int(digits))

    def get_account_equity(self) -> float:
        trader_response = self.api.request(
            "ProtoOATraderReq",
            ctidTraderAccountId=self.account_id,
        )
        trader = trader_response.trader
        balance = self._money(trader.balance, getattr(trader, "moneyDigits", 0))
        pnl_response = self.api.request(
            "ProtoOAGetPositionUnrealizedPnLReq",
            ctidTraderAccountId=self.account_id,
        )
        pnl = sum(
            self._money(item.netUnrealizedPnL, pnl_response.moneyDigits)
            for item in pnl_response.positionUnrealizedPnL
        )
        equity = balance + pnl
        if not math.isfinite(equity) or equity <= 0:
            raise RuntimeError("cTrader returned invalid account equity")
        return equity

    def _positions(self) -> list[Any]:
        response = self.api.request(
            "ProtoOAReconcileReq",
            ctidTraderAccountId=self.account_id,
            returnProtectionOrders=False,
        )
        return [
            position
            for position in response.position
            if int(position.tradeData.symbolId) == self.symbol_id
            and str(getattr(position.tradeData, "label", "")) == self.label
        ]

    def get_current_state(self):
        positions = self._positions()
        if not positions:
            return PositionDir.FLAT, 0, 0.0
        sides = {int(position.tradeData.tradeSide) for position in positions}
        if len(sides) != 1:
            raise RuntimeError(
                "cTrader strategy label has simultaneous long and short positions"
            )
        total_volume = sum(int(position.tradeData.volume) for position in positions)
        if total_volume <= 0:
            return PositionDir.FLAT, 0, 0.0
        average_price = sum(
            int(position.tradeData.volume) * float(position.price)
            for position in positions
        ) / total_volume
        direction = (
            PositionDir.POSITIVE if sides == {self.BUY} else PositionDir.NEGATIVE
        )
        return direction, len(positions), average_price

    def get_server_time(self):
        return datetime.now(timezone.utc)

    def get_last_position_open_time(self):
        timestamps = [
            int(getattr(position.tradeData, "openTimestamp", 0) or 0)
            for position in self._positions()
        ]
        timestamps = [timestamp for timestamp in timestamps if timestamp > 0]
        if not timestamps:
            return None
        return datetime.fromtimestamp(max(timestamps) / 1000.0, tz=timezone.utc)

    def _normalize_volume(self, size: float) -> int:
        requested = int(
            (Decimal(str(size)) * 100).to_integral_value(rounding=ROUND_DOWN)
        )
        volume = requested // self.volume_step * self.volume_step
        if volume < self.volume_min:
            raise ValueError(
                f"cTrader order volume {volume / 100:g} is below minimum "
                f"{self.volume_min / 100:g}"
            )
        return volume

    def _volume_batches(self, total_volume: int) -> list[int]:
        maximum = self.volume_max or total_volume
        maximum = maximum // self.volume_step * self.volume_step
        if maximum < self.volume_min:
            raise RuntimeError("cTrader symbol has invalid maximum volume metadata")
        batches = []
        remaining = total_volume
        while remaining > maximum:
            batch = maximum
            remainder = remaining - batch
            if remainder < self.volume_min:
                adjustment = (
                    self.volume_min - remainder + self.volume_step - 1
                ) // self.volume_step * self.volume_step
                batch -= adjustment
            if batch < self.volume_min:
                raise ValueError("cTrader volume cannot be split into valid orders")
            batches.append(batch)
            remaining -= batch
        if remaining < self.volume_min:
            raise ValueError("cTrader order remainder is below the symbol minimum volume")
        batches.append(remaining)
        return batches

    @staticmethod
    def _order_percentage(value: Any, name: str) -> float | None:
        if value is None:
            return None
        percentage = float(value)
        if not math.isfinite(percentage) or percentage <= 0:
            raise ValueError(f"{name} must be a positive finite ratio")
        return percentage

    def submit_order(
        self,
        size,
        is_buy,
        stop_loss_pct=None,
        take_profit_pct=None,
        interval_ms=500,
        **legacy,
    ):
        if stop_loss_pct is None:
            stop_loss_pct = legacy.pop("stop_loss", None)
        if take_profit_pct is None:
            take_profit_pct = legacy.pop("take_profit", None)
        if legacy:
            raise TypeError(f"Unsupported cTrader order arguments: {sorted(legacy)}")
        stop_loss_pct = self._order_percentage(stop_loss_pct, "stop_loss_pct")
        take_profit_pct = self._order_percentage(
            take_profit_pct,
            "take_profit_pct",
        )
        if self._limited_risk and stop_loss_pct is None:
            raise ValueError("cTrader limited-risk accounts require a stop loss")

        total_volume = self._normalize_volume(float(size))
        batches = self._volume_batches(total_volume)
        price = self.api.latest_price(self.symbol_id, is_buy=bool(is_buy))
        relative_stop = (
            max(1, int(round(price * stop_loss_pct * 100_000)))
            if stop_loss_pct is not None
            else None
        )
        relative_take_profit = (
            max(1, int(round(price * take_profit_pct * 100_000)))
            if take_profit_pct is not None
            else None
        )
        responses = []
        for index, volume in enumerate(batches):
            fields = {
                "ctidTraderAccountId": self.account_id,
                "symbolId": self.symbol_id,
                "orderType": self.MARKET_ORDER,
                "tradeSide": self.BUY if is_buy else self.SELL,
                "volume": volume,
                "label": self.label,
                "comment": f"financial-ml-system batch {index + 1}/{len(batches)}",
            }
            if relative_stop is not None:
                fields["relativeStopLoss"] = relative_stop
            if relative_take_profit is not None:
                fields["relativeTakeProfit"] = relative_take_profit
            if self._limited_risk:
                if not self._guaranteed_stop_loss:
                    raise RuntimeError(
                        "cTrader limited-risk account requires a symbol with guaranteed stops"
                    )
                fields["guaranteedStopLoss"] = True
            responses.append(self.api.request("ProtoOANewOrderReq", **fields))
            if index + 1 < len(batches):
                time.sleep(max(0.0, float(interval_ms) / 1000.0))
        return responses

    def close_position(self, size=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported cTrader close arguments: {sorted(kwargs)}")
        positions = self._positions()
        if not positions:
            return []
        remaining = None if size is None else self._normalize_volume(abs(float(size)))
        responses = []
        for position in positions:
            available = int(position.tradeData.volume)
            volume = available if remaining is None else min(available, remaining)
            if volume <= 0:
                break
            responses.append(
                self.api.request(
                    "ProtoOAClosePositionReq",
                    ctidTraderAccountId=self.account_id,
                    positionId=int(position.positionId),
                    volume=volume,
                )
            )
            if remaining is not None:
                remaining -= volume
        return responses

    def shutdown(self) -> None:
        self.api.shutdown()
