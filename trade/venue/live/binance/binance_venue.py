"""Binance USD-M futures implementation of the shared live venue protocol."""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Iterable
from urllib.parse import urlencode

import requests

from trade.core.protocol import PositionDir
from trade.core.venue_base import VenueBase


class BinanceVenue(VenueBase):
    """One-way Binance USD-M futures venue with fail-closed bracket creation."""

    BASE_URL = "https://fapi.binance.com"
    API_KEY_FILES = ("hmac_api_key", "binance_api_key")
    API_SECRET_FILES = ("hmac_secret", "binance_api_secret")

    def __init__(
        self,
        key_path: str,
        symbol: str,
        *,
        logger: logging.Logger | None = None,
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ):
        self.symbol = str(symbol).upper()
        self.logger = logger or logging.getLogger("trade.binance")
        self.timeout = float(timeout)
        self.session = session or requests.Session()
        try:
            self.api_key = self._load_first(key_path, self.API_KEY_FILES)
            self.api_secret = self._load_first(key_path, self.API_SECRET_FILES)
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})
            self.quantity_step, self.minimum_quantity, self.price_tick = self._load_filters()
            self._require_one_way_mode()
        except Exception:
            self.session.close()
            raise

    @staticmethod
    def _load_first(directory: str, candidates: Iterable[str]) -> str:
        for filename in candidates:
            path = os.path.join(directory, filename)
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    value = handle.read().strip()
                if value:
                    return value
        raise FileNotFoundError(
            f"No Binance credential file found in {directory}: {tuple(candidates)}"
        )

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        signed: bool = False,
    ) -> Any:
        payload = dict(params or {})
        if signed:
            payload.setdefault("recvWindow", 5000)
            payload["timestamp"] = int(time.time() * 1000)
            query = urlencode(payload)
            payload["signature"] = hmac.new(
                self.api_secret.encode("utf-8"),
                query.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        response = self.session.request(
            method,
            f"{self.BASE_URL}{path}",
            params=payload if method.upper() in {"GET", "DELETE"} else None,
            data=payload if method.upper() not in {"GET", "DELETE"} else None,
            timeout=self.timeout,
        )
        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Binance returned a non-JSON response for {path}: HTTP {response.status_code}"
            ) from exc
        if not response.ok:
            raise RuntimeError(
                f"Binance request failed for {path}: HTTP {response.status_code}, "
                f"code={body.get('code')}, message={body.get('msg')}"
            )
        return body

    def _load_filters(self) -> tuple[Decimal, Decimal, Decimal]:
        payload = self._request("GET", "/fapi/v1/exchangeInfo")
        symbol_info = next(
            (item for item in payload.get("symbols", []) if item.get("symbol") == self.symbol),
            None,
        )
        if symbol_info is None:
            raise ValueError(f"Binance USD-M symbol is not available: {self.symbol}")
        filters = {item["filterType"]: item for item in symbol_info.get("filters", [])}
        lot = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE")
        price = filters.get("PRICE_FILTER")
        if not lot or not price:
            raise RuntimeError(f"Binance symbol filters are incomplete: {self.symbol}")
        return (
            Decimal(str(lot["stepSize"])),
            Decimal(str(lot["minQty"])),
            Decimal(str(price["tickSize"])),
        )

    def _require_one_way_mode(self) -> None:
        mode = self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
        dual_side = mode.get("dualSidePosition")
        if dual_side is True or str(dual_side).casefold() == "true":
            raise RuntimeError(
                "Binance hedge mode is not supported by this live runner; "
                "switch the account to one-way mode before starting"
            )

    @staticmethod
    def _floor_to_step(value: float, step: Decimal) -> Decimal:
        amount = Decimal(str(value))
        return (amount / step).to_integral_value(rounding=ROUND_DOWN) * step

    @staticmethod
    def _decimal_string(value: Decimal) -> str:
        return format(value.normalize(), "f")

    def _position(self) -> dict[str, Any] | None:
        positions = self._request(
            "GET",
            "/fapi/v3/positionRisk",
            {"symbol": self.symbol},
            signed=True,
        )
        if not isinstance(positions, list):
            raise RuntimeError("Binance returned an invalid position response")
        return next(
            (
                position
                for position in positions
                if position.get("symbol") == self.symbol
                and float(position.get("positionAmt", 0.0)) != 0.0
            ),
            None,
        )

    def get_account_equity(self) -> float:
        account = self._request("GET", "/fapi/v3/account", signed=True)
        equity = float(account.get("totalMarginBalance", 0.0))
        if not math.isfinite(equity) or equity <= 0:
            raise RuntimeError("Binance returned invalid account equity")
        return equity

    def get_current_state(self):
        position = self._position()
        if position is None:
            return PositionDir.FLAT, 0, 0.0
        quantity = float(position["positionAmt"])
        direction = PositionDir.POSITIVE if quantity > 0 else PositionDir.NEGATIVE
        return direction, 1, float(position.get("entryPrice", 0.0))

    def get_server_time(self):
        payload = self._request("GET", "/fapi/v1/time")
        return datetime.fromtimestamp(int(payload["serverTime"]) / 1000, tz=timezone.utc)

    def get_last_position_open_time(self):
        position = self._position()
        if position is None:
            return None
        update_time = int(position.get("updateTime", 0) or 0)
        if update_time <= 0:
            return None
        return datetime.fromtimestamp(update_time / 1000, tz=timezone.utc)

    def _latest_price(self) -> float:
        payload = self._request(
            "GET",
            "/fapi/v1/ticker/price",
            {"symbol": self.symbol},
        )
        price = float(payload["price"])
        if price <= 0 or not math.isfinite(price):
            raise RuntimeError("Binance returned invalid ticker price")
        return price

    def _trigger_price(self, raw_price: float) -> str:
        price = self._floor_to_step(raw_price, self.price_tick)
        if price <= 0:
            raise ValueError("Protective trigger price rounded to zero")
        return self._decimal_string(price)

    def _place_protective_order(self, side: str, order_type: str, trigger_price: str):
        return self._request(
            "POST",
            "/fapi/v1/algoOrder",
            {
                "algoType": "CONDITIONAL",
                "symbol": self.symbol,
                "side": side,
                "type": order_type,
                "triggerPrice": trigger_price,
                "closePosition": "true",
                "workingType": "MARK_PRICE",
            },
            signed=True,
        )

    def _assert_no_open_orders(self) -> None:
        regular = self._request(
            "GET",
            "/fapi/v1/openOrders",
            {"symbol": self.symbol},
            signed=True,
        )
        conditional = self._request(
            "GET",
            "/fapi/v1/openAlgoOrders",
            {"symbol": self.symbol},
            signed=True,
        )
        if regular or conditional:
            raise RuntimeError(
                f"Refusing to open with existing Binance orders: {self.symbol}"
            )

    def _cancel_protective_orders(self) -> None:
        self._request(
            "DELETE",
            "/fapi/v1/algoOpenOrders",
            {"symbol": self.symbol},
            signed=True,
        )

    def submit_order(
        self,
        size,
        is_buy,
        stop_loss_pct=None,
        take_profit_pct=None,
        **legacy,
    ):
        if stop_loss_pct is None:
            stop_loss_pct = legacy.pop("stop_loss", None)
        if take_profit_pct is None:
            take_profit_pct = legacy.pop("take_profit", None)
        if legacy:
            raise TypeError(f"Unsupported Binance order arguments: {sorted(legacy)}")
        if self._position() is not None:
            raise RuntimeError(f"Refusing to open over an existing Binance position: {self.symbol}")
        self._assert_no_open_orders()

        quantity = self._floor_to_step(float(size), self.quantity_step)
        if quantity < self.minimum_quantity:
            raise ValueError(
                f"Binance order quantity {quantity} is below minimum {self.minimum_quantity}"
            )
        side = "BUY" if is_buy else "SELL"
        exit_side = "SELL" if is_buy else "BUY"
        price = self._latest_price()
        order = self._request(
            "POST",
            "/fapi/v1/order",
            {
                "symbol": self.symbol,
                "side": side,
                "type": "MARKET",
                "quantity": self._decimal_string(quantity),
                "newOrderRespType": "RESULT",
            },
            signed=True,
        )
        executed_price = float(order.get("avgPrice", 0.0) or 0.0)
        if not math.isfinite(executed_price) or executed_price <= 0:
            executed_price = price

        try:
            if stop_loss_pct:
                stop_price = executed_price * (
                    1.0 - stop_loss_pct if is_buy else 1.0 + stop_loss_pct
                )
                self._place_protective_order(
                    exit_side,
                    "STOP_MARKET",
                    self._trigger_price(stop_price),
                )
            if take_profit_pct:
                take_profit_price = executed_price * (
                    1.0 + take_profit_pct if is_buy else 1.0 - take_profit_pct
                )
                self._place_protective_order(
                    exit_side,
                    "TAKE_PROFIT_MARKET",
                    self._trigger_price(take_profit_price),
                )
        except Exception:
            self.logger.exception(
                "Protective order creation failed; closing the new Binance position"
            )
            self.close_position()
            raise
        return order

    def close_position(self, size=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported Binance close arguments: {sorted(kwargs)}")
        position = self._position()
        if position is None:
            return None
        position_quantity = float(position["positionAmt"])
        close_quantity = abs(position_quantity) if size is None else min(
            abs(position_quantity),
            abs(float(size)),
        )
        quantity = self._floor_to_step(close_quantity, self.quantity_step)
        if quantity < self.minimum_quantity:
            raise ValueError("Binance close quantity is below the symbol minimum")
        result = self._request(
            "POST",
            "/fapi/v1/order",
            {
                "symbol": self.symbol,
                "side": "SELL" if position_quantity > 0 else "BUY",
                "type": "MARKET",
                "quantity": self._decimal_string(quantity),
                "reduceOnly": "true",
                "newOrderRespType": "RESULT",
            },
            signed=True,
        )
        self._cancel_protective_orders()
        return result

    def shutdown(self):
        self.session.close()
