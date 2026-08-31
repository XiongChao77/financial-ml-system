"""Binance USD-M futures implementation of the shared live venue protocol."""

from __future__ import annotations

import hashlib
import hmac
import itertools
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Iterable
from urllib.parse import urlencode

import requests

from trade.core.protocol import OrderType, PositionDir, PositionView
from trade.core.venue_base import VenueBase


class BinanceVenue(VenueBase):
    """Binance USD-M futures venue with fail-closed bracket creation."""

    BASE_URL = "https://fapi.binance.com"
    API_KEY_FILES = ("hmac_api_key", "binance_api_key")
    API_SECRET_FILES = ("hmac_secret", "binance_api_secret")

    def __init__(
        self,
        key_path: str,
        symbol: str,
        magic: str | None = None,
        *,
        logger: logging.Logger | None = None,
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ):
        self.symbol = str(symbol).upper()
        self.magic = str(magic or "financial-ml-system")
        self._order_id_sequence = itertools.count()
        self.logger = logger or logging.getLogger("trade.binance")
        self.timeout = float(timeout)
        self.session = session or requests.Session()
        try:
            self.api_key = self._load_first(key_path, self.API_KEY_FILES)
            self.api_secret = self._load_first(key_path, self.API_SECRET_FILES)
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})
            self.quantity_step, self.minimum_quantity, self.price_tick = self._load_filters()
            self.hedge_mode = self._load_hedge_mode()
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
        raise FileNotFoundError(f"No Binance credential file found in {directory}: {tuple(candidates)}")

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
            raise RuntimeError(f"Binance returned a non-JSON response for {path}: HTTP {response.status_code}") from exc
        if not response.ok:
            raise RuntimeError(f"Binance request failed for {path}: HTTP {response.status_code}, " f"code={body.get('code')}, message={body.get('msg')}")
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

    def _load_hedge_mode(self) -> bool:
        """Return whether the account uses Hedge Mode."""
        mode = self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
        dual_side = mode.get("dualSidePosition")
        if isinstance(dual_side, bool):
            return dual_side
        normalized = str(dual_side).casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise RuntimeError("Binance returned an invalid position mode response")

    @staticmethod
    def _floor_to_step(value: float, step: Decimal) -> Decimal:
        amount = Decimal(str(value))
        return (amount / step).to_integral_value(rounding=ROUND_DOWN) * step

    @staticmethod
    def _decimal_string(value: Decimal) -> str:
        return format(value.normalize(), "f")

    def _new_client_order_id(self, action: str) -> str:
        safe_magic = re.sub(r"[^.A-Za-z0-9_:/-]", "_", self.magic)
        nonce = f"{time.time_ns():x}{next(self._order_id_sequence):x}"[-14:]
        suffix = f":{action}:{nonce}"
        return f"{safe_magic[:36 - len(suffix)]}{suffix}"

    def _position(self) -> dict[str, Any] | None:
        positions = self._request(
            "GET",
            "/fapi/v3/positionRisk",
            {"symbol": self.symbol},
            signed=True,
        )
        if not isinstance(positions, list):
            raise RuntimeError("Binance returned an invalid position response")
        active_positions = [position for position in positions if position.get("symbol") == self.symbol and float(position.get("positionAmt", 0.0)) != 0.0]
        if len(active_positions) > 1:
            raise RuntimeError("Binance account has simultaneous LONG and SHORT positions for " f"{self.symbol}; this strategy supports one active position")
        if not active_positions:
            return None

        position = active_positions[0]
        if self.hedge_mode:
            position_side = position.get("positionSide")
            quantity = float(position.get("positionAmt", 0.0))
            if position_side not in {"LONG", "SHORT"}:
                raise RuntimeError("Binance Hedge Mode position has no valid positionSide")
            if (position_side == "LONG" and quantity <= 0) or (position_side == "SHORT" and quantity >= 0):
                raise RuntimeError("Binance Hedge Mode position direction is inconsistent")
        return position

    def get_account_equity(self) -> float:
        account = self._request("GET", "/fapi/v3/account", signed=True)
        equity = float(account.get("totalMarginBalance", 0.0))
        if not math.isfinite(equity) or equity <= 0:
            raise RuntimeError("Binance returned invalid account equity")
        return equity

    def get_current_state(self) -> PositionView:
        position = self._position()
        if position is None:
            return PositionView()
        if self.hedge_mode:
            direction = PositionDir.POSITIVE if position["positionSide"] == "LONG" else PositionDir.NEGATIVE
        else:
            quantity = float(position["positionAmt"])
            direction = PositionDir.POSITIVE if quantity > 0 else PositionDir.NEGATIVE
        return PositionView(
            dir=direction,
            size=abs(float(position["positionAmt"])),
            price=float(position.get("entryPrice", 0.0)),
        )

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

    def _place_stop_market_order(
        self,
        side: str,
        trigger_price: str,
        position_side: str | None = None,
    ):
        params = {
            "algoType": "CONDITIONAL",
            "symbol": self.symbol,
            "side": side,
            "type": "STOP_MARKET",
            "triggerPrice": trigger_price,
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "clientAlgoId": self._new_client_order_id("sl"),
        }
        if position_side is not None:
            params["positionSide"] = position_side
        return self._request(
            "POST",
            "/fapi/v1/algoOrder",
            params,
            signed=True,
        )

    def _place_take_profit_limit_order(
        self,
        side: str,
        trigger_price: str,
        quantity: Decimal,
        position_side: str | None = None,
    ):
        quantity_string = self._decimal_string(quantity)
        params = {
            "algoType": "CONDITIONAL",
            "symbol": self.symbol,
            "side": side,
            "type": "TAKE_PROFIT",
            "timeInForce": "GTC",
            "quantity": quantity_string,
            "price": trigger_price,
            "triggerPrice": trigger_price,
            "workingType": "MARK_PRICE",
            "clientAlgoId": self._new_client_order_id("tp"),
        }
        if position_side is not None:
            params["positionSide"] = position_side
        else:
            params["reduceOnly"] = "true"
        return self._request(
            "POST",
            "/fapi/v1/algoOrder",
            params,
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
            raise RuntimeError(f"Refusing to open with existing Binance orders: {self.symbol}")

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
        *,
        order_type=OrderType.MARKET,
        price=None,
    ):
        order_type, price = self.normalize_order_request(order_type, price)
        if order_type == OrderType.LIMIT and (stop_loss_pct or take_profit_pct):
            raise ValueError("Binance resting limit entries with protective orders require " "fill monitoring, which is not supported")
        if self._position() is not None:
            raise RuntimeError(f"Refusing to open over an existing Binance position: {self.symbol}")
        self._assert_no_open_orders()

        quantity = self._floor_to_step(float(size), self.quantity_step)
        if quantity < self.minimum_quantity:
            raise ValueError(f"Binance order quantity {quantity} is below minimum {self.minimum_quantity}")
        side = "BUY" if is_buy else "SELL"
        exit_side = "SELL" if is_buy else "BUY"
        position_side = ("LONG" if is_buy else "SHORT") if self.hedge_mode else None
        order_params = {
            "symbol": self.symbol,
            "side": side,
            "type": "MARKET" if order_type == OrderType.MARKET else "LIMIT",
            "quantity": self._decimal_string(quantity),
            "newOrderRespType": "RESULT",
            "newClientOrderId": self._new_client_order_id("open"),
        }
        if position_side is not None:
            order_params["positionSide"] = position_side
        if order_type == OrderType.LIMIT:
            limit_price = self._floor_to_step(price, self.price_tick)
            if limit_price <= 0:
                raise ValueError("Binance limit price rounded to zero")
            order_params.update(
                price=self._decimal_string(limit_price),
                timeInForce="GTC",
            )
        order = self._request(
            "POST",
            "/fapi/v1/order",
            order_params,
            signed=True,
        )
        executed_price = None
        if stop_loss_pct or take_profit_pct:
            executed_price = float(order.get("avgPrice", 0.0) or 0.0)
            if not math.isfinite(executed_price) or executed_price <= 0:
                executed_price = self._latest_price()

        try:
            if stop_loss_pct:
                stop_price = executed_price * (1.0 - stop_loss_pct if is_buy else 1.0 + stop_loss_pct)
                self._place_stop_market_order(
                    exit_side,
                    self._trigger_price(stop_price),
                    position_side,
                )
            if take_profit_pct:
                take_profit_price = executed_price * (1.0 + take_profit_pct if is_buy else 1.0 - take_profit_pct)
                self._place_take_profit_limit_order(
                    exit_side,
                    self._trigger_price(take_profit_price),
                    quantity,
                    position_side,
                )
        except Exception:
            self.logger.exception("Protective order creation failed; closing the new Binance position")
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
        close_quantity = (
            abs(position_quantity)
            if size is None
            else min(
                abs(position_quantity),
                abs(float(size)),
            )
        )
        quantity = self._floor_to_step(close_quantity, self.quantity_step)
        if quantity < self.minimum_quantity:
            raise ValueError("Binance close quantity is below the symbol minimum")
        order_params = {
            "symbol": self.symbol,
            "side": "SELL" if position_quantity > 0 else "BUY",
            "type": "MARKET",
            "quantity": self._decimal_string(quantity),
            "newOrderRespType": "RESULT",
            "newClientOrderId": self._new_client_order_id("close"),
        }
        if self.hedge_mode:
            order_params["positionSide"] = position["positionSide"]
        else:
            order_params["reduceOnly"] = "true"
        result = self._request(
            "POST",
            "/fapi/v1/order",
            order_params,
            signed=True,
        )
        self._cancel_protective_orders()
        return result

    def shutdown(self):
        self.session.close()
