"""
Restartable Martingale

Core idea: split the capital into two isolated sub-accounts, [trade account] + [reserve account].
  - the trade account runs the martingale (no hard stop, averages down, takes small profits often) and earns in normal regimes;
  - its unrealized profit is periodically "swept" into the reserve account (isolated, never traded, never eaten by a blow-up);
  - on a tail event the trade account is blown up (wiped out / account level stop) -> it is allowed to die;
  - after a death, trading pauses for a while (a week by default), then the reserve funds a restart of the trade account;
  - when the reserve can no longer fund a restart, both accounts are wiped out -> the whole strategy exits.

Expected edge: E[profit swept before death] > E[loss of one death + restart cost]

This file only makes decisions (strategy layer) and contains no backtrader API;
execution is left to the VenueBase implementations (backtest: MartingaleBtVenue / live: BybitVenue).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from datetime import datetime, timedelta
import logging
import math

import numpy as np

from trade.core.venue_base import VenueBase
from trade.core.strategy_base import StrategyBase
from trade.core.protocol import (
    TradeIntent,
    Observation,
    PositionDir,
    ActionType,
    Signal,
)


# ============================================================
# Data structures
# ============================================================

class AccountPhase(Enum):
    """Life cycle state of the trade account"""
    RUNNING = "running"     # trading normally
    PAUSED = "paused"       # just died, cooling off (a week by default)
    DEAD = "dead"           # the reserve cannot fund a restart either, definitely over


class ExitReason(Enum):
    TAKE_PROFIT = "tp"          # take profit on the average price
    CYCLE_STOP = "cycle_sl"     # stop loss of a single martingale cycle
    DEATH = "death"             # account level wipe-out
    TIMEOUT = "timeout"         # holding time exceeded
    SHUTDOWN = "shutdown"       # backtest / strategy ended


@dataclass
class DeathRecord:
    """Record of one death (trade account wiped out)"""
    index: int
    time: datetime
    trade_base: float          # starting capital of this life
    equity_at_death: float     # trade account equity at the moment of death
    loss: float                # money lost during this life
    swept: float               # profit already swept to the reserve before this death
    bars_alive: int
    layers: int


@dataclass
class CycleStat:
    """Record of one martingale cycle (from open to close)"""
    dir: int
    layers: int
    bars: int
    pnl: float
    reason: str


@dataclass(frozen=True)
class MartingaleStrategyConfig:
    """Static parameters owned by the restartable martingale decision logic."""

    reserve_pct: float = 0.7
    restart_capital_pct: float = 0.3
    min_restart_capital_pct: float = 0.05
    restart_cost_pct: float = 0.0
    pause_days: int = 7
    sweep_trigger_pct: float = 0.10
    compound_pct: float = 0.0
    sweep_min_interval_days: int = 0
    base_order_pct: float = 0.02
    max_safety_orders: int = 8
    price_deviation_pct: float = 0.01
    step_mult: float = 1.2
    volume_mult: float = 1.6
    tp_pct: float = 0.01
    atr_grid_mult: Optional[float] = None
    atr_tp_mult: Optional[float] = None
    max_hold_bars: Optional[int] = None
    margin_usage_cap_pct: float = 0.9
    death_equity_pct: float = 0.2
    cycle_stop_pct: Optional[float] = None
    entry_mode: str = "signal"
    prob_thresh: Optional[float] = None
    allow_long: bool = True
    allow_short: bool = True


# ============================================================
# RestartableMartingaleStrategy
# ============================================================

class RestartableMartingaleStrategy(StrategyBase):

    def __init__(
        self,
        venue: VenueBase,
        init_equity: float,
        config: MartingaleStrategyConfig,
        leverage: float = 1.0,
    ):
        self.logger = logging.getLogger("trade")
        super().__init__(venue)
        self.config = config
        self.init_equity = init_equity
        self.min_restart_capital = init_equity * config.min_restart_capital_pct
        self.leverage = max(1.0, float(leverage))

        # ---- the two accounts ----
        self.reserve = init_equity * self.config.reserve_pct  # reserve account (isolated capital)
        self.trade_base = init_equity - self.reserve     # starting capital of the current trade account
        self.trade_equity = self.trade_base              # current trade account equity (unrealized pnl included)

        # ---- life cycle ----
        self.phase = AccountPhase.RUNNING
        self.resume_time: Optional[datetime] = None
        self.bar_index = 0
        self.life_start_bar = 0

        # ---- state of the current martingale cycle ----
        self.cycle_dir = PositionDir.FLAT
        self.cycle_layers = 0            # filled layers (base order included)
        self.cycle_qty = 0.0
        self.cycle_cost = 0.0            # accumulated notional cost, used for the average price
        self.cycle_avg_price = 0.0
        self.cycle_last_entry_price = 0.0
        self.cycle_start_equity = 0.0
        self.cycle_bars = 0
        self.pending_layer_qty = 0.0     # order sent on this bar (confirmed by the fill callback on the next one)

        # ---- statistics ----
        self.deaths: List[DeathRecord] = []
        self.cycles: List[CycleStat] = []
        self.total_swept = 0.0           # profit swept to the reserve so far
        self.swept_this_life = 0.0
        self.restart_count = 0
        self.total_restart_cost = 0.0
        self.last_sweep_time: Optional[datetime] = None
        self.max_layers_seen = 0
        self.paused_bars = 0
        self.equity_curve: List[float] = []   # total equity = reserve + trade account

        self.logger.info(
            f"🎲 [MARTINGALE INIT] Total capital={init_equity:.2f} | Trading account={self.trade_base:.2f} "
            f"| Reserve account={self.reserve:.2f} | Max safety orders={self.config.max_safety_orders}"
        )

    # ------------------------------------------------------------------
    # Account / position bookkeeping
    # ------------------------------------------------------------------
    def _sync(self, state: Observation):
        """Derive the trade account equity from the broker total (the reserve is virtually isolated and never traded)"""
        self.trade_equity = state.account.equity - self.reserve

    @property
    def total_equity(self) -> float:
        return self.reserve + self.trade_equity

    def on_fill(self, price: float, size: float, is_buy: bool):
        """Called back by the venue after a fill, maintains the average price and the layer count"""
        qty = abs(size)
        if qty <= 0:
            return
        direction = PositionDir.POSITIVE if is_buy else PositionDir.NEGATIVE
        if self.cycle_dir == PositionDir.FLAT:
            self.cycle_dir = direction
            self.cycle_qty = 0.0
            self.cycle_cost = 0.0
            self.cycle_layers = 0
        if direction != self.cycle_dir:
            # A closing fill; the cycle state is handled by _reset_cycle
            return
        self.cycle_qty += qty
        self.cycle_cost += qty * price
        self.cycle_avg_price = self.cycle_cost / max(self.cycle_qty, 1e-12)
        self.cycle_last_entry_price = price
        self.cycle_layers += 1
        self.max_layers_seen = max(self.max_layers_seen, self.cycle_layers)
        self.logger.debug(
            f"🧱 [LAYER {self.cycle_layers}] price={price:.4f} qty={qty:.4f} "
            f"avg={self.cycle_avg_price:.4f} total_qty={self.cycle_qty:.4f}"
        )

    def _reset_cycle(self, reason: ExitReason):
        if self.cycle_layers > 0:
            pnl = self.trade_equity - self.cycle_start_equity
            self.cycles.append(
                CycleStat(
                    dir=int(self.cycle_dir),
                    layers=self.cycle_layers,
                    bars=self.cycle_bars,
                    pnl=pnl,
                    reason=reason.value,
                )
            )
        self.cycle_dir = PositionDir.FLAT
        self.cycle_layers = 0
        self.cycle_qty = 0.0
        self.cycle_cost = 0.0
        self.cycle_avg_price = 0.0
        self.cycle_last_entry_price = 0.0
        self.cycle_bars = 0

    # ------------------------------------------------------------------
    # Grid parameters
    # ------------------------------------------------------------------
    def _deviation(self, state: Observation, layer: int) -> float:
        base = self.config.price_deviation_pct
        if self.config.atr_grid_mult is not None and state.market.atr_pct and state.market.atr_pct > 0:
            base = state.market.atr_pct * self.config.atr_grid_mult
        return base * (self.config.step_mult ** max(0, layer - 1))

    def _tp_pct(self, state: Observation) -> float:
        if self.config.atr_tp_mult is not None and state.market.atr_pct and state.market.atr_pct > 0:
            return max(1e-4, state.market.atr_pct * self.config.atr_tp_mult)
        return self.config.tp_pct

    def _next_layer_qty(self, state: Observation) -> float:
        """Size of layer (cycle_layers+1), capped by the margin limit"""
        layer = self.cycle_layers  # 0 => base order
        base_notional = self.trade_base * self.config.base_order_pct
        notional = base_notional * (self.config.volume_mult ** layer)
        qty = notional / max(state.market.price, 1e-12)

        used_margin = self.cycle_qty * state.market.price / self.leverage
        free_margin = self.trade_equity * self.config.margin_usage_cap_pct - used_margin
        if free_margin <= 0:
            return 0.0
        max_qty = free_margin * self.leverage / max(state.market.price, 1e-12)
        if qty > max_qty:
            self.logger.debug(
                f"🛡️ [MARGIN CUT] layer={layer + 1} requested {qty:.4f} → {max_qty:.4f} "
                f"(free_margin={free_margin:.2f})"
            )
            qty = max_qty
        return max(0.0, qty)

    # ------------------------------------------------------------------
    # Death / pause / restart
    # ------------------------------------------------------------------
    def _is_dead(self) -> bool:
        return self.trade_equity <= self.trade_base * self.config.death_equity_pct

    def _kill(self, state: Observation):
        """Trade account wiped out: flatten + record + enter the cooling off period"""
        loss = self.trade_base - self.trade_equity
        rec = DeathRecord(
            index=len(self.deaths) + 1,
            time=state.current_time,
            trade_base=self.trade_base,
            equity_at_death=self.trade_equity,
            loss=loss,
            swept=self.swept_this_life,
            bars_alive=self.bar_index - self.life_start_bar,
            layers=self.cycle_layers,
        )
        self.deaths.append(rec)
        self.logger.warning(
            f"💀 [DEATH #{rec.index}] {state.current_time} | Capital={self.trade_base:.2f} "
            f"→ Equity={self.trade_equity:.2f} | Loss={loss:.2f} | Profit swept this life={self.swept_this_life:.2f} "
            f"| Layers={self.cycle_layers} | Survived={rec.bars_alive} bars | Reserve={self.reserve:.2f}"
        )
        self._reset_cycle(ExitReason.DEATH)
        self.phase = AccountPhase.PAUSED
        self.resume_time = state.current_time + timedelta(days=self.config.pause_days)
        self.swept_this_life = 0.0

    def _try_restart(self, state: Observation) -> bool:
        """After the cooling off period, fund a restart from the reserve; returns whether it succeeded"""
        draw = self.reserve * self.config.restart_capital_pct
        if draw > self.reserve:
            draw = self.reserve
        cost = draw * self.config.restart_cost_pct
        new_base = self.trade_equity + draw - cost   # trade_equity may be a remainder / negative

        if draw < self.min_restart_capital or new_base <= 0:
            self.phase = AccountPhase.DEAD
            self.logger.error(
                f"🪦 [GAME OVER] {state.current_time} | Reserve={self.reserve:.2f} cannot fund another start "
                f"(required >= {self.min_restart_capital:.2f}) | Deaths={len(self.deaths)} "
                f"| Total swept profit={self.total_swept:.2f}"
            )
            return False

        self.reserve -= draw
        if cost > 0:
            # The restart friction cost is real money leaving, let the execution layer debit the account
            withdraw = getattr(self.venue, "withdraw_cash", None)
            if callable(withdraw):
                withdraw(cost)
        self.total_restart_cost += cost
        self.trade_base = new_base
        self.restart_count += 1
        self.life_start_bar = self.bar_index
        self.phase = AccountPhase.RUNNING
        self.logger.warning(
            f"🔁 [RESTART #{self.restart_count}] {state.current_time} | Allocation={draw:.2f} "
            f"(cost={cost:.2f}) | New trading capital={self.trade_base:.2f} | Remaining reserve={self.reserve:.2f}"
        )
        return True

    # ------------------------------------------------------------------
    # Profit sweep (trade account -> reserve account)
    # ------------------------------------------------------------------
    def _try_sweep(self, state: Observation):
        """Only sweep while flat, so unrealized profit is never moved as if it were realized"""
        if self.cycle_dir != PositionDir.FLAT or self.cycle_qty > 0:
            return
        if self.trade_equity <= self.trade_base * (1.0 + self.config.sweep_trigger_pct):
            return
        if self.config.sweep_min_interval_days > 0 and self.last_sweep_time is not None:
            if (state.current_time - self.last_sweep_time).days < self.config.sweep_min_interval_days:
                return

        profit = self.trade_equity - self.trade_base
        keep = profit * self.config.compound_pct
        move = profit - keep
        if move <= 0:
            return

        self.reserve += move
        self.trade_base += keep
        self.total_swept += move
        self.swept_this_life += move
        self.last_sweep_time = state.current_time
        self.logger.info(
            f"💰 [SWEEP] {state.current_time} | Swept={move:.2f} → Reserve={self.reserve:.2f} "
            f"| Trading capital={self.trade_base:.2f} | Total swept={self.total_swept:.2f}"
        )

    # ------------------------------------------------------------------
    # Entry direction
    # ------------------------------------------------------------------
    def _entry_dir(self, state: Observation) -> PositionDir:
        if self.config.entry_mode == "long":
            return PositionDir.POSITIVE if self.config.allow_long else PositionDir.FLAT
        if self.config.entry_mode == "short":
            return PositionDir.NEGATIVE if self.config.allow_short else PositionDir.FLAT
        if self.config.entry_mode == "reversion":
            # Counter trend: buy the dip, sell the rip (the classic mean reverting martingale)
            prev = getattr(self, "_prev_price", None)
            self._prev_price = state.market.price
            if prev is None:
                return PositionDir.FLAT
            if state.market.price < prev and self.config.allow_long:
                return PositionDir.POSITIVE
            if state.market.price > prev and self.config.allow_short:
                return PositionDir.NEGATIVE
            return PositionDir.FLAT

        # signal mode: follow the ML prediction
        signal = state.market.signal
        if signal == Signal.INVALID:
            return PositionDir.FLAT
        if self.config.prob_thresh is not None and state.market.pred_prob < self.config.prob_thresh:
            return PositionDir.FLAT
        if signal == Signal.POSITIVE and self.config.allow_long:
            return PositionDir.POSITIVE
        if signal == Signal.NEGATIVE and self.config.allow_short:
            return PositionDir.NEGATIVE
        return PositionDir.FLAT

    # ------------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------------
    def process(self, state: Observation) -> TradeIntent:
        self.bar_index += 1
        self._sync(state)
        self.equity_curve.append(self.total_equity)

        if self.cycle_qty > 0:
            self.cycle_bars += 1

        # ---------- 0. definitely over ----------
        if self.phase == AccountPhase.DEAD:
            return self._emit(TradeIntent(ActionType.NOOP, reason="dead"), state)

        # ---------- 1. cooling off ----------
        if self.phase == AccountPhase.PAUSED:
            self.paused_bars += 1
            if state.position.dir != PositionDir.FLAT:
                # Flatten whatever position is left
                return self._emit(TradeIntent(ActionType.CLOSE, reason="paused_flatten"), state)
            if self.resume_time is not None and state.current_time >= self.resume_time:
                if not self._try_restart(state):
                    # Reserve exhausted -> tell the execution layer to end the backtest
                    self._shutdown()
            return self._emit(TradeIntent(ActionType.NOOP, reason="paused"), state)

        # ---------- 2. death check (before anything else) ----------
        if self._is_dead():
            self._kill(state)
            if state.position.dir != PositionDir.FLAT:
                return self._emit(TradeIntent(ActionType.CLOSE, reason="death"), state)
            return self._emit(TradeIntent(ActionType.NOOP, reason="death"), state)

        # ---------- 3. in position: take profit / stop loss / safety order ----------
        if state.position.dir != PositionDir.FLAT and self.cycle_qty > 0:
            tp = self._tp_pct(state)
            if self.cycle_dir == PositionDir.POSITIVE:
                hit_tp = state.market.price >= self.cycle_avg_price * (1.0 + tp)
                adverse = (self.cycle_last_entry_price - state.market.price) / max(self.cycle_last_entry_price, 1e-12)
                unreal = (state.market.price - self.cycle_avg_price) * self.cycle_qty
            else:
                hit_tp = state.market.price <= self.cycle_avg_price * (1.0 - tp)
                adverse = (state.market.price - self.cycle_last_entry_price) / max(self.cycle_last_entry_price, 1e-12)
                unreal = (self.cycle_avg_price - state.market.price) * self.cycle_qty

            if hit_tp:
                self.logger.debug(
                    f"🎯 [TP] avg={self.cycle_avg_price:.4f} price={state.market.price:.4f} "
                    f"layers={self.cycle_layers} bars={self.cycle_bars}"
                )
                self._reset_cycle(ExitReason.TAKE_PROFIT)
                return self._emit(TradeIntent(ActionType.CLOSE, reason="tp"), state)

            if self.config.cycle_stop_pct is not None and unreal <= -self.trade_base * self.config.cycle_stop_pct:
                self.logger.warning(
                    f"🛑 [CYCLE STOP] Unrealized loss={unreal:.2f} >= capital*{self.config.cycle_stop_pct:.0%} "
                    f"| layers={self.cycle_layers}"
                )
                self._reset_cycle(ExitReason.CYCLE_STOP)
                return self._emit(TradeIntent(ActionType.CLOSE, reason="cycle_stop"), state)

            if self.config.max_hold_bars is not None and self.cycle_bars >= self.config.max_hold_bars:
                self.logger.debug(f"⌛ [TIMEOUT] Held for {self.cycle_bars} bars; forcing exit")
                self._reset_cycle(ExitReason.TIMEOUT)
                return self._emit(TradeIntent(ActionType.CLOSE, reason="timeout"), state)

            # Safety order
            if self.cycle_layers <= self.config.max_safety_orders:
                need = self._deviation(state, self.cycle_layers)
                if adverse >= need:
                    qty = self._next_layer_qty(state)
                    if qty > 0:
                        return self._emit(
                            TradeIntent(
                                ActionType.PYRAMID,
                                target_dir=self.cycle_dir,
                                order_qty=qty,
                                layer=self.cycle_layers + 1,
                                reason=f"safety_order_{self.cycle_layers + 1}",
                            ),
                            state,
                        )
                    self.logger.debug("⚠️ [NO MARGIN] Cannot add another safety order; holding current exposure")
            return self._emit(TradeIntent(ActionType.NOOP, reason="hold"), state)

        # ---------- 4. flat: sweep the profit first, then start a new cycle ----------
        if state.position.dir == PositionDir.FLAT and self.cycle_qty > 0:
            # The position was closed on the venue side (e.g. liquidation), resync the state
            self._reset_cycle(ExitReason.SHUTDOWN)

        self._try_sweep(state)

        target_dir = self._entry_dir(state)
        if target_dir == PositionDir.FLAT:
            return self._emit(TradeIntent(ActionType.NOOP, reason="no_signal"), state)

        # Do not open a new cycle right before the close (when the data carries bars_to_close)
        if getattr(state.market, "bars_to_close", None) is not None and state.market.bars_to_close <= 2:
            return self._emit(TradeIntent(ActionType.NOOP, reason="near_close"), state)

        self.cycle_dir = target_dir
        qty = self._next_layer_qty(state)
        self.cycle_dir = PositionDir.FLAT  # the real direction is confirmed in the fill callback
        if qty <= 0:
            return self._emit(TradeIntent(ActionType.NOOP, reason="no_margin"), state)

        self.cycle_start_equity = self.trade_equity
        return self._emit(
            TradeIntent(
                ActionType.OPEN,
                target_dir=target_dir,
                order_qty=qty,
                layer=1,
                reason="base_order",
            ),
            state,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _emit(self, action: TradeIntent, state: Observation) -> TradeIntent:
        self.execute_action(action)
        return action

    def execute_action(self, action: TradeIntent):
        if action.action == ActionType.NOOP or action.action == ActionType.NOOP:
            return
        if action.action == ActionType.CLOSE:
            self.venue.close_position()
            return
        if action.action in (ActionType.OPEN, ActionType.PYRAMID):
            is_buy = action.target_dir == PositionDir.POSITIVE
            self.venue.submit_order(action.order_qty, is_buy=is_buy)

    def _shutdown(self):
        """Both accounts wiped out -> ask the execution layer to stop"""
        stop = getattr(self.venue, "request_halt", None)
        if callable(stop):
            stop()

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    def finalize(self):
        self.logger.info("\n" + "🎲" * 5 + " Restartable Martingale Summary " + "🎲" * 5)
        self.logger.info(
            f"[ACCOUNT] Reserve={self.reserve:.2f} | Trading equity={self.trade_equity:.2f} "
            f"| Total={self.total_equity:.2f} | Initial={self.init_equity:.2f} "
            f"| Total return={(self.total_equity / max(self.init_equity, 1e-9) - 1):.2%}"
        )
        self.logger.info(
            f"[LIFECYCLE] Deaths={len(self.deaths)} | Restarts={self.restart_count} "
            f"| Paused={self.paused_bars} bars | Final state={self.phase.value}"
        )
        self.logger.info(
            f"[PROFIT SWEEP] Total swept={self.total_swept:.2f} | Restart cost={self.total_restart_cost:.2f}"
        )

        if self.deaths:
            losses = np.array([d.loss for d in self.deaths])
            swept = np.array([d.swept for d in self.deaths])
            alive = np.array([d.bars_alive for d in self.deaths])
            self.logger.info(
                f"[DEATH DISTRIBUTION] Average loss={losses.mean():.2f} | Maximum={losses.max():.2f} "
                f"| Average swept before death={swept.mean():.2f} | Average survival={alive.mean():.0f} bars "
                f"| Minimum survival={alive.min()} bars"
            )
            edge = swept.mean() - losses.mean()
            verdict = "✅ Positive expectancy" if edge > 0 else "❌ Negative expectancy"
            self.logger.info(
                f"[CORE CHECK] E[swept before death] - E[death loss] = {swept.mean():.2f} - {losses.mean():.2f} "
                f"= {edge:.2f} → {verdict}"
            )
        else:
            self.logger.info("[DEATH DISTRIBUTION] The trading account did not die during the backtest.")

        if self.cycles:
            pnls = np.array([c.pnl for c in self.cycles])
            layers = np.array([c.layers for c in self.cycles])
            bars = np.array([c.bars for c in self.cycles])
            wins = int((pnls > 0).sum())
            self.logger.info(
                f"[MARTINGALE CYCLES] n={len(self.cycles)} | Win rate={wins / len(self.cycles):.1%} "
                f"| Average PnL={pnls.mean():.2f} | Worst={pnls.min():.2f} "
                f"| Average layers={layers.mean():.2f} | Deepest={self.max_layers_seen} "
                f"| Average holding period={bars.mean():.1f} bars"
            )
            reasons = {}
            for c in self.cycles:
                reasons[c.reason] = reasons.get(c.reason, 0) + 1
            self.logger.info(f"[EXIT REASONS] {reasons}")
        self.logger.info("=" * 60 + "\n")

    def report(self) -> dict:
        """Structured statistics for the layer above (backtest report / frontend)"""
        losses = [d.loss for d in self.deaths]
        swept = [d.swept for d in self.deaths]
        return {
            "final_reserve": self.reserve,
            "final_trade_equity": self.trade_equity,
            "final_total_equity": self.total_equity,
            "init_equity": self.init_equity,
            "total_return": self.total_equity / max(self.init_equity, 1e-9) - 1,
            "death_count": len(self.deaths),
            "restart_count": self.restart_count,
            "phase": self.phase.value,
            "total_swept": self.total_swept,
            "total_restart_cost": self.total_restart_cost,
            "avg_death_loss": float(np.mean(losses)) if losses else 0.0,
            "max_death_loss": float(np.max(losses)) if losses else 0.0,
            "avg_swept_before_death": float(np.mean(swept)) if swept else 0.0,
            "expectancy_edge": (float(np.mean(swept)) - float(np.mean(losses))) if losses else None,
            "cycle_count": len(self.cycles),
            "max_layers": self.max_layers_seen,
            "paused_bars": self.paused_bars,
            "deaths": [
                {
                    "index": d.index,
                    "time": d.time.isoformat() if isinstance(d.time, datetime) else str(d.time),
                    "trade_base": d.trade_base,
                    "equity_at_death": d.equity_at_death,
                    "loss": d.loss,
                    "swept": d.swept,
                    "bars_alive": d.bars_alive,
                    "layers": d.layers,
                }
                for d in self.deaths
            ],
        }
