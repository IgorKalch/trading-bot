"""Domain models shared by live trading and backtesting.

All datetimes in the system are timezone-aware UTC. Conversion from broker
server time happens once, at the data boundary (see core.clock).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1

    @property
    def opposite(self) -> Side:
        return Side.SHORT if self is Side.LONG else Side.LONG


class SignalKind(StrEnum):
    FIRST = "first"  # breakout in gap direction
    SECOND = "second"  # reversal breakout of the opposite OR boundary


class TpMode(StrEnum):
    FIXED_RR = "fixed_rr"  # static take profit at N * R
    TRAILING = "trailing"  # no static TP, stop trails in R-steps


@dataclass(frozen=True, slots=True)
class Bar:
    """One completed OHLC bar. `time` is the bar OPEN time, tz-aware UTC."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int = 0
    spread_points: int = 0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass(frozen=True, slots=True)
class Tick:
    """Current price snapshot used for position management."""

    time: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True, slots=True)
class OpeningRange:
    day: date
    high: float
    low: float
    start: datetime
    end: datetime

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0


@dataclass(frozen=True, slots=True)
class GapInfo:
    """Overnight gap: session open price vs previous session close price."""

    prev_close: float
    session_open: float

    @property
    def size(self) -> float:
        return self.session_open - self.prev_close

    @property
    def abs_size(self) -> float:
        return abs(self.size)

    @property
    def side(self) -> Side:
        """Direction the gap points to (up-gap -> LONG bias)."""
        return Side.LONG if self.size >= 0 else Side.SHORT


# --------------------------------------------------------------------------
# Strategy output events. The strategy is pure logic: it emits intents that
# the live runner or the backtest engine interpret.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntrySignal:
    kind: SignalKind
    side: Side
    time: datetime
    entry_ref: float  # close of the confirmation bar (reference, fill may differ)
    stop_loss: float
    take_profit: float | None  # reference TP from entry_ref; None in trailing mode
    tp_rr: float | None  # RR multiple; executor recomputes TP from actual fill
    risk_points: float  # |entry_ref - stop_loss|
    reason: str
    context: dict[str, float | str] = field(default_factory=dict)

    def take_profit_for_fill(self, fill_price: float) -> float | None:
        """TP recomputed from the actual fill so the RR is exact."""
        if self.tp_rr is None:
            return None
        risk = abs(fill_price - self.stop_loss)
        return fill_price + self.side.sign * self.tp_rr * risk


@dataclass(frozen=True, slots=True)
class SkipEvent:
    """A potential trade was skipped by a filter — logged and sent to Telegram."""

    time: datetime
    rule: str
    detail: str


@dataclass(frozen=True, slots=True)
class InfoEvent:
    """Notable strategy milestone (OR formed, breakout seen, day done, ...)."""

    time: datetime
    message: str


StrategyEvent = EntrySignal | SkipEvent | InfoEvent


# --------------------------------------------------------------------------
# Position lifecycle
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ManagedPosition:
    """Broker-agnostic state of a position the bot manages.

    Used identically by live trading (backed by an MT5 position ticket)
    and by the backtest simulator (ticket is None).
    """

    signal: EntrySignal
    entry_price: float  # actual fill price
    volume: float
    opened_at: datetime
    initial_stop: float
    stop_loss: float
    take_profit: float | None
    ticket: int | None = None
    trail_steps_done: int = 0
    closed: bool = False
    close_price: float | None = None
    closed_at: datetime | None = None
    close_reason: str | None = None

    @property
    def side(self) -> Side:
        return self.signal.side

    @property
    def risk_points(self) -> float:
        """1R in price points, from actual fill to the initial stop."""
        return abs(self.entry_price - self.initial_stop)

    def profit_points(self, price: float) -> float:
        return (price - self.entry_price) * self.side.sign

    def profit_r(self, price: float) -> float:
        if self.risk_points <= 0:
            return 0.0
        return self.profit_points(price) / self.risk_points

    @property
    def result_r(self) -> float | None:
        if not self.closed or self.close_price is None:
            return None
        return self.profit_r(self.close_price)


# --------------------------------------------------------------------------
# Trade-management actions (emitted by management.trade_manager)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModifyStop:
    new_stop: float
    reason: str


@dataclass(frozen=True, slots=True)
class CloseNow:
    reason: str


ManageAction = ModifyStop | CloseNow
