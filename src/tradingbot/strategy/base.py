"""Strategy interface.

A strategy is PURE bar-by-bar logic: it receives completed bars and emits
events (entry signals, skips, info). It performs no I/O, holds no broker
state and never sleeps — which is exactly what lets live trading and the
backtest engine share one implementation.

Day-level facts a strategy cannot compute from intraday bars alone
(previous session close, daily ATR, news calendar) are injected through
DayContext by the runner / backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from tradingbot.core.clock import SessionTimes
from tradingbot.core.models import Bar, ManagedPosition, StrategyEvent
from tradingbot.data.news import NewsCalendar


@dataclass
class DayContext:
    session: SessionTimes
    prev_session_close: float | None = None  # previous Xetra cash close
    atr_d1: float | None = None  # daily ATR for volatility filters
    or_window_avg_volume: float | None = None  # avg OR-window tick volume, for RVOL
    news: NewsCalendar = field(default_factory=NewsCalendar.empty)


class Strategy(Protocol):
    def on_day_start(self, ctx: DayContext) -> None:
        """Reset per-day state. Called before the first bar of each day."""
        ...

    def on_bar(self, bar: Bar) -> list[StrategyEvent]:
        """Process one COMPLETED bar, return zero or more events."""
        ...

    def on_position_closed(self, pos: ManagedPosition) -> None:
        """Feedback needed for e.g. one-position-at-a-time bookkeeping."""
        ...
