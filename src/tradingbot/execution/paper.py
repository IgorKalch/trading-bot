"""Paper executor for signal-only mode.

No orders reach the broker. Positions are simulated against live ticks so
Telegram receives the full lifecycle (entry, SL moves, exit) as if trading.
"""

from __future__ import annotations

import itertools
import logging

from tradingbot.core.models import EntrySignal, ManagedPosition, Side
from tradingbot.data.mt5_client import Mt5Client

log = logging.getLogger(__name__)


class PaperExecutor:
    def __init__(self, client: Mt5Client, symbol: str):
        self.client = client
        self.symbol = symbol
        self._ids = itertools.count(1)

    def open_market(self, signal: EntrySignal, volume: float) -> ManagedPosition | None:
        tick = self.client.current_tick(self.symbol)
        price = tick.ask if signal.side is Side.LONG else tick.bid
        pos = ManagedPosition(
            signal=signal,
            entry_price=price,
            volume=volume,
            opened_at=tick.time,
            initial_stop=signal.stop_loss,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            ticket=-next(self._ids),  # negative = paper
        )
        log.info("[PAPER] opened %s @ %.2f", signal.side.value, price)
        return pos

    def modify_stop(self, pos: ManagedPosition, new_stop: float) -> bool:
        pos.stop_loss = new_stop
        log.info("[PAPER] SL -> %.2f", new_stop)
        return True

    def close(self, pos: ManagedPosition, reason: str) -> bool:
        tick = self.client.current_tick(self.symbol)
        pos.closed = True
        pos.close_price = tick.bid if pos.side is Side.LONG else tick.ask
        pos.closed_at = tick.time
        pos.close_reason = reason
        log.info("[PAPER] closed @ %.2f (%s)", pos.close_price, reason)
        return True

    def list_broker_positions(self) -> list[ManagedPosition]:
        return []  # paper positions live only in the bot's own state file

    def sync(self, pos: ManagedPosition) -> None:
        """Simulate SL/TP hits using the current tick."""
        if pos.closed:
            return
        tick = self.client.current_tick(self.symbol)
        price = tick.bid if pos.side is Side.LONG else tick.ask
        sign = pos.side.sign
        if (price - pos.stop_loss) * sign <= 0:
            pos.closed = True
            pos.close_price = pos.stop_loss
            pos.closed_at = tick.time
            pos.close_reason = "stop loss hit (paper)"
        elif pos.take_profit is not None and (price - pos.take_profit) * sign >= 0:
            pos.closed = True
            pos.close_price = pos.take_profit
            pos.closed_at = tick.time
            pos.close_reason = "take profit hit (paper)"
