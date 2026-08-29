"""Execution abstraction: live MT5 or paper (signal-only mode).

The live runner talks only to this interface, so switching between
real trading and Telegram-signals-only is a config change.
"""

from __future__ import annotations

from typing import Protocol

from tradingbot.core.models import EntrySignal, ManagedPosition


class ExecutionClient(Protocol):
    def open_market(self, signal: EntrySignal, volume: float) -> ManagedPosition | None:
        """Open a market position with SL/TP from the signal. None on failure."""
        ...

    def modify_stop(self, pos: ManagedPosition, new_stop: float) -> bool:
        """Move the stop loss. Returns True on success (pos updated in place)."""
        ...

    def close(self, pos: ManagedPosition, reason: str) -> bool:
        """Close at market. Returns True on success (pos updated in place)."""
        ...

    def sync(self, pos: ManagedPosition) -> None:
        """Refresh position state from the venue (detect SL/TP hits)."""
        ...

    def list_broker_positions(self) -> list[ManagedPosition]:
        """Open positions at the venue belonging to this bot (orphan adoption)."""
        ...
