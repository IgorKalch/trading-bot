"""Persistence of live bot state so a restart never loses track of positions.

Stores, per MT5 ticket: the originating signal, initial stop (needed for
R-based trailing) and lifecycle fields. On restart the runner reconciles this
store against mt5.positions_get(magic=...).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from tradingbot.core.fsutil import atomic_write_text
from tradingbot.core.models import EntrySignal, ManagedPosition, Side, SignalKind

log = logging.getLogger(__name__)


class PositionStore:
    def __init__(self, state_dir: str | Path):
        self._file = Path(state_dir) / "positions.json"

    def save(self, positions: list[ManagedPosition]) -> None:
        data = [self._encode(p) for p in positions if not p.closed]
        atomic_write_text(self._file, json.dumps(data, indent=2, default=str))

    def load(self) -> list[ManagedPosition]:
        if not self._file.exists():
            return []
        try:
            raw = json.loads(self._file.read_text(encoding="utf-8"))
            return [self._decode(item) for item in raw]
        except (OSError, ValueError, KeyError) as exc:
            log.error("Position store unreadable (%s) — starting with empty local state", exc)
            return []

    @staticmethod
    def _encode(p: ManagedPosition) -> dict:
        d = asdict(p)
        d["signal"]["side"] = p.signal.side.value
        d["signal"]["kind"] = p.signal.kind.value
        return d

    @staticmethod
    def _decode(d: dict) -> ManagedPosition:
        s = d["signal"]
        signal = EntrySignal(
            kind=SignalKind(s["kind"]),
            side=Side(s["side"]),
            time=datetime.fromisoformat(s["time"]),
            entry_ref=s["entry_ref"],
            stop_loss=s["stop_loss"],
            take_profit=s.get("take_profit"),
            tp_rr=s.get("tp_rr"),
            risk_points=s["risk_points"],
            reason=s.get("reason", ""),
            context=s.get("context", {}),
        )
        return ManagedPosition(
            signal=signal,
            entry_price=d["entry_price"],
            volume=d["volume"],
            opened_at=datetime.fromisoformat(d["opened_at"]),
            initial_stop=d["initial_stop"],
            stop_loss=d["stop_loss"],
            take_profit=d.get("take_profit"),
            ticket=d.get("ticket"),
            trail_steps_done=d.get("trail_steps_done", 0),
        )


class DayStore:
    """Remembers which trading day was already initialised (survives restart)."""

    def __init__(self, state_dir: str | Path):
        self._file = Path(state_dir) / "day.json"

    def last_day(self) -> date | None:
        if not self._file.exists():
            return None
        try:
            return date.fromisoformat(json.loads(self._file.read_text(encoding="utf-8"))["day"])
        except (OSError, ValueError, KeyError):
            return None

    def set_day(self, day: date) -> None:
        atomic_write_text(self._file, json.dumps({"day": day.isoformat()}))
