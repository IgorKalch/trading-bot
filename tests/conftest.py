"""Shared test helpers: synthetic bar builders for a Xetra trading day."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tradingbot.config import AppConfig
from tradingbot.core.clock import SessionSchedule
from tradingbot.core.models import Bar
from tradingbot.strategy.base import DayContext

# 2025-06-05 is a Thursday; Berlin is UTC+2 in June -> 09:00 Berlin = 07:00 UTC.
DAY = datetime(2025, 6, 5, tzinfo=UTC).date()
OPEN_UTC = datetime(2025, 6, 5, 7, 0, tzinfo=UTC)


def bar(minutes_after_open: int, o: float, h: float, low: float, c: float, vol: int = 100) -> Bar:
    return Bar(
        time=OPEN_UTC + timedelta(minutes=minutes_after_open),
        open=o, high=h, low=low, close=c, tick_volume=vol,
    )


@pytest.fixture
def cfg() -> AppConfig:
    return AppConfig()


@pytest.fixture
def schedule(cfg) -> SessionSchedule:
    s = cfg.session
    return SessionSchedule(
        tz_name=s.timezone, session_open=s.open, or_minutes=s.or_minutes,
        pos1_cutoff=s.pos1_cutoff, pos2_cutoff=s.pos2_cutoff, flat_time=s.flat_time,
    )


@pytest.fixture
def day_ctx(schedule) -> DayContext:
    """Context with an up-gap: previous close 19980, session opens at 20000."""
    return DayContext(session=schedule.for_day(DAY), prev_session_close=19980.0)
