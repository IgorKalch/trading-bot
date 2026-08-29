"""Timezone and session-time handling.

MT5 returns bar/tick timestamps as *naive* datetimes in the broker server
timezone. Everything inside the bot uses tz-aware UTC, so conversion happens
exactly once — here.

Server timezone modes (config `mt5.server_timezone`):
  - an IANA name, e.g. "Europe/Kyiv" (EET with EU DST rules)
  - "fixed:+HH:MM", e.g. "fixed:+02:00" (no DST)
  - "eet_us_dst": UTC+2 in winter, UTC+3 in summer, switching on the US DST
    calendar (2nd Sunday of March / 1st Sunday of November). This is the most
    common convention among MT5 brokers ("New York close" charts).

Session times (Xetra open etc.) are configured in their own timezone
(default Europe/Kyiv, matching the source strategy description) and converted
to UTC per trading day, so EU DST transitions are handled correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

UTC = UTC


def _nth_sunday(year: int, month: int, n: int) -> date:
    d = date(year, month, 1)
    # weekday(): Monday=0 ... Sunday=6
    first_sunday = d + timedelta(days=(6 - d.weekday()) % 7)
    return first_sunday + timedelta(weeks=n - 1)


def us_dst_active(d: date) -> bool:
    """US DST: 2nd Sunday of March .. 1st Sunday of November."""
    return _nth_sunday(d.year, 3, 2) <= d < _nth_sunday(d.year, 11, 1)


@dataclass(frozen=True)
class ServerClock:
    """Converts naive broker-server datetimes to/from aware UTC."""

    mode: str  # IANA name | "fixed:+HH:MM" | "eet_us_dst"

    def _fixed_offset(self) -> timedelta | None:
        if not self.mode.startswith("fixed:"):
            return None
        raw = self.mode.removeprefix("fixed:")
        sign = -1 if raw.startswith("-") else 1
        hh, mm = raw.lstrip("+-").split(":")
        return sign * timedelta(hours=int(hh), minutes=int(mm))

    def utc_offset(self, server_naive: datetime) -> timedelta:
        if self.mode == "eet_us_dst":
            return timedelta(hours=3 if us_dst_active(server_naive.date()) else 2)
        fixed = self._fixed_offset()
        if fixed is not None:
            return fixed
        tz = ZoneInfo(self.mode)
        offset = server_naive.replace(tzinfo=tz).utcoffset()
        assert offset is not None
        return offset

    def to_utc(self, server_naive: datetime) -> datetime:
        return (server_naive - self.utc_offset(server_naive)).replace(tzinfo=UTC)

    def from_utc(self, dt_utc: datetime) -> datetime:
        # Offset depends on local date; one adjustment pass is enough for
        # whole-hour offsets far from midnight transitions.
        guess = dt_utc + self.utc_offset(dt_utc.replace(tzinfo=None))
        offset = self.utc_offset(guess.replace(tzinfo=None))
        return (dt_utc + offset).replace(tzinfo=None)


@dataclass(frozen=True)
class SessionTimes:
    """UTC timestamps of the strategy time marks for one trading day."""

    day: date
    session_open: datetime
    or_end: datetime
    pos1_cutoff: datetime
    pos2_cutoff: datetime
    flat_time: datetime


@dataclass(frozen=True)
class SessionSchedule:
    """Computes strategy time marks for a given calendar day.

    All times are configured as "HH:MM" strings in `tz` (IANA name).
    """

    tz_name: str
    session_open: str
    or_minutes: int
    pos1_cutoff: str
    pos2_cutoff: str
    flat_time: str

    def _at(self, day: date, hhmm: str) -> datetime:
        hh, mm = hhmm.split(":")
        local = datetime.combine(day, time(int(hh), int(mm)), tzinfo=ZoneInfo(self.tz_name))
        return local.astimezone(UTC)

    def for_day(self, day: date) -> SessionTimes:
        open_utc = self._at(day, self.session_open)
        return SessionTimes(
            day=day,
            session_open=open_utc,
            or_end=open_utc + timedelta(minutes=self.or_minutes),
            pos1_cutoff=self._at(day, self.pos1_cutoff),
            pos2_cutoff=self._at(day, self.pos2_cutoff),
            flat_time=self._at(day, self.flat_time),
        )

    def is_trading_day(self, day: date) -> bool:
        return day.weekday() < 5  # Mon..Fri; holidays handled by "no bars" naturally
