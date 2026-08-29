from datetime import UTC, datetime

from tradingbot.core.clock import ServerClock, SessionSchedule, us_dst_active


def test_us_dst_boundaries():
    assert not us_dst_active(datetime(2026, 3, 7).date())  # before 2nd Sunday of March
    assert us_dst_active(datetime(2026, 3, 8).date())  # 2nd Sunday of March 2026
    assert us_dst_active(datetime(2026, 10, 31).date())
    assert not us_dst_active(datetime(2026, 11, 1).date())  # 1st Sunday of November


def test_server_clock_eet_us_dst():
    clock = ServerClock("eet_us_dst")
    # Winter: server UTC+2. 10:00 server -> 08:00 UTC.
    winter = datetime(2026, 1, 15, 10, 0)
    assert clock.to_utc(winter) == datetime(2026, 1, 15, 8, 0, tzinfo=UTC)
    # Summer (US DST): server UTC+3. 10:00 server -> 07:00 UTC.
    summer = datetime(2026, 6, 15, 10, 0)
    assert clock.to_utc(summer) == datetime(2026, 6, 15, 7, 0, tzinfo=UTC)
    # Round-trip.
    assert clock.from_utc(clock.to_utc(summer)) == summer


def test_server_clock_fixed_offset():
    clock = ServerClock("fixed:+03:00")
    dt = datetime(2026, 1, 15, 12, 0)
    assert clock.to_utc(dt) == datetime(2026, 1, 15, 9, 0, tzinfo=UTC)


def test_server_clock_iana():
    clock = ServerClock("Europe/Kyiv")
    # Kyiv winter = UTC+2.
    assert clock.to_utc(datetime(2026, 1, 15, 12, 0)) == datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    # Kyiv summer = UTC+3.
    assert clock.to_utc(datetime(2026, 6, 15, 12, 0)) == datetime(2026, 6, 15, 9, 0, tzinfo=UTC)


def test_session_schedule_berlin_anchor():
    sched = SessionSchedule(
        tz_name="Europe/Berlin", session_open="09:00", or_minutes=5,
        pos1_cutoff="11:00", pos2_cutoff="17:00", flat_time="17:20",
    )
    # June: Berlin UTC+2 -> 09:00 Berlin = 07:00 UTC.
    summer = sched.for_day(datetime(2025, 6, 5).date())
    assert summer.session_open == datetime(2025, 6, 5, 7, 0, tzinfo=UTC)
    assert summer.or_end == datetime(2025, 6, 5, 7, 5, tzinfo=UTC)
    # January: Berlin UTC+1 -> 09:00 Berlin = 08:00 UTC.
    winter = sched.for_day(datetime(2025, 1, 15).date())
    assert winter.session_open == datetime(2025, 1, 15, 8, 0, tzinfo=UTC)
    assert winter.flat_time == datetime(2025, 1, 15, 16, 20, tzinfo=UTC)
