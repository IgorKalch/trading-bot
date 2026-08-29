from datetime import UTC, datetime

import pytest

from tradingbot.config import PropConfig
from tradingbot.data.mt5_client import AccountState
from tradingbot.data.news import NewsCalendar, NewsEvent
from tradingbot.risk.prop_guard import PropGuard

DAY = datetime(2025, 6, 5, tzinfo=UTC).date()
NOW = datetime(2025, 6, 5, 8, 0, tzinfo=UTC)  # Thursday morning


def account(balance=100_000.0, equity=None) -> AccountState:
    return AccountState(
        balance=balance, equity=equity if equity is not None else balance,
        margin_free=balance, currency="USD", login=1,
    )


@pytest.fixture
def guard(tmp_path) -> PropGuard:
    g = PropGuard(PropConfig(), state_dir=tmp_path)
    g.on_day_start(DAY, account())
    return g


def test_floors(guard):
    # 100k anchor: firm daily floor 95k, buffered 96.5k; overall 90k / 92k.
    assert guard.daily_loss_floor() == pytest.approx(95_000)
    assert guard.daily_soft_floor() == pytest.approx(96_500)
    assert guard.overall_floor() == pytest.approx(90_000)
    assert guard.overall_soft_floor() == pytest.approx(92_000)


def test_allows_normal_trade(guard):
    v = guard.can_open(account(), 1_000, NOW, NewsCalendar.empty(), ["EUR", "USD"])
    assert v.allowed


def test_blocks_when_daily_headroom_too_small(guard):
    # Equity already down to 97k: 1k more risk -> 96k < soft floor 96.5k.
    v = guard.can_open(account(equity=97_000), 1_000, NOW, NewsCalendar.empty(), ["EUR"])
    assert not v.allowed and v.rule == "daily_loss"


def test_blocks_in_news_window(guard):
    news = NewsCalendar([NewsEvent(time=NOW, currency="EUR", impact="high", title="ECB")])
    v = guard.can_open(account(), 1_000, NOW, news, ["EUR", "USD"])
    assert not v.allowed and v.rule == "news_window"


def test_news_of_other_currency_ignored(guard):
    news = NewsCalendar([NewsEvent(time=NOW, currency="GBP", impact="high", title="BoE")])
    v = guard.can_open(account(), 1_000, NOW, news, ["EUR", "USD"])
    assert v.allowed


def test_emergency_close(guard):
    assert not guard.emergency_close_needed(account(equity=97_000)).allowed
    verdict = guard.emergency_close_needed(account(equity=96_400))
    assert verdict.allowed and verdict.rule == "daily_loss"


def test_day_anchor_uses_max_of_balance_equity(tmp_path):
    g = PropGuard(PropConfig(), state_dir=tmp_path)
    g.on_day_start(DAY, account(balance=100_000, equity=101_500))
    assert g.day_anchor == pytest.approx(101_500)


def test_anchor_persists_across_restart(tmp_path):
    g1 = PropGuard(PropConfig(), state_dir=tmp_path)
    g1.on_day_start(DAY, account(balance=100_000))
    # "Restart" with a higher balance same day: anchor must NOT change.
    g2 = PropGuard(PropConfig(), state_dir=tmp_path)
    g2.on_day_start(DAY, account(balance=105_000))
    assert g2.day_anchor == pytest.approx(100_000)
    assert g2.initial_balance == pytest.approx(100_000)


def test_friday_evening_entries_blocked(tmp_path):
    g = PropGuard(PropConfig(), state_dir=tmp_path)
    friday = datetime(2025, 6, 6, tzinfo=UTC).date()
    g.on_day_start(friday, account())
    late_friday = datetime(2025, 6, 6, 19, 30, tzinfo=UTC)
    v = g.can_open(account(), 1_000, late_friday, NewsCalendar.empty(), ["EUR"])
    assert not v.allowed and v.rule == "weekend"


def test_disabled_guard_allows_everything(tmp_path):
    g = PropGuard(PropConfig(enabled=False), state_dir=tmp_path)
    v = g.can_open(account(equity=10), 1_000_000, NOW, NewsCalendar.empty(), ["EUR"])
    assert v.allowed
