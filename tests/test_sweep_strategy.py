"""Scenario tests for the overnight-sweep model and the FVG helper."""

from __future__ import annotations

from datetime import timedelta

from tests.conftest import OPEN_UTC, bar
from tradingbot.config import StrategyConfig
from tradingbot.core.models import Bar, EntrySignal, Side, SkipEvent
from tradingbot.strategy.imbalance import FvgTracker
from tradingbot.strategy.sweep import SweepStrategy


def pre(minutes_before: int, o: float, h: float, low: float, c: float) -> Bar:
    """A bar BEFORE the session open — builds the overnight range."""
    return Bar(
        time=OPEN_UTC - timedelta(minutes=minutes_before),
        open=o, high=h, low=low, close=c, tick_volume=100,
    )


def overnight(n: int = 40) -> list[Bar]:
    """n quiet bars before the open, range 19900..20000."""
    bars = [pre(5 * (n - i), 19950, 19960, 19940, 19950) for i in range(n)]
    bars[0] = pre(5 * n, 19950, 20000, 19900, 19950)  # the extremes
    return bars


def cfg_sweep() -> StrategyConfig:
    c = StrategyConfig()
    c.name = "sweep"
    c.filters.news_filter_enabled = False
    return c


def run(bars, ctx, cfg: StrategyConfig | None = None):
    strat = SweepStrategy(cfg or cfg_sweep())
    strat.on_day_start(ctx)
    events = []
    for b in bars:
        events.extend(strat.on_bar(b))
    return events


def signals(events):
    return [e for e in events if isinstance(e, EntrySignal)]


def rules(events):
    return [e.rule for e in events if isinstance(e, SkipEvent)]


def test_sweep_of_the_high_then_reclaim_goes_short(day_ctx):
    bars = overnight() + [
        bar(0, 19990, 20030, 19985, 20025),  # trades above 20000 -> sweep, no reclaim
        bar(5, 20025, 20028, 19980, 19985),  # closes back inside -> short
    ]
    sigs = signals(run(bars, day_ctx))
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.side is Side.SHORT
    assert sig.entry_ref == 19985
    assert sig.stop_loss == 20030 + 2.0  # sweep extreme plus the buffer
    assert sig.risk_points == 47.0


def test_sweep_of_the_low_then_reclaim_goes_long(day_ctx):
    bars = overnight() + [
        bar(0, 19910, 19915, 19870, 19875),  # trades below 19900
        bar(5, 19875, 19930, 19872, 19925),  # closes back inside -> long
    ]
    sigs = signals(run(bars, day_ctx))
    assert len(sigs) == 1
    assert sigs[0].side is Side.LONG
    assert sigs[0].stop_loss == 19870 - 2.0


def test_a_bar_that_sweeps_and_reclaims_by_itself_fires(day_ctx):
    bars = overnight() + [bar(0, 19990, 20040, 19960, 19970)]  # spike up, close inside
    sigs = signals(run(bars, day_ctx))
    assert len(sigs) == 1 and sigs[0].side is Side.SHORT
    assert sigs[0].stop_loss == 20042.0


def test_no_reclaim_times_out(day_ctx):
    cfg = cfg_sweep()
    cfg.sweep.max_reclaim_bars = 2
    bars = overnight() + [
        bar(0, 19990, 20030, 19985, 20025),
        bar(5, 20025, 20040, 20020, 20035),  # keeps going, never comes back
        bar(10, 20035, 20050, 20030, 20045),
        bar(15, 20045, 20060, 20040, 20055),
    ]
    events = run(bars, day_ctx, cfg)
    assert not signals(events)
    assert "sweep_timeout" in rules(events)


def test_min_sweep_depth_ignores_a_marginal_poke(day_ctx):
    cfg = cfg_sweep()
    cfg.sweep.min_sweep_points = 20.0
    bars = overnight() + [
        bar(0, 19990, 20005, 19985, 20002),  # only 5 points beyond the edge
        bar(5, 20002, 20004, 19980, 19985),
    ]
    assert not signals(run(bars, day_ctx, cfg))


def test_thin_overnight_session_is_skipped(day_ctx):
    cfg = cfg_sweep()
    cfg.sweep.min_pre_bars = 30
    bars = overnight(5) + [bar(0, 19990, 20030, 19985, 19985)]
    events = run(bars, day_ctx, cfg)
    assert not signals(events)
    assert "no_overnight_range" in rules(events)


def test_require_fvg_blocks_an_entry_without_imbalance(day_ctx):
    cfg = cfg_sweep()
    cfg.sweep.require_fvg = True
    bars = overnight() + [
        bar(0, 19990, 20030, 19985, 20025),
        bar(5, 20025, 20028, 19980, 19985),  # overlapping wicks -> no gap
    ]
    events = run(bars, day_ctx, cfg)
    assert not signals(events)
    assert "no_fvg" in rules(events)


# ------------------------------------------------------------------ FVG unit


def test_fvg_tracker_finds_a_bullish_gap():
    t = FvgTracker()
    t.update(bar(0, 100, 110, 90, 105))
    t.update(bar(5, 105, 160, 104, 155))  # displacement
    gap = t.update(bar(10, 155, 170, 120, 165))  # low 120 > first high 110
    assert gap is not None and gap.side is Side.LONG
    assert (gap.low, gap.high) == (110, 120)
    assert t.recent(Side.LONG, max_age_bars=5) is gap
    assert t.recent(Side.SHORT, max_age_bars=5) is None


def test_fvg_tracker_finds_a_bearish_gap_and_ages_it_out():
    t = FvgTracker()
    t.update(bar(0, 200, 210, 190, 195))
    t.update(bar(5, 195, 196, 150, 155))
    gap = t.update(bar(10, 155, 180, 150, 175))  # high 180 < first low 190
    assert gap is not None and gap.side is Side.SHORT
    assert (gap.low, gap.high) == (180, 190)
    for i in range(6):
        t.update(bar(15 + 5 * i, 175, 178, 172, 176))
    assert t.recent(Side.SHORT, max_age_bars=3) is None
    assert t.recent(Side.SHORT, max_age_bars=10) is gap


def test_fvg_min_size_filters_out_a_tiny_gap():
    t = FvgTracker()
    t.update(bar(0, 100, 110, 90, 105))
    t.update(bar(5, 105, 140, 104, 135))
    t.update(bar(10, 135, 150, 111, 145))  # gap is 110..111, one point wide
    assert t.recent(Side.LONG, max_age_bars=5, min_size=5.0) is None
    assert t.recent(Side.LONG, max_age_bars=5, min_size=0.5) is not None
