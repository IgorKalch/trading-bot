"""Scenario tests for the retest-and-absorb model (STRATEGY.md Додаток В).

Same synthetic day as the ORB tests: OR candle 20000..20050 (width 50).
"""

from __future__ import annotations

from tests.conftest import bar
from tradingbot.config import StrategyConfig
from tradingbot.core.models import EntrySignal, Side, SkipEvent
from tradingbot.strategy.retest import RetestStrategy

OR_BAR = bar(0, 20000, 20050, 20000, 20040)


def cfg_retest() -> StrategyConfig:
    c = StrategyConfig()
    c.name = "retest"
    return c


def run_day(bars, ctx, cfg: StrategyConfig | None = None):
    strat = RetestStrategy(cfg or cfg_retest())
    strat.on_day_start(ctx)
    events = []
    for b in bars:
        events.extend(strat.on_bar(b))
    return events


def signals(events):
    return [e for e in events if isinstance(e, EntrySignal)]


def skip_rules(events):
    return [e.rule for e in events if isinstance(e, SkipEvent)]


def test_full_long_sequence_break_retest_absorb(day_ctx):
    bars = [
        OR_BAR,
        bar(5, 20045, 20080, 20044, 20075),  # body close above 20050 -> break, extreme 20080
        bar(10, 20075, 20078, 20048, 20055),  # pulls back and touches the boundary
        bar(15, 20055, 20060, 20046, 20052),  # pullback extreme drops to 20046
        bar(20, 20052, 20090, 20051, 20085),  # closes above 20080 -> absorption
    ]
    sigs = signals(run_day(bars, day_ctx))
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.side is Side.LONG
    assert sig.entry_ref == 20085
    # Stop sits at the pullback extreme (20046) minus the 2.0 buffer, NOT at OR.low.
    assert sig.stop_loss == 20044.0
    assert sig.risk_points == 41.0
    assert sig.take_profit == 20085 + 41.0  # default fixed_rr 1.0


def test_stop_is_far_tighter_than_the_orb_stop(day_ctx):
    """The whole point of the model: 1R is a fraction of the range, not the range."""
    bars = [
        OR_BAR,
        bar(5, 20045, 20080, 20044, 20075),
        bar(10, 20075, 20078, 20049, 20055),  # shallow retest, pullback low 20049
        bar(15, 20055, 20090, 20054, 20085),
    ]
    sig = signals(run_day(bars, day_ctx))[0]
    orb_stop_distance = 20085 - 20000  # what the ORB model would have risked
    assert sig.risk_points == 20085 - (20049 - 2.0)
    assert sig.risk_points < orb_stop_distance / 2


def test_no_retest_times_out(day_ctx):
    cfg = cfg_retest()
    cfg.retest.max_pullback_bars = 2
    bars = [
        OR_BAR,
        bar(5, 20045, 20080, 20044, 20075),  # break
        bar(10, 20075, 20085, 20070, 20080),  # never comes back to the boundary
        bar(15, 20080, 20095, 20078, 20090),
        bar(20, 20090, 20100, 20088, 20095),
    ]
    events = run_day(bars, day_ctx, cfg)
    assert not signals(events)
    assert "retest_timeout" in skip_rules(events)


def test_pullback_through_the_range_invalidates_the_setup(day_ctx):
    bars = [
        OR_BAR,
        bar(5, 20045, 20080, 20044, 20075),  # break up
        bar(10, 20075, 20078, 20040, 20045),  # retest touches the boundary
        bar(15, 20045, 20046, 19990, 19995),  # closes below OR.low -> dead
        bar(20, 19995, 20090, 19990, 20085),  # even a later reclaim must not fire
    ]
    events = run_day(bars, day_ctx)
    assert not signals(events)
    assert "retest_invalidated" in skip_rules(events)


def test_absorption_needs_to_clear_the_pre_pullback_extreme(day_ctx):
    bars = [
        OR_BAR,
        bar(5, 20045, 20080, 20044, 20075),  # break, extreme 20080
        bar(10, 20075, 20078, 20048, 20055),  # retest
        bar(15, 20055, 20079, 20054, 20078),  # closes at 20078 — below the extreme
    ]
    assert not signals(run_day(bars, day_ctx))


def test_short_side_mirrors_the_long_logic(day_ctx):
    bars = [
        OR_BAR,
        bar(5, 20005, 20006, 19970, 19975),  # body close below 20000 -> break down
        bar(10, 19975, 20002, 19974, 19990),  # retest touches OR.low from below
        bar(15, 19990, 19996, 19960, 19965),  # closes below 19970 -> absorption
    ]
    sigs = signals(run_day(bars, day_ctx))
    assert len(sigs) == 1
    assert sigs[0].side is Side.SHORT
    assert sigs[0].stop_loss == 20002 + 2.0  # pullback high plus buffer


def test_break_requires_a_directional_body(day_ctx):
    bars = [
        OR_BAR,
        bar(5, 20080, 20085, 20050, 20055),  # closes above OR.high but the body is red
        bar(10, 20055, 20058, 20048, 20052),
        bar(15, 20052, 20095, 20051, 20090),
    ]
    # The red bar cannot arm the setup; the last bar arms it instead of entering.
    assert not signals(run_day(bars, day_ctx))


def test_min_stop_points_rejects_a_stop_tighter_than_the_spread(day_ctx):
    cfg = cfg_retest()
    cfg.retest.min_stop_points = 50.0
    bars = [
        OR_BAR,
        bar(5, 20045, 20080, 20044, 20075),
        bar(10, 20075, 20078, 20050, 20070),  # very shallow pullback -> tiny 1R
        bar(15, 20070, 20090, 20069, 20085),
    ]
    events = run_day(bars, day_ctx, cfg)
    assert not signals(events)
    assert "stop_size" in skip_rules(events)
