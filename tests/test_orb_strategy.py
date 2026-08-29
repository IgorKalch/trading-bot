"""Scenario tests for the ORB strategy against STRATEGY.md rules.

Synthetic day: session opens 07:00 UTC (09:00 Berlin, June). OR candle
20000..20050 (width 50). Previous close 19980 -> up-gap of +20 (open 20000).
"""

from __future__ import annotations

from tests.conftest import bar
from tradingbot.config import StrategyConfig
from tradingbot.core.models import EntrySignal, Side, SignalKind, SkipEvent
from tradingbot.strategy.orb import OrbStrategy


def run_day(bars, ctx, cfg: StrategyConfig | None = None):
    strat = OrbStrategy(cfg or StrategyConfig())
    strat.on_day_start(ctx)
    events = []
    for b in bars:
        events.extend(strat.on_bar(b))
    return events


OR_BAR = bar(0, 20000, 20050, 20000, 20040)  # 09:00-09:05, range 20000..20050


def entry_signals(events):
    return [e for e in events if isinstance(e, EntrySignal)]


def skips(events):
    return [e for e in events if isinstance(e, SkipEvent)]


def test_long_breakout_with_gap_up(day_ctx):
    bars = [
        OR_BAR,
        bar(5, 20040, 20055, 20030, 20045),  # inside
        bar(10, 20045, 20090, 20044, 20085),  # impulsive body close above OR.high
    ]
    events = run_day(bars, day_ctx)
    sigs = entry_signals(events)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig.side is Side.LONG and sig.kind is SignalKind.FIRST
    assert sig.stop_loss == 20000 - 2.0  # OR.low - buffer (§9)
    assert sig.take_profit is not None and sig.tp_rr == 1.0
    # 1R symmetric TP from the confirmation close (§10)
    assert abs((sig.take_profit - sig.entry_ref) - (sig.entry_ref - sig.stop_loss)) < 1e-9


def test_counter_gap_breakout_skipped(day_ctx):
    # Gap is UP (+20); a downside breakout must be skipped for position 1 (§4).
    bars = [
        OR_BAR,
        bar(5, 20000, 20005, 19940, 19950),  # impulsive close below OR.low
    ]
    events = run_day(bars, day_ctx)
    assert not entry_signals(events)
    assert any(s.rule == "gap_direction" for s in skips(events))


def test_counter_gap_allowed_for_small_gap(day_ctx):
    cfg = StrategyConfig()
    cfg.entry.counter_gap_max_points = 60.0  # gap is 20 <= 60 -> allowed
    bars = [OR_BAR, bar(5, 20000, 20005, 19940, 19950)]
    sigs = entry_signals(run_day(bars, day_ctx, cfg))
    assert len(sigs) == 1 and sigs[0].side is Side.SHORT


def test_weak_confirmation_skipped(day_ctx):
    # Candle closes above OR.high but body/range < 0.5 (long wick) — §5.
    bars = [OR_BAR, bar(5, 20048, 20095, 20040, 20055)]  # body 7, range 55
    events = run_day(bars, day_ctx)
    assert not entry_signals(events)
    assert any(s.rule == "confirmation" for s in skips(events))


def test_skipped_breakout_consumes_the_side(day_ctx):
    # Weak breakout burns the boundary; a later perfect candle on the same
    # side must NOT signal (base model trades only the first breakout, §3.4).
    bars = [
        OR_BAR,
        bar(5, 20048, 20095, 20040, 20055),  # weak -> skip
        bar(10, 20055, 20120, 20054, 20115),  # would be perfect
    ]
    events = run_day(bars, day_ctx)
    assert not entry_signals(events)


def test_second_position_on_reversal(day_ctx):
    bars = [
        OR_BAR,
        bar(5, 20045, 20090, 20044, 20085),  # pos1 long signal
        bar(10, 20085, 20087, 20020, 20025),  # back inside OR
        bar(15, 20025, 20026, 19930, 19940),  # impulsive close below OR.low
    ]
    sigs = entry_signals(run_day(bars, day_ctx))
    assert len(sigs) == 2
    assert sigs[0].kind is SignalKind.FIRST and sigs[0].side is Side.LONG
    assert sigs[1].kind is SignalKind.SECOND and sigs[1].side is Side.SHORT
    assert sigs[1].stop_loss == 20050 + 2.0  # behind OR.high (§9)


def test_no_third_signal(day_ctx):
    bars = [
        OR_BAR,
        bar(5, 20045, 20090, 20044, 20085),  # pos1 long
        bar(10, 20085, 20087, 19930, 19940),  # pos2 short (reversal)
        bar(15, 19940, 20120, 19939, 20110),  # back above OR — must NOT signal
    ]
    sigs = entry_signals(run_day(bars, day_ctx))
    assert len(sigs) == 2


def test_pos1_cutoff(day_ctx):
    # Confirmation at 11:05 Berlin (125 min after open) is past the 11:00 cutoff (§8).
    bars = [OR_BAR, bar(125, 20045, 20090, 20044, 20085)]
    events = run_day(bars, day_ctx)
    assert not entry_signals(events)
    assert any(s.rule == "time_window" for s in skips(events))


def test_no_gap_info_allows_both_directions(schedule):
    from tests.conftest import DAY
    from tradingbot.strategy.base import DayContext

    ctx = DayContext(session=schedule.for_day(DAY), prev_session_close=None)
    bars = [OR_BAR, bar(5, 20000, 20005, 19940, 19950)]  # downside breakout
    sigs = entry_signals(run_day(bars, ctx))
    assert len(sigs) == 1 and sigs[0].side is Side.SHORT


def test_or_width_filter_skips_day(day_ctx):
    cfg = StrategyConfig()
    cfg.filters.max_or_width_points = 40.0  # OR width is 50 -> skip day (§7.2)
    bars = [OR_BAR, bar(5, 20045, 20090, 20044, 20085)]
    events = run_day(bars, day_ctx, cfg)
    assert not entry_signals(events)
    assert any(s.rule == "or_width_max" for s in skips(events))


def test_max_positions_per_day_limit(day_ctx):
    cfg = StrategyConfig()
    cfg.entry.max_positions_per_day = 1
    bars = [
        OR_BAR,
        bar(5, 20045, 20090, 20044, 20085),  # pos1
        bar(10, 20085, 20087, 19930, 19940),  # reversal — must be ignored
    ]
    sigs = entry_signals(run_day(bars, day_ctx, cfg))
    assert len(sigs) == 1


def test_zero_gap_allows_both_directions(schedule):
    from tests.conftest import DAY
    from tradingbot.strategy.base import DayContext

    # Previous close exactly equals the session open (20000): no gap direction.
    ctx = DayContext(session=schedule.for_day(DAY), prev_session_close=20000.0)
    bars = [OR_BAR, bar(5, 20000, 20005, 19940, 19950)]  # downside breakout
    sigs = entry_signals(run_day(bars, ctx))
    assert len(sigs) == 1 and sigs[0].side is Side.SHORT
