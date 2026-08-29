"""End-to-end backtest engine tests on synthetic days.

Uses zero spread/slippage where exact R math is asserted.
"""

from __future__ import annotations

import pytest

from tests.conftest import bar
from tradingbot.backtest.engine import BacktestEngine
from tradingbot.backtest.metrics import compute_metrics
from tradingbot.config import AppConfig
from tradingbot.strategy.orb import OrbStrategy


def make_cfg(**backtest_overrides) -> AppConfig:
    cfg = AppConfig()
    cfg.backtest.spread_points = 0.0
    cfg.backtest.slippage_points = 0.0
    cfg.strategy.stops.sl_buffer_points = 0.0
    for k, v in backtest_overrides.items():
        setattr(cfg.backtest, k, v)
    return cfg


def run(bars, cfg=None):
    cfg = cfg or make_cfg()
    engine = BacktestEngine(cfg, OrbStrategy(cfg.strategy))
    return engine.run(bars), cfg


# Day skeleton: up-move day. OR 20000..20050. Long breakout confirmed at
# 09:10-09:15 closing 20085; entry next bar open 20085; SL 20000 (R=85);
# TP 1R = 20170.
UP_DAY = [
    bar(0, 20000, 20050, 20000, 20040),
    bar(5, 20040, 20055, 20030, 20045),
    bar(10, 20045, 20090, 20044, 20085),  # confirmation
    bar(15, 20085, 20120, 20080, 20110),  # entry fills at open 20085
    bar(20, 20110, 20180, 20105, 20160),  # TP 20170 hit intra-bar
    bar(25, 20160, 20165, 20150, 20155),
]


def test_take_profit_hit():
    result, cfg = run(UP_DAY)
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.kind == "first" and t.side == "long"
    assert t.entry == pytest.approx(20085)
    assert t.exit == pytest.approx(20170)
    assert t.result_r == pytest.approx(1.0)
    assert t.close_reason == "take profit hit"
    assert result.final_balance > result.initial_balance


def test_stop_loss_hit():
    bars = UP_DAY[:4] + [
        bar(20, 20110, 20115, 19990, 20000),  # collapses through SL 20000
    ]
    result, _ = run(bars)
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit == pytest.approx(20000)
    assert t.result_r == pytest.approx(-1.0)
    assert t.close_reason == "stop loss hit"


def test_pessimistic_same_bar_prefers_sl():
    # One giant bar hits both TP (20170) and SL (20000).
    bars = UP_DAY[:4] + [bar(20, 20110, 20200, 19990, 20100)]
    result, _ = run(bars)
    assert result.trades[0].close_reason == "stop loss hit"


def test_flat_time_forces_exit():
    # Position opened, price drifts sideways forever -> closed at flat_time
    # (17:20 Berlin = 15:20 UTC = 500 min after open).
    drift = [bar(15 + i * 5, 20085, 20090, 20080, 20086) for i in range(1, 120)]
    bars = UP_DAY[:4] + drift
    result, _ = run(bars)
    assert len(result.trades) == 1
    assert "flat_time" in result.trades[0].close_reason


def test_reversal_stops_out_first_position_by_sl():
    # With the default full-OR stop, a confirmed reversal breakout necessarily
    # trades through pos1's stop first — pos1 exits by SL, then pos2 opens.
    bars = [
        bar(0, 20000, 20050, 20000, 20040),
        bar(5, 20045, 20090, 20044, 20085),  # pos1 long confirmation
        bar(10, 20085, 20090, 20040, 20050),  # pos1 entry @20085
        bar(15, 20050, 20052, 19930, 19940),  # breaks OR.low: pos1 SL 20000 hit, reversal confirmed
        bar(20, 19940, 19950, 19850, 19870),  # pos2 entry @19940, then falls
        bar(25, 19870, 19880, 19700, 19750),  # pos2 TP (1R = 19940-(20050-19940)=19830) hit
    ]
    result, _ = run(bars)
    assert len(result.trades) == 2
    first, second = result.trades
    assert first.kind == "first" and first.close_reason == "stop loss hit"
    assert second.kind == "second" and second.side == "short"
    assert second.close_reason == "take profit hit"


def test_reversal_flips_surviving_first_position():
    # With an SL buffer, the reversal candle can confirm without touching
    # pos1's stop — the engine must close pos1 with the reversal reason.
    cfg = make_cfg()
    cfg.strategy.stops.sl_buffer_points = 5.0  # pos1 SL = 19995
    bars = [
        bar(0, 20000, 20050, 20000, 20040),
        bar(5, 20045, 20090, 20044, 20085),  # pos1 long confirmation
        bar(10, 20085, 20090, 20040, 20050),  # pos1 entry @20085
        bar(15, 20050, 20052, 19996, 19997),  # closes below OR.low, low stays above SL 19995
        bar(20, 19997, 20000, 19996, 19998),  # pos1 flipped at open, pos2 short entry @19997
        bar(25, 19995, 19996, 19930, 19940),  # pos2 TP (SL 20055, R=58 -> TP 19939) hit
    ]
    result, _ = run(bars, cfg)
    assert len(result.trades) == 2
    first, second = result.trades
    assert first.kind == "first" and "reversal" in first.close_reason
    assert second.kind == "second" and second.close_reason == "take profit hit"


def test_spread_and_commission_costs_reduce_pnl():
    cfg = make_cfg(spread_points=2.0, slippage_points=1.0, commission_per_lot=1.0)
    result, _ = run(UP_DAY, cfg)
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.entry == pytest.approx(20085 + 2.0)  # half-spread 1 + slippage 1
    # TP recomputed from the actual fill -> exactly +1R in price terms; the
    # commission still reduces the money PnL below volume * R points.
    assert t.result_r == pytest.approx(1.0)
    risk_points = t.entry - t.initial_stop
    assert t.pnl_money == pytest.approx(t.volume * risk_points - t.volume * 1.0)


def test_metrics_computation():
    result, _ = run(UP_DAY)
    m = compute_metrics(result)
    assert m.trades == 1 and m.wins == 1
    assert m.win_rate == pytest.approx(100.0)
    assert m.expectancy_r == pytest.approx(1.0)
    assert "first" in m.by_kind


def test_multi_day_run_and_gap_direction():
    # Day 1 establishes prev_session_close for day 2; day 2 opens with a gap
    # DOWN vs day-1 close, so an upside breakout on day 2 must be skipped.
    from datetime import timedelta

    day1 = UP_DAY  # closes around 20155
    day2 = [
        # opens at 20000 (< 20155 close): gap down; upside breakout -> skip
        bar(0, 20000, 20050, 20000, 20040),
        bar(5, 20045, 20090, 20044, 20085),
    ]
    day2 = [
        type(b)(time=b.time + timedelta(days=1), open=b.open, high=b.high,
                low=b.low, close=b.close, tick_volume=b.tick_volume,
                spread_points=b.spread_points)
        for b in day2
    ]
    result, _ = run(day1 + day2)
    assert len(result.trades) == 1  # only day 1 trade
    assert any(s.rule == "gap_direction" for s in result.skips)


def test_fill_bar_stop_out_detected():
    # The bar that fills the entry immediately collapses through the SL —
    # the engine must record the stop-out on the fill bar, not miss it.
    bars = UP_DAY[:3] + [
        bar(15, 20085, 20090, 19990, 20050),  # fill @20085, low 19990 < SL 20000
        bar(20, 20050, 20200, 20040, 20180),  # would tag TP if wrongly still open
    ]
    result, _ = run(bars)
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.close_reason == "stop loss hit"
    assert t.exit == pytest.approx(20000)


def test_gap_through_tp_overrides_pessimism():
    # Bar OPENS beyond TP (20170) and later collapses through SL: the open is
    # chronologically first, so TP fills at the open despite pessimistic mode.
    bars = UP_DAY[:4] + [bar(20, 20175, 20200, 19990, 20000)]
    result, _ = run(bars)
    t = result.trades[0]
    assert t.close_reason == "take profit hit"
    assert t.exit == pytest.approx(20175)


def test_tp_exit_not_charged_spread():
    cfg = make_cfg(spread_points=2.0, slippage_points=0.0)
    result, _ = run(UP_DAY, cfg)
    t = result.trades[0]
    # Entry pays half-spread (20086); TP recomputed from fill at 1R; the TP
    # exit itself is spread-free, so the trade closes at exactly +1R.
    assert t.result_r == pytest.approx(1.0)
