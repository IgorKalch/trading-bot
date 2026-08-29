"""Backtest performance metrics — the same set the source strategy PDF uses:
WinRate, PnL, avg monthly PnL, expectancy (R), profit factor, max losing streak,
plus max drawdown on the balance curve.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from tradingbot.backtest.engine import BacktestResult, TradeRecord


@dataclass
class Metrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_r: float = 0.0
    expectancy_r: float = 0.0
    profit_factor: float = 0.0
    max_losing_streak: int = 0
    total_pnl_money: float = 0.0
    total_pnl_pct: float = 0.0
    avg_monthly_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    monthly_pnl_pct: dict[str, float] = field(default_factory=dict)
    by_kind: dict[str, Metrics] = field(default_factory=dict)


def compute_metrics(result: BacktestResult) -> Metrics:
    m = _metrics_for(result.trades, result.initial_balance)
    m.by_kind = {
        kind: _metrics_for([t for t in result.trades if t.kind == kind], result.initial_balance)
        for kind in sorted({t.kind for t in result.trades})
    }
    return m


def _metrics_for(trades: list[TradeRecord], initial_balance: float) -> Metrics:
    m = Metrics(trades=len(trades))
    if not trades:
        return m

    gross_win = gross_loss = 0.0
    streak = 0
    balance = initial_balance
    peak = initial_balance
    max_dd = 0.0
    monthly: dict[str, float] = defaultdict(float)

    for t in trades:
        if t.pnl_money > 0:
            m.wins += 1
            gross_win += t.pnl_money
            streak = 0
        else:
            m.losses += 1
            gross_loss += -t.pnl_money
            streak += 1
            m.max_losing_streak = max(m.max_losing_streak, streak)
        m.total_r += t.result_r
        balance += t.pnl_money
        peak = max(peak, balance)
        max_dd = max(max_dd, (peak - balance) / peak * 100.0)
        monthly[t.day.strftime("%Y-%m")] += t.pnl_money

    m.win_rate = m.wins / m.trades * 100.0
    m.expectancy_r = m.total_r / m.trades
    m.profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")
    m.total_pnl_money = gross_win - gross_loss
    m.total_pnl_pct = m.total_pnl_money / initial_balance * 100.0
    m.monthly_pnl_pct = {k: v / initial_balance * 100.0 for k, v in sorted(monthly.items())}
    m.avg_monthly_pnl_pct = (
        sum(m.monthly_pnl_pct.values()) / len(m.monthly_pnl_pct) if m.monthly_pnl_pct else 0.0
    )
    m.max_drawdown_pct = max_dd
    return m
