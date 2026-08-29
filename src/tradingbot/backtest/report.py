"""Backtest report: console text + trades CSV + skip log."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from tradingbot.backtest.engine import BacktestResult
from tradingbot.backtest.metrics import Metrics, compute_metrics


def _fmt(m: Metrics, title: str) -> str:
    pf = f"{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "inf"
    lines = [
        f"--- {title} ---",
        f"Trades: {m.trades}  (wins {m.wins} / losses {m.losses})",
        f"WinRate: {m.win_rate:.1f}%   Expectancy: {m.expectancy_r:+.3f}R   ProfitFactor: {pf}",
        f"Total PnL: {m.total_pnl_money:+,.2f} ({m.total_pnl_pct:+.2f}%)   "
        f"Avg monthly: {m.avg_monthly_pnl_pct:+.2f}%",
        f"Max losing streak: {m.max_losing_streak}   Max drawdown: {m.max_drawdown_pct:.2f}%",
    ]
    return "\n".join(lines)


def render_report(result: BacktestResult) -> str:
    m = compute_metrics(result)
    parts = [
        "================ BACKTEST REPORT ================",
        f"Days processed: {result.days_processed}   "
        f"Balance: {result.initial_balance:,.0f} -> {result.final_balance:,.2f}",
        _fmt(m, "ALL TRADES"),
    ]
    for kind, km in m.by_kind.items():
        parts.append(_fmt(km, f"{kind.upper()} POSITION"))
    if m.monthly_pnl_pct:
        parts.append("--- MONTHLY PnL % ---")
        parts.extend(f"{month}: {pnl:+.2f}%" for month, pnl in m.monthly_pnl_pct.items())
    parts.append(f"Skipped setups: {len(result.skips)}")
    skip_counts: dict[str, int] = {}
    for s in result.skips:
        skip_counts[s.rule] = skip_counts.get(s.rule, 0) + 1
    parts.extend(f"  {rule}: {n}" for rule, n in sorted(skip_counts.items(), key=lambda x: -x[1]))
    parts.append("=================================================")
    return "\n".join(parts)


def save_report(result: BacktestResult, reports_dir: str | Path, tag: str) -> tuple[Path, Path]:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)

    txt_path = out / f"backtest_{tag}.txt"
    txt_path.write_text(render_report(result), encoding="utf-8")

    csv_path = out / f"trades_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if result.trades:
            writer = csv.DictWriter(f, fieldnames=list(asdict(result.trades[0]).keys()))
            writer.writeheader()
            for t in result.trades:
                writer.writerow(asdict(t))
    return txt_path, csv_path
