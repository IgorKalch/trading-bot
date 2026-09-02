"""Three legs on NQ, their correlations, and what diversification actually buys.

The single-leg search is exhausted (HANDOVER.md): seven instruments, 21,672
filter combinations, every exit scheme, the whole win-rate/stop-distance family.
Expectancy per trade cannot be pushed further on this data.

What has NOT been exhausted is the number of UNCORRELATED legs. Two legs took
the portfolio from MAR 1.67 to 2.57 while return per month barely moved, because
the gain came entirely from the drawdown. That is the one axis with headroom
left, and it scales roughly as the square root of the number of legs.

    leg 1  retest of the 5-minute opening range, M1, target 1R
    leg 2  IB confluence (close vs mid + which extreme printed first), 1.5R
    leg 3  IB extension on wide-IB days only - the width rule from
           reports/backtest_ib_open_types.txt, never before traded

Risk is split equally across whichever legs are enabled, so the R figures are
directly comparable to the single-leg numbers elsewhere in reports/.

    python tools/portfolio.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.disable(logging.WARNING)

from tradingbot.backtest.engine import BacktestEngine  # noqa: E402
from tradingbot.config import load_config  # noqa: E402
from tradingbot.data import history  # noqa: E402
from tradingbot.data.news import NewsCalendar  # noqa: E402
from tradingbot.strategy import build_strategy  # noqa: E402

CONFIG = "config/config.nq.yaml"
TZ = "America/New_York"
OPEN_M, CLOSE_M = 9 * 60 + 30, 15 * 60 + 55
COST = 0.5  # spread 0.25 + slippage 0.25, in points
CUT = pd.Period("2025-03", "M")


def m5_sessions() -> dict:
    """Session frames keyed by date, minutes-from-midnight in `m`."""
    cfg = load_config(CONFIG, ".env")
    bars = history.load_bars(cfg.backtest.data_dir, cfg.mt5.symbol, "M5")
    df = pd.DataFrame(
        [{"t": b.time, "o": b.open, "h": b.high, "l": b.low, "c": b.close} for b in bars]
    )
    loc = pd.to_datetime(df["t"], utc=True).dt.tz_convert(TZ)
    df["d"] = loc.dt.date
    df["m"] = loc.dt.hour * 60 + loc.dt.minute
    out = {}
    for d, g in df.groupby("d"):
        s = g[(g.m >= OPEN_M) & (g.m < CLOSE_M)].sort_values("m").reset_index(drop=True)
        if len(s) >= 60:
            out[d] = s
    return out


def _walk(rows: pd.DataFrame, side: int, entry: float, stop: float, rr: float) -> float:
    """Resolve one trade bar by bar; unresolved trades exit at the last close."""
    risk = abs(entry - stop)
    tp = entry + side * rr * risk
    for _, b in rows.iterrows():
        if (b.l <= stop) if side > 0 else (b.h >= stop):
            return -1.0 - COST / risk
        if (b.h >= tp) if side > 0 else (b.l <= tp):
            return rr - COST / risk
    return (rows.iloc[-1]["c"] - entry) * side / risk - COST / risk


def leg2_confluence(sess: dict) -> pd.Series:
    """Close of the last IB bar vs IB mid, agreeing with which extreme came first."""
    out: dict = {}
    for d, s in sess.items():
        ib = s[s.m < OPEN_M + 60].reset_index(drop=True)
        rest = s[s.m >= OPEN_M + 60].reset_index(drop=True)
        if len(ib) != 12 or len(rest) < 30:
            continue
        ibh, ibl = ib.h.max(), ib.l.min()
        il, ih = int(ib["l"].idxmin()), int(ib["h"].idxmax())
        if il == ih:
            continue
        bull = ib.iloc[-1]["c"] > (ibh + ibl) / 2 and il < ih
        bear = ib.iloc[-1]["c"] < (ibh + ibl) / 2 and ih < il
        if not (bull or bear):
            continue
        side = 1 if bull else -1
        entry = rest.iloc[0]["o"]
        stop = ibl if side > 0 else ibh
        if abs(entry - stop) <= 0:
            continue
        out[d] = _walk(rest, side, entry, stop, 1.5)
    return pd.Series(out)


def leg3_extension(sess: dict, min_width_pct: float = 0.9, rr: float = 1.0) -> pd.Series:
    """First close beyond the IB, on wide-IB days only, stop at the IB mid.

    The width rule is the one conditional variable that survived the open-type
    work, and it runs the opposite way to the folklore: a WIDE first hour halves
    the chance of a two-sided day, so the first extension is likelier to hold.
    Width is a percentage of price, never an absolute point threshold - that
    mistake turns any filter on an index into a date filter (Додаток И).
    """
    out: dict = {}
    for d, s in sess.items():
        ib = s[s.m < OPEN_M + 60].reset_index(drop=True)
        rest = s[s.m >= OPEN_M + 60].reset_index(drop=True)
        if len(ib) != 12 or len(rest) < 30:
            continue
        ibh, ibl = ib.h.max(), ib.l.min()
        mid = (ibh + ibl) / 2
        if mid <= 0 or (ibh - ibl) / mid * 100 < min_width_pct:
            continue
        for i, b in rest.iterrows():
            side = 1 if b.c > ibh else (-1 if b.c < ibl else 0)
            if side == 0:
                continue
            tail = rest.iloc[i + 1 :]
            if tail.empty:
                break
            out[d] = _walk(tail, side, b.c, mid, rr)
            break
    return pd.Series(out)


def leg1_retest() -> pd.Series:
    """The engine's own retest signals, summed per day."""
    cfg = load_config(CONFIG, ".env")
    c = cfg.model_copy(deep=True)
    c.strategy.name = "retest"
    c.strategy.timeframe = "M1"
    c.strategy.retest.max_pullback_bars = 6
    c.strategy.targets.tp_mode = "fixed_rr"
    c.strategy.targets.fixed_rr = 1.0
    c.strategy.filters.require_fvg = True
    c.strategy.filters.trend_ma_period = 89
    c.backtest.spread_points = 0.25
    c.backtest.slippage_points = 0.25
    trades = BacktestEngine(c, build_strategy(c.strategy), NewsCalendar.empty()).run(
        history.load_bars(c.backtest.data_dir, c.mt5.symbol, "M1")
    ).trades
    out: dict = {}
    for t in trades:
        out[t.day] = out.get(t.day, 0.0) + t.result_r
    return pd.Series(out)


def monthly(s: pd.Series) -> pd.Series:
    return s.groupby([pd.Period(d, "M") for d in s.index]).sum()


def stats(ser: pd.Series) -> tuple[float, float, float]:
    """Mean R per month, max drawdown in R, and annualised MAR."""
    cum = peak = worst = 0.0
    for x in ser:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    dd = -worst
    yrs = len(ser) / 12
    return ser.mean(), dd, (ser.sum() / yrs) / dd if dd else 0.0


def describe(name: str, daily: pd.Series) -> None:
    m = monthly(daily)
    mean, dd, mar = stats(m)
    wins = (daily > 0).sum()
    pos = daily[daily > 0].sum()
    neg = -daily[daily < 0].sum()
    print(
        f"  {name:<26}{len(daily):>6}{wins / len(daily) * 100:>7.1f}%"
        f"{pos / neg if neg else 0:>7.2f}{daily.mean():>+9.3f}{mean:>+9.3f}{dd:>8.2f}{mar:>7.2f}"
    )


def main() -> int:
    sess = m5_sessions()
    legs = {
        "leg 1 retest M1 1R": leg1_retest(),
        "leg 2 IB confluence 1.5R": leg2_confluence(sess),
        "leg 3 IB extension wide": leg3_extension(sess),
    }

    print(f"=== NQ, {len(sess)} sessions, cost {COST} points per side ===\n")
    print(f"  {'leg':<26}{'n':>6}{'WR':>8}{'PF':>7}{'expR':>9}{'R/mo':>9}{'DD_R':>8}{'MAR':>7}")
    for k, v in legs.items():
        describe(k, v)

    mo = pd.DataFrame({k: monthly(v) for k, v in legs.items()}).fillna(0.0)
    print("\n--- monthly correlation ---")
    print(mo.corr().round(3).to_string())

    print(
        f"\n--- portfolios, risk split equally across the enabled legs ---\n"
        f"  {'legs':<16}{'R/mo':>9}{'DD_R':>8}{'MAR':>7}{'+mo':>7}"
        f"{'worst':>9}{'IS':>9}{'OOS':>9}"
    )
    combos = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    names = list(legs)
    for combo in combos:
        cols = [names[i - 1] for i in combo]
        port = mo[cols].sum(axis=1) / len(cols)
        mean, dd, mar = stats(port)
        _, _, mari = stats(port[port.index < CUT])
        _, _, maro = stats(port[port.index >= CUT])
        tag = "+".join(str(c) for c in combo)
        print(
            f"  {tag:<16}{mean:>+9.3f}{dd:>8.2f}{mar:>7.2f}{(port > 0).mean() * 100:>6.0f}%"
            f"{port.min():>+9.2f}{mari:>9.2f}{maro:>9.2f}"
        )

    port = mo.sum(axis=1) / 3
    print("\n--- all three legs, equal risk ---")
    print("  by year: " + "  ".join(f"{y}:{g.mean():+.3f}" for y, g in port.groupby(port.index.year)))
    for risk in (0.5, 1.0, 1.5, 2.0):
        eq = peak = 1.0
        mdd = 0.0
        for x in port:
            eq *= 1 + risk / 100 * x
            peak = max(peak, eq)
            mdd = max(mdd, (peak - eq) / peak)
        cagr = eq ** (12 / len(port)) - 1
        print(f"  {risk}% risk per 1R: {cagr * 100:+6.1f}%/yr, max DD {mdd * 100:.1f}%")

    print("\n--- how much a leg is worth: same-return comparison ---")
    base = stats(monthly(legs[names[0]]))
    print(f"  1 leg  MAR {base[2]:.2f}")
    for n, combo in ((2, (1, 2)), (3, (1, 2, 3))):
        cols = [names[i - 1] for i in combo]
        p = mo[cols].sum(axis=1) / len(cols)
        print(f"  {n} legs MAR {stats(p)[2]:.2f}   sqrt(n) would predict "
              f"{base[2] * np.sqrt(n):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
