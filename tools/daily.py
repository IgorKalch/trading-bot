"""Multi-day horizon research across a diversified market universe.

Why this exists, in one line: cost/R is cost divided by stop distance, stop
distance grows with the square root of holding time, and cost does not grow at
all. Every market this project killed on cost - DAX at 4.5%, ES at 6.1%,
EuroStoxx at 23.8% - becomes cost-survivable once the stop is measured in daily
ATR instead of a five-minute range.

The intraday engine in src/ cannot express this: it flattens at flat_time by
construction. This is a separate, deliberately small backtester for positions
held across days.

Conventions kept identical to the intraday work so results are comparable:
  * everything is measured in R, where R is the initial stop distance
  * costs are charged on entry and exit
  * signals are evaluated on CLOSED daily bars, entry fills at the next open

    python tools/daily.py --systems donchian,tsmom --cost-atr 0.05
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA = Path("data/cache_daily")

CLASSES = {
    "NQ": "equity", "ES": "equity", "YM": "equity", "DAX": "equity",
    "SX5E": "equity", "FTSE": "equity", "NKY": "equity",
    "EURUSD": "fx", "GBPUSD": "fx", "USDJPY": "fx",
    "AUDUSD": "fx", "USDCAD": "fx", "USDCHF": "fx",
    "XAUUSD": "metal", "XAGUSD": "metal", "COPPER": "metal",
    "BRENT": "energy", "WTI": "energy", "NATGAS": "energy",
    "BUND": "bond", "GILT": "bond", "USTB": "bond",
    "BTCUSD": "crypto",
}


@dataclass
class Trade:
    market: str
    entry_day: pd.Timestamp
    exit_day: pd.Timestamp
    side: int
    entry: float
    stop: float
    exit: float
    r: float
    bars_held: int


def load(sym: str) -> pd.DataFrame | None:
    p = DATA / f"{sym}_D1.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df.dropna()


def signal(df: pd.DataFrame, system: str, n: int) -> pd.Series:
    """+1 long, -1 short, 0 flat. Uses only information available at the close."""
    c = df["close"]
    if system == "donchian":
        hi = df["high"].rolling(n).max().shift(1)
        lo = df["low"].rolling(n).min().shift(1)
        s = pd.Series(0, index=df.index)
        s[c > hi] = 1
        s[c < lo] = -1
        return s.replace(0, pd.NA).ffill().fillna(0).astype(int)
    if system == "tsmom":
        return (c > c.shift(n)).astype(int) * 2 - 1
    if system == "macross":
        fast = c.rolling(max(2, n // 4)).mean()
        slow = c.rolling(n).mean()
        return (fast > slow).astype(int) * 2 - 1
    raise ValueError(system)


def backtest(
    df: pd.DataFrame, market: str, system: str, n: int, atr_stop: float, cost_atr: float,
    trail: bool = True,
) -> list[Trade]:
    sig = signal(df, system, n)
    o, h, low_, c, atr = df["open"], df["high"], df["low"], df["close"], df["atr"]
    idx = df.index
    trades: list[Trade] = []
    pos = 0
    entry = stop = initial_risk = 0.0
    entry_i = 0
    for i in range(1, len(df) - 1):
        want = sig.iloc[i]
        if pos != 0:
            # stop check on the following bar, conservatively against us
            hit = low_.iloc[i] <= stop if pos > 0 else h.iloc[i] >= stop
            flip = want != 0 and want != pos
            if hit or flip:
                px = stop if hit else o.iloc[i + 1]
                # R is measured against the INITIAL stop. Using the trailed stop
                # shrinks the denominator on winners and inflates every metric.
                cost = cost_atr * atr.iloc[entry_i]
                r = ((px - entry) * pos - cost) / initial_risk
                trades.append(
                    Trade(market, idx[entry_i], idx[i], pos, entry, stop, px, r, i - entry_i)
                )
                pos = 0
            elif trail:
                new = (
                    c.iloc[i] - atr_stop * atr.iloc[i]
                    if pos > 0
                    else c.iloc[i] + atr_stop * atr.iloc[i]
                )
                stop = max(stop, new) if pos > 0 else min(stop, new)
        # Enter only on a FRESH signal. Without this the state stays "long"
        # after a stop-out and the system re-enters the next bar, which
        # manufactures whipsaw rather than trading a trend.
        fresh = want != 0 and want != sig.iloc[i - 1]
        if pos == 0 and fresh and i + 1 < len(df):
            pos = int(want)
            entry = o.iloc[i + 1]
            entry_i = i + 1
            stop = entry - pos * atr_stop * atr.iloc[i]
            initial_risk = abs(entry - stop)
    return trades


def stats(rs: list[float]) -> dict:
    n = len(rs)
    if n < 10:
        return {}
    m = sum(rs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (n - 1))
    gw = sum(x for x in rs if x > 0)
    gl = -sum(x for x in rs if x < 0)
    peak = cum = worst = 0.0
    for x in rs:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    return {
        "n": n, "wr": sum(1 for x in rs if x > 0) / n * 100, "pf": gw / gl if gl else 0,
        "exp": m, "se": sd / math.sqrt(n), "tot": sum(rs), "dd": -worst,
    }


def portfolio(all_trades: list[Trade], months: float) -> dict:
    """Aggregate R by calendar day across markets, then measure the joint curve."""
    if not all_trades:
        return {}
    by_day: dict[pd.Timestamp, float] = {}
    for t in all_trades:
        d = t.exit_day.normalize()
        by_day[d] = by_day.get(d, 0.0) + t.r
    days = sorted(by_day)
    series = [by_day[d] for d in days]
    peak = cum = worst = 0.0
    for x in series:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    tot = sum(series)
    return {
        "trades": len(all_trades), "tot": tot, "dd": -worst,
        "r_per_month": tot / months, "mar": tot / months / -worst if worst else 0,
        "per_month_trades": len(all_trades) / months,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="donchian,tsmom,macross")
    ap.add_argument("--lookbacks", default="20,55,100")
    ap.add_argument("--atr-stop", type=float, default=3.0)
    ap.add_argument("--cost-atr", type=float, default=0.05,
                    help="round-trip cost as a fraction of ATR(14)")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--markets", default="")
    ap.add_argument("--no-trail", action="store_true")
    args = ap.parse_args()

    syms = args.markets.split(",") if args.markets else list(CLASSES)
    data = {s: load(s) for s in syms}
    data = {s: d[d.index >= args.start] for s, d in data.items() if d is not None and len(d) > 300}
    span = max(d.index.max() for d in data.values()) - min(d.index.min() for d in data.values())
    months = span.days / 30.44
    print(f"{len(data)} markets, {months:.0f} months, cost {args.cost_atr}xATR round trip, "
          f"stop {args.atr_stop}xATR\n")

    best = []
    for system in args.systems.split(","):
        for n in [int(x) for x in args.lookbacks.split(",")]:
            allt: list[Trade] = []
            for sym, df in data.items():
                allt.extend(backtest(df, sym, system, n, args.atr_stop, args.cost_atr,
                                     trail=not args.no_trail))
            p = portfolio(allt, months)
            s = stats([t.r for t in allt])
            if not s:
                continue
            best.append((system, n, p, s, allt))
            print(f"{system:<9} n={n:<4} trades {s['n']:>5} ({p['per_month_trades']:>4.1f}/mo) "
                  f"WR {s['wr']:>4.1f}% PF {s['pf']:>4.2f} exp {s['exp']:>+6.3f}R "
                  f"| portfolio {p['r_per_month']:>5.2f}R/mo  DD {p['dd']:>5.1f}R  MAR {p['mar']:>4.2f}")

    if not best:
        print("nothing ran")
        return 1
    top = max(best, key=lambda b: b[2]["mar"])
    system, n, p, s, allt = top
    print(f"\nBEST BY MAR: {system} n={n} -> {p['r_per_month']:.2f}R/month, "
          f"DD {p['dd']:.1f}R, MAR {p['mar']:.2f}")
    print("\nper-market breakdown:")
    print(f"  {'market':<9}{'class':<8}{'n':>5}{'WR':>7}{'PF':>6}{'expR':>8}{'totR':>8}")
    for sym in data:
        rs = [t.r for t in allt if t.market == sym]
        st = stats(rs)
        if st:
            print(f"  {sym:<9}{CLASSES.get(sym,'?'):<8}{st['n']:>5}{st['wr']:>6.1f}%"
                  f"{st['pf']:>6.2f}{st['exp']:>+8.3f}{st['tot']:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
