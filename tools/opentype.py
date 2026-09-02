"""Market Profile open types, classified mechanically, and what they predict.

The "Type Open + Initial Balance" material (TO_IB/iborrotd.mp4) says to trade
only opens identifiable as Open Test Drive or Open Rejection Reverse, because
other types have a low chance of price extension. This measures that claim.

The measurement has to be against the right baseline. "The Initial Balance
extends" happens on 97% of days (reports/backtest_ib_open_types.txt), so it is
not a hypothesis anyone can be wrong about. The quantity that matters for
trading is a CLEAN day: the IB extends in the direction the open drove, and the
other side is never taken. That runs at 49.3% on NQ and 43.1% on DAX, which is
a baseline a classification can actually beat or fail to beat.

Thresholds are expressed in daily ATR so they do not silently become a date
filter as the index level changes - the mistake caught in Додаток И.

    python tools/opentype.py --markets NQ,DAX
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tradingbot.config import load_config  # noqa: E402
from tradingbot.data import history  # noqa: E402

MARKETS = {
    "NQ": ("config/config.nq.yaml", "America/New_York", "09:30"),
    "ES": ("config/config.es.yaml", "America/New_York", "09:30"),
    "DAX": ("config/config.duka.yaml", "Europe/Berlin", "09:00"),
}


def classify(
    sess: pd.DataFrame, open_px: float, atr: float,
    k_probe: float = 0.10, k_drive: float = 0.25, n_bars: int = 6,
) -> str | None:
    """One of OD / OTD / ORR / OA from the first n_bars 5-minute bars.

    OD  - drives from the open and never trades back through it
    OTD - a shallow probe against the eventual direction, then a drive
    ORR - a real excursion one way, rejected, then a drive back through the open
    OA  - no directional conviction
    """
    b = sess.iloc[:n_bars]
    if len(b) < n_bars:
        return None
    up = b.h.max() - open_px
    dn = open_px - b.l.min()
    close_n = b.c.iloc[-1]
    drive = 1 if close_n > open_px else -1
    after = b.iloc[1:]
    crossed = (after.l.min() <= open_px) if drive > 0 else (after.h.max() >= open_px)
    adverse = dn if drive > 0 else up
    favour = up if drive > 0 else dn
    if favour < k_drive * atr:
        return "OA"
    if not crossed and adverse < k_probe * atr:
        return "OD"
    if adverse >= k_drive * atr:
        return "ORR"
    return "OTD"


def sessions(market: str, k_probe: float, k_drive: float, n_bars: int) -> pd.DataFrame:
    cfgp, tzname, open_hm = MARKETS[market]
    cfg = load_config(cfgp, ".env")
    bars = history.load_bars(cfg.backtest.data_dir, cfg.mt5.symbol, "M5")
    df = pd.DataFrame([{"t": b.time, "o": b.open, "h": b.high, "l": b.low, "c": b.close} for b in bars])
    loc = pd.to_datetime(df["t"], utc=True).dt.tz_convert(tzname)
    df["d"] = loc.dt.date
    df["m"] = loc.dt.hour * 60 + loc.dt.minute
    oh, om = map(int, open_hm.split(":"))
    o0 = oh * 60 + om

    day_rng = df.groupby("d").agg(hi=("h", "max"), lo=("l", "min"))
    atr14 = (day_rng["hi"] - day_rng["lo"]).rolling(14).mean().shift(1)

    rows = []
    for d, g in df.groupby("d"):
        s = g[(g.m >= o0) & (g.m < o0 + 390)].reset_index(drop=True)
        if len(s) < 60:
            continue
        atr = atr14.get(d, np.nan)
        if not np.isfinite(atr) or atr <= 0:
            continue
        open_px = s.iloc[0]["o"]
        ib = s[s.m < o0 + 60]
        rest = s[s.m >= o0 + 60]
        if len(ib) < 10 or len(rest) < 12:
            continue
        t = classify(s, open_px, atr, k_probe, k_drive, n_bars)
        if t is None:
            continue
        up_ext = rest.h.max() > ib.h.max()
        dn_ext = rest.l.min() < ib.l.min()
        drive = 1 if s.iloc[:n_bars].c.iloc[-1] > open_px else -1
        rows.append(
            {
                "d": d, "type": t, "drive": drive,
                "ext_any": up_ext or dn_ext,
                "both": up_ext and dn_ext,
                "clean": (up_ext and not dn_ext) if drive > 0 else (dn_ext and not up_ext),
                "year": pd.Timestamp(d).year,
            }
        )
    return pd.DataFrame(rows)


def report(market: str, k_probe: float, k_drive: float, n_bars: int) -> None:
    r = sessions(market, k_probe, k_drive, n_bars)
    if r.empty:
        print(f"{market}: no sessions")
        return
    base = r.clean.mean() * 100
    print(f"\n=== {market}: {len(r)} sessions, probe {k_probe} ATR, drive {k_drive} ATR, "
          f"{n_bars} bars ===")
    print(f"  BASELINE: IB extends {r.ext_any.mean() * 100:.1f}%, both sides "
          f"{r.both.mean() * 100:.1f}%, CLEAN drive-direction day {base:.1f}%")
    print(f"  {'type':<6}{'n':>6}{'share':>8}{'both':>8}{'CLEAN':>8}{'vs base':>9}   by year")
    for t, g in r.groupby("type"):
        yr = "  ".join(f"{y}:{gg.clean.mean() * 100:.0f}%" for y, gg in g.groupby("year"))
        print(f"  {t:<6}{len(g):>6}{len(g) / len(r) * 100:>7.1f}%{g.both.mean() * 100:>7.1f}%"
              f"{g.clean.mean() * 100:>7.1f}%{g.clean.mean() * 100 - base:>+8.1f}   {yr}")
    picked = r[r.type.isin(["OTD", "ORR"])]
    rest = r[~r.type.isin(["OTD", "ORR"])]
    if len(picked) and len(rest):
        print(f"  the source's pick (OTD+ORR): n={len(picked)} clean {picked.clean.mean() * 100:.1f}%"
              f"  vs everything else n={len(rest)} clean {rest.clean.mean() * 100:.1f}%"
              f"  -> {picked.clean.mean() * 100 - rest.clean.mean() * 100:+.1f} points")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="NQ,DAX")
    ap.add_argument("--probe", type=float, default=0.10)
    ap.add_argument("--drive", type=float, default=0.25)
    ap.add_argument("--bars", type=int, default=6)
    args = ap.parse_args()
    for m in args.markets.split(","):
        if m in MARKETS:
            report(m, args.probe, args.drive, args.bars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
