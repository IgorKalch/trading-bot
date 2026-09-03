"""Four standalone strategies against seven index markets, each judged alone.

Everything before this was a portfolio question - how legs combine. This asks
the prior question the portfolio work skipped: does each MODEL have an edge on
each MARKET, standing on its own? A leg that only earns its keep through
correlation is not a strategy, and the distinction matters because the two are
usually confused.

    retest        M1 opening-range retest and absorption (the engine's own)
    confluence    IB close vs mid, agreeing with which IB extreme printed first
    extension     first close beyond a WIDE Initial Balance, stop at IB mid
    dalton        Market Profile open types (OD / OTD / ORR / OA), traded in the
                  drive direction - the "Type Open" claim from TO_IB/iborrotd.mp4
    drive         the same entry with the classification DELETED, on every
                  session - which began as dalton's null model and beat it

Costs are measured, never assumed. Each market is run twice: at futures cost
(one tick of spread plus one of slippage) and at the prop CFD cost (the broker
spread measured inside the entry window, plus half again). Reality for a given
account is one of the two, so neither is averaged - and on most markets the gap
between them decides the whole question.

    python tools/markets.py
    python tools/markets.py --models dalton --grid
    python tools/validate_drive.py
"""

from __future__ import annotations

import argparse
import logging
import math
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

DATA_DIR = "data/cache_duka"
CUT = pd.Period("2025-03", "M")  # IS / OOS split, fixed once and never moved

# symbol -> (label, timezone, cash open, cash close, futures tick)
# The tick is the real exchange minimum increment of the matching future:
# NQ/ES 0.25, RTY 0.10, YM 1.0, FDAX 0.5, FESX 1.0, FTSE-100 0.5.
MARKETS: dict[str, tuple[str, str, str, str, float]] = {
    "NDXUSD": ("NQ", "America/New_York", "09:30", "16:00", 0.25),
    "SPXUSD": ("ES", "America/New_York", "09:30", "16:00", 0.25),
    "RTYUSD": ("RTY", "America/New_York", "09:30", "16:00", 0.10),
    "YMUSD": ("YM", "America/New_York", "09:30", "16:00", 1.00),
    "DEUIDXEUR": ("DAX", "Europe/Berlin", "09:00", "17:30", 0.50),
    "SX5EEUR": ("SX5E", "Europe/Berlin", "09:00", "17:30", 1.00),
    "UK100GBP": ("FTSE", "Europe/London", "08:00", "16:30", 0.50),
}

# Four markets ship no spread file. The three that do all sit within a narrow
# band of price - 0.0067% (NQ), 0.0088% (ES), 0.0078% (DAX) - so the missing
# ones are interpolated at the middle of that band rather than invented.
SPREAD_PCT_OF_PRICE = 0.00008

_cache: dict[str, pd.DataFrame] = {}


def hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def load_m1(symbol: str) -> pd.DataFrame:
    """M1 bars localised to the market timezone, with minutes-from-midnight."""
    if symbol in _cache:
        return _cache[symbol]
    _, tz, _, _, _ = MARKETS[symbol]
    df = pd.read_parquet(f"{DATA_DIR}/{symbol}_M1.parquet")
    loc = pd.to_datetime(df["time"], utc=True).dt.tz_convert(tz)
    out = pd.DataFrame(
        {
            "d": loc.dt.date,
            "m": loc.dt.hour * 60 + loc.dt.minute,
            "o": df["open"].to_numpy(),
            "h": df["high"].to_numpy(),
            "l": df["low"].to_numpy(),
            "c": df["close"].to_numpy(),
            "sp": df["spread_points"].to_numpy(),
        }
    )
    _cache[symbol] = out
    return out


def to_m5(m1: pd.DataFrame) -> pd.DataFrame:
    """Resample to 5 minutes; only NQ/ES/DAX ship a ready M5 file."""
    g = m1.assign(b=m1.m // 5).groupby(["d", "b"], sort=True)
    out = g.agg(m=("m", "first"), o=("o", "first"), h=("h", "max"), l=("l", "min"),
                c=("c", "last")).reset_index()
    return out.drop(columns="b")


def measured_spread(symbol: str) -> tuple[float, str]:
    """Median CFD spread inside the entry window, and how it was obtained.

    The M1 files carry an all-zero spread_points column; the real Dukascopy
    spreads live in a companion *_M1_spread.parquet whose timestamps were lost,
    but whose rows are the same count and the same order - so they join
    positionally onto the M1 timestamps and can be windowed properly.

    Windowing matters: NQ reads 3.20 over the whole day and far less inside the
    entry window. The daily average is the trap recorded for DAX (3.91 against
    a measured 1.51).
    """
    m1 = load_m1(symbol)
    _, _, op, _, _ = MARKETS[symbol]
    o = hm(op)
    path = Path(DATA_DIR) / f"{symbol}_M1_spread.parquet"
    if path.exists():
        sp = pd.to_numeric(pd.read_parquet(path)["spread"], errors="coerce").to_numpy()
        if len(sp) == len(m1):
            w = sp[((m1.m >= o) & (m1.m < o + 120)).to_numpy()]
            w = w[np.isfinite(w)]
            if len(w):
                return float(np.median(w)), "measured"
    return float(np.nanmedian(m1["c"])) * SPREAD_PCT_OF_PRICE, "interpolated"


def costs_for(symbol: str) -> tuple[float, float, float, str]:
    """(futures cost, CFD cost, raw spread, provenance), all in index points.

    Futures: one tick of spread plus one tick of slippage - what an exchange
    fill actually costs, and the frame every NQ number in reports/ uses.
    CFD: the measured broker spread plus half of it again for slippage - what a
    prop account actually costs. Reality for a given account is one of the two,
    not something in between, so both are reported and neither is averaged.
    """
    tick = MARKETS[symbol][4]
    sp, how = measured_spread(symbol)
    return 2.0 * tick, sp * 1.5, sp, how


# --------------------------------------------------------------------- models


def _walk(rows: pd.DataFrame, side: int, entry: float, stop: float, rr: float,
          cost: float) -> float | None:
    """Resolve one trade bar by bar; unresolved trades exit at the last close.

    Same-bar ambiguity is resolved AGAINST the trade: the stop is checked first,
    so a bar that spans both levels counts as a loss.
    """
    risk = abs(entry - stop)
    if risk <= 0 or rows.empty:
        return None
    tp = entry + side * rr * risk
    for _, b in rows.iterrows():
        if (b.l <= stop) if side > 0 else (b.h >= stop):
            return -1.0 - cost / risk
        if (b.h >= tp) if side > 0 else (b.l <= tp):
            return rr - cost / risk
    return (rows.iloc[-1]["c"] - entry) * side / risk - cost / risk


def _daily_atr(m1: pd.DataFrame) -> pd.Series:
    """14-day mean range, shifted one day so today never sees itself."""
    r = m1.groupby("d").agg(hi=("h", "max"), lo=("l", "min"))
    return (r["hi"] - r["lo"]).rolling(14).mean().shift(1)


def _sessions(symbol: str) -> tuple[dict, pd.Series]:
    m1 = load_m1(symbol)
    _, _, op, cl, _ = MARKETS[symbol]
    o, c = hm(op), hm(cl) - 5
    m5 = to_m5(m1)
    out = {}
    for d, g in m5.groupby("d"):
        s = g[(g.m >= o) & (g.m < c)].sort_values("m").reset_index(drop=True)
        if len(s) >= 60:
            out[d] = s
    return out, _daily_atr(m1)


def model_confluence(symbol: str, cost: float, rr: float = 1.5) -> pd.Series:
    sess, _ = _sessions(symbol)
    o = hm(MARKETS[symbol][2])
    res = {}
    for d, s in sess.items():
        ib = s[s.m < o + 60].reset_index(drop=True)
        rest = s[s.m >= o + 60].reset_index(drop=True)
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
        r = _walk(rest, side, rest.iloc[0]["o"], ibl if side > 0 else ibh, rr, cost)
        if r is not None:
            res[d] = r
    return pd.Series(res, dtype=float)


def model_extension(symbol: str, cost: float, rr: float = 1.0,
                    min_width_pct: float = 0.9) -> pd.Series:
    sess, _ = _sessions(symbol)
    o = hm(MARKETS[symbol][2])
    res = {}
    for d, s in sess.items():
        ib = s[s.m < o + 60].reset_index(drop=True)
        rest = s[s.m >= o + 60].reset_index(drop=True)
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
            r = _walk(rest.iloc[i + 1:], side, b.c, mid, rr, cost)
            if r is not None:
                res[d] = r
            break
    return pd.Series(res, dtype=float)


def classify(b: pd.DataFrame, open_px: float, atr: float,
             k_probe: float = 0.10, k_drive: float = 0.25) -> tuple[str, int]:
    """Dalton's four open types from the first n closed bars, ATR-scaled.

    OD  drives from the open and never trades back through it
    OTD a shallow probe against the eventual direction, then a drive
    ORR a real excursion one way, rejected, then a drive back through the open
    OA  no directional conviction

    Thresholds are fractions of the prior 14-day range so they do not silently
    become a date filter as the index level grows - the mistake caught in
    Додаток И and the reason absolute point thresholds are banned here.
    """
    up = b.h.max() - open_px
    dn = open_px - b.l.min()
    drive = 1 if b.c.iloc[-1] > open_px else -1
    after = b.iloc[1:]
    crossed = (after.l.min() <= open_px) if drive > 0 else (after.h.max() >= open_px)
    adverse = dn if drive > 0 else up
    favour = up if drive > 0 else dn
    if favour < k_drive * atr:
        return "OA", drive
    if not crossed and adverse < k_probe * atr:
        return "OD", drive
    if adverse >= k_drive * atr:
        return "ORR", drive
    return "OTD", drive


def model_dalton(symbol: str, cost: float, rr: float = 1.0,
                 take: tuple[str, ...] = ("OTD", "ORR"), n_bars: int = 6,
                 k_probe: float = 0.10, k_drive: float = 0.25) -> pd.Series:
    """Classify the open, then trade the drive direction with a structural stop.

    Entry is the close of the classification window, so nothing after it is
    used. The stop goes beyond the window's adverse extreme - for OTD that is
    the shallow probe, for ORR the rejected excursion, which is exactly the
    level the type says should not trade again.
    """
    sess, atr14 = _sessions(symbol)
    o = hm(MARKETS[symbol][2])
    res = {}
    for d, s in sess.items():
        atr = atr14.get(d, np.nan)
        if not np.isfinite(atr) or atr <= 0:
            continue
        w = s[s.m < o + n_bars * 5].reset_index(drop=True)
        rest = s[s.m >= o + n_bars * 5].reset_index(drop=True)
        if len(w) != n_bars or len(rest) < 24:
            continue
        t, side = classify(w, w.iloc[0]["o"], atr, k_probe, k_drive)
        if t not in take:
            continue
        entry = w.iloc[-1]["c"]
        stop = w.l.min() if side > 0 else w.h.max()
        r = _walk(rest, side, entry, stop, rr, cost)
        if r is not None:
            res[d] = r
    return pd.Series(res, dtype=float)


def model_retest(symbol: str, cost: float) -> pd.Series:
    """The engine's retest model, re-pointed at another market's bars.

    Reuses tested code rather than reimplementing the state machine, so the NQ
    row here should reproduce the published NQ number.
    """
    cfg = load_config("config/config.nq.yaml", ".env").model_copy(deep=True)
    _, tz, op, cl, _ = MARKETS[symbol]
    cfg.mt5.symbol = symbol
    cfg.backtest.data_dir = DATA_DIR
    cfg.session.timezone = tz
    cfg.session.open = op
    cfg.session.close = cl
    cfg.session.pos1_cutoff = f"{(hm(op) + 120) // 60:02d}:{(hm(op) + 120) % 60:02d}"
    cfg.session.pos2_cutoff = f"{(hm(cl) - 30) // 60:02d}:{(hm(cl) - 30) % 60:02d}"
    cfg.session.flat_time = f"{(hm(cl) - 10) // 60:02d}:{(hm(cl) - 10) % 60:02d}"
    cfg.strategy.name = "retest"
    cfg.strategy.timeframe = "M1"
    cfg.strategy.retest.max_pullback_bars = 6
    cfg.strategy.targets.tp_mode = "fixed_rr"
    cfg.strategy.targets.fixed_rr = 1.0
    cfg.strategy.filters.require_fvg = True
    cfg.strategy.filters.trend_ma_period = 89
    cfg.backtest.spread_points = cost / 1.5
    cfg.backtest.slippage_points = cost / 3.0
    trades = BacktestEngine(cfg, build_strategy(cfg.strategy), NewsCalendar.empty()).run(
        history.load_bars(DATA_DIR, symbol, "M1")
    ).trades
    res: dict = {}
    for t in trades:
        res[t.day] = res.get(t.day, 0.0) + t.result_r
    return pd.Series(res, dtype=float)


def model_drive(symbol: str, cost: float, rr: float = 3.0, n_bars: int = 6) -> pd.Series:
    """Dalton with the classification deleted - trade the drive on EVERY session.

    This began as the null model for model_dalton and outlived it. If sorting
    opens into Dalton types carried information, the classified subsets would
    beat this; they do not, so what is left is the honest statement of the only
    thing that measured positive: at n_bars past the open, take the direction
    the window moved, stop at its opposite extreme, target rr, exit at the close.

    It is intraday momentum, which is a published and independently replicated
    effect rather than anything found here - a point in its favour, not against.

    The target defaults to 3.0R, changed from 2.0R after tools/drivetarget.py
    swept it: drawdown 12.1R -> 6.8R, better in both halves, 2023 turned
    positive, and at prop CFD cost the 2.0R lower bound is negative while 3.0R
    clears zero. Adopted for the DRAWDOWN only - the expectancy gain is not
    significant (paired t = 1.57) and out-of-sample expectancy actually falls.
    3.0R is the middle of a flat 2.5-3.5 plateau, not the best cell, and
    removing the target entirely is decisively worse, which is what makes this
    an interior optimum rather than the outlier-carried illusion that killed 22
    trailing variants. See reports/backtest_drive_target.txt.
    """
    sess, _ = _sessions(symbol)
    o = hm(MARKETS[symbol][2])
    res = {}
    for d, s in sess.items():
        w = s[s.m < o + n_bars * 5].reset_index(drop=True)
        rest = s[s.m >= o + n_bars * 5].reset_index(drop=True)
        if len(w) != n_bars or len(rest) < 24:
            continue
        side = 1 if w.c.iloc[-1] > w.iloc[0]["o"] else -1
        r = _walk(rest, side, w.iloc[-1]["c"], w.l.min() if side > 0 else w.h.max(), rr, cost)
        if r is not None:
            res[d] = r
    return pd.Series(res, dtype=float)


MODELS = {
    "retest": model_retest,
    "confluence": model_confluence,
    "extension": model_extension,
    "dalton": model_dalton,
    "drive": model_drive,
}


# ---------------------------------------------------------------------- report


def summarise(r: pd.Series) -> dict:
    if len(r) < 20:
        return {"n": len(r)}
    mo = r.groupby([pd.Period(d, "M") for d in r.index]).sum()
    cum = peak = worst = 0.0
    for x in mo:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    se = r.std(ddof=1) / math.sqrt(len(r))
    is_, oos = r[[pd.Period(d, "M") < CUT for d in r.index]], r[[pd.Period(d, "M") >= CUT for d in r.index]]
    yr = {y: g.mean() for y, g in r.groupby([pd.Timestamp(d).year for d in r.index])}
    return {
        "n": len(r), "wr": (r > 0).mean() * 100, "pf": pos / neg if neg else float("inf"),
        "exp": r.mean(), "lo": r.mean() - 1.96 * se, "dd": -worst,
        "is": is_.mean() if len(is_) > 20 else float("nan"),
        "oos": oos.mean() if len(oos) > 20 else float("nan"),
        "yr": yr, "rmo": mo.mean(),
    }


def line(label: str, s: dict) -> str:
    if "wr" not in s:
        return f"  {label:<22}{s['n']:>6}   too few trades"
    yr = " ".join(f"{y % 100:02d}:{v:+.2f}" for y, v in sorted(s["yr"].items()))
    return (f"  {label:<22}{s['n']:>6}{s['wr']:>7.1f}%{s['pf']:>7.2f}{s['exp']:>+9.3f}"
            f"{s['lo']:>+8.3f}{s['dd']:>7.1f}{s['is']:>+8.3f}{s['oos']:>+8.3f}  {yr}")


HEAD = (f"  {'market':<22}{'n':>6}{'WR':>8}{'PF':>7}{'expR':>9}{'CI lo':>8}"
        f"{'DD_R':>7}{'IS':>8}{'OOS':>8}  by year")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="retest,confluence,extension,dalton,drive")
    ap.add_argument("--markets", default=",".join(MARKETS))
    ap.add_argument("--grid", action="store_true", help="dalton: type sets x targets")
    args = ap.parse_args()

    syms = [s for s in args.markets.split(",") if s in MARKETS]
    fut, cfd = {}, {}
    print("=== cost per market, in index points ===")
    print(f"  {'':6}{'symbol':<12}{'price':>9}{'spread':>9}{'source':>13}"
          f"{'futures':>10}{'CFD':>8}")
    for s in syms:
        f, c, sp, how = costs_for(s)
        fut[s], cfd[s] = f, c
        px = float(np.nanmedian(load_m1(s)["c"]))
        print(f"  {MARKETS[s][0]:<6}{s:<12}{px:>9.0f}{sp:>9.2f}{how:>13}{f:>10.2f}{c:>8.2f}")

    for name in args.models.split(","):
        fn = MODELS.get(name)
        if fn is None:
            continue
        for label, table in (("FUTURES COST", fut), ("PROP CFD COST", cfd)):
            print(f"\n=== {name} - {label} ===")
            print(HEAD)
            for s in syms:
                try:
                    r = fn(s, table[s])
                except Exception as exc:  # a market may lack the bars a model needs
                    print(f"  {MARKETS[s][0]:<22}   failed: {type(exc).__name__}: {exc}")
                    continue
                print(line(MARKETS[s][0], summarise(r)))

    if args.grid:
        print("\n=== dalton grid: which open types, and which target (measured cost) ===")
        sets = [("OTD+ORR (source)", ("OTD", "ORR")), ("OD only", ("OD",)),
                ("OD+OTD+ORR", ("OD", "OTD", "ORR")), ("ORR only", ("ORR",)),
                ("OTD only", ("OTD",)), ("OA only", ("OA",))]
        for tag, take in sets:
            for rr in (1.0, 1.5, 2.0):
                print(f"\n  -- {tag}, target {rr}R (futures cost)")
                print(HEAD)
                for s in syms:
                    r = model_dalton(s, fut[s], rr=rr, take=take)
                    print(line(MARKETS[s][0], summarise(r)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
