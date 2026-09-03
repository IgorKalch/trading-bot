"""Three video-sourced changes to the drive model, none of which need a filter.

The 460 findings from the second reading of TO_IB/iborrotd.mp4 contain one class
of rule the thirty-seven filter cells could not reach, because it is not a
filter at all - it is about WHEN TO STOP HOLDING:

  [07:09] "within the first two hours we get the main move in most cases ... after
          two hours of the open forming I skip all subsequent work and no longer
          open or add anything inside the session"
  [56:11] "usually I looked within the backtest up to one o'clock"

Our drive model holds to 15:55. If the presenter is right that the move is spent
by late morning, a shorter hold should keep most of the return and shed the
drawdown that accumulates while a dead position sits open. This costs no trades,
which makes it strictly more interesting than anything the filter sweep tried.

Two of his rejection rules are also tested here, and as DOSE-RESPONSES rather
than thresholds, because that is the test that killed the last lead:

  [13:02] "everything that opened with an aggressive impulse I also tried to skip"
  [11:57] "if the Initial Balance range was abnormally large ... I could skip"

Both are claims that a bigger opening move is worse. That is the opposite of
what a momentum model wants, so they are real predictions, not restatements -
and our own open-type work already agrees in one place: the Open Drive type,
the most aggressive open of the four, was the weakest at +0.003R.

    python tools/driveexit.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from markets import MARKETS, costs_for, hm, load_m1, to_m5  # noqa: E402

CUT = pd.Period("2025-03", "M")


def drive(symbol: str, cost: float, rr: float = 2.0, n_bars: int = 6,
          exit_minutes: int | None = None) -> pd.DataFrame:
    """The drive model with an optional hard time exit, in minutes from the open.

    exit_minutes=None keeps the original behaviour: hold to the session close.
    The stop and target are unchanged, so any difference is purely the hold.
    """
    m1 = load_m1(symbol)
    o, c = hm(MARKETS[symbol][2]), hm(MARKETS[symbol][3]) - 5
    day = m1.groupby("d").agg(hi=("h", "max"), lo=("l", "min"))
    atr = (day["hi"] - day["lo"]).rolling(14).mean().shift(1)
    m5 = to_m5(m1)
    cutoff = c if exit_minutes is None else min(c, o + exit_minutes)
    rows = []
    for d, g in m5.groupby("d"):
        s = g[(g.m >= o) & (g.m < c)].sort_values("m").reset_index(drop=True)
        if len(s) < 60:
            continue
        a = atr.get(d, np.nan)
        w = s[s.m < o + n_bars * 5].reset_index(drop=True)
        rest = s[(s.m >= o + n_bars * 5) & (s.m < cutoff)].reset_index(drop=True)
        if len(w) != n_bars or len(rest) < 3:
            continue
        side = 1 if w.c.iloc[-1] > w.iloc[0]["o"] else -1
        entry = float(w.iloc[-1]["c"])
        stop = float(w.l.min() if side > 0 else w.h.max())
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        tp = entry + side * rr * risk
        r = None
        for b in rest.itertuples():
            if (b.l <= stop) if side > 0 else (b.h >= stop):
                r = -1.0 - cost / risk
                break
            if (b.h >= tp) if side > 0 else (b.l <= tp):
                r = rr - cost / risk
                break
        if r is None:
            r = (float(rest.iloc[-1]["c"]) - entry) * side / risk - cost / risk
        rows.append({
            "d": d, "r": r, "risk_atr": risk / a if np.isfinite(a) and a > 0 else np.nan,
            # how hard the open drove, and how wide it ranged - his two claims
            "displacement_atr": abs(entry - float(w.iloc[0]["o"])) / a
            if np.isfinite(a) and a > 0 else np.nan,
            "window_width_atr": (float(w.h.max()) - float(w.l.min())) / a
            if np.isfinite(a) and a > 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("d")


def stats(r: np.ndarray, dates) -> dict:
    mo = pd.Series(r).groupby([pd.Period(d, "M") for d in dates]).sum()
    cum = peak = worst = 0.0
    for x in mo:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    dd = -worst
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    se = r.std(ddof=1) / math.sqrt(len(r))
    m = np.array([pd.Period(d, "M") < CUT for d in dates])
    return {
        "n": len(r), "wr": (r > 0).mean() * 100, "pf": pos / neg if neg else float("inf"),
        "exp": r.mean(), "lo": r.mean() - 1.96 * se, "dd": dd, "rmo": mo.mean(),
        "mar": (mo.mean() * 12) / dd if dd > 0 else float("nan"),
        "is": r[m].mean(), "oos": r[~m].mean(),
        "trim20": np.sort(r)[:-20].mean(),
    }


def dose(r: pd.Series, x: pd.Series, q: int = 5) -> list[dict]:
    ok = x.notna() & r.notna()
    if ok.sum() < q * 25:
        return []
    xs, rs = x[ok], r[ok]
    b = pd.qcut(xs, q, labels=False, duplicates="drop")
    out = []
    for k in sorted(pd.Series(b).dropna().unique()):
        m = (b == k).to_numpy()
        v = rs[m].to_numpy()
        out.append({"n": len(v), "lo": xs[m].min(), "hi": xs[m].max(),
                    "exp": v.mean(), "wr": (v > 0).mean() * 100})
    return out


def rho(rows: list[dict]) -> float:
    if len(rows) < 3:
        return float("nan")
    a = pd.Series(np.arange(len(rows), dtype=float)).rank().to_numpy()
    b = pd.Series([x["exp"] for x in rows]).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="NDXUSD")
    args = ap.parse_args()
    for sym in args.markets.split(","):
        if sym not in MARKETS:
            continue
        cost = costs_for(sym)[0]
        lab = MARKETS[sym][0]
        o = hm(MARKETS[sym][2])

        print(f"\n=== {lab}: how long to hold the drive model ===")
        print("  His claim is that the move is spent inside two hours. A shorter")
        print("  hold costs no trades, so this is a free comparison.")
        print(f"  {'exit at':<14}{'n':>6}{'WR':>7}{'PF':>6}{'expR':>9}{'CI lo':>8}"
              f"{'DD_R':>7}{'R/mo':>8}{'MAR':>7}{'trim20':>9}{'IS':>8}{'OOS':>8}")
        for mins, label in ((60, "open+1h"), (90, "open+1h30"), (120, "open+2h"),
                            (150, "open+2h30"), (180, "open+3h"), (210, "open+3h30"),
                            (240, "open+4h"), (330, "open+5h30"), (None, "session close")):
            t = drive(sym, cost, exit_minutes=mins)
            if len(t) < 100:
                continue
            s = stats(t.r.to_numpy(), t.index)
            clock = "15:55" if mins is None else f"{(o + mins) // 60:02d}:{(o + mins) % 60:02d}"
            print(f"  {label:<9}{clock:>5}{s['n']:>6}{s['wr']:>6.1f}%{s['pf']:>6.2f}"
                  f"{s['exp']:>+9.4f}{s['lo']:>+8.4f}{s['dd']:>7.1f}{s['rmo']:>+8.3f}"
                  f"{s['mar']:>7.2f}{s['trim20']:>+9.4f}{s['is']:>+8.4f}{s['oos']:>+8.4f}")

        t = drive(sym, cost)
        print(f"\n=== {lab}: his two rejection claims, as dose-responses ===")
        print("  Both say a BIGGER opening move is worse. Quintiles, smallest first.")
        for var, why in (("displacement_atr", "how far the 30-minute window drove"),
                         ("window_width_atr", "how wide the 30-minute window ranged")):
            rows = dose(t.r, t[var])
            if not rows:
                print(f"  {var:<20} too few observations")
                continue
            r_ = rho(rows)
            cells = "  ".join(f"[{x['n']:>3}] {x['exp']:+.3f}" for x in rows)
            verdict = ("MONOTONE" if abs(r_) >= 0.9 else
                       "gradient" if abs(r_) >= 0.7 else "no gradient")
            print(f"  {var:<20} {cells}   rho {r_:+.2f}  {verdict}")
            print(f"  {'':20} ({why}, {rows[0]['lo']:.2f}..{rows[-1]['hi']:.2f} ATR)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
