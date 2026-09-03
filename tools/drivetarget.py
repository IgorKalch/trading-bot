"""The drive model's target width, swept properly, because nobody read the column.

tools/validate_drive.py has printed a target plateau since the model was built,
and the drawdown column in it was never read: 2.0R shows 12.1R of drawdown and
3.0R shows 6.8R. Twelve angles of external search produced twenty candidate
filters and twenty kills, while a parameter of our own model sat unswept. That
is worth stating plainly rather than burying.

This checks the finding with the machinery that has killed every previous
candidate, on both arms, at both cost levels:

  the plateau         is it an interior plateau across adjacent cells, or a
                      single lucky threshold, or the "let winners run" illusion
  the no-cap control  removing the target entirely MUST be worse. If it is not,
                      the effect is outlier-carried and dies like the 22
                      trailing variants did
  trim20              delete the twenty best trades. Every give-back exit ever
                      measured here goes negative on this; a wide fixed target
                      should not
  IS / OOS and years  split fixed at 2025-03, never moved
  random-side null    same days, same stops, same exits, side by coin flip
  paired bootstrap    quarterly blocks, both arms on the SAME days, so the
                      drawdown difference is not two independent draws
  the portfolio       what it does to the three-leg drawdown, which is the
                      number the whole brief is about

    python tools/drivetarget.py
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

from markets import (  # noqa: E402
    MARKETS,
    costs_for,
    hm,
    load_m1,
    model_confluence,
    model_retest,
    to_m5,
)

CUT = pd.Period("2025-03", "M")
RNG = np.random.default_rng(20260903)


def drive(symbol: str, cost: float, rr: float | None = 2.0, n_bars: int = 6,
          flip: dict | None = None) -> pd.DataFrame:
    """rr=None means no target at all - the control arm that must be worse."""
    m1 = load_m1(symbol)
    o, c = hm(MARKETS[symbol][2]), hm(MARKETS[symbol][3]) - 5
    m5 = to_m5(m1)
    rows = []
    for d, g in m5.groupby("d"):
        s = g[(g.m >= o) & (g.m < c)].sort_values("m").reset_index(drop=True)
        if len(s) < 60:
            continue
        w = s[s.m < o + n_bars * 5].reset_index(drop=True)
        rest = s[s.m >= o + n_bars * 5].reset_index(drop=True)
        if len(w) != n_bars or len(rest) < 24:
            continue
        side = 1 if w.c.iloc[-1] > w.iloc[0]["o"] else -1
        if flip is not None:
            side = flip.get(d, side)
        entry = float(w.iloc[-1]["c"])
        stop = float(w.l.min() if side > 0 else w.h.max())
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        tp = None if rr is None else entry + side * rr * risk
        r, how = None, "close"
        for b in rest.itertuples():
            if (b.l <= stop) if side > 0 else (b.h >= stop):
                r, how = -1.0 - cost / risk, "stop"
                break
            if tp is not None and ((b.h >= tp) if side > 0 else (b.l <= tp)):
                r, how = rr - cost / risk, "target"
                break
        if r is None:
            r = (float(rest.iloc[-1]["c"]) - entry) * side / risk - cost / risk
        rows.append({"d": d, "r": r, "how": how})
    return pd.DataFrame(rows).set_index("d")


def monthly(r: pd.Series) -> pd.Series:
    return r.groupby([pd.Period(d, "M") for d in r.index]).sum()


def dd_of(mo: pd.Series) -> float:
    cum = peak = worst = 0.0
    for x in mo:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    return -worst


def summarise(t: pd.DataFrame) -> dict:
    r = t.r.to_numpy()
    mo = monthly(t.r)
    dd = dd_of(mo)
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    se = r.std(ddof=1) / math.sqrt(len(r))
    m = np.array([pd.Period(d, "M") < CUT for d in t.index])
    return {
        "n": len(r), "wr": (r > 0).mean() * 100, "pf": pos / neg if neg else float("inf"),
        "exp": r.mean(), "lo": r.mean() - 1.96 * se, "dd": dd, "rmo": mo.mean(),
        "mar": (mo.mean() * 12) / dd if dd > 0 else float("nan"),
        "trim20": np.sort(r)[:-20].mean(),
        "is": r[m].mean(), "oos": r[~m].mean(),
        "is_dd": dd_of(monthly(t.r[m])), "oos_dd": dd_of(monthly(t.r[~m])),
        "hit": (t.how == "target").mean() * 100,
        "at_close": (t.how == "close").mean() * 100,
        "yrs": {y: g.mean() for y, g in t.r.groupby([pd.Timestamp(d).year for d in t.index])},
    }


def line(tag: str, s: dict) -> str:
    return (f"  {tag:<10}{s['n']:>5}{s['wr']:>7.1f}%{s['pf']:>6.2f}{s['exp']:>+9.4f}"
            f"{s['lo']:>+8.4f}{s['dd']:>7.1f}{s['rmo']:>+8.3f}{s['mar']:>7.2f}"
            f"{s['trim20']:>+9.4f}{s['is']:>+8.4f}{s['oos']:>+8.4f}{s['hit']:>6.0f}%"
            f"{s['at_close']:>7.0f}%")


HEAD = (f"  {'target':<10}{'n':>5}{'WR':>8}{'PF':>6}{'expR':>9}{'CI lo':>8}{'DD_R':>7}"
        f"{'R/mo':>8}{'MAR':>7}{'trim20':>9}{'IS':>8}{'OOS':>8}{'hit':>7}{'close':>7}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NDXUSD")
    args = ap.parse_args()
    sym = args.symbol
    fut, cfd = costs_for(sym)[0], costs_for(sym)[1]

    for label, cost in (("FUTURES COST", fut), ("PROP CFD COST", cfd)):
        print(f"\n=== {MARKETS[sym][0]} drive, target sweep - {label} ({cost:.2f} pts) ===")
        print(HEAD)
        for rr in (1.0, 2.0, 2.5, 3.0, 3.5, 4.0):
            print(line(f"{rr:.1f}R", summarise(drive(sym, cost, rr=rr))))
        print(line("no cap", summarise(drive(sym, cost, rr=None))) + "   <- control")

    print("\n=== the two arms in detail, futures cost ===")
    a2 = drive(sym, fut, rr=2.0)
    a3 = drive(sym, fut, rr=3.0)
    for tag, t in (("2.0R (live)", a2), ("3.0R", a3)):
        s = summarise(t)
        print(f"  {tag:<12} by year: " + "  ".join(f"{y % 100:02d}:{v:+.3f}"
                                                   for y, v in sorted(s["yrs"].items())))
        print(f"  {'':12} drawdown in-sample {s['is_dd']:.1f}R, "
              f"out-of-sample {s['oos_dd']:.1f}R")

    # -- paired quarterly block bootstrap of the DRAWDOWN difference. Both arms
    # -- resampled on the SAME quarters, so this is one difference, not two draws.
    common = a2.index.intersection(a3.index)
    d2, d3 = a2.loc[common], a3.loc[common]
    q = pd.Series([pd.Period(d, "Q") for d in common])
    qs = list(q.unique())
    diffs, worse = [], 0
    for _ in range(3000):
        pick = [qs[i] for i in RNG.integers(0, len(qs), len(qs))]
        idx = np.concatenate([(q == p).to_numpy().nonzero()[0] for p in pick])
        m2 = monthly(pd.Series(d2.r.to_numpy()[idx], index=common[idx]))
        m3 = monthly(pd.Series(d3.r.to_numpy()[idx], index=common[idx]))
        dl = dd_of(m3) - dd_of(m2)
        diffs.append(dl)
        worse += dl > 0
    diffs = np.array(diffs)
    print("\n  paired quarterly bootstrap of the drawdown change (3.0R minus 2.0R), "
          "3000 draws:")
    print(f"    median {np.median(diffs):+.2f}R   95% [{np.percentile(diffs, 2.5):+.2f}, "
          f"{np.percentile(diffs, 97.5):+.2f}]   P(3.0R has the WORSE drawdown) = "
          f"{worse / len(diffs) * 100:.1f}%")

    # -- paired per-trade expectancy: the same days, so pair them
    both = d3.r.to_numpy() - d2.r.to_numpy()
    changed = (np.abs(both) > 1e-9).mean() * 100
    tstat = both.mean() / (both.std(ddof=1) / math.sqrt(len(both)))
    print(f"  paired per-trade expectancy change: {both.mean():+.4f}R, t = {tstat:.2f}, "
          f"outcome changed on {changed:.0f}% of trades")

    # -- random-side null on the chosen arm
    nulls = []
    for _ in range(120):
        flip = {d: (1 if RNG.random() < 0.5 else -1) for d in a3.index}
        nulls.append(drive(sym, fut, rr=3.0, flip=flip).r.mean())
    nulls = np.array(nulls)
    e3 = a3.r.mean()
    print(f"  random-side null at 3.0R: {nulls.mean():+.4f}R (sd {nulls.std(ddof=1):.4f}); "
          f"real is {(e3 - nulls.mean()) / nulls.std(ddof=1):.1f} sigma above it")

    # -- and the number the brief is actually about
    print("\n=== the three-leg portfolio, drive at 2.0R against 3.0R ===")
    legs = {"retest": model_retest(sym, fut), "confluence": model_confluence(sym, fut)}
    print(f"  {'drive target':<14}{'R/mo':>9}{'DD_R':>8}{'MAR':>7}{'R/mo at 10R DD':>16}")
    for tag, arm in (("2.0R", a2), ("3.0R", a3)):
        mo = pd.DataFrame({k: monthly(v) for k, v in {**legs, "drive": arm.r}.items()}).fillna(0.0)
        p = mo.sum(axis=1) / 3
        dd = dd_of(p)
        print(f"  {tag:<14}{p.mean():>+9.3f}{dd:>8.2f}{(p.mean() * 12) / dd:>7.2f}"
              f"{p.mean() * (10.0 / dd):>+16.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
