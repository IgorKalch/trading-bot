"""Validate the one candidate the Dalton grid actually produced.

The candidate is NOT Dalton's: it is what is left when the classification is
deleted. At 30 minutes past the open, go in the direction the first 30 minutes
moved, stop at that window's opposite extreme, target 2R, forced exit at the
session close. No open type, no filter.

The checks are the ones that killed earlier candidates: a random-side null with
identical stops and exit machinery, trim20, IS/OOS, and a block bootstrap by
quarter so autocorrelation inside a quarter cannot inflate the interval.
"""
import sys

sys.path.insert(0, "tools")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from markets import MARKETS, _sessions, _walk, costs_for, hm

ALL = ("OD", "OTD", "ORR", "OA")
RNG = np.random.default_rng(20260902)


def drive_trades(symbol, cost, rr=2.0, n_bars=6, side_override=None):
    """Returns a DataFrame of one trade per session: date and R."""
    sess, _ = _sessions(symbol)
    o = hm(MARKETS[symbol][2])
    rows = []
    for d, s in sess.items():
        w = s[s.m < o + n_bars * 5].reset_index(drop=True)
        rest = s[s.m >= o + n_bars * 5].reset_index(drop=True)
        if len(w) != n_bars or len(rest) < 24:
            continue
        side = 1 if w.c.iloc[-1] > w.iloc[0]["o"] else -1
        if side_override is not None:
            side = side_override(d)
        entry = w.iloc[-1]["c"]
        stop = w.l.min() if side > 0 else w.h.max()
        r = _walk(rest, side, entry, stop, rr, cost)
        if r is not None:
            rows.append({"d": d, "r": r, "side": side})
    return pd.DataFrame(rows)


def dd_of(df):
    mo = df.groupby([pd.Period(d, "M") for d in df.d]).r.sum()
    cum = peak = worst = 0.0
    for x in mo:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    return -worst, mo.mean()


def main():
    sym = "NDXUSD"
    cost = costs_for(sym)[0]
    t = drive_trades(sym, cost)
    r = t.r.to_numpy()
    n = len(r)
    exp = r.mean()
    se = r.std(ddof=1) / np.sqrt(n)
    dd, rmo = dd_of(t)
    print(f"=== NQ, 30-minute drive direction, target 2R, cost {cost} ===")
    print(f"  n={n}  WR {(r > 0).mean() * 100:.1f}%  expectancy {exp:+.4f}R  "
          f"SE {se:.4f}  95% CI [{exp - 1.96 * se:+.4f}, {exp + 1.96 * se:+.4f}]")
    print(f"  R/month {rmo:+.3f}  max DD {dd:.1f}R  MAR {(rmo * 12) / dd:.2f}")
    print(f"  longs {int((t.side > 0).sum())} at {r[t.side > 0].mean():+.3f}R, "
          f"shorts {int((t.side < 0).sum())} at {r[t.side < 0].mean():+.3f}R")

    trim = np.sort(r)[:-20].mean()
    print(f"\n  trim20 (delete the 20 best trades): {trim:+.4f}R  "
          f"{'survives' if trim > 0 else 'DIES - the edge was a few runners'}")

    yr = t.assign(y=[pd.Timestamp(d).year for d in t.d]).groupby("y").r.agg(["mean", "count"])
    print("  by year: " + "  ".join(f"{y}:{row['mean']:+.3f}(n={int(row['count'])})"
                                    for y, row in yr.iterrows()))

    cut = pd.Period("2025-03", "M")
    m = [pd.Period(d, "M") < cut for d in t.d]
    print(f"  IS n={sum(m)} {r[np.array(m)].mean():+.4f}   "
          f"OOS n={n - sum(m)} {r[~np.array(m)].mean():+.4f}")

    # -- null model: same days, same stops, same exits, side chosen by coin flip
    draws = 200
    print(f"\n  --- random-side null, {draws} draws, identical stops and exits ---")
    nulls = []
    for _ in range(draws):
        flip = {d: (1 if RNG.random() < 0.5 else -1) for d in t.d}
        nt = drive_trades(sym, cost, side_override=flip.__getitem__)
        nulls.append(nt.r.mean())
    nulls = np.array(nulls)
    sigma = (exp - nulls.mean()) / nulls.std(ddof=1)
    print(f"  null expectancy {nulls.mean():+.4f}R  sd {nulls.std(ddof=1):.4f}  "
          f"real is {sigma:.1f} sigma above null")

    # -- block bootstrap by quarter
    q = pd.Series([pd.Period(d, "Q") for d in t.d])
    blocks = [r[(q == qq).to_numpy()] for qq in q.unique()]
    boot = np.array([
        np.concatenate([blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))]).mean()
        for _ in range(4000)
    ])
    print(f"\n  quarterly block bootstrap: 95% CI [{np.percentile(boot, 2.5):+.4f}, "
          f"{np.percentile(boot, 97.5):+.4f}]  P(<=0) = {(boot <= 0).mean() * 100:.1f}%")

    # -- is the 2R target a plateau or a spike?
    print("\n  --- target plateau ---")
    for rr in (1.0, 1.5, 2.0, 2.5, 3.0):
        tt = drive_trades(sym, cost, rr=rr)
        d2, m2 = dd_of(tt)
        print(f"  {rr:>4.1f}R  n={len(tt)}  WR {(tt.r > 0).mean() * 100:>5.1f}%  "
              f"{tt.r.mean():+.4f}R  DD {d2:>5.1f}R  R/mo {m2:+.3f}")

    # -- and the window length
    print("\n  --- classification window (bars of 5 minutes) ---")
    for nb in (3, 6, 9, 12, 18):
        tt = drive_trades(sym, cost, n_bars=nb)
        d2, m2 = dd_of(tt)
        print(f"  {nb * 5:>3} min  n={len(tt)}  WR {(tt.r > 0).mean() * 100:>5.1f}%  "
              f"{tt.r.mean():+.4f}R  DD {d2:>5.1f}R  R/mo {m2:+.3f}")


if __name__ == "__main__":
    main()
