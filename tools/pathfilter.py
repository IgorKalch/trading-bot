"""Does a heavy prior-value zone sitting in the way actually stop a trade?

Every filter tested so far asks WHERE THE SESSION OPENED. This asks a different
question, taken from the video at [54:18]: "we can get an open rejection reverse
which leads us to the balance of the previous day - and I get a stop." That is a
claim about the PATH, not the origin - the trade has to travel through a zone
where a lot of time was already spent, and the claim is that it does not get
through.

It is a genuinely different mechanism from the thirty cells already measured, so
it earns its own test. The prior is not kind: thirty pre-specified Market
Profile cells produced one apparent positive, which is what chance predicts, and
that one failed a dose-response test.

Two families here, five hypotheses, all pre-specified:

  OBSTRUCTION   the prior POC, or the prior value area, lies between the entry
                and the target. Veto, or require a clear path.
  BALANCE       a multi-day balance zone - the overlap of the last N value areas,
                which is what Dalton means by balance, not one session's value
                area - contains the open. Veto, per the video at [68:01]:
                "while price opens inside the balance, in most cases it's a skip".

Nothing here uses the session being traded: the zones come from prior sessions
and the entry, stop and target are fixed at the moment of entry.

    python tools/pathfilter.py
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

from markets import MARKETS, _walk, costs_for, hm, to_m5  # noqa: E402
from markets import load_m1 as _load_m1  # noqa: E402
from mprofile import session_features  # noqa: E402

CUT = pd.Period("2025-03", "M")
BALANCE_DAYS = 3  # how many prior sessions must overlap to call it balance


def drive_trades(symbol: str, cost: float, rr: float = 2.0, n_bars: int = 6) -> pd.DataFrame:
    """The drive model, but returning the geometry each trade committed to."""
    m1 = _load_m1(symbol)
    o = hm(MARKETS[symbol][2])
    c = hm(MARKETS[symbol][3]) - 5
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
        entry = float(w.iloc[-1]["c"])
        stop = float(w.l.min() if side > 0 else w.h.max())
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        r = _walk(rest, side, entry, stop, rr, cost)
        if r is None:
            continue
        rows.append({
            "d": d, "r": r, "side": side, "entry": entry, "stop": stop,
            "target": entry + side * rr * risk, "risk": risk,
        })
    return pd.DataFrame(rows).set_index("d")


def multiday_balance(feats: pd.DataFrame, days: int = BALANCE_DAYS) -> pd.DataFrame:
    """Overlap of the last `days` value areas, known before today opens.

    Dalton's balance is an area several sessions agree on, not one session's
    value area. If the value areas do not overlap there is no balance, and the
    columns come back as NaN - which is itself the answer on those days.
    """
    vah = feats["vah"].shift(1)
    val = feats["val"].shift(1)
    hi = vah.rolling(days).min()  # the lowest of the recent VAHs
    lo = val.rolling(days).max()  # the highest of the recent VALs
    ok = hi > lo
    return pd.DataFrame({"bal_hi": hi.where(ok), "bal_lo": lo.where(ok)}, index=feats.index)


def between(a: float, b: float, x: float) -> bool:
    return (min(a, b) < x < max(a, b))


def overlaps(a: float, b: float, lo: float, hi: float) -> bool:
    return not (max(a, b) <= lo or min(a, b) >= hi)


def hypotheses(f: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    poc_in_path = np.array([
        between(r.entry, r.target, r.prior_poc) if np.isfinite(r.prior_poc) else False
        for r in f.itertuples()
    ])
    va_in_path = np.array([
        overlaps(r.entry, r.target, r.prior_val, r.prior_vah)
        if np.isfinite(r.prior_val) and np.isfinite(r.prior_vah) else False
        for r in f.itertuples()
    ])
    poc_before_stop = np.array([
        between(r.entry, r.stop, r.prior_poc) if np.isfinite(r.prior_poc) else False
        for r in f.itertuples()
    ])
    open_in_balance = np.array([
        (r.bal_lo <= r.entry <= r.bal_hi)
        if np.isfinite(getattr(r, "bal_lo", np.nan)) and np.isfinite(getattr(r, "bal_hi", np.nan))
        else False
        for r in f.itertuples()
    ])
    has_balance = np.array([
        np.isfinite(getattr(r, "bal_lo", np.nan)) for r in f.itertuples()
    ])
    return [
        ("path CLEAR of prior POC",
         "the video's claim: a trade that must travel through yesterday's point "
         "of control gets stopped instead",
         ~poc_in_path),
        ("path BLOCKED by prior POC",
         "the inverse; if the claim is right this should be much worse",
         poc_in_path),
        ("path CLEAR of prior value area",
         "the same idea with the whole value area rather than its single "
         "busiest price",
         ~va_in_path),
        ("prior POC behind the stop",
         "a point of control between entry and stop should pull price back "
         "through the stop",
         poc_before_stop),
        ("open NOT inside a 3-day balance",
         "the video at 68:01 - opening inside balance is mostly a skip. Balance "
         "here is the overlap of three value areas, not one",
         ~open_in_balance),
        ("open inside a 3-day balance",
         "the inverse of the video's rule",
         open_in_balance),
        ("a 3-day balance exists at all",
         "control: does merely having a defined balance predict anything",
         has_balance),
    ]


def stats(r: np.ndarray, dates) -> dict:
    if len(r) < 30:
        return {"n": len(r)}
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
        "is": r[m].mean() if m.sum() > 20 else float("nan"),
        "oos": r[~m].mean() if (~m).sum() > 20 else float("nan"),
        "losses": neg,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="NDXUSD")
    args = ap.parse_args()
    for sym in args.markets.split(","):
        if sym not in MARKETS:
            continue
        cost = costs_for(sym)[0]
        t = drive_trades(sym, cost)
        feats = session_features(sym).set_index("d")
        bal = multiday_balance(feats)
        f = t.join(feats[["prior_poc", "prior_vah", "prior_val"]]).join(bal).dropna(subset=["r"])
        base = stats(f.r.to_numpy(), f.index)
        print(f"\n=== {MARKETS[sym][0]} / drive: {base['n']} trades, {base['exp']:+.4f}R, "
              f"drawdown {base['dd']:.1f}R, MAR {base['mar']:.2f} ===")
        hs = hypotheses(f)
        print(f"  {len(hs)} new pre-specified hypotheses, on top of the 30 already "
              f"tested - so 37 cells total in this family")
        inside = np.mean([
            (r.bal_lo <= r.entry <= r.bal_hi) if np.isfinite(r.bal_lo) else False
            for r in f.itertuples()
        ])
        print(f"  a 3-day balance is defined on {np.isfinite(f.bal_lo).mean() * 100:.0f}% "
              f"of sessions; the open sits inside it on {inside * 100:.0f}%")
        print(f"  {'filter':<32}{'kept':>6}{'%tr':>6}{'%loss':>7}{'WR':>7}{'PF':>6}"
              f"{'expR':>9}{'CI lo':>8}{'DD_R':>7}{'R/mo':>8}{'MAR':>7}{'IS':>8}{'OOS':>8}")
        print(f"  {'(unfiltered)':<32}{base['n']:>6}{100:>6.0f}{100:>7.0f}{base['wr']:>6.1f}%"
              f"{base['pf']:>6.2f}{base['exp']:>+9.4f}{base['lo']:>+8.4f}{base['dd']:>7.1f}"
              f"{base['rmo']:>+8.3f}{base['mar']:>7.2f}{base['is']:>+8.4f}{base['oos']:>+8.4f}")
        for name, _why, mask in hs:
            kept = f[mask]
            if len(kept) < 30:
                print(f"  {name:<32}{len(kept):>6}   too few kept")
                continue
            s = stats(kept.r.to_numpy(), kept.index)
            lk = -kept.r[kept.r < 0].sum()
            print(f"  {name:<32}{s['n']:>6}{s['n'] / base['n'] * 100:>6.0f}"
                  f"{lk / base['losses'] * 100:>7.0f}{s['wr']:>6.1f}%{s['pf']:>6.2f}"
                  f"{s['exp']:>+9.4f}{s['lo']:>+8.4f}{s['dd']:>7.1f}{s['rmo']:>+8.3f}"
                  f"{s['mar']:>7.2f}{s['is']:>+8.4f}{s['oos']:>+8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
