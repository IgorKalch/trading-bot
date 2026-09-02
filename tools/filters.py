"""Profile-derived filters, applied to the live models, judged on what they reject.

The brief is specific: raise profitability, or cut drawdown by REJECTING trades
that lose. That makes the interesting number not expectancy but the trade-off -
what fraction of trades a rule removes against what fraction of the LOSSES it
removes. A rule that removes 30% of trades and 30% of the losses has done
nothing except shrink the sample.

Every filter here is decidable at the moment the model decides, and that is
enforced by construction rather than by care: the features come from
profile.session_features and only the `prior_*`, `open_*` and naked-POC columns
are used, all of which are complete before the session's first tick. The one
exception is flagged in the table - `window_width_atr` is measured on the drive
model's own 30-minute window, which closes before its entry.

The hypotheses are PRE-SPECIFIED from Market Profile theory rather than scanned,
which is what keeps the multiple-testing count honest: nine rules per model,
each with a stated reason to exist, and the count is printed with the results.

    python tools/filters.py
    python tools/filters.py --models drive --markets NDXUSD
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
    model_drive,
    model_retest,
    to_m5,
)
from mprofile import session_features  # noqa: E402

CUT = pd.Period("2025-03", "M")

MODELS = {"drive": model_drive, "retest": model_retest, "confluence": model_confluence}


def window_width(symbol: str, n_bars: int = 6) -> pd.Series:
    """Width of the drive model's own classification window, in prior ATR."""
    m1 = load_m1(symbol)
    o = hm(MARKETS[symbol][2])
    day = m1.groupby("d").agg(hi=("h", "max"), lo=("l", "min"))
    atr = (day["hi"] - day["lo"]).rolling(14).mean().shift(1)
    m5 = to_m5(m1)
    out = {}
    for d, g in m5.groupby("d"):
        w = g[(g.m >= o) & (g.m < o + n_bars * 5)]
        a = atr.get(d, np.nan)
        if len(w) == n_bars and np.isfinite(a) and a > 0:
            out[d] = (w.h.max() - w.l.min()) / a
    return pd.Series(out, dtype=float)


def trade_side(symbol: str, n_bars: int = 6) -> pd.Series:
    """+1/-1 direction the drive model would take, so filters can use it."""
    m1 = load_m1(symbol)
    o = hm(MARKETS[symbol][2])
    m5 = to_m5(m1)
    out = {}
    for d, g in m5.groupby("d"):
        w = g[(g.m >= o) & (g.m < o + n_bars * 5)].reset_index(drop=True)
        if len(w) == n_bars:
            out[d] = 1 if w.c.iloc[-1] > w.iloc[0]["o"] else -1
    return pd.Series(out, dtype=float)


# Each entry: label, reason it should work, and a mask over the joined frame.
def hypotheses(f: pd.DataFrame) -> list[tuple[str, str, pd.Series]]:
    return [
        ("open outside value",
         "an open beyond yesterday's value area says price is no longer accepted "
         "where it was - Dalton's precondition for a directional day",
         f.open_vs_value != "inside"),
        ("open inside value",
         "the inverse of the above; if the theory is right this should be WORSE",
         f.open_vs_value == "inside"),
        ("open above VAH only",
         "opening above value is the long-side version of the same precondition",
         f.open_vs_value == "above"),
        ("side agrees with open vs prior POC",
         "trade only when the session opened on the same side of yesterday's "
         "point of control as the direction taken",
         np.sign(f.open_dist_poc_atr) == np.sign(f.side)),
        ("yesterday was a balance day",
         "balance precedes breakout - the most repeated claim in the tradition",
         f.prior_balance_day.fillna(False).astype(bool)),
        ("yesterday was NOT a trend day",
         "a trend day rarely follows a trend day, so skip the day after one",
         ~f.prior_trend_day.fillna(False).astype(bool)),
        ("gap under 0.25 ATR",
         "a large gap has already made the move the model is trying to catch",
         f.open_gap_atr.abs() < 0.25),
        ("heading toward a naked POC",
         "an untested point of control is claimed to act as a magnet, so the "
         "direction pointing at one should reach further",
         (f.naked_poc_side == f.side) & (f.naked_poc_dist_atr.abs() < 1.0)),
        ("yesterday's range under 1.0 ATR",
         "a quiet prior session leaves room; an exhausted one does not",
         f.prior_range_atr < 1.0),
        ("window width 0.15-0.45 ATR",
         "too narrow means no conviction, too wide means the move is spent",
         (f.window_width_atr > 0.15) & (f.window_width_atr < 0.45)),
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
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    se = r.std(ddof=1) / math.sqrt(len(r))
    m = np.array([pd.Period(d, "M") < CUT for d in dates])
    yrs = {}
    for y, g in pd.Series(r).groupby([pd.Timestamp(d).year for d in dates]):
        yrs[y] = g.mean()
    dd = -worst
    return {
        "n": len(r), "wr": (r > 0).mean() * 100, "pf": pos / neg if neg else float("inf"),
        "exp": r.mean(), "lo": r.mean() - 1.96 * se, "dd": dd, "rmo": mo.mean(),
        # MAR is the only metric that settles a filter: it holds the drawdown
        # constant, so a rule cannot look good merely by trading less.
        "mar": (mo.mean() * 12) / dd if dd > 0 else float("nan"),
        "is": r[m].mean() if m.sum() > 20 else float("nan"),
        "oos": r[~m].mean() if (~m).sum() > 20 else float("nan"),
        "yrs": yrs, "losses": neg,
    }


def run(symbol: str, model: str) -> None:
    cost = costs_for(symbol)[0]
    tr = MODELS[model](symbol, cost)
    if len(tr) < 60:
        print(f"  {MARKETS[symbol][0]} / {model}: only {len(tr)} trades, skipped")
        return
    feats = session_features(symbol).set_index("d")
    f = pd.DataFrame({"r": tr})
    f = f.join(feats, how="inner")
    f["window_width_atr"] = window_width(symbol)
    f["side"] = trade_side(symbol)
    f = f.dropna(subset=["r"])

    base = stats(f.r.to_numpy(), f.index)
    lab = MARKETS[symbol][0]
    print(f"\n=== {lab} / {model}: {base['n']} trades, {base['exp']:+.4f}R, "
          f"drawdown {base['dd']:.1f}R, losses total {base['losses']:.1f}R ===")
    hs = hypotheses(f)
    print(f"  {len(hs)} pre-specified hypotheses; at 5% roughly "
          f"{len(hs) * 0.05:.1f} false positives are expected by chance")
    print(f"  {'filter':<34}{'kept':>6}{'%tr':>6}{'%loss':>7}{'WR':>7}{'PF':>6}"
          f"{'expR':>9}{'CI lo':>8}{'DD_R':>7}{'R/mo':>8}{'MAR':>7}{'IS':>8}{'OOS':>8}")
    print(f"  {'(unfiltered)':<34}{base['n']:>6}{100:>6.0f}{100:>7.0f}{base['wr']:>6.1f}%"
          f"{base['pf']:>6.2f}{base['exp']:>+9.4f}{base['lo']:>+8.4f}{base['dd']:>7.1f}"
          f"{base['rmo']:>+8.3f}{base['mar']:>7.2f}{base['is']:>+8.4f}{base['oos']:>+8.4f}")
    rows = []
    for name, _why, mask in hs:
        m = mask.fillna(False).to_numpy().astype(bool)
        kept = f[m]
        if len(kept) < 30:
            print(f"  {name:<34}{len(kept):>6}   too few kept")
            continue
        s = stats(kept.r.to_numpy(), kept.index)
        rej = f[~m]
        loss_kept = -kept.r[kept.r < 0].sum()
        print(f"  {name:<34}{s['n']:>6}{s['n'] / base['n'] * 100:>6.0f}"
              f"{loss_kept / base['losses'] * 100:>7.0f}{s['wr']:>6.1f}%{s['pf']:>6.2f}"
              f"{s['exp']:>+9.4f}{s['lo']:>+8.4f}{s['dd']:>7.1f}{s['rmo']:>+8.3f}"
              f"{s['mar']:>7.2f}{s['is']:>+8.4f}{s['oos']:>+8.4f}")
        rows.append((name, s, loss_kept / base["losses"] * 100,
                     s["n"] / base["n"] * 100, rej.r.mean() if len(rej) else float("nan")))

    # A filter earns its place only by raising MAR, so rank on that. Ranking on
    # expectancy would crown whichever rule trades least.
    print("\n  --- ranked by MAR, the only test a filter can pass ---")
    for name, s, lpct, tpct, rej_exp in sorted(
        rows, key=lambda x: -(x[1]["mar"] if np.isfinite(x[1]["mar"]) else -9)
    )[:4]:
        verdict = "BEATS the unfiltered model" if s["mar"] > base["mar"] else \
            "loses to the unfiltered model"
        print(f"  {name}: MAR {base['mar']:.2f} -> {s['mar']:.2f} - {verdict}")
        sep = ("separates winners from losers" if abs(lpct - tpct) > 8
               else "no separation - it only shrinks the sample")
        print(f"      keeps {tpct:.0f}% of trades and {lpct:.0f}% of the losses ({sep})")
        print(f"      expectancy {base['exp']:+.4f} -> {s['exp']:+.4f}, "
              f"drawdown {base['dd']:.1f}R -> {s['dd']:.1f}R, "
              f"R/mo {base['rmo']:+.3f} -> {s['rmo']:+.3f}")
        rej_note = ("the rejected trades LOST money, so rejecting them was right"
                    if rej_exp < 0 else
                    f"the rejected trades still MADE {rej_exp:+.4f}R, so this "
                    "throws away profit")
        print(f"      {rej_note}")
        print("      by year: " + "  ".join(f"{y % 100:02d}:{v:+.3f}" for y, v in sorted(s["yrs"].items())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="drive,retest,confluence")
    ap.add_argument("--markets", default="NDXUSD")
    args = ap.parse_args()
    for model in args.models.split(","):
        if model not in MODELS:
            continue
        for sym in args.markets.split(","):
            if sym in MARKETS:
                run(sym, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
