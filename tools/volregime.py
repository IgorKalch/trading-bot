"""Follow-up on the one lead the profile-filter sweep left: prior volatility.

Thirty pre-specified Market Profile cells produced exactly one where the
rejected trades actually lost money - the confluence model with yesterday's
range under 1.0 ATR. That rule is not Market-Profile-shaped at all; it is a
volatility-regime filter. This file tests it as what it is.

The test is deliberately NOT another threshold. A threshold scan on a lead found
in thirty cells would find a better-looking number and mean nothing. What is
hard to fake is a DOSE-RESPONSE: sort every trade into quintiles of the
conditioning variable and ask whether expectancy moves monotonically across
them. A single lucky bucket is noise; a gradient across five is a relationship.
Thresholds are reported afterwards, and only for the variables that show a
gradient first.

Everything here is known before the session opens: yesterday's range, and ratios
of trailing average ranges. Nothing uses the session being traded.

    python tools/volregime.py
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
    load_m1,
    model_confluence,
    model_drive,
    model_retest,
)

CUT = pd.Period("2025-03", "M")
MODELS = {"drive": model_drive, "retest": model_retest, "confluence": model_confluence}


def regime_features(symbol: str) -> pd.DataFrame:
    """Trailing volatility measures, every one lagged so today cannot see itself."""
    m1 = load_m1(symbol)
    day = m1.groupby("d").agg(hi=("h", "max"), lo=("l", "min"), c=("c", "last"))
    rng = day["hi"] - day["lo"]
    atr14 = rng.rolling(14).mean()
    atr5 = rng.rolling(5).mean()
    atr40 = rng.rolling(40).mean()
    ret = np.log(day["c"]).diff()
    out = pd.DataFrame(
        {
            # yesterday's range against the trailing mean it belongs to
            "prior_range_atr": (rng / atr14.shift(1)).shift(1),
            # is volatility rising or falling into today
            "atr5_over_atr40": (atr5 / atr40).shift(1),
            "atr14_over_atr40": (atr14 / atr40).shift(1),
            # close-to-close realised vol, short over long
            "rv5_over_rv20": (ret.rolling(5).std() / ret.rolling(20).std()).shift(1),
            # absolute level of volatility, as a fraction of price
            "atr14_pct": (atr14 / day["c"]).shift(1) * 100,
        }
    )
    return out


def dose_response(r: pd.Series, x: pd.Series, q: int = 5) -> list[dict]:
    """Expectancy per quintile of x, lowest bucket first."""
    ok = x.notna() & r.notna()
    if ok.sum() < q * 25:
        return []
    xs, rs = x[ok], r[ok]
    try:
        bins = pd.qcut(xs, q, labels=False, duplicates="drop")
    except ValueError:
        return []
    rows = []
    for b in sorted(pd.Series(bins).dropna().unique()):
        m = (bins == b).to_numpy()
        v = rs[m].to_numpy()
        if len(v) < 20:
            continue
        se = v.std(ddof=1) / math.sqrt(len(v))
        rows.append({
            "bucket": int(b), "n": len(v), "lo": xs[m].min(), "hi": xs[m].max(),
            "exp": v.mean(), "se": se, "wr": (v > 0).mean() * 100,
        })
    return rows


def spearman(rows: list[dict]) -> float:
    """Rank correlation between bucket order and expectancy - the gradient test."""
    if len(rows) < 3:
        return float("nan")
    a = np.arange(len(rows), dtype=float)
    b = pd.Series([x["exp"] for x in rows]).rank().to_numpy()
    a = pd.Series(a).rank().to_numpy()
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="NDXUSD")
    ap.add_argument("--models", default="confluence,retest,drive")
    args = ap.parse_args()

    VARS = ["prior_range_atr", "atr5_over_atr40", "atr14_over_atr40",
            "rv5_over_rv20", "atr14_pct"]

    for sym in args.markets.split(","):
        if sym not in MARKETS:
            continue
        feats = regime_features(sym)
        cost = costs_for(sym)[0]
        for model in args.models.split(","):
            if model not in MODELS:
                continue
            tr = MODELS[model](sym, cost)
            if len(tr) < 150:
                continue
            df = pd.DataFrame({"r": tr}).join(feats, how="left")
            base = df.r.mean()
            print(f"\n=== {MARKETS[sym][0]} / {model}: {len(df)} trades, "
                  f"unconditional {base:+.4f}R ===")
            print("  quintiles of each trailing-volatility measure, lowest first."
                  "\n  A gradient across five buckets is a relationship; one good "
                  "bucket is noise.")
            for v in VARS:
                rows = dose_response(df.r, df[v])
                if not rows:
                    print(f"  {v:<20} too few observations")
                    continue
                rho = spearman(rows)
                cells = "  ".join(
                    f"[{x['n']:>3}] {x['exp']:+.3f}" for x in rows
                )
                rng = f"{rows[0]['lo']:.2f}..{rows[-1]['hi']:.2f}"
                verdict = ("MONOTONE" if abs(rho) >= 0.9 else
                           "gradient" if abs(rho) >= 0.7 else "no gradient")
                print(f"  {v:<20} {cells}   rho {rho:+.2f}  {verdict}   range {rng}")

            # Only for variables with a gradient, show what a split would do -
            # and always with the out-of-sample half separated.
            for v in VARS:
                rows = dose_response(df.r, df[v])
                if not rows or abs(spearman(rows)) < 0.7:
                    continue
                print(f"\n  --- {v} showed a gradient, so here is the split, "
                      f"with IS/OOS ---")
                med = df[v].median()
                for label, mask in (("below median", df[v] < med), ("above median", df[v] >= med)):
                    sub = df[mask.fillna(False)]
                    if len(sub) < 50:
                        continue
                    m = np.array([pd.Period(d, "M") < CUT for d in sub.index])
                    se = sub.r.std(ddof=1) / math.sqrt(len(sub))
                    yrs = "  ".join(
                        f"{y % 100:02d}:{g.mean():+.3f}"
                        for y, g in sub.r.groupby([pd.Timestamp(d).year for d in sub.index])
                    )
                    print(f"    {label:<14} n={len(sub):>4}  {sub.r.mean():+.4f}R  "
                          f"CI lo {sub.r.mean() - 1.96 * se:+.4f}  "
                          f"IS {sub.r[m].mean() if m.sum() > 20 else float('nan'):+.4f}  "
                          f"OOS {sub.r[~m].mean() if (~m).sum() > 20 else float('nan'):+.4f}  "
                          f"| {yrs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
