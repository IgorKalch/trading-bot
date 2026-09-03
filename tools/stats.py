"""The three legs, per leg and combined: how often they trade and what they pay.

One table, honestly denominated. Two things are easy to get wrong here and both
have been got wrong in this project before:

  RISK PER LEG vs RISK TOTAL. A three-leg portfolio at "1% risk" can mean 1% on
  each leg or 1% split across them. The combined rows below split the risk, so
  their R figures are directly comparable to the single-leg rows and the money
  column answers the question actually asked - what does the account do.

  TRADES PER MONTH IS NOT TRADES PER SESSION. The legs fire at very different
  rates: the retest leg needs a specific sequence and skips most days, while
  drive and confluence take something almost every session.

    python tools/stats.py
    python tools/stats.py --risk 0.6
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
    model_confluence,
    model_drive,
    model_retest,
)

CUT = pd.Period("2025-03", "M")


def monthly(r: pd.Series) -> pd.Series:
    return r.groupby([pd.Period(d, "M") for d in r.index]).sum()


def dd_of(mo: pd.Series) -> float:
    cum = peak = worst = 0.0
    for x in mo:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    return -worst


def money(mo: pd.Series, risk_pct: float) -> tuple[float, float]:
    """Compounded annual return and peak-to-trough equity drawdown, in percent."""
    eq = peak = 1.0
    mdd = 0.0
    for x in mo:
        eq *= 1 + risk_pct / 100 * x
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak)
    return (eq ** (12 / len(mo)) - 1) * 100, mdd * 100


def leg_stats(r: pd.Series, sessions: int) -> dict:
    v = r.to_numpy()
    mo = monthly(r)
    dd = dd_of(mo)
    pos, neg = v[v > 0].sum(), -v[v < 0].sum()
    se = v.std(ddof=1) / math.sqrt(len(v))
    m = np.array([pd.Period(d, "M") < CUT for d in r.index])
    return {
        "n": len(v), "months": len(mo), "per_mo": len(v) / len(mo),
        "per_session": len(v) / sessions,
        "wr": (v > 0).mean() * 100, "pf": pos / neg if neg else float("inf"),
        "exp": v.mean(), "lo": v.mean() - 1.96 * se,
        "rmo": mo.mean(), "dd": dd, "mar": (mo.mean() * 12) / dd if dd > 0 else float("nan"),
        "best": mo.max(), "worst": mo.min(), "pos_mo": (mo > 0).mean() * 100,
        "is": v[m].mean() if m.sum() > 20 else float("nan"),
        "oos": v[~m].mean() if (~m).sum() > 20 else float("nan"),
        "yrs": {y: g.sum() for y, g in r.groupby([pd.Timestamp(d).year for d in r.index])},
        "mo": mo,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NDXUSD")
    ap.add_argument("--risk", type=float, default=0.6,
                    help="percent of the account risked per 1R, per leg")
    args = ap.parse_args()
    sym, lab = args.symbol, MARKETS[args.symbol][0]
    fut, cfd = costs_for(sym)[0], costs_for(sym)[1]

    for cost_label, cost in (("FUTURES COST", fut), ("PROP CFD COST", cfd)):
        legs = {
            "1 retest (M1, 1R)": model_retest(sym, cost),
            "2 IB confluence (1.5R)": model_confluence(sym, cost),
            "3 drive (30min, 3R)": model_drive(sym, cost),
        }
        sessions = max(len(v) for v in legs.values())
        st = {k: leg_stats(v, sessions) for k, v in legs.items()}
        mo = pd.DataFrame({k: v["mo"] for k, v in st.items()}).fillna(0.0)
        n_months = len(mo)

        print(f"\n{'=' * 104}")
        print(f"=== {lab}, {cost_label} ({cost:.2f} points) - "
              f"{n_months} months, 2023-01 to 2026-08 ===")
        print(f"{'=' * 104}")
        print(f"  {'leg':<24}{'trades':>7}{'per mo':>8}{'WR':>7}{'PF':>6}{'expR':>9}"
              f"{'CI lo':>8}{'R/mo':>8}{'DD_R':>7}{'MAR':>6}{'+mo':>6}{'best':>7}{'worst':>7}")
        for k, s in st.items():
            print(f"  {k:<24}{s['n']:>7}{s['per_mo']:>8.1f}{s['wr']:>6.1f}%{s['pf']:>6.2f}"
                  f"{s['exp']:>+9.4f}{s['lo']:>+8.4f}{s['rmo']:>+8.3f}{s['dd']:>7.1f}"
                  f"{s['mar']:>6.2f}{s['pos_mo']:>5.0f}%{s['best']:>+7.1f}{s['worst']:>+7.1f}")

        # -- combined, risk SPLIT across the enabled legs so R stays comparable
        print(f"\n  {'combined (risk split equally)':<24}{'trades':>7}{'per mo':>8}"
              f"{'':>7}{'':>6}{'':>9}{'':>8}{'R/mo':>8}{'DD_R':>7}{'MAR':>6}{'+mo':>6}"
              f"{'best':>7}{'worst':>7}")
        combos = [("1+2", ["1 retest (M1, 1R)", "2 IB confluence (1.5R)"]),
                  ("1+3", ["1 retest (M1, 1R)", "3 drive (30min, 3R)"]),
                  ("2+3", ["2 IB confluence (1.5R)", "3 drive (30min, 3R)"]),
                  ("all three", list(st))]
        rows = {}
        for tag, cols in combos:
            p = mo[cols].sum(axis=1) / len(cols)
            dd = dd_of(p)
            trades = sum(st[c]["n"] for c in cols)
            rows[tag] = (p, dd)
            print(f"  {tag:<24}{trades:>7}{trades / n_months:>8.1f}{'':>7}{'':>6}{'':>9}"
                  f"{'':>8}{p.mean():>+8.3f}{dd:>7.1f}{(p.mean() * 12) / dd:>6.2f}"
                  f"{(p > 0).mean() * 100:>5.0f}%{p.max():>+7.1f}{p.min():>+7.1f}")

        print(f"\n  monthly correlation between the legs "
              f"(mean pairwise {mo.corr().values[np.triu_indices(3, 1)].mean():+.2f}):")
        print("   " + mo.corr().round(2).to_string().replace("\n", "\n   "))

        # -- the money, which is what the question is about
        p, dd = rows["all three"]
        print(f"\n  --- money on all three legs, risk SPLIT so each leg takes "
              f"{args.risk / 3:.2f}% per 1R ---")
        print(f"  {'total risk per 1R':<22}{'R/mo':>8}{'%/year':>9}{'max DD %':>10}"
              f"{'worst month %':>15}")
        for tot in (0.5, 1.0, 1.5, 2.0, 3.0):
            cagr, mdd = money(p, tot)
            print(f"  {f'{tot:.1f}% total':<22}{p.mean():>+8.3f}{cagr:>+9.1f}{mdd:>10.1f}"
                  f"{p.min() * tot / 100 * 100:>15.1f}")

        print(f"\n  --- and each leg alone at {args.risk:.1f}% per 1R, for comparison ---")
        print(f"  {'leg':<24}{'R/mo':>8}{'%/year':>9}{'max DD %':>10}")
        for k, s in st.items():
            cagr, mdd = money(s["mo"], args.risk)
            print(f"  {k:<24}{s['rmo']:>+8.3f}{cagr:>+9.1f}{mdd:>10.1f}")
        cagr, mdd = money(p, args.risk)
        print(f"  {'all three (split)':<24}{p.mean():>+8.3f}{cagr:>+9.1f}{mdd:>10.1f}")

        print("\n  --- R per calendar year, per leg and combined ---")
        yrs = sorted({y for s in st.values() for y in s["yrs"]})
        print(f"  {'leg':<24}" + "".join(f"{y:>10}" for y in yrs))
        for k, s in st.items():
            print(f"  {k:<24}" + "".join(f"{s['yrs'].get(y, 0.0):>+10.1f}" for y in yrs))
        py = p.groupby(p.index.year).sum()
        print(f"  {'all three (split)':<24}" + "".join(f"{py.get(y, 0.0):>+10.1f}" for y in yrs))
        print(f"  {'IS -> OOS expectancy':<24}" + "  ".join(
            f"{k.split()[1][:6]} {s['is']:+.3f}->{s['oos']:+.3f}" for k, s in st.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
