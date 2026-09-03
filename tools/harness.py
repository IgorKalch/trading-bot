"""A harness that accepts rules which CHANGE an outcome, not only rules that delete a trade.

Why this exists. tools/filters.py is a boolean mask over a fixed vector of trade
outcomes. That shape can express "skip this trade" and nothing else, so across
two sessions twenty proposals were all written as entry gates, all failed with
the identical signature - percent of trades removed equal to percent of losses
removed - and meanwhile the exit surface of the newest and largest leg was never
swept. When it finally was, one number cut the drawdown by 44%. The harness
shaped the research, so the harness is the thing to fix.

The primitive here is not (R, mask). It is (trade geometry, remaining bar path),
and a RULE computes R itself. That makes these all expressible in one place:

    a filter          skip the trade            -> returns None
    a target change   1R vs 3R vs no cap        -> different exit price
    a time exit       flat at 11:30             -> truncated path
    a trailing stop   move the stop as it goes  -> mutating stop
    a partial         half off at 0.5R          -> blended outcome
    a composition     gate(predicate, exit_rule)

Four things are enforced structurally rather than left to the caller, because
each one has already cost this project a retraction or a wasted week:

 1. THE R DENOMINATOR IS FROZEN AT ENTRY. Trade.risk is set once from the
    initial stop and every rule divides by it. Measuring R against a trailed
    stop once turned -0.003R into +0.55R here.
 2. THE STOP IS CHECKED BEFORE THE TARGET on any bar that spans both, so
    same-bar ambiguity always resolves against the trade.
 3. EVERY COMPARISON IS PAIRED. Both arms run on the same days, so the drawdown
    difference is one distribution and not two independent draws.
 4. A RANDOM-REJECTION NULL IS AUTOMATIC for any rule that skips trades. A
    filter must beat the 95th percentile of deleting the same fraction of
    sessions at random; three earlier kills needed this and computed it ad hoc.

The separation gate reports N/A rather than a number when a rule mutates
outcomes, because "percent of losses removed" is undefined when no trade was
removed. Reporting a plausible-looking number there is how a harness lies.

    python tools/harness.py                      # demo: reproduce the 3.0R result
    python tools/harness.py --leg confluence
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from markets import MARKETS, costs_for, hm, load_m1, to_m5  # noqa: E402

CUT = pd.Period("2025-03", "M")
RNG = np.random.default_rng(20260903)


# ------------------------------------------------------------------- the trade


@dataclass(frozen=True)
class Trade:
    """One trade's committed geometry, plus the bars it will live through.

    risk is frozen here and is the ONLY denominator any rule may use. bars is
    the path after entry, with columns m/o/h/l/c.
    """

    day: object
    side: int  # +1 long, -1 short
    entry: float
    stop: float
    risk: float
    cost: float  # in price points, charged once
    bars: pd.DataFrame
    atr: float

    def r_at(self, price: float) -> float:
        return (price - self.entry) * self.side / self.risk - self.cost / self.risk


Rule = Callable[[Trade], "Outcome | None"]


@dataclass(frozen=True)
class Outcome:
    r: float
    how: str  # stop | target | time | trail | partial | close
    minutes_held: int


# ------------------------------------------------------------------ the rules


def _hits_stop(b, side: int, stop: float) -> bool:
    return (b.l <= stop) if side > 0 else (b.h >= stop)


def _hits_target(b, side: int, tp: float) -> bool:
    return (b.h >= tp) if side > 0 else (b.l <= tp)


def fixed_target(rr: float | None) -> Rule:
    """The baseline family. rr=None means no target: hold to the last bar."""

    def rule(t: Trade) -> Outcome | None:
        tp = None if rr is None else t.entry + t.side * rr * t.risk
        for i, b in enumerate(t.bars.itertuples()):
            if _hits_stop(b, t.side, t.stop):
                return Outcome(-1.0 - t.cost / t.risk, "stop", (i + 1) * 5)
            if tp is not None and _hits_target(b, t.side, tp):
                return Outcome(rr - t.cost / t.risk, "target", (i + 1) * 5)
        last = float(t.bars.iloc[-1]["c"])
        return Outcome(t.r_at(last), "close", len(t.bars) * 5)

    return rule


def time_exit(minutes_after_entry: int, rr: float | None = None) -> Rule:
    """Flat after a fixed hold, with an optional target still active."""

    def rule(t: Trade) -> Outcome | None:
        n = max(1, minutes_after_entry // 5)
        path = t.bars.iloc[:n]
        tp = None if rr is None else t.entry + t.side * rr * t.risk
        for i, b in enumerate(path.itertuples()):
            if _hits_stop(b, t.side, t.stop):
                return Outcome(-1.0 - t.cost / t.risk, "stop", (i + 1) * 5)
            if tp is not None and _hits_target(b, t.side, tp):
                return Outcome(rr - t.cost / t.risk, "target", (i + 1) * 5)
        return Outcome(t.r_at(float(path.iloc[-1]["c"])), "time", len(path) * 5)


    return rule


def trail(start_r: float, step_r: float, rr: float | None = None) -> Rule:
    """Ratchet the stop once price has travelled start_r, then every step_r.

    The stop MOVES but the denominator does not - t.risk stays the initial risk,
    which is the bug this harness exists partly to make impossible.
    """

    def rule(t: Trade) -> Outcome | None:
        stop = t.stop
        best = 0.0
        tp = None if rr is None else t.entry + t.side * rr * t.risk
        for i, b in enumerate(t.bars.itertuples()):
            if _hits_stop(b, t.side, stop):
                return Outcome((stop - t.entry) * t.side / t.risk - t.cost / t.risk,
                               "trail" if stop != t.stop else "stop", (i + 1) * 5)
            if tp is not None and _hits_target(b, t.side, tp):
                return Outcome(rr - t.cost / t.risk, "target", (i + 1) * 5)
            reach = (b.h if t.side > 0 else b.l)
            best = max(best, (reach - t.entry) * t.side / t.risk)
            if best >= start_r:
                steps = math.floor((best - start_r) / step_r) + 1
                lifted = t.entry + t.side * (steps - 1) * step_r * t.risk
                stop = max(stop, lifted) if t.side > 0 else min(stop, lifted)
        return Outcome(t.r_at(float(t.bars.iloc[-1]["c"])), "close", len(t.bars) * 5)

    return rule


def partial(at_r: float, frac: float, breakeven: bool, rr: float) -> Rule:
    """Take `frac` off at at_r, optionally move the rest to breakeven."""

    def rule(t: Trade) -> Outcome | None:
        first = t.entry + t.side * at_r * t.risk
        tp = t.entry + t.side * rr * t.risk
        stop = t.stop
        booked = 0.0
        remaining = 1.0
        for i, b in enumerate(t.bars.itertuples()):
            if _hits_stop(b, t.side, stop):
                r_stop = (stop - t.entry) * t.side / t.risk
                return Outcome(booked + remaining * r_stop - t.cost / t.risk,
                               "stop", (i + 1) * 5)
            if remaining == 1.0 and _hits_target(b, t.side, first):
                booked = frac * at_r
                remaining = 1.0 - frac
                if breakeven:
                    stop = t.entry
            if _hits_target(b, t.side, tp):
                return Outcome(booked + remaining * rr - t.cost / t.risk,
                               "target", (i + 1) * 5)
        last = (float(t.bars.iloc[-1]["c"]) - t.entry) * t.side / t.risk
        return Outcome(booked + remaining * last - t.cost / t.risk, "close",
                       len(t.bars) * 5)

    return rule


def gate(predicate: Callable[[Trade], bool], inner: Rule) -> Rule:
    """Compose a skip condition with an outcome rule - a filter, expressed here."""

    def rule(t: Trade) -> Outcome | None:
        return inner(t) if predicate(t) else None

    return rule


# --------------------------------------------------------------- trade sources


def drive_trades(symbol: str, cost: float, n_bars: int = 6) -> list[Trade]:
    """The drive leg: direction of the first n_bars, stop at its far extreme."""
    m1 = load_m1(symbol)
    o, c = hm(MARKETS[symbol][2]), hm(MARKETS[symbol][3]) - 5
    day = m1.groupby("d").agg(hi=("h", "max"), lo=("l", "min"))
    atr14 = (day["hi"] - day["lo"]).rolling(14).mean().shift(1)
    m5 = to_m5(m1)
    out = []
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
        out.append(Trade(d, side, entry, stop, risk, cost, rest,
                         float(atr14.get(d, np.nan))))
    return out


def confluence_trades(symbol: str, cost: float) -> list[Trade]:
    """The IB confluence leg: close vs IB mid agreeing with which extreme came first."""
    m1 = load_m1(symbol)
    o, c = hm(MARKETS[symbol][2]), hm(MARKETS[symbol][3]) - 5
    day = m1.groupby("d").agg(hi=("h", "max"), lo=("l", "min"))
    atr14 = (day["hi"] - day["lo"]).rolling(14).mean().shift(1)
    m5 = to_m5(m1)
    out = []
    for d, g in m5.groupby("d"):
        s = g[(g.m >= o) & (g.m < c)].sort_values("m").reset_index(drop=True)
        if len(s) < 60:
            continue
        ib = s[s.m < o + 60].reset_index(drop=True)
        rest = s[s.m >= o + 60].reset_index(drop=True)
        if len(ib) != 12 or len(rest) < 30:
            continue
        ibh, ibl = float(ib.h.max()), float(ib.l.min())
        il, ih = int(ib["l"].idxmin()), int(ib["h"].idxmax())
        if il == ih:
            continue
        bull = ib.iloc[-1]["c"] > (ibh + ibl) / 2 and il < ih
        bear = ib.iloc[-1]["c"] < (ibh + ibl) / 2 and ih < il
        if not (bull or bear):
            continue
        side = 1 if bull else -1
        entry = float(rest.iloc[0]["o"])
        stop = ibl if side > 0 else ibh
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        out.append(Trade(d, side, entry, stop, risk, cost, rest,
                         float(atr14.get(d, np.nan))))
    return out


SOURCES = {"drive": drive_trades, "confluence": confluence_trades}


# ------------------------------------------------------------------ evaluation


def run(trades: list[Trade], rule: Rule) -> pd.DataFrame:
    rows = []
    for t in trades:
        o = rule(t)
        if o is None:
            continue
        rows.append({"d": t.day, "r": o.r, "how": o.how, "held": o.minutes_held})
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


def describe(t: pd.DataFrame) -> dict:
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
        "trim20": np.sort(r)[:-20].mean() if len(r) > 25 else float("nan"),
        "is": r[m].mean() if m.sum() > 20 else float("nan"),
        "oos": r[~m].mean() if (~m).sum() > 20 else float("nan"),
        "losses": neg, "months": len(mo),
    }


def compare(trades: list[Trade], base: Rule, test: Rule, base_name: str,
            test_name: str, draws: int = 2000) -> None:
    """Paired evaluation of a rule against a named benchmark on the same days."""
    b, t = run(trades, base), run(trades, test)
    sb, st = describe(b), describe(t)
    denom = np.median([x.risk for x in trades])
    print(f"\n  benchmark: {base_name}    candidate: {test_name}")
    print(f"  R denominator: median initial risk {denom:.1f} index points "
          f"({np.median([x.risk / x.atr for x in trades if np.isfinite(x.atr)]):.2f} of prior ATR)")
    hd = (f"  {'arm':<26}{'n':>5}{'WR':>7}{'PF':>6}{'expR':>9}{'CI lo':>8}{'DD_R':>7}"
          f"{'R/mo':>8}{'MAR':>7}{'trim20':>9}{'IS':>8}{'OOS':>8}")
    print(hd)
    for tag, s in ((base_name, sb), (test_name, st)):
        print(f"  {tag[:25]:<26}{s['n']:>5}{s['wr']:>6.1f}%{s['pf']:>6.2f}{s['exp']:>+9.4f}"
              f"{s['lo']:>+8.4f}{s['dd']:>7.1f}{s['rmo']:>+8.3f}{s['mar']:>7.2f}"
              f"{s['trim20']:>+9.4f}{s['is']:>+8.4f}{s['oos']:>+8.4f}")

    # -- separation gate: only meaningful when the rule DELETES trades
    skipped = sb["n"] - st["n"]
    if skipped > 0:
        kept_loss = -t.r[t.r < 0].sum()
        pct_tr = st["n"] / sb["n"] * 100
        pct_loss = kept_loss / sb["losses"] * 100
        sep = pct_loss - pct_tr
        print(f"  separation: keeps {pct_tr:.0f}% of trades and {pct_loss:.0f}% of the "
              f"losses -> {sep:+.1f} points "
              f"({'PASSES the 8-point gate' if sep <= -8 else 'FAILS - no separation'})")
        rejected = b.loc[b.index.difference(t.index)]
        if len(rejected):
            verdict = ("they lost money, so rejecting them was right"
                       if rejected.r.mean() < 0
                       else "they MADE money - this discards profit")
            print(f"  the rejected trades averaged {rejected.r.mean():+.4f}R ({verdict})")
        # -- random-rejection null: delete the same fraction at random
        frac = st["n"] / sb["n"]
        rnd = []
        for _ in range(400):
            keep = RNG.random(len(b)) < frac
            if keep.sum() < 30:
                continue
            rnd.append(b.r.to_numpy()[keep].mean())
        rnd = np.array(rnd)
        pct = (rnd < st["exp"]).mean() * 100
        print(f"  random-rejection null at the same {frac * 100:.0f}% keep rate: "
              f"mean {rnd.mean():+.4f}R, 95th pct {np.percentile(rnd, 95):+.4f}R; "
              f"the rule sits at the {pct:.0f}th percentile "
              f"({'beats it' if pct >= 95 else 'does NOT beat random rejection'})")
    else:
        print("  separation: N/A - this rule mutates outcomes and deletes no trades, "
              "so 'percent of losses removed' is undefined")

    # -- paired bootstrap on the shared days
    common = b.index.intersection(t.index)
    if len(common) < 100:
        print("  paired bootstrap: too few shared days")
        return
    bb, tt = b.loc[common], t.loc[common]
    paired = tt.r.to_numpy() - bb.r.to_numpy()
    changed = (np.abs(paired) > 1e-9).mean() * 100
    tstat = paired.mean() / (paired.std(ddof=1) / math.sqrt(len(paired))) \
        if paired.std(ddof=1) > 0 else float("nan")
    print(f"  paired per-trade change on {len(common)} shared days: {paired.mean():+.4f}R, "
          f"t = {tstat:.2f}, outcome changed on {changed:.0f}% of them")
    q = pd.Series([pd.Period(d, "Q") for d in common])
    qs = list(q.unique())
    dds, exps, worse = [], [], 0
    for _ in range(draws):
        pick = [qs[i] for i in RNG.integers(0, len(qs), len(qs))]
        idx = np.concatenate([(q == p).to_numpy().nonzero()[0] for p in pick])
        mb = monthly(pd.Series(bb.r.to_numpy()[idx], index=common[idx]))
        mt = monthly(pd.Series(tt.r.to_numpy()[idx], index=common[idx]))
        d = dd_of(mt) - dd_of(mb)
        dds.append(d)
        exps.append(tt.r.to_numpy()[idx].mean() - bb.r.to_numpy()[idx].mean())
        worse += d > 0
    dds, exps = np.array(dds), np.array(exps)
    print(f"  paired quarterly bootstrap, {draws} draws, both arms on the same quarters:")
    print(f"    drawdown change  median {np.median(dds):+.2f}R   "
          f"95% [{np.percentile(dds, 2.5):+.2f}, {np.percentile(dds, 97.5):+.2f}]   "
          f"P(candidate WORSE) = {worse / len(dds) * 100:.1f}%")
    print(f"    expectancy change median {np.median(exps):+.4f}R   "
          f"95% [{np.percentile(exps, 2.5):+.4f}, {np.percentile(exps, 97.5):+.4f}]   "
          f"P(<=0) = {(exps <= 0).mean() * 100:.1f}%")
    print(f"  trim20: {sb['trim20']:+.4f} -> {st['trim20']:+.4f} "
          f"({'survives' if st['trim20'] > 0 else 'DIES - the edge is a few runners'})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NDXUSD")
    ap.add_argument("--leg", default="drive", choices=sorted(SOURCES))
    args = ap.parse_args()
    cost = costs_for(args.symbol)[0]
    trades = SOURCES[args.leg](args.symbol, cost)
    print(f"=== harness demo: {MARKETS[args.symbol][0]} / {args.leg}, "
          f"{len(trades)} trades, cost {cost:.2f} points ===")
    print("  Each block below is a rule the OLD harness could not express at all.")

    base = fixed_target(2.0)
    compare(trades, base, fixed_target(3.0), "target 2.0R", "target 3.0R")
    compare(trades, base, fixed_target(None), "target 2.0R", "no target (control)")
    compare(trades, base, time_exit(120, rr=2.0), "target 2.0R", "flat 2h after entry")
    compare(trades, base, trail(1.0, 0.5, rr=None), "target 2.0R", "trail from 1R step 0.5R")
    compare(trades, base, partial(1.0, 0.5, True, 3.0), "target 2.0R",
            "half at 1R, rest to 3R, BE")
    compare(trades, base, gate(lambda t: t.risk < t.atr * 0.35, fixed_target(3.0)),
            "target 2.0R", "3.0R, only risk < 0.35 ATR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
