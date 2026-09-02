"""Market Profile structures built from M1 bars, and used only where legal.

There is no volume-at-price feed here, but a TPO profile - time at price - needs
nothing more than one-minute highs and lows, and that is what Steidlmayer's
construction actually is. So POC, the value area, VAH, VAL, single prints, poor
highs and lows, naked POCs and the balance/trend distinction are all computable
from what we have. The volume-weighted variant is available too by spreading
each bar's tick_volume across its own range.

THE LOOKAHEAD RULE, which is the whole reason this file is careful:
a session's own profile is not known until the session ends, so it can never
filter a trade taken inside that session. What IS known at the open is the
PRIOR session's profile, complete, and the open's position relative to it.
Every feature this module exposes for filtering is therefore derived from
yesterday, and `session_features` separates the two explicitly - `prior_*`
fields are legal at the open, bare fields are for post-hoc study only.

Bin height is the prior 14-day range over 120, so a profile has roughly a
hundred rows on every market at every price level. A fixed point bin would be
120 rows on 2023 NQ and 190 rows on 2026 NQ - the same date-filter trap that
absolute thresholds create, moved into the histogram.

    python tools/mprofile.py --markets NDXUSD,DEUIDXEUR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from markets import MARKETS, hm, load_m1  # noqa: E402

VALUE_AREA = 0.70  # Steidlmayer's one standard deviation proxy
ROWS_TARGET = 120  # profile rows per prior-ATR of range


# ------------------------------------------------------------------ one profile


def build_profile(low: np.ndarray, high: np.ndarray, bin_h: float,
                  weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """TPO counts per price bin. Returns (bin_lows, counts, bin_h).

    Each bar contributes 1 (or its weight) to every bin its range covers, which
    is the letter-per-30-minutes construction at one-minute resolution.
    """
    lo, hi = float(np.min(low)), float(np.max(high))
    # floor, not ceil: the session high must land IN the top bin. With ceil the
    # profile always carries one empty row above the high, which reads as a
    # single print and makes a poor high impossible to detect.
    n = max(1, int(np.floor((hi - lo) / bin_h)) + 1)
    counts = np.zeros(n, dtype=float)
    i0 = np.floor((low - lo) / bin_h).astype(int)
    i1 = np.floor((high - lo) / bin_h).astype(int)
    w = np.ones(len(low)) if weight is None else weight
    for a, b, ww in zip(i0, i1, w, strict=True):
        counts[a : b + 1] += ww
    return lo + np.arange(n) * bin_h, counts, bin_h


def value_area(bin_lows: np.ndarray, counts: np.ndarray,
               frac: float = VALUE_AREA) -> tuple[float, float, float]:
    """(POC, VAH, VAL) by expanding from the POC toward the busier neighbour.

    This is the standard construction: start at the point of control and add
    whichever adjacent row holds more time, until `frac` of all time is
    enclosed. Expanding by pairs is the textbook version; single-row expansion
    is used here so a lopsided profile is not forced to grow symmetrically.
    """
    total = counts.sum()
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    p = int(np.argmax(counts))
    lo = hi = p
    acc = counts[p]
    target = frac * total
    while acc < target and (lo > 0 or hi < len(counts) - 1):
        left = counts[lo - 1] if lo > 0 else -1.0
        right = counts[hi + 1] if hi < len(counts) - 1 else -1.0
        if right >= left:
            hi += 1
            acc += counts[hi]
        else:
            lo -= 1
            acc += counts[lo]
    step = bin_lows[1] - bin_lows[0] if len(bin_lows) > 1 else 0.0
    return float(bin_lows[p] + step / 2), float(bin_lows[hi] + step), float(bin_lows[lo])


def extremes(counts: np.ndarray, single_max: float = 1.0) -> tuple[int, int, bool, bool]:
    """(single prints at the top, at the bottom, poor high, poor low).

    Excess - a tail of single prints - says the extreme was rejected and is
    unlikely to be revisited. A POOR high is the opposite: two or more rows of
    time right at the extreme, which Dalton reads as unfinished business and a
    level likely to be taken out later.
    """
    top = 0
    for c in counts[::-1]:
        if c <= single_max:
            top += 1
        else:
            break
    bot = 0
    for c in counts:
        if c <= single_max:
            bot += 1
        else:
            break
    return top, bot, top == 0, bot == 0


# -------------------------------------------------------------- session tables


def session_features(symbol: str) -> pd.DataFrame:
    """One row per session: its own profile, plus yesterday's shifted forward.

    Columns prefixed `prior_` and the `open_*` columns are known at the open and
    are the only ones legal as filters. Everything else describes the session
    itself and exists for measurement, not for trading.
    """
    m1 = load_m1(symbol)
    _, _, op, cl, tick = MARKETS[symbol]
    o, c = hm(op), hm(cl)

    day_rng = m1.groupby("d").agg(hi=("h", "max"), lo=("l", "min"))
    atr = (day_rng["hi"] - day_rng["lo"]).rolling(14).mean().shift(1)

    rows = []
    for d, g in m1.groupby("d"):
        s = g[(g.m >= o) & (g.m < c)]
        if len(s) < 120:
            continue
        a = atr.get(d, np.nan)
        if not np.isfinite(a) or a <= 0:
            continue
        bin_h = max(tick, a / ROWS_TARGET)
        lows = s["l"].to_numpy()
        highs = s["h"].to_numpy()
        bl, counts, _ = build_profile(lows, highs, bin_h)
        poc, vah, val = value_area(bl, counts)
        sp_top, sp_bot, poor_hi, poor_lo = extremes(counts)
        rng = float(highs.max() - lows.min())
        close = float(s["c"].iloc[-1])
        opn = float(s["o"].iloc[0])
        ib = s[s.m < o + 60]
        rows.append(
            {
                "d": d, "atr": a, "bin_h": bin_h,
                "open": opn, "close": close, "high": float(highs.max()), "low": float(lows.min()),
                "range": rng, "range_atr": rng / a,
                "poc": poc, "vah": vah, "val": val,
                "va_width": vah - val, "va_frac": (vah - val) / rng if rng > 0 else np.nan,
                "sp_top": sp_top, "sp_bot": sp_bot, "poor_high": poor_hi, "poor_low": poor_lo,
                "close_pos": (close - lows.min()) / rng if rng > 0 else np.nan,
                "ib_high": float(ib["h"].max()) if len(ib) else np.nan,
                "ib_low": float(ib["l"].min()) if len(ib) else np.nan,
            }
        )
    df = pd.DataFrame(rows).sort_values("d").reset_index(drop=True)
    if df.empty:
        return df

    df["ib_width_atr"] = (df.ib_high - df.ib_low) / df.atr

    # -- yesterday, shifted forward: this is what the open can legally see
    for col in ("poc", "vah", "val", "high", "low", "close", "range_atr", "va_frac", "close_pos",
                "poor_high", "poor_low"):
        df[f"prior_{col}"] = df[col].shift(1)

    # A trend day closes at an extreme on a wide range; a balance day holds most
    # of its time inside a narrow value area. Both are read off YESTERDAY.
    df["prior_trend_day"] = (
        (df.prior_range_atr > 1.1)
        & ((df.prior_close_pos > 0.8) | (df.prior_close_pos < 0.2))
    )
    df["prior_balance_day"] = (df.prior_va_frac > 0.55) & (df.prior_range_atr < 1.0)

    # Where did we open relative to yesterday's value?
    df["open_vs_value"] = np.where(
        df.open > df.prior_vah, "above",
        np.where(df.open < df.prior_val, "below", "inside"),
    )
    df["open_gap_atr"] = (df.open - df.prior_close) / df.atr
    df["open_dist_poc_atr"] = (df.open - df.prior_poc) / df.atr

    # -- naked POC: a prior POC no later session has traded through, and how far
    # away it is at today's open. Purely backward-looking at every step.
    naked_dist, naked_side = [], []
    pending: list[tuple[float, float]] = []  # (poc price, day index it came from)
    for i, r in df.iterrows():
        alive = [(p, j) for p, j in pending if not (r.prior_low <= p <= r.prior_high)] \
            if i > 0 and np.isfinite(r.prior_low) else list(pending)
        if alive:
            k = int(np.argmin([abs(r.open - p) for p, _ in alive]))
            p = alive[k][0]
            naked_dist.append((p - r.open) / r.atr)
            naked_side.append(1 if p > r.open else -1)
        else:
            naked_dist.append(np.nan)
            naked_side.append(0)
        if np.isfinite(r.prior_poc):
            alive.append((float(r.prior_poc), i))
        pending = alive[-40:]
    df["naked_poc_dist_atr"] = naked_dist
    df["naked_poc_side"] = naked_side
    return df


# ---------------------------------------------------------------------- report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="NDXUSD,DEUIDXEUR")
    args = ap.parse_args()
    for sym in args.markets.split(","):
        if sym not in MARKETS:
            continue
        df = session_features(sym)
        lab = MARKETS[sym][0]
        print(f"\n=== {lab} ({sym}): {len(df)} sessions with a full profile ===")
        print(f"  bin height: median {df.bin_h.median():.2f} points "
              f"({df.bin_h.median() / df.open.median() * 100:.4f}% of price)")
        print(f"  value area holds a median {df.va_frac.median() * 100:.0f}% of the session range")
        print(f"  session range: median {df.range_atr.median():.2f} of prior ATR")
        print(f"  IB width: median {df.ib_width_atr.median():.2f} of prior ATR")
        print(f"  poor high on {df.poor_high.mean() * 100:.0f}% of days, "
              f"poor low on {df.poor_low.mean() * 100:.0f}%")
        print(f"  single prints at the top: median {df.sp_top.median():.0f} rows, "
              f"bottom {df.sp_bot.median():.0f}")
        vc = df.open_vs_value.value_counts(normalize=True) * 100
        print("  open relative to yesterday's value area: "
              + "  ".join(f"{k} {v:.0f}%" for k, v in vc.items()))
        print(f"  yesterday was a trend day on {df.prior_trend_day.mean() * 100:.0f}% of days, "
              f"a balance day on {df.prior_balance_day.mean() * 100:.0f}%")
        nk = df.naked_poc_dist_atr.notna().mean() * 100
        print(f"  a naked POC exists on {nk:.0f}% of opens, median distance "
              f"{df.naked_poc_dist_atr.abs().median():.2f} ATR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
