"""Exhaustive exit x filter grid search over the cached backtest data.

Analysis tooling, not part of the bot: it never touches strategy logic, it only
builds config variants and runs the same BacktestEngine the CLI runs.

The grid is large by construction, so the point of this script is NOT the
leaderboard — with hundreds of variants over one year of trades the best row is
almost certainly noise (STRATEGY.md §7). Every combination is therefore scored
on an in-sample slice AND on a held-out out-of-sample tail, so a result can be
asked the only question that matters: did it survive on data it never saw.

    python tools/sweep.py --out reports/sweep.csv
"""

from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tradingbot.backtest.engine import BacktestEngine  # noqa: E402
from tradingbot.config import AppConfig, load_config  # noqa: E402
from tradingbot.data import history  # noqa: E402
from tradingbot.data.news import NewsCalendar  # noqa: E402
from tradingbot.strategy.orb import OrbStrategy  # noqa: E402

# --- grid -------------------------------------------------------------------

EXITS = [
    ("fixed_1.0R", {"tp_mode": "fixed_rr", "fixed_rr": 1.0}),
    ("fixed_1.5R", {"tp_mode": "fixed_rr", "fixed_rr": 1.5}),
    ("fixed_2.0R", {"tp_mode": "fixed_rr", "fixed_rr": 2.0}),
    ("trail_0.5/0.5", {"tp_mode": "trailing", "trail_start_r": 0.5, "trail_step_r": 0.5}),
    ("trail_0.5/0.25", {"tp_mode": "trailing", "trail_start_r": 0.5, "trail_step_r": 0.25}),
    ("trail_0.5/1.0", {"tp_mode": "trailing", "trail_start_r": 0.5, "trail_step_r": 1.0}),
    ("trail_1.0/0.5", {"tp_mode": "trailing", "trail_start_r": 1.0, "trail_step_r": 0.5}),
]

# Each axis keeps its disabled value first, so the all-first combination is the
# unfiltered baseline.
FILTER_AXES = {
    "rvol": ("min_or_rvol", [0.0, 0.8, 1.0, 1.2]),
    "or_max": ("max_or_width_points", [0.0, 70.0, 100.0]),
    "or_min": ("min_or_width_points", [0.0, 30.0]),
    "gap_max": ("max_gap_points", [0.0, 40.0, 80.0]),
    "or_dir": ("or_direction_filter", [False, True]),
    "vwap": ("vwap_filter", [False, True]),
    "skip_wd": ("skip_weekdays", [[], [0], [3], [4]]),
}

_CTX: dict = {}


def _init(config_path: str, split_iso: str) -> None:
    cfg = load_config(config_path)
    bars = history.load_bars(cfg.backtest.data_dir, cfg.mt5.symbol, cfg.strategy.timeframe)
    _CTX["cfg"] = cfg
    _CTX["bars"] = bars
    _CTX["split"] = date.fromisoformat(split_iso)


def _variant(cfg: AppConfig, exit_kw: dict, filt_kw: dict) -> AppConfig:
    c = cfg.model_copy(deep=True)
    for k, v in exit_kw.items():
        setattr(c.strategy.targets, k, v)
    for k, v in filt_kw.items():
        setattr(c.strategy.filters, k, v)
    return c


def _stats(rs: list[float]) -> dict:
    n = len(rs)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "exp": 0.0, "sumR": 0.0}
    wins = [r for r in rs if r > 0]
    gross_win = sum(wins)
    gross_loss = -sum(r for r in rs if r < 0)
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    return {
        "n": n,
        "wr": len(wins) / n * 100.0,
        "pf": pf,
        "exp": sum(rs) / n,
        "sumR": sum(rs),
    }


def _max_dd_r(rs: list[float]) -> float:
    """Max drawdown of the cumulative R curve, in R."""
    peak = cum = 0.0
    worst = 0.0
    for r in rs:
        cum += r
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    return -worst


def _run(job: tuple) -> dict:
    exit_name, exit_kw, filt_label, filt_kw = job
    cfg = _variant(_CTX["cfg"], exit_kw, filt_kw)
    res = BacktestEngine(cfg, OrbStrategy(cfg.strategy), NewsCalendar.empty()).run(_CTX["bars"])
    split = _CTX["split"]
    all_r = [t.result_r for t in res.trades]
    is_r = [t.result_r for t in res.trades if t.day < split]
    oos_r = [t.result_r for t in res.trades if t.day >= split]
    row = {"exit": exit_name, "filters": filt_label}
    for tag, rs in (("all", all_r), ("is", is_r), ("oos", oos_r)):
        for k, v in _stats(rs).items():
            row[f"{tag}_{k}"] = v
    row["all_maxdd_r"] = _max_dd_r(all_r)
    return row


def build_jobs() -> list[tuple]:
    names = list(FILTER_AXES)
    combos = itertools.product(*(FILTER_AXES[n][1] for n in names))
    jobs = []
    for values in combos:
        kw, parts = {}, []
        for n, v in zip(names, values, strict=True):
            field, levels = FILTER_AXES[n]
            kw[field] = v
            if v != levels[0]:
                parts.append(f"{n}={v}")
        label = "+".join(parts) if parts else "none"
        for exit_name, exit_kw in EXITS:
            jobs.append((exit_name, exit_kw, label, kw))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--out", default="reports/sweep.csv")
    ap.add_argument("--split", default="2026-05-01", help="first day of the out-of-sample tail")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    jobs = build_jobs()
    print(f"grid: {len(EXITS)} exits x {len(jobs) // len(EXITS)} filter combos = {len(jobs)} runs")

    import csv
    import os
    import time

    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(args.config, args.split)) as ex:
        for i, row in enumerate(ex.map(_run, jobs, chunksize=8), 1):
            rows.append(row)
            if i % 250 == 0 or i == len(jobs):
                el = time.time() - t0
                left = el / i * (len(jobs) - i)
                print(f"  {i}/{len(jobs)}  {el:.0f}s elapsed, ~{left:.0f}s left", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
