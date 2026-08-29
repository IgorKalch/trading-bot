"""Strategy x exit x filter grid over one cached dataset.

Where tools/sweep.py explores filter combinations for a single strategy, this
runs several strategies side by side on the same bars and the same exit schemes,
which is what a fair comparison between models needs.

Every cell reports a standard error and a 95% interval, because with samples of
a few hundred trades the point estimate on its own has repeatedly proved
worthless in this project (STRATEGY.md Додатки Д, Е).

    python tools/grid.py --config config/config.duka.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tradingbot.backtest.engine import BacktestEngine  # noqa: E402
from tradingbot.config import load_config  # noqa: E402
from tradingbot.data import history  # noqa: E402
from tradingbot.data.news import NewsCalendar  # noqa: E402
from tradingbot.strategy import build_strategy  # noqa: E402

EXITS = [
    ("fixed 1.0R", {"tp_mode": "fixed_rr", "fixed_rr": 1.0}),
    ("fixed 1.5R", {"tp_mode": "fixed_rr", "fixed_rr": 1.5}),
    ("fixed 2.0R", {"tp_mode": "fixed_rr", "fixed_rr": 2.0}),
    ("trail 0.5/0.25", {"tp_mode": "trailing", "trail_start_r": 0.5, "trail_step_r": 0.25}),
    ("trail 0.5/0.5", {"tp_mode": "trailing", "trail_start_r": 0.5, "trail_step_r": 0.5}),
    ("trail 0.5/1.0", {"tp_mode": "trailing", "trail_start_r": 0.5, "trail_step_r": 1.0}),
    ("trail 1.0/0.5", {"tp_mode": "trailing", "trail_start_r": 1.0, "trail_step_r": 0.5}),
    ("trail 1.0/1.0", {"tp_mode": "trailing", "trail_start_r": 1.0, "trail_step_r": 1.0}),
    ("trail 1.5/0.5", {"tp_mode": "trailing", "trail_start_r": 1.5, "trail_step_r": 0.5}),
]

# (label, strategy name, timeframe, {dotted config path: value})
VARIANTS = [
    ("orb", "orb", "M5", {}),
    ("orb +fvg", "orb", "M5", {"filters.require_fvg": True}),
    ("orb +break0.2", "orb", "M5", {"confirmation.min_break_or_frac": 0.2}),
    ("retest pb3", "retest", "M1", {"retest.max_pullback_bars": 3}),
    ("retest pb6", "retest", "M1", {"retest.max_pullback_bars": 6}),
    ("retest pb3 +fvg", "retest", "M1", {"retest.max_pullback_bars": 3, "filters.require_fvg": True}),
    ("sweep M5", "sweep", "M5", {}),
    ("sweep M5 +fvg", "sweep", "M5", {"sweep.require_fvg": True}),
    ("sweep M5 deep", "sweep", "M5", {"sweep.min_sweep_range_frac": 0.05}),
    ("sweep M1", "sweep", "M1", {"sweep.max_reclaim_bars": 15}),
    ("sweep M1 +fvg", "sweep", "M1", {"sweep.max_reclaim_bars": 15, "sweep.require_fvg": True}),
]

_CTX: dict = {}


def _init(config_path: str) -> None:
    cfg = load_config(config_path, ".env")
    _CTX["cfg"] = cfg
    _CTX["bars"] = {
        tf: history.load_bars(cfg.backtest.data_dir, cfg.mt5.symbol, tf) for tf in ("M1", "M5")
    }


def _set(strategy_cfg, dotted: str, value) -> None:
    """Apply e.g. "filters.require_fvg" to a StrategyConfig."""
    head, _, tail = dotted.partition(".")
    setattr(getattr(strategy_cfg, head), tail, value)


def _stats(rs: list[float]) -> dict:
    n = len(rs)
    if n < 2:
        return {"n": n, "wr": 0.0, "pf": 0.0, "exp": 0.0, "se": 0.0, "dd": 0.0}
    mean = sum(rs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in rs) / (n - 1))
    gw = sum(x for x in rs if x > 0)
    gl = -sum(x for x in rs if x < 0)
    peak = cum = worst = 0.0
    for x in rs:
        cum += x
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    return {
        "n": n,
        "wr": sum(1 for x in rs if x > 0) / n * 100.0,
        "pf": gw / gl if gl else 0.0,
        "exp": mean,
        "se": sd / math.sqrt(n),
        "dd": -worst,
    }


def _run(job: tuple) -> dict:
    vlabel, sname, tf, overrides, elabel, ekw = job
    cfg = _CTX["cfg"].model_copy(deep=True)
    cfg.strategy.name = sname
    cfg.strategy.timeframe = tf
    for k, v in ekw.items():
        setattr(cfg.strategy.targets, k, v)
    for k, v in overrides.items():
        _set(cfg.strategy, k, v)
    res = BacktestEngine(cfg, build_strategy(cfg.strategy), NewsCalendar.empty()).run(_CTX["bars"][tf])
    row = {"variant": vlabel, "exit": elabel, "tf": tf}
    row.update(_stats([t.result_r for t in res.trades]))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.duka.yaml")
    ap.add_argument("--out", default="reports/grid.csv")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    jobs = [(vl, sn, tf, ov, el, ek) for vl, sn, tf, ov in VARIANTS for el, ek in EXITS]
    print(f"{len(VARIANTS)} variants x {len(EXITS)} exits = {len(jobs)} runs")

    import csv
    import os
    import time

    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(args.config,)) as ex:
        for i, row in enumerate(ex.map(_run, jobs), 1):
            rows.append(row)
            print(f"  [{i}/{len(jobs)}] {row['variant']:<16} {row['exit']:<15} "
                  f"n={row['n']:<5} PF {row['pf']:.2f}  {row['exp']:+.3f}R +-{row['se']:.3f}", flush=True)
    print(f"done in {time.time() - t0:.0f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
