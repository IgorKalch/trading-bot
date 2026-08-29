"""Open-position management: trailing stop, breakeven, time exit.

Pure logic — consumed by the live runner (which applies actions via MT5)
and by the backtest simulator (which applies them to simulated positions).

Trailing algorithm (STRATEGY.md §9):
  Let R = |entry - initial_stop|, start = trail_start_r, step = trail_step_r.
  When profit reaches start + k*step (k = 0, 1, 2, ...):
      stop moves to entry + k*step*R (k=0 -> breakeven).
  With defaults start=0.5, step=0.5:
      +0.5R -> stop to breakeven, +1.0R -> stop to +0.5R, +1.5R -> +1.0R, ...
  The stop only ever tightens, never loosens.
"""

from __future__ import annotations

import math
from datetime import datetime

from tradingbot.config import TargetsConfig
from tradingbot.core.models import CloseNow, ManageAction, ManagedPosition, ModifyStop, Side, TpMode


def manage_position(
    pos: ManagedPosition,
    price: float,
    now: datetime,
    flat_time: datetime | None,
    cfg: TargetsConfig,
) -> list[ManageAction]:
    """Decide management actions for one open position at current `price`.

    `price` must be the closing side of the position: bid for LONG, ask for SHORT.
    """
    if pos.closed:
        return []

    if flat_time is not None and now >= flat_time:
        return [CloseNow(reason="flat_time: end-of-day forced exit")]

    if TpMode(cfg.tp_mode) is not TpMode.TRAILING:
        return []

    profit_r = pos.profit_r(price)
    if profit_r < cfg.trail_start_r or pos.risk_points <= 0:
        return []

    # Number of completed trailing steps (k starts at 0 for breakeven).
    # Config validates trail_step_r > 0; guard anyway (breakeven-only fallback).
    step = cfg.trail_step_r if cfg.trail_step_r > 0 else float("inf")
    k = math.floor((profit_r - cfg.trail_start_r) / step + 1e-9)
    sign = pos.side.sign
    target_stop = pos.entry_price + sign * k * cfg.trail_step_r * pos.risk_points
    if k == 0 and cfg.breakeven_buffer_points > 0:
        target_stop = pos.entry_price + sign * cfg.breakeven_buffer_points

    improves = (target_stop > pos.stop_loss) if pos.side is Side.LONG else (target_stop < pos.stop_loss)
    if not improves:
        return []

    label = "breakeven" if k == 0 else f"+{k * cfg.trail_step_r:.2f}R locked"
    return [ModifyStop(new_stop=target_stop, reason=f"trailing at +{profit_r:.2f}R -> {label}")]
