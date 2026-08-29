"""Position sizing: fixed fractional risk.

volume (lots) = risk_money / (stop_distance_in_ticks * tick_value_per_lot)
rounded DOWN to the broker's volume step so real risk never exceeds the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tradingbot.data.mt5_client import SymbolSpec


class SizingError(ValueError):
    pass


@dataclass(frozen=True)
class SizingResult:
    volume: float
    risk_money_target: float
    risk_money_actual: float  # after rounding volume down
    stop_distance: float


def calc_volume(
    balance: float,
    risk_pct: float,
    stop_distance: float,
    spec: SymbolSpec,
    min_volume_fallback: bool = False,
) -> SizingResult:
    """Compute lot size so that hitting the stop loses ~risk_pct% of balance.

    stop_distance is in PRICE units (e.g. DAX points if quote step is 1.0).
    """
    if balance <= 0:
        raise SizingError(f"balance must be positive, got {balance}")
    if stop_distance <= 0:
        raise SizingError(f"stop distance must be positive, got {stop_distance}")
    if not 0 < risk_pct <= 100:
        raise SizingError(f"risk_pct out of range: {risk_pct}")

    risk_money = balance * risk_pct / 100.0
    money_per_lot = (stop_distance / spec.tick_size) * spec.tick_value
    if money_per_lot <= 0:
        raise SizingError(f"invalid symbol economics: money_per_lot={money_per_lot}")

    raw = risk_money / money_per_lot
    steps = math.floor(raw / spec.volume_step + 1e-9)
    volume = round(steps * spec.volume_step, 8)

    if volume < spec.volume_min:
        if not min_volume_fallback:
            raise SizingError(
                f"computed volume {volume} is below broker minimum {spec.volume_min}; "
                f"risk {risk_pct}% of {balance:.2f} cannot be respected with stop {stop_distance}"
            )
        volume = spec.volume_min
    volume = min(volume, spec.volume_max)

    return SizingResult(
        volume=volume,
        risk_money_target=risk_money,
        risk_money_actual=volume * money_per_lot,
        stop_distance=stop_distance,
    )
