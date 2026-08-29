"""Fair Value Gap / imbalance detection, shared by the strategies.

A three-bar imbalance in the smart-money sense: the middle bar displaces far
enough that the wicks of its neighbours do not overlap.

    bullish FVG at bar i:  bars[i-2].high < bars[i].low
    bearish FVG at bar i:  bars[i-2].low  > bars[i].high

The gap between those two levels is the unfilled area. `FvgTracker` keeps the
most recent gap per direction and how many bars ago it formed, which is all the
entry filters need: "was there displacement in my direction, recently".

Pure functions on closed bars — no I/O, no config dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingbot.core.models import Bar, Side


@dataclass(frozen=True)
class Fvg:
    side: Side  # direction of the displacement
    low: float  # bottom of the unfilled gap
    high: float  # top of the unfilled gap
    bar_index: int  # index of the third bar, for ageing

    @property
    def size(self) -> float:
        return self.high - self.low

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high


class FvgTracker:
    """Rolling three-bar window; remembers the latest gap on each side."""

    def __init__(self) -> None:
        self._w: list[Bar] = []
        self._i = -1
        self.last: dict[Side, Fvg] = {}

    def update(self, bar: Bar) -> Fvg | None:
        """Feed one closed bar; returns a gap if this bar completed one."""
        self._i += 1
        self._w.append(bar)
        if len(self._w) > 3:
            self._w.pop(0)
        if len(self._w) < 3:
            return None
        first, _mid, third = self._w
        found: Fvg | None = None
        if first.high < third.low:
            found = Fvg(Side.LONG, first.high, third.low, self._i)
        elif first.low > third.high:
            found = Fvg(Side.SHORT, third.high, first.low, self._i)
        if found is not None:
            self.last[found.side] = found
        return found

    def recent(self, side: Side, max_age_bars: int, min_size: float = 0.0) -> Fvg | None:
        """Latest gap on `side`, if it formed within `max_age_bars` and is big enough."""
        g = self.last.get(side)
        if g is None:
            return None
        if self._i - g.bar_index > max_age_bars:
            return None
        if min_size and g.size < min_size:
            return None
        return g
