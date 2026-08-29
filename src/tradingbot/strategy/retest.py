"""Retest-and-absorb entry model (STRATEGY.md Додаток В).

Source: Cryptology Key, "Open Range NASDAQ". Same opening range as the ORB
model, but the entry is not the breakout itself:

    1. a bar CLOSES with its body beyond the opening range  -> direction is set
    2. price RETURNS and tests the range boundary            -> pullback
    3. the pullback is ABSORBED: price closes back beyond the
       extreme made before it                                -> entry

The stop is what makes this a separate strategy rather than a filter on ORB:
it sits at the pullback extreme, not at the far side of the opening range, so
1R is typically a fraction of the range width. The target is a fixed RR taken
from cfg.targets, so the same exit grid applies to both models.

Pure logic on closed bars, no I/O — same contract as strategy/orb.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum

from tradingbot.config import StrategyConfig
from tradingbot.core.models import (
    Bar,
    EntrySignal,
    InfoEvent,
    ManagedPosition,
    OpeningRange,
    Side,
    SignalKind,
    SkipEvent,
    StrategyEvent,
    TpMode,
)
from tradingbot.strategy.base import DayContext

_TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30}


class _Phase(Enum):
    WAIT_BREAK = "wait_break"
    WAIT_RETEST = "wait_retest"
    WAIT_ABSORB = "wait_absorb"
    DONE = "done"


@dataclass
class _SideState:
    """One side's progress through break -> retest -> absorption."""

    phase: _Phase = _Phase.WAIT_BREAK
    break_extreme: float | None = None  # best price reached before the pullback
    pullback_extreme: float | None = None  # worst price during the pullback
    bars_since_break: int = 0


@dataclass
class _DayState:
    ctx: DayContext
    or_high: float | None = None
    or_low: float | None = None
    opening_range: OpeningRange | None = None
    day_skipped: str | None = None
    signals_emitted: int = 0
    sides: dict[Side, _SideState] = field(default_factory=dict)


class RetestStrategy:
    """One instance handles consecutive days; on_day_start resets state."""

    name = "retest"

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self._st: _DayState | None = None

    # ------------------------------------------------------------------ API

    def on_day_start(self, ctx: DayContext) -> None:
        self._st = _DayState(ctx=ctx, sides={s: _SideState() for s in Side})

    def on_position_closed(self, pos: ManagedPosition) -> None:
        pass

    def on_bar(self, bar: Bar) -> list[StrategyEvent]:
        st = self._st
        if st is None:
            return []
        s = st.ctx.session
        if bar.time < s.session_open or st.day_skipped:
            return []

        # -- opening range: same definition as the ORB model (§2)
        if bar.time < s.or_end:
            st.or_high = bar.high if st.or_high is None else max(st.or_high, bar.high)
            st.or_low = bar.low if st.or_low is None else min(st.or_low, bar.low)
            return []

        events: list[StrategyEvent] = []
        if st.opening_range is None:
            if st.or_high is None or st.or_low is None:
                st.day_skipped = "no bars in the OR window (holiday / data gap)"
                return [SkipEvent(bar.time, "no_or", st.day_skipped)]
            st.opening_range = OpeningRange(
                day=s.day, high=st.or_high, low=st.or_low, start=s.session_open, end=s.or_end
            )
            # This bar is also the first one that can break the range, so fall
            # through to the state machine instead of consuming it here.
            events.append(
                InfoEvent(
                    bar.time,
                    f"OR formed: {st.or_low:.1f}..{st.or_high:.1f} "
                    f"(width {st.opening_range.width:.1f}) — retest model",
                )
            )

        if st.signals_emitted >= self.cfg.retest.max_positions_per_day:
            return events

        for side in (Side.LONG, Side.SHORT):
            event = self._advance(bar, st, side)
            if event is None:
                continue
            events.append(event)
            if isinstance(event, EntrySignal):
                break  # at most one entry per bar
        return events

    # ------------------------------------------------------------ state machine

    def _advance(self, bar: Bar, st: _DayState, side: Side) -> StrategyEvent | None:
        r = self.cfg.retest
        orng = st.opening_range
        assert orng is not None
        state = st.sides[side]
        if state.phase is _Phase.DONE:
            return None

        boundary = orng.high if side is Side.LONG else orng.low
        sign = side.sign

        if state.phase is _Phase.WAIT_BREAK:
            return self._try_break(bar, state, side, boundary, orng.width)

        state.bars_since_break += 1
        if state.bars_since_break > r.max_pullback_bars:
            state.phase = _Phase.DONE
            return SkipEvent(
                bar.time,
                "retest_timeout",
                f"{side.value}: no absorbed retest within {r.max_pullback_bars} bars",
            )

        if state.phase is _Phase.WAIT_RETEST:
            # Extend the extreme only while the move is still running. Once the
            # pullback starts it must FREEZE: it is the level absorption has to
            # reclaim, so folding in later bars would make it unreachable.
            state.break_extreme = (
                max(state.break_extreme or bar.high, bar.high)
                if side is Side.LONG
                else min(state.break_extreme or bar.low, bar.low)
            )
            # A retest is price coming back to touch the range boundary again.
            touched = bar.low <= boundary if side is Side.LONG else bar.high >= boundary
            if not touched:
                return None
            state.phase = _Phase.WAIT_ABSORB
            state.pullback_extreme = bar.low if side is Side.LONG else bar.high
            return None

        # -- WAIT_ABSORB
        state.pullback_extreme = (
            min(state.pullback_extreme or bar.low, bar.low)
            if side is Side.LONG
            else max(state.pullback_extreme or bar.high, bar.high)
        )
        opposite = orng.low if side is Side.LONG else orng.high
        if (bar.close - opposite) * sign < 0:
            state.phase = _Phase.DONE
            return SkipEvent(
                bar.time, "retest_invalidated", f"{side.value}: pullback closed through the range"
            )
        if (bar.close - (state.break_extreme or 0.0)) * sign <= 0:
            return None
        return self._signal(bar, st, side, state)

    def _try_break(
        self, bar: Bar, state: _SideState, side: Side, boundary: float, or_width: float
    ) -> StrategyEvent | None:
        r = self.cfg.retest
        beyond = (bar.close - boundary) * side.sign
        if beyond <= 0:
            return None
        if r.require_body_close:
            directional = bar.close > bar.open if side is Side.LONG else bar.close < bar.open
            if not directional:
                return None
        min_break = r.min_break_or_frac * or_width
        if min_break and beyond < min_break:
            return None
        state.phase = _Phase.WAIT_RETEST
        state.break_extreme = bar.high if side is Side.LONG else bar.low
        return InfoEvent(
            bar.time,
            f"retest model {side.value}: body close {bar.close:.1f} beyond "
            f"{boundary:.1f} by {beyond:.1f} — waiting for the retest",
        )

    # ----------------------------------------------------------------- signal

    def _signal(self, bar: Bar, st: _DayState, side: Side, state: _SideState) -> StrategyEvent:
        cfg = self.cfg
        r = cfg.retest
        orng = st.opening_range
        assert orng is not None and state.pullback_extreme is not None
        s = st.ctx.session

        def done(rule: str, detail: str) -> SkipEvent:
            state.phase = _Phase.DONE
            return SkipEvent(bar.time, rule, f"retest {side.value}: {detail}")

        kind = SignalKind.FIRST if st.signals_emitted == 0 else SignalKind.SECOND
        cutoff = s.pos1_cutoff if kind is SignalKind.FIRST else s.pos2_cutoff
        if bar.time >= cutoff:
            return done("time_window", f"absorption at {bar.time:%H:%M} UTC is past the cutoff")

        stop = state.pullback_extreme - side.sign * r.stop_buffer_points
        risk_points = abs(bar.close - stop)
        if risk_points <= 0:
            return done("stop_size", "degenerate stop distance")
        if r.max_stop_points and risk_points > r.max_stop_points:
            return done("stop_size", f"stop distance {risk_points:.1f} > max {r.max_stop_points}")
        if r.min_stop_points and risk_points < r.min_stop_points:
            return done("stop_size", f"stop distance {risk_points:.1f} < min {r.min_stop_points}")

        f = cfg.filters
        if f.news_filter_enabled:
            entry_time = bar.time + timedelta(minutes=_TF_MINUTES.get(cfg.timeframe, 5))
            event = st.ctx.news.blocking_event(
                entry_time,
                currencies=f.news_currencies,
                min_impact=f.news_min_impact,
                before_min=f.news_buffer_before_min,
                after_min=f.news_buffer_after_min,
            )
            if event is not None:
                return done("news", f"news blackout: {event.currency} {event.title}")

        tp_rr = cfg.targets.fixed_rr if TpMode(cfg.targets.tp_mode) is TpMode.FIXED_RR else None
        take_profit = bar.close + side.sign * tp_rr * risk_points if tp_rr else None

        state.phase = _Phase.DONE
        st.signals_emitted += 1
        return EntrySignal(
            kind=kind,
            side=side,
            time=bar.time,
            entry_ref=bar.close,
            stop_loss=stop,
            take_profit=take_profit,
            tp_rr=tp_rr,
            risk_points=risk_points,
            reason=(
                f"retest absorbed {side.value}: close {bar.close:.1f} reclaimed "
                f"{state.break_extreme:.1f}, stop at pullback extreme "
                f"{state.pullback_extreme:.1f} (risk {risk_points:.1f})"
            ),
            context={
                "or_high": orng.high,
                "or_low": orng.low,
                "or_width": round(orng.width, 2),
                "break_extreme": round(state.break_extreme or 0.0, 2),
                "pullback_extreme": round(state.pullback_extreme, 2),
                "model": "retest",
            },
        )
