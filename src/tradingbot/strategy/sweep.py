"""Liquidity sweep of the overnight range, then reclaim (STRATEGY.md Додаток Ж).

Sources: the TradingView layouts in grafik-creen/ (overnight/Asia range boxes on
GER40 M5), Tom Hougaard shorting "against the overnight range top" at the open,
and the smart-money framing of the same move — stops resting beyond an obvious
level get taken, price fails to hold, and the failure is the signal.

    1. build the range of everything BEFORE the session open (Asia/overnight)
    2. after the open, price trades BEYOND one edge of it        -> the sweep
    3. price closes back INSIDE the range within a few bars      -> the reclaim
    4. enter AGAINST the sweep, stop beyond its extreme

This is the first strategy here that fades rather than continues, which is the
point: ORB and the retest model are both continuation models and both measured
as edgeless (Додатки Д, Е).

Pure logic on closed bars — same contract as strategy/orb.py.
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
    Side,
    SignalKind,
    SkipEvent,
    StrategyEvent,
    TpMode,
)
from tradingbot.strategy.base import DayContext
from tradingbot.strategy.imbalance import FvgTracker

_TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30}


class _Phase(Enum):
    WAIT_SWEEP = "wait_sweep"
    WAIT_RECLAIM = "wait_reclaim"
    DONE = "done"


@dataclass
class _SideState:
    """One edge of the overnight range: swept yet, and how far."""

    phase: _Phase = _Phase.WAIT_SWEEP
    sweep_extreme: float | None = None  # furthest price reached beyond the edge
    bars_since_sweep: int = 0


@dataclass
class _DayState:
    ctx: DayContext
    pre_high: float | None = None  # overnight range, built before the open
    pre_low: float | None = None
    pre_bars: int = 0
    opened: bool = False
    signals_emitted: int = 0
    sides: dict[Side, _SideState] = field(default_factory=dict)
    fvg: FvgTracker = field(default_factory=FvgTracker)


class SweepStrategy:
    """One instance handles consecutive days; on_day_start resets state."""

    name = "sweep"

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self._st: _DayState | None = None
        # Rolling session extremes, carried across days so the model can sweep
        # PDH/PDL and PWH/PWL as well as the overnight range (Додаток З).
        self._sessions: list[tuple[float, float]] = []
        self._cur: tuple[float, float] | None = None

    # ------------------------------------------------------------------ API

    def on_day_start(self, ctx: DayContext) -> None:
        if self._cur is not None:
            self._sessions.append(self._cur)
            self._sessions = self._sessions[-5:]
            self._cur = None
        # SHORT fades a sweep of the high, LONG fades a sweep of the low.
        self._st = _DayState(ctx=ctx, sides={s: _SideState() for s in Side})

    def on_position_closed(self, pos: ManagedPosition) -> None:
        pass

    def on_bar(self, bar: Bar) -> list[StrategyEvent]:
        st = self._st
        if st is None:
            return []
        s = st.ctx.session
        st.fvg.update(bar)
        self._cur = (
            (bar.high, bar.low)
            if self._cur is None
            else (max(self._cur[0], bar.high), min(self._cur[1], bar.low))
        )

        # -- overnight range: everything before the cash open
        if bar.time < s.session_open:
            st.pre_high = bar.high if st.pre_high is None else max(st.pre_high, bar.high)
            st.pre_low = bar.low if st.pre_low is None else min(st.pre_low, bar.low)
            st.pre_bars += 1
            return []

        w = self.cfg.sweep
        events: list[StrategyEvent] = []
        if not st.opened:
            st.opened = True
            if w.reference != "overnight":
                # Replace the overnight range with the previous session's or the
                # previous week's extremes; the state machine is unchanged.
                back = 1 if w.reference == "prev_day" else 5
                hist = self._sessions[-back:]
                if len(hist) < back:
                    st.sides[Side.LONG].phase = _Phase.DONE
                    st.sides[Side.SHORT].phase = _Phase.DONE
                    return [SkipEvent(bar.time, "no_history", f"need {back} prior sessions")]
                st.pre_high = max(h for h, _ in hist)
                st.pre_low = min(low for _, low in hist)
                st.pre_bars = w.min_pre_bars
            if st.pre_bars < w.min_pre_bars or st.pre_high is None or st.pre_low is None:
                st.sides[Side.LONG].phase = _Phase.DONE
                st.sides[Side.SHORT].phase = _Phase.DONE
                return [
                    SkipEvent(
                        bar.time,
                        "no_overnight_range",
                        f"only {st.pre_bars} bars before the open, need {w.min_pre_bars}",
                    )
                ]
            # The first bar of the session can already sweep, so record the
            # range and fall through instead of consuming the bar here.
            events.append(
                InfoEvent(
                    bar.time,
                    f"overnight range {st.pre_low:.1f}..{st.pre_high:.1f} "
                    f"(width {st.pre_high - st.pre_low:.1f}) from {st.pre_bars} bars",
                )
            )

        if st.signals_emitted >= w.max_positions_per_day:
            return events

        for side in (Side.SHORT, Side.LONG):
            ev = self._advance(bar, st, side)
            if ev is None:
                continue
            events.append(ev)
            if isinstance(ev, EntrySignal):
                break
        return events

    # ------------------------------------------------------------ state machine

    def _advance(self, bar: Bar, st: _DayState, side: Side) -> StrategyEvent | None:
        """`side` is the direction we would TRADE, i.e. against the sweep."""
        w = self.cfg.sweep
        state = st.sides[side]
        if state.phase is _Phase.DONE:
            return None
        assert st.pre_high is not None and st.pre_low is not None
        # Fading a sweep of the HIGH means going short.
        edge = st.pre_high if side is Side.SHORT else st.pre_low
        width = st.pre_high - st.pre_low
        beyond_sign = 1.0 if side is Side.SHORT else -1.0  # sweep direction

        if state.phase is _Phase.WAIT_SWEEP:
            reach = bar.high if side is Side.SHORT else bar.low
            depth = (reach - edge) * beyond_sign
            if depth <= 0:
                return None
            min_depth = max(w.min_sweep_points, w.min_sweep_range_frac * width)
            if min_depth and depth < min_depth:
                return None
            state.phase = _Phase.WAIT_RECLAIM
            state.sweep_extreme = reach
            # The sweeping bar can reclaim on its own close — fall through.
        else:
            state.bars_since_sweep += 1
            if state.bars_since_sweep > w.max_reclaim_bars:
                state.phase = _Phase.DONE
                return SkipEvent(
                    bar.time,
                    "sweep_timeout",
                    f"{side.value}: no reclaim within {w.max_reclaim_bars} bars",
                )
            reach = bar.high if side is Side.SHORT else bar.low
            state.sweep_extreme = (
                max(state.sweep_extreme or reach, reach)
                if side is Side.SHORT
                else min(state.sweep_extreme or reach, reach)
            )

        # -- reclaim: close back inside the range, on the trading side of the edge
        inside = (edge - bar.close) * beyond_sign > 0
        if not inside:
            return None
        if w.require_body_close:
            directional = bar.close < bar.open if side is Side.SHORT else bar.close > bar.open
            if not directional:
                return None
        return self._signal(bar, st, side, state, edge, width)

    # ----------------------------------------------------------------- signal

    def _signal(
        self, bar: Bar, st: _DayState, side: Side, state: _SideState, edge: float, width: float
    ) -> StrategyEvent:
        cfg = self.cfg
        w = cfg.sweep
        s = st.ctx.session
        assert state.sweep_extreme is not None

        def done(rule: str, detail: str) -> SkipEvent:
            state.phase = _Phase.DONE
            return SkipEvent(bar.time, rule, f"sweep {side.value}: {detail}")

        kind = SignalKind.FIRST if st.signals_emitted == 0 else SignalKind.SECOND
        cutoff = s.pos1_cutoff if kind is SignalKind.FIRST else s.pos2_cutoff
        if bar.time >= cutoff:
            return done("time_window", f"reclaim at {bar.time:%H:%M} UTC is past the cutoff")

        if w.require_fvg:
            gap = st.fvg.recent(side, w.fvg_max_age_bars, w.fvg_min_size_points)
            if gap is None:
                return done("no_fvg", "no imbalance in the trade direction")

        # SHORT (sign -1) puts the stop above the sweep, LONG below it.
        stop = state.sweep_extreme - side.sign * w.stop_buffer_points
        risk_points = abs(bar.close - stop)
        if risk_points <= 0:
            return done("stop_size", "degenerate stop distance")
        if w.max_stop_points and risk_points > w.max_stop_points:
            return done("stop_size", f"stop {risk_points:.1f} > max {w.max_stop_points}")
        if w.min_stop_points and risk_points < w.min_stop_points:
            return done("stop_size", f"stop {risk_points:.1f} < min {w.min_stop_points}")

        f = cfg.filters
        if f.news_filter_enabled:
            entry_time = bar.time + timedelta(minutes=_TF_MINUTES.get(cfg.timeframe, 5))
            ev = st.ctx.news.blocking_event(
                entry_time,
                currencies=f.news_currencies,
                min_impact=f.news_min_impact,
                before_min=f.news_buffer_before_min,
                after_min=f.news_buffer_after_min,
            )
            if ev is not None:
                return done("news", f"news blackout: {ev.currency} {ev.title}")

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
                f"swept {edge:.1f} to {state.sweep_extreme:.1f} and reclaimed: "
                f"close {bar.close:.1f} back inside, stop {stop:.1f} (risk {risk_points:.1f})"
            ),
            context={
                "overnight_high": st.pre_high or 0.0,
                "overnight_low": st.pre_low or 0.0,
                "range_width": round(width, 2),
                "sweep_extreme": round(state.sweep_extreme, 2),
                "model": "sweep",
            },
        )
