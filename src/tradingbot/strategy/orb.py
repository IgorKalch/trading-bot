"""ORB (Opening Range Breakout) strategy — implementation of STRATEGY.md.

Every rule implemented here MUST be described in STRATEGY.md (§ references in
comments). Pure logic: no I/O, no broker calls, no clocks — driven by
completed bars from either the live runner or the backtest engine.

Day lifecycle (state machine):
    WAIT_OR -> OR forming (§2) -> WAIT_BREAKOUT (first breakout of either
    side, §3-§5) -> after first breakout: WAIT_REVERSAL (opposite side only,
    §6) -> DONE (both sides used, or cutoffs passed).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta

from tradingbot.config import StrategyConfig
from tradingbot.core.models import (
    Bar,
    EntrySignal,
    GapInfo,
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
from tradingbot.strategy.imbalance import FvgTracker

log = logging.getLogger(__name__)

_TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}


@dataclass
class _DayState:
    ctx: DayContext
    or_high: float | None = None
    or_low: float | None = None
    or_close: float | None = None
    or_volume: int = 0
    opening_range: OpeningRange | None = None
    gap: GapInfo | None = None
    session_open_price: float | None = None
    day_skipped: str | None = None  # reason, if the whole day is filtered out
    first_breakout_side: Side | None = None
    signals_emitted: int = 0
    sides_signalled: set[Side] = field(default_factory=set)
    # session VWAP accumulators (§7.6)
    vwap_pv: float = 0.0
    vwap_vol: float = 0.0
    # rolling bar volumes for the breakout-candle volume filter (§7.10)
    recent_vols: deque[float] = field(default_factory=deque)
    bar_rvol: float | None = None  # current bar's volume / recent average
    # three-bar imbalance tracker for the FVG filter (§7.11)
    fvg: FvgTracker = field(default_factory=FvgTracker)


class OrbStrategy:
    """One instance handles consecutive days; on_day_start resets state."""

    name = "orb"

    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self._st: _DayState | None = None

    # ------------------------------------------------------------------ API

    def on_day_start(self, ctx: DayContext) -> None:
        self._st = _DayState(ctx=ctx)

    def on_position_closed(self, pos: ManagedPosition) -> None:
        # Bookkeeping hook; entry limits are enforced via sides_signalled.
        pass

    def on_bar(self, bar: Bar) -> list[StrategyEvent]:
        st = self._st
        if st is None:
            return []
        s = st.ctx.session
        events: list[StrategyEvent] = []

        if bar.time < s.session_open or st.day_skipped:
            return events

        st.fvg.update(bar)  # §7.11

        # -- VWAP accumulation over the whole session (§7.6)
        typical = (bar.high + bar.low + bar.close) / 3.0
        st.vwap_pv += typical * max(bar.tick_volume, 1)
        st.vwap_vol += max(bar.tick_volume, 1)

        # -- relative volume of THIS bar vs the recent ones (§7.10). Measured
        # before the bar is folded in, so a bar is never compared to itself.
        lookback = max(self.cfg.filters.break_rvol_lookback_bars, 1)
        st.bar_rvol = (
            bar.tick_volume / (sum(st.recent_vols) / len(st.recent_vols))
            if len(st.recent_vols) >= 2
            else None
        )
        st.recent_vols.append(float(bar.tick_volume))
        while len(st.recent_vols) > lookback:
            st.recent_vols.popleft()

        # -- Opening Range formation (§2)
        if bar.time < s.or_end:
            if st.or_high is None:
                st.session_open_price = bar.open
            st.or_high = bar.high if st.or_high is None else max(st.or_high, bar.high)
            st.or_low = bar.low if st.or_low is None else min(st.or_low, bar.low)
            st.or_close = bar.close
            st.or_volume += bar.tick_volume
            return events

        if st.opening_range is None:
            if st.or_high is None or st.or_low is None:
                st.day_skipped = "no bars in the OR window (holiday / data gap)"
                return [SkipEvent(bar.time, "no_or", st.day_skipped)]
            st.opening_range = OpeningRange(
                day=s.day, high=st.or_high, low=st.or_low, start=s.session_open, end=s.or_end
            )
            if st.ctx.prev_session_close is not None and st.session_open_price is not None:
                st.gap = GapInfo(prev_close=st.ctx.prev_session_close, session_open=st.session_open_price)
            events.extend(self._on_or_formed(bar, st))
            if st.day_skipped:
                return events

        # -- Breakout detection (§3-§6)
        events.extend(self._check_breakout(bar, st))
        return events

    # ------------------------------------------------------- OR-level filters

    def _on_or_formed(self, bar: Bar, st: _DayState) -> list[StrategyEvent]:
        f = self.cfg.filters
        orng = st.opening_range
        assert orng is not None
        events: list[StrategyEvent] = [
            InfoEvent(
                bar.time,
                f"OR formed: {orng.low:.1f}..{orng.high:.1f} (width {orng.width:.1f}), "
                + (f"gap {st.gap.size:+.1f}" if st.gap else "gap unknown"),
            )
        ]

        def skip(rule: str, detail: str) -> None:
            st.day_skipped = detail
            events.append(SkipEvent(bar.time, rule, detail))

        weekday = orng.day.weekday()
        if weekday in f.skip_weekdays:  # §7.5
            skip("weekday", f"weekday {weekday} is in skip_weekdays")
        elif f.min_or_width_points and orng.width < f.min_or_width_points:  # §7.2
            skip("or_width_min", f"OR width {orng.width:.1f} < min {f.min_or_width_points}")
        elif f.max_or_width_points and orng.width > f.max_or_width_points:  # §7.2
            skip("or_width_max", f"OR width {orng.width:.1f} > max {f.max_or_width_points}")
        elif f.min_or_width_pct and orng.width / orng.high * 100 < f.min_or_width_pct:  # §7.2
            skip("or_width_pct_min",
                 f"OR width {orng.width / orng.high * 100:.3f}% < min {f.min_or_width_pct}%")
        elif f.max_or_width_pct and orng.width / orng.high * 100 > f.max_or_width_pct:  # §7.2
            skip("or_width_pct_max",
                 f"OR width {orng.width / orng.high * 100:.3f}% > max {f.max_or_width_pct}%")
        elif (
            f.max_or_width_atr_mult
            and st.ctx.atr_d1
            and orng.width > f.max_or_width_atr_mult * st.ctx.atr_d1
        ):  # §7.2
            skip(
                "or_width_atr",
                f"OR width {orng.width:.1f} > {f.max_or_width_atr_mult} x ATR({f.atr_period}) "
                f"{st.ctx.atr_d1:.1f}",
            )
        elif f.max_gap_points and st.gap and st.gap.abs_size > f.max_gap_points:  # §7.3
            skip("gap_size", f"|gap| {st.gap.abs_size:.1f} > max {f.max_gap_points}")
        elif f.min_or_rvol and st.ctx.or_window_avg_volume:  # §7.4
            rvol = st.or_volume / st.ctx.or_window_avg_volume
            if rvol < f.min_or_rvol:
                skip("rvol", f"OR relative volume {rvol:.2f} < min {f.min_or_rvol}")
        return events

    # ---------------------------------------------------------- breakouts

    def _check_breakout(self, bar: Bar, st: _DayState) -> list[StrategyEvent]:
        orng = st.opening_range
        assert orng is not None
        if st.signals_emitted >= self.cfg.entry.max_positions_per_day:
            return []

        for side, boundary in ((Side.LONG, orng.high), (Side.SHORT, orng.low)):
            beyond = (bar.close - boundary) * side.sign
            if beyond <= 0 or side in st.sides_signalled:
                continue

            # This bar closed beyond an unused OR boundary.
            is_first = st.first_breakout_side is None
            kind = SignalKind.FIRST if is_first else SignalKind.SECOND
            if is_first:
                st.first_breakout_side = side
            elif side == st.first_breakout_side:
                continue  # same side can never signal twice

            event = self._evaluate_breakout(bar, st, side, boundary, beyond, kind)
            if event is not None:
                return [event]  # at most one signal per bar
        return []

    def _evaluate_breakout(
        self, bar: Bar, st: _DayState, side: Side, boundary: float, beyond: float, kind: SignalKind
    ) -> StrategyEvent | None:
        cfg = self.cfg
        orng = st.opening_range
        assert orng is not None
        s = st.ctx.session

        def skip(rule: str, detail: str) -> SkipEvent:
            # A skipped breakout still consumes the side: the base model trades
            # only the FIRST valid breakout of each boundary (§3.4).
            st.sides_signalled.add(side)
            return SkipEvent(bar.time, rule, f"{kind.value} {side.value}: {detail}")

        # -- time windows (§8)
        cutoff = s.pos1_cutoff if kind is SignalKind.FIRST else s.pos2_cutoff
        if bar.time >= cutoff:
            return skip("time_window", f"confirmation at {bar.time:%H:%M} UTC is past the cutoff")

        if kind is SignalKind.SECOND and not cfg.entry.allow_second_position:
            return skip("second_disabled", "second position disabled in config")

        # -- gap direction rule, first position only (§4). A zero gap has no
        # direction: the rule is inactive, either side allowed.
        gap_rule_active = (
            kind is SignalKind.FIRST
            and cfg.entry.first_in_gap_direction_only
            and st.gap is not None
            and st.gap.abs_size > 0
        )
        if gap_rule_active and side != st.gap.side:  # type: ignore[union-attr]
            allowed_counter = (
                cfg.entry.counter_gap_max_points > 0
                and st.gap.abs_size <= cfg.entry.counter_gap_max_points
            )
            if not allowed_counter:
                return skip(
                    "gap_direction",
                    f"breakout {side.value} against gap ({st.gap.size:+.1f}) — first position "
                    "is taken only in the gap direction",
                )

        # -- confirmation quality (§5)
        c = cfg.confirmation
        if c.require_body_close:
            directional = bar.close > bar.open if side is Side.LONG else bar.close < bar.open
            if not directional:
                return skip("confirmation", "breakout candle is not directional (body against breakout)")
        if bar.range > 0 and bar.body / bar.range < c.min_body_to_range:
            return skip(
                "confirmation",
                f"weak candle: body/range {bar.body / bar.range:.2f} < {c.min_body_to_range}",
            )
        min_break = max(c.min_break_points, c.min_break_or_frac * orng.width)
        if min_break and beyond < min_break:
            return skip("confirmation", f"marginal breakout: {beyond:.1f} < required {min_break:.1f}")

        # -- optional filters (§7)
        f = cfg.filters
        if f.or_direction_filter and st.or_close is not None and st.session_open_price is not None:
            # OR candle direction = close of the OR window vs the session open (§7.7).
            or_bullish = st.or_close >= st.session_open_price
            if (side is Side.LONG) != or_bullish:
                return skip("or_direction", "breakout against the OR candle direction")
        if f.vwap_filter and st.vwap_vol > 0:
            vwap = st.vwap_pv / st.vwap_vol
            if (bar.close - vwap) * side.sign < 0:
                return skip("vwap", f"close {bar.close:.1f} on the wrong side of VWAP {vwap:.1f}")
        if f.min_break_bar_rvol and st.bar_rvol is not None and st.bar_rvol < f.min_break_bar_rvol:
            return skip(  # §7.10
                "break_rvol",
                f"breakout candle volume {st.bar_rvol:.2f}x session average "
                f"< required {f.min_break_bar_rvol}",
            )
        if f.trend_ma_period and st.ctx.trend_ma is not None:  # §7.9
            if (bar.close - st.ctx.trend_ma) * side.sign < 0:
                return skip(
                    "trend_ma",
                    f"close {bar.close:.1f} on the wrong side of "
                    f"MA{f.trend_ma_period} {st.ctx.trend_ma:.1f}",
                )
            slope_bad = (
                f.trend_ma_require_slope
                and st.ctx.trend_ma_prev is not None
                and (st.ctx.trend_ma - st.ctx.trend_ma_prev) * side.sign < 0
            )
            if slope_bad:
                return skip("trend_ma", f"MA{f.trend_ma_period} slope is against the breakout")
        if f.require_fvg and st.fvg.recent(side, f.fvg_max_age_bars, f.fvg_min_size_points) is None:
            return skip("no_fvg", "no imbalance in the breakout direction")  # §7.11
        if f.news_filter_enabled:
            # The actual entry happens at the NEXT bar open = this bar's close
            # time, so the blackout window is tested at entry time (§7.1).
            entry_time = bar.time + timedelta(minutes=_TF_MINUTES.get(cfg.timeframe, 5))
            event = st.ctx.news.blocking_event(
                entry_time,
                currencies=f.news_currencies,
                min_impact=f.news_min_impact,
                before_min=f.news_buffer_before_min,
                after_min=f.news_buffer_after_min,
            )
            if event is not None:
                return skip(
                    "news", f"news blackout: {event.currency} {event.title} @ {event.time:%H:%M} UTC"
                )

        # -- entry distance / stop size (§5.4, §9)
        max_dist = cfg.entry.max_entry_distance_or_mult
        if max_dist and beyond > max_dist * orng.width:
            return skip(
                "entry_distance",
                f"confirmation too far from OR: {beyond:.1f} > "
                f"{cfg.entry.max_entry_distance_or_mult} x width {orng.width:.1f}",
            )

        stop = self._stop_loss(side, orng)
        risk_points = abs(bar.close - stop)
        if cfg.stops.max_sl_points and risk_points > cfg.stops.max_sl_points:
            return skip("stop_size", f"stop distance {risk_points:.1f} > max {cfg.stops.max_sl_points}")

        # -- build the signal (§3.3, §9, §10)
        tp_rr = cfg.targets.fixed_rr if TpMode(cfg.targets.tp_mode) is TpMode.FIXED_RR else None
        take_profit = bar.close + side.sign * tp_rr * risk_points if tp_rr else None

        st.sides_signalled.add(side)
        st.signals_emitted += 1
        gap_str = f"{st.gap.size:+.1f}" if st.gap else "n/a"
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
                f"{kind.value} breakout {side.value}: close {bar.close:.1f} beyond OR "
                f"{orng.low:.1f}..{orng.high:.1f} by {beyond:.1f}, gap {gap_str}"
            ),
            context={
                "or_high": orng.high,
                "or_low": orng.low,
                "or_width": round(orng.width, 2),
                "gap": round(st.gap.size, 2) if st.gap else "n/a",
                "boundary": boundary,
            },
        )

    def _stop_loss(self, side: Side, orng: OpeningRange) -> float:
        # §9: stop behind the FULL opening range (opposite boundary) + buffer.
        buf = self.cfg.stops.sl_buffer_points
        return orng.low - buf if side is Side.LONG else orng.high + buf
