"""Event-driven backtest engine.

Replays cached M5 bars through the SAME OrbStrategy and manage_position()
code used in live trading; only fills are simulated (broker_sim-like logic
inlined here for bar data).

Execution model (documented in STRATEGY.md §15 / config `backtest.*`):
  - entry at next bar OPEN after the confirmation bar, adverse-adjusted by
    half-spread + slippage;
  - SL/TP checked against every bar's high/low; if both hit within one bar,
    `pessimistic_same_bar` decides (default: SL first);
  - gap through a level fills at the bar open (worse) price;
  - trailing evaluated on bar CLOSE only (conservative: no intra-bar trailing);
  - forced flat at `session.flat_time`.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

from tradingbot.config import AppConfig
from tradingbot.core.clock import SessionSchedule
from tradingbot.core.models import (
    Bar,
    CloseNow,
    EntrySignal,
    ManagedPosition,
    ModifyStop,
    Side,
    SkipEvent,
)
from tradingbot.data.mt5_client import SymbolSpec
from tradingbot.data.news import NewsCalendar
from tradingbot.management.trade_manager import manage_position
from tradingbot.risk.position_sizing import SizingError, calc_volume
from tradingbot.strategy.base import DayContext, Strategy

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    day: date
    kind: str
    side: str
    entry_time: datetime
    entry: float
    initial_stop: float
    take_profit: float | None
    exit_time: datetime
    exit: float
    volume: float
    result_r: float
    pnl_money: float
    balance_after: float
    close_reason: str
    signal_reason: str


@dataclass
class BacktestResult:
    trades: list[TradeRecord]
    skips: list[SkipEvent]
    initial_balance: float
    final_balance: float
    days_processed: int


def _session_schedule(cfg: AppConfig) -> SessionSchedule:
    s = cfg.session
    return SessionSchedule(
        tz_name=s.timezone,
        session_open=s.open,
        or_minutes=s.or_minutes,
        pos1_cutoff=s.pos1_cutoff,
        pos2_cutoff=s.pos2_cutoff,
        flat_time=s.flat_time,
    )


class BacktestEngine:
    def __init__(self, cfg: AppConfig, strategy: Strategy, news: NewsCalendar | None = None):
        self.cfg = cfg
        self.strategy = strategy
        self.news = news or NewsCalendar.empty()
        bt = cfg.backtest
        self.spec = SymbolSpec(
            name=cfg.mt5.symbol,
            point=1.0,
            tick_size=1.0,
            tick_value=bt.point_value_per_lot,
            volume_min=bt.volume_min,
            volume_max=min(bt.volume_max, cfg.risk.max_volume_lots),
            volume_step=bt.volume_step,
            digits=1,
            contract_size=1.0,
        )
        self._half_spread = bt.spread_points / 2.0
        self._slip = bt.slippage_points

    # ------------------------------------------------------------------ run

    def run(self, bars: list[Bar]) -> BacktestResult:
        cfg = self.cfg
        schedule = _session_schedule(cfg)
        tz = ZoneInfo(cfg.session.timezone)
        days = self._group_by_day(bars, tz)
        # Flat series for the pre-session trend MA (§7.9); bars are already sorted.
        ma_times = [b.time for b in bars]
        ma_closes = [b.close for b in bars]

        balance = cfg.backtest.initial_balance
        trades: list[TradeRecord] = []
        skips: list[SkipEvent] = []
        prev_session_close: float | None = None
        session_ranges: list[tuple[float, float, float]] = []  # (high, low, prev_close) per day
        or_volumes: list[float] = []
        processed = 0

        for day, day_bars in days:
            if not schedule.is_trading_day(day):
                continue
            session = schedule.for_day(day)
            ctx = DayContext(
                session=session,
                prev_session_close=prev_session_close,
                atr_d1=self._atr(session_ranges, cfg.strategy.filters.atr_period),
                or_window_avg_volume=self._avg_or_volume(
                    or_volumes, cfg.strategy.filters.rvol_lookback_days
                ),
                news=self.news,
            )
            ctx.trend_ma, ctx.trend_ma_prev = self._trend_ma(
                ma_times, ma_closes, session.session_open, cfg.strategy.filters.trend_ma_period
            )
            self.strategy.on_day_start(ctx)

            day_trades, day_skips = self._run_day(day, day_bars, ctx, balance)
            for t in day_trades:
                balance += t.pnl_money
                t.balance_after = balance
            trades.extend(day_trades)
            skips.extend(day_skips)
            processed += 1

            # -- roll day-level aggregates used by next days' context
            close_utc = self._close_utc(session, tz)
            sess_bars = [b for b in day_bars if session.session_open <= b.time < close_utc]
            if sess_bars:
                hi = max(b.high for b in sess_bars)
                lo = min(b.low for b in sess_bars)
                session_ranges.append((hi, lo, prev_session_close if prev_session_close else lo))
                prev_session_close = sess_bars[-1].close
            or_bars = [b for b in day_bars if session.session_open <= b.time < session.or_end]
            if or_bars:
                or_volumes.append(sum(b.tick_volume for b in or_bars))

        return BacktestResult(
            trades=trades,
            skips=skips,
            initial_balance=cfg.backtest.initial_balance,
            final_balance=balance,
            days_processed=processed,
        )

    # -------------------------------------------------------------- one day

    def _run_day(
        self, day: date, day_bars: list[Bar], ctx: DayContext, balance: float
    ) -> tuple[list[TradeRecord], list[SkipEvent]]:
        cfg = self.cfg
        trades: list[TradeRecord] = []
        skips: list[SkipEvent] = []
        open_pos: ManagedPosition | None = None
        pending: EntrySignal | None = None

        for bar in day_bars:
            # 1) manage open position against this bar (SL/TP/flat/trailing)
            if open_pos is not None:
                closed = self._check_exit(open_pos, bar, ctx)
                if not closed:
                    for action in manage_position(
                        open_pos, bar.close, bar.time, ctx.session.flat_time, cfg.strategy.targets
                    ):
                        if isinstance(action, ModifyStop):
                            open_pos.stop_loss = action.new_stop
                        elif isinstance(action, CloseNow):
                            self._close_at(open_pos, bar.close, bar.time, action.reason)
                            closed = True
                if closed:
                    trades.append(self._record(day, open_pos))
                    self.strategy.on_position_closed(open_pos)
                    open_pos = None

            # 2) fill a pending entry at this bar's open
            if pending is not None:
                if open_pos is not None and cfg.strategy.entry.close_open_position_on_reversal:
                    self._close_at(open_pos, bar.open, bar.time, "reversal signal: position flipped")
                    trades.append(self._record(day, open_pos))
                    self.strategy.on_position_closed(open_pos)
                    open_pos = None
                if open_pos is None:
                    open_pos = self._fill(pending, bar, balance + sum(t.pnl_money for t in trades))
                    if open_pos is None:
                        skips.append(SkipEvent(bar.time, "sizing", "volume below broker minimum"))
                    elif self._check_exit(open_pos, bar, ctx):
                        # The fill bar itself can run through SL/TP after the
                        # open — test its full range immediately.
                        trades.append(self._record(day, open_pos))
                        self.strategy.on_position_closed(open_pos)
                        open_pos = None
                pending = None

            # 3) feed the completed bar to the strategy
            for event in self.strategy.on_bar(bar):
                if isinstance(event, EntrySignal):
                    pending = event
                elif isinstance(event, SkipEvent):
                    skips.append(event)

        # day ended with a position still open (data gap before flat_time)
        if open_pos is not None and day_bars:
            last = day_bars[-1]
            self._close_at(open_pos, last.close, last.time, "end of day data")
            trades.append(self._record(day, open_pos))
            self.strategy.on_position_closed(open_pos)
        return trades, skips

    # ------------------------------------------------------------- fills

    def _fill(self, sig: EntrySignal, bar: Bar, balance: float) -> ManagedPosition | None:
        cost = self._half_spread + self._slip
        fill = bar.open + sig.side.sign * cost
        stop_distance = abs(fill - sig.stop_loss)
        if stop_distance <= 0:
            return None
        try:
            sizing = calc_volume(
                balance=balance,
                risk_pct=self.cfg.risk.risk_per_trade_pct,
                stop_distance=stop_distance,
                spec=self.spec,
                min_volume_fallback=self.cfg.risk.min_volume_fallback,
            )
        except SizingError as exc:
            log.warning("Sizing failed on %s: %s", bar.time, exc)
            return None
        return ManagedPosition(
            signal=sig,
            entry_price=fill,
            volume=sizing.volume,
            opened_at=bar.time,
            initial_stop=sig.stop_loss,
            stop_loss=sig.stop_loss,
            take_profit=sig.take_profit_for_fill(fill),
        )

    def _check_exit(self, pos: ManagedPosition, bar: Bar, ctx: DayContext) -> bool:
        """SL/TP/flat checks for this bar. Returns True if position closed."""
        sign = pos.side.sign

        if bar.time >= ctx.session.flat_time:
            self._close_at(pos, bar.open, bar.time, "flat_time: end-of-day forced exit")
            return True

        if pos.side is Side.LONG:
            sl_hit = bar.low <= pos.stop_loss
            tp_hit = pos.take_profit is not None and bar.high >= pos.take_profit
            gap_through_sl = bar.open <= pos.stop_loss
            gap_through_tp = pos.take_profit is not None and bar.open >= pos.take_profit
        else:
            sl_hit = bar.high >= pos.stop_loss
            tp_hit = pos.take_profit is not None and bar.low <= pos.take_profit
            gap_through_sl = bar.open >= pos.stop_loss
            gap_through_tp = pos.take_profit is not None and bar.open <= pos.take_profit

        if sl_hit and tp_hit:
            # The bar OPEN is chronologically first — a gap through a level is
            # deterministic and overrides the pessimistic tie-break.
            if gap_through_sl:
                first_sl = True
            elif gap_through_tp:
                first_sl = False
            else:
                first_sl = self.cfg.backtest.pessimistic_same_bar
        elif sl_hit or tp_hit:
            first_sl = sl_hit
        else:
            return False

        if first_sl:
            price = bar.open if gap_through_sl else pos.stop_loss
            self._close_at(pos, price - sign * self._slip, bar.time, "stop loss hit")
        else:
            assert pos.take_profit is not None
            price = bar.open if gap_through_tp else pos.take_profit
            # TP is a server-side limit order: fills at the level, no spread cost
            # (bid-quoted bars make this slightly optimistic for shorts).
            self._close_at(pos, price, bar.time, "take profit hit", spread_free=not gap_through_tp)
        return True

    def _close_at(
        self, pos: ManagedPosition, price: float, when: datetime, reason: str, spread_free: bool = False
    ) -> None:
        # Market/stop exits pay half-spread against you; limit (TP) exits don't.
        exit_price = price if spread_free else price - pos.side.sign * self._half_spread
        pos.closed = True
        pos.close_price = exit_price
        pos.closed_at = when
        pos.close_reason = reason

    def _record(self, day: date, pos: ManagedPosition) -> TradeRecord:
        assert pos.close_price is not None and pos.closed_at is not None
        points = (pos.close_price - pos.entry_price) * pos.side.sign
        pnl = points * pos.volume * self.spec.value_per_point
        pnl -= pos.volume * self.cfg.backtest.commission_per_lot
        return TradeRecord(
            day=day,
            kind=pos.signal.kind.value,
            side=pos.side.value,
            entry_time=pos.opened_at,
            entry=pos.entry_price,
            initial_stop=pos.initial_stop,
            take_profit=pos.take_profit,
            exit_time=pos.closed_at,
            exit=pos.close_price,
            volume=pos.volume,
            result_r=pos.result_r or 0.0,
            pnl_money=pnl,
            balance_after=0.0,  # filled by run()
            close_reason=pos.close_reason or "",
            signal_reason=pos.signal.reason,
        )

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _group_by_day(bars: list[Bar], tz: ZoneInfo) -> list[tuple[date, list[Bar]]]:
        grouped: dict[date, list[Bar]] = {}
        for b in bars:
            grouped.setdefault(b.time.astimezone(tz).date(), []).append(b)
        return sorted(grouped.items())

    def _close_utc(self, session, tz: ZoneInfo) -> datetime:
        hh, mm = self.cfg.session.close.split(":")
        local = datetime.combine(session.day, datetime.min.time(), tzinfo=tz).replace(
            hour=int(hh), minute=int(mm)
        )
        return local.astimezone(ZoneInfo("UTC"))

    @staticmethod
    def _atr(session_ranges: list[tuple[float, float, float]], period: int) -> float | None:
        """Simple ATR over session-hours daily ranges (true range vs prev close)."""
        if len(session_ranges) < max(period, 2):
            return None
        window = session_ranges[-period:]
        trs = [max(hi - lo, abs(hi - pc), abs(lo - pc)) for hi, lo, pc in window]
        return sum(trs) / len(trs)

    @staticmethod
    def _trend_ma(
        times: list[datetime], closes: list[float], session_open: datetime, period: int
    ) -> tuple[float | None, float | None]:
        """MA of the `period` closes preceding the session open, and one bar earlier.

        Uses only bars strictly before the open, so the value is knowable at the
        moment the session starts and carries no look-ahead.
        """
        if period <= 0:
            return None, None
        end = bisect_left(times, session_open)
        if end < period + 1:
            return None, None
        ma = sum(closes[end - period : end]) / period
        ma_prev = sum(closes[end - period - 1 : end - 1]) / period
        return ma, ma_prev

    @staticmethod
    def _avg_or_volume(or_volumes: list[float], lookback: int) -> float | None:
        if len(or_volumes) < max(lookback // 2, 3):
            return None
        window = or_volumes[-lookback:]
        return sum(window) / len(window)
