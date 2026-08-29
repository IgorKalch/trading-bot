"""Live trading loop.

Drives the same OrbStrategy + manage_position() as the backtest, against
real-time MT5 data. Every action is logged and mirrored to Telegram.

Cycle (every `bot.poll_seconds`):
  1. day rollover -> refresh news, build DayContext (transactional: retried
     until it fully succeeds); prop-guard anchors keyed to platform time UTC+3
  2. feed newly CLOSED M5 bars to the strategy (stale replayed signals are
     rebuilt for state but never executed)
  3. execute entry signals (news gate -> spread -> sizing -> guard -> close
     reversal -> open -> persist -> notify)
  4. manage open positions (sync SL/TP hits, pre-news flatten, trailing,
     flat_time, emergency guard flat)
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from tradingbot.config import AppConfig
from tradingbot.core.clock import SessionSchedule, SessionTimes
from tradingbot.core.models import (
    Bar,
    CloseNow,
    EntrySignal,
    InfoEvent,
    ManagedPosition,
    ModifyStop,
    Side,
    SkipEvent,
)
from tradingbot.data.mt5_client import Mt5Client
from tradingbot.data.news import NewsCalendar
from tradingbot.execution.base import ExecutionClient
from tradingbot.live.state import DayStore, PositionStore
from tradingbot.management.trade_manager import manage_position
from tradingbot.notify import formatter as fmt
from tradingbot.notify.telegram import TelegramNotifier
from tradingbot.risk.position_sizing import SizingError, calc_volume
from tradingbot.risk.prop_guard import PropGuard
from tradingbot.strategy.base import DayContext, Strategy

log = logging.getLogger(__name__)

PLATFORM_TZ = timezone(timedelta(hours=3))  # Funding Pips daily reset: 00:00 UTC+3
STALE_SIGNAL_TOLERANCE = timedelta(seconds=90)


class LiveRunner:
    def __init__(
        self,
        cfg: AppConfig,
        client: Mt5Client,
        executor: ExecutionClient,
        strategy: Strategy,
        notifier: TelegramNotifier,
        guard: PropGuard,
    ):
        self.cfg = cfg
        self.client = client
        self.executor = executor
        self.strategy = strategy
        self.notifier = notifier
        self.guard = guard
        self.symbol = cfg.mt5.symbol
        s = cfg.session
        self.schedule = SessionSchedule(
            tz_name=s.timezone, session_open=s.open, or_minutes=s.or_minutes,
            pos1_cutoff=s.pos1_cutoff, pos2_cutoff=s.pos2_cutoff, flat_time=s.flat_time,
        )
        self._tf_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60}.get(
            cfg.strategy.timeframe, 5
        )
        self._pos_store = PositionStore(cfg.bot.state_dir)
        self._day_store = DayStore(cfg.bot.state_dir)
        self.positions: list[ManagedPosition] = []
        self.news = NewsCalendar.empty()
        self._session: SessionTimes | None = None
        self._last_bar_time: datetime | None = None
        self._trading_halted_today = False
        self._last_heartbeat = datetime.now(tz=UTC)

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        self.client.connect()
        self.client.select_symbol(self.symbol)
        self.positions = self._reconcile_positions()
        mode = self.cfg.bot.mode
        if self.cfg.prop.enabled and self.cfg.prop.initial_balance <= 0:
            self.notifier.send(
                "⚠️ <b>prop.initial_balance не задано</b> — guard зафіксує ПОТОЧНИЙ баланс "
                "як стартовий. Якщо рахунок уже має історію угод, задай стартовий баланс "
                "рахунку в config.yaml явно!"
            )
        self.notifier.send(
            f"🤖 <b>ORB бот запущено</b> ({mode} mode)\n"
            f"Символ: {self.symbol}, ризик {self.cfg.risk.risk_per_trade_pct}%/угоду, "
            f"TP: {self.cfg.strategy.targets.tp_mode} {self.cfg.strategy.targets.fixed_rr}R"
        )
        try:
            while True:
                try:
                    self._tick()
                except Exception as exc:  # noqa: BLE001 — keep the loop alive
                    log.exception("Main loop iteration failed")
                    self.notifier.send(fmt.fmt_error("у головному циклі", exc))
                    time.sleep(10)
                time.sleep(self.cfg.bot.poll_seconds)
        except KeyboardInterrupt:
            log.info("Shutdown requested")
        finally:
            self.notifier.send("🛑 <b>ORB бот зупинено</b>")
            self.notifier.close()
            self.client.shutdown()

    # ----------------------------------------------------------------- tick

    def _tick(self) -> None:
        self.client.ensure_connected()
        now = datetime.now(tz=UTC)
        today = now.astimezone(ZoneInfo(self.cfg.session.timezone)).date()

        # Prop-guard anchors follow the PLATFORM day (00:00 UTC+3), which is
        # not the same boundary as the session (Berlin) day.
        self.guard.on_day_start(now.astimezone(PLATFORM_TZ).date(), self.client.account())

        if self._session is None or self._session.day != today:
            self._start_day(today)  # raises on failure -> retried next tick
        assert self._session is not None

        self._heartbeat(now)
        self._manage_positions(now)

        if not self.schedule.is_trading_day(today) or self._trading_halted_today:
            return

        for bar in self._new_closed_bars(now):
            for event in self.strategy.on_bar(bar):
                if isinstance(event, EntrySignal):
                    self._handle_signal(event, now)
                elif isinstance(event, SkipEvent):
                    log.info("Skip [%s]: %s", event.rule, event.detail)
                    self.notifier.send(fmt.fmt_skip(event, self.symbol))
                elif isinstance(event, InfoEvent):
                    log.info("Strategy: %s", event.message)
                    self.notifier.send(f"ℹ️ {event.message}")

    # ------------------------------------------------------------ day start

    def _start_day(self, today) -> None:
        """Initialise the trading day. Transactional: self._session is only
        assigned after every step succeeded, so a transient MT5/feed failure
        is retried on the next tick instead of silently killing the day."""
        session = self.schedule.for_day(today)
        account = self.client.account()

        news = NewsCalendar.empty()
        if self.cfg.strategy.filters.news_filter_enabled or self.cfg.prop.restrict_news_trading:
            news = NewsCalendar.fetch_forexfactory(cache_dir="data/news")
            if not news.events and self.cfg.prop.restrict_news_trading:
                self.notifier.send(
                    "⚠️ <b>Календар новин порожній</b> — фільтр новин сьогодні не має даних! "
                    + ("Входи заблоковано (prop.require_news_data)."
                       if self.cfg.prop.require_news_data
                       else "Торгівля триває БЕЗ захисту від новин — будь уважний.")
                )

        ctx = DayContext(session=session, news=news)
        try:
            ctx.prev_session_close = self._prev_session_close(session)
            ctx.atr_d1, ctx.or_window_avg_volume = self._daily_aggregates(session)
            ctx.trend_ma, ctx.trend_ma_prev = self._trend_ma(session)
        except Exception as exc:  # noqa: BLE001 — context extras are optional
            log.warning("Day context extras unavailable: %s", exc)
        self.strategy.on_day_start(ctx)

        # Commit only now — everything above may raise and will be retried.
        self.news = news
        self._session = session
        self._trading_halted_today = False
        self._last_bar_time = None
        self._day_store.set_day(today)

        if self.schedule.is_trading_day(today):
            prev = "невідоме" if ctx.prev_session_close is None else f"{ctx.prev_session_close:.1f}"
            self.notifier.send(
                f"📅 <b>Новий торговий день {today}</b>\n"
                f"Баланс: {account.balance:,.2f}  Еквіті: {account.equity:,.2f}\n"
                f"Закриття минулої сесії: {prev}\n"
                f"Денний стоп бота: {self.guard.daily_soft_floor():,.2f} "
                f"(ліміт фірми {self.guard.daily_loss_floor():,.2f})"
            )

    def _prev_session_close(self, session: SessionTimes) -> float | None:
        """Close of the last M5 bar of the previous session, scoped to that
        day's cash session so a holiday's evening CFD bars are never used."""
        bars = self.client.get_recent_bars(self.symbol, "M5", 3000)
        bar_len = timedelta(minutes=self._tf_minutes)
        for days_back in range(1, 8):
            day = session.day - timedelta(days=days_back)
            if not self.schedule.is_trading_day(day):
                continue
            open_utc = self._local_time_utc(day, self.cfg.session.open)
            close_utc = self._local_time_utc(day, self.cfg.session.close)
            in_session = [b for b in bars if open_utc <= b.time and b.time + bar_len <= close_utc]
            if in_session:
                return in_session[-1].close
            # Exchange holiday (weekday without session bars) — look further back.
        return None

    def _daily_aggregates(self, session: SessionTimes) -> tuple[float | None, float | None]:
        """(ATR of session-hours daily ranges, avg OR-window volume)."""
        f = self.cfg.strategy.filters
        need_atr = f.max_or_width_atr_mult > 0
        need_rvol = f.min_or_rvol > 0
        if not (need_atr or need_rvol):
            return None, None
        bars = self.client.get_recent_bars(self.symbol, "M5", 8000)
        ranges: list[tuple[float, float, float]] = []
        or_vols: list[float] = []
        prev_close: float | None = None
        for days_back in range(max(f.atr_period, f.rvol_lookback_days) + 10, 0, -1):
            day = session.day - timedelta(days=days_back)
            if not self.schedule.is_trading_day(day):
                continue
            open_utc = self._local_time_utc(day, self.cfg.session.open)
            close_utc = self._local_time_utc(day, self.cfg.session.close)
            or_end = open_utc + timedelta(minutes=self.cfg.session.or_minutes)
            sess = [b for b in bars if open_utc <= b.time < close_utc]
            if not sess:
                continue
            hi, lo = max(b.high for b in sess), min(b.low for b in sess)
            ranges.append((hi, lo, prev_close if prev_close is not None else lo))
            prev_close = sess[-1].close
            or_vols.append(sum(b.tick_volume for b in sess if b.time < or_end))
        atr = None
        if need_atr and len(ranges) >= f.atr_period:
            window = ranges[-f.atr_period:]
            atr = sum(max(h - lo, abs(h - pc), abs(lo - pc)) for h, lo, pc in window) / len(window)
        rvol_avg = None
        if need_rvol and len(or_vols) >= 3:
            window_v = or_vols[-f.rvol_lookback_days:]
            rvol_avg = sum(window_v) / len(window_v)
        return atr, rvol_avg

    def _trend_ma(self, session: SessionTimes) -> tuple[float | None, float | None]:
        """MA of the traded timeframe over the bars before today's open (§7.9)."""
        period = self.cfg.strategy.filters.trend_ma_period
        if period <= 0:
            return None, None
        tf = self.cfg.strategy.timeframe
        bars = self.client.get_recent_bars(self.symbol, tf, period + 50)
        prior = [b.close for b in bars if b.time < session.session_open]
        if len(prior) < period + 1:
            log.warning("Trend MA(%d) unavailable: only %d bars before the open", period, len(prior))
            return None, None
        ma = sum(prior[-period:]) / period
        ma_prev = sum(prior[-period - 1 : -1]) / period
        return ma, ma_prev

    def _local_time_utc(self, day, hhmm: str) -> datetime:
        hh, mm = hhmm.split(":")
        local = datetime(day.year, day.month, day.day, int(hh), int(mm),
                         tzinfo=ZoneInfo(self.cfg.session.timezone))
        return local.astimezone(UTC)

    # ------------------------------------------------------------------ bars

    def _new_closed_bars(self, now: datetime) -> list[Bar]:
        """Fetch bars and return the newly CLOSED ones since the last call."""
        first_poll = self._last_bar_time is None
        if first_poll:
            count = 300  # covers a full day for intraday restarts
        else:
            # Scale to the actual gap so a long outage never drops bars.
            gap_bars = int((now - self._last_bar_time).total_seconds() / 60 / self._tf_minutes)
            count = max(10, gap_bars + 5)
        bars = self.client.get_recent_bars(self.symbol, self.cfg.strategy.timeframe, count)
        if len(bars) < 2:
            return []
        closed = bars[:-1]  # last element is the still-forming bar
        if first_poll:
            # First poll of the day: replay today's bars so an intraday restart
            # rebuilds OR/breakout state. Stale signals emitted during the
            # replay are suppressed in _handle_signal.
            assert self._session is not None
            day_start = self._session.session_open - timedelta(hours=2)
            fresh = [b for b in closed if b.time >= day_start]
        else:
            fresh = [b for b in closed if b.time > self._last_bar_time]  # type: ignore[operator]
        if fresh:
            self._last_bar_time = fresh[-1].time
        return fresh

    # --------------------------------------------------------------- signals

    def _handle_signal(self, sig: EntrySignal, now: datetime) -> None:
        cfg = self.cfg

        # Replayed/stale signal (restart or late start): the entry moment was
        # the close of the confirmation bar — if that is already in the past,
        # rebuild-only: never execute at today's price (STRATEGY.md §3.3).
        entry_moment = sig.time + timedelta(minutes=self._tf_minutes)
        if now - entry_moment > STALE_SIGNAL_TOLERANCE:
            log.info("Suppressing stale signal from %s (now %s)", sig.time, now)
            self.notifier.send(
                f"♻️ Сигнал {sig.side.value} від {sig.time:%H:%M} UTC відновлено після рестарту — "
                "НЕ виконується (застарілий)."
            )
            return

        self.notifier.send(fmt.fmt_signal(sig, self.symbol, cfg.bot.mode))

        # News blackout gates BOTH the reversal close and the new entry —
        # closing inside a red-folder window violates the Master rule too.
        account = self.client.account()
        verdict = self.guard.can_open(
            account, 0.0, now, self.news, cfg.strategy.filters.news_currencies
        )
        if not verdict.allowed and verdict.rule == "news_window":
            self.notifier.send(fmt.fmt_guard_block(verdict.rule, verdict.detail))
            return
        if (
            cfg.prop.enabled
            and cfg.prop.restrict_news_trading
            and cfg.prop.require_news_data
            and not self.news.events
        ):
            self.notifier.send(fmt.fmt_guard_block(
                "news_data", "календар новин недоступний, входи заборонено (require_news_data)"))
            return

        # Spread sanity check (§7.8).
        tick = self.client.current_tick(self.symbol)
        spread = tick.ask - tick.bid
        if cfg.strategy.filters.max_spread_points and spread > cfg.strategy.filters.max_spread_points:
            self.notifier.send(fmt.fmt_guard_block(
                "spread", f"спред {spread:.1f} п. > ліміту {cfg.strategy.filters.max_spread_points}"))
            return

        # Position sizing (§13) — the 20-lot cap is folded into the spec so the
        # reported/guarded risk matches the actually traded volume.
        from dataclasses import replace

        spec = self.client.symbol_spec(self.symbol)
        spec = replace(spec, volume_max=min(spec.volume_max, cfg.risk.max_volume_lots))
        entry_est = tick.ask if sig.side is Side.LONG else tick.bid
        stop_distance = abs(entry_est - sig.stop_loss)
        try:
            sizing = calc_volume(
                balance=account.balance,
                risk_pct=cfg.risk.risk_per_trade_pct,
                stop_distance=stop_distance,
                spec=spec,
                min_volume_fallback=cfg.risk.min_volume_fallback,
            )
        except SizingError as exc:
            self.notifier.send(fmt.fmt_error("розрахунок обсягу", exc))
            return

        # Funding Pips guard with the real planned risk (§14).
        verdict = self.guard.can_open(
            account, sizing.risk_money_actual, now, self.news, cfg.strategy.filters.news_currencies
        )
        if not verdict.allowed:
            self.notifier.send(fmt.fmt_guard_block(verdict.rule, verdict.detail))
            if verdict.rule in ("daily_loss", "max_drawdown"):
                self._trading_halted_today = True
                self.notifier.send("⛔️ Торгівлю на сьогодні зупинено (ліміт ризику).")
            return

        # All checks passed — only now flip the open opposite position (§12.4).
        for pos in self._open_positions():
            flip = cfg.strategy.entry.close_open_position_on_reversal and pos.side != sig.side
            if flip and self.executor.close(pos, "reversal signal: position flipped"):
                self.notifier.send(fmt.fmt_closed(pos, self.symbol))
                self.strategy.on_position_closed(pos)
        self._pos_store.save(self.positions)
        if self._open_positions():
            log.warning("Signal while a position is still open — skipping entry")
            self.notifier.send(fmt.fmt_guard_block("one_position", "позиція вже відкрита"))
            return

        pos = self.executor.open_market(sig, sizing.volume)
        if pos is None:
            self.notifier.send(fmt.fmt_error("відкриття позиції", "ордер не виконано (див. логи)"))
            return
        # Persist FIRST — a failure below must never orphan a live position.
        self.positions.append(pos)
        self._pos_store.save(self.positions)
        try:
            # Exact-RR take profit from the actual fill (§10).
            new_tp = sig.take_profit_for_fill(pos.entry_price)
            if new_tp is not None and abs((pos.take_profit or 0) - new_tp) > spec.point:
                pos.take_profit = new_tp
                self.executor.modify_stop(pos, pos.stop_loss)
                self._pos_store.save(self.positions)
        except Exception as exc:  # noqa: BLE001 — position is already tracked
            log.error("TP adjust failed (position stays tracked): %s", exc)
            self.notifier.send(fmt.fmt_error("коригування TP (позиція під контролем)", exc))
        self.notifier.send(
            fmt.fmt_opened(pos, self.symbol, sizing.risk_money_actual, cfg.risk.risk_per_trade_pct)
        )

    # ------------------------------------------------------- position mgmt

    def _open_positions(self) -> list[ManagedPosition]:
        return [p for p in self.positions if not p.closed]

    def _manage_positions(self, now: datetime) -> None:
        if not self._open_positions():
            return
        assert self._session is not None
        changed = False

        # Sync with the broker FIRST so we never act on already-closed positions.
        for pos in self._open_positions():
            old_sl = pos.stop_loss
            self.executor.sync(pos)
            if pos.closed:
                self.notifier.send(fmt.fmt_closed(pos, self.symbol))
                self.strategy.on_position_closed(pos)
                changed = True
            elif pos.stop_loss != old_sl:
                changed = True  # SL changed outside the bot — persist it

        open_now = self._open_positions()
        if not open_now:
            if changed:
                self._pos_store.save(self.positions)
            return

        # Emergency prop-guard flat (§12.2).
        account = self.client.account()
        emergency = self.guard.emergency_close_needed(account)
        if emergency.allowed:
            for pos in open_now:
                if self.executor.close(pos, f"EMERGENCY {emergency.rule}: {emergency.detail}"):
                    self.notifier.send(fmt.fmt_closed(pos, self.symbol))
                    self.strategy.on_position_closed(pos)
                else:
                    self.notifier.send(fmt.fmt_error(
                        "аварійне закриття", f"тікет {pos.ticket} не закрився — повтор через "
                        f"{self.cfg.bot.poll_seconds}с"))
            self._trading_halted_today = True
            self.notifier.send(f"⛔️ <b>Аварійне закриття</b>: {emergency.detail}. Стоп до кінця дня.")
            self._pos_store.save(self.positions)
            return

        # Pre-news flatten: close before the red-folder window opens so no
        # close ever lands inside it (§14).
        p = self.cfg.prop
        if p.enabled and p.restrict_news_trading and p.flatten_before_news:
            lookahead = now + timedelta(minutes=self.cfg.prop.news_window_before_min + 2)
            event = self.guard.news_event_ahead(
                lookahead, self.news, self.cfg.strategy.filters.news_currencies
            )
            if event is not None:
                for pos in open_now:
                    if self.executor.close(
                        pos, f"pre-news flatten: {event.currency} {event.title} @ {event.time:%H:%M} UTC"
                    ):
                        self.notifier.send(fmt.fmt_closed(pos, self.symbol))
                        self.strategy.on_position_closed(pos)
                self._pos_store.save(self.positions)
                return

        tick = self.client.current_tick(self.symbol)
        for pos in self._open_positions():
            price = tick.bid if pos.side is Side.LONG else tick.ask
            for action in manage_position(
                pos, price, now, self._session.flat_time, self.cfg.strategy.targets
            ):
                if isinstance(action, ModifyStop):
                    if self.executor.modify_stop(pos, action.new_stop):
                        self.notifier.send(fmt.fmt_sl_moved(pos, self.symbol, action.reason))
                        changed = True
                elif isinstance(action, CloseNow) and self.executor.close(pos, action.reason):
                    self.notifier.send(fmt.fmt_closed(pos, self.symbol))
                    self.strategy.on_position_closed(pos)
                    changed = True
        if changed:
            self._pos_store.save(self.positions)

    def _reconcile_positions(self) -> list[ManagedPosition]:
        """On start: merge persisted positions with what the broker reports,
        and ADOPT any broker position carrying our magic that we lost track of."""
        stored = self._pos_store.load()
        for pos in stored:
            try:
                self.executor.sync(pos)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not sync stored position %s: %s", pos.ticket, exc)
        known_tickets = {p.ticket for p in stored}
        try:
            for orphan in self.executor.list_broker_positions():
                if orphan.ticket not in known_tickets:
                    stored.append(orphan)
                    log.warning("Adopted orphan broker position ticket=%s", orphan.ticket)
                    self.notifier.send(
                        f"♻️ <b>Прийнято «осиротілу» позицію</b> #{orphan.ticket} "
                        f"({orphan.side.value}, SL {orphan.stop_loss:.1f}) — беру під керування."
                    )
        except Exception as exc:  # noqa: BLE001
            log.warning("Broker position scan failed: %s", exc)
        alive = [p for p in stored if not p.closed]
        if alive:
            log.info("Recovered %d open position(s) from state", len(alive))
            self.notifier.send(f"♻️ Відновлено відкритих позицій після рестарту: {len(alive)}")
        self._pos_store.save(stored)
        return stored

    def _heartbeat(self, now: datetime) -> None:
        hb = self.cfg.telegram.heartbeat_minutes
        if hb and now - self._last_heartbeat >= timedelta(minutes=hb):
            self._last_heartbeat = now
            account = self.client.account()
            self.notifier.send(
                f"💓 Бот працює. Еквіті: {account.equity:,.2f}, "
                f"відкрито позицій: {len(self._open_positions())}"
            )
