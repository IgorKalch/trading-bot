"""Live order execution on MT5 with retry on transient rejects.

Retry behavior: transient retcodes (requote, price change, no connection...)
are retried with a FRESH price from the current tick — resending a stale
price is exactly what keeps failing on a fast open.

Partial fills (TRADE_RETCODE_DONE_PARTIAL, normal for IOC) are treated as
success for the filled volume; close() reduces the tracked volume and lets
the next management cycle retry the remainder.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from tradingbot.core.models import EntrySignal, ManagedPosition, Side, SignalKind
from tradingbot.data.mt5_client import Mt5Client, Mt5Error

log = logging.getLogger(__name__)

# Retcodes worth retrying (transient market conditions).
_TRANSIENT_RETCODES = frozenset({
    10004,  # REQUOTE
    10018,  # MARKET_CLOSED (may open within the retry window at session start)
    10020,  # PRICE_CHANGED (market-execution twin of a requote)
    10021,  # PRICE_OFF (no quotes to process request)
    10024,  # TOO_MANY_REQUESTS
    10031,  # NO_CONNECTION
})


class Mt5Executor:
    def __init__(self, client: Mt5Client, symbol: str, magic: int, deviation_points: int = 100):
        self.client = client
        self.symbol = symbol
        self.magic = magic
        self.deviation = deviation_points

    # -- helpers -------------------------------------------------------------

    def _mt5(self):
        import MetaTrader5 as mt5  # noqa: PLC0415

        return mt5

    def _filling_mode(self) -> int:
        """Pick a filling mode the symbol supports (IOC preferred)."""
        mt5 = self._mt5()
        info = mt5.symbol_info(self.symbol)
        if info is None:
            raise Mt5Error(f"symbol_info({self.symbol}) failed")
        modes = info.filling_mode  # bitmask: 1=FOK, 2=IOC
        if modes & 2:
            return mt5.ORDER_FILLING_IOC
        if modes & 1:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def _send_with_retry(
        self,
        request: dict,
        attempts: int = 4,
        refresh_price: Callable[[dict], None] | None = None,
    ) -> object | None:
        mt5 = self._mt5()
        done_partial = getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)
        for attempt in range(1, attempts + 1):
            self.client.ensure_connected()
            if refresh_price is not None and attempt > 1:
                try:
                    refresh_price(request)
                except Mt5Error as exc:
                    log.warning("Price refresh failed before retry: %s", exc)
            result = mt5.order_send(request)
            if result is None:
                log.error("order_send returned None: %s (attempt %d)", mt5.last_error(), attempt)
            elif result.retcode in (mt5.TRADE_RETCODE_DONE, done_partial):
                if result.retcode == done_partial:
                    log.warning("order_send PARTIAL fill: %.2f of %.2f lots",
                                result.volume, request.get("volume", 0.0))
                return result
            elif result.retcode in _TRANSIENT_RETCODES:
                log.warning("order_send transient retcode=%s (%s), attempt %d/%d",
                            result.retcode, result.comment, attempt, attempts)
            else:
                log.error("order_send rejected: retcode=%s comment=%s request=%s",
                          result.retcode, result.comment, request)
                return None
            time.sleep(min(2.0 * attempt, 5.0))
        return None

    # -- ExecutionClient -----------------------------------------------------

    def open_market(self, signal: EntrySignal, volume: float) -> ManagedPosition | None:
        mt5 = self._mt5()
        tick = self.client.current_tick(self.symbol)
        price = tick.ask if signal.side is Side.LONG else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if signal.side is Side.LONG else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit or 0.0,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": f"ORB {signal.kind.value}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(),
        }

        def refresh(req: dict) -> None:
            t = self.client.current_tick(self.symbol)
            req["price"] = t.ask if signal.side is Side.LONG else t.bid

        result = self._send_with_retry(request, refresh_price=refresh)
        if result is None:
            return None
        fill_price = result.price or price
        fill_volume = result.volume or volume
        pos = ManagedPosition(
            signal=signal,
            entry_price=fill_price,
            volume=fill_volume,
            opened_at=tick.time,
            initial_stop=signal.stop_loss,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            ticket=getattr(result, "order", None) or getattr(result, "deal", None),
        )
        log.info("Opened %s %s %.2f lots @ %.2f (ticket=%s)",
                 signal.side.value, self.symbol, pos.volume, fill_price, pos.ticket)
        return pos

    def modify_stop(self, pos: ManagedPosition, new_stop: float) -> bool:
        mt5 = self._mt5()
        if pos.ticket is None:
            return False
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": pos.ticket,
            "sl": new_stop,
            "tp": pos.take_profit or 0.0,
            "magic": self.magic,
        }
        result = self._send_with_retry(request)
        if result is None:
            return False
        pos.stop_loss = new_stop
        log.info("Modified SL of ticket=%s to %.2f", pos.ticket, new_stop)
        return True

    def close(self, pos: ManagedPosition, reason: str) -> bool:
        mt5 = self._mt5()
        if pos.ticket is None or pos.closed:
            return False
        tick = self.client.current_tick(self.symbol)
        price = tick.bid if pos.side is Side.LONG else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.side is Side.LONG else mt5.ORDER_TYPE_BUY,
            "position": pos.ticket,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": f"close: {reason[:20]}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(),
        }

        def refresh(req: dict) -> None:
            t = self.client.current_tick(self.symbol)
            req["price"] = t.bid if pos.side is Side.LONG else t.ask

        result = self._send_with_retry(request, refresh_price=refresh)
        if result is None:
            return False
        filled = result.volume or pos.volume
        if filled + 1e-9 < pos.volume:
            # Partial close: keep the remainder under management, retry later.
            pos.volume = round(pos.volume - filled, 8)
            log.warning("Partial close of ticket=%s: %.2f lots remain", pos.ticket, pos.volume)
            return False
        pos.closed = True
        pos.close_price = result.price or price
        pos.closed_at = tick.time
        pos.close_reason = reason
        log.info("Closed ticket=%s @ %.2f (%s)", pos.ticket, pos.close_price, reason)
        return True

    def sync(self, pos: ManagedPosition) -> None:
        """If the broker closed the position (SL/TP hit), reflect it locally.

        Transient API failures must NOT be mistaken for 'position closed':
        we only mark a position closed when the closing deal is found.
        """
        mt5 = self._mt5()
        if pos.ticket is None or pos.closed:
            return
        self.client.ensure_connected()
        live = mt5.positions_get(ticket=pos.ticket)
        if live is None:
            log.warning("positions_get(ticket=%s) failed: %s — keeping position under management",
                        pos.ticket, mt5.last_error())
            return
        if live:
            # Still open; pick up SL changes made outside the bot, if any.
            pos.stop_loss = live[0].sl or pos.stop_loss
            return
        # Position is gone — find the closing deal in recent history.
        now_srv = self.client.clock.from_utc(datetime.now(tz=UTC))
        deals = mt5.history_deals_get(
            (now_srv - timedelta(days=3)).replace(tzinfo=UTC),
            (now_srv + timedelta(days=1)).replace(tzinfo=UTC),
            position=pos.ticket,
        )
        if not deals:
            # History may lag right after an SL/TP hit — retry on the next poll.
            log.warning("Ticket %s not in positions and closing deal not found yet — will re-check",
                        pos.ticket)
            return
        closing = deals[-1]
        pos.closed = True
        pos.close_price = closing.price
        server_naive = datetime.fromtimestamp(int(closing.time), tz=UTC).replace(tzinfo=None)
        pos.closed_at = self.client.clock.to_utc(server_naive)
        reason_map = {
            getattr(mt5, "DEAL_REASON_SL", 4): "stop loss hit",
            getattr(mt5, "DEAL_REASON_TP", 5): "take profit hit",
        }
        pos.close_reason = reason_map.get(closing.reason, "closed on broker side")
        log.info("Position ticket=%s detected closed: %s @ %s",
                 pos.ticket, pos.close_reason, pos.close_price)

    # -- broker reconciliation -------------------------------------------------

    def list_broker_positions(self) -> list[ManagedPosition]:
        """All open positions carrying this bot's magic number, rebuilt as
        ManagedPosition (used to adopt orphans after a crash)."""
        mt5 = self._mt5()
        self.client.ensure_connected()
        raw = mt5.positions_get(symbol=self.symbol)
        if raw is None:
            return []
        adopted = []
        for p in raw:
            if p.magic != self.magic:
                continue
            side = Side.LONG if p.type == mt5.ORDER_TYPE_BUY else Side.SHORT
            kind = SignalKind.SECOND if "second" in (p.comment or "") else SignalKind.FIRST
            server_naive = datetime.fromtimestamp(int(p.time), tz=UTC).replace(tzinfo=None)
            opened_at = self.client.clock.to_utc(server_naive)
            sl = p.sl or (p.price_open - side.sign * 1.0)  # degenerate fallback
            signal = EntrySignal(
                kind=kind, side=side, time=opened_at, entry_ref=p.price_open,
                stop_loss=sl, take_profit=p.tp or None, tp_rr=None,
                risk_points=abs(p.price_open - sl),
                reason="adopted from broker after restart",
            )
            adopted.append(
                ManagedPosition(
                    signal=signal, entry_price=p.price_open, volume=p.volume,
                    opened_at=opened_at, initial_stop=sl, stop_loss=sl,
                    take_profit=p.tp or None, ticket=p.ticket,
                )
            )
        return adopted
