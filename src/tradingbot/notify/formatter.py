"""Telegram message formatting (Ukrainian, HTML parse mode)."""

from __future__ import annotations

from tradingbot.core.models import EntrySignal, ManagedPosition, SkipEvent

SIDE_EMOJI = {"long": "🟢 LONG", "short": "🔴 SHORT"}
KIND_UA = {"first": "перша (за гепом)", "second": "друга (реверс)"}


def fmt_signal(sig: EntrySignal, symbol: str, mode: str) -> str:
    head = "📣 <b>СИГНАЛ</b>" if mode == "signal" else "🎯 <b>ВХІД</b>"
    tp = f"{sig.take_profit:.1f} ({sig.tp_rr}R)" if sig.take_profit else "trailing"
    return (
        f"{head} {SIDE_EMOJI[sig.side.value]} <b>{symbol}</b>\n"
        f"Позиція: {KIND_UA[sig.kind.value]}\n"
        f"Ціна (закриття підтвердження): {sig.entry_ref:.1f}\n"
        f"SL: {sig.stop_loss:.1f} ({sig.risk_points:.1f} п.)  TP: {tp}\n"
        f"<i>{sig.reason}</i>"
    )


def fmt_opened(pos: ManagedPosition, symbol: str, risk_money: float, risk_pct: float) -> str:
    tp = f"{pos.take_profit:.1f}" if pos.take_profit else "trailing"
    return (
        f"✅ <b>ПОЗИЦІЮ ВІДКРИТО</b> {SIDE_EMOJI[pos.side.value]} <b>{symbol}</b>\n"
        f"Обсяг: {pos.volume:.2f} лот  Вхід: {pos.entry_price:.1f}\n"
        f"SL: {pos.stop_loss:.1f}  TP: {tp}\n"
        f"Ризик: {risk_money:,.2f} ({risk_pct:.2f}%)  Тікет: {pos.ticket}"
    )


def fmt_sl_moved(pos: ManagedPosition, symbol: str, reason: str) -> str:
    return (
        f"🔧 <b>SL ПЕРЕСУНУТО</b> {symbol} #{pos.ticket}\n"
        f"Новий SL: {pos.stop_loss:.1f}\n<i>{reason}</i>"
    )


def fmt_closed(pos: ManagedPosition, symbol: str) -> str:
    r = pos.result_r
    r_str = f"{r:+.2f}R" if r is not None else "?"
    exit_str = f"{pos.close_price:.1f}" if pos.close_price is not None else "?"
    emoji = "🏆" if (r or 0) > 0 else "🛑"
    return (
        f"{emoji} <b>ПОЗИЦІЮ ЗАКРИТО</b> {symbol} #{pos.ticket}\n"
        f"{SIDE_EMOJI[pos.side.value]}  Вхід: {pos.entry_price:.1f} → Вихід: "
        f"{exit_str}\nРезультат: <b>{r_str}</b>\n<i>{pos.close_reason}</i>"
    )


def fmt_skip(skip: SkipEvent, symbol: str) -> str:
    return f"⏭ <b>ПРОПУСК</b> {symbol} [{skip.rule}]\n<i>{skip.detail}</i>"


def fmt_guard_block(rule: str, detail: str) -> str:
    return f"🚫 <b>PROP GUARD</b> заблокував вхід [{rule}]\n<i>{detail}</i>"


def fmt_error(what: str, exc: Exception | str) -> str:
    return f"❗️ <b>ПОМИЛКА</b> {what}\n<code>{exc}</code>"
