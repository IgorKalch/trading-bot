"""Telegram notifications via Bot API (plain HTTPS, no extra framework).

Every bot action is mirrored here: signals, entries, SL moves, closes, skips,
errors, prop-guard triggers. Failures to notify NEVER break trading — they are
logged and swallowed.
"""

from __future__ import annotations

import logging
import queue
import threading

import requests

from tradingbot.config import TelegramConfig
from tradingbot.core.retry import retry

log = logging.getLogger(__name__)

API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    """Non-blocking sender: messages go through a background queue so the
    trading loop is never delayed by Telegram latency."""

    def __init__(self, cfg: TelegramConfig):
        self._cfg = cfg
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=1000)
        self._worker: threading.Thread | None = None
        if self.enabled:
            self._worker = threading.Thread(target=self._run, name="telegram", daemon=True)
            self._worker.start()

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enabled and self._cfg.bot_token.get_secret_value() and self._cfg.chat_id)

    def send(self, text: str) -> None:
        """Queue a message. Never raises, never blocks the caller."""
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            log.warning("Telegram queue full, dropping message: %s", text[:80])

    def close(self) -> None:
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout=10)

    # -- internals ---------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._post(item)
            except Exception as exc:  # noqa: BLE001 — notification loss is acceptable
                log.error("Telegram send failed permanently: %s", exc)

    @retry(attempts=4, delay=1.0, backoff=2.0, exceptions=(requests.RequestException,))
    def _post(self, text: str) -> None:
        resp = requests.post(
            API_URL.format(token=self._cfg.bot_token.get_secret_value()),
            json={
                "chat_id": self._cfg.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if resp.status_code == 429:
            retry_after = resp.json().get("parameters", {}).get("retry_after", 3)
            raise requests.RequestException(f"rate limited, retry_after={retry_after}")
        resp.raise_for_status()


class NullNotifier(TelegramNotifier):
    """Used in backtests and tests."""

    def __init__(self):  # noqa: D107 — intentionally no super().__init__
        self._messages: list[str] = []

    @property
    def enabled(self) -> bool:
        return False

    def send(self, text: str) -> None:
        self._messages.append(text)

    def close(self) -> None:
        pass
