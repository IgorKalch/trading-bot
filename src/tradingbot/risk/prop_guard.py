"""Funding Pips (prop firm) hard limits enforcement.

The guard is deliberately MORE conservative than the firm's real limits
(configurable buffers): the bot must stop itself before the account is at
risk of violation, including floating (equity) losses.

Checked before every entry and continuously while positions are open:
  - daily loss limit (anchored to day-start balance/equity, incl. buffer)
  - overall max drawdown (anchored to initial balance, incl. buffer)
  - planned trade risk must fit inside the remaining daily headroom
  - news blackout window for entries (funded accounts)
  - weekend holding prohibition (flat before Friday close is enforced by
    flat_time; this guard refuses late-Friday entries as a second belt)

Day anchors are persisted to disk so a bot restart cannot reset them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from tradingbot.config import PropConfig
from tradingbot.core.fsutil import atomic_write_text
from tradingbot.data.mt5_client import AccountState
from tradingbot.data.news import NewsCalendar, NewsEvent

log = logging.getLogger(__name__)


class GuardStateError(RuntimeError):
    """The persisted guard state is unreadable — trading must not proceed
    on silently re-created (possibly looser) anchors."""


@dataclass(frozen=True)
class GuardVerdict:
    allowed: bool
    rule: str = ""
    detail: str = ""


class PropGuard:
    def __init__(self, cfg: PropConfig, state_dir: str | Path = "state"):
        self.cfg = cfg
        self._state_file = Path(state_dir) / "prop_guard.json"
        self._state: dict = self._load_state()

    # -- persistence ---------------------------------------------------------

    def _load_state(self) -> dict:
        if self._state_file.exists():
            try:
                return json.loads(self._state_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                # Fail CLOSED: a corrupt file must not silently re-anchor the
                # floors at the current (possibly drawn-down) equity.
                if self.cfg.enabled:
                    raise GuardStateError(
                        f"prop_guard state {self._state_file} is unreadable ({exc}). "
                        "Verify the account anchors, then delete or fix the file to proceed."
                    ) from exc
                log.error("prop_guard state unreadable (%s) — guard disabled, starting fresh", exc)
        return {}

    def _save_state(self) -> None:
        atomic_write_text(self._state_file, json.dumps(self._state, indent=2))

    # -- anchors ---------------------------------------------------------------

    def on_day_start(self, day: date, account: AccountState) -> None:
        """Anchor the daily loss limit. Idempotent per calendar day."""
        if self._state.get("day") == day.isoformat():
            return
        if not self._state.get("initial_balance"):
            initial = self.cfg.initial_balance or account.balance
            self._state["initial_balance"] = initial
            log.info("PropGuard: initial balance anchored at %.2f", initial)
        # Funding Pips daily drawdown anchors to the day-start value; using
        # max(balance, equity) is the conservative choice.
        self._state["day"] = day.isoformat()
        self._state["day_anchor"] = max(account.balance, account.equity)
        self._save_state()
        log.info("PropGuard: day %s anchored at %.2f", day, self._state["day_anchor"])

    @property
    def initial_balance(self) -> float:
        return float(self._state.get("initial_balance", 0.0))

    @property
    def day_anchor(self) -> float:
        return float(self._state.get("day_anchor", 0.0))

    # -- limits ----------------------------------------------------------------

    def daily_loss_floor(self) -> float:
        """Equity level at which the FIRM's daily limit is breached."""
        return self.day_anchor * (1 - self.cfg.daily_loss_limit_pct / 100.0)

    def daily_soft_floor(self) -> float:
        """Equity level at which the BOT stops trading (buffered)."""
        pct = self.cfg.daily_loss_limit_pct - self.cfg.daily_loss_buffer_pct
        return self.day_anchor * (1 - pct / 100.0)

    def overall_floor(self) -> float:
        return self.initial_balance * (1 - self.cfg.max_drawdown_pct / 100.0)

    def overall_soft_floor(self) -> float:
        pct = self.cfg.max_drawdown_pct - self.cfg.max_drawdown_buffer_pct
        return self.initial_balance * (1 - pct / 100.0)

    # -- checks ------------------------------------------------------------------

    def can_open(
        self,
        account: AccountState,
        planned_risk_money: float,
        now: datetime,
        news: NewsCalendar,
        news_currencies: list[str],
    ) -> GuardVerdict:
        if not self.cfg.enabled:
            return GuardVerdict(True)

        if self.day_anchor <= 0 or self.initial_balance <= 0:
            return GuardVerdict(False, "not_anchored", "PropGuard has no day anchor — call on_day_start")

        # Worst case after this trade: current equity minus its full risk.
        worst_equity = account.equity - planned_risk_money

        if worst_equity <= self.daily_soft_floor():
            return GuardVerdict(
                False, "daily_loss",
                f"worst-case equity {worst_equity:.2f} would breach buffered daily floor "
                f"{self.daily_soft_floor():.2f} (firm floor {self.daily_loss_floor():.2f})",
            )
        if worst_equity <= self.overall_soft_floor():
            return GuardVerdict(
                False, "max_drawdown",
                f"worst-case equity {worst_equity:.2f} would breach buffered overall floor "
                f"{self.overall_soft_floor():.2f} (firm floor {self.overall_floor():.2f})",
            )
        if self.cfg.restrict_news_trading:
            event = news.blocking_event(
                now,
                currencies=news_currencies,
                min_impact="high",
                before_min=self.cfg.news_window_before_min,
                after_min=self.cfg.news_window_after_min,
            )
            if event is not None:
                return GuardVerdict(
                    False, "news_window",
                    f"high-impact news blackout: {event.currency} {event.title} at "
                    f"{event.time:%H:%M UTC} (±{self.cfg.news_window_before_min}m)",
                )
        if self.cfg.forbid_weekend_holding and now.weekday() == 4 and now.hour >= 19:
            # Friday evening UTC: refuse fresh entries close to market close.
            return GuardVerdict(False, "weekend", "no new entries late on Friday (weekend holding ban)")

        return GuardVerdict(True)

    def emergency_close_needed(self, account: AccountState) -> GuardVerdict:
        """While positions are open: force-flat if equity nears a hard floor."""
        if not self.cfg.enabled or self.day_anchor <= 0:
            return GuardVerdict(False)
        if account.equity <= self.daily_soft_floor():
            return GuardVerdict(
                True, "daily_loss",
                f"equity {account.equity:.2f} at buffered daily floor {self.daily_soft_floor():.2f}",
            )
        if account.equity <= self.overall_soft_floor():
            return GuardVerdict(
                True, "max_drawdown",
                f"equity {account.equity:.2f} at buffered overall floor {self.overall_soft_floor():.2f}",
            )
        return GuardVerdict(False)

    def news_event_ahead(self, now: datetime, news: NewsCalendar, currencies: list[str]) -> NewsEvent | None:
        """Used to decide whether to flatten positions before news (optional)."""
        if not (self.cfg.enabled and self.cfg.restrict_news_trading):
            return None
        return news.blocking_event(
            now, currencies=currencies, min_impact="high",
            before_min=self.cfg.news_window_before_min, after_min=self.cfg.news_window_after_min,
        )
