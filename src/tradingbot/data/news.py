"""Economic calendar for the news filter.

Two sources:
  - live: ForexFactory weekly JSON feed (free, no key), cached to disk
  - backtest: local CSV (data/news/calendar.csv) maintained by the user

CSV format (UTC):
    datetime_utc,currency,impact,title
    2026-07-03 12:30,USD,high,Non-Farm Payrolls
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

from tradingbot.core.retry import retry

log = logging.getLogger(__name__)

UTC = UTC

FF_FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

_IMPACT_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class NewsEvent:
    time: datetime  # UTC
    currency: str
    impact: str  # low | medium | high
    title: str


class NewsCalendar:
    def __init__(self, events: list[NewsEvent]):
        self.events = sorted(events, key=lambda e: e.time)

    # -- constructors --------------------------------------------------------

    @classmethod
    def empty(cls) -> NewsCalendar:
        return cls([])

    @classmethod
    def from_csv(cls, path: str | Path) -> NewsCalendar:
        p = Path(path)
        if not p.exists():
            log.warning("News calendar CSV not found: %s — news filter has no data", p)
            return cls.empty()
        events: list[NewsEvent] = []
        with p.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                events.append(
                    NewsEvent(
                        time=datetime.fromisoformat(row["datetime_utc"]).replace(tzinfo=UTC),
                        currency=row["currency"].strip().upper(),
                        impact=row["impact"].strip().lower(),
                        title=row.get("title", "").strip(),
                    )
                )
        log.info("Loaded %d news events from %s", len(events), p)
        return cls(events)

    @classmethod
    def fetch_forexfactory(cls, cache_dir: str | Path = "data/news") -> NewsCalendar:
        """Fetch this week's calendar; fall back to cached copy on failure."""
        cache = Path(cache_dir) / "ff_thisweek.json"
        try:
            raw = _download_ff_feed()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 — degrade to cache, then to empty
            log.error("ForexFactory feed fetch failed: %s", exc)
            if cache.exists():
                log.warning("Using cached news feed from %s", cache)
                raw = json.loads(cache.read_text(encoding="utf-8"))
            else:
                log.warning("No cached news feed — news filter has no data this week")
                return cls.empty()
        events = []
        for item in raw:
            try:
                t = datetime.fromisoformat(item["date"])  # feed carries tz offset
                events.append(
                    NewsEvent(
                        time=t.astimezone(UTC),
                        currency=str(item.get("country", "")).upper(),
                        impact=str(item.get("impact", "")).lower(),
                        title=str(item.get("title", "")),
                    )
                )
            except (KeyError, ValueError) as exc:
                log.debug("Skipping malformed news item %s: %s", item, exc)
        log.info("News feed loaded: %d events this week", len(events))
        return cls(events)

    # -- queries --------------------------------------------------------------

    def blocking_event(
        self,
        at: datetime,
        currencies: list[str],
        min_impact: str = "high",
        before_min: int = 5,
        after_min: int = 5,
    ) -> NewsEvent | None:
        """Return the event that makes `at` fall inside a news blackout window."""
        rank = _IMPACT_RANK.get(min_impact, 2)
        for e in self.events:
            if e.currency not in currencies or _IMPACT_RANK.get(e.impact, 0) < rank:
                continue
            if e.time - timedelta(minutes=before_min) <= at <= e.time + timedelta(minutes=after_min):
                return e
        return None


@retry(attempts=3, delay=2.0, exceptions=(requests.RequestException,))
def _download_ff_feed() -> list:
    resp = requests.get(FF_FEED_URL, timeout=15, headers={"User-Agent": "orb-bot/1.0"})
    resp.raise_for_status()
    return resp.json()
