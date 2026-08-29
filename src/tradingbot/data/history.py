"""Historical data pipeline for backtests.

Downloads M5 bars from MT5 (copy_rates_range) and caches them as Parquet.
Backtests read only the cache, so they run without a terminal (and in CI).

NOTE: how much history MT5 returns depends on the broker AND the terminal
setting Tools > Options > Charts > "Max bars in chart" (set to Unlimited).
Verify coverage with `tradingbot download ... --check`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from tradingbot.core.models import Bar

log = logging.getLogger(__name__)

UTC = UTC

COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread_points"]


def cache_path(data_dir: str | Path, symbol: str, timeframe: str) -> Path:
    return Path(data_dir) / f"{symbol}_{timeframe}.parquet"


def bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "time": [b.time for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "tick_volume": [b.tick_volume for b in bars],
            "spread_points": [b.spread_points for b in bars],
        }
    )
    return df


def frame_to_bars(df: pd.DataFrame) -> list[Bar]:
    return [
        Bar(
            time=t.to_pydatetime(),
            open=float(o), high=float(h), low=float(lo), close=float(c),
            tick_volume=int(v), spread_points=int(s),
        )
        for t, o, h, lo, c, v, s in zip(
            df["time"], df["open"], df["high"], df["low"], df["close"],
            df["tick_volume"], df["spread_points"], strict=True,
        )
    ]


def save_bars(bars: list[Bar], data_dir: str | Path, symbol: str, timeframe: str) -> Path:
    """Merge new bars into the cache (dedup by time), keep sorted."""
    path = cache_path(data_dir, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = bars_to_frame(bars)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset="time", keep="last").sort_values("time").reset_index(drop=True)
    df.to_parquet(path, index=False)
    log.info("Cache %s: %d bars (%s .. %s)", path, len(df), df["time"].iloc[0], df["time"].iloc[-1])
    return path


def load_bars(
    data_dir: str | Path,
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[Bar]:
    path = cache_path(data_dir, symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached data at {path}. Run: tradingbot download --symbol {symbol} first."
        )
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if start is not None:
        df = df[df["time"] >= start]
    if end is not None:
        df = df[df["time"] <= end]
    return frame_to_bars(df)


def download(
    client,  # Mt5Client; untyped to keep this module importable without MT5
    data_dir: str | Path,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    chunk_days: int = 30,
) -> Path:
    """Pull bars from MT5 in chunks and merge into the cache."""
    from datetime import timedelta

    client.ensure_connected()
    client.select_symbol(symbol)
    all_bars: list[Bar] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        bars = client.get_bars_range(symbol, timeframe, cursor, chunk_end)
        all_bars.extend(bars)
        log.info("Downloaded %s %s..%s: %d bars", symbol, cursor.date(), chunk_end.date(), len(bars))
        cursor = chunk_end
    if not all_bars:
        raise RuntimeError(
            f"MT5 returned no {symbol} bars for {start}..{end}. Check symbol name, "
            "broker history depth and 'Max bars in chart' terminal setting."
        )
    return save_bars(all_bars, data_dir, symbol, timeframe)
