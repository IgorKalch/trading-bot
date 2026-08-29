"""Thin, robust wrapper around the MetaTrader5 Python package.

Responsibilities:
  - connection lifecycle (initialize / login / reconnect with retries)
  - fetching bars & ticks, converted to tz-aware UTC at this boundary
  - symbol specification needed for position sizing
  - account state for risk guards

Order placement lives in execution.mt5_executor, not here.

The `MetaTrader5` package only works on Windows with a running terminal;
it is imported lazily so the rest of the codebase (backtests, tests) works
anywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from tradingbot.config import Mt5Config
from tradingbot.core.clock import ServerClock
from tradingbot.core.models import Bar, Tick
from tradingbot.core.retry import retry

log = logging.getLogger(__name__)

TIMEFRAMES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


class Mt5Error(RuntimeError):
    pass


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    point: float  # price change of one "point" (e.g. 0.01)
    tick_size: float  # minimal price change
    tick_value: float  # account-currency value of one tick per 1.0 lot
    volume_min: float
    volume_max: float
    volume_step: float
    digits: int
    contract_size: float

    @property
    def value_per_point(self) -> float:
        """Account-currency value of a 1-point move per 1.0 lot."""
        if self.tick_size <= 0:
            raise Mt5Error(f"{self.name}: invalid tick_size {self.tick_size}")
        return self.tick_value * (self.point / self.tick_size)


@dataclass(frozen=True)
class AccountState:
    balance: float
    equity: float
    margin_free: float
    currency: str
    login: int


def _mt5():
    try:
        import MetaTrader5 as mt5  # noqa: PLC0415 — lazy, Windows-only
    except ImportError as exc:  # pragma: no cover
        raise Mt5Error(
            "MetaTrader5 package is not installed. Run: pip install MetaTrader5 (Windows only)"
        ) from exc
    return mt5


class Mt5Client:
    def __init__(self, cfg: Mt5Config):
        self.cfg = cfg
        self.clock = ServerClock(cfg.server_timezone)
        self._connected = False

    # -- connection ---------------------------------------------------------

    @retry(attempts=5, delay=2.0, backoff=2.0, exceptions=(Mt5Error,))
    def connect(self) -> None:
        mt5 = _mt5()
        kwargs: dict = {}
        if self.cfg.terminal_path:
            kwargs["path"] = self.cfg.terminal_path
        if self.cfg.login:
            kwargs.update(
                login=self.cfg.login,
                password=self.cfg.password.get_secret_value(),
                server=self.cfg.server,
            )
        if not mt5.initialize(**kwargs):
            raise Mt5Error(f"MT5 initialize failed: {mt5.last_error()}")
        info = mt5.account_info()
        if info is None:
            raise Mt5Error(f"MT5 account_info failed: {mt5.last_error()}")
        self._connected = True
        log.info("MT5 connected: login=%s server=%s balance=%.2f", info.login, info.server, info.balance)

    def ensure_connected(self) -> None:
        mt5 = _mt5()
        term = mt5.terminal_info()
        # terminal_info() stays non-None while the terminal merely lost its
        # broker link — the `connected` flag reflects actual connectivity.
        if not self._connected or term is None or not getattr(term, "connected", True):
            log.warning("MT5 connection lost — reconnecting")
            self._connected = False
            self.connect()

    def shutdown(self) -> None:
        if self._connected:
            _mt5().shutdown()
            self._connected = False

    # -- market data ---------------------------------------------------------

    def select_symbol(self, symbol: str) -> None:
        mt5 = _mt5()
        if not mt5.symbol_select(symbol, True):
            raise Mt5Error(f"symbol_select({symbol}) failed: {mt5.last_error()}")

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        mt5 = _mt5()
        self.select_symbol(symbol)
        info = mt5.symbol_info(symbol)
        if info is None:
            raise Mt5Error(f"symbol_info({symbol}) failed: {mt5.last_error()}")
        return SymbolSpec(
            name=symbol,
            point=info.point,
            tick_size=info.trade_tick_size,
            tick_value=info.trade_tick_value,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            digits=info.digits,
            contract_size=info.trade_contract_size,
        )

    def _tf_const(self, timeframe: str) -> int:
        mt5 = _mt5()
        try:
            return getattr(mt5, f"TIMEFRAME_{timeframe}")
        except AttributeError as exc:
            raise Mt5Error(f"Unknown timeframe {timeframe}") from exc

    @retry(attempts=3, delay=1.0, exceptions=(Mt5Error,))
    def get_recent_bars(self, symbol: str, timeframe: str, count: int) -> list[Bar]:
        """Last `count` bars including the (possibly unfinished) current bar."""
        mt5 = _mt5()
        rates = mt5.copy_rates_from_pos(symbol, self._tf_const(timeframe), 0, count)
        if rates is None:
            raise Mt5Error(f"copy_rates_from_pos({symbol}) returned None: {mt5.last_error()}")
        return [self._to_bar(r) for r in rates]

    @retry(attempts=3, delay=1.0, exceptions=(Mt5Error,))
    def get_bars_range(
        self, symbol: str, timeframe: str, start_utc: datetime, end_utc: datetime
    ) -> list[Bar]:
        """Bars in [start_utc, end_utc]. MT5 expects server-time datetimes."""
        mt5 = _mt5()
        # MT5 interprets the passed datetime's clock value as server time; pass
        # naive server-time datetimes widened by a day on both sides, then trim.
        srv_from = self.clock.from_utc(start_utc) - timedelta(days=1)
        srv_to = self.clock.from_utc(end_utc) + timedelta(days=1)
        rates = mt5.copy_rates_range(
            symbol, self._tf_const(timeframe),
            srv_from.replace(tzinfo=UTC), srv_to.replace(tzinfo=UTC),
        )
        if rates is None:
            raise Mt5Error(f"copy_rates_range({symbol}) returned None: {mt5.last_error()}")
        bars = [self._to_bar(r) for r in rates]
        return [b for b in bars if start_utc <= b.time <= end_utc]

    def _to_bar(self, rate) -> Bar:
        server_naive = datetime.fromtimestamp(int(rate["time"]), tz=UTC).replace(tzinfo=None)
        return Bar(
            time=self.clock.to_utc(server_naive),
            open=float(rate["open"]),
            high=float(rate["high"]),
            low=float(rate["low"]),
            close=float(rate["close"]),
            tick_volume=int(rate["tick_volume"]),
            spread_points=int(rate["spread"]),
        )

    @retry(attempts=3, delay=0.5, exceptions=(Mt5Error,))
    def current_tick(self, symbol: str) -> Tick:
        mt5 = _mt5()
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise Mt5Error(f"symbol_info_tick({symbol}) returned None: {mt5.last_error()}")
        server_naive = datetime.fromtimestamp(int(t.time), tz=UTC).replace(tzinfo=None)
        return Tick(time=self.clock.to_utc(server_naive), bid=t.bid, ask=t.ask)

    # -- account -------------------------------------------------------------

    def account(self) -> AccountState:
        mt5 = _mt5()
        info = mt5.account_info()
        if info is None:
            raise Mt5Error(f"account_info failed: {mt5.last_error()}")
        return AccountState(
            balance=info.balance,
            equity=info.equity,
            margin_free=info.margin_free,
            currency=info.currency,
            login=info.login,
        )
