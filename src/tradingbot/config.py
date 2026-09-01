"""Configuration: YAML file for behaviour, .env for secrets.

Usage:
    cfg = load_config("config/config.yaml")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr


class Mt5Config(BaseModel):
    # Secrets come from .env, not from YAML.
    terminal_path: str = ""
    login: int = 0
    password: SecretStr = SecretStr("")
    server: str = ""
    server_timezone: str = "eet_us_dst"  # see core.clock.ServerClock
    symbol: str = "GER40"
    # Max slippage for market orders, in BROKER points (symbol_info.point);
    # with point=0.1 a value of 100 allows 10 index points.
    deviation_points: int = 100


class TelegramConfig(BaseModel):
    enabled: bool = True
    bot_token: SecretStr = SecretStr("")
    chat_id: str = ""
    # Send heartbeat message this often (minutes); 0 disables.
    heartbeat_minutes: int = 60


class SessionConfig(BaseModel):
    # Anchored to the exchange timezone (Xetra = Europe/Berlin). 09:00 Berlin
    # is always 10:00 Kyiv — both follow EU DST.
    timezone: str = "Europe/Berlin"
    open: str = "09:00"  # Xetra cash open
    close: str = "17:30"  # Xetra cash close — reference price for the overnight gap
    or_minutes: int = 5
    pos1_cutoff: str = "11:00"  # 12:00 Kyiv
    pos2_cutoff: str = "17:00"  # 18:00 Kyiv
    flat_time: str = "17:20"  # force-close before the cash close


class ConfirmationConfig(BaseModel):
    require_body_close: bool = True  # candle must CLOSE beyond the OR boundary
    min_body_to_range: float = 0.5  # impulse: body >= 50% of candle range
    # "No marginal breakouts": close must clear the boundary by at least
    # max(min_break_points, min_break_or_frac * OR width). 0 = off.
    min_break_points: float = 0.0
    min_break_or_frac: float = 0.0


class EntryConfig(BaseModel):
    first_in_gap_direction_only: bool = True
    # Counter-gap (gap-fill direction) first entry allowed only when
    # |gap| <= this many points; 0 disables counter-gap entries entirely.
    counter_gap_max_points: float = 0.0
    allow_second_position: bool = True
    close_open_position_on_reversal: bool = True
    max_positions_per_day: int = 2
    # Skip entry if confirmation close is further than this multiple of the
    # OR width beyond the broken boundary ("nowhere to put the stop"); 0 = off.
    max_entry_distance_or_mult: float = 0.0


class FiltersConfig(BaseModel):
    # 0 disables a numeric filter (all thresholds in DAX index points).
    max_gap_points: float = 0.0  # skip the whole day if |gap| larger than this
    max_or_width_points: float = 0.0
    min_or_width_points: float = 0.0
    max_or_width_atr_mult: float = 0.0  # OR width <= mult * ATR(atr_period, D1)
    # Opening-range width as a PERCENTAGE of price. Volatility-neutral, unlike
    # the absolute point thresholds above: on an index that doubles, a fixed
    # point threshold silently becomes a date filter (Додаток И).
    min_or_width_pct: float = 0.0
    max_or_width_pct: float = 0.0
    atr_period: int = 14
    # Relative volume of the OR window vs the same window on previous days.
    min_or_rvol: float = 0.0  # e.g. 1.0 = require at least average volume
    rvol_lookback_days: int = 14
    # Only trade breakouts aligned with the OR candle's own close direction.
    or_direction_filter: bool = False
    # Only long above session VWAP / short below.
    vwap_filter: bool = False
    # §7.9 Trend filter: trade only on the side of a long moving average of the
    # traded timeframe, measured before the session opens. 0 = off (Hougaard: 89).
    trend_ma_period: int = 0
    trend_ma_require_slope: bool = False  # also require the MA itself to point that way
    # §7.10 The breakout candle must carry at least this multiple of the average
    # volume of the preceding bars ("high volume extension"). 0 = off. The window
    # is rolling and short on purpose: measured against the whole session the
    # opening spike dominates and every later bar scores below 1.0.
    min_break_bar_rvol: float = 0.0
    break_rvol_lookback_bars: int = 12
    # §7.11 Imbalance (smart-money FVG): demand a three-bar gap in the trade
    # direction within the last N bars. False = off.
    require_fvg: bool = False
    fvg_max_age_bars: int = 12
    fvg_min_size_points: float = 0.0
    # Weekdays to skip entirely: 0=Mon .. 4=Fri.
    skip_weekdays: list[int] = Field(default_factory=list)
    # Live entry sanity: skip entry if current spread wider than this.
    max_spread_points: float = 6.0
    news_filter_enabled: bool = True
    news_min_impact: Literal["high", "medium"] = "high"
    news_buffer_before_min: int = 5
    news_buffer_after_min: int = 5
    news_currencies: list[str] = Field(default_factory=lambda: ["EUR", "USD"])


class StopsConfig(BaseModel):
    # Stop always goes behind the FULL opening range (STRATEGY.md §9).
    sl_buffer_points: float = 2.0  # extra distance beyond the OR boundary
    max_sl_points: float = 0.0  # skip trade if stop distance larger; 0 = off


class TargetsConfig(BaseModel):
    tp_mode: Literal["fixed_rr", "trailing"] = "fixed_rr"
    fixed_rr: float = Field(default=1.0, gt=0, le=2.0)  # >2R forbidden by STRATEGY.md §10
    trail_start_r: float = Field(default=0.5, ge=0)  # start trailing after this profit in R
    trail_step_r: float = Field(default=0.5, gt=0)  # move stop every additional step in R
    breakeven_buffer_points: float = 0.0  # extra points when moving stop to BE


class RetestConfig(BaseModel):
    """Retest-and-absorb model (Додаток В). Used only when strategy.name = retest."""

    require_body_close: bool = True  # the break bar must close with a directional body
    min_break_or_frac: float = 0.0  # decisive break, as a fraction of OR width
    max_pullback_bars: int = 12  # give up if the retest is not absorbed within this
    stop_buffer_points: float = 2.0  # beyond the pullback extreme
    min_stop_points: float = 0.0  # skip if 1R is so tight the spread dominates
    max_stop_points: float = 0.0
    max_positions_per_day: int = 2


class SweepConfig(BaseModel):
    """Overnight-range liquidity sweep (Додаток Ж). strategy.name = sweep."""

    # Which level the sweep is measured against (Додаток З):
    #   overnight  - range built before the cash open (default)
    #   prev_day   - previous session high/low  (PDH/PDL)
    #   prev_week  - highest high / lowest low of the last 5 sessions (PWH/PWL)
    reference: Literal["overnight", "prev_day", "prev_week"] = "overnight"
    min_pre_bars: int = 30  # skip the day if the overnight session is too thin
    min_sweep_points: float = 0.0  # how far past the edge counts as a sweep
    min_sweep_range_frac: float = 0.0  # ... or as a fraction of the range width
    max_reclaim_bars: int = 6  # give up if price does not come back inside
    require_body_close: bool = True  # the reclaim bar must close directionally
    require_fvg: bool = False  # demand an imbalance in the trade direction
    fvg_max_age_bars: int = 12
    fvg_min_size_points: float = 0.0
    stop_buffer_points: float = 2.0  # beyond the sweep extreme
    min_stop_points: float = 0.0
    max_stop_points: float = 0.0
    max_positions_per_day: int = 2


class StrategyConfig(BaseModel):
    name: str = "orb"
    timeframe: str = "M5"
    confirmation: ConfirmationConfig = ConfirmationConfig()
    entry: EntryConfig = EntryConfig()
    filters: FiltersConfig = FiltersConfig()
    stops: StopsConfig = StopsConfig()
    targets: TargetsConfig = TargetsConfig()
    retest: RetestConfig = RetestConfig()
    sweep: SweepConfig = SweepConfig()


class RiskConfig(BaseModel):
    risk_per_trade_pct: float = 1.0  # % of account balance risked per trade
    min_volume_fallback: bool = False  # if computed lot < min lot: trade min lot?
    max_volume_lots: float = 20.0  # Funding Pips: hard 20-lot-per-click cap


class PropConfig(BaseModel):
    """Funding Pips (or any prop firm) hard guards.

    Buffers make the bot stop BEFORE the firm's real limit is touched.
    """

    enabled: bool = True
    daily_loss_limit_pct: float = 5.0
    daily_loss_buffer_pct: float = 1.5  # bot stops at limit - buffer (i.e. -3.5%)
    max_drawdown_pct: float = 10.0
    max_drawdown_buffer_pct: float = 2.0
    initial_balance: float = 0.0  # account starting balance; 0 = read at first start
    restrict_news_trading: bool = True  # funded accounts: no entries around news
    news_window_before_min: int = 5
    news_window_after_min: int = 5
    # Close open positions shortly BEFORE a red-folder event window so no
    # open/close ever happens inside it (Master account profit-deduction rule).
    flatten_before_news: bool = True
    # If the news calendar could not be loaded: refuse entries (fail closed)?
    require_news_data: bool = False
    forbid_weekend_holding: bool = True


class BacktestConfig(BaseModel):
    data_dir: str = "data/cache"
    reports_dir: str = "reports"
    news_csv: str = "data/news/calendar.csv"  # optional; empty calendar if missing
    initial_balance: float = 100_000.0
    # Execution model (conservative): every fill pays half-spread + slippage
    # against you; commission charged per lot round-turn at close.
    spread_points: float = 2.0
    slippage_points: float = 1.0
    commission_per_lot: float = 0.0  # Funding Pips: indices are commission-free
    # If SL and TP fall inside the same bar, assume SL hit first.
    pessimistic_same_bar: bool = True
    # Simulated GER40 CFD contract: 1.0 lot = point_value account currency per point.
    point_value_per_lot: float = 1.0
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 20.0


class BotConfig(BaseModel):
    mode: Literal["live", "signal"] = "signal"
    magic: int = 20260706
    poll_seconds: float = 2.0  # position-management poll interval
    state_dir: str = "state"
    log_dir: str = "logs"
    log_level: str = "INFO"


class AppConfig(BaseModel):
    bot: BotConfig = BotConfig()
    mt5: Mt5Config = Mt5Config()
    telegram: TelegramConfig = TelegramConfig()
    session: SessionConfig = SessionConfig()
    strategy: StrategyConfig = StrategyConfig()
    risk: RiskConfig = RiskConfig()
    prop: PropConfig = PropConfig()
    backtest: BacktestConfig = BacktestConfig()


def load_config(path: str | Path = "config/config.yaml", env_file: str | Path = ".env") -> AppConfig:
    """Load YAML config and overlay secrets from .env / environment."""
    load_dotenv(env_file)
    raw: dict = {}
    p = Path(path)
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    cfg = AppConfig.model_validate(raw)

    # Secrets and machine-specific values always come from the environment.
    cfg.mt5.terminal_path = os.getenv("MT5_TERMINAL_PATH", cfg.mt5.terminal_path)
    if os.getenv("MT5_LOGIN"):
        cfg.mt5.login = int(os.environ["MT5_LOGIN"])
    if os.getenv("MT5_PASSWORD"):
        cfg.mt5.password = SecretStr(os.environ["MT5_PASSWORD"])
    cfg.mt5.server = os.getenv("MT5_SERVER", cfg.mt5.server)
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        cfg.telegram.bot_token = SecretStr(os.environ["TELEGRAM_BOT_TOKEN"])
    cfg.telegram.chat_id = os.getenv("TELEGRAM_CHAT_ID", cfg.telegram.chat_id)
    return cfg
