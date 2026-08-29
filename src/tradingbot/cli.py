"""Command-line interface.

    tradingbot check                         — verify MT5/Telegram/config wiring
    tradingbot download --months 12          — cache history for backtests
    tradingbot backtest --months 12          — run the ORB backtest
    tradingbot run                           — live/signal trading (per config)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta

from tradingbot.config import AppConfig, load_config
from tradingbot.logging_setup import setup_logging

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradingbot", description="ORB trading bot for MT5")
    p.add_argument("--config", default="config/config.yaml", help="path to config YAML")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="verify environment: MT5 connection, symbol, Telegram")

    d = sub.add_parser("download", help="download & cache historical bars from MT5")
    d.add_argument("--months", type=int, default=12, help="how many months back")
    d.add_argument("--symbol", default=None, help="override symbol from config")
    d.add_argument("--timeframe", default=None, help="override timeframe (default M5)")

    b = sub.add_parser("backtest", help="run backtest on cached data")
    b.add_argument("--months", type=int, default=12, help="period length back from --end")
    b.add_argument("--start", default=None, help="YYYY-MM-DD (overrides --months)")
    b.add_argument("--end", default=None, help="YYYY-MM-DD (default: now)")
    b.add_argument("--tag", default=None, help="report file tag")

    sub.add_parser("run", help="start live/signal trading (mode from config bot.mode)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg.bot.log_dir, cfg.bot.log_level)
    try:
        if args.command == "check":
            return _cmd_check(cfg)
        if args.command == "download":
            return _cmd_download(cfg, args)
        if args.command == "backtest":
            return _cmd_backtest(cfg, args)
        if args.command == "run":
            return _cmd_run(cfg)
    except KeyboardInterrupt:
        return 130
    return 2


# ---------------------------------------------------------------- commands


def _cmd_check(cfg: AppConfig) -> int:
    from tradingbot.data.mt5_client import Mt5Client
    from tradingbot.notify.telegram import TelegramNotifier

    ok = True
    print(f"Config OK. Mode: {cfg.bot.mode}, symbol: {cfg.mt5.symbol}")

    try:
        client = Mt5Client(cfg.mt5)
        client.connect()
        account = client.account()
        print(f"MT5 OK: login={account.login}, balance={account.balance:.2f} {account.currency}")
        spec = client.symbol_spec(cfg.mt5.symbol)
        print(
            f"Symbol OK: {spec.name} point={spec.point} tick_value={spec.tick_value} "
            f"vol {spec.volume_min}..{spec.volume_max} step {spec.volume_step}"
        )
        tick = client.current_tick(cfg.mt5.symbol)
        offset = datetime.now(tz=UTC) - tick.time
        print(f"Tick OK: bid={tick.bid} ask={tick.ask} time={tick.time:%H:%M:%S} UTC")
        if abs(offset.total_seconds()) > 300:
            print(
                f"!! Tick time differs from UTC-now by {offset} — check mt5.server_timezone "
                f"(current: {cfg.mt5.server_timezone})"
            )
            ok = False
        client.shutdown()
    except Exception as exc:  # noqa: BLE001
        print(f"MT5 FAILED: {exc}")
        ok = False

    notifier = TelegramNotifier(cfg.telegram)
    if notifier.enabled:
        notifier.send("✅ tradingbot check: Telegram працює")
        notifier.close()
        print("Telegram OK: test message sent")
    else:
        print("Telegram DISABLED (no token/chat_id or enabled=false)")
    return 0 if ok else 1


def _cmd_download(cfg: AppConfig, args) -> int:
    from tradingbot.data import history
    from tradingbot.data.mt5_client import Mt5Client

    symbol = args.symbol or cfg.mt5.symbol
    timeframe = args.timeframe or cfg.strategy.timeframe
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=31 * args.months)

    client = Mt5Client(cfg.mt5)
    client.connect()
    try:
        path = history.download(client, cfg.backtest.data_dir, symbol, timeframe, start, end)
        bars = history.load_bars(cfg.backtest.data_dir, symbol, timeframe)
        print(f"Saved to {path}: {len(bars)} bars, {bars[0].time} .. {bars[-1].time}")
    finally:
        client.shutdown()
    return 0


def _cmd_backtest(cfg: AppConfig, args) -> int:
    from tradingbot.backtest.engine import BacktestEngine
    from tradingbot.backtest.report import render_report, save_report
    from tradingbot.data import history
    from tradingbot.data.news import NewsCalendar
    from tradingbot.strategy import build_strategy

    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC) if args.end else datetime.now(tz=UTC)
    if args.start:
        start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    else:
        start = end - timedelta(days=31 * args.months)

    bars = history.load_bars(cfg.backtest.data_dir, cfg.mt5.symbol, cfg.strategy.timeframe, start, end)
    if not bars:
        print("No bars in the requested range — run `tradingbot download` first.")
        return 1
    print(f"Backtesting {cfg.mt5.symbol} on {len(bars)} bars: {bars[0].time} .. {bars[-1].time}")

    news = NewsCalendar.from_csv(cfg.backtest.news_csv)
    engine = BacktestEngine(cfg, build_strategy(cfg.strategy), news)
    result = engine.run(bars)

    print(render_report(result))
    tag = args.tag or f"{start:%Y%m%d}_{end:%Y%m%d}"
    txt, csv_path = save_report(result, cfg.backtest.reports_dir, tag)
    print(f"Saved: {txt}\n       {csv_path}")
    return 0


def _cmd_run(cfg: AppConfig) -> int:
    from tradingbot.data.mt5_client import Mt5Client
    from tradingbot.execution.mt5_executor import Mt5Executor
    from tradingbot.execution.paper import PaperExecutor
    from tradingbot.live.runner import LiveRunner
    from tradingbot.notify.telegram import TelegramNotifier
    from tradingbot.risk.prop_guard import PropGuard
    from tradingbot.strategy import build_strategy

    client = Mt5Client(cfg.mt5)
    if cfg.bot.mode == "live":
        executor = Mt5Executor(client, cfg.mt5.symbol, cfg.bot.magic, cfg.mt5.deviation_points)
        log.info("LIVE mode: real orders WILL be sent to MT5")
    else:
        executor = PaperExecutor(client, cfg.mt5.symbol)
        log.info("SIGNAL mode: no real orders, Telegram signals only")

    runner = LiveRunner(
        cfg=cfg,
        client=client,
        executor=executor,
        strategy=build_strategy(cfg.strategy),
        notifier=TelegramNotifier(cfg.telegram),
        guard=PropGuard(cfg.prop, cfg.bot.state_dir),
    )
    runner.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
