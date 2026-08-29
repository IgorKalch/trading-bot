"""Logging: console + rotating file. Telegram gets its own channel (notify/)."""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(console)

    file_handler = TimedRotatingFileHandler(
        Path(log_dir) / "bot.log", when="midnight", backupCount=30, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(file_handler)

    # Third-party noise down.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
