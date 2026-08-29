"""Retry with exponential backoff for flaky I/O (MT5 calls, Telegram HTTP)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

log = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def retry(
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            wait = delay
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == attempts:
                        raise
                    log.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                        fn.__qualname__, attempt, attempts, exc, wait,
                    )
                    time.sleep(wait)
                    wait *= backoff
            raise RuntimeError("unreachable")

        return wrapper

    return decorator
