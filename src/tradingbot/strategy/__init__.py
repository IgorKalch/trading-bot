"""Strategy registry: pick the model by name from config."""

from __future__ import annotations

from tradingbot.config import StrategyConfig
from tradingbot.strategy.base import Strategy
from tradingbot.strategy.orb import OrbStrategy
from tradingbot.strategy.retest import RetestStrategy
from tradingbot.strategy.sweep import SweepStrategy

_REGISTRY = {"orb": OrbStrategy, "retest": RetestStrategy, "sweep": SweepStrategy}


def build_strategy(cfg: StrategyConfig) -> Strategy:
    try:
        return _REGISTRY[cfg.name](cfg)
    except KeyError:
        raise ValueError(f"unknown strategy '{cfg.name}'; known: {sorted(_REGISTRY)}") from None


__all__ = ["OrbStrategy", "RetestStrategy", "SweepStrategy", "build_strategy"]
