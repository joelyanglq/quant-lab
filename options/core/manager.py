"""StrategyManager — registry, lifecycle, status."""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Type

from options.core.base import Strategy, StrategyStatus

log = logging.getLogger(__name__)

# ── strategy registry ────────────────────────────────────────────
_REGISTRY: Dict[str, Type[Strategy]] = {}


def register(strategy_cls: Type[Strategy]):
    """Register a strategy class by its strategy_name attribute."""
    name = getattr(strategy_cls, "strategy_name", None)
    if name is None:
        raise ValueError(f"{strategy_cls} must have a strategy_name attribute")
    _REGISTRY[name] = strategy_cls
    return strategy_cls


def get_registry() -> Dict[str, Type[Strategy]]:
    return dict(_REGISTRY)


class StrategyManager:
    """Manages lifecycle of multiple Strategy instances.

    Usage:
        mgr = StrategyManager()
        mgr.start("monthly_ic")
        mgr.start("dte0_ic")
        mgr.status()
        mgr.stop_all()
    """

    def __init__(self):
        self._running: Dict[str, Strategy] = {}
        self._lock = threading.Lock()

    def available(self) -> List[str]:
        return sorted(_REGISTRY.keys())

    def start(self, name: str, **overrides) -> bool:
        with self._lock:
            if name in self._running and self._running[name].is_alive():
                log.warning("[%s] already running", name)
                return False

            cls = _REGISTRY.get(name)
            if cls is None:
                log.error("Unknown strategy: %s. Available: %s",
                          name, list(_REGISTRY.keys()))
                return False

            strategy = cls(**overrides)
            strategy._thread = threading.Thread(
                target=strategy._run,
                name=f"strategy-{name}",
                daemon=True,
            )
            strategy._thread.start()
            self._running[name] = strategy
            log.info("Started strategy: %s", name)
            return True

    def stop(self, name: str, timeout: float = 30) -> bool:
        with self._lock:
            strategy = self._running.get(name)
        if strategy is None:
            log.warning("[%s] not running", name)
            return False

        strategy.request_stop()
        strategy._thread.join(timeout=timeout)

        with self._lock:
            if not strategy.is_alive():
                del self._running[name]
                return True
            else:
                log.error("[%s] did not stop within %ss", name, timeout)
                return False

    def stop_all(self, timeout: float = 30) -> None:
        with self._lock:
            names = list(self._running.keys())
        # signal all first
        for name in names:
            self._running[name].request_stop()
        # then wait
        for name in names:
            self.stop(name, timeout=timeout)

    def status(self) -> List[StrategyStatus]:
        with self._lock:
            strategies = list(self._running.values())
        return [s.status() for s in strategies]
