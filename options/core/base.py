"""Strategy base class — lifecycle interface + threading."""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class StrategyState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class StrategyStatus:
    """Thread-safe status snapshot for CLI/GUI."""
    name: str
    state: StrategyState
    open_positions: int
    last_poll: Optional[str] = None
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """Base class for all options strategies.

    Each strategy runs in its own thread with its own IBKR connection.
    Subclasses implement the abstract methods; the base class handles
    threading, state management, and the polling loop.
    """

    def __init__(self):
        self._state = StrategyState.STOPPED
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None
        self._last_poll: Optional[str] = None
        self._lock = threading.Lock()

    # ── identity ─────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy name, e.g. 'monthly_ic'."""
        ...

    # ── lifecycle hooks (subclass implements) ────────────────────

    @abstractmethod
    def make_broker(self):
        """Create and return a Broker instance."""
        ...

    @abstractmethod
    def make_logger(self):
        """Create and return a Logger instance."""
        ...

    @abstractmethod
    def setup(self, broker, logger) -> None:
        """Called once after connect, before polling."""
        ...

    @abstractmethod
    def poll_once(self, broker, logger) -> None:
        """One polling cycle — risk checks, exits, entries."""
        ...

    def teardown(self, broker, logger) -> None:
        """Called once after loop exits, before disconnect."""
        pass

    @abstractmethod
    def poll_interval(self) -> int:
        """Seconds between polling cycles."""
        ...

    @abstractmethod
    def in_trading_hours(self) -> bool:
        """Return True if within this strategy's trading window."""
        ...

    @abstractmethod
    def get_open_position_count(self) -> int:
        ...

    # ── status (thread-safe) ─────────────────────────────────────

    def status(self) -> StrategyStatus:
        with self._lock:
            return StrategyStatus(
                name=self.name,
                state=self._state,
                open_positions=self.get_open_position_count(),
                last_poll=self._last_poll,
                error=self._error,
            )

    # ── threading ────────────────────────────────────────────────

    def _run(self) -> None:
        """Thread entry point. Do not call directly."""
        log = logging.getLogger(f"strategy.{self.name}")

        with self._lock:
            self._state = StrategyState.STARTING

        broker = self.make_broker()
        logger = self.make_logger()

        if not broker.connect():
            with self._lock:
                self._state = StrategyState.ERROR
                self._error = "IBKR connection failed"
            return

        with self._lock:
            self._state = StrategyState.RUNNING
        log.info("[%s] started", self.name)

        try:
            self.setup(broker, logger)

            while not self._stop_event.is_set():
                if not self.in_trading_hours():
                    # sleep shorter outside RTH to check stop_event
                    broker.ib.sleep(30)
                    continue

                self.poll_once(broker, logger)
                self._last_poll = datetime.now(ET).isoformat()

                # sleep in small chunks so stop_event is checked
                remaining = self.poll_interval()
                while remaining > 0 and not self._stop_event.is_set():
                    chunk = min(remaining, 5)
                    broker.ib.sleep(chunk)
                    remaining -= chunk

        except Exception as e:
            log.exception("[%s] error: %s", self.name, e)
            with self._lock:
                self._state = StrategyState.ERROR
                self._error = str(e)
        finally:
            try:
                self.teardown(broker, logger)
            except Exception:
                log.exception("[%s] teardown error", self.name)
            broker.disconnect()
            with self._lock:
                if self._state != StrategyState.ERROR:
                    self._state = StrategyState.STOPPED
            log.info("[%s] stopped", self.name)

    def request_stop(self) -> None:
        self._stop_event.set()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
