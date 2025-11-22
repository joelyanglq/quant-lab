"""Clock implementations for time progression."""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Literal
import pandas as pd

from qedp.events.base import ClockEvent


class Clock(ABC):
    """Abstract base class for clock implementations."""
    
    @abstractmethod
    def now(self) -> pd.Timestamp:
        """Get current time."""
        pass
    
    @abstractmethod
    def tick(self) -> ClockEvent | None:
        """Advance time and return ClockEvent if phase boundary crossed."""
        pass


class BacktestClock(Clock):
    """
    Backtest clock with discrete time steps.
    
    Advances time by fixed increments and emits ClockEvent at session boundaries.
    """
    
    def __init__(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        step: str = "1s",  # pandas frequency string
        timezone: str = "America/New_York",
        market_open: str = "09:30",
        market_close: str = "16:00"
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.step = pd.Timedelta(step)
        self.timezone = timezone
        
        # Parse market hours
        self.market_open_time = pd.Timestamp(market_open).time()
        self.market_close_time = pd.Timestamp(market_close).time()
        
        self._current_time = start_time
        self._session_id = self._get_session_id(start_time)
        self._last_phase = None
        
    def now(self) -> pd.Timestamp:
        return self._current_time
    
    def tick(self) -> ClockEvent | None:
        """Advance time by one step."""
        if self._current_time >= self.end_time:
            return None
            
        # Advance time
        self._current_time += self.step
        
        # Check for phase transitions
        current_phase = self._get_phase(self._current_time)
        current_session = self._get_session_id(self._current_time)
        
        # Emit ClockEvent on phase change
        if current_phase != self._last_phase or current_session != self._session_id:
            self._last_phase = current_phase
            self._session_id = current_session
            
            return ClockEvent(
                ts=self._current_time,
                phase=current_phase,
                session_id=current_session
            )
        
        return None
    
    def _get_phase(self, ts: pd.Timestamp) -> Literal["pre", "open", "close", "after"]:
        """Determine market phase for given timestamp."""
        time = ts.time()
        
        if time < self.market_open_time:
            return "pre"
        elif time < self.market_close_time:
            return "open"
        elif time == self.market_close_time:
            return "close"
        else:
            return "after"
    
    def _get_session_id(self, ts: pd.Timestamp) -> str:
        """Get trading session ID (date string)."""
        return ts.strftime("%Y-%m-%d")


class LiveClock(Clock):
    """
    Live trading clock.
    
    Does not advance time automatically; time is driven by external events.
    """
    
    def __init__(self, timezone: str = "America/New_York"):
        self.timezone = timezone
        
    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz=self.timezone)
    
    def tick(self) -> ClockEvent | None:
        """Live clock does not tick; events are driven externally."""
        return None
