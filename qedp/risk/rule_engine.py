"""Market and account rule engine."""
from typing import Dict, Any
from dataclasses import dataclass
import pandas as pd

from qedp.events.base import SignalEvent, ControlEvent


@dataclass
class RuleViolation:
    """Represents a rule violation."""
    rule_name: str
    reason: str
    severity: str  # "warn" or "reject"


class RuleEngine:
    """
    Enforces market and account rules.
    
    Examples:
    - T+1 settlement rules
    - Short sale restrictions
    - Circuit breakers
    - Position limits per symbol
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.t_plus_n = config.get("t_plus_n", 0)  # 0 = T+0, 1 = T+1
        self.allow_short = config.get("allow_short", True)
        self.max_symbol_position = config.get("max_symbol_position", {})
        self.circuit_breaker_pct = config.get("circuit_breaker_pct", 0.20)  # 20%
        
        # Track positions for rule checking
        self._positions: Dict[str, int] = {}
        self._pending_settlements: Dict[str, int] = {}
    
    def check_signal(self, signal: SignalEvent, current_positions: Dict[str, int]) -> RuleViolation | None:
        """
        Check if a signal violates any rules.
        
        Args:
            signal: The signal to check
            current_positions: Current position quantities by symbol
            
        Returns:
            RuleViolation if violated, None otherwise
        """
        self._positions = current_positions
        symbol = signal.symbol
        
        # Check position limits
        if symbol in self.max_symbol_position:
            max_pos = self.max_symbol_position[symbol]
            current_pos = current_positions.get(symbol, 0)
            target_qty = signal.target.get("qty", 0)
            
            if signal.intent == "open":
                new_pos = current_pos + target_qty
                if abs(new_pos) > max_pos:
                    return RuleViolation(
                        rule_name="max_symbol_position",
                        reason=f"Position limit exceeded for {symbol}: {new_pos} > {max_pos}",
                        severity="reject"
                    )
        
        # Check short sale restrictions
        if not self.allow_short:
            current_pos = current_positions.get(symbol, 0)
            target_qty = signal.target.get("qty", 0)
            
            if signal.intent == "open" and target_qty < 0:
                return RuleViolation(
                    rule_name="no_short_sales",
                    reason=f"Short sales not allowed for {symbol}",
                    severity="reject"
                )
            
            if signal.intent == "open" and current_pos + target_qty < 0:
                return RuleViolation(
                    rule_name="no_short_sales",
                    reason=f"Would create short position in {symbol}",
                    severity="reject"
                )
        
        # T+N settlement check
        if self.t_plus_n > 0:
            pending = self._pending_settlements.get(symbol, 0)
            if pending > 0 and signal.intent == "close":
                return RuleViolation(
                    rule_name="t_plus_n_settlement",
                    reason=f"Cannot sell {symbol} - pending T+{self.t_plus_n} settlement",
                    severity="reject"
                )
        
        return None
    
    def update_position(self, symbol: str, qty_change: int):
        """Update position tracking."""
        self._positions[symbol] = self._positions.get(symbol, 0) + qty_change
        
        # Track pending settlement for T+N
        if self.t_plus_n > 0 and qty_change > 0:
            self._pending_settlements[symbol] = self._pending_settlements.get(symbol, 0) + qty_change
    
    def settle_pending(self, symbol: str):
        """Mark pending settlements as settled (called after T+N days)."""
        if symbol in self._pending_settlements:
            del self._pending_settlements[symbol]
