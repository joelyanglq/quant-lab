"""Risk checks for portfolio limits."""
from typing import Dict, Any
from dataclasses import dataclass

from qedp.events.base import SignalEvent


@dataclass
class RiskViolation:
    """Represents a risk limit violation."""
    check_name: str
    reason: str
    severity: str  # "warn" or "reject"


class RiskCheck:
    """
    Portfolio-level risk checks.
    
    Examples:
    - Gross exposure limits
    - Net exposure limits
    - Single order notional limits
    - VAR limits
    - Margin requirements
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_gross_exposure = config.get("max_gross_exposure", 2.0)  # 2x capital
        self.max_net_exposure = config.get("max_net_exposure", 1.0)  # 1x capital
        self.max_order_notional = config.get("max_order_notional", float('inf'))
        self.soft_limits_action = config.get("soft_limits_action", "warn")
        self.hard_limits_action = config.get("hard_limits_action", "reject")
    
    def check_signal(
        self,
        signal: SignalEvent,
        portfolio_value: float,
        current_positions: Dict[str, int],
        current_prices: Dict[str, float]
    ) -> RiskViolation | None:
        """
        Check if a signal violates risk limits.
        
        Args:
            signal: The signal to check
            portfolio_value: Current portfolio value
            current_positions: Current positions {symbol: qty}
            current_prices: Current prices {symbol: price}
            
        Returns:
            RiskViolation if violated, None otherwise
        """
        symbol = signal.symbol
        target_qty = signal.target.get("qty", 0)
        price = current_prices.get(symbol, 0)
        
        # Check order notional limit
        order_notional = abs(target_qty * price)
        if order_notional > self.max_order_notional:
            return RiskViolation(
                check_name="max_order_notional",
                reason=f"Order notional ${order_notional:,.0f} exceeds limit ${self.max_order_notional:,.0f}",
                severity=self.hard_limits_action
            )
        
        # Calculate new exposure
        new_positions = current_positions.copy()
        current_qty = new_positions.get(symbol, 0)
        
        if signal.intent == "open":
            new_positions[symbol] = current_qty + target_qty
        elif signal.intent == "close":
            new_positions[symbol] = 0
        elif signal.intent == "reduce":
            new_positions[symbol] = current_qty - target_qty
        elif signal.intent == "flip":
            new_positions[symbol] = -current_qty
        
        # Calculate gross and net exposure
        gross_exposure = sum(abs(qty * current_prices.get(sym, 0)) 
                           for sym, qty in new_positions.items())
        net_exposure = sum(qty * current_prices.get(sym, 0) 
                         for sym, qty in new_positions.items())
        
        gross_ratio = gross_exposure / portfolio_value if portfolio_value > 0 else 0
        net_ratio = abs(net_exposure) / portfolio_value if portfolio_value > 0 else 0
        
        # Check gross exposure limit
        if gross_ratio > self.max_gross_exposure:
            return RiskViolation(
                check_name="max_gross_exposure",
                reason=f"Gross exposure {gross_ratio:.2f}x exceeds limit {self.max_gross_exposure:.2f}x",
                severity=self.hard_limits_action
            )
        
        # Check net exposure limit
        if net_ratio > self.max_net_exposure:
            return RiskViolation(
                check_name="max_net_exposure",
                reason=f"Net exposure {net_ratio:.2f}x exceeds limit {self.max_net_exposure:.2f}x",
                severity=self.soft_limits_action
            )
        
        return None
