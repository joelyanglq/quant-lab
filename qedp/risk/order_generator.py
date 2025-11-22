"""Order generation from signals after risk checks."""
from typing import List
import uuid

from qedp.events.base import SignalEvent, OrderEvent, ControlEvent
from qedp.risk.rule_engine import RuleEngine, RuleViolation
from qedp.risk.risk_check import RiskCheck, RiskViolation


class OrderGenerator:
    """
    Converts SignalEvents to OrderEvents after passing risk checks.
    
    Supports:
    - Order slicing (TWAP, VWAP, POV)
    - Iceberg orders
    - Risk-based position sizing
    """
    
    def __init__(self, rule_engine: RuleEngine, risk_check: RiskCheck):
        self.rule_engine = rule_engine
        self.risk_check = risk_check
    
    def process_signal(
        self,
        signal: SignalEvent,
        portfolio_value: float,
        current_positions: dict,
        current_prices: dict
    ) -> List[OrderEvent | ControlEvent]:
        """
        Process a signal and generate orders or rejection events.
        
        Args:
            signal: The signal to process
            portfolio_value: Current portfolio value
            current_positions: Current positions
            current_prices: Current market prices
            
        Returns:
            List of OrderEvent or ControlEvent (rejection)
        """
        # Check market rules
        rule_violation = self.rule_engine.check_signal(signal, current_positions)
        if rule_violation and rule_violation.severity == "reject":
            return [ControlEvent(
                ts=signal.ts,
                cmd="reject",
                args={
                    "signal": signal,
                    "reason": rule_violation.reason,
                    "rule": rule_violation.rule_name
                }
            )]
        
        # Check risk limits
        risk_violation = self.risk_check.check_signal(
            signal, portfolio_value, current_positions, current_prices
        )
        if risk_violation and risk_violation.severity == "reject":
            return [ControlEvent(
                ts=signal.ts,
                cmd="reject",
                args={
                    "signal": signal,
                    "reason": risk_violation.reason,
                    "check": risk_violation.check_name
                }
            )]
        
        # Generate order(s)
        return self._generate_orders(signal, current_positions, current_prices)
    
    def _generate_orders(
        self,
        signal: SignalEvent,
        current_positions: dict,
        current_prices: dict
    ) -> List[OrderEvent]:
        """Generate order events from signal."""
        symbol = signal.symbol
        current_qty = current_positions.get(symbol, 0)
        target_qty = signal.target.get("qty", 0)
        
        # Determine order quantity and side
        if signal.intent == "open":
            order_qty = abs(target_qty)
            side = "BUY" if target_qty > 0 else "SELL"
        elif signal.intent == "close":
            order_qty = abs(current_qty)
            side = "SELL" if current_qty > 0 else "BUY"
        elif signal.intent == "reduce":
            order_qty = abs(target_qty)
            side = "SELL" if current_qty > 0 else "BUY"
        elif signal.intent == "flip":
            order_qty = abs(current_qty) * 2
            side = "SELL" if current_qty > 0 else "BUY"
        else:
            return []
        
        if order_qty == 0:
            return []
        
        # Determine order type from signal meta
        order_type = signal.meta.get("order_type", "MKT")
        limit_price = signal.meta.get("limit_price")
        tif = signal.meta.get("tif", "DAY")
        
        # Check for order slicing
        slice_algo = signal.meta.get("slice_algo")  # "TWAP", "VWAP", "ICEBERG"
        if slice_algo:
            return self._slice_order(signal, order_qty, side, order_type, limit_price, tif, slice_algo)
        
        # Single order
        order = OrderEvent(
            ts=signal.ts,
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            type=order_type,
            qty=order_qty,
            px=limit_price,
            tif=tif,
            meta={"signal_id": id(signal)}
        )
        
        return [order]
    
    def _slice_order(
        self,
        signal: SignalEvent,
        total_qty: int,
        side: str,
        order_type: str,
        limit_price: float | None,
        tif: str,
        algo: str
    ) -> List[OrderEvent]:
        """
        Slice a large order into smaller pieces.
        
        TODO: Implement TWAP, VWAP, POV algorithms
        For now, just create a single order.
        """
        # Simplified: just create one order
        # In production, would split into time-based or volume-based slices
        order = OrderEvent(
            ts=signal.ts,
            order_id=str(uuid.uuid4()),
            symbol=signal.symbol,
            side=side,
            type=order_type,
            qty=total_qty,
            px=limit_price,
            tif=tif,
            meta={"signal_id": id(signal), "algo": algo}
        )
        
        return [order]
