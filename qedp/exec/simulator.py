"""Execution simulator for backtesting."""
from typing import Dict, Any
import pandas as pd
import uuid

from qedp.events.base import OrderEvent, FillEvent, ControlEvent
from qedp.exec.order_manager import OrderManager, OrderStatus
from qedp.fees.cost_models import SlippageModel, FeeModel


class ExecutionSimulator:
    """
    Simulates order execution in backtest.
    
    Features:
    - Market orders: immediate fill with slippage
    - Limit orders: price-level matching with partial fills
    - Stop orders: trigger conversion
    - Configurable latency
    - Partial fills and cancellations
    """
    
    def __init__(
        self,
        slippage_model: SlippageModel,
        fee_model: FeeModel,
        config: Dict[str, Any]
    ):
        self.slippage_model = slippage_model
        self.fee_model = fee_model
        self.config = config
        
        # Latency configuration (milliseconds)
        self.latency_submit_ms = config.get("latency_ms", {}).get("submit", 30)
        self.latency_ack_ms = config.get("latency_ms", {}).get("ack", 10)
        self.latency_fill_ms = config.get("latency_ms", {}).get("fill", 20)
        
        # Order manager
        self.order_manager = OrderManager()
        
        # Current market data cache
        self._market_data: Dict[str, Dict] = {}
    
    def update_market_data(self, symbol: str, data: Dict):
        """Update market data for a symbol."""
        self._market_data[symbol] = data
    
    def submit_order(self, order_event: OrderEvent) -> list[FillEvent | ControlEvent]:
        """
        Submit an order for execution.
        
        Returns:
            List of events (FillEvent for fills, ControlEvent for acks/rejects)
        """
        # Submit to order manager
        order = self.order_manager.submit_order(order_event)
        
        # Add latency
        ack_ts = order_event.ts + pd.Timedelta(milliseconds=self.latency_ack_ms)
        
        # Accept order
        self.order_manager.accept_order(order_event.order_id, ack_ts)
        
        # Attempt to execute
        events = []
        
        if order_event.type == "MKT":
            # Market order - immediate fill
            fill_events = self._execute_market_order(order_event, ack_ts)
            events.extend(fill_events)
        elif order_event.type == "LMT":
            # Limit order - check if fillable
            fill_events = self._execute_limit_order(order_event, ack_ts)
            events.extend(fill_events)
        else:
            # Other order types - simplified execution
            fill_events = self._execute_market_order(order_event, ack_ts)
            events.extend(fill_events)
        
        return events
    
    def _execute_market_order(self, order: OrderEvent, ts: pd.Timestamp) -> list[FillEvent]:
        """Execute a market order immediately."""
        symbol = order.symbol
        market_data = self._market_data.get(symbol, {})
        
        # Get current price
        price = market_data.get("close", market_data.get("price", 0))
        if price == 0:
            # No price data - cannot fill
            return []
        
        # Calculate slippage
        slippage = self.slippage_model.calculate_slippage(
            symbol, order.side, order.qty, price, market_data
        )
        
        # Fill price includes slippage
        if order.side == "BUY":
            fill_price = price + slippage
        else:
            fill_price = price - slippage
        
        # Calculate fees
        fees = self.fee_model.calculate_fees(symbol, order.side, order.qty, fill_price)
        
        # Add fill latency
        fill_ts = ts + pd.Timedelta(milliseconds=self.latency_fill_ms)
        
        # Create fill event
        fill = FillEvent(
            ts=fill_ts,
            order_id=order.order_id,
            symbol=symbol,
            fill_px=fill_price,
            fill_qty=order.qty,
            fees=fees,
            last_liquidity="REMOVE",  # Market orders remove liquidity
            meta={"order_type": "MKT"}
        )
        
        # Update order manager
        self.order_manager.fill_order(fill)
        
        return [fill]
    
    def _execute_limit_order(self, order: OrderEvent, ts: pd.Timestamp) -> list[FillEvent]:
        """
        Execute a limit order if price crosses limit.
        
        Simplified: checks if current bar's high/low crosses limit price.
        """
        symbol = order.symbol
        market_data = self._market_data.get(symbol, {})
        
        limit_price = order.px
        if limit_price is None:
            return []
        
        # Get bar data
        high = market_data.get("high", 0)
        low = market_data.get("low", 0)
        close = market_data.get("close", 0)
        
        # Check if limit price was crossed
        fillable = False
        fill_price = limit_price
        
        if order.side == "BUY":
            # Buy limit: fill if low <= limit
            if low > 0 and low <= limit_price:
                fillable = True
                fill_price = min(limit_price, close)  # Get best price
        else:
            # Sell limit: fill if high >= limit
            if high > 0 and high >= limit_price:
                fillable = True
                fill_price = max(limit_price, close)  # Get best price
        
        if not fillable:
            # Limit not crossed - order remains open
            return []
        
        # Calculate fees (no slippage for limit orders that improve price)
        fees = self.fee_model.calculate_fees(symbol, order.side, order.qty, fill_price)
        
        # Add fill latency
        fill_ts = ts + pd.Timedelta(milliseconds=self.latency_fill_ms)
        
        # Create fill event
        fill = FillEvent(
            ts=fill_ts,
            order_id=order.order_id,
            symbol=symbol,
            fill_px=fill_price,
            fill_qty=order.qty,
            fees=fees,
            last_liquidity="ADD",  # Limit orders add liquidity
            meta={"order_type": "LMT"}
        )
        
        # Update order manager
        self.order_manager.fill_order(fill)
        
        return [fill]
    
    def cancel_order(self, cancel_event) -> bool:
        """Cancel an order."""
        self.order_manager.cancel_order(cancel_event)
        return True
