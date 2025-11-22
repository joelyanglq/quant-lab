import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import queue

from event import EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent, OrderType, OrderSide

class Portfolio:
    """
    The Portfolio class handles the positions and market value of all instruments
    at a resolution of a "bar", i.e. secondly, minutely, 5-min, 30-min,
    60-min or EOD.
    """
    def __init__(self, data_handler, events: queue.Queue, initial_capital=100000.0):
        """
        Initializes the portfolio with bars and an event queue.
        Also includes a starting capital.

        Parameters:
        data_handler: The DataHandler object with current market data.
        events: The Event Queue object.
        initial_capital: The starting capital in USD.
        """
        self.data_handler = data_handler
        self.events = events
        self.initial_capital = initial_capital

        self.all_positions = self.construct_all_positions()
        self.current_positions = dict()
        
        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()

    def construct_all_positions(self):
        """
        Constructs the positions list using the start_date to determine
        when the time index will begin.
        """
        return []

    def construct_all_holdings(self):
        """
        Constructs the holdings list using the start_date to determine
        when the time index will begin.
        """
        return []

    def construct_current_holdings(self):
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        d = dict( (s, 0.0) for s in self.data_handler.symbols )
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_market(self, event: MarketEvent):
        """
        Update the portfolio to reflect the market value of the assets
        at the current time index.
        """
        if event.event_type == EventType.MARKET:
            # Create a new holdings object (copy of current)
            # In a real system we might append to a list or dataframe
            # For now, we update current_holdings based on latest prices
            
            # Note: In a full implementation we would append to all_holdings history
            # self.all_holdings.append(self.current_holdings.copy())
            
            # Update market value
            total_market_value = 0.0
            for s in self.data_handler.symbols:
                # Get latest close price
                bars = self.data_handler.get_latest_bars(s, N=1)
                if not bars.empty:
                    price = bars.iloc[-1]['close']
                    market_value = self.current_positions.get(s, 0) * price
                    self.current_holdings[s] = market_value
                    total_market_value += market_value
            
            self.current_holdings['total'] = self.current_holdings['cash'] + total_market_value
            
            # Append to history (simplified)
            # We should add timestamp here
            self.all_holdings.append(self.current_holdings.copy())

    def update_signal(self, event: SignalEvent):
        """
        Acts on a SignalEvent to generate new orders.
        """
        if event.event_type == EventType.SIGNAL:
            order_event = self.generate_order(event)
            self.events.put(order_event)

    def generate_order(self, signal: SignalEvent) -> OrderEvent:
        """
        Simply files an Order object as a constant quantity sizing of the signal object,
        without risk management or position sizing considerations.
        """
        order = None
        
        symbol = signal.symbol
        direction = signal.signal_type
        strength = signal.strength
        
        mkt_quantity = 100 # Default quantity
        
        cur_quantity = self.current_positions.get(symbol, 0)
        
        order_type = OrderType.MARKET # Default to Market Order
        
        if direction == 'long':
            side = OrderSide.BUY
        elif direction == 'short':
            side = OrderSide.SELL
        elif direction == 'exit':
            if cur_quantity > 0:
                side = OrderSide.SELL
                mkt_quantity = abs(cur_quantity)
            elif cur_quantity < 0:
                side = OrderSide.BUY
                mkt_quantity = abs(cur_quantity)
            else:
                # No position to exit
                return None
        
        order = OrderEvent(
            order_id=str(id(object())), # Simple ID generation
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=mkt_quantity
        )
        
        return order

    def update_fill(self, event: FillEvent):
        """
        Updates the portfolio current positions and holdings 
        from a FillEvent.
        """
        if event.event_type == EventType.FILL:
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)

    def update_positions_from_fill(self, fill: FillEvent):
        """
        Takes a Fill object and updates the position matrix to reflect
        the new position.
        """
        fill_dir = 0
        if fill.side == OrderSide.BUY:
            fill_dir = 1
        if fill.side == OrderSide.SELL:
            fill_dir = -1
            
        self.current_positions[fill.symbol] = self.current_positions.get(fill.symbol, 0) + fill_dir * fill.quantity

    def update_holdings_from_fill(self, fill: FillEvent):
        """
        Takes a Fill object and updates the holdings matrix to reflect
        the holdings value.
        """
        fill_dir = 0
        if fill.side == OrderSide.BUY:
            fill_dir = 1
        if fill.side == OrderSide.SELL:
            fill_dir = -1
            
        fill_cost = self.data_handler.get_latest_bars(fill.symbol, N=1).iloc[-1]['close'] # Use latest close for cost estimate or fill price
        cost = fill_dir * fill_cost * fill.quantity
        
        self.current_holdings[fill.symbol] += cost
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] -= (cost + fill.commission)
        self.current_holdings['total'] -= (fill.commission)
