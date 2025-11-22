import queue
from datetime import datetime
from typing import Optional

from event import EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent

class BacktestEngine:
    """
    Event-driven Backtest Engine
    """
    def __init__(self, data_handler, strategy, portfolio, execution_handler):
        """
        Initializes the backtest engine.

        Parameters:
        data_handler: Handles market data feed.
        strategy: Generates signals based on market data.
        portfolio: Manages positions and holdings.
        execution_handler: Executes orders.
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.events = queue.Queue()
        
        # Inject event queue into components
        self.data_handler.events = self.events
        self.strategy.events = self.events
        self.portfolio.events = self.events
        self.execution_handler.events = self.events

    def run_backtest(self):
        """
        Executes the backtest loop.
        """
        print("Starting Backtest...")
        while True:
            # Update data handler (feed new bars)
            if self.data_handler.continue_backtest:
                self.data_handler.update_bars()
            else:
                break

            # Handle events
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    if event is not None:
                        if event.event_type == EventType.MARKET:
                            self.strategy.calculate_signals(event)
                            self.portfolio.update_market(event)

                        elif event.event_type == EventType.SIGNAL:
                            self.portfolio.update_signal(event)

                        elif event.event_type == EventType.ORDER:
                            self.execution_handler.execute_order(event)

                        elif event.event_type == EventType.FILL:
                            self.portfolio.update_fill(event)

        print("Backtest finished.")
