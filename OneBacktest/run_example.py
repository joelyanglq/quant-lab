import queue
from datetime import datetime

from backtest.engine import BacktestEngine
from data.handler import HistoricDataHandler
from data.yfinance_feed import YFinanceDataFeed
from data.manager import DataManager
from strategy.examples.simple_ma import SimpleMAStrategy
from strategy.examples.buy_and_hold import BuyAndHoldStrategy
from strategy.portfolio import Portfolio
from execution.handler import SimulatedExecutionHandler
from event import EventType

def run_ma_backtest():
    print("Running Simple Moving Average Backtest...")
    events = queue.Queue()
    
    # Define symbols and dates
    symbols = ['SPY']
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 1, 1)
    
    # Data Feed and Manager
    data_feed = YFinanceDataFeed()
    data_manager = DataManager(data_feed, use_cache=False) # Disable cache for simplicity or enable if needed
    
    # Data Handler
    data_handler = HistoricDataHandler(events, data_manager, symbols, start_date, end_date)
    
    # Strategy
    strategy = SimpleMAStrategy(data_handler, events, short_window=20, long_window=50)
    
    # Portfolio
    portfolio = Portfolio(data_handler, events, initial_capital=100000.0)
    
    # Execution Handler
    execution_handler = SimulatedExecutionHandler(events, data_handler)
    
    # Engine
    engine = BacktestEngine(data_handler, strategy, portfolio, execution_handler)
    
    # Run
    engine.run_backtest()
    
    # Print Results
    print("Final Portfolio Value: ${:.2f}".format(portfolio.current_holdings['total']))
    print("Return: {:.2f}%".format((portfolio.current_holdings['total'] - 100000.0) / 100000.0 * 100))
    
    # Print transaction history (simplified)
    # In a real system we would inspect portfolio.all_holdings or a transaction log

if __name__ == "__main__":
    run_ma_backtest()
