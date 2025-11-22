# OneBacktest

OneBacktest is an event-driven backtesting framework designed for quantitative trading strategies. It provides a modular and extensible architecture to simulate trading strategies with historical data.

## Architecture

The framework follows a classic event-driven architecture where components interact through a central event queue. The main components are:

1.  **BacktestEngine**: The core of the system. It runs the event loop, dispatching events to the appropriate components.
2.  **DataHandler**: Manages the data feed. It drips historical data into the system bar-by-bar, generating `MarketEvent`s.
3.  **Strategy**: The brain of the system. It receives `MarketEvent`s, analyzes data, and generates `SignalEvent`s.
4.  **Portfolio**: Manages the state of the account. It tracks positions, holdings, and cash. It receives `SignalEvent`s to generate `OrderEvent`s and processes `FillEvent`s to update positions.
5.  **ExecutionHandler**: Simulates the execution of orders. It receives `OrderEvent`s and generates `FillEvent`s.

### Event Flow

1.  `DataHandler` updates bars -> `MarketEvent`
2.  `Strategy` receives `MarketEvent` -> `SignalEvent`
3.  `Portfolio` receives `SignalEvent` -> `OrderEvent`
4.  `ExecutionHandler` receives `OrderEvent` -> `FillEvent`
5.  `Portfolio` receives `FillEvent` -> Updates Positions & Holdings

## Components

### Data
-   **DataHandler**: Abstract base class for data handling.
-   **HistoricDataHandler**: Loads historical data from CSVs or other sources and replays it.
-   **YFinanceDataFeed**: Fetches historical data from Yahoo Finance.
-   **DataManager**: A helper to manage data storage and caching (optional usage).

### Strategy
-   **Strategy**: Abstract base class. Users should inherit from this to implement their logic in `calculate_signals`.
-   **Examples**:
    -   `SimpleMAStrategy`: A dual moving average crossover strategy.
    -   `BuyAndHoldStrategy`: A simple buy-and-hold benchmark.

### Portfolio
-   **Portfolio**: Handles position sizing (currently simple fixed quantity) and risk management (basic). It marks-to-market the portfolio on every market event.

### Execution
-   **ExecutionHandler**: Abstract base class for execution.
-   **SimulatedExecutionHandler**: Simulates order execution. It assumes instant fill at the current closing price (simplified).

## Usage

See `run_example.py` for a complete example of how to set up and run a backtest.

```python
# Basic setup
data_feed = YFinanceDataFeed()
data_handler = HistoricDataHandler(events, data_manager, symbols, start_date, end_date)
strategy = SimpleMAStrategy(data_handler, events)
portfolio = Portfolio(data_handler, events)
execution = SimulatedExecutionHandler(events, data_handler)
engine = BacktestEngine(data_handler, strategy, portfolio, execution)

engine.run_backtest()
```
