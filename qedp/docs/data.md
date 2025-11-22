# Data Module

## Overview

The `data` module is responsible for providing market data to the backtesting framework. It handles data ingestion, processing, and delivery through various data feeds. This module abstracts away the complexities of data sources, allowing strategies to focus on trading logic rather than data management.

## Key Components

### DataFeed (Abstract Base Class)

The `DataFeed` class defines the interface for all data sources. It supports both pull-based (iterator) and push-based (callback) data consumption patterns.

**Key Features:**
- Subscription management for specific symbols and data fields
- Event-driven data delivery through callbacks
- Iterator interface for sequential data processing

### FileFeed

A concrete implementation that loads historical market data from files (CSV, Parquet, HDF5).

**Supported Formats:**
- Parquet (recommended for performance)
- CSV (with timestamp parsing)
- HDF5

**Usage Example:**
```python
feed = FileFeed(
    name="historical_data",
    file_path="market_data.parquet",
    symbols=["AAPL", "GOOGL"]
)

for event in feed:
    # Process market events
    print(f"Received {event.etype} for {event.symbol}")
```

## Detailed Example with Input/Output Visualization

### Input: Sample CSV Data File (`market_data.csv`)
```csv
timestamp,symbol,open,high,low,close,volume
2023-01-01 09:30:00,AAPL,150.00,151.50,149.80,151.20,1000000
2023-01-01 09:31:00,AAPL,151.20,152.00,151.00,151.80,800000
2023-01-01 09:30:00,GOOGL,2800.00,2820.00,2790.00,2815.00,50000
2023-01-01 09:31:00,GOOGL,2815.00,2830.00,2810.00,2825.00,45000
```

### Processing Code
```python
from qedp.data.feed import FileFeed

# Create feed from CSV file
feed = FileFeed(
    name="sample_data",
    file_path="market_data.csv",
    file_format="csv",
    symbols=["AAPL", "GOOGL"]
)

# Subscribe to specific symbols
feed.subscribe("AAPL")
feed.subscribe("GOOGL")

# Process events
events = list(feed)  # Convert to list for demonstration
```

### Output: Generated MarketEvent Objects
```python
# Event 1: AAPL first bar
MarketEvent(
    ts=pd.Timestamp('2023-01-01 09:30:00'),
    symbol='AAPL',
    etype='bar',
    payload={
        'open': 150.00,
        'high': 151.50,
        'low': 149.80,
        'close': 151.20,
        'volume': 1000000
    },
    priority=10
)

# Event 2: AAPL second bar
MarketEvent(
    ts=pd.Timestamp('2023-01-01 09:31:00'),
    symbol='AAPL',
    etype='bar',
    payload={
        'open': 151.20,
        'high': 152.00,
        'low': 151.00,
        'close': 151.80,
        'volume': 800000
    },
    priority=10
)

# Event 3: GOOGL first bar
MarketEvent(
    ts=pd.Timestamp('2023-01-01 09:30:00'),
    symbol='GOOGL',
    etype='bar',
    payload={
        'open': 2800.00,
        'high': 2820.00,
        'low': 2790.00,
        'close': 2815.00,
        'volume': 50000
    },
    priority=10
)

# Event 4: GOOGL second bar
MarketEvent(
    ts=pd.Timestamp('2023-01-01 09:31:00'),
    symbol='GOOGL',
    etype='bar',
    payload={
        'open': 2815.00,
        'high': 2830.00,
        'low': 2810.00,
        'close': 2825.00,
        'volume': 45000
    },
    priority=10
)
```

### Data Flow Visualization
```
CSV File (Input)
    ↓
FileFeed._load_data()
    ↓
Pandas DataFrame
    ↓
FileFeed.__iter__()
    ↓
MarketEvent Objects (Output)
    ↓
Event Queue → Strategy Processing
```

### ReplayFeed

A specialized feed for replaying logged events, useful for audit trails and consistency testing.

## Design Principles

### Abstraction
The module provides a unified interface regardless of the underlying data source (files, live feeds, APIs).

### Flexibility
Supports different data granularities (ticks, bars, order books) and subscription patterns.

### Performance
Optimized for backtesting workloads with efficient data loading and minimal memory footprint.

## Integration with Framework

Data feeds integrate with the event-driven engine by generating `MarketEvent` objects that are queued and processed by strategies and other components.

## Common Use Cases

1. **Historical Backtesting**: Load CSV/Parquet files with historical price data
2. **Live Testing**: Connect to real-time data feeds (future extension)
3. **Event Replay**: Audit and debug trading sessions using logged events

## Assumptions for Beginners

If you're new to quantitative trading:

- **Market Data**: This is the price and volume information for financial instruments (stocks, futures, etc.)
- **Data Feeds**: Think of them as "data suppliers" that provide real-time or historical market information
- **Events**: Data is delivered as discrete events (like "AAPL price changed to $150") rather than continuous streams
- **Backtesting**: The framework simulates trading with historical data to test strategies before using real money
