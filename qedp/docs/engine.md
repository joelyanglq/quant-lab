# Engine Module

## Overview

The `engine` module is the heart of the event-driven backtesting framework. It orchestrates the entire simulation by managing time progression, event routing, and component coordination. Think of it as the "conductor" that ensures all parts of the system work together in harmony.

## Key Components

### Engine (Main Class)

The `Engine` class implements the core event loop and handles the lifecycle of a backtesting run.

**Core Responsibilities:**
- Event routing: Directs events to appropriate handlers
- Time management: Coordinates with the clock for time progression
- Control flow: Manages start/stop/pause operations
- Error handling: Isolates component failures

### Clock

Manages simulation time and session boundaries (market open/close).

**Key Features:**
- Time progression in discrete steps
- Session phase tracking (pre-open, open, close, after-hours)
- Clock events for time-based triggers

### EventQueue

A priority queue that orders events by timestamp and priority.

**Priority System:**
- Control events (highest priority)
- Clock events
- Fill events
- Cancel events
- Order events
- Signal events
- Market events (lowest priority)

## How It Works

### Event Loop

The engine runs a continuous loop:

1. **Check Queue**: Look for pending events
2. **Process Events**: Route events to registered handlers
3. **Pump**: Allow handlers to generate new events in the same time step
4. **Advance Time**: If no events, tick the clock forward

### Handler Registration

Components register event handlers with the engine:

```python
engine.register_handler(MarketEvent, strategy.on_market_event)
engine.register_handler(SignalEvent, portfolio.on_signal_event)
```

## Detailed Example with Input/Output Visualization

### Input: System Configuration
```python
from qedp.engine.engine import Engine
from qedp.engine.clock import Clock
from qedp.engine.queue import EventQueue
from qedp.events.base import MarketEvent, SignalEvent, OrderEvent

# Initialize components
clock = Clock()
event_queue = EventQueue()
engine = Engine(clock, event_queue)

# Register event handlers
def handle_market_event(event):
    print(f"Strategy received market data: {event.symbol} @ {event.payload['close']}")
    # Strategy logic would generate signals here
    if event.payload['close'] > 150:
        signal = SignalEvent(
            ts=event.ts,
            symbol=event.symbol,
            intent="open",
            target={"qty": 100}
        )
        event_queue.push(signal)

def handle_signal_event(event):
    print(f"Portfolio received signal: {event.intent} {event.target['qty']} {event.symbol}")
    # Portfolio logic would generate orders here
    order = OrderEvent(
        ts=event.ts,
        order_id="ORD001",
        symbol=event.symbol,
        side="BUY",
        type="MKT",
        qty=event.target['qty']
    )
    event_queue.push(order)

engine.register_handler(MarketEvent, handle_market_event)
engine.register_handler(SignalEvent, handle_signal_event)
```

### Input: Market Data Events
```python
# Simulate incoming market data
market_events = [
    MarketEvent(
        ts=pd.Timestamp('2023-01-01 09:30:00'),
        symbol='AAPL',
        etype='bar',
        payload={'open': 150.0, 'high': 151.5, 'low': 149.8, 'close': 151.2, 'volume': 1000000}
    ),
    MarketEvent(
        ts=pd.Timestamp('2023-01-01 09:31:00'),
        symbol='AAPL',
        etype='bar',
        payload={'open': 151.2, 'high': 152.0, 'low': 151.0, 'close': 152.5, 'volume': 800000}
    )
]

# Add events to queue
for event in market_events:
    event_queue.push(event)
```

### Processing: Engine Run
```python
# Start engine (simplified example - normally runs continuously)
print("Starting engine...")
print(f"Initial queue size: {event_queue.size()}")

# Process first event
engine.pump()  # Process MarketEvent → Strategy → SignalEvent
print(f"After first pump - queue size: {event_queue.size()}")

# Process derived events
engine.pump()  # Process SignalEvent → Portfolio → OrderEvent
print(f"After second pump - queue size: {event_queue.size()}")
```

### Output: Console Log and Event Flow
```
Starting engine...
Initial queue size: 2

Strategy received market data: AAPL @ 151.2
After first pump - queue size: 2
[MarketEvent processed, SignalEvent added]

Strategy received market data: AAPL @ 152.5
Portfolio received signal: open 100 AAPL
After second pump - queue size: 2
[SignalEvent processed, OrderEvent added]
```

### Final Event Queue State
```python
# Events remaining in queue
remaining_events = []
while not event_queue.is_empty():
    event = event_queue.pop()
    remaining_events.append(event)
    print(f"Remaining: {type(event).__name__} for {event.symbol}")

# Output:
# Remaining: OrderEvent for AAPL
# Remaining: OrderEvent for AAPL
```

### Event Flow Visualization
```
Input Events → Event Queue
    ↓
Engine.pump() → Route to Handlers
    ↓
Strategy Handler → Generate SignalEvent → Queue
    ↓
Engine.pump() → Route SignalEvent
    ↓
Portfolio Handler → Generate OrderEvent → Queue
    ↓
Ready for Execution Module Processing
```

### Control Events

The engine responds to control commands:
- `stop`: End the simulation
- `pause`/`resume`: Suspend/resume execution
- `reload`: Reconfigure components

## Design Principles

### Event-Driven Architecture
Everything is event-based, allowing loose coupling between components.

### Priority-Based Processing
Critical events (like fills) are processed before less urgent ones.

### Time Synchronization
All components operate on the same timeline, ensuring deterministic results.

## Integration with Framework

The engine coordinates:
- **Data feeds** → Generate market events
- **Strategies** → Process market data, generate signals
- **Portfolio** → Track positions, generate orders
- **Execution** → Handle order lifecycle
- **Risk management** → Validate trades

## Common Use Cases

1. **Standard Backtest**: Run complete simulation from start to finish
2. **Interactive Debugging**: Pause at specific points to inspect state
3. **Partial Runs**: Test specific time periods or market conditions

## Assumptions for Beginners

If you're new to backtesting:

- **Event-Driven**: Instead of a linear script, the system reacts to "events" (price changes, order fills, etc.)
- **Simulation Time**: The framework advances time in steps, processing events as they occur
- **Component Coordination**: Different parts (strategy, portfolio, execution) communicate through events
- **Deterministic**: Same inputs always produce same results, crucial for testing strategies

## Performance Considerations

- Event processing is optimized for high-frequency data
- Memory usage scales with event volume
- CPU usage depends on handler complexity
