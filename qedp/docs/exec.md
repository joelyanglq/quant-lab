# Execution Module

## Overview

The `exec` module handles the complete lifecycle of trading orders, from submission to final execution. It acts as the bridge between trading decisions (signals/orders) and actual trade confirmations (fills). This module ensures orders are properly tracked, managed, and settled.

## Key Components

### OrderManager

The central coordinator for all order-related operations.

**Core Responsibilities:**
- Order state tracking and lifecycle management
- Fill processing and position updates
- Audit trail generation
- Order status reporting

### Order (State Class)

Represents the current state of a trading order.

**Lifecycle States:**
- `NEW`: Order submitted but not yet accepted
- `ACCEPTED`: Order acknowledged by execution system
- `PARTIALLY_FILLED`: Some but not all quantity executed
- `FILLED`: Complete execution
- `CANCELED`: Order canceled before completion
- `REJECTED`: Order rejected by system/broker
- `EXPIRED`: Order expired due to time limits

**Key Properties:**
- `remaining_qty`: Unfilled quantity
- `avg_fill_price`: Volume-weighted average execution price
- `is_complete`: Whether order is in terminal state

### Order Lifecycle

1. **Submission**: Strategy/portfolio creates `OrderEvent`
2. **Acceptance**: Order acknowledged and assigned ID
3. **Execution**: Partial or full fills occur via `FillEvent`
4. **Completion**: Order reaches terminal state

## How It Works

### Order Submission
```python
order = order_manager.submit_order(order_event)
print(f"Order {order.event.order_id} submitted")
```

### Fill Processing
When fills arrive, they're automatically applied to orders:
```python
order_manager.fill_order(fill_event)
# Order status and P&L updated automatically
```

### State Tracking
The manager maintains complete order history for audit and analysis.

## Detailed Example with Input/Output Visualization

### Input: Order Submission
```python
from qedp.exec.order_manager import OrderManager
from qedp.events.base import OrderEvent
import pandas as pd

# Initialize order manager
order_manager = OrderManager()

# Create order event from portfolio
order_event = OrderEvent(
    ts=pd.Timestamp('2023-01-01 10:00:00'),
    order_id='AAPL_BUY_001',
    symbol='AAPL',
    side='BUY',
    type='MKT',
    qty=100,
    meta={'strategy': 'mean_reversion', 'signal_id': 'rsi_oversold'}
)

print("Input OrderEvent:")
print(order_event)
```

**Output:**
```
OrderEvent(ts=Timestamp('2023-01-01 10:00:00'), order_id='AAPL_BUY_001', symbol='AAPL', side='BUY', type='MKT', qty=100, px=None, tif='DAY', meta={'strategy': 'mean_reversion', 'signal_id': 'rsi_oversold'}, priority=30)
```

### Step 1: Order Submission Processing
```python
# Submit order to manager
order = order_manager.submit_order(order_event)

print("Order submitted successfully:")
print(f"Order ID: {order.event.order_id}")
print(f"Status: {order.status}")
print(f"Quantity: {order.event.qty}")
print(f"Remaining: {order.remaining_qty}")
```

**Output:**
```
Order submitted successfully:
Order ID: AAPL_BUY_001
Status: OrderStatus.NEW
Quantity: 100
Remaining: 100
```

### Step 2: Order Acceptance
```python
# Broker accepts the order
order_manager.accept_order('AAPL_BUY_001', pd.Timestamp('2023-01-01 10:00:01'))

print("After acceptance:")
print(f"Status: {order.status}")
print(f"Updated: {order.updated_at}")
```

**Output:**
```
After acceptance:
Status: OrderStatus.ACCEPTED
Updated: 2023-01-01 10:00:01
```

### Step 3: Partial Fill Processing
```python
from qedp.events.base import FillEvent

# First partial fill
fill1 = FillEvent(
    ts=pd.Timestamp('2023-01-01 10:00:02'),
    order_id='AAPL_BUY_001',
    symbol='AAPL',
    fill_px=145.50,
    fill_qty=50,  # Partial fill
    fees=0.725,   # Commission
    last_liquidity='ADD'
)

order_manager.fill_order(fill1)

print("After first fill:")
print(f"Status: {order.status}")
print(f"Filled: {order.filled_qty}")
print(f"Remaining: {order.remaining_qty}")
print(f"Avg Price: ${order.avg_fill_price:.2f}")
```

**Output:**
```
After first fill:
Status: OrderStatus.PARTIALLY_FILLED
Quantity: 50
Remaining: 50
Avg Price: $145.50
```

### Step 4: Second Fill (Complete Order)
```python
# Second fill completes the order
fill2 = FillEvent(
    ts=pd.Timestamp('2023-01-01 10:00:03'),
    order_id='AAPL_BUY_001',
    symbol='AAPL',
    fill_px=145.55,
    fill_qty=50,  # Remaining quantity
    fees=0.728,   # Commission
    last_liquidity='REMOVE'
)

order_manager.fill_order(fill2)

print("After second fill (complete):")
print(f"Status: {order.status}")
print(f"Filled: {order.filled_qty}")
print(f"Remaining: {order.remaining_qty}")
print(f"Avg Price: ${order.avg_fill_price:.2f}")
print(f"Total Fees: ${sum(f.fees for f in order.fills):.3f}")
```

**Output:**
```
After second fill (complete):
Status: OrderStatus.FILLED
Filled: 100
Remaining: 0
Avg Price: $145.525
Total Fees: $1.453
```

### Step 5: Order Audit Log
```python
# Get complete order history
audit_log = order_manager.get_order_log()

print("Order Audit Trail:")
for entry in audit_log:
    print(f"{entry['ts']} - {entry['event_type']}: {entry['status']} "
          f"(Filled: {entry['filled_qty']}, Remaining: {entry['remaining_qty']})")
```

**Output:**
```
Order Audit Trail:
2023-01-01 10:00:00 - submit: NEW (Filled: 0, Remaining: 100)
2023-01-01 10:00:01 - accept: ACCEPTED (Filled: 0, Remaining: 100)
2023-01-01 10:00:02 - fill: PARTIALLY_FILLED (Filled: 50, Remaining: 50)
2023-01-01 10:00:03 - fill: FILLED (Filled: 100, Remaining: 0)
```

### Order Lifecycle Visualization
```
Input: OrderEvent
    ↓
1. submit_order() → OrderStatus.NEW
    ↓
2. accept_order() → OrderStatus.ACCEPTED
    ↓
3. fill_order() → OrderStatus.PARTIALLY_FILLED
    ↓
4. fill_order() → OrderStatus.FILLED
    ↓
Complete Audit Log Available
```

### Fill Details Breakdown
```python
print("Individual Fill Details:")
for i, fill in enumerate(order.fills, 1):
    print(f"Fill {i}: {fill.fill_qty} @ ${fill.fill_px} "
          f"(Fees: ${fill.fees}, Liquidity: {fill.last_liquidity})")

# Output:
# Fill 1: 50 @ $145.50 (Fees: $0.725, Liquidity: ADD)
# Fill 2: 50 @ $145.55 (Fees: $0.728, Liquidity: REMOVE)
```

### Active Orders Query
```python
# Check for any active orders
active_orders = order_manager.get_active_orders()
print(f"Active orders: {len(active_orders)}")

# Since order is complete, should be empty
# Output: Active orders: 0
```

## Integration with Framework

### Receives Events:
- `OrderEvent`: New orders to process
- `FillEvent`: Execution confirmations
- `CancelEvent`: Cancellation requests

### Generates Events:
- `AccountEvent`: Portfolio updates after fills
- Status updates for monitoring systems

## Design Principles

### State Machine Pattern
Orders follow a strict state transition diagram, preventing invalid states.

### Audit Trail
Every order action is logged with timestamps for compliance and debugging.

### Idempotent Operations
Processing the same event multiple times is safe.

## Common Use Cases

1. **Market Orders**: Immediate execution at current prices
2. **Limit Orders**: Execute only at specified price or better
3. **Partial Fills**: Handle orders that execute in multiple pieces
4. **Order Cancellation**: Stop unfilled orders
5. **Position Tracking**: Monitor real-time position changes

## Assumptions for Beginners

If you're new to trading execution:

- **Orders vs. Fills**: An order is a request to trade; a fill is the actual execution
- **Partial Fills**: Large orders often execute in pieces over time
- **Slippage**: The difference between expected and actual execution price
- **Market Impact**: Large orders can move prices against you
- **Time in Force**: How long an order remains active ("DAY", "GTC", etc.)

## Performance Considerations

- Order tracking scales with active order count
- Fill processing is optimized for high-frequency execution
- Memory usage depends on order history retention period

## Risk Management Integration

The execution module works closely with risk management:
- Validates orders against position limits
- Tracks real-time exposure changes
- Provides data for margin calculations

## Best Practices

- Monitor order status actively
- Handle partial fills appropriately
- Implement proper error handling for rejections
- Maintain comprehensive audit logs
- Use appropriate order types for different market conditions
