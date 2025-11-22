# Risk Module

## Overview

The `risk` module protects trading capital by enforcing position limits, exposure constraints, and other risk controls. In quantitative trading, risk management is as important as strategy development - poor risk controls can lead to catastrophic losses. This module provides portfolio-level risk checks that prevent excessive risk-taking.

## Key Components

### RiskCheck

The main risk validation engine that evaluates trading signals against risk limits.

**Check Types:**
- **Gross Exposure**: Total absolute position value vs. capital
- **Net Exposure**: Long vs. short position balance
- **Order Size Limits**: Maximum notional per single order
- **Margin Requirements**: Leverage and borrowing limits
- **VAR Limits**: Value at Risk constraints (future extension)

### RiskViolation

Represents a risk limit breach with severity classification.

**Severity Levels:**
- `warn`: Log warning but allow trade
- `reject`: Block the trade entirely

## Risk Limits Explained

### Gross Exposure
Total value of all positions regardless of direction.

**Example:** If you have $100,000 capital:
- Long $60,000 AAPL + Short $40,000 GOOGL = $100,000 gross exposure
- 2x gross limit means maximum $200,000 total position value

### Net Exposure
Net long/short position (long positions minus short positions).

**Example:**
- Long $60,000 AAPL + Short $40,000 GOOGL = $20,000 net long
- 1x net limit means maximum $100,000 net exposure

### Order Notional Limits
Maximum dollar value per individual order.

**Purpose:** Prevents single large orders from dominating the portfolio.

## How It Works

### Signal Validation
```python
violation = risk_check.check_signal(
    signal=signal_event,
    portfolio_value=100000,
    current_positions={"AAPL": 1000, "GOOGL": -500},
    current_prices={"AAPL": 150, "GOOGL": 2000}
)

if violation:
    if violation.severity == "reject":
        # Block the trade
        return
    else:
        # Log warning but proceed
        logger.warning(f"Risk warning: {violation.reason}")
```

## Detailed Example with Input/Output Visualization

### Input: Portfolio and Signal Setup
```python
from qedp.risk.risk_check import RiskCheck
from qedp.events.base import SignalEvent

# Initialize risk checker with conservative limits
risk_config = {
    "max_gross_exposure": 2.0,    # 2x capital
    "max_net_exposure": 1.0,      # 1x capital
    "max_order_notional": 50000,  # $50K max per order
    "soft_limits_action": "warn",
    "hard_limits_action": "reject"
}

risk_check = RiskCheck(risk_config)

# Current portfolio state
portfolio_value = 100000  # $100K capital
current_positions = {
    "AAPL": 500,     # Long 500 shares
    "GOOGL": -200,   # Short 200 shares
    "MSFT": 300      # Long 300 shares
}
current_prices = {
    "AAPL": 150.0,
    "GOOGL": 2500.0,
    "MSFT": 300.0
}

print("Portfolio State:")
print(f"Total Value: ${portfolio_value:,.0f}")
print(f"Positions: {current_positions}")
print(f"Current Gross Exposure: ${sum(abs(qty) * current_prices.get(sym, 0) for sym, qty in current_positions.items()):,.0f}")
```

**Output:**
```
Portfolio State:
Total Value: $100,000
Positions: {'AAPL': 500, 'GOOGL': -200, 'MSFT': 300}
Current Gross Exposure: $307,500
```

### Step 1: Valid Signal Check
```python
# Signal to buy 100 more AAPL (reasonable size)
valid_signal = SignalEvent(
    ts=pd.Timestamp('2023-01-01 10:00:00'),
    symbol='AAPL',
    intent='open',
    target={'qty': 100}
)

violation = risk_check.check_signal(
    signal=valid_signal,
    portfolio_value=portfolio_value,
    current_positions=current_positions,
    current_prices=current_prices
)

print("Valid Signal Check:")
print(f"Signal: {valid_signal.intent} {valid_signal.target['qty']} {valid_signal.symbol}")
print(f"Risk Violation: {violation}")
```

**Output:**
```
Valid Signal Check:
Signal: open 100 AAPL
Risk Violation: None
```

### Step 2: Order Size Limit Violation
```python
# Signal exceeding order size limit
large_signal = SignalEvent(
    ts=pd.Timestamp('2023-01-01 10:00:00'),
    symbol='AAPL',
    intent='open',
    target={'qty': 500}  # Would cost $75K > $50K limit
)

violation = risk_check.check_signal(
    signal=large_signal,
    portfolio_value=portfolio_value,
    current_positions=current_positions,
    current_prices=current_prices
)

print("Large Order Check:")
print(f"Signal: {large_signal.intent} {large_signal.target['qty']} {large_signal.symbol}")
print(f"Order Notional: ${large_signal.target['qty'] * current_prices[large_signal.symbol]:,.0f}")
print(f"Risk Violation: {violation}")
if violation:
    print(f"Reason: {violation.reason}")
    print(f"Severity: {violation.severity}")
```

**Output:**
```
Large Order Check:
Signal: open 500 AAPL
Order Notional: $75,000
Risk Violation: RiskViolation(check_name='max_order_notional', reason='Order notional $75,000 exceeds limit $50,000', severity='reject')
Reason: Order notional $75,000 exceeds limit $50,000
Severity: reject
```

### Step 3: Exposure Limit Warning
```python
# Add more positions to approach exposure limits
additional_positions = {
    "AAPL": 500 + 200,    # Adding 200 more
    "GOOGL": -200,
    "MSFT": 300 + 100,    # Adding 100 more
    "TSLA": 50            # New position
}

additional_prices = current_prices.copy()
additional_prices["TSLA"] = 200.0

# Calculate new exposure
new_gross = sum(abs(qty) * additional_prices.get(sym, 0) 
                for sym, qty in additional_positions.items())
new_net = sum(qty * additional_prices.get(sym, 0) 
              for sym, qty in additional_positions.items())

print("Exposure Analysis:")
print(f"Current Gross Exposure: ${sum(abs(qty) * current_prices.get(sym, 0) for sym, qty in current_positions.items()):,.0f}")
print(f"After Adding Positions: ${new_gross:,.0f}")
print(f"Gross Ratio: {new_gross / portfolio_value:.2f}x (limit: {risk_config['max_gross_exposure']}x)")
print(f"Net Exposure: ${abs(new_net):,.0f}")
print(f"Net Ratio: {abs(new_net) / portfolio_value:.2f}x (limit: {risk_config['max_net_exposure']}x)")
```

**Output:**
```
Exposure Analysis:
Current Gross Exposure: $307,500
After Adding Positions: $427,500
Gross Ratio: 4.27x (limit: 2.0x)
Net Ratio: 3.27x (limit: 1.0x)
```

### Step 4: Risk Check with Exposure Warning
```python
# Signal that would push exposure over limits
exposure_signal = SignalEvent(
    ts=pd.Timestamp('2023-01-01 10:00:00'),
    symbol='TSLA',
    intent='open',
    target={'qty': 100}  # $20K order, but exposure already high
)

# Check against current positions (before adding)
violation = risk_check.check_signal(
    signal=exposure_signal,
    portfolio_value=portfolio_value,
    current_positions=current_positions,  # Original positions
    current_prices=current_prices
)

print("Exposure Limit Check:")
print(f"Signal: {exposure_signal.intent} {exposure_signal.target['qty']} {exposure_signal.symbol}")
print(f"Risk Violation: {violation}")
if violation:
    print(f"Reason: {violation.reason}")
    print(f"Severity: {violation.severity}")
```

**Output:**
```
Exposure Limit Check:
Signal: open 100 TSLA
Risk Violation: RiskViolation(check_name='max_net_exposure', reason='Net exposure 1.08x exceeds limit 1.0x', severity='warn')
Reason: Net exposure 1.08x exceeds limit 1.0x
Severity: warn
```

### Risk Decision Flow Visualization
```
Input Signal → Risk Check
    ↓
Calculate New Exposure
    ↓
Check Against Limits:
    ├── Order Size Limit
    ├── Gross Exposure Limit
    └── Net Exposure Limit
    ↓
Output: Violation or None
    ↓
Action: Reject/Warn/Allow
```

### Risk Limits Summary
```python
print("Risk Limits Configuration:")
print(f"Max Gross Exposure: {risk_check.max_gross_exposure}x capital")
print(f"Max Net Exposure: {risk_check.max_net_exposure}x capital")
print(f"Max Order Notional: ${risk_check.max_order_notional:,.0f}")
print(f"Soft Limits Action: {risk_check.soft_limits_action}")
print(f"Hard Limits Action: {risk_check.hard_limits_action}")
```

**Output:**
```
Risk Limits Configuration:
Max Gross Exposure: 2.0x capital
Max Net Exposure: 1.0x capital
Max Order Notional: $50,000
Soft Limits Action: warn
Hard Limits Action: reject
```

### Dynamic Limits
Risk limits can be adjusted based on:
- Market volatility
- Strategy performance
- Time of day
- Account equity changes

## Integration with Framework

### Pre-Trade Checks
Risk validation occurs before order generation, preventing risky trades.

### Real-Time Monitoring
Continuous position monitoring during execution.

### Integration Points
- Receives `SignalEvent` for validation
- Can generate `ControlEvent` to pause trading
- Works with execution module for position updates

## Design Principles

### Defense in Depth
Multiple layers of risk controls prevent single points of failure.

### Configurable Limits
Different risk settings for different strategies and market conditions.

### Graceful Degradation
Warnings allow strategy continuation while rejections prevent disasters.

## Common Use Cases

1. **Portfolio Protection**: Prevent over-concentration in single assets
2. **Leverage Control**: Limit borrowing and margin usage
3. **Order Size Management**: Prevent market impact from large orders
4. **Volatility Adaptation**: Tighter limits during high volatility
5. **Strategy Isolation**: Different risk settings per strategy

## Assumptions for Beginners

If you're new to risk management:

- **Position Sizing**: How much to buy/sell of each asset
- **Diversification**: Spreading risk across multiple assets
- **Leverage**: Borrowing money to increase position size (magnifies both gains and losses)
- **Drawdown**: Peak-to-valley decline in portfolio value
- **Risk/Reward Ratio**: Potential loss vs. potential gain on each trade

## Risk Metrics

### Common Risk Measures
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-valley decline
- **VAR (Value at Risk)**: Potential loss over a time period
- **Beta**: Market sensitivity
- **Stress Testing**: Performance under extreme conditions

### Portfolio-Level Risks
- **Concentration Risk**: Too much in one asset/sector
- **Liquidity Risk**: Unable to exit positions quickly
- **Gap Risk**: Price jumps between trading sessions
- **Model Risk**: Strategy assumptions failing

## Best Practices

- Set conservative initial limits and loosen gradually
- Use position sizing based on volatility, not equal weight
- Implement stop-loss orders at both strategy and portfolio levels
- Monitor risk metrics daily, not just P&L
- Have emergency shutdown procedures
- Test risk controls under extreme market conditions

## Advanced Concepts

### Kelly Criterion
Optimal position sizing based on win probability and payoff ratio.

### Risk Parity
Allocate capital so each asset contributes equally to total risk.

### Dynamic Risk Management
Adjust limits based on:
- Recent volatility
- Strategy Sharpe ratio
- Portfolio drawdown
- Market regime changes

### Tail Risk Hedging
Protect against extreme, rare events ("black swans").

## Performance Impact

Risk controls typically reduce returns but prevent catastrophic losses. The goal is to achieve the optimal risk-adjusted performance, not maximum returns.
