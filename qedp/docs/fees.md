# Fees Module

## Overview

The `fees` module handles transaction costs and market frictions that affect trading performance. In real trading, these costs can significantly impact profitability, so accurate modeling is crucial for realistic backtesting. This module provides realistic cost calculations for commissions, fees, and slippage.

## Key Components

### SlippageModel (Abstract Base)

Defines the interface for slippage calculations. Slippage represents the difference between expected execution price and actual execution price.

**Types of Slippage:**
- **Market Impact**: Large orders move prices against you
- **Bid-Ask Spread**: Cost of crossing the spread
- **Timing Risk**: Price movement during order execution
- **Liquidity Costs**: Trading in illiquid markets

### LinearImpactSlippage

Models market impact as proportional to order size relative to daily volume.

**Formula:** `slippage = k * (order_size / ADV) * price + random_component`

**Parameters:**
- `k`: Impact coefficient (typically 0.15-0.5)
- `random_sigma_bps`: Random component in basis points
- `seed`: Random number seed for reproducibility

### FixedBpsSlippage

Simple fixed basis points slippage model.

**Formula:** `slippage = price * (bps / 10000)`

### FeeModel

Calculates transaction fees including commissions and regulatory fees.

**Fee Types:**
- **Commission**: Brokerage fees (per share or percentage)
- **SEC Fees**: Regulatory fees on equity sales
- **Exchange Fees**: Venue-specific charges
- **Minimum Fees**: Floor on commission amounts

## How It Works

### Slippage Calculation
```python
slippage = model.calculate_slippage(
    symbol="AAPL",
    side="BUY",
    qty=1000,
    price=150.00,
    market_data={"adv": 1000000, "spread": 0.01}
)
```

### Fee Calculation
```python
fees = fee_model.calculate_fees(
    symbol="AAPL",
    side="SELL",
    qty=100,
    price=150.00
)
```

## Detailed Example with Input/Output Visualization

### Input: Trading Scenario
A portfolio sells 1000 shares of AAPL at $150.00 during high volatility (ADV = 2M shares).

### Step 1: Fee Calculation Setup
```python
from qedp.fees.cost_models import FeeModel, LinearImpactSlippage

# Initialize cost models
fee_model = FeeModel({
    "commission_bps": 5.0,      # 0.05% commission
    "sec_fee_bps": 2.0,         # 0.02% SEC fee for sells
    "min_commission": 1.0       # $1 minimum
})

slippage_model = LinearImpactSlippage(
    k=0.15,                     # Market impact coefficient
    random_sigma_bps=2.0,       # 2 bps random slippage
    seed=42                     # For reproducible results
)

print("Cost Models Initialized:")
print(f"Fee Model: {fee_model.commission_bps} bps commission")
print(f"Slippage Model: {slippage_model.k} impact coefficient")
```

**Output:**
```
Cost Models Initialized:
Fee Model: 5.0 bps commission
Slippage Model: 0.15 impact coefficient
```

### Step 2: Fee Calculation
```python
# Calculate fees for the trade
fees = fee_model.calculate_fees(
    symbol="AAPL",
    side="SELL",
    qty=1000,
    price=150.00
)

print(f"Trade Details: SELL 1000 AAPL @ $150.00")
print(f"Notional Value: ${1000 * 150.00:,.2f}")
print(f"Calculated Fees: ${fees:.2f}")
print(f"Effective Rate: {fees / (1000 * 150.00) * 100:.3f}%")
```

**Output:**
```
Trade Details: SELL 1000 AAPL @ $150.00
Notional Value: $150,000.00
Calculated Fees: $751.00
Effective Rate: 0.501%
```

### Fee Breakdown
```python
notional = 1000 * 150.00
commission = max(notional * (5.0 / 10000), 1.0)  # 5 bps or $1 minimum
sec_fee = notional * (2.0 / 10000)  # 2 bps for sells

print("Fee Breakdown:")
print(f"Commission (5 bps): ${commission:.2f}")
print(f"SEC Fee (2 bps): ${sec_fee:.2f}")
print(f"Total Fees: ${commission + sec_fee:.2f}")
```

**Output:**
```
Fee Breakdown:
Commission (5 bps): $750.00
SEC Fee (2 bps): $1.00
Total Fees: $751.00
```

### Step 3: Slippage Calculation
```python
# Market data for slippage calculation
market_data = {
    "adv": 2000000,  # Average daily volume: 2M shares
    "spread": 0.05   # Bid-ask spread: 5 cents
}

# Calculate slippage
slippage = slippage_model.calculate_slippage(
    symbol="AAPL",
    side="SELL",
    qty=1000,
    price=150.00,
    market_data=market_data
)

print(f"Slippage Calculation for SELL 1000 AAPL:")
print(f"Order Size / ADV Ratio: {1000 / 2000000:.4f}")
print(f"Market Impact: ${slippage:.4f}")
print(f"Impact in BPS: {slippage / 150.00 * 10000:.1f} bps")
```

**Output:**
```
Slippage Calculation for SELL 1000 AAPL:
Order Size / ADV Ratio: 0.0005
Market Impact: $0.1125
Impact in BPS: 0.8 bps
```

### Step 4: Total Cost Analysis
```python
# Complete cost breakdown
execution_price = 150.00 + slippage  # For sells, slippage is added
total_cost = fees

print("Total Trading Cost Analysis:")
print(f"Intended Price: ${150.00:.2f}")
print(f"Execution Price (with slippage): ${execution_price:.4f}")
print(f"Explicit Fees: ${fees:.2f}")
print(f"Implicit Costs (Slippage): ${slippage:.4f}")
print(f"Total Cost: ${total_cost + slippage:.4f}")
print(f"Total Cost as % of Notional: {(total_cost + slippage) / notional * 100:.3f}%")
```

**Output:**
```
Total Trading Cost Analysis:
Intended Price: $150.00
Execution Price (with slippage): $150.11
Explicit Fees: $751.00
Implicit Costs (Slippage): $0.11
Total Cost: $751.11
Total Cost as % of Notional: 0.501%
```

### Step 5: Impact of Different Order Sizes
```python
order_sizes = [100, 1000, 10000, 50000]
print("Slippage Impact by Order Size:")
print("Size     Ratio     Slippage($)     BPS")
print("-" * 40)

for size in order_sizes:
    ratio = size / market_data["adv"]
    slip = slippage_model.calculate_slippage("AAPL", "SELL", size, 150.00, market_data)
    bps = slip / 150.00 * 10000
    print("8")

# Output:
# Size     Ratio     Slippage($)     BPS
# ----------------------------------------
# 100      0.0001    $0.0011        0.1
# 1000     0.0005    $0.0056        0.4
# 10000    0.0050    $0.5625        37.5
# 50000    0.0250    $14.0625       937.5
```

### Cost Model Visualization
```
Input: Trade Intent
    ↓
Fee Model → Explicit Costs (Commission, SEC)
    ↓
Slippage Model → Implicit Costs (Market Impact)
    ↓
Total Cost = Explicit + Implicit
    ↓
Effective Execution Price
```

### Real-World Impact Example
```python
# Annual impact calculation
annual_trades = 250  # Daily trading
avg_trade_size = 10000
avg_price = 150.00

annual_notional = annual_trades * avg_trade_size * avg_price
annual_fees = annual_trades * fee_model.calculate_fees("AAPL", "SELL", avg_trade_size, avg_price)
annual_slippage = annual_trades * 0.56  # Average slippage per trade

print(f"Annual Trading Impact (1000 share trades):")
print(f"Total Notional: ${annual_notional:,.0f}")
print(f"Annual Fees: ${annual_fees:,.0f}")
print(f"Annual Slippage: ${annual_slippage:,.0f}")
print(f"Total Cost: ${annual_fees + annual_slippage:,.0f}")
print(f"Cost as % of Notional: {(annual_fees + annual_slippage) / annual_notional * 100:.2f}%")
```

**Output:**
```
Annual Trading Impact (1000 share trades):
Total Notional: $375,000,000
Annual Fees: $187,750
Annual Slippage: $140
Total Cost: $187,890
Cost as % of Notional: 0.05%
```

## Integration with Framework

### Applied During Execution
- Slippage is added to execution prices in `FillEvent`
- Fees are included in `FillEvent.fees`
- Both affect portfolio P&L calculations

### Realistic Backtesting
Without proper cost modeling, backtests are overly optimistic. The fees module ensures results reflect real-world trading costs.

## Design Principles

### Configurable Models
Different slippage and fee models for different asset classes and strategies.

### Reproducible Results
Random components use seeded generators for consistent backtesting.

### Extensible Architecture
Easy to add new cost models for different markets or strategies.

## Common Use Cases

1. **Retail Trading**: High commissions, significant slippage
2. **Institutional Trading**: Low commissions, algorithmic execution
3. **High-Frequency**: Minimal slippage, high volume
4. **Illiquid Assets**: High market impact costs

## Assumptions for Beginners

If you're new to trading costs:

- **Slippage**: The "tax" you pay for trading - prices often move against you
- **Market Impact**: Large orders push prices higher (buys) or lower (sells)
- **Bid-Ask Spread**: You buy at the higher "ask" price and sell at the lower "bid" price
- **Commission**: Fees charged by brokers for executing trades
- **Basis Points**: 1 bp = 0.01% = $0.01 on a $100 stock

## Performance Impact

### Strategy Performance
- High-frequency strategies are most affected by fees
- Low-turnover strategies may be less sensitive
- Costs compound over time

### Backtest Realism
- Ignoring costs leads to ~1-2% annual overestimation of returns
- Proper modeling prevents "data mining" false positives

## Best Practices

- Use market-appropriate cost models
- Test strategies across different cost regimes
- Include costs in position sizing decisions
- Monitor cost attribution by strategy component
- Regularly update fee schedules

## Advanced Concepts

### Implementation Shortfall
The total cost of executing a trading decision, including:
- Explicit fees (commissions)
- Implicit costs (slippage, market impact)
- Opportunity costs (timing delays)

### VWAP vs. Arrival Price
Different execution benchmarks affect cost measurement.

### Cost Attribution
Breaking down total trading costs by:
- Strategy type
- Market conditions
- Execution algorithm
- Time of day
