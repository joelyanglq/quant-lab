# Portfolio Module

## Overview

The `portfolio` module manages investment positions, cash balances, and overall account state. This is the "book of record" for the trading system, tracking what assets are owned, cash available, and unrealized gains/losses. The module is currently under development but will provide comprehensive portfolio accounting and position management.

## Planned Components

### Position Tracking
- Real-time position updates from fills
- Average cost basis calculation
- Unrealized P&L computation
- Position reconciliation

### Cash Management
- Cash balance tracking
- Margin and borrowing calculations
- Cash flow attribution (dividends, fees, etc.)
- Currency handling for multi-currency portfolios

### Account Valuation
- Total portfolio value calculation
- Asset allocation reporting
- Sector/industry breakdowns
- Performance attribution by position

### Transaction History
- Complete trade log with timestamps
- Position-level audit trail
- Tax lot tracking for cost basis
- Corporate action processing

## Design Principles

### Double-Entry Accounting
Every transaction has equal debit/credit impact.

### Real-Time Updates
Portfolio state updated immediately on fills.

### Multi-Asset Support
Equities, options, futures, forex, etc.

## Integration with Framework

### Receives Events:
- `FillEvent`: Position updates from executions
- `AccountEvent`: External cash flows or adjustments
- `MarketEvent`: Mark-to-market pricing

### Generates Events:
- `SignalEvent`: Rebalancing signals
- `OrderEvent`: Risk reduction orders
- `AccountEvent`: Status updates

## Assumptions for Beginners

If you're new to portfolio management:

- **Positions**: What stocks/bonds you own and how many
- **Cost Basis**: What you paid for your holdings (important for taxes)
- **Unrealized P&L**: Paper gains/losses on current holdings
- **Mark-to-Market**: Revaluing positions at current prices
- **Asset Allocation**: How your money is distributed across different investments

## Key Concepts

### Long vs. Short Positions
- **Long**: You own the asset (betting on price increase)
- **Short**: You borrowed and sold the asset (betting on price decrease)

### Margin Trading
- Using borrowed money to increase position size
- Requires maintenance margin to avoid forced liquidation

### Portfolio Turnover
- How frequently positions are bought/sold
- Affects transaction costs and tax efficiency

## Future Development

This module will be implemented alongside strategy development to provide accurate position tracking and performance measurement.
