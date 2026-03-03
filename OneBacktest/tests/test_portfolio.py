import pandas as pd

from data.types import Bar
from event import FillEvent, OrderSide
from strategy.portfolio import Portfolio


def test_portfolio_updates_positions_cash_and_market_value():
    latest_prices = {
        "AAPL": Bar(
            timestamp=pd.Timestamp("2025-01-02"),
            symbol="AAPL",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1000,
        )
    }
    portfolio = Portfolio(["AAPL"], latest_prices, initial_capital=1000.0)

    buy_fill = FillEvent(
        timestamp=pd.Timestamp("2025-01-02"),
        symbol="AAPL",
        side=OrderSide.BUY,
        fill_price=100.0,
        fill_quantity=2,
        commission=1.0,
    )
    portfolio.update_fill(buy_fill)

    assert portfolio.current_positions["AAPL"] == 2
    assert portfolio.current_holdings["cash"] == 799.0
    assert portfolio.current_holdings["commission"] == 1.0
    assert portfolio.current_holdings["total"] == 999.0

    # Price moves from 100 to 110, total equity should reflect mark-to-market.
    latest_prices["AAPL"] = Bar(
        timestamp=pd.Timestamp("2025-01-03"),
        symbol="AAPL",
        open=109.0,
        high=111.0,
        low=108.0,
        close=110.0,
        volume=1000,
    )
    portfolio.update_market(latest_prices["AAPL"])

    assert portfolio.current_holdings["AAPL"] == 220.0
    assert portfolio.current_holdings["total"] == 1019.0

    curve = portfolio.get_equity_curve()
    assert not curve.empty
    assert curve.iloc[-1]["total"] == 1019.0
