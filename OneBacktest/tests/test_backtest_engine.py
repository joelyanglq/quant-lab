import pandas as pd

from backtest.engine import BacktestEngine
from data.feed import DataFeed
from data.types import Bar
from event import OrderType
from execution.handler import SimulatedExecutionHandler
from strategy.base import Strategy
from strategy.portfolio import Portfolio


class DummyFeed(DataFeed):
    def __init__(self, bars):
        self._bars = list(bars)
        self._idx = 0

    def subscribe(self, symbols, start, end):
        return None

    def next(self):
        if self._idx >= len(self._bars):
            return None
        bar = self._bars[self._idx]
        self._idx += 1
        return bar

    def has_next(self):
        return self._idx < len(self._bars)


class BuyOnceStrategy(Strategy):
    def __init__(self):
        self.order_timestamps = []
        self.fill_timestamps = []
        self.market_close_dates = []
        self.week_end_dates = []
        self.month_end_dates = []
        self._ordered = False

    def on_bar(self, bar):
        if not self._ordered:
            self.buy(bar.symbol, 1, order_type=OrderType.MARKET)
            self.order_timestamps.append(bar.timestamp)
            self._ordered = True

    def on_fill(self, fill):
        self.fill_timestamps.append(fill.timestamp)

    def on_market_close(self, dt):
        self.market_close_dates.append(dt)

    def on_week_end(self, dt):
        self.week_end_dates.append(dt)

    def on_month_end(self, dt):
        self.month_end_dates.append(dt)


def _bar(ts, close):
    return Bar(
        timestamp=pd.Timestamp(ts),
        symbol="AAPL",
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100,
    )


def test_engine_executes_orders_on_next_bar_and_fires_aggregate_callbacks():
    bars = [
        _bar("2025-01-31 16:00:00", 100.0),  # Friday month-end
        _bar("2025-02-03 16:00:00", 110.0),  # Monday next week / next month
    ]
    latest_prices = {}
    strategy = BuyOnceStrategy()
    portfolio = Portfolio(["AAPL"], latest_prices, initial_capital=1000.0)
    execution = SimulatedExecutionHandler(latest_prices)
    engine = BacktestEngine(
        data_feed=DummyFeed(bars),
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution,
        latest_prices=latest_prices,
    )

    engine.run_backtest()

    assert strategy.order_timestamps == [pd.Timestamp("2025-01-31 16:00:00")]
    assert strategy.fill_timestamps == [pd.Timestamp("2025-02-03 16:00:00")]
    assert portfolio.current_positions["AAPL"] == 1
    assert portfolio.trade_log[0].fill_price == 110.0

    assert strategy.market_close_dates == [
        pd.Timestamp("2025-01-31").date(),
        pd.Timestamp("2025-02-03").date(),
    ]
    assert strategy.week_end_dates == [
        pd.Timestamp("2025-01-31").date(),
        pd.Timestamp("2025-02-03").date(),
    ]
    assert strategy.month_end_dates == [
        pd.Timestamp("2025-01-31").date(),
        pd.Timestamp("2025-02-03").date(),
    ]
