"""Integration test: Strategy signals + Combiner + Sizer + Engine full pipeline."""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from data.context import DataContext
from data.feed import DataFeed
from data.types import Bar
from execution.handler import SimulatedExecutionHandler
from strategy.base import Strategy
from strategy.combiner import WeightedAvgCombiner
from strategy.composite import CompositeStrategy
from strategy.portfolio import Portfolio
from strategy.sizer import VolTargetSizer


# ── Test helpers ──

class DummyFeed(DataFeed):
    def __init__(self, bars):
        self._bars = list(bars)
        self._idx = 0

    def subscribe(self, symbols, start, end):
        pass

    def next(self):
        if self._idx >= len(self._bars):
            return None
        bar = self._bars[self._idx]
        self._idx += 1
        return bar

    def has_next(self):
        return self._idx < len(self._bars)


class AlwaysBullish(Strategy):
    """Constant +0.7 for AAPL."""
    def on_market_close(self, dt):
        return {'AAPL': 0.7}


class AlwaysBearish(Strategy):
    """Constant -0.3 for AAPL."""
    def on_market_close(self, dt):
        return {'AAPL': -0.3}


def _make_bars(n_days=30, start_price=100.0):
    np.random.seed(42)
    bars = []
    price = start_price
    base = pd.Timestamp('2025-01-02')
    for i in range(n_days):
        price *= (1 + np.random.normal(0, 0.01))
        dt = base + pd.Timedelta(days=i)
        # skip weekends
        while dt.weekday() >= 5:
            dt += pd.Timedelta(days=1)
        bars.append(Bar(
            timestamp=dt + pd.Timedelta(hours=16),
            symbol='AAPL',
            open=price * 0.999,
            high=price * 1.005,
            low=price * 0.995,
            close=price,
            volume=10000,
        ))
    return bars


# ── Tests ──

def test_composite_runs():
    bars = _make_bars(30)
    latest = {}
    strategy = CompositeStrategy(
        strategies=[AlwaysBullish()],
        combiner=WeightedAvgCombiner(),
        sizer=VolTargetSizer(target_vol=0.15, vol_lookback=10),
    )
    portfolio = Portfolio(['AAPL'], latest, initial_capital=100_000.0)
    execution = SimulatedExecutionHandler(latest)
    engine = BacktestEngine(
        data_feed=DummyFeed(bars),
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution,
        latest_prices=latest,
    )
    engine.run_backtest()
    assert portfolio.current_positions.get('AAPL', 0) > 0


def test_composite_with_data_context():
    bars = _make_bars(30)
    latest = {}
    ctx = DataContext()
    ctx.register('signal', pd.DataFrame(
        {'AAPL': [0.5]},
        index=pd.to_datetime(['2025-01-01']),
    ))
    strategy = CompositeStrategy(
        strategies=[AlwaysBullish()],
        combiner=WeightedAvgCombiner(),
        sizer=VolTargetSizer(target_vol=0.15, vol_lookback=10),
    )
    portfolio = Portfolio(['AAPL'], latest, initial_capital=100_000.0)
    execution = SimulatedExecutionHandler(latest)
    engine = BacktestEngine(
        data_feed=DummyFeed(bars),
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution,
        latest_prices=latest,
        data_context=ctx,
    )
    engine.run_backtest()
    # Verify injection
    assert strategy.data is ctx
    assert strategy.alphas[0].data is ctx


def test_two_strategies_combined():
    bars = _make_bars(30)
    latest = {}
    # (+0.7 + -0.3) / 2 = +0.2 → mildly bullish
    strategy = CompositeStrategy(
        strategies=[AlwaysBullish(), AlwaysBearish()],
        combiner=WeightedAvgCombiner(),
        sizer=VolTargetSizer(target_vol=0.15, vol_lookback=10),
        weights=[1.0, 1.0],
    )
    portfolio = Portfolio(['AAPL'], latest, initial_capital=100_000.0)
    execution = SimulatedExecutionHandler(latest)
    engine = BacktestEngine(
        data_feed=DummyFeed(bars),
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution,
        latest_prices=latest,
    )
    engine.run_backtest()
    # Combined +0.2 → small long
    assert portfolio.current_positions.get('AAPL', 0) > 0


def test_v1_strategy_backward_compat():
    """v1 Strategy still works unchanged with the engine changes."""
    from strategy.base import Strategy

    class SimpleV1(Strategy):
        def __init__(self):
            self._bought = False

        def on_bar(self, bar):
            if not self._bought:
                self.buy(bar.symbol, 10)
                self._bought = True

    bars = _make_bars(5)
    latest = {}
    strategy = SimpleV1()
    portfolio = Portfolio(['AAPL'], latest, initial_capital=100_000.0)
    execution = SimulatedExecutionHandler(latest)
    engine = BacktestEngine(
        data_feed=DummyFeed(bars),
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution,
        latest_prices=latest,
    )
    engine.run_backtest()
    assert portfolio.current_positions.get('AAPL', 0) == 10


def test_invalid_rebalance_freq():
    with pytest.raises(ValueError, match="rebalance_freq"):
        CompositeStrategy(
            strategies=[AlwaysBullish()],
            combiner=WeightedAvgCombiner(),
            sizer=VolTargetSizer(),
            rebalance_freq='hourly',
        )


def test_weight_length_mismatch():
    with pytest.raises(ValueError, match="weights"):
        CompositeStrategy(
            strategies=[AlwaysBullish()],
            combiner=WeightedAvgCombiner(),
            sizer=VolTargetSizer(),
            weights=[1.0, 2.0],  # 2 weights for 1 strategy
        )
