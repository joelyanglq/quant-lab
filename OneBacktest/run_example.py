"""
回测示例：BuyAndHold on AAPL + SPY

演示新架构：Feed → Engine → Strategy/Portfolio/Execution
"""
import sys
import os
import shutil

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import ParquetStorage, HistoricFeed
from backtest.engine import BacktestEngine
from strategy.examples.buy_and_hold import BuyAndHoldStrategy
from strategy.portfolio import Portfolio
from execution.handler import SimulatedExecutionHandler


def generate_demo_data(storage: ParquetStorage):
    """生成演示数据"""
    dates = pd.bdate_range('2023-01-03', '2023-06-30')
    np.random.seed(42)

    for symbol, base_price in [('AAPL', 130.0), ('SPY', 390.0)]:
        close = base_price + np.cumsum(np.random.randn(len(dates)) * 2)
        df = pd.DataFrame({
            'open': close - np.random.rand(len(dates)),
            'high': close + np.abs(np.random.randn(len(dates))) * 1.5,
            'low': close - np.abs(np.random.randn(len(dates))) * 1.5,
            'close': close,
            'volume': np.random.randint(50_000_000, 100_000_000, len(dates)),
        }, index=dates)
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        storage.save(symbol, df, '1d')

    print(f"Generated demo data: {len(dates)} bars per symbol")


def main():
    symbols = ['AAPL', 'SPY']
    data_dir = './demo_data_cache'

    # 准备数据
    storage = ParquetStorage(data_dir)
    generate_demo_data(storage)

    # 创建 Feed
    feed = HistoricFeed(storage, frequency='1d')
    start = pd.Timestamp('2023-01-03')
    end = pd.Timestamp('2023-06-30')
    feed.subscribe(symbols, start, end)

    # latest_prices 由 Engine 维护，Portfolio 和 Execution 共享引用
    latest_prices = {}

    # 创建组件
    strategy = BuyAndHoldStrategy(symbols)
    portfolio = Portfolio(symbols, latest_prices, initial_capital=100000.0)
    execution = SimulatedExecutionHandler(latest_prices)

    # 创建并运行引擎
    engine = BacktestEngine(feed, strategy, portfolio, execution, latest_prices)
    engine.run_backtest()

    # 结果
    total = portfolio.current_holdings['total']
    ret = (total - 100000.0) / 100000.0 * 100
    print(f"\nFinal Portfolio Value: ${total:,.2f}")
    print(f"Return: {ret:.2f}%")

    # 清理
    shutil.rmtree(data_dir)


if __name__ == "__main__":
    main()
