"""
回测示例：5 个策略对比（真实 NASDAQ 日线数据）

1. BuyAndHold      - 基准
2. RSI Reversal    - RSI 均值回归
3. Momentum+Vol    - 动量 + 波动率配仓
4. HHT Timing      - Hilbert 变换择时
5. QRS Timing      - 阻力支撑择时

执行模式: T 日信号 → T+1 日收盘价成交
"""
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from data import ParquetStorage, HistoricFeed
from backtest.engine import BacktestEngine
from backtest.analytics import calc_metrics, print_report
from strategy.examples.buy_and_hold import BuyAndHoldStrategy
from strategy.examples.rsi_reversal import RSIReversalStrategy
from strategy.examples.momentum_vol_sized import MomentumVolSizedStrategy
from strategy.examples.hht_timing import HHTTimingStrategy
from strategy.examples.qrs_timing import QRSTimingStrategy
from strategy.portfolio import Portfolio
from execution.handler import SimulatedExecutionHandler


DATA_DIR = r'D:\04_Project\quant-lab\data\processed\bars_1d'
SYMBOLS = ['AVGO', 'AAPL', 'NBIS', 'TSM', 'GOOG', 'COHR']
START = pd.Timestamp('2020-01-02')
END = pd.Timestamp('2026-02-06')
INITIAL_CAPITAL = 100000.0

STRATEGIES = [
    lambda syms: (BuyAndHoldStrategy(syms), "BuyHold"),
    lambda syms: (RSIReversalStrategy(syms, rsi_period=14, oversold=30, overbought=70), "RSI"),
    lambda syms: (MomentumVolSizedStrategy(syms, momentum_window=20, vol_window=60, rebalance_days=20), "Mom+Vol"),
    lambda syms: (HHTTimingStrategy(syms, ma_period=60, ht_period=30, position_frac=0.95), "HHT"),
    lambda syms: (QRSTimingStrategy(syms, regression_window=18, zscore_window=250,
                                     upper_bound=0.7, lower_bound=-0.7, position_frac=0.95), "QRS"),
]


def run_single(strategy, symbols, name: str):
    """运行单个策略并返回 metrics"""
    storage = ParquetStorage(DATA_DIR)
    feed = HistoricFeed(storage, frequency='1d')
    feed.subscribe(symbols, START, END)

    latest_prices = {}
    portfolio = Portfolio(symbols, latest_prices, INITIAL_CAPITAL)
    execution = SimulatedExecutionHandler(latest_prices)
    engine = BacktestEngine(feed, strategy, portfolio, execution, latest_prices)
    engine.run_backtest()

    equity = portfolio.get_equity_curve()
    metrics = calc_metrics(equity, portfolio.trade_log, INITIAL_CAPITAL)
    return metrics


def main():
    print(f"Period: {START.date()} ~ {END.date()}  |  Capital: ${INITIAL_CAPITAL:,.0f}  |  Execution: T+1 close")
    print()

    # 表头
    header = f"{'Symbol':<8}"
    strat_names = []
    for make_strat in STRATEGIES:
        _, name = make_strat(['_'])
        strat_names.append(name)
        header += f" | {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'PF':>6}"
    # 打印策略名行
    name_line = f"{'':8}"
    for name in strat_names:
        name_line += f" | {name:^33}"
    print(name_line)
    print(header)
    print("-" * len(header))

    for symbol in SYMBOLS:
        row = f"{symbol:<8}"
        for make_strat in STRATEGIES:
            strat, name = make_strat([symbol])
            metrics = run_single(strat, [symbol], f"{name}-{symbol}")
            ret = metrics['total_return']
            sharpe = metrics['sharpe_ratio']
            maxdd = metrics['max_drawdown']
            pf = metrics['profit_factor']
            pf_str = f"{pf:6.2f}" if pf < 1000 else "   inf"
            row += f" |  {ret:>7.1%} {sharpe:>7.2f} {maxdd:>8.1%} {pf_str}"
        print(row)

    print()


if __name__ == "__main__":
    main()
