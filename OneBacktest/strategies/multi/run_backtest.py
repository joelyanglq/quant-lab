"""
多策略组合回测 — CrossSection × Regime × Timing

架构:
    Layer 0 (CrossSectionAlpha):    多因子选股 → top N → forecast ∈ (0, 1]
    Layer 1 (HMMRegimeStrategy):    SPY HMM → 仓位缩放 ∈ [0, 1]
    Layer 2 (IndexTimingStrategy):  SPY HHT + QRS → 择时信号 ∈ [0, 1]

    LayeredCombiner: CrossSection × Regime × Timing
    VolTargetSizer: 波动率目标 15%, 最大杠杆 1.0

执行:
    cd OneBacktest
    python -m strategies.multi.run_backtest
    python -m strategies.multi.run_backtest --top-n 20 --start 2022-01-01
    python -m strategies.multi.run_backtest --no-alpha101 --rebalance weekly
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backtest.analytics import calc_metrics, print_report
from backtest.engine import BacktestEngine
from data import HistoricFeed, ParquetStorage, load_index_symbols
from data.context import DataContext
from execution.handler import SimulatedExecutionHandler
from strategies.multi.cross_section_alpha import CrossSectionAlpha
from strategies.multi.index_timing_strategy import IndexTimingStrategy
from strategies.regime_alloc.hmm.strategy import HMMRegimeStrategy
from strategy.combiner import LayeredCombiner
from strategy.composite import CompositeStrategy
from strategy.portfolio import Portfolio
from strategy.sizer import VolTargetSizer

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / 'data' / 'processed' / 'bars_1d'


def build_data_context(symbols, storage, start, end):
    """Pre-compute fundamental factors and register in DataContext."""
    print('  Loading fundamental factors ...')
    from strategies.cross_section.factors import compute_fundamental_factors

    # Load close panel from storage for fundamental factor computation
    df = storage.load(symbols, start, end, '1d')
    close = df.pivot_table(index=df.index, columns='symbol', values='close')
    close = close.sort_index()

    # Only compute for symbols that have fundamental data
    fund_factors = compute_fundamental_factors(symbols, close)

    ctx = DataContext()
    registered = []
    for name, panel in fund_factors.items():
        if panel.dropna(how='all').empty:
            continue
        ctx.register(name, panel)
        registered.append(name)

    print(f'  Registered {len(registered)} fundamental factors: {registered}')
    return ctx


def build_strategy(top_n=10, use_alpha101=True, rebalance_freq='weekly',
                   target_vol=0.15) -> CompositeStrategy:
    """Assemble the three-layer multi-strategy."""

    cross_section = CrossSectionAlpha(
        top_n=top_n,
        min_history=252,
        use_alpha101=use_alpha101,
    )

    regime = HMMRegimeStrategy(
        index_symbol='SPY',
        n_states=3,
        gamma=6.0,
        min_history=252,
        model_path=str(ROOT / 'output' / 'models' / 'hmm_3states.joblib'),
    )

    timing = IndexTimingStrategy(
        index_symbol='SPY',
        hht_ma=60,
        hht_ht=30,
        qrs_reg_w=18,
        qrs_zscore_w=250,
    )

    combiner = LayeredCombiner(
        groups=[[0], [1], [2]],
        require_all_groups=True,
    )

    sizer = VolTargetSizer(
        target_vol=target_vol,
        vol_lookback=20,
        max_leverage=1.0,
        min_forecast=0.02,
    )

    return CompositeStrategy(
        strategies=[cross_section, regime, timing],
        combiner=combiner,
        sizer=sizer,
        rebalance_freq=rebalance_freq,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Multi-strategy backtest: CrossSection × Regime × Timing')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top stock picks (default: 10)')
    parser.add_argument('--start', type=str, default='2022-01-01',
                        help='Backtest start date')
    parser.add_argument('--end', type=str, default='2026-12-31',
                        help='Backtest end date')
    parser.add_argument('--capital', type=float, default=1_000_000.0,
                        help='Initial capital (default: 1M)')
    parser.add_argument('--no-alpha101', action='store_true',
                        help='Skip alpha101 factors')
    parser.add_argument('--rebalance', type=str, default='weekly',
                        choices=['daily', 'weekly', 'monthly'],
                        help='Rebalance frequency (default: weekly)')
    parser.add_argument('--target-vol', type=float, default=0.15,
                        help='Annual target volatility (default: 0.15)')
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    # ── 1. Load universe ──
    print('[1/4] Loading universe ...')
    symbols = load_index_symbols()
    # SPY must be in the universe for Regime + Timing
    if 'SPY' not in symbols:
        symbols.append('SPY')
    print(f'  Universe: {len(symbols)} symbols (including SPY)')

    # ── 2. Build data feed ──
    print('[2/5] Loading data feed ...')
    if not DATA_DIR.exists():
        print(f'ERROR: Data directory not found: {DATA_DIR}')
        return
    storage = ParquetStorage(str(DATA_DIR))
    feed = HistoricFeed(storage, frequency='1d')
    feed.subscribe(symbols, start, end)

    # ── 3. Build DataContext (fundamental factors) ──
    print('[3/5] Building DataContext ...')
    data_context = build_data_context(symbols, storage, start, end)

    # ── 4. Build strategy ──
    print('[4/5] Building strategy ...')
    strategy = build_strategy(
        top_n=args.top_n,
        use_alpha101=not args.no_alpha101,
        rebalance_freq=args.rebalance,
        target_vol=args.target_vol,
    )
    print(f'  Top-N: {args.top_n} | Rebalance: {args.rebalance} | '
          f'Target Vol: {args.target_vol:.0%}')

    # ── 5. Run backtest ──
    print('[5/5] Running backtest ...')
    latest_prices = {}
    portfolio = Portfolio(symbols, latest_prices, args.capital)
    execution = SimulatedExecutionHandler(latest_prices)
    engine = BacktestEngine(
        data_feed=feed,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution,
        latest_prices=latest_prices,
        history_lookback=504,
        data_context=data_context,
    )
    engine.run_backtest()

    # ── Results ──
    equity = portfolio.get_equity_curve()
    metrics = calc_metrics(equity, portfolio.trade_log, args.capital)
    print_report(metrics)

    # Final positions
    active = {s: q for s, q in portfolio.current_positions.items() if q != 0}
    if active:
        print(f'\nFinal positions ({len(active)} stocks):')
        for sym, qty in sorted(active.items(), key=lambda x: -abs(x[1])):
            bar = latest_prices.get(sym)
            value = qty * bar.close if bar else 0
            print(f'  {sym:<6} {qty:>6} shares  ${value:>12,.0f}')


if __name__ == '__main__':
    main()
