"""Backtest runtime entry point.

Usage:
    python -m trading_platform.runtime.backtest \\
        --start 2020-01-01 --end 2024-12-31 \\
        --strategy timing  --symbols AAPL,MSFT,NVDA  --rule HHT
    python -m trading_platform.runtime.backtest \\
        --strategy cross_section --rebalance W-FRI
    python -m trading_platform.runtime.backtest \\
        --strategy rotation --rebalance M

For multi-strategy composition, use a Python entry script — see
`templates/multi_strategy_composite.py`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..analytics.metrics import compute_metrics, print_report
from ..core.context import BacktestDataContext
from ..core.engine import Engine
from ..core.events import Frequency
from ..data.feed import BacktestFeed
from ..execution.simulated import SimulatedExecutionHandler
from ..risk.portfolio import Portfolio
from ..strategy.archetypes.cross_section import CrossSectionAlpha
from ..strategy.archetypes.pairs import PairsAlpha
from ..strategy.archetypes.rotation import RotationAlpha
from ..strategy.archetypes.single_name_timing import SingleNameTimingAlpha
from ..strategy.combiner import WeightedCombiner
from ..strategy.composite import CompositeStrategy
from ..strategy.sizer import RiskSizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


DEFAULT_DATA_ROOT = Path("data")


def build_alpha(args) -> object:
    if args.strategy == "timing":
        return SingleNameTimingAlpha(
            symbols=args.symbols.split(","),
            rule=args.rule,
            strategy_id=f"timing_{args.rule}",
        )
    if args.strategy == "cross_section":
        return CrossSectionAlpha(
            rebalance_freq=args.rebalance,
            sector_neutral=not args.no_sector_neutral,
        )
    if args.strategy == "pairs":
        return PairsAlpha()
    if args.strategy == "rotation":
        return RotationAlpha(rebalance_freq=args.rebalance)
    raise ValueError(f"Unknown strategy: {args.strategy}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--strategy", required=True,
                    choices=["timing", "cross_section", "pairs", "rotation"])
    ap.add_argument("--symbols", default="AAPL,MSFT,NVDA",
                    help="comma-separated, used for timing")
    ap.add_argument("--rule", default="HHT", help="single-name timing rule")
    ap.add_argument("--rebalance", default="W-FRI",
                    help="rebalance freq for cross_section / rotation")
    ap.add_argument("--no-sector-neutral", action="store_true")
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument("--max-leverage", type=float, default=1.0)
    ap.add_argument("--initial-capital", type=float, default=100_000.0)
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    ctx = BacktestDataContext(args.data_root)
    universe = ctx.universe(end)

    # Determine which symbols to feed.
    if args.strategy == "timing":
        feed_syms = args.symbols.split(",")
    elif args.strategy == "rotation":
        from ..strategy.archetypes.rotation import DEFAULT_SECTOR_ETFS
        feed_syms = DEFAULT_SECTOR_ETFS
    else:
        feed_syms = universe

    feed = BacktestFeed(args.data_root, feed_syms, start, end, Frequency.EOD)
    portfolio = Portfolio(initial_capital=args.initial_capital)
    execution = SimulatedExecutionHandler()
    execution.on_fill(portfolio.on_fill)

    alpha = build_alpha(args)
    sizer = RiskSizer(target_vol=args.target_vol, max_leverage=args.max_leverage)
    composite = CompositeStrategy(
        alphas=[alpha],
        combiner=WeightedCombiner(),
        sizer=sizer,
        ctx=ctx,
        execution=execution,
        portfolio=portfolio,
        initial_capital=args.initial_capital,
        trigger_freq=Frequency.EOD,
        strategy_id=alpha.strategy_id,
    )

    engine = Engine([feed], [composite], execution, portfolio)
    log.info("Running backtest: %s %s..%s", args.strategy, args.start, args.end)
    engine.run()

    eq = portfolio.equity_curve()
    if eq.empty:
        log.warning("No equity curve produced — strategy may not have traded")
        return
    report = compute_metrics(eq["equity"], is_portfolio=(args.strategy != "timing"))
    print_report(report, name=alpha.strategy_id)


if __name__ == "__main__":
    main()
