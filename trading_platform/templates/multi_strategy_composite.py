"""模板：多策略组合 (Multi-Strategy Composite)
=================================================
把 4 类策略组合成 Carver Ch5-Ch11 的"乘性三层"投资组合：

    Layer 1 (Stock-Picking): CrossSectionAlpha (多因子选股)
    Layer 2 (Regime / Macro): RotationAlpha (作为 SPY/risk-off 指标)
    Layer 3 (Timing): SingleNameTimingAlpha (HHT)

LayeredCombiner 把三层乘起来; HandcraftedCombiner 处理同层多策略相关性。

如何使用：
    复制本文件，按你的策略列表替换 alphas / groups / combiner，跑 main()。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from trading_platform.analytics.metrics import compute_metrics, print_report
from trading_platform.core.context import BacktestDataContext
from trading_platform.core.engine import Engine
from trading_platform.core.events import Frequency
from trading_platform.data.feed import BacktestFeed
from trading_platform.execution.simulated import SimulatedExecutionHandler
from trading_platform.risk.portfolio import Portfolio
from trading_platform.strategy.archetypes.cross_section import CrossSectionAlpha
from trading_platform.strategy.archetypes.rotation import RotationAlpha, DEFAULT_SECTOR_ETFS
from trading_platform.strategy.archetypes.single_name_timing import SingleNameTimingAlpha
from trading_platform.strategy.combiner import LayeredCombiner, WeightedCombiner
from trading_platform.strategy.composite import CompositeStrategy
from trading_platform.strategy.sizer import RiskSizer

logging.basicConfig(level=logging.INFO)


def main():
    DATA_ROOT = "data"
    START = pd.Timestamp("2020-01-01")
    END = pd.Timestamp("2024-12-31")
    INITIAL = 100_000.0

    ctx = BacktestDataContext(DATA_ROOT)

    # ── 三层 Alpha ──────────────────────────────────────────────────
    universe = ctx.universe(END)[:50]  # 节省时间，前 50

    cs_alpha = CrossSectionAlpha(
        rebalance_freq="W-FRI",
        sector_neutral=True,
        strategy_id="cs_picking",
    )

    # 用 SPY 做"regime"近似——价格上方 200MA -> 正 forecast。
    class SPYRegimeAlpha(SingleNameTimingAlpha):
        strategy_id = "spy_regime"

    spy_alpha = SPYRegimeAlpha(symbols=["SPY"], rule="MA_cross",
                               rule_kwargs={"fast": 50, "slow": 200},
                               strategy_id="spy_regime")

    # 单票择时 (HHT 在选出来的股票上)。简化为对全部 universe 做 HHT。
    timing_alpha = SingleNameTimingAlpha(
        symbols=universe,
        rule="HHT",
        strategy_id="symbol_timing",
    )

    # ── 组合: LayeredCombiner 三组乘性 ──────────────────────────────
    # alphas 列表索引:
    #   0 = cs_alpha (picking)
    #   1 = spy_alpha (regime, 仅 SPY 信号)
    #   2 = timing_alpha (timing on universe)
    alphas = [cs_alpha, spy_alpha, timing_alpha]
    combiner = LayeredCombiner(
        groups=[[0], [1], [2]],
        require_all_groups=False,  # 一个组缺信号时不强制清仓
    )

    # ── Engine + Portfolio + Execution ──────────────────────────────
    # Feed 包含三个 alpha 的 union universe.
    feed_syms = sorted(set(universe) | {"SPY"})
    feed = BacktestFeed(DATA_ROOT, feed_syms, START, END, Frequency.EOD)
    portfolio = Portfolio(initial_capital=INITIAL)
    execution = SimulatedExecutionHandler()
    execution.on_fill(portfolio.on_fill)

    composite = CompositeStrategy(
        alphas=alphas,
        combiner=combiner,
        sizer=RiskSizer(target_vol=0.12, max_leverage=1.0),
        ctx=ctx,
        execution=execution,
        portfolio=portfolio,
        initial_capital=INITIAL,
        trigger_freq=Frequency.EOD,
        strategy_id="multi_strategy",
    )

    Engine([feed], [composite], execution, portfolio).run()
    eq = portfolio.equity_curve()
    if not eq.empty:
        print_report(compute_metrics(eq["equity"], is_portfolio=True), name="multi_strategy")


if __name__ == "__main__":
    main()
