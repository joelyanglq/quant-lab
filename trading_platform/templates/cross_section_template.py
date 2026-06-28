"""模板：横截面多因子策略 (Cross-Section Multi-Factor)
========================================================
适用：你想加一个新的截面因子到多因子打分系统。

如何使用：
    1. 复制并改类名 MyCrossSectionAlpha -> 自己的名字
    2. 在 _compute_my_factor() 里实现新因子（输入是 close panel，输出 panel）
    3. 在 self.factors 里注册它
    4. 框架自动做：sector-neutral z-score → expanding-window IC → handcrafted 加权 → 合成

关键约束：
    - 因子函数签名: factor_fn(close: DataFrame) -> DataFrame (相同 shape)
    - PIT-safe: 框架已在 ctx.as_of(dt) 层面保证不返未来数据，因子函数不需自己防护
    - 因子方向: 我们假设"高分 = 看多"——如果是反向因子，记得在函数里取负号
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_platform.core.events import Frequency
from trading_platform.strategy.archetypes.cross_section import (
    CrossSectionAlpha,
    DEFAULT_FACTORS,
    factor_momentum_12_1,
    factor_short_reversal,
    factor_low_vol,
)


# ── 自定义因子示例：流动性因子 (Amihud illiquidity 反向) ─────────────────────
def factor_liquidity(close: pd.DataFrame) -> pd.DataFrame:
    """Amihud illiquidity 反向: -mean(|return| / dollar_volume).

    我们没有 dollar volume，简化为 -mean(|return|) 在 60d 窗口内。
    高流动性（小 |return|）-> 高分。
    """
    ret = close.pct_change()
    illiq = ret.abs().rolling(60).mean()
    return -illiq  # 取负: 高流动性 → 高 score


def factor_size_proxy(close: pd.DataFrame) -> pd.DataFrame:
    """Size 因子代理: log(price) — 价格高的可能是大盘股或被拆分前。
    真实使用应该用市值；这里只是模板示意，新策略请 ctx.as_of(dt, 'fundamentals') 拿市值。
    """
    return np.log(close)


# ── 自定义 Alpha 类 ────────────────────────────────────────────────────────
class MyCrossSectionAlpha(CrossSectionAlpha):
    """组合默认因子 + 你自己加的两个因子。"""

    strategy_id = "my_cross_section"

    def __init__(self, **kw):
        my_factors = {
            **DEFAULT_FACTORS,        # momentum_12_1 / short_reversal / low_vol
            "liquidity": factor_liquidity,
            "size_proxy": factor_size_proxy,
        }
        super().__init__(factors=my_factors, **kw)


# ── 独立运行示例 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    from trading_platform.analytics.metrics import compute_metrics, print_report
    from trading_platform.core.context import BacktestDataContext
    from trading_platform.core.engine import Engine
    from trading_platform.data.feed import BacktestFeed
    from trading_platform.execution.simulated import SimulatedExecutionHandler
    from trading_platform.risk.portfolio import Portfolio
    from trading_platform.strategy.combiner import WeightedCombiner
    from trading_platform.strategy.composite import CompositeStrategy
    from trading_platform.strategy.sizer import RiskSizer

    logging.basicConfig(level=logging.INFO)

    DATA_ROOT = "data"
    START = pd.Timestamp("2022-01-01")
    END = pd.Timestamp("2024-12-31")

    ctx = BacktestDataContext(DATA_ROOT)
    universe = ctx.universe(END)
    # 为节省时间，只取前 100 个 symbol
    universe = universe[:100]

    feed = BacktestFeed(DATA_ROOT, universe, START, END, Frequency.EOD)
    portfolio = Portfolio(initial_capital=100_000.0)
    execution = SimulatedExecutionHandler()
    execution.on_fill(portfolio.on_fill)

    alpha = MyCrossSectionAlpha(rebalance_freq="W-FRI", sector_neutral=True)
    composite = CompositeStrategy(
        alphas=[alpha],
        combiner=WeightedCombiner(),
        sizer=RiskSizer(target_vol=0.15, max_leverage=1.0),
        ctx=ctx,
        execution=execution,
        portfolio=portfolio,
        initial_capital=100_000.0,
        trigger_freq=Frequency.EOD,
        strategy_id=alpha.strategy_id,
    )
    Engine([feed], [composite], execution, portfolio).run()

    eq = portfolio.equity_curve()
    print_report(compute_metrics(eq["equity"], is_portfolio=True), name=alpha.strategy_id)
