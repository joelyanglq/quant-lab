"""模板：板块/风格轮动策略 (Rotation)
=======================================
适用：你想自定义 rotation 信号（比如基于 macro indicator、估值反转、低波因子）。

如何使用：
    1. 复制并改类名 MyRotationAlpha -> 自己的名字
    2. 重写 _compute_signal() 输出 cross-sectional score
    3. 选择 long_only=True 还是 long-short
    4. 框架自动做：z-score 标准化、scaling、ERC 权重（在 sizer 端）

关键约束：
    - signal 越高越看好（z-score 在更高位）
    - 用 ctx.as_of(dt, 'price_1d') 取价格；用 fundamentals / macro 也可以
    - long_only=True 时负 forecast 在 sizer 里被裁剪为 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_platform.core.events import Frequency
from trading_platform.strategy.archetypes.rotation import (
    RotationAlpha,
    DEFAULT_SECTOR_ETFS,
)


class MyRotationAlpha(RotationAlpha):
    """举例：相对强度反转 — 用过去 1 个月最弱的板块作为下个月的多头。"""

    strategy_id = "my_rotation_reversal"

    def __init__(self, lookback_days: int = 21, **kw):
        super().__init__(**kw)
        self.lookback_days = lookback_days

    def forecast(self, dt: pd.Timestamp, ctx) -> dict[str, float]:
        if not self._is_rebalance_day(dt):
            return {}
        prices = ctx.as_of(dt, "price_1d", lookback=self.lookback_days + 30, symbols=self.symbols)
        if prices is None or prices.empty:
            return {s: float("nan") for s in self.symbols}
        if len(prices) < self.lookback_days + 1:
            return {s: float("nan") for s in self.symbols}

        # 1-month return.
        ret = prices[self.symbols].iloc[-1] / prices[self.symbols].iloc[-self.lookback_days - 1] - 1.0

        # 反转: 过去越弱的板块 forecast 越高.
        if ret.std() == 0 or ret.std() != ret.std():
            return {s: float("nan") for s in self.symbols}
        z = -(ret - ret.mean()) / ret.std()  # 注意: 取负 = 反转

        raw = {s: (float(z[s]) if s in z.index and not np.isnan(z[s]) else float("nan"))
               for s in self.symbols}
        if self.long_only:
            raw = {s: max(v, 0.0) if not np.isnan(v) else float("nan") for s, v in raw.items()}
        return self._scale_and_cap(raw)


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
    START = pd.Timestamp("2018-01-01")
    END = pd.Timestamp("2024-12-31")

    ctx = BacktestDataContext(DATA_ROOT)
    feed = BacktestFeed(DATA_ROOT, DEFAULT_SECTOR_ETFS, START, END, Frequency.EOD)
    portfolio = Portfolio(initial_capital=100_000.0)
    execution = SimulatedExecutionHandler()
    execution.on_fill(portfolio.on_fill)

    alpha = MyRotationAlpha(rebalance_freq="M", lookback_days=21, long_only=True)
    composite = CompositeStrategy(
        alphas=[alpha],
        combiner=WeightedCombiner(),
        sizer=RiskSizer(target_vol=0.12, max_leverage=1.0),
        ctx=ctx,
        execution=execution,
        portfolio=portfolio,
        initial_capital=100_000.0,
        trigger_freq=Frequency.EOD,
        strategy_id=alpha.strategy_id,
    )
    Engine([feed], [composite], execution, portfolio).run()
    eq = portfolio.equity_curve()
    if not eq.empty:
        print_report(compute_metrics(eq["equity"], is_portfolio=True), name=alpha.strategy_id)
