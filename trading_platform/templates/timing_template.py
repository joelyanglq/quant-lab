"""模板：单票择时策略 (Single-Name Timing)
================================================
适用：你想为一组股票（按 symbol 独立）开发一条新的择时规则。

如何使用：
    1. 复制本文件到 trading_platform/strategies_user/<your_name>.py（自己建目录）
    2. 改类名 MyTimingAlpha -> 自己的名字
    3. 在 _compute_raw_for_symbol() 里实现你的规则（输入是单 symbol 的 close 序列）
    4. 在 runtime/backtest.py 里用你的 Alpha 替换 SingleNameTimingAlpha 即可

关键约束（forecast 协议）：
    - 返回值必须是 dict[symbol -> float in [-20, +20] or NaN]
    - 不依赖价格量级、波动率、资金量
    - 用 ScalingMixin 自动把 raw 信号映射到 E[|f|] ≈ 10
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_platform.core.events import Frequency
from trading_platform.strategy.alpha import Alpha, ScalingMixin


class MyTimingAlpha(Alpha, ScalingMixin):
    """举例：基于 20/60 日均线斜率比的择时规则。

    raw = slope_20 - slope_60 (per-symbol)
    -> 短期上升斜率 > 长期，看多
    """

    trigger_freq = Frequency.EOD
    min_history = 80     # 至少需要 80 天历史
    strategy_id = "my_timing"

    def __init__(self, symbols: list[str], short_window: int = 20, long_window: int = 60):
        super().__init__()
        # IMPORTANT: scaling 在 252 天滚动窗口内学习
        self._init_scaling(window=252)
        self.symbols = list(symbols)
        self.short_window = short_window
        self.long_window = long_window

    def universe(self, dt, ctx):
        return list(self.symbols)

    def _compute_raw_for_symbol(self, close: pd.Series) -> float:
        """单 symbol 的原始信号——任意尺度，scaling 会自动校准。

        Args:
            close: 单 symbol 的 close 序列 (含至少 min_history 长度)

        Returns:
            float — 任意尺度，正 = 看多，负 = 看空，NaN = 不出信号
        """
        if len(close) < self.long_window + 5:
            return float("nan")
        # 拟合最后 short_window / long_window 个点的线性斜率
        x = np.arange(self.short_window)
        slope_short = np.polyfit(x, close.tail(self.short_window).values, 1)[0]
        x = np.arange(self.long_window)
        slope_long = np.polyfit(x, close.tail(self.long_window).values, 1)[0]
        # 用价格归一化，去掉量级
        norm = close.iloc[-1]
        if norm <= 0:
            return float("nan")
        raw = (slope_short - slope_long) / norm * 1000.0
        return float(raw)

    # ── Alpha ABC 实现 ──────────────────────────────────────────────
    def forecast(self, dt: pd.Timestamp, ctx) -> dict[str, float]:
        prices = ctx.as_of(dt, "price_1d", lookback=self.long_window * 4, symbols=self.symbols)
        if prices is None or prices.empty:
            return {s: float("nan") for s in self.symbols}

        raw = {}
        for sym in self.symbols:
            if sym not in prices.columns:
                raw[sym] = float("nan")
                continue
            series = prices[sym].dropna()
            if len(series) < self.min_history:
                raw[sym] = float("nan")
                continue
            raw[sym] = self._compute_raw_for_symbol(series)
        # 自动 scaling + ±20 cap
        return self._scale_and_cap(raw)


# ── 独立运行示例（直接 python timing_template.py 看一次回测） ─────────────
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
    SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOG"]
    START = pd.Timestamp("2022-01-01")
    END = pd.Timestamp("2024-12-31")

    ctx = BacktestDataContext(DATA_ROOT)
    feed = BacktestFeed(DATA_ROOT, SYMBOLS, START, END, Frequency.EOD)
    portfolio = Portfolio(initial_capital=100_000.0)
    execution = SimulatedExecutionHandler()
    execution.on_fill(portfolio.on_fill)

    alpha = MyTimingAlpha(SYMBOLS)
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
    print_report(compute_metrics(eq["equity"], is_portfolio=False), name=alpha.strategy_id)
