"""模板：协整对策略 (Pairs / Statistical Arbitrage)
=====================================================
适用：你想用自定义信号代替默认的 z-score 反转（比如 Kalman filter spread、
Ornstein-Uhlenbeck 半生命期建模、bid-ask 微观结构等）。

如何使用：
    1. 离线扫描出协整对，写入 data/processed/cointegration_pairs/valid_pairs.parquet:
       python -m trading_platform.runtime.pairs_scanner --start 2020-01-01 --end 2024-01-01
    2. 复制本文件，改类名 MyPairsAlpha -> 自己的名字
    3. 重写 _compute_pair_signal() 用你的信号逻辑
    4. 框架会自动做：spread vol 估计、stop-loss、双边 forecast 输出

关键约束：
    - forecast 必须为对偶符号: spread 高 → 卖 A 买 B → forecast_A 负, forecast_B 正
    - 用 hedge_ratio (来自协整) 决定相对持仓比例
    - 强制 stop_z 触发时返回 0 (清仓)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trading_platform.core.events import Frequency
from trading_platform.strategy.archetypes.pairs import PairsAlpha


class MyPairsAlpha(PairsAlpha):
    """举例：用 EMA 替代 rolling mean 计算 spread z-score。

    EMA 对最近的 mean shift 反应更快，可能减少滞后但更易过度交易。
    """

    strategy_id = "my_pairs_ema"

    def __init__(self, ema_halflife: int = 30, **kw):
        super().__init__(**kw)
        self.ema_halflife = ema_halflife

    def _compute_pair_signal(self, log_pa: pd.Series, log_pb: pd.Series, beta: float) -> float:
        """计算单对的 spread z-score。返回 z（不是 forecast，由 framework 转 forecast）。"""
        spread = log_pa - beta * log_pb
        ema_mean = spread.ewm(halflife=self.ema_halflife).mean().iloc[-1]
        ema_std = spread.ewm(halflife=self.ema_halflife).std().iloc[-1]
        if ema_std == 0 or np.isnan(ema_std):
            return float("nan")
        return float((spread.iloc[-1] - ema_mean) / ema_std)

    def forecast(self, dt: pd.Timestamp, ctx) -> dict[str, float]:
        try:
            pairs = ctx.as_of(dt, "cointegration_pairs")
        except Exception:
            return {}
        if pairs is None or pairs.empty:
            return {}
        pairs = pairs.head(self.max_pairs)

        symbols = sorted(set(pairs["symbol_a"]).union(pairs["symbol_b"]))
        prices = ctx.as_of(dt, "price_1d", lookback=self.spread_window * 4, symbols=symbols)
        if prices is None or prices.empty:
            return {}

        raw: dict[str, float] = {s: 0.0 for s in symbols}
        for _, row in pairs.iterrows():
            a, b = row["symbol_a"], row["symbol_b"]
            beta = float(row.get("hedge_ratio", 1.0))
            if a not in prices.columns or b not in prices.columns:
                continue
            pair = prices[[a, b]].dropna()
            if len(pair) < self.spread_window + 5:
                continue
            log_a = np.log(pair[a])
            log_b = np.log(pair[b])
            z = self._compute_pair_signal(log_a, log_b, beta)
            if np.isnan(z):
                continue

            pair_key = (a, b)
            in_pos = pair_key in self._open_z
            if in_pos and abs(z) > self.stop_z:
                self._open_z.pop(pair_key, None)
                continue
            if in_pos and abs(z) < self.exit_z:
                self._open_z.pop(pair_key, None)
                continue
            if not in_pos and abs(z) >= self.entry_z:
                self._open_z[pair_key] = float(z)

            if pair_key in self._open_z:
                raw[a] = raw.get(a, 0.0) + (-float(z))
                raw[b] = raw.get(b, 0.0) + (float(z) * beta)

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
    START = pd.Timestamp("2024-01-01")
    END = pd.Timestamp("2024-12-31")

    ctx = BacktestDataContext(DATA_ROOT)
    try:
        pairs_df = ctx.as_of(END, "cointegration_pairs")
    except Exception:
        pairs_df = pd.DataFrame()
    if pairs_df.empty:
        print("No cointegration pairs file found. Run pairs_scanner first:")
        print("    python -m trading_platform.runtime.pairs_scanner --start 2020-01-01 --end 2024-01-01")
        raise SystemExit(0)

    syms = sorted(set(pairs_df["symbol_a"]).union(pairs_df["symbol_b"]))
    feed = BacktestFeed(DATA_ROOT, syms, START, END, Frequency.EOD)
    portfolio = Portfolio(initial_capital=100_000.0)
    execution = SimulatedExecutionHandler()
    execution.on_fill(portfolio.on_fill)

    alpha = MyPairsAlpha()
    composite = CompositeStrategy(
        alphas=[alpha],
        combiner=WeightedCombiner(),
        sizer=RiskSizer(target_vol=0.10, max_leverage=1.0),
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
