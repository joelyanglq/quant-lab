"""
CompositeStrategy -- 组合策略编排器

管理多个子 Strategy, 通过 Combiner 和 Sizer 产出最终订单.
继承自 Strategy, 对 Engine 而言和普通策略没有区别.

    Strategy ×N → ForecastCombiner → VolTargetSizer → buy/sell
"""
from typing import Dict, List, Optional

from strategy.base import Strategy
from strategy.combiner import ForecastCombiner
from strategy.sizer import VolTargetSizer
from data.types import Bar


class CompositeStrategy(Strategy):
    """
    组合策略: 子 Strategy ×N → Combiner → Sizer → buy/sell.

    每个子 Strategy 的 on_market_close(dt) 返回 Dict[str, float]
    作为 forecast, 由 Combiner 合成, Sizer 定仓, 最终差量下单.
    """

    def __init__(self,
                 strategies: List[Strategy],
                 combiner: ForecastCombiner,
                 sizer: VolTargetSizer,
                 weights: Optional[List[float]] = None,
                 rebalance_freq: str = 'daily'):
        """
        Args:
            strategies:      子 Strategy 实例列表 (on_market_close 返回 forecast)
            combiner:        ForecastCombiner 实例
            sizer:           VolTargetSizer 实例
            weights:         各子策略权重 (None → 等权)
            rebalance_freq:  'daily' | 'weekly' | 'monthly'
        """
        self.sub_strategies = strategies
        self.combiner = combiner
        self.sizer = sizer
        self.weights = weights or [1.0] * len(strategies)
        self.rebalance_freq = rebalance_freq

        if len(self.weights) != len(self.sub_strategies):
            raise ValueError(
                f"weights length ({len(self.weights)}) "
                f"!= strategies length ({len(self.sub_strategies)})")
        if rebalance_freq not in ('daily', 'weekly', 'monthly'):
            raise ValueError(
                f"rebalance_freq must be 'daily'|'weekly'|'monthly', "
                f"got '{rebalance_freq}'")

    # ── 向后兼容 ──

    @property
    def alphas(self):
        """Backward compat: .alphas 仍可访问子策略列表."""
        return self.sub_strategies

    # ── Engine 回调 ──

    def on_init(self):
        """注入 data/history/latest_prices 到每个子策略, 然后初始化."""
        for s in self.sub_strategies:
            s.data = getattr(self, 'data', None)
            s.history = self.history
            s.latest_prices = self.latest_prices
            s.on_init()

    def on_bar(self, bar: Bar):
        """转发 bar 到每个子策略 (更新内部状态)."""
        for s in self.sub_strategies:
            s.on_bar(bar)

    def on_market_close(self, dt):
        if self.rebalance_freq == 'daily':
            self._rebalance(dt)

    def on_week_end(self, dt):
        # 转发给子策略 (子策略可能在 on_week_end 更新内部状态)
        for s in self.sub_strategies:
            s.on_week_end(dt)
        if self.rebalance_freq == 'weekly':
            self._rebalance(dt)

    def on_month_end(self, dt):
        # 转发给子策略 (如 HMMRegimeStrategy 在 on_month_end 重拟合)
        for s in self.sub_strategies:
            s.on_month_end(dt)
        if self.rebalance_freq == 'monthly':
            self._rebalance(dt)

    # ── 内部 ──

    def _rebalance(self, dt):
        """收集 forecast → 合成 → 定仓 → 差量下单."""
        # 1. 收集所有子策略的 forecast
        forecasts = []
        for s in self.sub_strategies:
            f = s.on_market_close(dt)
            forecasts.append(f if f is not None else {})

        # 2. 合成
        combined = self.combiner.combine(forecasts, self.weights)

        # 3. 计算目标仓位
        targets = self.sizer.size(
            combined,
            self.history,
            self.get_portfolio_value(),
            self.latest_prices,
        )

        # 4. 差量下单
        self._rebalance_to_targets(targets)

    def _rebalance_to_targets(self, targets: Dict[str, int]):
        """根据目标股数与当前持仓的差异, 生成 buy/sell 订单."""
        all_symbols = set(targets.keys())
        if self.positions:
            all_symbols.update(self.positions.keys())

        for sym in all_symbols:
            target = targets.get(sym, 0)
            current = self.get_position(sym)
            diff = target - current
            if diff > 0:
                self.buy(sym, diff)
            elif diff < 0:
                self.sell(sym, abs(diff))
