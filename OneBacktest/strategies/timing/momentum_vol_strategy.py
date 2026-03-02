from collections import deque
from typing import List, Dict
import numpy as np

from strategy.base import Strategy
from data.types import Bar


class MomentumVolSizedStrategy(Strategy):
    """
    动量 + 波动率配仓策略

    - N 日收益率为正时持仓，为负时平仓
    - 仓位大小按波动率反比分配（低波动多配）
    - 定期再平衡
    """

    def __init__(self, symbols: List[str], momentum_window=20,
                 vol_window=60, rebalance_days=20):
        self.symbols = symbols
        self.momentum_window = momentum_window
        self.vol_window = vol_window
        self.rebalance_days = rebalance_days

        self._closes: Dict[str, deque] = {
            s: deque(maxlen=max(momentum_window, vol_window) + 1) for s in symbols
        }
        self._bar_count = 0

    def on_bar(self, bar: Bar):
        s = bar.symbol
        if s not in self._closes:
            return

        self._closes[s].append(bar.close)
        self._bar_count += 1

        # 只在最后一个 symbol 的 bar 到达时做再平衡判断
        if s != self.symbols[-1]:
            return
        if self._bar_count < self.rebalance_days * len(self.symbols):
            return
        if (self._bar_count // len(self.symbols)) % self.rebalance_days != 0:
            return

        self._rebalance()

    def _rebalance(self):
        # 计算每个 symbol 的动量和波动率
        momentum_signals = {}
        volatilities = {}

        for s in self.symbols:
            closes = list(self._closes[s])
            if len(closes) < self.vol_window:
                continue

            # 动量：N 日收益率
            mom = closes[-1] / closes[-self.momentum_window] - 1
            momentum_signals[s] = mom

            # 波动率：日收益率标准差
            log_returns = np.diff(np.log(closes[-self.vol_window:]))
            vol = np.std(log_returns)
            volatilities[s] = vol if vol > 0 else 1e-6

        if not volatilities:
            return

        # 只持有动量为正的标的
        longs = {s: v for s, v in volatilities.items() if momentum_signals.get(s, 0) > 0}

        if not longs:
            # 全部平仓
            for s in self.symbols:
                pos = self.get_position(s)
                if pos > 0:
                    self.sell(s, pos)
            return

        # 波动率反比权重
        inv_vol = {s: 1.0 / v for s, v in longs.items()}
        total_inv = sum(inv_vol.values())
        weights = {s: iv / total_inv for s, iv in inv_vol.items()}

        portfolio_value = self.get_portfolio_value()

        # 调仓
        for s in self.symbols:
            target_weight = weights.get(s, 0.0)
            latest = self.latest_prices.get(s)
            if latest is None:
                continue

            target_qty = int(portfolio_value * target_weight / latest.close)
            current_qty = self.get_position(s)
            diff = target_qty - current_qty

            if diff > 0:
                self.buy(s, diff)
            elif diff < 0:
                self.sell(s, abs(diff))
