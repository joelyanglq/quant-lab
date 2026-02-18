from collections import deque
from typing import List, Dict
import numpy as np

from strategy.base import Strategy
from data.types import Bar


class RSIReversalStrategy(Strategy):
    """
    RSI 均值回归策略

    RSI < oversold 时买入，RSI > overbought 时卖出。
    """

    def __init__(self, symbols: List[str], rsi_period=14,
                 oversold=30, overbought=70):
        self.symbols = symbols
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

        # 缓存 close 用于计算 RSI
        self._closes: Dict[str, deque] = {
            s: deque(maxlen=rsi_period + 2) for s in symbols
        }

    def _calc_rsi(self, symbol: str) -> float:
        closes = list(self._closes[symbol])
        if len(closes) < self.rsi_period + 1:
            return 50.0  # 数据不足，返回中性值

        deltas = np.diff(closes)[-self.rsi_period:]
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def on_bar(self, bar: Bar):
        s = bar.symbol
        if s not in self._closes:
            return

        self._closes[s].append(bar.close)
        rsi = self._calc_rsi(s)
        pos = self.get_position(s)

        if rsi < self.oversold and pos == 0:
            # 用 10% 资金买入
            cash_for_trade = self.get_portfolio_value() * 0.10
            qty = int(cash_for_trade / bar.close)
            if qty > 0:
                self.buy(s, qty)

        elif rsi > self.overbought and pos > 0:
            self.sell(s, pos)
