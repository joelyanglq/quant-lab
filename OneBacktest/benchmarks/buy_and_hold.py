from typing import List

from strategy.base import Strategy
from data.types import Bar


class BuyAndHoldStrategy(Strategy):
    """
    买入持有策略

    第一根 bar 出现时买入 100 股，持有到结束。
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols

    def on_bar(self, bar: Bar):
        if bar.symbol in self.symbols and self.get_position(bar.symbol) == 0:
            self.buy(bar.symbol, 100)
