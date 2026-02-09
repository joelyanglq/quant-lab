import queue
from typing import Dict

from data.feed import DataFeed
from data.types import Bar
from event import EventType


class BacktestEngine:
    """
    回测引擎

    直接从 DataFeed 拉取 Bar，同步分发给 strategy 和 portfolio。
    事件队列用于 Order → Fill 交易链。
    """

    def __init__(self, data_feed, strategy, portfolio, execution_handler,
                 latest_prices: Dict[str, Bar] = None):
        self.data_feed = data_feed
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.events = queue.Queue()

        # 共享的最新价格字典
        self.latest_prices = latest_prices if latest_prices is not None else {}

        # 注入依赖
        self.strategy.events = self.events
        self.strategy.positions = portfolio.current_positions
        self.strategy.holdings = portfolio.current_holdings
        self.strategy.latest_prices = self.latest_prices
        self.portfolio.events = self.events
        self.execution_handler.events = self.events

    def run_backtest(self):
        print("Starting Backtest...")
        self.strategy.on_init()

        while self.data_feed.has_next():
            bar = self.data_feed.next()
            if bar is None:
                break

            # 更新最新价格
            self.latest_prices[bar.symbol] = bar

            # 同步分发
            self.strategy.on_bar(bar)
            self.portfolio.update_market(bar)

            # 处理交易事件链: Order → Fill
            while not self.events.empty():
                event = self.events.get(False)
                if event is None:
                    continue

                if event.event_type == EventType.ORDER:
                    self.execution_handler.execute_order(event)

                elif event.event_type == EventType.FILL:
                    self.portfolio.update_fill(event)
                    self.strategy.on_fill(event)

        print("Backtest finished.")
