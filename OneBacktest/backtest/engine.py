import queue
from typing import Dict

from data.feed import DataFeed
from data.types import Bar
from event import EventType


class BacktestEngine:
    """
    回测引擎

    直接从 DataFeed 拉取 Bar，同步分发给 strategy 和 portfolio。
    事件队列仅用于 Signal → Order → Fill 交易链。
    """

    def __init__(self, data_feed, strategy, portfolio, execution_handler,
                 latest_prices: Dict[str, Bar] = None):
        self.data_feed = data_feed
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        self.events = queue.Queue()

        # 共享的最新价格字典（Portfolio/Execution 持有同一引用）
        self.latest_prices = latest_prices if latest_prices is not None else {}

        # 注入 event queue
        self.strategy.events = self.events
        self.portfolio.events = self.events
        self.execution_handler.events = self.events

    def run_backtest(self):
        print("Starting Backtest...")

        while self.data_feed.has_next():
            bar = self.data_feed.next()
            if bar is None:
                break

            # 更新最新价格
            self.latest_prices[bar.symbol] = bar

            # 同步分发
            self.strategy.calculate_signals(bar)
            self.portfolio.update_market(bar)

            # 处理交易事件链: Signal → Order → Fill
            while not self.events.empty():
                event = self.events.get(False)
                if event is None:
                    continue

                if event.event_type == EventType.SIGNAL:
                    self.portfolio.update_signal(event)

                elif event.event_type == EventType.ORDER:
                    self.execution_handler.execute_order(event)

                elif event.event_type == EventType.FILL:
                    self.portfolio.update_fill(event)

        print("Backtest finished.")
