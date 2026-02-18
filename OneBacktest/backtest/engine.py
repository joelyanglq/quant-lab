import queue
from datetime import date
from typing import Dict, List, Optional

from data.feed import DataFeed
from data.types import Bar
from event import EventType, OrderEvent
from data.history import HistoryManager


class BacktestEngine:
    """
    回测引擎

    直接从 DataFeed 拉取 Bar，同步分发给 strategy 和 portfolio。
    订单延迟到下一根 bar 执行（T 日信号 → T+1 日成交）。

    聚合回调:
        on_market_close(dt)  每个交易日所有 bar 处理完后
        on_week_end(dt)      每周最后一个交易日
        on_month_end(dt)     每月最后一个交易日
    """

    def __init__(self, data_feed, strategy, portfolio, execution_handler,
                 latest_prices: Dict[str, Bar] = None,
                 history_lookback: int = 504,
                 storage_1min=None):
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

        # HistoryManager: 策略通过 self.history 访问
        history = HistoryManager(portfolio.symbols, history_lookback, storage_1min)
        self.history = history
        self.strategy.history = history

    def run_backtest(self):
        print("Starting Backtest...")
        self.strategy.on_init()

        # 上一轮 on_bar 产生的 pending orders，延迟到下一根 bar 执行
        pending_orders: List[OrderEvent] = []
        current_date: Optional[date] = None

        while self.data_feed.has_next():
            bar = self.data_feed.next()
            if bar is None:
                break

            bar_date = bar.timestamp.date()

            # ── 日期边界检测: 新日期出现 → 前一日结束 ──
            if current_date is not None and bar_date != current_date:
                self._fire_aggregate_callbacks(
                    current_date, bar_date, pending_orders)
            current_date = bar_date

            # 更新最新价格
            self.latest_prices[bar.symbol] = bar

            # ① 执行 pending orders（上一根 bar 产生的，用当前 bar 价格成交）
            remaining = []
            for order in pending_orders:
                if order.symbol == bar.symbol:
                    self.execution_handler.execute_order(order)
                else:
                    remaining.append(order)
            pending_orders = remaining

            # ② 处理 Fill 事件
            while not self.events.empty():
                event = self.events.get(False)
                if event is None:
                    continue
                if event.event_type == EventType.FILL:
                    self.portfolio.update_fill(event)
                    self.strategy.on_fill(event)

            # ③ 更新组合市值
            self.portfolio.update_market(bar)

            # ④ 喂入 HistoryManager
            self.history._on_bar(bar)

            # ⑤ 策略处理当前 bar（可能产生新 order）
            self.strategy.on_bar(bar)

            # ⑥ 收集新产生的 orders 到 pending（下一根 bar 执行）
            while not self.events.empty():
                event = self.events.get(False)
                if event is None:
                    continue
                if event.event_type == EventType.ORDER:
                    pending_orders.append(event)

        # ── 最后一天: 触发聚合回调 ──
        if current_date is not None:
            self._fire_aggregate_callbacks(current_date, None, pending_orders)

        print("Backtest finished.")

    def _fire_aggregate_callbacks(self, completed_date: date,
                                  next_date: Optional[date],
                                  pending_orders: List[OrderEvent]):
        """
        日期边界触发聚合回调, 并收集回调产生的 orders.

        Args:
            completed_date: 刚结束的交易日
            next_date: 下一个交易日 (None = 数据末尾)
            pending_orders: 待执行订单列表 (会被 in-place 追加)
        """
        # 每日必触发
        self.strategy.on_market_close(completed_date)

        # 周末: ISO (year, week) 不同 → 上一日是该周最后交易日
        is_week_end = (
            next_date is None
            or completed_date.isocalendar()[:2] != next_date.isocalendar()[:2]
        )
        if is_week_end:
            self.strategy.on_week_end(completed_date)

        # 月末: (year, month) 不同
        is_month_end = (
            next_date is None
            or (completed_date.year, completed_date.month)
               != (next_date.year, next_date.month)
        )
        if is_month_end:
            self.strategy.on_month_end(completed_date)

        # 收集回调产生的 orders
        while not self.events.empty():
            event = self.events.get(False)
            if event is not None and event.event_type == EventType.ORDER:
                pending_orders.append(event)
