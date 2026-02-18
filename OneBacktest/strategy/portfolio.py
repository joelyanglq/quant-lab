from typing import Dict, List
import queue

import pandas as pd

from data.types import Bar
from event import EventType, FillEvent, OrderSide


class Portfolio:
    """
    组合管理（只记账）

    跟踪持仓、市值、P&L。不生成订单。
    """

    def __init__(self, symbols: List[str], latest_prices: Dict[str, Bar],
                 initial_capital=100000.0):
        self.symbols = symbols
        self.latest_prices = latest_prices
        self.events: queue.Queue = None  # 由 Engine 注入
        self.initial_capital = initial_capital

        self.current_positions = dict()
        self.current_holdings = self._init_holdings()

        self.all_holdings: List[dict] = []
        self.trade_log: List[FillEvent] = []

    def _init_holdings(self):
        d = {s: 0.0 for s in self.symbols}
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_market(self, bar: Bar):
        """根据最新 bar 更新组合市值"""
        total_market_value = 0.0
        for s in self.symbols:
            latest = self.latest_prices.get(s)
            if latest is not None:
                market_value = self.current_positions.get(s, 0) * latest.close
                self.current_holdings[s] = market_value
                total_market_value += market_value

        self.current_holdings['total'] = self.current_holdings['cash'] + total_market_value

        snapshot = self.current_holdings.copy()
        snapshot['timestamp'] = bar.timestamp
        self.all_holdings.append(snapshot)

    def update_fill(self, event: FillEvent):
        if event.event_type == EventType.FILL:
            self._update_positions(event)
            self._update_holdings(event)
            self.trade_log.append(event)

    def _update_positions(self, fill: FillEvent):
        fill_dir = 1 if fill.side == OrderSide.BUY else -1
        self.current_positions[fill.symbol] = (
            self.current_positions.get(fill.symbol, 0) + fill_dir * fill.fill_quantity
        )

    def _update_holdings(self, fill: FillEvent):
        fill_dir = 1 if fill.side == OrderSide.BUY else -1
        latest = self.latest_prices.get(fill.symbol)
        fill_cost = latest.close if latest else 0.0
        cost = fill_dir * fill_cost * fill.fill_quantity

        self.current_holdings[fill.symbol] += cost
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] -= (cost + fill.commission)
        self.current_holdings['total'] -= fill.commission

    def get_equity_curve(self) -> pd.DataFrame:
        """返回权益曲线 DataFrame（index=timestamp, columns 含 total/cash 等）"""
        if not self.all_holdings:
            return pd.DataFrame()
        df = pd.DataFrame(self.all_holdings)
        df = df.set_index('timestamp')
        # 同一时间戳可能有多条（多 symbol），取最后一条
        df = df[~df.index.duplicated(keep='last')]
        return df
