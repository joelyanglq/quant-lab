from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import queue

from data.types import Bar
from event import FillEvent, OrderEvent, OrderType, OrderSide


class Strategy(ABC):
    """
    策略抽象基类 — 生命周期回调 + 下单 API

    回调:
        on_init()              引擎启动时调用一次
        on_bar(bar)            新 bar 到达（必须实现）
        on_fill(fill)          成交回报
        on_market_close(dt)    每日收盘后（所有 symbol bar 处理完毕）
        on_week_end(dt)        每周最后一个交易日收盘后
        on_month_end(dt)       每月最后一个交易日收盘后

    下单:
        buy(symbol, qty)       买入
        sell(symbol, qty)      卖出
        rebalance_to(symbols)  一键调仓到目标持仓

    查询:
        get_position(symbol)     当前持仓数量
        get_portfolio_value()    组合总市值
        get_cash()               可用现金
    """

    events: queue.Queue = None              # Engine 注入
    positions: Dict[str, int] = None        # Engine 注入（Portfolio.current_positions 引用）
    holdings: dict = None                   # Engine 注入（Portfolio.current_holdings 引用）
    latest_prices: Dict[str, Bar] = None    # Engine 注入
    history = None                          # Engine 注入（HistoryManager）

    # ==================== 生命周期回调 ====================

    def on_init(self):
        """引擎启动时调用一次"""
        pass

    @abstractmethod
    def on_bar(self, bar: Bar):
        """新 bar 到达，核心决策逻辑"""
        ...

    def on_fill(self, fill: FillEvent):
        """成交回报"""
        pass

    # ---- 聚合回调（opt-in, 默认不做任何事）----

    def on_market_close(self, dt):
        """每日收盘后, 所有 symbol 的 bar 都处理完毕后触发."""
        pass

    def on_week_end(self, dt):
        """每周最后一个交易日收盘后触发 (在 on_market_close 之后)."""
        pass

    def on_month_end(self, dt):
        """每月最后一个交易日收盘后触发 (在 on_market_close 之后)."""
        pass

    # ==================== 下单 API ====================

    def buy(self, symbol: str, quantity: int,
            order_type: OrderType = OrderType.MARKET,
            price: Optional[float] = None):
        """买入"""
        self._send_order(symbol, OrderSide.BUY, quantity, order_type, price)

    def sell(self, symbol: str, quantity: int,
             order_type: OrderType = OrderType.MARKET,
             price: Optional[float] = None):
        """卖出"""
        self._send_order(symbol, OrderSide.SELL, quantity, order_type, price)

    def rebalance_to(self, target_symbols: List[str],
                     weights: Optional[Dict[str, float]] = None):
        """
        一键调仓到目标持仓.

        Args:
            target_symbols: 目标持仓列表
            weights: {symbol: weight}, 权重加总 ~1.0。None 则等权。
        """
        if not target_symbols:
            for sym, qty in list((self.positions or {}).items()):
                if qty > 0:
                    self.sell(sym, qty)
            return

        target_set = set(target_symbols)

        if weights is None:
            w = 1.0 / len(target_symbols)
            weights = {s: w for s in target_symbols}

        portfolio_value = self.get_portfolio_value()

        # 先卖出不在目标中的持仓
        if self.positions:
            for sym, qty in list(self.positions.items()):
                if qty > 0 and sym not in target_set:
                    self.sell(sym, qty)

        # 调整目标持仓
        for sym in target_symbols:
            latest = (self.latest_prices or {}).get(sym)
            if latest is None or latest.close <= 0:
                continue
            target_value = portfolio_value * weights.get(sym, 0.0)
            target_qty = int(target_value / latest.close)
            current_qty = self.get_position(sym)
            diff = target_qty - current_qty
            if diff > 0:
                self.buy(sym, diff)
            elif diff < 0:
                self.sell(sym, abs(diff))

    def _send_order(self, symbol: str, side: OrderSide, quantity: int,
                    order_type: OrderType, price: Optional[float]):
        latest = self.latest_prices.get(symbol)
        timestamp = latest.timestamp if latest else None
        order = OrderEvent(
            timestamp=timestamp,
            order_id=str(id(object())),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )
        self.events.put(order)

    # ==================== 查询 ====================

    def get_position(self, symbol: str) -> int:
        """查当前持仓数量"""
        if self.positions is None:
            return 0
        return self.positions.get(symbol, 0)

    def get_portfolio_value(self) -> float:
        """查组合总市值（持仓市值 + 现金）"""
        if self.holdings is None:
            return 0.0
        return self.holdings.get('total', 0.0)

    def get_cash(self) -> float:
        """查可用现金"""
        if self.holdings is None:
            return 0.0
        return self.holdings.get('cash', 0.0)
