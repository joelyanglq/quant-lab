"""IBKR broker wrapper — connection, market data, order placement."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from ib_insync import IB, ComboLeg, Contract, Index, LimitOrder, MarketOrder, Option

log = logging.getLogger(__name__)


@dataclass
class CondorLegs:
    expiry: str            # YYYYMMDD
    short_put: float
    long_put: float
    short_call: float
    long_call: float
    short_put_con: Optional[Option] = None
    long_put_con: Optional[Option] = None
    short_call_con: Optional[Option] = None
    long_call_con: Optional[Option] = None


@dataclass
class FillResult:
    """Returned by broker after waiting for fill."""
    filled: bool              # True if order was fully filled
    avg_price: float          # average fill price (0.0 if not filled)
    total_commission: float   # sum of commission reports (0.0 if unavailable)
    filled_qty: int           # actual filled quantity
    status: str               # ib_insync order status string


class Broker:
    """Duck-typed: accepts any config with host/port/client_id/symbol/exchange/currency."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ib = IB()

    # -- connection -----------------------------------------------------------
    def connect(self) -> bool:
        try:
            self.ib.connect(
                self.cfg.host, self.cfg.port, clientId=self.cfg.client_id,
                timeout=10, readonly=False,
            )
            log.info("Connected to IBKR  port=%s  clientId=%s",
                     self.cfg.port, self.cfg.client_id)
            return True
        except Exception as e:
            log.error("IBKR connection failed: %s", e)
            return False

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()
            log.info("Disconnected from IBKR")

    # -- market data ----------------------------------------------------------
    def get_vix(self) -> tuple[float, float]:
        """Return (current_price, prev_close) for VIX."""
        vix = Index("VIX", "CBOE")
        self.ib.qualifyContracts(vix)
        self.ib.reqMktData(vix, "", False, False)
        self.ib.sleep(2)
        ticker = self.ib.ticker(vix)
        current = ticker.marketPrice()
        prev_close = ticker.close
        self.ib.cancelMktData(vix)
        return current, prev_close

    def get_underlying_price(self) -> float:
        """Return current price of the underlying."""
        und = Index(self.cfg.symbol, self.cfg.exchange)
        self.ib.qualifyContracts(und)
        self.ib.reqMktData(und, "", False, False)
        self.ib.sleep(2)
        ticker = self.ib.ticker(und)
        price = ticker.marketPrice()
        self.ib.cancelMktData(und)
        return price

    # -- option chain ---------------------------------------------------------
    def get_chain_expirations(self, trading_class: str = None) -> list[str]:
        """Return sorted expiration strings (YYYYMMDD).

        trading_class: override filter (e.g. 'SPXW' for 0DTE weeklys).
        """
        und = Index(self.cfg.symbol, self.cfg.exchange)
        self.ib.qualifyContracts(und)
        chains = self.ib.reqSecDefOptParams(
            und.symbol, "", und.secType, und.conId,
        )
        tc = trading_class or self.cfg.symbol
        for ch in chains:
            if ch.tradingClass == tc and ch.exchange == "SMART":
                return sorted(ch.expirations)
        # fallback
        if chains:
            return sorted(chains[0].expirations)
        return []

    def get_option_delta(self, contracts: list[Option]) -> dict[int, float]:
        """Request greeks for a batch of option contracts.

        Returns {conId: delta}.
        """
        self.ib.qualifyContracts(*contracts)
        tickers = [self.ib.reqMktData(c, "", False, False) for c in contracts]

        for _ in range(16):
            self.ib.sleep(0.5)
            if all(t.modelGreeks for t in tickers):
                break

        result: dict[int, float] = {}
        for t in tickers:
            if t.modelGreeks and t.modelGreeks.delta is not None:
                result[t.contract.conId] = t.modelGreeks.delta
            self.ib.cancelMktData(t.contract)
        return result

    def get_combo_mid_price(self, legs: CondorLegs) -> Optional[float]:
        """Return current debit-to-close for the condor (positive = pay to close).

        Fetches bid/ask for all 4 legs, computes cost to reverse the position.
        Returns None if quotes unavailable.
        """
        contracts = [
            legs.short_put_con, legs.long_put_con,
            legs.short_call_con, legs.long_call_con,
        ]
        actions = ["short", "long", "short", "long"]
        self.ib.qualifyContracts(*[c for c in contracts if c])
        tickers = [self.ib.reqMktData(c, "", False, False) for c in contracts]
        self.ib.sleep(3)

        total = 0.0
        for t, action in zip(tickers, actions):
            bid, ask = t.bid, t.ask
            self.ib.cancelMktData(t.contract)
            if bid is None or ask is None or bid <= 0 or ask <= 0:
                return None
            mid = (bid + ask) / 2.0
            # to close: buy back shorts (pay), sell longs (receive)
            total += mid if action == "short" else -mid

        return total

    # -- order placement ------------------------------------------------------
    def place_condor(self, legs: CondorLegs, qty: int,
                     credit_limit: float,
                     fill_timeout: float = 30.0) -> FillResult:
        """Place Iron Condor as a 4-leg combo limit order, wait for fill."""
        combo = Contract()
        combo.symbol = self.cfg.symbol
        combo.secType = "BAG"
        combo.currency = self.cfg.currency
        combo.exchange = "SMART"

        combo.comboLegs = [
            ComboLeg(conId=legs.short_put_con.conId,
                     ratio=1, action="SELL", exchange="SMART"),
            ComboLeg(conId=legs.long_put_con.conId,
                     ratio=1, action="BUY", exchange="SMART"),
            ComboLeg(conId=legs.short_call_con.conId,
                     ratio=1, action="SELL", exchange="SMART"),
            ComboLeg(conId=legs.long_call_con.conId,
                     ratio=1, action="BUY", exchange="SMART"),
        ]

        order = LimitOrder("BUY", qty, -credit_limit)
        order.tif = "GTC"
        trade = self.ib.placeOrder(combo, order)
        log.info("Placed condor  credit=%.2f  qty=%d  expiry=%s",
                 credit_limit, qty, legs.expiry)
        return self._wait_for_fill(trade, fill_timeout)

    def close_condor(self, legs: CondorLegs, qty: int,
                     use_market: bool = False,
                     fill_timeout: float = 30.0) -> FillResult:
        """Close (reverse) an existing Iron Condor position, wait for fill."""
        combo = Contract()
        combo.symbol = self.cfg.symbol
        combo.secType = "BAG"
        combo.currency = self.cfg.currency
        combo.exchange = "SMART"

        combo.comboLegs = [
            ComboLeg(conId=legs.short_put_con.conId,
                     ratio=1, action="BUY", exchange="SMART"),
            ComboLeg(conId=legs.long_put_con.conId,
                     ratio=1, action="SELL", exchange="SMART"),
            ComboLeg(conId=legs.short_call_con.conId,
                     ratio=1, action="BUY", exchange="SMART"),
            ComboLeg(conId=legs.long_call_con.conId,
                     ratio=1, action="SELL", exchange="SMART"),
        ]

        if use_market:
            order = MarketOrder("BUY", qty)
        else:
            order = LimitOrder("BUY", qty, 20.0)
        order.tif = "GTC"
        trade = self.ib.placeOrder(combo, order)
        log.info("Closing condor  qty=%d  expiry=%s  market=%s",
                 qty, legs.expiry, use_market)
        return self._wait_for_fill(trade, fill_timeout)

    def _wait_for_fill(self, trade, timeout: float) -> FillResult:
        """Block until order is filled, cancelled, or timeout expires."""
        elapsed = 0.0
        poll = 0.5

        while elapsed < timeout:
            self.ib.sleep(poll)
            elapsed += poll
            status = trade.orderStatus.status
            if status == "Filled":
                break
            if status in ("Cancelled", "Inactive"):
                log.warning("Order status=%s", status)
                break

        status = trade.orderStatus.status

        if status != "Filled":
            log.warning("Fill timeout after %.1fs  status=%s  filled=%d/%d",
                        elapsed, status,
                        trade.orderStatus.filled, trade.order.totalQuantity)
            self.ib.cancelOrder(trade.order)

        # extra sleep to let commission reports arrive
        if status == "Filled":
            self.ib.sleep(1.0)

        avg_price = 0.0
        total_commission = 0.0
        filled_qty = int(trade.orderStatus.filled)

        if trade.fills:
            avg_price = trade.fills[-1].execution.avgPrice
            for fill in trade.fills:
                cr = fill.commissionReport
                if cr and cr.commission is not None and cr.commission < 1e9:
                    total_commission += cr.commission

        return FillResult(
            filled=(status == "Filled"),
            avg_price=avg_price,
            total_commission=total_commission,
            filled_qty=filled_qty,
            status=status,
        )

    # -- position query -------------------------------------------------------
    def get_option_positions(self) -> list:
        return [p for p in self.ib.positions()
                if p.contract.secType == "OPT"
                and p.contract.symbol == self.cfg.symbol]

    def get_account_value(self, tag: str = "NetLiquidation") -> float:
        vals = self.ib.accountValues()
        for v in vals:
            if v.tag == tag and v.currency == self.cfg.currency:
                return float(v.value)
        return 0.0
