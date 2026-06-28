"""Test IBKR order execution — shadow / paper / live modes.

Fixed test parameters: AAPL, BUY, 1 share, MKT order.
Shadow test runs offline; paper/live tests require IBKR TWS/Gateway.
"""
from __future__ import annotations

import json
import time

import pandas as pd
import pytest

from trading_platform.core.events import (
    Bar,
    FillEvent,
    Frequency,
    OrderEvent,
    OrderSide,
    OrderType,
)
from trading_platform.execution.shadow import ShadowExecutionHandler
from trading_platform.risk.monitoring import OrderAuditLog
from trading_platform.risk.portfolio import Portfolio

try:
    from trading_platform.execution.live_ibkr import LiveExecutionHandler
    HAS_IB = True
except ImportError:
    HAS_IB = False


def _make_order(
    side: OrderSide = OrderSide.BUY,
) -> OrderEvent:
    return OrderEvent(
        timestamp=pd.Timestamp.utcnow(),
        symbol="AAPL",
        side=side,
        quantity=1,
        order_type=OrderType.MKT,
        strategy_id="test_order",
    )


# ── Shadow mode (offline, no IBKR needed) ──────────────────────────────


class TestShadowOrder:

    def test_shadow_order_produces_log(self, tmp_path):
        """Submit order -> feed bar -> verify ledger entries + positions + fill callback."""
        ledger = tmp_path / "shadow.jsonl"
        handler = ShadowExecutionHandler(ledger_path=ledger)

        fills: list[FillEvent] = []
        handler.on_fill(fills.append)

        order = _make_order()
        order_id = handler.submit_order(order)
        assert order_id is not None
        assert len(order_id) > 0

        assert handler.get_open_orders(), "order should be pending before fill"

        bar = Bar(
            symbol="AAPL",
            timestamp=pd.Timestamp("2025-01-02 16:00:00", tz="UTC"),
            frequency=Frequency.EOD,
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1e6,
        )
        handler.process_bar(bar)

        assert ledger.exists(), "ledger file should be created"
        lines = [json.loads(l) for l in ledger.read_text().strip().splitlines()]

        submitted = [l for l in lines if l["status"] == "submitted"]
        assert len(submitted) == 1
        assert submitted[0]["symbol"] == "AAPL"
        assert submitted[0]["side"] == "BUY"
        assert submitted[0]["qty"] == 1

        filled = [l for l in lines if l["status"] == "filled"]
        assert len(filled) == 1
        assert "fill_price" in filled[0]
        assert filled[0]["fill_price"] > 0

        positions = handler.get_positions()
        assert positions.get("AAPL") == 1, f"Expected 1 AAPL share, got {positions}"

        assert len(fills) == 1
        fill = fills[0]
        assert fill.symbol == "AAPL"
        assert fill.side == OrderSide.BUY
        assert fill.quantity == 1
        assert fill.fill_price > 0

        assert not handler.get_open_orders(), "no pending orders after fill"

    def test_shadow_order_with_portfolio(self, tmp_path):
        """Verify fill callback updates Portfolio positions and equity."""
        ledger = tmp_path / "shadow.jsonl"
        handler = ShadowExecutionHandler(ledger_path=ledger)
        portfolio = Portfolio(initial_capital=100_000.0)
        handler.on_fill(portfolio.on_fill)

        order = _make_order()
        handler.submit_order(order)

        bar = Bar("AAPL", pd.Timestamp("2025-01-02 16:00", tz="UTC"),
                   Frequency.EOD, 150.0, 152.0, 149.0, 151.0, 1e6)
        portfolio.update_market("AAPL", 151.0, bar.timestamp)
        handler.process_bar(bar)

        assert portfolio.positions().get("AAPL", 0) >= 1
        assert portfolio.equity() > 0

    def test_shadow_audit_log(self, tmp_path):
        """Verify OrderAuditLog records submissions alongside shadow ledger."""
        ledger = tmp_path / "shadow.jsonl"
        handler = ShadowExecutionHandler(ledger_path=ledger)
        audit = OrderAuditLog(log_dir=tmp_path / "logs")

        order = _make_order()
        oid = handler.submit_order(order)
        audit.log_order(order, status="submitted")

        bar = Bar("AAPL", pd.Timestamp("2025-01-02 16:00", tz="UTC"),
                   Frequency.EOD, 150.0, 152.0, 149.0, 151.0, 1e6)

        fills: list[FillEvent] = []
        handler.on_fill(fills.append)
        handler.process_bar(bar)

        assert len(fills) == 1
        audit.log_fill(fills[0])

        log_files = list((tmp_path / "logs").glob("orders_*.jsonl"))
        assert len(log_files) == 1
        entries = [json.loads(l) for l in log_files[0].read_text().strip().splitlines()]
        types = [e["type"] for e in entries]
        assert "order" in types
        assert "fill" in types


# ── Paper mode (IBKR paper account on port 7497) ──────────────────────


@pytest.mark.ibkr
@pytest.mark.skipif(not HAS_IB, reason="ib_insync not installed")
class TestPaperOrder:

    def test_paper_order_with_position_snapshot(self, require_ibkr, tmp_path):
        """Submit MKT BUY 1 AAPL on paper account, verify fill + positions."""
        handler = LiveExecutionHandler(port=7497, client_id=60)
        handler.connect()
        audit = OrderAuditLog(log_dir=tmp_path / "logs")

        fills: list[FillEvent] = []
        handler.on_fill(fills.append)

        initial_pos = handler.get_positions().get("AAPL", 0)

        order = _make_order()
        oid = handler.submit_order(order)
        audit.log_order(order, status="submitted")
        assert oid is not None

        deadline = time.time() + 30
        while time.time() < deadline:
            handler.ib.sleep(0.5)
            if fills:
                break
        try:
            assert len(fills) >= 1, "No fill received within 30s"
            fill = fills[0]
            assert fill.symbol == "AAPL"
            assert fill.fill_price > 0
            audit.log_fill(fill)

            log_files = list((tmp_path / "logs").glob("orders_*.jsonl"))
            assert len(log_files) == 1
            entries = [json.loads(l) for l in log_files[0].read_text().strip().splitlines()]
            assert any(e["type"] == "order" for e in entries)
            assert any(e["type"] == "fill" for e in entries)

            current_pos = handler.get_positions().get("AAPL", 0)
            assert current_pos >= initial_pos + 1, (
                f"Position should increase: was {initial_pos}, now {current_pos}"
            )
        finally:
            # cleanup: sell back
            sell_order = _make_order(side=OrderSide.SELL)
            handler.submit_order(sell_order)
            time.sleep(3)
            handler.disconnect()


# ── Live mode (IBKR live account on port 7496, real money) ─────────────


@pytest.mark.ibkr_live
@pytest.mark.skipif(not HAS_IB, reason="ib_insync not installed")
class TestLiveOrder:

    def test_live_order_with_position_snapshot(self, require_ibkr_live, tmp_path):
        """Submit MKT BUY 1 AAPL on live account, verify fill + positions."""
        handler = LiveExecutionHandler(
            port=7496, client_id=61, confirm_live=True,
        )
        handler.connect()
        audit = OrderAuditLog(log_dir=tmp_path / "logs")

        fills: list[FillEvent] = []
        handler.on_fill(fills.append)

        initial_pos = handler.get_positions().get("AAPL", 0)

        order = _make_order()
        oid = handler.submit_order(order)
        audit.log_order(order, status="submitted")
        assert oid is not None

        deadline = time.time() + 30
        while time.time() < deadline:
            handler.ib.sleep(0.5)
            if fills:
                break

        try:
            assert len(fills) >= 1, "No fill received within 30s"
            fill = fills[0]
            assert fill.symbol == "AAPL"
            assert fill.fill_price > 0
            audit.log_fill(fill)

            log_files = list((tmp_path / "logs").glob("orders_*.jsonl"))
            assert len(log_files) == 1
            entries = [json.loads(l) for l in log_files[0].read_text().strip().splitlines()]
            assert any(e["type"] == "order" for e in entries)
            assert any(e["type"] == "fill" for e in entries)

            current_pos = handler.get_positions().get("AAPL", 0)
            assert current_pos >= initial_pos + 1, (
                f"Position should increase: was {initial_pos}, now {current_pos}"
            )
        finally:
            sell_order = _make_order(side=OrderSide.SELL)
            handler.submit_order(sell_order)
            time.sleep(3)
            handler.disconnect()
