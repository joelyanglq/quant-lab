import pandas as pd

from backtest.analytics import _calc_trade_pnls, calc_metrics
from event import FillEvent, OrderSide


def test_calc_trade_pnls_fifo_partial_close():
    trade_log = [
        FillEvent(
            timestamp=pd.Timestamp("2025-01-01"),
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=10.0,
            fill_quantity=5,
        ),
        FillEvent(
            timestamp=pd.Timestamp("2025-01-02"),
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=12.0,
            fill_quantity=5,
        ),
        FillEvent(
            timestamp=pd.Timestamp("2025-01-03"),
            symbol="AAPL",
            side=OrderSide.SELL,
            fill_price=11.0,
            fill_quantity=7,
        ),
    ]

    pnls = _calc_trade_pnls(trade_log)
    assert pnls == [5.0, -2.0]


def test_calc_metrics_basic_values():
    equity = pd.DataFrame(
        {"total": [100000.0, 101000.0, 99000.0, 102000.0]},
        index=pd.to_datetime(
            ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]
        ),
    )
    trade_log = [
        FillEvent(
            timestamp=pd.Timestamp("2025-01-02"),
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=100.0,
            fill_quantity=10,
        ),
        FillEvent(
            timestamp=pd.Timestamp("2025-01-06"),
            symbol="AAPL",
            side=OrderSide.SELL,
            fill_price=110.0,
            fill_quantity=10,
        ),
    ]

    metrics = calc_metrics(equity, trade_log, initial_capital=100000.0)

    assert metrics["final_value"] == 102000.0
    assert metrics["total_return"] == 0.02
    assert metrics["total_trades"] == 2
    assert metrics["win_rate"] == 1.0
    assert metrics["profit_factor"] > 1.0
    assert metrics["max_drawdown"] < 0.0
