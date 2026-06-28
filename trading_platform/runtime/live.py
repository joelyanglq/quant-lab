"""Live runtime — connects to IBKR (paper or live), runs strategies in real time.

Usage:
    # Paper (default port 7497)
    python -m trading_platform.runtime.live \\
        --strategy timing --symbols AAPL --rule HHT --bar-size '5 mins'

    # Live (port 7496) — requires explicit confirmation
    python -m trading_platform.runtime.live \\
        --strategy timing --symbols AAPL --rule HHT \\
        --port 7496 --i-understand-this-uses-real-money

The live runtime loads historical context from Parquet, then subscribes to
IBKR real-time bars. Each incoming bar drives the same Engine -> Strategy
-> ExecutionHandler chain as the backtest, so strategy code is unchanged.

Risk subsystem (kill-switch, reconciliation, slippage monitoring, audit log)
runs in a sidecar thread and can halt new order submission at any time.
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

import pandas as pd

from ..analytics.metrics import compute_metrics, print_report
from ..core.context import BacktestDataContext
from ..core.engine import Engine
from ..core.events import Frequency, OrderEvent
from ..data.live_feed import LiveDataFeed
from ..execution.live_ibkr import LiveExecutionHandler
from ..execution.shadow import ShadowExecutionHandler
from ..risk.kill_switch import KillSwitch
from ..risk.monitoring import OrderAuditLog, SlippageMonitor
from ..risk.portfolio import Portfolio
from ..risk.reconciliation import Reconciler
from ..strategy.archetypes.cross_section import CrossSectionAlpha
from ..strategy.archetypes.pairs import PairsAlpha
from ..strategy.archetypes.rotation import RotationAlpha
from ..strategy.archetypes.single_name_timing import SingleNameTimingAlpha
from ..strategy.combiner import WeightedCombiner
from ..strategy.composite import CompositeStrategy
from ..strategy.sizer import RiskSizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


def _bar_size_to_frequency(s: str) -> Frequency:
    return {
        "1 min": Frequency.MIN_1,
        "5 mins": Frequency.MIN_5,
        "30 mins": Frequency.MIN_30,
        "1 day": Frequency.EOD,
    }.get(s, Frequency.MIN_5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--mode", choices=["shadow", "paper", "live"], default="paper")
    ap.add_argument("--port", type=int, default=7497, help="7497=paper, 7496=live")
    ap.add_argument("--client-id", type=int, default=2)
    ap.add_argument("--strategy", required=True,
                    choices=["timing", "cross_section", "pairs", "rotation"])
    ap.add_argument("--symbols", default="AAPL")
    ap.add_argument("--rule", default="HHT")
    ap.add_argument("--rebalance", default="W-FRI")
    ap.add_argument("--bar-size", default="5 mins")
    ap.add_argument("--target-vol", type=float, default=0.15)
    ap.add_argument("--max-leverage", type=float, default=0.5,
                    help="conservative initial value; raise after stable run")
    ap.add_argument("--max-daily-loss", type=float, default=0.01)
    ap.add_argument("--initial-capital", type=float, default=10_000.0)
    ap.add_argument("--i-understand-this-uses-real-money", action="store_true",
                    help="REQUIRED for --port 7496 (live trading)")
    args = ap.parse_args()

    if args.port == 7496 and not args.i_understand_this_uses_real_money:
        log.error("Refusing to connect to live port without explicit acknowledgment")
        log.error("Re-run with --i-understand-this-uses-real-money")
        return

    # ── Build strategy ────────────────────────────────────────────────
    ctx = BacktestDataContext(args.data_root)
    if args.strategy == "timing":
        alpha = SingleNameTimingAlpha(args.symbols.split(","), rule=args.rule,
                                      trigger_freq=_bar_size_to_frequency(args.bar_size),
                                      strategy_id=f"timing_{args.rule}")
        feed_syms = args.symbols.split(",")
    elif args.strategy == "rotation":
        from ..strategy.archetypes.rotation import DEFAULT_SECTOR_ETFS
        alpha = RotationAlpha(rebalance_freq=args.rebalance)
        feed_syms = DEFAULT_SECTOR_ETFS
    elif args.strategy == "cross_section":
        alpha = CrossSectionAlpha(rebalance_freq=args.rebalance)
        feed_syms = ctx.universe(pd.Timestamp.utcnow())
    else:
        alpha = PairsAlpha()
        try:
            pairs = ctx.as_of(pd.Timestamp.utcnow(), "cointegration_pairs")
        except Exception:
            pairs = pd.DataFrame()
        feed_syms = sorted(set(pairs.get("symbol_a", [])).union(pairs.get("symbol_b", [])))

    portfolio = Portfolio(initial_capital=args.initial_capital)

    # ── Choose execution handler ──────────────────────────────────────
    if args.mode == "shadow":
        execution = ShadowExecutionHandler()
    else:
        execution = LiveExecutionHandler(
            port=args.port,
            client_id=args.client_id,
            confirm_live=args.i_understand_this_uses_real_money,
        )
        execution.connect()
    execution.on_fill(portfolio.on_fill)

    # ── Risk subsystem ────────────────────────────────────────────────
    kill = KillSwitch(
        max_leverage=args.max_leverage,
        max_daily_loss_pct=args.max_daily_loss,
    )
    if kill.is_active():
        log.error("KILL-SWITCH already active: %s", kill.reason)
        log.error("Reset manually before retrying.")
        return

    audit = OrderAuditLog()
    slippage = SlippageMonitor()

    # Wrap execution.submit_order to gate on kill-switch + audit.
    original_submit = execution.submit_order

    def guarded_submit(order: OrderEvent):
        if kill.is_active():
            log.error("Kill-switch active; refusing to submit %s", order)
            return None
        if kill.is_strategy_paused(order.strategy_id or ""):
            log.warning("Strategy %s paused; skipping order", order.strategy_id)
            return None
        oid = original_submit(order)
        audit.log_order(order, status="submitted")
        return oid

    execution.submit_order = guarded_submit  # type: ignore

    # Reconciliation in sidecar thread (live mode only).
    if args.mode in ("paper", "live"):
        recon = Reconciler(
            local_positions_fn=lambda: portfolio.positions(),
            broker_positions_fn=execution.get_positions,
            market_price_fn=lambda s: None,  # filled by the engine via portfolio market prices
            on_mismatch=lambda msg: kill.activate(msg),
        )
        t = threading.Thread(target=recon.run_loop, kwargs={"interval_seconds": 60}, daemon=True)
        t.start()

    # ── Compose strategy ──────────────────────────────────────────────
    sizer = RiskSizer(target_vol=args.target_vol, max_leverage=args.max_leverage)
    composite = CompositeStrategy(
        alphas=[alpha],
        combiner=WeightedCombiner(),
        sizer=sizer,
        ctx=ctx,
        execution=execution,
        portfolio=portfolio,
        initial_capital=args.initial_capital,
        trigger_freq=_bar_size_to_frequency(args.bar_size),
        strategy_id=alpha.strategy_id,
    )

    # ── Live data feed ────────────────────────────────────────────────
    feed = LiveDataFeed(
        symbols=feed_syms,
        port=args.port,
        client_id=args.client_id + 100,  # different from execution
        bar_size=args.bar_size,
    )

    engine = Engine([feed], [composite], execution, portfolio)
    log.info("Live runtime starting: mode=%s strategy=%s symbols=%s",
             args.mode, args.strategy, ",".join(feed_syms[:5]))
    try:
        engine.run()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — shutting down")
    finally:
        if hasattr(execution, "disconnect"):
            execution.disconnect()
        if hasattr(feed, "close"):
            feed.close()


if __name__ == "__main__":
    main()
