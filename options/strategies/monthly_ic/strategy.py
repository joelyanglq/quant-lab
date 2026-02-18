"""Monthly Iron Condor strategy — poll-based with VIX spike trigger."""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from options.core.base import Strategy
from options.core.broker import Broker
from options.core.logger import Logger
from .config import MonthlyICConfig
from .scanner import check_entry_trigger, find_condor_legs
from .risk import check_risk

ET = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)


class MonthlyICStrategy(Strategy):
    strategy_name = "monthly_ic"

    def __init__(self, **overrides):
        super().__init__()
        self.cfg = MonthlyICConfig(**{
            k: v for k, v in overrides.items()
            if hasattr(MonthlyICConfig, k)
        })
        self.open_positions: list[dict] = []

    @property
    def name(self) -> str:
        return self.strategy_name

    def make_broker(self) -> Broker:
        return Broker(self.cfg)

    def make_logger(self) -> Logger:
        return Logger(self.cfg.log_dir, strategy=self.strategy_name)

    def poll_interval(self) -> int:
        return self.cfg.poll_interval_sec

    def in_trading_hours(self) -> bool:
        now = datetime.now(ET).time()
        return self.cfg.rth_start <= now <= self.cfg.rth_end

    def get_open_position_count(self) -> int:
        return len(self.open_positions)

    def setup(self, broker, logger):
        log.info("[monthly_ic] symbol=%s  port=%d  poll=%ds",
                 self.cfg.symbol, self.cfg.port, self.cfg.poll_interval_sec)

    def poll_once(self, broker, logger):
        log.info("--- poll %s ET ---",
                 datetime.now(ET).strftime("%H:%M:%S"))

        # 1. risk check (highest priority)
        if self.open_positions:
            self._check_risk_exit(broker, logger)

        # 2. time exit
        if self.open_positions:
            self._check_time_exit(broker, logger)

        # 3. new entry
        self._try_entry(broker, logger)

        log.info("Open positions: %d", len(self.open_positions))

    def teardown(self, broker, logger):
        s = logger.summary()
        if s.get("total_trades", 0) > 0:
            log.info("=== SUMMARY ===")
            for k, v in s.items():
                if k != "tail_losses":
                    log.info("  %s: %s", k, v)

    # -- private logic ---------------------------------------------------------

    def _try_entry(self, broker, logger):
        if len(self.open_positions) >= self.cfg.max_positions:
            return

        if not check_entry_trigger(broker, self.cfg):
            return

        log.info("=== VIX ENTRY TRIGGERED ===")
        legs = find_condor_legs(broker, self.cfg)
        if legs is None:
            log.warning("No valid condor found")
            return

        oid = logger.log_order("open", legs,
                               limit_price=self.cfg.min_credit,
                               qty=self.cfg.default_qty)

        result = broker.place_condor(legs, self.cfg.default_qty,
                                     self.cfg.min_credit)
        if not result.filled:
            log.warning("Open order not filled  status=%s", result.status)
            return

        logger.log_fill(oid, result.avg_price,
                        result.filled_qty, result.total_commission)

        self.open_positions.append({
            "legs": legs,
            "qty": result.filled_qty,
            "credit": result.avg_price,
            "entry_ts": datetime.now().isoformat(),
        })

    def _check_time_exit(self, broker, logger):
        to_remove = []
        for i, pos in enumerate(self.open_positions):
            dte = self._dte(pos["legs"].expiry)
            if dte <= self.cfg.exit_dte:
                log.info("TIME EXIT  expiry=%s  DTE=%d",
                         pos["legs"].expiry, dte)

                oid = logger.log_order("close", pos["legs"],
                                       limit_price=0.0,
                                       qty=pos["qty"],
                                       reason="time_exit")
                result = broker.close_condor(pos["legs"], pos["qty"])

                logger.log_fill(oid, result.avg_price,
                                result.filled_qty, result.total_commission)
                to_remove.append(i)

        for i in reversed(to_remove):
            self.open_positions.pop(i)

    def _check_risk_exit(self, broker, logger):
        to_remove = []
        for i, pos in enumerate(self.open_positions):
            alerts = check_risk(broker, pos["legs"], self.cfg)
            if not alerts:
                continue

            close_qty = max(1, pos["qty"] // 2)
            log.warning("RISK EXIT  alerts=%s  closing %d of %d",
                        [a.rule for a in alerts], close_qty, pos["qty"])

            oid = logger.log_order("close", pos["legs"],
                                   limit_price=0.0,
                                   qty=close_qty,
                                   reason="risk_exit")
            result = broker.close_condor(pos["legs"], close_qty,
                                         use_market=False)

            logger.log_fill(oid, result.avg_price,
                            result.filled_qty, result.total_commission)

            for a in alerts:
                logger.log_risk(a.rule, a.detail, f"close_{close_qty}")

            pos["qty"] -= result.filled_qty
            if pos["qty"] <= 0:
                to_remove.append(i)

        for i in reversed(to_remove):
            self.open_positions.pop(i)

    @staticmethod
    def _dte(expiry: str) -> int:
        exp = datetime.strptime(expiry, "%Y%m%d").date()
        return (exp - datetime.now().date()).days
