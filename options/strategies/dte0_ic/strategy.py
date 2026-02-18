"""0DTE Iron Condor strategy — daily entry with P&L-based exit."""

from __future__ import annotations

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

from options.core.base import Strategy
from options.core.broker import Broker
from options.core.logger import Logger
from .config import ZeroDTEConfig
from .scanner import (check_vix_condition, is_entry_window,
                      find_condor_legs, check_risk_reward)
from .risk import check_risk, check_pnl_exit

ET = ZoneInfo("America/New_York")
log = logging.getLogger(__name__)


class ZeroDTEStrategy(Strategy):
    strategy_name = "dte0_ic"

    def __init__(self, **overrides):
        super().__init__()
        self.cfg = ZeroDTEConfig(**{
            k: v for k, v in overrides.items()
            if hasattr(ZeroDTEConfig, k)
        })
        self.open_position: dict | None = None
        self._entries_today: int = 0
        self._last_entry_date: date | None = None

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
        return 1 if self.open_position else 0

    def setup(self, broker, logger):
        log.info("[dte0_ic] symbol=%s  port=%d  poll=%ds",
                 self.cfg.symbol, self.cfg.port, self.cfg.poll_interval_sec)

    def poll_once(self, broker, logger):
        now = datetime.now(ET)
        today = now.date()

        # reset daily counter
        if self._last_entry_date != today:
            self._entries_today = 0
            self._last_entry_date = today

        log.info("--- poll %s ET  pos=%s ---",
                 now.strftime("%H:%M:%S"),
                 "OPEN" if self.open_position else "none")

        # 1. EOD forced close (highest priority)
        if self.open_position and now.time() >= self.cfg.eod_exit_time:
            self._close_position(broker, logger, "eod_exit",
                                 use_market=True)
            return

        # 2. Risk check (proximity + delta)
        if self.open_position:
            alerts = check_risk(broker, self.open_position["legs"],
                                self.cfg)
            if alerts:
                log.warning("RISK EXIT  alerts=%s",
                            [a.rule for a in alerts])
                for a in alerts:
                    logger.log_risk(a.rule, a.detail, "close_all")
                self._close_position(broker, logger, "risk_exit",
                                     use_market=True)
                return

        # 3. P&L exit (profit target / stop loss)
        if self.open_position:
            reason = check_pnl_exit(
                broker, self.open_position["legs"],
                self.open_position["credit"], self.cfg)
            if reason:
                self._close_position(broker, logger, reason)
                return

        # 4. New entry (once per day, in entry window)
        if (self.open_position is None
                and self._entries_today < self.cfg.max_entries_per_day
                and is_entry_window(self.cfg, now)):
            self._try_entry(broker, logger)

    def teardown(self, broker, logger):
        if self.open_position:
            log.warning("Shutdown with open position — force closing")
            oid = logger.log_order(
                "close", self.open_position["legs"],
                limit_price=0.0,
                qty=self.open_position["qty"],
                reason="shutdown")
            result = broker.close_condor(
                self.open_position["legs"],
                self.open_position["qty"],
                use_market=True)
            logger.log_fill(oid, result.avg_price,
                            result.filled_qty, result.total_commission)
            self.open_position = None

        s = logger.summary()
        if s.get("total_trades", 0) > 0:
            log.info("=== SUMMARY ===")
            for k, v in s.items():
                if k != "tail_losses":
                    log.info("  %s: %s", k, v)

    # -- private ---------------------------------------------------------------

    def _try_entry(self, broker, logger):
        if not check_vix_condition(broker, self.cfg):
            return

        log.info("=== 0DTE ENTRY — VIX OK ===")
        legs = find_condor_legs(broker, self.cfg)
        if legs is None:
            log.warning("No valid 0DTE condor found")
            return

        if not check_risk_reward(self.cfg.min_credit, self.cfg):
            log.info("R:R check failed, skip entry")
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

        self.open_position = {
            "legs": legs,
            "qty": result.filled_qty,
            "credit": result.avg_price,
            "entry_ts": datetime.now(ET).isoformat(),
        }
        self._entries_today += 1

    def _close_position(self, broker, logger, reason: str,
                        use_market: bool = False):
        pos = self.open_position
        if pos is None:
            return

        log.info("CLOSING  reason=%s  expiry=%s", reason,
                 pos["legs"].expiry)

        oid = logger.log_order("close", pos["legs"],
                               limit_price=0.0,
                               qty=pos["qty"],
                               reason=reason)

        result = broker.close_condor(pos["legs"], pos["qty"],
                                     use_market=use_market)

        logger.log_fill(oid, result.avg_price,
                        result.filled_qty, result.total_commission)

        if not result.filled:
            log.warning("Close order not fully filled  status=%s  "
                        "filled=%d/%d",
                        result.status, result.filled_qty, pos["qty"])

        self.open_position = None
