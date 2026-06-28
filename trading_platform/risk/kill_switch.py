"""KillSwitch — central runtime safety with persistent state.

Triggers (any of):
    - max_leverage breached
    - daily P&L below -max_daily_loss * NetLiquidation
    - per-strategy 7-day cumulative loss > 5% of allocation
    - IBKR disconnection > disconnect_grace_seconds
    - anomalous price (>50% gap from previous bar)
    - manual operator command

State persists to JSON; restart respects active kill-switch and refuses
new orders until manually reset.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


class KillSwitch:
    def __init__(
        self,
        state_path: str | Path = "runtime/state/kill_switch.json",
        max_leverage: float = 1.0,
        max_daily_loss_pct: float = 0.02,
        per_strategy_loss_pct: float = 0.05,
        disconnect_grace_seconds: int = 300,
        anomaly_pct: float = 0.50,
    ):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_leverage = max_leverage
        self.max_daily_loss_pct = max_daily_loss_pct
        self.per_strategy_loss_pct = per_strategy_loss_pct
        self.disconnect_grace_seconds = disconnect_grace_seconds
        self.anomaly_pct = anomaly_pct

        self._active = False
        self._reason: Optional[str] = None
        self._activated_at: Optional[pd.Timestamp] = None
        self._strategy_pnl_7d: dict[str, deque] = defaultdict(lambda: deque(maxlen=7))
        self._strategy_paused: set = set()
        self._disconnect_since: Optional[float] = None
        self._last_bar_price: dict[str, float] = {}

        self._load_state()

    # ── state persistence ────────────────────────────────────────────
    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._active = bool(data.get("active", False))
            self._reason = data.get("reason")
            ts = data.get("activated_at")
            self._activated_at = pd.Timestamp(ts) if ts else None
            if self._active:
                log.warning("Loaded ACTIVE kill-switch state from %s: %s", self.state_path, self._reason)
        except Exception as e:
            log.exception("Failed to load kill-switch state: %s", e)

    def _persist(self) -> None:
        data = {
            "active": self._active,
            "reason": self._reason,
            "activated_at": self._activated_at.isoformat() if self._activated_at else None,
        }
        self.state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ── activation ───────────────────────────────────────────────────
    def activate(self, reason: str) -> None:
        if self._active:
            return
        self._active = True
        self._reason = reason
        self._activated_at = pd.Timestamp.utcnow()
        log.critical("KILL-SWITCH ACTIVATED: %s", reason)
        self._persist()

    def is_active(self) -> bool:
        return self._active

    def reset(self, operator_confirm: bool = False) -> None:
        if not operator_confirm:
            raise RuntimeError("Kill-switch reset requires explicit operator_confirm=True")
        self._active = False
        self._reason = None
        self._activated_at = None
        self._strategy_paused.clear()
        self._persist()
        log.info("Kill-switch reset")

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    # ── per-trigger checks ───────────────────────────────────────────
    def check_leverage(self, total_notional: float, net_liquidation: float) -> bool:
        if net_liquidation <= 0:
            return True
        leverage = total_notional / net_liquidation
        if leverage > self.max_leverage:
            self.activate(f"LEVERAGE_BREACH: {leverage:.2f} > {self.max_leverage}")
            return False
        return True

    def check_daily_loss(self, intraday_pnl: float, net_liquidation: float) -> bool:
        if net_liquidation <= 0:
            return True
        loss_pct = -intraday_pnl / net_liquidation
        if loss_pct > self.max_daily_loss_pct:
            self.activate(f"DAILY_LOSS_BREACH: -{loss_pct:.2%}")
            return False
        return True

    def update_strategy_pnl(self, strategy_id: str, daily_pnl: float, allocation: float) -> None:
        self._strategy_pnl_7d[strategy_id].append(daily_pnl)
        cum = sum(self._strategy_pnl_7d[strategy_id])
        if allocation > 0 and cum < -self.per_strategy_loss_pct * allocation:
            self._strategy_paused.add(strategy_id)
            log.warning("Strategy %s paused (7d loss %.2f > %.0f%% of $%s)",
                        strategy_id, cum, self.per_strategy_loss_pct * 100, allocation)

    def is_strategy_paused(self, strategy_id: str) -> bool:
        return strategy_id in self._strategy_paused

    def check_connection(self, is_connected: bool) -> bool:
        now = time.time()
        if is_connected:
            self._disconnect_since = None
            return True
        if self._disconnect_since is None:
            self._disconnect_since = now
            return True
        if now - self._disconnect_since > self.disconnect_grace_seconds:
            self.activate(f"DISCONNECT_TIMEOUT: > {self.disconnect_grace_seconds}s")
            return False
        return True

    def check_anomalous_price(self, symbol: str, price: float) -> bool:
        prev = self._last_bar_price.get(symbol)
        self._last_bar_price[symbol] = price
        if prev is None or prev <= 0:
            return True
        change = abs(price - prev) / prev
        if change > self.anomaly_pct:
            log.warning("Anomalous price for %s: %.2f -> %.2f (%.0f%%)",
                        symbol, prev, price, change * 100)
            # We pause submission for this symbol via the runtime layer; do not
            # auto-trip kill-switch on a single bar to avoid false positives.
            return False
        return True
