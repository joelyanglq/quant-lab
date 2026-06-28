"""Reconciliation — periodic comparison of local Portfolio against IBKR.

Mismatch beyond tolerance (1 share or $100 market value) triggers
kill-switch activation via the supplied callback.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

log = logging.getLogger(__name__)


class Reconciler:
    def __init__(
        self,
        local_positions_fn: Callable[[], dict[str, float]],
        broker_positions_fn: Callable[[], dict[str, float]],
        market_price_fn: Callable[[str], Optional[float]],
        on_mismatch: Callable[[str], None],
        share_tolerance: float = 1.0,
        notional_tolerance: float = 100.0,
    ):
        self.local = local_positions_fn
        self.broker = broker_positions_fn
        self.price = market_price_fn
        self.on_mismatch = on_mismatch
        self.share_tol = share_tolerance
        self.notional_tol = notional_tolerance

    def reconcile_once(self) -> list[tuple[str, float, float, float]]:
        """Run one reconciliation cycle. Returns list of (symbol, local, broker, notional_diff)."""
        local = self.local()
        broker = self.broker()
        symbols = sorted(set(local) | set(broker))
        mismatches = []
        for sym in symbols:
            l = local.get(sym, 0.0)
            b = broker.get(sym, 0.0)
            share_diff = abs(l - b)
            price = self.price(sym) or 0.0
            notional_diff = share_diff * price
            if share_diff > self.share_tol or notional_diff > self.notional_tol:
                mismatches.append((sym, l, b, notional_diff))
        if mismatches:
            details = "; ".join(f"{s}: local={l}, broker={b}, ${nd:.0f}"
                                for s, l, b, nd in mismatches)
            self.on_mismatch(f"RECON_MISMATCH: {details}")
        return mismatches

    def run_loop(self, interval_seconds: int = 60) -> None:
        """Blocking reconciliation loop. Run in a dedicated thread."""
        while True:
            try:
                self.reconcile_once()
            except Exception:
                log.exception("Reconciliation cycle failed")
            time.sleep(interval_seconds)
