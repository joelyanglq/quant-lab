"""Trade and risk event logger — JSON Lines format.

Events:
    ORDER  — records intent (order submitted)
    FILL   — records reality (order filled, with price + commission)
    RISK   — records risk rule violation
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class OrderEvent:
    event: str = "ORDER"
    order_id: str = ""
    strategy: str = ""        # e.g. "dte0_ic", "monthly_ic"
    action: str = ""          # "open" | "close"
    expiry: str = ""
    short_put: float = 0.0
    long_put: float = 0.0
    short_call: float = 0.0
    long_call: float = 0.0
    limit_price: float = 0.0
    qty: int = 0
    reason: str = ""          # "" for open; exit reason for close
    ts: str = ""


@dataclass
class FillEvent:
    event: str = "FILL"
    order_id: str = ""        # links to OrderEvent
    fill_price: float = 0.0
    qty: int = 0
    commission: float = 0.0
    ts: str = ""


class Logger:
    def __init__(self, log_dir: str = "logs", strategy: str = ""):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trade_file = self.log_dir / "trades.jsonl"
        self.risk_file = self.log_dir / "risk_events.jsonl"
        self.strategy = strategy

    def _append(self, path: Path, record: dict):
        record["ts"] = datetime.now().isoformat()
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -- order / fill events --------------------------------------------------

    def log_order(self, action: str, legs, limit_price: float,
                  qty: int, reason: str = "") -> str:
        """Log an order submission. Returns generated order_id."""
        oid = uuid.uuid4().hex[:12]
        evt = OrderEvent(
            order_id=oid,
            strategy=self.strategy,
            action=action,
            expiry=legs.expiry,
            short_put=legs.short_put,
            long_put=legs.long_put,
            short_call=legs.short_call,
            long_call=legs.long_call,
            limit_price=limit_price,
            qty=qty,
            reason=reason,
        )
        self._append(self.trade_file, asdict(evt))
        return oid

    def log_fill(self, order_id: str, fill_price: float,
                 qty: int, commission: float = 0.0):
        """Log a fill received for a previously logged order."""
        evt = FillEvent(
            order_id=order_id,
            fill_price=fill_price,
            qty=qty,
            commission=commission,
        )
        self._append(self.trade_file, asdict(evt))

    # -- risk events (unchanged) ----------------------------------------------

    def log_risk(self, rule: str, detail: str, action: str):
        self._append(self.risk_file, {
            "event": "RISK",
            "strategy": self.strategy,
            "rule": rule, "detail": detail, "action": action,
        })

    # -- summary --------------------------------------------------------------

    def get_trades(self) -> List[Dict[str, Any]]:
        if not self.trade_file.exists():
            return []
        trades = []
        with open(self.trade_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))
        return trades

    def summary(self) -> Dict[str, Any]:
        """Compute performance summary from trade log."""
        trades = self.get_trades()
        if not trades:
            return {"total_trades": 0, "net_pnl": 0.0}

        # Detect format: old (OPEN/CLOSE) vs new (ORDER/FILL)
        events = {t.get("event") for t in trades}
        if "ORDER" in events or "FILL" in events:
            return self._summary_order_fill(trades)
        return self._summary_legacy(trades)

    def _summary_order_fill(self, trades: List[dict]) -> Dict[str, Any]:
        """Summary from ORDER+FILL event pairs."""
        orders: Dict[str, dict] = {}
        fills: Dict[str, List[dict]] = {}

        for t in trades:
            if t["event"] == "ORDER":
                orders[t["order_id"]] = t
            elif t["event"] == "FILL":
                fills.setdefault(t["order_id"], []).append(t)

        # Collect open fills and close fills by expiry
        open_fills: Dict[str, List[dict]] = {}  # expiry -> list
        round_trips: List[dict] = []

        for oid, order in orders.items():
            order_fills = fills.get(oid, [])
            if not order_fills:
                continue

            for f in order_fills:
                if order["action"] == "open":
                    open_fills.setdefault(order["expiry"], []).append({
                        "fill_price": f["fill_price"],
                        "qty": f["qty"],
                        "commission": f["commission"],
                    })
                elif order["action"] == "close":
                    opens = open_fills.get(order["expiry"], [])
                    if opens:
                        o = opens.pop(0)
                        qty = min(o["qty"], f["qty"])
                        gross = (o["fill_price"] - f["fill_price"]) * 100 * qty
                        comm = o["commission"] + f["commission"]
                        round_trips.append({
                            "expiry": order["expiry"],
                            "reason": order.get("reason", ""),
                            "credit": o["fill_price"],
                            "debit": f["fill_price"],
                            "qty": qty,
                            "gross_pnl": gross,
                            "commission": comm,
                            "net_pnl": gross - comm,
                        })

        if not round_trips:
            return {"total_trades": 0, "net_pnl": 0.0}

        pnls = [rt["net_pnl"] for rt in round_trips]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # max drawdown
        peak = running = 0.0
        max_dd = 0.0
        for p in pnls:
            running += p
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd

        gross_total = sum(rt["gross_pnl"] for rt in round_trips)
        comm_total = sum(rt["commission"] for rt in round_trips)

        return {
            "total_trades": len(round_trips),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(round_trips),
            "gross_pnl": gross_total,
            "total_commission": comm_total,
            "net_pnl": gross_total - comm_total,
            "avg_pnl": sum(pnls) / len(pnls),
            "max_win": max(pnls),
            "max_loss": min(pnls),
            "max_drawdown": max_dd,
            "tail_losses": [rt for rt in round_trips if rt["net_pnl"] < -500],
        }

    def _summary_legacy(self, trades: List[dict]) -> Dict[str, Any]:
        """Backward-compatible summary for old OPEN/CLOSE format."""
        closes = [t for t in trades if t["event"] == "CLOSE"]
        if not closes:
            return {"total_trades": 0, "net_pnl": 0.0}

        pnls = [t["pnl"] for t in closes]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        peak = running = 0.0
        max_dd = 0.0
        for p in pnls:
            running += p
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd

        return {
            "total_trades": len(closes),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closes),
            "gross_pnl": sum(pnls),
            "total_commission": 0.0,
            "net_pnl": sum(pnls),
            "avg_pnl": sum(pnls) / len(pnls),
            "max_win": max(pnls),
            "max_loss": min(pnls),
            "max_drawdown": max_dd,
            "tail_losses": [t for t in closes if t["pnl"] < -500],
        }
