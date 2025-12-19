from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional

import pandas as pd

from .bill import Bill
from .bond import Bond, Note

def _to_dt(x: Any) -> Optional[datetime]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()

def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default

def _to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return int(x)
    except Exception:
        return default

@dataclass(frozen=True)
class InstrumentFactory:
    """
    CRSP row -> Instrument object.
    Keep CRSP-specific column names here (single choke point).
    """

    @staticmethod
    def from_crsp_row(row: Mapping[str, Any]) -> Any:
        itype = _to_int(row.get("ITYPE"))
        freq = _to_int(row.get("TNIPPY"))

        val_date = _to_dt(row.get("CALDT"))
        maturity = _to_dt(row.get("TMATDT"))
        dated = _to_dt(row.get("TDATDT"))
        first_cp = _to_dt(row.get("TFCPDT"))

        # prefer KYTREASNO as stable key; fallback to CUSIP/CRSPID
        key = str(row.get("KYTREASNO") or row.get("TCUSIP") or row.get("CRSPID"))
        cusip = str(row.get("TCUSIP") or "")

        clean = _to_float(row.get("TDNOMPRC"))
        ai = _to_float(row.get("TDACCINT"))

        # coupon in CRSP is percentage (e.g., 4.5), convert to decimal
        coupon_rate = _to_float(row.get("TCOUPRT")) / 100.0

        if val_date is None or maturity is None:
            raise ValueError("Missing CALDT or TMATDT; cannot build instrument.")

        # ITYPE handling for v1:
        # 4 = Bill, 2 = Note, 1 = Bond (ignore callable/TIPS/etc for v1)
        if itype == 4 or freq == 0 or coupon_rate == 0.0:
            return Bill(
                key=key, cusip=cusip,
                val_date=val_date, maturity_date=maturity,
                clean_price=clean, accrued_interest=ai
            )

        # For coupon instruments: need TDATDT for schedule fallback anchor
        if dated is None:
            # v1 fallback: if dated missing, use val_date as anchor (keeps schedule generator from blowing up)
            dated = val_date

        base_kwargs = dict(
            key=key, cusip=cusip,
            val_date=val_date, dated_date=dated, maturity_date=maturity,
            coupon_rate=coupon_rate, freq=max(freq, 2),  # usually 2
            clean_price=clean, accrued_interest=ai,
            first_coupon_date=first_cp
        )

        if itype == 2:
            return Note(**base_kwargs)
        elif itype == 1:
            return Bond(**base_kwargs)

        # default: treat as Bond-like if you still want to include it
        # (but for v1 you likely filter these out before calling factory)
        return Bond(**base_kwargs)
