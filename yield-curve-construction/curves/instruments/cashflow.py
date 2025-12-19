from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List

@dataclass(frozen=True)
class Cashflow:
    pay_date: datetime
    amount: float

def filter_cashflows_after(val_date: datetime, cfs: Iterable[Cashflow]) -> List[Cashflow]:
    """Keep only cashflows strictly after valuation date."""
    return [cf for cf in cfs if cf.pay_date > val_date]
