from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List

from .cashflow import Cashflow, filter_cashflows_after

FACE = 100.0

@dataclass(frozen=True)
class Bill:
    key: str                 # e.g., KYTREASNO or TCUSIP
    cusip: str
    val_date: datetime       # CALDT
    maturity_date: datetime  # TMATDT
    clean_price: float       # TDNOMPRC
    accrued_interest: float  # TDACCINT

    @property
    def dirty_price(self) -> float:
        return float(self.clean_price) + float(self.accrued_interest)

    def cashflows(self) -> List[Cashflow]:
        cfs = [Cashflow(self.maturity_date, FACE)]
        return filter_cashflows_after(self.val_date, cfs)
