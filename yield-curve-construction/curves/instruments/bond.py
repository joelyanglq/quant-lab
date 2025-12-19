from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd
from dateutil.relativedelta import relativedelta

from .cashflow import Cashflow, filter_cashflows_after

FACE = 100.0

@dataclass(frozen=True)
class Bond:
    key: str
    cusip: str
    val_date: datetime       # CALDT
    dated_date: datetime     # TDATDT (used as proxy / schedule anchor)
    maturity_date: datetime  # TMATDT
    coupon_rate: float       # annual coupon rate in decimal, e.g. 0.045
    freq: int                # TNIPPY, usually 2
    clean_price: float       # TDNOMPRC
    accrued_interest: float  # TDACCINT
    first_coupon_date: Optional[datetime] = None  # TFCPDT (optional)

    @property
    def dirty_price(self) -> float:
        return float(self.clean_price) + float(self.accrued_interest)

    def _coupon_amount(self) -> float:
        return FACE * self.coupon_rate / self.freq

    def _generate_coupon_dates(self) -> List[datetime]:
        """
        Minimal robust schedule:
        - If first_coupon_date is available: step forward by 12/freq months until maturity.
        - Else: step backward from maturity by 12/freq months (common fallback).
        """
        months = int(round(12 / self.freq))
        dates: List[datetime] = []

        if self.first_coupon_date is not None and pd.notna(self.first_coupon_date):
            d = self.first_coupon_date
            # march forward to maturity
            while d < self.maturity_date:
                dates.append(d)
                d = (pd.Timestamp(d) + relativedelta(months=months)).to_pydatetime()
            dates.append(self.maturity_date)
        else:
            # fallback: go backwards from maturity
            d = self.maturity_date
            while True:
                dates.append(d)
                d_prev = (pd.Timestamp(d) - relativedelta(months=months)).to_pydatetime()
                if d_prev <= self.dated_date:
                    break
                d = d_prev
            dates = sorted(set(dates))

        return dates

    def cashflows(self) -> List[Cashflow]:
        coupon = self._coupon_amount()
        dates = self._generate_coupon_dates()

        cfs: List[Cashflow] = []
        for d in dates:
            if d < self.maturity_date:
                cfs.append(Cashflow(d, coupon))
            else:
                cfs.append(Cashflow(d, coupon + FACE))

        return filter_cashflows_after(self.val_date, cfs)

@dataclass(frozen=True)
class Note(Bond):
    """Semantically a Note (<=10Y). Same cashflow mechanics as Bond in v1."""
    pass
