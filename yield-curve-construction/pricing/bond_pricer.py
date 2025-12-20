from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Protocol, runtime_checkable

from curves.bootstrapping.daycount import yearfrac
from curves.curve import YieldCurve
from curves.instruments.cashflow import Cashflow


@runtime_checkable
class PricedInstrument(Protocol):
    val_date: datetime
    accrued_interest: float

    def cashflows(self) -> Iterable[Cashflow]:
        ...


@dataclass
class CashflowPV:
    """Per-cashflow present value diagnostics."""

    pay_date: datetime
    t: float
    amount: float
    df: float
    pv: float


@dataclass
class PriceBreakdown:
    dirty_price: float
    clean_price: float
    accrued_interest: float
    cashflow_pvs: List[CashflowPV]


class BondPricer:
    """
    Discount Treasury bills/notes/bonds using a fitted YieldCurve.
    """

    def __init__(self, curve: YieldCurve):
        self.curve = curve

    def price(self, instrument: PricedInstrument) -> PriceBreakdown:
        """
        Returns:
            PriceBreakdown with dirty price (PV of future flows) and clean price.
        """
        cfs = list(instrument.cashflows())
        if not cfs:
            raise ValueError("Instrument has no cashflows after valuation date.")

        breakdown: List[CashflowPV] = []
        dirty_price = 0.0
        for cf in cfs:
            t = yearfrac(instrument.val_date, cf.pay_date)
            if t <= 0:
                # Cashflows should already be filtered, but skip just in case.
                continue
            df = self.curve.df(t)
            pv = cf.amount * df
            breakdown.append(CashflowPV(cf.pay_date, t, cf.amount, df, pv))
            dirty_price += pv

        accrued = float(getattr(instrument, "accrued_interest", 0.0))
        clean = dirty_price - accrued
        return PriceBreakdown(
            dirty_price=dirty_price,
            clean_price=clean,
            accrued_interest=accrued,
            cashflow_pvs=breakdown,
        )


def price_bond(instrument: PricedInstrument, curve: YieldCurve) -> PriceBreakdown:
    """Convenience wrapper."""
    return BondPricer(curve).price(instrument)
