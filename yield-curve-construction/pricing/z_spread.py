from __future__ import annotations

import math
from typing import Iterable

from curves.bootstrapping.daycount import yearfrac
from curves.bootstrapping.root_finding import solve_bisection
from curves.curve import YieldCurve
from curves.instruments.cashflow import Cashflow


def price_with_z_spread(
    instrument,
    curve: YieldCurve,
    spread: float,
) -> float:
    """
    Present value of an instrument when adding a parallel spread (continuous comp).

    Args:
        instrument: Treasury bill/note/bond
        curve: fitted YieldCurve
        spread: constant spread in decimal (0.001 == 10 bps)
    """
    pv = 0.0
    for cf in instrument.cashflows():
        t = yearfrac(instrument.val_date, cf.pay_date)
        if t <= 0:
            continue
        base_df = curve.df(t)
        adj_df = base_df * math.exp(-spread * t)
        pv += cf.amount * adj_df
    return pv


def solve_z_spread(
    instrument,
    curve: YieldCurve,
    target_dirty_price: float | None = None,
    *,
    lo: float = -0.05,
    hi: float = 0.05,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """
    Compute the constant spread needed so PV equals target dirty price.
    Returns the spread in decimal (0.001 == 10 bps).
    """
    target = float(target_dirty_price or instrument.dirty_price)

    def f(spread: float) -> float:
        return price_with_z_spread(instrument, curve, spread) - target

    return solve_bisection(f, lo=lo, hi=hi, tol=tol, max_iter=max_iter)
