from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Tuple, Any, Optional, Callable, Sequence
import pandas as pd
import numpy as np

from .daycount import yearfrac
from ..curve import YieldCurve

@dataclass(frozen=True)
class BootstrapConfig:
    yearfrac: Callable[[Any, Any], float] = yearfrac
    tol: float = 1e-12
    allow_df_increase: bool = False
    return_report: bool = True

class Bootstrapper:
    def __init__(self, config: BootstrapConfig = BootstrapConfig()):
        self.cfg = config

    def build_discount_curve(self, instruments: List[Any]):
        """
        Bootstrap discount curve using linear equation method (not bisection).
        
        Returns:
            YieldCurve object (and optionally report)
        """
        if len(instruments) == 0:
            raise ValueError("No instruments provided.")

        # Determine valuation date
        val_date = instruments[0].val_date
        
        # Build a list with extracted cashflows and prices
        items = []
        for inst in instruments:
            # Validate same valuation date if present
            if hasattr(inst, "val_date"):
                vd = inst.val_date
                if vd != val_date:
                    raise ValueError("All instruments must share the same valuation date.")

            cfs = inst.cashflows()
            if not cfs:
                continue

            # Filter strictly after valuation date
            cfs = [(cf.pay_date, cf.amount) for cf in cfs if cf.pay_date > val_date]
            if not cfs:
                continue

            # Price
            price = inst.dirty_price

            mat_date = max(d for d, _ in cfs)
            mat_t = self.cfg.yearfrac(val_date, mat_date)

            items.append({
                "inst": inst,
                "inst_id": getattr(inst, "cusip", str(id(inst))),
                "price": price,
                "cashflows": cfs,
                "maturity_date": mat_date,
                "maturity_t": float(mat_t),
            })

        if len(items) == 0:
            raise ValueError("After filtering, no instruments with future cashflows remain.")

        # Sort by maturity time (short to long)
        items.sort(key=lambda x: x["maturity_t"])

        # Bootstrap nodes
        nodes_t: List[float] = []
        nodes_df: List[float] = []
        report_rows: List[dict] = []

        prev_df: Optional[float] = None
        
        for k, it in enumerate(items, start=1):
            inst = it["inst"]
            price = it["price"]
            cfs = it["cashflows"]
            mat_date = it["maturity_date"]
            t_n = it["maturity_t"]

            # If we already have nodes, build/update curve for interpolation of earlier cashflows
            temp_curve = None
            if len(nodes_t) >= 1:
                temp_curve = YieldCurve(
                    val_date=val_date,
                    nodes=list(zip(nodes_t, nodes_df)),
                    yearfrac=self.cfg.yearfrac
                )

            # PV of all cashflows except those on maturity date (unknown DF at t_n)
            pv_known = 0.0
            if temp_curve is not None:
                pv_known = self._pv_known(val_date, cfs, temp_curve, mat_date)

            # Sum of amounts on maturity date
            cf_n = sum(amt for d, amt in cfs if d == mat_date)

            # Solve DF(t_n) linearly
            numer = price - pv_known
            df_n = numer / cf_n if cf_n != 0 else np.nan

            # Sanity checks
            status = "ok"
            warn = ""

            if not np.isfinite(df_n):
                status = "fail"
                warn = "df_nonfinite"
            elif df_n <= 0:
                status = "fail"
                warn = "df_nonpositive"
            elif not self.cfg.allow_df_increase and prev_df is not None and df_n > prev_df + 1e-10:
                status = "warn"
                warn = "df_increasing"

            # Also check numer positivity (should be > 0 for positive df)
            if status != "fail" and numer <= 0:
                status = "fail"
                warn = "price_minus_pv_known_nonpositive"

            # Append node if ok/warn (fail -> you may want to drop instrument)
            if status == "fail":
                report_rows.append({
                    "step": k,
                    "inst_id": it["inst_id"],
                    "maturity_t": t_n,
                    "maturity_date": mat_date,
                    "price": price,
                    "pv_known": pv_known,
                    "cf_maturity": cf_n,
                    "df_solved": float(df_n) if np.isfinite(df_n) else np.nan,
                    "zero_cont": np.nan,
                    "status": status,
                    "warn": warn,
                })
                continue

            nodes_t.append(float(t_n))
            nodes_df.append(float(df_n))
            prev_df = float(df_n)

            # Build curve with the new node to compute residual (reprice instrument)
            curve_now = YieldCurve(
                val_date=val_date,
                nodes=list(zip(nodes_t, nodes_df)),
                yearfrac=self.cfg.yearfrac
            )

            pv_all = 0.0
            for d, amt in cfs:
                tt = self.cfg.yearfrac(val_date, d)
                if tt <= 0:
                    continue
                pv_all += amt * curve_now.df(tt)

            residual = pv_all - price
            zero_cont = curve_now.zero_rate_cc(t_n)

            report_rows.append({
                "step": k,
                "inst_id": it["inst_id"],
                "maturity_t": t_n,
                "maturity_date": mat_date,
                "price": price,
                "pv_known": pv_known,
                "cf_maturity": cf_n,
                "df_solved": float(df_n),
                "zero_cont": float(zero_cont),
                "residual": float(residual),
                "status": status,
                "warn": warn,
            })

        if len(nodes_t) == 0:
            raise ValueError("Bootstrapping failed: no valid nodes produced.")

        final_curve = YieldCurve(
            val_date=val_date,
            nodes=list(zip(nodes_t, nodes_df)),
            yearfrac=self.cfg.yearfrac
        )
        
        if self.cfg.return_report:
            report_df = pd.DataFrame(report_rows)
            return final_curve, report_df
        return final_curve

    def _pv_known(self, val_date, cashflows: List[Tuple[Any, float]], curve: YieldCurve, unknown_last_date: Any) -> float:
        """
        Present value of all cashflows except those occurring on unknown_last_date.
        Use curve.df(t) for their discounting (with interpolation).
        """
        pv = 0.0
        for d, amt in cashflows:
            if d == unknown_last_date:
                continue
            t = self.cfg.yearfrac(val_date, d)
            if t <= 0:
                continue
            pv += amt * curve.df(t)
        return float(pv)
