from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional, Callable
import math

@dataclass
class YieldCurve:
    val_date: datetime
    nodes: List[Tuple[float, float]]           # (t, df)
    interpolator: Optional[object] = None      # DFInterpolator after fit
    yearfrac: Callable[[datetime, datetime], float] = None  # Year fraction function

    def __post_init__(self):
        # Sort nodes by time
        self.nodes = sorted(self.nodes, key=lambda x: x[0])
        
        # Default yearfrac function if not provided
        if self.yearfrac is None:
            self.yearfrac = self._default_yearfrac

    def _default_yearfrac(self, d0: datetime, d1: datetime) -> float:
        """Default ACT/365.25 year fraction."""
        return (d1 - d0).days / 365.25

    def fit(self, interpolator):
        """Fit an interpolator to the nodes."""
        times = [t for t, _ in self.nodes]
        dfs = [df for _, df in self.nodes]
        self.interpolator = interpolator.fit(times, dfs)
        return self

    def df(self, x: float) -> float:
        """
        Discount factor at time t (years) or at a future date.
        
        Args:
            x: Time in years (float) or datetime object
        """
        if isinstance(x, datetime):
            t = self.yearfrac(self.val_date, x)
        else:
            t = float(x)

        if t <= 0:
            return 1.0
            
        # If we have an interpolator, use it
        if self.interpolator is not None:
            return float(self.interpolator.df(t))
            
        # Fallback: piecewise log-linear on nodes
        return self._df_loglinear(t)

    def zero_rate_cc(self, x: float) -> float:
        """
        Continuous compounding zero rate at time t (years) or at a future date.
        
        Args:
            x: Time in years (float) or datetime object
        """
        if isinstance(x, datetime):
            t = self.yearfrac(self.val_date, x)
        else:
            t = float(x)

        if t <= 0:
            return 0.0
        return -math.log(self.df(t)) / t

    def _df_loglinear(self, t: float) -> float:
        """Linear interpolation on ln(df)."""
        if not self.nodes:
            raise ValueError("No nodes available.")
            
        if t <= self.nodes[0][0]:
            return self.nodes[0][1]
        if t >= self.nodes[-1][0]:
            return self.nodes[-1][1]
            
        # Find the interval containing t
        for i in range(1, len(self.nodes)):
            t0, df0 = self.nodes[i-1]
            t1, df1 = self.nodes[i]
            if t0 <= t <= t1:
                w = (t - t0) / (t1 - t0)
                return math.exp((1-w)*math.log(df0) + w*math.log(df1))
                
        return self.nodes[-1][1]

    def zero_rate_simple(self, x: float) -> float:
        """
        Simple annualized zero rate at time t (years) or at a future date.
        
        Args:
            x: Time in years (float) or datetime object
        """
        if isinstance(x, datetime):
            t = self.yearfrac(self.val_date, x)
        else:
            t = float(x)

        if t <= 0:
            return 0.0
        return (1.0 / self.df(t) - 1.0) / t
