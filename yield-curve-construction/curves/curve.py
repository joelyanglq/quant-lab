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
        
        # Remove duplicate time points (keep first occurrence)
        seen = set()
        unique_nodes = []
        for t, df in self.nodes:
            if t not in seen:
                seen.add(t)
                unique_nodes.append((t, df))
        self.nodes = unique_nodes
        
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
        """Linear interpolation on ln(df) with proper extrapolation."""
        if not self.nodes:
            raise ValueError("No nodes available.")
            
        if t <= self.nodes[0][0]:
            # Extrapolate left using first segment
            return self._extrapolate_left(t)
        if t >= self.nodes[-1][0]:
            # Extrapolate right using last segment
            return self._extrapolate_right(t)
            
        # Interpolate within range
        for i in range(1, len(self.nodes)):
            t0, df0 = self.nodes[i-1]
            t1, df1 = self.nodes[i]
            if t0 <= t <= t1:
                w = (t - t0) / (t1 - t0)
                return math.exp((1-w)*math.log(df0) + w*math.log(df1))
                
        return self.nodes[-1][1]

    def _extrapolate_left(self, t: float) -> float:
        """Extrapolate ln(df) linearly using the first segment."""
        if len(self.nodes) == 1:
            # Single node: flat forward extrapolation
            t1, df1 = self.nodes[0]
            return math.exp(math.log(df1) * (t / t1))
        
        # Use first two nodes for extrapolation
        t0, df0 = self.nodes[0]
        t1, df1 = self.nodes[1]
        
        # Linear extrapolation on ln(df)
        ln0, ln1 = math.log(df0), math.log(df1)
        slope = (ln1 - ln0) / (t1 - t0)
        ln = ln0 + slope * (t - t0)
        return math.exp(ln)

    def _extrapolate_right(self, t: float) -> float:
        """Extrapolate ln(df) linearly using the last segment."""
        if len(self.nodes) == 1:
            # Single node: flat forward extrapolation
            t1, df1 = self.nodes[0]
            return math.exp(math.log(df1) * (t / t1))
        
        # Use last two nodes for extrapolation
        t0, df0 = self.nodes[-2]
        t1, df1 = self.nodes[-1]
        
        # Linear extrapolation on ln(df)
        ln0, ln1 = math.log(df0), math.log(df1)
        slope = (ln1 - ln0) / (t1 - t0)
        ln = ln0 + slope * (t - t0)
        return math.exp(ln)

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
