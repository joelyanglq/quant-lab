"""Single-name timing archetype — per-symbol directional forecasts.

A single Alpha instance subscribes to N symbols and computes an independent
forecast for each one (HHT, QRS, MA-cross, RSI, momentum, etc.). Each rule
operates on per-symbol price history.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from ...core.events import Frequency
from ..alpha import Alpha, ScalingMixin


def rule_ma_cross(close: pd.Series, fast: int = 20, slow: int = 60) -> float:
    """Faster MA above slower MA → positive."""
    if len(close) < slow:
        return float("nan")
    fast_ma = close.tail(fast).mean()
    slow_ma = close.tail(slow).mean()
    return float((fast_ma - slow_ma) / slow_ma * 100.0)


def rule_rsi(close: pd.Series, period: int = 14) -> float:
    """RSI 50 → 0; RSI > 70 → strongly negative (mean-reversion); RSI < 30 → strongly positive."""
    if len(close) < period + 1:
        return float("nan")
    diff = close.diff().dropna()
    gain = diff.clip(lower=0).rolling(period).mean().iloc[-1]
    loss = -diff.clip(upper=0).rolling(period).mean().iloc[-1]
    if loss == 0 or np.isnan(loss):
        rsi = 100.0
    else:
        rs = gain / loss
        rsi = 100.0 - 100.0 / (1.0 + rs)
    return float(50.0 - rsi)  # mean-reversion: low RSI → positive forecast


def rule_momentum(close: pd.Series, lookback: int = 252) -> float:
    """12-1 month momentum: ret over [-252, -21] (skipping last 21 days)."""
    if len(close) < lookback + 21:
        return float("nan")
    p_now = close.iloc[-21]
    p_then = close.iloc[-lookback - 21]
    return float((p_now / p_then - 1.0) * 100.0)


def rule_hht(close: pd.Series, ma: int = 60, ht: int = 30) -> float:
    """Simplified Hilbert filter: deviation from `ma` MA over `ht` window."""
    if len(close) < ma + ht:
        return float("nan")
    ma_series = close.rolling(ma).mean()
    dev = (close - ma_series) / ma_series
    smoothed = dev.rolling(ht).mean().iloc[-1]
    return float(smoothed * 100.0)


def rule_qrs(close: pd.Series, reg_w: int = 18, zscore_w: int = 250, threshold: float = 0.7) -> float:
    """Quantile regression slope vs threshold (simplified to OLS slope z-score).

    Real QRS uses statsmodels QuantReg; here we use an OLS slope as proxy.
    """
    if len(close) < zscore_w + reg_w:
        return float("nan")
    log_p = np.log(close.values)
    series = []
    for end in range(reg_w, len(log_p) + 1):
        x = np.arange(reg_w)
        y = log_p[end - reg_w:end]
        slope = np.polyfit(x, y, 1)[0]
        series.append(slope)
    s = pd.Series(series).tail(zscore_w)
    if s.std() == 0:
        return 0.0
    z = (s.iloc[-1] - s.mean()) / s.std()
    if abs(z) < threshold:
        return 0.0
    return float(z)


RULES: dict[str, Callable[[pd.Series], float]] = {
    "MA_cross": rule_ma_cross,
    "RSI": rule_rsi,
    "momentum": rule_momentum,
    "HHT": rule_hht,
    "QRS": rule_qrs,
}


class SingleNameTimingAlpha(Alpha, ScalingMixin):
    """Per-symbol timing rule.

    Args:
        symbols: list of symbols to trade.
        rule: one of 'MA_cross', 'RSI', 'momentum', 'HHT', 'QRS', or a callable.
        rule_kwargs: kwargs passed to the rule function.
        trigger_freq: when to evaluate (default EOD).
        lookback: how many bars of history to fetch from ctx.
        strategy_id: identifier.
    """

    min_history = 252

    def __init__(
        self,
        symbols: list[str],
        rule: str | Callable = "MA_cross",
        rule_kwargs: Optional[dict] = None,
        trigger_freq: Frequency = Frequency.EOD,
        lookback: int = 504,
        strategy_id: str = "timing",
    ):
        super().__init__()
        self._init_scaling(window=252)
        self.symbols = list(symbols)
        if isinstance(rule, str):
            if rule not in RULES:
                raise ValueError(f"Unknown rule {rule}; available: {list(RULES)}")
            self._rule = RULES[rule]
            self.rule_name = rule
        else:
            self._rule = rule
            self.rule_name = getattr(rule, "__name__", "custom")
        self.rule_kwargs = rule_kwargs or {}
        self.trigger_freq = trigger_freq
        self.lookback = lookback
        self.strategy_id = strategy_id

    def universe(self, dt, ctx):
        return list(self.symbols)

    def forecast(self, dt: pd.Timestamp, ctx) -> dict[str, float]:
        prices = ctx.as_of(dt, "price_1d", lookback=self.lookback, symbols=self.symbols)
        if prices is None or prices.empty:
            return {s: float("nan") for s in self.symbols}

        raw = {}
        for sym in self.symbols:
            if sym not in prices.columns:
                raw[sym] = float("nan")
                continue
            series = prices[sym].dropna()
            if len(series) < self.min_history:
                raw[sym] = float("nan")
                continue
            try:
                raw[sym] = float(self._rule(series, **self.rule_kwargs))
            except Exception:
                raw[sym] = float("nan")
        return self._scale_and_cap(raw)
