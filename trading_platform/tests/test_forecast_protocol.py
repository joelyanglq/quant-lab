"""Unit tests for forecast-protocol contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_platform.strategy.alpha import (
    Alpha, ScalingMixin, FORECAST_CAP, TARGET_ABS_FORECAST,
)
from trading_platform.core.events import Frequency


class DummyAlpha(Alpha, ScalingMixin):
    """Emits raw Gaussian values with user-defined mean/std."""
    trigger_freq = Frequency.EOD
    min_history = 1

    def __init__(self, mean=0.0, std=1.0, seed=0):
        super().__init__()
        self._init_scaling(window=252)
        self.rng = np.random.default_rng(seed)
        self.mean = mean
        self.std = std

    def forecast(self, dt, ctx):
        raw = {s: float(self.rng.normal(self.mean, self.std)) for s in ctx.universe(dt)}
        return self._scale_and_cap(raw)


class _Ctx:
    def __init__(self, syms): self._s = syms
    def universe(self, dt): return list(self._s)
    def as_of(self, dt, key, **kw): raise NotImplementedError


@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (0.0, 5.0), (0.0, 0.1), (1.0, 3.0)])
def test_forecast_scaling_in_target_range(mean, std):
    """After 500 bars, E[|forecast|] should be in [7, 13]."""
    alpha = DummyAlpha(mean=mean, std=std, seed=42)
    ctx = _Ctx(["A", "B", "C"])
    for i in range(500):
        alpha.forecast(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), ctx)
    valid, mean_abs = alpha.validate_scaling()
    assert valid, f"mean_abs={mean_abs} outside [7,13]"


def test_forecast_capped_at_20():
    """Extreme raw values must clip to ±20."""
    alpha = DummyAlpha(mean=0.0, std=100.0)
    ctx = _Ctx(["A"])
    for _ in range(300):
        out = alpha.forecast(pd.Timestamp("2024-01-01"), ctx)
        for v in out.values():
            if not np.isnan(v):
                assert -FORECAST_CAP <= v <= FORECAST_CAP


def test_nan_passes_through():
    """NaN raw should remain NaN after scaling."""
    alpha = DummyAlpha()
    alpha._raw_history.extend([1.0] * 100)
    out = alpha._scale_and_cap({"A": float("nan"), "B": 5.0})
    assert np.isnan(out["A"])
    assert not np.isnan(out["B"])
