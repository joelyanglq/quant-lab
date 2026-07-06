"""
Time-series rolling percentile for ETF valuation metrics.

Tracks where the current value of a metric sits relative to its OWN history
(not cross-sectional vs peers). This is the core of the valuation timing strategy.
"""
import numpy as np
import pandas as pd


def rolling_percentile(
    series: pd.Series,
    window_years: int = 5,
    min_obs: int = 60,
) -> pd.Series:
    """
    Compute rolling percentile of a time series vs its trailing window.

    For each point, calculates: what fraction of observations in the
    trailing window are <= the current value.

    Args:
        series: time series (DatetimeIndex, daily or monthly frequency)
        window_years: lookback window in years
        min_obs: minimum observations required for valid percentile

    Returns:
        Series of percentile values in [0, 1]
    """
    window_days = window_years * 252  # trading days
    result = pd.Series(np.nan, index=series.index)

    values = series.values
    n = len(values)

    for i in range(min_obs, n):
        start = max(0, i - window_days)
        window = values[start:i+1]
        valid = window[np.isfinite(window)]

        if len(valid) < min_obs:
            continue

        current = values[i]
        if not np.isfinite(current):
            continue

        result.iloc[i] = np.mean(valid <= current)

    return result


def expanding_percentile(
    series: pd.Series,
    min_obs: int = 60,
) -> pd.Series:
    """
    Compute expanding (all-history) percentile.

    Uses all available history up to each point, not a fixed window.
    Better when history is short (< 5 years).

    Args:
        series: time series
        min_obs: minimum observations required

    Returns:
        Series of percentile values in [0, 1]
    """
    result = pd.Series(np.nan, index=series.index)
    values = series.values
    n = len(values)

    for i in range(min_obs, n):
        window = values[:i+1]
        valid = window[np.isfinite(window)]

        if len(valid) < min_obs:
            continue

        current = values[i]
        if not np.isfinite(current):
            continue

        result.iloc[i] = np.mean(valid <= current)

    return result


def compute_percentiles(
    etf_history: pd.DataFrame,
    metric_col: str,
    window_years: int = 5,
    min_obs: int = 60,
) -> pd.Series:
    """
    Compute percentile for an ETF metric time series.

    Automatically chooses rolling vs expanding based on available history:
    - If history >= window_years: use rolling percentile
    - If history < window_years but >= min_obs: use expanding percentile

    Args:
        etf_history: DataFrame with DatetimeIndex
        metric_col: column name for the metric
        window_years: preferred lookback window
        min_obs: minimum observations

    Returns:
        Series of percentile values
    """
    if metric_col not in etf_history.columns:
        return pd.Series(np.nan, index=etf_history.index)

    series = etf_history[metric_col].dropna()
    if len(series) < min_obs:
        return pd.Series(np.nan, index=etf_history.index)

    available_years = (series.index[-1] - series.index[0]).days / 365.25
    if available_years >= window_years:
        pct = rolling_percentile(series, window_years, min_obs)
    else:
        pct = expanding_percentile(series, min_obs)

    return pct.reindex(etf_history.index)
