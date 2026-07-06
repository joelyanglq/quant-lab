"""
Valuation metric aggregation for ETFs.

Computes ETF-level valuation metrics from individual stock ratios
and ETF holdings weights.

Aggregation methods:
  - Weighted harmonic mean: for ratio metrics (PE, PB, PS, EV/EBITDA)
    Harmonic mean is the industry standard for index P/E because
    1/avg(1/PE_i) = TotalMarketCap / TotalEarnings
  - Weighted arithmetic mean: for yield metrics (DivYield, FCF Yield)
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from etf_valuation.config import METRICS

logger = logging.getLogger(__name__)

MIN_COVERAGE = 0.70  # minimum weight coverage for reliable metric


def weighted_harmonic_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Weighted harmonic mean, excluding non-positive values.

    Returns:
        (result, coverage, n_valid)
        coverage = sum of included weights / sum of all weights
    """
    mask = (values > 0) & np.isfinite(values) & (weights > 0)
    if not mask.any():
        return np.nan, 0.0, 0

    v = values[mask]
    w = weights[mask]
    total_w = weights.sum()
    coverage = w.sum() / total_w if total_w > 0 else 0.0

    # harmonic mean = sum(w) / sum(w/v)
    result = w.sum() / (w / v).sum()
    return result, coverage, int(mask.sum())


def weighted_arithmetic_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Weighted arithmetic mean, excluding NaN/Inf values.

    Returns:
        (result, coverage, n_valid)
    """
    mask = np.isfinite(values) & (weights > 0)
    if not mask.any():
        return np.nan, 0.0, 0

    v = values[mask]
    w = weights[mask]
    total_w = weights.sum()
    coverage = w.sum() / total_w if total_w > 0 else 0.0

    result = np.average(v, weights=w)
    return result, coverage, int(mask.sum())


def aggregate_etf_metrics(
    holdings: pd.DataFrame,
    ratios: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    treasury_10y: Optional[float] = None,
) -> Dict[str, Dict]:
    """
    Compute ETF-level valuation metrics from holdings + stock ratios.

    Args:
        holdings: DataFrame with columns [ticker, weight]
        ratios: DataFrame with columns [symbol, priceToEarningsRatioTTM, ...]
        metrics: list of metric keys to compute (default: all)
        treasury_10y: 10-year Treasury rate (for ERP calculation)

    Returns:
        {metric_key: {"value": float, "coverage": float, "n_valid": int, "reliable": bool}}
    """
    if metrics is None:
        metrics = list(METRICS.keys())

    # Join holdings with ratios on ticker
    merged = holdings.merge(
        ratios.rename(columns={"symbol": "ticker"}),
        on="ticker",
        how="left",
    )

    results = {}
    for metric_key in metrics:
        meta = METRICS.get(metric_key)
        if not meta:
            continue

        if metric_key == "erp":
            # ERP = Earnings Yield - 10Y Treasury
            pe_result = _compute_single_metric(merged, "pe_ttm")
            if pe_result["value"] is not None and not np.isnan(pe_result["value"]) and treasury_10y is not None:
                earnings_yield = 1.0 / pe_result["value"]
                erp = earnings_yield - treasury_10y
                results["erp"] = {
                    "value": erp,
                    "coverage": pe_result["coverage"],
                    "n_valid": pe_result["n_valid"],
                    "reliable": pe_result["coverage"] >= MIN_COVERAGE,
                }
            else:
                results["erp"] = {"value": np.nan, "coverage": 0.0, "n_valid": 0, "reliable": False}
            continue

        if metric_key == "fcf_yield":
            # FCF Yield = 1 / Price-to-FCF ratio
            fmp_field = meta["fmp_field"]
            if fmp_field not in merged.columns:
                results[metric_key] = {"value": np.nan, "coverage": 0.0, "n_valid": 0, "reliable": False}
                continue

            vals = merged[fmp_field].values.astype(float)
            ws = merged["weight"].values.astype(float)

            # Convert P/FCF to FCF Yield (1/P_FCF) then aggregate with arithmetic mean
            mask = (vals > 0) & np.isfinite(vals) & (ws > 0)
            if mask.any():
                fcf_yields = 1.0 / vals[mask]
                w = ws[mask]
                total_w = ws.sum()
                coverage = w.sum() / total_w if total_w > 0 else 0.0
                value = np.average(fcf_yields, weights=w)
                results[metric_key] = {
                    "value": value,
                    "coverage": coverage,
                    "n_valid": int(mask.sum()),
                    "reliable": coverage >= MIN_COVERAGE,
                }
            else:
                results[metric_key] = {"value": np.nan, "coverage": 0.0, "n_valid": 0, "reliable": False}
            continue

        results[metric_key] = _compute_single_metric(merged, metric_key)

    return results


def _compute_single_metric(merged: pd.DataFrame, metric_key: str) -> dict:
    """Compute a single metric from merged holdings+ratios data."""
    meta = METRICS.get(metric_key)
    if not meta or not meta.get("fmp_field"):
        return {"value": np.nan, "coverage": 0.0, "n_valid": 0, "reliable": False}

    fmp_field = meta["fmp_field"]
    if fmp_field not in merged.columns:
        return {"value": np.nan, "coverage": 0.0, "n_valid": 0, "reliable": False}

    vals = merged[fmp_field].values.astype(float)
    ws = merged["weight"].values.astype(float)

    agg_method = meta.get("agg", "harmonic")
    if agg_method == "harmonic":
        value, coverage, n_valid = weighted_harmonic_mean(vals, ws)
    else:
        value, coverage, n_valid = weighted_arithmetic_mean(vals, ws)

    return {
        "value": value,
        "coverage": coverage,
        "n_valid": n_valid,
        "reliable": coverage >= MIN_COVERAGE,
    }


def compute_all_etfs(
    etf_holdings: Dict[str, pd.DataFrame],
    ratios: pd.DataFrame,
    etf_metrics: Dict[str, List[str]],
    treasury_10y: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute valuation metrics for all ETFs.

    Args:
        etf_holdings: {ticker: holdings_df with [ticker, weight]}
        ratios: stock-level ratios DataFrame
        etf_metrics: {etf_ticker: [metric_keys to compute]}
        treasury_10y: 10-year Treasury rate

    Returns:
        DataFrame indexed by ETF ticker with metric values as columns
    """
    rows = []
    for etf_ticker, metrics_list in etf_metrics.items():
        holdings = etf_holdings.get(etf_ticker)
        if holdings is None or holdings.empty:
            continue

        result = aggregate_etf_metrics(
            holdings, ratios, metrics=metrics_list, treasury_10y=treasury_10y
        )

        row = {"etf": etf_ticker}
        for metric_key, data in result.items():
            row[metric_key] = data["value"]
            row[f"{metric_key}_coverage"] = data["coverage"]
            row[f"{metric_key}_reliable"] = data["reliable"]
        rows.append(row)

    return pd.DataFrame(rows).set_index("etf") if rows else pd.DataFrame()
