"""
Mean reversion validation for ETF valuation metrics.

Tests whether ETF-level valuation metrics (PE, PB, PS, etc.) exhibit
mean-reverting behavior — the statistical foundation of the valuation
timing strategy.

Tests:
  1. ADF (Augmented Dickey-Fuller): H0 = unit root (non-stationary)
     Reject → stationary → mean-reverting
  2. Hurst exponent: H < 0.5 → mean-reverting, H = 0.5 → random walk, H > 0.5 → trending
  3. AR(1) half-life: How many periods until a deviation is halved
"""
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


def adf_test(series: pd.Series) -> Dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    Returns dict with test_statistic, p_value, is_stationary (at 5% level).
    """
    from statsmodels.tsa.stattools import adfuller

    clean = series.dropna()
    if len(clean) < 8:
        return {"test_statistic": np.nan, "p_value": np.nan, "is_stationary": False, "n_obs": len(clean)}

    result = adfuller(clean.values, maxlag=min(4, len(clean) // 4), autolag=None)
    return {
        "test_statistic": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < 0.05,
        "n_obs": len(clean),
    }


def hurst_exponent(series: pd.Series) -> float:
    """
    Estimate Hurst exponent via rescaled range (R/S) analysis.

    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending / persistent
    """
    clean = series.dropna().values
    n = len(clean)
    if n < 20:
        return np.nan

    max_k = min(n // 2, 64)
    lags = range(2, max_k + 1)
    rs_values = []

    for lag in lags:
        sub = clean[:lag]
        mean = np.mean(sub)
        deviations = np.cumsum(sub - mean)
        r = np.max(deviations) - np.min(deviations)
        s = np.std(sub, ddof=1)
        if s > 0:
            rs_values.append((lag, r / s))

    if len(rs_values) < 3:
        return np.nan

    log_lags = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])

    coeffs = np.polyfit(log_lags, log_rs, 1)
    return coeffs[0]


def ar1_halflife(series: pd.Series) -> float:
    """
    AR(1) half-life of mean reversion.

    Fits: x(t) - x(t-1) = phi * (x(t-1) - mean) + epsilon
    Half-life = -log(2) / log(1 + phi)

    Returns half-life in periods (quarters).
    """
    clean = series.dropna()
    if len(clean) < 5:
        return np.nan

    y = clean.diff().iloc[1:]
    x = (clean.iloc[:-1] - clean.mean()).values
    y = y.values

    if np.std(x) == 0:
        return np.nan

    phi = np.sum(x * y) / np.sum(x * x)

    if phi >= 0:
        return np.inf  # not mean-reverting

    half_life = -np.log(2) / np.log(1 + phi)
    return half_life


def validate_metric(
    series: pd.Series, name: str = ""
) -> Dict:
    """Run all three mean-reversion tests on a single metric series."""
    adf = adf_test(series)
    h = hurst_exponent(series)
    hl = ar1_halflife(series)

    is_mean_reverting = (
        adf["is_stationary"]
        or (not np.isnan(h) and h < 0.5)
        or (not np.isnan(hl) and not np.isinf(hl) and hl < 20)
    )

    return {
        "metric": name,
        "n_obs": adf["n_obs"],
        "adf_stat": adf["test_statistic"],
        "adf_pval": adf["p_value"],
        "adf_stationary": adf["is_stationary"],
        "hurst": h,
        "hurst_label": "mean-rev" if h < 0.45 else "random" if h < 0.55 else "trending" if not np.isnan(h) else "N/A",
        "halflife_q": hl,
        "halflife_years": hl / 4 if not np.isnan(hl) and not np.isinf(hl) else np.nan,
        "is_mean_reverting": is_mean_reverting,
    }


def validate_etf(history: pd.DataFrame, ticker: str) -> List[Dict]:
    """Validate mean reversion for all metrics of one ETF."""
    metrics_to_test = ["pe_ttm", "pb_lf", "ps_ttm", "div_yield", "fcf_yield", "ev_ebitda", "erp"]
    results = []
    for metric in metrics_to_test:
        if metric in history.columns:
            series = history[metric].dropna()
            if len(series) >= 8:
                r = validate_metric(series, metric)
                r["ticker"] = ticker
                results.append(r)
    return results


def run_validation(snapshots_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Run mean reversion validation on all reconstructed ETF histories.

    Returns DataFrame with test results per (ETF, metric).
    """
    if snapshots_dir is None:
        from etf_valuation.config import load_config
        config = load_config()
        snapshots_dir = config.get_storage_path("snapshots")

    print("=" * 90)
    print("  MEAN REVERSION VALIDATION")
    print("=" * 90)

    all_results = []
    for hist_file in sorted(snapshots_dir.glob("*_history.parquet")):
        ticker = hist_file.stem.replace("_history", "")
        df = pd.read_parquet(hist_file)
        if len(df) < 8:
            print(f"  {ticker:<5} — {len(df)} obs, too few for statistical tests")
            continue

        results = validate_etf(df, ticker)
        all_results.extend(results)

    if not all_results:
        print("  No ETFs with sufficient history for testing.")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_results)

    # Print formatted table
    print()
    print(f"  {'ETF':<6} {'Metric':<10} {'N':>3} {'ADF stat':>9} {'p-val':>7} {'Stat?':>5} {'Hurst':>6} {'Type':<8} {'HL(yr)':>7} {'MR?':>4}")
    print(f"  {'─'*6} {'─'*10} {'─'*3} {'─'*9} {'─'*7} {'─'*5} {'─'*6} {'─'*8} {'─'*7} {'─'*4}")

    for _, r in result_df.iterrows():
        adf_s = f"{r['adf_stat']:9.2f}" if not np.isnan(r['adf_stat']) else "      N/A"
        pval_s = f"{r['adf_pval']:7.3f}" if not np.isnan(r['adf_pval']) else "    N/A"
        stat_s = "  YES" if r['adf_stationary'] else "   no"
        hurst_s = f"{r['hurst']:6.3f}" if not np.isnan(r['hurst']) else "   N/A"
        hl_s = f"{r['halflife_years']:7.1f}" if not np.isnan(r['halflife_years']) else "    inf" if np.isinf(r['halflife_q']) else "    N/A"
        mr_s = " YES" if r['is_mean_reverting'] else "  no"

        print(f"  {r['ticker']:<6} {r['metric']:<10} {r['n_obs']:>3} {adf_s} {pval_s} {stat_s} {hurst_s} {r['hurst_label']:<8} {hl_s} {mr_s}")

    # Summary
    print()
    metrics = result_df["metric"].unique()
    print("  SUMMARY BY METRIC (across ETFs with 27Q history):")
    print(f"  {'Metric':<10} {'#ETFs':>5} {'ADF pass%':>9} {'Avg Hurst':>10} {'Avg HL(yr)':>11} {'Mean-Rev%':>10}")
    print(f"  {'─'*10} {'─'*5} {'─'*9} {'─'*10} {'─'*11} {'─'*10}")

    for metric in metrics:
        sub = result_df[(result_df["metric"] == metric) & (result_df["n_obs"] >= 20)]
        if sub.empty:
            continue
        n = len(sub)
        adf_pass = sub["adf_stationary"].mean() * 100
        avg_hurst = sub["hurst"].mean()
        valid_hl = sub[sub["halflife_years"].notna() & ~np.isinf(sub["halflife_q"])]
        avg_hl = valid_hl["halflife_years"].mean() if not valid_hl.empty else np.nan
        mr_pct = sub["is_mean_reverting"].mean() * 100

        hl_s = f"{avg_hl:11.1f}" if not np.isnan(avg_hl) else "        N/A"
        print(f"  {metric:<10} {n:>5} {adf_pass:>8.0f}% {avg_hurst:>10.3f} {hl_s} {mr_pct:>9.0f}%")

    return result_df


if __name__ == "__main__":
    run_validation()
