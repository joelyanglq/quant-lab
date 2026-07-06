"""
Historical ETF metric reconstruction.

Joins quarterly N-PORT holdings weights with per-stock ratios from
edgar_financials to reconstruct ETF-level valuation metrics over time.

Data sources:
  - N-PORT holdings: E:/stocks/etf_valuation/nport/{TICKER}/{YEAR}Q{N}.parquet
  - Per-stock ratios: E:/raw/edgar_financials/ratios/{period}_{symbol}.json
  - CUSIP→ticker cache: E:/stocks/etf_valuation/cusip_cache/cusip_to_ticker.parquet
  - Treasury rates: E:/stocks/treasury/treasury_rates.parquet

Output:
  - E:/stocks/etf_valuation/snapshots/{TICKER}_history.parquet
    columns: date, pe_ttm, pb_lf, ps_ttm, div_yield, fcf_yield, ev_ebitda, erp, coverage
"""
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from etf_valuation.config import METRICS, load_config
from etf_valuation.metrics import weighted_harmonic_mean, weighted_arithmetic_mean

logger = logging.getLogger(__name__)

EDGAR_FIELD_MAP = {
    "pe_ttm": "priceToEarningsRatio",
    "pb_lf": "priceToBookRatio",
    "ps_ttm": "priceToSalesRatio",
    "div_yield": "dividendYield",
    "fcf_yield": "priceToFreeCashFlowRatio",
    "ev_ebitda": "enterpriseValueMultiple",
}

EDGAR_RATIOS_DIR = Path("E:/raw/edgar_financials/ratios")


class EdgarRatiosLoader:
    """Loads per-stock ratios from edgar_financials JSON files."""

    def __init__(self, ratios_dir: Path = EDGAR_RATIOS_DIR):
        self.ratios_dir = ratios_dir
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """Load all quarterly ratio records for a symbol, sorted by date."""
        if symbol in self._cache:
            return self._cache[symbol]

        records = []
        for period in ["Q1", "Q2", "Q3", "Q4", "annual"]:
            fpath = self.ratios_dir / f"{period}_{symbol}.json"
            if fpath.exists():
                with open(fpath, "r") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    continue
                for r in data:
                    if isinstance(r, dict):
                        records.append(r)

        if not records:
            self._cache[symbol] = pd.DataFrame()
            return self._cache[symbol]

        df = pd.DataFrame(records)
        if "date" not in df.columns:
            self._cache[symbol] = pd.DataFrame()
            return self._cache[symbol]

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        self._cache[symbol] = df
        return df

    def get_ratios_at_date(
        self, symbol: str, target_date: date, max_days: int = 120
    ) -> Optional[Dict[str, float]]:
        """Get the closest ratio record for a symbol near a target date."""
        df = self.load_symbol(symbol)
        if df.empty:
            return None

        target = pd.Timestamp(target_date)
        deltas = (df["date"] - target).abs()
        idx = deltas.idxmin()
        if deltas[idx].days > max_days:
            return None

        row = df.loc[idx]
        result = {}
        for metric_key, edgar_field in EDGAR_FIELD_MAP.items():
            val = row.get(edgar_field)
            if val is not None and pd.notna(val):
                result[metric_key] = float(val)
        return result


def reconstruct_quarter(
    holdings: pd.DataFrame,
    cusip_cache: pd.DataFrame,
    edgar_loader: EdgarRatiosLoader,
    report_date: date,
    treasury_10y: Optional[float] = None,
) -> Dict[str, float]:
    """
    Reconstruct ETF-level metrics for one quarter.

    Args:
        holdings: N-PORT holdings with cusip, weight columns
        cusip_cache: CUSIP→ticker mapping
        edgar_loader: EdgarRatiosLoader instance
        report_date: quarter-end date from N-PORT
        treasury_10y: 10Y Treasury rate at that date

    Returns:
        {metric_key: value, metric_key_coverage: float, ...}
    """
    if "ticker" not in holdings.columns:
        holdings = holdings.merge(cusip_cache, on="cusip", how="left")

    mapped = holdings[holdings["ticker"].notna()].copy()
    if mapped.empty:
        return {}

    metric_keys = list(EDGAR_FIELD_MAP.keys())
    ratio_rows = []
    for _, row in mapped.iterrows():
        ticker = row["ticker"]
        ratios = edgar_loader.get_ratios_at_date(ticker, report_date)
        if ratios:
            ratios["ticker"] = ticker
            ratio_rows.append(ratios)

    if not ratio_rows:
        return {}

    ratio_df = pd.DataFrame(ratio_rows)
    merged = mapped.merge(ratio_df, on="ticker", how="left")

    result = {"date": report_date}
    weights = merged["weight"].values.astype(float)

    for metric_key in metric_keys:
        if metric_key not in merged.columns:
            result[metric_key] = np.nan
            result[f"{metric_key}_coverage"] = 0.0
            continue

        vals = merged[metric_key].values.astype(float)
        meta = METRICS.get(metric_key, {})
        agg = meta.get("agg", "harmonic")

        if metric_key == "fcf_yield":
            # P/FCF → FCF Yield = 1/P_FCF
            mask = (vals > 0) & np.isfinite(vals) & (weights > 0)
            if mask.any():
                fcf_yields = 1.0 / vals[mask]
                w = weights[mask]
                total_w = weights.sum()
                coverage = w.sum() / total_w if total_w > 0 else 0.0
                value = np.average(fcf_yields, weights=w)
            else:
                value, coverage = np.nan, 0.0
        elif agg == "harmonic":
            value, coverage, _ = weighted_harmonic_mean(vals, weights)
        else:
            value, coverage, _ = weighted_arithmetic_mean(vals, weights)

        result[metric_key] = value
        result[f"{metric_key}_coverage"] = coverage

    # ERP = 1/PE - treasury
    pe = result.get("pe_ttm")
    if pe and not np.isnan(pe) and pe > 0 and treasury_10y is not None:
        result["erp"] = 1.0 / pe - treasury_10y
    else:
        result["erp"] = np.nan

    # Overall coverage = weight of tickers that had edgar data
    tickers_with_data = set(ratio_df["ticker"].unique())
    matched_weight = merged.loc[merged["ticker"].isin(tickers_with_data), "weight"].sum()
    total_weight = merged["weight"].sum()
    result["coverage"] = matched_weight / total_weight if total_weight > 0 else 0.0

    return result


def load_treasury_rates(treasury_path: Optional[Path] = None) -> pd.DataFrame:
    """Load treasury rates for historical ERP calculation."""
    if treasury_path is None:
        treasury_path = Path("E:/stocks/treasury/treasury_rates.parquet")
    if not treasury_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(treasury_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df


def get_treasury_at_date(treasury_df: pd.DataFrame, target_date: date) -> Optional[float]:
    """Get the 10Y treasury rate closest to a target date."""
    if treasury_df.empty:
        return None

    # Look for 'year10' or 'month120' column
    rate_col = None
    for col in ["year10", "month120", "10Y"]:
        if col in treasury_df.columns:
            rate_col = col
            break
    if rate_col is None:
        return None

    target = pd.Timestamp(target_date)
    idx = treasury_df.index.get_indexer([target], method="nearest")[0]
    if idx < 0 or idx >= len(treasury_df):
        return None

    val = treasury_df.iloc[idx][rate_col]
    if pd.isna(val):
        return None
    # Treasury rates are in percentage (e.g. 4.5 = 4.5%), convert to decimal
    return float(val) / 100.0 if float(val) > 1.0 else float(val)


def reconstruct_etf_history(
    ticker: str,
    nport_dir: Path,
    cusip_cache: pd.DataFrame,
    edgar_loader: EdgarRatiosLoader,
    treasury_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct full quarterly history for one ETF.

    Returns DataFrame with quarterly metric values, indexed by date.
    """
    etf_dir = nport_dir / ticker
    if not etf_dir.exists():
        return pd.DataFrame()

    quarter_files = sorted(etf_dir.glob("*.parquet"))
    if not quarter_files:
        return pd.DataFrame()

    rows = []
    for qf in quarter_files:
        holdings = pd.read_parquet(qf)
        report_date = holdings["report_date"].iloc[0]
        if isinstance(report_date, str):
            report_date = datetime.strptime(report_date[:10], "%Y-%m-%d").date()
        elif hasattr(report_date, "date"):
            report_date = report_date.date()

        treasury_10y = get_treasury_at_date(treasury_df, report_date)

        row = reconstruct_quarter(
            holdings, cusip_cache, edgar_loader, report_date, treasury_10y
        )
        if row:
            rows.append(row)
            logger.info(
                "  %s %s: PE=%.1f PB=%.1f cov=%.0f%%",
                ticker, qf.stem,
                row.get("pe_ttm", float("nan")),
                row.get("pb_lf", float("nan")),
                row.get("coverage", 0) * 100,
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


def append_today_snapshot(
    ticker: str,
    today_metrics: Dict[str, float],
    snapshots_dir: Path,
) -> None:
    """Append today's computed metrics to the ETF history file."""
    out_path = snapshots_dir / f"{ticker}_history.parquet"
    today = date.today()

    new_row = pd.DataFrame([{
        "date": pd.Timestamp(today),
        **{k: v for k, v in today_metrics.items() if k != "date"},
    }]).set_index("date")

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        if "date" in existing.columns:
            existing = existing.set_index("date")
        # Dedup: drop today if already present
        existing = existing[existing.index != pd.Timestamp(today)]
        combined = pd.concat([existing, new_row]).sort_index()
    else:
        combined = new_row

    combined.to_parquet(out_path, compression="snappy")


def run_reconstruction(
    tickers: Optional[List[str]] = None,
    force: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Reconstruct historical ETF metrics for all specified tickers.

    Saves results to snapshots/{TICKER}_history.parquet.

    Returns dict of {ticker: history_df}.
    """
    config = load_config()
    nport_dir = config.get_storage_path("nport")
    snapshots_dir = config.get_storage_path("snapshots")

    if tickers is None:
        from etf_valuation.run_backfill import PRIORITY_TICKERS
        tickers = PRIORITY_TICKERS

    # Load shared resources
    cusip_cache = pd.read_parquet(
        config.get_storage_path("cusip_cache") / "cusip_to_ticker.parquet"
    )
    edgar_loader = EdgarRatiosLoader()
    treasury_df = load_treasury_rates()

    print("=" * 70)
    print("HISTORICAL ETF METRIC RECONSTRUCTION")
    print(f"  edgar_financials: {len(list(EDGAR_RATIOS_DIR.glob('*.json')))} ratio files")
    print(f"  CUSIP cache: {len(cusip_cache)} mappings")
    print(f"  Treasury rates: {len(treasury_df)} records")
    print("=" * 70)

    results = {}
    for i, ticker in enumerate(tickers, 1):
        out_path = snapshots_dir / f"{ticker}_history.parquet"
        if out_path.exists() and not force:
            print(f"  [{i}/{len(tickers)}] {ticker:<5} — already exists, skipping")
            results[ticker] = pd.read_parquet(out_path)
            continue

        etf_dir = nport_dir / ticker
        if not etf_dir.exists():
            print(f"  [{i}/{len(tickers)}] {ticker:<5} — no N-PORT data")
            continue

        n_quarters = len(list(etf_dir.glob("*.parquet")))
        print(f"  [{i}/{len(tickers)}] {ticker:<5} — {n_quarters} quarters...", end=" ", flush=True)

        history = reconstruct_etf_history(
            ticker, nport_dir, cusip_cache, edgar_loader, treasury_df
        )

        if history.empty:
            print("no data")
            continue

        history.to_parquet(out_path, compression="snappy")
        results[ticker] = history

        pe_range = f"PE {history['pe_ttm'].min():.1f}-{history['pe_ttm'].max():.1f}" if "pe_ttm" in history else ""
        cov_avg = history["coverage"].mean() * 100 if "coverage" in history else 0
        print(f"{len(history)} quarters, {pe_range}, avg cov {cov_avg:.0f}%")

    return results


def print_history_summary(results: Dict[str, pd.DataFrame]) -> None:
    """Print summary table of reconstructed histories."""
    print()
    print("=" * 90)
    print("  HISTORICAL RECONSTRUCTION SUMMARY")
    print("=" * 90)
    print(f"  {'ETF':<6} {'Quarters':>8} {'Date Range':<25} {'PE Range':<16} {'PB Range':<16} {'Avg Cov':>8}")
    print(f"  {'─'*6} {'─'*8} {'─'*25} {'─'*16} {'─'*16} {'─'*8}")

    for ticker, df in sorted(results.items()):
        if df.empty:
            continue
        n = len(df)
        dr = f"{df.index[0].strftime('%Y-%m')} → {df.index[-1].strftime('%Y-%m')}"
        pe_min = df["pe_ttm"].min() if "pe_ttm" in df else np.nan
        pe_max = df["pe_ttm"].max() if "pe_ttm" in df else np.nan
        pb_min = df["pb_lf"].min() if "pb_lf" in df else np.nan
        pb_max = df["pb_lf"].max() if "pb_lf" in df else np.nan
        cov = df["coverage"].mean() * 100 if "coverage" in df else 0

        pe_s = f"{pe_min:.1f}–{pe_max:.1f}" if not np.isnan(pe_min) else "N/A"
        pb_s = f"{pb_min:.1f}–{pb_max:.1f}" if not np.isnan(pb_min) else "N/A"

        print(f"  {ticker:<6} {n:>8} {dr:<25} {pe_s:<16} {pb_s:<16} {cov:>7.0f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    results = run_reconstruction(force=True)
    print_history_summary(results)
