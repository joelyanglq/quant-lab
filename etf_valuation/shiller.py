"""
Shiller/Yale S&P 500 historical data loader.

Robert Shiller's dataset provides monthly S&P 500 data from 1871:
  - Price, Earnings, P/E ratio
  - CAPE (Cyclically Adjusted P/E, aka Shiller P/E)
  - 10-Year Treasury rate
  - CPI, Real Price, Real Earnings

Source: http://www.econ.yale.edu/~shiller/data/ie_data.xls
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"


class ShillerLoader:
    """Download and parse Shiller's S&P 500 historical dataset."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._parquet_path = storage_dir / "sp500_shiller.parquet"

    def download_and_parse(self) -> pd.DataFrame:
        """Download Shiller Excel and parse into clean DataFrame."""
        logger.info("Downloading Shiller dataset from %s", SHILLER_URL)

        df = pd.read_excel(
            SHILLER_URL,
            sheet_name="Data",
            skiprows=range(7),
            header=0,
        )

        # The columns are typically:
        # Date, S&P Comp. Price, Dividend, Earnings, CPI,
        # Date Fraction, Long Interest Rate GS10, Real Price,
        # Real Dividend, Real Total Return Price, Real Earnings,
        # Real TR Scaled Earnings, CAPE, TR CAPE, ...
        # But column names are messy, so use positional
        col_map = {}
        cols = list(df.columns)

        # First column is the date (year.month format like 2023.01)
        col_map[cols[0]] = "date_frac"
        if len(cols) > 1:
            col_map[cols[1]] = "price"
        if len(cols) > 2:
            col_map[cols[2]] = "dividend"
        if len(cols) > 3:
            col_map[cols[3]] = "earnings"
        if len(cols) > 4:
            col_map[cols[4]] = "cpi"
        if len(cols) > 6:
            col_map[cols[6]] = "gs10"  # 10-year treasury rate
        if len(cols) > 7:
            col_map[cols[7]] = "real_price"
        if len(cols) > 10:
            col_map[cols[10]] = "real_earnings"

        # Find CAPE column (usually has "CAPE" in header)
        for i, c in enumerate(cols):
            if isinstance(c, str) and "CAPE" in c.upper() and "TR" not in c.upper():
                col_map[c] = "cape"
                break

        df = df.rename(columns=col_map)
        keep_cols = [c for c in ["date_frac", "price", "dividend", "earnings",
                                  "cpi", "gs10", "real_price", "real_earnings", "cape"]
                     if c in df.columns]
        df = df[keep_cols].copy()

        # Parse date fraction (e.g. 2023.01 → 2023-01-01)
        df = df.dropna(subset=["date_frac"])
        df = df[df["date_frac"].apply(lambda x: isinstance(x, (int, float)))]
        df["year"] = df["date_frac"].astype(float).astype(int)
        df["month"] = ((df["date_frac"].astype(float) % 1) * 100).round().astype(int).clip(1, 12)
        df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
        df = df.drop(columns=["date_frac", "year", "month"])

        # Compute P/E
        for col in ["price", "earnings", "gs10", "cape"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "price" in df.columns and "earnings" in df.columns:
            df["pe"] = df["price"] / df["earnings"]

        # Compute ERP (Earnings Yield - 10Y rate)
        if "pe" in df.columns and "gs10" in df.columns:
            df["earnings_yield"] = 1.0 / df["pe"]
            df["erp"] = df["earnings_yield"] - df["gs10"] / 100.0

        df = df.set_index("date").sort_index()
        df = df[df.index.year >= 1871]

        return df

    def sync(self) -> int:
        """Download, parse, and save Shiller data."""
        df = self.download_and_parse()
        df.to_parquet(self._parquet_path, compression="snappy")
        logger.info("Saved Shiller data: %d rows (%s to %s)",
                     len(df), df.index.min().date(), df.index.max().date())
        return len(df)

    def load(self) -> pd.DataFrame:
        """Load saved Shiller data."""
        if not self._parquet_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self._parquet_path)

    def get_spy_pe_history(self) -> pd.DataFrame:
        """Get S&P 500 P/E time series for SPY historical extension."""
        df = self.load()
        if df.empty:
            return df
        keep = ["pe", "cape", "earnings_yield", "erp", "gs10"]
        return df[[c for c in keep if c in df.columns]]
