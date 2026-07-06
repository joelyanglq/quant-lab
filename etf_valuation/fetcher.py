"""
Data fetchers for ETF valuation system.

- FMP /ratios-ttm-bulk: daily snapshot of all stock ratios (PE, PB, PS, DivYield, etc.)
- yfinance: ETF-level P/E, P/B, Div Yield for cross-validation
"""
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

RATIO_FIELDS = [
    "symbol",
    "priceToEarningsRatioTTM",
    "priceToBookRatioTTM",
    "priceToSalesRatioTTM",
    "dividendYieldTTM",
    "enterpriseValueMultipleTTM",
    "priceToFreeCashFlowRatioTTM",
]


class RatiosBulkFetcher:
    """Fetches FMP /ratios-ttm-bulk and stores daily snapshots."""

    def __init__(self, storage_dir: Path, fmp_api_key: str):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._api_key = fmp_api_key

    def fetch_today(self) -> pd.DataFrame:
        """Fetch today's bulk ratios from FMP."""
        from data_sync.client.fmp import FMPClient

        client = FMPClient(api_key=self._api_key)
        try:
            df = client.get_csv("/ratios-ttm-bulk")
        finally:
            client.close()

        available_cols = [c for c in RATIO_FIELDS if c in df.columns]
        df = df[available_cols].copy()

        df["fetch_date"] = date.today().isoformat()
        return df

    def save_snapshot(self, df: pd.DataFrame, snapshot_date: Optional[date] = None):
        """Save a ratios snapshot, appending to the year-partitioned file."""
        if snapshot_date is None:
            snapshot_date = date.today()

        year_file = self.storage_dir / f"{snapshot_date.year}_ratios_ttm.parquet"

        if year_file.exists():
            existing = pd.read_parquet(year_file)
            existing = existing[existing["fetch_date"] != snapshot_date.isoformat()]
            df = pd.concat([existing, df], ignore_index=True)

        df.to_parquet(year_file, compression="snappy", index=False)
        logger.info("Saved ratios snapshot: %s (%d rows)", snapshot_date, len(df))

    def sync(self) -> int:
        """Fetch and save today's ratios. Returns row count."""
        today = date.today()
        year_file = self.storage_dir / f"{today.year}_ratios_ttm.parquet"

        if year_file.exists():
            existing = pd.read_parquet(year_file)
            if today.isoformat() in existing["fetch_date"].values:
                logger.info("Ratios already fetched for %s", today)
                return 0

        df = self.fetch_today()
        self.save_snapshot(df, today)
        return len(df)

    def load_snapshot(self, snapshot_date: date) -> pd.DataFrame:
        """Load ratios for a specific date."""
        year_file = self.storage_dir / f"{snapshot_date.year}_ratios_ttm.parquet"
        if not year_file.exists():
            return pd.DataFrame()

        df = pd.read_parquet(year_file)
        return df[df["fetch_date"] == snapshot_date.isoformat()]

    def load_latest(self) -> pd.DataFrame:
        """Load the most recent ratios snapshot."""
        files = sorted(self.storage_dir.glob("*_ratios_ttm.parquet"), reverse=True)
        if not files:
            return pd.DataFrame()

        df = pd.read_parquet(files[0])
        if df.empty:
            return df

        latest_date = df["fetch_date"].max()
        return df[df["fetch_date"] == latest_date]

    def available_dates(self) -> List[str]:
        """List all dates with stored snapshots."""
        dates = []
        for f in sorted(self.storage_dir.glob("*_ratios_ttm.parquet")):
            df = pd.read_parquet(f, columns=["fetch_date"])
            dates.extend(df["fetch_date"].unique().tolist())
        return sorted(set(dates))


class FMPHoldingsFetcher:
    """Fetch current ETF holdings from FMP /stable/etf/holdings."""

    def __init__(self, fmp_api_key: str):
        self._api_key = fmp_api_key

    def fetch(self, ticker: str) -> pd.DataFrame:
        """Fetch current holdings for one ETF.

        Returns DataFrame with columns: ticker, weight, name, cusip, isin,
        shares, market_value.
        """
        import requests

        url = (
            f"https://financialmodelingprep.com/stable/etf/holdings"
            f"?symbol={ticker}&apikey={self._api_key}"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list) or not data:
            logger.warning("FMP etf/holdings returned empty for %s", ticker)
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.rename(columns={
            "asset": "ticker",
            "weightPercentage": "weight",
            "securityCusip": "cusip",
            "sharesNumber": "shares",
            "marketValue": "market_value",
        })

        keep = ["ticker", "weight", "name", "cusip", "isin", "shares", "market_value"]
        df = df[[c for c in keep if c in df.columns]]

        # Normalize: empty ticker → NaN
        if "ticker" in df.columns:
            df.loc[df["ticker"].str.strip() == "", "ticker"] = pd.NA

        return df

    def fetch_all(
        self,
        tickers: list[str],
        delay: float = 0.2,
    ) -> dict[str, pd.DataFrame]:
        """Fetch holdings for multiple ETFs.

        Returns {ticker: holdings_df}.
        """
        import time

        results = {}
        for i, ticker in enumerate(tickers, 1):
            try:
                df = self.fetch(ticker)
                if not df.empty:
                    results[ticker] = df
                    n = len(df)
                    wt = df["weight"].sum() if "weight" in df.columns else 0
                    logger.info(
                        "  [%d/%d] %s: %d holdings, wt=%.1f%%",
                        i, len(tickers), ticker, n, wt,
                    )
            except Exception as e:
                logger.warning("  [%d/%d] %s: %s", i, len(tickers), ticker, e)

            if i < len(tickers) and delay > 0:
                time.sleep(delay)

        return results


class YFinanceValidator:
    """Cross-validate ETF valuations against yfinance."""

    def fetch_etf_ratios(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch current P/E, P/B, Div Yield from yfinance for ETFs."""
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed, skipping validation")
            return pd.DataFrame()

        rows = []
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                info = t.info
                rows.append({
                    "ticker": ticker,
                    "yf_pe": info.get("trailingPE"),
                    "yf_pb": info.get("priceToBook"),
                    "yf_div_yield": info.get("trailingAnnualDividendYield"),
                })
            except Exception as e:
                logger.warning("yfinance %s: %s", ticker, e)
                rows.append({"ticker": ticker})

        return pd.DataFrame(rows)

    def compare(
        self,
        computed: Dict[str, Dict[str, float]],
        tickers: List[str],
    ) -> pd.DataFrame:
        """
        Compare computed ETF valuations with yfinance values.

        Args:
            computed: {ticker: {pe_ttm: X, pb_lf: Y, ...}}
            tickers: list of ETF tickers to validate

        Returns:
            DataFrame with computed vs yfinance values and pct difference
        """
        yf_df = self.fetch_etf_ratios(tickers)
        if yf_df.empty:
            return pd.DataFrame()

        rows = []
        for _, yf_row in yf_df.iterrows():
            ticker = yf_row["ticker"]
            comp = computed.get(ticker, {})

            row = {"ticker": ticker}
            for metric, yf_col in [("pe_ttm", "yf_pe"), ("pb_lf", "yf_pb"), ("div_yield", "yf_div_yield")]:
                comp_val = comp.get(metric)
                yf_val = yf_row.get(yf_col)
                row[f"{metric}_computed"] = comp_val
                row[f"{metric}_yfinance"] = yf_val
                if comp_val and yf_val and yf_val != 0:
                    row[f"{metric}_diff_pct"] = abs(comp_val - yf_val) / yf_val * 100
            rows.append(row)

        return pd.DataFrame(rows)
