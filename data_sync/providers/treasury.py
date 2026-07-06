"""
Treasury Rates Provider — US Treasury yield curve

FMP endpoint: /treasury
Returns daily treasury rates for multiple maturities.
"""
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import pandas as pd

from data_sync.client.fmp import FMPClient
from data_sync.config import ETLConfig
from data_sync.providers.base import Provider
from data_sync.providers import register

if TYPE_CHECKING:
    from data_sync.status import StatusManager

logger = logging.getLogger(__name__)


class FMPTreasuryProvider(Provider):
    name = "treasury"

    def __init__(self, config: ETLConfig):
        super().__init__(config)
        self._storage_dir = config.get_storage_path("treasury")
        self._client = FMPClient(
            api_key=config.get_api_key("fmp"),
            max_per_minute=config.get_rate_limit("fmp"),
        )
        self._file_path = self._storage_dir / "treasury_rates.parquet"

    def sync_with_status(
        self,
        status_mgr: "StatusManager",
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        force: bool = False,
        limit: Optional[int] = None,
        stale_days: int = 1,
    ):
        print("=" * 60)
        print("[treasury] FMP Treasury Rates")
        print("=" * 60)

        existing = self._load_existing()

        if not force and not existing.empty and "date" in existing.columns:
            latest = pd.to_datetime(existing["date"]).max().date()
            if latest >= date.today():
                print(f"  Already up-to-date (latest: {latest})")
                return

        if from_date is None and not existing.empty and "date" in existing.columns:
            from_date = pd.to_datetime(existing["date"]).max().date()

        data = self._client.get_json("/treasury-rates", **self._build_params(from_date, to_date))
        if not data:
            print("  No data returned from FMP /treasury")
            return

        new_df = pd.DataFrame(data)
        new_df["fetched_at"] = datetime.now(timezone.utc).isoformat()

        if not existing.empty:
            existing_dates = set(existing["date"].values)
            new_df = new_df[~new_df["date"].isin(existing_dates)]

        if new_df.empty:
            print("  No new data")
            return

        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        self._save(combined)

        print(f"  Added {len(new_df)} new dates, total: {len(combined)}")

        if status_mgr:
            status_mgr.update_symbol("treasury", "rates", records=len(combined), status="ok")

    def _build_params(self, from_date: Optional[date], to_date: Optional[date]) -> dict:
        params = {}
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()
        return params

    def _load_existing(self) -> pd.DataFrame:
        if self._file_path.exists():
            return pd.read_parquet(self._file_path)
        return pd.DataFrame()

    def _save(self, df: pd.DataFrame):
        df.to_parquet(self._file_path, compression="snappy", index=False)

    def get_rate(self, rate_date: date, maturity: str = "year10") -> Optional[float]:
        """Get treasury rate for a specific date and maturity."""
        df = self._load_existing()
        if df.empty:
            return None

        row = df[df["date"] == rate_date.isoformat()]
        if row.empty:
            # Find nearest prior date
            df["date_dt"] = pd.to_datetime(df["date"])
            prior = df[df["date_dt"] <= pd.Timestamp(rate_date)]
            if prior.empty:
                return None
            row = prior.iloc[-1:]

        if maturity in row.columns:
            val = row[maturity].values[0]
            if pd.notna(val):
                return float(val) / 100.0  # Convert from percentage
        return None

    def status(self) -> dict:
        if not self._file_path.exists():
            return {"dates": 0, "size_mb": 0}
        df = pd.read_parquet(self._file_path)
        return {
            "dates": len(df),
            "latest": df["date"].max() if "date" in df.columns else None,
            "size_mb": round(self._file_path.stat().st_size / 1e6, 1),
        }


register("treasury", "fmp", FMPTreasuryProvider)
