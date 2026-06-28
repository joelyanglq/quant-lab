"""
Analysts Provider — Analyst Estimates

FMP 实现：
    /stable/analyst-estimates?symbol=X (annual + quarterly consensus)

存储: analysts/estimates.parquet，append 模式 (每次 sync 带 fetched_date)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from threading import Lock
from typing import List, Optional, TYPE_CHECKING

import pandas as pd

from data_sync.client.fmp import FMPClient
from data_sync.config import ETLConfig
from data_sync.providers.base import Provider
from data_sync.providers import register
from data_sync.symbols import resolve_symbol_selector

if TYPE_CHECKING:
    from data_sync.status import StatusManager

logger = logging.getLogger(__name__)


class FMPAnalystsProvider(Provider):
    name = "analysts"

    def __init__(self, config: ETLConfig):
        super().__init__(config)
        self._storage_dir = config.get_storage_path("analysts")
        self._client = FMPClient(
            api_key=config.get_api_key("fmp"),
            max_per_minute=config.get_rate_limit("fmp"),
        )
        self._file_path = self._storage_dir / "estimates.parquet"

    def sync_with_status(
        self,
        status_mgr: "StatusManager",
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        force: bool = False,
        limit: Optional[int] = None,
        stale_days: int = 7,
    ):
        """Sync with StatusManager tracking, concurrency, and --limit support."""
        print("=" * 60)
        print("[analysts] FMP Analyst Estimates")
        print("=" * 60)

        if symbols is None:
            symbols = self._load_default_symbols()

        existing = self._load_existing()

        if force:
            queue = list(symbols)
        else:
            queue = self._get_stale_symbols(existing, symbols, stale_days)

        if limit and len(queue) > limit:
            queue = queue[:limit]
            print(f"  Target: {len(symbols)}, this run: {len(queue)} (limit={limit})")
        else:
            print(f"  Target: {len(symbols)}, stale: {len(queue)}")

        if not queue:
            print("  All symbols up-to-date.\n")
            return

        fetched_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        new_dfs = []
        done_count = 0
        dfs_lock = Lock()

        def fetch_one(sym):
            try:
                all_data = []
                for period in ["annual", "quarter"]:
                    data = self._client.get_json(
                        "/analyst-estimates", symbol=sym, period=period
                    )
                    if data:
                        for row in data:
                            row["period_type"] = period
                        all_data.extend(data)

                if all_data:
                    df = pd.DataFrame(all_data)
                    df["fetched_date"] = fetched_date
                    return sym, df, len(all_data), None
                return sym, None, 0, None
            except Exception as e:
                return sym, None, 0, str(e)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(fetch_one, sym): sym for sym in queue}

            for fut in as_completed(futures):
                sym, df, n_rows, error = fut.result()

                if error:
                    print(f"  {sym} ERROR: {error}")
                    if status_mgr:
                        status_mgr.update_symbol(
                            "analysts", sym, status="error", error_msg=error
                        )
                elif df is not None:
                    print(f"  {sym} {n_rows} rows")
                    done_count += 1
                    with dfs_lock:
                        new_dfs.append(df)
                    if status_mgr:
                        status_mgr.update_symbol(
                            "analysts", sym, records=n_rows, status="ok"
                        )
                else:
                    print(f"  {sym} no data")
                    done_count += 1
                    if status_mgr:
                        status_mgr.update_symbol("analysts", sym, status="ok")

                if len(new_dfs) >= 50:
                    with dfs_lock:
                        to_append = new_dfs[:]
                        new_dfs.clear()
                    batch = pd.concat(to_append, ignore_index=True)
                    existing = self._append(existing, batch)
                    self._save(existing)

        if new_dfs:
            batch = pd.concat(new_dfs, ignore_index=True)
            existing = self._append(existing, batch)
            self._save(existing)

        n_total = (
            existing["symbol"].nunique()
            if not existing.empty and "symbol" in existing.columns
            else 0
        )
        print(f"\n  [analysts] Done: {done_count} symbols, {n_total} total in store\n")

    def _append(self, existing: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        if existing.empty:
            return new_data

        combined = pd.concat([existing, new_data], ignore_index=True)
        dedup_cols = ["symbol", "date", "fetched_date"]
        dedup_cols = [c for c in dedup_cols if c in combined.columns]
        if dedup_cols:
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
        return combined

    def _load_existing(self) -> pd.DataFrame:
        if self._file_path.exists():
            return pd.read_parquet(self._file_path)
        return pd.DataFrame()

    def _save(self, df: pd.DataFrame):
        df.to_parquet(self._file_path, compression="snappy", index=False)

    def _get_stale_symbols(
        self, existing: pd.DataFrame, symbols: List[str], stale_days: int
    ) -> List[str]:
        if existing.empty or "fetched_date" not in existing.columns:
            return list(symbols)

        cutoff = (datetime.now(timezone.utc) - pd.Timedelta(days=stale_days)).strftime(
            "%Y-%m-%d"
        )
        latest = existing.groupby("symbol")["fetched_date"].max().reset_index()
        fresh = set(latest[latest["fetched_date"] >= cutoff]["symbol"].values)
        return [s for s in symbols if s not in fresh]

    def _load_default_symbols(self) -> List[str]:
        return resolve_symbol_selector(self.config, provider_name=self.name)

    def status(self) -> dict:
        if not self._file_path.exists():
            return {"symbols": 0, "rows": 0, "size_mb": 0, "fetched_dates": []}
        df = pd.read_parquet(self._file_path)
        dates = (
            sorted(df["fetched_date"].unique().tolist())
            if "fetched_date" in df.columns
            else []
        )
        return {
            "symbols": df["symbol"].nunique() if "symbol" in df.columns else 0,
            "rows": len(df),
            "size_mb": round(self._file_path.stat().st_size / 1e6, 1),
            "fetched_dates": dates[-5:],
        }


register("analysts", "fmp", FMPAnalystsProvider)
