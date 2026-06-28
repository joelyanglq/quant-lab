"""
Financials Provider — 财报三表

FMP 实现：
    /stable/income-statement?symbol=X&period=annual|quarter
    /stable/balance-sheet-statement?symbol=X&period=annual|quarter
    /stable/cash-flow-statement?symbol=X&period=annual|quarter

存储: 单文件 financials.parquet，三表 merge on (symbol, date, period)
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

STATEMENT_ENDPOINTS = {
    "income": "/income-statement",
    "balance_sheet": "/balance-sheet-statement",
    "cash_flow": "/cash-flow-statement",
}


class FMPFinancialsProvider(Provider):
    name = "financials"

    def __init__(self, config: ETLConfig):
        super().__init__(config)
        self._storage_dir = config.get_storage_path("financials")
        self._client = FMPClient(
            api_key=config.get_api_key("fmp"),
            max_per_minute=config.get_rate_limit("fmp"),
        )
        self._file_path = self._storage_dir / "financials.parquet"

    def sync_with_status(
        self,
        status_mgr: "StatusManager",
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        force: bool = False,
        limit: Optional[int] = None,
        stale_days: int = 30,
    ):
        """Sync with StatusManager tracking, concurrency, and --limit support."""
        print("=" * 60)
        print("[financials] FMP Financial Statements (Income + BS + CF)")
        print("=" * 60)

        if symbols is None:
            symbols = self._load_default_symbols()

        existing = self._load_existing()

        if force:
            queue = list(symbols)
        else:
            queue = self._get_stale_symbols(existing, symbols, stale_days)

        n_existing = existing["symbol"].nunique() if not existing.empty else 0

        if limit and len(queue) > limit:
            queue = queue[:limit]
            print(
                f"  Target: {len(symbols)}, in store: {n_existing}, this run: {len(queue)} (limit={limit})"
            )
        else:
            print(
                f"  Target: {len(symbols)}, in store: {n_existing}, stale: {len(queue)}"
            )

        if not queue:
            print("  All symbols up-to-date.\n")
            return

        new_dfs = []
        done_count = 0
        fail_count = 0
        dfs_lock = Lock()

        def fetch_one(sym):
            try:
                df = self._fetch_symbol(sym)
                if df is not None and not df.empty:
                    n_a = len(df[df["period"] == "FY"])
                    n_q = len(df[df["period"] != "FY"])
                    return sym, df, f"{n_a}A + {n_q}Q", None
                return sym, None, "no data", None
            except Exception as e:
                return sym, None, None, str(e)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(fetch_one, sym): sym for sym in queue}

            for fut in as_completed(futures):
                sym, df, info, error = fut.result()

                if error:
                    print(f"  {sym} ERROR: {error}")
                    fail_count += 1
                    if status_mgr:
                        status_mgr.update_symbol(
                            "financials", sym, status="error", error_msg=error
                        )
                elif df is not None:
                    print(f"  {sym} {info}")
                    done_count += 1
                    with dfs_lock:
                        new_dfs.append(df)
                    if status_mgr:
                        status_mgr.update_symbol(
                            "financials", sym, records=len(df), status="ok"
                        )
                else:
                    print(f"  {sym} {info}")
                    done_count += 1
                    if status_mgr:
                        status_mgr.update_symbol("financials", sym, status="ok")

                if len(new_dfs) >= 20:
                    with dfs_lock:
                        to_merge = new_dfs[:]
                        new_dfs.clear()
                    merged = pd.concat(to_merge, ignore_index=True)
                    existing = self._merge_symbol_data(existing, merged)
                    self._save(existing)

        if new_dfs:
            merged = pd.concat(new_dfs, ignore_index=True)
            existing = self._merge_symbol_data(existing, merged)
            self._save(existing)

        n_total = existing["symbol"].nunique() if not existing.empty else 0
        print(
            f"\n  [financials] Done: {done_count} ok, {fail_count} failed, {n_total} total symbols\n"
        )

    def _fetch_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        dfs_by_statement = {}

        for stmt_name, endpoint in STATEMENT_ENDPOINTS.items():
            for period in ["annual", "quarter"]:
                data = self._client.get_json(
                    endpoint, symbol=symbol, period=period, limit=100
                )
                if data:
                    df = pd.DataFrame(data)
                    dfs_by_statement.setdefault(stmt_name, []).append(df)

        if not dfs_by_statement:
            return None

        stmt_dfs = {}
        for stmt_name, df_list in dfs_by_statement.items():
            stmt_dfs[stmt_name] = pd.concat(df_list, ignore_index=True)

        merge_keys = ["symbol", "date", "period"]
        result = None
        for stmt_name, df in stmt_dfs.items():
            if result is None:
                result = df
            else:
                overlap_cols = set(result.columns) & set(df.columns) - set(merge_keys)
                df_clean = df.drop(
                    columns=[c for c in overlap_cols if c in df.columns],
                    errors="ignore",
                )
                result = result.merge(df_clean, on=merge_keys, how="outer")

        if result is not None:
            result["fetched_at"] = datetime.now(timezone.utc).isoformat()

        return result

    def _load_existing(self) -> pd.DataFrame:
        if self._file_path.exists():
            return pd.read_parquet(self._file_path)
        return pd.DataFrame()

    def _save(self, df: pd.DataFrame):
        df.to_parquet(self._file_path, compression="snappy", index=False)

    def _merge_symbol_data(
        self, existing: pd.DataFrame, new_data: pd.DataFrame
    ) -> pd.DataFrame:
        if existing.empty:
            return new_data
        if new_data.empty:
            return existing
        new_symbols = set(new_data["symbol"].unique())
        kept = existing[~existing["symbol"].isin(new_symbols)]
        return pd.concat([kept, new_data], ignore_index=True)

    def _get_stale_symbols(
        self, existing: pd.DataFrame, symbols: List[str], stale_days: int
    ) -> List[str]:
        if existing.empty or "symbol" not in existing.columns:
            return list(symbols)

        if "fetched_at" not in existing.columns:
            existing_syms = set(existing["symbol"].unique())
            return [s for s in symbols if s not in existing_syms]

        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=stale_days)
        latest_ts = existing.groupby("symbol")["fetched_at"].max().reset_index()
        latest_ts["fetched_at"] = pd.to_datetime(latest_ts["fetched_at"], utc=True)
        fresh = set(latest_ts[latest_ts["fetched_at"] >= cutoff]["symbol"].values)
        return [s for s in symbols if s not in fresh]

    def _load_default_symbols(self) -> List[str]:
        return resolve_symbol_selector(self.config, provider_name=self.name)

    def status(self) -> dict:
        if not self._file_path.exists():
            return {"symbols": 0, "rows": 0, "size_mb": 0}
        df = pd.read_parquet(self._file_path)
        return {
            "symbols": df["symbol"].nunique() if "symbol" in df.columns else 0,
            "rows": len(df),
            "size_mb": round(self._file_path.stat().st_size / 1e6, 1),
        }


register("financials", "fmp", FMPFinancialsProvider)
