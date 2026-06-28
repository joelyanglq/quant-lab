"""
Universe Provider — 股票宇宙管理

FMP 实现：stock list、company profile、index constituents (当前 + 历史变更)。

存储：
    universe/stock_list.parquet     — 全市场 tradeable stocks
    universe/profiles.parquet       — company profiles (sector, industry, etc.)
    universe/index_current.parquet  — 当前指数成分 (覆盖写)
    universe/index_changes.parquet  — 指数成分历史变更 (覆盖写)
"""
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import pandas as pd

from data_sync.client.fmp import FMPClient
from data_sync.config import ETLConfig
from data_sync.providers.base import Provider
from data_sync.providers import register

if TYPE_CHECKING:
    from data_sync.status import StatusManager

logger = logging.getLogger(__name__)

INDEX_ENDPOINTS = {
    "sp500": {
        "current": "/sp500-constituent",
        "historical": "/historical-sp500-constituent",
    },
    "nasdaq": {
        "current": "/nasdaq-constituent",
        "historical": "/historical-nasdaq-constituent",
    },
    "dowjones": {
        "current": "/dowjones-constituent",
        "historical": "/historical-dowjones-constituent",
    },
}


class FMPUniverseProvider(Provider):
    name = "universe"

    def __init__(self, config: ETLConfig):
        super().__init__(config)
        self._storage_dir = config.get_storage_path("universe")
        self._client = FMPClient(
            api_key=config.get_api_key("fmp"),
            max_per_minute=config.get_rate_limit("fmp"),
        )

    # ── sync ──────────────────────────────────────────────────

    def sync(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        force: bool = False,
    ):
        """同步 stock list + profiles + index constituents。"""
        print("=" * 60)
        print("[universe] Syncing stock list, profiles, index constituents")
        print("=" * 60)

        self.sync_stock_list()
        self.sync_index("sp500")
        self.sync_index("nasdaq")

        if symbols is None:
            symbols = self.get_index_symbols("sp500") + self.get_index_symbols("nasdaq")
            symbols = sorted(set(symbols))
        self.sync_profiles(symbols)

        print("[universe] Done.\n")

    def sync_with_status(
        self,
        status_mgr: "StatusManager",
        from_date=None, to_date=None, symbols=None, force=False, limit=None,
    ):
        self.sync(from_date=from_date, to_date=to_date, symbols=symbols, force=force)

        records = 0
        stock_list = self._storage_dir / "stock_list.parquet"
        if stock_list.exists():
            records = len(pd.read_parquet(stock_list))
        status_mgr.update_bulk("universe", records=records)

    def sync_stock_list(self):
        """拉取 /stable/stock-list → stock_list.parquet"""
        print("  Fetching stock list...", end=" ", flush=True)
        data = self._client.get_json("/stock-list")
        if not data:
            print("empty response")
            return

        df = pd.DataFrame(data)
        path = self._storage_dir / "stock_list.parquet"
        df.to_parquet(path, compression="snappy", index=False)
        print(f"{len(df)} stocks")

    def sync_profiles(self, symbols: List[str]):
        print(f"  Fetching profiles for {len(symbols)} symbols...", flush=True)
        all_rows = []

        for i, sym in enumerate(symbols):
            try:
                data = self._client.get_json("/profile", symbol=sym)
                if data:
                    all_rows.extend(data)
            except Exception as e:
                logger.warning("Profile failed for %s: %s", sym, e)

            if (i + 1) % 100 == 0:
                print(f"    {i+1}/{len(symbols)}", flush=True)

        if all_rows:
            df = pd.DataFrame(all_rows)
            path = self._storage_dir / "profiles.parquet"
            df.to_parquet(path, compression="snappy", index=False)
            print(f"  Saved {len(df)} profiles")
        else:
            print("  No profile data")

    def sync_index(self, index_name: str):
        endpoints = INDEX_ENDPOINTS.get(index_name)
        if not endpoints:
            raise ValueError(f"Unknown index: {index_name}. Available: {list(INDEX_ENDPOINTS)}")

        print(f"  Fetching {index_name} current constituents...", end=" ", flush=True)
        current = self._client.get_json(endpoints["current"])
        if current:
            df_current = pd.DataFrame(current)
            df_current["index_name"] = index_name
            path = self._storage_dir / f"index_current_{index_name}.parquet"
            df_current.to_parquet(path, compression="snappy", index=False)
            print(f"{len(df_current)} symbols")
        else:
            print("empty")

        print(f"  Fetching {index_name} historical changes...", end=" ", flush=True)
        changes = self._client.get_json(endpoints["historical"])
        if changes:
            df_changes = pd.DataFrame(changes)
            df_changes["index_name"] = index_name
            path = self._storage_dir / f"index_changes_{index_name}.parquet"
            df_changes.to_parquet(path, compression="snappy", index=False)
            print(f"{len(df_changes)} changes")
        else:
            print("empty")

    # ── 查询 ──────────────────────────────────────────────────

    def get_index_symbols(
        self, index_name: str, as_of: Optional[date] = None
    ) -> List[str]:
        current_path = self._storage_dir / f"index_current_{index_name}.parquet"
        if not current_path.exists():
            return []

        df_current = pd.read_parquet(current_path)
        current_symbols = set(df_current["symbol"].tolist())

        if as_of is None:
            return sorted(current_symbols)

        changes_path = self._storage_dir / f"index_changes_{index_name}.parquet"
        if not changes_path.exists():
            return sorted(current_symbols)

        df_changes = pd.read_parquet(changes_path)
        if df_changes.empty:
            return sorted(current_symbols)

        date_col = "date" if "date" in df_changes.columns else "dateAdded"
        df_changes[date_col] = pd.to_datetime(df_changes[date_col]).dt.date

        future_changes = df_changes[df_changes[date_col] > as_of].sort_values(
            date_col, ascending=False
        )

        symbols = set(current_symbols)
        for _, row in future_changes.iterrows():
            sym = row.get("symbol") or row.get("addedSecurity") or row.get("removedTicker")
            if sym is None:
                continue

            added = row.get("addedSecurity") or row.get("symbol")
            removed = row.get("removedTicker") or row.get("removedSecurity")

            if added and added in symbols:
                symbols.discard(added)
            if removed:
                symbols.add(removed)

        return sorted(symbols)

    def get_rebalance_dates(self, index_name: str) -> List[date]:
        changes_path = self._storage_dir / f"index_changes_{index_name}.parquet"
        if not changes_path.exists():
            return []

        df = pd.read_parquet(changes_path)
        if df.empty:
            return []

        date_col = "date" if "date" in df.columns else "dateAdded"
        dates = pd.to_datetime(df[date_col]).dt.date.unique()
        return sorted(dates)

    def get_universe(
        self,
        exchange: Optional[List[str]] = None,
        active_only: bool = True,
    ) -> List[str]:
        path = self._storage_dir / "stock_list.parquet"
        if not path.exists():
            return []

        df = pd.read_parquet(path)

        if exchange:
            exchange_upper = [e.upper() for e in exchange]
            df = df[df["exchangeShortName"].str.upper().isin(exchange_upper)]

        if active_only and "isActivelyTrading" in df.columns:
            df = df[df["isActivelyTrading"] == True]

        if "type" in df.columns:
            df = df[df["type"] == "stock"]

        return sorted(df["symbol"].tolist())

    def get_sector(self, symbol: str) -> Optional[str]:
        return self._profile_lookup(symbol, "sector")

    def get_industry(self, symbol: str) -> Optional[str]:
        return self._profile_lookup(symbol, "industry")

    def _profile_lookup(self, symbol: str, field: str) -> Optional[str]:
        path = self._storage_dir / "profiles.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path, filters=[("symbol", "==", symbol)])
        if df.empty:
            return None
        return df.iloc[0].get(field)

    # ── status ────────────────────────────────────────────────

    def status(self) -> dict:
        info = {}
        for name in ["stock_list", "profiles", "index_current_sp500", "index_changes_sp500"]:
            path = self._storage_dir / f"{name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                info[name] = {
                    "rows": len(df),
                    "size_mb": round(path.stat().st_size / 1e6, 1),
                }
            else:
                info[name] = None
        return info


register("universe", "fmp", FMPUniverseProvider)
