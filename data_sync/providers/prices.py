"""
Prices Provider — 日线 / 分钟线价格数据

FMP 实现：
    日线: /stable/eod-bulk?date=YYYY-MM-DD (全市场，一次一天)
    分钟线: /stable/historical-chart/1min?symbol=X&from=&to= (per-symbol)

存储: ParquetStorage (日线按年；1min 按月 parquet)
"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from threading import Lock
from typing import List, Optional, Set, TYPE_CHECKING

import pandas as pd

from data_sync.client.fmp import FMPClient
from data_sync.config import ETLConfig
from data_sync.providers.base import Provider
from data_sync.providers import register
from data_sync.storage.parquet import ParquetStorage
from data_sync.symbols import resolve_symbol_selector

if TYPE_CHECKING:
    from data_sync.status import StatusManager

logger = logging.getLogger(__name__)


def _trading_dates(start: date, end: date) -> List[date]:
    """工作日列表（跳过周末）"""
    dates = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


# ── 日线 ──────────────────────────────────────────────────────


class FMPPricesDailyProvider(Provider):
    name = "prices_1d"

    def __init__(self, config: ETLConfig):
        super().__init__(config)
        self._storage = ParquetStorage(str(config.get_storage_path("prices_1d")))
        self._client = FMPClient(
            api_key=config.get_api_key("fmp"),
            max_per_minute=config.get_rate_limit("fmp"),
        )

    def sync(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        force: bool = False,
    ):
        print("=" * 60)
        print("[prices_1d] FMP eod-bulk — Full Market Daily Bars")
        print("=" * 60)

        if to_date is None:
            to_date = date.today()

        if from_date is None:
            latest = self._storage.latest_date("1d")
            if latest is not None:
                from_date = (latest + timedelta(days=1)).date()
                print(f"  Latest data: {latest.date()}")
            else:
                from_date = date(2018, 1, 2)
                print("  No existing data, starting from 2018-01-02")

        dates = _trading_dates(from_date, to_date)
        if not dates:
            print("  Already up to date.\n")
            return

        print(f"  Fetching {len(dates)} date(s): {dates[0]} ~ {dates[-1]}\n")

        total_bars = 0
        for d in dates:
            date_str = d.isoformat()
            print(f"  {date_str} ...", end=" ", flush=True)
            try:
                df_raw = self._client.get_csv("/eod-bulk", date=date_str)
                if df_raw.empty:
                    print("no data (holiday?)")
                    continue

                df = self._normalize_eod(df_raw, d)
                if symbols:
                    df = df[df["symbol"].isin(symbols)]

                if df.empty:
                    print("no matching data")
                    continue

                n_syms = df["symbol"].nunique()
                print(f"{len(df):,} bars, {n_syms:,} symbols")
                self._storage.save(df, "1d")
                total_bars += len(df)

            except Exception as e:
                print(f"FAILED ({e})")

        if total_bars > 0:
            new_latest = self._storage.latest_date("1d")
            print(f"\n  [prices_1d] Done: {total_bars:,} bars. Latest: {new_latest.date()}\n")
        else:
            print("\n  [prices_1d] No new data.\n")

    def sync_with_status(
        self,
        status_mgr: "StatusManager",
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        force: bool = False,
        limit: Optional[int] = None,
    ):
        """Sync with StatusManager tracking. limit is ignored for bulk daily (date-based)."""
        self.sync(from_date=from_date, to_date=to_date, symbols=symbols, force=force)

        latest = self._storage.latest_date("1d")
        storage_path = self.config.get_storage_path("prices_1d")
        files = sorted(storage_path.glob("*_1d.parquet"))
        n_syms = len(self._storage.list_symbols("1d")) if self._storage.exists("1d") else 0
        status_mgr.update_bulk(
            "prices_1d",
            latest_date=str(latest.date()) if latest else None,
            records=n_syms,
        )

    def _normalize_eod(self, df_raw: pd.DataFrame, trade_date: date) -> pd.DataFrame:
        df = df_raw.copy()
        df = df.dropna(subset=["close", "volume"])
        df = df[df["volume"] > 0]

        col_map = {"adjClose": "adj_close"}
        df = df.rename(columns=col_map)

        keep_cols = ["symbol", "open", "high", "low", "close", "volume"]
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]

        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype(int)

        df.index = pd.DatetimeIndex([pd.Timestamp(trade_date)] * len(df))
        return df

    def status(self) -> dict:
        latest = self._storage.latest_date("1d")
        storage_path = self.config.get_storage_path("prices_1d")
        files = sorted(storage_path.glob("*_1d.parquet"))
        total_size = sum(f.stat().st_size for f in files) / 1e6
        return {
            "latest_date": str(latest.date()) if latest else None,
            "n_files": len(files),
            "total_size_mb": round(total_size, 1),
        }


# ── 分钟线 ────────────────────────────────────────────────────


class FMPPricesIntradayProvider(Provider):
    name = "prices_1min"

    def __init__(self, config: ETLConfig):
        super().__init__(config)
        self._storage = ParquetStorage(str(config.get_storage_path("prices_1min")))
        self._client = FMPClient(
            api_key=config.get_api_key("fmp"),
            max_per_minute=config.get_rate_limit("fmp"),
        )
        self._workers = 64
        self._progress_file = config.get_storage_path("prices_1min") / "_sync_progress.json"

    def sync(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        force: bool = False,
    ):
        self.sync_with_status(None, from_date, to_date, symbols, force)

    def sync_with_status(
        self,
        status_mgr: Optional["StatusManager"],
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        force: bool = False,
        limit: Optional[int] = None,
    ):
        print("=" * 60)
        print("[prices_1min] FMP historical-chart/1min — Intraday Bars")
        print("=" * 60)

        if to_date is None:
            to_date = date.today()

        if symbols is None:
            symbols = self._load_default_symbols()

        print(f"  Symbols: {len(symbols)}")
        print(f"  Workers: {self._workers}")

        progress = self._load_progress()
        today_str = date.today().isoformat()
        if progress.get("date") != today_str or force:
            progress = {"date": today_str, "done": []}

        done_set = set(progress["done"])
        remaining = [s for s in symbols if s not in done_set]

        if limit and len(remaining) > limit:
            remaining = remaining[:limit]
            print(f"  Done: {len(done_set)}, this run: {len(remaining)} (limit={limit})")
        else:
            print(f"  Done: {len(done_set)}, remaining: {len(remaining)}")

        if not remaining:
            print("  All symbols already synced.\n")
            return

        sync_from = from_date
        if sync_from is None:
            latest = self._storage.latest_date("1m")
            if latest is not None:
                sync_from = (latest + timedelta(days=1)).date()
                print(f"  Storage latest: {latest.date()}")
            else:
                sync_from = date(2024, 1, 1)
                print("  No existing 1min data detected, starting from 2024-01-01")

        if sync_from > to_date:
            print("  Already up to date.\n")
            return

        # FMP /historical-chart/1min caps at ~3 trading days regardless of the
        # from/to span (it returns only the most recent 3 days before `to`).
        # Roll 3-calendar-day windows across [sync_from, to_date] to cover the
        # whole range; dedup by (timestamp, symbol) at save time.
        windows = self._rolling_windows(sync_from, to_date, step_days=3)
        print(f"  Fetching: {sync_from} ~ {to_date} ({len(windows)} windows)\n")

        total_bars = 0
        batch_dfs = []
        batch_lock = Lock()
        storage_lock = Lock()

        def fetch_one(sym: str) -> tuple:
            seen = {}
            for w_from, w_to in windows:
                try:
                    data = self._client.get_json(
                        "/historical-chart/1min",
                        symbol=sym,
                        **{"from": w_from.isoformat(), "to": w_to.isoformat()},
                    )
                except Exception as e:
                    return sym, None, 0, str(e)
                if not data:
                    continue
                for row in data:
                    seen[row["date"]] = row
            if not seen:
                return sym, None, 0, None
            df = self._normalize_1min(list(seen.values()), sym)
            if df.empty:
                return sym, None, 0, None
            return sym, df, len(df), None

        n_done_before = len(done_set)
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {pool.submit(fetch_one, sym): sym for sym in remaining}

            for fut in as_completed(futures):
                sym, df, n_bars, error = fut.result()
                idx = n_done_before + len(done_set) - n_done_before + 1

                if error:
                    print(f"  [{idx}/{len(symbols)}] {sym} ERROR: {error}")
                    if status_mgr:
                        status_mgr.update_symbol("prices_1min", sym, status="error", error_msg=error)
                elif df is not None:
                    print(f"  [{idx}/{len(symbols)}] {sym} +{n_bars} bars")
                    total_bars += n_bars
                    with batch_lock:
                        batch_dfs.append(df)

                    if status_mgr:
                        latest = str(df.index.max().date())
                        status_mgr.update_symbol("prices_1min", sym, latest_date=latest, records=n_bars)

                    if len(batch_dfs) >= 30:
                        with batch_lock:
                            to_flush = batch_dfs[:]
                            batch_dfs.clear()
                        combined = pd.concat(to_flush)
                        with storage_lock:
                            self._storage.save(combined, "1m", partition="monthly")
                        print(f"    [flushed {len(combined):,} bars]")
                else:
                    print(f"  [{idx}/{len(symbols)}] {sym} no data")
                    if status_mgr:
                        status_mgr.update_symbol("prices_1min", sym, status="ok")

                done_set.add(sym)

                if len(done_set) % 20 == 0:
                    progress["done"] = list(done_set)
                    self._save_progress(progress)

        if batch_dfs:
            combined = pd.concat(batch_dfs)
            self._storage.save(combined, "1m", partition="monthly")
            print(f"    [flushed {len(combined):,} bars]")

        progress["done"] = list(done_set)
        self._save_progress(progress)
        print(f"\n  [prices_1min] Done: {total_bars:,} new bars\n")

    @staticmethod
    def _rolling_windows(start: date, end: date, step_days: int = 3):
        """Yield (window_start, window_end) covering [start, end] in steps."""
        windows = []
        cur = start
        while cur <= end:
            w_end = min(cur + timedelta(days=step_days - 1), end)
            windows.append((cur, w_end))
            cur = cur + timedelta(days=step_days)
        return windows

    def _normalize_1min(self, data: list, symbol: str) -> pd.DataFrame:
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df["date"] = df["date"].dt.tz_localize("US/Eastern").dt.tz_localize(None)
        df = df.rename(columns={"date": "timestamp"})
        df = df.set_index("timestamp")

        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype(int)

        df["symbol"] = symbol
        df = df[["symbol", "open", "high", "low", "close", "volume"]]
        df = df.sort_index()
        return df

    def _load_default_symbols(self) -> List[str]:
        return resolve_symbol_selector(self.config, provider_name=self.name)

    def _load_progress(self) -> dict:
        if self._progress_file.exists():
            return json.loads(self._progress_file.read_text(encoding="utf-8"))
        return {"date": None, "done": []}

    def _save_progress(self, progress: dict):
        self._progress_file.write_text(
            json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def status(self) -> dict:
        latest = self._storage.latest_date("1m")
        storage_path = self.config.get_storage_path("prices_1min")
        files = sorted(storage_path.glob("*_1m.parquet"))
        total_size = sum(f.stat().st_size for f in files) / 1e6
        return {
            "latest_date": str(latest.date()) if latest else None,
            "n_files": len(files),
            "total_size_mb": round(total_size, 1),
        }


register("prices_1d", "fmp", FMPPricesDailyProvider)
register("prices_1min", "fmp", FMPPricesIntradayProvider)
