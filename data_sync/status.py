"""
Status Manager — SQLite-backed sync progress tracking + Rich CLI display

每个 provider 在 sync 完成后调用 StatusManager.update_symbol() 记录进度。
CLI 通过 StatusManager.summary() / StatusManager.symbol_detail() 快速查询。

SQLite DB 位于 {storage_root}/_sync_status.db
"""
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from data_sync.config import ETLConfig

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS sync_status (
    provider    TEXT    NOT NULL,
    symbol      TEXT    NOT NULL,
    latest_date TEXT,
    records     INTEGER DEFAULT 0,
    status      TEXT    DEFAULT 'pending',
    error_msg   TEXT,
    updated_at  TEXT    NOT NULL,
    PRIMARY KEY (provider, symbol)
);

CREATE TABLE IF NOT EXISTS sync_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    provider    TEXT    NOT NULL,
    started_at  TEXT    NOT NULL,
    finished_at TEXT,
    symbols_ok  INTEGER DEFAULT 0,
    symbols_err INTEGER DEFAULT 0,
    total_records INTEGER DEFAULT 0,
    duration_sec REAL
);

CREATE INDEX IF NOT EXISTS idx_status_provider ON sync_status(provider);
CREATE INDEX IF NOT EXISTS idx_status_updated ON sync_status(provider, updated_at);
"""


class StatusManager:

    def __init__(self, db_path: Path):
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_SQL)

    @classmethod
    def from_config(cls, config: ETLConfig) -> "StatusManager":
        db_path = config.storage_root / "_sync_status.db"
        return cls(db_path)

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Write API (called by providers during sync) ──────────

    def update_symbol(
        self,
        provider: str,
        symbol: str,
        latest_date: Optional[str] = None,
        records: int = 0,
        status: str = "ok",
        error_msg: Optional[str] = None,
    ):
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO sync_status (provider, symbol, latest_date, records, status, error_msg, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, symbol) DO UPDATE SET
                latest_date = COALESCE(excluded.latest_date, latest_date),
                records = CASE WHEN excluded.records > 0 THEN excluded.records ELSE records END,
                status = excluded.status,
                error_msg = excluded.error_msg,
                updated_at = excluded.updated_at
            """,
            (provider, symbol, latest_date, records, status, error_msg, now),
        )
        self._conn.commit()

    def update_bulk(
        self,
        provider: str,
        symbol: str = "__bulk__",
        latest_date: Optional[str] = None,
        records: int = 0,
    ):
        """For bulk providers like prices_1d / universe that don't have per-symbol granularity."""
        self.update_symbol(provider, symbol, latest_date, records, "ok")

    def start_run(self, provider: str) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            "INSERT INTO sync_runs (provider, started_at) VALUES (?, ?)",
            (provider, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def finish_run(
        self, run_id: int, symbols_ok: int = 0, symbols_err: int = 0,
        total_records: int = 0, duration_sec: float = 0,
    ):
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            UPDATE sync_runs
            SET finished_at = ?, symbols_ok = ?, symbols_err = ?,
                total_records = ?, duration_sec = ?
            WHERE id = ?
            """,
            (now, symbols_ok, symbols_err, total_records, duration_sec, run_id),
        )
        self._conn.commit()

    # ── Read API (called by status CLI) ──────────────────────

    def summary(self) -> List[dict]:
        """Per-provider summary for the overview table."""
        rows = self._conn.execute("""
            SELECT
                provider,
                COUNT(*) as symbol_count,
                SUM(records) as total_records,
                MAX(latest_date) as latest_date,
                MAX(updated_at) as last_sync,
                SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) as ok_count,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_count
            FROM sync_status
            WHERE symbol != '__bulk__'
            GROUP BY provider
            ORDER BY provider
        """).fetchall()

        # Also include bulk entries
        bulk_rows = self._conn.execute("""
            SELECT provider, latest_date, records, updated_at
            FROM sync_status
            WHERE symbol = '__bulk__'
        """).fetchall()
        bulk_map = {r["provider"]: dict(r) for r in bulk_rows}

        result = []
        for r in rows:
            d = dict(r)
            if d["provider"] in bulk_map:
                b = bulk_map[d["provider"]]
                d["latest_date"] = b["latest_date"] or d["latest_date"]
                d["last_sync"] = max(d["last_sync"] or "", b["updated_at"] or "")
            result.append(d)

        # Add bulk-only providers (e.g. universe, prices_1d)
        seen = {r["provider"] for r in result}
        for prov, b in bulk_map.items():
            if prov not in seen:
                result.append({
                    "provider": prov,
                    "symbol_count": 0,
                    "total_records": b["records"],
                    "latest_date": b["latest_date"],
                    "last_sync": b["updated_at"],
                    "ok_count": 0,
                    "error_count": 0,
                    "pending_count": 0,
                })

        result.sort(key=lambda x: x["provider"])
        return result

    def symbol_detail(self, provider: str, symbols: Optional[List[str]] = None) -> List[dict]:
        """Per-symbol detail for drill-down."""
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            rows = self._conn.execute(
                f"""
                SELECT symbol, latest_date, records, status, error_msg, updated_at
                FROM sync_status
                WHERE provider = ? AND symbol IN ({placeholders}) AND symbol != '__bulk__'
                ORDER BY symbol
                """,
                [provider] + symbols,
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT symbol, latest_date, records, status, error_msg, updated_at
                FROM sync_status
                WHERE provider = ? AND symbol != '__bulk__'
                ORDER BY symbol
                """,
                (provider,),
            ).fetchall()
        return [dict(r) for r in rows]

    def stale_symbols(self, provider: str, stale_days: int) -> List[dict]:
        """Symbols not updated within stale_days."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=stale_days)).isoformat()
        rows = self._conn.execute(
            """
            SELECT symbol, latest_date, records, status, updated_at
            FROM sync_status
            WHERE provider = ? AND updated_at < ? AND symbol != '__bulk__'
            ORDER BY updated_at ASC
            """,
            (provider, cutoff),
        ).fetchall()
        return [dict(r) for r in rows]

    def last_run(self, provider: str) -> Optional[dict]:
        row = self._conn.execute(
            """
            SELECT * FROM sync_runs
            WHERE provider = ?
            ORDER BY id DESC LIMIT 1
            """,
            (provider,),
        ).fetchone()
        return dict(row) if row else None

    def pending_count(self, provider: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM sync_status WHERE provider = ? AND status = 'pending'",
            (provider,),
        ).fetchone()
        return row["cnt"] if row else 0


def refresh_from_storage(config: ETLConfig, status_mgr: "StatusManager"):
    """Rebuild status DB from existing data on disk. Useful for bootstrapping."""
    import pandas as pd
    from data_sync.storage.parquet import ParquetStorage

    now = datetime.now(timezone.utc).isoformat()

    # universe
    uni_dir = config.storage_paths.get("universe")
    if uni_dir and uni_dir.exists():
        stock_list = uni_dir / "stock_list.parquet"
        records = 0
        if stock_list.exists():
            records = len(pd.read_parquet(stock_list))
        status_mgr.update_bulk("universe", records=records)
        print(f"  universe: {records} stocks")

    # prices_1d
    p1d_dir = config.storage_paths.get("prices_1d")
    if p1d_dir and p1d_dir.exists():
        storage = ParquetStorage(str(p1d_dir))
        latest = storage.latest_date("1d")
        syms = storage.list_symbols("1d")
        status_mgr.update_bulk(
            "prices_1d",
            latest_date=str(latest.date()) if latest else None,
            records=len(syms),
        )
        print(f"  prices_1d: {len(syms)} symbols, latest={latest.date() if latest else 'none'}")

    # prices_1min
    p1m_dir = config.storage_paths.get("prices_1min")
    if p1m_dir and p1m_dir.exists():
        storage = ParquetStorage(str(p1m_dir))
        syms = storage.list_symbols("1m")
        latest = storage.latest_date("1m")
        for sym in syms:
            status_mgr.update_symbol(
                "prices_1min", sym,
                latest_date=str(latest.date()) if latest else None,
                status="ok",
            )
        print(f"  prices_1min: {len(syms)} symbols, latest={latest.date() if latest else 'none'}")

    # financials
    fin_dir = config.storage_paths.get("financials")
    if fin_dir:
        fin_file = fin_dir / "financials.parquet"
        if fin_file.exists():
            df = pd.read_parquet(fin_file)
            if "symbol" in df.columns:
                for sym, grp in df.groupby("symbol"):
                    fetched = grp["fetched_at"].max() if "fetched_at" in grp.columns else now
                    status_mgr.update_symbol(
                        "financials", sym,
                        records=len(grp),
                        status="ok",
                    )
                n = df["symbol"].nunique()
                print(f"  financials: {n} symbols, {len(df)} rows")

    # analysts
    ana_dir = config.storage_paths.get("analysts")
    if ana_dir:
        ana_file = ana_dir / "estimates.parquet"
        if ana_file.exists():
            df = pd.read_parquet(ana_file)
            if "symbol" in df.columns:
                for sym, grp in df.groupby("symbol"):
                    status_mgr.update_symbol(
                        "analysts", sym,
                        records=len(grp),
                        status="ok",
                    )
                n = df["symbol"].nunique()
                print(f"  analysts: {n} symbols, {len(df)} rows")

    print("  Status DB refreshed.\n")


def _format_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n:,}"
    return str(n)


def _freshness(updated_at: Optional[str]) -> str:
    if not updated_at:
        return "never"
    try:
        ts = datetime.fromisoformat(updated_at)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        days = delta.days
        if days == 0:
            hours = delta.seconds // 3600
            if hours == 0:
                return f"{delta.seconds // 60}m ago"
            return f"{hours}h ago"
        if days == 1:
            return "1 day ago"
        return f"{days} days ago"
    except (ValueError, TypeError):
        return "?"


def print_summary(status_mgr: StatusManager, provider_order: List[str]):
    """Print the overview table using rich."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    summaries = status_mgr.summary()
    summary_map = {s["provider"]: s for s in summaries}

    table = Table(title="Data Sync Status", show_lines=False)
    table.add_column("Provider", style="bold")
    table.add_column("Last Sync", justify="right")
    table.add_column("Records", justify="right")
    table.add_column("Symbols", justify="right")
    table.add_column("OK", justify="right", style="green")
    table.add_column("Err", justify="right", style="red")
    table.add_column("Latest Date", justify="right")
    table.add_column("Freshness", justify="right")

    for prov in provider_order:
        s = summary_map.get(prov)
        if s is None:
            table.add_row(prov, "-", "-", "-", "-", "-", "-", "[dim]no data[/dim]")
            continue

        freshness = _freshness(s.get("last_sync"))
        fresh_style = ""
        if "day" in freshness:
            try:
                days = int(freshness.split()[0])
                if days > 7:
                    fresh_style = "[yellow]"
                if days > 30:
                    fresh_style = "[red]"
            except ValueError:
                pass

        last_sync_short = ""
        if s.get("last_sync"):
            try:
                last_sync_short = s["last_sync"][:10]
            except (TypeError, IndexError):
                pass

        table.add_row(
            prov,
            last_sync_short,
            _format_number(s.get("total_records") or 0),
            str(s.get("symbol_count") or "-"),
            str(s.get("ok_count") or 0),
            str(s.get("error_count") or 0),
            s.get("latest_date") or "-",
            f"{fresh_style}{freshness}{'[/]' if fresh_style else ''}",
        )

    console.print()
    console.print(table)

    # Last run info
    for prov in provider_order:
        run = status_mgr.last_run(prov)
        if run and run.get("duration_sec"):
            dur = run["duration_sec"]
            if dur > 60:
                dur_str = f"{dur/60:.1f}min"
            else:
                dur_str = f"{dur:.0f}s"
            console.print(
                f"  [dim]{prov}[/dim] last run: {run['symbols_ok']} ok, "
                f"{run['symbols_err']} err, {_format_number(run['total_records'])} records, "
                f"{dur_str}"
            )
    console.print()


def print_symbol_detail(status_mgr: StatusManager, provider: str, symbols: Optional[List[str]] = None):
    """Print per-symbol detail table using rich."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    rows = status_mgr.symbol_detail(provider, symbols)

    if not rows:
        console.print(f"[yellow]No status data for provider '{provider}'[/yellow]")
        return

    table = Table(title=f"{provider} — Symbol Status ({len(rows)} symbols)")
    table.add_column("Symbol", style="bold")
    table.add_column("Latest Date", justify="right")
    table.add_column("Records", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Freshness", justify="right")
    table.add_column("Error", max_width=40)

    for r in rows:
        status_style = {"ok": "[green]", "error": "[red]", "pending": "[yellow]"}.get(r["status"], "")
        table.add_row(
            r["symbol"],
            r.get("latest_date") or "-",
            _format_number(r.get("records") or 0),
            f"{status_style}{r['status']}{'[/]' if status_style else ''}",
            _freshness(r.get("updated_at")),
            (r.get("error_msg") or "")[:40],
        )

    console.print()
    console.print(table)
    console.print()


def print_stale(status_mgr: StatusManager, provider: str, stale_days: int):
    """Print stale symbols for a provider."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    rows = status_mgr.stale_symbols(provider, stale_days)

    if not rows:
        console.print(f"[green]No stale symbols for '{provider}' (threshold: {stale_days}d)[/green]")
        return

    table = Table(title=f"{provider} — Stale Symbols ({len(rows)}, >{stale_days}d)")
    table.add_column("Symbol", style="bold")
    table.add_column("Latest Date", justify="right")
    table.add_column("Last Updated", justify="right")
    table.add_column("Records", justify="right")

    for r in rows:
        table.add_row(
            r["symbol"],
            r.get("latest_date") or "-",
            _freshness(r.get("updated_at")),
            _format_number(r.get("records") or 0),
        )

    console.print()
    console.print(table)
    console.print()
