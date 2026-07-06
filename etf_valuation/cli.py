"""
CLI entry point for ETF Valuation Monitoring System.

Usage:
    python -m etf_valuation sync              # Fetch today's ratios + treasury
    python -m etf_valuation backfill          # Historical N-PORT + ratios
    python -m etf_valuation report            # Full valuation report
    python -m etf_valuation report --tier sectors
    python -m etf_valuation screen --signal BUY
    python -m etf_valuation history SPY       # Percentile history
"""
import argparse
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def cmd_sync(args, config):
    """Fetch today's data: ratios-ttm-bulk + treasury rates."""
    from etf_valuation.fetcher import RatiosBulkFetcher

    print("=" * 60)
    print("[etf_valuation] Daily sync")
    print("=" * 60)

    # 1. Fetch FMP ratios
    fmp_key = _get_fmp_key()
    ratios_dir = config.get_storage_path("ratios_bulk")
    fetcher = RatiosBulkFetcher(ratios_dir, fmp_key)
    n = fetcher.sync()
    if n:
        print(f"  Ratios: fetched {n} stocks")
    else:
        print("  Ratios: already up-to-date")

    # 2. Fetch treasury rates
    try:
        from data_sync.config import load_config as load_ds_config
        from data_sync.providers.treasury import FMPTreasuryProvider

        ds_config = load_ds_config()
        treasury = FMPTreasuryProvider(ds_config)
        treasury.sync_with_status(None)
    except Exception as e:
        print(f"  Treasury: {e}")

    print()


def cmd_backfill(args, config):
    """Historical backfill: N-PORT holdings + CUSIP mapping + Shiller."""
    from etf_valuation.nport import NPortDownloader
    from etf_valuation.cusip_map import CusipMapper
    from etf_valuation.shiller import ShillerLoader

    print("=" * 60)
    print("[etf_valuation] Historical backfill")
    print("=" * 60)

    fmp_key = _get_fmp_key()
    equity_etfs = config.equity_etfs()
    tickers = [e.ticker for e in equity_etfs]

    if args.only:
        tickers = [t for t in tickers if t in args.only]

    # 1. Download N-PORT filings
    print("\n--- Phase 1: SEC N-PORT Holdings ---")
    nport_dir = config.get_storage_path("nport")
    with NPortDownloader(nport_dir) as dl:
        results = dl.sync_all(tickers, force=args.force)
        total_new = sum(v for v in results.values() if v > 0)
        errors = sum(1 for v in results.values() if v < 0)
        print(f"\n  N-PORT: {total_new} new quarters, {errors} errors")

    # 2. Build CUSIP → ticker mapping
    print("\n--- Phase 2: CUSIP Mapping ---")
    cache_dir = config.get_storage_path("cusip_cache")
    mapper = CusipMapper(cache_dir, fmp_api_key=fmp_key)

    # Collect all unique CUSIPs from downloaded N-PORT filings
    all_cusips = set()
    for ticker in tickers:
        etf_dir = nport_dir / ticker
        if not etf_dir.exists():
            continue
        for qf in etf_dir.glob("*.parquet"):
            df = pd.read_parquet(qf, columns=["cusip"])
            all_cusips.update(c for c in df["cusip"].unique() if c != "000000000")

    unmapped = mapper.get_unmapped_cusips(
        pd.DataFrame({"cusip": list(all_cusips)})
    )
    print(f"  Total unique CUSIPs: {len(all_cusips)}, unmapped: {len(unmapped)}")

    if unmapped or args.force:
        n = mapper.build_from_figi(unmapped if unmapped else list(all_cusips))
        print(f"  CUSIP map: {n} total entries")

    # 3. Map CUSIPs to tickers in all holdings
    print("\n--- Phase 3: Map Holdings ---")
    for ticker in tickers:
        etf_dir = nport_dir / ticker
        if not etf_dir.exists():
            continue
        for qf in etf_dir.glob("*.parquet"):
            df = pd.read_parquet(qf)
            if "ticker" in df.columns and df["ticker"].notna().sum() > 0:
                continue
            df = mapper.batch_map(df)
            stats = mapper.coverage_stats(df)
            df.to_parquet(qf, compression="snappy", index=False)
            print(f"  {ticker}/{qf.stem}: {stats['matched']}/{stats['total_holdings']} mapped "
                  f"({stats['weight_coverage']:.0%} weight)")

    # 4. Shiller data
    print("\n--- Phase 4: Shiller S&P 500 ---")
    shiller_dir = config.get_storage_path("shiller")
    loader = ShillerLoader(shiller_dir)
    shiller_path = shiller_dir / "sp500_shiller.parquet"
    if not shiller_path.exists() or args.force:
        n = loader.sync()
        print(f"  Shiller: {n} months")
    else:
        print(f"  Shiller data exists (use --force to rebuild)")

    print("\n  Backfill complete.\n")


def cmd_report(args, config):
    """Generate valuation report."""
    from etf_valuation.fetcher import RatiosBulkFetcher
    from etf_valuation.nport import NPortDownloader
    from etf_valuation.cusip_map import CusipMapper
    from etf_valuation.metrics import aggregate_etf_metrics
    from etf_valuation.scoring import score_etf
    from etf_valuation.report import print_valuation_report, export_csv, export_json

    fmp_key = _get_fmp_key()

    # Load latest ratios
    ratios_dir = config.get_storage_path("ratios_bulk")
    fetcher = RatiosBulkFetcher(ratios_dir, fmp_key)
    ratios = fetcher.load_latest()
    if ratios.empty:
        print("No ratios data. Run 'python -m etf_valuation sync' first.")
        return

    # Load treasury rate for ERP
    treasury_10y = _get_treasury_10y(config)

    # Load holdings and compute
    nport_dir = config.get_storage_path("nport")
    dl = NPortDownloader(nport_dir)

    scores = []
    equity_etfs = config.equity_etfs()

    for etf_def in equity_etfs:
        holdings = dl.load_holdings(etf_def.ticker)
        if holdings.empty or "ticker" not in holdings.columns:
            continue

        metrics_to_compute = [etf_def.primary]
        if etf_def.secondary:
            metrics_to_compute.append(etf_def.secondary)

        result = aggregate_etf_metrics(
            holdings, ratios,
            metrics=metrics_to_compute,
            treasury_10y=treasury_10y,
        )

        pri_data = result.get(etf_def.primary, {})
        sec_data = result.get(etf_def.secondary, {}) if etf_def.secondary else {}

        # For now, use current value as percentile placeholder (needs history)
        # TODO: compute actual percentile from historical snapshots
        score = score_etf(
            primary_pct=float("nan"),  # placeholder until history built
            secondary_pct=float("nan"),
            primary_metric=etf_def.primary,
            secondary_metric=etf_def.secondary,
            primary_weight=config.primary_weight,
            secondary_weight=config.secondary_weight,
        )

        score.update({
            "ticker": etf_def.ticker,
            "primary_value": pri_data.get("value"),
            "secondary_value": sec_data.get("value"),
            "primary_coverage": pri_data.get("coverage"),
            "primary_reliable": pri_data.get("reliable"),
        })
        scores.append(score)

    print_valuation_report(
        scores, config,
        tier_filter=args.tier,
        signal_filter=args.signal,
    )

    if args.csv:
        export_csv(scores, Path(args.csv))
    if args.json:
        export_json(scores, Path(args.json))


def cmd_screen(args, config):
    """Screen ETFs by signal."""
    args.tier = None
    args.csv = None
    args.json = None
    cmd_report(args, config)


def cmd_status(args, config):
    """Show data availability status."""
    from etf_valuation.nport import NPortDownloader

    print("=" * 60)
    print("[etf_valuation] Data Status")
    print("=" * 60)

    # N-PORT coverage
    nport_dir = config.get_storage_path("nport")
    dl = NPortDownloader(nport_dir)

    equity_etfs = config.equity_etfs()
    print(f"\nN-PORT Holdings ({len(equity_etfs)} equity ETFs):")
    for etf in equity_etfs:
        quarters = dl.list_quarters(etf.ticker)
        if quarters:
            print(f"  {etf.ticker:<6} {len(quarters):>3} quarters  ({quarters[0]} → {quarters[-1]})")
        else:
            print(f"  {etf.ticker:<6}   - no data")

    # Ratios coverage
    ratios_dir = config.get_storage_path("ratios_bulk")
    ratios_files = sorted(ratios_dir.glob("*_ratios_ttm.parquet"))
    print(f"\nRatios Snapshots:")
    for f in ratios_files:
        df = pd.read_parquet(f, columns=["fetch_date"])
        dates = sorted(df["fetch_date"].unique())
        print(f"  {f.name}: {len(dates)} snapshots ({dates[0]} → {dates[-1]})")

    # Shiller
    shiller_path = config.get_storage_path("shiller") / "sp500_shiller.parquet"
    if shiller_path.exists():
        df = pd.read_parquet(shiller_path)
        print(f"\nShiller: {len(df)} months ({df.index.min().date()} → {df.index.max().date()})")
    else:
        print("\nShiller: not downloaded")

    # Treasury
    treasury_path = config.storage_root / "treasury" / "treasury_rates.parquet"
    if treasury_path.exists():
        df = pd.read_parquet(treasury_path)
        print(f"Treasury: {len(df)} dates")
    else:
        print("Treasury: not synced")

    print()


def _get_fmp_key() -> str:
    try:
        from data_sync.config import load_config as load_ds_config
        ds_config = load_ds_config()
        return ds_config.get_api_key("fmp")
    except Exception:
        raise RuntimeError("FMP API key not found. Configure data_sync/config.yaml first.")


def _get_treasury_10y(config) -> Optional[float]:
    """Get latest 10Y treasury rate."""
    treasury_path = config.storage_root / "treasury" / "treasury_rates.parquet"
    if not treasury_path.exists():
        return None

    try:
        df = pd.read_parquet(treasury_path)
        if df.empty or "year10" not in df.columns:
            return None
        latest = df.sort_values("date").iloc[-1]
        rate = latest.get("year10")
        if pd.notna(rate):
            return float(rate) / 100.0
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        prog="etf_valuation",
        description="US ETF Valuation Monitoring System",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to etf_universe.yaml",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    sub = parser.add_subparsers(dest="command")

    # sync
    sub.add_parser("sync", help="Fetch today's ratios + treasury rates")

    # backfill
    bf = sub.add_parser("backfill", help="Historical backfill (N-PORT + Shiller)")
    bf.add_argument("--only", nargs="+", help="Only backfill these tickers")
    bf.add_argument("--force", action="store_true", help="Force re-download")

    # report
    rp = sub.add_parser("report", help="Valuation report")
    rp.add_argument("--tier", type=str, help="Filter by tier (broad/sectors/themes/factors/international)")
    rp.add_argument("--signal", type=str, help="Filter by signal (STRONG_BUY/BUY/etc)")
    rp.add_argument("--csv", type=str, help="Export to CSV file")
    rp.add_argument("--json", type=str, help="Export to JSON file")

    # screen
    sc = sub.add_parser("screen", help="Screen ETFs by signal")
    sc.add_argument("--signal", type=str, required=True, help="Signal to filter (BUY/SELL/etc)")

    # status
    sub.add_parser("status", help="Show data availability")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(0)

    from etf_valuation.config import load_config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if args.command == "sync":
        cmd_sync(args, config)
    elif args.command == "backfill":
        cmd_backfill(args, config)
    elif args.command == "report":
        cmd_report(args, config)
    elif args.command == "screen":
        cmd_screen(args, config)
    elif args.command == "status":
        cmd_status(args, config)
