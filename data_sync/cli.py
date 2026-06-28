"""
data_sync CLI — subcommand-based entry point

Usage:
    python -m data_sync status
    python -m data_sync status --provider prices_1d --symbols AAPL,MSFT
    python -m data_sync status --stale 7
    python -m data_sync sync
    python -m data_sync sync --only prices_1d --from 2018-01-01
    python -m data_sync sync --only prices_1min --limit 50
    python -m data_sync dry-run --only prices_1d
"""
import argparse
import time
import sys
from datetime import date
from typing import List, Optional

from data_sync.config import load_config, ETLConfig
from data_sync.providers import get_provider
from data_sync.status import StatusManager, print_summary, print_symbol_detail, print_stale, refresh_from_storage
from data_sync.symbols import resolve_symbol_selector

PROVIDER_ORDER = ["universe", "prices_1d", "prices_1min", "financials", "analysts"]


def cmd_status(args, config: ETLConfig):
    """Show sync status."""
    with StatusManager.from_config(config) as sm:
        if args.refresh:
            print("Refreshing status from storage...")
            refresh_from_storage(config, sm)

        if args.provider and args.symbols:
            syms = [s.strip() for s in args.symbols.split(",")]
            print_symbol_detail(sm, args.provider, syms)
        elif args.provider and args.stale:
            print_stale(sm, args.provider, args.stale)
        elif args.provider:
            print_symbol_detail(sm, args.provider)
        elif args.stale:
            for prov in PROVIDER_ORDER:
                print_stale(sm, prov, args.stale)
        else:
            print_summary(sm, PROVIDER_ORDER)


def cmd_sync(args, config: ETLConfig):
    """Run sync."""
    from_date = date.fromisoformat(args.from_date) if args.from_date else None
    to_date = date.fromisoformat(args.to_date) if args.to_date else None
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None

    provider_names = args.only if args.only else PROVIDER_ORDER

    with StatusManager.from_config(config) as sm:
        for name in provider_names:
            if name not in PROVIDER_ORDER:
                print(f"[WARN] Unknown provider: {name}, skipping")
                continue

            try:
                provider = get_provider(name, config)
            except ValueError as e:
                print(f"[ERROR] {e}")
                continue

            run_id = sm.start_run(name)
            t0 = time.time()

            try:
                if hasattr(provider, 'sync_with_status'):
                    provider.sync_with_status(
                        sm,
                        from_date=from_date,
                        to_date=to_date,
                        symbols=symbols,
                        force=args.force,
                        limit=args.limit,
                    )
                else:
                    provider.sync(
                        from_date=from_date,
                        to_date=to_date,
                        symbols=symbols,
                        force=args.force,
                    )

                duration = time.time() - t0
                sm.finish_run(run_id, duration_sec=duration)

            except KeyboardInterrupt:
                duration = time.time() - t0
                sm.finish_run(run_id, duration_sec=duration)
                print(f"\n[{name}] Interrupted after {duration:.0f}s. Progress saved.")
                sys.exit(1)
            except Exception as e:
                duration = time.time() - t0
                sm.finish_run(run_id, symbols_err=1, duration_sec=duration)
                print(f"[ERROR] {name}: {e}")


def cmd_dry_run(args, config: ETLConfig):
    """Preview sync plan without executing."""
    from_date = date.fromisoformat(args.from_date) if args.from_date else None
    to_date = date.fromisoformat(args.to_date) if args.to_date else None
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None

    provider_names = args.only if args.only else PROVIDER_ORDER

    print("=" * 60)
    print("DRY RUN — Preview Only (no API calls)")
    print("=" * 60)

    for name in provider_names:
        if name not in PROVIDER_ORDER:
            print(f"  [SKIP] Unknown provider: {name}")
            continue

        try:
            provider = get_provider(name, config)
        except ValueError as e:
            print(f"  [ERROR] {e}")
            continue

        print(f"\n  [{name}]")
        print(f"    from_date: {from_date or 'auto'}")
        print(f"    to_date:   {to_date or 'today'}")
        print(f"    symbols:   {symbols or 'default universe'}")
        print(f"    force:     {args.force if hasattr(args, 'force') else False}")

        try:
            st = provider.status()
            for k, v in st.items():
                print(f"    {k}: {v}")
            provider_cfg = config.providers.get(name)
            if provider_cfg and hasattr(provider_cfg, 'symbols'):
                print(f"    default_selector: {provider_cfg.symbols}")
                if provider_cfg.symbols:
                    try:
                        n_syms = len(resolve_symbol_selector(config, provider_name=name))
                        print(f"    default_symbol_count: {n_syms}")
                    except Exception:
                        pass
        except Exception as e:
            print(f"    status: error ({e})")

    print()


def main():
    parser = argparse.ArgumentParser(
        prog="data_sync",
        description="Market data ETL pipeline — sync, monitor, and manage",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (default: data_sync/config.yaml)",
    )
    sub = parser.add_subparsers(dest="command")

    # ── status ────────────────────────────────────────────────
    st = sub.add_parser("status", help="Show data sync status")
    st.add_argument("--provider", "-p", type=str, default=None,
                     help="Show detail for one provider")
    st.add_argument("--symbols", "-s", type=str, default=None,
                     help="Comma-separated symbols to inspect")
    st.add_argument("--stale", type=int, default=None,
                     help="Show symbols not updated within N days")
    st.add_argument("--refresh", action="store_true",
                     help="Rebuild status DB from existing storage data")

    # ── sync ──────────────────────────────────────────────────
    sy = sub.add_parser("sync", help="Run data sync")
    sy.add_argument("--only", nargs="+", default=None,
                     help="Only sync these providers")
    sy.add_argument("--from", type=str, dest="from_date",
                     help="Start date (YYYY-MM-DD)")
    sy.add_argument("--to", type=str, dest="to_date",
                     help="End date (YYYY-MM-DD)")
    sy.add_argument("--symbols", "-s", type=str, default=None,
                     help="Comma-separated symbol list")
    sy.add_argument("--force", action="store_true",
                     help="Ignore stale checks, force re-fetch")
    sy.add_argument("--limit", type=int, default=None,
                     help="Max symbols to sync per provider (for resumable runs)")

    # ── dry-run ───────────────────────────────────────────────
    dr = sub.add_parser("dry-run", help="Preview sync plan (no API calls)")
    dr.add_argument("--only", nargs="+", default=None)
    dr.add_argument("--from", type=str, dest="from_date")
    dr.add_argument("--to", type=str, dest="to_date")
    dr.add_argument("--symbols", "-s", type=str, default=None)
    dr.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if args.command == "status":
        cmd_status(args, config)
    elif args.command == "sync":
        cmd_sync(args, config)
    elif args.command == "dry-run":
        cmd_dry_run(args, config)
