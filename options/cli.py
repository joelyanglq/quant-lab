"""
Options Strategy Manager CLI.

Usage:
    python -m options list                        # available strategies
    python -m options run monthly_ic dte0_ic      # run both (Ctrl+C to stop)
    python -m options run dte0_ic --port 7496     # run one with override
    python -m options summary monthly_ic          # show performance
"""

import argparse
import logging
import signal
import sys
import time

from options.core.manager import StrategyManager, get_registry
import options.strategies  # noqa: F401  triggers registration


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def cmd_list(args, mgr):
    registry = get_registry()
    print(f"\nAvailable strategies ({len(registry)}):")
    for name in sorted(registry):
        print(f"  - {name}")
    print()


def cmd_run(args, mgr):
    overrides = {}
    if args.port:
        overrides["port"] = args.port

    for name in args.names:
        ok = mgr.start(name, **overrides)
        if not ok:
            print(f"Failed to start: {name}")
            mgr.stop_all()
            sys.exit(1)

    print(f"\nRunning: {', '.join(args.names)}")
    print("Press Ctrl+C to stop all.\n")

    def on_sigint(sig, frame):
        print("\nStopping all strategies...")
        mgr.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    while True:
        time.sleep(60)
        for s in mgr.status():
            logging.getLogger("cli").info(
                "[%s] state=%s  positions=%d",
                s.name, s.state.value, s.open_positions,
            )


def cmd_status(args, mgr):
    statuses = mgr.status()
    if not statuses:
        print("No strategies running.")
        return
    print(f"\n{'Name':<15} {'State':<10} {'Positions':<10} {'Last Poll'}")
    print("-" * 55)
    for s in statuses:
        print(f"{s.name:<15} {s.state.value:<10} "
              f"{s.open_positions:<10} {s.last_poll or 'N/A'}")
    print()


def cmd_summary(args, mgr):
    registry = get_registry()
    cls = registry.get(args.name)
    if cls is None:
        print(f"Unknown strategy: {args.name}")
        return
    instance = cls()
    logger = instance.make_logger()
    s = logger.summary()
    print(f"\n=== {args.name} Summary ===")
    for k, v in s.items():
        if k == "tail_losses":
            continue
        if isinstance(v, float):
            print(f"  {k}: {v:,.2f}")
        else:
            print(f"  {k}: {v}")
    print()


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Options Strategy Manager")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available strategies")

    p_run = sub.add_parser("run", help="Run strategies")
    p_run.add_argument("names", nargs="+")
    p_run.add_argument("--port", type=int)

    sub.add_parser("status", help="Show running strategies")

    p_summary = sub.add_parser("summary", help="Show performance summary")
    p_summary.add_argument("name")

    args = parser.parse_args()
    mgr = StrategyManager()

    commands = {
        "list": cmd_list,
        "run": cmd_run,
        "status": cmd_status,
        "summary": cmd_summary,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
    else:
        handler(args, mgr)


if __name__ == "__main__":
    main()
