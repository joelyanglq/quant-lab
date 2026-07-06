"""
Report generator for ETF valuation system.

Outputs console tables and CSV/JSON exports.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from etf_valuation.config import METRICS, ValuationConfig
from etf_valuation.scoring import SIGNALS


def format_pct(val: float) -> str:
    if pd.isna(val):
        return "N/A"
    return f"{val:.0%}"


def format_value(val: float, metric: str) -> str:
    if pd.isna(val):
        return "N/A"
    if metric in ("div_yield", "fcf_yield", "erp"):
        return f"{val:.2%}"
    return f"{val:.1f}"


def signal_color(signal: str) -> str:
    """ANSI color code for signal."""
    colors = {
        "STRONG_BUY": "\033[92m",   # bright green
        "BUY": "\033[32m",          # green
        "LEAN_BUY": "\033[33m",     # yellow
        "HOLD": "\033[37m",         # white
        "LEAN_SELL": "\033[33m",    # yellow
        "SELL": "\033[31m",         # red
        "STRONG_SELL": "\033[91m",  # bright red
    }
    return colors.get(signal, "\033[0m")


RESET = "\033[0m"


def print_valuation_report(
    scores: List[Dict],
    config: ValuationConfig,
    tier_filter: Optional[str] = None,
    signal_filter: Optional[str] = None,
    sort_by: str = "composite_pct",
):
    """
    Print a formatted valuation report to console.

    Args:
        scores: list of score dicts from scoring.score_etf() with etf ticker
        config: valuation config
        tier_filter: only show ETFs in this tier
        signal_filter: only show ETFs with this signal
        sort_by: column to sort by
    """
    if tier_filter:
        tier_tickers = {e.ticker for e in config.etfs_by_tier(tier_filter)}
        scores = [s for s in scores if s.get("ticker") in tier_tickers]

    if signal_filter:
        scores = [s for s in scores if s.get("composite_signal") == signal_filter]

    scores = sorted(scores, key=lambda s: s.get(sort_by, 0.5))

    # Print header
    print()
    print(f"{'ETF':<6} {'Name':<16} {'Pri.Metric':<10} {'Value':>8} {'Pct':>6} "
          f"{'Signal':<12} {'Sec.Metric':<10} {'Value':>8} {'Pct':>6} "
          f"{'Comp':>6} {'Action':<12}")
    print("-" * 110)

    for s in scores:
        ticker = s.get("ticker", "?")
        etf_def = config.etfs.get(ticker)
        name = etf_def.name[:14] if etf_def else ""

        pri_metric = s.get("primary_metric", "")
        pri_val = s.get("primary_value", float("nan"))
        pri_pct = s.get("primary_pct", float("nan"))
        pri_sig = s.get("primary_signal", "N/A")

        sec_metric = s.get("secondary_metric", "")
        sec_val = s.get("secondary_value", float("nan"))
        sec_pct = s.get("secondary_pct", float("nan"))

        comp_pct = s.get("composite_pct", float("nan"))
        comp_sig = s.get("composite_signal", "N/A")

        color = signal_color(comp_sig)

        print(
            f"{ticker:<6} {name:<16} {pri_metric:<10} "
            f"{format_value(pri_val, pri_metric):>8} {format_pct(pri_pct):>6} "
            f"{color}{pri_sig:<12}{RESET} "
            f"{sec_metric or '-':<10} "
            f"{format_value(sec_val, sec_metric) if sec_metric else '-':>8} "
            f"{format_pct(sec_pct) if sec_metric else '-':>6} "
            f"{format_pct(comp_pct):>6} "
            f"{color}{comp_sig:<12}{RESET}"
        )

    print("-" * 110)
    print(f"Total: {len(scores)} ETFs")

    # Summary by signal
    signal_counts = {}
    for s in scores:
        sig = s.get("composite_signal", "N/A")
        signal_counts[sig] = signal_counts.get(sig, 0) + 1

    print("\nSignal Distribution:")
    for sig in ["STRONG_BUY", "BUY", "LEAN_BUY", "HOLD", "LEAN_SELL", "SELL", "STRONG_SELL"]:
        count = signal_counts.get(sig, 0)
        if count:
            label = SIGNALS.get(sig)
            color = signal_color(sig)
            print(f"  {color}{sig:<12}{RESET} {label.label if label else sig}: {count}")


def export_csv(scores: List[Dict], output_path: Path):
    """Export scores to CSV."""
    df = pd.DataFrame(scores)
    df.to_csv(output_path, index=False)
    print(f"Exported to {output_path}")


def export_json(scores: List[Dict], output_path: Path):
    """Export scores to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, default=str, ensure_ascii=False)
    print(f"Exported to {output_path}")
