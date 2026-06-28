"""Offline cointegration scanner for pairs.

Scans candidate symbol pairs over a 2-year window using the Engle-Granger
two-step test and persists qualifying pairs (p-value < 0.05) to
`data/processed/cointegration_pairs/valid_pairs.parquet`.

Usage:
    python -m trading_platform.runtime.pairs_scanner \\
        --start 2023-01-01 --end 2025-01-01 --max-pairs 100

Note: this is the standalone scanner. The PairsAlpha consumes its output via
DataContext.as_of(dt, 'cointegration_pairs').
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import coint
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


def scan_pairs(
    prices: pd.DataFrame,
    candidates: Optional[Iterable[tuple[str, str]]] = None,
    p_threshold: float = 0.05,
    min_obs: int = 252,
) -> pd.DataFrame:
    """Engle-Granger cointegration scan.

    Args:
        prices: wide price panel (date × symbol).
        candidates: iterable of (a, b) symbol pairs; default = all pairs.
        p_threshold: max p-value for "cointegrated" classification.
        min_obs: minimum overlapping observations.

    Returns:
        DataFrame with columns: symbol_a, symbol_b, p_value, hedge_ratio,
        valid_from, valid_until.
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels is required for cointegration scanning")

    cols = list(prices.columns)
    if candidates is None:
        candidates = itertools.combinations(cols, 2)

    rows = []
    for a, b in candidates:
        if a not in prices.columns or b not in prices.columns:
            continue
        pair = prices[[a, b]].dropna()
        if len(pair) < min_obs:
            continue
        log_a = np.log(pair[a].values)
        log_b = np.log(pair[b].values)
        try:
            score, p, _ = coint(log_a, log_b)
        except Exception:
            continue
        if p > p_threshold:
            continue
        # Hedge ratio via OLS log_a ~ β log_b.
        beta = np.polyfit(log_b, log_a, 1)[0]
        rows.append({
            "symbol_a": a,
            "symbol_b": b,
            "p_value": float(p),
            "hedge_ratio": float(beta),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--symbols", default=None,
                    help="comma-separated list; default reads _index_symbols.json")
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--min-obs", type=int, default=252)
    ap.add_argument("--out", default="data/processed/cointegration_pairs/valid_pairs.parquet")
    args = ap.parse_args()

    from ..core.events import Frequency
    from ..data.storage.parquet import ParquetStorage

    if args.symbols:
        symbols = args.symbols.split(",")
    else:
        import json
        idx_file = Path(args.data_root) / "_index_symbols.json"
        data = json.loads(idx_file.read_text(encoding="utf-8"))
        symbols = sorted(set(data.get("sp500", []) + data.get("ndx100", [])))

    storage = ParquetStorage(args.data_root, Frequency.EOD)
    df = storage.load(symbols, pd.Timestamp(args.start), pd.Timestamp(args.end))
    if df.empty:
        print("No price data in range")
        return

    wide = df.pivot_table(index=df.index, columns="symbol", values="close").sort_index()
    print(f"Scanning {len(wide.columns)} symbols, {len(wide)} dates...")
    result = scan_pairs(wide, p_threshold=args.p, min_obs=args.min_obs)
    if result.empty:
        print("No cointegrated pairs found")
        return

    result["valid_from"] = pd.Timestamp(args.start)
    result["valid_until"] = pd.Timestamp(args.end) + pd.Timedelta(days=90)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"Wrote {len(result)} pairs to {out}")


if __name__ == "__main__":
    main()
