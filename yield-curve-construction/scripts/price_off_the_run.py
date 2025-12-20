#!/usr/bin/env python3
"""
Price off-the-run Treasuries against the bootstrapped discount curve.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import pandas as pd

from build_curve import bootstrap_curve_from_dataframe, dataframe_to_instruments
from select_otr_treasuries import (
    DEFAULT_BUCKETS,
    load_data,
    split_otr_offtherun,
)
from curves.bootstrapping.daycount import yearfrac
from pricing import BondPricer
from pricing.z_spread import solve_z_spread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Price off-the-run Treasuries with the fitted curve.")
    parser.add_argument("-i", "--input", required=True, help="Input CRSP file (csv or parquet).")
    parser.add_argument("-d", "--date", required=True, help="Trading date (YYYY-MM-DD).")
    parser.add_argument(
        "-o",
        "--output",
        default="off_the_run_pricing.csv",
        help="Output CSV containing pricing diagnostics.",
    )
    parser.add_argument(
        "--interpolator",
        choices=["loglinear", "cubic"],
        default="loglinear",
        help="Interpolator for the bootstrapped curve.",
    )
    parser.add_argument(
        "--no-liquidity",
        action="store_true",
        help="Disable liquidity tiebreak when selecting OTR securities.",
    )
    parser.add_argument("--date-col", default="CALDT", help="Date column name.")
    parser.add_argument("--maturity-col", default="TMATDT", help="Maturity column name.")
    parser.add_argument("--id-col", default="KYTREASNO", help="Security id column name.")
    parser.add_argument("--bid-col", default="TDBID", help="Bid column name.")
    parser.add_argument("--ask-col", default="TDASK", help="Ask column name.")
    parser.add_argument("--out-col", default="TDPUBOUT", help="Outstanding column name.")
    parser.add_argument(
        "--spread-bracket",
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        default=(-0.05, 0.05),
        help="Bracket (decimal) for the z-spread solver.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="Number of priced bonds to display in stdout summary.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    day = load_data(args.input, args.date)
    if len(day) == 0:
        print(f"No records found for {args.date}.", file=sys.stderr)
        sys.exit(1)

    otr_df, off_df, _ = split_otr_offtherun(
        day=day,
        buckets=DEFAULT_BUCKETS,
        date_col=args.date_col,
        maturity_col=args.maturity_col,
        id_col=args.id_col,
        bid_col=args.bid_col,
        ask_col=args.ask_col,
        out_col=args.out_col,
        use_liquidity_tiebreak=not args.no_liquidity,
    )

    if len(otr_df) == 0:
        print("No OTR securities available to bootstrap.", file=sys.stderr)
        sys.exit(1)
    if len(off_df) == 0:
        print("No off-the-run securities to price.", file=sys.stderr)
        sys.exit(1)

    curve, nodes, _, _, errors = bootstrap_curve_from_dataframe(
        otr_df, interpolator=args.interpolator
    )
    if errors:
        print(f"Skipped {len(errors)} OTR entries during conversion.", file=sys.stderr)

    off_instruments, off_errors = dataframe_to_instruments(off_df)
    if off_errors:
        print(f"Skipped {len(off_errors)} off-the-run entries.", file=sys.stderr)
    if not off_instruments:
        print("No valid off-the-run instruments to price.", file=sys.stderr)
        sys.exit(1)

    pricer = BondPricer(curve)
    priced_rows = []
    spread_lo, spread_hi = args.spread_bracket
    for inst in off_instruments:
        result = pricer.price(inst)
        try:
            spread = solve_z_spread(
                inst,
                curve,
                target_dirty_price=inst.dirty_price,
                lo=spread_lo,
                hi=spread_hi,
            )
        except ValueError as exc:
            spread = float("nan")
            print(f"Warning: z-spread failed for {inst.cusip}: {exc}", file=sys.stderr)

        ttm = yearfrac(inst.val_date, inst.maturity_date)
        priced_rows.append(
            {
                "key": inst.key,
                "cusip": inst.cusip,
                "ttm": ttm,
                "dirty_market": inst.dirty_price,
                "dirty_model": result.dirty_price,
                "clean_model": result.clean_price,
                "price_diff": result.dirty_price - inst.dirty_price,
                "z_spread_bps": spread * 10000 if math.isfinite(spread) else float("nan"),
            }
        )

    priced_df = pd.DataFrame(priced_rows).sort_values("ttm").reset_index(drop=True)
    priced_df.to_csv(args.output, index=False)

    diffs = priced_df["price_diff"].dropna().tolist()
    avg_diff = statistics.mean(diffs) if diffs else 0.0
    print("=== Off-the-run pricing summary ===")
    print(f"Valuation date: {curve.val_date.date()}")
    print(f"OTR securities bootstrapped: {len(otr_df)}")
    print(f"Off-the-run priced: {len(priced_df)} saved to {Path(args.output).resolve()}")
    print(f"Mean dirty price diff (model - market): {avg_diff:.4f}")
    print(priced_df.head(args.preview).to_string(index=False))


if __name__ == "__main__":
    main()
