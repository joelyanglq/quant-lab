#!/usr/bin/env python3
"""
Bootstrap a Treasury discount curve from OTR securities and persist the nodes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from select_otr_treasuries import (
    DEFAULT_BUCKETS,
    load_data,
    split_otr_offtherun,
)
from curves.bootstrapping.bootstrapper import Bootstrapper
from curves.curve import YieldCurve
from curves.instruments.factory import InstrumentFactory


def dataframe_to_instruments(day_df: pd.DataFrame) -> Tuple[List[object], List[str]]:
    instruments: List[object] = []
    errors: List[str] = []
    for idx, row in day_df.iterrows():
        try:
            inst = InstrumentFactory.from_crsp_row(row)
            instruments.append(inst)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"row={idx}: {exc}")
    return instruments, errors


def bootstrap_curve_from_dataframe(
    day_df: pd.DataFrame,
    *,
    interpolator: str = "loglinear",
) -> Tuple[YieldCurve, List[Tuple[float, float]], List[dict], List[object], List[str]]:
    instruments, errors = dataframe_to_instruments(day_df)
    if not instruments:
        raise ValueError("No valid instruments available to bootstrap.")

    bootstrapper = Bootstrapper()
    nodes, report = bootstrapper.build_discount_curve(instruments)
    curve = YieldCurve(val_date=instruments[0].val_date, nodes=nodes)

    if interpolator == "cubic":
        from curves.interpolation.cubic_spline import CubicSplineLogDF

        curve.fit(CubicSplineLogDF())

    return curve, nodes, report, instruments, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a discount curve from OTR Treasuries.")
    parser.add_argument("-i", "--input", required=True, help="Input CRSP file (csv or parquet).")
    parser.add_argument("-d", "--date", required=True, help="Trading date (YYYY-MM-DD).")
    parser.add_argument(
        "-o",
        "--output",
        default="curve_nodes.csv",
        help="Output CSV to store (t, df) nodes.",
    )
    parser.add_argument(
        "--report-out",
        help="Optional CSV for bootstrap diagnostics (per-instrument PV residuals).",
    )
    parser.add_argument(
        "--selection-report",
        help="Optional CSV for the OTR selection diagnostics.",
    )
    parser.add_argument(
        "--interpolator",
        choices=["loglinear", "cubic"],
        default="loglinear",
        help="Interpolator used for the YieldCurve fit.",
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
    return parser.parse_args()


def main():
    args = parse_args()
    day = load_data(args.input, args.date)
    if len(day) == 0:
        print(f"No records found for {args.date}.", file=sys.stderr)
        sys.exit(1)

    otr_df, off_df, selection_report = split_otr_offtherun(
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
        print("No OTR securities selected; cannot bootstrap.", file=sys.stderr)
        sys.exit(1)

    curve, nodes, bootstrap_report, _, errors = bootstrap_curve_from_dataframe(
        otr_df, interpolator=args.interpolator
    )

    if errors:
        print(f"Skipped {len(errors)} entries during instrument conversion.", file=sys.stderr)

    nodes_df = pd.DataFrame(nodes, columns=["t", "df"])
    nodes_df.to_csv(args.output, index=False)

    if args.report_out:
        pd.DataFrame(bootstrap_report).to_csv(args.report_out, index=False)
    if args.selection_report:
        selection_report.to_csv(args.selection_report, index=False)

    print("=== Curve summary ===")
    print(f"Valuation date: {curve.val_date.date()}")
    print(f"Nodes: {len(nodes)} saved to {Path(args.output).resolve()}")
    print(f"OTR securities used: {len(otr_df)}")
    print(f"Off-the-run inventory: {len(off_df)}")


if __name__ == "__main__":
    main()
