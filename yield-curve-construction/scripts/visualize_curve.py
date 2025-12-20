#!/usr/bin/env python3
"""
Visualize discount factors and zero rates from a saved nodes CSV.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot discount curve nodes.")
    parser.add_argument(
        "-c",
        "--curve-file",
        required=True,
        help="CSV file containing columns 't' and 'df' (output of build_curve.py).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="curve.png",
        help="Output image filename (PNG).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    return parser.parse_args()


def load_nodes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "t" not in df.columns or "df" not in df.columns:
        raise ValueError("Curve file must have columns 't' and 'df'.")
    return df


def main():
    args = parse_args()
    nodes_df = load_nodes(args.curve_file).sort_values("t")

    zero_rates = []
    for _, row in nodes_df.iterrows():
        t = row["t"]
        df = row["df"]
        if t <= 0 or df <= 0:
            zero_rates.append(float("nan"))
        else:
            zero_rates.append(-math.log(df) / t)
    nodes_df["zero_rate"] = zero_rates

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(nodes_df["t"], nodes_df["df"], marker="o")
    ax1.set_ylabel("Discount factor")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(nodes_df["t"], nodes_df["zero_rate"] * 100.0, marker="o", color="tab:orange")
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Zero rate (%)")
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Discount Curve")
    fig.tight_layout()

    output_path = Path(args.output)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path.resolve()}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
