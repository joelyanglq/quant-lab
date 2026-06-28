"""
Build reusable symbol layers for tiered / stratified universe selection.

Deprecated for routine use: `rus1000` is treated as a fixed list stored in
`universe/symbol_layer_rus1000.json`. Only run this if you explicitly want to
re-bootstrap that fixed list from the monthly merged 1min coverage universe.

Usage:
    python -m data_sync.build_symbol_layers
    python -m data_sync.build_symbol_layers --storage-root E:/stocks
"""
import argparse

from data_sync.config import load_config
from data_sync.symbols import build_rus1000_layer


def main():
    parser = argparse.ArgumentParser(description="Re-bootstrap the fixed rus1000 symbol list from storage")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (default: data_sync/config.yaml)")
    parser.add_argument("--storage-root", type=str, default=None,
                        help="Override storage root (default from config)")
    args = parser.parse_args()

    config = load_config(args.config)
    storage_root = args.storage_root or str(config.storage_root)

    df = build_rus1000_layer(storage_root)
    print("=" * 60)
    print("Rebuilt fixed symbol layer: rus1000")
    print("=" * 60)
    print("  NOTE: this command is deprecated for routine runs.")
    print("  It does NOT fetch SP500/Nasdaq lists from FMP.")
    print("  It only scans existing monthly 1min parquet files under bars_1min/.")
    print(f"  Symbols:     {len(df)}")
    print(f"  First month: {df['first_month'].min()}")
    print(f"  Last month:  {df['last_month'].max()}")
    print(f"  Output:      {config.storage_root / 'universe' / 'symbol_layer_rus1000.json'}")


if __name__ == "__main__":
    main()
