"""
Symbol universe / selector helpers.

`rus1000` is currently treated as a fixed list persisted in:
    universe/symbol_layer_rus1000.json

The monthly merged 1min coverage was only used to bootstrap this list; routine
syncs do not rebuild it.
"""
import json
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from data_sync.config import ETLConfig

_FIXED_LIST_FILES = {
    "rus1000": "symbol_layer_rus1000.json",
}

_INDEX_FILE_MAP = {
    "sp500": "index_current_sp500.parquet",
    "nasdaq": "index_current_nasdaq.parquet",
    "ndx100": "index_current_nasdaq.parquet",
    "dowjones": "index_current_dowjones.parquet",
    "dow": "index_current_dowjones.parquet",
}


def resolve_symbol_selector(
    config: ETLConfig,
    selector: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> List[str]:
    """Resolve a configured symbol selector into a concrete symbol list.

    Supported selectors:
      - index         → SP500 ∪ Nasdaq 100
      - sp500         → SP500 only
      - nasdaq/ndx100 → Nasdaq 100 only
      - all           → stock_list.parquet active stocks
      - rus1000       → fixed symbol list stored in universe/symbol_layer_rus1000.json
      - comma/plus unions, e.g. "sp500,nasdaq" or "index+rus1000"
    """
    if selector is None and provider_name is not None:
        provider_cfg = config.providers.get(provider_name)
        selector = provider_cfg.symbols if provider_cfg else "index"

    selector = (selector or "index").strip().lower()
    parts = [p.strip() for p in re.split(r"[,+]", selector) if p.strip()]
    if not parts:
        return []

    symbols = set()
    for part in parts:
        symbols.update(_resolve_one(config, part))
    return sorted(symbols)



def _resolve_one(config: ETLConfig, selector: str) -> List[str]:
    universe_dir = config.get_storage_path("universe")

    if selector == "index":
        symbols = set()
        for idx in ["sp500", "nasdaq"]:
            path = universe_dir / _INDEX_FILE_MAP[idx]
            if path.exists():
                symbols.update(_read_symbol_list(path))
        if symbols:
            return sorted(symbols)

        fallback = config.storage_root / "_index_symbols.json"
        if fallback.exists():
            data = json.loads(fallback.read_text(encoding="utf-8"))
            return sorted(set(data.get("symbols", [])))

        raise FileNotFoundError(
            f"No index universe files found in {universe_dir} and no fallback _index_symbols.json"
        )

    if selector in _INDEX_FILE_MAP:
        path = universe_dir / _INDEX_FILE_MAP[selector]
        if not path.exists():
            raise FileNotFoundError(f"Index universe file not found: {path}")
        return _read_symbol_list(path)

    if selector == "all":
        path = universe_dir / "stock_list.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Stock list file not found: {path}")
        df = pd.read_parquet(path)
        if "isActivelyTrading" in df.columns:
            df = df[df["isActivelyTrading"] == True]
        if "type" in df.columns:
            df = df[df["type"] == "stock"]
        return sorted(set(df["symbol"].dropna().tolist()))

    if selector in _FIXED_LIST_FILES:
        path = universe_dir / _FIXED_LIST_FILES[selector]
        if not path.exists():
            raise FileNotFoundError(
                f"Fixed symbol list not found: {path}. "
                f"If you intentionally want to regenerate it, run: python -m data_sync.build_symbol_layers"
            )
        return _read_symbol_list(path)

    raise ValueError(
        f"Unknown symbol selector: {selector!r}. "
        f"Supported: index, sp500, nasdaq, ndx100, all, rus1000"
    )



def _read_symbol_list(path: Path) -> List[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            symbols = data.get("symbols", [])
        elif isinstance(data, list):
            symbols = data
        else:
            raise ValueError(f"Unsupported JSON symbol list structure in {path}")
        return sorted(set(s for s in symbols if s))

    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=["symbol"])
        return sorted(set(df["symbol"].dropna().tolist()))

    raise ValueError(f"Unsupported symbol list file format: {path}")



def build_rus1000_layer(storage_root: Path) -> pd.DataFrame:
    """Manual bootstrap / rebuild utility for the fixed rus1000 list.

    Deprecated for routine use: `rus1000` should normally remain fixed. This
    helper exists only when you explicitly want to regenerate the list from the
    monthly merged 1min parquet universe.

    Output file:
      - universe/symbol_layer_rus1000.json
    """
    storage_root = Path(storage_root)
    bars_dir = storage_root / "bars_1min"
    universe_dir = storage_root / "universe"
    universe_dir.mkdir(parents=True, exist_ok=True)

    month_files = sorted(bars_dir.glob("????-??_1m.parquet"))
    if not month_files:
        raise FileNotFoundError(f"No monthly 1min parquet files found in {bars_dir}")

    stats = {}
    for fp in month_files:
        year_month = fp.stem.split("_")[0]
        df = pd.read_parquet(fp, columns=["symbol"])
        symbols = set(df["symbol"].dropna().tolist())

        for sym in symbols:
            row = stats.get(sym)
            if row is None:
                stats[sym] = {
                    "symbol": sym,
                    "layer": "rus1000",
                    "source": "bars_1min_monthly",
                    "first_month": year_month,
                    "last_month": year_month,
                    "month_count": 1,
                }
            else:
                row["first_month"] = min(row["first_month"], year_month)
                row["last_month"] = max(row["last_month"], year_month)
                row["month_count"] += 1

    out = pd.DataFrame(stats.values()).sort_values(["symbol"]).reset_index(drop=True)

    json_path = universe_dir / "symbol_layer_rus1000.json"
    payload = {
        "layer": "rus1000",
        "source": "bars_1min_monthly",
        "symbol_count": int(len(out)),
        "first_month": str(out["first_month"].min()),
        "last_month": str(out["last_month"].max()),
        "symbols": out["symbol"].tolist(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return out
