"""DataContext — PIT-safe heterogeneous data registry.

All strategies read data via `ctx.as_of(dt, key)` and the framework guarantees
no future data leaks. Backtest backend reads Parquet; live backend mixes
real-time IBKR bars with historical Parquet (see runtime/live.py).

Supported keys:
    'price_1d', 'price_1min'  -> DataFrame indexed by timestamp, columns=symbols
    'fundamentals'            -> DataFrame indexed by (symbol, acceptedDate)
    'gics_sector'             -> Series mapping symbol -> sector at dt
    'ewma_vol'                -> Series of per-symbol annualized EWMA vol
    'cointegration_pairs'     -> DataFrame of currently-valid pairs
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .events import Frequency


class DataContext(ABC):
    """Abstract heterogeneous data registry with PIT semantics."""

    @abstractmethod
    def as_of(self, dt: pd.Timestamp, key: str, **kwargs) -> Any:
        """Return data for `key` known no later than `dt`."""

    @abstractmethod
    def universe(self, dt: pd.Timestamp) -> list[str]:
        """Return tradable symbol list at `dt`."""


class _LRUCache:
    """Bounded LRU cache for DataFrame-valued entries."""

    def __init__(self, maxsize: int = 64):
        self._d: OrderedDict = OrderedDict()
        self._max = maxsize

    def get(self, key):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self._max:
            self._d.popitem(last=False)


class BacktestDataContext(DataContext):
    """Parquet-backed PIT data context.

    Args:
        data_root: project data dir, e.g. 'D:/04_Project/quant-lab/data'.
        symbols: full universe; if omitted, derived from index_symbols.json.
        ewma_halflife: EWMA half-life in days for vol estimation (default 36 per Carver Ch9).
    """

    def __init__(
        self,
        data_root: str | Path,
        symbols: Optional[list[str]] = None,
        ewma_halflife: int = 36,
    ):
        self.root = Path(data_root)
        self._symbols = symbols
        self._ewma_halflife = ewma_halflife
        self._cache = _LRUCache(maxsize=8)
        self._price_1d: pd.DataFrame | None = None
        self._price_1min: pd.DataFrame | None = None

    # ── universe ────────────────────────────────────────────────────────
    def universe(self, dt: pd.Timestamp) -> list[str]:
        if self._symbols is not None:
            return list(self._symbols)
        idx_file = self.root / "_index_symbols.json"
        if idx_file.exists():
            import json
            data = json.loads(idx_file.read_text(encoding="utf-8"))
            syms = sorted(set(data.get("sp500", []) + data.get("ndx100", [])))
            self._symbols = syms
            return syms
        raise FileNotFoundError(f"No symbol list available at {idx_file}")

    # ── price panels ────────────────────────────────────────────────────
    def _load_prices(self, frequency: Frequency) -> pd.DataFrame:
        """Load price data for the given frequency. Cached in memory.

        - 1d: read yearly files matching *_{frequency}.parquet
        - 1min: read monthly files matching YYYY-MM_1m.parquet and ignore yearly 1m files
        """
        attr = "_price_1d" if frequency == Frequency.EOD else "_price_1min"
        cached = getattr(self, attr)
        if cached is not None:
            return cached

        subdir = "bars_1d" if frequency == Frequency.EOD else "bars_1min"
        candidates = [self.root / "processed" / subdir, self.root / subdir]
        bars_dir = next((c for c in candidates if c.exists()), None)
        if bars_dir is None or not bars_dir.exists():
            raise FileNotFoundError(
                f"Bar data dir not found under {self.root} (tried {' or '.join(str(c) for c in candidates)})"
            )

        frames = []
        if frequency == Frequency.EOD:
            files = sorted(bars_dir.glob(f"*_{frequency.value}.parquet"))
        elif frequency == Frequency.MIN_1:
            files = sorted(bars_dir.glob("????-??_1m.parquet"))
        else:
            files = []

        for f in files:
            df = pd.read_parquet(f)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            elif "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            frames.append(df)

        if not frames:
            raise FileNotFoundError(f"No parquet files in {bars_dir}")

        df = pd.concat(frames, ignore_index=False)
        df = df.sort_index()
        setattr(self, attr, df)
        return df

    def _price_panel(
        self,
        dt: pd.Timestamp,
        frequency: Frequency,
        field: str = "close",
        lookback: int = 504,
        symbols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        df = self._load_prices(frequency)
        df = df[df.index <= dt]
        if lookback is not None:
            df = df.tail(lookback * (len(df["symbol"].unique()) if "symbol" in df.columns else 1))
        if "symbol" in df.columns:
            wide = df.pivot_table(index=df.index, columns="symbol", values=field)
        else:
            wide = df[[field]] if field in df.columns else df
        if symbols is not None:
            wide = wide.reindex(columns=symbols)
        return wide.sort_index().tail(lookback) if lookback else wide.sort_index()

    # ── fundamentals (PIT) ──────────────────────────────────────────────
    def _load_fundamentals(self) -> pd.DataFrame:
        cached = self._cache.get("fundamentals")
        if cached is not None:
            return cached
        fund_dir = self.root / "fundamentals" / "massive"
        if not fund_dir.exists():
            return pd.DataFrame()
        frames = []
        for f in sorted(fund_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(f)
                df["symbol"] = f.stem
                frames.append(df)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        all_f = pd.concat(frames, ignore_index=True)
        # Use acceptedDate (filing date) for PIT; fall back to fillingDate / date.
        for col in ("acceptedDate", "fillingDate", "filingDate", "date"):
            if col in all_f.columns:
                all_f["effective_date"] = pd.to_datetime(all_f[col])
                break
        else:
            return pd.DataFrame()
        self._cache.put("fundamentals", all_f)
        return all_f

    def _fundamentals_as_of(self, dt: pd.Timestamp) -> pd.DataFrame:
        df = self._load_fundamentals()
        if df.empty:
            return df
        return df[df["effective_date"] <= dt]

    # ── sector mapping ──────────────────────────────────────────────────
    def _sector_as_of(self, dt: pd.Timestamp) -> pd.Series:
        f = self.root / "reference" / "gics_sector_map.parquet"
        if not f.exists():
            return pd.Series(dtype=object)
        df = pd.read_parquet(f)
        if "effective_date" in df.columns:
            df = df[pd.to_datetime(df["effective_date"]) <= dt]
        return df.groupby("symbol")["sector"].last()

    # ── EWMA volatility ─────────────────────────────────────────────────
    def _ewma_vol_as_of(
        self,
        dt: pd.Timestamp,
        symbols: Optional[list[str]] = None,
        floor: float = 0.05,
    ) -> pd.Series:
        close = self._price_panel(dt, Frequency.EOD, "close", lookback=252, symbols=symbols)
        ret = close.pct_change()
        ewma = ret.ewm(halflife=self._ewma_halflife, min_periods=20).std()
        annual = ewma.iloc[-1] * np.sqrt(252)
        return annual.clip(lower=floor)

    # ── cointegration pairs ─────────────────────────────────────────────
    def _pairs_as_of(self, dt: pd.Timestamp) -> pd.DataFrame:
        f = self.root / "processed" / "cointegration_pairs" / "valid_pairs.parquet"
        if not f.exists():
            return pd.DataFrame(columns=["symbol_a", "symbol_b", "hedge_ratio", "valid_until"])
        df = pd.read_parquet(f)
        if "valid_from" in df.columns:
            df = df[pd.to_datetime(df["valid_from"]) <= dt]
        if "valid_until" in df.columns:
            df = df[pd.to_datetime(df["valid_until"]) >= dt]
        return df

    # ── unified entry point ─────────────────────────────────────────────
    def as_of(self, dt: pd.Timestamp, key: str, **kwargs) -> Any:
        if key == "price_1d":
            return self._price_panel(dt, Frequency.EOD, **kwargs)
        if key == "price_1min":
            return self._price_panel(dt, Frequency.MIN_1, **kwargs)
        if key == "fundamentals":
            return self._fundamentals_as_of(dt)
        if key == "gics_sector":
            return self._sector_as_of(dt)
        if key == "ewma_vol":
            return self._ewma_vol_as_of(dt, **kwargs)
        if key == "cointegration_pairs":
            return self._pairs_as_of(dt)
        raise KeyError(f"Unknown DataContext key: {key}")
