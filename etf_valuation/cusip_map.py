"""
CUSIP → ticker mapping.

SEC N-PORT filings identify holdings by CUSIP (9-char) and optionally ISIN.
We need to map these to tickers to join with FMP ratio data.

Strategy (multi-layer):
  1. FMP /profile per-symbol — has cusip + isin fields (batch via concurrent requests)
  2. OpenFIGI API — free bulk CUSIP→ticker lookup (10 items/req)
  3. Name fuzzy matching — fallback for remaining misses
"""
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FIGI_URL = "https://api.openfigi.com/v3/mapping"


class CusipMapper:
    """Maps CUSIP codes to stock tickers."""

    def __init__(self, cache_dir: Path, fmp_api_key: Optional[str] = None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = cache_dir / "cusip_to_ticker.parquet"
        self._fmp_api_key = fmp_api_key
        self._map: Optional[Dict[str, str]] = None

    def _load_cache(self) -> Dict[str, str]:
        if self._cache_path.exists():
            df = pd.read_parquet(self._cache_path)
            return dict(zip(df["cusip"], df["ticker"]))
        return {}

    def _save_cache(self, mapping: Dict[str, str]):
        df = pd.DataFrame(list(mapping.items()), columns=["cusip", "ticker"])
        df.to_parquet(self._cache_path, compression="snappy", index=False)

    def build_from_fmp_profiles(self, symbols: Optional[List[str]] = None):
        """
        Build CUSIP→ticker map by querying FMP /profile for each symbol.

        Uses concurrent requests (6 workers) with rate limiting.
        If symbols is None, fetches all from FMP /stock-list.
        """
        if not self._fmp_api_key:
            raise ValueError("FMP API key required")

        from data_sync.client.fmp import FMPClient

        client = FMPClient(api_key=self._fmp_api_key)

        if symbols is None:
            stock_list = client.get_json("/stock-list")
            symbols = [s["symbol"] for s in stock_list if s.get("symbol")]
            # Filter to likely equities (no dots, reasonable length)
            symbols = [s for s in symbols if "." not in s and len(s) <= 5]

        mapping = self._load_cache()
        known_cusips = set(mapping.keys())

        def fetch_profile(sym):
            try:
                data = client.get_json("/profile", symbol=sym)
                if data and isinstance(data, list) and data[0].get("cusip"):
                    return data[0]["cusip"], sym
            except Exception:
                pass
            return None, sym

        new_count = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(fetch_profile, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                cusip, sym = fut.result()
                if cusip and cusip not in known_cusips:
                    mapping[cusip] = sym
                    known_cusips.add(cusip)
                    new_count += 1

                if new_count > 0 and new_count % 500 == 0:
                    self._save_cache(mapping)
                    print(f"    CUSIP map checkpoint: {len(mapping)} entries ({new_count} new)")

        client.close()
        self._save_cache(mapping)
        self._map = mapping
        logger.info("Built CUSIP map: %d entries (%d new)", len(mapping), new_count)
        return len(mapping)

    def build_from_figi(self, cusips: List[str]) -> int:
        """
        Build CUSIP→ticker map via OpenFIGI API (free, no key needed).

        Rate limit: unauthenticated = 5 req/min, 10 items/req.
        So max 50 CUSIPs/min. For 500 CUSIPs takes ~10 min.
        """
        mapping = self._load_cache()
        unmapped = [c for c in cusips if c not in mapping]

        if not unmapped:
            self._map = mapping
            return len(mapping)

        print(f"    OpenFIGI: looking up {len(unmapped)} unmapped CUSIPs...")
        session = requests.Session()
        new_count = 0

        for i in range(0, len(unmapped), 10):
            batch = unmapped[i:i+10]
            payload = [{"idType": "ID_CUSIP", "idValue": c} for c in batch]

            for attempt in range(3):
                try:
                    resp = session.post(FIGI_URL, json=payload, timeout=30)
                    if resp.status_code == 200:
                        for j, item in enumerate(resp.json()):
                            data = item.get("data", [])
                            if data:
                                ticker = data[0].get("ticker")
                                if ticker and batch[j] not in mapping:
                                    # Normalize tickers: BRK/B → BRK-B
                                    ticker = ticker.replace("/", "-")
                                    mapping[batch[j]] = ticker
                                    new_count += 1
                        break
                    elif resp.status_code == 429:
                        time.sleep(15)
                    else:
                        break
                except Exception as e:
                    logger.warning("FIGI error: %s", e)
                    time.sleep(5)

            # Rate limit: ~5 req/min for unauthenticated
            if i + 10 < len(unmapped):
                time.sleep(12)

            if new_count > 0 and new_count % 100 == 0:
                self._save_cache(mapping)

        session.close()
        self._save_cache(mapping)
        self._map = mapping
        print(f"    OpenFIGI: mapped {new_count} new CUSIPs, total: {len(mapping)}")
        return len(mapping)

    def build_from_name_match(
        self, holdings_df: pd.DataFrame, ratios_df: pd.DataFrame
    ) -> int:
        """
        Fallback: match N-PORT holdings to FMP ratios by company name.

        Simple approach: normalize names and find exact matches.
        """
        mapping = self._load_cache()
        unmapped = holdings_df[
            ~holdings_df["cusip"].isin(mapping) & holdings_df["name"].notna()
        ]

        if unmapped.empty:
            self._map = mapping
            return len(mapping)

        # Build name → symbol lookup from ratios data
        if "symbol" not in ratios_df.columns:
            self._map = mapping
            return len(mapping)

        ratios_symbols = set(ratios_df["symbol"].unique())

        # Common name normalization patterns
        name_to_cusip = {}
        for _, row in unmapped.iterrows():
            name = _normalize_name(row["name"])
            if name:
                name_to_cusip[name] = row["cusip"]

        new_count = 0
        # For each FMP symbol, try to match against N-PORT names
        # This is a heuristic — won't catch everything
        for sym in ratios_symbols:
            # Try obvious matches: symbol itself as part of the name
            for name, cusip in name_to_cusip.items():
                if cusip in mapping:
                    continue
                # Check if the symbol appears as a word in the name
                norm_sym = sym.lower().replace("-", "")
                if norm_sym in name.split():
                    mapping[cusip] = sym
                    new_count += 1

        self._save_cache(mapping)
        self._map = mapping
        return len(mapping)

    def get_ticker(self, cusip: str, isin: Optional[str] = None) -> Optional[str]:
        """Look up ticker for a CUSIP, with ISIN fallback."""
        if self._map is None:
            self._map = self._load_cache()

        ticker = self._map.get(cusip)
        if ticker:
            return ticker

        # Try ISIN → CUSIP extraction
        if isin and len(isin) >= 11:
            alt_cusip = isin[2:11]
            ticker = self._map.get(alt_cusip)
            if ticker:
                return ticker

        return None

    def batch_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'ticker' column to a holdings DataFrame."""
        if self._map is None:
            self._map = self._load_cache()

        tickers = []
        for _, row in df.iterrows():
            cusip = row.get("cusip", "")
            isin = row.get("isin")
            tickers.append(self.get_ticker(cusip, isin))

        df = df.copy()
        df["ticker"] = tickers
        return df

    def coverage_stats(self, df: pd.DataFrame) -> dict:
        """Report mapping coverage for a holdings DataFrame."""
        mapped = df if "ticker" in df.columns else self.batch_map(df)
        total = len(mapped)
        matched = int(mapped["ticker"].notna().sum())
        weight_matched = mapped.loc[mapped["ticker"].notna(), "weight"].sum()
        weight_total = mapped["weight"].sum()

        return {
            "total_holdings": total,
            "matched": matched,
            "unmatched": total - matched,
            "match_rate": matched / total if total else 0,
            "weight_coverage": weight_matched / weight_total if weight_total else 0,
        }

    def get_unmapped_cusips(self, df: pd.DataFrame) -> List[str]:
        """Get list of CUSIPs not yet in the mapping."""
        if self._map is None:
            self._map = self._load_cache()
        return [c for c in df["cusip"].unique() if c not in self._map and c != "000000000"]


def _normalize_name(name: str) -> str:
    """Normalize company name for matching."""
    name = name.lower().strip()
    # Remove common suffixes
    for suffix in [" inc", " inc.", " corp", " corp.", " ltd", " ltd.",
                   " co", " co.", " plc", " sa", " ag", " se",
                   " class a", " class b", " class c",
                   " cl a", " cl b", " cl c"]:
        name = name.replace(suffix, "")
    # Remove punctuation
    name = re.sub(r"[^a-z0-9\s]", "", name)
    return name.strip()
