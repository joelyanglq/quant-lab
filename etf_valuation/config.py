"""
Config loader for etf_valuation.

Loads etf_universe.yaml and provides typed access to ETF definitions,
metric assignments, storage paths, and scoring parameters.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

METRICS = {
    "pe_ttm":     {"fmp_field": "priceToEarningsRatioTTM",       "direction": "higher_expensive", "agg": "harmonic"},
    "pb_lf":      {"fmp_field": "priceToBookRatioTTM",           "direction": "higher_expensive", "agg": "harmonic"},
    "ps_ttm":     {"fmp_field": "priceToSalesRatioTTM",          "direction": "higher_expensive", "agg": "harmonic"},
    "div_yield":  {"fmp_field": "dividendYieldTTM",              "direction": "higher_cheap",     "agg": "arithmetic"},
    "erp":        {"fmp_field": None,                            "direction": "higher_cheap",     "agg": "arithmetic"},
    "fcf_yield":  {"fmp_field": "priceToFreeCashFlowRatioTTM",   "direction": "higher_cheap",     "agg": "arithmetic"},
    "ev_ebitda":  {"fmp_field": "enterpriseValueMultipleTTM",    "direction": "higher_expensive", "agg": "harmonic"},
}


@dataclass
class ETFDef:
    ticker: str
    name: str
    tier: str
    primary: Optional[str]
    secondary: Optional[str]


@dataclass
class ValuationConfig:
    storage_root: Path
    storage_paths: Dict[str, Path]
    etfs: Dict[str, ETFDef]
    tiers: Dict[str, str]  # tier_key → label
    window_years: int = 5
    min_observations: int = 60
    primary_weight: float = 0.7
    secondary_weight: float = 0.3

    def get_storage_path(self, key: str) -> Path:
        path = self.storage_paths[key]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def equity_etfs(self) -> List[ETFDef]:
        return [e for e in self.etfs.values() if e.primary is not None]

    def etfs_by_tier(self, tier: str) -> List[ETFDef]:
        return [e for e in self.etfs.values() if e.tier == tier]


def load_config(config_path: Optional[str] = None) -> ValuationConfig:
    if config_path is None:
        config_path = Path(__file__).parent / "etf_universe.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    storage_root = Path(raw["storage_root"])
    storage_paths = {k: storage_root / v for k, v in raw["storage"].items()}

    pct_cfg = raw.get("percentile", {})
    score_cfg = raw.get("scoring", {}).get("composite_weights", {})

    etfs: Dict[str, ETFDef] = {}
    tier_labels: Dict[str, str] = {}
    for tier_key, tier_data in raw.get("tiers", {}).items():
        tier_labels[tier_key] = tier_data.get("label", tier_key)
        for ticker, edef in tier_data.get("etfs", {}).items():
            etfs[ticker] = ETFDef(
                ticker=ticker,
                name=edef["name"],
                tier=tier_key,
                primary=edef.get("primary"),
                secondary=edef.get("secondary"),
            )

    return ValuationConfig(
        storage_root=storage_root,
        storage_paths=storage_paths,
        etfs=etfs,
        tiers=tier_labels,
        window_years=pct_cfg.get("window_years", 5),
        min_observations=pct_cfg.get("min_observations", 60),
        primary_weight=score_cfg.get("primary", 0.7),
        secondary_weight=score_cfg.get("secondary", 0.3),
    )
