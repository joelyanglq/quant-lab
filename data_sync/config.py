"""
ETL 配置加载

从 config.yaml 加载配置，提供 typed 访问。
存储路径相对于 config.yaml 所在目录解析。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class ProviderConfig:
    source: str
    # prices_1d
    bulk: bool = False
    # symbol selection: index | sp500 | nasdaq | ndx100 | all | rus1000 | unions via comma/plus
    symbols: str = "index"
    # financials
    statements: List[str] = field(default_factory=lambda: ["income", "balance_sheet", "cash_flow"])
    period: List[str] = field(default_factory=lambda: ["annual", "quarter"])
    # update interval (days)
    update_interval_days: int = 1


@dataclass
class ETLConfig:
    api_keys: Dict[str, str]
    providers: Dict[str, ProviderConfig]
    storage_root: Path
    storage_paths: Dict[str, Path]
    rate_limits: Dict[str, int]

    def get_api_key(self, source: str) -> str:
        key = self.api_keys.get(source)
        if not key:
            raise ValueError(f"No API key configured for source: {source}")
        return key

    def get_storage_path(self, data_type: str) -> Path:
        path = self.storage_paths.get(data_type)
        if not path:
            raise ValueError(f"No storage path configured for: {data_type}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_rate_limit(self, source: str) -> int:
        return self.rate_limits.get(source, 60)


def load_config(config_path: Optional[str] = None) -> ETLConfig:
    """
    加载 ETL 配置。

    Args:
        config_path: YAML 配置文件路径。None 时使用默认路径 (data_sync/config.yaml)。

    Returns:
        ETLConfig 实例
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 解析存储路径 (相对于 config.yaml)
    config_dir = config_path.parent
    storage_raw = raw.get("storage", {})
    root_rel = storage_raw.pop("root", "../data")
    storage_root = (config_dir / root_rel).resolve()

    storage_paths = {}
    for key, rel_path in storage_raw.items():
        storage_paths[key] = storage_root / rel_path

    # 解析 providers
    providers = {}
    for name, cfg in raw.get("providers", {}).items():
        providers[name] = ProviderConfig(
            source=cfg.get("source", "fmp"),
            bulk=cfg.get("bulk", False),
            symbols=cfg.get("symbols", "index"),
            statements=cfg.get("statements", ["income", "balance_sheet", "cash_flow"]),
            period=cfg.get("period", ["annual", "quarter"]),
            update_interval_days=cfg.get("update_interval_days", 1),
        )

    return ETLConfig(
        api_keys=raw.get("api_keys", {}),
        providers=providers,
        storage_root=storage_root,
        storage_paths=storage_paths,
        rate_limits=raw.get("rate_limits", {}),
    )
