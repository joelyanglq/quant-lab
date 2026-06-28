"""
Provider 注册表

根据 config.yaml 的 providers.<type>.source 映射到具体 Provider 类。
"""
from data_sync.config import ETLConfig
from data_sync.providers.base import Provider

# (data_type, source) → Provider class
# 在各 provider 模块底部调用 register() 注册
_REGISTRY: dict = {}


def register(data_type: str, source: str, provider_cls: type):
    """注册一个 Provider 实现。"""
    _REGISTRY[(data_type, source)] = provider_cls


def get_provider(data_type: str, config: ETLConfig) -> Provider:
    """
    根据 config 获取对应的 Provider 实例。

    Args:
        data_type: 数据类型 (e.g. "prices_1d", "financials")
        config: ETL 配置

    Returns:
        Provider 实例
    """
    provider_cfg = config.providers.get(data_type)
    if not provider_cfg:
        raise ValueError(
            f"No provider configured for '{data_type}'. "
            f"Available: {list(config.providers.keys())}"
        )

    source = provider_cfg.source
    key = (data_type, source)

    if key not in _REGISTRY:
        _import_providers()
        if key not in _REGISTRY:
            available = [s for (dt, s) in _REGISTRY if dt == data_type]
            raise ValueError(
                f"No provider registered for ({data_type!r}, {source!r}). "
                f"Available sources for '{data_type}': {available}"
            )

    return _REGISTRY[key](config)


def _import_providers():
    """延迟 import 所有 provider 模块，触发注册。"""
    try:
        import data_sync.providers.prices  # noqa: F401
    except ImportError:
        pass
    try:
        import data_sync.providers.financials  # noqa: F401
    except ImportError:
        pass
    try:
        import data_sync.providers.analysts  # noqa: F401
    except ImportError:
        pass
    try:
        import data_sync.providers.universe  # noqa: F401
    except ImportError:
        pass
