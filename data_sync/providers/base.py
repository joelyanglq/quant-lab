"""
Provider 基类

所有数据类型的 Provider 继承此 ABC，统一 sync / status 接口。
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional

from data_sync.config import ETLConfig


class Provider(ABC):
    """数据 Provider 抽象基类"""

    name: str = ""

    def __init__(self, config: ETLConfig):
        self.config = config

    def sync(
        self,
        from_date: Optional[date] = None,
        to_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
        force: bool = False,
    ):
        """
        同步数据（回填 + 增量更新共用; 无 StatusManager 版本）。

        CLI 优先调用 sync_with_status()（含并发+状态追踪），
        如未实现则回退到此方法。遗留 provider 可直接覆写此方法，
        新 provider 建议实现 sync_with_status()。

        Args:
            from_date: 起始日期。None → 自动从 latest_date + 1 开始。
            to_date: 结束日期。None → today。
            symbols: 指定 symbol 列表。None → 使用默认 universe。
            force: True 时忽略 stale 检查，强制重拉。
        """
        raise NotImplementedError(
            f"{self.name} must implement sync() or sync_with_status()"
        )

    @abstractmethod
    def status(self) -> dict:
        """
        返回当前数据状态。

        Returns:
            dict with keys like: latest_date, symbol_count, file_size, etc.
        """
        ...
