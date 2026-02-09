"""
data/ 模块

架构：
    DataFeed (抽象接口)
        └── HistoricFeed (从 Parquet 文件加载历史数据)

    ParquetStorage (存储后端)
        - 读写 Parquet 文件

目录结构：
    data/
    ├── __init__.py
    ├── types.py           # Bar, Frequency, AdjustType
    ├── feed.py            # DataFeed 抽象基类
    ├── sources/
    │   └── historic.py    # HistoricFeed 实现
    └── storage/
        └── parquet.py     # ParquetStorage
"""
from .types import Bar, Frequency, AdjustType
from .feed import DataFeed
from .sources.historic import HistoricFeed
from .storage.parquet import ParquetStorage

__all__ = [
    'Bar',
    'Frequency',
    'AdjustType',
    'DataFeed',
    'HistoricFeed',
    'ParquetStorage',
]
