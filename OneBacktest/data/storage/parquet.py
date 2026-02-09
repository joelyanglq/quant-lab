"""
Parquet 存储后端

用于缓存下载的数据，避免重复请求 API
"""
from pathlib import Path
from typing import Optional

import pandas as pd


class ParquetStorage:
    """
    Parquet 文件存储

    使用方式：
        storage = ParquetStorage('./data_cache')
        storage.save('AAPL', df, '1d')
        df = storage.load('AAPL', start, end, '1d')
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save(self, symbol: str, df: pd.DataFrame, frequency: str = '1d'):
        """保存 DataFrame 为 Parquet 文件"""
        file_path = self.data_dir / f"{symbol}_{frequency}.parquet"
        df.to_parquet(file_path, compression='snappy')

    def load(
        self,
        symbol: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        frequency: str = '1d',
    ) -> pd.DataFrame:
        """加载 Parquet 文件"""
        file_path = self.data_dir / f"{symbol}_{frequency}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"No cached data for {symbol} at {file_path}")

        df = pd.read_parquet(file_path)

        # 过滤日期范围
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]

        return df

    def exists(self, symbol: str, frequency: str = '1d') -> bool:
        """检查缓存是否存在"""
        file_path = self.data_dir / f"{symbol}_{frequency}.parquet"
        return file_path.exists()

    def delete(self, symbol: str, frequency: str = '1d'):
        """删除缓存"""
        file_path = self.data_dir / f"{symbol}_{frequency}.parquet"
        if file_path.exists():
            file_path.unlink()
