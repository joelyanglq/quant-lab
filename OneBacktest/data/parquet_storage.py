import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime


class ParquetStorage:
    """
    Parquet格式存储
    
    优点：
    - 压缩率高（节省磁盘空间）
    - 读写速度快
    - 支持列式查询
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def save(self, symbol: str, df: pd.DataFrame, frequency: str = '1d'):
        """保存数据"""
        file_path = self.storage_path / f"{symbol}_{frequency}.parquet"
        
        # 确保datetime是索引
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 写入parquet
        df.to_parquet(file_path, compression='snappy')
        print(f"Saved {symbol} data to {file_path}")
    
    def load(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        frequency: str = '1d'
    ) -> pd.DataFrame:
        """加载数据"""
        file_path = self.storage_path / f"{symbol}_{frequency}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found: {file_path}")
        
        # 读取parquet
        df = pd.read_parquet(file_path)
        
        # 过滤日期
        if start or end:
            start = start or df.index.min()
            end = end or df.index.max()
            df = df.loc[start:end]
        
        return df
    
    def exists(self, symbol: str, frequency: str = '1d') -> bool:
        """检查数据是否存在"""
        file_path = self.storage_path / f"{symbol}_{frequency}.parquet"
        return file_path.exists()
    
    def get_date_range(self, symbol: str, frequency: str = '1d') -> tuple:
        """获取数据的日期范围"""
        df = self.load(symbol, frequency=frequency)
        return df.index.min(), df.index.max()