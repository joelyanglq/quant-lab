from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

from .base import DataFeed, Frequency, AdjustType


class CSVDataFeed(DataFeed):
    """
    CSV文件数据源
    
    适合本地测试和小规模回测
    
    文件格式要求：
    - 文件名: {symbol}.csv
    - 列名: date, open, high, low, close, volume
    """
    
    def __init__(self, data_dir: str):
        super().__init__('CSVDataFeed')
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        # 缓存已加载的数据
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        frequency: Frequency = Frequency.DAY,
        adjust: AdjustType = AdjustType.NONE
    ) -> pd.DataFrame:
        """从CSV文件读取数据"""
        
        # 检查缓存
        if symbol not in self._cache:
            csv_path = self.data_dir / f"{symbol}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            # 读取CSV
            df = pd.read_csv(csv_path, parse_dates=['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # 标准化列名
            df.columns = df.columns.str.lower()
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV missing required columns: {required_cols}")
            
            self._cache[symbol] = df
        
        # 过滤日期范围
        df = self._cache[symbol]
        mask = (df.index >= start) & (df.index <= end)
        result = df.loc[mask].copy()
        
        # 处理复权
        if adjust != AdjustType.NONE:
            result = self._adjust_price(result, adjust)
        
        return result
    
    def get_symbols(self, market: Optional[str] = None) -> List[str]:
        """获取目录下所有CSV文件对应的股票代码"""
        csv_files = self.data_dir.glob("*.csv")
        symbols = [f.stem for f in csv_files]
        return sorted(symbols)
    
    def _adjust_price(self, df: pd.DataFrame, adjust: AdjustType) -> pd.DataFrame:
        """
        复权处理（简化版）
        
        实际应用中需要获取分红送股数据
        """
        # TODO: 实现真正的复权逻辑
        return df