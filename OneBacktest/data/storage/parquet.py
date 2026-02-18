"""
Parquet 存储后端（按年存储）

每个 parquet 文件存储一年内全部 symbol 的数据。
文件名格式: {year}_{frequency}.parquet（如 2024_1d.parquet）

使用方式:
    storage = ParquetStorage('./processed/bars_1d')
    storage.save(df, '1d')            # df 含 symbol 列, index=DatetimeIndex
    df = storage.load(['AAPL', 'MSFT'], start, end, '1d')
"""
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


class ParquetStorage:
    """
    按年分文件的 Parquet 存储

    存储格式:
        - 文件: {year}_{frequency}.parquet
        - 列: symbol, open, high, low, close, volume (及其他可选列)
        - index: DatetimeIndex (timestamp)
        - 读取时用 pyarrow filters 按 symbol 过滤
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _year_file(self, year: int, frequency: str) -> Path:
        return self.data_dir / f"{year}_{frequency}.parquet"

    def _list_year_files(self, frequency: str) -> List[Path]:
        """列出所有年份文件，按年份排序"""
        pattern = f"*_{frequency}.parquet"
        files = sorted(self.data_dir.glob(pattern))
        return files

    def _parse_year(self, path: Path) -> int:
        """从文件名提取年份"""
        return int(path.stem.split('_')[0])

    # ── 写入 ──────────────────────────────────────────────────────

    def save(self, df: pd.DataFrame, frequency: str = '1d'):
        """
        保存 DataFrame，自动按年分文件。

        Args:
            df: 必须含 'symbol' 列，index 为 DatetimeIndex
            frequency: 数据频率标识
        """
        if 'symbol' not in df.columns:
            raise ValueError("DataFrame must contain 'symbol' column")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        # 按年分组保存
        for year, group in df.groupby(df.index.year):
            file_path = self._year_file(year, frequency)

            if file_path.exists():
                # 增量合并: 读旧数据 → concat → 去重
                old = pd.read_parquet(file_path)
                merged = pd.concat([old, group])
                # 按 (timestamp, symbol) 去重，保留最新
                merged = merged.reset_index()
                ts_col = merged.columns[0]  # DatetimeIndex 变成的列
                merged = merged.drop_duplicates(
                    subset=[ts_col, 'symbol'], keep='last'
                )
                merged = merged.set_index(ts_col).sort_index()
                merged.to_parquet(file_path, compression='snappy')
            else:
                group.sort_index().to_parquet(file_path, compression='snappy')

    # ── 读取 ──────────────────────────────────────────────────────

    def load(
        self,
        symbols: Union[str, List[str]],
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        frequency: str = '1d',
    ) -> pd.DataFrame:
        """
        加载指定 symbol 和日期范围的数据。

        Args:
            symbols: 单个或多个 ticker
            start: 起始时间（含）
            end: 结束时间（含）
            frequency: 数据频率

        Returns:
            DataFrame, index=DatetimeIndex, columns=[symbol, open, high, low, close, volume, ...]
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # 确定需要读哪些年份文件
        year_files = self._list_year_files(frequency)
        if not year_files:
            raise FileNotFoundError(
                f"No {frequency} data files in {self.data_dir}"
            )

        # 按日期范围过滤年份
        if start is not None:
            year_files = [f for f in year_files
                          if self._parse_year(f) >= start.year]
        if end is not None:
            year_files = [f for f in year_files
                          if self._parse_year(f) <= end.year]

        if not year_files:
            raise FileNotFoundError(
                f"No data files for date range [{start}, {end}]"
            )

        # 用 pyarrow filters 过滤 symbol（谓词下推，避免全量读取）
        filters = [('symbol', 'in', symbols)]

        dfs = []
        for fp in year_files:
            df = pd.read_parquet(fp, filters=filters)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            raise FileNotFoundError(
                f"No data for symbols {symbols} in [{start}, {end}]"
            )

        result = pd.concat(dfs)

        # 日期范围过滤（年份过滤是粗粒度的，这里精确裁剪）
        if start is not None:
            result = result[result.index >= start]
        if end is not None:
            result = result[result.index <= end]

        return result.sort_index()

    # ── 查询 ──────────────────────────────────────────────────────

    def list_symbols(self, frequency: str = '1d') -> List[str]:
        """返回所有可用 symbol（扫描最近一个年份文件）"""
        year_files = self._list_year_files(frequency)
        if not year_files:
            return []
        # 读最近一年的 symbol 列
        df = pd.read_parquet(year_files[-1], columns=['symbol'])
        symbols = [s for s in df['symbol'].unique() if s is not None]
        return sorted(symbols)

    def exists(self, frequency: str = '1d') -> bool:
        """检查是否有该频率的数据文件"""
        return len(self._list_year_files(frequency)) > 0

    def latest_date(self, frequency: str = '1d') -> Optional[pd.Timestamp]:
        """返回已有数据的最新日期"""
        year_files = self._list_year_files(frequency)
        if not year_files:
            return None
        df = pd.read_parquet(year_files[-1], columns=[])
        if len(df) == 0:
            return None
        return pd.Timestamp(df.index.max())
