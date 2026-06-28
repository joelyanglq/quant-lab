"""
Parquet 存储后端（支持按年 / 按月分片）

文件名格式:
    按年: {year}_{frequency}.parquet    — 如 2024_1d.parquet
    按月: {YYYY-MM}_{frequency}.parquet — 如 2024-01_1m.parquet

两种格式可以共存于同一目录（如 1min 数据迁移期间）。

使用方式:
    storage = ParquetStorage('./processed/bars_1d')
    storage.save(df, '1d')                        # 按年分片（默认）
    storage.save(df, '1m', partition='monthly')    # 按月分片
    df = storage.load(['AAPL', 'MSFT'], start, end, '1d')
"""
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

_YEARLY_RE = re.compile(r'^(\d{4})_')
_MONTHLY_RE = re.compile(r'^(\d{4})-(\d{2})_')


class ParquetStorage:
    """
    按年/月分文件的 Parquet 存储

    存储格式:
        - 文件: {year}_{freq}.parquet 或 {YYYY-MM}_{freq}.parquet
        - 列: symbol, open, high, low, close, volume (及其他可选列)
        - index: DatetimeIndex (timestamp)
        - 读取时用 pyarrow filters 按 symbol 过滤
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _year_file(self, year: int, frequency: str) -> Path:
        return self.data_dir / f"{year}_{frequency}.parquet"

    def _month_file(self, year: int, month: int, frequency: str) -> Path:
        return self.data_dir / f"{year:04d}-{month:02d}_{frequency}.parquet"

    def _list_data_files(self, frequency: str) -> List[Path]:
        """列出所有数据文件（年份+月份），按文件名排序"""
        pattern = f"*_{frequency}.parquet"
        files = sorted(self.data_dir.glob(pattern))
        return files

    def _list_year_files(self, frequency: str) -> List[Path]:
        """兼容旧接口"""
        return self._list_data_files(frequency)

    def _parse_year(self, path: Path) -> int:
        """从文件名提取年份（兼容 YYYY 和 YYYY-MM 格式）"""
        prefix = path.stem.split('_')[0]
        return int(prefix[:4])

    def _parse_period(self, path: Path) -> Tuple[int, Optional[int]]:
        """返回 (year, month_or_None)"""
        stem = path.stem
        m = _MONTHLY_RE.match(stem)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = _YEARLY_RE.match(stem)
        if m:
            return int(m.group(1)), None
        raise ValueError(f"Cannot parse period from {path.name}")

    # ── 写入 ──────────────────────────────────────────────────────

    def save(self, df: pd.DataFrame, frequency: str = '1d', partition: str = 'yearly'):
        """
        保存 DataFrame，自动按年或月分文件。

        新数据日期在已有数据之后时使用 pyarrow 流式追加（避免读入全量旧数据）；
        有日期重叠时回退到 read → concat → dedup 路径。

        Args:
            df: 必须含 'symbol' 列，index 为 DatetimeIndex
            frequency: 数据频率标识
            partition: 'yearly' (default) 或 'monthly'
        """
        import pyarrow.parquet as pq

        if 'symbol' not in df.columns:
            raise ValueError("DataFrame must contain 'symbol' column")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")

        if partition == 'monthly':
            for (year, month), group in df.groupby([df.index.year, df.index.month]):
                file_path = self._month_file(year, month, frequency)
                self._save_to_file(group, file_path)
        else:
            for year, group in df.groupby(df.index.year):
                file_path = self._year_file(year, frequency)
                self._save_to_file(group, file_path)

    def _save_to_file(self, group: pd.DataFrame, file_path: Path):
        """Save a group of data to a single parquet file (append or create)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        group = group.sort_index()

        if not file_path.exists():
            group.to_parquet(file_path, compression='snappy')
            return

        # 快速路径: 新数据的最小日期 > 旧文件最大日期 → 直接追加
        existing_pf = pq.ParquetFile(file_path)
        old_schema = existing_pf.schema_arrow

        old_idx = pd.read_parquet(file_path, columns=[])
        old_max = old_idx.index.max()
        del old_idx

        if group.index.min() > old_max:
            all_cols = [f.name for f in old_schema]
            data_cols = [c for c in all_cols if c in group.columns]
            idx_name = [c for c in all_cols if c not in set(group.columns)]
            if idx_name:
                group.index.name = idx_name[0]
            group = group[data_cols]

            new_table = pa.Table.from_pandas(group)
            try:
                new_table = new_table.cast(old_schema)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                pass

            tmp_out = file_path.with_suffix('.tmp')
            writer = pq.ParquetWriter(tmp_out, old_schema, compression='snappy')
            for i in range(existing_pf.metadata.num_row_groups):
                writer.write_table(existing_pf.read_row_group(i))
            writer.write_table(new_table)
            writer.close()
            del existing_pf, new_table
            file_path.unlink()
            tmp_out.rename(file_path)
        else:
            del existing_pf
            old = pd.read_parquet(file_path)
            merged = pd.concat([old, group])
            del old
            merged = merged.reset_index()
            ts_col = merged.columns[0]
            merged = merged.drop_duplicates(subset=[ts_col, 'symbol'], keep='last')
            merged = merged.set_index(ts_col).sort_index()
            merged.to_parquet(file_path, compression='snappy')

    # ── 读取 ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize parquet data to DatetimeIndex + symbol column.

        Handles two schemas:
          - FMP format: DatetimeIndex (pandas metadata), symbol column
          - Merged format: flat 'date' column, symbol column, RangeIndex
        Also strips timezone to ensure all data is tz-naive for consistent comparison.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
                df.index = pd.DatetimeIndex(df.index)
                df.index.name = None
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df

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

        # 确定需要读哪些数据文件
        data_files = self._list_data_files(frequency)
        if not data_files:
            raise FileNotFoundError(
                f"No {frequency} data files in {self.data_dir}"
            )

        # 按日期范围过滤（年级别粗筛）
        if start is not None:
            data_files = [f for f in data_files
                          if self._parse_year(f) >= start.year]
        if end is not None:
            data_files = [f for f in data_files
                          if self._parse_year(f) <= end.year]

        if not data_files:
            raise FileNotFoundError(
                f"No data files for date range [{start}, {end}]"
            )

        # 用 pyarrow filters 过滤 symbol（谓词下推，避免全量读取）
        filters = [('symbol', 'in', symbols)]

        dfs = []
        for fp in data_files:
            df = pd.read_parquet(fp, filters=filters)
            if not df.empty:
                dfs.append(self._normalize_df(df))

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

        return result

    # ── 查询 ──────────────────────────────────────────────────────

    def list_symbols(self, frequency: str = '1d') -> List[str]:
        """返回所有可用 symbol（扫描最近一个文件）"""
        data_files = self._list_data_files(frequency)
        if not data_files:
            return []
        df = pd.read_parquet(data_files[-1], columns=['symbol'])
        symbols = [s for s in df['symbol'].unique() if s is not None]
        return sorted(symbols)

    def exists(self, frequency: str = '1d') -> bool:
        """检查是否有该频率的数据文件"""
        return len(self._list_data_files(frequency)) > 0

    def latest_date(self, frequency: str = '1d') -> Optional[pd.Timestamp]:
        """返回已有数据的最新日期（tz-naive）"""
        data_files = self._list_data_files(frequency)
        if not data_files:
            return None

        latest = None
        for fp in data_files:
            df = pd.read_parquet(fp, columns=[])
            if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
                ts = pd.Timestamp(df.index.max())
                ts = ts.tz_localize(None) if ts.tzinfo else ts
            else:
                df = pd.read_parquet(fp, columns=['date'])
                if len(df) == 0:
                    continue
                ts = pd.Timestamp(df['date'].max())

            if latest is None or ts > latest:
                latest = ts

        return latest
