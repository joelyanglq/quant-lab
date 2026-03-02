"""
数据加载层

- 价格面板: ParquetStorage → pivot 宽表
- Point-in-Time 财报: filing_date 对齐, 避免前视偏差
- TTM 计算: flow 项滚动4季求和, stock 项取最新值
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── 路径常量 ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _ROOT / 'data'
BARS_1D_DIR = DATA_DIR / 'processed' / 'bars_1d'
FUNDAMENTALS_DIR = DATA_DIR / 'fundamentals' / 'massive'
INDEX_SYMBOLS = DATA_DIR / '_index_symbols.json'


def load_index_symbols() -> List[str]:
    """从缓存加载 S&P500 + NASDAQ100 成分股"""
    data = json.loads(INDEX_SYMBOLS.read_text(encoding='utf-8'))
    return data['symbols']


# ── 价格面板 ────────────────────────────────────────────────────

def load_price_panel(
    symbols: List[str],
    start: str = '2018-01-01',
    end: str = '2026-12-31',
) -> Dict[str, pd.DataFrame]:
    """
    加载日线价格并 pivot 成宽表。

    Returns:
        {'close': df, 'high': df, 'low': df, 'volume': df}
        每个 df: index=DatetimeIndex, columns=symbols
    """
    from data.storage.parquet import ParquetStorage

    storage = ParquetStorage(str(BARS_1D_DIR))
    raw = storage.load(symbols, pd.Timestamp(start), pd.Timestamp(end), '1d')

    panels = {}
    for field in ['open', 'close', 'high', 'low', 'volume']:
        wide = raw.pivot_table(index=raw.index, columns='symbol', values=field)
        wide.index.name = 'date'
        panels[field] = wide

    return panels


# ── 财报加载 (Point-in-Time) ────────────────────────────────────

# flow 项: 利润表 + 现金流量表 (TTM = 滚动4季求和)
FLOW_SECTIONS = ('income_statement', 'cash_flow_statement')
# stock 项: 资产负债表 (取最新值)
STOCK_SECTIONS = ('balance_sheet',)
# 虽然在 flow section 但不应求和的字段 (取最新值)
NON_FLOW_OVERRIDES = {
    'income_statement__diluted_average_shares',
    'income_statement__basic_average_shares',
    'income_statement__diluted_earnings_per_share',
    'income_statement__basic_earnings_per_share',
}


def _load_single_fundamental(symbol: str) -> Optional[pd.DataFrame]:
    """加载一个 symbol 的财报 parquet"""
    fp = FUNDAMENTALS_DIR / f'{symbol}.parquet'
    if not fp.exists():
        return None
    return pd.read_parquet(fp)


def _fix_filing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    修复 Q4 的 filing_date 缺失:
    - Q4 行 filing_date 为 None → 取同 end_date 的 FY 行的 filing_date
    - 仍缺失 → end_date + 90天
    """
    df = df.copy()

    # 建立 FY 行的 end_date → filing_date 映射
    fy_mask = df['fiscal_period'] == 'FY'
    fy_map = df.loc[fy_mask].dropna(subset=['filing_date']).set_index('end_date')['filing_date'].to_dict()

    # 修复 Q4 缺失
    missing = df['filing_date'].isna()
    for idx in df.index[missing]:
        ed = df.loc[idx, 'end_date']
        if ed in fy_map:
            df.loc[idx, 'filing_date'] = fy_map[ed]

    # 仍缺失 → end_date + 90天
    still_missing = df['filing_date'].isna()
    if still_missing.any():
        df.loc[still_missing, 'filing_date'] = (
            pd.to_datetime(df.loc[still_missing, 'end_date']) + pd.Timedelta(days=90)
        ).astype(str)

    return df


def _compute_ttm(quarterly_df: pd.DataFrame, col: str, is_flow: bool) -> pd.Series:
    """
    计算 TTM:
    - flow 项: 最近4个季度滚动求和
    - stock 项: 直接取最新值
    """
    if not is_flow:
        return quarterly_df[col]
    return quarterly_df[col].rolling(4, min_periods=4).sum()


def build_fundamental_panel(
    symbols: List[str],
    fields: List[str],
    trading_dates: pd.DatetimeIndex,
) -> Dict[str, pd.DataFrame]:
    """
    构建 Point-in-Time 基本面面板。

    Args:
        symbols: ticker 列表
        fields: 需要的字段列表, e.g. ['income_statement__revenues', ...]
        trading_dates: 交易日序列 (用于 reindex)

    Returns:
        {field_name: DataFrame(index=trading_dates, columns=symbols)}
        flow 项已做 TTM (滚动4季求和)
    """
    panels = {f: pd.DataFrame(index=trading_dates, columns=symbols, dtype=float)
              for f in fields}

    for sym in symbols:
        raw = _load_single_fundamental(sym)
        if raw is None:
            continue

        # 只用季度数据做 TTM
        qtr = raw[raw['timeframe'] == 'quarterly'].copy()
        if qtr.empty:
            continue

        # 修复 filing_date
        qtr = _fix_filing_dates(qtr)
        qtr['filing_date'] = pd.to_datetime(qtr['filing_date'])
        qtr = qtr.sort_values('filing_date')

        # 去重: 同一个 (fiscal_year, fiscal_period) 保留最新的 filing
        qtr = qtr.drop_duplicates(subset=['fiscal_year', 'fiscal_period'], keep='last')

        for field in fields:
            if field not in qtr.columns:
                continue

            section = field.split('__')[0]
            is_flow = section in FLOW_SECTIONS and field not in NON_FLOW_OVERRIDES

            # NON_FLOW_OVERRIDES: Q4 行的值是 FY-(Q1+Q2+Q3) 残差，不可用
            if field in NON_FLOW_OVERRIDES:
                work = qtr[qtr['fiscal_period'] != 'Q4'].copy()
            else:
                work = qtr

            if work.empty:
                continue

            # 计算 TTM
            vals = _compute_ttm(work, field, is_flow)

            # 按 filing_date 创建时间序列 (PIT)
            ts = pd.Series(vals.values, index=work['filing_date'].values, dtype=float)
            ts = ts[~ts.index.duplicated(keep='last')]
            ts = ts.sort_index()

            # reindex 到交易日 + ffill
            ts = ts.reindex(trading_dates, method='ffill')

            panels[field][sym] = ts

    return panels


def build_shares_panel(
    symbols: List[str],
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    构建稀释后流通股数面板 (用于计算市值)。

    Returns:
        DataFrame(index=trading_dates, columns=symbols)
    """
    field = 'income_statement__diluted_average_shares'
    result = build_fundamental_panel(symbols, [field], trading_dates)
    return result[field]


def build_quarterly_series(
    symbol: str,
    field: str,
    n_quarters: int = 8,
) -> Optional[pd.Series]:
    """
    获取某 symbol 最近 N 个季度的原始值 (不做 TTM)。
    用于 EPS Score / Growth Stability 等需要逐季数据的因子。

    Returns:
        Series(index=end_date, values=field values), 按时间排序
    """
    raw = _load_single_fundamental(symbol)
    if raw is None:
        return None

    qtr = raw[raw['timeframe'] == 'quarterly'].copy()
    if field not in qtr.columns:
        return None

    qtr['end_date'] = pd.to_datetime(qtr['end_date'])
    qtr = qtr.sort_values('end_date')
    qtr = qtr.drop_duplicates(subset=['fiscal_year', 'fiscal_period'], keep='last')
    qtr = qtr.dropna(subset=[field])

    if len(qtr) < n_quarters:
        return None

    recent = qtr.tail(n_quarters)
    return pd.Series(recent[field].values, index=recent['end_date'].values)
