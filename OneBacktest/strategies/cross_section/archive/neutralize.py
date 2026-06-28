"""
行业中性化 — 共享模块

提供 GICS 行业分类加载 + 因子行业中性化（减行业均值）。
供 pick_stocks.py (实盘选股) 和 cross_section_alpha.py (回测) 共用。

需要中性化的因子:
  - 基本面因子: ROE, ROIC, EV_EBITDA, PS 等 — 估值/盈利水平有天然行业差异
  - 量价偏差因子: 使用绝对 volume/turnover 或价格幂次的因子
"""
from pathlib import Path
from typing import Optional, Set

import pandas as pd

_GICS_PATH = (Path(__file__).resolve().parent.parent.parent.parent
              / 'data' / '_gics_sectors.parquet')

# 使用绝对 volume/turnover 或价格幂次, 具有行业系统性偏差的价格因子
# (基本面因子由 factor_registry input_data=='fundamental' 标记, 不在此集合)
VOLUME_BIASED_FACTORS = frozenset({
    'alpha_054',       # (low-close)*open^5 / ((low-high)*close^5) — 价格幂次
    'alpha_060',       # rank(position * volume) — 绝对 volume
    'vol_battle_pos',  # cumsum_battle(volume, position) — 绝对 volume
    'fuzzy_corr',      # corr(ambiguity, turnover) — 绝对 turnover
    'lone_goose',      # 截面 turnover 相关性 — 绝对 turnover
    'moderate_risk',   # volume surge 检测 — 绝对 volume 阈值
    'peak_climbing',   # cov(ret/better_vol, better_vol) — OHLC 绝对值
})


def load_gics_sector_map(path: Path = _GICS_PATH) -> Optional[pd.Series]:
    """
    加载 GICS 行业分类映射。

    Returns:
        Series(index=symbol, values=gics_sector), 或 None
    """
    try:
        if not path.exists():
            return None
        gics = pd.read_parquet(path)
        col_map = {}
        for c in gics.columns:
            cl = c.lower().replace(' ', '_')
            if 'symbol' in cl or 'ticker' in cl:
                col_map[c] = 'symbol'
            elif 'sector' in cl:
                col_map[c] = 'gics_sector'
        if col_map:
            gics = gics.rename(columns=col_map)
        if 'symbol' in gics.columns and 'gics_sector' in gics.columns:
            return gics.set_index('symbol')['gics_sector']
    except Exception:
        pass
    return None


def neutralize_factors(
    df: pd.DataFrame,
    sector_map: pd.Series,
    cols_to_neutralize: Set[str],
) -> pd.DataFrame:
    """
    对指定因子做行业中性化（减行业均值）。

    不在 cols_to_neutralize 中的列不变。每个行业至少 2 只股票才做中性化。

    Args:
        df:                  index=symbol, columns=因子名
        sector_map:          Series(index=symbol, values=sector)
        cols_to_neutralize:  需要中性化的列名集合

    Returns:
        中性化后的 DataFrame (copy)
    """
    df = df.copy()
    for col in df.columns:
        if col not in cols_to_neutralize:
            continue
        s = df[col]
        sectors = sector_map.reindex(s.dropna().index).dropna()
        if len(sectors) < 5:
            continue
        valid = s.index.intersection(sectors.index)
        for sector in sectors[valid].unique():
            sec_syms = sectors[valid][sectors[valid] == sector].index
            if len(sec_syms) >= 2:
                df.loc[sec_syms, col] = s[sec_syms] - s[sec_syms].mean()
    return df
