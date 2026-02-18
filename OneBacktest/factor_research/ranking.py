"""
横截面排名

- MAD winsorize (3σ)
- z-score 标准化
- 五分位分组 (1=最差, 5=最好)
"""
import numpy as np
import pandas as pd


def _mad_winsorize(row: pd.Series, n_sigma: float = 3.0) -> pd.Series:
    """MAD-based winsorization for a single cross-section"""
    median = row.median()
    mad = (row - median).abs().median()
    if mad == 0:
        return row
    cutoff = n_sigma * 1.4826 * mad  # 1.4826 converts MAD to std equiv
    return row.clip(median - cutoff, median + cutoff)


def cross_sectional_zscore(
    factor: pd.DataFrame,
    winsorize_sigma: float = 3.0,
) -> pd.DataFrame:
    """
    横截面 z-score: 每一行 (日期) 独立标准化。

    1. MAD winsorize
    2. (x - mean) / std
    """
    # Winsorize
    winsorized = factor.apply(_mad_winsorize, axis=1, n_sigma=winsorize_sigma)

    # z-score per row
    row_mean = winsorized.mean(axis=1)
    row_std = winsorized.std(axis=1)
    zscore = winsorized.sub(row_mean, axis=0).div(row_std.replace(0, np.nan), axis=0)

    return zscore


def assign_quantiles(
    factor: pd.DataFrame,
    n_quantiles: int = 5,
    min_stocks: int = 10,
) -> pd.DataFrame:
    """
    横截面五分位分组: 1=最差, n_quantiles=最好。

    每行 (日期) 独立分组。不足 min_stocks 只的日期全部设 NaN。
    """
    def _qcut_row(row):
        valid = row.dropna()
        if len(valid) < min_stocks:
            return pd.Series(np.nan, index=row.index)
        try:
            labels = pd.qcut(valid.rank(method='first'), n_quantiles, labels=False) + 1
            return labels.reindex(row.index)
        except ValueError:
            return pd.Series(np.nan, index=row.index)

    return factor.apply(_qcut_row, axis=1)
