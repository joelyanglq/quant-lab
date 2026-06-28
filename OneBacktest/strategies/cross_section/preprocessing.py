"""
统一预处理 pipeline

两种标准化 + OLS 行业/市值中性化，在因子计算后、评估前统一执行。

Usage:
    from strategies.cross_section.preprocessing import preprocess_factor
    factor_clean = preprocess_factor(factor, method="mad_zscore", neutralize=True,
                                     mktcap=mktcap, sector_map=sector_map)
"""
import numpy as np
import pandas as pd
from scipy import stats

from .ranking import cross_sectional_zscore


def _rank_zscore(factor: pd.DataFrame) -> pd.DataFrame:
    """
    Rank + z-score 标准化。

    每行(日期)独立: rank → 百分位 → 正态逆变换 (保留正态分布)。
    """
    def _row_rank_zscore(row):
        valid = row.dropna()
        if len(valid) < 5:
            return row * np.nan
        ranked = valid.rank() / (len(valid) + 1)  # 开区间 (0, 1)
        zscore = pd.Series(stats.norm.ppf(ranked), index=valid.index)
        return zscore.reindex(row.index)

    return factor.apply(_row_rank_zscore, axis=1)


def _ols_neutralize(
    factor: pd.DataFrame,
    mktcap: pd.DataFrame,
    sector_map: pd.Series,
) -> pd.DataFrame:
    """
    OLS 中性化: factor ~ log(mktcap) + Σ Industry_dummies, 取残差。

    比减行业均值更标准 — 同时去除市值和行业效应。
    """
    result = factor.copy()
    log_mktcap = np.log(mktcap.replace(0, np.nan))

    for dt in factor.index:
        f = factor.loc[dt].dropna()
        if len(f) < 20:
            continue

        # 对齐 mktcap 和 sector
        if dt not in log_mktcap.index:
            continue
        lmc = log_mktcap.loc[dt].reindex(f.index).dropna()
        sec = sector_map.reindex(f.index).dropna()
        common = f.index.intersection(lmc.index).intersection(sec.index)
        if len(common) < 20:
            continue

        f_aligned = f[common]
        lmc_aligned = lmc[common]
        sec_aligned = sec[common]

        # 构建 X: [log_mktcap, industry_dummy_1, ..., industry_dummy_k-1]
        dummies = pd.get_dummies(sec_aligned, drop_first=True, dtype=float)
        X = pd.concat([lmc_aligned.rename("log_mktcap"), dummies], axis=1)
        X = X.values
        y = f_aligned.values

        # OLS: y = Xβ + ε, 残差 = y - Xβ
        try:
            X_with_const = np.column_stack([np.ones(len(X)), X])
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            residuals = y - X_with_const @ beta
            result.loc[dt, common] = residuals
        except np.linalg.LinAlgError:
            continue

    return result


def preprocess_factor(
    factor: pd.DataFrame,
    method: str = "mad_zscore",
    neutralize: bool = True,
    mktcap: pd.DataFrame = None,
    sector_map: pd.Series = None,
    winsorize_sigma: float = 3.0,
) -> pd.DataFrame:
    """
    统一因子预处理 pipeline: standardize → neutralize。

    Args:
        factor: 因子面板 (index=dates, columns=symbols)
        method: "mad_zscore" | "rank_zscore"
        neutralize: 是否做行业/市值中性化
        mktcap: 市值面板 (index=dates, columns=symbols)，neutralize=True 时必需
        sector_map: Series(index=symbol, values=sector)，neutralize=True 时必需
        winsorize_sigma: MAD winsorize 的 sigma 倍数 (仅 mad_zscore)

    Returns:
        预处理后的因子面板 (same shape)
    """
    # Step 1: 标准化
    if method == "mad_zscore":
        standardized = cross_sectional_zscore(factor, winsorize_sigma=winsorize_sigma)
    elif method == "rank_zscore":
        standardized = _rank_zscore(factor)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mad_zscore' or 'rank_zscore'.")

    # Step 2: 中性化
    if neutralize:
        if mktcap is None or sector_map is None:
            raise ValueError("neutralize=True requires both mktcap and sector_map")
        return _ols_neutralize(standardized, mktcap, sector_map)

    return standardized
