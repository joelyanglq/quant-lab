"""
因子筛选与正交化

- IC 分析: 逐期 Spearman Rank IC + 汇总统计
- 筛选: |IC| 过滤 + 相关性去冗余
- 正交化: 逐截面 OLS 回归取残差
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════
# IC 分析
# ═══════════════════════════════════════════════════════════════

def compute_rank_ic(
    factor_w: pd.DataFrame,
    fwd_ret_w: pd.DataFrame,
    min_obs: int = 20,
) -> pd.Series:
    """
    逐期 Spearman Rank IC.

    Args:
        factor_w: 因子面板 (index=dates, columns=symbols)
        fwd_ret_w: 前瞻收益面板 (同结构)
        min_obs: 最小有效观测数

    Returns:
        Series(index=dates, values=ic)
    """
    common_idx = factor_w.index.intersection(fwd_ret_w.index)
    common_cols = factor_w.columns.intersection(fwd_ret_w.columns)

    ics = {}
    for dt in common_idx:
        fv = factor_w.loc[dt, common_cols].dropna()
        rv = fwd_ret_w.loc[dt].reindex(fv.index).dropna()
        common = fv.index.intersection(rv.index)
        if len(common) < min_obs:
            continue
        ic = fv[common].corr(rv[common], method='spearman')
        if not np.isnan(ic):
            ics[dt] = ic

    return pd.Series(ics)


def compute_ic_summary(
    factors_w: Dict[str, pd.DataFrame],
    fwd_ret_w: pd.DataFrame,
    min_periods: int = 3,
) -> pd.DataFrame:
    """
    批量计算所有因子的 IC 统计.

    Returns:
        DataFrame(index=factor_name,
                  columns=[mean_ic, ic_std, icir, n_periods])
        按 |ICIR| 降序排列.
    """
    ic_stats = {}
    for name, panel in factors_w.items():
        ics = compute_rank_ic(panel, fwd_ret_w)
        if len(ics) >= min_periods:
            ic_stats[name] = {
                'mean_ic': ics.mean(),
                'ic_std': ics.std(),
                'icir': ics.mean() / ics.std() if ics.std() > 0 else 0,
                'n_periods': len(ics),
            }

    if not ic_stats:
        return pd.DataFrame(columns=['mean_ic', 'ic_std', 'icir', 'n_periods'])

    return pd.DataFrame(ic_stats).T.sort_values('icir', ascending=False)


# ═══════════════════════════════════════════════════════════════
# 筛选
# ═══════════════════════════════════════════════════════════════

def ic_filter(
    ic_summary: pd.DataFrame,
    min_abs_ic: float = 0.005,
) -> List[str]:
    """
    IC 绝对值过滤.

    Returns:
        通过筛选的因子名列表 (保持 |ICIR| 降序).
    """
    mask = ic_summary['mean_ic'].abs() >= min_abs_ic
    return ic_summary[mask].index.tolist()


def correlation_dedup(
    factors_w: Dict[str, pd.DataFrame],
    ic_summary: pd.DataFrame,
    passed: List[str],
    max_corr: float = 0.7,
) -> List[str]:
    """
    相关性去冗余: 高相关对中保留 |ICIR| 更高者.

    Args:
        factors_w: 周频因子面板
        ic_summary: IC 统计 (需含 'icir' 列)
        passed: IC 过滤后的因子名列表
        max_corr: 最大允许相关系数

    Returns:
        去冗余后的因子名列表.
    """
    if len(passed) <= 1:
        return list(passed)

    # 构建 stacked 长表 → corr matrix
    common_idx = None
    common_cols = None
    for name in passed:
        idx = factors_w[name].index
        cols = factors_w[name].columns
        common_idx = idx if common_idx is None else common_idx.intersection(idx)
        common_cols = cols if common_cols is None else common_cols.intersection(cols)

    stacked = pd.DataFrame()
    for name in passed:
        vals = factors_w[name].loc[common_idx, common_cols].stack()
        stacked[name] = vals

    stacked = stacked.dropna()
    corr = stacked.corr()

    # 贪心去冗余
    removed = set()
    for i in range(len(passed)):
        if passed[i] in removed:
            continue
        for j in range(i + 1, len(passed)):
            if passed[j] in removed:
                continue
            a, b = passed[i], passed[j]
            if abs(corr.loc[a, b]) > max_corr:
                ic_a = abs(ic_summary.loc[a, 'icir'])
                ic_b = abs(ic_summary.loc[b, 'icir'])
                drop = b if ic_a >= ic_b else a
                removed.add(drop)

    return [n for n in passed if n not in removed]


def screen_factors(
    factors_w: Dict[str, pd.DataFrame],
    fwd_ret_w: pd.DataFrame,
    min_abs_ic: float = 0.005,
    max_corr: float = 0.7,
) -> Tuple[List[str], pd.DataFrame]:
    """
    因子筛选 (便捷包装):
    1. 计算 IC 汇总
    2. |IC| 过滤
    3. 相关性去冗余

    Returns:
        (selected_names, ic_summary_df)
    """
    ic_df = compute_ic_summary(factors_w, fwd_ret_w)

    if ic_df.empty:
        return [], ic_df

    # IC 过滤
    passed = ic_filter(ic_df, min_abs_ic)

    if len(passed) <= 1:
        return passed, ic_df

    # 相关性去冗余
    selected = correlation_dedup(factors_w, ic_df, passed, max_corr)

    return selected, ic_df


# ═══════════════════════════════════════════════════════════════
# 正交化
# ═══════════════════════════════════════════════════════════════

def orthogonalize(
    new_factor: pd.DataFrame,
    existing_factors: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    单因子正交化: 逐截面 OLS 回归取残差.

    对每个截面 (日期 t):
        new_factor[t] = β₀ + Σ βᵢ · existing_i[t] + ε[t]
    返回 ε (去除已有因子线性成分后的纯净因子).

    Args:
        new_factor: 待正交化因子 (dates × symbols)
        existing_factors: 已有因子 {name: dates × symbols}

    Returns:
        正交化后的因子面板 (dates × symbols)
    """
    if not existing_factors:
        return new_factor.copy()

    residuals = {}
    for dt in new_factor.index:
        y = new_factor.loc[dt].dropna()
        if len(y) == 0:
            continue

        # 收集该截面的已有因子
        X_parts = {}
        for name, panel in existing_factors.items():
            if dt in panel.index:
                X_parts[name] = panel.loc[dt]

        if not X_parts:
            residuals[dt] = y
            continue

        X = pd.DataFrame(X_parts).reindex(y.index).dropna()
        common = y.index.intersection(X.index)

        # 需要足够观测 (至少 > 自变量数 + 2)
        if len(common) < max(10, X.shape[1] + 2):
            residuals[dt] = y
            continue

        yc = y[common].values
        Xc = np.column_stack([X.loc[common].values, np.ones(len(common))])
        beta, _, _, _ = np.linalg.lstsq(Xc, yc, rcond=None)
        residuals[dt] = pd.Series(yc - Xc @ beta, index=common)

    return pd.DataFrame(residuals).T


def orthogonalize_sequential(
    factors: Dict[str, pd.DataFrame],
    order: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    顺序正交化 (类似 Gram-Schmidt):
    按 order 排序, 第 i 个因子对前 i-1 个因子正交化.

    典型用法: order 按 |ICIR| 降序, 高 ICIR 因子保持原样,
    低 ICIR 因子被正交化为纯增量信息.

    Args:
        factors: 因子面板 {name: dates × symbols}
        order: 因子优先级排序

    Returns:
        {name: 正交化后的面板}
    """
    result = {}
    for i, name in enumerate(order):
        if name not in factors:
            continue
        if i == 0:
            result[name] = factors[name]
        else:
            existing = {k: result[k] for k in order[:i] if k in result}
            result[name] = orthogonalize(factors[name], existing)
    return result
