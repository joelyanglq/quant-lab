"""
复合因子

加权组合多个子因子的 z-score:
- 盈利 25%: ROE 12.5%, ROIC 12.5%
- 估值 30%: EV/EBITDA 10%, FCF Yield 10%, P/S 10%
- 成长 20%: FCF Growth 5%, EPS Score 10%, Growth Stability 5%
- 动量 25%: RS12M 10%, RS6M 5%, 52WRange 5%, RSI28W 5%
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional

from .ranking import cross_sectional_zscore

# 默认权重
DEFAULT_WEIGHTS = {
    'ROE': 0.125,
    'ROIC': 0.125,
    'EV_EBITDA': 0.10,
    'FCF_Yield': 0.10,
    'PS': 0.10,
    'FCF_Growth': 0.05,
    'EPS_Score': 0.10,
    'Growth_Stability': 0.05,
    'RS_12M': 0.10,
    'RS_6M': 0.05,
    'Range_52W': 0.05,
    'RSI_28W': 0.05,
}


def build_composite_factor(
    factors: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None,
    min_factor_pct: float = 0.5,
) -> pd.DataFrame:
    """
    加权复合因子。

    Args:
        factors: {name: DataFrame} 所有子因子
        weights: {name: weight}，默认 DEFAULT_WEIGHTS
        min_factor_pct: 某股票在某日期有效因子数占比低于此值则排除

    Returns:
        DataFrame(index=dates, columns=symbols) 复合因子值
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # 只用有权重的因子
    active = {k: v for k, v in factors.items() if k in weights}

    # z-score 标准化
    zscores = {name: cross_sectional_zscore(df) for name, df in active.items()}

    # 对齐 index
    common_idx = None
    common_cols = None
    for name, df in zscores.items():
        if common_idx is None:
            common_idx = df.index
            common_cols = df.columns
        else:
            common_idx = common_idx.intersection(df.index)
            common_cols = common_cols.intersection(df.columns)

    if common_idx is None or len(common_idx) == 0:
        raise ValueError("No common dates across factors")

    # 加权求和
    composite = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
    weight_sum = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
    valid_count = pd.DataFrame(0, index=common_idx, columns=common_cols)
    total_factors = len(active)

    for name, zdf in zscores.items():
        w = weights[name]
        z = zdf.reindex(index=common_idx, columns=common_cols)
        mask = z.notna()
        composite += z.fillna(0) * w
        weight_sum += mask.astype(float) * w
        valid_count += mask.astype(int)

    # 归一化: 用实际权重之和缩放
    composite = composite / weight_sum.replace(0, np.nan)

    # 排除有效因子过少的股票
    min_count = int(total_factors * min_factor_pct)
    composite[valid_count < min_count] = np.nan

    return composite
