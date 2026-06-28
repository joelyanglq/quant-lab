"""
换手率分析模块

分组换手率 + 行业换手率。
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..ranking import assign_quantiles


@dataclass
class TurnoverReport:
    """换手率分析结果"""
    group_turnover: pd.Series      # index=Q1..Q5, values=平均换手率
    industry_turnover: pd.Series   # index=industry, values=平均权重变化
    n_rebalances: int


def compute_turnover_analysis(
    factor: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex = None,
    n_quantiles: int = 5,
    sector_map: pd.Series = None,
) -> TurnoverReport:
    """
    换手率分析。

    Args:
        factor: 预处理后因子面板 (daily)
        rebalance_dates: 调仓日期。None 时按月末。
        n_quantiles: 分组数
        sector_map: sector 映射 (用于行业换手)

    Returns:
        TurnoverReport
    """
    # 确定调仓日期
    if rebalance_dates is None:
        s = pd.Series(range(len(factor.index)), index=factor.index)
        idx = s.resample("M").last().values
        rebalance_dates = factor.index[idx]

    factor_rebal = factor.loc[factor.index.isin(rebalance_dates)]
    quantiles = assign_quantiles(factor_rebal, n_quantiles)

    # ── 分组换手率 ──
    group_turnovers = {q: [] for q in range(1, n_quantiles + 1)}
    dates = quantiles.index.tolist()

    for i in range(1, len(dates)):
        prev = quantiles.loc[dates[i - 1]]
        curr = quantiles.loc[dates[i]]

        for q in range(1, n_quantiles + 1):
            prev_members = set(prev[prev == q].dropna().index)
            curr_members = set(curr[curr == q].dropna().index)
            if not prev_members and not curr_members:
                continue
            union = prev_members | curr_members
            if len(union) == 0:
                continue
            # 换手率: |w_t - w_{t-1}| 的和
            # 等权组合: w = 1/n if in group, 0 otherwise
            n_prev = len(prev_members) or 1
            n_curr = len(curr_members) or 1
            turnover = 0
            for sym in union:
                w_prev = 1.0 / n_prev if sym in prev_members else 0
                w_curr = 1.0 / n_curr if sym in curr_members else 0
                turnover += abs(w_curr - w_prev)
            group_turnovers[q].append(turnover)

    group_avg = pd.Series(
        {f"Q{q}": np.mean(v) if v else 0 for q, v in group_turnovers.items()},
        name="avg_turnover"
    )

    # ── 行业换手率 (top group) ──
    industry_turnover = pd.Series(dtype=float)
    if sector_map is not None:
        industry_changes = []
        for i in range(1, len(dates)):
            prev = quantiles.loc[dates[i - 1]]
            curr = quantiles.loc[dates[i]]
            prev_top = set(prev[prev == n_quantiles].dropna().index)
            curr_top = set(curr[curr == n_quantiles].dropna().index)

            # 行业权重
            prev_weights = _industry_weights(prev_top, sector_map)
            curr_weights = _industry_weights(curr_top, sector_map)

            all_industries = set(prev_weights.keys()) | set(curr_weights.keys())
            change = {ind: abs(curr_weights.get(ind, 0) - prev_weights.get(ind, 0))
                      for ind in all_industries}
            industry_changes.append(change)

        if industry_changes:
            industry_df = pd.DataFrame(industry_changes)
            industry_turnover = industry_df.mean().sort_values(ascending=False)
            industry_turnover.name = "avg_industry_turnover"

    report = TurnoverReport(
        group_turnover=group_avg,
        industry_turnover=industry_turnover,
        n_rebalances=len(dates) - 1,
    )
    print_turnover_report(report)
    return report


def _industry_weights(symbols: set, sector_map: pd.Series) -> Dict[str, float]:
    """计算一组 symbols 的行业权重（等权）。"""
    if not symbols:
        return {}
    sectors = sector_map.reindex(list(symbols)).dropna()
    if sectors.empty:
        return {}
    return (sectors.value_counts() / len(sectors)).to_dict()


def print_turnover_report(report: TurnoverReport):
    """打印换手率指标。"""
    print("\n" + "-" * 40)
    print(f"TURNOVER ANALYSIS ({report.n_rebalances} rebalances)")
    print("-" * 40)
    print("  Group avg turnover:")
    for q, v in report.group_turnover.items():
        print(f"    {q}: {v:.1%}")
    if not report.industry_turnover.empty:
        print("  Top group industry turnover (top 5):")
        for ind, v in report.industry_turnover.head(5).items():
            print(f"    {ind}: {v:.1%}")
    print("-" * 40 + "\n")
