"""
回归分析模块

截面 OLS 因子收益率 + 累计因子收益率。
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RegressionReport:
    """回归分析结果"""
    factor_returns: pd.Series      # 每日因子收益率 (回归系数 β)
    t_stats: pd.Series             # 每日 t-stat
    cumsum: pd.Series              # 累计因子收益
    max_cumsum: pd.Series          # running max cumsum
    min_cumsum: pd.Series          # running min cumsum
    mean_return: float
    mean_t: float
    ann_return: float


def compute_regression_analysis(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int = 21,
) -> RegressionReport:
    """
    截面回归因子收益率分析。

    每日: ret_i = α + β * factor_i + ε，β 即 factor return。

    Args:
        factor: 预处理后因子面板
        close: 收盘价面板
        horizon: 前向收益周期

    Returns:
        RegressionReport
    """
    fwd_ret = close.shift(-horizon) / close - 1

    common_dates = factor.index.intersection(fwd_ret.index)
    betas = []
    tstats = []
    dates = []

    for dt in common_dates:
        f = factor.loc[dt].dropna()
        r = fwd_ret.loc[dt].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 20:
            continue

        x = f[common].values
        y = r[common].values

        # OLS: y = α + β*x
        X = np.column_stack([np.ones(len(x)), x])
        try:
            beta_vec = np.linalg.lstsq(X, y, rcond=None)[0]
            beta = beta_vec[1]
            resid = y - X @ beta_vec
            se = np.sqrt(np.sum(resid ** 2) / (len(y) - 2)) / np.sqrt(np.sum((x - x.mean()) ** 2))
            t = beta / se if se > 0 else 0
        except np.linalg.LinAlgError:
            continue

        betas.append(beta)
        tstats.append(t)
        dates.append(dt)

    factor_returns = pd.Series(betas, index=pd.DatetimeIndex(dates), name="factor_return")
    t_stats = pd.Series(tstats, index=pd.DatetimeIndex(dates), name="t_stat")

    cumsum = factor_returns.cumsum()
    max_cumsum = cumsum.cummax()
    min_cumsum = cumsum.cummin()

    mean_ret = factor_returns.mean()
    mean_t = t_stats.mean()
    ann_factor = 252 / max(horizon, 1)
    ann_return = mean_ret * ann_factor

    report = RegressionReport(
        factor_returns=factor_returns,
        t_stats=t_stats,
        cumsum=cumsum,
        max_cumsum=max_cumsum,
        min_cumsum=min_cumsum,
        mean_return=round(mean_ret, 6),
        mean_t=round(mean_t, 4),
        ann_return=round(ann_return, 4),
    )

    print_regression_report(report)
    return report


def print_regression_report(report: RegressionReport):
    """打印回归分析指标。"""
    print("\n" + "-" * 40)
    print("REGRESSION ANALYSIS")
    print("-" * 40)
    print(f"  Mean factor return: {report.mean_return:.6f}")
    print(f"  Mean t-stat:        {report.mean_t:.4f}")
    print(f"  Annualized return:  {report.ann_return:.4f}")
    print("-" * 40 + "\n")
