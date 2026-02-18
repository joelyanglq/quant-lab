"""
因子回测引擎

- 周频/月频/任意周期调仓, T+1 执行
- 五分位等权组合
- IC (Information Coefficient) 计算
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

from .ranking import cross_sectional_zscore, assign_quantiles


def _get_month_end_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """获取每月最后一个交易日"""
    s = pd.Series(range(len(dates)), index=dates)
    return s.groupby(s.index.to_period('M')).last().values


def _get_periodic_dates(dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """
    获取指定频率的周期末交易日。

    Args:
        dates: 交易日序列
        freq: pandas resample 频率, 如 'W-FRI', 'M', 'Q', 'W', 'ME'

    Returns:
        周期末日期的索引位置
    """
    s = pd.Series(range(len(dates)), index=dates)
    return s.resample(freq).last().values


def run_factor_backtest(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_quantiles: int = 5,
    min_stocks: int = 10,
) -> Dict:
    """
    单因子回测。

    Args:
        factor: 因子值面板 (index=dates, columns=symbols)
        forward_returns: 未来1月收益 (index=rebalance_dates, columns=symbols)
        n_quantiles: 分位数
        min_stocks: 最小持仓数

    Returns:
        {
            'quantile_returns': DataFrame(index=dates, columns=[1..n_quantiles]),
            'long_short': Series,
            'ic_series': Series (Spearman rank IC per month),
            'quantiles': DataFrame (分组结果),
        }
    """
    # 横截面分组
    quantiles = assign_quantiles(factor, n_quantiles, min_stocks)

    # 月度分位组合收益
    quantile_returns = {}
    for q in range(1, n_quantiles + 1):
        # 每月: 属于 q 分位的股票等权平均收益
        mask = (quantiles == q).astype(float).replace(0, np.nan)
        # 加权平均: 等权 = 简单 mean of forward_returns where mask
        qret = (forward_returns * mask).mean(axis=1)
        quantile_returns[q] = qret

    qret_df = pd.DataFrame(quantile_returns)
    qret_df = qret_df.dropna(how='all')

    # Long-Short: Q5 - Q1
    long_short = qret_df[n_quantiles] - qret_df[1]

    # IC: Spearman rank correlation (factor vs forward returns per month)
    ic_series = _compute_ic(factor, forward_returns)

    return {
        'quantile_returns': qret_df,
        'long_short': long_short,
        'ic_series': ic_series,
        'quantiles': quantiles,
    }


def _compute_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """
    Spearman rank IC: 每个日期的因子值 rank 与未来收益 rank 的相关系数
    """
    common_idx = factor.index.intersection(forward_returns.index)
    ic_vals = []
    ic_dates = []

    for dt in common_idx:
        f = factor.loc[dt].dropna()
        r = forward_returns.loc[dt].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 10:
            continue
        corr = f[common].rank().corr(r[common].rank())
        ic_vals.append(corr)
        ic_dates.append(dt)

    return pd.Series(ic_vals, index=pd.DatetimeIndex(ic_dates), name='IC')


def compute_forward_returns(
    close: pd.DataFrame,
    periods: int = 21,
) -> pd.DataFrame:
    """
    计算未来 N 日收益 (shift forward)。

    月度回测用 periods=21 (约一个月)。
    """
    return close.shift(-periods) / close - 1


def build_monthly_rebalance(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    n_quantiles: int = 5,
    holding_period: int = 21,
    min_stocks: int = 10,
) -> Dict:
    """
    完整的月度调仓回测流程。

    1. 取月末交易日的因子值
    2. T+1 执行 (用下一交易日的 close)
    3. 持有到下个月末
    4. 计算分位收益 + IC

    Args:
        factor: 每日因子面板
        close: 每日收盘价
        n_quantiles: 分位数
        holding_period: 持仓天数 (默认21)
        min_stocks: 每期最少持仓数

    Returns: same as run_factor_backtest
    """
    dates = close.index

    # 月末交易日的 index
    month_end_idx = _get_month_end_dates(dates)
    month_end_dates = dates[month_end_idx]

    # 月末因子值
    factor_monthly = factor.loc[factor.index.isin(month_end_dates)]

    # 未来 holding_period 日收益 (from T+1)
    fwd_ret = close.shift(-holding_period - 1) / close.shift(-1) - 1

    # 只在月末日期上评估
    fwd_ret_monthly = fwd_ret.loc[fwd_ret.index.isin(month_end_dates)]

    # 对齐
    common = factor_monthly.index.intersection(fwd_ret_monthly.index)
    factor_monthly = factor_monthly.loc[common]
    fwd_ret_monthly = fwd_ret_monthly.loc[common]

    return run_factor_backtest(
        factor_monthly, fwd_ret_monthly,
        n_quantiles=n_quantiles, min_stocks=min_stocks,
    )


def build_weekly_rebalance(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    n_quantiles: int = 5,
    holding_period: int = 5,
    min_stocks: int = 10,
) -> Dict:
    """
    完整的周频调仓回测流程。

    1. 取每周五的因子值
    2. T+1 执行 (用下一交易日的 close)
    3. 持有 5 天 (约一周)
    4. 计算分位收益 + IC

    Args:
        factor: 每日因子面板
        close: 每日收盘价
        n_quantiles: 分位数
        holding_period: 持仓天数 (默认5)
        min_stocks: 每期最少持仓数

    Returns: same as run_factor_backtest
    """
    dates = close.index

    # 每周五的 index
    week_end_idx = _get_periodic_dates(dates, 'W-FRI')
    week_end_dates = dates[week_end_idx]

    # 周五因子值
    factor_weekly = factor.loc[factor.index.isin(week_end_dates)]

    # 未来 holding_period 日收益 (from T+1)
    fwd_ret = close.shift(-holding_period - 1) / close.shift(-1) - 1

    # 只在周五日期上评估
    fwd_ret_weekly = fwd_ret.loc[fwd_ret.index.isin(week_end_dates)]

    # 对齐
    common = factor_weekly.index.intersection(fwd_ret_weekly.index)
    factor_weekly = factor_weekly.loc[common]
    fwd_ret_weekly = fwd_ret_weekly.loc[common]

    return run_factor_backtest(
        factor_weekly, fwd_ret_weekly,
        n_quantiles=n_quantiles, min_stocks=min_stocks,
    )


def build_periodic_rebalance(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    rebalance_freq: str = 'M',
    n_quantiles: int = 5,
    holding_period: Optional[int] = None,
    min_stocks: int = 10,
) -> Dict:
    """
    通用的周期性调仓回测流程。

    1. 按指定频率取因子值
    2. T+1 执行 (用下一交易日的 close)
    3. 持有指定天数
    4. 计算分位收益 + IC

    Args:
        factor: 每日因子面板
        close: 每日收盘价
        rebalance_freq: pandas resample 频率
            - 'W-FRI': 每周五
            - 'M' or 'ME': 每月末
            - 'Q' or 'QE': 每季末
            - 'W': 每周日 (最后一个交易日)
        n_quantiles: 分位数
        holding_period: 持仓天数, None 时自动推断:
            - 'W-FRI' or 'W': 5 天
            - 'M' or 'ME': 21 天
            - 'Q' or 'QE': 63 天
        min_stocks: 每期最少持仓数

    Returns: same as run_factor_backtest
    """
    dates = close.index

    # 自动推断持仓周期
    if holding_period is None:
        if rebalance_freq in ('W-FRI', 'W'):
            holding_period = 5
        elif rebalance_freq in ('M', 'ME'):
            holding_period = 21
        elif rebalance_freq in ('Q', 'QE'):
            holding_period = 63
        else:
            # 默认 21 天
            holding_period = 21

    # 周期末交易日的 index
    period_end_idx = _get_periodic_dates(dates, rebalance_freq)
    period_end_dates = dates[period_end_idx]

    # 周期末因子值
    factor_periodic = factor.loc[factor.index.isin(period_end_dates)]

    # 未来 holding_period 日收益 (from T+1)
    fwd_ret = close.shift(-holding_period - 1) / close.shift(-1) - 1

    # 只在周期末日期上评估
    fwd_ret_periodic = fwd_ret.loc[fwd_ret.index.isin(period_end_dates)]

    # 对齐
    common = factor_periodic.index.intersection(fwd_ret_periodic.index)
    factor_periodic = factor_periodic.loc[common]
    fwd_ret_periodic = fwd_ret_periodic.loc[common]

    return run_factor_backtest(
        factor_periodic, fwd_ret_periodic,
        n_quantiles=n_quantiles, min_stocks=min_stocks,
    )
