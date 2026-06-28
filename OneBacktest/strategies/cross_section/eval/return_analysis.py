"""
收益分析模块

分层回测 + top 组合评价（含回撤、fitness、超额收益）。
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..backtest import build_periodic_rebalance


@dataclass
class ReturnReport:
    """收益分析结果"""
    quantile_cumrets: pd.DataFrame   # columns=Q1..Q5 + L/S
    top_metrics: Dict[str, float]    # ann_return, ann_vol, sharpe, maxdd, fitness, margin
    top_drawdown: pd.Series          # drawdown series
    ls_cumret: pd.Series             # L/S cumulative return


def compute_return_analysis(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int = 21,
    rebalance_freq: str = "M",
    n_quantiles: int = 5,
    benchmark_close: pd.Series = None,
) -> ReturnReport:
    """
    分层回测 + top 组合评价。

    Args:
        factor: 预处理后因子面板
        close: 收盘价面板
        horizon: 持仓周期
        rebalance_freq: 调仓频率
        n_quantiles: 分组数
        benchmark_close: 基准收盘价 (用于计算超额收益)

    Returns:
        ReturnReport
    """
    bt = build_periodic_rebalance(
        factor, close,
        rebalance_freq=rebalance_freq,
        n_quantiles=n_quantiles,
        holding_period=horizon,
    )

    qret = bt["quantile_returns"]
    ls = bt["long_short"]

    # 累计收益
    cum_rets = {}
    for q in range(1, n_quantiles + 1):
        if q in qret.columns:
            cum_rets[f"Q{q}"] = (1 + qret[q]).cumprod() - 1
    cum_rets["L/S"] = (1 + ls).cumprod() - 1
    quantile_cumrets = pd.DataFrame(cum_rets)

    # Top 组合 (Q5) 指标
    top_ret = qret[n_quantiles].dropna()
    top_metrics = _compute_top_metrics(top_ret, horizon, benchmark_close, close)

    # Drawdown
    top_cumret = (1 + top_ret).cumprod()
    top_drawdown = top_cumret / top_cumret.cummax() - 1

    report = ReturnReport(
        quantile_cumrets=quantile_cumrets,
        top_metrics=top_metrics,
        top_drawdown=top_drawdown,
        ls_cumret=quantile_cumrets["L/S"],
    )

    print_return_report(report)
    return report


def _compute_top_metrics(
    top_ret: pd.Series,
    horizon: int,
    benchmark_close: pd.Series = None,
    close: pd.DataFrame = None,
) -> Dict[str, float]:
    """计算 top 组合的评价指标。"""
    n = len(top_ret)
    if n < 2:
        return {}

    ann_factor = 252 / max(horizon, 1)
    ann_return = top_ret.mean() * ann_factor
    ann_vol = top_ret.std() * np.sqrt(ann_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cumret = (1 + top_ret).cumprod()
    maxdd = (cumret / cumret.cummax() - 1).min()

    # Turnover proxy (简单估算)
    turnover = 1.0  # 占位

    # Fitness: sharpe / max(1, turnover)
    fitness = sharpe / max(1, turnover)

    # Margin (超额收益)
    margin = np.nan
    if benchmark_close is not None:
        bm_ret = benchmark_close.pct_change().reindex(top_ret.index)
        bm_ann = bm_ret.mean() * 252
        margin = ann_return - bm_ann

    return {
        "ann_return": round(ann_return, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(maxdd, 4),
        "fitness": round(fitness, 4),
        "margin": round(margin, 4) if not np.isnan(margin) else "N/A",
    }


def print_return_report(report: ReturnReport):
    """打印收益分析指标。"""
    print("\n" + "-" * 40)
    print("RETURN ANALYSIS — Top Group Metrics")
    print("-" * 40)
    for k, v in report.top_metrics.items():
        if isinstance(v, float):
            if "return" in k or "vol" in k or "margin" in k or "drawdown" in k:
                print(f"  {k:15s}: {v:>8.2%}")
            else:
                print(f"  {k:15s}: {v:>8.4f}")
        else:
            print(f"  {k:15s}: {v}")
    print("-" * 40 + "\n")
